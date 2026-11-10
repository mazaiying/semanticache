"""
SemantiCache v2 — KV engine with token-level reuse and selective recompute.

Core idea
---------
A prompt is a sequence of T tokens.  For each position we either
  * reuse a materialized KV entry (pre-RoPE key + value, re-rotated to the
    target position), or
  * compute it fresh.
Optionally a fraction r of the reused tokens with the largest layer-0 value
deviation are recomputed in all layers (CacheBlend-style selective recompute)
to repair cross-chunk attention.

Everything (full prefill, prefix caching, chunk reuse, delta reuse, naive
substitution) is expressed through `prefill()` so every method shares exactly
the same kernels and timing code.

Supports HF Llama-family decoder layers (Qwen2, Llama, Mistral) that expose
self_attn.{q,k,v,o}_proj, input_layernorm, post_attention_layernorm, mlp.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


@dataclass
class ChunkKV:
    """Materialized KV of a token span; keys are stored PRE-RoPE."""
    token_ids: List[int]
    k: List[torch.Tensor]  # per layer [Hkv, T, D]
    v: List[torch.Tensor]  # per layer [Hkv, T, D]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def to(self, device, non_blocking=True) -> "ChunkKV":
        return ChunkKV(self.token_ids,
                       [t.to(device, non_blocking=non_blocking) for t in self.k],
                       [t.to(device, non_blocking=non_blocking) for t in self.v])


@dataclass
class PrefillResult:
    first_token: int
    cache: object               # transformers DynamicCache (post-RoPE)
    pre_k: List[torch.Tensor]   # per layer [Hkv, T, D] pre-RoPE (for materialization)
    v: List[torch.Tensor]
    n_computed: int             # tokens computed in layers >= 1
    n_total: int
    ttft_ms: float


class KVEngine:
    def __init__(self, model_path: str, device: str = "cuda:0",
                 dtype=torch.bfloat16):
        self.model_path, self.device, self.dtype = model_path, device, dtype

    # ------------------------------------------------------------------ load
    def load(self):
        self.tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=self.dtype, device_map=self.device,
            attn_implementation="sdpa", trust_remote_code=True)
        self.model.eval()
        cfg = self.model.config
        self.cfg = cfg
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = getattr(cfg, "num_key_value_heads", self.n_heads)
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.layers = self.model.model.layers
        self.rotary = self.model.model.rotary_emb
        self.embed = self.model.model.embed_tokens
        self.norm = self.model.model.norm
        self.lm_head = self.model.lm_head
        self.eos_ids = set()
        for e in [self.tok.eos_token_id, getattr(self.model.generation_config, "eos_token_id", None)]:
            if isinstance(e, int):
                self.eos_ids.add(e)
            elif isinstance(e, (list, tuple)):
                self.eos_ids.update(e)
        return self

    # --------------------------------------------------------------- helpers
    def _cos_sin(self, T: int):
        pos = torch.arange(T, device=self.device).unsqueeze(0)
        dummy = torch.empty(1, 1, self.head_dim, device=self.device, dtype=self.dtype)
        cos, sin = self.rotary(dummy, pos)          # [1, T, D]
        return cos[0], sin[0]                       # [T, D]

    @staticmethod
    def _rope(x, cos, sin):
        # x [H, N, D], cos/sin [N, D]
        return x * cos.unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0)

    def kv_bytes_per_token(self) -> int:
        return 2 * self.n_layers * self.n_kv * self.head_dim * torch.finfo(self.dtype).bits // 8

    # --------------------------------------------------------------- prefill
    @torch.no_grad()
    def prefill(self,
                token_ids: List[int],
                reuse_mask: Optional[torch.Tensor] = None,     # bool [T] (cpu or gpu)
                reuse_k: Optional[List[torch.Tensor]] = None,  # per layer [Hkv, T, D] pre-RoPE
                reuse_v: Optional[List[torch.Tensor]] = None,
                recompute_ratio: float = 0.0,
                build_cache: bool = True,
                t_start: Optional[float] = None) -> PrefillResult:
        """Prefill with partial reuse. Timing starts at t_start (if given)."""
        dev = self.device
        if t_start is None:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        T = len(token_ids)
        ids = torch.tensor(token_ids, device=dev)
        cos, sin = self._cos_sin(T)
        h = self.embed(ids)                                  # [T, hidden]

        if reuse_mask is None or not bool(reuse_mask.any()):
            reuse_mask = torch.zeros(T, dtype=torch.bool, device=dev)
            has_reuse = False
        else:
            reuse_mask = reuse_mask.to(dev)
            has_reuse = True
        if not bool((~reuse_mask)[-1]):
            raise ValueError("last token must be computed")

        compute_idx = torch.nonzero(~reuse_mask).squeeze(1)  # always computed
        do_select = has_reuse and recompute_ratio > 0
        active = torch.arange(T, device=dev) if do_select else compute_idx

        out_k, out_v, pre_k_all = [], [], []
        amask = None
        CHK = 1  # layer at which value deviation is measured (layer 0 is context-free)
        H, Hk, D = self.n_heads, self.n_kv, self.head_dim
        rep = H // Hk
        n_computed = len(compute_idx)

        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h[active])             # [A, hidden]
            A = x.shape[0]
            q = attn.q_proj(x).view(A, H, D).transpose(0, 1)   # [H, A, D]
            k = attn.k_proj(x).view(A, Hk, D).transpose(0, 1)  # [Hk, A, D]
            v = attn.v_proj(x).view(A, Hk, D).transpose(0, 1)

            if (not has_reuse) or A == T:
                # every token computed in this layer (no reuse, or layer 0 of
                # selective recompute)
                Kpre, Vf = k, v
                if do_select and li == CHK:
                    # pick reused tokens with the largest layer-0 value deviation
                    Vc = reuse_v[CHK].to(dev, non_blocking=True)
                    score = (v.float() - Vc.float()).norm(dim=-1).mean(0)  # [T]
                    score[~reuse_mask] = -1.0
                    n_reused = int(reuse_mask.sum())
                    n_sel = max(1, int(round(recompute_ratio * n_reused)))
                    sel = torch.topk(score, n_sel).indices
                    keep = (~reuse_mask).clone()
                    keep[sel] = True
                    next_active = torch.nonzero(keep).squeeze(1)
                    n_computed = len(next_active)
            else:
                # NOTE: reuse_k/reuse_v are consumed (modified in place)
                Kpre = reuse_k[li].to(dev, non_blocking=True)
                Vf = reuse_v[li].to(dev, non_blocking=True)
                Kpre[:, active] = k
                Vf[:, active] = v

            Kr = self._rope(Kpre, cos, sin)                   # [Hk, T, D]
            qr = self._rope(q, cos[active], sin[active])      # [H, A, D]
            Kh = Kr.repeat_interleave(rep, dim=0) if rep > 1 else Kr
            Vh = Vf.repeat_interleave(rep, dim=0) if rep > 1 else Vf
            if A == T:
                o = F.scaled_dot_product_attention(qr.unsqueeze(0), Kh.unsqueeze(0),
                                                   Vh.unsqueeze(0), is_causal=True)[0]
            else:
                if amask is None:
                    allowed = active.unsqueeze(1) >= torch.arange(T, device=dev).unsqueeze(0)
                    amask = torch.zeros(A, T, device=dev, dtype=self.dtype)
                    amask.masked_fill_(~allowed, float("-inf"))
                    amask = amask.unsqueeze(0).unsqueeze(0)
                o = F.scaled_dot_product_attention(qr.unsqueeze(0), Kh.unsqueeze(0),
                                                   Vh.unsqueeze(0), attn_mask=amask)[0]
            o = o.transpose(0, 1).reshape(A, H * D)
            hs = h[active] + attn.o_proj(o)
            hs = hs + layer.mlp(layer.post_attention_layernorm(hs))
            h[active] = hs

            out_k.append(Kr)
            out_v.append(Vf)
            pre_k_all.append(Kpre)
            if do_select and li == CHK:
                active = next_active

        last = self.norm(h[T - 1: T])
        logits = self.lm_head(last)
        first = int(torch.argmax(logits[0]).item())
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - t_start) * 1000.0

        cache = None
        if build_cache:
            from transformers import DynamicCache
            cache = DynamicCache()
            for li in range(self.n_layers):
                cache.update(out_k[li].unsqueeze(0), out_v[li].unsqueeze(0), li)
        return PrefillResult(first, cache, pre_k_all, out_v, n_computed, T, ttft)

    # ---------------------------------------------------------------- decode
    @torch.no_grad()
    def decode(self, pr: PrefillResult, max_new_tokens: int = 32) -> Tuple[List[int], float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen = [pr.first_token]
        cache = pr.cache
        cur = torch.tensor([[pr.first_token]], device=self.device)
        pos = pr.n_total
        for _ in range(max_new_tokens - 1):
            if gen[-1] in self.eos_ids:
                break
            out = self.model(input_ids=cur, past_key_values=cache, use_cache=True,
                             position_ids=torch.tensor([[pos]], device=self.device),
                             cache_position=torch.tensor([pos], device=self.device))
            nxt = int(torch.argmax(out.logits[0, -1]).item())
            gen.append(nxt)
            cache = out.past_key_values
            cur = torch.tensor([[nxt]], device=self.device)
            pos += 1
        torch.cuda.synchronize()
        return gen, (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------- materialization
    @torch.no_grad()
    def materialize(self, token_ids: List[int], to_cpu: bool = True) -> ChunkKV:
        """Encode a chunk independently (positions 0..T-1) and keep pre-RoPE KV.
        The final token is computed as usual; no first-token is needed."""
        pr = self.prefill(token_ids, build_cache=False)
        if to_cpu:
            k = [t.to("cpu").pin_memory() for t in pr.pre_k]
            v = [t.to("cpu").pin_memory() for t in pr.v]
        else:
            k, v = pr.pre_k, pr.v
        return ChunkKV(list(token_ids), k, v)
