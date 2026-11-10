"""
Methods compared in SemantiCache v2. Every method returns a per-request record.

  full        : no reuse (prefix of the system prompt is NOT reused)
  apc         : block-level longest-prefix reuse against recent requests
                (vLLM Automatic Prefix Caching semantics, block = 16)
  cacheblend  : exact chunk reuse (hash of passage text) + selective recompute
  naive_sem   : exact chunk reuse; on miss, substitute the most similar base
                chunk (its tokens AND KV) if cos >= tau  (SemShareKV-style /
                our VLDB'26 design)
  gptcache    : response cache keyed by question embedding (cos >= tau_q)
  semanticache: exact chunk reuse; on miss, find most similar base chunk,
                token-align it with the request chunk, reuse aligned tokens,
                compute the delta tokens, plus selective recompute.
"""
from __future__ import annotations

import difflib
import hashlib
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from engine import ChunkKV, KVEngine
from workload import PromptBuilder, Request, best_f1


def _h(ids) -> str:
    return hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


# --------------------------------------------------------------- KV store
class QChunk:
    """Compressed materialized view: per-(head, token) absmax-scaled int8 / fp8."""

    def __init__(self, token_ids, qk, sk, qv, sv):
        self.token_ids, self.qk, self.sk, self.qv, self.sv = token_ids, qk, sk, qv, sv

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.qk + self.sk + self.qv + self.sv)

    def to(self, device, non_blocking=True) -> "QChunk":
        mv = lambda ts: [t.to(device, non_blocking=non_blocking) for t in ts]
        return QChunk(self.token_ids, mv(self.qk), mv(self.sk), mv(self.qv), mv(self.sv))


def _quant(x: torch.Tensor, mode: str):
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    if mode == "int8":
        s = amax / 127.0
        q = (xf / s).round().clamp_(-127, 127).to(torch.int8)
    elif mode == "fp8":
        s = amax / 448.0
        q = (xf / s).to(torch.float8_e4m3fn)
    else:
        raise ValueError(mode)
    return q, s.to(torch.float16)


def _dequant(q: torch.Tensor, s: torch.Tensor, dtype) -> torch.Tensor:
    return (q.to(torch.float32) * s.to(torch.float32)).to(dtype)


class TieredKVStore:
    """Materialized chunk-KV views. GPU tier (LRU, byte budget) over a CPU tier.
    Materialization happens offline (ingestion) and is not charged to TTFT.
    kv_quant in {none, int8, fp8}: views are stored compressed in BOTH tiers and
    dequantized on every fetch (dequantization time IS charged to TTFT)."""

    def __init__(self, engine: KVEngine, gpu_budget_gb: float = 20.0, kv_quant: str = "none"):
        self.e = engine
        self.cpu: Dict[str, ChunkKV] = {}
        self.gpu: "OrderedDict[str, ChunkKV]" = OrderedDict()
        self.gpu_bytes = 0
        self.budget = gpu_budget_gb * (1 << 30)
        self.quant = kv_quant
        self.stats = {"gpu_hit": 0, "cpu_hit": 0, "bytes_h2d": 0}

    def ensure(self, key: str, token_ids: List[int]):
        if key in self.cpu:
            return
        if self.quant == "none":
            self.cpu[key] = self.e.materialize(token_ids, to_cpu=True)
            return
        c = self.e.materialize(token_ids, to_cpu=False)
        qk, sk, qv, sv = [], [], [], []
        for k, v in zip(c.k, c.v):
            a, b = _quant(k, self.quant)
            qk.append(a.to("cpu").pin_memory()); sk.append(b.to("cpu").pin_memory())
            a, b = _quant(v, self.quant)
            qv.append(a.to("cpu").pin_memory()); sv.append(b.to("cpu").pin_memory())
        self.cpu[key] = QChunk(list(token_ids), qk, sk, qv, sv)

    def _view(self, c):
        if isinstance(c, QChunk):
            dt = self.e.dtype
            return ChunkKV(c.token_ids, [_dequant(q, s, dt) for q, s in zip(c.qk, c.sk)],
                           [_dequant(q, s, dt) for q, s in zip(c.qv, c.sv)])
        return c

    def drop_cpu(self, keys):
        for k in keys:
            self.cpu.pop(k, None)
            if k in self.gpu:
                c = self.gpu.pop(k)
                self.gpu_bytes -= c.nbytes()

    def fetch(self, key: str) -> ChunkKV:
        if key in self.gpu:
            self.gpu.move_to_end(key)
            self.stats["gpu_hit"] += 1
            return self._view(self.gpu[key])
        c = self.cpu[key]
        g = c.to(self.e.device)
        self.stats["cpu_hit"] += 1
        self.stats["bytes_h2d"] += c.nbytes()
        nb = g.nbytes()
        while self.gpu and self.gpu_bytes + nb > self.budget:
            _, old = self.gpu.popitem(last=False)
            self.gpu_bytes -= old.nbytes()
        if nb <= self.budget:
            self.gpu[key] = g
            self.gpu_bytes += nb
        return self._view(g)

    def footprint(self) -> Dict[str, float]:
        cpu_b = sum(c.nbytes() for c in self.cpu.values())
        return {"cpu_gb": cpu_b / 2**30, "gpu_gb": self.gpu_bytes / 2**30,
                "n_chunks": len(self.cpu)}


# ---------------------------------------------------------- semantic index
class SemanticIndex:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda:0"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model_name, device=device)
        self.keys: List[str] = []
        self.mat: Optional[torch.Tensor] = None
        self.device = device

    def embed(self, texts: List[str]) -> torch.Tensor:
        return self.m.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                             batch_size=64, show_progress_bar=False)

    def build(self, keys: List[str], texts: List[str]):
        self.keys = list(keys)
        self.mat = self.embed(texts)

    def query(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        q = self.embed(texts)
        sims = q @ self.mat.T
        v, i = sims.max(dim=1)
        return [self.keys[j] for j in i.tolist()], v.tolist()


# ------------------------------------------------------------------ runner
class Runner:
    def __init__(self, engine: KVEngine, pb: PromptBuilder, corpus, index: SemanticIndex,
                 store: TieredKVStore, max_new_tokens=32):
        self.e, self.pb, self.corpus, self.idx, self.store = engine, pb, corpus, index, store
        self.mnt = max_new_tokens
        # base text -> pid (exact chunk lookup is by text hash of BASE corpus)
        self.text2pid = {_h(pb.passage_ids(p.text)): pid for pid, p in corpus.items()}
        self.prefix_key = "__prefix__"
        self.store.ensure(self.prefix_key, pb.prefix_ids)
        # APC state: recent requests' token ids + pre-RoPE KV on GPU
        self.apc: "OrderedDict[str, Tuple[List[int], List[torch.Tensor], List[torch.Tensor]]]" = OrderedDict()
        self.apc_cap = 4
        self.gpt: Dict[str, Tuple[torch.Tensor, str]] = {}
        # request-level diff reuse state (KVShare-style baseline)
        self.rq: "OrderedDict[str, Tuple[List[int], List[torch.Tensor], List[torch.Tensor]]]" = OrderedDict()
        self.profile = False

    # -------------------------------------------------------- assembly
    def _assemble(self, segs):
        """segs: list of (token_ids, kv: ChunkKV-on-gpu or None, src_idx or None)
        src_idx: LongTensor of positions in kv for each token (-1 => compute)."""
        e = self.e
        ids, masks = [], []
        for tok, kv, src in segs:
            ids.extend(tok)
            if kv is None:
                masks.append(torch.zeros(len(tok), dtype=torch.bool))
            else:
                masks.append(src >= 0)
        T = len(ids)
        mask = torch.cat(masks).to(e.device)
        if not bool(mask.any()):
            return ids, None, None, None
        Ks, Vs = [], []
        # precompute per-segment copy plans ONCE (no per-layer CPU syncs)
        plans, off = [], 0
        for tok, kv, src in segs:
            n = len(tok)
            if kv is not None:
                m = src >= 0
                if bool(m.all()):
                    plans.append((off, n, kv, None, None))
                else:
                    dst = (torch.nonzero(m).squeeze(1) + off).to(e.device)
                    plans.append((off, n, kv, dst, src[m].to(e.device)))
                if bool(m.all()) and not bool((src == torch.arange(n)).all()):
                    plans[-1] = (off, n, kv, (torch.arange(n) + off).to(e.device), src.to(e.device))
            off += n
        for li in range(e.n_layers):
            K = torch.empty(e.n_kv, T, e.head_dim, device=e.device, dtype=e.dtype)
            V = torch.empty_like(K)
            for off, n, kv, dst, s in plans:
                if dst is None:
                    K[:, off:off + n] = kv.k[li][:, :n]
                    V[:, off:off + n] = kv.v[li][:, :n]
                else:
                    K[:, dst] = kv.k[li][:, s]
                    V[:, dst] = kv.v[li][:, s]
            Ks.append(K)
            Vs.append(V)
        return ids, mask, Ks, Vs

    def _full_src(self, n):
        return torch.arange(n)

    def _finish(self, req: Request, method: str, pr, t0, extra=None, text=None):
        if text is None:
            gen, dec_ms = self.e.decode(pr, self.mnt)
            text = self.e.tok.decode(gen, skip_special_tokens=True)
            ttft = pr.ttft_ms
            n_comp, n_tot = pr.n_computed, pr.n_total
        else:
            dec_ms, ttft = 0.0, (time.perf_counter() - t0) * 1000
            n_comp, n_tot = 0, 0
        rec = {"rid": req.rid, "kind": req.kind, "method": method,
               "ttft_ms": ttft, "e2e_ms": ttft + dec_ms,
               "n_computed": n_comp, "n_tokens": n_tot,
               "pred": text.strip(), "f1": best_f1(text, req.golds)}
        if "orig_answer" in req.meta:
            rec["f1_old_answer"] = best_f1(text, [req.meta["orig_answer"]])
        if extra:
            rec.update(extra)
        return rec

    # ---------------------------------------------------------- methods
    def full(self, req: Request):
        ids = list(self.pb.prefix_ids)
        for p in req.passages:
            ids += self.pb.passage_ids(p.text)
        ids += self.pb.suffix_ids(req.question)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pr = self.e.prefill(ids, t_start=t0)
        return self._finish(req, "full", pr, t0)

    def apc_method(self, req: Request, block=16):
        ids = list(self.pb.prefix_ids)
        for p in req.passages:
            ids += self.pb.passage_ids(p.text)
        ids += self.pb.suffix_ids(req.question)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        best, best_key = 0, None
        for key, (pids, _, _) in self.apc.items():
            n = 0
            m = min(len(pids), len(ids)) - 1
            while n < m and pids[n] == ids[n]:
                n += 1
            n = (n // block) * block
            if n > best:
                best, best_key = n, key
        if best >= 256:
            _, Kc, Vc = self.apc[best_key]
            self.apc.move_to_end(best_key)
            T = len(ids)
            mask = torch.zeros(T, dtype=torch.bool, device=self.e.device)
            mask[:best] = True
            Ks, Vs = [], []
            for li in range(self.e.n_layers):
                K = torch.zeros(self.e.n_kv, T, self.e.head_dim, device=self.e.device, dtype=self.e.dtype)
                V = torch.zeros_like(K)
                K[:, :best] = Kc[li][:, :best]
                V[:, :best] = Vc[li][:, :best]
                Ks.append(K)
                Vs.append(V)
            pr = self.e.prefill(ids, mask, Ks, Vs, 0.0, t_start=t0)
        else:
            pr = self.e.prefill(ids, t_start=t0)
        rec = self._finish(req, "apc", pr, t0, {"prefix_hit_tokens": best})
        self.apc[req.rid] = (ids, pr.pre_k, pr.v)
        while len(self.apc) > self.apc_cap:
            self.apc.popitem(last=False)
        return rec

    def gptcache(self, req: Request, tau_q=0.9):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        q = self.idx.embed([req.question])[0]
        best, ans = -1.0, None
        for _, (emb, a) in self.gpt.items():
            s = float(emb @ q)
            if s > best:
                best, ans = s, a
        if ans is not None and best >= tau_q:
            torch.cuda.synchronize()
            rec = self._finish(req, "gptcache", None, t0, {"hit": True, "sim": best}, text=ans)
            return rec
        rec = self.full(req)
        rec["method"] = "gptcache"
        rec["hit"] = False
        self.gpt[req.rid] = (q, rec["pred"])
        return rec

    def _tick(self) -> float:
        if self.profile:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _chunk_plan(self, req: Request, mode: str, tau: float, omega: float, halo: int = 0):
        """Return segs and stats. mode in {cacheblend, naive_sem, semanticache}.
        stats also carries a latency breakdown (ms): t_lookup (hash + semantic
        index), t_align (token alignment + halo), t_fetch (view fetch, H2D,
        dequantization). Exact attribution requires --profile (adds syncs)."""
        tl = ta = tf = 0.0
        t = self._tick()
        pre = self.store.fetch(self.prefix_key)
        t2 = self._tick(); tf += t2 - t
        segs = [(self.pb.prefix_ids, pre, self._full_src(len(self.pb.prefix_ids)))]
        stats = {"exact": 0, "semantic": 0, "miss": 0, "reused_tok": 0, "delta_tok": 0,
                 "sem_lookup": 0, "sem_correct": 0, "sem_distractor": 0}
        t = self._tick()
        p_ids = [self.pb.passage_ids(p.text) for p in req.passages]
        keys = [self.text2pid.get(_h(tk)) for tk in p_ids]
        miss_idx = [i for i, k in enumerate(keys) if k is None]
        cand, sims = {}, {}
        if mode != "cacheblend" and miss_idx:
            ck, cs = self.idx.query([req.passages[i].text for i in miss_idx])
            for i, k, s in zip(miss_idx, ck, cs):
                cand[i], sims[i] = k, s
                stats["sem_lookup"] += 1
                stats["sem_correct"] += int(k == req.passages[i].pid)
                stats["sem_distractor"] += int(k.startswith("__dist__"))
        tl += self._tick() - t
        for i, (p, tok) in enumerate(zip(req.passages, p_ids)):
            k = keys[i]
            if k is not None:
                t = self._tick()
                kv = self.store.fetch(k)
                tf += self._tick() - t
                segs.append((tok, kv, self._full_src(len(tok))))
                stats["exact"] += 1
                stats["reused_tok"] += len(tok)
                continue
            if i in cand and sims[i] >= tau and not cand[i].startswith("__dist__"):
                base_pid = cand[i]
                base_tok = self.pb.passage_ids(self.corpus[base_pid].text)
                t = self._tick()
                kv = self.store.fetch(base_pid)
                tf += self._tick() - t
                if mode == "naive_sem":
                    segs.append((base_tok, kv, self._full_src(len(base_tok))))
                    stats["semantic"] += 1
                    stats["reused_tok"] += len(base_tok)
                    continue
                t = self._tick()
                sm = difflib.SequenceMatcher(None, base_tok, tok, autojunk=False)
                src = torch.full((len(tok),), -1, dtype=torch.long)
                for a, b, n in sm.get_matching_blocks():
                    if n:
                        src[b:b + n] = torch.arange(a, a + n)
                ov = float((src >= 0).float().mean())
                if ov >= omega and halo > 0:
                    # boundary halo: reused tokens within `halo` positions of any
                    # unmatched token are recomputed (their context changed)
                    bad = (src < 0).numpy().astype(np.int8)
                    kk = np.ones(2 * halo + 1, dtype=np.int8)
                    near = np.convolve(bad, kk, mode="same") > 0
                    src[torch.from_numpy(near)] = -1
                ta += self._tick() - t
                if ov >= omega:
                    segs.append((tok, kv, src))
                    stats["semantic"] += 1
                    nr = int((src >= 0).sum())
                    stats["reused_tok"] += nr
                    stats["delta_tok"] += len(tok) - nr
                    continue
            segs.append((tok, None, None))
            stats["miss"] += 1
        segs.append((self.pb.suffix_ids(req.question), None, None))
        stats.update({"t_lookup": tl * 1000, "t_align": ta * 1000, "t_fetch": tf * 1000})
        return segs, stats

    def chunked(self, req: Request, mode: str, ratio=0.15, tau=0.8, omega=0.5, halo=0, tag=None):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        segs, stats = self._chunk_plan(req, mode, tau, omega, halo)
        t1 = self._tick()
        ids, mask, Ks, Vs = self._assemble(segs)
        t2 = self._tick()
        pr = self.e.prefill(ids, mask, Ks, Vs, ratio, t_start=t0)
        stats["t_assemble"] = (t2 - t1) * 1000
        stats["t_plan"] = (t1 - t0) * 1000
        stats["t_prefill"] = pr.ttft_ms - (t2 - t0) * 1000
        return self._finish(req, tag or mode, pr, t0, stats)

    # ----------------------------------------------- request-level diff reuse
    def reqdiff(self, req: Request, ratio=0.15, halo=8, omega=0.5, min_jacc=0.3):
        """KVShare-style baseline (request granularity): keep the contextualized
        pre-RoPE KV of the last `apc_cap` FULL prompts; align the new prompt
        against the most similar one (token Jaccard pre-filter + difflib), reuse
        matched tokens (re-rotated), recompute unmatched tokens + halo, then the
        same layer-1 selective recompute. No chunk-level views, no semantic index."""
        ids = list(self.pb.prefix_ids)
        for p in req.passages:
            ids += self.pb.passage_ids(p.text)
        ids += self.pb.suffix_ids(req.question)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sids = set(ids)
        best, bj = None, -1.0
        for key, (pids, _, _) in self.rq.items():
            sp = set(pids)
            j = len(sids & sp) / max(1, len(sids | sp))
            if j > bj:
                best, bj = key, j
        extra = {"reused_tok": 0, "delta_tok": len(ids), "rq_hit": False}
        pr = None
        if best is not None and bj >= min_jacc:
            pids, Kc, Vc = self.rq[best]
            sm = difflib.SequenceMatcher(None, pids, ids, autojunk=False)
            src = torch.full((len(ids),), -1, dtype=torch.long)
            for a, b, n in sm.get_matching_blocks():
                if n:
                    src[b:b + n] = torch.arange(a, a + n)
            ov = float((src >= 0).float().mean())
            if ov >= omega:
                if halo > 0:
                    bad = (src < 0).numpy().astype(np.int8)
                    near = np.convolve(bad, np.ones(2 * halo + 1, dtype=np.int8), mode="same") > 0
                    src[torch.from_numpy(near)] = -1
                src[-1] = -1
                self.rq.move_to_end(best)
                segs = [(ids, ChunkKV(pids, Kc, Vc), src)]
                _, mask, Ks, Vs = self._assemble(segs)
                if mask is not None:
                    pr = self.e.prefill(ids, mask, Ks, Vs, ratio, t_start=t0)
                    nr = int((src >= 0).sum())
                    extra = {"reused_tok": nr, "delta_tok": len(ids) - nr, "rq_hit": True}
        if pr is None:
            pr = self.e.prefill(ids, t_start=t0)
        rec = self._finish(req, "reqdiff", pr, t0, extra)
        self.rq[req.rid] = (ids, pr.pre_k, pr.v)
        while len(self.rq) > self.apc_cap:
            self.rq.popitem(last=False)
        return rec
