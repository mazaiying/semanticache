"""
HuggingFace Transformers Inference Engine for SemantiCache
==========================================================

Integrates Qwen2.5-7B (or any CausalLM) with SemantiCache by:
  1. Separating prefill and decode phases
  2. Injecting cached past_key_values on semantic/exact cache hits
     → skips prefill entirely, directly starts decoding
  3. Measuring true TTFT (time-to-first-token) for each path

Compatible with CUDA 11.5+. No vllm required.

Cache hit flow:
  lookup SemantiCache → hit → load past_kv from TSM → inject → decode only
                      → miss → full prefill → store past_kv → decode

TTFT definition:
  - Cache hit:  time to load KV tensors + 1 decode step   (fast)
  - Cache miss: time for full prefill + 1 decode step      (baseline)
"""

import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class QwenInferenceEngine:
    """
    Inference engine wrapping Qwen2.5-7B-Instruct (or any HF CausalLM).

    Key methods
    -----------
    load()
        Load model + tokenizer into GPU memory.
    full_generate(input_ids, max_new_tokens)
        Run prefill + decode. Returns (past_key_values, output_text, ttft_ms).
        Used for cache miss path and "No Cache" baseline.
    decode_from_kv(past_key_values, max_new_tokens)
        Decode using an injected KV cache, skipping prefill.
        Returns (output_text, ttft_ms).
        Used for exact / semantic cache hit path.
    tokenize(text)
        Returns input_ids tensor [1, seq_len] on CPU.
    embed(text)
        Returns unit-norm semantic embedding (numpy float32, dim=128).
        Used to query / populate the HSI.
    kv_to_numpy(past_key_values)
        Serialize past_key_values to a numpy array for TSM storage.
    numpy_to_kv(array, meta)
        Deserialize numpy array back to past_key_values tuple.
    get_kv_size_gb(past_key_values)
        Returns the storage size of a KV cache in GB.
    """

    def __init__(
        self,
        model_path: str = "~/models/qwen2.5-7b",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 2048,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 128,
    ):
        self.model_path = str(Path(model_path).expanduser())
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._embedder = None
        self._proj_matrix: Optional[np.ndarray] = None  # for dim reduction

    # ──────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────

    def load(self) -> None:
        """Load Qwen model and tokenizer. Call once before benchmarking."""
        logger.info(f"Loading tokenizer from {self.model_path} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model (dtype={self.dtype}, device={self.device}) …")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Model ready.")

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call engine.load() first.")

    def _synchronize(self) -> None:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

    # ──────────────────────────────────────────────────────
    # Core inference paths
    # ──────────────────────────────────────────────────────

    def full_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> Tuple[list, str, float]:
        """
        Full prefill + decode (cache miss / No Cache baseline).

        Parameters
        ----------
        input_ids : [1, seq_len] tensor (CPU or GPU)
        max_new_tokens : int

        Returns
        -------
        kv_list    : List[Tuple[Tensor, Tensor]] — packed KV (CPU, cloned)
                     Use decode_from_kv() to inject this on a cache hit.
        output_text : str
        ttft_ms    : float — time to first token in milliseconds
        """
        self._ensure_loaded()
        input_ids = input_ids.to(self.device)

        # ── Prefill ──────────────────────────────────────
        self._synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            prefill_out = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )

        # First generated token (completes TTFT measurement)
        first_logits = prefill_out.logits[:, -1, :]
        first_token = torch.argmax(first_logits, dim=-1, keepdim=True)
        self._synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000

        # Snapshot KV after TTFT so cache admission does not inflate prefill
        # latency, but before decode can extend DynamicCache in place.
        kv_list = self._pack_kv(prefill_out.past_key_values)

        # ── Decode ───────────────────────────────────────
        generated_ids = [first_token.item()]
        cur_past_kv = prefill_out.past_key_values
        cur_input = first_token

        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                if cur_input.item() == self.tokenizer.eos_token_id:
                    break
                out = self.model(
                    input_ids=cur_input,
                    past_key_values=cur_past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(next_tok.item())
                cur_past_kv = out.past_key_values
                cur_input = next_tok

        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return kv_list, output_text, ttft_ms

    def decode_from_kv(
        self,
        kv_list: list,
        last_prompt_token_id: int,
        max_new_tokens: int = 128,
    ) -> Tuple[str, float]:
        """
        Decode from a packed KV list (cache-hit path), skipping prefill.

        Parameters
        ----------
        kv_list              : List[Tuple[Tensor, Tensor]] from full_generate or _pack_kv
        last_prompt_token_id : the last token ID of the cached prompt (decode start)
        max_new_tokens       : int

        Returns
        -------
        output_text : str
        ttft_ms     : float — time for first decode step (no prefill)
        """
        self._ensure_loaded()

        # Hit TTFT includes restoration/upload from the selected storage tier.
        self._synchronize()
        t0 = time.perf_counter()
        past_kv = self._unpack_kv(kv_list, self.device, self.dtype)

        start_token = torch.tensor(
            [[last_prompt_token_id]],
            dtype=torch.long,
            device=self.device,
        )

        # ── First decode step = TTFT ──────────────────────
        with torch.no_grad():
            out = self.model(
                input_ids=start_token,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
        first_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        self._synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000

        # ── Continue decoding ─────────────────────────────
        generated_ids = [first_token.item()]
        cur_past_kv = out.past_key_values
        cur_input = first_token

        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                if cur_input.item() == self.tokenizer.eos_token_id:
                    break
                out = self.model(
                    input_ids=cur_input,
                    past_key_values=cur_past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(next_tok.item())
                cur_past_kv = out.past_key_values
                cur_input = next_tok

        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text, ttft_ms

    # ──────────────────────────────────────────────────────
    # Tokenizer helpers
    # ──────────────────────────────────────────────────────

    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text → input_ids [1, seq_len] (CPU).

        Applies chat template if available (Qwen-Instruct expects it).
        """
        self._ensure_loaded()
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": text}]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                enc = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_len,
                )
                return enc["input_ids"]
            except Exception:
                pass  # fall through to plain tokenize
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )
        return enc["input_ids"]

    # ──────────────────────────────────────────────────────
    # Semantic embedding (for HSI)
    # ──────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """
        Return a unit-norm semantic embedding of shape (embedding_dim,).
        Used to query the Hierarchical Semantic Index (HSI).
        """
        embedder = self._get_embedder()
        raw = embedder.encode(text, normalize_embeddings=True)  # (768,)
        proj = self._get_proj()
        reduced = proj @ raw  # (128,)
        norm = np.linalg.norm(reduced)
        return (reduced / (norm + 1e-8)).astype(np.float32)

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedder: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _get_proj(self) -> np.ndarray:
        """Fixed random projection matrix for dim reduction (reproducible)."""
        if self._proj_matrix is None:
            rng = np.random.default_rng(42)
            raw_dim = self._get_embedder().get_sentence_embedding_dimension()
            P = rng.normal(0, 1, (self.embedding_dim, raw_dim)).astype(np.float32)
            self._proj_matrix = P / np.linalg.norm(P, axis=1, keepdims=True)
        return self._proj_matrix

    # ──────────────────────────────────────────────────────
    # KV packing / unpacking (stable storage format)
    # ──────────────────────────────────────────────────────

    def _pack_kv(self, past_key_values: Any) -> list:
        """
        Pack past_key_values into a stable CPU list of (key, value) pairs.
        Always clones and detaches to prevent mutation from the decode loop.

        Handles:
          - DynamicCache / HybridCache (transformers >= 4.38): has .key_cache
          - Legacy tuple-of-tuples: each layer may have >= 2 tensors; take [0],[1]
        """
        # ── DynamicCache / HybridCache ─────────────────────────────────────
        if hasattr(past_key_values, 'key_cache'):
            return [
                (k.detach().cpu().clone(), v.detach().cpu().clone())
                for k, v in zip(past_key_values.key_cache,
                                past_key_values.value_cache)
                if k is not None
            ]

        # ── Legacy tuple-of-tuples ─────────────────────────────────────────
        # Use index access to avoid "too many values to unpack" on GQA/SWA layers
        result = []
        for layer in past_key_values:
            k = layer[0]
            v = layer[1]
            if k is not None:
                result.append((k.detach().cpu().clone(), v.detach().cpu().clone()))
        return result

    def _unpack_kv(self, kv_list: list, device: str, dtype: torch.dtype) -> Any:
        """
        Restore a proper DynamicCache from a packed list.
        Uses DynamicCache.update() which sets _seen_tokens correctly.

        Parameters
        ----------
        kv_list : List[Tuple[Tensor, Tensor]] from _pack_kv()
        device  : target device string (e.g. 'cuda')
        dtype   : target dtype (e.g. torch.bfloat16)

        Returns
        -------
        DynamicCache ready to pass to the model as past_key_values
        """
        from transformers import DynamicCache
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv_list):
            k_dev = k.to(device=device, dtype=dtype)
            v_dev = v.to(device=device, dtype=dtype)
            cache.update(k_dev, v_dev, layer_idx)
        return cache

    # ── Legacy serialization helpers (for TSM on-disk storage) ───────────

    def kv_to_numpy(
        self, kv_list: list
    ) -> Tuple[np.ndarray, Dict]:
        """
        Serialize a packed kv_list to a flat numpy array for disk/SSD storage.
        Returns (array, meta) where meta holds shape info.
        """
        arrays, shapes = [], []
        for k, v in kv_list:
            k_np = k.float().numpy()
            v_np = v.float().numpy()
            arrays.extend([k_np.flatten(), v_np.flatten()])
            shapes.append(list(k_np.shape))
        array = np.concatenate(arrays)
        meta  = {"shapes": shapes, "num_layers": len(shapes),
                 "dtype": str(self.dtype)}
        return array, meta

    def numpy_to_kv(
        self, array: np.ndarray, meta: Dict
    ) -> list:
        """
        Deserialize a flat numpy array back to a packed kv_list (CPU).
        """
        shapes, num_layers = meta["shapes"], meta["num_layers"]
        kv_list, offset = [], 0
        for shape in shapes:
            size = int(np.prod(shape))
            k = torch.tensor(array[offset: offset+size].reshape(shape),
                             dtype=self.dtype)
            offset += size
            v = torch.tensor(array[offset: offset+size].reshape(shape),
                             dtype=self.dtype)
            offset += size
            kv_list.append((k, v))
        return kv_list

    def get_kv_size_gb(self, kv_list: list) -> float:
        """Return total storage size of a packed kv_list in GB."""
        total = sum(
            k.nelement() * k.element_size() + v.nelement() * v.element_size()
            for k, v in kv_list
        )
        return total / (1024 ** 3)

    # ── (removed) _kv_to_device — replaced by _pack_kv / _unpack_kv ─────
    # Kept as a no-op alias for backward compat in case benchmark calls it.
    def _kv_to_device(self, past_key_values: Any, device: str) -> Any:
        """Alias: pack to CPU list then immediately unpack to target device."""
        kv_list = self._pack_kv(past_key_values)
        return self._unpack_kv(kv_list, device, self.dtype)
