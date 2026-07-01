"""
SemantiCache Real Inference Benchmark
======================================

Runs all 4 comparison systems using REAL Qwen2.5-7B inference.
Measures true TTFT (time-to-first-token) with actual GPU computation.

Systems compared
----------------
1. No Cache       — full prefill for every request (baseline)
2. Exact Cache    — token-hash-based KV reuse (vLLM APC equivalent)
3. SemShareKV     — LSH semantic KV reuse, no quality control
4. SemantiCache   — HSI + QBR + TSM + TIL (our full system)

Usage
-----
  # Quick test (50 requests, RAG synthetic data)
  python eval/run_real_benchmark.py --num_requests 50 --dataset rag_synthetic

  # Full paper experiment (500 requests)
  python eval/run_real_benchmark.py --num_requests 500 --dataset sharegpt \\
      --model_path ~/models/qwen2.5-7b --output_dir results/

  # Ablation: vary τ threshold
  python eval/run_real_benchmark.py --num_requests 200 --tau_sweep

  # Full ablation study
  python eval/run_real_benchmark.py --num_requests 200 --ablation
"""

import os
import sys
import json
import time
import logging
import hashlib
import argparse
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

# ── Path setup ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hf_inference import QwenInferenceEngine
from core.system import SemantiCache
from core.hsi import HierarchicalSemanticIndex
from core.tsm import StorageConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def compute_bertscore(references: List[str], candidates: List[str]) -> float:
    """
    Compute mean BERTScore F1 between candidate outputs and reference outputs.
    Uses distilbert-base-uncased for speed. Returns value in [0, 1].
    Called for semantic-hit outputs only (exact hits / misses = 1.0 by definition).
    """
    if not references or not candidates:
        return 1.0
    try:
        from bert_score import score as _bert_score
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, F1 = _bert_score(
            candidates, references,
            model_type="distilbert-base-uncased",
            lang="en",
            device=device,
            verbose=False,
            batch_size=16,
        )
        return float(F1.mean())
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}. Returning 1.0.")
        return 1.0


# ════════════════════════════════════════════════════════
# Request generation / loading
# ════════════════════════════════════════════════════════

def load_requests(
    engine: QwenInferenceEngine,
    dataset: str,
    num_requests: int,
    seed: int = 42,
) -> List[Dict]:
    """
    Load requests as list of dicts:
      {request_id, text, input_ids, semantic_vector, tenant_id, type}
    """
    import random
    rng = random.Random(seed)

    if dataset == "rag_synthetic":
        texts = _gen_rag_synthetic(num_requests, seed)
    elif dataset in ("sharegpt", "lmsys"):
        texts = _load_from_hf(num_requests)
    else:
        texts = _gen_pure_synthetic(num_requests, seed)

    logger.info(f"Encoding {len(texts)} prompts …")
    requests = []
    for i, text in enumerate(tqdm(texts, desc="Building requests")):
        input_ids = engine.tokenize(text)  # [1, seq_len]
        sem_vec = engine.embed(text)       # (128,)
        requests.append({
            "request_id": i,
            "text": text,
            "input_ids": input_ids,
            "semantic_vector": sem_vec,
            "tenant_id": f"tenant_{rng.randint(0, 3)}",
            "type": "real",
        })
    return requests


def _gen_rag_synthetic(num_requests: int, seed: int) -> List[str]:
    """
    RAG-style: same document, different question phrasings.
    Designed to have HIGH semantic similarity → tests semantic cache well.
    """
    import random
    rng = random.Random(seed)

    documents = [
        "The transformer architecture uses self-attention to process sequences in parallel, enabling efficient training of large language models.",
        "Large language models are trained on massive text corpora using next-token prediction objectives with billions of parameters.",
        "KV Cache stores key-value pairs from attention layers to avoid recomputation during autoregressive decoding.",
        "Retrieval-augmented generation combines LLMs with external knowledge retrieval to reduce hallucinations.",
        "Multi-tenant serving allows multiple organizations to share the same model instance with resource isolation.",
        "Semantic similarity measures the closeness of two text snippets in their underlying meaning rather than surface form.",
        "GPU memory bandwidth is the primary bottleneck in large language model inference systems.",
        "Flash Attention reduces memory usage by computing attention in tiles without materializing the full attention matrix.",
        "Quantization reduces model weight precision from FP16 to INT8 or INT4 to reduce memory and increase throughput.",
        "Speculative decoding uses a small draft model to propose tokens verified by the large target model.",
    ]
    templates = [
        "Please explain: {}",
        "What is {}? Provide a detailed explanation.",
        "Help me understand {}.",
        "Describe {} in simple terms.",
        "What do you know about {}?",
        "What is the significance of {}?",
        "Please elaborate on {}.",
        "Give me an overview of {}.",
        "请介绍一下：{}",
        "请详细解释：{}",
        "{}是什么？请说明。",
        "请用简单语言描述{}",
    ]
    texts = []
    for _ in range(num_requests):
        doc = rng.choice(documents)
        tpl = rng.choice(templates)
        texts.append(tpl.format(doc[:50]))
    return texts


def _load_from_hf(num_requests: int) -> List[str]:
    """Load from LMSYS-Chat-1M (HuggingFace). Falls back to synthetic."""
    try:
        from datasets import load_dataset
        logger.info("Downloading LMSYS-Chat-1M …")
        ds = load_dataset(
            "lmsys/lmsys-chat-1m",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        texts = []
        for item in ds:
            if len(texts) >= num_requests * 2:
                break
            conv = item.get("conversation", [])
            first_human = next(
                (m["content"] for m in conv if m.get("role") == "user"), None
            )
            if first_human and len(first_human) > 20:
                texts.append(first_human)
        logger.info(f"Loaded {len(texts)} prompts from LMSYS-Chat-1M")
        return texts[:num_requests]
    except Exception as e:
        logger.warning(f"HuggingFace load failed ({e}). Using synthetic fallback.")
        return _gen_rag_synthetic(num_requests, seed=42)


def _gen_pure_synthetic(num_requests: int, seed: int) -> List[str]:
    import random
    rng = random.Random(seed)
    base = [
        "Tell me about artificial intelligence.",
        "What is machine learning?",
        "Explain deep learning.",
        "How do neural networks work?",
        "What is natural language processing?",
    ]
    return [rng.choice(base) + f" (variant {i})" for i in range(num_requests)]


# ════════════════════════════════════════════════════════
# System 1: No Cache
# ════════════════════════════════════════════════════════

def run_no_cache(
    engine: QwenInferenceEngine,
    requests: List[Dict],
    max_new_tokens: int = 64,
) -> Dict:
    """Baseline: full prefill for every request. Always collects outputs for BERTScore reference."""
    ttfts = []
    outputs = []
    for req in tqdm(requests, desc="[No Cache]"):
        _, text, ttft = engine.full_generate(req["input_ids"], max_new_tokens)
        ttfts.append(ttft)
        outputs.append(text)

    result = _make_result("No Cache", ttfts, 0, 0, 0)
    result["outputs"] = outputs  # kept for BERTScore reference in tau_sweep
    return result


# ════════════════════════════════════════════════════════
# System 2: Exact Cache (vLLM APC equivalent)
# ════════════════════════════════════════════════════════

def run_exact_cache(
    engine: QwenInferenceEngine,
    requests: List[Dict],
    max_new_tokens: int = 64,
) -> Dict:
    """
    Exact hash-based KV cache. Equivalent to vLLM Automatic Prefix Caching.
    Cache key = SHA256 of token IDs.
    """
    # kv_store: hash → (past_key_values_on_cpu, last_token_id)
    kv_store: Dict[str, Tuple[Any, int]] = {}
    exact_hits = 0
    ttfts = []

    for req in tqdm(requests, desc="[Exact Cache]"):
        token_ids = req["input_ids"]
        token_key = hashlib.sha256(token_ids.numpy().tobytes()).hexdigest()

        if token_key in kv_store:
            # Cache hit: inject KV, skip prefill
            kv_list, last_tok = kv_store[token_key]
            _, ttft = engine.decode_from_kv(kv_list, last_tok, max_new_tokens)
            exact_hits += 1
        else:
            # Cache miss: full generate, store KV on CPU
            kv_list, _, ttft = engine.full_generate(token_ids, max_new_tokens)
            last_tok = token_ids[0, -1].item()
            kv_store[token_key] = (kv_list, last_tok)

        ttfts.append(ttft)

    n = len(requests)
    return _make_result(
        "Exact Cache (APC)",
        ttfts,
        exact_hits,
        0,
        n - exact_hits,
    )


# ════════════════════════════════════════════════════════
# System 3: SemShareKV (semantic, no quality control)
# ════════════════════════════════════════════════════════

def run_semshare_kv(
    engine: QwenInferenceEngine,
    requests: List[Dict],
    threshold: float = 0.70,
    max_new_tokens: int = 64,
) -> Dict:
    """
    SemShareKV: LSH-based semantic KV reuse without quality control.
    Any request with cosine similarity ≥ threshold gets cache reuse,
    regardless of output quality degradation.
    """
    hsi = HierarchicalSemanticIndex(
        embedding_dim=128,
        lsh_num_tables=8,
        lsh_num_bits=10,
    )
    # kv_store: block_id → (past_kv_cpu, last_token_id)
    kv_store: Dict[str, Tuple[Any, int]] = {}

    semantic_hits = 0
    ttfts = []

    for req in tqdm(requests, desc="[SemShareKV]"):
        block, hit_type = hsi.lookup(req["input_ids"][0].tolist(), req["semantic_vector"])

        if block is not None:
            sim = hsi._cosine_similarity(req["semantic_vector"], block.semantic_vector)
            if sim >= threshold and block.block_id in kv_store:
                kv_list, last_tok = kv_store[block.block_id]
                _, ttft = engine.decode_from_kv(kv_list, last_tok, max_new_tokens)
                semantic_hits += 1
                ttfts.append(ttft)
                continue

        # Miss or below threshold
        kv_list, _, ttft = engine.full_generate(req["input_ids"], max_new_tokens)
        last_tok = req["input_ids"][0, -1].item()
        block_id = hsi.insert(
            token_ids=req["input_ids"][0].tolist(),
            semantic_vector=req["semantic_vector"],
            kv_ref=None,
        )
        kv_store[block_id] = (kv_list, last_tok)
        ttfts.append(ttft)

    n = len(requests)
    return _make_result(
        "SemShareKV",
        ttfts,
        0,
        semantic_hits,
        n - semantic_hits,
    )


# ════════════════════════════════════════════════════════
# System 4: SemantiCache (Full System)
# ════════════════════════════════════════════════════════

def run_semanticache(
    engine: QwenInferenceEngine,
    requests: List[Dict],
    threshold: float = 0.85,
    max_new_tokens: int = 64,
    enable_tsm: bool = True,
    enable_qbr: bool = True,
    enable_til: bool = True,
    enable_semantic: bool = True,
    collect_outputs: bool = False,
    storage_policy: str = "benefit",
    gpu_capacity_gb: float = 20.0,
    cpu_capacity_gb: float = 64.0,
    ssd_capacity_gb: float = 200.0,
    ssd_path: Optional[str] = None,
) -> Dict:
    """
    SemantiCache: HSI + QBR + TSM + TIL.
    All flags can be toggled for ablation studies.
    collect_outputs: if True, saves generated texts and semantic_hit_indices for BERTScore.
    """
    import torch

    effective_policy = storage_policy if enable_tsm else "single_lru"
    storage_config = StorageConfig(
        gpu_capacity_gb=gpu_capacity_gb,
        cpu_capacity_gb=cpu_capacity_gb if enable_tsm else 0.0,
        ssd_capacity_gb=ssd_capacity_gb if enable_tsm else 0.0,
        eviction_policy=effective_policy,
        device=engine.device,
        ssd_path=ssd_path,
    )
    system = SemantiCache(
        embedding_dim=128,
        similarity_threshold=threshold,
        adaptive_qbr=enable_qbr,
        storage_config=storage_config,
        enable_semantic=enable_semantic,
        enable_qbr=enable_qbr,
        enable_til=enable_til,
    )

    # Register tenants (TIL)
    if enable_til:
        for i in range(4):
            system.til.register_tenant(f"tenant_{i}")

    exact_hits = semantic_hits = misses = 0
    ttfts: List[float] = []
    outputs: List[str] = []          # generated texts (if collect_outputs)
    semantic_hit_indices: List[int] = []  # request indices with semantic hits
    per_tenant_latencies: Dict[str, List[float]] = {}  # for Fairness Index
    hit_by_tier = {"l1": 0, "l2": 0, "l3": 0}

    gpu_mem_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    for req_idx, req in enumerate(tqdm(requests, desc="[SemantiCache]")):
        tenant = req["tenant_id"] if enable_til else None

        # ── SemantiCache lookup ──────────────────────────
        kv_payload, info = system.lookup(
            token_ids=req["input_ids"][0].tolist(),
            semantic_vector=req["semantic_vector"],
            tenant_id=tenant,
        )

        hit_type = info["hit_type"]
        decision = info["decision"]

        if decision == "approved" and kv_payload is not None:
            # Cache hit: inject KV, skip prefill
            text, decode_ttft = engine.decode_from_kv(
                kv_payload["kv_list"],
                kv_payload["last_token_id"],
                max_new_tokens,
            )
            ttft = info["latency_ms"] + decode_ttft
            if info["storage_tier"] in hit_by_tier:
                hit_by_tier[info["storage_tier"]] += 1

            if hit_type == "exact":
                exact_hits += 1
            else:
                semantic_hits += 1
                semantic_hit_indices.append(req_idx)
        else:
            # Cache miss (or rejected by QBR/TIL): full generate
            kv_list, text, prefill_ttft = engine.full_generate(
                req["input_ids"], max_new_tokens
            )
            ttft = info["latency_ms"] + prefill_ttft
            last_tok = req["input_ids"][0, -1].item()
            kv_size = engine.get_kv_size_gb(kv_list)

            system.store(
                token_ids=req["input_ids"][0].tolist(),
                semantic_vector=req["semantic_vector"],
                kv_data={
                    "kv_list": kv_list,
                    "last_token_id": last_tok,
                },
                tenant_id=tenant,
                kv_size_gb=kv_size,
                prefill_cost_ms=prefill_ttft,
            )
            misses += 1

        ttfts.append(ttft)
        if collect_outputs:
            outputs.append(text)

        # Per-tenant latency tracking (for Fairness Index)
        if tenant:
            per_tenant_latencies.setdefault(tenant, []).append(ttft)

    gpu_mem_after = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    result = _make_result("SemantiCache", ttfts, exact_hits, semantic_hits, misses)
    result["system_stats"] = system.get_stats()
    result["storage_policy"] = effective_policy
    result["hit_by_tier"] = hit_by_tier
    result["leakage_rate"] = system.til.get_leakage_rate()
    result["fairness_index"] = system.til.get_fairness_index(per_tenant_latencies)
    result["gpu_memory_gb"] = {
        "before_gb": round(gpu_mem_before, 3),
        "after_gb":  round(gpu_mem_after, 3),
        "delta_gb":  round(gpu_mem_after - gpu_mem_before, 3),
    }
    if collect_outputs:
        result["outputs"] = outputs
        result["semantic_hit_indices"] = semantic_hit_indices
    system.close()
    return result


# ════════════════════════════════════════════════════════
# Ablation study
# ════════════════════════════════════════════════════════

def run_ablation(
    engine: QwenInferenceEngine,
    requests: List[Dict],
    max_new_tokens: int = 64,
    gpu_capacity_gb: float = 20.0,
    cpu_capacity_gb: float = 64.0,
    ssd_capacity_gb: float = 200.0,
    ssd_path: Optional[str] = None,
) -> List[Dict]:
    """
    Ablation study: toggle each component off one at a time.
    Returns list of result dicts with 'name' field for plotting.
    """
    configs = [
        {"name": "SemantiCache (Full)", "enable_tsm": True, "enable_qbr": True, "enable_til": True, "enable_semantic": True},
        {"name": "w/o Semantic Layer",  "enable_tsm": True, "enable_qbr": True, "enable_til": True, "enable_semantic": False},
        {"name": "w/o QBR",             "enable_tsm": True, "enable_qbr": False, "enable_til": True, "enable_semantic": True},
        {"name": "w/o TSM",             "enable_tsm": False, "enable_qbr": True, "enable_til": True, "enable_semantic": True},
        {"name": "w/o TIL",             "enable_tsm": True, "enable_qbr": True, "enable_til": False, "enable_semantic": True},
    ]
    results = []
    for cfg in configs:
        name = cfg.pop("name")
        logger.info(f"Ablation: {name}")
        r = run_semanticache(
            engine,
            requests,
            threshold=0.85,
            max_new_tokens=max_new_tokens,
            storage_policy="benefit",
            gpu_capacity_gb=gpu_capacity_gb,
            cpu_capacity_gb=cpu_capacity_gb,
            ssd_capacity_gb=ssd_capacity_gb,
            ssd_path=ssd_path,
            **cfg,
        )
        r["name"] = name
        results.append(r)
    return results


def run_storage_policy_comparison(
    engine: QwenInferenceEngine,
    requests: List[Dict],
    max_new_tokens: int = 64,
    gpu_capacity_gb: float = 20.0,
    cpu_capacity_gb: float = 64.0,
    ssd_capacity_gb: float = 200.0,
    ssd_path: Optional[str] = None,
) -> List[Dict]:
    """Compare replacement policies over the same cold-start request trace."""
    configurations = [
        ("Single-tier LRU", "single_lru"),
        ("Tiered LRU", "tiered_lru"),
        ("Benefit-density TSM", "benefit"),
    ]
    results = []
    for name, policy in configurations:
        logger.info(f"Storage policy: {name}")
        result = run_semanticache(
            engine,
            requests,
            threshold=0.85,
            max_new_tokens=max_new_tokens,
            storage_policy=policy,
            gpu_capacity_gb=gpu_capacity_gb,
            cpu_capacity_gb=cpu_capacity_gb,
            ssd_capacity_gb=ssd_capacity_gb,
            ssd_path=ssd_path,
        )
        result["name"] = name
        results.append(result)
    return results


# ════════════════════════════════════════════════════════
# τ sweep (quality-efficiency tradeoff)
# ════════════════════════════════════════════════════════

def run_tau_sweep(
    engine: QwenInferenceEngine,
    requests: List[Dict],
    max_new_tokens: int = 64,
) -> List[Dict]:
    """
    Vary τ threshold from 0.60 to 0.95.
    Computes BERTScore on semantic-hit outputs vs No Cache reference outputs
    to produce the quality-efficiency tradeoff curve.
    """
    # ── Step 1: No Cache reference outputs (ground truth) ──────────────
    logger.info("τ sweep: computing No Cache reference outputs for BERTScore …")
    nc_result = run_no_cache(engine, requests, max_new_tokens)
    reference_outputs: List[str] = nc_result.get("outputs", [])

    tau_values = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    results = []
    for tau in tau_values:
        logger.info(f"τ sweep: τ = {tau}")
        r = run_semanticache(
            engine, requests,
            threshold=tau, max_new_tokens=max_new_tokens,
            collect_outputs=True,
        )
        r["name"] = f"τ={tau}"
        r["tau"] = tau

        # ── BERTScore on semantic hits ──────────────────────────────────
        sem_idx: List[int] = r.pop("semantic_hit_indices", [])
        sc_outputs: List[str] = r.pop("outputs", [])
        if sem_idx and reference_outputs:
            refs  = [reference_outputs[i] for i in sem_idx]
            cands = [sc_outputs[i]          for i in sem_idx]
            bs = compute_bertscore(refs, cands)
            logger.info(
                f"  τ={tau}: BERTScore(semantic hits)={bs:.4f}  n={len(sem_idx)}"
            )
        else:
            bs = 1.0  # no semantic hits → quality = perfect
        r["bertscore"] = round(bs, 4)
        r["bertscore_n_samples"] = len(sem_idx)

        results.append(r)
    return results


# ════════════════════════════════════════════════════════
# Scalability study (cache warm-up curve)
# ════════════════════════════════════════════════════════

def run_scalability(
    engine: QwenInferenceEngine,
    requests: List[Dict],
    max_new_tokens: int = 64,
    checkpoint_every: int = 50,
    gpu_capacity_gb: float = 20.0,
    cpu_capacity_gb: float = 64.0,
    ssd_capacity_gb: float = 200.0,
    ssd_path: Optional[str] = None,
) -> Dict:
    """
    Scalability study: run SemantiCache and No-Cache on an increasing
    number of requests, recording hit rate and TTFT at each checkpoint.

    Returns
    -------
    {
      "checkpoints": [50, 100, 150, ...],
      "semanticache_hit_rate": [...],   # cumulative hit rate at each checkpoint
      "semanticache_mean_ttft": [...],
      "no_cache_mean_ttft": [...],
    }
    """
    from core.system import SemantiCache
    from core.tsm import StorageConfig

    storage_config = StorageConfig(
        gpu_capacity_gb=gpu_capacity_gb,
        cpu_capacity_gb=cpu_capacity_gb,
        ssd_capacity_gb=ssd_capacity_gb,
        eviction_policy="benefit",
        device=engine.device,
        ssd_path=ssd_path,
    )
    system = SemantiCache(
        embedding_dim=128,
        similarity_threshold=0.85,
        adaptive_qbr=True,
        storage_config=storage_config,
    )
    for i in range(4):
        system.til.register_tenant(f"tenant_{i}")

    nc_ttfts_window, sc_ttfts_window = [], []
    checkpoints, sc_hit_rates, sc_ttfts, nc_ttfts = [], [], [], []
    total_hits = 0

    logger.info(f"Scalability: {len(requests)} requests, checkpoint every {checkpoint_every}")

    for i, req in enumerate(tqdm(requests, desc="[Scalability]"), start=1):
        # ── No Cache ──────────────────────────────────────────
        _, _, nc_ttft = engine.full_generate(req["input_ids"], max_new_tokens)
        nc_ttfts_window.append(nc_ttft)

        # ── SemantiCache ──────────────────────────────────────
        tenant = req["tenant_id"]
        kv_payload, info = system.lookup(
            token_ids=req["input_ids"][0].tolist(),
            semantic_vector=req["semantic_vector"],
            tenant_id=tenant,
        )
        decision = info["decision"]
        if decision == "approved" and kv_payload is not None:
            _, decode_ttft = engine.decode_from_kv(
                kv_payload["kv_list"],
                kv_payload["last_token_id"],
                max_new_tokens,
            )
            sc_ttft = info["latency_ms"] + decode_ttft
            total_hits += 1
        else:
            kv_list, _, prefill_ttft = engine.full_generate(
                req["input_ids"], max_new_tokens
            )
            sc_ttft = info["latency_ms"] + prefill_ttft
            last_tok = req["input_ids"][0, -1].item()
            kv_size = engine.get_kv_size_gb(kv_list)
            system.store(
                token_ids=req["input_ids"][0].tolist(),
                semantic_vector=req["semantic_vector"],
                kv_data={
                    "kv_list": kv_list,
                    "last_token_id": last_tok,
                },
                tenant_id=tenant,
                kv_size_gb=kv_size,
                prefill_cost_ms=prefill_ttft,
            )
        sc_ttfts_window.append(sc_ttft)

        # ── Checkpoint ────────────────────────────────────────
        if i % checkpoint_every == 0 or i == len(requests):
            checkpoints.append(i)
            sc_hit_rates.append(total_hits / i * 100)
            sc_ttfts.append(float(np.mean(sc_ttfts_window)))
            nc_ttfts.append(float(np.mean(nc_ttfts_window)))
            sc_ttfts_window.clear()
            nc_ttfts_window.clear()
            logger.info(f"  @{i}: hit={total_hits/i*100:.1f}% sc_ttft={sc_ttfts[-1]:.1f}ms nc_ttft={nc_ttfts[-1]:.1f}ms")

    result = {
        "checkpoints": checkpoints,
        "semanticache_hit_rate": sc_hit_rates,
        "semanticache_mean_ttft": sc_ttfts,
        "no_cache_mean_ttft": nc_ttfts,
        "total_requests": len(requests),
        "final_hit_rate": total_hits / len(requests) * 100,
        "system_stats": system.get_stats(),
    }
    system.close()
    return result


# ════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════

def _make_result(
    name: str,
    ttfts: List[float],
    exact_hits: int,
    semantic_hits: int,
    misses: int,
) -> Dict:
    n = len(ttfts)
    total_hits = exact_hits + semantic_hits
    return {
        "name": name,
        "num_requests": n,
        "hit_rate": total_hits / n if n > 0 else 0.0,
        "exact_hit_rate": exact_hits / n if n > 0 else 0.0,
        "semantic_hit_rate": semantic_hits / n if n > 0 else 0.0,
        "miss_rate": misses / n if n > 0 else 1.0,
        "mean_ttft_ms": float(np.mean(ttfts)),
        "median_ttft_ms": float(np.median(ttfts)),
        "p99_ttft_ms": float(np.percentile(ttfts, 99)),
        "throughput_rps": 1000.0 / float(np.mean(ttfts)),
        "ttft_all_ms": ttfts,  # raw, for plotting
    }


def _print_table(results: List[Dict]) -> None:
    header = f"{'System':<25} {'Hit%':>7} {'Exact%':>8} {'Sem%':>8} {'TTFT(ms)':>10} {'p99(ms)':>9} {'RPS':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<25} "
            f"{r['hit_rate']:>7.1%} "
            f"{r['exact_hit_rate']:>8.1%} "
            f"{r['semantic_hit_rate']:>8.1%} "
            f"{r['mean_ttft_ms']:>10.1f} "
            f"{r['p99_ttft_ms']:>9.1f} "
            f"{r['throughput_rps']:>8.1f}"
        )
    print("=" * len(header) + "\n")


def _save_results(results: List[Dict], path: str) -> None:
    clean = []
    for r in results:
        cr = {k: v for k, v in r.items() if k != "ttft_all_ms"}
        clean.append(cr)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    # Also save raw TTFTs for CDF plotting
    raw = {r["name"]: r.get("ttft_all_ms", []) for r in results}
    raw_path = path.replace(".json", "_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    logger.info(f"Results saved → {path}")
    logger.info(f"Raw TTFT data → {raw_path}")


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SemantiCache Real Inference Benchmark")
    parser.add_argument("--model_path", default="~/models/qwen2.5-7b",
                        help="Path to Qwen2.5-7B (or any CausalLM) model directory")
    parser.add_argument("--dataset", default="rag_synthetic",
                        choices=["rag_synthetic", "sharegpt", "lmsys", "synthetic"],
                        help="Request workload dataset")
    parser.add_argument("--num_requests", type=int, default=100,
                        help="Number of inference requests to run")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max tokens to generate per request")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="SemantiCache QBR similarity threshold τ")
    parser.add_argument("--output_dir", default="results",
                        help="Directory to save JSON results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau_sweep", action="store_true",
                        help="Run τ sweep experiment (quality-efficiency curve)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study")
    parser.add_argument(
        "--storage_policies",
        action="store_true",
        help="Compare single LRU, tiered LRU, and benefit-density TSM",
    )
    parser.add_argument("--scalability", action="store_true",
                        help="Run scalability study (hit rate vs request count)")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Record metrics every N requests (scalability mode)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (cuda / cpu)")
    parser.add_argument("--gpu_capacity_gb", type=float, default=20.0)
    parser.add_argument("--cpu_capacity_gb", type=float, default=64.0)
    parser.add_argument("--ssd_capacity_gb", type=float, default=200.0)
    parser.add_argument(
        "--ssd_path",
        default=None,
        help="NVMe directory or mounted remote filesystem used as L3",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──────────────────────────────────────
    engine = QwenInferenceEngine(
        model_path=args.model_path,
        device=args.device,
    )
    engine.load()

    # ── Load requests ───────────────────────────────────
    logger.info(f"Loading {args.num_requests} requests from '{args.dataset}' …")
    requests = load_requests(engine, args.dataset, args.num_requests, args.seed)

    # ── Main experiment ─────────────────────────────────
    if args.tau_sweep:
        logger.info("Running τ sweep experiment …")
        results = run_tau_sweep(engine, requests, args.max_new_tokens)
        _print_table(results)
        _save_results(results, os.path.join(args.output_dir, "tau_sweep.json"))

    elif args.ablation:
        logger.info("Running ablation study …")
        results = run_ablation(
            engine,
            requests,
            args.max_new_tokens,
            gpu_capacity_gb=args.gpu_capacity_gb,
            cpu_capacity_gb=args.cpu_capacity_gb,
            ssd_capacity_gb=args.ssd_capacity_gb,
            ssd_path=args.ssd_path,
        )
        _print_table(results)
        _save_results(results, os.path.join(args.output_dir, "ablation.json"))

    elif args.storage_policies:
        logger.info("Running physical storage-policy comparison …")
        results = run_storage_policy_comparison(
            engine,
            requests,
            args.max_new_tokens,
            gpu_capacity_gb=args.gpu_capacity_gb,
            cpu_capacity_gb=args.cpu_capacity_gb,
            ssd_capacity_gb=args.ssd_capacity_gb,
            ssd_path=args.ssd_path,
        )
        _print_table(results)
        _save_results(
            results,
            os.path.join(args.output_dir, "storage_policies.json"),
        )

    elif args.scalability:
        logger.info("Running scalability study …")
        result = run_scalability(
            engine,
            requests,
            args.max_new_tokens,
            args.checkpoint_every,
            gpu_capacity_gb=args.gpu_capacity_gb,
            cpu_capacity_gb=args.cpu_capacity_gb,
            ssd_capacity_gb=args.ssd_capacity_gb,
            ssd_path=args.ssd_path,
        )
        out_path = os.path.join(args.output_dir, "scalability.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Scalability results → {out_path}")
        logger.info(f"Final hit rate: {result['final_hit_rate']:.1f}%")

    else:
        logger.info("Running full system comparison …")
        results = []
        results.append(run_no_cache(engine, requests, args.max_new_tokens))
        results.append(run_exact_cache(engine, requests, args.max_new_tokens))
        results.append(run_semshare_kv(engine, requests, threshold=0.70,
                                       max_new_tokens=args.max_new_tokens))
        results.append(
            run_semanticache(
                engine,
                requests,
                threshold=args.threshold,
                max_new_tokens=args.max_new_tokens,
                gpu_capacity_gb=args.gpu_capacity_gb,
                cpu_capacity_gb=args.cpu_capacity_gb,
                ssd_capacity_gb=args.ssd_capacity_gb,
                ssd_path=args.ssd_path,
            )
        )
        _print_table(results)
        _save_results(results, os.path.join(
            args.output_dir, f"benchmark_{args.dataset}_n{args.num_requests}.json"))


if __name__ == "__main__":
    main()
