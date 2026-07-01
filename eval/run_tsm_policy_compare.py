#!/usr/bin/env python3
"""
Standalone TSM Policy Comparison - no external run_real_benchmark dependency.
Three policies: single_lru vs tiered_lru vs benefit
"""
import os
# Force A100 (GPU 1) - GPU 0 is RTX 4080 (16GB, not enough for 7B model)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys, json, time, logging, hashlib
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/semanticache"))

from core.hf_inference import QwenInferenceEngine
from core.system import SemantiCache
from core.tsm import StorageConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def gen_rag_requests(engine, n=500, seed=42):
    import random
    rng = random.Random(seed)
    documents = [
        "The transformer architecture uses self-attention to process sequences in parallel.",
        "Large language models are trained on massive text corpora using next-token prediction.",
        "KV Cache stores key-value pairs from attention layers to avoid recomputation.",
        "Retrieval-augmented generation combines LLMs with external knowledge retrieval.",
        "Multi-tenant serving allows multiple organizations to share the same model instance.",
        "Semantic similarity measures the closeness of two text snippets in meaning.",
        "GPU memory bandwidth is the primary bottleneck in large language model inference.",
        "Flash Attention reduces memory usage by computing attention in tiles.",
        "Quantization reduces model weight precision to reduce memory and increase throughput.",
        "Speculative decoding uses a small draft model to propose tokens verified by the large model.",
    ]
    templates = [
        "Please explain: {}", "What is {}? Provide a detailed explanation.",
        "Help me understand {}.", "Describe {} in simple terms.",
        "What do you know about {}?", "Please elaborate on {}.",
    ]
    texts = []
    for _ in range(n):
        doc = rng.choice(documents)
        tpl = rng.choice(templates)
        texts.append(tpl.format(doc[:50]))

    requests = []
    for i, text in enumerate(tqdm(texts, desc="Encoding")):
        input_ids = engine.tokenize(text)
        sem_vec = engine.embed(text)
        requests.append({
            "request_id": i, "text": text,
            "input_ids": input_ids, "semantic_vector": sem_vec,
            "tenant_id": f"tenant_{i % 4}",
        })
    return requests


def run_policy(engine, requests, policy_name, label,
               gpu_gb=0.5, cpu_gb=2.0, ssd_gb=20.0, tau=0.85, max_new=64):
    logger.info(f"\n{'='*60}\nPolicy: {label}\n{'='*60}")

    storage_config = StorageConfig(
        gpu_capacity_gb=gpu_gb,
        cpu_capacity_gb=cpu_gb,
        ssd_capacity_gb=ssd_gb,
        eviction_policy=policy_name,
    )
    system = SemantiCache(
        embedding_dim=128,
        similarity_threshold=tau,
        adaptive_qbr=True,
        storage_config=storage_config,
    )
    for i in range(4):
        system.til.register_tenant(f"tenant_{i}")

    exact_hits = semantic_hits = misses = 0
    ttfts = []
    hit_by_tier = {"l1": 0, "l2": 0, "l3": 0}

    for req in tqdm(requests, desc=f"[{label}]"):
        kv_payload, info = system.lookup(
            token_ids=req["input_ids"][0].tolist(),
            semantic_vector=req["semantic_vector"],
            tenant_id=req["tenant_id"],
        )
        if info["decision"] == "approved" and kv_payload is not None:
            text, decode_ttft = engine.decode_from_kv(
                kv_payload["kv_list"], kv_payload["last_token_id"], max_new)
            ttft = info["latency_ms"] + decode_ttft
            if info.get("storage_tier") in hit_by_tier:
                hit_by_tier[info["storage_tier"]] += 1
            if info["hit_type"] == "exact":
                exact_hits += 1
            else:
                semantic_hits += 1
        else:
            kv_list, text, prefill_ttft = engine.full_generate(req["input_ids"], max_new)
            ttft = info["latency_ms"] + prefill_ttft
            last_tok = req["input_ids"][0, -1].item()
            kv_size = engine.get_kv_size_gb(kv_list)
            system.store(
                token_ids=req["input_ids"][0].tolist(),
                semantic_vector=req["semantic_vector"],
                kv_data={"kv_list": kv_list, "last_token_id": last_tok},
                tenant_id=req["tenant_id"],
                kv_size_gb=kv_size,
            )
            misses += 1
        ttfts.append(ttft)

    n = len(requests)
    result = {
        "name": label, "policy": policy_name,
        "num_requests": n,
        "hit_rate": (exact_hits + semantic_hits) / n,
        "exact_hit_rate": exact_hits / n,
        "semantic_hit_rate": semantic_hits / n,
        "miss_rate": misses / n,
        "mean_ttft_ms": float(np.mean(ttfts)),
        "p99_ttft_ms": float(np.percentile(ttfts, 99)),
        "throughput_rps": 1000.0 / float(np.mean(ttfts)),
        "hit_by_tier": hit_by_tier,
    }
    logger.info(f"  hit={result['hit_rate']*100:.1f}%  TTFT={result['mean_ttft_ms']:.1f}ms  RPS={result['throughput_rps']:.1f}")
    system.close()
    return result


def main():
    logger.info("Loading model...")
    engine = QwenInferenceEngine(
        model_path=os.path.expanduser("~/models/qwen2.5-7b"),
        device="cuda",
    )
    engine.load()
    logger.info("Model loaded. Generating requests (n=500)...")
    requests = gen_rag_requests(engine, n=500, seed=42)

    policies = [
        ("single_lru", "Single-tier LRU"),
        ("tiered_lru", "Tiered LRU (warm eviction)"),
        ("benefit",    "Benefit-Density (theory policy)"),
    ]

    results = []
    for policy_name, label in policies:
        r = run_policy(engine, requests, policy_name, label,
                       gpu_gb=0.5, cpu_gb=2.0, ssd_gb=20.0)
        results.append(r)
        # Save incrementally so partial results survive crashes
        _out = os.path.expanduser("~/semanticache/results/tsm_policy_comparison.json")
        Path(_out).parent.mkdir(parents=True, exist_ok=True)
        with open(_out, "w") as _f:
            json.dump(results, _f, indent=2)
        print(f"  [Saved {len(results)} result(s) so far → {_out}]")

    out_path = os.path.expanduser("~/semanticache/results/tsm_policy_comparison.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_path}")
    print("\n=== FINAL SUMMARY ===")
    for r in results:
        print(f"{r['name']:40s}  hit={r['hit_rate']*100:.1f}%  exact={r['exact_hit_rate']*100:.1f}%  "
              f"sem={r['semantic_hit_rate']*100:.1f}%  TTFT={r['mean_ttft_ms']:.1f}ms")


if __name__ == "__main__":
    main()
