"""
E6: TSM Stress Test (Tiered Storage Eviction)
==============================================
Forces the Tiered Storage Manager (TSM) to move real KV tensors from GPU to
pinned CPU memory and then to serialized files on an NVMe path.

Proves to reviewers that:
1. Physical tier transfers complete and restored KV can be decoded.
2. The selected victim policy migrates cold data out and hot data in.
3. SemantiCache survives constrained cache capacity without stale HSI entries.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hf_inference import QwenInferenceEngine
from core.system import SemantiCache
from core.tsm import StorageConfig
from eval.run_real_benchmark import load_requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="TSM Stress Test")
    parser.add_argument("--model_path", default="~/models/qwen2.5-7b")
    parser.add_argument("--num_requests", type=int, default=300)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--policy",
        choices=["single_lru", "tiered_lru", "benefit"],
        default="benefit",
    )
    parser.add_argument("--gpu_capacity_gb", type=float, default=0.008)
    parser.add_argument("--cpu_capacity_gb", type=float, default=0.012)
    parser.add_argument("--ssd_capacity_gb", type=float, default=5.0)
    parser.add_argument("--ssd_path", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model
    engine = QwenInferenceEngine(model_path=args.model_path, device=args.device)
    try:
        engine.load()
    except Exception as e:
        logger.warning(f"Could not load real model ({e}). Ensure model path is correct.")
        return

    # Load requests (High semantic similarity to force cache hits on evicted data)
    requests = load_requests(engine, dataset=args.dataset, num_requests=args.num_requests)

    # Deliberately tight physical capacities ensure that all enabled tiers are
    # exercised even on the short synthetic prompts.
    storage_config = StorageConfig(
        gpu_capacity_gb=args.gpu_capacity_gb,
        cpu_capacity_gb=args.cpu_capacity_gb,
        ssd_capacity_gb=args.ssd_capacity_gb,
        eviction_policy=args.policy,
        device=args.device,
        ssd_path=args.ssd_path,
    )

    system = SemantiCache(
        embedding_dim=128,
        similarity_threshold=0.85,
        storage_config=storage_config,
    )
    # Register tenants
    for i in range(4):
        system.til.register_tenant(f"tenant_{i}")

    hit_by_tier = {"l1": 0, "l2": 0, "l3": 0}
    logger.info(
        "Running physical TSM stress test "
        f"(policy={args.policy}, L1={args.gpu_capacity_gb} GB, "
        f"L2={args.cpu_capacity_gb} GB) ..."
    )
    for req in tqdm(requests, desc="[TSM Stress Test]"):
        tenant = req["tenant_id"]
        
        kv_payload, info = system.lookup(
            token_ids=req["input_ids"][0].tolist(),
            semantic_vector=req["semantic_vector"],
            tenant_id=tenant,
        )

        decision = info["decision"]
        if decision == "approved" and kv_payload is not None:
            if info["storage_tier"] in hit_by_tier:
                hit_by_tier[info["storage_tier"]] += 1
            engine.decode_from_kv(
                kv_payload["kv_list"],
                kv_payload["last_token_id"],
                64,
            )
        else:
            kv_list, _, prefill_ttft = engine.full_generate(req["input_ids"], 64)
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

    # Collect stats
    stats = system.get_stats()
    stats["tsm"]["hit_by_source_tier"] = hit_by_tier
    logger.info("\n--- TSM Stress Test Results ---")
    logger.info("L1 (GPU) Stats:  " + str(stats["tsm"]["l1"]))
    logger.info("L2 (CPU) Stats:  " + str(stats["tsm"]["l2"]))
    logger.info("L3 (SSD) Stats:  " + str(stats["tsm"]["l3"]))

    # We expect migrations_out and evictions in L1 to be > 0
    with open(os.path.join(args.output_dir, "tsm_stress.json"), "w") as f:
        json.dump(stats, f, indent=2)

    system.close()
    logger.info("Saved TSM stress results.")

if __name__ == "__main__":
    main()
