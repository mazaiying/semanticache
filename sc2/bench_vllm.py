"""
Anchor our HF-based KVEngine against real vLLM on the SAME prompts.

Measures, per request, the latency of a 1-token generation (≈ TTFT, sequential,
batch size 1) in vLLM with prefix caching OFF and ON. Purpose: show that the
in-house `full` / `apc` baselines are in a realistic range (not strawmen), and
report the absolute gap honestly.

Usage (GPU 1 only, nothing else running):
  CUDA_VISIBLE_DEVICES=1 python bench_vllm.py --model ~/models/qwen2.5-7b-instruct \
      --dataset hotpotqa --n_samples 60 --out ~/sc2/runs/q8_vllm_anchor
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workload import PromptBuilder, build_requests, load_longbench


def prompts_for(reqs, pb):
    out = []
    for r in reqs:
        ids = list(pb.prefix_ids)
        for p in r.passages:
            ids += pb.passage_ids(p.text)
        ids += pb.suffix_ids(r.question)
        out.append((r.rid, r.kind, ids))
    return out


def run(llm, prompts, SamplingParams, TokensPrompt):
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    for _, _, ids in prompts[:3]:  # warm-up
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
    res = []
    for rid, kind, ids in prompts:
        t = time.perf_counter()
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        res.append({"rid": rid, "kind": kind, "n_tokens": len(ids),
                    "ttft_ms": (time.perf_counter() - t) * 1000})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/qwen2.5-7b-instruct"))
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/sc2/data/longbench/data"))
    ap.add_argument("--n_samples", type=int, default=60)
    ap.add_argument("--max_ctx", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_mem", type=float, default=0.6)
    ap.add_argument("--mode", choices=["full", "apc"], required=True,
                    help="run each mode in a SEPARATE process (vLLM does not free GPU memory reliably)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        from vllm import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    samples = load_longbench(os.path.join(a.data_dir, f"{a.dataset}.jsonl"))
    reqs, _ = build_requests(samples, a.dataset, tok, a.max_ctx, a.n_samples, a.seed)
    prompts = prompts_for(reqs, PromptBuilder(tok))
    print(f"requests={len(prompts)}", flush=True)

    summary = {}
    for apc in ([False] if a.mode == "full" else [True]):
        tag = "vllm_apc" if apc else "vllm_full"
        llm = LLM(model=a.model, dtype="bfloat16", enable_prefix_caching=apc,
                  gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_ctx * 2 + 512,
                  enforce_eager=False, seed=0)
        res = run(llm, prompts, SamplingParams, TokensPrompt)
        with open(os.path.join(a.out, f"{tag}.jsonl"), "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        by = {}
        for r in res:
            by.setdefault(r["kind"], []).append(r["ttft_ms"])
            by.setdefault("ALL", []).append(r["ttft_ms"])
        summary[tag] = {k: {"n": len(v), "ttft_ms": float(np.mean(v)),
                            "ttft_p95_ms": float(np.percentile(v, 95))} for k, v in by.items()}
        print(tag, json.dumps(summary[tag]["ALL"]), flush=True)
        del llm
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)
    json.dump(summary, open(os.path.join(a.out, f"summary_{a.mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
