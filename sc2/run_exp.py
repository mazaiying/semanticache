"""
Main experiment driver for SemantiCache v2.

Example:
  python run_exp.py --model ~/models/qwen2.5-7b-instruct --dataset hotpotqa \
      --n_samples 60 --max_ctx 4096 --out runs/q7b_hotpot \
      --methods full apc gptcache cacheblend naive_sem semanticache

Outputs (in --out):
  records.jsonl   one json per (request, method)
  summary.json    aggregated table
  summary.txt     human-readable table
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import KVEngine
from methods import Runner, SemanticIndex, TieredKVStore
from workload import PromptBuilder, build_requests, load_longbench

ALL = ["full", "apc", "gptcache", "cacheblend", "cacheblend_r0",
       "naive_sem", "semanticache", "semanticache_r0", "semanticache_halo0",
       "semanticache_always", "reqdiff"]


def run_method(name, runner, req, a):
    if name == "full":
        return runner.full(req)
    if name == "apc":
        return runner.apc_method(req)
    if name == "gptcache":
        return runner.gptcache(req, tau_q=a.tau_q)
    if name == "cacheblend":
        return runner.chunked(req, "cacheblend", ratio=a.ratio, tag=name)
    if name == "cacheblend_r0":
        return runner.chunked(req, "cacheblend", ratio=0.0, tag=name)
    if name == "naive_sem":
        return runner.chunked(req, "naive_sem", ratio=0.0, tau=a.tau, tag=name)
    if name == "semanticache":
        return runner.chunked(req, "semanticache", ratio=a.ratio, tau=a.tau, omega=a.omega, halo=a.halo, tag=name)
    if name == "semanticache_halo0":
        return runner.chunked(req, "semanticache", ratio=a.ratio, tau=a.tau, omega=a.omega, halo=0, tag=name)
    if name == "semanticache_r0":
        return runner.chunked(req, "semanticache", ratio=0.0, tau=a.tau, omega=a.omega, halo=a.halo, tag=name)
    if name == "semanticache_always":   # never fall back to recompute (omega = 0)
        return runner.chunked(req, "semanticache", ratio=a.ratio, tau=0.0, omega=0.0, halo=a.halo, tag=name)
    if name == "reqdiff":
        return runner.reqdiff(req, ratio=a.ratio, halo=a.halo, omega=a.omega)
    raise ValueError(name)


def load_distractors(data_dir, exclude_texts, n, seed=0):
    """Distractor passages for the semantic-index scalability test: real passages
    from every LongBench file in data_dir (not in the materialized corpus); if
    fewer than n exist, the rest are synthesized by sentence recombination.
    Distractors are indexed (embedding only) but have NO materialized KV."""
    import glob
    import random
    import re
    from workload import split_passages
    rng = random.Random(seed)
    excl = set(exclude_texts)
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        for line in open(f):
            try:
                ctx = json.loads(line).get("context", "")
            except Exception:
                continue
            parts = split_passages(ctx)
            if len(parts) < 2:
                w = ctx.split()
                parts = [" ".join(w[i:i + 200]) for i in range(0, len(w), 200)]
            out.extend(p for p in parts if len(p) > 200 and p not in excl)
    rng.shuffle(out)
    n_real = min(n, len(out))
    out = out[:n_real]
    sents = [s for p in out[:20000] for s in re.split(r"(?<=[.!?])\s+", p) if len(s) > 20]
    while len(out) < n and sents:
        out.append(" ".join(rng.choice(sents) for _ in range(8)))
    return out, n_real


def summarize(recs):
    by = defaultdict(list)
    for r in recs:
        by[(r["method"], r["kind"])].append(r)
        by[(r["method"], "ALL")].append(r)
    out = {}
    for (m, k), rs in by.items():
        d = {
            "n": len(rs),
            "f1": float(np.mean([x["f1"] for x in rs])),
            "ttft_ms": float(np.mean([x["ttft_ms"] for x in rs])),
            "ttft_p95_ms": float(np.percentile([x["ttft_ms"] for x in rs], 95)),
            "e2e_ms": float(np.mean([x["e2e_ms"] for x in rs])),
        }
        tot = sum(x["n_tokens"] for x in rs)
        if tot:
            d["compute_frac"] = sum(x["n_computed"] for x in rs) / tot
        if rs and "f1_old_answer" in rs[0]:
            d["stale_f1"] = float(np.mean([x["f1_old_answer"] for x in rs]))
        for key in ("exact", "semantic", "miss", "reused_tok", "delta_tok"):
            if key in rs[0]:
                d[key] = float(np.mean([x[key] for x in rs]))
        for key in ("t_lookup", "t_align", "t_fetch", "t_assemble", "t_plan", "t_prefill"):
            if key in rs[0]:
                d[key] = float(np.mean([x[key] for x in rs]))
        if rs and "sem_lookup" in rs[0]:
            nl = sum(x["sem_lookup"] for x in rs)
            d["sem_lookup"] = nl
            d["sem_precision"] = (sum(x["sem_correct"] for x in rs) / nl) if nl else None
            d["sem_distractor_rate"] = (sum(x["sem_distractor"] for x in rs) / nl) if nl else None
        if rs and "rq_hit" in rs[0]:
            d["rq_hit_rate"] = float(np.mean([bool(x["rq_hit"]) for x in rs]))
        out[f"{m}|{k}"] = d
    return out


def fmt(summary, methods):
    kinds = ["orig", "reorder", "format", "edit", "drift", "ALL"]
    lines = []
    for metric, title in (("f1", "QA F1 (higher better)"), ("ttft_ms", "TTFT ms (lower better)"),
                          ("e2e_ms", "End-to-end ms")):
        lines.append(f"\n== {title} ==")
        lines.append(f"{'method':<18}" + "".join(f"{k:>10}" for k in kinds))
        for m in methods:
            row = f"{m:<18}"
            for k in kinds:
                v = summary.get(f"{m}|{k}", {}).get(metric)
                row += f"{v:>10.3f}" if v is not None else f"{'-':>10}"
            lines.append(row)
    lines.append("\n== edit requests: stale-answer F1 (lower better) ==")
    for m in methods:
        v = summary.get(f"{m}|edit", {}).get("stale_f1")
        if v is not None:
            lines.append(f"{m:<18}{v:>10.3f}")
    lines.append("\n== fraction of tokens computed ==")
    for m in methods:
        v = summary.get(f"{m}|ALL", {}).get("compute_frac")
        if v is not None:
            lines.append(f"{m:<18}{v:>10.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/qwen2.5-7b-instruct"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/sc2/data/longbench/data"))
    ap.add_argument("--n_samples", type=int, default=60)
    ap.add_argument("--max_ctx", type=int, default=4096)
    ap.add_argument("--methods", nargs="+", default=["full", "apc", "gptcache", "cacheblend",
                                                       "naive_sem", "semanticache"])
    ap.add_argument("--ratio", type=float, default=0.15)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--tau_q", type=float, default=0.9)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--halo", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--gpu_budget_gb", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fmt_prob", type=float, default=0.5)
    ap.add_argument("--drift_frac", type=float, default=0.0)
    ap.add_argument("--kinds", nargs="+", default=None)
    ap.add_argument("--kv_quant", default="none", choices=["none", "int8", "fp8"])
    ap.add_argument("--n_distractors", type=int, default=0)
    ap.add_argument("--profile", action="store_true",
                    help="cuda-synchronize at stage boundaries for an exact latency breakdown")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    eng = KVEngine(a.model, a.device).load()
    samples = load_longbench(os.path.join(a.data_dir, f"{a.dataset}.jsonl"))
    reqs, corpus = build_requests(samples, a.dataset, eng.tok, a.max_ctx, a.n_samples, a.seed,
                                  fmt_prob=a.fmt_prob, drift_frac=a.drift_frac, kinds=a.kinds)
    print(f"requests={len(reqs)} base_passages={len(corpus)}", flush=True)
    pb = PromptBuilder(eng.tok)
    idx = SemanticIndex(device=a.device)
    keys, texts = list(corpus.keys()), [p.text for p in corpus.values()]
    if a.n_distractors > 0:
        dist, n_real = load_distractors(a.data_dir, texts, a.n_distractors, a.seed)
        keys += [f"__dist__{i}" for i in range(len(dist))]
        texts += dist
        print(f"distractors={len(dist)} (real={n_real})", flush=True)
    t_idx = time.time()
    idx.build(keys, texts)
    print(f"index built: {len(keys)} entries in {time.time() - t_idx:.1f}s", flush=True)
    store = TieredKVStore(eng, a.gpu_budget_gb, a.kv_quant)

    t0 = time.time()
    for pid, p in corpus.items():          # offline ingestion (materialized views)
        store.ensure(pid, pb.passage_ids(p.text))
    t_ingest = time.time() - t0
    print(f"ingested {len(corpus)} chunks in {t_ingest:.0f}s; "
          f"footprint={store.footprint()}", flush=True)

    recs = []
    rec_path = os.path.join(a.out, "records.jsonl")
    open(rec_path, "w").close()
    for m in a.methods:
        runner = Runner(eng, pb, corpus, idx, store, a.max_new_tokens)
        runner.profile = a.profile
        for w in reqs[:3]:                   # warm-up
            runner.full(w)
        tm = time.time()
        for i, rq in enumerate(reqs):
            rec = run_method(m, runner, rq, a)
            recs.append(rec)
            with open(rec_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if (i + 1) % 40 == 0:
                print(f"[{m}] {i + 1}/{len(reqs)} elapsed {time.time() - tm:.0f}s", flush=True)
        print(f"[{m}] done in {time.time() - tm:.0f}s", flush=True)

    summ = summarize(recs)
    summ["_store"] = store.footprint()
    summ["_store"].update({"kv_bytes_per_token": eng.kv_bytes_per_token(), **store.stats,
                           "ingest_s": t_ingest, "kv_quant": a.kv_quant,
                           "index_entries": len(keys)})
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    txt = fmt(summ, a.methods)
    open(os.path.join(a.out, "summary.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
