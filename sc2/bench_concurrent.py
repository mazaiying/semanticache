"""
E4: open-loop serving benchmark on ONE GPU (real wall-clock, FCFS single worker).

Requests arrive as a Poisson process at rate `lambda` (req/s).  A single worker
executes them one at a time (no continuous batching -- same engine/kernels for every
method).  Latency = finish_time - arrival_time = queueing delay + service time.
This measures how much load each method sustains before latency explodes; it does NOT
claim anything about multi-GPU or batched serving.

Example:
  python bench_concurrent.py --out runs/q7_conc --methods full apc cacheblend semanticache \
      --rates 0.4 0.8 1.2 1.6 2.0 --n_req 150
Outputs: results.json (per method/rate raw latencies + summary), summary.txt
"""
import argparse, json, os, random, sys, time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import KVEngine
from methods import Runner, SemanticIndex, TieredKVStore
from run_exp import run_method
from workload import PromptBuilder, build_requests, load_longbench


def pct(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/qwen2.5-7b-instruct"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/sc2/data/longbench/data"))
    ap.add_argument("--n_samples", type=int, default=60)
    ap.add_argument("--max_ctx", type=int, default=4096)
    ap.add_argument("--n_req", type=int, default=150, help="requests per (method, rate) run")
    ap.add_argument("--methods", nargs="+", default=["full", "apc", "cacheblend", "semanticache"])
    ap.add_argument("--rates", nargs="+", type=float, default=[0.4, 0.8, 1.2, 1.6, 2.0])
    ap.add_argument("--ratio", type=float, default=0.15)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--tau_q", type=float, default=0.9)
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--halo", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--gpu_budget_gb", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    eng = KVEngine(a.model, a.device).load()
    samples = load_longbench(os.path.join(a.data_dir, f"{a.dataset}.jsonl"))
    reqs, corpus = build_requests(samples, a.dataset, eng.tok, a.max_ctx, a.n_samples, a.seed)
    pb = PromptBuilder(eng.tok)
    idx = SemanticIndex(device=a.device)
    idx.build(list(corpus.keys()), [p.text for p in corpus.values()])
    store = TieredKVStore(eng, a.gpu_budget_gb)
    t0 = time.time()
    for pid, p in corpus.items():
        store.ensure(pid, pb.passage_ids(p.text))
    print(f"requests={len(reqs)} chunks={len(corpus)} ingest={time.time()-t0:.0f}s "
          f"footprint={store.footprint()}", flush=True)

    rng = random.Random(a.seed)
    order = list(range(len(reqs)))
    results, flat = {}, []
    for m in a.methods:
        runner = Runner(eng, pb, corpus, idx, store, a.max_new_tokens)
        for w in reqs[:3]:
            runner.full(w)
        # closed-loop saturation throughput (back-to-back)
        rng.shuffle(order)
        sel = [reqs[i] for i in order[:min(a.n_req, len(reqs))]]
        torch.cuda.synchronize(); tb = time.perf_counter()
        for rq in sel:
            run_method(m, runner, rq, a)
        torch.cuda.synchronize()
        sat = len(sel) / (time.perf_counter() - tb)
        results[f"{m}|closed_loop"] = {"throughput_rps": sat, "n": len(sel)}
        print(f"[{m}] closed-loop max throughput {sat:.3f} req/s", flush=True)
        for lam in a.rates:
            rng.shuffle(order)
            sel = [reqs[i] for i in order[:min(a.n_req, len(reqs))]]
            gaps = np.random.RandomState(a.seed + int(lam * 1000)).exponential(1.0 / lam, len(sel))
            arr = np.cumsum(gaps)
            lat, qd, svc, kinds = [], [], [], []
            torch.cuda.synchronize(); t_start = time.perf_counter()
            for at, rq in zip(arr, sel):
                now = time.perf_counter() - t_start
                if now < at:
                    time.sleep(at - now)
                s = time.perf_counter() - t_start
                run_method(m, runner, rq, a)
                torch.cuda.synchronize()
                f = time.perf_counter() - t_start
                lat.append((f - at) * 1000); qd.append((s - at) * 1000)
                svc.append((f - s) * 1000); kinds.append(rq.kind)
            wall = time.perf_counter() - t_start
            d = {"rate": lam, "n": len(sel), "achieved_rps": len(sel) / wall,
                 "lat_mean_ms": float(np.mean(lat)), "lat_p50_ms": pct(lat, 50),
                 "lat_p95_ms": pct(lat, 95), "lat_p99_ms": pct(lat, 99),
                 "queue_mean_ms": float(np.mean(qd)), "service_mean_ms": float(np.mean(svc)),
                 "lat": lat, "kinds": kinds}
            results[f"{m}|{lam}"] = d
            flat.append((m, lam, d))
            print(f"[{m}] rate={lam}: p50={d['lat_p50_ms']:.0f} p95={d['lat_p95_ms']:.0f} "
                  f"p99={d['lat_p99_ms']:.0f} ms  queue={d['queue_mean_ms']:.0f} "
                  f"service={d['service_mean_ms']:.0f}  achieved={d['achieved_rps']:.2f} rps", flush=True)
            json.dump(results, open(os.path.join(a.out, "results.json"), "w"))
    lines = ["method        rate  achieved  lat_p50  lat_p95  lat_p99  queue_mean  service_mean"]
    for m, lam, d in flat:
        lines.append(f"{m:<13}{lam:>5.2f}{d['achieved_rps']:>10.2f}{d['lat_p50_ms']:>9.0f}"
                     f"{d['lat_p95_ms']:>9.0f}{d['lat_p99_ms']:>9.0f}{d['queue_mean_ms']:>12.0f}"
                     f"{d['service_mean_ms']:>14.0f}")
    lines.append("")
    for m in a.methods:
        lines.append(f"closed-loop max throughput {m}: {results[m + '|closed_loop']['throughput_rps']:.3f} req/s")
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
