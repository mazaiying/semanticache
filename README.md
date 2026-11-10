# SemantiCache: Incremental Semantic KV View Maintenance for Long-Context RAG via Token-Delta Repair

This repository contains the core engine, workload builder, evaluation/profiling harness, completed experimental outputs (`results/`), consolidated statistical tables (`results_tables.md`), and figure generation scripts (`plot_paper.py`) for **SemantiCache**.

---

## Overview

Long-context retrieval-augmented generation (RAG) repeatedly prefills the same retrieved passages across requests, yet conventional prefix/chunk KV caches discard cached states whenever a passage is reordered, reformatted, or locally edited. Conversely, whole-chunk semantic substitution can silently reinstate stale facts when an answer-bearing span changes.

**SemantiCache** formulates reusable document KV states as **versioned semantic materialized views** (`(t(c), K_pre, V_val)`) and maintains them incrementally across three modules:

1. **Module I — Versioned Chunk-View Layout and Discovery**: Separates persistent passage identity, exact token digests (SHA-1), dense semantic candidate nomination (`--tau`, default `0.8`), and position-independent pre-RoPE key/value storage.
2. **Module II — Token-Delta Maintenance Pipeline**: Validates reusable tokens via `SequenceMatcher` alignment (`--omega`, default `0.5`), expands unmatched deltas with a bidirectional boundary halo (`--halo`, default `8`), rerotates reused pre-RoPE keys at target offsets, and selectively recomputes the top-`r` highest-deviation aligned tokens at Layer 1 (`--ratio`, default `0.15`).
3. **Module III — Cost-Aware Planning and Budgeted Fallback**: Enforces similarity and overlap profitability gates to revert heavily drifted chunks to full prefill, and manages a byte-budgeted GPU LRU tier (`--gpu_budget_gb`) backed by pinned CPU views in BF16 or per-(head, token) absmax `INT8`/`FP8` quantization (`--kv_quant`).

---

## Repository Structure

```text
semanticache/
├── sc2/                         # Core KV maintenance engine & benchmark drivers
│   ├── engine.py                # Pre-RoPE KV capture, partial KV assembly, Layer-1 selective repair
│   ├── methods.py               # TieredKVStore (BF16/INT8/FP8), SemanticIndex, Runner (all 11 methods)
│   ├── workload.py              # LongBench multi-doc QA loader & controlled reorder/format/edit/drift perturbations
│   ├── run_exp.py               # Main experiment driver (--tau, --omega, --halo, --ratio, --gpu_budget_gb, --kv_quant, --profile)
│   ├── bench_concurrent.py      # Open-loop Poisson arrival & closed-loop throughput benchmark
│   ├── bench_vllm.py            # Standalone vLLM comparison driver
│   ├── profile_engine.py        # Prefill phase micro-profiler
│   └── test_engine.py           # Engine invariant & unit smoke tests
├── queues/                      # Batch scripts for Q1–Q9 experiment suites
│   ├── common.sh                # Shared environment & resumable run() helper
│   ├── q1_main.sh               # Q1: Main 3 datasets (HotpotQA, 2WikiMQA, MuSiQue) on Qwen2.5-7B
│   ├── q2_sweeps.sh             # Q2: Sensitivity sweeps (ratio, tau, omega, halo)
│   ├── q3_models.sh             # Q3: Cross-model generalization (Qwen2.5-3B/14B, Llama-3.1-8B, Mistral-7B-v0.3)
│   ├── q4_ctx_seed.sh           # Q4: Context-length scaling (2K, 8K) & multi-seed HotpotQA runs
│   ├── q5_extra.sh              # Q5: 16K context scaling & multi-seed 2WikiMQA / MuSiQue / Qwen2.5-14B runs
│   ├── q6_mem.sh                # Q6: GPU-tier budget sweep (2–40 GB)
│   ├── q7_conc.sh               # Q7: Open-loop Poisson arrivals & closed-loop throughput
│   ├── q8_supp.sh               # Q8: Stage breakdown, reqdiff (KVShare-style), drift_frac, fmt_prob, low GPU budget, INT8/FP8, 200K index
│   └── q9_optional.sh           # Q9: 150-sample (618-view) corpus scale & extra seeds
├── analyze.py                   # Aggregates results/ + paired bootstrap 95% CIs -> results_tables.md
├── plot_paper.py                # Generates all paper figures (figures/*.pdf) directly from results/
├── results_tables.md            # Consolidated ground-truth tables (Sections A–N)
├── figures/                     # Generated PDF figures used in the paper
└── results/                     # Complete experimental outputs (summary.json, summary.txt, records.jsonl, args.json)
```

---

## Quick Verification (Reproducing Tables & Figures from Included Artifacts)

All raw per-request records (`results/*/records.jsonl`) and run summaries (`results/*/summary.json`) are included in `results/`. You can regenerate the consolidated markdown tables (with 2,000-resample paired request-level bootstrap 95% confidence intervals) and all paper PDF figures on any machine without a GPU:

```bash
pip install -r requirements.txt

# 1. Regenerate results_tables.md (Sections A–N)
python3 analyze.py

# 2. Regenerate all paper figures into figures/
python3 plot_paper.py
```

---

## Running Experiments on GPU

### 1. Environment & Unit Smoke Test

Experiments were conducted on a single NVIDIA A100-SXM4-80GB GPU with PyTorch (`bfloat16`) and HuggingFace Transformers.

```bash
# Run engine invariant & RoPE/assembly smoke test
python3 sc2/test_engine.py --model /path/to/qwen2.5-7b-instruct
```

### 2. Main Experiment Driver (`sc2/run_exp.py`)

```bash
python3 sc2/run_exp.py \
  --model /path/to/qwen2.5-7b-instruct \
  --dataset hotpotqa \
  --n_samples 100 \
  --max_ctx 4096 \
  --tau 0.8 \
  --omega 0.5 \
  --halo 8 \
  --ratio 0.15 \
  --gpu_budget_gb 20.0 \
  --kv_quant none \
  --methods full apc gptcache cacheblend_r0 cacheblend naive_sem semanticache_r0 semanticache_halo0 semanticache \
  --out results/q1_hotpotqa_q7b_s0
```

Supported `--methods`:
- `full`: Full prompt prefill
- `apc`: 16-token-block exact prefix caching (vLLM APC semantics)
- `gptcache`: Question-embedding response cache ($\tau_q = 0.9$)
- `cacheblend_r0`: Exact-chunk KV reuse with positional repair only ($r = 0$)
- `cacheblend`: CacheBlend-style exact-chunk KV reuse with Layer-1 selective recomputation ($r = 0.15$)
- `naive_sem`: Whole-chunk semantic KV substitution without token-delta repair
- `reqdiff`: KVShare-style full-request sequence alignment and selective recomputation
- `semanticache_r0`: SemantiCache without Layer-1 selective contextual repair ($r = 0$)
- `semanticache_halo0`: SemantiCache without boundary halo expansion ($h = 0$)
- `semanticache_always`: SemantiCache without the $\omega$ overlap fallback gate ($\omega = 0$)
- `semanticache`: Full SemantiCache pipeline ($\tau = 0.8, \omega = 0.5, h = 8, r = 0.15$)

### 3. Reproducing the Full Q1–Q9 Suites

Edit the model paths (`M7`, `M3`, `M14`, `ML`, `MM`) in `queues/common.sh` to point to your local model weights and LongBench JSONL directory (`data/longbench/data`), then run:

```bash
bash queues/q1_main.sh
bash queues/q2_sweeps.sh
bash queues/q3_models.sh
bash queues/q4_ctx_seed.sh
bash queues/q5_extra.sh
bash queues/q6_mem.sh
bash queues/q7_conc.sh
bash queues/q8_supp.sh
bash queues/q9_optional.sh
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
