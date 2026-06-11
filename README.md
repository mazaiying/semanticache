# SemantiCache

**SemantiCache: Quality-Guaranteed, Storage-Efficient, and Multi-Tenant-Safe Semantic KV Cache Reuse in Production LLM Serving**

> Anonymous submission to ICDE 2027. Code is provided for reproducibility review.

---

## Overview

SemantiCache is a semantic-aware KV cache management system for multi-tenant LLM serving. It reuses KV cache blocks across **semantically equivalent but lexically different** prompts, providing:

- **Quality guarantee**: formal BERTScore lower bound via Quality-Bounded Reuse (QBR)
- **Storage scalability**: cost-model-driven tiered storage (GPU HBM → CPU DRAM → NVMe SSD)
- **Multi-tenant privacy**: structural zero-leakage isolation via Tenant Isolation Layer (TIL)

### Key Results

| Metric | Exact Cache (vLLM) | **SemantiCache** | Improvement |
|--------|:-----------------:|:-----------------:|:-----------:|
| Cache Hit Rate (RAG) | 88.0% | **96.7%** | +8.7 pp |
| TTFT | 34.67 ms | **30.58 ms** | −11.8% |
| Throughput (RPS) | 28.84 | **32.70** | +13.4% |
| BERTScore | 1.000 | **≥ 0.858** | guaranteed |
| Cross-tenant leakage | 0% | **0%** | ✓ |

---

## Repository Structure

```
semanticache/
├── core/                        # Core system components
│   ├── system.py                # SemantiCache main orchestrator
│   ├── hsi.py                   # Hierarchical Semantic Index (HSI)
│   ├── qbr.py                   # Quality-Bounded Reuse (QBR) gate
│   ├── tsm.py                   # Tiered Storage Manager (TSM)
│   ├── til.py                   # Tenant Isolation Layer (TIL)
│   └── hf_inference.py          # HuggingFace LLM inference backend
├── eval/                        # Evaluation scripts
│   ├── run_benchmark.py         # Main benchmark (RAG Synthetic workload)
│   ├── run_real_benchmark.py    # LMSYS Real-World workload
│   ├── run_overhead_microbench.py  # Overhead breakdown (Fig. 7)
│   ├── run_tsm_stress.py        # TSM scalability stress test
│   ├── run_all_experiments.sh   # One-command full reproduction
│   └── data_loader.py           # Dataset loading utilities
├── configs/                     # YAML experiment configurations
│   └── default.yaml
├── plot_results.py              # Figure generation (reproduces all paper figures)
└── requirements.txt
```

---

## Environment Setup

### Requirements

- Python 3.10+
- CUDA 12.1+, PyTorch 2.2+
- 2× NVIDIA A100 80GB GPUs (for full reproduction; single GPU OK for small-scale tests)

### Installation

```bash
# 1. Create conda environment
conda create -n semanticache python=3.10 -y
conda activate semanticache

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install flash-attention for faster prefill
pip install flash-attn --no-build-isolation
```

### Model Download

```bash
# Download Qwen2.5-7B-Instruct (used in all paper experiments)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

---

## Quick Start

```bash
# Run the main RAG benchmark (reproduces Table 2 / Fig. 3)
CUDA_VISIBLE_DEVICES=0,1 python eval/run_benchmark.py \
    --config configs/default.yaml \
    --output results/rag_benchmark.json

# Run LMSYS real-world benchmark (reproduces Fig. 3 right panel)
CUDA_VISIBLE_DEVICES=0,1 python eval/run_real_benchmark.py \
    --config configs/default.yaml \
    --output results/lmsys_benchmark.json

# Run overhead microbenchmark (reproduces Fig. 7)
CUDA_VISIBLE_DEVICES=0 python eval/run_overhead_microbench.py \
    --output results/overhead_breakdown.json
```

---

## Full Reproduction

To reproduce all figures and tables from the paper with a single command:

```bash
bash eval/run_all_experiments.sh
python plot_results.py
```

Output figures are saved to `results/figures/`. The script runs all experiments sequentially and takes approximately 2–4 hours on dual A100 GPUs.

### Experiment → Figure Mapping

| Script | Paper Figure/Table |
|--------|-------------------|
| `run_benchmark.py` | Table 2, Fig. 3 (hit rate & TTFT) |
| `run_real_benchmark.py` | Fig. 3 (LMSYS panel) |
| `run_benchmark.py --tau-sweep` | Fig. 4 (τ vs. quality/latency) |
| `run_benchmark.py --qps-sweep` | Fig. 5 (scalability) |
| `run_benchmark.py --ablation` | Fig. 6 (ablation study) |
| `run_overhead_microbench.py` | Fig. 7 (overhead breakdown) |

---

## Configuration

Key parameters in `configs/default.yaml`:

```yaml
model:
  name: Qwen/Qwen2.5-7B-Instruct
  device: cuda

semanticache:
  tau: 0.85              # QBR similarity threshold (trade-off: quality vs. hit rate)
  lsh_bands: 20          # LSH bands (higher = more recall, more memory)
  lsh_rows: 4            # LSH rows per band
  tsm_l1_gb: 10.0        # GPU HBM budget for KV cache (GB)
  tsm_l2_gb: 40.0        # CPU DRAM budget (GB)
  tsm_l3_path: /tmp/kvcache  # NVMe SSD path for L3 tier

benchmark:
  n_requests: 1000       # Number of requests
  qps: 5                 # Queries per second
  workload: rag_synthetic  # Options: rag_synthetic | lmsys_real
```

---

## Core Component API

```python
from core.system import SemantiCacheSystem

# Initialize the system
system = SemantiCacheSystem.from_config("configs/default.yaml")

# Process a request
response = system.serve(
    prompt="What are the side effects of ibuprofen?",
    tenant_id="tenant_A"
)

# Check cache statistics
stats = system.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Mean TTFT: {stats['mean_ttft_ms']:.2f} ms")
```

---

## Baselines

The following baselines are implemented for comparison (see Table 2):

| Baseline | Description |
|----------|-------------|
| `No Cache` | Standard LLM inference without any caching |
| `Exact Cache` | vLLM-style Radix-tree exact prefix caching |
| `SemShareKV` | LSH-based semantic cache without quality guarantees |
| **SemantiCache** | Our full system (QBR + HSI + TSM + TIL) |

---

## Citation

```bibtex
@inproceedings{semanticache2027,
  title     = {SemantiCache: Quality-Guaranteed, Storage-Efficient, and
               Multi-Tenant-Safe Semantic KV Cache Reuse in Production LLM Serving},
  booktitle = {Proceedings of the 43rd IEEE International Conference on
               Data Engineering (ICDE)},
  year      = {2027},
  note      = {Anonymous submission}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
