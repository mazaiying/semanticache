#!/bin/bash
# =============================================================
#  SemantiCache — Full Experiment Suite
#  ICDE 2027 Paper: SemantiCache
#
#  Usage (on the server):
#    cd ~/semanticache
#    chmod +x eval/run_all_experiments.sh
#    nohup bash eval/run_all_experiments.sh > results/all_experiments.log 2>&1 &
#    echo "PID: $!"
#
#  Monitor:
#    tail -f results/all_experiments.log
#
#  Total estimated time: ~8-10 hours on A100 80GB
# =============================================================

set -e   # Exit on first error
set -o pipefail

# ── Config ────────────────────────────────────────────────
MODEL=~/models/qwen2.5-7b
DATASET=rag_synthetic
export CUDA_VISIBLE_DEVICES=1   # Use A100 (GPU index 1)
DEVICE=cuda
OUT=results
LOG=$OUT/all_experiments.log
SSD_PATH=${SSD_PATH:-$OUT/nvme_cache}

mkdir -p $OUT $OUT/figures

# ── Logging helper ────────────────────────────────────────
log() {
    echo ""
    echo "=========================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "=========================================================="
}

log_done() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ DONE: $1"
}

# ── Sanity check ──────────────────────────────────────────
log "SemantiCache Full Experiment Suite — Starting"
echo "  Model  : $MODEL"
echo "  Dataset: $DATASET"
echo "  GPU    : $CUDA_VISIBLE_DEVICES (CUDA_VISIBLE_DEVICES)"
echo "  L3 path: $SSD_PATH (place this directory on NVMe)"
echo "  Output : $OUT/"
echo ""

# Verify model exists
if [ ! -d "$HOME/models/qwen2.5-7b" ]; then
    echo "ERROR: Model not found at ~/models/qwen2.5-7b"
    exit 1
fi

# ──────────────────────────────────────────────────────────
# E1: τ Sensitivity Analysis (quality-efficiency tradeoff)
#     200 requests × 8 τ values = ~1600 SemantiCache inferences
#     Estimated: ~55 minutes
# ──────────────────────────────────────────────────────────
log "E1: τ Sensitivity Analysis (200 req × 8 τ values)"

python eval/run_real_benchmark.py \
    --model_path $MODEL \
    --dataset $DATASET \
    --num_requests 200 \
    --max_new_tokens 64 \
    --tau_sweep \
    --device $DEVICE \
    --output_dir $OUT

log_done "E1: τ sweep → $OUT/tau_sweep.json"

# ──────────────────────────────────────────────────────────
# E2: Ablation Study (component-wise contribution)
#     1000 requests × 5 configurations, identical to the main trace length
#     Estimated: ~3 hours
# ──────────────────────────────────────────────────────────
log "E2: Ablation Study (1000 req × 5 configs)"

python eval/run_real_benchmark.py \
    --model_path $MODEL \
    --dataset $DATASET \
    --num_requests 1000 \
    --max_new_tokens 64 \
    --ablation \
    --device $DEVICE \
    --ssd_path $SSD_PATH \
    --output_dir $OUT

log_done "E2: Ablation → $OUT/ablation.json"

# ──────────────────────────────────────────────────────────
# E3: Physical storage policy comparison
#     Tight physical budgets force GPU -> pinned CPU -> NVMe movement.
# ──────────────────────────────────────────────────────────
log "E3: Physical Storage Policies (1000 req × 3 policies)"

python eval/run_real_benchmark.py \
    --model_path $MODEL \
    --dataset $DATASET \
    --num_requests 1000 \
    --max_new_tokens 64 \
    --storage_policies \
    --gpu_capacity_gb 0.008 \
    --cpu_capacity_gb 0.012 \
    --ssd_capacity_gb 5.0 \
    --ssd_path $SSD_PATH \
    --device $DEVICE \
    --output_dir $OUT

log_done "E3: Storage policies → $OUT/storage_policies.json"

# ──────────────────────────────────────────────────────────
# E4: Direct physical tier stress test
# ──────────────────────────────────────────────────────────
log "E4: Physical TSM Stress Test (300 req)"

python eval/run_tsm_stress.py \
    --model_path $MODEL \
    --dataset synthetic \
    --num_requests 300 \
    --policy benefit \
    --gpu_capacity_gb 0.008 \
    --cpu_capacity_gb 0.012 \
    --ssd_capacity_gb 5.0 \
    --ssd_path $SSD_PATH \
    --device $DEVICE \
    --output_dir $OUT

log_done "E4: TSM stress → $OUT/tsm_stress.json"

# ──────────────────────────────────────────────────────────
# E5: Scalability Study (cache warm-up curve)
#     1000 requests, checkpoint every 50
#     Runs SemantiCache + No Cache in parallel per request
#     Estimated: ~67 minutes
# ──────────────────────────────────────────────────────────
log "E5: Scalability Study (1000 req, checkpoint=50)"

python eval/run_real_benchmark.py \
    --model_path $MODEL \
    --dataset $DATASET \
    --num_requests 1000 \
    --max_new_tokens 64 \
    --scalability \
    --checkpoint_every 50 \
    --device $DEVICE \
    --output_dir $OUT

log_done "E5: Scalability → $OUT/scalability.json"

# ──────────────────────────────────────────────────────────
# E6: Extended Main Comparison (1000 requests, 4 systems)
#     More statistically significant than 500-req run
#     Estimated: ~135 minutes
# ──────────────────────────────────────────────────────────
log "E6: Extended Main Comparison (1000 req, 4 systems)"

python eval/run_real_benchmark.py \
    --model_path $MODEL \
    --dataset $DATASET \
    --num_requests 1000 \
    --max_new_tokens 64 \
    --threshold 0.85 \
    --device $DEVICE \
    --ssd_path $SSD_PATH \
    --seed 123 \
    --output_dir $OUT

log_done "E6: Main comparison → $OUT/benchmark_${DATASET}_n1000.json"

# ──────────────────────────────────────────────────────────
# E7: Generate All Paper Figures
#     Uses all JSON results produced above
#     Estimated: ~2 minutes
# ──────────────────────────────────────────────────────────
log "E7: Generating all paper figures"

python eval/plot_results.py \
    --results_dir $OUT \
    --output_dir $OUT/figures

log_done "E7: Figures → $OUT/figures/"

# ──────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────
log "ALL EXPERIMENTS COMPLETE"
echo ""
echo "Results:"
ls -lh $OUT/*.json 2>/dev/null || echo "  (no JSON files)"
echo ""
echo "Figures:"
ls -lh $OUT/figures/*.pdf 2>/dev/null || echo "  (no PDF files)"
echo ""
echo "Done at: $(date '+%Y-%m-%d %H:%M:%S')"
