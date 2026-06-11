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
#  Total estimated time: ~3.5 hours on A100 80GB
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
echo "  GPU    : $GPU (CUDA_VISIBLE_DEVICES)"
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
#     200 requests × 5 configurations
#     Estimated: ~35 minutes
# ──────────────────────────────────────────────────────────
log "E2: Ablation Study (200 req × 5 configs)"

python eval/run_real_benchmark.py \
    --model_path $MODEL \
    --dataset $DATASET \
    --num_requests 200 \
    --max_new_tokens 64 \
    --ablation \
    --device $DEVICE \
    --output_dir $OUT

log_done "E2: Ablation → $OUT/ablation.json"

# ──────────────────────────────────────────────────────────
# E3: Scalability Study (cache warm-up curve)
#     1000 requests, checkpoint every 50
#     Runs SemantiCache + No Cache in parallel per request
#     Estimated: ~67 minutes
# ──────────────────────────────────────────────────────────
log "E3: Scalability Study (1000 req, checkpoint=50)"

python eval/run_real_benchmark.py \
    --model_path $MODEL \
    --dataset $DATASET \
    --num_requests 1000 \
    --max_new_tokens 64 \
    --scalability \
    --checkpoint_every 50 \
    --device $DEVICE \
    --output_dir $OUT

log_done "E3: Scalability → $OUT/scalability.json"

# ──────────────────────────────────────────────────────────
# E4: Extended Main Comparison (1000 requests, 4 systems)
#     More statistically significant than 500-req run
#     Estimated: ~135 minutes
# ──────────────────────────────────────────────────────────
log "E4: Extended Main Comparison (1000 req, 4 systems)"

python eval/run_real_benchmark.py \
    --model_path $MODEL \
    --dataset $DATASET \
    --num_requests 1000 \
    --max_new_tokens 64 \
    --threshold 0.85 \
    --device $DEVICE \
    --seed 123 \
    --output_dir $OUT

log_done "E4: Main comparison → $OUT/benchmark_${DATASET}_n1000.json"

# ──────────────────────────────────────────────────────────
# E5: Generate All Paper Figures
#     Uses all JSON results produced above
#     Estimated: ~2 minutes
# ──────────────────────────────────────────────────────────
log "E5: Generating all paper figures"

python eval/plot_results.py \
    --results_dir $OUT \
    --output_dir $OUT/figures

log_done "E5: Figures → $OUT/figures/"

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
