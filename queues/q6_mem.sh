# Q6: E5 memory / tiered-storage sweep (GPU-tier budget). One method per process so the
# GPU LRU state is not shared across methods.  Footprint (cpu_gb, kv_bytes_per_token,
# gpu_hit/cpu_hit/bytes_h2d) is recorded in summary.json -> _store.
source "$(dirname "$0")/common.sh"
B="--model $M7 --dataset hotpotqa --n_samples 60 --seed 0"
for g in 2 5 10 20 40; do
  for m in semanticache cacheblend; do
    run q6_gpu${g}_${m} $B --gpu_budget_gb $g --methods $m
  done
done
