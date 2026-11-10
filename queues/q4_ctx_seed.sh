# Q4: context length scaling and seeds for confidence intervals (7B)
source "$(dirname "$0")/common.sh"
M="full apc cacheblend naive_sem semanticache"
for c in 2048 8192; do
  run q4_ctx${c} --model $M7 --dataset hotpotqa --n_samples 60 --max_ctx $c --methods $M
done
for s in 1 2; do
  run q4_hotpotqa_q7b_s$s --model $M7 --dataset hotpotqa --n_samples 100 --methods $ALLM --seed $s
done
