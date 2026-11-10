# Q9: optional (P1). Run ONLY after Q8 finished and only if `free -g` shows >= 150 GB RAM free.
source "$(dirname "$0")/common.sh"
# larger corpus: 150 HotpotQA samples (~1000 materialized views, ~75 GB CPU RAM for 7B)
run q9_scale_hotpotqa_n150 --model $M7 --dataset hotpotqa --n_samples 150 --seed 0 \
    --methods full cacheblend reqdiff semanticache
# extra seed for the closest-baseline comparison
run q9_reqdiff_hotpotqa_s1 --model $M7 --dataset hotpotqa --n_samples 60 --seed 1 \
    --methods cacheblend reqdiff semanticache
echo "Q9 ALL DONE"
