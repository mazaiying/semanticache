# Q5: extra evidence (CIs on other datasets / big model, 16K context). Order = priority.
source "$(dirname "$0")/common.sh"
M="full apc cacheblend naive_sem semanticache"
run q5_ctx16384 --model $M7 --dataset hotpotqa --n_samples 40 --max_ctx 16384 --methods $M
run q5_2wikimqa_q7b_s1 --model $M7 --dataset 2wikimqa --n_samples 100 --methods $ALLM --seed 1
run q5_musique_q7b_s1  --model $M7 --dataset musique  --n_samples 100 --methods $ALLM --seed 1
run q5_hotpotqa_q14b_s1 --model $M14 --dataset hotpotqa --n_samples 100 --methods $M --seed 1
