# Q3: other models (needs test_engine.py to pass on each model first!)
source "$(dirname "$0")/common.sh"
M="full apc gptcache cacheblend naive_sem semanticache"
run q3_hotpotqa_q3b  --model $M3  --dataset hotpotqa --n_samples 100 --methods $M
run q3_hotpotqa_q14b --model $M14 --dataset hotpotqa --n_samples 100 --methods $M
run q3_hotpotqa_llama8b   --model $ML --dataset hotpotqa --n_samples 100 --methods $M
run q3_hotpotqa_mistral7b --model $MM --dataset hotpotqa --n_samples 100 --methods $M
