# Q1: main comparison, Qwen2.5-7B, three multi-doc QA datasets, 100 samples (400 requests)
source "$(dirname "$0")/common.sh"
for ds in hotpotqa 2wikimqa musique; do
  run q1_${ds}_q7b_s0 --model $M7 --dataset $ds --n_samples 100 --methods $ALLM --seed 0
done
