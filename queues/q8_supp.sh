# Q8: ICDE round-2 supplementary experiments (P0). Est. GPU time ~4 h on one A100.
# Order = priority: if time runs out, the most important results already exist.
source "$(dirname "$0")/common.sh"
B="--model $M7 --dataset hotpotqa --n_samples 60 --seed 0"

# --- S0: smoke test of every new code path (tiny; must succeed before anything else)
run q8_smoke $B --n_samples 3 --drift_frac 0.1 --kv_quant int8 --n_distractors 500 --profile \
    --methods full cacheblend semanticache semanticache_always reqdiff

# --- S1 (closest related work): request-level diff reuse (KVShare-style) vs chunk-level views
for ds in hotpotqa 2wikimqa musique; do
  run q8_reqdiff_$ds --model $M7 --dataset $ds --n_samples 100 --seed 0 \
      --methods full cacheblend reqdiff semanticache
done

# --- S2: latency breakdown (exact attribution with --profile)
run q8_breakdown_4k  $B --profile --methods full cacheblend semanticache reqdiff
run q8_breakdown_16k --model $M7 --dataset hotpotqa --n_samples 40 --max_ctx 16384 --seed 0 \
    --profile --methods cacheblend semanticache

# --- S3: maintenance cost vs revision magnitude (drift requests only)
for f in 0.02 0.05 0.1 0.2 0.3 0.5; do
  run q8_drift_$f $B --drift_frac $f --kinds drift \
      --methods full cacheblend semanticache_always semanticache
done

# --- S4: share of near-duplicate passages per request
for p in 0.0 0.25 0.5 0.75 1.0; do
  run q8_fmt_$p $B --fmt_prob $p --kinds format edit --methods full cacheblend semanticache
done

# --- S5: real memory pressure (one method per process; 2 GB already exists as q6_gpu2_*)
for g in 0.25 0.5 1; do
  for m in semanticache cacheblend; do
    run q8_gpu${g}_${m} $B --gpu_budget_gb $g --methods $m
  done
done

# --- S6: compressed materialized views (100 samples => paired with q1_hotpotqa_q7b_s0 / q3_hotpotqa_q14b)
Q100="--dataset hotpotqa --n_samples 100 --seed 0"
run q8_quant_int8_q7b  --model $M7  $Q100 --kv_quant int8 --methods cacheblend semanticache
run q8_quant_fp8_q7b   --model $M7  $Q100 --kv_quant fp8  --methods cacheblend semanticache
run q8_quant_int8_q14b --model $M14 $Q100 --kv_quant int8 --methods cacheblend semanticache

# --- S7: semantic-index scalability
for n in 10000 50000 200000; do
  run q8_index_$n $B --n_distractors $n --profile --methods semanticache
done
echo "Q8 ALL DONE"
