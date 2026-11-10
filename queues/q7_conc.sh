# Q7: E4 open-loop concurrency benchmark (single A100, FCFS worker, Poisson arrivals).
source "$(dirname "$0")/common.sh"
if [ ! -f runs/q7_conc/summary.txt ]; then
  echo "$(date +%T) START q7_conc"
  $PY bench_concurrent.py --out runs/q7_conc --model $M7 --dataset hotpotqa --n_samples 60 \
      --n_req 150 --methods full apc cacheblend semanticache --rates 0.4 0.8 1.2 1.6 2.0 \
      > runs/q7_conc.log 2>&1 || echo "$(date +%T) FAILED q7_conc"
  echo "$(date +%T) END   q7_conc"
else echo "SKIP q7_conc"; fi
