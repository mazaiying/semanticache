# Q2: sensitivity sweeps for SemantiCache (7B, hotpotqa, 60 samples). Only the methods needed.
source "$(dirname "$0")/common.sh"
B="--model $M7 --dataset hotpotqa --n_samples 60 --seed 0"
for r in 0.05 0.10 0.15 0.30 0.50; do run q2_ratio_$r $B --ratio $r --methods semanticache cacheblend; done
for t in 0.5 0.6 0.7 0.8 0.9; do run q2_tau_$t $B --tau $t --methods semanticache naive_sem; done
for o in 0.2 0.3 0.5 0.7 0.9; do run q2_omega_$o $B --omega $o --methods semanticache; done
for h in 0 2 4 8 16; do run q2_halo_$h $B --halo $h --methods semanticache; done
