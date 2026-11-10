"""Correctness checks for the KV engine. Run: python test_engine.py <model_path>"""
import sys
import time

import torch

sys.path.insert(0, ".")
from engine import KVEngine

mp = sys.argv[1]
e = KVEngine(mp, "cuda:0").load()
text = ("The Eiffel Tower is a wrought-iron lattice tower in Paris. It was designed by "
        "Gustave Eiffel's company and completed in 1889. ") * 40 + "Question: Who designed it? Answer:"
ids = e.tok(text).input_ids
T = len(ids)
print("T =", T)

# 1. engine full prefill vs HF forward
with torch.no_grad():
    ref = e.model(torch.tensor([ids], device=e.device)).logits[0, -1].float()
pr = e.prefill(ids)
print("[1] HF argmax", int(ref.argmax()), "engine", pr.first_token,
      "OK" if int(ref.argmax()) == pr.first_token else "MISMATCH")

# 2. reuse all but last 8 tokens from the full prefill itself -> must match
mask = torch.ones(T, dtype=torch.bool)
mask[-8:] = False
pr2 = e.prefill(ids, mask, pr.pre_k, pr.v, 0.0)
print("[2] self-reuse first token", pr2.first_token, "OK" if pr2.first_token == pr.first_token else "MISMATCH",
      "computed", pr2.n_computed, "ttft full %.1f reuse %.1f" % (pr.ttft_ms, pr2.ttft_ms))

# 3. selective recompute with ratio 1.0 == full
pr3 = e.prefill(ids, mask, pr.pre_k, pr.v, 1.0)
print("[3] ratio=1.0 first token", pr3.first_token, "OK" if pr3.first_token == pr.first_token else "MISMATCH")

# 4. chunk materialized independently at position 0, reused at offset (RoPE re-rotation)
half = T // 2
c = e.materialize(ids[half:-8], to_cpu=False)
pre = e.materialize(ids[:half], to_cpu=False)
Ks = [torch.cat([pre.k[l], c.k[l], torch.zeros_like(c.k[l][:, :8])], 1) for l in range(e.n_layers)]
Vs = [torch.cat([pre.v[l], c.v[l], torch.zeros_like(c.v[l][:, :8])], 1) for l in range(e.n_layers)]
pr4 = e.prefill(ids, mask.to(e.device), Ks, Vs, 0.0)
pr5 = e.prefill(ids, mask.to(e.device), Ks, Vs, 0.15)
print("[4] independent chunks r=0:", pr4.first_token, " r=0.15:", pr5.first_token, " full:", pr.first_token)

# 5. decode
gen, ms = e.decode(pr, 16)
print("[5] decode:", repr(e.tok.decode(gen)), "%.0f ms" % ms)
gen4, _ = e.decode(pr5, 16)
print("    decode chunk-reuse:", repr(e.tok.decode(gen4)))
print("KV bytes/token:", e.kv_bytes_per_token())
