"""Calibrate engine speed vs HF. usage: python profile_engine.py <model>"""
import sys, time, torch
sys.path.insert(0, ".")
from engine import KVEngine
e = KVEngine(sys.argv[1], "cuda:0").load()
txt = ("The quick brown fox jumps over the lazy dog near the river bank. " * 600)
ids = e.tok(txt).input_ids[:4096]
T = len(ids); print("T", T)
def tm(f, n=5):
    f(); torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000
x = torch.tensor([ids], device=e.device)
with torch.no_grad():
    print("HF forward      %.0f ms" % tm(lambda: e.model(x, use_cache=False, logits_to_keep=1)))
print("engine full     %.0f ms" % tm(lambda: e.prefill(ids, build_cache=False)))
pr = e.prefill(ids, build_cache=False)
half = T - 600
mask = torch.zeros(T, dtype=torch.bool, device=e.device); mask[:half] = True
def reuse(r):
    K=[t.clone() for t in pr.pre_k]; V=[t.clone() for t in pr.v]
    return e.prefill(ids, mask, K, V, r, build_cache=False)
print("reuse 85%% r=0   %.0f ms" % tm(lambda: reuse(0.0)))
print("reuse 85%% r=.15 %.0f ms" % tm(lambda: reuse(0.15)))
print("peak mem GB", torch.cuda.max_memory_allocated()/2**30)
