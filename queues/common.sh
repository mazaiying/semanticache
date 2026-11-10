# sourced by queue scripts; runs on the SERVER inside ~/sc2
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1          # A100. NEVER use GPU 0 (someone else's vLLM) 
PY=$HOME/miniconda3/envs/semanticache/bin/python
cd $HOME/sc2; mkdir -p runs
run() {  # run <name> <args...>   (skips if summary already exists => resumable)
  name=$1; shift
  if [ -f runs/$name/summary.json ]; then echo "SKIP $name"; return; fi
  echo "$(date +%T) START $name"
  $PY run_exp.py --out runs/$name "$@" > runs/$name.log 2>&1 || echo "$(date +%T) FAILED $name"
  echo "$(date +%T) END   $name"
}
M7=$HOME/models/qwen2.5-7b-instruct
M3=$HOME/models/qwen2.5-3b-instruct
M14=$HOME/models/qwen2.5-14b-instruct
ML=$HOME/models/llama3.1-8b-instruct
MM=$HOME/models/mistral-7b-instruct-v0.3-c170c708
ALLM="full apc gptcache cacheblend_r0 cacheblend naive_sem semanticache_r0 semanticache_halo0 semanticache"
