#!/usr/bin/env bash
# ============================================================
# 消融实验重跑脚本 (n=500)
# 用途：修复消融图与主实验数字不一致的问题
# 在 A100 服务器上运行，约 30-60 分钟
# ============================================================

set -e

# 进入代码目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "=========================================="
echo " SemantiCache 消融实验重跑 (n=500)"
echo " 目录: $REPO_DIR"
echo " 时间: $(date)"
echo "=========================================="

# 备份旧结果
if [ -f results/ablation.json ]; then
    cp results/ablation.json results/ablation_n200_backup.json
    echo "[备份] 旧 ablation.json 已备份到 ablation_n200_backup.json"
fi

# 重跑消融实验 (n=500)
echo ""
echo "[运行] 消融实验 n=500 ..."
python eval/run_real_benchmark.py \
    --num_requests 500 \
    --ablation \
    --output_dir results \
    --seed 42

echo ""
echo "[完成] ablation.json 已更新"

# 重新生成消融图 (fig6_ablation.pdf / .png)
echo ""
echo "[绘图] 重新生成 fig6_ablation ..."

# 优先用 eval/plot_results.py（更新版本）
if [ -f eval/plot_results.py ]; then
    python eval/plot_results.py --output_dir results/figures
elif [ -f plot_results.py ]; then
    python plot_results.py --output_dir results/figures
fi

# 把新图复制到 overleaf figures 目录
OVERLEAF_FIGS="../../semanticache_overleaf/figures"
if [ -d "$OVERLEAF_FIGS" ]; then
    cp results/figures/fig6_ablation.pdf "$OVERLEAF_FIGS/fig6_ablation.pdf" 2>/dev/null || true
    cp results/figures/fig6_ablation.png "$OVERLEAF_FIGS/fig6_ablation.png" 2>/dev/null || true
    echo "[同步] 新图已复制到 semanticache_overleaf/figures/"
fi

echo ""
echo "=========================================="
echo " 全部完成！$(date)"
echo " 下一步：把新的 ablation.json 中的"
echo " Full System 和 w/o Semantic 命中率数字"
echo " 告诉 Antigravity 以更新论文正文数字。"
echo "=========================================="

# 打印新结果摘要
echo ""
echo "--- 新消融结果摘要 ---"
python3 -c "
import json
with open('results/ablation.json') as f:
    data = json.load(f)
for r in data:
    name = r.get('name', r.get('system', 'Unknown'))
    hr = r.get('hit_rate', 0) * 100
    ehr = r.get('exact_hit_rate', 0) * 100
    shr = r.get('semantic_hit_rate', 0) * 100
    print(f'  {name:<30} total={hr:.1f}%  exact={ehr:.1f}%  semantic={shr:.1f}%')
"
