"""
SemantiCache ICDE Paper Figures
Style: Tang et al. PVLDB 2025 — pastel fills + hatch patterns
Color scheme:
  - No Cache    : light salmon  (#FFBBBB) + '////' hatch
  - Exact Cache : light green   (#BBDDBB) + 'xxxx' hatch
  - SemShareKV  : light yellow  (#EEEEAA) + '\\\\' hatch
  - SemantiCache: light blue    (#BBCCFF) + '....' hatch  (our method)
Architecture diagram: light blue-gray tint outer box
NOTE: No set_title() calls — captions are handled by LaTeX.
"""

import json, os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.45,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

RESULTS = 'results'
FIGS = os.path.join(RESULTS, 'figures')
os.makedirs(FIGS, exist_ok=True)

def load(fname):
    with open(os.path.join(RESULTS, fname)) as f:
        return json.load(f)

def save(fig, name):
    fig.savefig(os.path.join(FIGS, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, f'{name}.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  {name}  \u2713')

# ── Pastel color palette ──────────────────────────────────────────
C = {
    'no':    '#FFBBBB',   # light salmon  — No Cache
    'exact': '#BBDDBB',   # light green   — Exact Cache (APC)
    'sem':   '#EEEEAA',   # light yellow  — SemShareKV
    'ours':  '#BBCCFF',   # light blue    — SemantiCache (ours)
}
H = {
    'no':    '////',
    'exact': 'xxxx',
    'sem':   '\\\\',
    'ours':  '....',
}
LABELS = ['No Cache', 'Exact Cache (APC)', 'SemShareKV', 'SemantiCache (ours)']
CL = [C['no'], C['exact'], C['sem'], C['ours']]
HL = [H['no'], H['exact'], H['sem'], H['ours']]

def legend_patches():
    return [mpatches.Patch(fc=CL[i], ec='black', hatch=HL[i], linewidth=0.9,
                           label=LABELS[i]) for i in range(4)]

def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=11))

def circle_num(ax, x, y, num, r=0.12, fontsize=8):
    c = plt.Circle((x, y), r, fc='#333333', ec='black', lw=0.8, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, str(num), ha='center', va='center',
            fontsize=fontsize, color='white', fontweight='bold', zorder=6)

def fancy_box(ax, x, y, w, h, label, sub='',
              fc='#EEEEEE', ec='#444444', lw=1.2, ls='solid',
              fontsize=9, bold=True, subfontsize=7.0, linespacing=1.55):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04',
                       fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2)
    ax.add_patch(p)
    ty = y + h/2 + (0.14 if sub else 0)
    ax.text(x+w/2, ty, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold' if bold else 'normal', zorder=3)
    if sub:
        ax.text(x+w/2, y+h/2 - 0.20, sub, ha='center', va='center',
                fontsize=subfontsize, color='#333333', zorder=3,
                linespacing=linespacing)


# ═══════════════════════════════════════════════════════════════════
# Fig 1 — Motivation: Pipeline
# ═══════════════════════════════════════════════════════════════════
def fig_motivation():
    """Pipeline figure — single-column vertical layout (original style)."""
    fig = plt.figure(figsize=(3.5, 4.5))
    ax = fig.add_axes([0.04, 0.02, 0.92, 0.96])
    ax.set_xlim(0, 10); ax.set_ylim(4.1, 17.0); ax.axis('off')

    # ── Row 1: Multi-Tenant LLM Requests ──────────────────────────
    ax.text(5.0, 16.55, 'Multi-Tenant LLM Requests',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    p = FancyBboxPatch((0.15, 14.60), 9.70, 1.85,
                       boxstyle='round,pad=0.05',
                       fc='white', ec='black', lw=1.1, linestyle='dashed', zorder=1)
    ax.add_patch(p)
    fancy_box(ax, 0.30, 14.72, 2.90, 1.60, 'Tenant A', 'RAG queries',
              fc='#F0F4FF', fontsize=8.5)
    fancy_box(ax, 3.55, 14.72, 2.90, 1.60, 'Tenant B', 'Chat queries',
              fc='#F0F4FF', fontsize=8.5)
    fancy_box(ax, 6.80, 14.72, 3.05, 1.60, 'Tenant C \u2026', '',
              fc='#F0F4FF', fontsize=8.5, bold=False)

    # ── Arrow: Tenants → SemantiCache ─────────────────────────────
    arrow(ax, 5.0, 14.60, 5.0, 13.80)
    ax.text(5.4, 14.20, 'requests', ha='left', va='center',
            fontsize=7.0, color='#444')

    # ── Row 2: SemantiCache core ───────────────────────────────────
    p2 = FancyBboxPatch((0.15, 9.35), 9.70, 4.35,
                        boxstyle='round,pad=0.05',
                        fc='#EBF0FA', ec='#3A5FA0', lw=1.5,
                        linestyle='dashed', zorder=1)
    ax.add_patch(p2)
    ax.text(5.0, 13.60, 'SemantiCache',
            ha='center', fontsize=11, fontweight='bold', color='#1A2A6C')

    # HSI  |  QBR (side by side)
    fancy_box(ax, 0.40, 11.50, 4.25, 1.80, 'HSI',
              'Radix Tree + LSH', fc='#D8E4F8', fontsize=10, subfontsize=6.5)
    fancy_box(ax, 5.35, 11.50, 4.25, 1.80, 'QBR',
              r'$\tau$-gate, $\tau$=0.85', fc='#D8E4F8', fontsize=10, subfontsize=6.5)
    # ① ② circles
    arrow(ax, 2.52, 11.50, 2.52, 11.22); circle_num(ax, 2.52, 11.10, 1, r=0.14)
    arrow(ax, 4.65, 12.40, 5.35, 12.40); circle_num(ax, 5.08, 12.40, 2, r=0.14)

    # TSM
    fancy_box(ax, 0.40, 10.08, 9.20, 1.20, 'TSM',
              'L1: HBM  |  L2: DRAM  |  L3: NVMe SSD',
              fc='#C8D8F0', fontsize=9.5)
    arrow(ax, 5.0, 10.08, 5.0, 9.82); circle_num(ax, 5.0, 9.68, 3, r=0.14)

    # ── Arrow: SemantiCache → LLM Engine ─────────────────────────
    arrow(ax, 5.0, 9.35, 5.0, 8.55)
    ax.text(5.4, 8.95, 'hit\u2192skip', ha='left', va='center',
            fontsize=7.0, color='#444')

    # ── Row 3: LLM Engine ─────────────────────────────────────────
    fancy_box(ax, 1.50, 7.00, 7.00, 1.45, 'LLM Engine',
              'prefill on miss only', fc='#F8F0E0', fontsize=10)

    # ── TIL — full-width at bottom ────────────────────────────────
    fancy_box(ax, 0.40, 5.50, 9.20, 1.30, 'TIL',
              'Tenant Isolation  (leakage = 0%)', fc='#D8E4F8', fontsize=9.5)

    # ── Response ──────────────────────────────────────────────────
    arrow(ax, 5.0, 5.50, 5.0, 4.80)
    ax.text(5.0, 4.60, 'Response', ha='center', va='top',
            fontsize=9, style='italic')

    save(fig, 'fig1_motivation')



def fig_hitrate():
    """Standalone cache hit-rate bar chart (single-column, placed in §5 RQ1)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    systems = ['No\nCache', 'Exact\nCache', 'Sem\nShareKV', 'Semanti\nCache']
    hits = [0.0, 88.0, 97.9, 96.7]
    x = np.arange(4)
    w = 0.48
    bars = ax.bar(x, hits, w, color=CL, edgecolor='black', linewidth=1.0)
    for bar, h in zip(bars, HL):
        bar.set_hatch(h)
    ax.set_ylabel('Cache Hit Rate (%)', fontsize=10.5)
    ax.set_ylim(0, 122)
    ax.set_xlim(-0.6, 3.8)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_patches(), fontsize=8.5, ncol=2,
              loc='upper left', bbox_to_anchor=(0.01, 1.0),
              frameon=True, framealpha=0.9,
              handlelength=1.5, handletextpad=0.4, columnspacing=0.5)
    plt.tight_layout()
    save(fig, 'fig2_hitrate')



# ═══════════════════════════════════════════════════════════════════
# Fig 2 — Architecture framework
# ═══════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0.45, 8.2); ax.axis('off')

    # Top: LLM Engine
    fancy_box(ax, 0.8, 7.40, 8.4, 0.60,
              'LLM Serving Engine  (vLLM-compatible)',
              fc='white', ec='black', lw=1.5, fontsize=10.5)
    arrow(ax, 2.8, 7.40, 2.8, 6.78, lw=1.4)
    ax.text(2.2, 7.08, 'Request', fontsize=8.5)
    arrow(ax, 7.2, 6.78, 7.2, 7.40, lw=1.4)
    ax.text(7.4, 7.08, 'KV Block / Miss', fontsize=8.5)

    # SemantiCache Core outer box
    outer = FancyBboxPatch((0.35, 1.40), 9.30, 5.25,
                           boxstyle='round,pad=0.06',
                           fc='#EBF0FA', ec='#3A5FA0',
                           lw=2.0, linestyle='dashed', zorder=1)
    ax.add_patch(outer)
    ax.text(5.0, 6.76, 'SemantiCache Core',
            ha='center', fontsize=11.5, fontweight='bold',
            color='black', zorder=5,
            bbox=dict(fc='#EBF0FA', ec='none', pad=2, alpha=1.0))

    # TIL sidebar
    til = FancyBboxPatch((0.50, 1.54), 1.28, 4.90,
                         boxstyle='round,pad=0.04',
                         fc='#CBD8EE', ec='#3A5FA0', lw=1.3, zorder=2)
    ax.add_patch(til)
    ax.text(1.14, 6.24, 'TIL', ha='center', va='center',
            fontsize=9, fontweight='bold', color='black', zorder=3)
    ax.text(1.14, 5.96, 'Tenant Isolation\nLayer',
            ha='center', va='center', fontsize=6.5, color='black', zorder=3)
    fancy_box(ax, 0.57, 5.04, 1.14, 0.72, 'Public Pool',
              fc='white', ec='#555', fontsize=7.5, bold=False)
    fancy_box(ax, 0.57, 4.10, 1.14, 0.72, 'Private\nSub-trees',
              fc='white', ec='#555', fontsize=7.5, bold=False)
    fancy_box(ax, 0.57, 1.65, 1.14, 2.20, 'Permission\nBitmap',
              fc='white', ec='#555', fontsize=7.5, bold=False)

    # HSI block
    hsi_c = FancyBboxPatch((2.05, 3.80), 3.35, 2.78,
                           boxstyle='round,pad=0.04',
                           fc='#D0DCEE', ec='#3A5FA0', lw=1.3, zorder=2)
    ax.add_patch(hsi_c)
    ax.text(3.725, 6.46, 'HSI \u2014 Hierarchical Semantic Index',
            ha='center', fontsize=9, fontweight='bold',
            color='black', zorder=3)
    fancy_box(ax, 2.14, 3.96, 1.48, 1.92,
              'Exact Layer',
              'Radix Tree, O(L)\nSHA-256 hash',
              fc='white', ec='#666', fontsize=8.5, subfontsize=7.8)
    fancy_box(ax, 3.82, 3.96, 1.48, 1.92,
              'LSH Layer',
              'SimHash 64-bit\nTop-k \u00b7 O(1) lookup',
              fc='white', ec='#666', fontsize=8.5, subfontsize=7.8)
    arrow(ax, 3.62, 4.92, 3.82, 4.92, lw=1.1)
    ax.text(3.72, 5.12, 'miss', ha='center', fontsize=7.5, color='black')

    # QBR block
    qbr_c = FancyBboxPatch((5.60, 3.80), 3.65, 2.78,
                           boxstyle='round,pad=0.04',
                           fc='#D0DCEE', ec='#3A5FA0', lw=1.3, zorder=2)
    ax.add_patch(qbr_c)
    ax.text(7.425, 6.46, 'QBR \u2014 Quality-Bounded Reuse',
            ha='center', fontsize=9, fontweight='bold',
            color='black', zorder=3)
    fancy_box(ax, 5.70, 3.96, 1.55, 1.92,
              'Similarity Gate',
              'cos(E(p),E(p\'))>=tau\ntau=0.85 (default)',
              fc='white', ec='#666', fontsize=8.5, subfontsize=7.8)
    fancy_box(ax, 7.52, 3.96, 1.60, 1.92,
              'Quality Monitor',
              'BERTScore F1\nAdaptive \u03c4 window',
              fc='white', ec='#666', fontsize=8.5, subfontsize=7.8)
    arrow(ax, 7.25, 4.92, 7.52, 4.92, lw=1.1)

    # TSM block
    tsm_c = FancyBboxPatch((2.05, 1.54), 7.50, 2.08,
                           boxstyle='round,pad=0.04',
                           fc='#C8D4EC', ec='#3A5FA0', lw=1.5, zorder=2)
    ax.add_patch(tsm_c)
    ax.text(5.0, 3.50, 'TSM — Tiered Storage Manager',
            ha='center', fontsize=9, fontweight='bold', color='black', zorder=5)
    fancy_box(ax, 2.14, 1.68, 2.18, 1.68,
              'L1  GPU HBM',
              '~2 TB/s · hot KV blocks\n+ HSI metadata',
              fc='white', ec='#666', fontsize=8.5, subfontsize=7.8)
    fancy_box(ax, 4.62, 1.68, 2.18, 1.68,
              'L2  CPU DRAM',
              '~50 GB/s · warm blocks\nPCIe 4.0',
              fc='white', ec='#666', fontsize=8.5, subfontsize=7.8)
    fancy_box(ax, 7.00, 1.68, 2.42, 1.68,
              'L3  NVMe SSD',
              '~7 GB/s · cold blocks\nTB-scale cap.',
              fc='white', ec='#666', fontsize=8.5, subfontsize=7.8)
    arrow(ax, 4.32, 2.70, 4.62, 2.70, lw=1.2)
    ax.text(4.47, 2.88, 'demote', ha='center', fontsize=7, color='black')
    arrow(ax, 4.62, 2.36, 4.32, 2.36, lw=1.2)
    ax.text(4.47, 2.20, 'promote', ha='center', fontsize=7, color='black')
    arrow(ax, 6.80, 2.70, 7.00, 2.70, lw=1.2)
    ax.text(6.90, 2.88, 'demote', ha='center', fontsize=7, color='black')
    arrow(ax, 7.00, 2.36, 6.80, 2.36, lw=1.2)
    ax.text(6.90, 2.20, 'promote', ha='center', fontsize=7, color='black')

    # Numbered data-flow circles
    arrow(ax, 1.85, 5.16, 2.05, 5.16, lw=1.3, color='black')
    circle_num(ax, 1.74, 5.16, 1, r=0.12)
    arrow(ax, 1.85, 4.58, 2.05, 4.58, lw=1.3, color='black')
    circle_num(ax, 1.74, 4.58, 2, r=0.12)
    arrow(ax, 5.40, 5.16, 5.62, 5.16, lw=1.3, color='black')
    circle_num(ax, 5.26, 5.16, 3, r=0.12)
    ax.text(5.26, 4.92, 'top-k', ha='center', fontsize=7, color='black')
    arrow(ax, 6.85, 4.05, 6.85, 3.64, lw=1.6, color='black')
    circle_num(ax, 6.85, 4.05, 4, r=0.13)
    ax.text(7.05, 3.85, 'fetch block', ha='left', va='center', fontsize=7, color='black')

    ax.text(5.0, 0.90,
            u'\u2460 Request \u2192 TIL \u2192 \u2461 HSI (Exact\u2192LSH) \u2192 '
            u'\u2462 QBR gate \u2192 \u2463 TSM fetch \u2192 LLM Decoder',
            ha='center', fontsize=8.5, color='black', style='italic')

    save(fig, 'fig2_architecture')


# ═══════════════════════════════════════════════════════════════════
# Fig 3 — End-to-end performance
# ═══════════════════════════════════════════════════════════════════
def fig_end_to_end():
    rag   = load('benchmark_rag_synthetic_n1000.json')
    lmsys = load('benchmark_lmsys_n500.json')

    def pick(data, key, mult=1):
        return [d.get(key, 0)*mult for d in data[:4]]

    rag_hit  = pick(rag,   'hit_rate', 100)
    rag_ttft = pick(rag,   'mean_ttft_ms')
    lms_hit  = pick(lmsys, 'hit_rate', 100)
    lms_ttft = pick(lmsys, 'mean_ttft_ms')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))   # larger for readability
    w = 0.35; xs = np.arange(4)

    titles = ['(a) RAG Synthetic ($n$=1,000)', '(b) LMSYS Real-World ($n$=500)']
    for ax, hits, ttfts, title in zip(
        axes,
        [rag_hit, lms_hit],
        [rag_ttft, lms_ttft],
        titles
    ):
        ax2 = ax.twinx()
        bars = ax.bar(xs, hits, w, color=CL, edgecolor='black', linewidth=0.9, zorder=3)
        for b, h in zip(bars, HL):
            b.set_hatch(h)
        ax2.plot(xs, ttfts, color='black', marker='D', markersize=6,
                 linewidth=1.8, linestyle='--', zorder=4, label='TTFT (ms)')
        ax2.set_ylabel('Mean TTFT (ms)', fontsize=10.5)
        ax2.set_ylim(20, 52)
        ax2.tick_params(axis='y', labelsize=10)
        ax.set_ylabel('Cache Hit Rate (%)', fontsize=10.5)
        ax.set_ylim(0, 120)
        ax.set_xticks(xs)
        ax.set_xticklabels(['No\nCache', 'Exact\nCache', 'Sem-\nShareKV', 'Semanti-\nCache'],
                           fontsize=9.5)
        ax.tick_params(axis='y', labelsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        # subfigure title below x-axis
        ax.text(0.5, -0.20, title, transform=ax.transAxes,
                fontsize=10, fontweight='bold', ha='center', va='top', clip_on=False)

    fig.legend(handles=legend_patches() + [
        plt.Line2D([0], [0], color='black', marker='D', markersize=5,
                   linestyle='--', label='TTFT (ms)')
    ], loc='upper center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=9.5)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, 'fig3_end_to_end')


# ═══════════════════════════════════════════════════════════════════
# Fig 4 — Tau sweep
# ═══════════════════════════════════════════════════════════════════
def fig_tau_sweep():
    data  = load('tau_sweep.json')
    taus  = [d['tau']          for d in data]
    hits  = [d['hit_rate']*100 for d in data]
    berts = [d['bertscore']    for d in data]

    fig, ax1 = plt.subplots(figsize=(5.8, 3.6))
    bw = 0.038
    bars = ax1.bar(taus, hits, bw, color=C['ours'], edgecolor='black',
                   hatch=H['ours'], linewidth=0.9, label='Hit Rate (%)')
    ax1.set_xlabel('Reuse Safety Threshold ($\\tau$)', fontsize=10)
    ax1.set_ylabel('Cache Hit Rate (%)', fontsize=10)
    ax1.set_ylim(60, 105)
    ax1.set_xticks(taus)
    ax1.set_xticklabels([str(t) for t in taus], fontsize=8.5)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(taus, berts, color='black', marker='o', markersize=6,
             linewidth=2.0, label='BERTScore')
    ax2.set_ylabel('Generation Quality (BERTScore)', fontsize=10)
    ax2.set_ylim(0.800, 0.895)
    ax2.grid(False)

    ax1.axvline(0.85, color='gray', linestyle=':', linewidth=1.2)
    ax1.text(0.855, 103, 'default $\\tau$', ha='left', fontsize=7.5, color='#555')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left', frameon=True, framealpha=0.9, fontsize=8.5)
    plt.tight_layout()
    save(fig, 'fig4_tau_sweep')


# ═══════════════════════════════════════════════════════════════════
# Fig 5 — Scalability
# ═══════════════════════════════════════════════════════════════════
def fig_scalability():
    data   = load('scalability.json')
    x      = data['checkpoints']
    hits   = data['semanticache_hit_rate']
    ttft_s = data['semanticache_mean_ttft']
    ttft_n = data['no_cache_mean_ttft']

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    axes[0].plot(x, hits, color='#2255AA', marker='o', markersize=3.5,
                 linewidth=2.0, label='SemantiCache')
    axes[0].fill_between(x, hits, 50, alpha=0.12, color='#BBCCFF')
    axes[0].set_xlabel('Number of Requests', fontsize=10)
    axes[0].set_ylabel('Cumulative Hit Rate (%)', fontsize=10)
    axes[0].set_ylim(50, 100); axes[0].set_xlim(0, 1050)
    axes[0].legend(loc='lower right', frameon=True, framealpha=0.9)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.4)

    axes[1].plot(x, ttft_n, color='#CC3333', marker='s', markersize=3.5,
                 linewidth=1.8, linestyle='--', label='No Cache')
    axes[1].plot(x, ttft_s, color='#2255AA', marker='o', markersize=3.5,
                 linewidth=2.0, label='SemantiCache')
    axes[1].fill_between(x, ttft_n, ttft_s, alpha=0.15, color='#BBCCFF',
                         label='TTFT savings')
    axes[1].set_xlabel('Number of Requests', fontsize=10)
    axes[1].set_ylabel('Mean TTFT (ms)', fontsize=10)
    axes[1].set_ylim(28, 46); axes[1].set_xlim(0, 1050)
    axes[1].legend(loc='upper right', frameon=True, framealpha=0.9)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.4)

    # subfigure labels — below each panel, centered (matches fig1 style)
    for ax, lbl in zip(axes, ['(a)', '(b)']):
        ax.text(0.5, -0.18, lbl, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top', clip_on=False)

    plt.tight_layout()
    save(fig, 'fig5_scalability')


# ═══════════════════════════════════════════════════════════════════
# Fig 6 — Ablation
# ═══════════════════════════════════════════════════════════════════
def fig_ablation():
    data  = load('ablation.json')
    names = ['Full\nSystem', 'w/o\nSemantic', 'w/o\nQBR', 'w/o\nTSM', 'w/o\nTIL']
    exact = [d['exact_hit_rate']*100    for d in data]
    sem   = [d['semantic_hit_rate']*100 for d in data]
    total = [d['hit_rate']*100          for d in data]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = np.arange(5); w = 0.48

    b1 = ax.bar(xs, exact, w, label='Exact Hit', color=C['exact'],
                edgecolor='black', hatch=H['exact'], linewidth=0.9)
    b2 = ax.bar(xs, sem, w, bottom=exact, label='Semantic Hit',
                color=C['ours'], edgecolor='black', hatch=H['ours'], linewidth=0.9)

    for i, t in enumerate(total):
        ax.text(xs[i], t+1.2, f'{t:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Cache Hit Rate (%)', fontsize=11)
    ax.set_ylim(0, 105)
    # Legend placed below the plot — never overlaps bars
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, frameon=True, framealpha=0.95, fontsize=10,
              handlelength=1.8, handleheight=1.0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, 'fig6_ablation')


# ═══════════════════════════════════════════════════════════════════
# Fig 7 — Overhead breakdown
# ═══════════════════════════════════════════════════════════════════
def fig_overhead():
    data  = load('overhead_breakdown.json')
    comps = ['Exact Hash', 'Embedding\nExtraction', 'LSH Lookup', 'QBR\nCheck']
    times = [data['exact_hash_avg_ms'], data['embedding_avg_ms'],
             data['lsh_lookup_avg_ms'], data['qbr_similarity_avg_ms']]
    fcs  = [C['no'], C['sem'], C['exact'], C['ours']]
    hats = [H['no'], H['sem'], H['exact'], H['ours']]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    left = 0
    for t, fc, h, lbl in zip(times, fcs, hats, comps):
        ax.barh('Overhead', t, left=left, color=fc, hatch=h,
                edgecolor='black', linewidth=0.9, label=lbl)
        if t > 0.05:
            ax.text(left + t/2, 0, f'{t:.3f}ms',
                    ha='center', va='center', fontsize=7.8, fontweight='bold')
        left += t
    ax.set_xlabel('Latency (ms)', fontsize=10)
    ax.set_xlim(0, 11.0); ax.set_yticks([])
    ax.text(left+0.15, 0, f'Total:\n{left:.2f} ms', va='center', fontsize=9, fontweight='bold')
    ax.legend(loc='upper center', ncol=4, frameon=False,
              bbox_to_anchor=(0.5, 1.42), fontsize=8.5)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    save(fig, 'fig7_overhead')




if __name__ == '__main__':
    print('Generating figures (Tang et al. pastel+hatch style)...\n')
    fig_motivation()
    fig_hitrate()
    fig_architecture()
    fig_end_to_end()
    fig_tau_sweep()
    fig_scalability()
    fig_ablation()
    fig_overhead()
    print('\nAll figures saved to', FIGS)
