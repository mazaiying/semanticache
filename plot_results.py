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

def arrow(ax, x1, y1, x2, y2, color='#4A4A4A', lw=1.15, ms=10):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=ms,
                                shrinkA=0, shrinkB=0),
                zorder=4)

def line(ax, x1, y1, x2, y2, color='#4A4A4A', lw=1.15):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
            solid_capstyle='round', zorder=4)

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
    """Motivating comparison between exact-prefix and semantic KV reuse."""
    fig = plt.figure(figsize=(3.55, 4.55))
    ax = fig.add_axes([0.02, 0.012, 0.96, 0.976])
    ax.set_xlim(0, 10); ax.set_ylim(2.00, 13.1); ax.axis('off')

    ax.text(5.0, 12.83, 'Same Intent, Different Tokens',
            ha='center', va='center', fontsize=11.2, fontweight='bold')

    fancy_box(ax, 0.35, 11.25, 9.30, 1.05, 'Tenant A - Request 1',
              '"Summarize the termination clauses"',
              fc='#F4F7FC', fontsize=8.8, subfontsize=7.8, lw=1.15)
    fancy_box(ax, 0.35, 9.88, 9.30, 1.05, 'Tenant A - Request 2',
              '"What are the exit provisions?"',
              fc='#F4F7FC', fontsize=8.8, subfontsize=7.8, lw=1.15)
    arrow(ax, 5.0, 11.23, 5.0, 10.95, lw=1.05, ms=8)
    ax.text(2.55, 9.66, 'low lexical overlap',
            ha='center', va='center', fontsize=7.0, style='italic',
            color='#333333', bbox=dict(fc='white', ec='none', pad=0.2))
    ax.text(7.15, 9.66, 'semantic similarity $\\geq\\tau$',
            ha='center', va='center', fontsize=7.0, style='italic',
            color='#333333', bbox=dict(fc='white', ec='none', pad=0.2))

    # Exact-prefix baseline exposes the missed opportunity.
    fancy_box(ax, 0.35, 7.90, 4.10, 1.18, 'Exact Prefix Cache',
              r'token hash differs  $\rightarrow$  MISS',
              fc='#FBE7E5', ec='#9E4A42', fontsize=9.0,
              subfontsize=7.4, lw=1.25)
    arrow(ax, 1.65, 9.86, 1.65, 9.10, lw=1.1, ms=9)

    # SemantiCache decision path.
    core = FancyBboxPatch((4.78, 7.55), 4.87, 1.88,
                          boxstyle='round,pad=0.05',
                          fc='#EAF1FB', ec='#315F9E', lw=1.35, zorder=1)
    ax.add_patch(core)
    ax.text(7.215, 9.18, 'SemantiCache',
            ha='center', va='center', fontsize=10.2, fontweight='bold',
            color='#173F78', bbox=dict(fc='#EAF1FB', ec='none', pad=0.3))

    module_x = [4.98, 6.13, 7.28, 8.43]
    module_names = ['TIL', 'HSI', 'QBR', 'TSM']
    module_subs = ['authorize', 'Exact/LSH', r'$\sigma\geq\tau$', 'L1/L2/L3']
    for i, (x, name, sub) in enumerate(zip(module_x, module_names, module_subs)):
        fancy_box(ax, x, 7.82, 1.02, 0.92, name, sub,
                  fc='white', ec='#4B5E78', fontsize=8.4,
                  subfontsize=6.2, lw=1.0)
        if i < 3:
            arrow(ax, x + 1.03, 8.28, module_x[i + 1] - 0.03, 8.28,
                  lw=1.0, ms=8)
    arrow(ax, 8.65, 9.86, 8.65, 9.45, lw=1.0, ms=8)

    # Two explicit outcomes make the benefit and fallback semantics visible.
    fancy_box(ax, 0.55, 5.68, 3.75, 1.16, 'Miss Path',
              r'full prefill  $\rightarrow$  admit new KV',
              fc='#F9EEDC', ec='#9A7241', fontsize=9.0,
              subfontsize=7.2, lw=1.2)
    fancy_box(ax, 5.70, 5.68, 3.75, 1.16, 'Approved Hit',
              r'fetch KV  $\rightarrow$  skip prefill',
              fc='#E2F2E8', ec='#3F7D5C', fontsize=9.0,
              subfontsize=7.2, lw=1.2)
    arrow(ax, 2.40, 7.88, 2.40, 6.86, color='#8B554D', lw=1.15, ms=9)
    arrow(ax, 8.94, 7.80, 8.94, 6.86, color='#3F7D5C', lw=1.15, ms=9)

    # Both paths converge on the same decoder interface.
    fancy_box(ax, 1.10, 3.63, 7.80, 1.10, 'LLM Decoder',
              'fresh response under current decoding policy',
              fc='#F4F4F4', ec='#444444', fontsize=9.8,
              subfontsize=7.4, lw=1.25)
    arrow(ax, 2.40, 5.66, 3.75, 4.75, lw=1.05, ms=8)
    arrow(ax, 7.60, 5.66, 6.25, 4.75, lw=1.05, ms=8)

    arrow(ax, 5.0, 3.62, 5.0, 2.75, lw=1.15, ms=9)
    ax.text(5.0, 2.43, 'Response', ha='center', va='center',
            fontsize=9.5, style='italic')
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
    """Compact double-column architecture with a fully explicit request path."""
    fig, ax = plt.subplots(figsize=(11.4, 5.15))
    ax.set_xlim(0, 12); ax.set_ylim(0.35, 6.65); ax.axis('off')

    fancy_box(ax, 0.65, 5.86, 11.00, 0.58,
              'LLM Serving Engine',
              fc='white', ec='#222222', lw=1.5, fontsize=10.5)


    outer = FancyBboxPatch((0.18, 0.62), 11.64, 4.86,
                           boxstyle='round,pad=0.06',
                           fc='#EEF3FB', ec='#315F9E',
                           lw=1.8, linestyle='dashed', zorder=1)
    ax.add_patch(outer)
    ax.text(6.0, 5.48, 'SemantiCache Core',
            ha='center', va='center', fontsize=11.5, fontweight='bold',
            color='#173F78', zorder=6,
            bbox=dict(fc='#EEF3FB', ec='none', pad=1.5))

    # TIL: authorization data and the candidate mask.
    til = FancyBboxPatch((0.40, 0.88), 1.95, 4.30,
                         boxstyle='round,pad=0.04',
                         fc='#D9E5F5', ec='#315F9E', lw=1.35, zorder=2)
    ax.add_patch(til)
    ax.text(1.375, 4.88, 'TIL', ha='center', fontsize=11.0,
            fontweight='bold', zorder=3)
    ax.text(1.375, 4.57, 'Tenant Isolation', ha='center',
            fontsize=9.5, zorder=3)
    fancy_box(ax, 0.57, 3.55, 1.61, 0.72, 'Public Pool', 'shareable',
              fc='white', ec='#52657F', fontsize=10.0, subfontsize=8.5, lw=1.0)
    fancy_box(ax, 0.57, 2.55, 1.61, 0.72, 'Private Pool', 'owner match',
              fc='white', ec='#52657F', fontsize=10.0, subfontsize=8.5, lw=1.0)
    fancy_box(ax, 0.57, 1.20, 1.61, 0.98, 'Metadata',
              'label + owner + tier',
              fc='white', ec='#52657F', fontsize=10.0, subfontsize=8.5, lw=1.0)

    # HSI: exact first, semantic only on an exact miss.
    hsi = FancyBboxPatch((2.68, 2.72), 3.38, 2.46,
                         boxstyle='round,pad=0.04',
                         fc='#D9E5F5', ec='#315F9E', lw=1.35, zorder=2)
    ax.add_patch(hsi)
    ax.text(4.37, 4.88, 'HSI - Hierarchical Index',
            ha='center', fontsize=11.0, fontweight='bold', zorder=3)
    fancy_box(ax, 2.86, 3.18, 1.40, 1.25, 'Exact',
              'SHA-256\nhash map',
              fc='white', ec='#52657F', fontsize=10.0, subfontsize=8.5, lw=1.0)
    fancy_box(ax, 4.48, 3.18, 1.40, 1.25, 'Semantic',
              'SimHash LSH\ncosine top-$k$',
              fc='white', ec='#52657F', fontsize=10.0, subfontsize=8.5, lw=1.0)
    arrow(ax, 4.27, 3.80, 4.46, 3.80, lw=1.10, ms=8)
    ax.text(4.36, 4.03, 'miss', ha='center', fontsize=9.8)

    # QBR: online decision plus a monitoring hook.
    qbr = FancyBboxPatch((6.38, 2.72), 2.56, 2.46,
                         boxstyle='round,pad=0.04',
                         fc='#D9E5F5', ec='#315F9E', lw=1.35, zorder=2)
    ax.add_patch(qbr)
    ax.text(7.66, 4.88, 'QBR - Quality Gate',
            ha='center', fontsize=11.0, fontweight='bold', zorder=3)
    fancy_box(ax, 6.62, 3.60, 2.08, 0.82, 'Similarity Decision',
              r'$\cos(E(p),E(p^\prime))\geq\tau$',
              fc='white', ec='#52657F', fontsize=10.0, subfontsize=8.8, lw=1.0)
    fancy_box(ax, 6.62, 2.95, 2.08, 0.48, 'Quality Monitor',
              'sampled BERTScore',
              fc='white', ec='#52657F', fontsize=9.5, subfontsize=8.5, lw=1.0)

    # Decision router makes hit and miss behavior explicit.
    router = FancyBboxPatch((9.25, 2.72), 2.28, 2.46,
                            boxstyle='round,pad=0.04',
                            fc='#D9E5F5', ec='#315F9E', lw=1.35, zorder=2)
    ax.add_patch(router)
    ax.text(10.39, 4.88, 'Reuse Decision',
            ha='center', fontsize=11.0, fontweight='bold', zorder=3)
    fancy_box(ax, 9.48, 3.64, 1.82, 0.72, 'Approved Hit',
              'fetch KV block',
              fc='#E2F2E8', ec='#3F7D5C', fontsize=10.0,
              subfontsize=8.5, lw=1.0)
    fancy_box(ax, 9.82, 2.78, 1.45, 0.52, 'Rejected / Miss',
              'run prefill',
              fc='#F9EEDC', ec='#9A7241', fontsize=9.0,
              subfontsize=8.5, lw=1.0)

    # TSM occupies a dense lower band rather than three oversized empty boxes.
    tsm = FancyBboxPatch((2.68, 0.88), 8.85, 1.55,
                         boxstyle='round,pad=0.04',
                         fc='#CBD9EE', ec='#315F9E', lw=1.35, zorder=2)
    ax.add_patch(tsm)
    ax.text(7.10, 2.22, 'TSM - Tiered Storage Manager',
            ha='center', fontsize=11.0, fontweight='bold', zorder=3)
    tier_specs = [
        (2.90, 2.35, 'L1  GPU HBM', 'hot blocks  |  ~2 TB/s'),
        (5.65, 2.35, 'L2  CPU DRAM', 'warm blocks  |  ~50 GB/s'),
        (8.40, 2.60, 'L3  NVMe SSD', 'cold blocks  |  ~7 GB/s'),
    ]
    for x, w, name, sub in tier_specs:
        fancy_box(ax, x, 1.08, w, 0.88, name, sub,
                  fc='white', ec='#52657F', fontsize=10.0,
                  subfontsize=8.5, lw=1.0)
    arrow(ax, 5.27, 1.70, 5.62, 1.70, lw=1.05, ms=8)
    arrow(ax, 5.62, 1.35, 5.27, 1.35, lw=1.05, ms=8)
    arrow(ax, 8.02, 1.70, 8.37, 1.70, lw=1.05, ms=8)
    arrow(ax, 8.37, 1.35, 8.02, 1.35, lw=1.05, ms=8)
    ax.text(5.45, 1.88, 'demote', ha='center', fontsize=8.8)
    ax.text(5.45, 1.16, 'promote', ha='center', fontsize=8.8)
    ax.text(8.20, 1.88, 'demote', ha='center', fontsize=8.8)
    ax.text(8.20, 1.16, 'promote', ha='center', fontsize=8.8)

    # Numbered request flow.
    arrow(ax, 1.38, 5.86, 1.38, 5.20, lw=1.3, ms=10)
    ax.text(0.82, 5.55, 'request', fontsize=11.0)
    circle_num(ax, 2.50, 4.02, 1, r=0.15, fontsize=10)
    arrow(ax, 2.35, 3.78, 2.68, 3.78, lw=1.3, color='black', ms=10)
    circle_num(ax, 6.20, 4.02, 2, r=0.15, fontsize=10)
    arrow(ax, 6.06, 3.78, 6.38, 3.78, lw=1.3, color='black', ms=10)
    circle_num(ax, 9.08, 4.02, 3, r=0.15, fontsize=10)
    arrow(ax, 8.94, 3.78, 9.25, 3.78, lw=1.3, color='black', ms=10)
    circle_num(ax, 9.38, 3.12, 4, r=0.15, fontsize=10)
    arrow(ax, 9.62, 3.62, 9.62, 2.45, lw=1.3, color='black', ms=10)

    # Rejected requests return directly to the serving engine for full prefill.
    arrow(ax, 11.10, 3.30, 11.10, 5.86, lw=1.20, ms=10)
    ax.text(10.98, 5.55, 'prefill',
            fontsize=10.2, color='#8A5E2D',
            bbox=dict(fc='white', ec='none', pad=0.4, alpha=0.9))

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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))   # larger for readability
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
        ax2.plot(xs, ttfts, color='black', marker='D', markersize=7,
                 linewidth=2.0, linestyle='--', zorder=4, label='TTFT (ms)')
        ax2.set_ylabel('Mean TTFT (ms)', fontsize=13)
        ax2.set_ylim(20, 52)
        ax2.tick_params(axis='y', labelsize=12)
        ax.set_ylabel('Cache Hit Rate (%)', fontsize=13)
        ax.set_ylim(0, 120)
        ax.set_xticks(xs)
        ax.set_xticklabels(['No\nCache', 'Exact\nCache', 'Sem-\nShareKV', 'Semanti-\nCache'],
                           fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        # subfigure title below x-axis
        ax.text(0.5, -0.22, title, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top', clip_on=False)

    fig.legend(handles=legend_patches() + [
        plt.Line2D([0], [0], color='black', marker='D', markersize=6,
                   linestyle='--', label='TTFT (ms)')
    ], loc='upper center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=11)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, 'fig3_end_to_end')

    # ── Also save individual panels for LaTeX \subfigure ──────────
    for panel_idx, (hits, ttfts, suffix) in enumerate([
        (rag_hit,  rag_ttft,  'fig3a_rag'),
        (lms_hit,  lms_ttft,  'fig3b_lmsys'),
    ]):
        fig_p, ax_p = plt.subplots(1, 1, figsize=(5.2, 4.4))
        ax2_p = ax_p.twinx()
        bars = ax_p.bar(xs, hits, w, color=CL, edgecolor='black', linewidth=0.9, zorder=3)
        for b, h in zip(bars, HL):
            b.set_hatch(h)
        ax2_p.plot(xs, ttfts, color='black', marker='D', markersize=7,
                   linewidth=2.0, linestyle='--', zorder=4)
        ax2_p.set_ylabel('Mean TTFT (ms)', fontsize=13)
        ax2_p.set_ylim(20, 52)
        ax2_p.tick_params(axis='y', labelsize=12)
        ax_p.set_ylabel('Cache Hit Rate (%)', fontsize=13)
        ax_p.set_ylim(0, 120)
        ax_p.set_xticks(xs)
        ax_p.set_xticklabels(['No\nCache', 'Exact\nCache', 'Sem-\nShareKV', 'Semanti-\nCache'],
                             fontsize=12)
        ax_p.tick_params(axis='y', labelsize=12)
        ax_p.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax_p.set_axisbelow(True)
        plt.tight_layout()
        save(fig_p, suffix)


# ═══════════════════════════════════════════════════════════════════
# Fig 4 — Tau sweep
# ═══════════════════════════════════════════════════════════════════
def fig_tau_sweep():
    data  = load('tau_sweep.json')
    taus  = [d['tau']          for d in data]
    hits  = [d['hit_rate']*100 for d in data]
    berts = [d['bertscore']    for d in data]

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    bw = 0.038
    bars = ax1.bar(taus, hits, bw, color=C['ours'], edgecolor='black',
                   hatch=H['ours'], linewidth=0.9, label='Hit Rate (%)')
    ax1.set_xlabel('Reuse Safety Threshold ($\\tau$)', fontsize=14)
    ax1.set_ylabel('Cache Hit Rate (%)', fontsize=14)
    ax1.set_ylim(60, 105)
    ax1.set_xticks(taus)
    ax1.set_xticklabels([str(t) for t in taus], fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(taus, berts, color='black', marker='o', markersize=6,
             linewidth=2.0, label='BERTScore')
    ax2.set_ylabel('Generation Quality (BERTScore)', fontsize=14)
    ax2.set_ylim(0.800, 0.895)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.grid(False)

    ax1.axvline(0.85, color='gray', linestyle=':', linewidth=1.2)
    ax1.text(0.855, 103, 'default $\\tau$', ha='left', fontsize=11, color='#555')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left', frameon=True, framealpha=0.9, fontsize=12)
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
    axes[0].set_xlabel('Number of Requests', fontsize=12)
    axes[0].set_ylabel('Cumulative Hit Rate (%)', fontsize=12)
    axes[0].set_ylim(50, 100); axes[0].set_xlim(0, 1050)
    axes[0].tick_params(labelsize=10.5)
    axes[0].legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=10.5)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.4)

    axes[1].plot(x, ttft_n, color='#CC3333', marker='s', markersize=3.5,
                 linewidth=1.8, linestyle='--', label='No Cache')
    axes[1].plot(x, ttft_s, color='#2255AA', marker='o', markersize=3.5,
                 linewidth=2.0, label='SemantiCache')
    axes[1].fill_between(x, ttft_n, ttft_s, alpha=0.15, color='#BBCCFF',
                         label='TTFT savings')
    axes[1].set_xlabel('Number of Requests', fontsize=12)
    axes[1].set_ylabel('Mean TTFT (ms)', fontsize=12)
    axes[1].set_ylim(28, 46); axes[1].set_xlim(0, 1050)
    axes[1].tick_params(labelsize=10.5)
    axes[1].legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10.5)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.4)

    # subfigure labels — below each panel, centered (matches fig1 style)
    for ax, lbl in zip(axes, ['(a)', '(b)']):
        ax.text(0.5, -0.18, lbl, transform=ax.transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top', clip_on=False)

    plt.tight_layout()
    save(fig, 'fig5_scalability')

    # ── Also save individual panels for LaTeX \subfigure ──────────
    # Panel (a): cumulative hit rate
    fig_a, ax_a = plt.subplots(1, 1, figsize=(4.8, 3.6))
    ax_a.plot(x, hits, color='#2255AA', marker='o', markersize=3.5,
              linewidth=2.0, label='SemantiCache')
    ax_a.fill_between(x, hits, 50, alpha=0.12, color='#BBCCFF')
    ax_a.set_xlabel('Number of Requests', fontsize=12)
    ax_a.set_ylabel('Cumulative Hit Rate (%)', fontsize=12)
    ax_a.set_ylim(50, 100); ax_a.set_xlim(0, 1050)
    ax_a.tick_params(labelsize=10.5)
    ax_a.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=10.5)
    ax_a.yaxis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    save(fig_a, 'fig5a_hit_rate')

    # Panel (b): TTFT comparison
    fig_b, ax_b = plt.subplots(1, 1, figsize=(4.8, 3.6))
    ax_b.plot(x, ttft_n, color='#CC3333', marker='s', markersize=3.5,
              linewidth=1.8, linestyle='--', label='No Cache')
    ax_b.plot(x, ttft_s, color='#2255AA', marker='o', markersize=3.5,
              linewidth=2.0, label='SemantiCache')
    ax_b.fill_between(x, ttft_n, ttft_s, alpha=0.15, color='#BBCCFF',
                      label='TTFT savings')
    ax_b.set_xlabel('Number of Requests', fontsize=12)
    ax_b.set_ylabel('Mean TTFT (ms)', fontsize=12)
    ax_b.set_ylim(28, 46); ax_b.set_xlim(0, 1050)
    ax_b.tick_params(labelsize=10.5)
    ax_b.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10.5)
    ax_b.yaxis.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    save(fig_b, 'fig5b_ttft')


# ═══════════════════════════════════════════════════════════════════
# Fig 6 — Ablation
# ═══════════════════════════════════════════════════════════════════
def fig_ablation():
    data  = load('ablation.json')
    names = ['Full\nSystem', 'w/o\nSemantic', 'w/o\nQBR', 'w/o\nTSM', 'w/o\nTIL']
    exact = [d['exact_hit_rate']*100    for d in data]
    sem   = [d['semantic_hit_rate']*100 for d in data]
    total = [d['hit_rate']*100          for d in data]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    xs = np.arange(5); w = 0.48

    b1 = ax.bar(xs, exact, w, label='Exact Hit', color=C['exact'],
                edgecolor='black', hatch=H['exact'], linewidth=0.9)
    b2 = ax.bar(xs, sem, w, bottom=exact, label='Semantic Hit',
                color=C['ours'], edgecolor='black', hatch=H['ours'], linewidth=0.9)

    for i, t in enumerate(total):
        ax.text(xs[i], t+1.4, f'{t:.0f}%',
                ha='center', va='bottom', fontsize=14.0, fontweight='bold')

    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=13.0)
    ax.tick_params(axis='y', labelsize=13.0)
    ax.set_ylabel('Cache Hit Rate (%)', fontsize=14.0)
    ax.set_ylim(0, 112)
    # Keep the legend above the plot so the visual reads top-to-bottom.
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=2, frameon=False, fontsize=13.0,
              handlelength=1.7, handleheight=1.0, columnspacing=1.6)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
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
        if t > 0.35:
            ax.text(left + t/2, 0, f'{t:.3f}ms',
                    ha='center', va='center', fontsize=10, fontweight='bold')
        elif t > 0.05:
            ax.text(left + t/2, 0.52, f'{t:.3f} ms',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.95))
        left += t
    exact_mid = times[0] / 2
    qbr_mid = sum(times[:-1]) + times[-1] / 2
    ax.annotate(f'Exact Hash\n{times[0]:.3f} ms',
                xy=(exact_mid, 0.39), xytext=(0.85, 0.72),
                ha='center', va='center', fontsize=9.5, fontweight='bold',
                bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.9,
                                shrinkA=1, shrinkB=1))
    ax.annotate(f'QBR Check\n{times[-1]:.3f} ms',
                xy=(qbr_mid, 0.39), xytext=(10.00, 0.72),
                ha='center', va='center', fontsize=9.5, fontweight='bold',
                bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.9,
                                shrinkA=1, shrinkB=1))
    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_xlim(0, 11.0); ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=10.5)
    ax.set_ylim(-0.72, 1.00)
    ax.text(left+0.32, 0, f'Total:\n{left:.2f} ms',
            va='center', fontsize=10.5, fontweight='bold')
    ax.legend(loc='upper center', ncol=4, frameon=False,
              bbox_to_anchor=(0.5, 1.42), fontsize=10.5)
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
