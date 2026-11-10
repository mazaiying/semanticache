#!/usr/bin/env python3
"""Generate the ICDE paper figures from the completed SemantiCache runs.

This script is intentionally read-only with respect to experiment outputs.  It
loads ``results/*/summary.json`` and ``results/q7_conc/results.json``.  The only
legacy result that is not present as a JSON file (the LMSYS trace table) is
parsed from ``paper/results_tables.md``, the consolidated source of truth.

No experiment is launched and no missing value is imputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.transforms import Bbox


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RESULTS = HERE / "results" if (HERE / "results").is_dir() else ROOT / "code" / "results"
DEFAULT_OUTPUT = (ROOT / "paper" / "figures") if (ROOT / "paper").is_dir() else (HERE / "figures")
DEFAULT_TABLES = (ROOT / "paper" / "results_tables.md") if (ROOT / "paper" / "results_tables.md").is_file() else (HERE / "results_tables.md")

# Palette sampled from ``semanticache (2).pdf``: the original paper maps the
# four principal systems to red, green, yellow, and blue pastel fills.  Black
# borders and the original hatch families preserve the mapping in grayscale.
COLORS = {
    "full": "#FFBBBB",
    "apc": "#BBDDBB",
    "gptcache": "#DCCBE8",
    "cacheblend": "#BBDDBB",
    "naive_sem": "#EEEEAA",
    "semanticache_r0": "#DDE6FF",
    "semanticache_halo0": "#CCD9FF",
    "semanticache": "#BBCCFF",
}
LINE_COLORS = {
    "full": "#C83E3E",
    "apc": "#467246",
    "gptcache": "#7A4F91",
    "cacheblend": "#467246",
    "naive_sem": "#837F18",
    "semanticache_r0": "#5D7FAF",
    "semanticache_halo0": "#3F6DAE",
    "semanticache": "#235DB2",
}
HATCHES = {
    "full": "////",
    "apc": "xxxx",
    "gptcache": "oo",
    "cacheblend": "xxxx",
    "naive_sem": "\\\\",
    "semanticache_r0": "xx",
    "semanticache_halo0": "//",
    "semanticache": "....",
}
LABELS = {
    "full": "Full",
    "apc": "Prefix cache",
    "gptcache": "Response cache",
    "cacheblend": "CacheBlend-style",
    "naive_sem": "Naive semantic",
    "semanticache_r0": r"SemantiCache ($r=0$)",
    "semanticache_halo0": "SemantiCache (no halo)",
    "semanticache": "SemantiCache",
}
MARKERS = {
    "full": "s",
    "apc": "X",
    "cacheblend": "^",
    "naive_sem": "D",
    "semanticache": "o",
}

plt.rcParams.update(
    {
        # The lab's ICDE/SIGMOD plotting convention uses Arial inside figures
        # even though the IEEE body text is Times. Keep Type-42 embedding so
        # labels remain vector text in the submission PDF.
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.axisbelow": True,
        "grid.alpha": 0.42,
        "grid.linewidth": 0.55,
        "grid.linestyle": "--",
        "hatch.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required completed result is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_summary(results_dir: Path, run: str) -> dict:
    return load_json(results_dir / run / "summary.json")


def metric(summary: Mapping, method: str, key: str, kind: str = "ALL") -> float:
    row = summary.get(f"{method}|{kind}")
    if row is None or key not in row:
        raise KeyError(f"Missing completed metric {method}|{kind}:{key}")
    return float(row[key])


def aggregate(
    results_dir: Path,
    runs: Sequence[str],
    methods: Sequence[str],
    key: str,
    kind: str = "ALL",
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [
            [metric(load_summary(results_dir, run), method, key, kind) for method in methods]
            for run in runs
        ],
        dtype=float,
    )
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
    return mean, std


def save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"saved {output_dir / filename}")


def save_axes_panels(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    output_dir: Path,
    filenames: Sequence[str],
    pad: float = 0.065,
) -> None:
    """Export individual axes for LaTeX ``subfigure`` composition.

    Panel letters are intentionally omitted here.  The ICDE manuscript places
    them underneath each panel through ``\\subfigure`` so labels, spacing, and
    font size are controlled by the conference template.
    """
    if len(axes) != len(filenames):
        raise ValueError("Each exported axis needs one output filename")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()
    s_boxes = [ax.get_window_extent(renderer).transformed(inv) for ax in axes]
    t_boxes = [
        Bbox.union([b for b in [ax.get_window_extent(renderer), ax.get_tightbbox(renderer)] if b is not None]).transformed(inv)
        for ax in axes
    ]
    max_left = max(s.x0 - t.x0 for s, t in zip(s_boxes, t_boxes)) + pad
    max_right = max(t.x1 - s.x1 for s, t in zip(s_boxes, t_boxes)) + pad
    max_bottom = max(s.y0 - t.y0 for s, t in zip(s_boxes, t_boxes)) + pad
    max_top = max(t.y1 - s.y1 for s, t in zip(s_boxes, t_boxes)) + pad
    for s, filename in zip(s_boxes, filenames):
        bbox = Bbox.from_extents(
            s.x0 - max_left, s.y0 - max_bottom, s.x1 + max_right, s.y1 + max_top
        )
        fig.savefig(output_dir / filename, bbox_inches=bbox, pad_inches=0)
        print(f"saved {output_dir / filename}")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.035, label, transform=ax.transAxes, fontweight="bold",
            fontsize=9.2, ha="left", va="bottom", clip_on=False)


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, linestyle="--", color="#D3D3D3",
            alpha=0.80, linewidth=0.50)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.9)
    ax.tick_params(width=0.8, color="#333333")


def styled_bar(ax: plt.Axes, x, values, width: float, method: str, **kwargs):
    return ax.bar(
        x, values, width, color=COLORS[method], edgecolor="black",
        linewidth=0.75, hatch=HATCHES[method], label=LABELS[method],
        zorder=3, **kwargs,
    )


def legend_handles(methods: Sequence[str]) -> list[Patch]:
    return [
        Patch(facecolor=COLORS[m], edgecolor="black", hatch=HATCHES[m],
              linewidth=0.75, label=LABELS[m])
        for m in methods
    ]


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float,
          color: str = "black", linewidth: float = 1.05,
          connectionstyle: str = "arc3", linestyle: str = "-",
          mutation_scale: float = 7.5) -> None:
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=linewidth, color=color, linestyle=linestyle,
        connectionstyle=connectionstyle, shrinkA=0, shrinkB=0, zorder=6,
    ))


def numbered_circle(ax: plt.Axes, x: float, y: float, number: int) -> None:
    circ = plt.Circle((x, y), 0.105, facecolor="#333333", edgecolor="black",
                      linewidth=0.6, zorder=8)
    ax.add_patch(circ)
    ax.text(x, y, str(number), ha="center", va="center", color="white",
            fontsize=6.6, fontweight="bold", zorder=9)


def labeled_box(
    ax: plt.Axes, x: float, y: float, width: float, height: float,
    title: str, body: str = "", face: str = "#F8F8F8",
    edge: str = "#555555", linewidth: float = 0.9,
    dashed: bool = False, title_size: float = 7.6,
    body_size: float = 6.5,
) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.035",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
        linestyle="--" if dashed else "-", zorder=2,
    ))
    ty = y + height * (0.64 if body else 0.50)
    ax.text(x + width / 2, ty, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", zorder=4)
    if body:
        ax.text(x + width / 2, y + height * 0.28, body, ha="center", va="center",
                fontsize=body_size, color="#333333", linespacing=1.25, zorder=4)


def parse_lmsys_table(results_tables: Path) -> list[tuple[str, float, float]]:
    text = results_tables.read_text(encoding="utf-8")
    try:
        section = text.split("## D. LMSYS-Chat real-trace reuse", 1)[1].split("## E.", 1)[0]
    except IndexError as exc:
        raise ValueError("Could not parse Table D from results_tables.md")
    rows = []
    for line in section.splitlines():
        if not line.startswith("|") or "method" in line or set(line) <= {"|", "-", " "}:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5:
            continue
        rows.append((cells[0], float(cells[1]), float(cells[4])))
    if len(rows) != 4:
        raise ValueError(f"Expected 4 LMSYS rows, found {len(rows)}")
    return rows


def parse_markdown_table(results_tables: Path, section: str) -> list[dict[str, str]]:
    """Read the first Markdown table in one results_tables.md section.

    The supplemental figures deliberately parse the paper's consolidated
    source of truth instead of duplicating G--N values in plotting code.
    """
    text = results_tables.read_text(encoding="utf-8")
    marker = f"## {section}."
    if marker not in text:
        raise ValueError(f"Could not find section {section} in {results_tables}")
    block = text.split(marker, 1)[1]
    if "\n## " in block:
        block = block.split("\n## ", 1)[0]
    lines = [line for line in block.splitlines() if line.startswith("|")]
    if len(lines) < 3:
        raise ValueError(f"Could not parse a table from section {section}")
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return rows


def as_float(value: str) -> float:
    return float(value.replace("%", "").replace("×", ""))


def _draw_cylinder(ax: plt.Axes, cx: float, by: float, w: float, h: float,
                   face: str = "white", edge: str = "black", lw: float = 1.0,
                   bands: int = 2) -> None:
    """Draw a database/cache cylinder icon in Prof. Zhang's HARMONY style."""
    from matplotlib.patches import Ellipse
    ry = h * 0.16
    body_h = h - 2 * ry
    ax.add_patch(plt.Rectangle((cx - w / 2, by + ry), w, body_h,
                               facecolor=face, edgecolor="none", zorder=3))
    ax.add_patch(Ellipse((cx, by + ry), w, 2 * ry,
                         facecolor=face, edgecolor=edge, linewidth=lw, zorder=3))
    ax.plot([cx - w / 2, cx - w / 2], [by + ry, by + ry + body_h],
            color=edge, linewidth=lw, zorder=4)
    ax.plot([cx + w / 2, cx + w / 2], [by + ry, by + ry + body_h],
            color=edge, linewidth=lw, zorder=4)
    for b in range(1, bands + 1):
        yy = by + ry + body_h * (b / (bands + 1))
        theta = np.linspace(np.pi, 2 * np.pi, 40)
        ax.plot(cx + (w / 2) * np.cos(theta), yy + ry * np.sin(theta),
                color=edge, linewidth=lw * 0.85, zorder=4)
    ax.add_patch(Ellipse((cx, by + ry + body_h), w, 2 * ry,
                         facecolor=face, edgecolor=edge, linewidth=lw, zorder=5))


def _draw_person_icon(ax: plt.Axes, cx: float, by: float, scale: float = 1.0) -> None:
    """Draw a clean line-art user icon as in HARMONY Fig. 1 & Fig. 3."""
    head_r = 0.15 * scale
    ax.add_patch(plt.Circle((cx, by + 0.52 * scale), head_r,
                            facecolor="white", edgecolor="black", linewidth=1.15, zorder=5))
    theta = np.linspace(0, np.pi, 50)
    bw = 0.28 * scale
    bh = 0.28 * scale
    xs = cx + bw * np.cos(theta)
    ys = by + bh * np.sin(theta)
    ax.fill(xs, ys, facecolor="white", edgecolor="black", linewidth=1.15, zorder=5)


def figure0_paradigm_intro(output_dir: Path) -> None:
    """Three-way paradigm comparison in Prof. Zhang's HARMONY Fig. 1 style.

    Open white canvas (no outer card borders), top legend for arrow/block
    semantics, contiguous sharp-cornered token and layer-KV grids, long-shafted
    arrows, and plain-text bottom subcaptions (a), (b), (c).
    """
    fig, ax = plt.subplots(figsize=(7.50, 2.15))
    ax.set_axis_off()
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 3.92)

    # -------------------------------------------------------------------------
    # Top Shared Legend (HARMONY Fig. 1 style: arrows on left, block colors on right)
    # -------------------------------------------------------------------------
    ly = 3.68
    arrow(ax, 0.35, ly, 1.05, ly, color="black", linewidth=1.05, linestyle="--")
    ax.text(1.15, ly, "View lookup / KV copy", fontsize=6.9, va="center", ha="left")

    arrow(ax, 3.25, ly, 3.95, ly, color="black", linewidth=1.10, linestyle="-")
    ax.text(4.05, ly, "GPU prefill / repair", fontsize=6.9, va="center", ha="left")

    leg_boxes = [
        (6.05, "Reused KV", "#E8F4E5", ""),
        (8.05, "Recomputed KV", "#72B0E5", ""),
        (10.35, "Stale fact KV", "#F4A582", "//"),
    ]
    for lx, ltxt, lface, lhatch in leg_boxes:
        ax.add_patch(plt.Rectangle((lx, ly - 0.11), 0.42, 0.22,
                                   facecolor=lface, edgecolor="black",
                                   linewidth=0.9, hatch=lhatch, zorder=4))
        ax.text(lx + 0.50, ly, ltxt, fontsize=6.9, va="center", ha="left")

    def _draw_contiguous_strip(x0: float, y0: float, cw: float, ch: float,
                               cells: list[tuple[str, str]]) -> None:
        for i, (txt, face) in enumerate(cells):
            cx = x0 + i * cw
            ax.add_patch(plt.Rectangle((cx, y0), cw, ch, facecolor=face,
                                       edgecolor="black", linewidth=0.95, zorder=4))
            ax.text(cx + cw / 2, y0 + ch / 2, txt, ha="center", va="center",
                    fontsize=6.6, fontweight="bold", color="black", zorder=6)

    def _draw_kv_grid(x0: float, y0: float, cw: float, rh: float,
                      grid_spec: list[list[tuple[str, str]]]) -> None:
        """Draw a 3-layer x 4-token contiguous sharp-cornered KV matrix."""
        row_labels = ["L1", "L2", "L3"]
        for r, row in enumerate(grid_spec):
            ry = y0 + (2 - r) * rh
            ax.text(x0 - 0.10, ry + rh / 2, row_labels[r], ha="right", va="center",
                    fontsize=6.2, fontweight="bold", color="#333333")
            for c, (face, hatch) in enumerate(row):
                cx = x0 + c * cw
                ax.add_patch(plt.Rectangle(
                    (cx, ry), cw, rh, facecolor=face, edgecolor="black",
                    linewidth=0.9, hatch=hatch, zorder=4,
                ))

    # Three paradigm x-origins (open canvas, subtle vertical dashed separator)
    ax.plot([4.12, 4.12], [0.15, 3.40], color="#CCCCCC", linestyle="--", linewidth=0.8)
    ax.plot([8.32, 8.32], [0.15, 3.40], color="#CCCCCC", linestyle="--", linewidth=0.8)

    paradigms = [
        (0.38, "(a) Exact-chunk reuse (42.0\u201345.0% computed)"),
        (4.56, "(b) Whole-view semantic (Stale F1: 0.24\u20130.51)"),
        (8.74, "(c) SemantiCache token-delta (19.9\u201321.1% computed)"),
    ]

    for idx, (px, subcap) in enumerate(paradigms):
        cw = 0.72
        gx = px + 0.52
        # 1. Top: Cached Base View B (contiguous token strip)
        _draw_contiguous_strip(gx, 3.08, cw, 0.27, [
            ("a", "white"), ("old", "white"), ("c", "white"), ("d", "white"),
        ])
        ax.text(gx - 0.08, 3.215, r"Base $B$:", ha="right", va="center", fontsize=6.4, fontweight="bold")

        # 2. Target Chunk B' (contiguous token strip with 'new' at pos 2)
        _draw_contiguous_strip(gx, 2.38, cw, 0.27, [
            ("a", "white"), ("new", "#FFF2CC"), ("c", "white"), ("d", "white"),
        ])
        ax.text(gx - 0.08, 2.515, r"Target $B'$:", ha="right", va="center", fontsize=6.4, fontweight="bold")

        # Long vertical dashed arrow from Base B ('old') to Target B' ('new') (shaft = 0.43 units!)
        arrow(ax, gx + 1.5 * cw, 3.08, gx + 1.5 * cw, 2.65, color="black", linewidth=1.0, linestyle="--")

        # 3. Middle gate label + long vertical arrows into Layer-Token KV grid (y: 2.38 -> 1.56, shaft = 0.82 units!)
        gy_top = 1.56
        rh = 0.32
        gy_bot = gy_top - 3 * rh  # 0.60

        if idx == 0:
            # (a) Hash miss -> all 4 columns recomputed via solid arrows
            ax.add_patch(plt.Rectangle((gx + 0.18, 1.88), 2.52, 0.25,
                                       facecolor="white", edgecolor="black", linewidth=0.85, zorder=7))
            ax.text(gx + 1.44, 2.005, r"Hash Miss: $H(B') \neq H(B)$",
                    ha="center", va="center", fontsize=6.3, fontweight="bold", zorder=8)
            for c in range(4):
                cx = gx + (c + 0.5) * cw
                arrow(ax, cx, 2.38, cx, gy_top, color="black", linewidth=1.0, linestyle="-")
            grid = [[("#72B0E5", "") for _ in range(4)] for _ in range(3)]
            _draw_kv_grid(gx, gy_bot, cw, rh, grid)

        elif idx == 1:
            # (b) Whole-view semantic -> all 4 columns copied via dashed arrows, col 1 is stale!
            ax.add_patch(plt.Rectangle((gx + 0.14, 1.88), 2.60, 0.25,
                                       facecolor="white", edgecolor="black", linewidth=0.85, zorder=7))
            ax.text(gx + 1.44, 2.005, r"Whole Copy: $\mathrm{sim}(B', B) \geq \tau$",
                    ha="center", va="center", fontsize=6.3, fontweight="bold", zorder=8)
            for c in range(4):
                cx = gx + (c + 0.5) * cw
                arrow(ax, cx, 2.38, cx, gy_top, color="black", linewidth=1.0, linestyle="--")
            grid = [
                [("#E8F4E5", ""), ("#F4A582", "//"), ("#E8F4E5", ""), ("#E8F4E5", "")]
                for _ in range(3)
            ]
            _draw_kv_grid(gx, gy_bot, cw, rh, grid)
            for r_idx in range(3):
                ax.text(gx + 1.5 * cw, gy_bot + (r_idx + 0.5) * rh, "old!",
                        ha="center", va="center", fontsize=6.0, fontweight="bold",
                        color="#6E1414", zorder=8)

        else:
            # (c) SemantiCache -> cols 0,3 dashed reuse; col 1 solid delta; col 2 L2-L3 halo/drift repair
            ax.add_patch(plt.Rectangle((gx + 0.04, 1.88), 2.80, 0.25,
                                       facecolor="white", edgecolor="black", linewidth=0.85, zorder=7))
            ax.text(gx + 1.44, 2.005, r"Align + RoPE + Repair $R\!=\!D\!\cup\!H_h\!\cup\!S_r$",
                    ha="center", va="center", fontsize=6.2, fontweight="bold", zorder=8)
            for c, lstyle in [(0, "--"), (1, "-"), (2, "-"), (3, "--")]:
                cx = gx + (c + 0.5) * cw
                arrow(ax, cx, 2.38, cx, gy_top, color="black", linewidth=1.0, linestyle=lstyle)
            grid = [
                [("#E8F4E5", ""), ("#72B0E5", ""), ("#E8F4E5", ""), ("#E8F4E5", "")],
                [("#E8F4E5", ""), ("#72B0E5", ""), ("#72B0E5", ""), ("#E8F4E5", "")],
                [("#E8F4E5", ""), ("#72B0E5", ""), ("#72B0E5", ""), ("#E8F4E5", "")],
            ]
            _draw_kv_grid(gx, gy_bot, cw, rh, grid)
            for r_idx in range(3):
                ax.text(gx + 1.5 * cw, gy_bot + (r_idx + 0.5) * rh, r"$D$",
                        ha="center", va="center", fontsize=6.2, fontweight="bold", color="black", zorder=8)
            ax.text(gx + 2.5 * cw, gy_bot + 1.5 * rh, r"$H_h$",
                    ha="center", va="center", fontsize=6.1, fontweight="bold", color="black", zorder=8)
            ax.text(gx + 2.5 * cw, gy_bot + 0.5 * rh, r"$S_r$",
                    ha="center", va="center", fontsize=6.1, fontweight="bold", color="black", zorder=8)

        # Plain-text subcaption at the bottom (exact HARMONY Fig. 1 style)
        ax.text(gx + 1.44, 0.22, subcap, ha="center", va="center",
                fontsize=6.8, fontweight="bold", color="black")

    fig.subplots_adjust(left=0.004, right=0.996, bottom=0.01, top=0.99)
    save(fig, output_dir, "fig1_paradigm_intro.pdf")


def figure1_motivation(results_dir: Path, output_dir: Path) -> None:
    """The two quantitative pain points required by the paper narrative.

    Panel (a) measures the redundant-prefill wall left by exact-chunk reuse.
    Panel (b) measures silent stale-state reuse after a factual edit.  Every
    value is aggregated from the completed runs used by Table A.
    """
    datasets = list(DATASET_RUNS)
    fig, axes = plt.subplots(1, 2, figsize=(6.60, 2.05))

    # Observation 1: an exact-chunk path still computes much more of a
    # near-duplicate prompt than token-delta maintenance.
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.32
    for i, method in enumerate(["cacheblend", "semanticache"]):
        means = [
            aggregate(results_dir, DATASET_RUNS[name], [method], "compute_frac")[0][0]
            for name in datasets
        ]
        bars = styled_bar(ax, x + (i - 0.5) * width, means, width, method)
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.014,
                    f"{100 * value:.0f}%", ha="center", va="bottom",
                    fontsize=7.6, fontweight="bold")
    ax.set_ylabel("Computed fraction", fontsize=9.8)
    ax.set_xticks(x, datasets, fontsize=8.8)
    ax.set_ylim(0, 0.68)
    ax.legend(handles=legend_handles(["cacheblend", "semanticache"]),
              frameon=False, fontsize=7.6, ncol=1, loc="upper right")
    style_axes(ax)

    # Observation 2: whole-view semantic substitution reproduces the old
    # answer more strongly than the post-edit target answer.
    ax = axes[1]
    methods = ["full", "naive_sem"]
    colors = {"full": COLORS["full"], "naive_sem": COLORS["naive_sem"]}
    edge = {"full": LINE_COLORS["full"], "naive_sem": LINE_COLORS["naive_sem"]}
    new_answer, stale_answer = [], []
    for method in methods:
        new_answer.append([
            aggregate(results_dir, DATASET_RUNS[name], [method], "f1", "edit")[0][0]
            for name in datasets
        ])
        stale_answer.append([
            aggregate(results_dir, DATASET_RUNS[name], [method], "stale_f1", "edit")[0][0]
            for name in datasets
        ])
    positions = np.arange(len(datasets))
    offsets = [-0.27, -0.09, 0.09, 0.27]
    series = [
        (new_answer[0], "Full: new", methods[0], ""),
        (new_answer[1], "Naive: new", methods[1], ""),
        (stale_answer[0], "Full: old", methods[0], "//"),
        (stale_answer[1], "Naive: old", methods[1], "//"),
    ]
    for offset, (values, label, method, hatch) in zip(offsets, series):
        ax.bar(positions + offset, values, 0.17, color=colors[method],
               edgecolor=edge[method], linewidth=0.75, hatch=hatch,
               label=label, zorder=3)
    ax.set_ylabel("Edit-answer F1")
    ax.set_xticks(positions, datasets)
    ax.set_ylim(0, 0.62)
    ax.legend(ncol=2, loc="upper right", frameon=False, fontsize=7.0,
              columnspacing=0.8, handlelength=1.3)
    style_axes(ax)

    fig.tight_layout(w_pad=1.65)
    save_axes_panels(
        fig, axes, output_dir,
        ["fig1a_redundant_prefill.pdf", "fig1b_stale_state.pdf"], pad=0.065,
    )
    save(fig, output_dir, "fig1_motivation.pdf")


def _module_badge(ax: plt.Axes, x: float, y: float, w: float, h: float,
                  num: int, title: str) -> None:
    """Draw a pure-white HARMONY Fig. 3 container with badge OUTSIDE and ABOVE."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.14",
        facecolor="white", edgecolor="black", linewidth=1.45, zorder=1,
    ))
    cx = x + 0.22
    cy = y + h + 0.24
    circ = plt.Circle((cx, cy), 0.145, facecolor="black", edgecolor="black",
                      linewidth=0.8, zorder=8)
    ax.add_patch(circ)
    ax.text(cx, cy, str(num), ha="center", va="center", color="white",
            fontsize=7.8, fontweight="bold", zorder=9)
    ax.text(cx + 0.20, cy, title, ha="left", va="center", color="black",
            fontsize=8.0, fontweight="bold", zorder=8)


def _stage_circle(ax: plt.Axes, x: float, y: float, num: int) -> None:
    """Draw a white hollow circle with black border for workflow stages (①-⑤)."""
    circ = plt.Circle((x, y), 0.135, facecolor="white", edgecolor="black",
                      linewidth=1.15, zorder=10)
    ax.add_patch(circ)
    ax.text(x, y, str(num), ha="center", va="center", color="black",
            fontsize=7.3, fontweight="bold", zorder=11)


def figure2_architecture(output_dir: Path) -> None:
    """Publication-grade system architecture in Prof. Zhang's HARMONY Fig. 3 style.

    Features:
    - Module badges (❶, ❷, ❸) placed OUTSIDE and ABOVE thick rounded white containers
    - Left-to-right main flow + bottom tiered-store loop with long-shafted arrows
    - Physical icons: User + Query Chunks (left), Magnifying Glass & Vector Grid (❶),
      Alignment Tree (between ❶ and ❷), Contiguous Token Grid & Pipeline Stages (❷),
      3 Database Cylinders + Cost Gate (❸ at bottom), Checkmark Results (right).
    """
    fig, ax = plt.subplots(figsize=(7.50, 2.55))
    ax.set_axis_off()
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 4.45)

    # =========================================================================
    # 0. Far Left: User & Retrieved Chunks [A, B', C] + Q (HARMONY Fig. 3 style)
    # =========================================================================
    _draw_person_icon(ax, 0.34, 2.52, scale=0.95)
    _draw_cylinder(ax, 0.88, 2.48, 0.44, 0.68, face="white", lw=1.1, bands=2)
    ax.text(0.62, 2.15, "Target prompt\n" + r"$[A, B', C] + Q$",
            ha="center", va="top", fontsize=6.6, fontweight="bold")

    # Long arrow from Input into Module ❶ (shaft = 0.56 units!)
    arrow(ax, 1.16, 2.82, 1.72, 2.82, linewidth=1.15, color="black")

    # =========================================================================
    # MODULE I (❶): Versioned View Discovery (Left Container)
    # =========================================================================
    _module_badge(ax, 1.72, 1.62, 2.96, 2.28, 1, "Versioned View Discovery")

    # Sub-icon 1 (Top): Exact Digest Lookup (box with Q + magnifying glass)
    ax.add_patch(plt.Rectangle((1.92, 2.94), 0.58, 0.64, facecolor="white",
                               edgecolor="black", linewidth=1.05, zorder=3))
    ax.text(2.21, 3.28, r"$H(B')$", ha="center", va="center", fontsize=6.5, fontweight="bold")
    ax.add_patch(plt.Circle((2.44, 3.02), 0.11, facecolor="white", edgecolor="black", linewidth=1.0, zorder=5))
    ax.plot([2.52, 2.64], [2.94, 2.82], color="black", linewidth=1.15, zorder=5)
    ax.text(3.64, 3.26, "Exact digest\n(A, C hit; B' miss)",
            ha="center", va="center", fontsize=6.6, fontweight="bold")

    # Sub-icon 2 (Bottom): Semantic Index & Candidate Base B (bar/vector icon)
    ax.add_patch(FancyBboxPatch((1.92, 1.84), 0.58, 0.72,
                                boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor="white", edgecolor="black", linewidth=1.05, zorder=3))
    for bi, bh in enumerate([0.42, 0.26, 0.18, 0.34]):
        ax.add_patch(plt.Rectangle((2.00 + bi * 0.11, 1.92), 0.07, bh,
                                   facecolor="#72B0E5" if bi == 0 else "white",
                                   edgecolor="black", linewidth=0.75, zorder=4))
    ax.text(3.64, 2.20, "Semantic index\n" + r"($\mathrm{sim}(B', B) \geq \tau$)",
            ha="center", va="center", fontsize=6.6, fontweight="bold")

    # Arrow from top check to bottom semantic index inside Module ❶
    arrow(ax, 2.21, 2.94, 2.21, 2.56, linewidth=0.95, color="black")

    # =========================================================================
    # Inter-Module Gap (❶ -> ❷): Stage ① & ② with Alignment Plan Icon
    # =========================================================================
    arrow(ax, 4.68, 2.82, 5.38, 2.82, linewidth=1.15, color="black")
    _stage_circle(ax, 4.98, 3.16, 1)

    # Alignment tree/mapping icon in the gap (like 'Query plan' in HARMONY Fig. 3)
    ax.add_patch(plt.Rectangle((5.58, 3.10), 0.42, 0.24, facecolor="white", edgecolor="black", lw=1.0, zorder=4))
    ax.text(5.79, 3.22, r"$B \leftrightarrow B'$", ha="center", va="center", fontsize=5.8, fontweight="bold")
    ax.plot([5.79, 5.79], [3.10, 2.92], color="black", lw=1.0)
    ax.plot([5.46, 6.12], [2.92, 2.92], color="black", lw=1.0)
    for cx, tok, col in [(5.46, "a", "#E8F4E5"), (5.79, "new", "#72B0E5"), (6.12, "c", "#E8F4E5")]:
        ax.plot([cx, cx], [2.92, 2.80], color="black", lw=1.0)
        ax.add_patch(plt.Rectangle((cx - 0.13, 2.56), 0.26, 0.24, facecolor=col, edgecolor="black", lw=0.85, zorder=4))
        ax.text(cx, 2.68, tok, ha="center", va="center", fontsize=5.6, fontweight="bold")
    ax.text(5.79, 2.34, "Token map", ha="center", va="center", fontsize=6.5, fontweight="bold")

    arrow(ax, 6.26, 2.82, 6.96, 2.82, linewidth=1.15, color="black")
    _stage_circle(ax, 6.60, 3.16, 2)

    # =========================================================================
    # MODULE II (❷): Token-Delta Maintenance Engine (Right Container)
    # =========================================================================
    _module_badge(ax, 6.96, 1.62, 4.18, 2.28, 2, "Token-Delta Maintenance Engine")

    # Three horizontal pipeline stage bars (exact HARMONY Fig. 3 right-box motif!)
    stages_spec = [
        (3.22, "white", r"Layer-1 drift check ($S_r$)"),
        (2.60, "#E5E7EB", "Pre-RoPE key rerotation"),
        (1.98, "white", r"Selective layers 2..L on $R$"),
    ]
    for sy, sface, slabel in stages_spec:
        ax.add_patch(FancyBboxPatch((7.14, sy), 2.40, 0.42,
                                    boxstyle="round,pad=0.01,rounding_size=0.07",
                                    facecolor=sface, edgecolor="black", linewidth=1.05, zorder=3))
        ax.add_patch(plt.Circle((7.26, sy + 0.21), 0.042, facecolor="white", edgecolor="black", lw=0.9, zorder=5))
        ax.add_patch(plt.Circle((9.42, sy + 0.21), 0.042, facecolor="white", edgecolor="black", lw=0.9, zorder=5))
        ax.text(8.34, sy + 0.21, slabel, ha="center", va="center", fontsize=6.3, fontweight="bold", zorder=5)

    _stage_circle(ax, 9.84, 3.43, 3)
    _stage_circle(ax, 9.84, 2.81, 4)
    _stage_circle(ax, 9.84, 2.19, 5)
    ax.text(10.54, 2.81, "Selective\ntoken-delta\nprefill", ha="center", va="center",
            fontsize=6.6, fontweight="bold")
    arrow(ax, 9.84, 3.25, 9.84, 2.97, linewidth=0.95, color="black")
    arrow(ax, 9.84, 2.63, 9.84, 2.35, linewidth=0.95, color="black")

    # =========================================================================
    # MODULE III (❸): Cost-Aware Planner & Tiered View Store (Bottom Pipeline)
    # =========================================================================
    # L-shaped routing from Input down to Cost Gate & View Store
    ax.plot([0.62, 0.62], [1.62, 0.56], color="black", linewidth=1.15)
    arrow(ax, 0.62, 0.56, 2.75, 0.56, linewidth=1.15, color="black")
    ax.text(1.68, 0.86, "Profitability gate\n" + r"$(\mathrm{sim}\geq\tau,\, \rho\geq\omega)$",
            ha="center", va="center", fontsize=6.4, fontweight="bold")

    # Module ❸ badge & 3 Database Cylinders at the bottom (exact HARMONY Fig. 3 style!)
    circ3 = plt.Circle((3.02, 1.18), 0.145, facecolor="black", edgecolor="black", lw=0.8, zorder=8)
    ax.add_patch(circ3)
    ax.text(3.02, 1.18, "3", ha="center", va="center", color="white", fontsize=7.8, fontweight="bold", zorder=9)
    ax.text(3.22, 1.18, "Planner & Tiered View Store", ha="left", va="center", fontsize=7.8, fontweight="bold")

    _draw_cylinder(ax, 3.18, 0.22, 0.48, 0.68, face="#E8F4E5", lw=1.05, bands=2)
    _draw_cylinder(ax, 3.74, 0.14, 0.48, 0.68, face="white", lw=1.05, bands=2)
    _draw_cylinder(ax, 4.30, 0.22, 0.48, 0.68, face="white", lw=1.05, bands=2)

    # Arrow from Alignment Plan down to View Store
    arrow(ax, 5.79, 2.18, 4.46, 0.96, linewidth=1.05, color="black")

    # Long dashed L-shaped arrow from View Store up into Module ❷
    ax.plot([4.62, 8.34], [0.56, 0.56], color="black", linewidth=1.15, linestyle="--")
    arrow(ax, 8.34, 0.56, 8.34, 1.62, linewidth=1.15, color="black", linestyle="--")
    ax.text(6.35, 0.80, "Fetch aligned pre-RoPE KV (GPU LRU / CPU INT8)",
            ha="center", va="center", fontsize=6.3, fontweight="bold")

    # =========================================================================
    # Far Right: Output Checkmarks & Assembled Request KV (HARMONY Fig. 3 style)
    # =========================================================================
    arrow(ax, 11.14, 2.82, 11.68, 2.82, linewidth=1.15, color="black")

    for ri, (ry, rtxt, rcol) in enumerate([
        (3.18, "Reuse A, C", "#E8F4E5"),
        (2.72, "Maint B' (R)", "#72B0E5"),
        (2.26, "Decode Q", "white"),
    ]):
        # Checkmark
        ax.plot([11.74, 11.82, 11.96], [ry + 0.14, ry + 0.04, ry + 0.24],
                color="black", linewidth=1.25, zorder=5)
        ax.add_patch(plt.Rectangle((12.04, ry), 0.50, 0.30, facecolor=rcol,
                                   edgecolor="black", linewidth=0.95, zorder=4))
    ax.text(12.14, 1.88, "Target KV\n& First Token", ha="center", va="top",
            fontsize=6.6, fontweight="bold")

    fig.subplots_adjust(left=0.004, right=0.996, bottom=0.01, top=0.99)
    save(fig, output_dir, "fig2_architecture.pdf")


DATASET_RUNS = {
    "HotpotQA": ["q1_hotpotqa_q7b_s0", "q4_hotpotqa_q7b_s1", "q4_hotpotqa_q7b_s2"],
    "2WikiMQA": ["q1_2wikimqa_q7b_s0", "q5_2wikimqa_q7b_s1"],
    "MuSiQue": ["q1_musique_q7b_s0", "q5_musique_q7b_s1"],
}


def figure3_main_results(results_dir: Path, output_dir: Path) -> None:
    methods = ["full", "cacheblend", "naive_sem", "semanticache"]
    fig, axes = plt.subplots(1, 2, figsize=(6.40, 2.25))
    x = np.arange(len(DATASET_RUNS))
    width = 0.19
    for i, method in enumerate(methods):
        f1_mean, f1_std, ttft_mean, ttft_std = [], [], [], []
        for runs in DATASET_RUNS.values():
            mean, std = aggregate(results_dir, runs, [method], "f1")
            f1_mean.append(mean[0]); f1_std.append(std[0])
            mean, std = aggregate(results_dir, runs, [method], "ttft_ms")
            ttft_mean.append(mean[0]); ttft_std.append(std[0])
        pos = x + (i - 1.5) * width
        styled_bar(axes[0], pos, f1_mean, width, method, yerr=f1_std,
                   capsize=2.0, error_kw={"elinewidth": 0.75, "capthick": 0.75})
        styled_bar(axes[1], pos, ttft_mean, width, method, yerr=ttft_std,
                   capsize=2.0, error_kw={"elinewidth": 0.75, "capthick": 0.75})
    axes[0].set_ylabel("QA F1", fontsize=10.0)
    axes[0].set_ylim(0, 0.66)
    axes[1].set_ylabel("TTFT (ms)", fontsize=10.0)
    axes[1].set_ylim(0, 800)
    for ax in axes:
        ax.set_xticks(x, DATASET_RUNS.keys(), fontsize=9.0)
        style_axes(ax)
    axes[0].legend(
        handles=legend_handles(methods), ncol=2, loc="upper left",
        fontsize=7.2, frameon=True, framealpha=0.92,
        handlelength=1.2, columnspacing=0.6,
    )
    axes[1].legend(
        handles=legend_handles(methods), ncol=2, loc="upper left",
        fontsize=7.2, frameon=True, framealpha=0.92,
        handlelength=1.2, columnspacing=0.6,
    )
    fig.tight_layout(w_pad=1.5)
    save_axes_panels(
        fig, axes, output_dir,
        ["fig3a_main_f1.pdf", "fig3b_main_ttft.pdf"],
    )
    save(fig, output_dir, "fig3_main_results.pdf")


def figure4_pareto(results_dir: Path, output_dir: Path) -> None:
    ratios = ["0.05", "0.10", "0.15", "0.30", "0.50"]
    methods = ["semanticache", "cacheblend"]
    fig, axes = plt.subplots(1, 2, figsize=(6.40, 2.25))
    for method in methods:
        f1, frac, ttft = [], [], []
        for ratio in ratios:
            summary = load_summary(results_dir, f"q2_ratio_{ratio}")
            f1.append(metric(summary, method, "f1"))
            frac.append(metric(summary, method, "compute_frac"))
            ttft.append(metric(summary, method, "ttft_ms"))
        for ax, xs, xlabel in [(axes[0], frac, "Computed-token fraction"), (axes[1], ttft, "TTFT (ms)")]:
            linestyle = "-" if method == "semanticache" else "--"
            ax.plot(xs, f1, marker=MARKERS[method], linestyle=linestyle,
                    linewidth=1.7, markersize=5.0, color=LINE_COLORS[method],
                    markerfacecolor=COLORS[method], markeredgecolor="black",
                    markeredgewidth=0.6, label=LABELS[method], zorder=4)
            for xv, yv, ratio in zip(xs, f1, ratios):
                ax.annotate(ratio, (xv, yv), xytext=(3, 3),
                            textcoords="offset points", fontsize=7.2)
            ax.set_xlabel(xlabel, fontsize=9.8)
            ax.set_ylabel("QA F1", fontsize=10.0)
            ax.set_ylim(0.41, 0.505)
            style_axes(ax, grid_axis="both")
    axes[0].legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.6)
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.6)
    fig.tight_layout(w_pad=1.6)
    save_axes_panels(
        fig, axes, output_dir,
        ["fig4a_compute_frontier.pdf", "fig4b_latency_frontier.pdf"],
    )
    save(fig, output_dir, "fig4_pareto.pdf")


def figure5_request_types(results_dir: Path, output_dir: Path) -> None:
    runs = DATASET_RUNS["HotpotQA"]
    methods = ["cacheblend", "semanticache"]
    kinds = ["orig", "reorder", "format", "edit"]
    fig, axes = plt.subplots(1, 2, figsize=(6.40, 2.25))
    x = np.arange(len(kinds))
    width = 0.34
    for i, method in enumerate(methods):
        ttft = [aggregate(results_dir, runs, [method], "ttft_ms", kind)[0][0] for kind in kinds]
        f1 = [aggregate(results_dir, runs, [method], "f1", kind)[0][0] for kind in kinds]
        pos = x + (i - 0.5) * width
        styled_bar(axes[0], pos, ttft, width, method)
        styled_bar(axes[1], pos, f1, width, method)
    axes[0].set_ylabel("TTFT (ms)", fontsize=10.0)
    axes[1].set_ylabel("QA F1", fontsize=10.0)
    axes[1].set_ylim(0, 0.68)
    for ax in axes:
        ax.set_xticks(x, ["Orig.", "Reorder", "Format", "Edit"], fontsize=8.8)
        style_axes(ax)
    axes[0].legend(handles=legend_handles(methods), loc="upper left",
                   frameon=True, framealpha=0.92, fontsize=7.6)
    axes[1].legend(handles=legend_handles(methods), loc="upper right",
                   frameon=True, framealpha=0.92, fontsize=7.6)
    fig.tight_layout(w_pad=1.6)
    save_axes_panels(
        fig, axes, output_dir,
        ["fig4c_request_ttft.pdf", "fig4d_request_f1.pdf"],
    )
    save(fig, output_dir, "fig5_request_types.pdf")


MODEL_RUNS = {
    "Qwen-3B": ["q3_hotpotqa_q3b"],
    "Qwen-7B": DATASET_RUNS["HotpotQA"],
    "Qwen-14B": ["q3_hotpotqa_q14b", "q5_hotpotqa_q14b_s1"],
    "Llama-8B": ["q3_hotpotqa_llama8b"],
    "Mistral-7B": ["q3_hotpotqa_mistral7b"],
}


def figure6_generalization(results_dir: Path, output_dir: Path) -> None:
    speedup, delta_f1 = [], []
    for runs in MODEL_RUNS.values():
        sc_ttft = aggregate(results_dir, runs, ["semanticache"], "ttft_ms")[0][0]
        cb_ttft = aggregate(results_dir, runs, ["cacheblend"], "ttft_ms")[0][0]
        sc_f1 = aggregate(results_dir, runs, ["semanticache"], "f1")[0][0]
        cb_f1 = aggregate(results_dir, runs, ["cacheblend"], "f1")[0][0]
        speedup.append(cb_ttft / sc_ttft)
        delta_f1.append(100 * (sc_f1 - cb_f1))
    fig, axes = plt.subplots(2, 2, figsize=(6.40, 4.30))
    x = np.arange(len(MODEL_RUNS))
    short_models = ["Q-3B", "Q-7B", "Q-14B", "L3-8B", "Mistral"]
    bars = styled_bar(axes[0, 0], x, speedup, 0.56, "semanticache")
    for bar, value in zip(bars, speedup):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2, value + 0.015,
            f"{value:.2f}" + r"$\times$", ha="center", fontsize=7.6,
            fontweight="bold",
        )
    axes[0, 0].axhline(1, color="#333333", linewidth=0.9)
    axes[0, 0].set_ylabel("Speedup over CB", fontsize=9.8)
    axes[0, 0].set_ylim(0.95, 1.62)
    for i, value in enumerate(delta_f1):
        method = "semanticache" if value >= 0 else "cacheblend"
        styled_bar(axes[0, 1], [x[i]], [value], 0.56, method)
    axes[0, 1].axhline(0, color="#333333", linewidth=0.9)
    axes[0, 1].set_ylabel(r"$\Delta$F1 (pts)", fontsize=9.8)
    axes[0, 1].set_ylim(-2.4, 1.8)
    for ax in axes[0]:
        ax.set_xlabel("Model family", fontsize=9.5)
        ax.set_xticks(x, short_models, fontsize=8.4)
        style_axes(ax)

    run_by_ctx = {2: "q4_ctx2048", 8: "q4_ctx8192", 16: "q5_ctx16384"}
    methods = ["full", "cacheblend", "semanticache"]
    for method in methods:
        ttft, f1 = [], []
        for run in run_by_ctx.values():
            summary = load_summary(results_dir, run)
            ttft.append(metric(summary, method, "ttft_ms"))
            f1.append(metric(summary, method, "f1"))
        linestyle = {"full": "--", "cacheblend": "-.", "semanticache": "-"}[method]
        for ax, values in [(axes[1, 0], ttft), (axes[1, 1], f1)]:
            ax.plot(
                run_by_ctx.keys(), values, marker=MARKERS[method],
                color=LINE_COLORS[method], linestyle=linestyle,
                linewidth=1.65, markersize=5.0, label=LABELS[method],
                markerfacecolor=COLORS[method], markeredgecolor="black",
                markeredgewidth=0.55, zorder=4,
            )
    axes[1, 0].set_ylabel("TTFT (ms)", fontsize=9.8)
    axes[1, 1].set_ylabel("QA F1", fontsize=9.8)
    axes[1, 1].set_ylim(0.25, 0.53)
    for ax in axes[1]:
        ax.set_xlabel("Max context (K tokens)", fontsize=9.5)
        ax.set_xticks(list(run_by_ctx))
        style_axes(ax, grid_axis="both")
    axes[1, 0].legend(frameon=True, framealpha=0.92, loc="upper left",
                      fontsize=7.4)
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    save_axes_panels(
        fig, list(axes.ravel()), output_dir,
        [
            "fig5a_model_speedup.pdf", "fig5b_model_f1.pdf",
            "fig5c_context_ttft.pdf", "fig5d_context_f1.pdf",
        ],
    )
    save(fig, output_dir, "fig6_generalization.pdf")


def figure7_context_scaling(results_dir: Path, output_dir: Path) -> None:
    run_by_ctx = {2: "q4_ctx2048", 8: "q4_ctx8192", 16: "q5_ctx16384"}
    methods = ["full", "cacheblend", "semanticache"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35))
    for method in methods:
        ttft, f1 = [], []
        for run in run_by_ctx.values():
            summary = load_summary(results_dir, run)
            ttft.append(metric(summary, method, "ttft_ms"))
            f1.append(metric(summary, method, "f1"))
        linestyle = {"full": "--", "cacheblend": "-.", "semanticache": "-"}[method]
        for ax, values in [(axes[0], ttft), (axes[1], f1)]:
            ax.plot(
                run_by_ctx.keys(), values, marker=MARKERS[method],
                color=LINE_COLORS[method], linestyle=linestyle,
                linewidth=1.65, markersize=5.0, label=LABELS[method],
                markerfacecolor=COLORS[method], markeredgecolor="black",
                markeredgewidth=0.6, zorder=4,
            )
    axes[0].set_ylabel("TTFT (ms)")
    axes[1].set_ylabel("QA F1")
    axes[1].set_ylim(0.25, 0.53)
    for ax in axes:
        ax.set_xlabel("Maximum context (K tokens)")
        ax.set_xticks(list(run_by_ctx))
        style_axes(ax, grid_axis="both")
    axes[0].legend(frameon=True, framealpha=0.92, loc="upper left")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout(w_pad=2.2)
    save(fig, output_dir, "fig7_context_scaling.pdf")


def figure8_concurrency(results_dir: Path, output_dir: Path) -> None:
    data = load_json(results_dir / "q7_conc" / "results.json")
    methods = ["full", "cacheblend", "semanticache"]
    rates = [0.4, 0.8, 1.2, 1.6, 2.0]
    fig, axes = plt.subplots(
        1, 3, figsize=(7.15, 1.95),
        gridspec_kw={"width_ratios": [1.05, 1.05, 0.85]},
    )
    for method in methods:
        p50 = [data[f"{method}|{rate:.1f}"]["lat_p50_ms"] / 1000 for rate in rates]
        p95 = [data[f"{method}|{rate:.1f}"]["lat_p95_ms"] / 1000 for rate in rates]
        linestyle = {"full": "--", "cacheblend": "-.", "semanticache": "-"}[method]
        for ax, values in [(axes[0], p50), (axes[1], p95)]:
            ax.plot(
                rates, values, marker=MARKERS[method],
                color=LINE_COLORS[method], linestyle=linestyle,
                linewidth=1.55, markersize=4.7, label=LABELS[method],
                markerfacecolor=COLORS[method], markeredgecolor="black",
                markeredgewidth=0.55, zorder=4,
            )
    for ax, ylabel in [(axes[0], "p50 latency (s)"),
                       (axes[1], "p95 latency (s)")]:
        ax.set_xlabel("Offered load (req/s)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(rates)
        style_axes(ax, grid_axis="both")
    closed = [data[f"{method}|closed_loop"]["throughput_rps"] for method in methods]
    for i, (method, value) in enumerate(zip(methods, closed)):
        bar = styled_bar(axes[2], [i], [value], 0.62, method)[0]
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, value + 0.03,
            f"{value:.2f}", ha="center", fontsize=7.4, fontweight="bold",
        )
    axes[2].set_ylabel("Max throughput (req/s)")
    axes[2].set_xticks(np.arange(len(methods)), ["Full", "CB", "SC"])
    axes[2].set_ylim(0, 2.08)
    style_axes(axes[2])
    axes[0].legend(frameon=True, framealpha=0.92, loc="upper left", fontsize=7.0)
    axes[1].legend(frameon=True, framealpha=0.92, loc="upper left", fontsize=7.0)
    fig.tight_layout(w_pad=1.45)
    save_axes_panels(
        fig, axes, output_dir,
        ["fig6a_p50.pdf", "fig6b_p95.pdf", "fig6c_throughput.pdf"],
    )
    save(fig, output_dir, "fig8_concurrency.pdf")


def figure9_ablation(results_dir: Path, output_dir: Path) -> None:
    ablations = ["semanticache_r0", "semanticache_halo0", "semanticache"]
    fig, axes = plt.subplots(1, 2, figsize=(6.40, 2.25))
    x = np.arange(len(DATASET_RUNS))
    width = 0.25
    for i, method in enumerate(ablations):
        means = [aggregate(results_dir, runs, [method], "f1")[0][0] for runs in DATASET_RUNS.values()]
        styled_bar(axes[0], x + (i - 1) * width, means, width, method)
    axes[0].set_xticks(x, DATASET_RUNS.keys(), fontsize=8.8)
    axes[0].set_ylabel("QA F1", fontsize=10.0)
    axes[0].set_ylim(0.20, 0.56)
    style_axes(axes[0])
    axes[0].legend(frameon=True, framealpha=0.92, fontsize=7.0,
                   loc="upper left", handlelength=1.4)

    halos = [0, 2, 4, 8, 16]
    f1, ttft = [], []
    for halo in halos:
        summary = load_summary(results_dir, f"q2_halo_{halo}")
        f1.append(metric(summary, "semanticache", "f1"))
        ttft.append(metric(summary, "semanticache", "ttft_ms"))
    ax = axes[1]
    ax2 = ax.twinx()
    line1 = ax.plot(
        halos, f1, "o-", color=LINE_COLORS["semanticache"], linewidth=1.65,
        markersize=4.8, markerfacecolor=COLORS["semanticache"],
        markeredgecolor="black", markeredgewidth=0.55, label="QA F1", zorder=4,
    )[0]
    line2 = ax2.plot(
        halos, ttft, "D--", color="#333333", linewidth=1.35,
        markersize=4.2, markerfacecolor="#EEEEAA", markeredgecolor="black",
        markeredgewidth=0.55, label="TTFT", zorder=4,
    )[0]
    ax.set_xlabel("Halo width $h$ (tokens/side)", fontsize=9.5)
    ax.set_ylabel("QA F1", color=LINE_COLORS["semanticache"], fontsize=9.8)
    ax2.set_ylabel("TTFT (ms)", color="#333333", fontsize=9.8)
    ax.set_xticks(halos)
    ax.set_ylim(0.45, 0.48)
    ax2.set_ylim(260, 345)
    style_axes(ax, grid_axis="both")
    ax2.spines["right"].set_color("#333333")
    ax2.spines["right"].set_linewidth(0.9)
    ax.tick_params(axis="y", colors=LINE_COLORS["semanticache"])
    ax2.tick_params(axis="y", colors="#333333")
    ax.legend([line1, line2], [line1.get_label(), line2.get_label()],
              frameon=True, framealpha=0.92, loc="lower right", fontsize=7.2)
    fig.tight_layout(w_pad=1.6)
    save_axes_panels(
        fig, axes, output_dir,
        ["fig9a_ablation_components.pdf", "fig9b_halo_sweep.pdf"],
    )
    save(fig, output_dir, "fig9_ablation_halo.pdf")


def figure10_supplemental_diagnostics(results_tables: Path, output_dir: Path) -> None:
    """Compact G--J evidence: costs, closest baseline, and robustness."""
    g = parse_markdown_table(results_tables, "G")
    h = parse_markdown_table(results_tables, "H")
    irows = parse_markdown_table(results_tables, "I")
    jrows = parse_markdown_table(results_tables, "J")
    nrows = parse_markdown_table(results_tables, "N")

    fig, axes = plt.subplots(2, 2, figsize=(6.40, 4.30))

    # (a) TTFT stage decomposition at 4K and 16K.
    ax = axes[0, 0]
    selected = [
        next(row for row in g if row["ctx"] == ctx and row["method"] == method
             and row["kind"] == "ALL")
        for ctx, method in [("4K", "cacheblend"), ("4K", "semanticache"),
                            ("16K", "cacheblend"), ("16K", "semanticache")]
    ]
    stage_cols = ["lookup (hash+index)", "token align + halo", "view fetch (H2D)",
                  "KV assemble", "prefill compute"]
    stage_labels = ["Lookup", "Align", "Fetch", "Assemble", "Prefill"]
    stage_colors = ["#D9D9D9", "#F2CC8F", "#9DC3E6", "#B4A7D6", "#7EA36A"]
    x = np.arange(len(selected))
    bottom = np.zeros(len(selected))
    for column, label, color in zip(stage_cols, stage_labels, stage_colors):
        values = [as_float(row[column]) for row in selected]
        ax.bar(x, values, 0.60, bottom=bottom, color=color, edgecolor="black",
               linewidth=0.55, label=label, zorder=3)
        bottom += values
    totals = [as_float(row["TTFT"]) for row in selected]
    for xi, total in zip(x, totals):
        ax.text(xi, total + 22, f"{total:.0f}", ha="center", fontsize=7.4, fontweight="bold")
    ax.set_xticks(x, ["CB\n4K", "SC\n4K", "CB\n16K", "SC\n16K"], fontsize=8.6)
    ax.set_ylabel("Stage time (ms)", fontsize=9.8)
    ax.set_ylim(0, 1240)
    ax.legend(ncol=3, loc="upper left", fontsize=6.6, frameon=False,
              columnspacing=0.6, handlelength=1.0)
    style_axes(ax)

    # (b) Chunk-view maintenance versus request-granularity diff.
    ax = axes[0, 1]
    paired_block = results_tables.read_text(encoding="utf-8").split(
        "Paired bootstrap (SemantiCache vs reqdiff):", 1
    )[1].split("## I.", 1)[0]
    paired_lines = [line for line in paired_block.splitlines() if line.startswith("|")]
    paired_header = [c.strip() for c in paired_lines[0].strip("|").split("|")]
    paired = []
    for line in paired_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(paired_header):
            paired.append(dict(zip(paired_header, cells)))
    speed = [as_float(row["TTFT ratio reqdiff/SC [95% CI]"].split()[0]) for row in paired]
    delta = [as_float(row["ΔF1 (SC − reqdiff) [95% CI]"].split()[0]) for row in paired]
    large = {row["method"]: row for row in nrows}
    speed.append(as_float(large["reqdiff"]["TTFT ms"]) /
                 as_float(large["semanticache"]["TTFT ms"]))
    delta.append(as_float(large["semanticache"]["F1"]) -
                 as_float(large["reqdiff"]["F1"]))
    names = ["Hotpot", "2Wiki", "MuSiQue", "150-smp"]
    bars = styled_bar(ax, np.arange(4), speed, 0.58, "semanticache")
    ax.axhline(1, color="#333333", linewidth=0.8)
    for bar, value, df1 in zip(bars, speed, delta):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035,
                f"{value:.2f}×\n{df1:+.3f}", ha="center", va="bottom",
                fontsize=6.8, fontweight="bold")
    ax.set_xticks(np.arange(4), names, fontsize=8.6)
    ax.set_ylabel("Speedup over reqdiff", fontsize=9.8)
    ax.set_ylim(0.9, 2.32)
    style_axes(ax)

    # (c) Cost gate under increasing revision magnitude.
    ax = axes[1, 0]
    drifts = sorted({as_float(row["drift frac"]) for row in irows})
    drift_idx = np.arange(len(drifts))
    for method, label, color, linestyle, marker in [
        ("cacheblend", "CacheBlend-style", LINE_COLORS["cacheblend"], "--", "^"),
        ("semanticache_always", r"SC ($\omega=0$)", "#B06B2E", ":", "s"),
        ("semanticache", r"SC ($\omega=0.5$)", LINE_COLORS["semanticache"], "-", "o"),
    ]:
        values = [as_float(next(row["TTFT ms"] for row in irows
                                if as_float(row["drift frac"]) == drift
                                and row["method"] == method)) for drift in drifts]
        ax.plot(drift_idx, values, linestyle=linestyle, marker=marker, color=color,
                linewidth=1.55, markersize=4.6, markeredgecolor="black",
                markeredgewidth=0.45, label=label, zorder=4)
    ax.set_xlabel("Rewritten-word fraction", fontsize=9.5)
    ax.set_ylabel("TTFT (ms)", fontsize=9.8)
    ax.set_xticks(drift_idx, [f"{d:.2f}" for d in drifts], fontsize=8.6)
    ax.legend(loc="lower right", fontsize=6.8, frameon=True, framealpha=0.92)
    style_axes(ax, grid_axis="both")

    # (d) How the near-duplicate share changes recomputation.
    ax = axes[1, 1]
    probs = sorted({as_float(row["fmt_prob"]) for row in jrows})
    for method, color, linestyle, marker in [
        ("cacheblend", LINE_COLORS["cacheblend"], "--", "^"),
        ("semanticache", LINE_COLORS["semanticache"], "-", "o"),
    ]:
        values = [as_float(next(row["computed frac"] for row in jrows
                                if as_float(row["fmt_prob"]) == prob
                                and row["method"] == method
                                and row["kind"] == "format")) for prob in probs]
        ax.plot(probs, values, linestyle=linestyle, marker=marker, color=color,
                linewidth=1.6, markersize=4.6, markerfacecolor=COLORS[method],
                markeredgecolor="black", markeredgewidth=0.5,
                label=LABELS[method], zorder=4)
    ax.set_xlabel("Near-duplicate probability", fontsize=9.5)
    ax.set_ylabel("Computed fraction", fontsize=9.8)
    ax.set_xticks(probs)
    ax.set_ylim(0.1, 1.07)
    ax.legend(loc="upper left", fontsize=7.0, frameon=False)
    style_axes(ax, grid_axis="both")

    fig.tight_layout(w_pad=1.5, h_pad=1.6)
    save_axes_panels(
        fig, list(axes.ravel()), output_dir,
        ["fig7a_breakdown.pdf", "fig7b_reqdiff.pdf",
         "fig7c_drift.pdf", "fig7d_fmtprob.pdf"], pad=0.065,
    )
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes.ravel()):
        panel_label(ax, label)
    save(fig, output_dir, "fig10_supplemental_diagnostics.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-tables", type=Path, default=DEFAULT_TABLES)
    args = parser.parse_args()

    figure0_paradigm_intro(args.output_dir)
    figure1_motivation(args.results_dir, args.output_dir)
    figure2_architecture(args.output_dir)
    figure3_main_results(args.results_dir, args.output_dir)
    figure4_pareto(args.results_dir, args.output_dir)
    figure5_request_types(args.results_dir, args.output_dir)
    figure6_generalization(args.results_dir, args.output_dir)
    figure8_concurrency(args.results_dir, args.output_dir)
    figure9_ablation(args.results_dir, args.output_dir)
    figure10_supplemental_diagnostics(args.results_tables, args.output_dir)


if __name__ == "__main__":
    main()
