#!/usr/bin/env python3
"""
Generate a 4×4 figure: 4 topologies (rows) × 4 network sizes (columns).
Y-axis is shared across all 16 panels so tick labels align everywhere.

Usage:
    python3 paper/generate_4by4.py
    python3 paper/generate_4by4.py --show
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ──────────────────────── rc overrides (publication style) ─────────

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":  "dejavuserif",
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.35,
    "ytick.minor.width": 0.35,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "xtick.minor.size":  2.0,
    "ytick.minor.size":  2.0,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "legend.frameon":    True,
    "legend.fancybox":   False,
    "legend.edgecolor":  "#888888",
    "legend.framealpha":  0.95,
})

# ──────────────────────── configuration ────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figs"

ALGORITHMS = ["bine", "glf", "pipe", "srda", "mpi", "bbs"]
N_VALUES = [128, 256, 512, 1024]
TOPOLOGIES = ["2Dmesh", "Butterfly", "Dragonfly", "FatTree"]

ALGO_STYLE = {
    "bine": ("BInE",      "#CD2032", "o",  "-", "#9A1624"),
    "glf":  ("GLF",       "#1560BD", "s",  "-", "#0E4382"),
    "bbs":  ("BBS",       "#1FAD3F", "^",  "-", "#14762B"),
    "srda": ("SRDA",      "#DAA520", "D",  "-", "#A07B18"),
    "mpi":  ("MPI_Bcast", "#8B45A6", "v",  "-", "#5E2E71"),
    "pipe": ("Pipeline",  "#E05500", "P",  "-", "#A03D00"),
}

TOPOLOGY_LABEL = {
    "2Dmesh":    "2D Mesh",
    "Butterfly": "Butterfly",
    "Dragonfly": "Dragonfly",
    "FatTree":   "Fat-Tree",
}

# ──────────────────────── helpers ──────────────────────────────────

def _bytes_label(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b / 1024:g} KB"
    else:
        return f"{b / 1024**2:g} MB"


def _time_label(y, _pos=None):
    if y <= 0:
        return ""
    if y < 1e-6:
        return f"{y*1e9:.0f} ns"
    if y < 1e-3:
        v = y * 1e6
        return f"{v:.1f} \u00b5s" if v < 100 else f"{v:.0f} \u00b5s"
    if y < 1:
        v = y * 1e3
        return f"{v:.2f} ms" if v < 1 else f"{v:.1f} ms"
    return f"{y:.2f} s"


def load_data(topology, algorithm, N):
    d = DATA / topology / algorithm
    if not d.is_dir():
        return None, None
    msgs, means = [], []
    for f in sorted(d.glob(f"N{N}_MSG*.json")):
        if "_R" in f.stem.split("MSG")[1]:
            continue
        with open(f) as fh:
            rec = json.load(fh)
        if rec["msg_bytes"] < 16384:
            continue
        msgs.append(rec["msg_bytes"])
        means.append(rec["mean_sec"])
    if not msgs:
        return None, None
    order = np.argsort(msgs)
    return np.array(msgs)[order], np.array(means)[order]


# ──────────────────────── plotting ─────────────────────────────────

def generate_4by4(show=False):
    """Create a 3×4 grid (N=128,256,1024) with '...' between rows 2 and 3."""

    PLOT_N_VALUES = [128, 256, 1024]

    # First pass: collect global y-range and ALL message sizes
    y_min_global, y_max_global = np.inf, -np.inf
    all_msgs_global = set()
    for topo in TOPOLOGIES:
        for N in PLOT_N_VALUES:
            for algo in ALGORITHMS:
                msg, mu = load_data(topo, algo, N)
                if msg is None:
                    continue
                y_min_global = min(y_min_global, mu.min())
                y_max_global = max(y_max_global, mu.max())
                all_msgs_global.update(msg.tolist())

    if y_min_global == np.inf:
        print("  (no data found)")
        return None

    y_min_global /= 2.0
    y_max_global *= 2.0
    global_xticks = sorted(all_msgs_global)
    global_xlim = (global_xticks[0] / 1.5, global_xticks[-1] * 1.5)

    # 4 rows in gridspec: row0=N128, row1=N256, row2=ellipsis gap, row3=N1024
    fig = plt.figure(figsize=(20, 13.5))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 0.12, 1],
                          wspace=0.08, hspace=0.35,
                          left=0.04, right=0.99, bottom=0.05, top=0.97)

    # Build axes for the 3 data rows (gs rows 0, 1, 3)
    data_gs_rows = [0, 1, 3]
    axes = []
    first_ax = None
    for gr, N in zip(data_gs_rows, PLOT_N_VALUES):
        row_axes = []
        for col in range(4):
            if first_ax is None:
                ax = fig.add_subplot(gs[gr, col])
                first_ax = ax
            else:
                ax = fig.add_subplot(gs[gr, col], sharex=first_ax, sharey=first_ax)
            row_axes.append(ax)
        axes.append(row_axes)

    # Set scales and limits once on the first axes (shared propagates)
    first_ax.set_xscale("log", base=2)
    first_ax.set_yscale("log")
    first_ax.set_xlim(global_xlim)
    first_ax.set_ylim(y_min_global, y_max_global)
    first_ax.set_xticks(global_xticks)
    first_ax.set_xticklabels([_bytes_label(int(x)) for x in global_xticks],
                              rotation=40, ha="right", fontsize=9)
    first_ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    first_ax.xaxis.set_minor_locator(ticker.NullLocator())
    first_ax.yaxis.set_major_formatter(ticker.FuncFormatter(_time_label))
    first_ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    for row, N in enumerate(PLOT_N_VALUES):
        for col, topo in enumerate(TOPOLOGIES):
            topo_label = TOPOLOGY_LABEL.get(topo, topo)
            ax = axes[row][col]

            for algo in ALGORITHMS:
                label, color, marker, ls, ecolor = ALGO_STYLE[algo]
                msg, mu = load_data(topo, algo, N)
                if msg is None:
                    continue
                ax.plot(msg, mu, ls, color=color,
                        marker=marker, markersize=4.5, markeredgecolor=ecolor,
                        markeredgewidth=0.5, linewidth=1.3,
                        label=label, zorder=3)

            # y-label only on leftmost column
            if col == 0:
                ax.set_ylabel("Broadcast time (mean)", fontsize=12, labelpad=0)

            if col == 0:
                plt.setp(ax.get_yticklabels(), fontsize=9)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

            # x-label only on bottom row
            if row == 2:
                ax.set_xlabel("Message size", fontsize=12, labelpad=4)

            plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)

            ax.set_title(f"{topo_label},  $N = {N}$",
                         fontsize=12, fontweight="bold", pad=6)

            # Legend only on top-left panel
            if row == 0 and col == 0:
                ax.legend(fontsize=6.5, loc="upper left", handlelength=2.2,
                          borderpad=0.4, labelspacing=0.35)

            ax.grid(True, which="major", axis="y", linewidth=0.8,
                    color="#888888", alpha=0.7)
            ax.grid(True, which="minor", axis="y", linewidth=0.3,
                    color="#bbbbbb", alpha=0.4)
            ax.grid(True, which="major", axis="x", linewidth=0.3,
                    color="#b0b0b0", alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color("#333333")
                spine.set_linewidth(1.2)

    # Draw "..." ellipsis in the gap row
    for col in range(4):
        ax_gap = fig.add_subplot(gs[2, col])
        ax_gap.set_axis_off()
        ax_gap.text(0.5, 0.5, r"$\vdots$", transform=ax_gap.transAxes,
                    fontsize=18, ha="center", va="center", color="#555555")

    FIGS.mkdir(exist_ok=True)
    pdf = FIGS / "all_4by4.pdf"
    png = FIGS / "all_4by4.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300, facecolor="white", edgecolor="none")
    print(f"  {pdf}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return pdf


# ──────────────────────── main ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate 4×4 figure (4 topologies × 4 sizes)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    generate_4by4(show=args.show)
    print("\nDone.")


if __name__ == "__main__":
    main()
