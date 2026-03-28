#!/usr/bin/env python3
"""
Generate a 1×4 figure for a single topology across 4 network sizes.
Y-axis is shared so tick labels align across all columns.

Usage:
    python3 paper/generate_1by4.py                     # all topologies
    python3 paper/generate_1by4.py --topo FatTree       # one topology
    python3 paper/generate_1by4.py --show               # open window
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

ALGO_STYLE = {
    "bine": ("Bine",      "#CD2032", "o",  "-", "#9A1624"),
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


def _time_label_ms(y, _pos=None):
    """Format seconds as milliseconds using 10^x notation."""
    if y <= 0:
        return ""
    import math
    v = y * 1e3  # convert to ms
    exp = math.log10(v)
    if abs(exp - round(exp)) < 1e-9:
        e = int(round(exp))
        return f"$10^{{{e}}}$"
    return ""


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

def generate_1by4(topology, show=False):
    """Create a 1×4 figure for one topology with shared y-axis."""

    # First pass: collect global y-range and ALL message sizes
    y_min_global, y_max_global = np.inf, -np.inf
    all_msgs_global = set()
    for N in N_VALUES:
        for algo in ALGORITHMS:
            msg, mu = load_data(topology, algo, N)
            if msg is None:
                continue
            y_min_global = min(y_min_global, mu.min())
            y_max_global = max(y_max_global, mu.max())
            all_msgs_global.update(msg.tolist())

    if y_min_global == np.inf:
        print(f"  (no data for {topology})")
        return None

    # Pad the range a bit on log scale
    y_min_global /= 2.0
    y_max_global *= 2.0
    global_xticks = sorted(all_msgs_global)
    global_xlim = (global_xticks[0] / 1.5, global_xticks[-1] * 1.5)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.0),
                             sharey=True, sharex=True)

    topo_label = TOPOLOGY_LABEL.get(topology, topology)

    # Set scales and limits once on the first axes (shared propagates)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlim(global_xlim)
    axes[0].set_ylim(y_min_global, y_max_global)
    axes[0].set_xticks(global_xticks)
    axes[0].set_xticklabels([_bytes_label(int(x)) for x in global_xticks],
                             rotation=40, ha="right", fontsize=9)
    axes[0].xaxis.set_minor_formatter(ticker.NullFormatter())
    axes[0].xaxis.set_minor_locator(ticker.NullLocator())
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(_time_label_ms))
    axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())

    for col, (N, ax) in enumerate(zip(N_VALUES, axes)):
        for algo in ALGORITHMS:
            label, color, marker, ls, ecolor = ALGO_STYLE[algo]
            msg, mu = load_data(topology, algo, N)
            if msg is None:
                continue
            ax.plot(msg, mu, ls, color=color,
                    marker=marker, markersize=4.5, markeredgecolor=ecolor,
                    markeredgewidth=0.5, linewidth=1.3,
                    label=label, zorder=3)

        if col == 0:
            ax.set_ylabel("Broadcast time (mean)", fontsize=12, labelpad=0)

        plt.setp(ax.get_yticklabels(), fontsize=9)
        ax.set_xlabel("Message size", fontsize=12, labelpad=4)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
        ax.set_title(f"{topo_label},  $N = {N}$",
                     fontsize=12, fontweight="bold", pad=8)

        # Legend only on first panel
        if col == 0:
            ax.legend(fontsize=9, loc="upper left", handlelength=2.2,
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

    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.18, top=0.88,
                        wspace=0.08)

    FIGS.mkdir(exist_ok=True)
    pdf = FIGS / f"{topology}_1by4.pdf"
    png = FIGS / f"{topology}_1by4.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300, facecolor="white", edgecolor="none")
    print(f"  {pdf}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return pdf


TOPOLOGIES = ["2Dmesh", "Butterfly", "Dragonfly", "FatTree"]


def generate_1by4_byN(N, show=False):
    """Create a 1×4 figure for one N value across 4 topologies."""

    # First pass: collect global y-range and ALL message sizes
    y_min_global, y_max_global = np.inf, -np.inf
    all_msgs_global = set()
    for topo in TOPOLOGIES:
        for algo in ALGORITHMS:
            msg, mu = load_data(topo, algo, N)
            if msg is None:
                continue
            y_min_global = min(y_min_global, mu.min())
            y_max_global = max(y_max_global, mu.max())
            all_msgs_global.update(msg.tolist())

    if y_min_global == np.inf:
        print(f"  (no data for N={N})")
        return None

    y_min_global /= 2.0
    y_max_global *= 2.0
    global_xticks = sorted(all_msgs_global)
    global_xlim = (global_xticks[0] / 1.5, global_xticks[-1] * 1.5)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6.5),
                             sharey=True, sharex=True)

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlim(global_xlim)
    axes[0].set_ylim(y_min_global, y_max_global)
    axes[0].set_xticks(global_xticks)
    axes[0].set_xticklabels([_bytes_label(int(x)) for x in global_xticks],
                             rotation=40, ha="right", fontsize=16)
    axes[0].xaxis.set_minor_formatter(ticker.NullFormatter())
    axes[0].xaxis.set_minor_locator(ticker.NullLocator())
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(_time_label_ms))
    axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())

    for col, (topo, ax) in enumerate(zip(TOPOLOGIES, axes)):
        topo_label = TOPOLOGY_LABEL.get(topo, topo)
        for algo in ALGORITHMS:
            label, color, marker, ls, ecolor = ALGO_STYLE[algo]
            msg, mu = load_data(topo, algo, N)
            if msg is None:
                continue
            ax.plot(msg, mu, ls, color=color,
                    marker=marker, markersize=6, markeredgecolor=ecolor,
                    markeredgewidth=0.6, linewidth=2.0,
                    label=label, zorder=3)

        if col != 0:
            plt.setp(ax.get_yticklabels(), visible=False)

        plt.setp(ax.get_yticklabels(), fontsize=16)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=16)
        ax.text(0.5, 0.96, topo_label, transform=ax.transAxes,
                fontsize=16, fontweight="bold", ha="center", va="top",
                clip_on=True, zorder=5)

        if col == 0:
            ax.legend(fontsize=13, loc="upper left", handlelength=2.2,
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

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.97,
                        wspace=0.02)

    fig.text(0.52, 0.01, "Message Size", ha="center", fontsize=16)
    fig.text(0.02, 0.55, "Mean Broadcast Time (ms)", va="center",
             rotation="vertical", fontsize=16)

    FIGS.mkdir(exist_ok=True)
    pdf = FIGS / f"N{N}_1by4.pdf"
    png = FIGS / f"N{N}_1by4.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300, facecolor="white", edgecolor="none")
    print(f"  {pdf}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return pdf


def generate_2by2_byN(N, show=False):
    """Create a 2×2 figure for one N value across 4 topologies."""

    # First pass: collect global y-range and ALL message sizes
    y_min_global, y_max_global = np.inf, -np.inf
    all_msgs_global = set()
    for topo in TOPOLOGIES:
        for algo in ALGORITHMS:
            msg, mu = load_data(topo, algo, N)
            if msg is None:
                continue
            y_min_global = min(y_min_global, mu.min())
            y_max_global = max(y_max_global, mu.max())
            all_msgs_global.update(msg.tolist())

    if y_min_global == np.inf:
        print(f"  (no data for N={N})")
        return None

    y_min_global /= 2.0
    y_max_global *= 2.0
    global_xticks = sorted(all_msgs_global)
    global_xlim = (global_xticks[0] / 1.5, global_xticks[-1] * 1.5)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                             sharey=True, sharex=True)

    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlim(global_xlim)
    axes[0, 0].set_ylim(y_min_global, y_max_global)
    axes[0, 0].set_xticks(global_xticks)
    axes[0, 0].set_xticklabels([_bytes_label(int(x)) for x in global_xticks],
                                rotation=40, ha="right", fontsize=16)
    axes[0, 0].xaxis.set_minor_formatter(ticker.NullFormatter())
    axes[0, 0].xaxis.set_minor_locator(ticker.NullLocator())
    axes[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(_time_label_ms))
    axes[0, 0].yaxis.set_minor_formatter(ticker.NullFormatter())

    for idx, (topo, ax) in enumerate(zip(TOPOLOGIES, axes.flat)):
        topo_label = TOPOLOGY_LABEL.get(topo, topo)
        row, col = divmod(idx, 2)
        for algo in ALGORITHMS:
            label, color, marker, ls, ecolor = ALGO_STYLE[algo]
            msg, mu = load_data(topo, algo, N)
            if msg is None:
                continue
            ax.plot(msg, mu, ls, color=color,
                    marker=marker, markersize=6, markeredgecolor=ecolor,
                    markeredgewidth=0.6, linewidth=2.0,
                    label=label, zorder=3)

        if col != 0:
            plt.setp(ax.get_yticklabels(), visible=False)

        plt.setp(ax.get_yticklabels(), fontsize=16)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=16)
        ax.text(0.5, 0.96, topo_label, transform=ax.transAxes,
                fontsize=16, fontweight="bold", ha="center", va="top",
                clip_on=True, zorder=5)

        if idx == 0:
            ax.legend(fontsize=13, loc="upper left", handlelength=2.2,
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

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.97,
                        wspace=0.02, hspace=0.02)

    fig.text(0.54, 0.01, "Message Size", ha="center", fontsize=20)
    fig.text(0.02, 0.52, "Mean Broadcast Time (ms)", va="center",
             rotation="vertical", fontsize=20)

    FIGS.mkdir(exist_ok=True)
    pdf = FIGS / f"N{N}_2by2.pdf"
    png = FIGS / f"N{N}_2by2.png"
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
        description="Generate 1×4 figures (1 topology, 4 sizes)")
    parser.add_argument("--topo", type=str, default=None,
                        help="Single topology to plot (default: all)")
    parser.add_argument("--N", type=int, default=None,
                        help="Generate a 1×4 by N (4 topologies for one N)")
    parser.add_argument("--layout", choices=["1by4", "2by2"], default="1by4",
                        help="Layout for --N mode (default: 1by4)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.N is not None:
        if args.layout == "2by2":
            generate_2by2_byN(args.N, show=args.show)
        else:
            generate_1by4_byN(args.N, show=args.show)
    else:
        topologies = sorted(
            d.name for d in DATA.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        if args.topo:
            topologies = [args.topo]

        for topo in topologies:
            generate_1by4(topo, show=args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
