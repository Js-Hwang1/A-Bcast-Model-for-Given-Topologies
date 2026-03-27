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

def generate_1by4(topology, show=False):
    """Create a 1×4 figure for one topology with shared y-axis."""

    # First pass: collect global y-range across all 4 sizes
    y_min_global, y_max_global = np.inf, -np.inf
    for N in N_VALUES:
        for algo in ALGORITHMS:
            msg, mu = load_data(topology, algo, N)
            if msg is None:
                continue
            y_min_global = min(y_min_global, mu.min())
            y_max_global = max(y_max_global, mu.max())

    if y_min_global == np.inf:
        print(f"  (no data for {topology})")
        return None

    # Pad the range a bit on log scale
    y_min_global /= 2.0
    y_max_global *= 2.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.0), sharey=True)

    topo_label = TOPOLOGY_LABEL.get(topology, topology)

    for col, (N, ax) in enumerate(zip(N_VALUES, axes)):
        any_data = False
        all_msgs = set()

        for algo in ALGORITHMS:
            label, color, marker, ls, ecolor = ALGO_STYLE[algo]
            msg, mu = load_data(topology, algo, N)
            if msg is None:
                continue
            any_data = True
            all_msgs.update(msg.tolist())
            ax.plot(msg, mu, ls, color=color,
                    marker=marker, markersize=4.5, markeredgecolor=ecolor,
                    markeredgewidth=0.5, linewidth=1.3,
                    label=label, zorder=3)

        if not any_data:
            ax.set_visible(False)
            continue

        # x-axis
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        xticks = sorted(all_msgs)
        ax.set_xticks(xticks)
        ax.set_xticklabels([_bytes_label(int(x)) for x in xticks],
                           rotation=40, ha="right", fontsize=7)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.xaxis.set_minor_locator(ticker.NullLocator())

        # y-axis: shared limits
        ax.set_ylim(y_min_global, y_max_global)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_time_label))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        if col == 0:
            plt.setp(ax.get_yticklabels(), fontsize=7)
            ax.set_ylabel("Broadcast time (mean)", fontsize=9, labelpad=4)

        ax.set_xlabel("Message size", fontsize=9, labelpad=4)
        ax.set_title(f"{topo_label},  $N = {N}$",
                     fontsize=11, fontweight="bold", pad=8)

        # Legend only on first panel
        if col == 0:
            ax.legend(fontsize=7, loc="upper left", handlelength=2.2,
                      borderpad=0.4, labelspacing=0.35)

        ax.grid(True, which="major", linewidth=0.3, color="#b0b0b0", alpha=0.6)
        ax.grid(True, which="minor", linewidth=0.15, color="#d0d0d0", alpha=0.4)
        for spine in ax.spines.values():
            spine.set_color("#555555")

    fig.tight_layout(pad=1.0)

    FIGS.mkdir(exist_ok=True)
    pdf = FIGS / f"{topology}_1by4.pdf"
    png = FIGS / f"{topology}_1by4.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
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
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

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
