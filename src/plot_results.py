#!/usr/bin/env python3
"""
Generate publication-quality broadcast latency plots.

One figure per (topology, N) pair.
  x-axis : message size  (bytes, log-scale)
  y-axis : mean broadcast time  (seconds, log-scale)

Usage:
    python3 src/plot_results.py              # saves PDFs into figs/
    python3 src/plot_results.py --show       # also opens a window
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

# Metallic palette — vivid, shiny tones
#   Metallic red, metallic blue, metallic green, metallic gold, metallic purple
ALGO_STYLE = {
    #          label        color      marker  ls     ecolor (deeper shade)
    "bine": ("BInE",      "#CD2032", "o",    "-",   "#9A1624"),
    "glf":  ("GLF",       "#1560BD", "s",    "-",   "#0E4382"),
    "bbs": ("BBS",  "#1FAD3F", "^",    "-",   "#14762B"),
    "srda": ("SRDA",      "#DAA520", "D",    "-",   "#A07B18"),
    "mpi":  ("MPI_Bcast", "#8B45A6", "v",    "-",  "#5E2E71"),
    "pipe":  ("Pipeline",       "#E05500", "P",    "-",  "#A03D00"),
}

TOPOLOGY_LABEL = {
    "2Dmesh":    "2D Mesh",
    "Butterfly": "Butterfly",
    "Dragonfly": "Dragonfly",
    "FatTree":   "Fat-Tree",
}


# ──────────────────────── helpers ──────────────────────────────────

def _bytes_label(b):
    """Human-readable byte string: 256 B, 1 KB, 1 MB, ..."""
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        v = b / 1024
        return f"{v:g} KB"
    else:
        v = b / 1024**2
        return f"{v:g} MB"


def _time_label(y, _pos=None):
    """Format seconds for log-scale y-axis ticks."""
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
    """Return sorted arrays (msg_bytes, mean, stdev) for one series."""
    d = DATA / topology / algorithm
    if not d.is_dir():
        return None, None

    msgs, means = [], []
    for f in sorted(d.glob(f"N{N}_MSG*.json")):
        if "_R" in f.stem.split("MSG")[1]:
            continue  # skip per-root files
        with open(f) as fh:
            rec = json.load(fh)
        if rec["msg_bytes"] < 16384:
            continue  # skip small messages (too noisy)
        msgs.append(rec["msg_bytes"])
        means.append(rec["mean_sec"])

    if not msgs:
        return None, None

    order = np.argsort(msgs)
    return (np.array(msgs)[order],
            np.array(means)[order])


# ──────────────────────── plotting ─────────────────────────────────

def plot_topology(topology, N, show=False):
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

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
        plt.close(fig)
        return None

    # ── axes ──
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    # x-ticks at actual data points
    xticks = sorted(all_msgs)
    ax.set_xticks(xticks)
    ax.set_xticklabels([_bytes_label(int(x)) for x in xticks],
                       rotation=40, ha="right", fontsize=7)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_time_label))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    plt.setp(ax.get_yticklabels(), fontsize=7)

    ax.set_xlabel("Message size", fontsize=9, labelpad=4)
    ax.set_ylabel("Broadcast time (mean)", fontsize=9,
                  labelpad=4)

    topo_label = TOPOLOGY_LABEL.get(topology, topology)
    ax.set_title(f"{topo_label},  $N = {N}$",
                 fontsize=11, fontweight="bold", pad=8)

    ax.legend(fontsize=7, loc="upper left", handlelength=2.2,
              borderpad=0.4, labelspacing=0.35)

    ax.grid(True, which="major", linewidth=0.3, color="#b0b0b0", alpha=0.6)
    ax.grid(True, which="minor", linewidth=0.15, color="#d0d0d0", alpha=0.4)

    # Subtle spine styling
    for spine in ax.spines.values():
        spine.set_color("#555555")

    fig.tight_layout(pad=1.0)

    FIGS.mkdir(exist_ok=True)
    pdf = FIGS / f"{topology}_N{N}.pdf"
    png = FIGS / f"{topology}_N{N}.png"
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
        description="Plot broadcast latency vs message size")
    parser.add_argument("--show", action="store_true",
                        help="Open figures in a window")
    args = parser.parse_args()

    topologies = sorted(
        d.name for d in DATA.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    # Discover all N values present
    n_values = set()
    for topo in topologies:
        for algo in ALGORITHMS:
            d = DATA / topo / algo
            if d.is_dir():
                for f in d.glob("N*_MSG*.json"):
                    if "_R" in f.stem.split("MSG")[1]:
                        continue
                    n_values.add(int(f.stem.split("_")[0][1:]))

    print(f"Topologies : {topologies}")
    print(f"N values   : {sorted(n_values)}")
    print(f"Algorithms : {ALGORITHMS}")
    print()

    for topo in topologies:
        for N in sorted(n_values):
            pdf = plot_topology(topo, N, show=args.show)
            if pdf is None:
                print(f"  (no data for {topo} N={N})")

    print("\nDone.")


if __name__ == "__main__":
    main()
