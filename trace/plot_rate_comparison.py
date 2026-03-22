#!/usr/bin/env python3
"""
Plot aggregate receiving-rate curves for multiple algorithms in one figure.

Usage:
    python3 trace/plot_rate_comparison.py              # saves PDF+PNG into trace/
    python3 trace/plot_rate_comparison.py --show        # also opens a window
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

TOPOLOGY = "Dragonfly"
N = 128
MSG = 4194304
ROOT = 0

ALGORITHMS = ["bine", "mpi", "srda", "glf", "pipe", "bbs"]

ALGO_STYLE = {
    "bine": ("BInE",      "#CD2032", "-"),
    "glf":  ("GLF",       "#1560BD", "-"),
    "bbs":  ("BBS",       "#1FAD3F", "-"),
    "srda": ("SRDA",      "#DAA520", "-"),
    "mpi":  ("MPI_Bcast", "#8B45A6", "-"),
    "pipe": ("Pipeline",  "#E05500", "-"),
}

RATE_STATE_NAMES = {"PMPI_Recv", "PMPI_Bcast", "PMPI_Irecv",
                    "PMPI_Scatter", "PMPI_Sendrecv"}

# BBS uses Isend/Irecv/Waitany for its pipelined broadcast.
# PMPI_Irecv is zero-duration (non-blocking); the actual data arrival
# is captured by PMPI_Waitany.  The PMPI_Bcast that appears in the trace
# is a coordination broadcast (MPI_Bcast(&need_gen,...)), NOT data transfer.
ALGO_RATE_STATES = {
    "bbs": {"PMPI_Waitany"},
}

DROP_ZERO_DURATION = True

FIGURE_WIDTH = 8.0
FIGURE_HEIGHT = 4.5
DPI = 300

TRACE_DIR = Path(__file__).resolve().parent


# ──────────────────────── helpers (from plot_state_timeline.py) ────

def rank_number(rank_name):
    match = re.search(r"(\d+)$", rank_name)
    return int(match.group(1)) if match else 10**9


def load_rows(csv_file):
    rows = []
    with open(csv_file, newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        for row in reader:
            if not row or row[0] != "State":
                continue
            rank = row[1]
            start = float(row[3])
            end = float(row[4])
            duration = float(row[5])
            state_name = row[7]

            if DROP_ZERO_DURATION and duration <= 0.0:
                continue
            rows.append((rank, rank_number(rank), start, end, duration, state_name))
    return rows


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        prev = merged[-1]
        if start <= prev[1]:
            if end > prev[1]:
                prev[1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def build_rate_curve(rows, message_bytes, rate_states=None):
    if rate_states is None:
        rate_states = RATE_STATE_NAMES
    intervals_by_rank = defaultdict(list)
    for rank, _rid, start, end, _dur, state_name in rows:
        if state_name in rate_states:
            intervals_by_rank[rank].append((start, end))

    events = defaultdict(float)
    for rank, intervals in intervals_by_rank.items():
        merged = merge_intervals(intervals)
        if not merged:
            continue
        union_duration = sum(e - s for s, e in merged)
        if union_duration <= 0.0:
            continue
        avg_rate = message_bytes / union_duration
        for s, e in merged:
            events[s] += avg_rate
            events[e] -= avg_rate

    times = sorted(events)
    curve_t, curve_r = [], []
    rate_now = 0.0
    if times:
        curve_t.append(times[0])
        curve_r.append(0.0)
    for t in times:
        rate_now += events[t]
        if abs(rate_now) < 1e-9:
            rate_now = 0.0
        curve_t.append(t)
        curve_r.append(rate_now)
    return curve_t, curve_r


# ──────────────────────── main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare aggregate receiving-rate across algorithms")
    parser.add_argument("--show", action="store_true",
                        help="Open figure in a window")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    any_data = False
    for algo in ALGORITHMS:
        csv_name = f"{TOPOLOGY}_N{N}_{algo}_MSG{MSG}_R{ROOT}.states.csv"
        csv_path = TRACE_DIR / csv_name
        if not csv_path.exists():
            print(f"  (skip {algo}: {csv_name} not found)")
            continue

        rows = load_rows(str(csv_path))
        if not rows:
            print(f"  (skip {algo}: no state rows)")
            continue

        rate_states = ALGO_RATE_STATES.get(algo, RATE_STATE_NAMES)
        curve_t, curve_r = build_rate_curve(rows, MSG, rate_states)
        if not curve_t:
            print(f"  (skip {algo}: no receiving intervals)")
            continue

        label, color, ls = ALGO_STYLE[algo]
        rates_gbps = [r / 1e9 for r in curve_r]
        ax.step(curve_t, rates_gbps, where="post",
                color=color, linestyle=ls, linewidth=1.5, label=label, zorder=3)
        any_data = True

    if not any_data:
        print("No data found.")
        plt.close(fig)
        return

    ax.set_xlabel("Time (seconds)", fontsize=9, labelpad=4)
    ax.set_ylabel("Aggregate receiving rate (GB/s)", fontsize=9, labelpad=4)
    ax.set_title(
        f"{TOPOLOGY},  $N = {N}$,  MSG = {MSG // (1024**2)} MB",
        fontsize=11, fontweight="bold", pad=8)

    ax.legend(fontsize=7, loc="upper right", handlelength=2.2,
              borderpad=0.4, labelspacing=0.35)

    ax.grid(True, which="major", linewidth=0.3, color="#b0b0b0", alpha=0.6)
    ax.set_ylim(bottom=0)

    for spine in ax.spines.values():
        spine.set_color("#555555")

    fig.tight_layout(pad=1.0)

    out_base = TRACE_DIR / f"{TOPOLOGY}_N{N}_MSG{MSG}_R{ROOT}_rate_comparison"
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"  {out_base}.pdf")
    print(f"  {out_base}.png")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
