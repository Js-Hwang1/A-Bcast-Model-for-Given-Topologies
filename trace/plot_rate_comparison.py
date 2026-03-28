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

TOPOLOGY = "16K3"
N = 16
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

# Base names of MPI states that represent actual data reception.
# PMPI_Irecv is deliberately EXCLUDED: it is a non-blocking post that
# returns near-instantly.  The actual data arrival is captured by Wait*.
RATE_STATE_BASES = {"PMPI_Recv", "PMPI_Bcast",
                    "PMPI_Scatter", "PMPI_Sendrecv"}

ALGO_RATE_BASES = {
    "bbs": {"PMPI_Waitany"},
}

# Algorithms whose receive states are near-instant completions of pipelined
# chunks (e.g. Waitany).  For these we use the *envelope* (span from first
# to last completion per rank) instead of the union of tiny durations.
ALGO_USE_ENVELOPE = {"bbs"}

DROP_ZERO_DURATION = True

FIGURE_WIDTH = 8.0
FIGURE_HEIGHT = 4.5
DPI = 300

TRACE_DIR = Path(__file__).resolve().parent


# ──────────────────────── helpers (from plot_state_timeline.py) ────

def rank_number(rank_name):
    match = re.search(r"(\d+)$", rank_name)
    return int(match.group(1)) if match else 10**9


_SIZE_RE = re.compile(r"^(\w+)\((\d+)\)$")   # e.g. "PMPI_Recv(4194304)"


def parse_state(state_name):
    """Return (base_name, byte_size) — byte_size is None if not annotated."""
    m = _SIZE_RE.match(state_name)
    if m:
        return m.group(1), int(m.group(2))
    return state_name, None


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
            base, nbytes = parse_state(state_name)
            rows.append((rank, rank_number(rank), start, end, duration,
                         base, nbytes))
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


def build_rate_curve(rows, message_bytes, rate_bases=None, root_rank=0,
                     use_envelope=False, n_compute=None):
    """Build an aggregate receiving-rate sweep-line curve.

    Per-rank approach: each non-root compute rank receives exactly
    message_bytes in total.  We spread that uniformly across the rank's
    receive span.

    n_compute: if set, only include ranks with rank_number < n_compute
               (filters out relay/switch ranks used by BBS on FatTree/Dragonfly).
    use_envelope: for pipelined algorithms whose individual receive
    completions are near-instant (e.g. Waitany), use the full envelope
    per rank instead of the union of tiny individual durations.
    """
    if rate_bases is None:
        rate_bases = RATE_STATE_BASES

    # Check if we have per-call annotations
    has_annotations = any(nb is not None
                         for _, _, _, _, _, base, nb in rows
                         if base in rate_bases)

    if has_annotations:
        # ── Exact per-interval rates from annotated byte sizes ──
        events = defaultdict(float)
        for rank, rid, start, end, _dur, base, nbytes in rows:
            if base not in rate_bases or nbytes is None:
                continue
            if n_compute is not None and rid >= n_compute:
                continue
            dur = end - start
            if dur <= 0.0:
                continue
            rate = nbytes / dur
            events[start] += rate
            events[end]   -= rate
    else:
        # ── Per-rank averaging (correct when each rank receives MSG) ──
        intervals_by_rank = defaultdict(list)
        for rank, rid, start, end, _dur, base, _nb in rows:
            if rid == root_rank:
                continue                # root sends, does not receive
            if n_compute is not None and rid >= n_compute:
                continue                # skip relay/switch ranks
            if base not in rate_bases:
                continue
            intervals_by_rank[rank].append((start, end))

        events = defaultdict(float)
        for rank, intervals in intervals_by_rank.items():
            if use_envelope:
                # Envelope: single span from first start to last end
                first = min(s for s, _ in intervals)
                last  = max(e for _, e in intervals)
                span  = last - first
                if span <= 0.0:
                    continue
                avg_rate = message_bytes / span
                events[first] += avg_rate
                events[last]  -= avg_rate
            else:
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

def plot_one(ax, n_compute, subtitle):
    """Plot rate curves for all algorithms onto a single axes."""
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

        rate_bases = ALGO_RATE_BASES.get(algo, RATE_STATE_BASES)
        envelope = algo in ALGO_USE_ENVELOPE
        curve_t, curve_r = build_rate_curve(rows, MSG, rate_bases,
                                            root_rank=ROOT,
                                            use_envelope=envelope,
                                            n_compute=n_compute)
        if not curve_t:
            print(f"  (skip {algo}: no receiving intervals)")
            continue

        label, color, ls = ALGO_STYLE[algo]
        rates_gbps = [r / 1e9 for r in curve_r]
        ax.step(curve_t, rates_gbps, where="post",
                color=color, linestyle=ls, linewidth=1.5, label=label, zorder=3)
        any_data = True

    if not any_data:
        return False

    ax.set_xlabel("Time (seconds)", fontsize=9, labelpad=4)
    ax.set_ylabel("Aggregate receiving rate (GB/s)", fontsize=9, labelpad=4)
    ax.set_title(subtitle, fontsize=10, fontweight="bold", pad=6)
    ax.legend(fontsize=6.5, loc="upper right", handlelength=2.0,
              borderpad=0.4, labelspacing=0.3)
    ax.grid(True, which="major", linewidth=0.3, color="#b0b0b0", alpha=0.6)
    ax.set_ylim(bottom=0)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compare aggregate receiving-rate across algorithms")
    parser.add_argument("--show", action="store_true",
                        help="Open figure in a window")
    args = parser.parse_args()

    # ── Check if side-by-side is needed (only for topos with relay ranks) ──
    # Topologies like Dragonfly/FatTree have relay ranks (NP > N);
    # flat topologies (16K3, 2Dmesh, Butterfly) do not.
    needs_sidebyside = TOPOLOGY in ("Dragonfly", "FatTree")

    if needs_sidebyside:
        fig, (ax_l, ax_r) = plt.subplots(1, 2,
                                          figsize=(FIGURE_WIDTH * 2, FIGURE_HEIGHT))
        suptitle = f"{TOPOLOGY},  $N = {N}$,  MSG = {MSG // (1024**2)} MB"
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02)

        ok_l = plot_one(ax_l, n_compute=N,
                        subtitle=f"Compute ranks only (< {N})")
        ok_r = plot_one(ax_r, n_compute=None,
                        subtitle="All ranks (incl. relay/router)")
        ax_r.set_ylabel("")

        if not (ok_l or ok_r):
            print("No data found.")
            plt.close(fig)
            return
    else:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        title = f"{TOPOLOGY},  $N = {N}$,  MSG = {MSG // (1024**2)} MB"
        ok = plot_one(ax, n_compute=N, subtitle=title)
        if not ok:
            print("No data found.")
            plt.close(fig)
            return

    fig.tight_layout(pad=1.2)

    suffix = "_sidebyside" if needs_sidebyside else ""
    out_base = TRACE_DIR / f"{TOPOLOGY}_N{N}_MSG{MSG}_R{ROOT}_rate_comparison{suffix}"
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
