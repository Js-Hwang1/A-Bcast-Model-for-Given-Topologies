#!/usr/bin/env python3
"""Simple Spyder-friendly plots for MPI state CSV files."""

import csv
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---- Edit these values directly in Spyder ----
PLOT_MODE = "average_receiving_rate"
CSV_FILE = "FatTree_N128_bbs_MSG4194304_R0.states.csv"
MESSAGE_BYTES = None

# For the receiving-rate estimate, merge intervals where any of these states
# are active on a rank. Edit this list manually for each algorithm/experiment.
RATE_STATE_NAMES = ["PMPI_Recv", "PMPI_Bcast", "PMPI_Irecv", "PMPI_Scatter", "PMPI_Sendrecv"]

# BBS override: actual data arrives during PMPI_Waitany, not PMPI_Bcast/Irecv.
# Auto-detected from CSV_FILE containing "_bbs_".
BBS_RATE_STATE_NAMES = ["PMPI_Waitany"]

# Show only these states. Set to None to keep all raw State rows.
STATE_NAMES = None

# Ignore zero-duration events. This usually makes the plot much cleaner.
DROP_ZERO_DURATION = True

# Optional time window. Use None to keep the full trace.
TIME_START = None
TIME_END = None

# Optional rank window. Example: 0, 31. Use None, None to keep all ranks.
RANK_START = None
RANK_END = None

# Figure size and output quality.
FIGURE_WIDTH = 28
FIGURE_HEIGHT = 24
DPI = 300


COLORS = {
    "PMPI_Bcast": "#d62728",
    "PMPI_Irecv": "#d62728",
    "PMPI_Recv": "#d62728",
    "PMPI_Isend": "#1f77b4",
    "PMPI_Wait": "#ff7f0e",
    "PMPI_Waitany": "#ff7f0e",
    "PMPI_Waitall": "#9467bd",
    "PMPI_Barrier": "#7f7f7f",
    "PMPI_Reduce": "#8c564b",
    "PMPI_Init": "#2ca02c",
    "PMPI_Finalize": "#17becf",
}


def rank_number(rank_name):
    match = re.search(r"(\d+)$", rank_name)
    if match:
        return int(match.group(1))
    return 10**9


def infer_message_bytes(csv_file):
    match = re.search(r"_MSG(\d+)_", csv_file)
    if match:
        return float(match.group(1))
    raise ValueError(
        "Could not infer MESSAGE_BYTES from CSV_FILE. Set MESSAGE_BYTES explicitly."
    )


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
            rank_id = rank_number(rank)

            if DROP_ZERO_DURATION and duration <= 0.0:
                continue
            if RANK_START is not None and rank_id < RANK_START:
                continue
            if RANK_END is not None and rank_id > RANK_END:
                continue
            if TIME_START is not None and end < TIME_START:
                continue
            if TIME_END is not None and start > TIME_END:
                continue

            if TIME_START is not None and start < TIME_START:
                start = TIME_START
            if TIME_END is not None and end > TIME_END:
                end = TIME_END

            duration = end - start
            if DROP_ZERO_DURATION and duration <= 0.0:
                continue

            rows.append((rank, rank_id, start, end, duration, state_name))
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
    return [(start, end) for start, end in merged]


def build_rate_estimate(rows, message_bytes, rate_state_names):
    intervals_by_rank = defaultdict(list)
    for rank, _rank_id, start, end, _duration, state_name in rows:
        if state_name in rate_state_names:
            intervals_by_rank[rank].append((start, end))

    merged_by_rank = {}
    summary_rows = []
    events = defaultdict(float)

    for rank, intervals in intervals_by_rank.items():
        merged = merge_intervals(intervals)
        if not merged:
            continue

        union_duration = sum(end - start for start, end in merged)
        if union_duration <= 0.0:
            continue

        avg_rate_bps = message_bytes / union_duration
        merged_by_rank[rank] = merged
        summary_rows.append(
            (
                rank,
                merged[0][0],
                merged[-1][1],
                union_duration,
                avg_rate_bps,
                len(merged),
            )
        )

        for start, end in merged:
            events[start] += avg_rate_bps
            events[end] -= avg_rate_bps

    times = sorted(events)
    total_rate_rows = []
    rate_now = 0.0
    if times:
        total_rate_rows.append((times[0], 0.0))
    for t in times:
        rate_now += events[t]
        if abs(rate_now) < 1e-9:
            rate_now = 0.0
        total_rate_rows.append((t, rate_now))

    return merged_by_rank, summary_rows, total_rate_rows


rows = load_rows(CSV_FILE)
if not rows:
    raise ValueError("No rows left after filtering. Adjust the settings at the top.")

if PLOT_MODE == "states":
    plot_rows = rows
    if STATE_NAMES is not None:
        plot_rows = [row for row in rows if row[5] in STATE_NAMES]
    if not plot_rows:
        raise ValueError("No state rows left after applying STATE_NAMES.")

    rank_names = sorted({row[0] for row in plot_rows}, key=rank_number)
    rank_to_y = {rank: i for i, rank in enumerate(rank_names)}

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor="white")
    ax.set_facecolor("white")

    for rank, _rank_id, start, _end, duration, state_name in plot_rows:
        y = rank_to_y[rank]
        color = COLORS.get(state_name, "#bdbdbd")
        ax.broken_barh([(start, duration)], (y - 0.4, 0.8), facecolors=color)

    x_min = min(row[2] for row in plot_rows)
    x_max = max(row[3] for row in plot_rows)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, len(rank_names))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Rank")
    ax.set_yticks(range(len(rank_names)))
    ax.set_yticklabels(rank_names, fontsize=7)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_title("MPI state timeline")

    used_states = sorted({row[5] for row in plot_rows})
    legend_handles = []
    for state_name in used_states:
        legend_handles.append(
            Patch(facecolor=COLORS.get(state_name, "#bdbdbd"), label=state_name)
        )

    ax.legend(
        handles=legend_handles,
        title="Color legend",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )
elif PLOT_MODE == "average_receiving_rate":
    message_bytes = float(MESSAGE_BYTES) if MESSAGE_BYTES is not None else infer_message_bytes(CSV_FILE)
    effective_rate_states = BBS_RATE_STATE_NAMES if "_bbs_" in CSV_FILE else RATE_STATE_NAMES
    _merged_by_rank, summary_rows, total_rate_rows = build_rate_estimate(
        rows, message_bytes, set(effective_rate_states)
    )
    if not total_rate_rows:
        raise ValueError("No receiving-rate intervals found. Adjust RATE_STATE_NAMES.")

    print("rank,first_start,last_end,union_duration,average_rate_Bps,merged_intervals")
    for rank, first_start, last_end, union_duration, avg_rate_bps, nmerged in sorted(
        summary_rows, key=lambda item: rank_number(item[0])
    ):
        print(
            f"{rank},{first_start:.12g},{last_end:.12g},{union_duration:.12g},"
            f"{avg_rate_bps:.12g},{nmerged}"
        )

    plot_times = [row[0] for row in total_rate_rows]
    plot_rates_gbps = [row[1] / 1e9 for row in total_rate_rows]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT / 2), facecolor="white")
    ax.set_facecolor("white")
    ax.step(plot_times, plot_rates_gbps, where="post", color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Aggregate receiving-rate estimate (GB/s)")
    ax.set_title("States-only aggregate receiving-rate estimate")
    ax.grid(axis="both", color="#dddddd", linewidth=0.8)
    ax.set_xlim(min(plot_times), max(plot_times))
    ax.set_ylim(bottom=0)
else:
    raise ValueError("PLOT_MODE must be 'states' or 'average_receiving_rate'.")

plt.tight_layout()
plt.show()
