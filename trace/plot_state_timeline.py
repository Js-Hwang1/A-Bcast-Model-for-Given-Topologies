#!/usr/bin/env python3
"""Simple Spyder-friendly Gantt plot from a Paje State CSV."""

import csv
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---- Edit these values directly in Spyder ----
CSV_FILE = "2Dmesh_N128_mpi_MSG512_R0.states.csv"

# Show only these states. Set to None to keep everything.
STATE_NAMES = ["PMPI_Bcast", "PMPI_Recv", "PMPI_Isend", "PMPI_Irecv",
               "PMPI_Wait", "PMPI_Waitall", "PMPI_Init", "PMPI_Finalize",
               "PMPI_Barrier", "PMPI_Reduce"]

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
DPI = 220


COLORS = {
    "PMPI_Bcast": "#d62728",
    "PMPI_Irecv": "#d62728",
    "PMPI_Recv": "#d62728",
    "PMPI_Isend": "#1f77b4",
    "PMPI_Wait": "#ff7f0e",
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


rows = []
last_time = 0

with open(CSV_FILE, newline="") as handle:
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

        if STATE_NAMES is not None and state_name not in STATE_NAMES:
            continue

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
        
        if end >= last_time:
            last_time = end
print(last_time)

if not rows:
    raise ValueError("No rows left after filtering. Adjust the settings at the top.")


rank_names = sorted({row[0] for row in rows}, key=rank_number)
rank_to_y = {rank: i for i, rank in enumerate(rank_names)}


fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), facecolor="white")
ax.set_facecolor("white")

for rank, rank_id, start, end, duration, state_name in rows:
    y = rank_to_y[rank]
    color = COLORS.get(state_name, "#bdbdbd")
    ax.broken_barh([(start, duration)], (y - 0.4, 0.8), facecolors=color)


x_min = min(row[2] for row in rows)
x_max = max(row[3] for row in rows)

ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, len(rank_names))
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Rank")
ax.set_yticks(range(len(rank_names)))
ax.set_yticklabels(rank_names, fontsize=7)
ax.grid(axis="x", color="#dddddd", linewidth=0.8)
ax.set_title("MPI state timeline")


used_states = sorted({row[5] for row in rows})
legend_handles = []
for state_name in used_states:
    legend_handles.append(
        Patch(facecolor=COLORS.get(state_name, "#bdbdbd"), label=state_name)
    )

ax.legend(
    handles=legend_handles,
    title="Color legend",
    loc="upper left",
    bbox_to_anchor=(0.9, 1.0),
    frameon=True,
)


plt.tight_layout()
plt.show()
