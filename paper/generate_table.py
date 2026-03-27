#!/usr/bin/env python3
"""
Generate LaTeX tables of broadcast results — one table per N value.
Each table has 4 topology groups x 6 algorithms.
Columns: 7 message sizes, each with mean, max, min, stdev sub-columns.
Best (lowest mean) per topology+msg_size is bolded.

Output: paper/paste.tex
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "paper" / "paste.tex"

ALGORITHMS = ["bbs", "bine", "glf", "pipe", "srda", "mpi"]
ALGO_LABEL = {
    "bbs":  "BBS",
    "bine": "BInE",
    "glf":  "GLF",
    "pipe": "Pipeline",
    "srda": "SRDA",
    "mpi":  r"MPI\_Bcast",
}

TOPOLOGIES = ["2Dmesh", "Butterfly", "Dragonfly", "FatTree"]
TOPOLOGY_LABEL = {
    "2Dmesh":    "2D Mesh",
    "Butterfly": "Butterfly",
    "Dragonfly": "Dragonfly",
    "FatTree":   "Fat-Tree",
}

N_VALUES = [128, 256, 512, 1024]
MSG_SIZES = [65536, 262144, 1048576, 4194304, 16777216, 67108864, 134217728]

STATS = ["mean_sec", "max_sec", "min_sec", "stdev_sec"]
STAT_HEADER = [r"$\bar{t}$", "max", "min", r"$\sigma$"]


def bytes_label(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b // 1024} KB"
    else:
        return f"{b // (1024**2)} MB"


def fmt_time(sec):
    """Format seconds as a plain number+unit string, no special commands."""
    if sec is None:
        return "---"
    if sec < 1e-6:
        v = sec * 1e9
        if v < 10:
            return f"{v:.2f} ns"
        elif v < 100:
            return f"{v:.1f} ns"
        else:
            return f"{v:.0f} ns"
    if sec < 1e-3:
        v = sec * 1e6
        if v < 10:
            return f"{v:.2f} us"
        elif v < 100:
            return f"{v:.1f} us"
        else:
            return f"{v:.0f} us"
    if sec < 1:
        v = sec * 1e3
        if v < 10:
            return f"{v:.2f} ms"
        elif v < 100:
            return f"{v:.1f} ms"
        else:
            return f"{v:.0f} ms"
    return f"{sec:.2f} s"


def load_record(topology, algorithm, N, msg_bytes):
    f = DATA / topology / algorithm / f"N{N}_MSG{msg_bytes}.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)


def generate_table(N):
    n_algo = len(ALGORITHMS)
    n_msg = len(MSG_SIZES)
    n_stat = len(STATS)

    col_spec = "ll" + "rrrr" * n_msg

    lines = []
    lines.append(r"\begin{tableorg}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Broadcast time for $N = " + str(N) + r"$.}")
    lines.append(r"\label{tab:results_N" + str(N) + r"}")
    lines.append(r"\rotatebox{90}{\resizebox{0.9\textheight}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Header row 1: msg size labels
    hdr1 = ["", ""]
    for msg in MSG_SIZES:
        hdr1.append(r"\multicolumn{4}{c}{" + bytes_label(msg) + r"}")
    lines.append(" & ".join(hdr1) + r" \\")

    # Cmidrules
    rules = []
    for i in range(n_msg):
        s = 3 + i * 4
        rules.append(r"\cmidrule(lr){" + f"{s}-{s+3}" + r"}")
    lines.append(" ".join(rules))

    # Header row 2: stat labels
    hdr2 = ["Topology", "Algorithm"]
    for _ in range(n_msg):
        hdr2.extend(STAT_HEADER)
    lines.append(" & ".join(hdr2) + r" \\")
    lines.append(r"\midrule")

    for t_idx, topo in enumerate(TOPOLOGIES):
        topo_lbl = TOPOLOGY_LABEL[topo]

        # Find best mean per msg size
        best_mean = {}
        for msg in MSG_SIZES:
            vals = []
            for algo in ALGORITHMS:
                rec = load_record(topo, algo, N, msg)
                if rec is not None:
                    vals.append(rec["mean_sec"])
            if vals:
                best_mean[msg] = min(vals)

        for a_idx, algo in enumerate(ALGORITHMS):
            parts = []
            if a_idx == 0:
                parts.append(r"\multirow{" + str(n_algo) + r"}{*}{" + topo_lbl + r"}")
            else:
                parts.append("")
            parts.append(ALGO_LABEL[algo])

            for msg in MSG_SIZES:
                rec = load_record(topo, algo, N, msg)
                for stat in STATS:
                    if rec is None:
                        parts.append("---")
                    else:
                        val = rec[stat]
                        s = fmt_time(val)
                        if stat == "mean_sec" and msg in best_mean and val <= best_mean[msg]:
                            s = r"\textbf{" + s + r"}"
                        parts.append(s)

            lines.append(" & ".join(parts) + r" \\")

        if t_idx < len(TOPOLOGIES) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}}")
    lines.append(r"\end{tableorg}")

    return "\n".join(lines)


def main():
    all_tables = []
    for N in N_VALUES:
        all_tables.append(generate_table(N))
        print(f"  Generated table for N={N}")

    with open(OUT, "w") as f:
        f.write("% Auto-generated by paper/generate_table.py\n\n")
        f.write("\n\n".join(all_tables))
        f.write("\n")

    print(f"\n  Output: {OUT}")


if __name__ == "__main__":
    main()
