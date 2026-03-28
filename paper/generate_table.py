#!/usr/bin/env python3
"""
Generate LaTeX tables of broadcast results — one table per N value.
Format: rows = Topology > Message Size > Metric, columns = algorithms.
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
    "bine": "Bine",
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
STAT_LABEL = [r"$\bar{T}$", r"$T_{\max}$", r"$T_{\min}$", r"$\sigma$"]


def bytes_label(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b // 1024} KB"
    else:
        return f"{b // (1024**2)} MB"


def fmt_time(sec):
    """Format seconds as a plain number+unit string."""
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
            return f"{v:.2f} \\textmu s"
        elif v < 100:
            return f"{v:.1f} \\textmu s"
        else:
            return f"{v:.0f} \\textmu s"
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
    n_rows_per_topo = n_msg * n_stat  # 28

    # Column spec: Topology | Message Size | Metric | 6 algorithm columns
    col_spec = "lll" + "r" * n_algo

    lines = []
    lines.append(r"\begin{tableorg}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Broadcast time for $N = " + str(N) + r"$.}")
    lines.append(r"\label{table" + str(N) + r"}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\resizebox{!}{0.5\textheight}{%")
    lines.append(r"\setlength{\tabcolsep}{20pt}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Header row
    hdr = ["Topology", "Message Size", "Metric"]
    for algo in ALGORITHMS:
        hdr.append(ALGO_LABEL[algo])
    lines.append(" & ".join(hdr) + r" \\")
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

        for m_idx, msg in enumerate(MSG_SIZES):
            for s_idx, (stat, stat_lbl) in enumerate(zip(STATS, STAT_LABEL)):
                parts = []

                # Topology column (multirow on first stat of first msg)
                if m_idx == 0 and s_idx == 0:
                    parts.append(r"\multirow{" + str(n_rows_per_topo) + r"}{*}{" + topo_lbl + r"}")
                else:
                    parts.append("")

                # Message size column (multirow on first stat)
                if s_idx == 0:
                    parts.append(r"\multirow{" + str(n_stat) + r"}{*}{" + bytes_label(msg) + r"}")
                    # First stat row gets the metric with leading whitespace
                    parts.append("  " + stat_lbl)
                else:
                    parts.append("")
                    parts.append(stat_lbl)

                # Algorithm values
                for algo in ALGORITHMS:
                    rec = load_record(topo, algo, N, msg)
                    if rec is None:
                        parts.append("---")
                    else:
                        val = rec[stat]
                        s = fmt_time(val)
                        if stat == "mean_sec" and msg in best_mean and val <= best_mean[msg]:
                            s = r"\textbf{" + s + r"}"
                        parts.append(s)

                # Build the row
                # For non-first stat rows, prefix with "& &" style
                if s_idx == 0 and m_idx == 0:
                    # First msg, first stat: topology multirow & msg multirow & stat
                    row = parts[0] + "\n& " + parts[1] + "\n  & " + parts[2]
                elif s_idx == 0:
                    # First stat of subsequent msgs
                    row = "& " + parts[1] + "\n  & " + parts[2]
                else:
                    # Subsequent stats
                    row = "& & " + parts[2]

                # Append algorithm values
                for p in parts[3:]:
                    row += " & " + p
                row += r" \\"

                lines.append(row)

            # cmidrule between message sizes (but not after the last one)
            if m_idx < n_msg - 1:
                lines.append(r"\cmidrule{2-9}")
                lines.append("")

        # midrule between topologies, blank line after last msg of a topo
        if t_idx < len(TOPOLOGIES) - 1:
            lines.append("")
            lines.append(r"\midrule")
            lines.append("")

    lines.append("")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
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
