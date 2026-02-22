#!/usr/bin/env python3
"""
aggregate.py — Compute mean ± stdev of time_sec across all roots
for each (topology, algorithm, N, MSG) combo.

Only includes combos where ALL N roots are present.
Outputs a markdown table to stdout.
"""

import json
import os
import sys
import math
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TOPOS = ["2Dmesh", "Butterfly", "Dragonfly", "FatTree"]
ALGOS = ["mpi", "srda", "pipe"]
SIZES = [128, 256, 512, 1024]
MSGS = [256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]


def human_bytes(b):
    """Format bytes as human-readable."""
    if b >= 1048576:
        return f"{b // 1048576}MB"
    elif b >= 1024:
        return f"{b // 1024}KB"
    return f"{b}B"


def collect():
    """Collect all results, return dict keyed by (topo, algo, N, msg) -> list of time_sec."""
    results = defaultdict(list)

    for topo in TOPOS:
        for algo in ALGOS:
            for n in SIZES:
                for msg in MSGS:
                    times = []
                    complete = True
                    for root in range(n):
                        fpath = DATA_DIR / topo / algo / f"N{n}_MSG{msg}_R{root}.json"
                        if not fpath.exists():
                            complete = False
                            break
                        try:
                            with open(fpath) as f:
                                d = json.load(f)
                            times.append(d["time_sec"])
                        except (json.JSONDecodeError, KeyError):
                            complete = False
                            break

                    if complete and len(times) == n:
                        results[(topo, algo, n, msg)] = times

    return results


def main():
    results = collect()

    complete_count = len(results)
    total_possible = len(TOPOS) * len(ALGOS) * len(SIZES) * len(MSGS)
    print(f"Complete combos: {complete_count} / {total_possible}", file=sys.stderr)

    # Format times in scientific notation for small values, otherwise fixed
    def fmt_time(mean, std):
        if mean < 0.001:
            return f"{mean:.3e} ± {std:.3e}"
        elif mean < 1.0:
            return f"{mean:.6f} ± {std:.6f}"
        else:
            return f"{mean:.4f} ± {std:.4f}"

    # Group by topology for organized output
    for topo in TOPOS:
        print(f"\n### {topo}\n")
        print("| N | MSG | MPI | SRDA | Pipe | Best Speedup |")
        print("|---|-----|-----|------|------|-------------:|")

        for n in SIZES:
            for msg in MSGS:
                mpi_key = (topo, "mpi", n, msg)
                srda_key = (topo, "srda", n, msg)
                pipe_key = (topo, "pipe", n, msg)

                # Need at least MPI as baseline
                if mpi_key not in results:
                    continue

                mpi_times = results[mpi_key]
                mpi_mean = sum(mpi_times) / len(mpi_times)
                mpi_std = math.sqrt(sum((t - mpi_mean) ** 2 for t in mpi_times) / len(mpi_times))

                cols = [f"{n}", human_bytes(msg), fmt_time(mpi_mean, mpi_std)]
                best_mean = mpi_mean

                for key in [srda_key, pipe_key]:
                    if key in results:
                        times = results[key]
                        mean = sum(times) / len(times)
                        std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
                        cols.append(fmt_time(mean, std))
                        if mean < best_mean:
                            best_mean = mean
                    else:
                        cols.append("—")

                speedup = mpi_mean / best_mean if best_mean > 0 else float('inf')
                cols.append(f"{speedup:.2f}x")

                print("| " + " | ".join(cols) + " |")

    # Also output a CSV for further analysis
    csv_path = DATA_DIR / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("topology,algorithm,nodes,msg_bytes,mean_sec,stdev_sec,n_roots\n")
        for (topo, algo, n, msg), times in sorted(results.items()):
            mean = sum(times) / len(times)
            std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
            f.write(f"{topo},{algo},{n},{msg},{mean:.9f},{std:.9f},{len(times)}\n")

    print(f"\nCSV written to {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
