#!/usr/bin/env python3
"""Compare BBS (weighted adaptive) against MPI and SRDA baselines for root=0, N=128."""
import json
import os

DATA_DIR = "data"
TOPOS = ["2Dmesh", "Butterfly", "FatTree"]
MSGS = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456]
ALGOS = ["mpi", "srda", "bbs"]
N = 128
ROOT = 0

def human_bytes(b):
    if b >= 1048576: return f"{b // 1048576}MB"
    elif b >= 1024: return f"{b // 1024}KB"
    return f"{b}B"

def load_time(topo, algo, n, msg, root):
    path = os.path.join(DATA_DIR, topo, algo, f"N{n}_MSG{msg}_R{root}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return d["time_sec"]
    except:
        return None

def fmt_time(t):
    if t is None: return "—"
    if t < 0.001: return f"{t:.3e}"
    elif t < 1.0: return f"{t:.6f}"
    return f"{t:.4f}"

for topo in TOPOS:
    print(f"\n### {topo} (N={N}, root={ROOT})\n")
    print("| MSG | MPI (s) | SRDA (s) | BBS (s) | BBS vs MPI | BBS vs SRDA |")
    print("|-----|---------|----------|---------|-----------|------------|")
    for msg in MSGS:
        times = {}
        for algo in ALGOS:
            times[algo] = load_time(topo, algo, N, msg, ROOT)

        row = [human_bytes(msg)]
        for algo in ALGOS:
            row.append(fmt_time(times[algo]))

        # Speedup vs MPI
        if times["bbs"] and times["mpi"] and times["bbs"] > 0:
            sp = times["mpi"] / times["bbs"]
            row.append(f"{sp:.2f}x")
        else:
            row.append("—")

        # Speedup vs SRDA
        if times["bbs"] and times["srda"] and times["bbs"] > 0:
            sp = times["srda"] / times["bbs"]
            row.append(f"{sp:.2f}x")
        else:
            row.append("—")

        print("| " + " | ".join(row) + " |")

print()
