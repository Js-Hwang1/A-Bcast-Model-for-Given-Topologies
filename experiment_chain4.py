#!/usr/bin/env python3
"""Sweep nchunks for pipelined broadcast on a 4-node chain and plot time vs chunks."""

import subprocess, re, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLATFORM = "topo/chain/platform_chain4.xml"
HOSTFILE = "topo/chain/hostfile_4"
BINARY   = "bin/runner"
ALGO     = "pipe"
MSG      = 52_428_800        # 50 MiB
ROOT     = 0

# ~150 points: dense near small K, sparser for large K
chunks = sorted(set(
    list(range(1, 51, 1)) +          # 1..50  step 1
    list(range(55, 200, 5)) +         # 55..195 step 5
    list(range(200, 501, 10)) +       # 200..500 step 10
    list(range(550, 2001, 50))        # 550..2000 step 50
))

print(f"Running {len(chunks)} simulations ...", flush=True)

times = []
for i, nc in enumerate(chunks):
    cmd = [
        "smpirun", "-np", "4",
        "-platform", PLATFORM,
        "-hostfile", HOSTFILE,
        "--cfg=smpi/host-speed:2000Gf",
        "--log=root.thres:warning",
        BINARY, ALGO, str(MSG), str(nc), str(ROOT), "_",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = re.search(r"time_sec\s*:\s*([\d.eE+-]+)", out)
    if m:
        times.append(float(m.group(1)))
    else:
        print(f"  FAILED at nchunks={nc}: {out[:200]}", flush=True)
        times.append(float("nan"))
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(chunks)}]", flush=True)

chunks = np.array(chunks)
times  = np.array(times)

# Find optimum
idx_min = np.nanargmin(times)
opt_k   = chunks[idx_min]
opt_t   = times[idx_min]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(chunks, times * 1e3, linewidth=1.2, color="#2563eb")
ax.axvline(opt_k, ls="--", color="#dc2626", lw=0.9, label=f"optimum K*={opt_k}")
ax.scatter([opt_k], [opt_t * 1e3], color="#dc2626", zorder=5, s=40)
ax.annotate(f"  K*={opt_k}  T={opt_t*1e3:.4f} ms",
            (opt_k, opt_t * 1e3), fontsize=9, color="#dc2626")

ax.set_xlabel("Number of chunks (K)", fontsize=12)
ax.set_ylabel("Broadcast time (ms)", fontsize=12)
ax.set_title("Pipelined Broadcast on 4-node chain  —  50 MiB from node 0", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("chain4_chunks_vs_time.png", dpi=180)
print(f"\nPlot saved: chain4_chunks_vs_time.png")
print(f"Optimum: K*={opt_k}, T={opt_t*1e6:.1f} µs  ({opt_t*1e3:.4f} ms)")
