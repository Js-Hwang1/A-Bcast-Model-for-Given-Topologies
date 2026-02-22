#!/usr/bin/env python3
"""Final plot of f(k) with warmup eliminated — full k=1..100."""

import numpy as np
import matplotlib.pyplot as plt

data = []
for line in open("results.txt"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    k, t = line.split()
    data.append((int(k), float(t)))

data.sort()
ks = np.array([d[0] for d in data])
ts = np.array([d[1] for d in data])

# Linear fit
coeffs = np.polyfit(ks, ts, 1)
a, b = coeffs
fitted = np.polyval(coeffs, ks)
residuals = ts - fitted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ts - np.mean(ts))**2)
r_squared = 1 - ss_res / ss_tot

# Theory: 1MB / 50GBps per child
theory_slope = 1048576 / 50e9
theory_intercept = 200e-9

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: f(k) with linear fit
ax = axes[0]
ax.scatter(ks, ts * 1e6, s=15, alpha=0.7, label="Measured f(k)", zorder=3)
ax.plot(ks, fitted * 1e6, "r-", linewidth=2,
        label=f"Fit: {a*1e6:.2f}k + {b*1e6:.2f} us")
ax.plot(ks, (theory_slope * ks + theory_intercept) * 1e6, "g--", linewidth=1.5,
        label=f"Theory: {theory_slope*1e6:.2f}k + {theory_intercept*1e6:.2f} us")
ax.set_xlabel("k (number of children)", fontsize=12)
ax.set_ylabel("Time (microseconds)", fontsize=12)
ax.set_title(f"f(k): node 0 sends 1MB to k children (shared 50GB/s uplink)\n"
             f"$R^2$ = {r_squared:.8f}", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Middle: residuals
ax = axes[1]
ax.scatter(ks, residuals * 1e6, s=15, alpha=0.7, color="green")
ax.axhline(y=0, color="r", linestyle="--")
ax.set_xlabel("k (number of children)", fontsize=12)
ax.set_ylabel("Residual (microseconds)", fontsize=12)
ax.set_title("Residuals from linear fit", fontsize=13)
ax.grid(True, alpha=0.3)

# Right: f(k)/f(1) vs k (should be identity)
ax = axes[2]
ratio = ts / ts[0]
ax.scatter(ks, ratio, s=15, alpha=0.7, label="f(k)/f(1)", color="purple")
ax.plot(ks, ks, "r--", linewidth=1.5, label="y = k (perfect scaling)")
ax.set_xlabel("k (number of children)", fontsize=12)
ax.set_ylabel("Ratio f(k)/f(1)", fontsize=12)
ax.set_title("Scaling: f(k)/f(1) vs k", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fk_final.png", dpi=150, bbox_inches="tight")
print(f"Saved fk_final.png")
print(f"\n=== f(k) = {a*1e6:.4f} * k + {b*1e6:.4f}  (microseconds) ===")
print(f"Theory:  {theory_slope*1e6:.4f} * k + {theory_intercept*1e6:.4f}  (microseconds)")
print(f"Overhead per child: {(a - theory_slope)*1e6:.2f} us ({(a/theory_slope - 1)*100:.1f}%)")
print(f"R-squared: {r_squared:.10f}")
print(f"\nf(1)   = {ts[0]*1e6:.2f} us")
print(f"f(10)  = {ts[9]*1e6:.2f} us")
print(f"f(50)  = {ts[49]*1e6:.2f} us")
print(f"f(100) = {ts[-1]*1e6:.2f} us")
print(f"f(100)/f(1) = {ts[-1]/ts[0]:.2f}x")
