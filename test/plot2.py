#!/usr/bin/env python3
"""Plot f(k) excluding k=1 warmup outlier, with linear fit and theory."""

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
ks_all = np.array([d[0] for d in data])
ts_all = np.array([d[1] for d in data])

# Exclude k=1 warmup outlier
mask = ks_all >= 2
ks = ks_all[mask]
ts = ts_all[mask]

# Linear fit on k >= 2
coeffs = np.polyfit(ks, ts, 1)
a, b = coeffs
fitted = np.polyval(coeffs, ks)
residuals = ts - fitted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ts - np.mean(ts))**2)
r_squared = 1 - ss_res / ss_tot

# Theoretical: 1MB / 50GBps = 20.97 us per child, + 200ns latency
theory_slope = 1048576 / 50e9  # seconds per child
theory_intercept = 200e-9  # 2 * 100ns latency

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: f(k) with fit (excluding warmup)
ax = axes[0]
ax.scatter(ks, ts * 1e6, s=12, alpha=0.7, label="Measured f(k)", zorder=3)
ax.plot(ks, fitted * 1e6, "r-", linewidth=2,
        label=f"Linear fit: {a*1e6:.2f}k + {b*1e6:.2f} us")
ax.plot(ks, (theory_slope * ks + theory_intercept) * 1e6, "g--", linewidth=1.5,
        label=f"Theory: {theory_slope*1e6:.2f}k + {theory_intercept*1e6:.2f} us")
ax.set_xlabel("k (number of children)", fontsize=12)
ax.set_ylabel("Time (microseconds)", fontsize=12)
ax.set_title(f"f(k) for k=2..100 (warmup excluded)\n$R^2$ = {r_squared:.8f}", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Middle: residuals
ax = axes[1]
ax.scatter(ks, residuals * 1e6, s=12, alpha=0.7, color="green")
ax.axhline(y=0, color="r", linestyle="--")
ax.set_xlabel("k (number of children)", fontsize=12)
ax.set_ylabel("Residual (microseconds)", fontsize=12)
ax.set_title("Residuals from linear fit", fontsize=13)
ax.grid(True, alpha=0.3)
max_res = max(abs(residuals)) * 1e6
ax.set_ylim(-max_res * 1.5, max_res * 1.5)

# Right: ratio f(k)/f(2) vs k/2 (should be linear)
ax = axes[2]
ratio = ts / ts[0]  # f(k)/f(2)
k_ratio = ks / 2.0   # k/2
ax.scatter(ks, ratio, s=12, alpha=0.7, label="f(k)/f(2)", color="purple")
ax.plot(ks, k_ratio, "r--", linewidth=1.5, label="k/2 (perfect linear)")
ax.set_xlabel("k (number of children)", fontsize=12)
ax.set_ylabel("Ratio", fontsize=12)
ax.set_title("f(k)/f(2) vs k/2", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fk_plot_clean.png", dpi=150, bbox_inches="tight")
print(f"Saved fk_plot_clean.png")
print(f"\n=== Results (k=2..100, warmup excluded) ===")
print(f"Linear fit:  f(k) = {a*1e6:.4f} * k + {b*1e6:.4f}  (microseconds)")
print(f"Theory:      f(k) = {theory_slope*1e6:.4f} * k + {theory_intercept*1e6:.4f}  (microseconds)")
print(f"Slope ratio: measured/theory = {a/theory_slope:.4f}")
print(f"R-squared:   {r_squared:.10f}")
print(f"\nf(2)   = {ts[0]*1e6:.2f} us")
print(f"f(50)  = {ts[48]*1e6:.2f} us")
print(f"f(100) = {ts[-1]*1e6:.2f} us")
print(f"f(100)/f(2) = {ts[-1]/ts[0]:.2f}x  (expected: 50.0x for perfect linear)")
