#!/usr/bin/env python3
"""Plot f(k) from test output and fit linear model."""

import sys
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

# Linear fit: f(k) = a*k + b
coeffs = np.polyfit(ks, ts, 1)
a, b = coeffs
fitted = np.polyval(coeffs, ks)
residuals = ts - fitted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ts - np.mean(ts))**2)
r_squared = 1 - ss_res / ss_tot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: f(k) with linear fit
ax1.scatter(ks, ts * 1e6, s=12, alpha=0.7, label="Measured f(k)", zorder=3)
ax1.plot(ks, fitted * 1e6, "r-", linewidth=2,
         label=f"Linear fit: {a*1e6:.2f}k + {b*1e6:.2f} us\n$R^2$ = {r_squared:.6f}")
ax1.set_xlabel("k (number of children)", fontsize=12)
ax1.set_ylabel("Time (microseconds)", fontsize=12)
ax1.set_title("f(k): Time for node 0 to send 1MB to k children\n(shared uplink, 50 GB/s)", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Right: residuals
ax2.scatter(ks, residuals * 1e6, s=12, alpha=0.7, color="green")
ax2.axhline(y=0, color="r", linestyle="--")
ax2.set_xlabel("k (number of children)", fontsize=12)
ax2.set_ylabel("Residual (microseconds)", fontsize=12)
ax2.set_title("Residuals from linear fit", fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fk_plot.png", dpi=150, bbox_inches="tight")
print(f"Saved fk_plot.png")
print(f"Linear fit: f(k) = {a*1e6:.4f} * k + {b*1e6:.4f}  (microseconds)")
print(f"R-squared:  {r_squared:.8f}")
print(f"f(1)  = {ts[0]*1e6:.2f} us")
print(f"f(100)= {ts[-1]*1e6:.2f} us")
print(f"Ratio f(100)/f(1) = {ts[-1]/ts[0]:.2f}x")
