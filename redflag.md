# Baseline Data Integrity Red Flags

## Summary

- **Datapoint counts**: ALL PASS. Every file has `n_roots == N` and `per_root_sec` has exactly N entries.
- **Monotonicity violations**: 77 cases where mean time *decreased* as message size grew.
- **Root cause**: 154 configs contain extreme outlier roots (>100x median). A single bad root can inflate the mean by 10-1000x, making a smaller message appear slower than a larger one.

## Root Cause: Extreme Per-Root Outliers

The non-monotonic means are **not** caused by incorrect data collection — the per-root timings are internally consistent. The problem is that certain roots produce times 100-22,000x the median, massively skewing the mean. Examples:

| Config | Root | Time | Median | Multiplier |
|--------|------|------|--------|------------|
| FatTree/mpi N=1024 MSG=256 | root 3 | 4.76e-01 | 2.10e-05 | **22,658x** |
| FatTree/bine N=256 MSG=256 | root 38 | 3.14e-01 | 1.38e-05 | **22,658x** |
| FatTree/mpi N=1024 MSG=1024 | root 726 | 3.08e-01 | 2.87e-05 | **10,733x** |
| FatTree/bine N=1024 MSG=4096 | root 332 | 6.25e-01 | 5.71e-05 | **10,947x** |
| Butterfly/srda N=1024 MSG=256 | root 72 | 4.99e-01 | 5.90e-05 | **8,446x** |
| FatTree/mpi N=512 MSG=4096 | root 260 | 1.18e-01 | 2.93e-05 | **4,032x** |
| 2Dmesh/mpi N=512 MSG=256 | root 380 | 1.04e-01 | 2.76e-05 | **3,753x** |
| FatTree/glf N=1024 MSG=256 | root 788 | 2.15e-01 | 5.95e-05 | **3,604x** |
| FatTree/srda N=1024 MSG=256 | root 237 | 2.71e-01 | 7.73e-05 | **3,498x** |

Total: 154 configs have at least one root >100x the median.

## Non-Monotonic Mean Times (77 violations)

Expected: larger message size -> larger broadcast time. Sorted by severity.

### Critical (>50% decrease) — 17 cases

| Topology/Algo | N | MSG transition | Mean before | Mean after | Drop |
|---------------|---|----------------|-------------|------------|------|
| FatTree/bine | 256 | 256 -> 1024 | 1.243e-03 | 1.805e-05 | 98.5% |
| FatTree/mpi | 1024 | 1024 -> 4096 | 5.047e-04 | 3.595e-05 | 92.9% |
| 2Dmesh/mpi | 512 | 256 -> 1024 | 2.300e-04 | 2.541e-05 | 89.0% |
| Butterfly/srda | 1024 | 256 -> 1024 | 6.444e-04 | 7.660e-05 | 88.1% |
| FatTree/srda | 512 | 4096 -> 16384 | 5.104e-04 | 9.593e-05 | 81.2% |
| FatTree/srda | 1024 | 256 -> 1024 | 6.170e-04 | 1.284e-04 | 79.2% |
| Butterfly/glf | 1024 | 1024 -> 4096 | 4.081e-04 | 8.802e-05 | 78.4% |
| 2Dmesh/srda | 512 | 16384 -> 65536 | 7.574e-04 | 1.858e-04 | 75.5% |
| FatTree/glf | 1024 | 256 -> 1024 | 2.820e-04 | 7.530e-05 | 73.3% |
| 2Dmesh/srda | 1024 | 16384 -> 65536 | 7.038e-04 | 2.008e-04 | 71.5% |
| Dragonfly/srda | 1024 | 256 -> 1024 | 3.391e-04 | 1.046e-04 | 69.2% |
| FatTree/mpi | 1024 | 16384 -> 65536 | 3.921e-04 | 1.272e-04 | 67.6% |
| 2Dmesh/glf | 1024 | 256 -> 1024 | 4.912e-04 | 1.716e-04 | 65.1% |
| FatTree/bine | 1024 | 4096 -> 16384 | 6.700e-04 | 2.355e-04 | 64.8% |
| 2Dmesh/glf | 512 | 256 -> 1024 | 2.903e-04 | 1.120e-04 | 61.4% |
| FatTree/mpi | 512 | 4096 -> 16384 | 2.604e-04 | 1.042e-04 | 60.0% |
| Butterfly/bine | 1024 | 256 -> 1024 | 1.181e-04 | 5.093e-05 | 56.9% |

### Major (20-50% decrease) — 18 cases

| Topology/Algo | N | MSG transition | Mean before | Mean after | Drop |
|---------------|---|----------------|-------------|------------|------|
| FatTree/bine | 1024 | 256 -> 1024 | 1.300e-04 | 5.859e-05 | 54.9% |
| Butterfly/glf | 512 | 4096 -> 16384 | 3.663e-04 | 1.816e-04 | 50.4% |
| Dragonfly/srda | 512 | 4096 -> 16384 | 6.923e-04 | 3.503e-04 | 49.4% |
| 2Dmesh/glf | 256 | 256 -> 1024 | 8.528e-05 | 4.520e-05 | 47.0% |
| Dragonfly/bine | 512 | 4096 -> 16384 | 4.764e-04 | 2.571e-04 | 46.0% |
| Dragonfly/bine | 1024 | 16384 -> 65536 | 4.840e-04 | 2.770e-04 | 42.8% |
| Dragonfly/bine | 512 | 256 -> 1024 | 1.784e-04 | 1.042e-04 | 41.6% |
| Butterfly/glf | 128 | 256 -> 1024 | 4.101e-05 | 2.447e-05 | 40.3% |
| 2Dmesh/glf | 512 | 1024 -> 4096 | 1.120e-04 | 6.788e-05 | 39.4% |
| Butterfly/glf | 1024 | 256 -> 1024 | 6.564e-04 | 4.081e-04 | 37.8% |
| Butterfly/glf | 256 | 1024 -> 4096 | 5.791e-05 | 3.942e-05 | 31.9% |
| Butterfly/pipe | 256 | 256 -> 1024 | 3.398e-04 | 2.409e-04 | 29.1% |
| Butterfly/pipe | 1024 | 1024 -> 4096 | 3.614e-03 | 2.585e-03 | 28.5% |
| 2Dmesh/srda | 512 | 1024 -> 4096 | 2.336e-04 | 1.729e-04 | 26.0% |
| Butterfly/srda | 128 | 256 -> 1024 | 2.078e-05 | 1.606e-05 | 22.7% |
| 2Dmesh/pipe | 512 | 1024 -> 4096 | 1.453e-03 | 1.121e-03 | 22.9% |
| FatTree/pipe | 512 | 4096 -> 16384 | 1.157e-03 | 9.225e-04 | 20.3% |
| Dragonfly/glf | 1024 | 256 -> 1024 | 1.317e-04 | 1.053e-04 | 20.1% |

### Moderate (5-20% decrease) — 30 cases

| Topology/Algo | N | MSG transition | Drop |
|---------------|---|----------------|------|
| 2Dmesh/bine N=1024 | 256 -> 1024 | 19.2% |
| Butterfly/bine N=128 | 1024 -> 4096 | 17.7% |
| 2Dmesh/glf N=128 | 1024 -> 4096 | 17.0% |
| Butterfly/glf N=256 | 256 -> 1024 | 16.5% |
| 2Dmesh/pipe N=256 | 256 -> 1024 | 16.3% |
| 2Dmesh/glf N=1024 | 1024 -> 4096 | 15.2% |
| Butterfly/srda N=256 | 256 -> 1024 | 15.0% |
| FatTree/glf N=1024 | 1024 -> 4096 | 14.4% |
| 2Dmesh/mpi N=128 | 1024 -> 4096 | 11.0% |
| FatTree/bine N=512 | 1024 -> 4096 | 11.1% |
| 2Dmesh/bine N=1024 | 1024 -> 4096 | 10.7% |
| Dragonfly/mpi N=1024 | 16384 -> 65536 | 10.8% |
| Dragonfly/mpi N=512 | 16384 -> 65536 | 10.0% |
| FatTree/pipe N=1024 | 262144 -> 1048576 | 9.5% |
| 2Dmesh/mpi N=1024 | 65536 -> 262144 | 9.1% |
| 2Dmesh/pipe N=256 | 1024 -> 4096 | 8.9% |
| Dragonfly/pipe N=1024 | 65536 -> 262144 | 8.8% |
| 2Dmesh/mpi N=1024 | 256 -> 1024 | 8.7% |
| Dragonfly/glf N=256 | 16384 -> 65536 | 8.6% |
| Butterfly/pipe N=128 | 256 -> 1024 | 8.3% |
| Dragonfly/glf N=512 | 256 -> 1024 | 8.2% |
| Dragonfly/bine N=1024 | 1024 -> 4096 | 7.7% |
| FatTree/srda N=1024 | 1024 -> 4096 | 7.4% |
| Butterfly/mpi N=512 | 1024 -> 4096 | 7.1% |
| Butterfly/srda N=256 | 4096 -> 16384 | 7.0% |
| Dragonfly/bine N=512 | 16384 -> 65536 | 6.4% |
| Butterfly/pipe N=512 | 1024 -> 4096 | 5.7% |
| FatTree/bine N=1024 | 16384 -> 65536 | 5.2% |
| FatTree/mpi N=1024 | 256 -> 1024 | 5.1% |
| Dragonfly/glf N=512 | 1024 -> 4096 | 4.6% |

### Minor (<5% decrease) — 12 cases

| Topology/Algo | N | MSG transition | Drop |
|---------------|---|----------------|------|
| 2Dmesh/srda N=128 | 1024 -> 4096 | 4.4% |
| 2Dmesh/bine N=128 | 1024 -> 4096 | 4.3% |
| 2Dmesh/pipe N=1024 | 1024 -> 4096 | 4.2% |
| Dragonfly/glf N=512 | 16384 -> 65536 | 4.1% |
| FatTree/pipe N=1024 | 65536 -> 262144 | 3.3% |
| 2Dmesh/pipe N=512 | 16384 -> 65536 | 2.1% |
| 2Dmesh/pipe N=1024 | 256 -> 1024 | 1.8% |
| Dragonfly/bine N=256 | 16384 -> 65536 | 1.2% |
| Dragonfly/glf N=128 | 16384 -> 65536 | 0.8% |
| FatTree/glf N=512 | 16384 -> 65536 | 0.8% |
| 2Dmesh/pipe N=128 | 256 -> 1024 | 0.5% |
| 2Dmesh/pipe N=1024 | 4096 -> 16384 | 0.1% |

## Configs Needing Re-run (Recommended)

These configs have the most severe outliers and should be re-run to get clean data. Prioritized by impact (outliers that cause non-monotonic violations):

1. `FatTree/bine N=256 MSG=256` — root 38 at 22,658x median (causes 98.5% drop)
2. `FatTree/mpi N=1024 MSG=256` — root 3 at 22,658x median
3. `FatTree/mpi N=1024 MSG=1024` — roots 726,732,738 at 420-10,733x median (causes 92.9% drop)
4. `2Dmesh/mpi N=512 MSG=256` — root 380 at 3,753x median (causes 89.0% drop)
5. `Butterfly/srda N=1024 MSG=256` — root 72 at 8,446x median (causes 88.1% drop)
6. `FatTree/srda N=512 MSG=4096` — roots 32,492 at 399-2,884x median (causes 81.2% drop)
7. `FatTree/srda N=1024 MSG=256` — roots 237,622 at 3,399-3,498x median (causes 79.2% drop)
8. `FatTree/mpi N=512 MSG=4096` — root 260 at 4,032x median (causes 60.0% drop)
9. `FatTree/mpi N=1024 MSG=16384` — root 837 at 2,057x median (causes 67.6% drop)
10. `FatTree/bine N=1024 MSG=4096` — root 332 at 10,947x median (causes 64.8% drop)
11. `Dragonfly/srda N=1024 MSG=256` — root 864 at 2,718x median (causes 69.2% drop)
12. `FatTree/glf N=1024 MSG=256` — root 788 at 3,604x median (causes 73.3% drop)
13. `2Dmesh/srda N=1024 MSG=16384` — roots 360,366 at 2,736-3,756x median (causes 71.5% drop)
14. `2Dmesh/srda N=512 MSG=16384` — roots 260,310,326,418 at 892-1,832x median (causes 75.5% drop)
15. `2Dmesh/glf N=1024 MSG=256` — root 13 at 405x median (causes 65.1% drop)
