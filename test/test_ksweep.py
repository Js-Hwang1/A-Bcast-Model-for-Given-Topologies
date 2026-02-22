#!/usr/bin/env python3
"""
test_ksweep.py — Verify physics-aware K-sweep and rationalization fix.

Tests:
  1. rationalize() preserves relative LP edge rates (not all-ones)
  2. bfs_diameter() returns correct values for known topologies
  3. compute_optimal_K() matches analytical formula K = sqrt(D*C*M/(L*B))
  4. T(K_opt) <= T(K) for all K in [1, 10*K_opt] (true minimum)
  5. write_params() produces valid JSON with sane K_opt values
  6. End-to-end: generate_plan() creates .plan + .params for 12+ cases
  7. K_opt scales as sqrt(M) when other params are fixed
"""

import sys
import os
import json
import math
import tempfile
import shutil

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from LP import (
    rationalize, bfs_diameter, bfs_distances, compute_optimal_K, T_model,
    write_params, generate_plan, solve_lp,
    adj_2dmesh, adj_butterfly, adj_fattree, adj_dragonfly,
    ADJ_BUILDERS, TOPO_PHYSICS, MSG_SIZES,
)

PASS = 0
FAIL = 0
TESTS = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        TESTS.append(("PASS", name, detail))
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        TESTS.append(("FAIL", name, detail))
        print(f"  FAIL  {name}  -- {detail}")


# =============================================================
# 1. Rationalize preserves relative LP rates
# =============================================================
print("\n=== 1. Rationalize: preserves relative edge importance ===")

# Synthetic LP output: edges with varying weights
synthetic_weights = {
    (0, 1): 0.50,
    (0, 2): 0.25,
    (1, 3): 0.50,
    (2, 3): 0.125,
    (3, 4): 0.50,
    (1, 4): 0.0625,  # just above 1% of max=0.50 → survives
}

int_w = rationalize(synthetic_weights, F_target=4)

# The old bug: all weights = 1 regardless of LP rate
all_ones = all(v == 1 for v in int_w.values())
check("rationalize: NOT all-ones",
      not all_ones,
      f"got weights {dict(int_w)}")

# max LP weight (0.50) should map to F_target=4
check("rationalize: max weight maps to F_target",
      int_w[(0, 1)] == 4,
      f"expected 4, got {int_w.get((0,1))}")

# 0.25 / 0.50 = 0.5 → round(0.5 * 4) = round(2.0) = 2
check("rationalize: half-weight edge gets ~2",
      int_w[(0, 2)] == 2,
      f"expected 2, got {int_w.get((0,2))}")

# 0.125 / 0.50 = 0.25 → round(0.25 * 4) = round(1.0) = 1
check("rationalize: quarter-weight edge gets 1",
      int_w[(2, 3)] == 1,
      f"expected 1, got {int_w.get((2,3))}")

# Check ordering is preserved: w(0,1) >= w(0,2) >= w(2,3)
check("rationalize: ordering preserved",
      int_w[(0, 1)] >= int_w[(0, 2)] >= int_w[(2, 3)],
      f"weights: {int_w[(0,1)]}, {int_w[(0,2)]}, {int_w[(2,3)]}")

# Noise edge below 1% of max should be dropped
noise_weights = {(0, 1): 1.0, (0, 2): 0.005}  # 0.005 < 0.01 * 1.0
int_noise = rationalize(noise_weights, F_target=4)
check("rationalize: noise edges dropped",
      (0, 2) not in int_noise,
      f"got {dict(int_noise)}")


# =============================================================
# 2. BFS diameter: known topology values
# =============================================================
print("\n=== 2. BFS diameter: correctness on known topologies ===")

# Butterfly N=128 (hypercube dim=7): diameter = 7
adj128 = adj_butterfly(128)
d128 = bfs_diameter(adj128, 128, 0)
check("bfs_diameter: Butterfly N=128 = 7",
      d128 == 7,
      f"got {d128}")

# Butterfly N=256 (dim=8): diameter = 8
adj256 = adj_butterfly(256)
d256 = bfs_diameter(adj256, 256, 0)
check("bfs_diameter: Butterfly N=256 = 8",
      d256 == 8,
      f"got {d256}")

# 2Dmesh N=128 (8x16): diameter = 7+15 = 22 from corner (0,0)
adj_m128 = adj_2dmesh(128)
dm128 = bfs_diameter(adj_m128, 128, 0)
check("bfs_diameter: 2Dmesh N=128 (8x16) = 22",
      dm128 == 22,
      f"got {dm128}")

# 2Dmesh N=256 (16x16): diameter from (0,0) = 15+15 = 30
adj_m256 = adj_2dmesh(256)
dm256 = bfs_diameter(adj_m256, 256, 0)
check("bfs_diameter: 2Dmesh N=256 (16x16) = 30",
      dm256 == 30,
      f"got {dm256}")

# FatTree N=128: check it's reasonable (should be small due to cliques)
adj_ft128 = adj_fattree(128)
dft128 = bfs_diameter(adj_ft128, 128, 0)
check("bfs_diameter: FatTree N=128 > 0 and <= 5",
      0 < dft128 <= 5,
      f"got {dft128}")


# =============================================================
# 3. compute_optimal_K: analytical formula verification
# =============================================================
print("\n=== 3. compute_optimal_K: matches analytical formula ===")

# Known case: Butterfly N=128, 16MB
# B=12.5e9, L=100e-9, C*≈0.142857 (1/7), D=7, F≈7
# K_anal = sqrt(7 * 0.142857 * 16777216 / (100e-9 * 12.5e9))
#        = sqrt(7 * 0.142857 * 16777216 / 1.25)
#        = sqrt(13421772.8) ≈ 3664 ... but let's test the formula directly

def analytical_K(D, C_star, M, L, B):
    return math.sqrt(D * C_star * M / (L * B))

# Test with clean numbers: D=10, C=0.5, M=1e6, L=1e-6, B=1e9
# K_anal = sqrt(10 * 0.5 * 1e6 / (1e-6 * 1e9)) = sqrt(5e6 / 1e3) = sqrt(5000) ≈ 70.7
K_opt, T_opt = compute_optimal_K(C_eff=0.5, F=5, D=10, B=1e9, L=1e-6, M=1_000_000)
K_anal_val = analytical_K(10, 0.5, 1e6, 1e-6, 1e9)
check("compute_optimal_K: clean numbers K≈71",
      abs(K_opt - round(K_anal_val)) <= 1,
      f"K_opt={K_opt}, K_anal={K_anal_val:.1f}")

# Verify T(K_opt) is actually the minimum by exhaustive check
T_at_opt = T_model(K_opt, 0.5, 5, 10, 1e9, 1e-6, 1e6)
check("compute_optimal_K: T_opt matches T(K_opt)",
      abs(T_at_opt - T_opt) / T_opt < 1e-9,
      f"T_opt={T_opt}, T(K_opt)={T_at_opt}")


# =============================================================
# 4. K_opt is true global minimum over wide range
# =============================================================
print("\n=== 4. K_opt is true minimum (exhaustive verification) ===")

test_cases = [
    # (C_eff, F, D, B, L, M, label)
    # C_eff = min in-degree of non-root in active subgraph (with F_target=1)
    (7, 7, 7, 12.5e9, 100e-9, 16777216, "Butterfly-128 16MB"),
    (7, 7, 7, 12.5e9, 100e-9, 1048576, "Butterfly-128 1MB"),
    (7, 7, 7, 12.5e9, 100e-9, 1024, "Butterfly-128 1KB"),
    (2, 4, 22, 50e9, 100e-9, 16777216, "2Dmesh-128 16MB"),
    (2, 4, 22, 50e9, 100e-9, 1048576, "2Dmesh-128 1MB"),
    (2, 4, 22, 50e9, 100e-9, 65536, "2Dmesh-128 64KB"),
    (8, 16, 3, 12.5e9, 100e-9, 67108864, "FatTree-128 64MB"),
    (8, 16, 3, 12.5e9, 100e-9, 4194304, "FatTree-128 4MB"),
    (2, 6, 5, 5.25e9, 400e-9, 16777216, "Dragonfly-128 16MB"),
    (2, 6, 5, 5.25e9, 400e-9, 262144, "Dragonfly-128 256KB"),
    (2, 4, 30, 50e9, 100e-9, 268435456, "2Dmesh-256 256MB"),
    (8, 8, 8, 12.5e9, 100e-9, 67108864, "Butterfly-256 64MB"),
]

for C_eff, F, D, B, L, M, label in test_cases:
    K_opt, T_opt = compute_optimal_K(C_eff, F, D, B, L, M)

    # Exhaustive search over [1, 20*K_opt] — K_opt must beat all
    is_minimum = True
    worst_ratio = 1.0
    for K in range(1, max(2, 20 * K_opt)):
        T_k = T_model(K, C_eff, F, D, B, L, M)
        if T_k < T_opt - 1e-15 * T_opt:  # allow float rounding
            is_minimum = False
            worst_ratio = T_k / T_opt
            break

    check(f"global minimum: {label} K_opt={K_opt}",
          is_minimum,
          f"found K={K} with T ratio {worst_ratio:.6f}" if not is_minimum else f"T_opt={T_opt:.6e}")


# =============================================================
# 5. K_opt scales as sqrt(M) (physics sanity)
# =============================================================
print("\n=== 5. K_opt scales as sqrt(M) ===")

# For fixed D, C_eff, B, L: K_opt ∝ sqrt(M)
# So K_opt(4M) / K_opt(M) ≈ 2
C_s, F_s, D_s, B_s, L_s = 7, 7, 7, 12.5e9, 100e-9
K_1mb, _ = compute_optimal_K(C_s, F_s, D_s, B_s, L_s, 1048576)
K_4mb, _ = compute_optimal_K(C_s, F_s, D_s, B_s, L_s, 4194304)
K_16mb, _ = compute_optimal_K(C_s, F_s, D_s, B_s, L_s, 16777216)
K_64mb, _ = compute_optimal_K(C_s, F_s, D_s, B_s, L_s, 67108864)

ratio_4x = K_4mb / K_1mb    # should be ~2
ratio_16x = K_16mb / K_1mb  # should be ~4
ratio_64x = K_64mb / K_1mb  # should be ~8

check("sqrt(M) scaling: K(4MB)/K(1MB) ≈ 2.0",
      1.5 < ratio_4x < 2.5,
      f"ratio = {ratio_4x:.2f}")

check("sqrt(M) scaling: K(16MB)/K(1MB) ≈ 4.0",
      3.0 < ratio_16x < 5.0,
      f"ratio = {ratio_16x:.2f}")

check("sqrt(M) scaling: K(64MB)/K(1MB) ≈ 8.0",
      6.0 < ratio_64x < 10.0,
      f"ratio = {ratio_64x:.2f}")


# =============================================================
# 6. K_opt >> 64 for large messages (shows hardcoded 64 is wrong)
# =============================================================
print("\n=== 6. K_opt vs hardcoded K=64 ===")

for C_eff, F, D, B, L, M, label in test_cases:
    if M >= 4194304:  # only large messages
        K_opt, T_opt = compute_optimal_K(C_eff, F, D, B, L, M)
        T_64 = T_model(64, C_eff, F, D, B, L, M)
        improvement = (T_64 - T_opt) / T_64 * 100

        check(f"K_opt beats K=64: {label}",
              T_opt <= T_64,
              f"K_opt={K_opt}, T_opt={T_opt:.6e} vs T(64)={T_64:.6e}, "
              f"improvement={improvement:.1f}%")


# =============================================================
# 7. End-to-end: generate_plan + write_params
# =============================================================
print("\n=== 7. End-to-end pipeline: generate_plan + .params ===")

tmpdir = tempfile.mkdtemp(prefix="bbs_test_")
e2e_cases = [
    ("Butterfly", 128, 0),
    ("Butterfly", 256, 0),
    ("2Dmesh", 128, 0),
    ("2Dmesh", 256, 0),
    ("FatTree", 128, 0),
    ("FatTree", 256, 0),
    ("Dragonfly", 128, 0),
    ("Dragonfly", 256, 0),
]

# Monkey-patch PLAN_DIR for isolated testing
import LP
old_plan_dir = LP.PLAN_DIR
LP.PLAN_DIR = tmpdir

try:
    for topo, n, root in e2e_cases:
        plan_path, C_val, nframes = generate_plan(topo, n, root, verbose=False)

        # Check .plan file exists and has frames
        check(f"e2e plan: {topo} N={n} .plan exists",
              os.path.isfile(plan_path),
              f"path={plan_path}")

        with open(plan_path) as f:
            plan_content = f.read()
        frame_count = plan_content.count("FRAME ")
        check(f"e2e plan: {topo} N={n} has {nframes} frames",
              frame_count == nframes,
              f"expected {nframes}, found {frame_count} FRAME lines")

        # Check .params file exists and has valid JSON
        params_path = plan_path.replace(".plan", ".params")
        check(f"e2e params: {topo} N={n} .params exists",
              os.path.isfile(params_path),
              f"path={params_path}")

        with open(params_path) as f:
            params = json.load(f)

        # Verify params structure
        check(f"e2e params: {topo} N={n} has C_star > 0",
              params["C_star"] > 0,
              f"C_star={params['C_star']}")

        check(f"e2e params: {topo} N={n} has all MSG_SIZES",
              len(params["optimal_K"]) == len(MSG_SIZES),
              f"got {len(params['optimal_K'])}, expected {len(MSG_SIZES)}")

        # Spot-check: K_opt for 16MB should be > 4 and < 100000
        k_16m = params["optimal_K"]["16777216"]["K_opt"]
        check(f"e2e params: {topo} N={n} K_opt(16MB) sane",
              4 < k_16m < 100000,
              f"K_opt={k_16m}")

        # C_star in params is C_eff (effective throughput from rationalization)
        # It should be >= 1 (every non-root node has at least 1 incoming edge)
        check(f"e2e params: {topo} N={n} C_eff={params['C_star']} >= 1",
              params["C_star"] >= 1,
              f"C_eff={params['C_star']}")

finally:
    LP.PLAN_DIR = old_plan_dir
    shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================
# 8. Rationalize on REAL LP output (not synthetic)
# =============================================================
print("\n=== 8. Rationalize on real LP output ===")

for topo, n in [("Butterfly", 128), ("2Dmesh", 128), ("FatTree", 128)]:
    adj = ADJ_BUILDERS[topo](n)
    edge_weights, C_val = solve_lp(adj, n, root=0)

    # Old behavior: all ones
    old_result = {e: 1 for e in edge_weights if edge_weights[e] > max(edge_weights.values()) * 0.01}

    # New behavior: scaled rounding
    new_result = rationalize(edge_weights, F_target=4)

    # New result should have SOME edges with weight > 1
    has_varied = any(v > 1 for v in new_result.values())
    check(f"real LP rationalize: {topo} N={n} has varied weights",
          has_varied,
          f"max_w={max(new_result.values())}, unique={len(set(new_result.values()))}")

    # New result should have MORE total weight (= more frame-slots for important edges)
    old_total = sum(old_result.values())
    new_total = sum(new_result.values())
    check(f"real LP rationalize: {topo} N={n} total weight increased",
          new_total >= old_total,
          f"old={old_total}, new={new_total}")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 60)

if FAIL > 0:
    print("\nFailed tests:")
    for status, name, detail in TESTS:
        if status == "FAIL":
            print(f"  - {name}: {detail}")
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
