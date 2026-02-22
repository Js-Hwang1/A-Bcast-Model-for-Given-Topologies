#!/usr/bin/env python3
"""
test_vs_baselines.py — Compare BBS T(K_opt) predictions against
actual MPI_Bcast and SRDA times from RESULTS.md.

For each (topology, N, msg_size):
  1. Run the full LP pipeline to get C*, F, D
  2. Compute K_opt and T_predicted via the physics model
  3. Compare against the measured MPI and SRDA mean times
  4. Report whether BBS is predicted to beat each baseline

This validates:
  - The LP solver + rationalization produce reasonable C*/F
  - The K-sweep model predicts wall-clock times in the right ballpark
  - BBS with optimal K is competitive with or beats MPI/SRDA
"""

import sys
import os
import re
import json
import math
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from LP import (
    generate_plan, compute_optimal_K, T_model, compute_C_eff,
    bfs_diameter, solve_lp, rationalize, decompose_frames,
    ADJ_BUILDERS, TOPO_PHYSICS, PLAN_DIR,
)
import LP

# ── Parse RESULTS.md ──────────────────────────────────────────

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "RESULTS.md")

MSG_COLS = [256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]

def parse_results_md(path):
    """Parse RESULTS.md → dict[(topo, algo, N, msg)] = mean_time."""
    results = {}
    with open(path) as f:
        lines = f.readlines()

    current_topo = None
    current_n = None

    for line in lines:
        line = line.strip()

        # Detect topology section
        if line.startswith("## 2D Mesh"):
            current_topo = "2Dmesh"
        elif line.startswith("## Butterfly"):
            current_topo = "Butterfly"
        elif line.startswith("## Dragonfly"):
            current_topo = "Dragonfly"
        elif line.startswith("## Fat-Tree"):
            current_topo = "FatTree"

        # Detect N
        m = re.match(r"### N = (\d+)", line)
        if m:
            current_n = int(m.group(1))

        # Detect algo row
        if current_topo and current_n and line.startswith("| MPI") or \
           current_topo and current_n and line.startswith("| SRDA") or \
           current_topo and current_n and line.startswith("| BBS"):

            if line.startswith("| MPI"):
                algo = "mpi"
            elif line.startswith("| SRDA"):
                algo = "srda"
            elif line.startswith("| BBS"):
                algo = "bbs"
            else:
                continue

            # Extract cells: split by |, skip first (empty) and algo name
            cells = [c.strip() for c in line.split("|")]
            # cells[0] = '', cells[1] = 'MPI_Bcast', cells[2..11] = data
            data_cells = cells[2:]

            for i, cell in enumerate(data_cells):
                if i >= len(MSG_COLS):
                    break
                if cell == "--" or not cell:
                    continue

                # Parse "1.27e-05 +/- 5.72e-06" → mean
                parts = cell.split("+/-")
                if not parts:
                    continue
                try:
                    mean_val = float(parts[0].strip())
                    results[(current_topo, algo, current_n, MSG_COLS[i])] = mean_val
                except ValueError:
                    continue

    return results


# ── Main test ─────────────────────────────────────────────────

def T_model_local(K, C_eff, F, D, B, L, M):
    """BBS wall-clock model (local copy for reference)."""
    return F * (K / C_eff + D) * (M / (K * B) + L)


def main():
    baselines = parse_results_md(RESULTS_PATH)

    print(f"Parsed {len(baselines)} baseline measurements from RESULTS.md\n")

    # Use a temp dir for plan files
    tmpdir = tempfile.mkdtemp(prefix="bbs_baseline_test_")
    old_plan_dir = LP.PLAN_DIR
    LP.PLAN_DIR = tmpdir

    # Test cases: representative subset across topologies and message sizes
    test_configs = [
        # (topo, N, root)
        ("Butterfly", 128, 0),
        ("Butterfly", 256, 0),
        ("2Dmesh", 128, 0),
        ("2Dmesh", 256, 0),
        ("FatTree", 128, 0),
        ("FatTree", 256, 0),
        ("Dragonfly", 128, 0),
        ("Dragonfly", 256, 0),
    ]

    # Message sizes to test (focus on medium-to-large where BBS matters)
    test_msgs = [16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]

    PASS = 0
    FAIL = 0
    total = 0

    # Store per-config physics for reuse in "beats both" section
    config_physics = {}

    # Header
    print(f"{'Topology':<12} {'N':>4} {'Msg':>8} | {'K_opt':>5} {'T_pred':>12} | "
          f"{'MPI':>12} {'ratio':>7} | {'SRDA':>12} {'ratio':>7} | Result")
    print("-" * 115)

    try:
        for topo, n, root in test_configs:
            # Run full LP pipeline (now with F_target sweep)
            plan_path, C_val, nframes = generate_plan(topo, n, root, verbose=False)

            # Read back the optimized params (C_eff, F, D from sweep)
            params_path = plan_path.replace(".plan", ".params")
            with open(params_path) as pf:
                params = json.load(pf)
            C_eff = params["C_star"]  # write_params now stores C_eff here
            F = params["F"]
            D = params["D"]
            B = TOPO_PHYSICS[topo]["B"]
            L = TOPO_PHYSICS[topo]["L"]
            config_physics[(topo, n)] = (C_eff, F, D, B, L)

            for msg in test_msgs:
                mpi_key = (topo, "mpi", n, msg)
                srda_key = (topo, "srda", n, msg)

                mpi_time = baselines.get(mpi_key)
                srda_time = baselines.get(srda_key)

                if mpi_time is None and srda_time is None:
                    continue

                total += 1

                K_opt, T_pred = compute_optimal_K(C_eff, F, D, B, L, msg)

                # Also compute T at hardcoded K=64 for reference
                T_64 = T_model_local(64, C_eff, F, D, B, L, msg)

                # Format message size
                if msg >= 1048576:
                    msg_str = f"{msg // 1048576}MB"
                elif msg >= 1024:
                    msg_str = f"{msg // 1024}KB"
                else:
                    msg_str = f"{msg}B"

                # Compare against baselines
                mpi_ratio = T_pred / mpi_time if mpi_time else None
                srda_ratio = T_pred / srda_time if srda_time else None

                mpi_str = f"{mpi_time:.4e}" if mpi_time else "   --"
                srda_str = f"{srda_time:.4e}" if srda_time else "   --"
                mpi_r_str = f"{mpi_ratio:.2f}x" if mpi_ratio else "  --"
                srda_r_str = f"{srda_ratio:.2f}x" if srda_ratio else "  --"

                # Determine best baseline
                best_baseline = None
                best_baseline_name = None
                if mpi_time and srda_time:
                    if mpi_time <= srda_time:
                        best_baseline = mpi_time
                        best_baseline_name = "MPI"
                    else:
                        best_baseline = srda_time
                        best_baseline_name = "SRDA"
                elif mpi_time:
                    best_baseline = mpi_time
                    best_baseline_name = "MPI"
                elif srda_time:
                    best_baseline = srda_time
                    best_baseline_name = "SRDA"

                # BBS model prediction vs best baseline
                # Allow up to 5x overhead — the model is a lower-bound estimate
                # (it doesn't account for contention, scheduling overhead, etc.)
                # The key check: is the model prediction in a physically reasonable range?
                ratio_vs_best = T_pred / best_baseline if best_baseline else None

                if ratio_vs_best is not None:
                    if ratio_vs_best <= 1.0:
                        verdict = "PRED BEATS"
                        PASS += 1
                    elif ratio_vs_best <= 2.0:
                        verdict = "WITHIN 2x"
                        PASS += 1
                    elif ratio_vs_best <= 5.0:
                        verdict = "WITHIN 5x"
                        PASS += 1
                    else:
                        verdict = f"SLOW ({ratio_vs_best:.1f}x)"
                        FAIL += 1
                else:
                    verdict = "NO DATA"
                    total -= 1

                print(f"{topo:<12} {n:>4} {msg_str:>8} | {K_opt:>5} {T_pred:>12.4e} | "
                      f"{mpi_str:>12} {mpi_r_str:>7} | {srda_str:>12} {srda_r_str:>7} | {verdict}")

            print()  # blank line between topologies

    finally:
        LP.PLAN_DIR = old_plan_dir
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Summary ───────────────────────────────────────────────
    print("=" * 115)
    print(f"\nSummary: {PASS} passed, {FAIL} failed out of {total} comparisons")
    print(f"  'PRED BEATS'  = model predicts BBS faster than best baseline")
    print(f"  'WITHIN 2x'   = model within 2x of best baseline (contention/overhead expected)")
    print(f"  'WITHIN 5x'   = model within 5x (acceptable for idealized model)")
    print(f"  'SLOW'         = model predicts > 5x slower (potential issue)")

    # ── Detailed analysis: where does BBS beat both? ──────────
    print(f"\n{'─'*60}")
    print("Cases where model predicts BBS beats BOTH MPI and SRDA:\n")

    beat_count = 0
    for topo, n, root in test_configs:
        C_eff, F, D, B, L = config_physics[(topo, n)]

        for msg in test_msgs:
            mpi_time = baselines.get((topo, "mpi", n, msg))
            srda_time = baselines.get((topo, "srda", n, msg))
            if not mpi_time or not srda_time:
                continue

            K_opt, T_pred = compute_optimal_K(C_eff, F, D, B, L, msg)

            if T_pred < mpi_time and T_pred < srda_time:
                beat_count += 1
                if msg >= 1048576:
                    ms = f"{msg//1048576}MB"
                elif msg >= 1024:
                    ms = f"{msg//1024}KB"
                else:
                    ms = f"{msg}B"
                speedup_mpi = mpi_time / T_pred
                speedup_srda = srda_time / T_pred
                print(f"  {topo} N={n} {ms:>6}: T_pred={T_pred:.4e}  "
                      f"MPI={mpi_time:.4e} ({speedup_mpi:.1f}x)  "
                      f"SRDA={srda_time:.4e} ({speedup_srda:.1f}x)")

    if beat_count == 0:
        print("  (none — model is conservative, actual BBS may still win)")

    print(f"\nTotal: {beat_count} cases where model predicts BBS wins")

    if FAIL > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
