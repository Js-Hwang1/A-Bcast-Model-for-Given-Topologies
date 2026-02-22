#!/usr/bin/env python3
"""Generate BBS plans for all roots of N=128 topologies, in parallel."""
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from LP import generate_weighted_plan, ALL_TOPOS

N = 128

def run_one(args):
    topo, n, root = args
    try:
        generate_weighted_plan(topo, n, root, verbose=False)
        return (topo, root, "OK")
    except Exception as e:
        return (topo, root, str(e))

if __name__ == "__main__":
    jobs = [(t, N, r) for t in ALL_TOPOS for r in range(N)]
    ncpu = os.cpu_count() or 8
    print(f"Generating {len(jobs)} plans across {ncpu} cores...")
    t0 = time.time()

    done = 0
    fails = []
    with ProcessPoolExecutor(max_workers=ncpu) as pool:
        futs = {pool.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            topo, root, status = fut.result()
            done += 1
            if status != "OK":
                fails.append((topo, root, status))
            if done % 64 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] ...", flush=True)

    dt = time.time() - t0
    print(f"Done: {done - len(fails)}/{done} OK in {dt:.0f}s")
    if fails:
        for t, r, e in fails[:10]:
            print(f"  FAIL {t} root={r}: {e}")
