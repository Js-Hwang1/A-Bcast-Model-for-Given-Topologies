#!/usr/bin/env python3
"""
Analyze per-edge concurrent flow count when tau BBS trees are overlaid
in the pipeline.

BBS pipeline model:
  - Root batch-Isends tau chunks (one per tree) → tau concurrent flows per edge
  - Non-root nodes have windowed Irecv of tau → tau concurrent incoming flows
  - At steady state, tau chunks (one per tree) are "in flight" at each depth level

For each edge (link), count: how many of the tau trees use that edge simultaneously?
An edge is "used" by tree t if it connects a parent→child pair in tree t.
"""

import sys

def analyze_fattree(Nc=128, npl=16, tau=3):
    Nl = Nc // npl
    Ns = npl  # = 16 for 128-node
    N = Nc + Nl + Ns

    LEAF = lambda l: Nc + l
    SPINE = lambda s: Nc + Nl + s

    print(f"FatTree: Nc={Nc} Nl={Nl} Ns={Ns} N={N} npl={npl} tau={tau}")
    print(f"Compute ranks: 0..{Nc-1}")
    print(f"Leaf ranks: {Nc}..{Nc+Nl-1}")
    print(f"Spine ranks: {Nc+Nl}..{N-1}")
    print()

    # Build tau trees (same as encode.c)
    root = 0
    rl = root // npl  # root's leaf index

    trees = []
    for t in range(tau):
        parent = [-1] * N
        parent[LEAF(rl)] = root

        # All spines are children of leaf_rl
        for s in range(Ns):
            parent[SPINE(s)] = LEAF(rl)

        # Remote leaves: child of anti-correlated spine
        for l in range(Nl):
            if l == rl:
                continue
            spine_idx = (l * tau + t) % Ns
            parent[LEAF(l)] = SPINE(spine_idx)

        # All computes → their leaf
        for c in range(Nc):
            if c == root:
                continue
            parent[c] = LEAF(c // npl)

        trees.append(parent)

    # Count per-edge usage across all tau trees
    # An "edge" is a link between two adjacent nodes (undirected)
    edge_usage = {}  # (min(a,b), max(a,b)) → count of trees using it

    for t in range(tau):
        for node in range(N):
            p = trees[t][node]
            if p < 0:
                continue
            edge = (min(node, p), max(node, p))
            edge_usage[edge] = edge_usage.get(edge, 0) + 1

    # Categorize edges
    root_uplink = []
    local_compute = []
    leaf_rl_spine = []
    spine_remote_leaf = []
    remote_compute = []

    for (a, b), count in sorted(edge_usage.items()):
        # Determine edge type
        if a == root and b == LEAF(rl):
            root_uplink.append((a, b, count))
        elif a < Nc and b == LEAF(rl) and a != root:
            local_compute.append((a, b, count))
        elif a == LEAF(rl) and Nc + Nl <= b < N:
            leaf_rl_spine.append((a, b, count))
        elif Nc <= a < Nc + Nl and a != LEAF(rl) and Nc + Nl <= b < N:
            spine_remote_leaf.append((a, b, count))
        elif a < Nc and Nc <= b < Nc + Nl and b != LEAF(rl):
            remote_compute.append((a, b, count))
        else:
            print(f"  UNCLASSIFIED: ({a},{b}) usage={count}")

    print("=" * 65)
    print(f"{'Edge type':<30} {'Count':<8} {'Usage per edge':<15} {'Max'}")
    print("=" * 65)

    def show(name, edges):
        if not edges:
            return
        usages = [c for _, _, c in edges]
        mx = max(usages)
        mn = min(usages)
        if mn == mx:
            print(f"{name:<30} {len(edges):<8} {mn:<15} {mx}")
        else:
            print(f"{name:<30} {len(edges):<8} {mn}-{mx:<13} {mx}")

    show("Root uplink", root_uplink)
    show("Local compute ↔ leaf_rl", local_compute)
    show("leaf_rl ↔ spine", leaf_rl_spine)
    show("spine ↔ remote leaf", spine_remote_leaf)
    show("Remote compute ↔ leaf", remote_compute)

    print("=" * 65)

    # Overall stats
    all_usages = list(edge_usage.values())
    print(f"\nTotal edges: {len(all_usages)}")
    print(f"Max usage (concurrent flows): {max(all_usages)}")
    print(f"Min usage: {min(all_usages)}")

    # Distribution
    from collections import Counter
    dist = Counter(all_usages)
    print(f"\nUsage distribution:")
    for usage, cnt in sorted(dist.items()):
        print(f"  usage={usage}: {cnt} edges")

    # Bottleneck analysis
    print(f"\n--- Pipeline Bottleneck Analysis ---")
    B = 12.5e9  # bytes/sec
    L = 100e-9  # seconds
    chunk = 8192  # bytes

    # The edge with max concurrent flows determines the pipeline stage time
    max_usage = max(all_usages)
    stage_time = max_usage * chunk / B + L
    K = 1048576 // chunk  # 1MB message
    n_stages = K / tau + (4 - 1)  # depth=4 for FatTree
    T = n_stages * stage_time

    print(f"Max concurrent flows on any edge: {max_usage}")
    print(f"Stage time = {max_usage}×{chunk}/B + L = {stage_time*1e6:.3f} μs")
    print(f"Pipeline steps (K/τ + d-1) = {n_stages:.1f}")
    print(f"Total time = {T*1e6:.1f} μs")

    # What if we could reduce leaf_rl↔spine to 1?
    print(f"\n--- What if leaf_rl↔spine edges had usage=1? ---")
    # Bottleneck would still be compute edges at tau
    print(f"Compute edges still have usage={tau} (degree-1, unavoidable)")
    print(f"Pipeline stage time UNCHANGED (compute edges are equally bottlenecked)")

    # Show spine assignment detail
    print(f"\n--- Spine assignments per tree ---")
    for t in range(tau):
        spines_used = []
        for l in range(Nl):
            if l == rl:
                continue
            s = (l * tau + t) % Ns
            spines_used.append(s)
        print(f"  Tree {t}: remote leaves use spines {spines_used}")

    # Per-spine load across trees (how many trees route through each spine)
    print(f"\n--- Per-spine load (as relay to remote leaves) ---")
    spine_relay_load = [0] * Ns
    for t in range(tau):
        for l in range(Nl):
            if l == rl:
                continue
            s = (l * tau + t) % Ns
            spine_relay_load[s] += 1
    for s in range(Ns):
        print(f"  spine-{s}: relays {spine_relay_load[s]} remote-leaf connections across {tau} trees")


if __name__ == "__main__":
    tau = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    analyze_fattree(tau=tau)
