#!/usr/bin/env python3
"""
LP.py — Unified Broadcast Convex Program + optimal BBS plan generator.

Generates plan files consumed by runner.c (BBS algorithm).

============================================================
THE BROADCAST CONVEX PROGRAM
============================================================

Given network G=(V,E), N nodes, root r, BFS diameter D, link
bandwidth B (bytes/sec), latency L (sec), message size M (bytes):

Variables: w_{ij} >= 0 for (i,j) in E; throughput C >= 0

Constraints:
  (S) sum_j w_{ij} <= 1           for all i          (send capacity)
  (R) sum_k w_{ki} <= 1           for all i          (recv capacity)
  (B) sum_j w_{ji} == C           for all i != root  (balanced incoming)
  (C) w_{ij} <= sum_k w_{ki}     for all i != root  (causality)
  (Z) w_{kr} == 0                for all k           (root boundary)

Objective: Minimize T*(C) = M/(CB) + 2*sqrt(DML/(CB)) + DL
Since T* is monotone decreasing in C: Minimize T* <=> Maximize C (LP).

============================================================
ANALYTICAL FORMULAS
============================================================

Optimal chunks:
  K*(M) = ceil(sqrt(D * C* * M / (B * L)))

Optimal time (three-term decomposition):
  T* = M/(C*B) + 2*sqrt(D*M*L/(C*B)) + D*L
       -------   --------------------   ---
       Term 1        Term 2            Term 3
       steady     ramp transient     pipeline
       state     (geometric mean)    latency

Round count: R = K + D (with maximum matching achieving C*=1/round)

============================================================
EXECUTION MODEL
============================================================

Maximum bipartite matching (Kuhn's algorithm) per round:
  - Ramp-up (rounds 1..D): data reaches all N nodes via BFS wavefront
  - Steady state (rounds D+1..K+D): each non-root receives 1 new
    chunk/round (maximum matching guarantees C*=1)
  - Total: K + D rounds (matches LP bound)

============================================================
PLAN FILE FORMATS
============================================================

Weighted (adaptive) mode — used by default (--weighted):
    # Weighted BBS plan for <topo> N=<n> root=<root>
    WEIGHTED
    <src> <dst> <weight>
    ...

  runner.c builds a CSR adjacency list from edges and uses Kuhn's
  maximum bipartite matching per round.

Legacy frame mode — generate_plan() (no --weighted):
    # BBS frames for <topology> N=<nodes> root=<root>
    FRAME 0
    <src> <dst>
    ...

  Uses rationalization + frame decomposition + round-robin cycling.

Usage:
    python3 LP.py <topology> <nodes> [root] --weighted
    python3 LP.py --all --weighted
    python3 LP.py --all              (legacy frame mode)
"""

import sys
import os
import argparse
import math
import time
from collections import defaultdict

import numpy as np
import cvxpy as cp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAN_DIR = os.path.join(SCRIPT_DIR, "..", "plans")

# ---- Topology configs (mirrored from topology_generater.py) ----
MESH_SIZES = {128: (8, 16), 256: (16, 16), 512: (16, 32), 1024: (32, 32)}
DRAGONFLY_CONFIGS = {
    128: (4, 4, 4, 2), 256: (4, 4, 4, 4),
    512: (8, 4, 4, 4), 1024: (16, 4, 4, 4),
}
ALL_TOPOS = ["2Dmesh", "Butterfly", "Dragonfly", "FatTree"]
ALL_SIZES = [128, 256, 512, 1024]

# ---- Topology physics (from platform XMLs) ----
TOPO_PHYSICS = {
    "2Dmesh":    {"B": 50e9,   "L": 100e-9},
    "Butterfly": {"B": 12.5e9, "L": 100e-9},
    "FatTree":   {"B": 12.5e9, "L": 100e-9},
    "Dragonfly": {"B": 5.25e9, "L": 400e-9},
}
MSG_SIZES = [1024, 4096, 16384, 65536, 262144,
             1048576, 4194304, 16777216, 67108864, 268435456]


# ============================================================
# A. Adjacency builders — one per topology
# ============================================================

def adj_2dmesh(n):
    """2D mesh: 4-connected grid (up/down/left/right)."""
    if n not in MESH_SIZES:
        raise ValueError(f"2Dmesh supports {sorted(MESH_SIZES.keys())}")
    p, q = MESH_SIZES[n]
    adj = defaultdict(set)
    for r in range(p):
        for c in range(q):
            node = r * q + c
            if c + 1 < q:
                adj[node].add(node + 1)
                adj[node + 1].add(node)
            if r + 1 < p:
                adj[node].add(node + q)
                adj[node + q].add(node)
    for i in range(n):
        _ = adj[i]  # ensure all nodes present
    return dict(adj)


def adj_butterfly(n):
    """Butterfly (hypercube): XOR neighbors i ^ (1<<k)."""
    dim = int(math.log2(n))
    if (1 << dim) != n or n < 4:
        raise ValueError(f"Butterfly needs power-of-2 >= 4, got {n}")
    adj = defaultdict(set)
    for i in range(n):
        for d in range(dim):
            adj[i].add(i ^ (1 << d))
    return dict(adj)


def adj_fattree(n):
    """Fat-tree: intra-leaf clique + inter-leaf bridge (node 0 of each leaf)."""
    hpl = 16
    while hpl > n // 2 and hpl > 2:
        hpl //= 2
    if n % hpl != 0:
        raise ValueError(f"{n} not divisible by hpl={hpl}")
    num_leaf = n // hpl
    adj = defaultdict(set)

    # Intra-leaf: complete graph within each leaf group
    for lf in range(num_leaf):
        base = lf * hpl
        for i in range(hpl):
            for j in range(i + 1, hpl):
                adj[base + i].add(base + j)
                adj[base + j].add(base + i)

    # Inter-leaf bridge: node 0 of each leaf <-> node 0 of every other leaf
    for l1 in range(num_leaf):
        for l2 in range(l1 + 1, num_leaf):
            a, b = l1 * hpl, l2 * hpl
            adj[a].add(b)
            adj[b].add(a)

    for i in range(n):
        _ = adj[i]
    return dict(adj)


def adj_dragonfly(n):
    """Dragonfly: intra-router + green/black/blue bridge edges."""
    if n not in DRAGONFLY_CONFIGS:
        raise ValueError(f"Dragonfly supports {sorted(DRAGONFLY_CONFIGS.keys())}")
    G, C, R, P = DRAGONFLY_CONFIGS[n]

    def nid(g, c, r, p):
        return g * (C * R * P) + c * (R * P) + r * P + p

    adj = defaultdict(set)

    # Intra-router: all nodes on same router are connected
    for g in range(G):
        for c in range(C):
            for r in range(R):
                nodes = [nid(g, c, r, p) for p in range(P)]
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        adj[nodes[i]].add(nodes[j])
                        adj[nodes[j]].add(nodes[i])

    # Green (intra-chassis): bridge node p=0 of each router pair in same chassis
    for g in range(G):
        for c in range(C):
            for r1 in range(R):
                for r2 in range(r1 + 1, R):
                    a, b = nid(g, c, r1, 0), nid(g, c, r2, 0)
                    adj[a].add(b)
                    adj[b].add(a)

    # Black (inter-chassis): bridge node r=0,p=0 of each chassis pair in group
    for g in range(G):
        for c1 in range(C):
            for c2 in range(c1 + 1, C):
                a, b = nid(g, c1, 0, 0), nid(g, c2, 0, 0)
                adj[a].add(b)
                adj[b].add(a)

    # Blue (inter-group): bridge node c=0,r=0,p=0 of each group pair
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            a, b = nid(g1, 0, 0, 0), nid(g2, 0, 0, 0)
            adj[a].add(b)
            adj[b].add(a)

    for i in range(n):
        _ = adj[i]
    return dict(adj)


ADJ_BUILDERS = {
    "2Dmesh":    adj_2dmesh,
    "Butterfly": adj_butterfly,
    "FatTree":   adj_fattree,
    "Dragonfly": adj_dragonfly,
}


# ============================================================
# B. LP solver using CVXPY
# ============================================================

def solve_lp(adj, n, root):
    """
    Solve the full-duplex BBS LP.

    Returns ({(i,j): rate}, C_value).
    """
    # Build directed edge list with precomputed index maps
    edges = []
    out_idx = defaultdict(list)
    in_idx = defaultdict(list)

    for i in range(n):
        for j in sorted(adj.get(i, set())):
            k = len(edges)
            edges.append((i, j))
            out_idx[i].append(k)
            in_idx[j].append(k)

    E = len(edges)
    O = cp.Variable(E, nonneg=True)
    C = cp.Variable()
    cons = [C >= 0]

    # 1. Send capacity: sum_j O[i,j] <= 1
    for i in range(n):
        if out_idx[i]:
            cons.append(cp.sum(O[out_idx[i]]) <= 1)

    # 2. Recv capacity: sum_k O[k,i] <= 1
    for i in range(n):
        if in_idx[i]:
            cons.append(cp.sum(O[in_idx[i]]) <= 1)

    # 3. Balanced incoming: sum_j O[j,i] == C for all non-root
    for i in range(n):
        if i == root:
            continue
        if in_idx[i]:
            cons.append(cp.sum(O[in_idx[i]]) == C)
        else:
            cons.append(C == 0)

    # 4. Causality: O[i,j] <= sum_k O[k,i] for all non-root i
    for i in range(n):
        if i == root or not in_idx[i]:
            continue
        in_sum = cp.sum(O[in_idx[i]])
        for eidx in out_idx[i]:
            cons.append(O[eidx] <= in_sum)

    # 5. Root does not receive
    for eidx in in_idx[root]:
        cons.append(O[eidx] == 0)

    prob = cp.Problem(cp.Maximize(C), cons)

    solved = False
    for solver in [cp.SCS, cp.CLARABEL, cp.ECOS]:
        try:
            kwargs = {"verbose": False}
            if solver == cp.SCS:
                kwargs["max_iters"] = 50000
            prob.solve(solver=solver, **kwargs)
            if prob.status in ("optimal", "optimal_inaccurate"):
                solved = True
                break
        except (cp.SolverError, Exception):
            continue

    if not solved:
        raise RuntimeError(f"LP failed: {prob.status}")

    edge_weights = {}
    for k, (i, j) in enumerate(edges):
        v = float(O.value[k])
        if v > 1e-6:
            edge_weights[(i, j)] = v

    return edge_weights, float(C.value)


# ============================================================
# C. Rationalization — continuous -> integer weights
# ============================================================

def rationalize(edge_weights, F_target=4):
    """Convert continuous LP weights to integer edge multiplicities.

    Scales LP weights so the maximum becomes F_target, then rounds to
    integers (minimum 1).  This preserves relative edge importance:
    an edge with 2x the LP rate gets ~2x the frame-slots.

    Edges below 1% of the maximum weight are dropped as LP noise.
    """
    weights = {e: v for e, v in edge_weights.items() if v > 1e-6}
    if not weights:
        return {}

    max_val = max(weights.values())
    weights = {e: v for e, v in weights.items() if v > max_val * 0.01}
    if not weights:
        return {}

    return {e: max(1, round(v * F_target / max_val)) for e, v in weights.items()}


def compute_C_eff(int_weights, n, root):
    """Effective throughput: min total incoming weight over non-root nodes.

    In the rationalized schedule, each non-root node i receives data
    in sum_j(w[j,i]) frame-slots per cycle of F frames.  The bottleneck
    node determines the effective throughput C_eff.
    """
    incoming = [0] * n
    for (src, dst), w in int_weights.items():
        incoming[dst] += w
    c_eff = min(incoming[i] for i in range(n) if i != root)
    return c_eff


# ============================================================
# D. Frame decomposition — greedy full-duplex matching
# ============================================================

def decompose_frames(int_weights):
    """
    Decompose integer edge weights into full-duplex frames.

    Each frame: out-degree <= 1 and in-degree <= 1 per node.
    Greedy: pick highest-weight edge first.
    """
    remaining = dict(int_weights)
    frames = []

    while any(v > 0 for v in remaining.values()):
        frame = []
        senders = set()
        receivers = set()

        for (src, dst) in sorted(remaining, key=remaining.get, reverse=True):
            if remaining[(src, dst)] <= 0:
                continue
            if src in senders or dst in receivers:
                continue
            frame.append((src, dst))
            senders.add(src)
            receivers.add(dst)

        if not frame:
            break

        for e in frame:
            remaining[e] -= 1
            if remaining[e] <= 0:
                del remaining[e]

        frames.append(frame)

    return frames


# ============================================================
# E. Frame ordering — wavefront urgency heuristic
# ============================================================

def bfs_distances(adj, n, root):
    """BFS distances from root (-1 = unreachable)."""
    dist = [-1] * n
    dist[root] = 0
    queue = [root]
    head = 0
    while head < len(queue):
        u = queue[head]; head += 1
        for v in adj.get(u, []):
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def bfs_diameter(adj, n, root):
    """BFS diameter: max distance from root to any reachable node."""
    dist = bfs_distances(adj, n, root)
    return max(d for d in dist if d >= 0)


def order_frames(frames, adj, n, root):
    """
    Order frames for optimal wavefront propagation.

    Greedy simulation: at each step pick the unplaced frame that
    activates the most new nodes and pushes data furthest from root.
    """
    if len(frames) <= 1:
        return list(frames)

    dist = bfs_distances(adj, n, root)
    max_dist = max(d for d in dist if d >= 0)
    if max_dist == 0:
        return list(frames)

    has_data = {root}
    ordered = []
    available = list(range(len(frames)))

    for _ in range(len(frames)):
        best_fi = -1
        best_score = -float("inf")

        for fi in available:
            score = 0.0
            for src, dst in frames[fi]:
                if src not in has_data:
                    continue
                d_dst = dist[dst] if dist[dst] >= 0 else 0
                d_src = dist[src] if dist[src] >= 0 else 0
                if dst not in has_data:
                    score += 100
                if d_dst > d_src:
                    score += 10
                score += d_dst

            if score > best_score:
                best_score = score
                best_fi = fi

        if best_fi < 0:
            best_fi = available[0]

        ordered.append(frames[best_fi])
        available.remove(best_fi)

        for src, dst in ordered[-1]:
            if src in has_data:
                has_data.add(dst)

    return ordered


# ============================================================
# F. Plan writer
# ============================================================

def write_plan(frames, topo, n, root, plan_dir=None):
    """Write .plan file in format expected by runner.c."""
    if plan_dir is None:
        plan_dir = PLAN_DIR
    outdir = os.path.join(plan_dir, topo)
    os.makedirs(outdir, exist_ok=True)

    path = os.path.join(outdir, f"{n}_root{root}.plan")
    with open(path, "w") as f:
        f.write(f"# BBS frames for {topo} N={n} root={root}\n")
        f.write(f"# nframes: {len(frames)}\n\n")
        for i, frame in enumerate(frames):
            f.write(f"FRAME {i}\n")
            for src, dst in frame:
                f.write(f"{src} {dst}\n")
            f.write("\n")
    return path


# ============================================================
# F2. Optimal K computation + params writer
# ============================================================

def T_model(K, C_eff, F, D, B, L, M):
    """BBS wall-clock time for K chunks.

    T(K) = F * (K/C_eff + D) * (M/(K*B) + L)

    where:
      K      = number of chunks the message is split into
      C_eff  = effective throughput (min non-root incoming degree)
      F      = number of frames (directed matchings) per cycle
      D      = BFS diameter from root
      B      = link bandwidth (bytes/sec)
      L      = link latency (sec)
      M      = message size (bytes)
    """
    return F * (K / C_eff + D) * (M / (K * B) + L)


def compute_optimal_K(C_eff, F, D, B, L, M):
    """Optimal chunk count K for the BBS wall-clock model.

    The wall-clock time is:
        T(K) = F * (K/C_eff + D) * (M/(K*B) + L)

    Setting dT/dK = 0 yields the closed-form optimum:
        K* = sqrt(D * C_eff * M / (B * L))

    At K*, the optimal time decomposes into three terms:
        T* = F * [M/(C_eff*B) + 2*sqrt(D*M*L/(C_eff*B)) + D*L]
              ─────────────   ────────────────────────────   ───
              steady state         ramp-up + ramp-down       pipeline
                                  (geometric mean)           latency

    Parameters:
        C_eff  effective throughput (min non-root incoming degree)
        F      number of frames per cycle
        D      BFS diameter from root
        B      link bandwidth (bytes/sec)
        L      link latency (sec)
        M      message size (bytes)

    Returns (K_opt, T_opt).
    """
    if C_eff <= 0 or B <= 0 or L <= 0 or M <= 0:
        return (4, float("inf"))

    # Closed-form optimal K
    K_anal = math.sqrt(D * C_eff * M / (B * L))

    # Best integer: check floor and ceil
    K_lo = max(1, math.floor(K_anal))
    K_hi = K_lo + 1
    T_lo = T_model(K_lo, C_eff, F, D, B, L, M)
    T_hi = T_model(K_hi, C_eff, F, D, B, L, M)

    if T_lo <= T_hi:
        return (K_lo, T_lo)
    else:
        return (K_hi, T_hi)


def write_params(topo, n, root, C_eff, F, D, plan_dir=None):
    """Write .params JSON with optimal K for each message size.

    File: plans/<Topo>/<N>_root<R>.params

    The JSON contains C_eff, F, D, link physics (B, L), and
    the optimal K for each message size computed via the
    closed-form formula K* = sqrt(D * C_eff * M / (B * L)).
    """
    import json

    if plan_dir is None:
        plan_dir = PLAN_DIR
    outdir = os.path.join(plan_dir, topo)
    os.makedirs(outdir, exist_ok=True)

    physics = TOPO_PHYSICS.get(topo)
    if physics is None:
        return None

    B = physics["B"]
    L = physics["L"]

    optimal_K = {}
    for M in MSG_SIZES:
        K_opt, T_opt = compute_optimal_K(C_eff, F, D, B, L, M)
        optimal_K[str(M)] = {"K_opt": K_opt, "T_predicted_sec": T_opt}

    params = {
        "topology": topo,
        "nodes": n,
        "root": root,
        "C_star": C_eff,   # effective throughput (min non-root incoming degree)
        "F": F,
        "D": D,
        "B": B,
        "L": L,
        "optimal_K": optimal_K,
    }

    path = os.path.join(outdir, f"{n}_root{root}.params")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path


# ============================================================
# F3. Weighted (adaptive) plan writer — no rationalization
# ============================================================

def write_weighted_plan(edge_weights, topo, n, root, plan_dir=None):
    """Write weighted .plan file for adaptive BBS scheduler.

    Filters noise edges (<1% of max), sorts by weight descending,
    writes new plan format:
        # Weighted BBS plan for <topo> N=<n> root=<root>
        # nedges: <count>
        WEIGHTED
        <src> <dst> <weight>
        ...
    """
    if plan_dir is None:
        plan_dir = PLAN_DIR
    outdir = os.path.join(plan_dir, topo)
    os.makedirs(outdir, exist_ok=True)

    # Filter noise edges
    weights = {e: v for e, v in edge_weights.items() if v > 1e-6}
    if not weights:
        return None
    max_val = max(weights.values())
    weights = {e: v for e, v in weights.items() if v > max_val * 0.01}
    if not weights:
        return None

    # Sort by weight descending (highest priority first)
    sorted_edges = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    path = os.path.join(outdir, f"{n}_root{root}.plan")
    with open(path, "w") as f:
        f.write(f"# Weighted BBS plan for {topo} N={n} root={root}\n")
        f.write(f"# nedges: {len(sorted_edges)}\n\n")
        f.write("WEIGHTED\n")
        for (src, dst), w in sorted_edges:
            f.write(f"{src} {dst} {w:.6f}\n")
    return path


def write_weighted_params(topo, n, root, D, plan_dir=None):
    """Write .params JSON for weighted (adaptive) BBS.

    Uses C_eff=1, F=1 (no frame factor).
    K* = sqrt(D * M / (B * L)).
    """
    import json

    if plan_dir is None:
        plan_dir = PLAN_DIR
    outdir = os.path.join(plan_dir, topo)
    os.makedirs(outdir, exist_ok=True)

    physics = TOPO_PHYSICS.get(topo)
    if physics is None:
        return None

    B = physics["B"]
    L = physics["L"]

    optimal_K = {}
    for M in MSG_SIZES:
        K_opt, T_opt = compute_optimal_K(1, 1, D, B, L, M)
        optimal_K[str(M)] = {"K_opt": K_opt, "T_predicted_sec": T_opt}

    params = {
        "topology": topo,
        "nodes": n,
        "root": root,
        "C_star": 1,
        "F": 1,
        "D": D,
        "B": B,
        "L": L,
        "optimal_K": optimal_K,
    }

    path = os.path.join(outdir, f"{n}_root{root}.params")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path


def generate_weighted_plan(topo, n, root=0, verbose=True):
    """Full pipeline: adjacency -> LP -> weighted plan (no rationalization).

    Skips rationalize(), decompose_frames(), order_frames() entirely.
    Uses LP edge weights directly as priorities for adaptive per-round
    greedy matching in runner.c.
    """
    t0 = time.time()
    if verbose:
        print(f"  {topo} N={n} root={root}: ", end="", flush=True)

    adj = ADJ_BUILDERS[topo](n)
    nedges = sum(len(v) for v in adj.values())
    if verbose:
        print(f"adj {nedges} dir-edges, ", end="", flush=True)

    edge_weights, C_val = solve_lp(adj, n, root)
    if verbose:
        print(f"C*={C_val:.4f}, ", end="", flush=True)

    D = bfs_diameter(adj, n, root)
    physics = TOPO_PHYSICS.get(topo)

    # Write weighted plan (no rationalization/decomposition)
    path = write_weighted_plan(edge_weights, topo, n, root)
    params_path = write_weighted_params(topo, n, root, D)

    dt = time.time() - t0
    if verbose:
        parts = [f"-> {path} ({dt:.1f}s)"]
        if params_path and physics:
            B, L = physics["B"], physics["L"]
            k1m, _ = compute_optimal_K(1, 1, D, B, L, 1048576)
            k64m, _ = compute_optimal_K(1, 1, D, B, L, 67108864)
            parts.append(f"D={D}, K_opt(1MB)={k1m}, K_opt(64MB)={k64m}")
        print(", ".join(parts))

    return path, C_val, 1  # F=1 for weighted mode


# ============================================================
# Main pipeline
# ============================================================

def generate_plan(topo, n, root=0, verbose=True):
    """Full pipeline: adjacency -> LP -> rationalize -> decompose -> order -> write.

    1. Build adjacency graph for the topology
    2. Solve LP to maximise balanced throughput C*
    3. Rationalize LP rates to uniform integer weights
    4. Decompose into F directed matchings (frames)
    5. Order frames for wavefront propagation
    6. Compute C_eff, D, and write plan + physics params

    The physics-aware model T(K) = F*(K/C_eff + D)*(M/(KB) + L)
    determines the optimal chunk count K* = sqrt(D*C_eff*M/(BL))
    for each message size, written to the .params file.
    """
    t0 = time.time()
    if verbose:
        print(f"  {topo} N={n} root={root}: ", end="", flush=True)

    adj = ADJ_BUILDERS[topo](n)
    nedges = sum(len(v) for v in adj.values())
    if verbose:
        print(f"adj {nedges} dir-edges, ", end="", flush=True)

    edge_weights, C_val = solve_lp(adj, n, root)
    if verbose:
        print(f"C*={C_val:.4f}, ", end="", flush=True)

    D = bfs_diameter(adj, n, root)
    physics = TOPO_PHYSICS.get(topo)

    # Rationalize to uniform weights and decompose into frames
    int_weights = rationalize(edge_weights, F_target=1)
    C_eff = compute_C_eff(int_weights, n, root)

    frames = decompose_frames(int_weights)
    frames = order_frames(frames, adj, n, root)
    F = len(frames)

    if verbose:
        print(f"F={F} C_eff={C_eff}, ", end="", flush=True)

    path = write_plan(frames, topo, n, root)
    params_path = write_params(topo, n, root, C_eff, F, D)

    dt = time.time() - t0
    if verbose:
        parts = [f"-> {path} ({dt:.1f}s)"]
        if params_path and physics:
            B, L = physics["B"], physics["L"]
            k1m, _ = compute_optimal_K(C_eff, F, D, B, L, 1048576)
            k64m, _ = compute_optimal_K(C_eff, F, D, B, L, 67108864)
            parts.append(f"D={D}, K_opt(1MB)={k1m}, K_opt(64MB)={k64m}")
        print(", ".join(parts))

    return path, C_val, F


# ============================================================
# G. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate optimal BBS broadcast plans via LP solver")
    parser.add_argument("topology", nargs="?",
                        help=f"Topology: {', '.join(ALL_TOPOS)}")
    parser.add_argument("nodes", nargs="?", type=int,
                        help="Number of nodes")
    parser.add_argument("root", nargs="?", type=int, default=0,
                        help="Root node (default: 0)")
    parser.add_argument("--all", action="store_true",
                        help="All topologies x all sizes x root=0")
    parser.add_argument("--weighted", action="store_true",
                        help="Use adaptive weighted plan (no rationalization)")
    args = parser.parse_args()

    gen_fn = generate_weighted_plan if args.weighted else generate_plan

    if args.all:
        mode = "weighted" if args.weighted else "frame-based"
        print(f"Generating all BBS plans ({mode})...")
        for topo in ALL_TOPOS:
            for n in ALL_SIZES:
                try:
                    gen_fn(topo, n, root=0)
                except Exception as e:
                    print(f"\n  FAILED {topo} N={n}: {e}")
        print("Done.")
    elif args.topology and args.nodes:
        if args.topology not in ADJ_BUILDERS:
            print(f"Unknown topology '{args.topology}'. "
                  f"Choose from: {ALL_TOPOS}", file=sys.stderr)
            sys.exit(1)
        gen_fn(args.topology, args.nodes, args.root)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
