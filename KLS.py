# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 17:48:57 2026

@author: Peter Lu
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.optimize import linprog
from scipy import sparse
from networkx.algorithms.shortest_paths.dense import (
    floyd_warshall_predecessor_and_distance,
    reconstruct_path,
)


# ============================================================
# 1) Build fixed Floyd paths on router layer
# ============================================================
def build_fixed_router_paths(
    router_ids,
    router_adj,
    node_ids,
    node_router_connections,
):
    """
    INPUTS
      router_ids: list of router IDs, length R
      router_adj: R x R numpy array / nested list of positive weights; <=0 means absent
      node_ids: list of node IDs, length N
      node_router_connections:
          either dict {node_id: router_id}
          or list aligned with node_ids giving router_id for each node

    OUTPUTS
      pair_router_edges[(i,j)] = list of router-edge keys (ra, rb) used by node-pair i<j
      router_edge_list = sorted list of router-edge keys
      node_to_router_idx = list length N
    """
    router_ids = list(router_ids)
    node_ids = list(node_ids)
    A = np.array(router_adj, dtype=float)
    R = len(router_ids)
    N = len(node_ids)

    if A.shape != (R, R):
        raise ValueError(f"router_adj must be shape ({R},{R}), got {A.shape}")

    router_index = {rid: i for i, rid in enumerate(router_ids)}

    if isinstance(node_router_connections, dict):
        node_to_router_idx = [router_index[node_router_connections[nid]] for nid in node_ids]
    else:
        if len(node_router_connections) != N:
            raise ValueError("node_router_connections list must align with node_ids")
        node_to_router_idx = [router_index[rid] for rid in node_router_connections]

    # Build explicit router graph
    RG = nx.Graph()
    RG.add_nodes_from(router_ids)
    for i in range(R):
        for j in range(i + 1, R):
            if A[i, j] > 0:
                RG.add_edge(router_ids[i], router_ids[j], weight=float(A[i, j]))

    if not nx.is_connected(RG):
        raise ValueError("Router graph is not connected.")

    # Floyd predecessor + distance
    pred, dist = floyd_warshall_predecessor_and_distance(RG, weight="weight")

    pair_router_edges = {}
    all_router_edges = set()

    for i in range(N):
        ri = node_to_router_idx[i]
        for j in range(i + 1, N):
            rj = node_to_router_idx[j]

            if ri == rj:
                path_router_ids = [router_ids[ri]]
            else:
                src = router_ids[ri]
                dst = router_ids[rj]
                path_router_ids = reconstruct_path(src, dst, pred)

            path_edges = []
            for a, b in zip(path_router_ids[:-1], path_router_ids[1:]):
                ia = router_index[a]
                ib = router_index[b]
                key = (ia, ib) if ia < ib else (ib, ia)
                path_edges.append(key)
                all_router_edges.add(key)

            pair_router_edges[(i, j)] = path_edges

    router_edge_list = sorted(all_router_edges)
    return pair_router_edges, router_edge_list, node_to_router_idx


# ============================================================
# 2) Single-tree KLS-style iterative relaxation
# ============================================================
def kls_single_tree_fixed_paths(
    num_nodes,
    pair_router_edges,
    router_edge_list,
    residual_budget,
    root=0,
    tol=1e-8,
    max_rounds=200,
):
    """
    Solve one crossing-spanning-tree subproblem with upper bounds only.

    This follows the KLS upper-bounds-only style:
      - solve LP relaxation
      - fix x_e = 0 or 1 when integral
      - drop a router-edge constraint when support_size <= budget + Delta - 1

    Returns:
      tree_edges   list of node-pair indices [(u,v),...], or None if failed
      tree_load    dict router_edge -> usage count by this tree
    """

    N = num_nodes
    undirected_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    m = len(undirected_edges)
    edge_index = {e: idx for idx, e in enumerate(undirected_edges)}

    # Router constraint incidence
    edges_by_router = {re: [] for re in router_edge_list}
    routers_by_edge = [[] for _ in range(m)]
    max_path_len = 0
    path_len = np.zeros(m, dtype=float)

    for e_idx, (u, v) in enumerate(undirected_edges):
        path = pair_router_edges[(u, v)]
        path_len[e_idx] = len(path)
        max_path_len = max(max_path_len, len(path))
        for re in path:
            edges_by_router[re].append(e_idx)
            routers_by_edge[e_idx].append(re)

    Delta = max_path_len if max_path_len > 0 else 1

    # Directed arcs for a rooted arborescence LP
    arc_list = []
    arc_of = {}
    for e_idx, (u, v) in enumerate(undirected_edges):
        a1 = len(arc_list)
        arc_list.append((u, v, e_idx))
        arc_of[(u, v)] = a1
        a2 = len(arc_list)
        arc_list.append((v, u, e_idx))
        arc_of[(v, u)] = a2

    Acount = len(arc_list)
    nxvar = m
    nzvar = Acount
    nfvar = Acount
    nvar = nxvar + nzvar + nfvar

    def x_idx(e): return e
    def z_idx(a): return nxvar + a
    def f_idx(a): return nxvar + nzvar + a

    incoming_arcs = [[] for _ in range(N)]
    outgoing_arcs = [[] for _ in range(N)]
    for a, (u, v, e_idx) in enumerate(arc_list):
        outgoing_arcs[u].append(a)
        incoming_arcs[v].append(a)

    # Fixed bounds on x
    x_lb = np.zeros(m, dtype=float)
    x_ub = np.ones(m, dtype=float)
    active_router_constraints = set(router_edge_list)

    # Dynamic cut constraints for spanning tree polytope separation:
    # x(delta(S)) >= 1  ->  -x(delta(S)) <= -1
    cut_constraints = []   # each item is a set/list of node indices defining S

    # A very mild objective: prefer shorter router paths and scarce routers
    def build_objective():
        c = np.zeros(nvar, dtype=float)
        for e_idx in range(m):
            scarcity = 0.0
            for re in routers_by_edge[e_idx]:
                b = residual_budget.get(re, 0)
                scarcity += 1.0 / max(1.0, b + 1.0)
            c[x_idx(e_idx)] = scarcity + 1e-3 * path_len[e_idx]
        return c

    def build_lp_matrices():
        # Equalities:
        # 1) z_uv + z_vu - x_e = 0      for each undirected edge
        # 2) indegree(v)=1 for v!=root, indegree(root)=0
        # 3) flow conservation from root
        eq_rows = m + N + N
        Aeq = sparse.lil_matrix((eq_rows, nvar))
        beq = np.zeros(eq_rows, dtype=float)

        row = 0
        for e_idx, (u, v) in enumerate(undirected_edges):
            a1 = arc_of[(u, v)]
            a2 = arc_of[(v, u)]
            Aeq[row, z_idx(a1)] = 1.0
            Aeq[row, z_idx(a2)] = 1.0
            Aeq[row, x_idx(e_idx)] = -1.0
            beq[row] = 0.0
            row += 1

        for v in range(N):
            for a in incoming_arcs[v]:
                Aeq[row, z_idx(a)] = 1.0
            beq[row] = 0.0 if v == root else 1.0
            row += 1

        for v in range(N):
            for a in outgoing_arcs[v]:
                Aeq[row, f_idx(a)] += 1.0
            for a in incoming_arcs[v]:
                Aeq[row, f_idx(a)] -= 1.0
            beq[row] = float(N - 1) if v == root else -1.0
            row += 1

        # Inequalities:
        # 1) f_a - (N-1) z_a <= 0
        # 2) router constraints sum x_e <= residual_budget[re] for active constraints
        # 3) cut constraints -sum_{e in delta(S)} x_e <= -1
        ineq_rows = Acount + len(active_router_constraints) + len(cut_constraints)
        Aub = sparse.lil_matrix((ineq_rows, nvar))
        bub = np.zeros(ineq_rows, dtype=float)

        row = 0
        for a in range(Acount):
            Aub[row, f_idx(a)] = 1.0
            Aub[row, z_idx(a)] = -(N - 1)
            bub[row] = 0.0
            row += 1

        for re in active_router_constraints:
            for e_idx in edges_by_router[re]:
                Aub[row, x_idx(e_idx)] = 1.0
            bub[row] = float(residual_budget[re])
            row += 1

        for S in cut_constraints:
            Sset = set(S)
            for e_idx, (u, v) in enumerate(undirected_edges):
                if (u in Sset) ^ (v in Sset):
                    Aub[row, x_idx(e_idx)] = -1.0
            bub[row] = -1.0
            row += 1

        bounds = []
        for e_idx in range(m):
            bounds.append((float(x_lb[e_idx]), float(x_ub[e_idx])))
        for _ in range(Acount):
            bounds.append((0.0, 1.0))
        for _ in range(Acount):
            bounds.append((0.0, float(N - 1)))

        return Aeq.tocsr(), beq, Aub.tocsr(), bub, bounds

    def separate_cut(x):
        """
        Check violated cut constraint via global min-cut on weighted graph with capacities x_e.
        Returns one violated S if mincut < 1 - tol, else None.
        """
        Gx = nx.Graph()
        Gx.add_nodes_from(range(N))
        for e_idx, (u, v) in enumerate(undirected_edges):
            w = x[e_idx]
            if w > 0:
                Gx.add_edge(u, v, weight=float(w))

        cut_value, partition = nx.stoer_wagner(Gx, weight="weight")
        if cut_value < 1.0 - 1e-7:
            A, B = partition
            if 0 < len(A) < N:
                return list(A)
        return None

    fixed_one = set()
    fixed_zero = set()

    for rnd in range(max_rounds):
        c = build_objective()
        Aeq, beq, Aub, bub, bounds = build_lp_matrices()

        res = linprog(
            c,
            A_ub=Aub,
            b_ub=bub,
            A_eq=Aeq,
            b_eq=beq,
            bounds=bounds,
            method="highs-ds",
        )

        if not res.success:
            return None, None

        x = res.x[:m]

        # First: cut separation loop
        violated_S = separate_cut(x)
        if violated_S is not None:
            cut_constraints.append(violated_S)
            continue

        # Then: fix integral variables
        changed = False
        for e_idx in range(m):
            if x_lb[e_idx] == x_ub[e_idx]:
                continue

            if x[e_idx] <= tol:
                x_ub[e_idx] = 0.0
                fixed_zero.add(e_idx)
                changed = True

            elif x[e_idx] >= 1.0 - tol:
                x_lb[e_idx] = 1.0
                fixed_one.add(e_idx)
                # contract/select effect on upper bounds: decrement residual budgets
                for re in routers_by_edge[e_idx]:
                    if re in active_router_constraints:
                        residual_budget[re] -= 1
                changed = True

        # If all x fixed, extract tree
        if np.allclose(x_lb, x_ub):
            chosen = [undirected_edges[e_idx] for e_idx in range(m) if x_lb[e_idx] > 0.5]
            if len(chosen) != N - 1:
                return None, None

            Gtree = nx.Graph()
            Gtree.add_nodes_from(range(N))
            Gtree.add_edges_from(chosen)
            if not nx.is_tree(Gtree):
                return None, None

            tree_load = {re: 0 for re in router_edge_list}
            for u, v in chosen:
                for re in pair_router_edges[(u, v)]:
                    tree_load[re] += 1
            return chosen, tree_load

        if changed:
            # If any residual budget already went negative, fail
            if any(residual_budget[re] < 0 for re in active_router_constraints):
                return None, None
            continue

        # KLS upper-bound dropping rule:
        # drop constraint re if current support size <= budget[re] + Delta - 1
        droppable = []
        for re in list(active_router_constraints):
            support_size = 0
            for e_idx in edges_by_router[re]:
                if x_lb[e_idx] < x_ub[e_idx]:
                    support_size += 1
            if support_size <= residual_budget[re] + Delta - 1:
                droppable.append(re)

        if droppable:
            for re in droppable:
                active_router_constraints.remove(re)
            continue

        # Fallback if HiGHS doesn't give a useful basic-looking point:
        # fix the variable closest to 0 or 1
        best_e = None
        best_gap = -1.0
        best_to_one = False
        for e_idx in range(m):
            if x_lb[e_idx] == x_ub[e_idx]:
                continue
            gap0 = abs(x[e_idx] - 0.0)
            gap1 = abs(x[e_idx] - 1.0)
            gap = max(1.0 - gap0, 1.0 - gap1)
            # equivalently, min distance to {0,1}
            min_dist = min(abs(x[e_idx]), abs(1.0 - x[e_idx]))
            score = -min_dist
            if best_e is None or score > best_gap:
                best_gap = score
                best_e = e_idx
                best_to_one = (x[e_idx] >= 0.5)

        if best_e is None:
            return None, None

        if best_to_one:
            x_lb[best_e] = 1.0
            for re in routers_by_edge[best_e]:
                if re in active_router_constraints:
                    residual_budget[re] -= 1
        else:
            x_ub[best_e] = 0.0

        if any(residual_budget[re] < 0 for re in active_router_constraints):
            return None, None

    return None, None


# ============================================================
# 3) Sequential multi-tree search for best k / L
# ============================================================
def select_trees_kls_style_with_floyd(
    router_ids,
    node_ids,
    router_adj,
    node_router_connections,
    max_trees=4,
    root_node_index=0,
):
    """
    Returns the best solution found over k = 1..max_trees.

    Output:
      {
        "num_trees": k,
        "ratio": k / L,
        "max_router_load": L,
        "tree_adjacency_matrices": [N x N numpy arrays],
        "tree_edge_lists": [[(node_id,node_id),...], ...],
        "router_edge_loads": {(router_id_a, router_id_b): load}
      }
    """
    router_ids = list(router_ids)
    node_ids = list(node_ids)
    N = len(node_ids)

    pair_router_edges, router_edge_list, _ = build_fixed_router_paths(
        router_ids=router_ids,
        router_adj=router_adj,
        node_ids=node_ids,
        node_router_connections=node_router_connections,
    )

    best = None

    # crude upper bound for L: each tree has N-1 edges, so one router edge can be used at most k*(N-1)
    for k in range(1, max_trees + 1):
        lo, hi = 1, k * (N - 1)
        best_for_k = None

        while lo <= hi:
            mid = (lo + hi) // 2
            import sys as _sys
            print(f"  k={k} binary search: L={mid} (lo={lo} hi={hi})", flush=True)

            residual_budget = {re: mid for re in router_edge_list}
            trees = []
            total_load = {re: 0 for re in router_edge_list}
            feasible = True

            for _ in range(k):
                tree_edges, tree_load = kls_single_tree_fixed_paths(
                    num_nodes=N,
                    pair_router_edges=pair_router_edges,
                    router_edge_list=router_edge_list,
                    residual_budget=dict(residual_budget),  # tree solver mutates, so pass copy
                    root=root_node_index,
                )

                if tree_edges is None:
                    feasible = False
                    break

                trees.append(tree_edges)

                # update true residuals after accepting this tree
                for re, add in tree_load.items():
                    total_load[re] += add
                    residual_budget[re] -= add
                    if residual_budget[re] < 0:
                        feasible = False
                if not feasible:
                    break

            if feasible:
                best_for_k = (mid, trees, total_load)
                hi = mid - 1
            else:
                lo = mid + 1

        if best_for_k is None:
            continue

        L, trees, total_load = best_for_k
        ratio = k / L

        if best is None or (ratio > best["ratio"]) or (
            abs(ratio - best["ratio"]) < 1e-12 and k > best["num_trees"]
        ):
            mats = []
            tree_edge_lists = []
            for tree in trees:
                A = np.zeros((N, N), dtype=np.int8)
                edges_named = []
                for u, v in tree:
                    A[u, v] = 1
                    A[v, u] = 1
                    edges_named.append((node_ids[u], node_ids[v]))
                mats.append(A)
                tree_edge_lists.append(edges_named)

            router_edge_loads = {}
            for (ia, ib), val in total_load.items():
                router_edge_loads[(router_ids[ia], router_ids[ib])] = val

            best = {
                "num_trees": k,
                "ratio": ratio,
                "max_router_load": L,
                "tree_adjacency_matrices": mats,
                "tree_edge_lists": tree_edge_lists,
                "router_edge_loads": router_edge_loads,
            }
            
            print(k,ratio)

    return best


# ============================================================
# 4) Small example
# ============================================================
def build_dragonfly_for_kls(cfg_path, mode="compute"):
    """Parse dragonfly cfg and build inputs for KLS.

    mode="compute": N = Nc compute nodes, routers form the constraint layer.
    mode="router":  N = Nr routers treated as nodes; each router is its own
                    "super-router" so router_adj is the identity mapping.
                    This avoids the C(128,2) blowup while still capturing
                    the Dragonfly link structure.
    """
    with open(cfg_path) as f:
        parts = f.read().strip().split()
    assert parts[0] == "dragonfly"
    G, C, R_per, Np = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

    Nc = G * C * R_per * Np
    Nr = G * C * R_per

    # Router IDs as (group, chassis, router) tuples
    router_ids = []
    rid_map = {}
    for g in range(G):
        for c in range(C):
            for r in range(R_per):
                rid = (g, c, r)
                rid_map[rid] = len(router_ids)
                router_ids.append(rid)

    router_adj = np.zeros((Nr, Nr), dtype=float)

    def add_rtr_edge(a, b, w=1.0):
        ia, ib = rid_map[a], rid_map[b]
        router_adj[ia, ib] = w
        router_adj[ib, ia] = w

    # Green links: all-to-all within each chassis
    for g in range(G):
        for c in range(C):
            for r1 in range(R_per):
                for r2 in range(r1 + 1, R_per):
                    add_rtr_edge((g, c, r1), (g, c, r2), 1.0)

    # Black links: router-0 of each chassis pair in same group
    for g in range(G):
        for c1 in range(C):
            for c2 in range(c1 + 1, C):
                add_rtr_edge((g, c1, 0), (g, c2, 0), 2.0)

    # Blue links: router-0-0 of each group pair
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            add_rtr_edge((g1, 0, 0), (g2, 0, 0), 4.0)

    if mode == "compute":
        node_ids = [f"n{i}" for i in range(Nc)]
        node_router_connections = {}
        for i in range(Nc):
            g = i // (C * R_per * Np)
            c = (i // (R_per * Np)) % C
            r = (i // Np) % R_per
            node_router_connections[f"n{i}"] = (g, c, r)
    else:
        # Router-as-node mode: each router IS a node, mapped to itself
        node_ids = [f"r{rid_map[rid]}" for rid in router_ids]
        node_router_connections = {f"r{rid_map[rid]}": rid for rid in router_ids}

    return router_ids, router_adj, node_ids, node_router_connections, G, C, R_per, Np


if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "topo/Dragonfly/topo_128.cfg"
    max_trees = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    mode = sys.argv[3] if len(sys.argv) > 3 else "router"

    router_ids, router_adj, node_ids, node_router_connections, G, C, R_per, Np = \
        build_dragonfly_for_kls(cfg_path, mode=mode)

    N = len(node_ids)
    Nr = len(router_ids)
    n_edges = int(np.sum(router_adj > 0)) // 2
    print(f"Dragonfly: G={G} C={C} R={R_per} Np={Np}")
    print(f"  Mode: {mode}")
    print(f"  N={N} nodes, {Nr} routers, {n_edges} router-router links")
    print(f"  Edge variables: {N*(N-1)//2}")
    print(f"  Solving KLS with max_trees={max_trees} ...")
    print(flush=True)

    ans = select_trees_kls_style_with_floyd(
        router_ids=router_ids,
        node_ids=node_ids,
        router_adj=router_adj,
        node_router_connections=node_router_connections,
        max_trees=max_trees,
        root_node_index=0,
    )

    if ans is None:
        print("No feasible solution found.")
    else:
        print()
        print(f"best k = {ans['num_trees']}")
        print(f"best ratio = {ans['ratio']:.6f}")
        print(f"max router load = {ans['max_router_load']}")
        print()
        for t, edges in enumerate(ans["tree_edge_lists"], start=1):
            print(f"Tree {t}: {len(edges)} edges")
        print()
        print("Router edge loads:")
        for (ra, rb), load in sorted(ans["router_edge_loads"].items()):
            if load > 0:
                print(f"  {ra} <-> {rb} : {load}")
