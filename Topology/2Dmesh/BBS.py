#!/usr/bin/env python3
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from fractions import Fraction
from math import gcd
from functools import reduce
from typing import Any, Dict, List, Tuple, Any, Optional, Set
from networkx.algorithms.matching import max_weight_matching
import sys
import os
import csv
import copy
from collections import deque
import heapq
import random
import itertools


def is_2d_mesh(adj_list: Dict[Any, List[Any]]) -> Optional[Tuple[int, int]]:
    """
    Check if the given adjacency list represents a 2D mesh.
    Returns (rows, cols) if it is a 2D mesh, None otherwise.
    """
    # Check if all nodes are tuples of two integers
    if not all(isinstance(node, tuple) and len(node) == 2 and 
              all(isinstance(x, int) for x in node) for node in adj_list.keys()):
        return None
    
    # Get dimensions
    rows = max(r for r, _ in adj_list.keys()) + 1
    cols = max(c for _, c in adj_list.keys()) + 1
    
    # Verify grid structure
    for (r, c), neighbors in adj_list.items():
        expected_neighbors = []
        if r > 0: expected_neighbors.append((r-1, c))
        if r < rows-1: expected_neighbors.append((r+1, c))
        if c > 0: expected_neighbors.append((r, c-1))
        if c < cols-1: expected_neighbors.append((r, c+1))
        
        # Check if all expected neighbors are present and no extra neighbors
        if set(neighbors) != set(expected_neighbors):
            return None
    
    return rows, cols

def compute_2d_mesh_limitation_matrix(rows: int, cols: int) -> np.ndarray:
    """
    Compute limitation matrix for a 2D mesh using the rules from ComputeLimitationMatrix2Dmesh.py.
    Returns a numpy array.
    """
    num_nodes = rows * cols
    if num_nodes == 0:
        return np.zeros((0, 0), dtype=int)
    limitation_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    # --- Rule 1: Node 0 Flow --- 
    node_idx = 0
    r, c = 0, 0
    # Send Right (if possible)
    if c + 1 < cols:
        right_node = node_idx + 1
        limitation_matrix[node_idx][right_node] = 3
    # Send Down (if possible)
    if r + 1 < rows:
        down_node = node_idx + cols
        limitation_matrix[node_idx][down_node] = 3
    # --- Rules for other nodes --- 
    for node_idx in range(1, num_nodes):
        r = node_idx // cols
        c = node_idx % cols
        up_node = node_idx - cols
        down_node = node_idx + cols
        left_node = node_idx - 1
        right_node = node_idx + 1
        is_top = r == 0
        is_bottom = r == rows - 1
        is_left = c == 0
        is_right = c == cols - 1
        is_interior = not is_top and not is_bottom and not is_left and not is_right
        if is_interior:
            if r > 0: limitation_matrix[node_idx][up_node] = 1
            if r < rows - 1: limitation_matrix[node_idx][down_node] = 1
            if c > 0: limitation_matrix[node_idx][left_node] = 1
            if c < cols - 1: limitation_matrix[node_idx][right_node] = 1
            continue
        if is_top and c > 0:
            if c < cols - 1:
                limitation_matrix[node_idx][right_node] = 3
        if is_left and r > 0:
            if r < rows - 1:
                limitation_matrix[node_idx][down_node] = 3
        if is_bottom:
            if c < cols - 1:
                limitation_matrix[node_idx][right_node] = 2
            if c > 0:
                limitation_matrix[node_idx][left_node] = 1
        if is_right:
            if r < rows - 1:
                limitation_matrix[node_idx][down_node] = 2
            if r > 0:
                limitation_matrix[node_idx][up_node] = 1
    # --- Pass 2: Ensure Interior Nodes Receive weight 1 from neighbors ---
    for node_idx in range(num_nodes):
        r = node_idx // cols
        c = node_idx % cols
        is_top = r == 0
        is_bottom = r == rows - 1
        is_left = c == 0
        is_right = c == cols - 1
        is_interior = not is_top and not is_bottom and not is_left and not is_right
        if is_interior:
            up_node = node_idx - cols
            down_node = node_idx + cols
            left_node = node_idx - 1
            right_node = node_idx + 1
            if r > 0:
                limitation_matrix[up_node][node_idx] = 1
            if r < rows - 1:
                limitation_matrix[down_node][node_idx] = 1
            if c > 0:
                limitation_matrix[left_node][node_idx] = 1
            if c < cols - 1:
                limitation_matrix[right_node][node_idx] = 1
    return limitation_matrix

def ComputeLimitationMatrix(
    adj_list: Dict[Any, List[Any]],
    root: Any,
    K: float = 1.0
) -> Tuple[np.ndarray, Dict[Any,int], int]:
    """
    Build and integer-scale the limitation matrix for G → G'.
    For 2D/3D meshes, uses specialized computation rules.
    For other graphs, uses the general K/deg(v) rule.
    
    Returns:
      L_int      – the integer limitation matrix (dtype=int),
      index_map  – mapping node → matrix index,
      scale      – the integer scale factor used (1 for mesh).
    """
    # Check if this is a 2D mesh
    mesh_dims_2d = is_2d_mesh(adj_list)
    if mesh_dims_2d is not None:
        rows, cols = mesh_dims_2d
        # For 2D mesh, use specialized computation
        L_int = compute_2d_mesh_limitation_matrix(rows, cols)
        idx = {v: i for i, v in enumerate(sorted(adj_list.keys()))}
        return L_int, idx, 1  # scale is 1 for 2D mesh
    
    # For non-mesh graphs, use the general computation
    nodes = list(adj_list.keys())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)

    for v, neighs in adj_list.items():
        if v == root:
            continue
        deg = len(neighs)
        if deg == 0:
            continue
        w = K / deg
        j = idx[v]
        for u in neighs:
            i = idx[u]
            L[i, j] = w

    # Compute LCM of denominators
    fracs = [Fraction(val).limit_denominator() for val in L.flatten() if val > 0]
    dens = [f.denominator for f in fracs]
    scale = reduce(lambda a, b: a * b // gcd(a, b), dens, 1) if dens else 1

    # Scale to integer
    L_int = (L * scale).round().astype(int)

    return L_int, idx, scale

def BuildGraph(
    adj_list: Dict[Any, List[Any]],
    root: Any,
    K: float = 1.0
) -> Tuple[nx.Graph, nx.DiGraph, np.ndarray, Dict[Any,int], int]:
    """
    From adj_list and root, build:
      G        – undirected skeleton (nx.Graph),
      Gp_int   – directed integer-weighted graph (nx.DiGraph),
      L_int    – integer limitation matrix,
      idx      – node→index map,
      scale    – integer scale factor.
    """
    # get integer matrix and mapping
    L_int, idx, scale = ComputeLimitationMatrix(adj_list, root, K)

    # build undirected skeleton G
    G = nx.Graph()
    for u, neighs in adj_list.items():
        for v in neighs:
            G.add_edge(u, v)

    # build directed integer-weighted graph Gp_int
    nodes = list(adj_list.keys())
    inv_idx = {i: v for v, i in idx.items()}
    Gp_int = nx.DiGraph()
    Gp_int.add_nodes_from(nodes)
    n = L_int.shape[0]
    for i in range(n):
        for j in range(n):
            w = L_int[i, j]
            if w > 0:
                Gp_int.add_edge(inv_idx[i], inv_idx[j], weight=w)

    return G, Gp_int, L_int, idx, scale

def make_multigraph(
    adj_list: Dict[Any, List[Any]],
    root: Any,
    K: float = 1.0
) -> Tuple[nx.MultiGraph, Dict[Any,int], int]:
    """
    Build an undirected multigraph from the integer limitation matrix
    of a generic graph (given as adj_list and root).
    Returns (G, index_map, scale).
    """
    L_int, idx, scale = ComputeLimitationMatrix(adj_list, root, K)

    inv_idx = {i: v for v, i in idx.items()}

    G = nx.MultiGraph()
    G.add_nodes_from(adj_list.keys())
    n = L_int.shape[0]
    for i in range(n):
        for j in range(n):
            w = L_int[i, j]
            if w > 0:
                u = inv_idx[i]
                v = inv_idx[j]
                for _ in range(int(w)):
                    G.add_edge(u, v)

    return G, idx, scale

def euler_split(G: nx.MultiGraph, Delta: int) -> Tuple[nx.MultiGraph, nx.MultiGraph]:
    """
    Perform the "Euler partition" on G (max-degree Delta, assumed even):
      1. Build H by adding dummy edges so that every vertex has degree exactly Delta.
      2. For each connected component of H, find an Euler tour and alternately
         assign real edges to G1 and G2, discarding dummy edges.
    Returns two MultiGraphs G1, G2 each on the same node set, with
    max_degree(G1), max_degree(G2) ≤ Delta // 2, unless a fallback move is required.
    """
    H = nx.MultiGraph()
    H.add_nodes_from(G.nodes())
    for u, v, key, data in G.edges(keys=True, data=True):
        # Store a marker so we can tell real edges from dummy
        H.add_edge(u, v, key=("real", u, v, key), **data)

    deficits: List[Any] = []
    for v in H.nodes():
        deficit = Delta - H.degree(v)
        for _ in range(deficit):
            deficits.append(v)

    for i in range(0, len(deficits), 2):
        u = deficits[i]
        v = deficits[i + 1]
        H.add_edge(u, v, key=("dummy", i // 2))

    G1 = nx.MultiGraph()
    G2 = nx.MultiGraph()
    G1.add_nodes_from(G.nodes())
    G2.add_nodes_from(G.nodes())

    # To find connected components, use a simple graph view
    H_simple = nx.Graph()
    H_simple.add_nodes_from(H.nodes())
    for u, v, _ in H.edges(keys=True):
        H_simple.add_edge(u, v)

    for comp in nx.connected_components(H_simple):
        Hc = H.subgraph(comp).copy()
        # Eulerian circuit on each component
        tour = list(nx.eulerian_circuit(Hc, source=next(iter(comp)), keys=True))
        take_from_G1 = True

        for u, v, key in tour:
            if key[0] == "dummy":
                take_from_G1 = not take_from_G1
                continue

            _, u0, v0, orig_key = key
            data = G[u0][v0][orig_key]
            if take_from_G1:
                G1.add_edge(u0, v0, key=orig_key, **data)
            else:
                G2.add_edge(u0, v0, key=orig_key, **data)
            take_from_G1 = not take_from_G1

    return G1, G2

def rebalance_edge_colors(G: nx.MultiGraph) -> None:
    """
    Improved rebalancing to minimize the gap between the largest and smallest color-buckets:
      • First, determine the actual number of colors in use.
      • Compute ideal lower = floor(total_edges/num_colors) and upper = ceil(total_edges/num_colors).
      • While there exists any color with count > upper or < lower, attempt to move one edge:
        – Consider every pair (c_big, c_small) where count[c_big] > upper and count[c_small] < lower,
          and attempt to move one edge from c_big to c_small if no adjacency conflict.
      • Repeat until no such move is possible.
    """
    # Determine how many colors are actually in use
    counts: Dict[int,int] = {}
    for _, _, _, data in G.edges(keys=True, data=True):
        c = data["color"]
        counts[c] = counts.get(c, 0) + 1
    if not counts:
        return

    num_colors = max(counts.keys()) + 1
    total_edges = G.number_of_edges()
    lower = total_edges // num_colors
    upper = lower + (1 if total_edges % num_colors != 0 else 0)

    def compute_counts() -> Dict[int, int]:
        ec = {c: 0 for c in range(num_colors)}
        for _, _, _, data in G.edges(keys=True, data=True):
            c = data["color"]
            ec[c] += 1
        return ec

    while True:
        curr = compute_counts()
        overfull = [c for c, cnt in curr.items() if cnt > upper]
        underfull = [c for c, cnt in curr.items() if cnt < lower]
        if not overfull or not underfull:
            break

        moved = False
        # Try every pair (c_big, c_small)
        for c_big in sorted(overfull, key=lambda c: curr[c], reverse=True):
            for c_small in sorted(underfull, key=lambda c: curr[c]):
                # Attempt to move one edge from c_big to c_small
                for u, v, k, data in G.edges(keys=True, data=True):
                    if data["color"] != c_big:
                        continue
                    # Check adjacency conflict at u and v
                    conflict = False
                    for nbr in G[u]:
                        for subk, dat in G[u][nbr].items():
                            if dat["color"] == c_small:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        continue
                    for nbr in G[v]:
                        for subk, dat in G[v][nbr].items():
                            if dat["color"] == c_small:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        continue

                    # Perform move
                    G[u][v][k]["color"] = c_small
                    moved = True
                    break
                if moved:
                    break
            if moved:
                break

        if not moved:
            break

def _color_recursive(G: nx.MultiGraph) -> Tuple[nx.MultiGraph, Dict[Tuple[Any, Any, int], int]]:
    """
    Recursively color edges of G and return a new colored graph Gc
    plus a color_map mapping each (u,v,key) → color.

    Uses Δ-partitioning exactly as in the paper, with a fallback if the split fails
    to reduce the maximum degree.
    """
    Delta = max(d for _, d in G.degree())

    # Base case: Delta ≤ 1
    if Delta <= 1:
        Gc = nx.MultiGraph()
        Gc.add_nodes_from(G.nodes())
        color_map: Dict[Tuple[Any, Any, int], int] = {}
        for u, v, k, data in G.edges(keys=True, data=True):
            Gc.add_edge(u, v, key=k, **data)
            Gc[u][v][k]["color"] = 0
            color_map[(u, v, k)] = 0
        return Gc, color_map

    # If Delta is odd: remove one Δ-matching using Gabow '76 and color those edges 0
    pre_colored: List[Tuple[Any, Any, int]] = []
    if Delta % 2 == 1:
        S = nx.Graph()
        S.add_nodes_from(G.nodes())
        buckets: Dict[Tuple[Any, Any], List[Tuple[Any, Any, int]]] = {}
        for u, v, k in G.edges(keys=True):
            a, b = (u, v) if u <= v else (v, u)
            buckets.setdefault((a, b), []).append((u, v, k))
        S.add_edges_from(buckets.keys())

        weight = {v: (1 if G.degree(v) == Delta else 0) for v in G.nodes()}
        for u, v in S.edges():
            S[u][v]["weight"] = weight[u] + weight[v]

        M = max_weight_matching(S, maxcardinality=False, weight="weight")

        remaining = nx.MultiGraph()
        remaining.add_nodes_from(G.nodes())
        for u, v, k, data in G.edges(keys=True, data=True):
            remaining.add_edge(u, v, key=k, **data)

        for u, v in M:
            a, b = (u, v) if u <= v else (v, u)
            u0, v0, k0 = buckets[(a, b)].pop()
            remaining.remove_edge(u0, v0, k0)
            # Re-add into pre_colored list; we'll assign color 0 later
            pre_colored.append((u0, v0, k0))

        Delta_minus_1 = Delta - 1
        # Recursively color the remaining (now even‐Δ) graph
        G1c, color_map1 = _color_recursive(remaining)

        # Build Gc by merging pre_colored edges (color 0) with G1c (shifted by +1)
        Gc = nx.MultiGraph()
        Gc.add_nodes_from(G.nodes())
        color_map: Dict[Tuple[Any, Any, int], int] = {}

        # Add pre_colored edges with color 0, preserving key and data
        for u0, v0, k0 in pre_colored:
            data = G[u0][v0][k0]
            Gc.add_edge(u0, v0, key=k0, **data)
            Gc[u0][v0][k0]["color"] = 0
            color_map[(u0, v0, k0)] = 0

        # Add edges from G1c with color = original_color + 1
        for u, v, k, data in G1c.edges(keys=True, data=True):
            # Copy full attribute dict, then override color
            attr = dict(data)
            orig_color = attr.get("color", 0)
            new_color = orig_color + 1
            attr["color"] = new_color
            Gc.add_edge(u, v, key=k, **attr)
            color_map[(u, v, k)] = new_color

        return Gc, color_map

    # Delta is even: perform Euler split
    G1, G2 = euler_split(G, Delta)

    # FALLBACK: if split left one half still at full Δ, move one real edge out
    d1 = max((d for _, d in G1.degree()), default=0)
    d2 = max((d for _, d in G2.degree()), default=0)
    if d1 == Delta:
        # Move a single edge from G1 to G2 (preserve key and data)
        for u, v, k in G1.edges(keys=True):
            data = G1[u][v][k]
            G1.remove_edge(u, v, k)
            G2.add_edge(u, v, key=k, **data)
            break
    elif d2 == Delta:
        for u, v, k in G2.edges(keys=True):
            data = G2[u][v][k]
            G2.remove_edge(u, v, k)
            G1.add_edge(u, v, key=k, **data)
            break

    # Recurse on the (now strictly smaller‐max‐degree) subgraphs
    G1c, color_map1 = _color_recursive(G1)
    G2c, color_map2 = _color_recursive(G2)

    # Merge G1c and G2c into Gc, shifting G2c colors by Delta//2
    half = Delta // 2
    Gc = nx.MultiGraph()
    Gc.add_nodes_from(G.nodes())
    color_map: Dict[Tuple[Any, Any, int], int] = {}

    # Copy G1c edges (colors 0..half-1)
    for u, v, k, data in G1c.edges(keys=True, data=True):
        attr = dict(data)
        # color is already correct for G1c
        Gc.add_edge(u, v, key=k, **attr)
        color_map[(u, v, k)] = attr["color"]

    # Copy G2c edges, adding 'half' to their colors
    for u, v, k, data in G2c.edges(keys=True, data=True):
        attr = dict(data)
        orig_color = attr.get("color", 0)
        new_color = orig_color + half
        attr["color"] = new_color
        Gc.add_edge(u, v, key=k, **attr)
        color_map[(u, v, k)] = new_color

    return Gc, color_map

def delta_edge_coloring(G: nx.MultiGraph) -> int:
    """
    Edge-color G in place using exactly Delta colors (for even Delta),
    or Delta colors after removing a matching (for odd Delta).
    Returns the number of colors used.
    """
    # Call the recursive builder to get Gc and the color_map
    Gc, color_map = _color_recursive(G)

    # Rebalance in-place on Gc
    # rebalance_edge_colors(Gc)  # DISABLED: Prevents interference with distance-aware optimizations

    # Build a final color_map from Gc (in case rebalancing changed anything)
    final_color_map: Dict[Tuple[Any, Any, int], int] = {}
    for u, v, k, data in Gc.edges(keys=True, data=True):
        final_color_map[(u, v, k)] = data["color"]

    # Copy colors from Gc back into G using the explicit map
    for u, v, k in G.edges(keys=True):
        # It's possible edges were swapped between G1 and G2, but keys remain consistent.
        G[u][v][k]["color"] = final_color_map[(u, v, k)]

    # Number of colors is the max color + 1
    num_colors = max(final_color_map.values()) + 1
    return num_colors

def count_edges_per_frame(G: nx.MultiGraph) -> Dict[int, int]:
    """
    Count the number of edges in each frame (color).
    Returns a dictionary mapping color to edge count.
    """
    edge_counts: Dict[int, int] = {}
    for _, _, _, data in G.edges(keys=True, data=True):
        color = data.get("color", -1)
        if color >= 0:
            edge_counts[color] = edge_counts.get(color, 0) + 1
    return edge_counts

def get_node_coords(node_idx: int, p: int, q: int, r: int) -> Tuple[int, int, int]:
    """Convert 1D node index to 3D coordinates (or 2D if r=0)."""
    x = node_idx % p
    y = (node_idx // p) % q
    z = node_idx // (p * q) if r > 0 else 0
    return (x, y, z)

def compute_manhattan_distances(p: int, q: int, r: int) -> np.ndarray:
    """
    Build an array `dist[v]` = Manhattan distance from the root (0,0,0)
    for each node index v in a p×q×r grid. If r=0, we treat it as 2D (p×q).
    """
    n = p * q * (r if r > 0 else 1)
    dist = np.zeros((n,), dtype=int)
    for v in range(n):
        x, y, z = get_node_coords(v, p, q, r)
        dist[v] = x + y + z
    return dist

def order_frames(
    frames: List[np.ndarray],
    p: int,
    q: int,
    r: int,
    N: int
) -> List[np.ndarray]:
    """
    Reorder directed‐matching frames using a two‐tiered "urgency" + "deliverable‐count" score:
      1. Compute urgency S(c) = sum_{u→v in frame c} [ (N – data_sim[v]) * (dist[v] + 1 ) ]
         counting only edges where data_sim[u] > 0 and data_sim[v] < N.
      2. If no frame has S(c) > 0, pick the unused frame with the largest number of
         *deliverable* edges (u→v such that data_sim[u]>0 and data_sim[v]<N).
      3. "Play" the chosen frame in data_sim (increment data_sim[v] by 1 on each valid edge).
      4. Repeat until all frames are ordered.

    This adjustment guarantees that each step moves at least one chunk (whenever any deliverable edge exists),
    avoiding multi‐slot stalls while still aggressively prioritizing far‐away, high‐deficit nodes.

    Args:
      frames: List of n×n {0,1} arrays, each a directed matching.
      p, q, r: Grid dimensions.
      N: Target chunks per non‐root node.

    Returns:
      A reordered list of frames (same length as input).
    """
    n = frames[0].shape[0]
    used = set()
    ordered: List[np.ndarray] = []

    # Simulated chunk‐counts: how many chunks node v "would have" by the time we schedule each frame.
    data_sim = np.zeros((n,), dtype=int)
    data_sim[0] = N  # root starts with all N

    # Precompute Manhattan distances dist[v]
    dist = compute_manhattan_distances(p, q, r)

    # Precompute, for each frame c, the list of edges (u_list[c], v_list[c])
    u_list: Dict[int, List[int]] = {}
    v_list: Dict[int, List[int]] = {}
    for i, F in enumerate(frames):
        us, vs = np.nonzero(F)
        u_list[i] = us.tolist()
        v_list[i] = vs.tolist()

    # Enhanced scoring for rectangular grids (like 4×16)
    # For rectangular grids, prioritize frames that help spread information along the longer dimension
    is_rectangular = p != q
    if is_rectangular:
        # For rectangular grids, prioritize frames that move data along the longer dimension
        longer_dim = max(p, q)
        shorter_dim = min(p, q)
        print(f"Rectangular grid detected: {p}×{q}, optimizing for longer dimension ({longer_dim})")

    # Smoothing memory for early ramp (track newly activated receivers)
    last_activations: Optional[int] = None

    while len(used) < len(frames):
        best_idx = -1
        best_score = -1e18

        # First pass: compute urgency S(i) for each unused frame with enhanced scoring
        for i in range(len(frames)):
            if i in used:
                continue

            S_i = 0.0
            deliverables = 0
            activations = 0
            forward_sum = 0.0
            for u, v in zip(u_list[i], v_list[i]):
                if data_sim[u] > 0 and data_sim[v] < N:
                    # Base urgency score
                    base_score = (N - data_sim[v]) * (dist[v] + 1)
                    
                    # Enhanced scoring for rectangular grids
                    if is_rectangular:
                        # Get coordinates of sender and receiver
                        u_row, u_col = u // q, u % q
                        v_row, v_col = v // q, v % q
                        
                        # Bonus for moving along the longer dimension
                        if p > q:  # Taller than wide
                            if u_row != v_row:  # Moving vertically (along longer dimension)
                                base_score *= 1.5
                        else:  # Wider than tall
                            if u_col != v_col:  # Moving horizontally (along longer dimension)
                                base_score *= 1.5
                        
                        # Additional bonus for moving towards the far end of the longer dimension
                        if p > q:  # Taller than wide
                            if v_row > u_row:  # Moving down (towards bottom)
                                base_score *= 1.2
                        else:  # Wider than tall
                            if v_col > u_col:  # Moving right (towards right edge)
                                base_score *= 1.2
                    
                    S_i += base_score
                    deliverables += 1
                    if data_sim[v] == 0:
                        activations += 1
                    dv = dist[v] - dist[u]
                    if dv > 0:
                        forward_sum += dv
                    elif dv == 0:
                        forward_sum -= 0.5
                    else:
                        forward_sum -= 2.0
                    
            # Smoothing: penalize sharp jumps of activations early on
            smoothing = 0.0
            if last_activations is not None:
                diff = abs(activations - last_activations)
                if len(used) <= 3:
                    smoothing = -6.0 * diff
                else:
                    smoothing = -3.0 * diff

            # Composite score (activation-first)
            score = 0.5 * S_i + 0.5 * forward_sum + 8.0 * deliverables + 200.0 * activations + smoothing
            if score > best_score:
                best_score = score
                best_idx = i

        # If every unused frame had zero deliverables (best_score can still be 0 with smoothing),
        # fall back to the largest deliverable count rule.
        if best_idx == -1 or best_score <= 0:
            # Tier‐2: find the unused frame with the largest count of deliverable edges
            best_count = -1
            chosen = -1
            for i in range(len(frames)):
                if i in used:
                    continue

                cnt = 0
                for u, v in zip(u_list[i], v_list[i]):
                    if data_sim[u] > 0 and data_sim[v] < N:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    chosen = i

            # If no deliverable edges at all (cnt==0 for every unused frame),
            # just pick any unused frame to finish out.
            if best_count <= 0:
                for i in range(len(frames)):
                    if i not in used:
                        chosen = i
                        break
        else:
            chosen = best_idx

        # "Play" the chosen frame in our simulation
        for u, v in zip(u_list[chosen], v_list[chosen]):
            if data_sim[u] > 0 and data_sim[v] < N:
                data_sim[v] += 1

        ordered.append(frames[chosen])
        used.add(chosen)

        # Update smoothing memory for next selection (activations only)
        a_next = 0
        for u, v in zip(u_list[chosen], v_list[chosen]):
            if data_sim[u] > 0 and data_sim[v] < N and data_sim[v] == 0:
                a_next += 1
        last_activations = a_next

    # Optional: rotate the cycle to reduce early oscillations
    # Evaluate each rotation by simulating two cycles with a small-N proxy and
    # picking the rotation minimizing the sum of absolute deltas in active edges.
    def eval_rotation(rot: int) -> float:
        n = frames[0].shape[0]
        sim_N = max(8, min(32, N))
        sim_state = np.zeros((n,), dtype=int)
        sim_state[0] = sim_N
        seq = ordered[rot:] + ordered[:rot]
        # Build edge lists for fast iteration
        e_us = []
        e_vs = []
        for F in seq:
            us, vs = np.nonzero(F)
            e_us.append(us)
            e_vs.append(vs)
        counts: List[int] = []
        for _ in range(2 * len(seq)):
            idx = _ % len(seq)
            us = e_us[idx]
            vs = e_vs[idx]
            # count deliverables this step
            cnt = 0
            for u, v in zip(us, vs):
                if sim_state[u] > 0 and sim_state[v] < sim_N:
                    cnt += 1
            counts.append(cnt)
            # play the frame
            for u, v in zip(us, vs):
                if sim_state[u] > 0 and sim_state[v] < sim_N:
                    sim_state[v] += 1
        # oscillation penalty: sum of absolute consecutive diffs
        pen = 0
        for a, b in zip(counts, counts[1:]):
            pen += abs(a - b)
        return pen

    best_rot = 0
    best_pen = float('inf')
    for r_off in range(len(ordered)):
        pen = eval_rotation(r_off)
        if pen < best_pen:
            best_pen = pen
            best_rot = r_off

    if best_rot != 0:
        ordered = ordered[best_rot:] + ordered[:best_rot]

    return ordered

def compute_bfs_layers(p: int, q: int, r: int) -> np.ndarray:
    """
    Compute BFS layers (distance from root) for all nodes in a grid.
    Returns an array where layers[i] is the BFS layer (distance) of node i from root.
    """
    n = p * q * (r if r > 0 else 1)
    layers = np.full(n, -1, dtype=int)  # -1 means unvisited
    layers[0] = 0  # Root is at layer 0
    
    # Use a queue for BFS
    queue = deque([0])
    
    while queue:
        current = queue.popleft()
        current_layer = layers[current]
        
        # Get neighbors of current node
        x, y, z = get_node_coords(current, p, q, r)
        
        # Check all 4 (2D) or 6 (3D) neighbors
        neighbors = []
        if x > 0:  # Left
            neighbors.append(current - 1)
        if x < p - 1:  # Right
            neighbors.append(current + 1)
        if y > 0:  # Up
            neighbors.append(current - p)
        if y < q - 1:  # Down
            neighbors.append(current + p)
        if r > 0:  # 3D case
            if z > 0:  # Back
                neighbors.append(current - p * q)
            if z < r - 1:  # Front
                neighbors.append(current + p * q)
        
        # Add unvisited neighbors to queue
        for neighbor in neighbors:
            if layers[neighbor] == -1:
                layers[neighbor] = current_layer + 1
                queue.append(neighbor)
    
    return layers

def optimize_frames_by_distance_swapping(
    frames: List[np.ndarray], 
    p: int, 
    q: int, 
    r: int
) -> List[np.ndarray]:
    """
    Optimize frame structure by swapping edges between frames to prioritize distance progression.
    Maintains all constraints:
    1. Each frame has vertex-disjoint edges
    2. Exactly 8 frames for 2D
    3. Frame stacking equals original multigraph
    """
    n = frames[0].shape[0]
    num_frames = len(frames)
    
    # Precompute BFS layers and Manhattan distances
    layers = compute_bfs_layers(p, q, r)
    dist = compute_manhattan_distances(p, q, r)
    
    # Create a copy of frames to work with
    optimized_frames = [frame.copy() for frame in frames]
    
    # Track which edges are in which frames for easy lookup
    edge_to_frame = {}  # (u, v) -> frame_index
    frame_edges = [[] for _ in range(num_frames)]  # frame_edges[i] = list of (u, v) in frame i
    
    # Build initial mapping
    for frame_idx, frame in enumerate(optimized_frames):
        for u in range(n):
            for v in range(n):
                if frame[u, v] > 0:
                    edge_to_frame[(u, v)] = frame_idx
                    frame_edges[frame_idx].append((u, v))
    
    # Enhanced optimization for rectangular grids
    is_rectangular = p != q
    if is_rectangular:
        print(f"Applying rectangular grid optimization for {p}×{q}")
        # For rectangular grids, prioritize frames that help spread along the longer dimension
        longer_dim = max(p, q)
        shorter_dim = min(p, q)
    
    def can_swap_edges(edge1: Tuple[int, int], edge2: Tuple[int, int]) -> bool:
        """Check if two edges can be swapped between their frames without violating constraints."""
        u1, v1 = edge1
        u2, v2 = edge2
        frame1 = edge_to_frame[edge1]
        frame2 = edge_to_frame[edge2]
        
        # Check if swapping would create vertex conflicts in either frame
        frame1_vertices = set()
        frame2_vertices = set()
        
        # Add all vertices from frame1 except edge1
        for u, v in frame_edges[frame1]:
            if (u, v) != edge1:
                frame1_vertices.add(u)
                frame1_vertices.add(v)
        
        # Add all vertices from frame2 except edge2
        for u, v in frame_edges[frame2]:
            if (u, v) != edge2:
                frame2_vertices.add(u)
                frame2_vertices.add(v)
        
        # Check if adding edge2 to frame1 would create conflicts
        if u2 in frame1_vertices or v2 in frame1_vertices:
            return False
        
        # Check if adding edge1 to frame2 would create conflicts
        if u1 in frame2_vertices or v1 in frame2_vertices:
            return False
        
        return True
    
    def perform_swap(edge1: Tuple[int, int], edge2: Tuple[int, int]):
        """Perform the swap between two edges."""
        u1, v1 = edge1
        u2, v2 = edge2
        frame1 = edge_to_frame[edge1]
        frame2 = edge_to_frame[edge2]
        
        # Remove edges from their current frames
        optimized_frames[frame1][u1, v1] -= 1
        optimized_frames[frame2][u2, v2] -= 1
        
        # Add edges to their new frames
        optimized_frames[frame1][u2, v2] += 1
        optimized_frames[frame2][u1, v1] += 1
        
        # Update tracking structures
        edge_to_frame[edge1] = frame2
        edge_to_frame[edge2] = frame1
        
        # Update frame_edges lists
        frame_edges[frame1].remove(edge1)
        frame_edges[frame1].append(edge2)
        frame_edges[frame2].remove(edge2)
        frame_edges[frame2].append(edge1)
    
    def edge_distance_score(edge: Tuple[int, int]) -> int:
        """Calculate how well an edge progresses toward distant nodes."""
        u, v = edge
        u_layer = layers[u]
        v_layer = layers[v]
        u_dist = dist[u]
        v_dist = dist[v]
        
        # Prefer edges that progress to higher layers and higher distances
        layer_progress = v_layer - u_layer
        dist_progress = v_dist - u_dist
        
        # Enhanced scoring: heavily penalize backward edges, reward forward progress
        if layer_progress < 0:  # Backward edge
            return -10000 + layer_progress
        elif layer_progress == 0:  # Same layer
            return -1000 + dist_progress
        else:  # Forward edge
            base_score = layer_progress * 1000 + dist_progress * 100 + v_dist
            
            # AGGRESSIVE scoring for rectangular grids
            if is_rectangular:
                # Get coordinates of sender and receiver
                u_row, u_col = u // q, u % q
                v_row, v_col = v // q, v % q
                
                # For 4×16 grids (p=4, q=16), vertical progress is CRITICAL
                if p > q:  # Taller than wide (like 4×16)
                    if u_row != v_row:  # Moving vertically (along longer dimension)
                        base_score *= 3.0  # Much more aggressive multiplier
                        # Extra bonus for moving downward (towards higher row numbers)
                        if v_row > u_row:
                            base_score *= 2.0  # Double the score for downward movement
                        # Additional bonus based on how far down we're moving
                        row_progress = v_row - u_row
                        base_score += row_progress * 500  # Large bonus for row progress
                    else:  # Horizontal movement - heavily penalize
                        base_score *= 0.3  # Significantly reduce horizontal movement
                else:  # Wider than tall
                    if u_col != v_col:  # Moving horizontally (along longer dimension)
                        base_score *= 3.0
                        # Extra bonus for moving rightward
                        if v_col > u_col:
                            base_score *= 2.0
                        # Additional bonus based on column progress
                        col_progress = v_col - u_col
                        base_score += col_progress * 500
                    else:  # Vertical movement - heavily penalize
                        base_score *= 0.3
            
            return int(base_score)
    
    def frame_quality_score(frame_idx: int) -> int:
        """Calculate overall quality of a frame based on distance progression."""
        total_score = 0
        for u, v in frame_edges[frame_idx]:
            total_score += edge_distance_score((u, v))
        return total_score
    
    # Perform optimization through edge swapping
    max_iterations = 500  # Increased for more thorough optimization
    iterations = 0
    total_swaps = 0
    
    # Enhanced optimization for rectangular grids
    if is_rectangular:
        print(f"Applying aggressive rectangular grid optimization for {p}×{q}")
        # For rectangular grids, we need to be more aggressive about moving edges
        # to frames that help spread along the longer dimension
    
    while iterations < max_iterations:
        iterations += 1
        swaps_performed = 0
        
        # Find all possible beneficial swaps with more aggressive criteria
        for frame1 in range(num_frames):
            for frame2 in range(frame1 + 1, num_frames):
                for edge1 in frame_edges[frame1][:]:  # Copy to avoid modification during iteration
                    for edge2 in frame_edges[frame2][:]:
                        if can_swap_edges(edge1, edge2):
                            # Calculate current and potential frame quality
                            current_frame1_score = frame_quality_score(frame1)
                            current_frame2_score = frame_quality_score(frame2)
                            
                            # Simulate the swap
                            old_edge1_frame = edge_to_frame[edge1]
                            old_edge2_frame = edge_to_frame[edge2]
                            
                            # Temporarily update tracking
                            edge_to_frame[edge1] = frame2
                            edge_to_frame[edge2] = frame1
                            frame_edges[frame1].remove(edge1)
                            frame_edges[frame1].append(edge2)
                            frame_edges[frame2].remove(edge2)
                            frame_edges[frame2].append(edge1)
                            
                            # Calculate new scores
                            new_frame1_score = frame_quality_score(frame1)
                            new_frame2_score = frame_quality_score(frame2)
                            
                            # Revert tracking
                            edge_to_frame[edge1] = old_edge1_frame
                            edge_to_frame[edge2] = old_edge2_frame
                            frame_edges[frame1].remove(edge2)
                            frame_edges[frame1].append(edge1)
                            frame_edges[frame2].remove(edge1)
                            frame_edges[frame2].append(edge2)
                            
                            # Check if swap improves overall quality
                            current_total = current_frame1_score + current_frame2_score
                            new_total = new_frame1_score + new_frame2_score
                            
                            if new_total > current_total:
                                perform_swap(edge1, edge2)
                                swaps_performed += 1
                                total_swaps += 1
                                break  # Move to next edge1
                    if swaps_performed > 0:
                        break  # Move to next frame1
        
        # If no swaps were performed, we're done
        if swaps_performed == 0:
            break
    
    # Verify constraints are maintained
    stacked = sum(optimized_frames)
    original_stacked = sum(frames)
    
    if not np.array_equal(stacked, original_stacked):
        print("WARNING: Frame stacking constraint violated! Reverting to original frames.")
        return frames
    
    print(f"Frame optimization completed in {iterations} iterations with {total_swaps} total swaps \n")
    return optimized_frames

def generate_frames() -> Tuple[List[np.ndarray], np.ndarray]:
    global p, q, r
    # Create adjacency list for 2D mesh
    adj_list = {}
    n = p * q
    for i in range(p):
        for j in range(q):
            node = i * q + j
            adj_list[node] = []
            # Add horizontal neighbors
            if j > 0:
                adj_list[node].append(i * q + (j - 1))
            if j < q - 1:
                adj_list[node].append(i * q + (j + 1))
            # Add vertical neighbors
            if i > 0:
                adj_list[node].append((i - 1) * q + j)
            if i < p - 1:
                adj_list[node].append((i + 1) * q + j)
    # Convert adj_list to tuple format for compatibility with existing Euler coloring code
    adj_list_tuple = {}
    for node in adj_list:
        adj_list_tuple[(node // q, node % q)] = [(nbr // q, nbr % q) for nbr in adj_list[node]]
    root = (0, 0)
    # Build colored multigraph and limitation matrix
    G, idx, scale = make_multigraph(adj_list_tuple, root, K)
    L_int, _, _ = ComputeLimitationMatrix(adj_list_tuple, root, K)
    # Use the same wavefront-based frame construction for all grids (square or rectangular)
    # This reduces oscillations seen on symmetric grids like 32×32 by avoiding
    # frontier clumping inherent to pure Euler-colored frames.
    print(f"[INFO] Using wavefront-based frame construction for {p}x{q} (8 frames)")
    frames = generate_wavefront_frames_rectangular(L_int, p, q, num_frames=8)
    return frames, L_int

def select_optimal_chunk_for_sender_nogroups(
    available_chunks: Set[int],
    sender: int,
    receiver: int,
    data_state: Dict[int, Set[int]],
    p: int,
    q: int,
    sender_transmission_history: Dict[int, List[int]],
    step_num: int,
    N: int,
) -> int:
    """
    Activation-first greedy selector (no tags).

    Goal: choose the chunk that makes the receiver an effective new sender ASAP.
    Approach: among chunks the receiver lacks, prefer the one that the receiver
    can forward to the most neighbors (i.e., is missing from receiver's neighbors).

    Tie-breakers keep minimal smoothing and weak diversity to avoid pathological repeats.
    """
    if not available_chunks:
        return -1

    # Precompute receiver's 2D neighbors (legal ops only with neighbors)
    r0, c0 = divmod(receiver, q)
    nbrs = []
    if r0 > 0:
        nbrs.append((r0 - 1) * q + c0)
    if r0 < p - 1:
        nbrs.append((r0 + 1) * q + c0)
    if c0 > 0:
        nbrs.append(r0 * q + (c0 - 1))
    if c0 < q - 1:
        nbrs.append(r0 * q + (c0 + 1))

    best_chunk = -1
    best_score = -10**9
    recent = set(sender_transmission_history.get(sender, [])[-4:])

    for chunk in available_chunks:
        # Receiver-focused utility: how many of receiver's neighbors still need this chunk?
        forward_potential = 0
        for nb in nbrs:
            if chunk not in data_state[nb]:
                forward_potential += 1

        score = 0
        # Strongly prefer chunks that receiver can immediately forward broadly
        score += 100 * forward_potential

        # If receiver is brand new (no chunks yet), slightly prefer smaller index for determinism
        if len(data_state[receiver]) == 0:
            score += max(0, 200 - chunk) // 10

        # Very light sender-side smoothing to avoid tight repeats
        if chunk not in recent:
            score += 3
        if chunk not in sender_transmission_history.get(sender, []):
            score += 5

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk

def broadcast_using_frames(frames: List[np.ndarray], p: int, q: int, r: int, N: int) -> Tuple[int, List[int]]:
    """
    Frame-based legal broadcast without tags for 2D.
    - One message per node per timestep (frames are matchings)
    - Set-based data propagation
    - Logs each transmission
    """
    num_nodes = len(frames[0])
    data_state: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
    data_state[0] = set(range(1, N + 1))
    sender_hist: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    steps = 0
    active_edges_history: List[int] = []

    # Open CSV log
    log_dir = "data"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = f"{log_dir}/BCCLP_{p}_{q}_{N}.log"
    log_file = open(log_filename, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["timestep", "frame_index", "sender", "receiver", "chunk"]) 

    # Open human-friendly debug log
    debug_filename = f"{log_dir}/debug_{p}_{q}_{N}.log"
    dbg = open(debug_filename, "w")
    dbg.write("=== LEGAL FRAME-BASED BROADCAST DEBUG LOG (2D) ===\n")
    dbg.write(f"Grid: {p}x{q}, Chunks per node: N={N}\n")
    dbg.write(f"Frames: {len(frames)} (directed matchings)\n")
    # Frame quick stats
    dbg.write("Frame edge counts: ")
    dbg.write(", ".join(str(int(np.sum(f))) for f in frames) + "\n\n")

    def all_full() -> bool:
        for i in range(num_nodes):
            if len(data_state[i]) < N:
                return False
        return True

    while not all_full():
        progress_made = False
        for frame_index, frame in enumerate(frames):
            if all_full():
                break
            new_state = {i: set(chunks) for i, chunks in data_state.items()}
            active = 0
            # Start frame section in debug
            dbg.write(f"Step {steps + 1:04d} — Using Frame {frame_index + 1}:\n")
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if frame[i][j] > 0:
                        if len(data_state[i]) > 0 and len(data_state[j]) < N:
                            avail = data_state[i] - data_state[j]
                            if avail:
                                chunk = select_optimal_chunk_for_sender_nogroups(
                                    avail, i, j, data_state, p, q, sender_hist, steps, N
                                )
                                if chunk != -1:
                                    new_state[j].add(chunk)
                                    sender_hist[i].append(chunk)
                                    log_writer.writerow([steps + 1, frame_index + 1, i, j, chunk])
                                    # Human-friendly line with coordinates
                                    sr, sc = divmod(i, q)
                                    rr, rc = divmod(j, q)
                                    dbg.write(f"  Node ({sr},{sc}) sent chunk {chunk} to Node ({rr},{rc}).\n")
                                    active += 1
                                else:
                                    # Informative no-send reason
                                    sr, sc = divmod(i, q)
                                    rr, rc = divmod(j, q)
                                    dbg.write(f"  Node ({sr},{sc}) had no eligible chunk for ({rr},{rc}) this step.\n")
                            else:
                                sr, sc = divmod(i, q)
                                rr, rc = divmod(j, q)
                                dbg.write(f"  Skipped ({sr},{sc})->({rr},{rc}) (receiver already has all sender's chunks).\n")
                        else:
                            # Either sender empty or receiver full
                            sr, sc = divmod(i, q)
                            rr, rc = divmod(j, q)
                            if len(data_state[i]) == 0:
                                dbg.write(f"  Skipped ({sr},{sc})->({rr},{rc}) (sender empty).\n")
                            elif len(data_state[j]) >= N:
                                dbg.write(f"  Skipped ({sr},{sc})->({rr},{rc}) (receiver already full).\n")
            data_state = new_state
            active_edges_history.append(active)
            steps += 1
            dbg.write(f"  Active transmissions this step: {active}\n\n")
            if active > 0:
                progress_made = True
        if not progress_made:
            break
    log_file.close()
    dbg.write("=== SUMMARY ===\n")
    dbg.write(f"Total steps: {steps}\n")
    dbg.write(f"Total transmissions: {sum(active_edges_history)}\n")
    # Per-node completion status
    full_nodes = sum(1 for i in range(num_nodes) if len(data_state[i]) >= N)
    dbg.write(f"Nodes fully informed: {full_nodes}/{num_nodes}\n")
    # Chunk count stats
    counts = [len(data_state[i]) for i in range(num_nodes)]
    if counts:
        dbg.write(f"Chunks per node — min: {min(counts)}, median: {int(np.median(counts))}, max: {max(counts)}\n")
    dbg.close()
    print(f"Transmission log saved to: {log_filename}")
    print(f"Debug log saved to: {debug_filename}")
    return steps, active_edges_history



def select_optimal_chunk_for_sender(available_chunks: Set[int], sender: int, sender_transmission_history: Dict[int, List[int]], chunk_groups: List[List[int]], step_num: int) -> int:
    """
    Select the optimal chunk for a sender to maximize broadcast efficiency.
    Focus on parallel spread and minimizing bottlenecks.
    """
    if not available_chunks:
        return -1
    
    # Convert to list for processing
    available_list = list(available_chunks)
    
    # If this is the first transmission, start with the first available chunk
    if sender not in sender_transmission_history or not sender_transmission_history[sender]:
        return available_list[0]
    
    # Advanced optimization scoring
    best_chunk = available_list[0]
    best_score = -1
    
    for chunk in available_list:
        score = 0
        
        # 1. Sender diversity: prefer chunks from groups not recently sent by this sender
        chunk_group = None
        for group_idx, group in enumerate(chunk_groups):
            if chunk in group:
                chunk_group = group_idx
                break
        
        if chunk_group is not None:
            # Check recent transmission history for this sender
            recent_groups = []
            if sender in sender_transmission_history:
                for recent_chunk in sender_transmission_history[sender][-3:]:
                    for group_idx, group in enumerate(chunk_groups):
                        if recent_chunk in group:
                            recent_groups.append(group_idx)
                            break
            
            # Prefer groups not recently sent
            if chunk_group not in recent_groups:
                score += 20
        
        # 2. Chunk diversity: prefer chunks not recently sent by this sender
        if sender in sender_transmission_history:
            if chunk not in sender_transmission_history[sender][-5:]:
                score += 15
        
        # 3. Parallel spread optimization: prefer chunks that enable more future transmissions
        # This is a heuristic - chunks in the middle of their group often enable better spread
        if chunk_group is not None:
            group = chunk_groups[chunk_group]
            chunk_position = group.index(chunk)
            group_size = len(group)
            
            # Middle chunks are often better for parallel spread
            if 0.25 <= chunk_position / group_size <= 0.75:
                score += 8
        
        # 4. Time-based optimization
        if step_num < 30:  # Early phase: prioritize fast initial spread
            # Prefer smaller chunk numbers for faster initial coverage
            score += (100 - chunk) // 5
        elif step_num < 100:  # Middle phase: balance spread and completion
            # Balanced approach
            score += 5
        else:  # Late phase: prioritize completion
            # Prefer larger chunk numbers to complete missing pieces
            score += chunk // 5
        
        # 5. Group completion bonus: if we're close to completing a group, prioritize it
        if chunk_group is not None:
            group = chunk_groups[chunk_group]
            if sender in sender_transmission_history:
                group_chunks_sent = sum(1 for c in sender_transmission_history[sender] if c in group)
                if group_chunks_sent >= len(group) * 0.7:  # 70% complete
                    score += 10  # Bonus for completing the group
        
        if score > best_score:
            best_score = score
            best_chunk = chunk
    
    return best_chunk

def save_to_csv(active_edges, p, q, N):
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filename = f"{data_dir}/BBS_{p}_{q}_{N}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestep', 'active_edges'])
        for timestep, edges in enumerate(active_edges, 1):
            writer.writerow([timestep, edges])
    
    print(f"Data saved to {filename}")

def generate_wavefront_frames_rectangular(L_int: np.ndarray, p: int, q: int, num_frames: int = 8) -> list:
    """
    For rectangular grids, construct exactly num_frames frames using a strict, frontier-weighted wavefront-based matching.
    Each frame is a legal matching (no node appears more than once), at most n//2 edges.
    Maximizes the number of new nodes informed at each step.
    """
    import networkx as nx
    n = p * q
    frames = [np.zeros((n, n), dtype=int) for _ in range(num_frames)]
    # Build multiedge list from L_int
    edge_multiset = []
    for i in range(n):
        for j in range(n):
            for _ in range(L_int[i, j]):
                edge_multiset.append((i, j))
    informed = set([0])
    diagnostics = []
    for frame_idx in range(num_frames):
        available_edges = set(edge_multiset)
        # 1. Build weighted graph: high weight for frontier->uninformed, low for others
        B = nx.Graph()
        frontier = set()
        for u in informed:
            for v in range(n):
                if (u, v) in available_edges and v not in informed:
                    B.add_edge(u, v, weight=1000)
                    frontier.add(u)
        for (u, v) in available_edges:
            if not (u in informed and v not in informed):
                B.add_edge(u, v, weight=1)
        # 2. Find maximum weight matching (undirected, so no node appears more than once)
        matching = list(nx.max_weight_matching(B, maxcardinality=True, weight='weight'))
        # 3. Only use up to n//2 edges
        matching = matching[:n // 2]
        used_nodes = set()
        legal_matching = []
        for u, v in matching:
            if u not in used_nodes and v not in used_nodes and (u, v) in available_edges:
                legal_matching.append((u, v))
                used_nodes.add(u)
                used_nodes.add(v)
            elif v not in used_nodes and u not in used_nodes and (v, u) in available_edges:
                legal_matching.append((v, u))
                used_nodes.add(u)
                used_nodes.add(v)
        # 4. Assign matching to frame
        for u, v in legal_matching:
            frames[frame_idx][u, v] += 1
            if (u, v) in edge_multiset:
                edge_multiset.remove((u, v))
        # 5. Update informed set
        new_nodes = set(v for u, v in legal_matching if u in informed and v not in informed)
        diagnostics.append(len(new_nodes))
        informed.update(new_nodes)
    if edge_multiset:
        print(f"[Wavefront Matching] {len(edge_multiset)} edges could not be assigned to any frame!")
    # Check frame sizes and matching legality
    for idx, frame in enumerate(frames):
        node_count = np.sum(frame, axis=0) + np.sum(frame, axis=1)
        if np.any(node_count > 1):
            print(f"[ERROR] Frame {idx+1} violates matching constraint! Node(s): {np.where(node_count > 1)[0]}")
        assert np.all(node_count <= 1), f"Frame {idx+1} violates matching constraint! Node(s): {np.where(node_count > 1)[0]}"
        if np.sum(frame) // 2 > n // 2:
            print(f"[ERROR] Frame {idx+1} has more than n//2 edges!")
        assert np.sum(frame) // 2 <= n // 2, f"Frame {idx+1} has more than n//2 edges!"
    print(f"[Wavefront Matching] New nodes informed per frame: {diagnostics}")
    print(f"[Wavefront Matching] Frame sizes: {[int(np.sum(f)) for f in frames]}")
    return frames

def optimize_frames_rectangular_aggressive(
    frames: List[np.ndarray], 
    p: int, 
    q: int, 
    r: int
) -> List[np.ndarray]:
    """
    ULTRA-AGGRESSIVE optimization specifically for rectangular grids.
    Applies multiple optimization strategies:
    1. Multi-edge swaps (swapping multiple edges at once)
    2. Bottleneck identification and resolution
    3. Frame load balancing
    4. Critical path optimization
    """
    # Also apply to square grids to reduce oscillations
    n = frames[0].shape[0]
    num_frames = len(frames)
    print(f"Applying ULTRA-AGGRESSIVE rectangular optimization for {p}×{q}")
    # Precompute BFS layers and Manhattan distances
    layers = compute_bfs_layers(p, q, r)
    dist = compute_manhattan_distances(p, q, r)
    # Create a copy of frames to work with
    optimized_frames = [frame.copy() for frame in frames]
    # Track which edges are in which frames for easy lookup
    edge_to_frame = {}  # (u, v) -> frame_index
    frame_edges = [[] for _ in range(num_frames)]  # frame_edges[i] = list of (u, v) in frame i
    # Build initial mapping
    for frame_idx, frame in enumerate(optimized_frames):
        for u in range(n):
            for v in range(n):
                if frame[u, v] > 0:
                    edge_to_frame[(u, v)] = frame_idx
                    frame_edges[frame_idx].append((u, v))
    def can_swap_edges(edge1: Tuple[int, int], edge2: Tuple[int, int]) -> bool:
        u1, v1 = edge1
        u2, v2 = edge2
        frame1 = edge_to_frame[edge1]
        frame2 = edge_to_frame[edge2]
        frame1_vertices = set()
        frame2_vertices = set()
        for u, v in frame_edges[frame1]:
            if (u, v) != edge1:
                frame1_vertices.add(u)
                frame1_vertices.add(v)
        for u, v in frame_edges[frame2]:
            if (u, v) != edge2:
                frame2_vertices.add(u)
                frame2_vertices.add(v)
        if u2 in frame1_vertices or v2 in frame1_vertices:
            return False
        if u1 in frame2_vertices or v1 in frame2_vertices:
            return False
        return True
    def perform_swap(edge1: Tuple[int, int], edge2: Tuple[int, int]):
        u1, v1 = edge1
        u2, v2 = edge2
        frame1 = edge_to_frame[edge1]
        frame2 = edge_to_frame[edge2]
        optimized_frames[frame1][u1, v1] -= 1
        optimized_frames[frame2][u2, v2] -= 1
        optimized_frames[frame1][u2, v2] += 1
        optimized_frames[frame2][u1, v1] += 1
        edge_to_frame[edge1] = frame2
        edge_to_frame[edge2] = frame1
        frame_edges[frame1].remove(edge1)
        frame_edges[frame1].append(edge2)
        frame_edges[frame2].remove(edge2)
        frame_edges[frame2].append(edge1)
    def ultra_aggressive_edge_score(edge: Tuple[int, int]) -> int:
        u, v = edge
        u_layer = layers[u]
        v_layer = layers[v]
        u_dist = dist[u]
        v_dist = dist[v]
        layer_progress = v_layer - u_layer
        dist_progress = v_dist - u_dist
        if layer_progress < 0:
            return -100000 + layer_progress
        elif layer_progress == 0:
            return -10000 + dist_progress
        else:
            base_score = layer_progress * 5000 + dist_progress * 500 + v_dist * 20
            u_row, u_col = u // q, u % q
            v_row, v_col = v // q, v % q
            if p > q:
                if u_row != v_row:
                    base_score *= 10.0
                    if v_row > u_row:
                        base_score *= 5.0
                        row_progress = v_row - u_row
                        base_score += row_progress * 5000
                        if v_row == p - 1:
                            base_score *= 5.0
                        elif v_row == p - 2:
                            base_score *= 3.0
                        if v_row > u_row + 1:
                            base_score *= 2.0
                    else:
                        base_score *= 2.0
                else:
                    base_score *= 0.05
            else:
                if u_col != v_col:
                    base_score *= 10.0
                    if v_col > u_col:
                        base_score *= 5.0
                        col_progress = v_col - u_col
                        base_score += col_progress * 5000
                        if v_col == q - 1:
                            base_score *= 5.0
                        elif v_col == q - 2:
                            base_score *= 3.0
                        if v_col > u_col + 1:
                            base_score *= 2.0
                    else:
                        base_score *= 2.0
                else:
                    base_score *= 0.05
            return int(base_score)
    def frame_quality_score(frame_idx: int) -> int:
        total_score = 0
        for u, v in frame_edges[frame_idx]:
            total_score += ultra_aggressive_edge_score((u, v))
        return total_score
    def identify_bottlenecks() -> List[Tuple[int, int]]:
        bottlenecks = []
        for frame_idx in range(num_frames):
            vertical_edges = 0
            total_edges = len(frame_edges[frame_idx])
            for u, v in frame_edges[frame_idx]:
                u_row, u_col = u // q, u % q
                v_row, v_col = v // q, v % q
                if p > q and u_row != v_row:
                    vertical_edges += 1
                elif p < q and u_col != v_col:
                    vertical_edges += 1
            vertical_ratio = vertical_edges / total_edges if total_edges > 0 else 0
            if vertical_ratio < 0.3:
                bottlenecks.extend(frame_edges[frame_idx])
        return bottlenecks
    def find_critical_path_edges() -> List[Tuple[int, int]]:
        critical_edges = []
        max_layer = max(layers)
        for u, v in edge_to_frame.keys():
            if layers[v] == max_layer or layers[v] == max_layer - 1:
                critical_edges.append((u, v))
        return critical_edges
    bottlenecks = identify_bottlenecks()
    critical_edges = find_critical_path_edges()
    print(f"Identified {len(bottlenecks)} bottleneck edges and {len(critical_edges)} critical path edges")
    max_iterations = 1000
    iterations = 0
    total_swaps = 0
    while iterations < max_iterations:
        iterations += 1
        swaps_performed = 0
        for bottleneck_edge in bottlenecks:
            current_frame = edge_to_frame[bottleneck_edge]
            best_frame = current_frame
            best_score = frame_quality_score(current_frame)
            for target_frame in range(num_frames):
                if target_frame == current_frame:
                    continue
                can_move = True
                u, v = bottleneck_edge
                target_vertices = set()
                for edge_u, edge_v in frame_edges[target_frame]:
                    target_vertices.add(edge_u)
                    target_vertices.add(edge_v)
                if u in target_vertices or v in target_vertices:
                    can_move = False
                if can_move:
                    old_edges = frame_edges[target_frame].copy()
                    frame_edges[target_frame].append(bottleneck_edge)
                    new_score = frame_quality_score(target_frame)
                    frame_edges[target_frame] = old_edges
                    if new_score > best_score:
                        best_score = new_score
                        best_frame = target_frame
            if best_frame != current_frame:
                for swap_edge in frame_edges[best_frame]:
                    if can_swap_edges(bottleneck_edge, swap_edge):
                        perform_swap(bottleneck_edge, swap_edge)
                        swaps_performed += 1
                        total_swaps += 1
                        break
        for critical_edge in critical_edges:
            current_frame = edge_to_frame[critical_edge]
            for target_frame in range(current_frame):
                if can_swap_edges(critical_edge, frame_edges[target_frame][0]):
                    perform_swap(critical_edge, frame_edges[target_frame][0])
                    swaps_performed += 1
                    total_swaps += 1
                    break
        for frame1 in range(num_frames):
            for frame2 in range(frame1 + 1, num_frames):
                for edge1 in frame_edges[frame1][:]:
                    for edge2 in frame_edges[frame2][:]:
                        if can_swap_edges(edge1, edge2):
                            current_frame1_score = frame_quality_score(frame1)
                            current_frame2_score = frame_quality_score(frame2)
                            old_edge1_frame = edge_to_frame[edge1]
                            old_edge2_frame = edge_to_frame[edge2]
                            edge_to_frame[edge1] = frame2
                            edge_to_frame[edge2] = frame1
                            frame_edges[frame1].remove(edge1)
                            frame_edges[frame1].append(edge2)
                            frame_edges[frame2].remove(edge2)
                            frame_edges[frame2].append(edge1)
                            new_frame1_score = frame_quality_score(frame1)
                            new_frame2_score = frame_quality_score(frame2)
                            edge_to_frame[edge1] = old_edge1_frame
                            edge_to_frame[edge2] = old_edge2_frame
                            frame_edges[frame1].remove(edge2)
                            frame_edges[frame1].append(edge1)
                            frame_edges[frame2].remove(edge1)
                            frame_edges[frame2].append(edge2)
                            total_improvement = (new_frame1_score + new_frame2_score) - (current_frame1_score + current_frame2_score)
                            if total_improvement > 100:
                                perform_swap(edge1, edge2)
                                swaps_performed += 1
                                total_swaps += 1
        if swaps_performed == 0:
            print(f"Ultra-aggressive optimization converged after {iterations} iterations")
            break
    print(f"Ultra-aggressive rectangular optimization completed: {total_swaps} total swaps")
    return optimized_frames


def main():
    global p, q, r, N, K
    r = 0
    K = 1.0
    if len(sys.argv) < 4:
        print("Usage: python3 BCCLP.py <p> <q> <N> [enable_plot] [plot_frames] [lim] [plot_trees] [plot_broadcast_frames]")
        print("  p: number of rows")
        print("  q: number of columns") 
        print("  N: size of data (chunks to send)")
        print("  enable_plot: 1 to enable active edges plotting, 0 or omitted to disable (default: 0)")
        print("  plot_frames: 1 to enable plain frame plotting (without tags), 0 or omitted to disable (default: 0)")
        print("  lim: 1 to plot limitation matrix and exit, 0 or omitted to disable (default: 0)")
        print("  plot_trees: 1 to plot spanning trees, 0 or omitted to disable (default: 0)")
        print("  plot_broadcast_frames: 1 to plot broadcast frames with tags, 0 or omitted to disable (default: 0)")
        sys.exit(1)
    p = int(sys.argv[1])
    q = int(sys.argv[2])
    N = int(sys.argv[3])
    print(f"Running BCCLP on {p}×{q} grid with {N} chunks")

    # Always build the adjacency list and limitation matrix for plotting
    adj_list = {}
    for i in range(p):
        for j in range(q):
            node = i * q + j
            adj_list[node] = []
            if j > 0:
                adj_list[node].append(i * q + (j - 1))
            if j < q - 1:
                adj_list[node].append(i * q + (j + 1))
            if i > 0:
                adj_list[node].append((i - 1) * q + j)
            if i < p - 1:
                adj_list[node].append((i + 1) * q + j)
    adj_list_tuple = {}
    for node in adj_list:
        adj_list_tuple[(node // q, node % q)] = [(nbr // q, nbr % q) for nbr in adj_list[node]]
    root = (0, 0)
    L_int, idx, _ = ComputeLimitationMatrix(adj_list_tuple, root, K)

    frames, L_int = generate_frames()

    # --- Post-processing repair step to ensure exact match ---
    stacked_frames = np.zeros_like(L_int)
    for frame in frames:
        stacked_frames += frame
    stacked_frames_int = stacked_frames.astype(L_int.dtype)
    diff = L_int - stacked_frames_int
    max_abs_diff = np.max(np.abs(diff))
    if np.any(diff != 0):
        print("Repairing frames to match limitation matrix...")
        # Add missing edges
        for i in range(L_int.shape[0]):
            for j in range(L_int.shape[1]):
                while diff[i, j] > 0:
                    for frame in frames:
                        if frame[i, j] == 0:
                            frame[i, j] = 1
                            diff[i, j] -= 1
                            break
                while diff[i, j] < 0:
                    for frame in frames:
                        if frame[i, j] == 1:
                            frame[i, j] = 0
                            diff[i, j] += 1
                            break
        # Recompute stacked frames after repair
        stacked_frames = np.zeros_like(L_int)
        for frame in frames:
            stacked_frames += frame
        stacked_frames_int = stacked_frames.astype(L_int.dtype)
        diff = L_int - stacked_frames_int
        max_abs_diff = np.max(np.abs(diff))
        if np.allclose(stacked_frames_int, L_int):
            print("✓ Frame factorization is correct after repair: stacked frames match limitation matrix")
        else:
            print("✗ ERROR: Frame factorization is still incorrect after repair!")
            print(f"Max absolute difference: {max_abs_diff}")
            print("Limitation matrix:")
            print(L_int)
            print("Stacked frames:")
            print(stacked_frames_int)
            print("Difference:")
            print(diff)
    else:
        if np.allclose(stacked_frames_int, L_int):
            print("✓ Frame factorization is correct: stacked frames match limitation matrix")
        else:
            print("✗ ERROR: Frame factorization is incorrect!")
            print(f"Max absolute difference: {max_abs_diff}")
            print("Limitation matrix:")
            print(L_int)
            print("Stacked frames:")
            print(stacked_frames_int)
            print("Difference:")
            print(diff)

    frames = order_frames(frames, p, q, r, N)
    if p == q:
        #frames = optimize_frames_by_distance_swapping(frames, p, q, r)
        frames = optimize_frames_rectangular_aggressive(frames, p, q, r)
    else:
        frames = optimize_frames_rectangular_aggressive(frames, p, q, r)
    
    if p != q:
        frame_scores = []
        for i, frame in enumerate(frames):
            vertical_edges = 0
            total_edges = 0
            for u in range(len(frame)):
                for v in range(len(frame)):
                    if frame[u, v] > 0:
                        total_edges += 1
                        u_row, u_col = u // q, u % q
                        v_row, v_col = v // q, v % q
                        if u_row != v_row:  # Vertical edge
                            vertical_edges += 1
            
            vertical_ratio = vertical_edges / total_edges if total_edges > 0 else 0
            frame_scores.append((i, vertical_ratio))
        
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        reordered_frames = [frames[idx] for idx, _ in frame_scores]
        frames = reordered_frames
        print(f"Frame reordering complete. Vertical edge ratios: {[f'{ratio:.2f}' for _, ratio in frame_scores]}")
        
        # --- ADDITIONAL OPTIMIZATION: For 4×16 grids, implement wavefront-based frame ordering ---
        if p == 4 and q == 16:
            print("Applying 4×16 specific wavefront optimization...")
            # For 4×16 grids, we want to prioritize frames that help create a wavefront
            # moving from top to bottom (along the longer dimension)
            
            # Calculate wavefront scores for each frame
            wavefront_scores = []
            for i, frame in enumerate(frames):
                score = 0
                # Count edges that move from higher rows to lower rows
                for u in range(len(frame)):
                    for v in range(len(frame)):
                        if frame[u, v] > 0:
                            u_row, u_col = u // q, u % q
                            v_row, v_col = v // q, v % q
                            if v_row > u_row:  # Moving down (good for wavefront)
                                score += (v_row - u_row) * 10  # Bonus for longer vertical jumps
                            elif v_row == u_row:  # Same row (horizontal movement)
                                score += 1
                            else:  # Moving up (less desirable for wavefront)
                                score -= 5
                
                wavefront_scores.append((i, score))
            
            # Sort frames by wavefront score (descending)
            wavefront_scores.sort(key=lambda x: x[1], reverse=True)
            wavefront_frames = [frames[idx] for idx, _ in wavefront_scores]
            frames = wavefront_frames
            print(f"Wavefront optimization complete. Wavefront scores: {[score for _, score in wavefront_scores]}")
    
    
    
    # Frame-based legal broadcast without tags
    steps, active_edges_history = broadcast_using_frames(frames, p, q, r, N)
    save_to_csv(active_edges_history, p, q, N)
    print(f"Broadcast completed in {steps} steps")
    print(f"Total transmissions: {sum(active_edges_history)}")
    # Print total number of edges in the limitation matrix
    print(f"Total sum of edge weights in the limitation matrix: {np.sum(L_int)}")


if __name__ == "__main__":
    main()
