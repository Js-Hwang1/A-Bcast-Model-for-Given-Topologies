#!/usr/bin/env python3
"""
visualize_trees.py — Visualize BBS spanning trees on a 2D mesh.

Generates a 6x6 mesh, runs the BBS tree-building algorithm (Python port
of encode.c's bbs_compute_trees), and draws each tree on the mesh grid.

Usage:
    python3 BBS/visualize_trees.py [--rows 6] [--cols 6] [--root 0]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque, Counter


# ── Build 2D mesh topology data ──────────────────────────────────────

def build_mesh(rows, cols):
    """Build adjacency, latency, flink, and path data for a 2D mesh.

    Mirrors what topo_preprocess.py produces: Floyd-Warshall over the
    mesh graph, storing latency, first-link, next-hop, and full path
    link arrays — exactly the data that encode.c expects.

    Returns:
        N          : number of nodes
        adj        : NxN bool adjacency (1-hop neighbors)
        lat        : NxN float latency matrix (hop count as proxy)
        flink      : NxN int first-link ID matrix (0 = self/no link)
        path_len   : NxN int number of physical links per path
        path_links : NxN x max_hops int full path link IDs
        max_hops   : int maximum path length
        pos        : dict node -> (col, row) for plotting
        link_id    : dict (a,b) -> unique link ID (1-based, a<b)
    """
    N = rows * cols

    # Assign grid positions: node i = (row i//cols, col i%cols)
    pos = {}
    for i in range(N):
        r, c = divmod(i, cols)
        pos[i] = (c, rows - 1 - r)  # (x, y) with y flipped for display

    # Build edges and assign link IDs
    edges = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            # right neighbor
            if c + 1 < cols:
                edges.append((node, node + 1))
            # down neighbor
            if r + 1 < rows:
                edges.append((node, (r + 1) * cols + c))

    link_id = {}
    for idx, (a, b) in enumerate(edges):
        lo, hi = min(a, b), max(a, b)
        link_id[(lo, hi)] = idx + 1  # 1-based

    # Adjacency
    adj = np.zeros((N, N), dtype=bool)
    for a, b in edges:
        adj[a, b] = True
        adj[b, a] = True

    # Floyd-Warshall for latency + first-link + next-hop
    INF = float('inf')
    lat = np.full((N, N), INF)
    flink = np.zeros((N, N), dtype=int)
    next_hop = np.full((N, N), -1, dtype=int)
    np.fill_diagonal(lat, 0.0)

    # Direct-edge link IDs for path reconstruction
    direct_links = {}  # (a, b) -> [link_ids]

    for a, b in edges:
        lo, hi = min(a, b), max(a, b)
        lid = link_id[(lo, hi)]
        lat[a, b] = 1.0
        lat[b, a] = 1.0
        flink[a, b] = lid
        flink[b, a] = lid
        next_hop[a, b] = b
        next_hop[b, a] = a
        direct_links[(a, b)] = [lid]
        direct_links[(b, a)] = [lid]

    for k in range(N):
        for i in range(N):
            for j in range(N):
                if lat[i, k] + lat[k, j] < lat[i, j]:
                    lat[i, j] = lat[i, k] + lat[k, j]
                    flink[i, j] = flink[i, k]
                    next_hop[i, j] = next_hop[i, k]

    # Reconstruct full paths (same logic as topo_preprocess.py)
    all_paths = {}
    for ci in range(N):
        for cj in range(N):
            if ci == cj:
                all_paths[(ci, cj)] = []
                continue
            path_lids = []
            current = ci
            seen = set()
            while current != cj and current >= 0:
                if current in seen:
                    break
                seen.add(current)
                nxt = next_hop[current, cj]
                if nxt < 0:
                    break
                key = (current, nxt)
                if key in direct_links:
                    path_lids.extend(direct_links[key])
                current = nxt
            all_paths[(ci, cj)] = path_lids

    max_hops = max(len(p) for p in all_paths.values()) if all_paths else 1
    if max_hops == 0:
        max_hops = 1

    path_len = np.zeros((N, N), dtype=int)
    path_links = np.zeros((N, N, max_hops), dtype=int)
    for (ci, cj), lids in all_paths.items():
        path_len[ci, cj] = len(lids)
        for k, lid in enumerate(lids):
            path_links[ci, cj, k] = lid

    return N, adj, lat, flink, path_len, path_links, max_hops, pos, link_id


# ── BBS tree computation (port of encode.c) ──────────────────────────

def tree_depth(parent, N):
    maxd = 0
    for i in range(N):
        d = 0
        v = i
        while parent[v] >= 0:
            v = parent[v]
            d += 1
            if d > N:
                return N
        maxd = max(maxd, d)
    return maxd


BBS_MAX_TREES = 8


def bbs_compute_trees(N, adj, lat, flink, path_len, path_links, max_hops,
                      root):
    """Build spanning trees via round-robin BFS with a shared edge pool.

    Algorithm:
        tau = 3 trees, each edge may be used by at most max_usage = 2 trees.
        All trees grow simultaneously: in round-robin order, each tree
        pops one node from its BFS frontier and expands all available
        neighbors (edges with usage < max_usage). Edge usage is updated
        globally after each expansion, so all trees see the same pool.

    Fully deterministic — no sorting, no randomness.

    Returns:
        tau     : number of trees
        trees   : list of parent arrays (each length N, -1 = root)
    """
    # ---- 1-hop threshold ----
    lat_1hop = 1e30
    for i in range(N):
        for j in range(N):
            l = lat[i, j]
            if l > 0.0 and l < lat_1hop:
                lat_1hop = l
    lat_thresh = lat_1hop * 1.01

    # ---- Build 1-hop adjacency matrix ----
    adj_1hop = np.zeros((N, N), dtype=bool)
    total_edges = 0
    for i in range(N):
        for j in range(i + 1, N):
            l = lat[i, j]
            if l > 0.0 and l <= lat_thresh:
                adj_1hop[i, j] = True
                adj_1hop[j, i] = True
                total_edges += 1

    # ---- Root's 1-hop neighbors ----
    root_nbrs = []
    for v in range(N):
        if adj_1hop[root, v]:
            root_nbrs.append(v)
    root_nn = len(root_nbrs)

    # ---- Compute baseline single-tree BFS depth ----
    visited = [False] * N
    queue = [0] * N
    t0_bfs = [-1] * N
    visited[root] = True
    qh, qt = 0, 0
    queue[qt] = root
    qt += 1
    while qh < qt:
        u = queue[qh]
        qh += 1
        for v in range(N):
            if visited[v] or not adj_1hop[u, v]:
                continue
            visited[v] = True
            t0_bfs[v] = u
            queue[qt] = v
            qt += 1
    depth0 = tree_depth(t0_bfs, N)

    # ---- Parameters ----
    tau = 3
    max_usage = 2

    # ---- Find max flink ID ----
    max_fl = int(np.max(flink)) + 2

    print(f"BBS: edges={total_edges} root_deg={root_nn} "
          f"tau={tau} max_usage={max_usage} "
          f"baseline_depth={depth0}")

    if root_nn < 1:
        print("BBS: isolated root, returning single BFS tree")
        return 1, [t0_bfs]

    # ---- Global edge pool ----
    edge_usage = [0] * max_fl

    # ---- Identify low-degree nodes (corners) — only ingoing allowed ----
    # For nodes with degree <= 2, their edge budget is too tight to be
    # parents of other nodes. Only allow them as children (ingoing edges).
    node_degree = [0] * N
    for i in range(N):
        for j in range(N):
            if adj_1hop[i, j]:
                node_degree[i] += 1
    # Nodes that cannot serve as parents (except root):
    # Corner nodes (degree <= 2) have tight edge budgets — all their
    # edge capacity is needed for their own parent assignments.
    no_parent = set()
    for i in range(N):
        if i == root:
            continue
        if node_degree[i] <= 2:
            no_parent.add(i)

    # ---- Compute BFS order from root ----
    bfs_order = []
    bfs_vis = [False] * N
    bfs_vis[root] = True
    bfs_q = deque([root])
    while bfs_q:
        u = bfs_q.popleft()
        bfs_order.append(u)
        for v in range(N):
            if not bfs_vis[v] and adj_1hop[u, v]:
                bfs_vis[v] = True
                bfs_q.append(v)

    # ---- Assign parents with constraint propagation ----
    # For each node in BFS order, assign a parent in each of the 3 trees.
    # The starting tree rotates per node (round-robin fairness).
    #
    # Each pass has two phases:
    #   Phase A (propagation): repeatedly assign pairs that have exactly
    #     one viable parent — these are forced and must be done first to
    #     avoid greedy choices consuming critical edges.
    #   Phase B (greedy): for pairs with 2+ options, pick lowest-usage
    #     edge, tie-break by lowest node ID.
    # Deferred pairs (no options yet) are retried in the next pass.
    trees = [[-1] * N for _ in range(tau)]

    # Build initial work list with round-robin tree ordering per node
    unassigned = []
    node_idx = 0
    for v in bfs_order:
        if v == root:
            continue
        start_t = node_idx % tau
        for offset in range(tau):
            t = (start_t + offset) % tau
            unassigned.append((v, t))
        node_idx += 1

    def find_options(v, t):
        """Return list of (parent, edge_usage) for viable parents."""
        opts = []
        for u in range(N):
            if not adj_1hop[v, u]:
                continue
            if u != root and trees[t][u] < 0:
                continue
            if u in no_parent:
                continue  # low-degree nodes can't be parents
            fl = int(flink[v, u])
            if 0 < fl < max_fl and edge_usage[fl] >= max_usage:
                continue
            usage = edge_usage[fl] if 0 < fl < max_fl else 0
            opts.append((u, usage))
        return opts

    def assign(v, t, parent):
        trees[t][v] = parent
        fl = int(flink[v, parent])
        if 0 < fl < max_fl:
            edge_usage[fl] += 1

    for pass_num in range(N):
        if not unassigned:
            break

        # Phase A: constraint propagation — assign forced pairs
        changed = True
        while changed:
            changed = False
            remaining = []
            for v, t in unassigned:
                opts = find_options(v, t)
                if len(opts) == 1:
                    assign(v, t, opts[0][0])
                    changed = True
                elif len(opts) == 0:
                    remaining.append((v, t))
                else:
                    remaining.append((v, t))
            unassigned = remaining

        # Phase B: greedy assignment for pairs with 2+ options
        new_unassigned = []
        progress = False
        for v, t in unassigned:
            opts = find_options(v, t)
            if opts:
                # Pick lowest usage, then lowest node ID
                best = min(opts, key=lambda x: (x[1], x[0]))
                assign(v, t, best[0])
                progress = True
            else:
                new_unassigned.append((v, t))

        if not progress and not new_unassigned != unassigned:
            # Check if propagation alone made progress
            if len(new_unassigned) == len(unassigned):
                break
        unassigned = new_unassigned

    # ---- Repair step: reparent to free saturated edges ----
    # Two-level repair: first try direct reparenting (1-deep),
    # then try chain reparenting (2-deep: free an edge for a victim
    # by reparenting one of victim's neighbors first).
    for repair_depth in range(2):
        if not unassigned:
            break
        still_unassigned = []
        for v, t in unassigned:
            resolved = False
            for u in range(N):
                if resolved:
                    break
                if not adj_1hop[v, u]:
                    continue
                fl = int(flink[v, u])
                if fl <= 0 or fl >= max_fl or edge_usage[fl] < max_usage:
                    continue
                if not (u == root or trees[t][u] >= 0):
                    continue
                for t2 in range(tau):
                    if t2 == t or resolved:
                        continue
                    victims = []
                    for w in range(N):
                        if trees[t2][w] < 0:
                            continue
                        if int(flink[w, trees[t2][w]]) == fl:
                            victims.append(w)
                    for victim in victims:
                        if resolved:
                            break
                        old_parent = trees[t2][victim]
                        # Try direct reparent
                        for alt in range(N):
                            if alt == old_parent or not adj_1hop[victim, alt]:
                                continue
                            if alt != root and trees[t2][alt] < 0:
                                continue
                            is_desc = False
                            a = alt
                            while a >= 0 and a != root:
                                if a == victim:
                                    is_desc = True
                                    break
                                a = trees[t2][a]
                            if is_desc:
                                continue
                            alt_fl = int(flink[victim, alt])
                            if 0 < alt_fl < max_fl and edge_usage[alt_fl] >= max_usage:
                                # Depth-2: try freeing alt_fl first
                                if repair_depth < 1:
                                    continue
                                # Find who uses alt_fl in another tree
                                freed = False
                                for t3 in range(tau):
                                    if t3 == t2 or freed:
                                        continue
                                    for w2 in range(N):
                                        if freed:
                                            break
                                        if trees[t3][w2] < 0:
                                            continue
                                        if int(flink[w2, trees[t3][w2]]) != alt_fl:
                                            continue
                                        old_p2 = trees[t3][w2]
                                        for alt2 in range(N):
                                            if alt2 == old_p2 or not adj_1hop[w2, alt2]:
                                                continue
                                            if alt2 != root and trees[t3][alt2] < 0:
                                                continue
                                            is_d2 = False
                                            a2 = alt2
                                            while a2 >= 0 and a2 != root:
                                                if a2 == w2:
                                                    is_d2 = True
                                                    break
                                                a2 = trees[t3][a2]
                                            if is_d2:
                                                continue
                                            alt2_fl = int(flink[w2, alt2])
                                            if 0 < alt2_fl < max_fl and edge_usage[alt2_fl] >= max_usage:
                                                continue
                                            # Chain reparent: w2 in t3, then victim in t2
                                            trees[t3][w2] = alt2
                                            edge_usage[alt_fl] -= 1
                                            if 0 < alt2_fl < max_fl:
                                                edge_usage[alt2_fl] += 1
                                            freed = True
                                            break
                                if not freed:
                                    continue
                            # Now alt_fl has budget; reparent victim
                            trees[t2][victim] = alt
                            edge_usage[fl] -= 1
                            if 0 < alt_fl < max_fl:
                                edge_usage[alt_fl] += 1
                            trees[t][v] = u
                            edge_usage[fl] += 1
                            resolved = True
                            break
            if not resolved:
                still_unassigned.append((v, t))
        unassigned = still_unassigned

    if unassigned:
        print(f"BBS: WARNING: {len(unassigned)} unresolved (node,tree) pairs")
        for v, t in unassigned:
            nbrs = [u for u in range(N) if adj_1hop[v, u]]
            print(f"  stuck: node={v} tree={t} neighbors={nbrs}")

    # ---- Per-tree stats ----
    for t in range(tau):
        tree_lids = set()
        for i in range(N):
            if trees[t][i] < 0:
                continue
            fl = int(flink[trees[t][i], i])
            if 0 < fl < max_fl:
                tree_lids.add(fl)
        excl = sum(1 for fl in tree_lids if edge_usage[fl] == 1)
        shared = len(tree_lids) - excl
        d = tree_depth(trees[t], N)
        rc = sum(1 for i in range(N) if trees[t][i] == root)
        print(f"BBS:   T{t}: depth={d} rc={rc} "
              f"exclusive={excl} shared={shared}")

    # ---- Link sharing distribution ----
    sharing_dist = Counter(c for c in edge_usage if c > 0)
    max_sharing = max(edge_usage) if max_fl > 1 else 0
    print(f"BBS: Link sharing: {dict(sorted(sharing_dist.items()))} "
          f"max={max_sharing}")

    # ---- Compute depths ----
    max_depth = max(tree_depth(trees[t], N) for t in range(tau))

    # ---- Count flink collisions between tree pairs ----
    flink_collisions = 0
    for a in range(tau):
        for b in range(a + 1, tau):
            ta = trees[a]
            tb = trees[b]
            for i in range(N):
                if ta[i] < 0:
                    continue
                fl_a = flink[ta[i], i]
                if fl_a == 0:
                    continue
                for j in range(N):
                    if tb[j] < 0:
                        continue
                    fl_b = flink[tb[j], j]
                    if fl_a == fl_b:
                        flink_collisions += 1
                        break

    print(f"BBS: tau={tau} max_depth={max_depth} baseline={depth0} "
          f"flink_collisions={flink_collisions}")

    # ---- Verify all trees are spanning ----
    for t in range(tau):
        covered = sum(1 for i in range(N) if trees[t][i] >= 0 or i == root)
        if covered != N:
            print(f"WARNING: Tree {t} covers only {covered}/{N} nodes!")

    return tau, trees


# ── Visualization ─────────────────────────────────────────────────────

# Qualitative colors for trees
TREE_COLORS = [
    '#e6194b',  # red
    '#3cb44b',  # green
    '#4363d8',  # blue
    '#f58231',  # orange
    '#911eb4',  # purple
    '#42d4f4',  # cyan
    '#f032e6',  # magenta
    '#bfef45',  # lime
]


def draw_tree(ax, tree, N, pos, root, color, label, rows, cols,
              draw_mesh=True):
    """Draw a single spanning tree on mesh grid."""
    ax.set_aspect('equal')
    ax.set_title(label, fontsize=13, fontweight='bold')

    # Draw mesh edges (light gray background)
    if draw_mesh:
        for i in range(N):
            for j in range(i + 1, N):
                ri, ci = divmod(i, cols)
                rj, cj = divmod(j, cols)
                if abs(ri - rj) + abs(ci - cj) == 1:
                    xi, yi = pos[i]
                    xj, yj = pos[j]
                    ax.plot([xi, xj], [yi, yj], '-',
                            color='#e0e0e0', linewidth=1.0, zorder=1)

    # Draw tree edges with arrows
    for i in range(N):
        p = tree[i]
        if p < 0:
            continue
        xp, yp = pos[p]
        xi, yi = pos[i]
        dx = xi - xp
        dy = yi - yp
        ax.annotate('', xy=(xi, yi), xytext=(xp, yp),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=2.2, shrinkA=6, shrinkB=6),
                    zorder=3)

    # Draw nodes
    for i in range(N):
        x, y = pos[i]
        if i == root:
            ax.plot(x, y, 'o', markersize=14, color=color,
                    markeredgecolor='black', markeredgewidth=2, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white', zorder=6)
        else:
            ax.plot(x, y, 'o', markersize=10, color='white',
                    markeredgecolor=color, markeredgewidth=1.5, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=6, color='#333333', zorder=6)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_overlay(ax, trees, N, pos, root, rows, cols):
    """Draw all trees overlaid with different colors + offsets."""
    ax.set_aspect('equal')
    ax.set_title('All Trees Overlaid', fontsize=13, fontweight='bold')
    tau = len(trees)

    # Draw mesh edges
    for i in range(N):
        for j in range(i + 1, N):
            ri, ci = divmod(i, cols)
            rj, cj = divmod(j, cols)
            if abs(ri - rj) + abs(ci - cj) == 1:
                xi, yi = pos[i]
                xj, yj = pos[j]
                ax.plot([xi, xj], [yi, yj], '-',
                        color='#e0e0e0', linewidth=1.0, zorder=1)

    # Offset each tree slightly so edges don't overlap
    offsets = []
    for t in range(tau):
        angle = 2 * np.pi * t / tau
        offsets.append((0.06 * np.cos(angle), 0.06 * np.sin(angle)))

    for t, tree in enumerate(trees):
        color = TREE_COLORS[t % len(TREE_COLORS)]
        ox, oy = offsets[t]
        for i in range(N):
            p = tree[i]
            if p < 0:
                continue
            xp, yp = pos[p]
            xi, yi = pos[i]
            ax.annotate('', xy=(xi + ox, yi + oy),
                        xytext=(xp + ox, yp + oy),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.8, shrinkA=5, shrinkB=5,
                                        alpha=0.8),
                        zorder=3)

    # Draw nodes (no offset)
    for i in range(N):
        x, y = pos[i]
        if i == root:
            ax.plot(x, y, 'o', markersize=14, color='#333333',
                    markeredgecolor='black', markeredgewidth=2, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white', zorder=6)
        else:
            ax.plot(x, y, 'o', markersize=10, color='white',
                    markeredgecolor='#666666', markeredgewidth=1.5, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=6, color='#333333', zorder=6)

    # Legend
    handles = [mpatches.Patch(color=TREE_COLORS[t % len(TREE_COLORS)],
                              label=f'Tree {t}')
               for t in range(tau)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_flink_usage(ax, trees, N, flink, root, link_id, pos, rows, cols):
    """Color each mesh edge by how many trees share it."""
    ax.set_aspect('equal')
    ax.set_title('Link Sharing Heatmap', fontsize=13, fontweight='bold')

    # Count how many trees use each physical link
    max_lid = max(link_id.values()) + 1
    usage = [0] * max_lid
    for tree in trees:
        used_in_tree = set()
        for i in range(N):
            p = tree[i]
            if p < 0:
                continue
            lo, hi = min(p, i), max(p, i)
            fl = flink[p, i]
            if fl > 0 and fl not in used_in_tree:
                used_in_tree.add(fl)
        for fl in used_in_tree:
            if fl < max_lid:
                usage[fl] += 1

    tau = len(trees)
    # Draw edges colored by usage count
    for (a, b), lid in link_id.items():
        xa, ya = pos[a]
        xb, yb = pos[b]
        u = usage[lid]
        if u == 0:
            color = '#e0e0e0'
            lw = 1.0
        elif u == 1:
            color = '#3cb44b'  # green = exclusive
            lw = 2.5
        elif u == 2:
            color = '#f58231'  # orange = shared by 2
            lw = 3.0
        else:
            color = '#e6194b'  # red = shared by 3+
            lw = 3.5
        ax.plot([xa, xb], [ya, yb], '-', color=color, linewidth=lw, zorder=2)

    # Draw nodes
    for i in range(N):
        x, y = pos[i]
        if i == root:
            ax.plot(x, y, 'o', markersize=12, color='#333333',
                    markeredgecolor='black', markeredgewidth=2, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white', zorder=6)
        else:
            ax.plot(x, y, 'o', markersize=9, color='white',
                    markeredgecolor='#666666', markeredgewidth=1.2, zorder=5)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=6, color='#333333', zorder=6)

    # Legend
    handles = [
        mpatches.Patch(color='#e0e0e0', label='Unused'),
        mpatches.Patch(color='#3cb44b', label='1 tree (exclusive)'),
        mpatches.Patch(color='#f58231', label='2 trees (shared)'),
        mpatches.Patch(color='#e6194b', label='3+ trees (contended)'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize BBS spanning trees on a 2D mesh')
    parser.add_argument('--rows', type=int, default=6)
    parser.add_argument('--cols', type=int, default=6)
    parser.add_argument('--root', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Save figure to file instead of showing')
    args = parser.parse_args()

    rows, cols, root = args.rows, args.cols, args.root
    N = rows * cols
    assert 0 <= root < N, f"root must be in [0, {N})"

    print(f"Building {rows}x{cols} mesh ({N} nodes)...")
    (N, adj, lat, flink, path_len, path_links, max_hops,
     pos, link_id) = build_mesh(rows, cols)

    print(f"Computing BBS trees (root={root})...")
    tau, trees = bbs_compute_trees(N, adj, lat, flink, path_len,
                                   path_links, max_hops, root)

    # Layout: individual trees + overlay + heatmap
    n_panels = tau + 2
    ncols_fig = min(4, n_panels)
    nrows_fig = (n_panels + ncols_fig - 1) // ncols_fig

    fig, axes = plt.subplots(nrows_fig, ncols_fig,
                             figsize=(5 * ncols_fig, 5 * nrows_fig))
    if nrows_fig == 1 and ncols_fig == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Hide extra axes
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    # Draw individual trees
    for t in range(tau):
        color = TREE_COLORS[t % len(TREE_COLORS)]
        draw_tree(axes[t], trees[t], N, pos, root, color,
                  f'Tree {t} (depth={tree_depth(trees[t], N)})',
                  rows, cols)

    # Overlay
    draw_overlay(axes[tau], trees, N, pos, root, rows, cols)

    # Heatmap
    draw_flink_usage(axes[tau + 1], trees, N, flink, root,
                     link_id, pos, rows, cols)

    fig.suptitle(f'BBS Trees on {rows}x{cols} Mesh  |  root={root}, '
                 f'tau={tau}', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
