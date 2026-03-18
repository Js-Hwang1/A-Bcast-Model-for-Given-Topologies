#!/usr/bin/env python3
"""Compare three trees: chain, dissemination, and balanced-hierarchy binary."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

N = 128

def dfly_group(v): return v // 32
def dfly_chassis(v): return (v % 32) // 8
def dfly_router(v): return (v % 8) // 2
def L(v): return f"{dfly_group(v)}{dfly_chassis(v)}{dfly_router(v)}{v%2}"

def edge_type(a, b):
    if dfly_group(a) != dfly_group(b): return 'blue'
    if dfly_chassis(a) != dfly_chassis(b): return 'black'
    if dfly_router(a) != dfly_router(b): return 'green'
    return 'red'

def latency(a, b):
    ag, ac, ar = dfly_group(a), dfly_chassis(a), dfly_router(a)
    bg, bc, br = dfly_group(b), dfly_chassis(b), dfly_router(b)
    lat = 200
    if ag != bg:
        lat += 400
        if ar != 0: lat += 100
        if ac != 0: lat += 200
        if bc != 0: lat += 200
        if br != 0: lat += 100
    elif ac != bc:
        if ar != 0: lat += 100
        lat += 200
        if br != 0: lat += 100
    elif ar != br:
        lat += 100
    return lat


# ─── Tree 1: Chain (current best, unbalanced binary) ───
def build_chain_tree(root=0):
    tree = [-1] * N
    queue = deque()
    queue.append((root, [i for i in range(N) if i != root]))
    while queue:
        par, nodes = queue.popleft()
        if not nodes: continue
        if len(nodes) == 1:
            tree[nodes[0]] = par; continue
        pg, pc, pr = dfly_group(par), dfly_chassis(par), dfly_router(par)
        groups = set(dfly_group(v) for v in nodes)
        chassis_set = set(dfly_chassis(v) for v in nodes if dfly_group(v) == pg)
        router_set = set(dfly_router(v) for v in nodes
                         if dfly_group(v) == pg and dfly_chassis(v) == pc)
        left, right = [], []
        if len(groups) > 1 and pg in groups:
            for v in nodes:
                (left if dfly_group(v) == pg else right).append(v)
        elif len(chassis_set) > 1 and pc in chassis_set:
            for v in nodes:
                (left if dfly_chassis(v) == pc else right).append(v)
        elif len(router_set) > 1 and pr in router_set:
            for v in nodes:
                (left if dfly_router(v) == pr else right).append(v)
        else:
            left.append(nodes[0])
            if len(nodes) > 1: right.append(nodes[1])
        def pick_rep(arr):
            if not arr: return None, arr
            i = min(range(len(arr)), key=lambda i: latency(par, arr[i]))
            return arr[i], arr[:i] + arr[i+1:]
        lr, lrest = pick_rep(left)
        if lr is not None:
            tree[lr] = par
            if lrest: queue.append((lr, lrest))
        rr, rrest = pick_rep(right)
        if rr is not None:
            tree[rr] = par
            if rrest: queue.append((rr, rrest))
    return tree


# ─── Tree 2: Dissemination (user's idea, high fanout) ───
def build_dissemination_tree(root=0):
    tree = [-1] * N
    tree[32] = 0; tree[64] = 0; tree[96] = 32
    for g in range(4):
        b = g * 32
        tree[b+8] = b; tree[b+16] = b; tree[b+24] = b+8
    for g in range(4):
        for c in range(4):
            b = g*32 + c*8
            tree[b+2] = b; tree[b+4] = b; tree[b+6] = b+2
    for g in range(4):
        for c in range(4):
            for r in range(4):
                b = g*32 + c*8 + r*2
                tree[b+1] = b
    return tree


# ─── Tree 3: HYBRID — balanced hierarchy binary (MY PROPOSAL) ───
def build_hybrid_tree(root=0):
    """Balanced binary splits at each hierarchy level.
    Gateway-aware rep selection to minimize contention."""
    tree = [-1] * N

    def split_and_build(parent, nodes):
        if not nodes:
            return
        if len(nodes) == 1:
            tree[nodes[0]] = parent
            return

        # Determine split dimension
        groups = sorted(set(dfly_group(v) for v in nodes))
        chassis = sorted(set(dfly_chassis(v) for v in nodes))
        routers = sorted(set(dfly_router(v) for v in nodes))

        spanning_groups = len(groups) >= 2
        spanning_chassis = len(chassis) >= 2

        if spanning_groups:
            mid = len(groups) // 2
            left_set = set(groups[:mid])
            left = [v for v in nodes if dfly_group(v) in left_set]
            right = [v for v in nodes if dfly_group(v) not in left_set]
        elif spanning_chassis:
            mid = len(chassis) // 2
            left_set = set(chassis[:mid])
            left = [v for v in nodes if dfly_chassis(v) in left_set]
            right = [v for v in nodes if dfly_chassis(v) not in left_set]
        elif len(routers) >= 2:
            mid = len(routers) // 2
            left_set = set(routers[:mid])
            left = [v for v in nodes if dfly_router(v) in left_set]
            right = [v for v in nodes if dfly_router(v) not in left_set]
        else:
            mid = len(nodes) // 2
            left = nodes[:mid]
            right = nodes[mid:]

        def pick_rep(arr, need_gateway_group=False, need_gateway_chassis=False):
            if not arr: return None, arr
            candidates = list(range(len(arr)))
            if need_gateway_group:
                # Prefer (*, 0, 0, 0) — group gateway nodes
                gw = [i for i in candidates if dfly_chassis(arr[i]) == 0
                      and dfly_router(arr[i]) == 0 and arr[i] % 2 == 0]
                if gw: candidates = gw
            elif need_gateway_chassis:
                # Prefer (g, *, 0, 0) — chassis gateway nodes
                gw = [i for i in candidates if dfly_router(arr[i]) == 0
                      and arr[i] % 2 == 0]
                if gw: candidates = gw
            i = min(candidates, key=lambda i: latency(parent, arr[i]))
            return arr[i], arr[:i] + arr[i+1:]

        lr, lrest = pick_rep(left, need_gateway_group=spanning_groups,
                             need_gateway_chassis=spanning_chassis)
        rr, rrest = pick_rep(right, need_gateway_group=spanning_groups,
                             need_gateway_chassis=spanning_chassis)

        if lr is not None:
            tree[lr] = parent
            split_and_build(lr, lrest)
        if rr is not None:
            tree[rr] = parent
            split_and_build(rr, rrest)

    all_nodes = [i for i in range(N) if i != root]
    split_and_build(root, all_nodes)
    return tree


# ─── Simulation (interleaved ordering, proven best) ───
def simulate(tree, root, k):
    children = {i: [] for i in range(N)}
    for i in range(N):
        if tree[i] >= 0:
            children[tree[i]].append(i)
    # Sort: blue > black > green > red
    for v in children:
        children[v].sort(key=lambda c: (
            0 if dfly_group(v) != dfly_group(c) else
            1 if dfly_chassis(v) != dfly_chassis(c) else
            2 if dfly_router(v) != dfly_router(c) else 3))

    recv = [[0]*k if i == root else [None]*k for i in range(N)]
    sends = []
    order = [root]
    q = deque([root]); vis = {root}
    while q:
        v = q.popleft()
        for c in children[v]:
            if c not in vis:
                vis.add(c); q.append(c); order.append(c)
    for v in order:
        ch = children[v]
        f = len(ch)
        if f == 0: continue
        prev_end = 0
        for j in range(k):
            start = max(recv[v][j], prev_end)
            for i, c in enumerate(ch):
                recv[c][j] = start + i + 1
                sends.append((start + i, v, c, j))
            prev_end = start + f
    return recv, sends, children


# ─── Tree stats ───
def tree_stats(tree, root, label):
    children = {i: [] for i in range(N)}
    for i in range(N):
        if tree[i] >= 0:
            children[tree[i]].append(i)

    # Depth via BFS
    depth = {root: 0}
    q = deque([root])
    while q:
        v = q.popleft()
        for c in children[v]:
            depth[c] = depth[v] + 1
            q.append(c)

    max_depth = max(depth.values())
    fanouts = [len(children[v]) for v in range(N)]
    max_fan = max(fanouts)

    # Count edge types
    blue = sum(1 for i in range(N) if tree[i] >= 0 and edge_type(tree[i], i) == 'blue')
    black = sum(1 for i in range(N) if tree[i] >= 0 and edge_type(tree[i], i) == 'black')
    green = sum(1 for i in range(N) if tree[i] >= 0 and edge_type(tree[i], i) == 'green')
    red = sum(1 for i in range(N) if tree[i] >= 0 and edge_type(tree[i], i) == 'red')

    # Max green per router
    green_per_router = {}
    for i in range(N):
        if tree[i] >= 0 and edge_type(tree[i], i) == 'green':
            r = (dfly_group(tree[i]), dfly_chassis(tree[i]))
            green_per_router[r] = green_per_router.get(r, 0) + 1
    max_green = max(green_per_router.values()) if green_per_router else 0

    # Uplink load: how many tree edges transit through each router's uplink
    uplink_load = {}
    for i in range(N):
        if tree[i] < 0: continue
        p = tree[i]
        if edge_type(p, i) in ('blue', 'black'):
            # Parent side: if parent not on router 0, needs uplink
            pr = dfly_router(p)
            if pr != 0:
                key = (dfly_group(p), dfly_chassis(p), pr)
                uplink_load[key] = uplink_load.get(key, 0) + 1
            # Child side: if child not on router 0, needs uplink
            cr = dfly_router(i)
            if cr != 0:
                key = (dfly_group(i), dfly_chassis(i), cr)
                uplink_load[key] = uplink_load.get(key, 0) + 1
    max_uplink = max(uplink_load.values()) if uplink_load else 0

    print(f"  {label}: depth={max_depth}, max_fan={max_fan}, "
          f"edges: {blue}B {black}K {green}G {red}R, "
          f"max_green/router={max_green}, max_uplink={max_uplink}")

    # Show high-fanout nodes
    high_fan = [(v, fanouts[v]) for v in range(N) if fanouts[v] >= 3]
    if high_fan:
        high_fan.sort(key=lambda x: -x[1])
        print(f"    High fanout: " + ", ".join(f"{L(v)}(f={f})" for v, f in high_fan[:6]))

    # Fanout distribution
    from collections import Counter
    fc = Counter(fanouts)
    print(f"    Fanout dist: " + ", ".join(f"f={f}:{n}" for f, n in sorted(fc.items())))

    return max_depth, max_fan


# ─── Main comparison ───
root = 0
chain = build_chain_tree(root)
dissem = build_dissemination_tree(root)
hybrid = build_hybrid_tree(root)

print("=" * 70)
print("  TREE STATISTICS")
print("=" * 70)
tree_stats(chain, root, "CHAIN")
tree_stats(dissem, root, "DISSEMINATION")
tree_stats(hybrid, root, "HYBRID")

print("\n" + "=" * 70)
print("  PIPELINE COMPLETION (interleaved ordering)")
print("=" * 70)
print(f"  {'k':>5s}  {'Chain':>8s}  {'Dissem':>8s}  {'Hybrid':>8s}  {'Best':>8s}")
print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

for k in [1, 2, 4, 8, 16, 32, 64, 128]:
    rc, _, _ = simulate(chain, root, k)
    rd, _, _ = simulate(dissem, root, k)
    rh, _, _ = simulate(hybrid, root, k)
    tc = max(r for i in range(N) for r in [rc[i][k-1]] if r is not None)
    td = max(r for i in range(N) for r in [rd[i][k-1]] if r is not None)
    th = max(r for i in range(N) for r in [rh[i][k-1]] if r is not None)
    best = min(tc, td, th)
    winner = "CHAIN" if best == tc else "DISSEM" if best == td else "HYBRID"
    print(f"  {k:5d}  {tc:8d}  {td:8d}  {th:8d}  {winner:>8s}")


# ─── Show root's children for hybrid ───
print("\n" + "=" * 70)
print("  HYBRID TREE: Root's subtree structure")
print("=" * 70)
children = {i: [] for i in range(N)}
for i in range(N):
    if hybrid[i] >= 0:
        children[hybrid[i]].append(i)

def show_subtree(v, indent=0, max_depth=3):
    ch = children[v]
    if not ch: return
    for c in ch:
        et = edge_type(v, c)
        subtree_size = 0
        q = deque([c]); vis = {c}
        while q:
            x = q.popleft()
            subtree_size += 1
            for cc in children[x]:
                if cc not in vis:
                    vis.add(cc); q.append(cc)
        print(f"  {'  '*indent}{L(v)} -> {L(c)} [{et}] (subtree: {subtree_size} nodes)")
        if indent < max_depth:
            show_subtree(c, indent + 1, max_depth)

show_subtree(0, 0, 3)


# ─── Plot: activity comparison ───
ECOLORS = {'blue': '#1565C0', 'black': '#455A64', 'green': '#2E7D32', 'red': '#C62828'}

fig, axes = plt.subplots(3, 1, figsize=(22, 14))

for ax, (label, tree) in zip(axes, [
    ("CHAIN", chain), ("DISSEMINATION", dissem), ("HYBRID (balanced binary)", hybrid)
]):
    recv, sends, _ = simulate(tree, root, 128)
    comp_t = max(r for i in range(N) for r in [recv[i][127]] if r is not None)
    max_t = comp_t + 5

    by_time = {et: [0]*(max_t+1) for et in ['blue','black','green','red']}
    for t, s, r, j in sends:
        if t <= max_t:
            by_time[edge_type(s, r)][t] += 1

    x = np.arange(max_t + 1)
    bottom = np.zeros(max_t + 1)
    for et, lbl in [('blue','Blue (inter-group)'), ('black','Black (inter-chassis)'),
                     ('green','Green (inter-router)'), ('red','Red (sibling)')]:
        vals = np.array(by_time[et])
        ax.bar(x, vals, bottom=bottom, color=ECOLORS[et], width=0.85,
               label=lbl, edgecolor='white', linewidth=0.3)
        bottom += vals

    ax.axvline(comp_t, color='red', ls='--', lw=3, alpha=0.8, label=f'Done: t={comp_t}')
    ax.set_title(f'{label} (k=128, done @ t={comp_t})', fontsize=14, fontweight='bold')
    ax.set_ylabel('# parallel sends', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.5, max_t + 0.5)
    ax.tick_params(labelsize=10)
    ax.grid(axis='y', alpha=0.2)

axes[-1].set_xlabel('Time step', fontsize=12)
fig.suptitle('Network Activity: Chain vs Dissemination vs Hybrid (k=128)',
             fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig('plot_hybrid_compare.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved plot_hybrid_compare.png")
