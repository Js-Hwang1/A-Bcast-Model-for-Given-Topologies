#!/usr/bin/env python3
"""Simulate the user's exact ordering: send ALL chunks to child1 first,
then ALL chunks to child2, etc. Overlapped send+receive allowed."""

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


def get_children(tree):
    children = {i: [] for i in range(N)}
    for i in range(N):
        if tree[i] >= 0:
            children[tree[i]].append(i)
    # Order: blue > black > green > red (groups first)
    for v in children:
        children[v].sort(key=lambda c: (
            0 if dfly_group(v) != dfly_group(c) else
            1 if dfly_chassis(v) != dfly_chassis(c) else
            2 if dfly_router(v) != dfly_router(c) else 3))
    return children


def simulate_allfirst(tree, root, k):
    """User's model: send ALL k chunks to child1 first, then ALL to child2, etc.
    Overlapped send+receive: a node can forward chunks while still receiving."""
    children = get_children(tree)

    # has[v][j] = step at which v can use chunk j
    has = [[None]*k for _ in range(N)]
    for j in range(k):
        has[root][j] = 0

    # Track sends for visualization
    sends = []

    # BFS order
    order = [root]
    q = deque([root]); vis = {root}
    while q:
        v = q.popleft()
        for c in children[v]:
            if c not in vis:
                vis.add(c); q.append(c); order.append(c)

    for v in order:
        ch = children[v]
        if not ch: continue

        # v sends ALL k chunks to child1, then ALL k to child2, etc.
        send_cursor = 0  # next available send step for v

        for ci, c in enumerate(ch):
            for j in range(k):
                # Earliest v can send c_j: max(has[v][j], send_cursor)
                t = max(has[v][j], send_cursor)
                has[c][j] = t + 1  # child receives next step
                sends.append((t, v, c, j))
                send_cursor = t + 1

    return has, sends, children


def simulate_interleaved(tree, root, k):
    """Previous model: round-robin chunks across children."""
    children = get_children(tree)
    has = [[None]*k for _ in range(N)]
    for j in range(k):
        has[root][j] = 0
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
            start = max(has[v][j], prev_end)
            for i, c in enumerate(ch):
                t = start + i
                has[c][j] = t + 1
                sends.append((t, v, c, j))
            prev_end = start + f

    return has, sends, children


ECOLORS = {'blue': '#1565C0', 'black': '#455A64', 'green': '#2E7D32', 'red': '#C62828'}

root = 0
chain = build_chain_tree(root)
dissem = build_dissemination_tree(root)

print("="*70)
print("  ALL-FIRST ordering (your idea): send ALL chunks to child1,")
print("  then ALL chunks to child2, etc.")
print("="*70)

for label, tree in [("DISSEMINATION", dissem), ("CHAIN", chain)]:
    print(f"\n{'─'*60}")
    print(f"  {label} tree")
    print(f"{'─'*60}")
    for k in [1, 2, 4, 8, 128]:
        has_af, sends_af, ch = simulate_allfirst(tree, root, k)
        has_il, sends_il, _  = simulate_interleaved(tree, root, k)
        t_af = max(has_af[i][k-1] for i in range(N) if has_af[i][k-1] is not None)
        t_il = max(has_il[i][k-1] for i in range(N) if has_il[i][k-1] is not None)
        print(f"  k={k:4d}:  all-first={t_af:5d}   interleaved={t_il:5d}   diff={t_af-t_il:+d}")


# Detailed step-by-step for k=4, dissemination, all-first
print(f"\n{'='*70}")
print(f"  DISSEMINATION k=4, ALL-FIRST: step-by-step")
print(f"{'='*70}")

has_af, sends_af, ch = simulate_allfirst(dissem, root, 4)
by_time = {}
for t, s, r, j in sends_af:
    by_time.setdefault(t, []).append((s, r, j))

max_t = max(t for t, _, _, _ in sends_af)
for t in range(max_t + 1):
    items = by_time.get(t, [])
    root_send = [(s,r,j) for s,r,j in items if s == 0]
    blues  = len([(s,r,j) for s,r,j in items if edge_type(s,r) == 'blue'])
    blacks = len([(s,r,j) for s,r,j in items if edge_type(s,r) == 'black'])
    greens = len([(s,r,j) for s,r,j in items if edge_type(s,r) == 'green'])
    reds   = len([(s,r,j) for s,r,j in items if edge_type(s,r) == 'red'])

    parts = []
    if blues:  parts.append(f"{blues}blu")
    if blacks: parts.append(f"{blacks}blk")
    if greens: parts.append(f"{greens}grn")
    if reds:   parts.append(f"{reds}red")

    rs = ""
    if root_send:
        r_s = root_send[0]
        rs = f"  ROOT→{L(r_s[1])}[c{r_s[2]}]"

    others = [(s,r,j) for s,r,j in items if s != 0][:3]
    ostr = ""
    if others:
        ostr = "  " + ", ".join(f"{L(s)}→{L(r)}[c{j}]" for s,r,j in others)
        if len([(s,r,j) for s,r,j in items if s != 0]) > 3:
            ostr += f" +{len([(s,r,j) for s,r,j in items if s != 0])-3}"

    print(f"  t={t:3d}: [{','.join(parts):>18s}] total={len(items):3d}{rs}{ostr}")

print(f"\n  >>> Complete at t={max_t+1}")


# ── Plot: network activity comparison for k=4 ──
fig, axes = plt.subplots(2, 2, figsize=(24, 12))

for col, (label, tree) in enumerate([("DISSEMINATION", dissem), ("CHAIN", chain)]):
    has_af, sends_af, _ = simulate_allfirst(tree, root, 4)
    has_il, sends_il, _ = simulate_interleaved(tree, root, 4)
    t_af = max(r for i in range(N) for r in [has_af[i][3]] if r is not None)
    t_il = max(r for i in range(N) for r in [has_il[i][3]] if r is not None)

    max_t_plot = max(t_af, t_il) + 2

    for row, (slabel, sends, comp) in enumerate([
        ("ALL-FIRST", sends_af, t_af),
        ("INTERLEAVED", sends_il, t_il)
    ]):
        ax = axes[row][col]
        by_t = {et: [0]*(max_t_plot+1) for et in ['blue','black','green','red']}
        for t, s, r, j in sends:
            if t <= max_t_plot:
                by_t[edge_type(s, r)][t] += 1
        x = np.arange(max_t_plot + 1)
        bottom = np.zeros(max_t_plot + 1)
        for et, lbl in [('blue','Blue'), ('black','Black'), ('green','Green'), ('red','Red')]:
            vals = np.array(by_t[et])
            ax.bar(x, vals, bottom=bottom, color=ECOLORS[et], width=0.85,
                   label=lbl, edgecolor='white', linewidth=0.3)
            bottom += vals
        ax.axvline(comp, color='red', ls='--', lw=3, alpha=0.8, label=f'Done: t={comp}')
        ax.set_title(f'{label} — {slabel} (done @ t={comp})', fontsize=13, fontweight='bold')
        ax.set_xlabel('Time step', fontsize=11)
        ax.set_ylabel('# sends', fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(-0.5, max_t_plot + 0.5)
        ax.tick_params(labelsize=10)

fig.suptitle('k=4: ALL-FIRST vs INTERLEAVED ordering\n'
             '(Top = your all-first, Bottom = round-robin interleaved)',
             fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig('plot_allfirst_vs_interleaved.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved plot_allfirst_vs_interleaved.png")
