#!/usr/bin/env python3
"""Step-by-step pipelined broadcast timeline: chain vs dissemination.

Model: sequential sends. Each node sends to ONE child per time step.
A node can receive from parent while sending to a child.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import deque

N = 128

def dfly_group(v): return v // 32
def dfly_chassis(v): return (v % 32) // 8
def dfly_router(v): return (v % 8) // 2
def L(v):
    return f"{dfly_group(v)}{dfly_chassis(v)}{dfly_router(v)}{v%2}"

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


# ── Tree builders ──

def build_chain_tree(root=0):
    """Chain split at every hierarchy level (current best, fanout=2)."""
    tree = [-1] * N
    queue = deque()
    queue.append((root, [i for i in range(N) if i != root]))
    while queue:
        par, nodes = queue.popleft()
        if not nodes: continue
        if len(nodes) == 1:
            tree[nodes[0]] = par
            continue
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
    """Strict hierarchical binary dissemination (depth=7, fanout=7 at root)."""
    tree = [-1] * N
    # Group level: binary dissemination among 4 groups
    tree[32] = 0;  tree[64] = 0;  tree[96] = 32
    # Chassis level per group
    for g in range(4):
        b = g * 32
        tree[b+8] = b;  tree[b+16] = b;  tree[b+24] = b+8
    # Router level per chassis
    for g in range(4):
        for c in range(4):
            b = g*32 + c*8
            tree[b+2] = b;  tree[b+4] = b;  tree[b+6] = b+2
    # Sibling
    for g in range(4):
        for c in range(4):
            for r in range(4):
                b = g*32 + c*8 + r*2
                tree[b+1] = b
    return tree


# ── Simulation ──

def simulate(tree, root, k):
    """Simulate pipelined bcast with sequential sends to ordered children.

    Returns:
      recv[node][chunk] = time when node has that chunk
      sends = list of (time, sender, receiver, chunk)
    """
    children = {i: [] for i in range(N)}
    for i in range(N):
        if tree[i] >= 0:
            children[tree[i]].append(i)

    # Order children: blue > black > green > red  (groups first)
    def prio(par, ch):
        if dfly_group(par) != dfly_group(ch): return 0
        if dfly_chassis(par) != dfly_chassis(ch): return 1
        if dfly_router(par) != dfly_router(ch): return 2
        return 3
    for v in children:
        children[v].sort(key=lambda c: prio(v, c))

    recv = [[0]*k if i == root else [None]*k for i in range(N)]
    sends = []

    # BFS order from root
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
                t = start + i
                recv[c][j] = t + 1
                sends.append((t, v, c, j))
            prev_end = start + f

    return recv, sends, children


# ── Print timeline ──

ETYPE_SYMBOL = {'blue': '⬤', 'black': '◼', 'green': '◆', 'red': '●'}

def print_timeline(sends, recv, k, title, max_t=None):
    completion = max(recv[i][k-1] for i in range(N) if recv[i][k-1] is not None)
    if max_t is None:
        max_t = completion

    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"  All {N} nodes have all {k} chunks at t = {completion}")
    print(f"{'='*78}")

    by_time = {}
    for t, s, r, j in sends:
        by_time.setdefault(t, []).append((s, r, j))

    for t in range(max_t + 1):
        items = by_time.get(t, [])
        root_send = [(s,r,j) for s,r,j in items if s == 0]
        blues  = [(s,r,j) for s,r,j in items if edge_type(s,r) == 'blue']
        blacks = [(s,r,j) for s,r,j in items if edge_type(s,r) == 'black']
        greens = [(s,r,j) for s,r,j in items if edge_type(s,r) == 'green']
        reds   = [(s,r,j) for s,r,j in items if edge_type(s,r) == 'red']

        root_str = ""
        if root_send:
            rs = root_send[0]
            et = edge_type(0, rs[1])
            root_str = f" ROOT→{L(rs[1])}[c{rs[2]},{et}]"

        parts = []
        if blues:  parts.append(f"{len(blues)} blue")
        if blacks: parts.append(f"{len(blacks)} blk")
        if greens: parts.append(f"{len(greens)} grn")
        if reds:   parts.append(f"{len(reds)} red")
        count_str = ", ".join(parts) if parts else "—"

        # Show a few interesting non-root sends
        others = [(s,r,j) for s,r,j in items if s != 0]
        other_str = ""
        if others:
            shown = others[:3]
            descs = [f"{L(s)}→{L(r)}[c{j}]" for s,r,j in shown]
            extra = f" +{len(others)-3}" if len(others) > 3 else ""
            other_str = "  " + ", ".join(descs) + extra

        print(f"  t={t:3d}: [{count_str:>20s}] total={len(items):3d}{root_str}{other_str}")

    print(f"\n  >>> Broadcast complete at t = {completion}")


# ── Plot ──

ECOLORS = {'blue': '#1565C0', 'black': '#37474F', 'green': '#2E7D32', 'red': '#C62828'}
CHUNK_COLORS = ['#E53935', '#FB8C00', '#43A047', '#1E88E5', '#8E24AA', '#00ACC1',
                '#FFB300', '#6D4C41']

def plot_root_gantt(ax, sends, k, fanout, title):
    """Gantt chart of root's sends: each bar is one send, colored by chunk."""
    root_sends = sorted([(t, r, j) for t, s, r, j in sends if s == 0])
    for t, r, j in root_sends:
        color = CHUNK_COLORS[j % len(CHUNK_COLORS)]
        et = edge_type(0, r)
        ec = ECOLORS[et]
        ax.barh(0, 1, left=t, height=0.6, color=color, edgecolor=ec, linewidth=1.5)
        ax.text(t + 0.5, 0, f"c{j}", ha='center', va='center', fontsize=6,
                fontweight='bold', color='white')

    ax.set_yticks([0])
    ax.set_yticklabels([f'Root\n(fan={fanout})'])
    ax.set_xlabel('Time step')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, max(t for t,_,_ in root_sends) + 2)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(axis='x', alpha=0.3)


def plot_completion_hist(ax, recv, k, title, color, max_x=None):
    """Histogram of when each node finishes (has all k chunks)."""
    completions = [recv[i][k-1] for i in range(N) if recv[i][k-1] is not None]
    bins = range(min(completions), max(completions) + 2)
    ax.hist(completions, bins=bins, color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.axvline(max(completions), color='red', linestyle='--', linewidth=2,
               label=f'Last node: t={max(completions)}')
    ax.set_xlabel('Completion time (time steps)')
    ax.set_ylabel('# nodes')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    if max_x: ax.set_xlim(0, max_x)


def plot_parallel_sends(ax, sends, k, title, max_t=None):
    """Stacked bar: # sends per time step, by link type."""
    if max_t is None:
        max_t = max(t for t,_,_,_ in sends)

    by_time = {et: [0]*(max_t+1) for et in ['blue','black','green','red']}
    for t, s, r, j in sends:
        if t <= max_t:
            by_time[edge_type(s, r)][t] += 1

    x = np.arange(max_t + 1)
    bottom = np.zeros(max_t + 1)
    for et in ['blue', 'black', 'green', 'red']:
        vals = np.array(by_time[et])
        ax.bar(x, vals, bottom=bottom, color=ECOLORS[et], width=0.9,
               label=et, edgecolor='white', linewidth=0.3)
        bottom += vals

    ax.set_xlabel('Time step')
    ax.set_ylabel('# parallel sends')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-0.5, max_t + 0.5)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

root = 0
chain = build_chain_tree(root)
dissem = build_dissemination_tree(root)

# Fanout stats
def max_fanout(tree):
    f = [0]*N
    for i in range(N):
        if tree[i] >= 0: f[tree[i]] += 1
    return max(f)

fan_c = max_fanout(chain)
fan_d = max_fanout(dissem)

for k_val in [1, 4]:
    recv_c, sends_c, _ = simulate(chain, root, k_val)
    recv_d, sends_d, _ = simulate(dissem, root, k_val)
    max_show = 35 if k_val <= 4 else 50
    print_timeline(sends_d, recv_d, k_val,
                   f"DISSEMINATION  k={k_val}  (depth=7, max_fanout={fan_d})", max_show)
    print_timeline(sends_c, recv_c, k_val,
                   f"CHAIN          k={k_val}  (depth=10, max_fanout={fan_c})", max_show)

# ── k=1 comparison plot ──
recv_c1, sends_c1, _ = simulate(chain, root, 1)
recv_d1, sends_d1, _ = simulate(dissem, root, 1)

fig, axes = plt.subplots(2, 2, figsize=(20, 8))

plot_parallel_sends(axes[0,0], sends_d1, 1,
    f'DISSEMINATION k=1: parallel sends per step', max_t=12)
plot_parallel_sends(axes[1,0], sends_c1, 1,
    f'CHAIN k=1: parallel sends per step', max_t=12)

comp_d1 = max(r for i in range(N) for r in [recv_d1[i][0]] if r is not None)
comp_c1 = max(r for i in range(N) for r in [recv_c1[i][0]] if r is not None)
max_x = max(comp_d1, comp_c1) + 2
plot_completion_hist(axes[0,1], recv_d1, 1,
    f'DISSEMINATION k=1: node completion (all done @ t={comp_d1})', '#1E88E5', max_x)
plot_completion_hist(axes[1,1], recv_c1, 1,
    f'CHAIN k=1: node completion (all done @ t={comp_c1})', '#E53935', max_x)

fig.suptitle('k=1 (NO pipelining): Dissemination finishes FASTER',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig('timeline_k1.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved timeline_k1.png")

# ── k=4 comparison plot ──
recv_c4, sends_c4, _ = simulate(chain, root, 4)
recv_d4, sends_d4, _ = simulate(dissem, root, 4)

fig2, axes2 = plt.subplots(2, 2, figsize=(24, 10))

comp_d4 = max(r for i in range(N) for r in [recv_d4[i][3]] if r is not None)
comp_c4 = max(r for i in range(N) for r in [recv_c4[i][3]] if r is not None)
max_t_show = max(comp_d4, comp_c4) + 2

plot_parallel_sends(axes2[0,0], sends_d4, 4,
    f'DISSEMINATION k=4: parallel sends per step', max_t=max_t_show)
plot_parallel_sends(axes2[1,0], sends_c4, 4,
    f'CHAIN k=4: parallel sends per step', max_t=max_t_show)

max_x4 = max(comp_d4, comp_c4) + 2
plot_completion_hist(axes2[0,1], recv_d4, 4,
    f'DISSEMINATION k=4: node completion (all done @ t={comp_d4})', '#1E88E5', max_x4)
plot_completion_hist(axes2[1,1], recv_c4, 4,
    f'CHAIN k=4: node completion (all done @ t={comp_c4})', '#E53935', max_x4)

fig2.suptitle('k=4 (pipelined): Which is faster?',
              fontsize=14, fontweight='bold')
fig2.tight_layout()
fig2.savefig('timeline_k4.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved timeline_k4.png")

# ── Root Gantt comparison for k=4 ──
fig3, (ax_d, ax_c) = plt.subplots(2, 1, figsize=(24, 4))
plot_root_gantt(ax_d, sends_d4, 4, fan_d,
    f'DISSEMINATION root sends (fan={fan_d}): 7 sends per chunk → period=7')
plot_root_gantt(ax_c, sends_c4, 4, fan_c,
    f'CHAIN root sends (fan={fan_c}): 2 sends per chunk → period=2')
# Sync x-axis
xmax = max(ax_d.get_xlim()[1], ax_c.get_xlim()[1])
ax_d.set_xlim(-0.5, xmax); ax_c.set_xlim(-0.5, xmax)
fig3.suptitle("Root node activity: k=4 chunks (each color = one chunk)",
              fontsize=13, fontweight='bold')
fig3.tight_layout()
fig3.savefig('timeline_root_gantt.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved timeline_root_gantt.png")

# ── k=128 (realistic) completion times ──
recv_c128, sends_c128, _ = simulate(chain, root, 128)
recv_d128, sends_d128, _ = simulate(dissem, root, 128)
comp_c128 = max(r for i in range(N) for r in [recv_c128[i][127]] if r is not None)
comp_d128 = max(r for i in range(N) for r in [recv_d128[i][127]] if r is not None)

fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
max_x128 = max(comp_c128, comp_d128) + 10
plot_completion_hist(ax1, recv_d128, 128,
    f'DISSEMINATION k=128 (all done @ t={comp_d128})', '#1E88E5', max_x128)
plot_completion_hist(ax2, recv_c128, 128,
    f'CHAIN k=128 (all done @ t={comp_c128})', '#E53935', max_x128)
fig4.suptitle(f'k=128: Dissemination t={comp_d128}  vs  Chain t={comp_c128}',
              fontsize=14, fontweight='bold')
fig4.tight_layout()
fig4.savefig('timeline_k128.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved timeline_k128.png")

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  k=1:   Dissem t={comp_d1:4d}   Chain t={comp_c1:4d}   winner: {'DISSEM' if comp_d1 < comp_c1 else 'CHAIN'}")
print(f"  k=4:   Dissem t={comp_d4:4d}   Chain t={comp_c4:4d}   winner: {'DISSEM' if comp_d4 < comp_c4 else 'CHAIN'}")
print(f"  k=128: Dissem t={comp_d128:4d}   Chain t={comp_c128:4d}   winner: {'DISSEM' if comp_d128 < comp_c128 else 'CHAIN'}")
