#!/usr/bin/env python3
"""Clear step-by-step timeline comparison: chain vs dissemination."""

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


def simulate(tree, root, k):
    children = {i: [] for i in range(N)}
    for i in range(N):
        if tree[i] >= 0:
            children[tree[i]].append(i)
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


ECOLORS = {'blue': '#1565C0', 'black': '#455A64', 'green': '#2E7D32', 'red': '#C62828'}
CHUNK_COLORS = ['#E53935', '#FB8C00', '#43A047', '#1E88E5',
                '#8E24AA', '#00ACC1', '#FFB300', '#6D4C41']

root = 0
chain = build_chain_tree(root)
dissem = build_dissemination_tree(root)

# ═══════════════════════════════════════════════════════
#  PLOT 1: Root Gantt chart — k=4
# ═══════════════════════════════════════════════════════
_, sends_d4, ch_d = simulate(dissem, root, 4)
_, sends_c4, ch_c = simulate(chain, root, 4)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 7))

for ax, sends, fan, label in [
    (ax1, sends_d4, 7, 'DISSEMINATION'),
    (ax2, sends_c4, 2, 'CHAIN')
]:
    root_sends = sorted([(t, r, j) for t, s, r, j in sends if s == 0])
    for t, r, j in root_sends:
        color = CHUNK_COLORS[j % len(CHUNK_COLORS)]
        et = edge_type(0, r)
        ec = ECOLORS[et]
        ax.barh(0, 0.95, left=t, height=0.7, color=color, edgecolor=ec, linewidth=2.5)
        dest = L(r)
        ax.text(t + 0.47, 0, f"c{j}\n→{dest}", ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    max_t = max(t for t, _, _ in root_sends) + 2
    ax.set_xlim(-0.5, max_t)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([0])
    ax.set_yticklabels([f'{label}\nRoot (fan={fan})'], fontsize=13, fontweight='bold')
    ax.set_xlabel('Time step', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='x', labelsize=11)

    # Mark chunk boundaries
    for j in range(4):
        t_start = j * fan
        ax.axvline(t_start, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(t_start + fan/2, 0.5, f'chunk {j}', ha='center', va='bottom',
                fontsize=10, color='#555', style='italic')

fig.suptitle('ROOT sends over time (k=4 chunks)\n'
             'Each colored bar = one send. Border color = link type (blue/black/green/red)',
             fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig('plot1_root_gantt.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plot1_root_gantt.png")


# ═══════════════════════════════════════════════════════
#  PLOT 2: Network activity per time step — k=4
# ═══════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 10))

for ax, sends, label, comp_t in [
    (ax1, sends_d4, 'DISSEMINATION (done @ t=28)', 28),
    (ax2, sends_c4, 'CHAIN (done @ t=17)', 17)
]:
    max_t = 30
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
               label=lbl, edgecolor='white', linewidth=0.5)
        bottom += vals

    ax.axvline(comp_t, color='red', linestyle='--', linewidth=3, alpha=0.8,
               label=f'All done: t={comp_t}')

    ax.set_ylabel('# parallel sends', fontsize=13)
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_title(label, fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(-0.5, max_t + 0.5)
    ax.tick_params(labelsize=11)
    ax.grid(axis='y', alpha=0.2)

    # Annotate root's chunk starts
    fan = 7 if 'DISSEM' in label else 2
    for j in range(4):
        t_s = j * fan
        ax.annotate(f'Root\nchunk {j}', xy=(t_s, 0), xytext=(t_s, -8),
                    fontsize=9, ha='center', color='#333', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

fig.suptitle('Network activity over time (k=4 chunks)\n'
             'Height = number of sends happening in parallel at each time step',
             fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig('plot2_activity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plot2_activity.png")


# ═══════════════════════════════════════════════════════
#  PLOT 3: Step-by-step text diagram for k=2
# ═══════════════════════════════════════════════════════
recv_d2, sends_d2, _ = simulate(dissem, root, 2)
recv_c2, sends_c2, _ = simulate(chain, root, 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))

for ax, sends, recv, label, k in [
    (ax1, sends_d2, recv_d2, 'DISSEMINATION (k=2)', 2),
    (ax2, sends_c2, recv_c2, 'CHAIN (k=2)', 2)
]:
    max_t_val = max(r for i in range(N) for r in [recv[i][k-1]] if r is not None)
    by_time = {}
    for t, s, r, j in sends:
        by_time.setdefault(t, []).append((s, r, j))

    ax.set_xlim(0, 42)
    ax.set_ylim(-max_t_val - 1, 1)
    ax.set_title(f'{label}\nComplete at t={max_t_val}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Time step', fontsize=14)

    for t in range(max_t_val + 1):
        items = by_time.get(t, [])
        # Background bar
        bg_color = '#f5f5f5' if t % 2 == 0 else '#e8e8e8'
        ax.axhspan(-t - 0.45, -t + 0.45, color=bg_color, zorder=0)
        ax.text(0.3, -t, f't={t}', ha='left', va='center', fontsize=10,
                fontweight='bold', color='#333')

        # Sort by type
        blues  = [(s,r,j) for s,r,j in items if edge_type(s,r) == 'blue']
        blacks = [(s,r,j) for s,r,j in items if edge_type(s,r) == 'black']
        greens = [(s,r,j) for s,r,j in items if edge_type(s,r) == 'green']
        reds   = [(s,r,j) for s,r,j in items if edge_type(s,r) == 'red']

        x = 3.5
        for group, et, clr in [(blues,'blue',ECOLORS['blue']),
                                (blacks,'black',ECOLORS['black']),
                                (greens,'green',ECOLORS['green']),
                                (reds,'red',ECOLORS['red'])]:
            if not group: continue
            # Show up to 4 sends, then count
            shown = group[:4]
            for s, r, j in shown:
                root_marker = '★' if s == 0 else ''
                txt = f"{root_marker}{L(s)}→{L(r)}"
                ax.text(x, -t, txt, ha='left', va='center', fontsize=8,
                        color=clr, fontweight='bold', fontfamily='monospace')
                x += 5.5
            if len(group) > 4:
                ax.text(x, -t, f'+{len(group)-4} more', ha='left', va='center',
                        fontsize=8, color=clr, style='italic')
                x += 4
            x += 1

        # Chunk label at right
        chunks_active = set(j for s,r,j in items)
        if chunks_active:
            ax.text(41, -t, f'c{",".join(str(c) for c in sorted(chunks_active))}',
                    ha='right', va='center', fontsize=9, color='#888')

    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.1)

fig.suptitle('Step-by-step sends (k=2 chunks, ★ = root sending)\n'
             'Format: sender→receiver',
             fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig('plot3_steps.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plot3_steps.png")

print("\n" + "="*50)
print("  SUMMARY (time steps to complete)")
print("="*50)
for k in [1, 2, 4, 8, 16, 128]:
    rd, _, _ = simulate(dissem, root, k)
    rc, _, _ = simulate(chain, root, k)
    td = max(r for i in range(N) for r in [rd[i][k-1]] if r is not None)
    tc = max(r for i in range(N) for r in [rc[i][k-1]] if r is not None)
    w = "DISSEM" if td < tc else "CHAIN" if tc < td else "TIE"
    print(f"  k={k:4d}:  Dissem={td:5d}  Chain={tc:5d}  ratio={td/tc:.2f}x  winner={w}")
