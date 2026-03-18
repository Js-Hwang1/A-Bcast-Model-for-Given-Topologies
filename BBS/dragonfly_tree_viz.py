#!/usr/bin/env python3
"""Visualize chain vs strict-dissemination dragonfly broadcast trees."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from collections import deque

NGROUPS = 4
NCHASSIS = 4
NROUTERS = 4

def dfly_group(v):   return v // 32
def dfly_chassis(v): return (v % 32) // 8
def dfly_router(v):  return (v % 8) // 2
def dfly_label(v):
    return f"({dfly_group(v)},{dfly_chassis(v)},{dfly_router(v)},{v%2})"

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


def _build_tree(root, N, balanced_groups, balanced_chassis, balanced_routers):
    """Generic tree builder with per-level chain vs balanced control."""
    tree = [-1] * N
    queue = deque()
    queue.append((root, [i for i in range(N) if i != root]))

    while queue:
        par, nodes = queue.popleft()
        if not nodes: continue
        if len(nodes) == 1:
            tree[nodes[0]] = par
            continue

        par_g, par_c, par_r = dfly_group(par), dfly_chassis(par), dfly_router(par)
        groups = set(dfly_group(v) for v in nodes)
        chassis_set = set(dfly_chassis(v) for v in nodes if dfly_group(v) == par_g)
        router_set = set(dfly_router(v) for v in nodes
                         if dfly_group(v) == par_g and dfly_chassis(v) == par_c)

        left, right = [], []

        if len(groups) > 1:
            if not balanced_groups and par_g in groups:
                for v in nodes:
                    (left if dfly_group(v) == par_g else right).append(v)
            else:
                gl = sorted(groups, key=lambda g: min(latency(par, v)
                            for v in nodes if dfly_group(v) == g))
                left_g = set(gl[:len(gl)//2])
                for v in nodes:
                    (left if dfly_group(v) in left_g else right).append(v)

        elif len(chassis_set) > 1:
            if not balanced_chassis and par_c in chassis_set:
                for v in nodes:
                    (left if dfly_chassis(v) == par_c else right).append(v)
            else:
                cl = sorted(chassis_set, key=lambda c: min(latency(par, v)
                            for v in nodes if dfly_group(v) == par_g and dfly_chassis(v) == c))
                left_c = set(cl[:len(cl)//2])
                for v in nodes:
                    (left if dfly_chassis(v) in left_c else right).append(v)

        elif len(router_set) > 1:
            if not balanced_routers and par_r in router_set:
                for v in nodes:
                    (left if dfly_router(v) == par_r else right).append(v)
            else:
                rl = sorted(router_set, key=lambda r: min(latency(par, v)
                            for v in nodes if dfly_group(v) == par_g
                            and dfly_chassis(v) == par_c and dfly_router(v) == r))
                left_r = set(rl[:len(rl)//2])
                for v in nodes:
                    (left if dfly_router(v) in left_r else right).append(v)
        else:
            left.append(nodes[0])
            if len(nodes) > 1: right.append(nodes[1])

        def pick_rep(arr):
            if not arr: return None, arr
            best_i = min(range(len(arr)), key=lambda i: latency(par, arr[i]))
            rep = arr[best_i]
            rest = arr[:best_i] + arr[best_i+1:]
            return rep, rest

        l_rep, l_rest = pick_rep(left)
        if l_rep is not None:
            tree[l_rep] = par
            if l_rest: queue.append((l_rep, l_rest))

        r_rep, r_rest = pick_rep(right)
        if r_rep is not None:
            tree[r_rep] = par
            if r_rest: queue.append((r_rep, r_rest))

    return tree

def build_chain_tree(root, N=128):
    return _build_tree(root, N, False, False, False)

def build_balanced_tree(root, N=128):
    return _build_tree(root, N, True, True, True)


def build_dissemination_tree(root=0, N=128):
    """Strict hierarchical binary dissemination.

    Level structure (each level completes before the next starts):
      Group level  (2 steps): binary dissemination among 4 group reps
      Chassis level (2 steps): binary dissemination among 4 chassis reps per group
      Router level  (2 steps): binary dissemination among 4 router reps per chassis
      Sibling level (1 step):  each router rep sends to its sibling

    Total depth = 7.  Max fanout = 7 (root is rep at every level).
    """
    tree = [-1] * N

    # Group-level binary dissemination:
    #   Step 1: G0_rep → G1_rep
    #   Step 2: G0_rep → G2_rep  AND  G1_rep → G3_rep
    group_rep = lambda g: g * 32   # (g,0,0,0)
    tree[group_rep(1)] = group_rep(0)  # 0000 → 1000
    tree[group_rep(2)] = group_rep(0)  # 0000 → 2000
    tree[group_rep(3)] = group_rep(1)  # 1000 → 3000

    # Chassis-level binary dissemination (parallel across all groups):
    #   Step 3: gXX → g1XX  (C0 rep → C1 rep)
    #   Step 4: gXX → g2XX  AND  g1XX → g3XX
    for g in range(4):
        chassis_rep = lambda c, g=g: g * 32 + c * 8  # (g,c,0,0)
        tree[chassis_rep(1)] = chassis_rep(0)  # gX00 → gX100
        tree[chassis_rep(2)] = chassis_rep(0)  # gX00 → gX200
        tree[chassis_rep(3)] = chassis_rep(1)  # gX100 → gX300

    # Router-level binary dissemination (parallel across all chassis):
    #   Step 5: gYXX → gY1XX  (R0 rep → R1 rep)
    #   Step 6: gYXX → gY2XX  AND  gY1XX → gY3XX
    for g in range(4):
        for c in range(4):
            router_rep = lambda r, g=g, c=c: g * 32 + c * 8 + r * 2  # (g,c,r,0)
            tree[router_rep(1)] = router_rep(0)  # gcX0 → gc10
            tree[router_rep(2)] = router_rep(0)  # gcX0 → gc20
            tree[router_rep(3)] = router_rep(1)  # gc10 → gc30

    # Sibling level:
    #   Step 7: each (g,c,r,0) → (g,c,r,1)
    for g in range(4):
        for c in range(4):
            for r in range(4):
                base = g * 32 + c * 8 + r * 2
                tree[base + 1] = base

    return tree


def get_depth(tree, node):
    d, v = 0, node
    while tree[v] >= 0:
        v = tree[v]; d += 1
    return d

def edge_type(a, b):
    if dfly_group(a) != dfly_group(b): return 'blue'
    if dfly_chassis(a) != dfly_chassis(b): return 'black'
    if dfly_router(a) != dfly_router(b): return 'green'
    return 'red'

EDGE_COLORS = {'blue': '#1565C0', 'black': '#37474F',
               'green': '#2E7D32', 'red': '#C62828'}
EDGE_NAMES  = {'blue': 'Inter-group (blue link)',
               'black': 'Inter-chassis (black link)',
               'green': 'Inter-router (green link)',
               'red': 'Sibling (uplink only)'}
GROUP_BG   = ['#BBDEFB', '#FFE0B2', '#C8E6C9', '#F8BBD0']
GROUP_DARK = ['#1565C0', '#E65100', '#2E7D32', '#AD1457']


def _count_subtree(children, v):
    count = 1
    for c in children[v]:
        count += _count_subtree(children, c)
    return count


def draw_logical_tree(tree, root, max_depth, ax, title=None):
    N = len(tree)
    children = {i: [] for i in range(N)}
    for i in range(N):
        if tree[i] >= 0:
            children[tree[i]].append(i)
    for v in children:
        children[v].sort()

    pos = {}
    x_counter = [0]

    def layout(v, depth):
        if depth > max_depth: return
        ch = [c for c in children[v] if depth + 1 <= max_depth]
        if not ch:
            pos[v] = (x_counter[0], -depth)
            x_counter[0] += 1
        else:
            for c in ch:
                layout(c, depth + 1)
            child_xs = [pos[c][0] for c in ch if c in pos]
            if child_xs:
                pos[v] = (sum(child_xs) / len(child_xs), -depth)
            else:
                pos[v] = (x_counter[0], -depth)
                x_counter[0] += 1

    layout(root, 0)
    if not pos: return
    total_w = x_counter[0]

    # Edges
    for v in pos:
        for c in children[v]:
            if c not in pos: continue
            et = edge_type(v, c)
            lw = 3.0 if et == 'blue' else 2.5 if et == 'black' else 2.0
            ax.plot([pos[v][0], pos[c][0]], [pos[v][1], pos[c][1]],
                    color=EDGE_COLORS[et], lw=lw, zorder=2, solid_capstyle='round')

    # Nodes
    fontsize = max(6, min(11, int(200 / (len(pos) ** 0.5))))
    for v, (x, y) in pos.items():
        g = dfly_group(v)
        fc = '#FFD600' if v == root else GROUP_BG[g]
        ec = '#F57F17' if v == root else GROUP_DARK[g]
        bbox = dict(boxstyle='round,pad=0.25', facecolor=fc, edgecolor=ec, linewidth=1.8)
        ax.text(x, y, dfly_label(v), ha='center', va='center',
                fontsize=fontsize, fontweight='bold', bbox=bbox, zorder=5, color='#212121')

        # Hidden subtree count
        ch_hidden = [c for c in children[v] if c not in pos]
        if ch_hidden:
            n_hidden = sum(_count_subtree(children, c) for c in ch_hidden)
            ax.text(x, y - 0.45, f'↓{n_hidden}',
                    ha='center', va='top', fontsize=fontsize - 2,
                    color='#666', style='italic')

    # Legend
    handles = []
    for et in ['blue', 'black', 'green', 'red']:
        handles.append(plt.Line2D([0], [0], color=EDGE_COLORS[et], lw=3,
                                   label=EDGE_NAMES[et]))
    for g in range(4):
        handles.append(mpatches.Patch(facecolor=GROUP_BG[g], edgecolor=GROUP_DARK[g],
                                       linewidth=1.5, label=f'Group {g}'))
    handles.append(mpatches.Patch(facecolor='#FFD600', edgecolor='#F57F17',
                                   linewidth=1.5, label='Root'))
    ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.95,
              edgecolor='#ccc')

    ax.set_xlim(-1, total_w + 1)
    ax.set_ylim(-max_depth - 0.8, 1.2)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)


def draw_physical_overlay(tree, root, ax, max_depth=None, title=None):
    N = len(tree)
    gx = [0, 14, 28, 42]
    node_pos = {}
    router_pos = {}

    for g in range(NGROUPS):
        for c in range(NCHASSIS):
            for r in range(NROUTERS):
                rx = gx[g] + r * 3
                ry = -c * 3.5
                router_pos[(g, c, r)] = (rx, ry)
                for s in range(2):
                    nid = g * 32 + c * 8 + r * 2 + s
                    node_pos[nid] = (rx + (s - 0.5) * 0.8, ry - 0.9)

    # Group bgs
    for g in range(NGROUPS):
        x0 = gx[g] - 1.5
        rect = mpatches.FancyBboxPatch((x0, -12.8), 11, 14.2,
                boxstyle='round,pad=0.4', facecolor=GROUP_BG[g], alpha=0.25,
                edgecolor=GROUP_DARK[g], linewidth=2)
        ax.add_patch(rect)
        ax.text(x0 + 5.5, 1.8, f'Group {g}', ha='center',
                fontsize=13, fontweight='bold', color=GROUP_DARK[g])

    # Chassis labels
    for g in range(NGROUPS):
        for c in range(NCHASSIS):
            ax.text(gx[g] - 0.6, -c * 3.5 + 0.15, f'C{c}', fontsize=7, color='#888')

    # Router circles
    for (g, c, r), (rx, ry) in router_pos.items():
        circ = plt.Circle((rx, ry), 0.55, facecolor='white',
                edgecolor='#666', linewidth=1.2, zorder=3, alpha=0.8)
        ax.add_patch(circ)
        ax.text(rx, ry, f'R{r}', ha='center', va='center',
                fontsize=7, color='#555', zorder=4)

    # Tree edges
    for i in range(N):
        if tree[i] < 0: continue
        if max_depth is not None and get_depth(tree, i) > max_depth: continue
        p = tree[i]
        et = edge_type(p, i)
        lw = 3.5 if et == 'blue' else 3.0 if et == 'black' else 2.0 if et == 'green' else 1.5
        x1, y1 = node_pos[p]; x2, y2 = node_pos[i]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=EDGE_COLORS[et], lw=lw,
                                    alpha=0.9, connectionstyle='arc3,rad=0.06'), zorder=6)

    # Compute nodes
    for nid, (nx, ny) in node_pos.items():
        d = get_depth(tree, nid)
        active = (max_depth is None or d <= max_depth)
        g = dfly_group(nid)
        if nid == root:
            color, ms, ec = '#FFD600', 10, '#F57F17'
        elif active:
            color, ms, ec = GROUP_BG[g], 7, GROUP_DARK[g]
        else:
            color, ms, ec = '#eee', 5, '#ccc'
        ax.plot(nx, ny, 'o', color=color, markersize=ms,
                markeredgecolor=ec, markeredgewidth=1.0, zorder=7)
        if active:
            ax.text(nx, ny + 0.55, dfly_label(nid), ha='center', va='bottom',
                    fontsize=5.5, color='#333', fontweight='bold', zorder=8,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    handles = []
    for et in ['blue', 'black', 'green', 'red']:
        handles.append(plt.Line2D([0], [0], color=EDGE_COLORS[et], lw=3,
                                   label=EDGE_NAMES[et]))
    ax.legend(handles=handles, loc='lower right', fontsize=9,
              framealpha=0.95, edgecolor='#ccc')
    ax.set_xlim(-3, 52); ax.set_ylim(-14, 3)
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, fontsize=13, fontweight='bold', pad=10)


def tree_stats(tree, N, label):
    depths = [get_depth(tree, i) for i in range(N)]
    max_d = max(depths)
    fanout = [0] * N
    for i in range(N):
        if tree[i] >= 0: fanout[tree[i]] += 1
    max_f = max(fanout)
    etypes = {'blue': 0, 'black': 0, 'green': 0, 'red': 0}
    for i in range(N):
        if tree[i] >= 0: etypes[edge_type(tree[i], i)] += 1

    # Green congestion
    green_cong = {}
    uplink_load = [0] * N
    for i in range(N):
        if tree[i] < 0: continue
        p = tree[i]
        ig, ic, ir = dfly_group(i), dfly_chassis(i), dfly_router(i)
        pg, pc, pr = dfly_group(p), dfly_chassis(p), dfly_router(p)
        uplink_load[i] += 1; uplink_load[p] += 1
        if ig != pg:
            if ir != 0: green_cong[(ig,ic,0,ir)] = green_cong.get((ig,ic,0,ir),0) + 1
            if pr != 0: green_cong[(pg,pc,0,pr)] = green_cong.get((pg,pc,0,pr),0) + 1
        elif ic != pc:
            if ir != 0: green_cong[(ig,ic,0,ir)] = green_cong.get((ig,ic,0,ir),0) + 1
            if pr != 0: green_cong[(ig,pc,0,pr)] = green_cong.get((ig,pc,0,pr),0) + 1
        elif ir != pr:
            lo, hi = min(ir,pr), max(ir,pr)
            green_cong[(ig,ic,lo,hi)] = green_cong.get((ig,ic,lo,hi),0) + 1

    max_green = max(green_cong.values()) if green_cong else 0
    max_uplink = max(uplink_load)

    # Nodes with high fanout
    high_fan = [(i, fanout[i]) for i in range(N) if fanout[i] >= 4]

    print(f"\n{label}:")
    print(f"  depth={max_d}, max_fanout={max_f}, max_uplink={max_uplink}, max_green={max_green}")
    print(f"  edges: {etypes}")
    if high_fan:
        for nid, f in sorted(high_fan, key=lambda x: -x[1]):
            print(f"  high-fanout: {dfly_label(nid)} fanout={f}")
    return max_d, max_f, max_green, max_uplink


def print_tree(tree, root, N, max_indent=6):
    children = {i: [] for i in range(N)}
    for i in range(N):
        if tree[i] >= 0: children[tree[i]].append(i)
    for v in children: children[v].sort()

    def _print(v, indent):
        if indent > max_indent:
            n = _count_subtree(children, v) - 1
            if n > 0: print(" " * indent + f"... ({n} more)")
            return
        g, c, r, s = dfly_group(v), dfly_chassis(v), dfly_router(v), v % 2
        et = f" [{edge_type(tree[v], v)}]" if tree[v] >= 0 else ""
        n = _count_subtree(children, v) - 1
        sub = f" ({n} desc)" if n > 0 else ""
        print(" " * indent + f"(G{g},C{c},R{r},S{s}){et}{sub}")
        for ch in children[v]:
            _print(ch, indent + 2)

    _print(root, 0)


if __name__ == '__main__':
    root = 0
    N = 128

    chain = build_chain_tree(root, N)
    dissem = build_dissemination_tree(root, N)

    d_c, f_c, mg_c, mu_c = tree_stats(chain, N, "CHAIN (current best, 0.746ms)")
    print_tree(chain, root, N, max_indent=5)

    d_d, f_d, mg_d, mu_d = tree_stats(dissem, N, "DISSEMINATION (your idea)")
    print_tree(dissem, root, N, max_indent=8)

    # === Side-by-side logical trees ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 12))

    draw_logical_tree(chain, root, max_depth=5, ax=ax1,
        title=f'CHAIN split (current best)\n'
              f'depth={d_c}, max_fanout={f_c}, max_green={mg_c}, max_uplink={mu_c} → 0.746ms')
    draw_logical_tree(dissem, root, max_depth=7, ax=ax2,
        title=f'DISSEMINATION (your idea)\n'
              f'depth={d_d}, max_fanout={f_d}, max_green={mg_d}, max_uplink={mu_d}')

    fig.suptitle('Dragonfly BBS Tree Comparison\n'
                 'Node labels: (Group, Chassis, Router, Sibling)',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('dragonfly_compare_logical.png', dpi=150, bbox_inches='tight')
    print("\nSaved dragonfly_compare_logical.png")

    # === Side-by-side physical overlay ===
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(28, 18))

    draw_physical_overlay(chain, root, ax3, max_depth=5,
        title=f'CHAIN — depth={d_c}, fanout={f_c}, max_green={mg_c}, max_uplink={mu_c} → 0.746ms')
    draw_physical_overlay(dissem, root, ax4,
        title=f'DISSEMINATION — depth={d_d}, fanout={f_d}, max_green={mg_d}, max_uplink={mu_d}')

    fig2.suptitle('Physical Layout Comparison',
                  fontsize=15, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig('dragonfly_compare_physical.png', dpi=150, bbox_inches='tight')
    print("Saved dragonfly_compare_physical.png")

    plt.close('all')
