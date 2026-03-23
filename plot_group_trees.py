#!/usr/bin/env python3
"""
Visualise two anti-correlated binary spanning trees for the group level.

Construction (same idea as FatTree inter-leaf):
  T0: binary tree  parent(pos) = pos // 2
  T1: permute interior (swap upper/lower halves), then same binary formula.
  Union degree profile: [2, 4, 4, ..., 4, 2]

Usage:  python3 plot_group_trees.py <ngroups> [root_g]
"""
import sys, math
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def build_binary_trees(ngroups, root_g=0):
    leaf_g = ngroups - 1 if root_g != ngroups - 1 else ngroups - 2
    interior = sorted(g for g in range(ngroups) if g != root_g and g != leaf_g)
    ord_ = [root_g] + interior + [leaf_g]
    n = len(ord_)

    # T0: binary heap tree
    t0 = {}
    for pos in range(1, n):
        t0[ord_[pos]] = ord_[pos // 2]

    # Permutation: swap upper half (1..h) with lower half (h+1..2h)
    h = (n - 2) // 2
    sigma = list(range(n))
    for i in range(1, h + 1):
        j = i + h
        if j < n - 1:
            sigma[i], sigma[j] = sigma[j], sigma[i]

    # T1: permuted binary tree
    t1 = {}
    for pos in range(1, n):
        t1[ord_[sigma[pos]]] = ord_[sigma[pos // 2]]

    return t0, t1, root_g, leaf_g


def tree_depth(par, root):
    d = 0
    for node in par:
        cur, steps = node, 0
        while cur != root:
            cur = par[cur]; steps += 1
        d = max(d, steps)
    return d


def layout(par, root, ngroups):
    """BFS layered positions."""
    children = defaultdict(list)
    for c, p in par.items():
        children[p].append(c)
    for k in children:
        children[k].sort()

    depth = {}
    depth[root] = 0
    queue = [root]
    while queue:
        node = queue.pop(0)
        for c in children[node]:
            depth[c] = depth[node] + 1
            queue.append(c)

    by_layer = defaultdict(list)
    for node, d in depth.items():
        by_layer[d].append(node)

    max_w = max(len(v) for v in by_layer.values())
    pos = {}
    for layer, nodes in by_layer.items():
        w = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (w - 1) / 2.0) * (max_w / max(w, 1))
            y = -layer
            pos[node] = (x, y)
    return pos, children


def draw_tree(ax, par, root, ngroups, root_g, leaf_g, title):
    pos, children = layout(par, root, ngroups)

    # Draw edges
    for c, p in par.items():
        x0, y0 = pos[p]
        x1, y1 = pos[c]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    # Draw nodes
    for g in range(ngroups):
        if g not in pos:
            continue
        x, y = pos[g]
        if g == root_g:
            color = '#4CAF50'
        elif g == leaf_g:
            color = '#F44336'
        else:
            color = '#2196F3'
        circle = plt.Circle((x, y), 0.35, color=color, ec='black', lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, str(g), ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=6)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.autoscale()
    ax.margins(0.15)
    ax.axis('off')


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)

    ngroups = int(sys.argv[1])
    root_g = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    t0, t1, root_g, leaf_g = build_binary_trees(ngroups, root_g)

    # Union degrees
    deg = [0] * ngroups
    for c, p in t0.items(): deg[c] += 1; deg[p] += 1
    for c, p in t1.items(): deg[c] += 1; deg[p] += 1

    ok = all(d == 2 if g in (root_g, leaf_g) else d == 4
             for g, d in enumerate(deg))
    d0, d1 = tree_depth(t0, root_g), tree_depth(t1, root_g)

    print(f"ngroups={ngroups}  ingress={root_g}  egress={leaf_g}")
    print(f"T0 depth={d0}  T1 depth={d1}  (log2={math.log2(ngroups):.1f})")
    print(f"Union degrees: {deg}")
    print(f"[2,4,...,4,2]: {'PASS' if ok else 'FAIL'}")

    # Plot
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3,
        figsize=(5 * 3, max(4, (d0 + 1) * 1.5)))

    draw_tree(ax0, t0, root_g, ngroups, root_g, leaf_g, 'T0 (binary tree)')
    draw_tree(ax1, t1, root_g, ngroups, root_g, leaf_g, 'T1 (permuted binary)')

    # Degree bar chart
    nc = ['#4CAF50' if g == root_g else '#F44336' if g == leaf_g else '#2196F3'
          for g in range(ngroups)]
    ax2.bar(range(ngroups), deg, color=nc, edgecolor='black')
    ax2.set_xticks(range(ngroups))
    ax2.set_xlabel('Group')
    ax2.set_ylabel('Union degree')
    ax2.set_title('Union degree profile', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(deg) + 1)
    ax2.axhline(y=4, color='gray', ls='--', alpha=0.5, label='target=4')
    ax2.axhline(y=2, color='gray', ls=':', alpha=0.5, label='target=2')
    ax2.legend(fontsize=9)

    fig.suptitle(f'Binary Group Trees  (n={ngroups}, ingress={root_g}, '
                 f'egress={leaf_g}, depth={max(d0,d1)})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = f'group_trees_n{ngroups}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
