#!/usr/bin/env python3
"""Generate BBS-specific hostfiles for FatTree topologies.

BBS uses two anti-correlated inter-leaf binary trees, each with Nl-1
edges.  Each edge gets its own spine MPI rank (total 2*(Nl-1)).

Edge coloring assigns each spine rank to one of 16 physical spine
hosts, ensuring no two edges at the same abstract leaf share a spine
(avoids up/down link contention).

Usage: python3 gen_bbs_hostfile.py <Nc> [npl=16]
Output: hostfile content to stdout

NP = Nc + Nl + 2*(Nl-1)  where Nl = Nc / npl
"""

import sys


def build_sigma(Nl):
    """Non-involutory permutation matching C ft_inter_node_of()."""
    h = Nl // 2 - 1
    sigma = list(range(Nl))
    pi1 = list(range(1, h + 1))
    if h >= 2:
        pi1[-1], pi1[-2] = pi1[-2], pi1[-1]
    pi2 = list(range(1, h + 1))
    if h >= 2:
        pi2[0], pi2[1] = pi2[1], pi2[0]
    for i in range(h):
        sigma[i + 1] = h + pi1[i]
    for j in range(h):
        sigma[h + 1 + j] = pi2[j]
    return sigma


def edge_color(Nl, sigma):
    """Greedy edge coloring of the union of T1 and T2 inter-leaf trees.

    Returns list of 2*(Nl-1) colors, indexed as:
        [0..Nl-2]      = T0 edges (al=1..Nl-1)
        [Nl-1..2*Nl-3] = T1 edges (al=1..Nl-1)
    """
    inv_sigma = [0] * Nl
    for i, s in enumerate(sigma):
        inv_sigma[s] = i

    used = [0] * Nl  # bitmask of colors used at each abstract leaf
    colors = []

    # T0: parent(al) = al // 2
    for al in range(1, Nl):
        parent_al = al // 2
        blocked = used[parent_al] | used[al]
        c = 0
        while blocked & (1 << c):
            c += 1
        colors.append(c)
        used[parent_al] |= (1 << c)
        used[al] |= (1 << c)

    # T1: parent via sigma permutation
    for al in range(1, Nl):
        pos = inv_sigma[al]
        parent_pos = pos // 2
        parent_al = sigma[parent_pos]
        blocked = used[parent_al] | used[al]
        c = 0
        while blocked & (1 << c):
            c += 1
        colors.append(c)
        used[parent_al] |= (1 << c)
        used[al] |= (1 << c)

    return colors


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <Nc> [npl=16]", file=sys.stderr)
        sys.exit(1)

    Nc = int(sys.argv[1])
    npl = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    Nl = Nc // npl

    if Nl < 2 or Nc % npl != 0:
        print(f"ERROR: Nc={Nc} not divisible by npl={npl} or Nl<2",
              file=sys.stderr)
        sys.exit(1)

    Ns = 2 * (Nl - 1)
    NP = Nc + Nl + Ns

    sigma = build_sigma(Nl)
    colors = edge_color(Nl, sigma)

    max_color = max(colors)
    print(f"# BBS FatTree hostfile: Nc={Nc} Nl={Nl} Ns={Ns} NP={NP} "
          f"max_spine_color={max_color}", file=sys.stderr)

    # Compute nodes
    for i in range(Nc):
        print(f"node-{i}")

    # Leaf switch hosts
    for i in range(Nl):
        print(f"leaf-{i}")

    # Spine ranks — each mapped to physical spine by edge coloring
    for c in colors:
        print(f"spine-{c}")


if __name__ == "__main__":
    main()
