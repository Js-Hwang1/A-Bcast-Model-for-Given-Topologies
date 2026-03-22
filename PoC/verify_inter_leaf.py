#!/usr/bin/env python3
"""Verify anti-correlated inter-leaf binary trees for Nl=8,16,32,64.

Construction:
  T1: parent(p) = p/2  (standard binary tree, root has 1 child)
  T2: apply permutation sigma that swaps internal/leaf nodes, then
      use the same tree structure.

  sigma swaps A={1..h} (T1 internals) with B={h+1..2h} (T1 leaves),
  where h = Nl/2 - 1.  Node 0 (root) and Nl-1 (leaf) are fixed.

  The permutation uses non-involutory mappings to avoid creating
  duplicate undirected edges.
"""

def build_sigma(Nl):
    """Build non-involutory swap permutation for Nl nodes."""
    h = Nl // 2 - 1  # |A| = |B| = h

    sigma = list(range(Nl))  # identity

    # pi1: mapping for A -> B (swap last two positions)
    # A = {1, ..., h}, B = {h+1, ..., 2h}
    pi1 = list(range(1, h + 1))  # [1, 2, ..., h]
    if h >= 2:
        pi1[-1], pi1[-2] = pi1[-2], pi1[-1]  # swap last two

    # pi2: mapping for B -> A (swap first two positions)
    pi2 = list(range(1, h + 1))  # [1, 2, ..., h]
    if h >= 2:
        pi2[0], pi2[1] = pi2[1], pi2[0]  # swap first two

    for i in range(h):
        a = i + 1        # node in A
        b = h + pi1[i]   # destination in B
        sigma[a] = b

    for j in range(h):
        b = h + 1 + j    # node in B
        a = pi2[j]       # destination in A
        sigma[b] = a

    # sigma(0) = 0, sigma(Nl-1) = Nl-1 (already set by identity init)
    return sigma


def build_trees_general(Nl):
    """Build T1 and T2 using the sigma permutation approach."""
    # T1: standard binary tree
    T1 = [-1] * Nl
    for p in range(1, Nl):
        T1[p] = p // 2

    # Build sigma
    sigma = build_sigma(Nl)

    # T2: T2[sigma(p)] = sigma(T1[p]) for p >= 1
    T2 = [-1] * Nl
    for p in range(1, Nl):
        sp = sigma[p]
        sparent = sigma[T1[p]]
        T2[sp] = sparent

    return T1, T2


def union_degrees(T1, T2, Nl):
    from collections import defaultdict
    neighbors = defaultdict(set)
    for p in range(Nl):
        if T1[p] >= 0:
            neighbors[p].add(T1[p])
            neighbors[T1[p]].add(p)
        if T2[p] >= 0:
            neighbors[p].add(T2[p])
            neighbors[T2[p]].add(p)
    return [len(neighbors[i]) for i in range(Nl)]


def check_spanning(tree, Nl):
    visited = set()
    for n in range(Nl):
        cur = n
        path = []
        while cur != 0 and cur not in visited:
            if cur in path:
                return False, f"cycle at {cur}"
            path.append(cur)
            cur = tree[cur]
        visited.update(path)
        visited.add(0)
    return len(visited) == Nl, f"{len(visited)}/{Nl}"


def check_edge_disjoint(T1, T2, Nl):
    edges1 = set()
    edges2 = set()
    for p in range(1, Nl):
        edges1.add((min(p, T1[p]), max(p, T1[p])))
        edges2.add((min(p, T2[p]), max(p, T2[p])))
    overlap = edges1 & edges2
    return len(overlap) == 0, overlap


def check_binary(tree, Nl):
    """Check each node has at most 2 children."""
    from collections import Counter
    children = Counter()
    for p in range(Nl):
        if tree[p] >= 0:
            children[tree[p]] += 1
    max_children = max(children.values()) if children else 0
    violators = {n: c for n, c in children.items() if c > 2}
    return max_children <= 2, violators


for Nl in [8, 16, 32, 64]:
    print(f"\n{'='*60}")
    print(f"Nl = {Nl}")
    print(f"{'='*60}")

    T1, T2 = build_trees_general(Nl)

    ok1, msg1 = check_spanning(T1, Nl)
    ok2, msg2 = check_spanning(T2, Nl)
    print(f"T1 spanning: {msg1} {'OK' if ok1 else 'FAIL'}")
    print(f"T2 spanning: {msg2} {'OK' if ok2 else 'FAIL'}")

    bin1, v1 = check_binary(T1, Nl)
    bin2, v2 = check_binary(T2, Nl)
    print(f"T1 binary: {'OK' if bin1 else 'FAIL ' + str(v1)}")
    print(f"T2 binary: {'OK' if bin2 else 'FAIL ' + str(v2)}")

    disjoint, overlap = check_edge_disjoint(T1, T2, Nl)
    print(f"Edge-disjoint: {'YES' if disjoint else 'NO — overlap: ' + str(overlap)}")

    degs = union_degrees(T1, T2, Nl)
    from collections import Counter
    dist = Counter(degs)
    print(f"Degree distribution: {dict(sorted(dist.items()))}")
    print(f"Max union degree: {max(degs)}")
    print(f"Node 0 degree: {degs[0]}, Node {Nl-1} degree: {degs[Nl-1]}")

    # Check the desired pattern: [2, 4, 4, ..., 4, 2]
    ok_pattern = (degs[0] == 2 and degs[Nl-1] == 2 and
                  all(d == 4 for d in degs[1:Nl-1]))
    print(f"Pattern [2,4,...,4,2]: {'YES' if ok_pattern else 'NO'}")
    if not ok_pattern:
        bad = [(i, degs[i]) for i in range(Nl) if degs[i] != 4 and i != 0 and i != Nl-1]
        if bad:
            print(f"  Non-4 interior nodes: {bad}")

    if Nl <= 16:
        print(f"\nT1: {T1}")
        print(f"T2: {T2}")
        print(f"Union degrees: {degs}")
