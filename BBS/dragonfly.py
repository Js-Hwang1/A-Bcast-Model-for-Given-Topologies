#!/usr/bin/env python3
"""
dragonfly.py — BBS spanning-tree construction for Dragonfly topologies.

Builds tau=3 spanning trees over the full compute+router graph where
every tree edge is exactly one physical link, giving exact contention
control via max_usage=2 on router-to-router links.

Node IDs:  compute 0..Nc-1,  routers Nc..Nc+Nr-1

Gateway pattern (matching the SimGrid XML):
  - Black links: router-0 in each chassis   (router-g-c1-0 ↔ router-g-c2-0)
  - Blue links:  router-0 of chassis-0       (router-g1-0-0 ↔ router-g2-0-0)
  - Green links: all-to-all within chassis

Usage:
    python3 BBS/dragonfly.py --cfg topo/Dragonfly/topo_128.cfg --root 0
"""

import argparse
import numpy as np
from collections import deque, Counter

# ── Constants ────────────────────────────────────────────────────────

COMPUTE = 0
ROUTER  = 1

LT_NODE_RTR = 0   # compute-node ↔ router
LT_GREEN    = 1   # intra-chassis router ↔ router
LT_BLACK    = 2   # inter-chassis (same group) router-0 ↔ router-0
LT_BLUE     = 3   # inter-group router-g-0-0 ↔ router-g-0-0

LT_NAME = {
    LT_NODE_RTR: "node-rtr",
    LT_GREEN:    "green",
    LT_BLACK:    "black",
    LT_BLUE:     "blue",
}


# ── Topology builder ─────────────────────────────────────────────────

def parse_cfg(path):
    """Parse 'dragonfly G C R Np' from config file."""
    with open(path) as f:
        parts = f.read().strip().split()
    assert parts[0] == "dragonfly", f"expected 'dragonfly', got '{parts[0]}'"
    return int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])


def build_dragonfly(G, C, R, Np):
    """Build full compute+router graph from dragonfly parameters.

    Returns a dict with all topology data.
    """
    Nc = G * C * R * Np          # compute nodes
    Nr = G * C * R               # routers
    N  = Nc + Nr                 # total

    def rid(g, c, r):
        """Router node ID for (group, chassis, router)."""
        return Nc + g * C * R + c * R + r

    # ── Node metadata ──
    ntype = np.zeros(N, dtype=np.int8)
    ntype[Nc:] = ROUTER

    rtr_of = np.zeros(Nc, dtype=np.int32)      # compute → its router
    rtr_gcr = {}                                 # router_id → (g,c,r)
    for g in range(G):
        for c in range(C):
            for r in range(R):
                rtr_gcr[rid(g, c, r)] = (g, c, r)
    for i in range(Nc):
        g = i // (C * R * Np)
        c = (i // (R * Np)) % C
        r = (i // Np) % R
        rtr_of[i] = rid(g, c, r)

    # ── Links ──
    neighbors = [[] for _ in range(N)]
    lid_mat  = np.zeros((N, N), dtype=np.int32)  # pair → link ID
    ltype    = {}                                  # lid → LT_*
    llat     = {}                                  # lid → latency (ns)
    next_lid = 1

    def add_link(a, b, lt, lat):
        nonlocal next_lid
        lid = next_lid; next_lid += 1
        lid_mat[a, b] = lid; lid_mat[b, a] = lid
        ltype[lid] = lt; llat[lid] = lat
        neighbors[a].append(b); neighbors[b].append(a)

    # 1. node ↔ router  (Nc links, 100 ns)
    for i in range(Nc):
        add_link(i, rtr_of[i], LT_NODE_RTR, 100)

    # 2. green: all router pairs in same chassis  (G*C*C(R,2) links, 100 ns)
    for g in range(G):
        for c in range(C):
            for r1 in range(R):
                for r2 in range(r1 + 1, R):
                    add_link(rid(g, c, r1), rid(g, c, r2), LT_GREEN, 100)

    # 3. black: router-0 of chassis pairs in same group  (G*C(C,2) links, 200 ns)
    for g in range(G):
        for c1 in range(C):
            for c2 in range(c1 + 1, C):
                add_link(rid(g, c1, 0), rid(g, c2, 0), LT_BLACK, 200)

    # 4. blue: router-g-0-0 between group pairs  (C(G,2) links, 400 ns)
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            add_link(rid(g1, 0, 0), rid(g2, 0, 0), LT_BLUE, 400)

    num_links = next_lid - 1
    return dict(N=N, Nc=Nc, Nr=Nr, G=G, C=C, R=R, Np=Np,
                ntype=ntype, rtr_of=rtr_of, rtr_gcr=rtr_gcr,
                neighbors=neighbors, lid_mat=lid_mat,
                ltype=ltype, llat=llat, num_links=num_links, rid=rid)


def node_label(topo, i):
    """Human-readable label for a node."""
    if topo['ntype'][i] == COMPUTE:
        return f"n{i}"
    g, c, r = topo['rtr_gcr'][i]
    return f"R{g}-{c}-{r}"


# ── Tree helpers ─────────────────────────────────────────────────────

def tree_depth(parent, N):
    maxd = 0
    for i in range(N):
        d = 0; v = i
        while parent[v] >= 0:
            v = parent[v]; d += 1
            if d > N: return N
        if d > maxd: maxd = d
    return maxd


# ── BBS tree construction ────────────────────────────────────────────

def bbs_compute_trees(topo, root, tau=3, max_usage_rr=2):
    """Build tau spanning trees on the full compute+router graph.

    max_usage_rr: max link sharing for router↔router links.
    node↔router links: max_usage = tau  (forced, unavoidable).
    """
    N   = topo['N']
    Nc  = topo['Nc']
    nbr = topo['neighbors']
    nt  = topo['ntype']
    lid_mat = topo['lid_mat']
    ltype   = topo['ltype']
    num_links = topo['num_links']
    max_lid = num_links + 2

    # ── per-link capacity ──
    link_cap = np.full(max_lid, max_usage_rr, dtype=np.int32)
    for lid, lt in ltype.items():
        if lt == LT_NODE_RTR:
            link_cap[lid] = tau            # all trees share these

    # ── BFS baseline depth ──
    vis = [False] * N
    bfs_par = [-1] * N
    bfs_order = []
    vis[root] = True
    q = deque([root])
    while q:
        u = q.popleft(); bfs_order.append(u)
        for v in nbr[u]:
            if not vis[v]:
                vis[v] = True; bfs_par[v] = u; q.append(v)
    depth0 = tree_depth(bfs_par, N)

    root_deg = len(nbr[root])
    tc = Counter(ltype.values())
    print(f"BBS-DF: N={N} (compute={Nc} routers={N-Nc})  "
          f"links={num_links} "
          f"(nr={tc[LT_NODE_RTR]} green={tc[LT_GREEN]} "
          f"black={tc[LT_BLACK]} blue={tc[LT_BLUE]})")
    print(f"BBS-DF: root={root} ({node_label(topo,root)}) deg={root_deg}  "
          f"tau={tau} max_usage_rr={max_usage_rr}  baseline_depth={depth0}")

    # ── edge usage ──
    edge_use = np.zeros(max_lid, dtype=np.int32)

    # ── no-parent set: non-root compute nodes (degree 1 → leaves only) ──
    no_par = set()
    for i in range(Nc):
        if i != root:
            no_par.add(i)

    # ── initialise trees ──
    trees = [[-1] * N for _ in range(tau)]

    # ── work list in BFS order, round-robin across trees ──
    unassigned = []
    ni = 0
    for v in bfs_order:
        if v == root:
            continue
        st = ni % tau
        for off in range(tau):
            unassigned.append((v, (st + off) % tau))
        ni += 1
    total_pairs = len(unassigned)

    # ── helpers ──
    def options(v, t):
        """Viable parents for node v in tree t."""
        opts = []
        for u in nbr[v]:
            if u != root and trees[t][u] < 0:  continue
            if u in no_par:                     continue
            lid = lid_mat[v, u]
            if lid > 0 and edge_use[lid] >= link_cap[lid]: continue
            opts.append((u, int(edge_use[lid]) if lid > 0 else 0))
        return opts

    def assign(v, t, par):
        trees[t][v] = par
        lid = lid_mat[v, par]
        if lid > 0: edge_use[lid] += 1

    # ── Phase A+B iterative solver ──
    for _ in range(N):
        if not unassigned: break

        # Phase A: constraint propagation (forced single-option pairs)
        changed = True
        while changed:
            changed = False
            rem = []
            for v, t in unassigned:
                opts = options(v, t)
                if len(opts) == 1:
                    assign(v, t, opts[0][0]); changed = True
                else:
                    rem.append((v, t))
            unassigned = rem

        # Phase B: greedy — pick lowest-usage edge, tie-break lowest ID
        nxt = []
        progress = False
        for v, t in unassigned:
            opts = options(v, t)
            if opts:
                best = min(opts, key=lambda x: (x[1], x[0]))
                assign(v, t, best[0]); progress = True
            else:
                nxt.append((v, t))
        if not progress and len(nxt) == len(unassigned):
            unassigned = nxt; break
        unassigned = nxt

    after_greedy = total_pairs - len(unassigned)
    print(f"BBS-DF: propagation+greedy: {after_greedy}/{total_pairs} assigned, "
          f"{len(unassigned)} remaining")

    # ── Repair: 2-deep chain reparenting ──
    if unassigned:
        # Index: which (node, tree) pairs currently use each link
        link_users = {}
        for t in range(tau):
            for i in range(N):
                if trees[t][i] < 0: continue
                lid = int(lid_mat[i, trees[t][i]])
                if lid > 0:
                    link_users.setdefault(lid, []).append((i, t))

        for depth in range(2):
            if not unassigned: break
            still = []
            for v, t in unassigned:
                resolved = False
                for u in nbr[v]:
                    if resolved: break
                    lid = int(lid_mat[v, u])
                    if lid <= 0 or edge_use[lid] < link_cap[lid]: continue
                    if u != root and trees[t][u] < 0:             continue
                    # lid saturated — try reparenting a user in another tree
                    for w, t2 in list(link_users.get(lid, [])):
                        if t2 == t or resolved: continue
                        old_p = trees[t2][w]
                        for alt in nbr[w]:
                            if alt == old_p or alt in no_par:     continue
                            if alt != root and trees[t2][alt] < 0: continue
                            # cycle check
                            cyc = False; a = alt
                            while a >= 0 and a != root:
                                if a == w: cyc = True; break
                                a = trees[t2][a]
                            if cyc: continue
                            alt_lid = int(lid_mat[w, alt])
                            if alt_lid > 0 and edge_use[alt_lid] >= link_cap[alt_lid]:
                                if depth < 1: continue
                                # depth-2: free alt_lid via chain
                                freed = False
                                for w2, t3 in list(link_users.get(alt_lid, [])):
                                    if t3 == t2 or freed: continue
                                    op2 = trees[t3][w2]
                                    for a2 in nbr[w2]:
                                        if a2 == op2 or a2 in no_par:     continue
                                        if a2 != root and trees[t3][a2] < 0: continue
                                        cy2 = False; x = a2
                                        while x >= 0 and x != root:
                                            if x == w2: cy2 = True; break
                                            x = trees[t3][x]
                                        if cy2: continue
                                        a2l = int(lid_mat[w2, a2])
                                        if a2l > 0 and edge_use[a2l] >= link_cap[a2l]:
                                            continue
                                        # chain reparent w2
                                        trees[t3][w2] = a2
                                        edge_use[alt_lid] -= 1
                                        if a2l > 0: edge_use[a2l] += 1
                                        lu = link_users.get(alt_lid, [])
                                        link_users[alt_lid] = [(n,tt) for n,tt in lu
                                                               if not (n==w2 and tt==t3)]
                                        link_users.setdefault(a2l, []).append((w2, t3))
                                        freed = True; break
                                    if freed: break
                                if not freed: continue
                            # reparent w in t2, then assign v in t
                            trees[t2][w] = alt
                            edge_use[lid] -= 1
                            if alt_lid > 0: edge_use[alt_lid] += 1
                            trees[t][v] = u
                            edge_use[lid] += 1
                            # update index
                            lu = link_users.get(lid, [])
                            link_users[lid] = [(n,tt) for n,tt in lu
                                               if not (n==w and tt==t2)]
                            link_users.setdefault(alt_lid, []).append((w, t2))
                            link_users.setdefault(lid, []).append((v, t))
                            resolved = True; break
                        if resolved: break
                if not resolved:
                    still.append((v, t))
            unassigned = still

    if unassigned:
        print(f"BBS-DF: WARNING — {len(unassigned)} unresolved pairs:")
        for v, t in unassigned[:20]:
            print(f"  node={v} ({node_label(topo,v)}) tree={t} "
                  f"nbrs=[{', '.join(node_label(topo,u) for u in nbr[v])}]")
    else:
        print(f"BBS-DF: All {total_pairs} pairs assigned")

    return tau, trees, edge_use, depth0


# ── Statistics ────────────────────────────────────────────────────────

def print_stats(topo, root, tau, trees, edge_use, depth0=None):
    N  = topo['N']
    Nc = topo['Nc']
    Nr = topo['Nr']
    lid_mat = topo['lid_mat']
    ltype   = topo['ltype']
    num_links = topo['num_links']
    max_lid = num_links + 2

    print(f"\n{'='*64}")
    print(f" Tree Statistics  (root={root}  {node_label(topo,root)})")
    print(f"{'='*64}")

    # ── per-tree breakdown ──
    for t in range(tau):
        d  = tree_depth(trees[t], N)
        rc = sum(1 for i in range(N) if trees[t][i] == root)
        by = Counter()
        for i in range(N):
            if trees[t][i] < 0: continue
            lid = int(lid_mat[i, trees[t][i]])
            if lid > 0 and lid in ltype:
                by[ltype[lid]] += 1
        parts = "  ".join(f"{LT_NAME[lt]}={by.get(lt,0)}"
                          for lt in [LT_NODE_RTR, LT_GREEN, LT_BLACK, LT_BLUE])
        print(f"  T{t}: depth={d:2d}  root_ch={rc:2d}  | {parts}")

    # ── link sharing by type ──
    print(f"\n  Link sharing (router↔router only):")
    for lt in [LT_GREEN, LT_BLACK, LT_BLUE]:
        lids = [lid for lid, l in ltype.items() if l == lt]
        usages = [int(edge_use[lid]) for lid in lids if lid < max_lid]
        dist = Counter(usages)
        mx = max(usages) if usages else 0
        viol = sum(1 for u in usages if u > 2)
        tag = "VIOLATION" if viol else "ok"
        print(f"    {LT_NAME[lt]:10s}: {len(lids):3d} links  "
              f"usage={dict(sorted(dist.items()))}  max={mx}  [{tag}]")

    # ── node-to-router (sanity: should all be tau) ──
    nr_lids = [lid for lid, l in ltype.items() if l == LT_NODE_RTR]
    nr_use  = [int(edge_use[lid]) for lid in nr_lids if lid < max_lid]
    nr_dist = Counter(nr_use)
    print(f"    {'node-rtr':10s}: {len(nr_lids):3d} links  "
          f"usage={dict(sorted(nr_dist.items()))}  (expected all={tau})")

    # ── blue link detail ──
    print(f"\n  Blue links:")
    blue_lids = sorted(lid for lid, l in ltype.items() if l == LT_BLUE)
    for lid in blue_lids:
        # find endpoints
        for a in range(N):
            for b in range(a + 1, N):
                if int(lid_mat[a, b]) == lid:
                    u = int(edge_use[lid])
                    # which trees use it?
                    tusers = []
                    for t in range(tau):
                        for i in range(N):
                            if trees[t][i] < 0: continue
                            if int(lid_mat[i, trees[t][i]]) == lid:
                                tusers.append(t); break
                    print(f"    {node_label(topo,a)} ↔ {node_label(topo,b)}"
                          f"  usage={u}/2  trees={tusers}")
                    break
            else: continue
            break

    # ── spanning check ──
    print(f"\n  Spanning:")
    all_ok = True
    for t in range(tau):
        cov = sum(1 for i in range(N) if trees[t][i] >= 0 or i == root)
        ok = (cov == N)
        if not ok: all_ok = False
        print(f"    T{t}: {cov}/{N} {'ok' if ok else 'INCOMPLETE'}")

    max_d = max(tree_depth(trees[t], N) for t in range(tau))
    bl = depth0 if depth0 is not None else "?"
    print(f"\n  Max depth across trees: {max_d}  (baseline BFS: {bl})")

    # ── budget summary ──
    rr_links  = sum(1 for l in ltype.values() if l != LT_NODE_RTR)
    rr_budget = rr_links * 2
    rr_needed = tau * (Nr - 1)
    print(f"\n  Budget: {rr_links} rr-links × 2 = {rr_budget} slots  "
          f"need {tau}×{Nr-1} = {rr_needed}  "
          f"{'feasible' if rr_budget >= rr_needed else 'INFEASIBLE'}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="BBS spanning-tree construction for Dragonfly topology")
    ap.add_argument("--cfg", required=True, help="Path to topo_*.cfg")
    ap.add_argument("--root", type=int, default=0, help="Root compute node")
    ap.add_argument("--tau", type=int, default=3)
    ap.add_argument("--max-usage", type=int, default=2,
                    help="Max usage for router↔router links")
    args = ap.parse_args()

    G, C, R, Np = parse_cfg(args.cfg)
    Nc = G * C * R * Np
    Nr = G * C * R
    print(f"Dragonfly: G={G} C={C} R={R} Np={Np}")
    print(f"  {Nc} compute + {Nr} routers = {Nc+Nr} nodes\n")

    topo = build_dragonfly(G, C, R, Np)

    root = args.root
    assert 0 <= root < Nc, f"root must be a compute node in [0, {Nc})"

    tau, trees, edge_use, depth0 = bbs_compute_trees(topo, root, args.tau, args.max_usage)
    print_stats(topo, root, tau, trees, edge_use, depth0)


if __name__ == "__main__":
    main()
