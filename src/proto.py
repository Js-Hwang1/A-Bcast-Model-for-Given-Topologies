import math
from collections import defaultdict
import networkx as nx
from networkx.algorithms.tree.branchings import minimum_spanning_arborescence
import matplotlib.pyplot as plt


# -----------------------------
# 1) Rooted min-cost arborescence oracle (Chu–Liu/Edmonds via NetworkX)
# -----------------------------
def min_cost_out_arborescence_rooted(G: nx.DiGraph, root, cost_attr: str = "cost") -> nx.DiGraph:
    """
    Returns a minimum-cost *spanning out-arborescence rooted at `root`*.
    Uses NetworkX's minimum_spanning_arborescence on an augmented graph with a super-root.

    Preconditions:
      - Every node is reachable from `root` in G (otherwise no spanning out-arborescence exists).
      - G edges have attribute `cost_attr` for costs.
    """
    # Copy graph to avoid mutating user object
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        H.add_edge(u, v, **data)

    # Add a super-root that has NO incoming edges, and a single outgoing edge to `root`.
    super_root = ("__super_root__", root)
    H.add_node(super_root)
    H.add_edge(super_root, root, **{cost_attr: 0.0})

    # Min spanning arborescence on H will have super_root as the unique root
    B = minimum_spanning_arborescence(H, attr=cost_attr)

    # Remove super-root; result should be rooted at `root`
    B2 = B.copy()
    B2.remove_node(super_root)

    if not nx.is_arborescence(B2):
        raise RuntimeError("Oracle returned a non-arborescence after stripping super-root.")

    roots = [n for n in B2.nodes() if B2.in_degree(n) == 0]
    if roots != [root]:
        raise RuntimeError(f"Oracle root mismatch: got {roots}, expected [{root}]")

    return B2


# -----------------------------
# 2) MWU arborescence packing
# -----------------------------
def mwu_arborescence_packing(
    G: nx.DiGraph,
    root,
    capacity: dict,          # (u,v) -> cap > 0
    C: int = 50,             # number of iterations / max number of trees in support
    eps: float = 0.2,        # MWU parameter
    eta: float | None = None,# fraction of bottleneck to push each iter (keeps feasibility)
    cost_on_residual: bool = False,  # if True, cost = price / residual (more "greedy on residual")
    tol: float = 1e-12,
):
    """
    Returns:
      trees: list of (edges_list, weight) with merged duplicates
      total_value: sum of weights
      used: dict (u,v) -> used capacity (<= capacity[(u,v)])
    """
    if eta is None:
        eta = min(0.5, eps)  # conservative default; larger eta packs faster but may be less stable

    # Work on the subgraph of positive-capacity edges
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        cap = capacity.get((u, v), 0.0)
        if cap > 0:
            H.add_edge(u, v)

    # Feasibility: must reach all nodes from root for an out-arborescence to exist
    reachable = nx.descendants(H, root) | {root}
    if len(reachable) != H.number_of_nodes():
        missing = list(set(H.nodes()) - reachable)
        raise ValueError(f"No spanning out-arborescence rooted at {root}; unreachable nodes exist, e.g. {missing[:10]}")

    # MWU prices and used capacities
    price = {(u, v): 1.0 for (u, v) in H.edges()}
    used = {(u, v): 0.0 for (u, v) in H.edges()}

    # store per-iteration tree and its pushed amount
    raw_packing = []

    for t in range(C):
        # (Step 3) Set costs for the oracle
        for u, v in H.edges():
            cap = capacity[(u, v)]
            resid = cap - used[(u, v)]
            denom = resid if cost_on_residual else cap
            denom = max(denom, tol)
            H[u][v]["cost"] = price[(u, v)] / denom

        # Extract minimum-cost spanning out-arborescence rooted at root
        T = min_cost_out_arborescence_rooted(H, root, cost_attr="cost")
        tree_edges = list(T.edges())

        # (Step 2) Push bottleneck residual (scaled by eta) and subtract from residual
        bottleneck = min(capacity[e] - used[e] for e in tree_edges)
        if bottleneck <= tol:
            break  # nothing left to pack

        delta = eta * bottleneck

        # MWU update on edges used by the tree
        for e in tree_edges:
            cap = capacity[e]
            used[e] += delta
            frac = delta / cap  # normalized usage
            # multiplicative increase (discourages repeatedly using the same tight edges)
            price[e] *= (1.0 + eps) ** frac

        # record this tree (canonicalize edge set so duplicates can be merged)
        raw_packing.append((tuple(sorted(tree_edges)), delta))

    # merge duplicates
    merged = defaultdict(float)
    for edges, w in raw_packing:
        merged[edges] += w

    trees = [(list(edges), w) for edges, w in merged.items()]
    total_value = sum(w for _, w in trees)

    # sanity: feasibility
    for e, u in used.items():
        if u - capacity[e] > 1e-9:
            raise RuntimeError(f"Capacity violated on edge {e}: used={u}, cap={capacity[e]}")

    return trees, total_value, used


# -----------------------------
# 3) Example: 5x5 grid as a directed graph (both directions) with unit capacities
# -----------------------------
def grid_5x5_digraph_unit_caps():
    coords = [(x, y) for x in range(-2, 3) for y in range(-2, 3)]
    G = nx.DiGraph()
    G.add_nodes_from(coords)

    for (x, y) in coords:
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx_, ny_ = x + dx, y + dy
            if (nx_, ny_) in set(coords):
                G.add_edge((x, y), (nx_, ny_))

    cap = {(u, v): 1.0 for (u, v) in G.edges()}  # unit capacities
    return G, cap


if __name__ == "__main__":
    G, cap = grid_5x5_digraph_unit_caps()
    P = (1, 1)  # your root example

    trees, value, used = mwu_arborescence_packing(
        G, P, cap,
        C=8,           # <= 8 trees in support (at most)
        eps=0.25,
        eta=0.5,       # push 50% of bottleneck each round
        cost_on_residual=False
    )

    print("Packed value =", value)
    print("#trees in support =", len(trees))
    for i, (edges, w) in enumerate(trees, 1):
        print(f"Tree {i}: weight={w:.6f}, edges={len(edges)}")

    # --- Plot trees in a 2x4 grid ---
    K = len(trees)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Node positions: use the (x, y) coordinates directly
    pos = {node: node for node in G.nodes()}

    for k in range(min(K, 8)):
        ax = axes[k]
        edge_list, weight = trees[k]

        T_graph = nx.DiGraph()
        T_graph.add_nodes_from(G.nodes())
        T_graph.add_edges_from(edge_list)

        nx.draw_networkx_edges(T_graph, pos, ax=ax, edge_color='black',
                               arrows=True, arrowsize=8, width=1.5, alpha=0.9,
                               arrowstyle='->', min_source_margin=3, min_target_margin=3)

        node_colors = ['red' if n == P else 'skyblue' for n in T_graph.nodes()]
        nx.draw_networkx_nodes(T_graph, pos, ax=ax, node_size=80,
                               node_color=node_colors, edgecolors='black', linewidths=0.5)

        ax.set_title(f"Tree {k+1}  (w={weight:.4f})", fontsize=11)
        ax.set_aspect('equal')
        ax.axis('off')

    for k in range(K, 8):
        axes[k].axis('off')

    fig.suptitle(f"MWU Arborescence Packing: {K} Trees on 5x5 Grid", fontsize=14)
    plt.tight_layout()
    plt.savefig("proto_trees.png", dpi=150, bbox_inches='tight')
    plt.show()
