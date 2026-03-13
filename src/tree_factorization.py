import cvxpy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque

def solve_O(A, E):
    """
    Solves for O_ij for an inputed adjacency matrix A and efficiency matrix E. Graph must be connected.
    Typically A and E will have nonzero elements in the same entries, so it should be possible to have the input be just A or E.
    Can also add a tol variable to allow flexibility
    """
    n = A.shape[0]
    O = cp.Variable((n, n))
    C = cp.Variable()

    constraints = []

    # Rate constraint
    for i in range(n):
        for j in range(n):
            if A[i, j] == 0:
                constraints.append(O[i, j] == 0)
            else: 
                constraints.append(O[i, j] >= 0)
                constraints.append(O[i, j] <= 1)

    # Row and column sum constraint
    for k in range(n):
        constraints.append(cp.sum(O[:, k]) <= 1)
        constraints.append(cp.sum(O[k, :]) <= 1)

    # Root constraint
    for k in range(n):
        constraints.append(O[k, 0] == 0)

    # Flow constraint maximization
    for j in range(n-1):
        constraints.append(cp.sum(cp.multiply(O[:, j+1], E[:, j+1])) == C)

    objective = cp.Maximize(C)

    # May want to break down C into a vector, may want the average C to be in the optimization
    # May also want to include maximizing the summation of O_ij, or the minimum of O_ij.
    # This constraint is most important in building the message passing schedule.
    # Need alternative constraints. This may force O_ij to be 0 when E_ij is sufficiently large, since maximizing the minimum C.
    # I would really like to penalize some sort of variance of O_ij, but may no longer be linear programming.
    
    # Other objectives:
    #objective = cp.Maximize(cp.min(C) + cp.min(O) - 0.5*cp.max(O)) # Maximize the minimum overall efficiency.
    #objective = cp.Maximize(cp.min(C)) # constraint sum E_ij O_ij >= C[j]
    #objective = cp.Maximize(C) # constraint sum E_ij O_ij >= C
    #objective = cp.Maximize(sum(C)) # constraint sum E_ij O_ij >= C[j]
    #objective = cp.Maximize(C or sum(C) + sum(O_ij))
    #objective = cp.Maximize(cp.min(C) + cp.min(O) - 0.5*cp.max(O)) # weighted sum

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    # Results
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("Warning: problem status =", prob.status)

    O_val = O.value if O.value is not None else np.zeros_like(A)
    C_val = C.value if C.value is not None else None

    return O_val, C_val

seed = 40
np.random.seed(seed=seed)
N = 10  # N x N mesh
n = N * N

G = nx.grid_2d_graph(N, N)
G = nx.convert_node_labels_to_integers(G)
E = nx.to_numpy_array(G, dtype=float)
A = nx.to_numpy_array(G, dtype=int)

O_s, C_s = solve_O(A, E)

'''print(O_s)
print(np.round(O_s, decimals = 3)[O_s>0.001])
print(C_s)'''

'''print("\nIncoming rates")
for j in range(n):
    print(sum(O_s[:, j]))

print("\nOutgoing rates")
for i in range(n):
    print(sum(O_s[i, :]))'''

def project_simplex(v):
    """
    Euclidean projection of v onto the probability simplex {x: x >= 0, sum x = 1}.
    (O(n log n) algorithm)
    """
    v = np.asarray(v).astype(float)
    if v.size == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u - (cssv - 1) / (np.arange(1, len(u) + 1)) > 0)[0]
    if rho.size == 0:
        # all entries projected to 0 except one
        theta = (cssv[-1] - 1) / len(u)
    else:
        rho = rho[-1]
        theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w

def flatten_matrix(M):
    return M.reshape((-1,), order='F')  # column-major like vec(M)

def unflatten_vec(v, n):
    return v.reshape((n, n), order='F')

def lmo_max_rooted_arborescence(residual_vec, n, root, allowed, eps=1):
    """
    LMO: max-weight rooted arborescence respecting allowed edges.
    Root has at most 1 outgoing edge.
    Adds eps edges only for connectivity (when a node would be disconnected).
    """
    R = unflatten_vec(residual_vec, n)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Pick best root->child edge
    best_root_child = None
    best_w = -np.inf
    for j in range(n):
        if j == root or not allowed[root, j]:
            continue
        w = float(R[root, j])
        if w > best_w:
            best_w = w
            best_root_child = j

    # Add all allowed edges except into root
    for i in range(n):
        for j in range(n):
            if i == j or j == root or not allowed[i,j]:
                continue
            if i == root:
                if best_root_child is None or j != best_root_child:
                    continue
            G.add_edge(i, j, weight=float(R[i,j]))

    # Add eps edges only for nodes with in-degree 0 (except root)
    for j in range(n):
        if j == root or G.in_degree(j) > 0:
            continue
        # pick any allowed parent other than root
        candidates = [i for i in range(n) if i != j and allowed[i,j] and i != root]
        if candidates:
            G.add_edge(candidates[0], j, weight=eps)

    # Compute maximum spanning arborescence
    Tbranch = nx.maximum_spanning_arborescence(G, attr="weight")
    T = np.zeros((n,n), dtype=int)
    for u,v in Tbranch.edges():
        T[u,v] = 1
    return T

def solve_qp_on_simplex(A, o, x0=None, max_iter=2000, tol=1e-8):
    """
    Solve min_{p in simplex} 0.5||o - A p||^2
    A: (d x m) matrix (columns are vec(T_k)), o: (d,)
    Returns p (m,)
    Uses projected gradient with step = 1/L where L = ||A||_2^2 (Lipschitz of grad).
    """
    d, m = A.shape
    if m == 0:
        return np.array([], dtype=float)
    if x0 is None:
        p = np.ones(m) / m
    else:
        p = project_simplex(x0)
    # Precompute
    ATA = A.T @ A
    ATy = A.T @ o
    # Lipschitz constant for grad = 2 * ||A||_2^2. We'll use L = 2 * sigma_max^2
    try:
        sigma_max = np.linalg.norm(A, ord=2)
    except Exception:
        sigma_max = np.sqrt(np.max(np.linalg.eigvals(ATA).real))
    L = 2.0 * (sigma_max ** 2) + 1e-12
    step = 1.0 / L
    prev_obj = None
    for it in range(max_iter):
        grad = 2 * (ATA @ p - ATy)   # gradient of ||o - A p||^2
        p = p - step * grad
        p = project_simplex(p)
        # objective
        r = o - A @ p
        obj = float(r.dot(r))
        if prev_obj is not None and abs(prev_obj - obj) < tol * max(1.0, prev_obj):
            break
        prev_obj = obj
    return p

def approximate_with_trees(O, allowed, root=0, K=10, max_iters=200, tol=1e-6, verbose=False):
    """
    Approximate matrix O (n x n) by convex combination of at most K rooted spanning trees.
    Returns:
      p_list: array of weights (m,) with m <= K
      T_list: list of binary adjacency matrices (n x n)
      recon: reconstructed matrix sum_k p_k T_k
      history: dict with 'objs' per iteration and 'residual_norms'
    Notes:
      - `root` is 0-based node index.
      - This uses fully-corrective steps (after adding a tree we re-solve for best p over current columns).
    """
    n = O.shape[0]
    assert O.shape == (n, n)
    o = flatten_matrix(O)
    columns = []      # will store vecs (d,)
    trees = []        # store adjacency matrices (n x n)
    objs = []
    residual_norms = []

    # Initial step: pick first tree from current residual = o
    for iteration in range(1, max_iters + 1):
        if iteration == 1:
            residual = o.copy()
        else:
            # compute residual from current A and p
            A = np.column_stack(columns) if len(columns) > 0 else np.zeros((n * n, 0))
            p_cur = p
            residual = o - A @ p_cur

        # LMO: find tree maximizing <residual, vec(T)>
        T_new = lmo_max_rooted_arborescence(residual, n, root, allowed)
        #R = unflatten_vec(residual, n)
        #allowed = O != 0
        #T_new = binary_arborescence(R, root, allowed)
        a_new = flatten_matrix(T_new)

        # Avoid adding duplicate columns (numerical equality)
        add_new = True
        for existing in columns:
            if np.array_equal(existing, a_new):
                add_new = False
                if verbose:
                    print("LMO returned duplicate tree; skipping add.")
                break

        if add_new:
            columns.append(a_new)
            trees.append(T_new)
        # Build A and solve best weights on simplex (fully-corrective)
        A = np.column_stack(columns)
        p = np.ones(len(columns)) / len(columns)
        #p = solve_qp_on_simplex(A, o)
        #recon_vec = A @ p

        recon_vec = np.mean(A, axis=1)
        
        rvec = o - recon_vec
        obj = float(rvec.dot(rvec))
        objs.append(obj)
        residual_norms.append(np.linalg.norm(rvec))

        if verbose:
            print(f"iter {iteration:3d}: m={len(columns):2d}, obj={obj:.6e}, ||res||={residual_norms[-1]:.6e}")

        # stopping criteria
        if len(columns) >= K:
            if verbose:
                print(f"Reached K={K} columns; stopping.")
            break
        if iteration > 1:
            # small improvement
            if abs(objs[-2] - objs[-1]) < tol * max(1.0, objs[-2]):
                if verbose:
                    print("Objective improvement below tol; stopping.")
                break

    # final return
    p_list = p
    T_list = trees
    recon = unflatten_vec((np.column_stack(columns) @ p_list), n)
    history = {"objs": objs, "residual_norms": residual_norms}
    return p_list, T_list, recon, history

def print_decomposition(O, p_list, T_list, precision=3):
    """
    Pretty-print:
      - O
      - each T_j
      - each p_j * T_j
      - reconstruction sum_j p_j T_j
      - residual O - reconstruction
    """
    np.set_printoptions(precision=precision, suppress=True)

    recon = np.zeros_like(O, dtype=float)

    for j, (p, T) in enumerate(zip(p_list, T_list)):
        print(f"\n---------------- T_{j} ----------------")
        print(T)

        degrees = np.sum(T != 0, axis=1)
        avg_deg = np.mean(degrees)
        max_deg = np.max(degrees)
        min_deg = np.min(degrees)

        print("\n--- Degree Statistics ---")
        print("Average degree:", avg_deg)
        print("Max degree:", max_deg)
        print("Min degree:", min_deg)

        weighted = p * T
        print(f"\n------------- p_{j} * T_{j} (p={p:.6f}) -------------")
        print(weighted)

        recon += weighted

    print("\n================ RECONSTRUCTION =================")
    print(recon)

    print("\n================ ORIGINAL O =================")
    print(O)

    residual = O - recon
    print("\n================ RESIDUAL (O - sum pT) =================")
    print(residual)

    print("\nFrobenius norm squared:", np.linalg.norm(residual)**2)

    mse = np.mean(residual**2)
    mae = np.mean(np.abs(residual))
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)

    mask = O != 0

    mse_nonzero = np.mean((residual[mask])**2)
    mae_nonzero = np.mean(np.abs(residual[mask]))

    print("MSE (nonzero O entries only):", mse_nonzero)
    print("MAE (nonzero O entries only):", mae_nonzero)

    high_mask = O > np.percentile(O[mask], 75)
    mae_high = np.mean(np.abs(O[high_mask] - recon[high_mask]))
    print("MAE (high O entries only):", mae_high)

    mean_nonzero = np.mean(np.abs(O[mask]))
    relative_mae = mae_nonzero / mean_nonzero
    print("Relative MAE:", relative_mae)

    rel_frob = np.linalg.norm(O - recon, 'fro') / np.linalg.norm(O, 'fro')
    print("Relative Frobenius error:", rel_frob)

    min_error_nonzero = np.min(residual[mask])
    max_error_nonzero = np.max(residual[mask])
    print("Min error (nonzero O entries):", min_error_nonzero)
    print("Max error (nonzero O entries):", max_error_nonzero)

def print_matrix_stats(O, eps=1e-8):
    O = np.asarray(O)

    print("========== MATRIX SUMMARY ==========")
    print("Shape:", O.shape)
    print("Total entries:", O.size)
    print("Nonzero entries:", np.sum(np.abs(O) > eps))
    print("Zero entries:", np.sum(np.abs(O) <= eps))

    flat = O.ravel()

    print("\n--- Statistics ---")
    print("Min:", np.min(flat))
    print("Max:", np.max(flat))
    print("Mean:", np.mean(flat))
    print("Std:", np.std(flat))
    print("Median:", np.median(flat))

    print("\n--- Quartiles ---")
    q1 = np.percentile(flat, 25)
    q2 = np.percentile(flat, 50)
    q3 = np.percentile(flat, 75)
    q90 = np.percentile(flat, 90)
    q95 = np.percentile(flat, 95)
    q99 = np.percentile(flat, 99)

    print("Q1 (25%):", q1)
    print("Median (50%):", q2)
    print("Q3 (75%):", q3)
    print("90th percentile:", q90)
    print("95th percentile:", q95)
    print("99th percentile:", q99)

    print("===================================\n")

def plot_trees(T_list, p_list, N, root=0):
    """
    Plot K trees in a 2x4 grid. Each subplot shows one spanning tree
    laid out on the NxN grid positions.
    """
    K = len(T_list)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Grid positions: node i -> (col, row) so it looks like a grid
    pos = {i: (i % N, N - 1 - i // N) for i in range(N * N)}

    for k in range(min(K, 8)):
        ax = axes[k]
        T = T_list[k]

        # Build a directed graph from the tree adjacency matrix
        T_graph = nx.DiGraph()
        T_graph.add_nodes_from(range(T.shape[0]))
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if T[i, j] == 1:
                    T_graph.add_edge(i, j)

        # Draw edges
        nx.draw_networkx_edges(T_graph, pos, ax=ax, edge_color='black',
                               arrows=True, arrowsize=8, width=1.5, alpha=0.9,
                               arrowstyle='->', min_source_margin=3, min_target_margin=3)

        # Draw nodes
        node_colors = ['red' if i == root else 'skyblue' for i in range(T.shape[0])]
        nx.draw_networkx_nodes(T_graph, pos, ax=ax, node_size=30,
                               node_color=node_colors, edgecolors='black', linewidths=0.5)

        ax.set_title(f"Tree {k+1}  (p={p_list[k]:.4f})", fontsize=11)
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide unused subplots if K < 8
    for k in range(K, 8):
        axes[k].axis('off')

    fig.suptitle(f"Tree Factorization: {K} Spanning Trees on {N}x{N} Grid", fontsize=14)
    plt.tight_layout()
    plt.savefig("tree_factorization.png", dpi=150, bbox_inches='tight')
    plt.show()

#print(sum(O_s))
eps = 1e-6
O_s[np.abs(O_s) < eps] = 0
allowed = A == 1
#print(sum(O_s))
#print(O_s[O_s>0])
root = 0

'''print("\nTest 1")
K = 8

p, trees, recon, history = approximate_with_trees(O_s, allowed, root=root, K=K, verbose=True)
print_decomposition(O_s, p, trees)

plot_trees(trees, p, N)
'''

'''print("\nTest 2")
K = 8
p, trees, recon, history = approximate_with_trees(O_s, allowed, root=root, K=K, verbose=True)
print_decomposition(O_s, p, trees)
'''

'''print_matrix_stats(O_s)

nonzero = np.abs(O_s) > eps
stats_nonzero = O_s[nonzero]
print_matrix_stats(stats_nonzero)'''




def _directed_reachable(edges, src, tgt, n):
    """
    Return True if tgt is reachable from src following directed edges.
    edges: list of (u,v) directed edges present in the partial tree.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    seen = [False]*n
    dq = deque([src])
    seen[src] = True
    while dq:
        x = dq.popleft()
        for nb in adj[x]:
            if not seen[nb]:
                if nb == tgt:
                    return True
                seen[nb] = True
                dq.append(nb)
    return False

def build_k_trees_round_robin_directed(O, k, root=0, max_children=2, seed=None,
                                       backtrack_limit=2000, verbose=False):
    """
    Build k directed spanning trees (arborescences) respecting directed O>0 topology.
    - O: (n,n) numpy array (directed). O[i,j] > 0 means directed edge i->j allowed.
    - k: number of trees
    - root: root node index (root must have exactly 1 outgoing child per tree)
    - max_children: max out-degree for non-root nodes (hard constraint)
    Returns: trees (list of lists of (u,v)), stats dict
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    O = np.asarray(O, dtype=float)
    if O.ndim != 2 or O.shape[0] != O.shape[1]:
        raise ValueError("O must be square (n,n).")
    n = O.shape[0]

    allowed = (O > 0)  # boolean directed mask
    if not allowed.any():
        raise ValueError("No allowed directed edges in O > 0.")

    # Normalize O so expected edges per tree sum to n-1 (optional but helps scoring)
    sumO = O.sum()
    if sumO <= 0:
        raise ValueError("Sum of O must be > 0.")
    desired_sum = n - 1
    O = O * (desired_sum / sumO)
    desired_counts = k * O  # float desired counts per directed edge

    # bookkeeping
    usage = defaultdict(int)                # usage[(u,v)] integer across all trees
    trees = [[] for _ in range(k)]          # each tree: list of directed (u,v)
    nodes_in_tree = [set([root]) for _ in range(k)]
    outdeg = [defaultdict(int) for _ in range(k)]
    in_nodes = [set([root]) for _ in range(k)]  # set of nodes present in tree

    # root has hard out-degree 1
    outdeg_limit = [defaultdict(lambda: max_children) for _ in range(k)]
    for t in range(k):
        outdeg_limit[t][root] = 1

    incomplete = set(range(k))
    iters = 0
    max_iters = k * (n - 1) * 20
    backtracks = 0

    while incomplete and iters < max_iters:
        progressed = False
        for t in list(incomplete):
            if len(trees[t]) == n - 1:
                incomplete.discard(t)
                continue
            iters += 1

            # build candidate directed edges: u in tree, v not in tree, allowed[u,v],
            # outdeg[u] < limit, and adding (u->v) doesn't create a directed cycle.
            candidates = []
            for u in list(nodes_in_tree[t]):
                if outdeg[t][u] >= outdeg_limit[t].get(u, max_children):
                    continue
                for v in range(n):
                    if v in nodes_in_tree[t]:
                        continue
                    if not allowed[u, v]:
                        continue
                    # don't create directed cycle: check if v reaches u already in current tree
                    if _directed_reachable(trees[t], v, u, n):
                        continue
                    key = (u, v)
                    rem = desired_counts[u, v] - usage[key]  # prefer edges with positive remaining desire
                    score = (rem, O[u, v], -usage[key])
                    candidates.append((score, u, v))

            if not candidates:
                # stuck: try small backtrack on other incomplete trees to free usages
                victim = None
                for tt in range(k):
                    if tt == t or len(trees[tt]) == 0 or len(trees[tt]) == n - 1:
                        continue
                    victim = tt
                    break
                if victim is None:
                    # no victim to backtrack -> cannot proceed for this tree; mark as stuck
                    if verbose:
                        print(f"[warn] tree {t} stuck with no candidates and no victims to backtrack.")
                    incomplete.discard(t)
                    continue
                # pop last edge from victim
                u_rem, v_rem = trees[victim].pop()
                usage[(u_rem, v_rem)] -= 1
                nodes_in_tree[victim].remove(v_rem)
                outdeg[victim][u_rem] -= 1
                backtracks += 1
                if verbose:
                    print(f"[backtrack] removed {(u_rem, v_rem)} from tree {victim} to free capacity.")
                # after backtracking, we'll continue and re-attempt t in next loop iteration
                continue

            # choose best candidate (greedy)
            candidates.sort(reverse=True, key=lambda x: x[0])
            chosen = None
            for _, u, v in candidates:
                key = (u, v)
                # Double-check cycle safety once more (race condition unlikely but safe)
                if _directed_reachable(trees[t], v, u, n):
                    continue
                # commit
                trees[t].append((u, v))
                nodes_in_tree[t].add(v)
                in_nodes[t].add(v)
                outdeg[t][u] += 1
                usage[key] += 1
                progressed = True
                break

        if not progressed:
            # If no tree progressed in this outer pass, break to avoid infinite loop
            if verbose:
                print("No progress made in an entire pass; stopping build loop.")
            break

    # final checks & repairs
    for t in range(k):
        # verify only allowed edges used
        for u, v in trees[t]:
            if not allowed[u, v]:
                raise RuntimeError(f"Tree {t} used disallowed directed edge {u}->{v}")

    # verify spanning: attempt to connect remaining nodes for incomplete trees using any allowed edges
    # (This is a fallback step: will not add disallowed edges.)
    for t in range(k):
        if len(trees[t]) < n - 1:
            # try greedy fill with any remaining legal edges (still disallow cycles)
            while len(trees[t]) < n - 1:
                added = False
                for u in list(nodes_in_tree[t]):
                    if outdeg[t][u] >= outdeg_limit[t].get(u, max_children):
                        continue
                    for v in range(n):
                        if v in nodes_in_tree[t]:
                            continue
                        if not allowed[u, v]:
                            continue
                        if _directed_reachable(trees[t], v, u, n):
                            continue
                        trees[t].append((u, v))
                        nodes_in_tree[t].add(v)
                        outdeg[t][u] += 1
                        usage[(u, v)] += 1
                        added = True
                        break
                    if added:
                        break
                if not added:
                    # cannot finish this tree
                    if verbose:
                        print(f"[warning] Could not produce full spanning tree for tree {t}.")
                    break

    # compute stats
    final_usage = dict(usage)
    stats = {
        "usage": final_usage,
        "iterations": iters,
        "backtracks": backtracks,
        "built_counts": [len(tr) for tr in trees]
    }
    return trees, stats

# ---------- Printing and plotting helpers ----------
def print_trees_readable(trees, n, root=0):
    """Print trees as children lists and adjacency matrices for visual inspection."""
    for idx, T in enumerate(trees):
        print(f"\n--- Tree {idx+1} (edges={len(T)}) ---")
        children = defaultdict(list)
        for u, v in T:
            children[u].append(v)
        # print children list for each node that has children
        for node in range(n):
            if node in children:
                print(f"{node} -> {children[node]}")
        # adjacency matrix
        A = np.zeros((n, n), dtype=int)
        for u, v in T:
            A[u, v] = 1
        print("Adjacency matrix (rows = parent u, cols = child v):")
        print(A)

def plot_trees_on_grid(trees, p_list=None, N=None, root=0, vmax_plots=8, figsize=(18,8)):
    """Simple plotting (up to 8 trees) on NxN grid using networkx. Trees are directed edge lists."""
    K = len(trees)
    show_k = min(K, vmax_plots)
    n = max(max((max(u, v) for u,v in T), default=0) for T in trees) + 1
    if N is None:
        sr = int(np.sqrt(n))
        N = sr if sr*sr == n else n
    pos = {i: (i % N, (N - 1) - (i // N)) for i in range(n)}
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for k in range(show_k):
        ax = axes[k]
        Gt = nx.DiGraph()
        Gt.add_nodes_from(range(n))
        Gt.add_edges_from(trees[k])
        nx.draw_networkx_edges(Gt, pos, ax=ax, edge_color='black', arrows=True,
                               arrowsize=8, width=1.2, connectionstyle='arc3,rad=0.0')
        node_colors = ['red' if i == root else 'skyblue' for i in range(n)]
        nx.draw_networkx_nodes(Gt, pos, ax=ax, node_size=50, node_color=node_colors,
                               edgecolors='black', linewidths=0.4)
        title = f"Tree {k+1}"
        if p_list is not None:
            title += f" (p={p_list[k]:.4f})"
        ax.set_title(title)
        ax.set_aspect('equal'); ax.axis('off')
    for kk in range(show_k, rows*cols):
        axes[kk].axis('off')
    plt.tight_layout()
    plt.show()

# ---------------- Example usage ----------------
k = 8
trees, stats = build_k_trees_round_robin_directed(O_s, k, root=0, max_children=2, seed=42, verbose=True)
print("stats:", stats)
print_trees_readable(trees, n, root=0)

# uniform p_list for plotting, or supply real weights
p_list = [1.0/k]*k
plot_trees_on_grid(trees, p_list=p_list, N=N, root=0)

import numpy as np

def compute_reconstruction_error(O, trees, p_list=None, verbose=True):
    """
    Compute reconstruction error of O from list of directed trees.
    - O: original (n,n) matrix
    - trees: list of edge lists [(u,v), ...]
    - p_list: optional list of tree weights (default uniform)
    Returns:
      error_dict = {
        'error_matrix': E,
        'MSE_nonzero': mse,
        'MAE_nonzero': mae,
        'max_error': max_e,
        'min_error': min_e,
        'mean_error': mean_e
      }
    """
    n = O.shape[0]
    K = len(trees)
    if p_list is None:
        p_list = [1.0/K]*K
    # Build weighted reconstruction
    O_hat = np.zeros((n, n))
    for t, T in enumerate(trees):
        A = np.zeros((n, n))
        for u, v in T:
            A[u, v] = 1
        O_hat += p_list[t] * A

    # Compute error
    E = O - O_hat
    mask = O > 1e-4  # consider only originally nonzero edges
    E_nonzero = E[mask]
    mse = np.mean(E_nonzero**2)
    mae = np.mean(np.abs(E_nonzero))
    max_e = np.max(E_nonzero)
    min_e = np.min(E_nonzero)

    if verbose:
        print("Reconstruction error stats (only O>0):")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  max error: {max_e:.4f}")
        print(f"  min error: {min_e:.4f}")

    return {
        "error_matrix": E,
        "MSE_nonzero": mse,
        "MAE_nonzero": mae,
        "max_error": max_e,
        "min_error": min_e,
        "O_hat": O_hat
    }

error_stats = compute_reconstruction_error(O_s, trees, p_list=p_list)
E = error_stats["error_matrix"]

# Optional: print the full error matrix
print("\nError matrix O - sum(p*T):")
np.set_printoptions(precision=3, suppress=True)
print(E)