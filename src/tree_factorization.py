import cvxpy as cp
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from networkx.generators import fast_gnp_random_graph

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

# --- Random bipartite ---
'''seed = 40
np.random.seed(seed=seed)
n1 = 6
n2 = 5
p = 0.8'''
seed = 40
np.random.seed(seed=seed)
n1 = 15
n2 = 15
p = 0.9

G = bipartite.random_graph(n1, n2, p, seed=seed)

n = 128
G = fast_gnp_random_graph(n, 0.05)
E = nx.to_numpy_array(G, dtype=float)

'''a = 0.8
b = 1/a
E = E * np.random.uniform(a, b, size=E.shape)'''
A = nx.to_numpy_array(G, dtype=int)

O_s, C_s = solve_O(A, E)
print(O_s)
print(np.round(O_s, decimals = 3)[O_s>0.001])
print(C_s)

print("\nIncoming rates")
for j in range(n):
    print(sum(O_s[:, j]))

print("\nOutgoing rates")
for i in range(n):
    print(sum(O_s[i, :]))


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

def lmo_max_rooted_arborescence(residual_vec, n, root):
    """
    residual_vec: vectorized residual of shape (n*n,)
    n: number of nodes
    root: integer in 0..n-1 (0-based)
    Returns binary adjacency matrix T (n x n) with 1 where edge i->j selected.
    """

    R = unflatten_vec(residual_vec, n)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if j == root:
                # forbid edges into root so root becomes the tree root
                continue
            G.add_edge(i, j, weight=float(R[i, j]))

    # New NetworkX 3.x API
    Tbranch = nx.maximum_spanning_arborescence(G, attr="weight")

    T = np.zeros((n, n), dtype=int)
    for u, v in Tbranch.edges():
        T[u, v] = 1

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

def approximate_with_trees(O, root=0, K=10, max_iters=200, tol=1e-6, verbose=False):
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
        T_new = lmo_max_rooted_arborescence(residual, n, root)
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
        p = solve_qp_on_simplex(A, o)
        # compute objective
        recon_vec = A @ p
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

print(sum(O_s))
eps = 1e-6
O_s[np.abs(O_s) < eps] = 0.0
print(sum(O_s))
print(O_s[O_s>0])
root = 0

''''print("\nTest 1")
K = 4

p, trees, recon, history = approximate_with_trees(O_s, root=root, K=K, verbose=True)
print_decomposition(O_s, p, trees)'''

print("\nTest 2")
K = 8
p, trees, recon, history = approximate_with_trees(O_s, root=root, K=K, verbose=True)
print_decomposition(O_s, p, trees)

def sample_weighted_rooted_tree(O, root, congestion, alpha):
    """
    Sample one rooted spanning arborescence.
    
    O           : (n x n) matrix of base communication rates
    root        : root node (0-based)
    congestion  : (n x n) matrix counting how many times edges used
    alpha       : congestion penalty strength
    """

    n = O.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Effective weights with congestion penalty
    w_eff = O * np.exp(-alpha * congestion)

    # Exponential clock sampling
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if j == root:
                continue  # forbid incoming to root
            if w_eff[i, j] <= 0:
                continue

            # Exponential random variable
            e = np.random.exponential(1.0)

            # Randomized score (bigger weight more likely)
            score = w_eff[i, j] / e

            G.add_edge(i, j, weight=float(score))

    # Maximum spanning arborescence
    Tbranch = nx.maximum_spanning_arborescence(G, attr="weight")

    T = np.zeros((n, n), dtype=int)
    for u, v in Tbranch.edges():
        T[u, v] = 1

    return T


def build_diverse_tree_pack(O, root=0, K=8, alpha=2.0):
    """
    Build K diverse rooted trees.
    
    alpha: larger → stronger penalty for reused edges
    """

    n = O.shape[0]
    congestion = np.zeros_like(O, dtype=float)

    trees = []

    for k in range(K):
        T = sample_weighted_rooted_tree(O, root, congestion, alpha)
        trees.append(T)

        # Update congestion
        congestion += T

    return trees, congestion

def optimize_p(O, trees):
    n = O.shape[0]
    o = O.reshape(-1, order='F')

    A = np.column_stack([T.reshape(-1, order='F') for T in trees])

    # Solve min ||o - A p||^2 s.t. p in simplex
    from scipy.optimize import minimize

    K = A.shape[1]

    def obj(p):
        r = o - A @ p
        return r @ r

    constraints = (
        {'type': 'eq', 'fun': lambda p: np.sum(p) - 1},
    )
    bounds = [(0, None)] * K

    p0 = np.ones(K) / K

    res = minimize(obj, p0, bounds=bounds, constraints=constraints)
    return res.x

'''print("\nTest 3")
trees, congestion = build_diverse_tree_pack(O_s, root=0, K=8, alpha=10.0)
p = optimize_p(O_s, trees)
print_decomposition(O_s, p, trees)'''

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

print_matrix_stats(O_s)

nonzero = np.abs(O_s) > eps
stats_nonzero = O_s[nonzero]
print_matrix_stats(stats_nonzero)