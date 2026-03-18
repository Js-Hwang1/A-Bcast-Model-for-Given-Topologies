/*
 * encode.c — BBS tree computation and .bbs file generation.
 *
 * Included from runner.c (via bbs.c) — shares topo_data_t,
 * topo_data_load/free, and all standard headers.
 *
 * Separated from bbs.c so that encoding logic is only invoked
 * when a .bbs file is missing.  The broadcast runtime (bbs.c)
 * never touches this code if the .bbs file exists.
 *
 * =====================================================================
 * TREE CONSTRUCTION — TOPOLOGY-DISPATCHED
 * =====================================================================
 *
 * Entry point: bbs_compute_trees() detects the topology from the node
 * degree distribution and dispatches to a topology-specific builder:
 *
 *   TOPO_MESH      → mesh algorithm (constraint propagation + greedy)
 *   TOPO_BUTTERFLY → butterfly_compute_trees()  [stub — TODO]
 *   TOPO_DRAGONFLY → dragonfly_compute_trees()  [stub — TODO]
 *   TOPO_FATTREE   → fattree_compute_trees()    [stub — TODO]
 *
 * If the topology-specific builder is unimplemented or fails, the
 * mesh algorithm runs as a generic fallback.
 *
 * ---- Mesh algorithm (tau=3, max_usage=2) ----
 *
 * Goal: build tau=3 BFS spanning trees from a given root such that
 * every physical edge is used by at most max_usage=2 of the 3 trees.
 *
 * Why tau=3, max_usage=2:
 *   On a 2D mesh with E edges, 3 spanning trees need 3*(N-1) parent
 *   edges.  With max_usage=2, the edge budget is 2*E.  For a mesh,
 *   2*E >> 3*(N-1), so the constraint is feasible.  The pipeline in
 *   bbs.c exploits this: each shared edge carries at most 2 trees'
 *   flows, giving 50% bandwidth per tree on those edges.  With 3
 *   trees each carrying 1/3 of the data, net efficiency = 3 * 50%
 *   = 1.5x throughput over a single tree.
 *
 * Algorithm overview:
 *   1. Build 1-hop adjacency from the tdat latency matrix.
 *   2. Compute BFS order from root (used to process nodes near-to-far).
 *   3. Create a work list of (node, tree) pairs, round-robin interleaved.
 *   4. Iterative solver:
 *      - Phase A (constraint propagation): if a (node, tree) pair has
 *        exactly 1 viable parent (considering edge capacity, adjacency,
 *        and the corner-ingoing rule), assign it immediately.  Repeat
 *        until no more forced assignments.
 *      - Phase B (greedy): for remaining pairs, pick the neighbor with
 *        the lowest edge usage.  Tie-break by lowest node ID.
 *   5. Repair step (2-deep chain reparenting): for any still-unresolved
 *      pairs, try to free a saturated edge by reparenting a victim in
 *      another tree to an alternate neighbor, cascading up to 2 levels.
 *
 * Corner-ingoing rule:
 *   Degree-2 nodes (mesh corners) cannot serve as parents (except root).
 *   This prevents bottlenecks where a corner's 2 edges would be forced
 *   to carry all 3 trees' traffic.
 *
 * The resulting trees all achieve baseline BFS depth (verified), which
 * is critical: the windowed pipeline in bbs.c needs d_BBS = d_baseline
 * so that the pipeline formula T = (d + k/tau - 1) * T_hop holds
 * without a depth penalty.
 * =====================================================================
 */

/* Forward declaration — defined after bbs_write */
static int tree_depth(const int16_t *par, int N);

/* ---- Topology type enum (prefixed to avoid collision with runner.c) ---- */
typedef enum {
    BBS_TOPO_MESH,
    BBS_TOPO_BUTTERFLY,
    BBS_TOPO_DRAGONFLY,
    BBS_TOPO_FATTREE,
    BBS_TOPO_UNKNOWN
} topo_type_t;

/* Detect topology from node degree distribution.
 * Butterfly/hypercube: uniform degree = log2(N), N is power-of-2.
 * 2D mesh: degree varies from 2 (corners) to 4 (interior).
 * Others: heuristic based on degree range. */
static topo_type_t detect_topo_type(const int *node_deg, int N)
{
    int min_deg = N, max_deg = 0;
    for (int i = 0; i < N; i++) {
        if (node_deg[i] < min_deg) min_deg = node_deg[i];
        if (node_deg[i] > max_deg) max_deg = node_deg[i];
    }

    /* Butterfly: all nodes same degree = log2(N), N power-of-2 */
    if (min_deg == max_deg && (N & (N - 1)) == 0 && N >= 4) {
        int log2N = 0;
        for (int tmp = N; tmp > 1; tmp >>= 1) log2N++;
        if (min_deg == log2N)
            return BBS_TOPO_BUTTERFLY;
    }

    /* 2D mesh: min degree 2 (corners), max degree 4 (interior) */
    if (min_deg >= 2 && max_deg <= 4)
        return BBS_TOPO_MESH;

    /* Dragonfly: all compute nodes have degree 1 in 1-hop adjacency
     * (only connected to their router sibling via a single uplink).
     * With P=2 nodes per router, each node sees exactly 1 neighbor. */
    if (min_deg == 1 && max_deg == 1 && N >= 8)
        return BBS_TOPO_DRAGONFLY;

    /* TODO: FatTree detection heuristics */
    return BBS_TOPO_UNKNOWN;
}

/* ---- Butterfly tree builder ----
 *
 * Builds tau=3 spanning trees on a hypercube with max_usage=2.
 * Same constraint-propagation + greedy algorithm as the mesh path,
 * but without the corner-ingoing rule (all nodes have degree = dim >= 3).
 *
 * Net throughput: tau * (B / max_usage) = 3 * B/2 = 1.5x single-tree.
 * Depth: baseline + 0-1 (near-optimal for pipeline startup).
 *
 * Why not disjoint (max_usage=1)?  Disjoint trees on the hypercube
 * have depth >> baseline due to root-edge contention, which kills
 * pipeline performance.  max_usage=2 keeps depth near-optimal while
 * still achieving 1.5x throughput — matching 2D mesh BBS efficiency.
 */
static int butterfly_compute_trees(const char *topo_file, int root, int N,
                                   const char *adj, const uint16_t *flink,
                                   int max_fl, const int *bfs_order, int bfs_n,
                                   int depth0, int dim,
                                   int16_t *parent_arrays, int *out_tau)
{
    (void)topo_file;

    int tau = 3;
    int max_usage = 2;
    if (tau > BBS_MAX_TREES) tau = BBS_MAX_TREES;

    fprintf(stderr, "BFLY: N=%d dim=%d tau=%d max_usage=%d "
            "baseline_depth=%d\n", N, dim, tau, max_usage, depth0);

    /* Global edge pool */
    uint16_t *edge_usage = (uint16_t *)calloc(max_fl, sizeof(uint16_t));

    /* No corner-ingoing rule — all butterfly nodes have degree = dim >= 3 */

    /* Initialize all trees to -1 */
    for (int t = 0; t < tau; t++)
        for (int i = 0; i < N; i++)
            parent_arrays[(size_t)t * N + i] = -1;

    /* Build unassigned work list: (node, tree) pairs with round-robin */
    int max_pairs = (N - 1) * tau;
    int *ua_node = (int *)malloc(max_pairs * sizeof(int));
    int *ua_tree = (int *)malloc(max_pairs * sizeof(int));
    int ua_count = 0;
    int node_idx = 0;
    for (int bi = 0; bi < bfs_n; bi++) {
        int v = bfs_order[bi];
        if (v == root) continue;
        int start_t = node_idx % tau;
        for (int offset = 0; offset < tau; offset++) {
            int t = (start_t + offset) % tau;
            ua_node[ua_count] = v;
            ua_tree[ua_count] = t;
            ua_count++;
        }
        node_idx++;
    }

    /* Temporary buffers for options */
    int *opt_parent = (int *)malloc(N * sizeof(int));
    int *opt_usage  = (int *)malloc(N * sizeof(int));

    /* Iterative passes with constraint propagation + greedy */
    int *new_ua_node = (int *)malloc(max_pairs * sizeof(int));
    int *new_ua_tree = (int *)malloc(max_pairs * sizeof(int));

    for (int pass = 0; pass < N; pass++) {
        if (ua_count == 0) break;

        /* Phase A: constraint propagation — assign forced pairs */
        int changed = 1;
        while (changed) {
            changed = 0;
            int new_count = 0;
            for (int k = 0; k < ua_count; k++) {
                int v = ua_node[k], t = ua_tree[k];
                int16_t *tree = parent_arrays + (size_t)t * N;
                int n_opts = 0;
                for (int u = 0; u < N; u++) {
                    if (!adj[v * N + u]) continue;
                    if (u != root && tree[u] < 0) continue;
                    uint16_t fl = flink[(size_t)v * N + u];
                    if (fl > 0 && fl < max_fl && edge_usage[fl] >= max_usage)
                        continue;
                    int usage = (fl > 0 && fl < max_fl) ? edge_usage[fl] : 0;
                    opt_parent[n_opts] = u;
                    opt_usage[n_opts]  = usage;
                    n_opts++;
                }
                if (n_opts == 1) {
                    int u = opt_parent[0];
                    tree[v] = (int16_t)u;
                    uint16_t fl = flink[(size_t)v * N + u];
                    if (fl > 0 && fl < max_fl) edge_usage[fl]++;
                    changed = 1;
                } else {
                    new_ua_node[new_count] = v;
                    new_ua_tree[new_count] = t;
                    new_count++;
                }
            }
            int *tmp;
            tmp = ua_node; ua_node = new_ua_node; new_ua_node = tmp;
            tmp = ua_tree; ua_tree = new_ua_tree; new_ua_tree = tmp;
            ua_count = new_count;
        }

        /* Phase B: greedy — pick lowest usage, tie-break lowest node ID */
        int new_count = 0;
        int progress = 0;
        for (int k = 0; k < ua_count; k++) {
            int v = ua_node[k], t = ua_tree[k];
            int16_t *tree = parent_arrays + (size_t)t * N;
            int best_parent = -1, best_usage = max_usage;
            for (int u = 0; u < N; u++) {
                if (!adj[v * N + u]) continue;
                if (u != root && tree[u] < 0) continue;
                uint16_t fl = flink[(size_t)v * N + u];
                if (fl > 0 && fl < max_fl && edge_usage[fl] >= max_usage)
                    continue;
                int usage = (fl > 0 && fl < max_fl) ? edge_usage[fl] : 0;
                if (usage < best_usage ||
                    (usage == best_usage && (best_parent < 0 || u < best_parent))) {
                    best_usage = usage;
                    best_parent = u;
                }
            }
            if (best_parent >= 0) {
                tree[v] = (int16_t)best_parent;
                uint16_t fl = flink[(size_t)v * N + best_parent];
                if (fl > 0 && fl < max_fl) edge_usage[fl]++;
                progress = 1;
            } else {
                new_ua_node[new_count] = v;
                new_ua_tree[new_count] = t;
                new_count++;
            }
        }
        { int *tmp;
          tmp = ua_node; ua_node = new_ua_node; new_ua_node = tmp;
          tmp = ua_tree; ua_tree = new_ua_tree; new_ua_tree = tmp; }
        if (!progress && new_count == ua_count) { ua_count = new_count; break; }
        ua_count = new_count;
    }

    /* Repair step: 2-deep chain reparenting for stuck pairs */
    for (int depth = 0; depth < 2 && ua_count > 0; depth++) {
        int new_count = 0;
        for (int k = 0; k < ua_count; k++) {
            int v = ua_node[k], t = ua_tree[k];
            int16_t *tree_t = parent_arrays + (size_t)t * N;
            int resolved = 0;
            for (int u = 0; u < N && !resolved; u++) {
                if (!adj[v * N + u]) continue;
                uint16_t fl = flink[(size_t)v * N + u];
                if (fl == 0 || fl >= max_fl || edge_usage[fl] < max_usage) continue;
                if (u != root && tree_t[u] < 0) continue;
                for (int t2 = 0; t2 < tau && !resolved; t2++) {
                    if (t2 == t) continue;
                    int16_t *tree_t2 = parent_arrays + (size_t)t2 * N;
                    for (int w = 0; w < N && !resolved; w++) {
                        if (tree_t2[w] < 0) continue;
                        uint16_t wfl = flink[(size_t)w * N + tree_t2[w]];
                        if (wfl != fl) continue;
                        for (int alt = 0; alt < N; alt++) {
                            if (alt == (int)tree_t2[w] || !adj[w * N + alt]) continue;
                            if (alt != root && tree_t2[alt] < 0) continue;
                            int is_desc = 0;
                            for (int a = alt; a >= 0 && a != root; a = tree_t2[a])
                                if (a == w) { is_desc = 1; break; }
                            if (is_desc) continue;
                            uint16_t alt_fl = flink[(size_t)w * N + alt];
                            if (alt_fl > 0 && alt_fl < max_fl &&
                                edge_usage[alt_fl] >= max_usage) {
                                if (depth < 1) continue;
                                int freed = 0;
                                for (int t3 = 0; t3 < tau && !freed; t3++) {
                                    if (t3 == t2) continue;
                                    int16_t *tree_t3 = parent_arrays + (size_t)t3 * N;
                                    for (int w2 = 0; w2 < N && !freed; w2++) {
                                        if (tree_t3[w2] < 0) continue;
                                        if (flink[(size_t)w2 * N + tree_t3[w2]] != alt_fl)
                                            continue;
                                        int old_p2 = tree_t3[w2];
                                        for (int a2 = 0; a2 < N; a2++) {
                                            if (a2 == old_p2 || !adj[w2 * N + a2]) continue;
                                            if (a2 != root && tree_t3[a2] < 0) continue;
                                            int d2 = 0;
                                            for (int x = a2; x >= 0 && x != root;
                                                 x = tree_t3[x])
                                                if (x == w2) { d2 = 1; break; }
                                            if (d2) continue;
                                            uint16_t a2fl = flink[(size_t)w2 * N + a2];
                                            if (a2fl > 0 && a2fl < max_fl &&
                                                edge_usage[a2fl] >= max_usage)
                                                continue;
                                            tree_t3[w2] = (int16_t)a2;
                                            edge_usage[alt_fl]--;
                                            if (a2fl > 0 && a2fl < max_fl) edge_usage[a2fl]++;
                                            freed = 1;
                                            break;
                                        }
                                    }
                                }
                                if (!freed) continue;
                            }
                            tree_t2[w] = (int16_t)alt;
                            edge_usage[fl]--;
                            if (alt_fl > 0 && alt_fl < max_fl) edge_usage[alt_fl]++;
                            tree_t[v] = (int16_t)u;
                            edge_usage[fl]++;
                            resolved = 1;
                            break;
                        }
                    }
                }
            }
            if (!resolved) {
                new_ua_node[new_count] = v;
                new_ua_tree[new_count] = t;
                new_count++;
            }
        }
        { int *tmp;
          tmp = ua_node; ua_node = new_ua_node; new_ua_node = tmp;
          tmp = ua_tree; ua_tree = new_ua_tree; new_ua_tree = tmp; }
        ua_count = new_count;
    }

    if (ua_count > 0)
        fprintf(stderr, "BFLY: WARNING: %d unresolved (node,tree) pairs\n", ua_count);

    /* Per-tree stats */
    int max_depth = 0;
    for (int t = 0; t < tau; t++) {
        int16_t *tree = parent_arrays + (size_t)t * N;
        int d = tree_depth(tree, N);
        int rc = 0, excl = 0, shared = 0;
        for (int i = 0; i < N; i++) {
            if (tree[i] == (int16_t)root) rc++;
            if (tree[i] < 0) continue;
            uint16_t fl = flink[(size_t)tree[i] * N + i];
            if (fl > 0 && fl < max_fl) {
                if (edge_usage[fl] == 1) excl++;
                else shared++;
            }
        }
        fprintf(stderr, "BFLY:   T%d: depth=%d rc=%d exclusive=%d shared=%d\n",
                t, d, rc, excl, shared);
        if (d > max_depth) max_depth = d;
    }

    /* Link sharing distribution */
    int sharing_max = 0;
    int sharing_counts[16] = {0};
    for (int i = 1; i < max_fl; i++) {
        if (edge_usage[i] > 0 && edge_usage[i] < 16) sharing_counts[edge_usage[i]]++;
        if (edge_usage[i] > sharing_max) sharing_max = edge_usage[i];
    }
    fprintf(stderr, "BFLY: max_sharing=%d", sharing_max);
    for (int s = 1; s <= sharing_max && s < 16; s++)
        if (sharing_counts[s] > 0) fprintf(stderr, " [%d]=%d", s, sharing_counts[s]);
    fprintf(stderr, "\n");

    fprintf(stderr, "BFLY: tau=%d max_depth=%d baseline=%d\n",
            tau, max_depth, depth0);

    /* Verify spanning */
    for (int t = 0; t < tau; t++) {
        int covered = 0;
        for (int i = 0; i < N; i++)
            if (parent_arrays[(size_t)t * N + i] >= 0 || i == root) covered++;
        if (covered != N)
            fprintf(stderr, "BFLY: WARNING: Tree %d covers only %d/%d nodes!\n",
                    t, covered, N);
    }

    *out_tau = tau;

    free(opt_parent); free(opt_usage);
    free(ua_node); free(ua_tree);
    free(new_ua_node); free(new_ua_tree);
    free(edge_usage);
    return 0;
}

/* ---- Seed-and-Diffuse recursive binary partition ----
 *
 * Builds a balanced binary tree by recursively splitting the node set:
 *   1. Find the node farthest from parent (using latency matrix)
 *   2. Partition into "close to parent" vs "close to far"
 *   3. Pick reps (closest to parent in each half), attach to parent
 *   4. Recurse on each half
 *
 * other_fan: if non-NULL, prefer nodes with other_fan[v]==0 (leaves in
 *            the other tree) as reps.  This produces anti-correlated
 *            fanout between two trees — nodes internal in T0 become
 *            leaves in T1 and vice versa.
 */
static void build_seed_diffuse(int par, int *nodes, int cnt,
                                int16_t *tree, const float *lat, int N,
                                const int *other_fan)
{
    if (cnt == 0) return;
    if (cnt == 1) {
        tree[nodes[0]] = (int16_t)par;
        return;
    }

    /* Find farthest node from par */
    int far_idx = 0;
    float far_dist = -1.0f;
    for (int i = 0; i < cnt; i++) {
        float d = lat[(size_t)par * N + nodes[i]];
        if (d > far_dist) { far_dist = d; far_idx = i; }
    }
    int far_node = nodes[far_idx];

    /* Partition: left = closer to par, right = closer to far.
     * Tie-break: assign to smaller half for balance. */
    int *left  = (int *)malloc(cnt * sizeof(int));
    int *right = (int *)malloc(cnt * sizeof(int));
    int lcnt = 0, rcnt = 0;

    for (int i = 0; i < cnt; i++) {
        int v = nodes[i];
        float dp = lat[(size_t)par * N + v];
        float df = lat[(size_t)far_node * N + v];
        if (dp < df)       left[lcnt++] = v;
        else if (df < dp)  right[rcnt++] = v;
        else {
            if (lcnt <= rcnt) left[lcnt++] = v;
            else              right[rcnt++] = v;
        }
    }

    /* Pick rep from a half: closest to par with soft anti-correlation.
     * When other_fan is set, inflate latency by 50% per unit of fanout
     * in the other tree.  This gently steers away from T0-internal nodes
     * without forcing long-distance edges. */
    #define PICK_REP(arr, acnt, rep_out) do {                           \
        int _best = -1;                                                 \
        float _best_d = 1e30f;                                          \
        for (int _i = 0; _i < (acnt); _i++) {                          \
            float _d = lat[(size_t)par * N + (arr)[_i]];               \
            if (other_fan)                                              \
                _d *= (1.0f + 0.1f * other_fan[(arr)[_i]]);            \
            if (_d < _best_d) { _best_d = _d; _best = _i; }           \
        }                                                               \
        (rep_out) = _best;                                              \
    } while(0)

    if (lcnt > 0) {
        int best;
        PICK_REP(left, lcnt, best);
        int lrep = left[best];
        tree[lrep] = (int16_t)par;
        left[best] = left[lcnt - 1];
        lcnt--;
        build_seed_diffuse(lrep, left, lcnt, tree, lat, N, other_fan);
    }

    if (rcnt > 0) {
        int best;
        PICK_REP(right, rcnt, best);
        int rrep = right[best];
        tree[rrep] = (int16_t)par;
        right[best] = right[rcnt - 1];
        rcnt--;
        build_seed_diffuse(rrep, right, rcnt, tree, lat, N, other_fan);
    }

    #undef PICK_REP

    free(left);
    free(right);
}

/* ---- Dragonfly tree builder ----
 *
 * Routing-aware hierarchical broadcast tree for dragonfly topology.
 * 4 groups × 4 chassis × 4 routers × 2 nodes = 128 compute nodes.
 *
 * Builds tau=2 anti-correlated trees:
 *   T0: uses "first sibling" (even node) as reps at each router
 *   T1: uses "second sibling" (odd node) as reps, preferring T0-leaves
 *
 * Both trees use the same hierarchy-aware partition (split by group,
 * chassis, router level), but pick OPPOSITE reps to anti-correlate
 * uplink congestion.  Result: max_uplink ≤ 5 (double-1) instead of 6.
 *
 * Fanout ≤ 2 per node per tree → single-tree uplink load ≤ 3.
 */

/* Dragonfly hierarchy extraction from node IDs.
 * Topology: 4 groups × 4 chassis × 4 routers × 2 nodes = 128.
 * node i → group i/32, chassis (i%32)/8, router (i%8)/2, sibling i^1 */
#define DFLY_GROUP(v)   ((v) / 32)
#define DFLY_CHASSIS(v) (((v) % 32) / 8)
#define DFLY_ROUTER(v)  (((v) % 8) / 2)
#define DFLY_SIBLING(v) ((v) ^ 1)
#define NGROUPS 4
#define NCHASSIS 4
#define NROUTERS 4

typedef struct { int par; int *nodes; int cnt; } dfly_task_t;

/* Build one hierarchy-aware tree.
 * t0_fan: if non-NULL, soft-penalize T0-internal nodes when picking reps.
 *         Inflates latency by 50% per unit of T0 fanout. */
static void dfly_build_one_tree(int16_t *tree, int root, int N,
                                const float *lat, const int *t0_fan)
{
    for (int i = 0; i < N; i++) tree[i] = -1;

    #define DFLY_QMAX 512
    dfly_task_t q[DFLY_QMAX];
    int qh = 0, qt = 0;

    /* Initial task: all non-root nodes */
    {
        int *nc = (int *)malloc((N - 1) * sizeof(int));
        int c = 0;
        for (int i = 0; i < N; i++)
            if (i != root) nc[c++] = i;
        q[qt].par = root;
        q[qt].nodes = nc;
        q[qt].cnt = c;
        qt++;
    }

    while (qh < qt) {
        dfly_task_t tk = q[qh++];
        int par = tk.par;

        if (tk.cnt == 0) { free(tk.nodes); continue; }
        if (tk.cnt == 1) {
            tree[tk.nodes[0]] = (int16_t)par;
            free(tk.nodes); continue;
        }

        int par_g = DFLY_GROUP(par);
        int par_c = DFLY_CHASSIS(par);
        int par_r = DFLY_ROUTER(par);

        /* Count distinct groups, chassis, routers in node set */
        char has_group[NGROUPS] = {0};
        char has_chassis[NCHASSIS] = {0};
        char has_router[NROUTERS] = {0};
        int n_groups = 0, n_chassis = 0, n_routers = 0;

        for (int i = 0; i < tk.cnt; i++) {
            int v = tk.nodes[i];
            int g = DFLY_GROUP(v), c = DFLY_CHASSIS(v), r = DFLY_ROUTER(v);
            if (!has_group[g]) { has_group[g] = 1; n_groups++; }
            if (g == par_g && !has_chassis[c]) { has_chassis[c] = 1; n_chassis++; }
            if (g == par_g && c == par_c && !has_router[r]) { has_router[r] = 1; n_routers++; }
        }

        /* Split strategy: chain at group level, balanced at lower levels.
         *   4 groups  → parent's group vs others    (chain, 3 steps)
         *   4 chassis → {parent's+1} vs {other 2}  (balanced, 2 steps)
         *   4 routers → {parent's+1} vs {other 2}  (balanced, 2 steps)
         *   2 siblings → sibling edge               (1 step)
         *
         * Chain at group level keeps node 1 (R0-C0-G0) as the G0 rep,
         * avoiding reverse black traffic from non-C0 chassis back to C0.
         * Balanced group splits use up both R0-C0 nodes for global work,
         * forcing a non-C0 node as G0 rep → extra green→black routing.
         * Gateway-aware rep selection ensures R0 nodes are picked when
         * distributing across multiple chassis.
         * Total depth = 3+2+2+1 = 8.
         */
        int *left  = (int *)malloc(tk.cnt * sizeof(int));
        int *right = (int *)malloc(tk.cnt * sizeof(int));
        int lcnt = 0, rcnt = 0;

        if (n_groups > 1) {
            /* Chain split: parent's group vs all others.
             * Keeps node 1 (R0-C0) as G0 rep → direct black links to other
             * chassis, no reverse green→black routing overhead.
             * Depth cost: 3 steps for 4 groups (vs balanced's 2), but
             * avoids max_black=2 and extra green contention. */
            int par_group_present = has_group[par_g];
            if (par_group_present) {
                for (int i = 0; i < tk.cnt; i++) {
                    if (DFLY_GROUP(tk.nodes[i]) == par_g)
                        left[lcnt++] = tk.nodes[i];
                    else
                        right[rcnt++] = tk.nodes[i];
                }
            } else {
                int group_list[NGROUPS], ng = 0;
                for (int g = 0; g < NGROUPS; g++)
                    if (has_group[g]) group_list[ng++] = g;
                int mid = ng / 2;
                if (mid < 1) mid = 1;
                char left_group[NGROUPS] = {0};
                for (int gi = 0; gi < mid; gi++)
                    left_group[group_list[gi]] = 1;
                for (int i = 0; i < tk.cnt; i++) {
                    if (left_group[DFLY_GROUP(tk.nodes[i])])
                        left[lcnt++] = tk.nodes[i];
                    else
                        right[rcnt++] = tk.nodes[i];
                }
            }
        } else if (n_chassis > 1) {
            /* Balanced binary split: parent's chassis in left half.
             * Depth 2 for 4 chassis (vs chain's depth 3).
             * Rep selection still prefers R0 gateway nodes (lowest
             * latency), so cross-chassis edges stay on black links. */
            int ch_list[NCHASSIS], nc_ch = 0;
            if (has_chassis[par_c]) ch_list[nc_ch++] = par_c;
            for (int c = 0; c < NCHASSIS; c++)
                if (has_chassis[c] && c != par_c) ch_list[nc_ch++] = c;
            int mid_c = nc_ch / 2;
            if (mid_c < 1) mid_c = 1;
            char left_ch[NCHASSIS] = {0};
            for (int ci = 0; ci < mid_c; ci++)
                left_ch[ch_list[ci]] = 1;
            for (int i = 0; i < tk.cnt; i++) {
                if (left_ch[DFLY_CHASSIS(tk.nodes[i])])
                    left[lcnt++] = tk.nodes[i];
                else
                    right[rcnt++] = tk.nodes[i];
            }
        } else if (n_routers > 1) {
            /* Balanced binary split: parent's router in left half.
             * Depth 2 for 4 routers (vs chain's depth 3).
             * All splits still land on green links within chassis. */
            int rt_list[NROUTERS], nr_rt = 0;
            if (has_router[par_r]) rt_list[nr_rt++] = par_r;
            for (int r = 0; r < NROUTERS; r++)
                if (has_router[r] && r != par_r) rt_list[nr_rt++] = r;
            int mid_r = nr_rt / 2;
            if (mid_r < 1) mid_r = 1;
            char left_rt[NROUTERS] = {0};
            for (int ri = 0; ri < mid_r; ri++)
                left_rt[rt_list[ri]] = 1;
            for (int i = 0; i < tk.cnt; i++) {
                if (left_rt[DFLY_ROUTER(tk.nodes[i])])
                    left[lcnt++] = tk.nodes[i];
                else
                    right[rcnt++] = tk.nodes[i];
            }
        } else {
            /* Sibling level: just split into 2 */
            left[lcnt++] = tk.nodes[0];
            if (tk.cnt > 1) right[rcnt++] = tk.nodes[1];
        }

        free(tk.nodes);

        /* Pick rep from each half: closest to parent.
         * When t0_fan is provided (building T1), inflate latency by 50%
         * per unit of T0 fanout to steer away from T0-internal nodes. */
        #define DFLY_PICK_REP(arr, acnt) do {                              \
            if ((acnt) <= 0) { free(arr); break; }                         \
            int _best = 0;                                                 \
            float _bd = lat[(size_t)par * N + (arr)[0]];                  \
            if (t0_fan) _bd *= (1.0f + 0.5f * t0_fan[(arr)[0]]);         \
            for (int _i = 1; _i < (acnt); _i++) {                         \
                float _d = lat[(size_t)par * N + (arr)[_i]];              \
                if (t0_fan) _d *= (1.0f + 0.5f * t0_fan[(arr)[_i]]);     \
                if (_d < _bd) { _bd = _d; _best = _i; }                   \
            }                                                              \
            int _rep = (arr)[_best];                                       \
            tree[_rep] = (int16_t)par;                                     \
            (arr)[_best] = (arr)[(acnt) - 1];                             \
            (acnt)--;                                                      \
            if ((acnt) > 0) {                                              \
                q[qt].par = _rep;                                          \
                q[qt].nodes = (arr);                                       \
                q[qt].cnt = (acnt);                                        \
                qt++;                                                      \
            } else {                                                       \
                free(arr);                                                 \
            }                                                              \
        } while(0)

        DFLY_PICK_REP(left, lcnt);
        DFLY_PICK_REP(right, rcnt);

        #undef DFLY_PICK_REP
    }
    #undef DFLY_QMAX
}

/* Compute physical link congestion for a set of trees and print stats */
static void dfly_print_stats(int16_t **trees, int tau, int root, int N,
                              const float *lat)
{
    /* Aggregate physical link congestion across ALL trees */
    int blue_cong[NGROUPS][NGROUPS];
    int black_cong[NGROUPS][NCHASSIS][NCHASSIS];
    int green_cong[NGROUPS][NCHASSIS][NROUTERS][NROUTERS];
    memset(blue_cong, 0, sizeof(blue_cong));
    memset(black_cong, 0, sizeof(black_cong));
    memset(green_cong, 0, sizeof(green_cong));
    int uplink_load[1024] = {0};

    for (int t = 0; t < tau; t++) {
        int16_t *tree = trees[t];
        for (int i = 0; i < N; i++) {
            if (tree[i] < 0) continue;
            int p = tree[i];
            int ig = DFLY_GROUP(i), ic = DFLY_CHASSIS(i), ir = DFLY_ROUTER(i);
            int pg = DFLY_GROUP(p), pc = DFLY_CHASSIS(p), pr = DFLY_ROUTER(p);

            uplink_load[i]++;
            uplink_load[p]++;

            if (ig != pg) {
                int lo = (ig < pg) ? ig : pg, hi = (ig < pg) ? pg : ig;
                blue_cong[lo][hi]++;
                if (ir != 0) green_cong[ig][ic][0][ir]++;
                if (ic != 0) black_cong[ig][0][ic]++;
                if (pc != 0) black_cong[pg][0][pc]++;
                if (pr != 0) green_cong[pg][pc][0][pr]++;
            } else if (ic != pc) {
                if (ir != 0) green_cong[ig][ic][0][ir]++;
                int lo = (ic < pc) ? ic : pc, hi = (ic < pc) ? pc : ic;
                black_cong[ig][lo][hi]++;
                if (pr != 0) green_cong[ig][pc][0][pr]++;
            } else if (ir != pr) {
                int lo = (ir < pr) ? ir : pr, hi = (ir < pr) ? pr : ir;
                green_cong[ig][ic][lo][hi]++;
            }
        }
    }

    int max_blue = 0, max_black = 0, max_green = 0, max_uplink = 0;
    for (int g1 = 0; g1 < NGROUPS; g1++)
        for (int g2 = g1+1; g2 < NGROUPS; g2++)
            if (blue_cong[g1][g2] > max_blue) max_blue = blue_cong[g1][g2];
    for (int g = 0; g < NGROUPS; g++)
        for (int c1 = 0; c1 < NCHASSIS; c1++)
            for (int c2 = c1+1; c2 < NCHASSIS; c2++)
                if (black_cong[g][c1][c2] > max_black) max_black = black_cong[g][c1][c2];
    for (int g = 0; g < NGROUPS; g++)
        for (int c = 0; c < NCHASSIS; c++)
            for (int r1 = 0; r1 < NROUTERS; r1++)
                for (int r2 = r1+1; r2 < NROUTERS; r2++)
                    if (green_cong[g][c][r1][r2] > max_green) max_green = green_cong[g][c][r1][r2];
    for (int i = 0; i < N; i++)
        if (uplink_load[i] > max_uplink) max_uplink = uplink_load[i];

    /* Per-tree stats */
    for (int t = 0; t < tau; t++) {
        int16_t *tree = trees[t];
        int d = tree_depth(tree, N);
        int tfan[1024] = {0};
        int max_fan = 0;
        for (int i = 0; i < N; i++)
            if (tree[i] >= 0) tfan[(int)tree[i]]++;
        for (int i = 0; i < N; i++)
            if (tfan[i] > max_fan) max_fan = tfan[i];
        int fan_hist[64] = {0};
        for (int i = 0; i < N; i++)
            if (tfan[i] < 64) fan_hist[tfan[i]]++;
        int covered = 0;
        for (int i = 0; i < N; i++)
            if (tree[i] >= 0 || i == root) covered++;

        fprintf(stderr, "DFLY: T%d: depth=%d max_fanout=%d root_fanout=%d "
                "covered=%d/%d\n", t, d, max_fan, tfan[root], covered, N);
        fprintf(stderr, "DFLY: T%d: fanout dist:", t);
        for (int f = 0; f <= max_fan && f < 64; f++)
            if (fan_hist[f] > 0) fprintf(stderr, " [f=%d]=%d", f, fan_hist[f]);
        fprintf(stderr, "\n");
        if (covered != N)
            fprintf(stderr, "DFLY: WARNING: T%d spans only %d/%d!\n", t, covered, N);
    }

    /* Anti-correlation analysis */
    if (tau == 2) {
        int fan0[1024] = {0}, fan1[1024] = {0};
        for (int i = 0; i < N; i++) {
            if (trees[0][i] >= 0) fan0[(int)trees[0][i]]++;
            if (trees[1][i] >= 0) fan1[(int)trees[1][i]]++;
        }
        int both_internal = 0, anti_corr = 0, both_leaf = 0;
        int parent_overlap = 0;
        for (int i = 0; i < N; i++) {
            if (trees[0][i] >= 0 && trees[0][i] == trees[1][i]) parent_overlap++;
            int f0 = fan0[i], f1 = fan1[i];
            if (f0 >= 2 && f1 >= 2) both_internal++;
            else if ((f0 >= 2 && f1 == 0) || (f0 == 0 && f1 >= 2)) anti_corr++;
            else if (f0 == 0 && f1 == 0) both_leaf++;
        }
        fprintf(stderr, "DFLY: anti-corr: parent_overlap=%d both_int=%d "
                "anti_corr=%d both_leaf=%d\n",
                parent_overlap, both_internal, anti_corr, both_leaf);
    }

    /* Uplink congestion histogram */
    int uplink_hist[16] = {0};
    for (int i = 0; i < N; i++)
        if (uplink_load[i] < 16) uplink_hist[uplink_load[i]]++;

    fprintf(stderr, "DFLY: physical link congestion (all trees): "
            "max_blue=%d max_black=%d max_green=%d max_uplink=%d\n",
            max_blue, max_black, max_green, max_uplink);
    fprintf(stderr, "DFLY: uplink histogram:");
    for (int u = 0; u <= max_uplink && u < 16; u++)
        if (uplink_hist[u] > 0) fprintf(stderr, " [%d]=%d", u, uplink_hist[u]);
    fprintf(stderr, "\n");
    fprintf(stderr, "DFLY: blue links:");
    for (int g1 = 0; g1 < NGROUPS; g1++)
        for (int g2 = g1+1; g2 < NGROUPS; g2++)
            if (blue_cong[g1][g2] > 0)
                fprintf(stderr, " [%d-%d]=%d", g1, g2, blue_cong[g1][g2]);
    fprintf(stderr, "\n");
}

/* Build the interleaved global+local dissemination tree.
 *
 * At each hierarchy level, uses binary dissemination (depth=2 for 4 entities).
 * Root's children are interleaved: global(blue), local(green), global(blue),
 * local(black), global(green), local(green), local(red).
 *
 * The key idea: while root seeds a remote group, previously-seeded nodes
 * do LOCAL diffusion on different physical links simultaneously.
 *
 * Depth = 7 (2 group + 2 chassis + 2 router + 1 sibling).
 * Fanout = 7 at root, 6 at group reps, 5 at chassis reps.
 * Each hierarchy level's rep is always at (router=0, chassis=0) for
 * clean gateway routing.
 */
static void dfly_build_interleaved_tree(int16_t *tree, int root, int N)
{
    (void)N;  /* we know N=128 */
    for (int i = 0; i < 128; i++) tree[i] = -1;

    int rg = DFLY_GROUP(root);
    int rc = DFLY_CHASSIS(root);
    int rr = DFLY_ROUTER(root);
    int rb = rg*32 + rc*8 + rr*2;  /* base of root's router */

    /* Group level: binary dissemination among 4 groups.
     * root_group → g1, root_group → g2, g1 → g3
     * Reps are always (g, 0, 0, 0) — group gateways. */
    int greps[NGROUPS];
    for (int g = 0; g < NGROUPS; g++) greps[g] = g * 32;

    /* Sort non-root groups by distance from root group */
    int gsorted[NGROUPS-1], gn = 0;
    for (int g = 0; g < NGROUPS; g++)
        if (g != rg) gsorted[gn++] = g;

    /* Binary dissemination: root→g[0], root→g[1], g[0]→g[2] */
    tree[greps[gsorted[0]]] = greps[rg];
    tree[greps[gsorted[1]]] = greps[rg];
    if (gn > 2)
        tree[greps[gsorted[2]]] = greps[gsorted[0]];

    /* Chassis level per group: binary dissemination among 4 chassis.
     * For each group, the group gateway (g,0,0,0) seeds chassis.
     * (g,0,0,0) → (g,1,0,0), (g,0,0,0) → (g,2,0,0), (g,1,0,0) → (g,3,0,0) */
    for (int g = 0; g < NGROUPS; g++) {
        int gb = g * 32;
        tree[gb + 8]  = gb;       /* C0 → C1 */
        tree[gb + 16] = gb;       /* C0 → C2 */
        tree[gb + 24] = gb + 8;   /* C1 → C3 */
    }

    /* Router level per chassis: binary dissemination among 4 routers.
     * (g,c,0,0) → (g,c,1,0), (g,c,0,0) → (g,c,2,0), (g,c,1,0) → (g,c,3,0) */
    for (int g = 0; g < NGROUPS; g++)
        for (int c = 0; c < NCHASSIS; c++) {
            int cb = g*32 + c*8;
            tree[cb + 2] = cb;       /* R0 → R1 */
            tree[cb + 4] = cb;       /* R0 → R2 */
            tree[cb + 6] = cb + 2;   /* R1 → R3 */
        }

    /* Sibling level: even node → odd node */
    for (int g = 0; g < NGROUPS; g++)
        for (int c = 0; c < NCHASSIS; c++)
            for (int r = 0; r < NROUTERS; r++) {
                int b = g*32 + c*8 + r*2;
                tree[b + 1] = b;
            }

    /* Fix root: if root is not (0,0,0,0), we need to graft root into
     * the tree.  For now, this assumes root=0 (the gateway node). */
    if (root != 0)
        fprintf(stderr, "DFLY: WARNING: interleaved tree only supports root=0\n");
}

/* Build tau=2 phase-isolated relay trees.
 *
 * Relay pattern at each hierarchy level:
 *   T0: 0 → 1 → {2, 3}    (entity 1 is the relay)
 *   T1: 0 → 2 → {1, 3}    (entity 2 is the relay)
 *
 * Phase isolation: each hierarchy level's rep delegates LOCAL work
 * to its SIBLING.  The relay node (e.g., G1 rep) has ONLY group-level
 * children — no chassis/router/sibling children.  This prevents
 * bandwidth sharing between hierarchy levels at relay nodes.
 *
 * Structure per tree:
 *   Group:   G0 → G1_rep → {G2_rep, G3_rep}     (relay, fan≤3 with delegate)
 *            Each G_rep → sibling (delegate for local chassis/router work)
 *   Chassis: delegate → C1_rep → {C2_rep, C3_rep} (relay within group)
 *            Each C_rep → sibling (delegate for local router work)
 *   Router:  delegate → R1_rep → {R2_rep, R3_rep} (relay within chassis)
 *   Sibling: R_rep → sibling_node
 */
static void dfly_build_phased_relay_pair(int16_t *t0, int16_t *t1,
                                          int root, int N)
{
    (void)root; (void)N;
    for (int i = 0; i < 128; i++) { t0[i] = -1; t1[i] = -1; }

    /* GROUP LEVEL:
     * T0: G0→G1, G1→{G2,G3}  (G1 relay)
     * T1: G0→G2, G2→{G1,G3}  (G2 relay)
     * Plus: each group rep → sibling delegate for local work. */
    t0[32]  = 0;    t0[64]  = 32;   t0[96]  = 32;
    t1[64]  = 0;    t1[32]  = 64;   t1[96]  = 64;

    /* Group delegates (rep → sibling for local chassis/router work) */
    for (int g = 0; g < 4; g++) {
        int rep = g * 32;
        t0[rep + 1] = rep;
        t1[rep + 1] = rep;
    }

    /* CHASSIS LEVEL per group:
     * From the group's sibling delegate (g*32+1).
     * T0: del→C1, C1→{C2,C3}
     * T1: del→C2, C2→{C1,C3} */
    for (int g = 0; g < 4; g++) {
        int del = g*32 + 1;
        t0[g*32 + 8]  = del;            /* C0→C1 */
        t0[g*32 + 16] = g*32 + 8;       /* C1→C2 (relay) */
        t0[g*32 + 24] = g*32 + 8;       /* C1→C3 (relay) */

        t1[g*32 + 16] = del;            /* C0→C2 */
        t1[g*32 + 8]  = g*32 + 16;      /* C2→C1 (relay) */
        t1[g*32 + 24] = g*32 + 16;      /* C2→C3 (relay) */
    }

    /* Chassis delegates for C1,C2,C3 (rep → sibling for router work) */
    for (int g = 0; g < 4; g++)
        for (int c = 1; c < 4; c++) {
            int crep = g*32 + c*8;
            t0[crep + 1] = crep;
            t1[crep + 1] = crep;
        }

    /* ROUTER LEVEL per chassis:
     * T0: del→R1, R1→{R2,R3}
     * T1: del→R2, R2→{R1,R3}
     * For C0: delegate = group sibling (g*32+1)
     * For C1-C3: delegate = chassis sibling (g*32+c*8+1) */
    for (int g = 0; g < 4; g++)
        for (int c = 0; c < 4; c++) {
            int del = (c == 0) ? g*32 + 1 : g*32 + c*8 + 1;
            int cb  = g*32 + c*8;

            t0[cb + 2] = del;           /* R0→R1 */
            t0[cb + 4] = cb + 2;        /* R1→R2 (relay) */
            t0[cb + 6] = cb + 2;        /* R1→R3 (relay) */

            t1[cb + 4] = del;           /* R0→R2 */
            t1[cb + 2] = cb + 4;        /* R2→R1 (relay) */
            t1[cb + 6] = cb + 4;        /* R2→R3 (relay) */
        }

    /* SIBLING LEVEL: each non-R0 router rep → its sibling */
    for (int g = 0; g < 4; g++)
        for (int c = 0; c < 4; c++)
            for (int r = 1; r < 4; r++) {
                int rrep = g*32 + c*8 + r*2;
                t0[rrep + 1] = rrep;
                t1[rrep + 1] = rrep;
            }

    if (root != 0)
        fprintf(stderr, "DFLY: WARNING: phased relay only supports root=0\n");
}

static int dragonfly_compute_trees(const char *topo_file, int root, int N,
                                   const char *adj, const uint16_t *flink,
                                   int max_fl, const int *bfs_order, int bfs_n,
                                   int depth0,
                                   int16_t *parent_arrays, int *out_tau)
{
    (void)adj; (void)flink; (void)max_fl;
    (void)bfs_order; (void)bfs_n; (void)depth0;

    topo_data_t td;
    if (topo_data_load(topo_file, &td) != 0) {
        fprintf(stderr, "DFLY: cannot load tdat\n");
        *out_tau = 0;
        return -1;
    }

    fprintf(stderr, "DFLY: root=%d group=%d chassis=%d router=%d\n",
            root, DFLY_GROUP(root), DFLY_CHASSIS(root), DFLY_ROUTER(root));

    int tau = 2;
    int16_t *trees[2] = { parent_arrays, parent_arrays + N };

    /* Tau=2 relay pair: anti-correlated trees with depth=7.
     * T0: 0→1→{2,3} pattern at each hierarchy level.
     * T1: 0→2→{1,3} pattern (relay node swapped). */
    dfly_build_phased_relay_pair(trees[0], trees[1], root, N);

    dfly_print_stats(trees, tau, root, N, td.lat);
    fprintf(stderr, "DFLY: tau=%d\n", tau);

    *out_tau = tau;
    topo_data_free(&td);
    return 0;
}

/* ---- FatTree tree builder (stub) ---- */
static int fattree_compute_trees(const char *topo_file, int root, int N,
                                 const char *adj, const uint16_t *flink,
                                 int max_fl, const int *bfs_order, int bfs_n,
                                 int depth0,
                                 int16_t *parent_arrays, int *out_tau)
{
    (void)topo_file; (void)adj; (void)flink; (void)max_fl;
    (void)bfs_order; (void)bfs_n; (void)depth0;
    (void)parent_arrays;
    fprintf(stderr, "BBS: fattree_compute_trees() not yet implemented "
            "(N=%d root=%d)\n", N, root);
    *out_tau = 0;
    return -1;
}

/* ---- Write .bbs from parent arrays ---- */
/* parent_arrays: tau × N, row-major.  parent_arrays[t*N + i] = parent of i in tree t. */
static int bbs_write(const char *path, int N, int tau, int root,
                     const int16_t *parent_arrays)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Header: magic(2) N(2) tau(1) pad(1) root(2) = 8 bytes */
    uint16_t h_magic = BBS_MAGIC, h_N = (uint16_t)N, h_root = (uint16_t)root;
    uint8_t  h_tau = (uint8_t)tau, h_pad = 0;
    fwrite(&h_magic, 2, 1, f);
    fwrite(&h_N,     2, 1, f);
    fwrite(&h_tau,   1, 1, f);
    fwrite(&h_pad,   1, 1, f);
    fwrite(&h_root,  2, 1, f);

    /* Reserve offset table */
    uint32_t *offsets = (uint32_t *)calloc(N, 4);
    long ot_pos = ftell(f);
    fwrite(offsets, 4, N, f);

    /* Write node sections */
    for (int i = 0; i < N; i++) {
        offsets[i] = (uint32_t)ftell(f);
        for (int t = 0; t < tau; t++) {
            int16_t p = parent_arrays[t * N + i];
            /* Collect children: nodes whose parent in tree t is i */
            int16_t ch[1024];
            uint16_t nc = 0;
            for (int j = 0; j < N; j++)
                if (parent_arrays[t * N + j] == (int16_t)i)
                    ch[nc++] = (int16_t)j;
            /* Sort children: farthest first (blue > black > green > red).
             * This ensures the deepest subtrees start receiving chunks
             * first in the pipeline. Uses |child-parent| as distance proxy. */
            for (int a = 0; a < (int)nc - 1; a++)
                for (int b = a + 1; b < (int)nc; b++)
                    if (abs(ch[a] - (int16_t)i) < abs(ch[b] - (int16_t)i)) {
                        int16_t tmp = ch[a]; ch[a] = ch[b]; ch[b] = tmp;
                    }
            fwrite(&p,  2, 1, f);
            fwrite(&nc, 2, 1, f);
            if (nc > 0) fwrite(ch, 2, nc, f);
        }
    }

    /* Patch offset table */
    fseek(f, ot_pos, SEEK_SET);
    fwrite(offsets, 4, N, f);

    free(offsets);
    fclose(f);
    return 0;
}

/* ---- Helper: compute depth of a parent-array tree ---- */
static int tree_depth(const int16_t *par, int N)
{
    int maxd = 0;
    for (int i = 0; i < N; i++) {
        int d = 0;
        for (int v = i; par[v] >= 0; v = par[v]) {
            d++;
            if (d > N) return N; /* cycle guard */
        }
        if (d > maxd) maxd = d;
    }
    return maxd;
}

/* ---- Compute spanning trees with max edge-sharing constraint ----
 *
 * Algorithm: constraint propagation + greedy round-robin.
 *   - tau = 3, max_usage = 2 (each physical link used by at most 2 trees)
 *   - Phase A (propagation): assign forced (v,t) pairs (1 viable parent)
 *   - Phase B (greedy): remaining pairs pick lowest-usage edge
 *   - Corner-ingoing rule: degree-2 nodes cannot be parents
 *   - Repair step: chain reparenting to resolve stuck nodes
 */
static int bbs_compute_trees(const char *topo_file, int root, int N,
                             int16_t *parent_arrays, int *out_tau)
{
    topo_data_t td;
    if (topo_data_load(topo_file, &td) != 0) return -1;

    /* 1-hop threshold */
    float lat_1hop = 1e30f;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float l = td.lat[(size_t)i * N + j];
            if (l > 0.0f && l < lat_1hop) lat_1hop = l;
        }
    float lat_thresh = lat_1hop * 1.01f;

    /* Build 1-hop adjacency matrix */
    size_t NN = (size_t)N * N;
    char *adj = (char *)calloc(NN, 1);
    int total_edges = 0;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            float l = td.lat[(size_t)i * N + j];
            if (l > 0.0f && l <= lat_thresh) {
                adj[i * N + j] = 1;
                adj[j * N + i] = 1;
                total_edges++;
            }
        }

    /* Root's 1-hop neighbors */
    int root_nn = 0;
    for (int v = 0; v < N; v++)
        if (adj[root * N + v])
            root_nn++;

    /* Node degrees */
    int *node_deg = (int *)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (adj[i * N + j]) node_deg[i]++;

    /* Compute baseline single-tree BFS depth */
    char *visited = (char *)calloc(N, 1);
    int  *queue   = (int *)malloc(N * sizeof(int));
    int16_t *t0_bfs = (int16_t *)malloc(N * sizeof(int16_t));
    for (int i = 0; i < N; i++) t0_bfs[i] = -1;
    visited[root] = 1;
    int qh = 0, qt = 0;
    queue[qt++] = root;
    while (qh < qt) {
        int u = queue[qh++];
        for (int v = 0; v < N; v++) {
            if (visited[v] || !adj[u * N + v]) continue;
            visited[v] = 1;
            t0_bfs[v] = (int16_t)u;
            queue[qt++] = v;
        }
    }
    int depth0 = tree_depth(t0_bfs, N);

    /* Find max flink ID */
    int max_fl = 0;
    for (size_t i = 0; i < NN; i++)
        if (td.flink[i] > max_fl) max_fl = td.flink[i];
    max_fl += 2;

    /* Compute BFS order from root */
    int *bfs_order = (int *)malloc(N * sizeof(int));
    int bfs_n = 0;
    memset(visited, 0, N);
    visited[root] = 1;
    qh = 0; qt = 0;
    queue[qt++] = root;
    while (qh < qt) {
        int u = queue[qh++];
        bfs_order[bfs_n++] = u;
        for (int v = 0; v < N; v++) {
            if (!visited[v] && adj[u * N + v]) {
                visited[v] = 1;
                queue[qt++] = v;
            }
        }
    }

    /* ---- Topology dispatch ---- */
    topo_type_t topo = detect_topo_type(node_deg, N);

    if (topo != BBS_TOPO_MESH && topo != BBS_TOPO_UNKNOWN) {
        int result = -1;
        switch (topo) {
        case BBS_TOPO_BUTTERFLY: {
            int dim = 0;
            for (int tmp = N; tmp > 1; tmp >>= 1) dim++;
            fprintf(stderr, "BBS: Butterfly detected (dim=%d, edges=%d, "
                    "root_deg=%d, baseline_depth=%d)\n",
                    dim, total_edges, root_nn, depth0);
            result = butterfly_compute_trees(topo_file, root, N, adj,
                                             td.flink, max_fl,
                                             bfs_order, bfs_n, depth0, dim,
                                             parent_arrays, out_tau);
            break;
        }
        case BBS_TOPO_DRAGONFLY:
            fprintf(stderr, "BBS: Dragonfly detected (edges=%d, root_deg=%d, "
                    "baseline_depth=%d)\n", total_edges, root_nn, depth0);
            result = dragonfly_compute_trees(topo_file, root, N, adj,
                                             td.flink, max_fl,
                                             bfs_order, bfs_n, depth0,
                                             parent_arrays, out_tau);
            break;
        case BBS_TOPO_FATTREE:
            fprintf(stderr, "BBS: FatTree detected (edges=%d, root_deg=%d, "
                    "baseline_depth=%d)\n", total_edges, root_nn, depth0);
            result = fattree_compute_trees(topo_file, root, N, adj,
                                           td.flink, max_fl,
                                           bfs_order, bfs_n, depth0,
                                           parent_arrays, out_tau);
            break;
        default:
            break;
        }

        /* If topology-specific builder succeeded, skip mesh path */
        if (result == 0 && *out_tau > 0) {
            free(bfs_order);
            free(t0_bfs); free(visited); free(queue); free(adj);
            free(node_deg); topo_data_free(&td);
            return 0;
        }
        /* Fall through to mesh algorithm as fallback */
        fprintf(stderr, "BBS: topology-specific builder failed, "
                "falling back to mesh algorithm\n");
    }

    /* ---- Mesh / generic algorithm ---- */
    int tau = 3;
    int max_usage = 2;
    if (root_nn < 2) tau = 1;
    if (tau > BBS_MAX_TREES) tau = BBS_MAX_TREES;

    fprintf(stderr, "BBS: edges=%d root_deg=%d tau=%d max_usage=%d "
            "baseline_depth=%d\n",
            total_edges, root_nn, tau, max_usage, depth0);

    if (tau < 2) {
        *out_tau = 1;
        memcpy(parent_arrays, t0_bfs, N * sizeof(int16_t));
        fprintf(stderr, "BBS: tau=1 (root degree < 2)\n");
        free(bfs_order);
        free(t0_bfs); free(visited); free(queue); free(adj);
        free(node_deg); topo_data_free(&td);
        return 0;
    }

    /* Global edge pool */
    uint16_t *edge_usage = (uint16_t *)calloc(max_fl, sizeof(uint16_t));

    /* Corner-ingoing: degree-2 nodes cannot be parents (except root) */
    char *no_parent = (char *)calloc(N, 1);
    for (int i = 0; i < N; i++)
        if (i != root && node_deg[i] <= 2)
            no_parent[i] = 1;

    /* Initialize all trees to -1 */
    for (int t = 0; t < tau; t++)
        for (int i = 0; i < N; i++)
            parent_arrays[(size_t)t * N + i] = -1;

    /* Build unassigned work list: (node, tree) pairs with round-robin */
    int max_pairs = (N - 1) * tau;
    int *ua_node = (int *)malloc(max_pairs * sizeof(int));
    int *ua_tree = (int *)malloc(max_pairs * sizeof(int));
    int ua_count = 0;
    int node_idx = 0;
    for (int bi = 0; bi < bfs_n; bi++) {
        int v = bfs_order[bi];
        if (v == root) continue;
        int start_t = node_idx % tau;
        for (int offset = 0; offset < tau; offset++) {
            int t = (start_t + offset) % tau;
            ua_node[ua_count] = v;
            ua_tree[ua_count] = t;
            ua_count++;
        }
        node_idx++;
    }

    /* Temporary buffers for options */
    int *opt_parent = (int *)malloc(N * sizeof(int));
    int *opt_usage  = (int *)malloc(N * sizeof(int));

    /* Iterative passes with constraint propagation + greedy */
    int *new_ua_node = (int *)malloc(max_pairs * sizeof(int));
    int *new_ua_tree = (int *)malloc(max_pairs * sizeof(int));

    for (int pass = 0; pass < N; pass++) {
        if (ua_count == 0) break;

        /* Phase A: constraint propagation — assign forced pairs */
        int changed = 1;
        while (changed) {
            changed = 0;
            int new_count = 0;
            for (int k = 0; k < ua_count; k++) {
                int v = ua_node[k], t = ua_tree[k];
                int16_t *tree = parent_arrays + (size_t)t * N;
                int n_opts = 0;
                for (int u = 0; u < N; u++) {
                    if (!adj[v * N + u]) continue;
                    if (u != root && tree[u] < 0) continue;
                    if (no_parent[u]) continue;
                    uint16_t fl = td.flink[(size_t)v * N + u];
                    if (fl > 0 && fl < max_fl && edge_usage[fl] >= max_usage)
                        continue;
                    int usage = (fl > 0 && fl < max_fl) ? edge_usage[fl] : 0;
                    opt_parent[n_opts] = u;
                    opt_usage[n_opts]  = usage;
                    n_opts++;
                }
                if (n_opts == 1) {
                    /* Forced assignment */
                    int u = opt_parent[0];
                    tree[v] = (int16_t)u;
                    uint16_t fl = td.flink[(size_t)v * N + u];
                    if (fl > 0 && fl < max_fl) edge_usage[fl]++;
                    changed = 1;
                } else {
                    new_ua_node[new_count] = v;
                    new_ua_tree[new_count] = t;
                    new_count++;
                }
            }
            /* Swap buffers */
            int *tmp;
            tmp = ua_node; ua_node = new_ua_node; new_ua_node = tmp;
            tmp = ua_tree; ua_tree = new_ua_tree; new_ua_tree = tmp;
            ua_count = new_count;
        }

        /* Phase B: greedy — pick lowest usage, tie-break lowest node ID */
        int new_count = 0;
        int progress = 0;
        for (int k = 0; k < ua_count; k++) {
            int v = ua_node[k], t = ua_tree[k];
            int16_t *tree = parent_arrays + (size_t)t * N;
            int best_parent = -1, best_usage = max_usage;
            for (int u = 0; u < N; u++) {
                if (!adj[v * N + u]) continue;
                if (u != root && tree[u] < 0) continue;
                if (no_parent[u]) continue;
                uint16_t fl = td.flink[(size_t)v * N + u];
                if (fl > 0 && fl < max_fl && edge_usage[fl] >= max_usage)
                    continue;
                int usage = (fl > 0 && fl < max_fl) ? edge_usage[fl] : 0;
                if (usage < best_usage ||
                    (usage == best_usage && (best_parent < 0 || u < best_parent))) {
                    best_usage = usage;
                    best_parent = u;
                }
            }
            if (best_parent >= 0) {
                tree[v] = (int16_t)best_parent;
                uint16_t fl = td.flink[(size_t)v * N + best_parent];
                if (fl > 0 && fl < max_fl) edge_usage[fl]++;
                progress = 1;
            } else {
                new_ua_node[new_count] = v;
                new_ua_tree[new_count] = t;
                new_count++;
            }
        }
        { int *tmp;
          tmp = ua_node; ua_node = new_ua_node; new_ua_node = tmp;
          tmp = ua_tree; ua_tree = new_ua_tree; new_ua_tree = tmp; }
        if (!progress && new_count == ua_count) { ua_count = new_count; break; }
        ua_count = new_count;
    }

    /* Repair step: 2-deep chain reparenting for stuck pairs */
    for (int depth = 0; depth < 2 && ua_count > 0; depth++) {
        int new_count = 0;
        for (int k = 0; k < ua_count; k++) {
            int v = ua_node[k], t = ua_tree[k];
            int16_t *tree_t = parent_arrays + (size_t)t * N;
            int resolved = 0;
            for (int u = 0; u < N && !resolved; u++) {
                if (!adj[v * N + u]) continue;
                uint16_t fl = td.flink[(size_t)v * N + u];
                if (fl == 0 || fl >= max_fl || edge_usage[fl] < max_usage) continue;
                if (u != root && tree_t[u] < 0) continue;
                /* Edge fl is saturated; try reparenting in another tree */
                for (int t2 = 0; t2 < tau && !resolved; t2++) {
                    if (t2 == t) continue;
                    int16_t *tree_t2 = parent_arrays + (size_t)t2 * N;
                    /* Find all victims in tree t2 using edge fl */
                    for (int w = 0; w < N && !resolved; w++) {
                        if (tree_t2[w] < 0) continue;
                        uint16_t wfl = td.flink[(size_t)w * N + tree_t2[w]];
                        if (wfl != fl) continue;
                        int old_parent = tree_t2[w];
                        /* Try reparenting w to an alternate neighbor */
                        for (int alt = 0; alt < N; alt++) {
                            if (alt == old_parent || !adj[w * N + alt]) continue;
                            if (alt != root && tree_t2[alt] < 0) continue;
                            /* Cycle check: alt must not descend from w */
                            int is_desc = 0;
                            for (int a = alt; a >= 0 && a != root; a = tree_t2[a])
                                if (a == w) { is_desc = 1; break; }
                            if (is_desc) continue;
                            uint16_t alt_fl = td.flink[(size_t)w * N + alt];
                            if (alt_fl > 0 && alt_fl < max_fl &&
                                edge_usage[alt_fl] >= max_usage) {
                                /* Depth-2: try freeing alt_fl first */
                                if (depth < 1) continue;
                                int freed = 0;
                                for (int t3 = 0; t3 < tau && !freed; t3++) {
                                    if (t3 == t2) continue;
                                    int16_t *tree_t3 = parent_arrays + (size_t)t3 * N;
                                    for (int w2 = 0; w2 < N && !freed; w2++) {
                                        if (tree_t3[w2] < 0) continue;
                                        if (td.flink[(size_t)w2 * N + tree_t3[w2]] != alt_fl)
                                            continue;
                                        int old_p2 = tree_t3[w2];
                                        for (int a2 = 0; a2 < N; a2++) {
                                            if (a2 == old_p2 || !adj[w2 * N + a2]) continue;
                                            if (a2 != root && tree_t3[a2] < 0) continue;
                                            int d2 = 0;
                                            for (int x = a2; x >= 0 && x != root;
                                                 x = tree_t3[x])
                                                if (x == w2) { d2 = 1; break; }
                                            if (d2) continue;
                                            uint16_t a2fl = td.flink[(size_t)w2 * N + a2];
                                            if (a2fl > 0 && a2fl < max_fl &&
                                                edge_usage[a2fl] >= max_usage)
                                                continue;
                                            /* Chain reparent */
                                            tree_t3[w2] = (int16_t)a2;
                                            edge_usage[alt_fl]--;
                                            if (a2fl > 0 && a2fl < max_fl) edge_usage[a2fl]++;
                                            freed = 1;
                                            break;
                                        }
                                    }
                                }
                                if (!freed) continue;
                            }
                            /* Reparent w in t2, assign v in t */
                            tree_t2[w] = (int16_t)alt;
                            edge_usage[fl]--;
                            if (alt_fl > 0 && alt_fl < max_fl) edge_usage[alt_fl]++;
                            tree_t[v] = (int16_t)u;
                            edge_usage[fl]++;
                            resolved = 1;
                            break;
                        }
                    }
                }
            }
            if (!resolved) {
                new_ua_node[new_count] = v;
                new_ua_tree[new_count] = t;
                new_count++;
            }
        }
        { int *tmp;
          tmp = ua_node; ua_node = new_ua_node; new_ua_node = tmp;
          tmp = ua_tree; ua_tree = new_ua_tree; new_ua_tree = tmp; }
        ua_count = new_count;
    }

    if (ua_count > 0)
        fprintf(stderr, "BBS: WARNING: %d unresolved (node,tree) pairs\n", ua_count);

    /* Per-tree stats */
    int max_depth = 0;
    for (int t = 0; t < tau; t++) {
        int16_t *tree = parent_arrays + (size_t)t * N;
        int d = tree_depth(tree, N);
        int rc = 0, excl = 0, shared = 0;
        for (int i = 0; i < N; i++) {
            if (tree[i] == (int16_t)root) rc++;
            if (tree[i] < 0) continue;
            uint16_t fl = td.flink[(size_t)tree[i] * N + i];
            if (fl > 0 && fl < max_fl) {
                if (edge_usage[fl] == 1) excl++;
                else shared++;
            }
        }
        fprintf(stderr, "BBS:   T%d: depth=%d rc=%d exclusive=%d shared=%d\n",
                t, d, rc, excl, shared);
        if (d > max_depth) max_depth = d;
    }

    /* Link sharing distribution */
    int sharing_max = 0;
    int sharing_counts[16] = {0};
    for (int i = 1; i < max_fl; i++) {
        if (edge_usage[i] > 0 && edge_usage[i] < 16) sharing_counts[edge_usage[i]]++;
        if (edge_usage[i] > sharing_max) sharing_max = edge_usage[i];
    }
    fprintf(stderr, "BBS: Link sharing: max=%d", sharing_max);
    for (int s = 1; s <= sharing_max && s < 16; s++)
        if (sharing_counts[s] > 0) fprintf(stderr, " [%d]=%d", s, sharing_counts[s]);
    fprintf(stderr, "\n");

    fprintf(stderr, "BBS: tau=%d max_depth=%d baseline=%d\n",
            tau, max_depth, depth0);

    /* Verify spanning */
    for (int t = 0; t < tau; t++) {
        int covered = 0;
        for (int i = 0; i < N; i++)
            if (parent_arrays[(size_t)t * N + i] >= 0 || i == root) covered++;
        if (covered != N)
            fprintf(stderr, "BBS: WARNING: Tree %d covers only %d/%d nodes!\n",
                    t, covered, N);
    }

    *out_tau = tau;

    free(opt_parent); free(opt_usage);
    free(ua_node); free(ua_tree);
    free(new_ua_node); free(new_ua_tree);
    free(bfs_order);
    free(edge_usage); free(no_parent); free(node_deg);
    free(t0_bfs); free(visited); free(queue); free(adj);
    topo_data_free(&td);
    return 0;
}
