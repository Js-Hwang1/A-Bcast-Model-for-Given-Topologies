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

    /* Dragonfly: all compute nodes have same degree = npn-1 in 1-hop
     * adjacency (connected to sibling nodes on the same router).
     * N=128 (npn=2) → deg=1, N=256 (npn=4) → deg=3, etc.
     * Must be N = 64*npn with npn = min_deg+1.
     * Check BEFORE mesh since npn=4 dragonfly has deg=3 which overlaps mesh range. */
    if (min_deg == max_deg && N >= 8) {
        int npn = min_deg + 1;
        /* 16 = NCHASSIS * NROUTERS (always 4*4, fixed) */
        if (npn >= 1 && N % (16 * npn) == 0)
            return BBS_TOPO_DRAGONFLY;
    }

    /* 2D mesh: min degree 2 (corners), max degree 4 (interior) */
    if (min_deg >= 2 && max_deg <= 4)
        return BBS_TOPO_MESH;

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
 * Topology: 4 groups × 4 chassis × 4 routers × npn nodes.
 * N=128 → npn=2, N=256 → npn=4, etc. */
static int dfly_npn = 2;         /* nodes per router — set in dragonfly_compute_trees */
static int dfly_use_shallow = 0; /* 0 = deep hierarchical, 1 = shallow relay pair */
#define DFLY_GROUP(v)   ((v) / (dfly_npn * 16))
#define DFLY_CHASSIS(v) (((v) % (dfly_npn * 16)) / (dfly_npn * 4))
#define DFLY_ROUTER(v)  (((v) % (dfly_npn * 4)) / dfly_npn)
#define DFLY_SIBLING(v) ((v) ^ 1)
#define DFLY_MAX_GROUPS 32
static int dfly_ngroups = 4;  /* set in dragonfly_compute_trees */
#define NCHASSIS 4
#define NROUTERS 4


/* Compute physical link congestion for a set of trees and print stats.
 * lat may be NULL (early detection path — skip physical congestion). */
static void dfly_print_stats(int16_t **trees, int tau, int root, int N,
                              const float *lat)
{
    (void)lat;

    /* Nc = number of compute nodes; router ranks are Nc..N-1 */
    int Nc = dfly_ngroups * NCHASSIS * NROUTERS * dfly_npn;
    int rtr_base = Nc;

    /* Aggregate physical link congestion across ALL trees */
    int blue_cong[DFLY_MAX_GROUPS][DFLY_MAX_GROUPS];
    int black_cong[DFLY_MAX_GROUPS][NCHASSIS][NCHASSIS];
    int green_cong[DFLY_MAX_GROUPS][NCHASSIS][NROUTERS][NROUTERS];
    memset(blue_cong, 0, sizeof(blue_cong));
    memset(black_cong, 0, sizeof(black_cong));
    memset(green_cong, 0, sizeof(green_cong));
    int *uplink_load = (int *)calloc(N, sizeof(int));

    for (int t = 0; t < tau; t++) {
        int16_t *tree = trees[t];
        for (int i = 0; i < N; i++) {
            if (tree[i] < 0) continue;
            int p = tree[i];
            int ig, ic, ir, pg, pc, pr;

            /* Extract g/c/r for node i */
            if (i < Nc) {
                ig = DFLY_GROUP(i); ic = DFLY_CHASSIS(i); ir = DFLY_ROUTER(i);
            } else {
                int ri = i - rtr_base;
                ig = ri / (NCHASSIS * NROUTERS);
                ic = (ri % (NCHASSIS * NROUTERS)) / NROUTERS;
                ir = ri % NROUTERS;
            }
            /* Extract g/c/r for parent p */
            if (p < Nc) {
                pg = DFLY_GROUP(p); pc = DFLY_CHASSIS(p); pr = DFLY_ROUTER(p);
            } else {
                int rp = p - rtr_base;
                pg = rp / (NCHASSIS * NROUTERS);
                pc = (rp % (NCHASSIS * NROUTERS)) / NROUTERS;
                pr = rp % NROUTERS;
            }

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
    for (int g1 = 0; g1 < dfly_ngroups; g1++)
        for (int g2 = g1+1; g2 < dfly_ngroups; g2++)
            if (blue_cong[g1][g2] > max_blue) max_blue = blue_cong[g1][g2];
    for (int g = 0; g < dfly_ngroups; g++)
        for (int c1 = 0; c1 < NCHASSIS; c1++)
            for (int c2 = c1+1; c2 < NCHASSIS; c2++)
                if (black_cong[g][c1][c2] > max_black) max_black = black_cong[g][c1][c2];
    for (int g = 0; g < dfly_ngroups; g++)
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
        int *tfan = (int *)calloc(N, sizeof(int));
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
        free(tfan);
    }

    /* Anti-correlation analysis */
    if (tau == 2) {
        int *fan0 = (int *)calloc(N, sizeof(int));
        int *fan1 = (int *)calloc(N, sizeof(int));
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
        free(fan0); free(fan1);
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
    for (int g1 = 0; g1 < dfly_ngroups; g1++)
        for (int g2 = g1+1; g2 < dfly_ngroups; g2++)
            if (blue_cong[g1][g2] > 0)
                fprintf(stderr, " [%d-%d]=%d", g1, g2, blue_cong[g1][g2]);
    fprintf(stderr, "\n");
    free(uplink_load);
}

/* Build tau=2 hierarchical anti-correlated trees for dragonfly.
 *
 * Algorithm (from algorithm.txt):
 *   0. Identify root/leaf at every level (node, router, chassis, group)
 *      based on the broadcast root — no renumbering.
 *   1. Node level:  within each router, T0=T1=root→leaf (npn=2 degenerate).
 *   2. Router level: K4 relay on 4 routers per chassis.
 *   3. Connect: leaf_node(src_router) → root_node(dst_router).
 *   4. Chassis level: K4 relay on 4 chassis per group.
 *   5. Connect: leaf_node(leaf_router(src_ch)) → root_node(root_router(dst_ch)).
 *   6. Group level: K4 relay on 4 groups.
 *   7. Connect: leaf_node(leaf_rtr(leaf_ch(src_g))) → root_node(root_rtr(root_ch(dst_g))).
 *
 * K4 relay pattern (root=R, leaf=L, interior={I0,I1} sorted):
 *   T0: R → I0 → {I1, L}
 *   T1: R → I1 → {I0, L}
 *   L is leaf in BOTH trees.  Anti-correlation on I0/I1.
 *
 * Max union degree = 2*tau = 4 across all N nodes.
 */
/* Shallow relay-pair trees (depth ≈ 9).
 * Binary-dissemination at each level: group, chassis, router, sibling.
 * Good for small messages where low depth dominates. */
/* Hierarchical trees with K4 relay pattern at each level.
 * Levels: node (within router), router (within chassis),
 *         chassis (within group), group (global).
 *
 * At each level, designate INGRESS (receives from above, union deg 2)
 * and EGRESS (sends to below, union deg 2).
 *
 * shallow=0 (deep):    cross-level edges go egress(src) → ingress(dst).
 *                      Data traverses full sub-tree before crossing levels.
 *                      Deeper tree, better pipeline throughput.
 * shallow=1 (shallow): cross-level edges go ingress(src) → ingress(dst).
 *                      Ingress does double duty (local sub-tree + cross-level).
 *                      Shallower tree, lower startup latency. */
static void dfly_build_hierarchical_trees(int16_t *t0, int16_t *t1,
                                           int root, int N, int shallow)
{
    int npn = dfly_npn;
    int Nc = dfly_ngroups * NCHASSIS * NROUTERS * npn;
    int n_rtrs = Nc / npn;
    int rtr_base = Nc;  /* first router rank */

    for (int i = 0; i < N; i++) { t0[i] = -1; t1[i] = -1; }

    int root_g = DFLY_GROUP(root);
    int root_c = DFLY_CHASSIS(root);
    int root_r = DFLY_ROUTER(root);

    /* --- Step 0: Ingress/Egress identification ---
     *
     * With router relays, cross-level edges insert router MPI ranks:
     *   ingress_node[dst_rtr] → rtr_rank[dst_rtr] → rtr_rank[src_rtr]
     * The root's own router connects to egress_node[root_rtr]. */

    /* Per-router: ingress/egress compute node IDs */
    int *ingress_node = (int *)malloc(n_rtrs * sizeof(int));
    int *egress_node  = (int *)malloc(n_rtrs * sizeof(int));
    for (int rtr = 0; rtr < n_rtrs; rtr++) {
        int base = rtr * npn;
        int g = rtr / (NCHASSIS * NROUTERS);
        int cl = (rtr % (NCHASSIS * NROUTERS)) / NROUTERS;
        int rl = rtr % NROUTERS;
        if (g == root_g && cl == root_c && rl == root_r) {
            ingress_node[rtr] = root;
            egress_node[rtr] = (npn == 2)
                ? (root == base ? base + 1 : base)
                : (root == base + npn - 1 ? base + npn - 2 : base + npn - 1);
        } else {
            ingress_node[rtr] = base;
            egress_node[rtr] = base + npn - 1;
        }
    }

    /* Per-chassis: ingress/egress router (local index 0..3) */
    int total_ch = dfly_ngroups * NCHASSIS;
    int *ingress_rtr = (int *)malloc(total_ch * sizeof(int));
    int *egress_rtr  = (int *)malloc(total_ch * sizeof(int));
    for (int ch = 0; ch < total_ch; ch++) {
        int g = ch / NCHASSIS, cl = ch % NCHASSIS;
        if (g == root_g && cl == root_c) {
            ingress_rtr[ch] = root_r;
            egress_rtr[ch] = (root_r == NROUTERS - 1)
                              ? NROUTERS - 2 : NROUTERS - 1;
        } else {
            ingress_rtr[ch] = 0;
            egress_rtr[ch] = NROUTERS - 1;
        }
    }

    /* Per-group: ingress/egress chassis (local index 0..3) */
    int ingress_ch[DFLY_MAX_GROUPS], egress_ch[DFLY_MAX_GROUPS];
    for (int g = 0; g < dfly_ngroups; g++) {
        if (g == root_g) {
            ingress_ch[g] = root_c;
            egress_ch[g] = (root_c == NCHASSIS - 1)
                            ? NCHASSIS - 2 : NCHASSIS - 1;
        } else {
            ingress_ch[g] = 0;
            egress_ch[g] = NCHASSIS - 1;
        }
    }

    /* Global: leaf group (egress target at group level) */
    int leaf_group = (root_g == dfly_ngroups - 1) ? dfly_ngroups - 2 : dfly_ngroups - 1;

    /* --- Step 1: Node-level trees (within each router) ---
     * Unchanged — only touches compute ranks 0..Nc-1.
     * No router relay needed (same physical router). */
    for (int rtr = 0; rtr < n_rtrs; rtr++) {
        int in_n = ingress_node[rtr], eg_n = egress_node[rtr];
        if (npn == 2) {
            t0[eg_n] = (int16_t)in_n;
            t1[eg_n] = (int16_t)in_n;
        } else if (npn >= 4) {
            int base = rtr * npn;
            int ord[64], on = 0;
            ord[on++] = in_n;
            for (int i = base; i < base + npn; i++)
                if (i != in_n && i != eg_n) ord[on++] = i;
            ord[on++] = eg_n;

            for (int i = 1; i < on; i++)
                t0[ord[i]] = (int16_t)ord[i / 2];

            int sigma[64];
            for (int i = 0; i < on; i++) sigma[i] = i;
            int half = (on - 2) / 2;
            for (int i = 1; i <= half; i++) {
                int j = i + half;
                if (j < on - 1) {
                    int tmp = sigma[i]; sigma[i] = sigma[j]; sigma[j] = tmp;
                }
            }
            for (int i = 1; i < on; i++)
                t1[ord[sigma[i]]] = (int16_t)ord[sigma[i / 2]];
        }
    }

    /* --- Steps 2+3: Router-level trees + cross-level connect ---
     * With router relays:
     *   ingress_node[dst] → rtr_rank[dst] → rtr_rank[src]
     * (instead of old: ingress_node[dst] → egress_node[src]) */
    for (int ch = 0; ch < total_ch; ch++) {
        int in_r = ingress_rtr[ch], eg_r = egress_rtr[ch];

        int ord[NROUTERS], on = 0;
        ord[on++] = in_r;
        for (int r = 0; r < NROUTERS; r++)
            if (r != in_r && r != eg_r) ord[on++] = r;
        ord[on++] = eg_r;

        int rpar[2][NROUTERS];
        for (int r = 0; r < NROUTERS; r++) { rpar[0][r] = -1; rpar[1][r] = -1; }
        for (int i = 1; i < on; i++)
            rpar[0][ord[i]] = ord[i / 2];

        int sigma[NROUTERS];
        for (int i = 0; i < on; i++) sigma[i] = i;
        int half = (on - 2) / 2;
        for (int i = 1; i <= half; i++) {
            int j = i + half;
            if (j < on - 1) {
                int tmp = sigma[i]; sigma[i] = sigma[j]; sigma[j] = tmp;
            }
        }
        for (int i = 1; i < on; i++)
            rpar[1][ord[sigma[i]]] = ord[sigma[i / 2]];

        /* Router relay cross-connect */
        int16_t *tt[2] = { t0, t1 };
        for (int t = 0; t < 2; t++)
            for (int r = 0; r < NROUTERS; r++) {
                if (rpar[t][r] < 0) continue;
                int dst = ch * NROUTERS + r;        /* global router index */
                int src = ch * NROUTERS + rpar[t][r];
                int dst_rtr_rank = rtr_base + dst;
                int src_rtr_rank = rtr_base + src;
                tt[t][ingress_node[dst]] = (int16_t)dst_rtr_rank;
                tt[t][dst_rtr_rank] = (int16_t)src_rtr_rank;
            }
    }

    /* --- Steps 4+5: Chassis-level trees + cross-level connect ---
     * With router relays:
     *   ingress_node[dst_rtr] → rtr_rank[dst_rtr] → rtr_rank[src_rtr] */
    for (int g = 0; g < dfly_ngroups; g++) {
        int in_c = ingress_ch[g], eg_c = egress_ch[g];

        int ord[NCHASSIS], on = 0;
        ord[on++] = in_c;
        for (int c = 0; c < NCHASSIS; c++)
            if (c != in_c && c != eg_c) ord[on++] = c;
        ord[on++] = eg_c;

        int cpar[2][NCHASSIS];
        for (int c = 0; c < NCHASSIS; c++) { cpar[0][c] = -1; cpar[1][c] = -1; }
        for (int i = 1; i < on; i++)
            cpar[0][ord[i]] = ord[i / 2];

        int sigma[NCHASSIS];
        for (int i = 0; i < on; i++) sigma[i] = i;
        int half = (on - 2) / 2;
        for (int i = 1; i <= half; i++) {
            int j = i + half;
            if (j < on - 1) {
                int tmp = sigma[i]; sigma[i] = sigma[j]; sigma[j] = tmp;
            }
        }
        for (int i = 1; i < on; i++)
            cpar[1][ord[sigma[i]]] = ord[sigma[i / 2]];

        int16_t *tt[2] = { t0, t1 };
        for (int t = 0; t < 2; t++)
            for (int c = 0; c < NCHASSIS; c++) {
                if (cpar[t][c] < 0) continue;
                int dst_ch = g * NCHASSIS + c;
                int src_ch = g * NCHASSIS + cpar[t][c];
                int dst_rtr = dst_ch * NROUTERS + ingress_rtr[dst_ch];
                int src_rtr = src_ch * NROUTERS + egress_rtr[src_ch];
                int dst_rtr_rank = rtr_base + dst_rtr;
                int src_rtr_rank = rtr_base + src_rtr;
                tt[t][dst_rtr_rank] = (int16_t)src_rtr_rank;
                tt[t][ingress_node[dst_rtr]] = (int16_t)dst_rtr_rank;
            }
    }

    /* --- Steps 6+7: Group-level trees + cross-level connect ---
     * With router relays:
     *   ingress_node[dst_rtr] → rtr_rank[dst_rtr] → rtr_rank[src_rtr] */
    {
        int ng = dfly_ngroups;
        int ord[DFLY_MAX_GROUPS];
        int n = 0;
        ord[n++] = root_g;
        int interior[DFLY_MAX_GROUPS];
        int ni = 0;
        for (int g = 0; g < ng; g++)
            if (g != root_g && g != leaf_group) interior[ni++] = g;
        for (int i = 0; i < ni - 1; i++)
            for (int j = i + 1; j < ni; j++)
                if (interior[i] > interior[j])
                    { int tmp = interior[i]; interior[i] = interior[j]; interior[j] = tmp; }
        for (int i = 0; i < ni; i++)
            ord[n++] = interior[i];
        ord[n++] = leaf_group;

        int gpar[2][DFLY_MAX_GROUPS];
        for (int g = 0; g < ng; g++) { gpar[0][g] = -1; gpar[1][g] = -1; }

        for (int i = 1; i < n; i++)
            gpar[0][ord[i]] = ord[i / 2];

        int sigma[DFLY_MAX_GROUPS];
        for (int i = 0; i < n; i++) sigma[i] = i;
        int h = (n - 2) / 2;
        for (int i = 1; i <= h; i++) {
            int j = i + h;
            if (j < n - 1) {
                int tmp = sigma[i]; sigma[i] = sigma[j]; sigma[j] = tmp;
            }
        }
        for (int i = 1; i < n; i++)
            gpar[1][ord[sigma[i]]] = ord[sigma[i / 2]];

        int16_t *tt[2] = { t0, t1 };
        for (int t = 0; t < 2; t++)
            for (int g = 0; g < ng; g++) {
                if (gpar[t][g] < 0) continue;
                int sg = gpar[t][g];
                int src_ch  = sg * NCHASSIS + egress_ch[sg];
                int src_rtr = src_ch * NROUTERS + egress_rtr[src_ch];
                int dst_ch  = g * NCHASSIS + ingress_ch[g];
                int dst_rtr = dst_ch * NROUTERS + ingress_rtr[dst_ch];
                int dst_rtr_rank = rtr_base + dst_rtr;
                int src_rtr_rank = rtr_base + src_rtr;
                tt[t][dst_rtr_rank] = (int16_t)src_rtr_rank;
                tt[t][ingress_node[dst_rtr]] = (int16_t)dst_rtr_rank;
            }
    }

    /* --- Root's router: connect to egress_node ---
     * The root's own router rank must have a parent in the tree.
     * It points to egress_node[root_rtr] (which is a child of root
     * in the node-level tree). */
    {
        int root_global_rtr = root_g * NCHASSIS * NROUTERS + root_c * NROUTERS + root_r;
        t0[rtr_base + root_global_rtr] = (int16_t)egress_node[root_global_rtr];
        t1[rtr_base + root_global_rtr] = (int16_t)egress_node[root_global_rtr];
    }

    (void)shallow;
    free(ingress_node); free(egress_node);
    free(ingress_rtr); free(egress_rtr);
}

static int dragonfly_compute_trees(const char *topo_file, int root, int N,
                                   const char *adj, const uint16_t *flink,
                                   int max_fl, const int *bfs_order, int bfs_n,
                                   int depth0,
                                   int16_t *parent_arrays, int *out_tau)
{
    (void)flink; (void)max_fl;
    (void)bfs_order; (void)bfs_n; (void)depth0;

    int Nc, Nr;

    if (!adj) {
        /* Early detection path: read G,C,R,P from cfg file */
        char dir[512] = {0};
        strncpy(dir, topo_file, sizeof(dir) - 1);
        char *sl = strrchr(dir, '/');
        if (sl) *(sl + 1) = '\0';
        else    strcpy(dir, "./");

        int nc_from_name = 0;
        const char *dp = strstr(topo_file, "dragonfly_");
        if (dp) {
            nc_from_name = atoi(dp + 10);
        } else {
            const char *tp = strstr(topo_file, "topo_");
            if (tp) nc_from_name = atoi(tp + 5);
        }
        if (nc_from_name <= 0) {
            fprintf(stderr, "DFLY: ERROR: cannot determine Nc from path\n");
            *out_tau = 0;
            return -1;
        }

        char cfg_path[512];
        snprintf(cfg_path, sizeof(cfg_path), "%stopo_%d.cfg", dir, nc_from_name);
        FILE *f = fopen(cfg_path, "r");
        if (!f) {
            fprintf(stderr, "DFLY: ERROR: cannot open %s\n", cfg_path);
            *out_tau = 0;
            return -1;
        }
        int dg = 0, dc = 0, dr = 0, dp2 = 0;
        char line[128] = {0};
        if (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "dragonfly", 9) == 0)
                sscanf(line + 10, "%d %d %d %d", &dg, &dc, &dr, &dp2);
        }
        fclose(f);

        if (dg <= 0 || dc != NCHASSIS || dr != NROUTERS || dp2 <= 0) {
            fprintf(stderr, "DFLY: ERROR: bad cfg G=%d C=%d R=%d P=%d\n",
                    dg, dc, dr, dp2);
            *out_tau = 0;
            return -1;
        }

        dfly_npn = dp2;
        dfly_ngroups = dg;
        Nc = dg * dc * dr * dp2;
        Nr = dg * dc * dr;

        if (Nc + Nr != N) {
            fprintf(stderr, "DFLY: ERROR: Nc(%d) + Nr(%d) = %d != N(%d)\n",
                    Nc, Nr, Nc + Nr, N);
            *out_tau = 0;
            return -1;
        }
    } else {
        /* Adjacency path: no longer supported — router relays require
         * the early detection path with NP-sized rank space. */
        fprintf(stderr, "DFLY: ERROR: adjacency path not supported "
                "(use early detection via hostfile_bbs)\n");
        *out_tau = 0;
        return -1;
    }

    fprintf(stderr, "DFLY: root=%d group=%d chassis=%d router=%d npn=%d "
            "ngroups=%d Nc=%d Nr=%d N=%d\n",
            root, DFLY_GROUP(root), DFLY_CHASSIS(root), DFLY_ROUTER(root),
            dfly_npn, dfly_ngroups, Nc, Nr, N);

    int tau = 2;
    int16_t *trees[2] = { parent_arrays, parent_arrays + N };

    dfly_build_hierarchical_trees(trees[0], trees[1], root, N,
                                   dfly_use_shallow ? 1 : 0);

    dfly_print_stats(trees, tau, root, N, NULL);
    fprintf(stderr, "DFLY: tau=%d variant=%s\n", tau,
            dfly_use_shallow ? "shallow" : "deep");

    *out_tau = tau;
    return 0;
}

/* ---- FatTree tree builder (tau=2, two-layer anti-correlated binary trees) ----
 *
 * Uses leaf/spine switch hosts as MPI relay ranks for explicit routing.
 * Rank mapping (for Nc compute, Nl leaf, Ns spine):
 *   0 .. Nc-1           : compute nodes
 *   Nc .. Nc+Nl-1       : leaf switches
 *   Nc+Nl .. Nc+Nl+Ns-1 : spine switches
 *
 * Layer 1 (intra-leaf): two hardcoded anti-correlated binary spanning trees
 * over the 16 compute nodes within each leaf router.
 *   - Abstract pos 0 = ingress (receives from leaf switch)
 *   - Abstract pos 15 = egress (sends to leaf switch / spines)
 *   - Max union degree = 4
 *
 * Layer 2 (inter-leaf): two anti-correlated binary spanning trees over the
 * Nl leaf routers, routed through explicit spine relays.
 *   - T1: standard binary tree parent(p) = p/2
 *   - T2: permuted — T1 leaves ↔ T1 internals, root and last node fixed
 *   - Spine assignment: spine = (parent_al + t * Ns/2) % Ns
 *     Anti-correlated: T1 and T2 use different spines for same parent.
 *
 * Inter-leaf edge (parent_leaf → child_leaf via spine S):
 *   egress(pos15) → leaf_switch(parent) → spine_S → leaf_switch(child) → ingress(pos0)
 *   [Root's leaf uses leaf_switch as relay; non-root leaves egress→spine directly]
 */

/* Hardcoded 16-node intra-leaf parent arrays */
static const int8_t FT_INTRA_T1[16] = {-1, 0, 1, 1, 5, 2, 2, 6, 5, 3, 3, 6, 9, 9, 10, 10};
static const int8_t FT_INTRA_T2[16] = {-1, 7, 7, 11, 0, 11, 13, 8, 4, 14, 13, 8, 4, 12, 12, 14};

/* Inter-leaf T2 permutation sigma: position → node.
 * Non-involutory swap of A={1..h} and B={h+1..2h} where h=Nl/2-1.
 * Nodes 0 and Nl-1 fixed.  Produces edge-disjoint trees with
 * union degree [2,4,...,4,2].  Requires Nl >= 8. */
static int ft_inter_node_of(int pos, int Nl) {
    int h = Nl / 2 - 1;
    if (pos == 0 || pos == Nl - 1) return pos;
    if (pos <= h - 2)  return h + pos;   /* A bulk → B */
    if (pos == h - 1)  return 2 * h;     /* A second-last → B last */
    if (pos == h)      return 2 * h - 1; /* A last → B second-last */
    if (pos == h + 1)  return 2;         /* B first → A second */
    if (pos == h + 2)  return 1;         /* B second → A first */
    return pos - h;                      /* B bulk → A */
}

/* Inter-leaf T2 permutation inverse: node → position. */
static int ft_inter_pos_of(int node, int Nl) {
    int h = Nl / 2 - 1;
    if (node == 0 || node == Nl - 1) return node;
    if (node == 1)      return h + 2;
    if (node == 2)      return h + 1;
    if (node >= 3 && node <= h) return node + h;  /* A → B positions */
    if (node <= 2*h - 2) return node - h;         /* B bulk → A */
    if (node == 2*h - 1) return h;
    return h - 1;                                 /* node == 2*h */
}

static int fattree_compute_trees(const char *topo_file, int root, int N,
                                 const char *adj, const uint16_t *flink,
                                 int max_fl, const int *bfs_order, int bfs_n,
                                 int depth0,
                                 int16_t *parent_arrays, int *out_tau)
{
    (void)adj; (void)flink; (void)max_fl;
    (void)bfs_order; (void)bfs_n; (void)depth0;

    /* ---- Read npl from cfg ---- */
    int npl = 0;
    {
        char dir[512] = {0};
        strncpy(dir, topo_file, sizeof(dir) - 1);
        char *sl = strrchr(dir, '/');
        if (sl) *(sl + 1) = '\0';
        else    strcpy(dir, "./");

        int nc_from_name = 0;
        const char *ft = strstr(topo_file, "fattree_");
        if (ft) {
            nc_from_name = atoi(ft + 8);
        } else {
            const char *tp = strstr(topo_file, "topo_");
            if (tp) nc_from_name = atoi(tp + 5);
        }
        if (nc_from_name > 0) {
            char cfg_path[512];
            snprintf(cfg_path, sizeof(cfg_path),
                     "%stopo_%d.cfg", dir, nc_from_name);
            FILE *f = fopen(cfg_path, "r");
            if (f) {
                char line[64] = {0};
                if (fgets(line, sizeof(line), f))
                    if (strncmp(line, "fattree", 7) == 0)
                        npl = atoi(line + 8);
                fclose(f);
            }
        }
        if (npl <= 0) {
            fprintf(stderr, "FTREE: ERROR: cannot determine npl "
                    "(N=%d root=%d)\n", N, root);
            *out_tau = 0;
            return -1;
        }
    }

    /* ---- Derive topology: N = Nc + Nl + Ns, Ns = 2*(Nl-1) ----
     * N = Nl*npl + Nl + 2*(Nl-1) = Nl*(npl+3) - 2
     * => Nl = (N+2) / (npl+3)
     */
    int Nl = (N + 2) / (npl + 3);
    int Nc = Nl * npl;
    int Ns = 2 * (Nl - 1);   /* one spine rank per inter-leaf edge */

    if (Nc + Nl + Ns != N || Nc <= 0 || Nl < 2) {
        fprintf(stderr, "FTREE: ERROR: topology mismatch Nc=%d Nl=%d Ns=%d "
                "!= N=%d  (expected N = Nl*(npl+3)-2 = %d)\n",
                Nc, Nl, Ns, N, Nl * (npl + 3) - 2);
        *out_tau = 0;
        return -1;
    }

    int leaf_base  = Nc;
    int spine_base = Nc + Nl;

    #define FT_LEAF(l)  (leaf_base  + (l))
    #define FT_SPINE(s) (spine_base + (s))

    int tau = 2;
    if (tau > BBS_MAX_TREES) tau = BBS_MAX_TREES;

    int root_leaf = root / npl;

    fprintf(stderr, "FTREE: N=%d Nc=%d Nl=%d Ns=%d npl=%d root=%d "
            "root_leaf=%d tau=%d\n", N, Nc, Nl, Ns, npl, root, root_leaf, tau);

    /* ---- Build rank maps ----
     * rank_map[leaf][abstract_pos] = actual compute rank
     *
     * Root's leaf: abstract pos 0 = root rank, pos 1..15 = remaining in order
     * Other leaves: abstract pos i = base + i (identity)
     */
    int rank_map[128][16];  /* max 128 leaves */
    for (int l = 0; l < Nl; l++) {
        int base = l * npl;
        if (l == root_leaf) {
            rank_map[l][0] = root;
            int slot = 1;
            for (int i = 0; i < npl; i++) {
                if (base + i == root) continue;
                rank_map[l][slot++] = base + i;
            }
        } else {
            for (int i = 0; i < npl; i++)
                rank_map[l][i] = base + i;
        }
    }

    /* ---- Build leaf_map: abstract inter-leaf index → actual leaf ----
     * Abstract leaf 0 = root's actual leaf, remaining in natural order.
     */
    int leaf_map[128];
    leaf_map[0] = root_leaf;
    {
        int slot = 1;
        for (int l = 0; l < Nl; l++) {
            if (l == root_leaf) continue;
            leaf_map[slot++] = l;
        }
    }

    /* ---- Build tau trees ---- */
    for (int t = 0; t < tau; t++) {
        int16_t *tree = parent_arrays + (size_t)t * N;
        for (int i = 0; i < N; i++) tree[i] = -1;

        const int8_t *intra = (t == 0) ? FT_INTRA_T1 : FT_INTRA_T2;

        /* 1. Intra-leaf edges for all leaves */
        for (int al = 0; al < Nl; al++) {
            int actual_leaf = leaf_map[al];
            for (int pos = 1; pos < npl; pos++) {
                int par_pos = (int)intra[pos];
                tree[rank_map[actual_leaf][pos]] =
                    (int16_t)rank_map[actual_leaf][par_pos];
            }
        }

        /* 2. Root's leaf switch: child of egress (pos 15) */
        tree[FT_LEAF(root_leaf)] = (int16_t)rank_map[root_leaf][15];

        /* 3. Inter-leaf edges: egress → spine_rank → child_leaf_sw → ingress
         *    Each edge gets a unique spine rank:
         *    spine_rank = spine_base + t*(Nl-1) + (al-1)
         */
        for (int al = 1; al < Nl; al++) {
            int actual_leaf = leaf_map[al];

            /* Compute inter-leaf parent for abstract leaf al */
            int parent_al;
            if (t == 0) {
                parent_al = al / 2;
            } else {
                int pos = ft_inter_pos_of(al, Nl);
                int parent_pos = pos / 2;
                parent_al = ft_inter_node_of(parent_pos, Nl);
            }

            int parent_leaf = leaf_map[parent_al];
            int spine_rank = spine_base + t * (Nl - 1) + (al - 1);

            /* Spine parent: root's leaf switch, or egress of non-root parent */
            if (parent_al == 0) {
                tree[spine_rank] = (int16_t)FT_LEAF(root_leaf);
            } else {
                tree[spine_rank] =
                    (int16_t)rank_map[parent_leaf][15];
            }

            /* Child leaf switch parent = spine */
            tree[FT_LEAF(actual_leaf)] = (int16_t)spine_rank;

            /* Ingress (pos 0) parent = child's leaf switch */
            tree[rank_map[actual_leaf][0]] = (int16_t)FT_LEAF(actual_leaf);
        }
    }

    /* 4. Attach unused switch ranks to childless compute nodes */
    for (int t = 0; t < tau; t++) {
        int16_t *tree = parent_arrays + (size_t)t * N;

        /* Find childless compute nodes */
        int *has_child = (int *)calloc(N, sizeof(int));
        for (int i = 0; i < N; i++)
            if (tree[i] >= 0) has_child[(int)tree[i]] = 1;

        int *childless = (int *)malloc(Nc * sizeof(int));
        int ncl = 0;
        for (int i = 0; i < Nc; i++)
            if (!has_child[i]) childless[ncl++] = i;

        /* Find unused switch ranks (not root, no parent) */
        int *unused_r = (int *)malloc((Nl + Ns) * sizeof(int));
        int nun = 0;
        for (int i = Nc; i < N; i++)
            if (tree[i] < 0 && i != root) unused_r[nun++] = i;

        /* Attach each unused rank as a child of a childless compute node */
        int ci = 0;
        for (int u = 0; u < nun && ci < ncl; u++, ci++)
            tree[unused_r[u]] = (int16_t)childless[ci];

        free(childless);
        free(unused_r);
        free(has_child);
    }

    /* ---- Per-tree stats ---- */
    for (int t = 0; t < tau; t++) {
        int16_t *tree = parent_arrays + (size_t)t * N;
        int d = tree_depth(tree, N);

        int *fan = (int *)calloc(N, sizeof(int));
        int max_fan = 0;
        for (int i = 0; i < N; i++)
            if (tree[i] >= 0) fan[(int)tree[i]]++;
        for (int i = 0; i < N; i++)
            if (fan[i] > max_fan) max_fan = fan[i];

        int covered = 0;
        for (int i = 0; i < N; i++)
            if (tree[i] >= 0 || i == root) covered++;

        fprintf(stderr, "FTREE: T%d: depth=%d max_fan=%d covered=%d/%d\n",
                t, d, max_fan, covered, N);
        if (covered != N)
            fprintf(stderr, "FTREE: WARNING: T%d spans only %d/%d!\n",
                    t, covered, N);

        free(fan);
    }

    /* ---- Spine usage per tree ---- */
    {
        for (int t = 0; t < tau; t++) {
            int16_t *tree = parent_arrays + (size_t)t * N;
            int used = 0;
            int base = spine_base + t * (Nl - 1);
            int count = Nl - 1;
            fprintf(stderr, "FTREE: T%d spines:", t);
            for (int s = 0; s < count; s++) {
                int sr = base + s;
                if (tree[sr] >= 0) {
                    used++;
                    int ch = 0;
                    for (int i = 0; i < N; i++)
                        if (tree[i] == (int16_t)sr) ch++;
                    fprintf(stderr, " r%d(%dch)", sr, ch);
                }
            }
            fprintf(stderr, " [%d/%d used]\n", used, count);
        }
    }

    #undef FT_LEAF
    #undef FT_SPINE

    *out_tau = tau;
    return 0;
}

/* ---- Write .bbs from parent arrays ---- */
/* parent_arrays: tau × N, row-major.  parent_arrays[t*N + i] = parent of i in tree t. */
static int bbs_write(const char *path, int N, int tau, int root,
                     const int16_t *parent_arrays)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Compute max depth across all trees (store in header for auto-chunking) */
    int maxdepth = 0;
    for (int t = 0; t < tau; t++) {
        int d = tree_depth(parent_arrays + (size_t)t * N, N);
        if (d > maxdepth) maxdepth = d;
    }
    if (maxdepth > 255) maxdepth = 255;

    /* Header: magic(2) N(2) tau(1) depth(1) root(2) = 8 bytes */
    uint16_t h_magic = BBS_MAGIC, h_N = (uint16_t)N, h_root = (uint16_t)root;
    uint8_t  h_tau = (uint8_t)tau, h_pad = (uint8_t)maxdepth;
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
    /* ---- Early FatTree detection ----
     * FatTree uses pure compute-node ranks (no switch ranks).
     * Detect from directory name and dispatch to the dedicated
     * two-layer anti-correlated tree builder. */
    {
        const char *base = strrchr(topo_file, '/');
        if (base) {
            const char *dir_end = base;
            const char *dir_start = topo_file;
            for (const char *p = topo_file; p < dir_end; p++)
                if (*p == '/') dir_start = p + 1;
            int dlen = (int)(dir_end - dir_start);
            if (dlen == 7 && memcmp(dir_start, "FatTree", 7) == 0) {
                fprintf(stderr, "BBS: FatTree topology detected (N=%d)\n", N);
                return fattree_compute_trees(topo_file, root, N,
                                             NULL, NULL, 0, NULL, 0, 0,
                                             parent_arrays, out_tau);
            }
            if (dlen == 9 && memcmp(dir_start, "Dragonfly", 9) == 0) {
                fprintf(stderr, "BBS: Dragonfly topology detected (N=%d)\n", N);
                return dragonfly_compute_trees(topo_file, root, N,
                                               NULL, NULL, 0, NULL, 0, 0,
                                               parent_arrays, out_tau);
            }
        }
    }

    topo_data_t td;
    if (topo_data_load(topo_file, &td) != 0) return -1;

    /* 1-hop threshold: find the natural gap between direct links and
     * multi-hop shortest paths.  Collect all unique nonzero latencies,
     * sort them, and pick the threshold at the first large gap (>25%).
     * Falls back to 1.01 × min_latency for uniform-link topologies. */
    float lat_thresh;
    {
        int ncand = 0;
        float *cand = (float *)malloc((size_t)N * N * sizeof(float));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float l = td.lat[(size_t)i * N + j];
                if (l > 0.0f) cand[ncand++] = l;
            }
        /* Simple insertion sort on unique values (N*N is small) */
        for (int i = 1; i < ncand; i++) {
            float key = cand[i];
            int j = i - 1;
            while (j >= 0 && cand[j] > key) { cand[j+1] = cand[j]; j--; }
            cand[j+1] = key;
        }
        /* Find first gap > 25% */
        lat_thresh = cand[0] * 1.01f;   /* default: uniform links */
        for (int i = 1; i < ncand; i++) {
            if (cand[i] > cand[i-1] * 1.25f) {
                lat_thresh = (cand[i-1] + cand[i]) * 0.5f;  /* midpoint */
                break;
            }
        }
        free(cand);
    }

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
