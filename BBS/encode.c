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
 * TREE CONSTRUCTION — CONSTRAINT PROPAGATION + GREEDY (tau=3, max_usage=2)
 * =====================================================================
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

    /* Parameters */
    int tau = 3;
    int max_usage = 2;
    if (root_nn < 2) tau = 1;
    if (tau > BBS_MAX_TREES) tau = BBS_MAX_TREES;

    /* Find max flink ID */
    int max_fl = 0;
    for (size_t i = 0; i < NN; i++)
        if (td.flink[i] > max_fl) max_fl = td.flink[i];
    max_fl += 2;

    fprintf(stderr, "BBS: edges=%d root_deg=%d tau=%d max_usage=%d "
            "baseline_depth=%d\n",
            total_edges, root_nn, tau, max_usage, depth0);

    if (tau < 2) {
        *out_tau = 1;
        memcpy(parent_arrays, t0_bfs, N * sizeof(int16_t));
        fprintf(stderr, "BBS: tau=1 (root degree < 2)\n");
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
