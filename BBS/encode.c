/*
 * encode.c — BBS tree computation and .bbs file generation.
 *
 * Included from runner.c (via bbs.c) — shares topo_data_t,
 * topo_data_load/free, and all standard headers.
 *
 * Separated from bbs.c so that encoding logic is only invoked
 * when a .bbs file is missing.  The broadcast runtime (bbs.c)
 * never touches this code if the .bbs file exists.
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

/* ---- Compute physically-disjoint spanning trees via flink ---- */
/*                                                                  */
/* Key innovation: use the first-link (flink) matrix from .tdat     */
/* to build trees whose edges use DIFFERENT physical links.         */
/*                                                                  */
/* Algorithm:                                                       */
/*   1. Split root's neighbors round-robin across tau trees         */
/*   2. For each tree, two-pass BFS:                                */
/*      Pass 1: BFS using only edges with UNUSED flink IDs          */
/*      Pass 2: BFS using any remaining edges (shared flinks)       */
/*   3. After each tree, mark its edge flinks as used               */
/*   4. Fallback if trees are too deep                              */
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
    int root_nbrs[256], root_nn = 0;
    for (int v = 0; v < N; v++)
        if (adj[root * N + v])
            root_nbrs[root_nn++] = v;

    /* Compute baseline single-tree BFS depth */
    char *visited = (char *)calloc(N, 1);
    int  *queue   = (int *)malloc(N * sizeof(int));
    int16_t *t0_bfs = (int16_t *)malloc(N * sizeof(int16_t));
    for (int i = 0; i < N; i++) t0_bfs[i] = -1;
    memset(visited, 0, N);
    int qh = 0, qt = 0;
    visited[root] = 1;
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

    /* Determine tau from root degree (allow shared edges) */
    int nw_bound = total_edges / (N - 1);
    int tau = root_nn;
    if (tau > BBS_MAX_TREES) tau = BBS_MAX_TREES;
    if (tau < 1) tau = 1;

    /* Find max flink ID and count distinct root flinks */
    int max_fl = 0;
    for (size_t i = 0; i < NN; i++)
        if (td.flink[i] > max_fl) max_fl = td.flink[i];
    max_fl += 2; /* +2 for safety margin */

    int root_distinct_fl = 0;
    {
        uint16_t seen_fl[256];
        for (int i = 0; i < root_nn; i++) {
            uint16_t fl = td.flink[(size_t)root * N + root_nbrs[i]];
            int dup = 0;
            for (int j = 0; j < root_distinct_fl; j++)
                if (seen_fl[j] == fl) { dup = 1; break; }
            if (!dup) seen_fl[root_distinct_fl++] = fl;
        }
    }

    fprintf(stderr, "BBS: edges=%d root_deg=%d NW=%d root_flinks=%d "
            "max_fl=%d → trying tau=%d (baseline_depth=%d)\n",
            total_edges, root_nn, nw_bound, root_distinct_fl,
            max_fl, tau, depth0);

    if (tau < 2) {
        *out_tau = 1;
        memcpy(parent_arrays, t0_bfs, N * sizeof(int16_t));
        fprintf(stderr, "BBS: tau=1 (root degree < 2)\n");
        free(t0_bfs); free(visited); free(queue); free(adj);
        topo_data_free(&td);
        return 0;
    }

    /* Flink usage tracking: how many trees use each physical link */
    uint16_t *flink_used = (uint16_t *)calloc(max_fl, sizeof(uint16_t));

    /* Build tau trees with flink-aware two-pass BFS */
    for (int t = 0; t < tau; t++) {
        int16_t *tree = parent_arrays + (size_t)t * N;
        for (int i = 0; i < N; i++) tree[i] = -1;

        memset(visited, 0, N);
        visited[root] = 1;
        qh = 0; qt = 0;
        queue[qt++] = root;

        /* Seed: root neighbors at indices t, t+tau, t+2*tau, ... */
        for (int i = t; i < root_nn; i += tau) {
            int v = root_nbrs[i];
            if (!visited[v]) {
                visited[v] = 1;
                tree[v] = (int16_t)root;
                queue[qt++] = v;
            }
        }

        /* Two-pass BFS for full-path disjointness.
         * Pass 1: only edges whose FULL physical path has no used links.
         * Pass 2: any remaining edges (shared paths). */
        qh = 1;
        if (td.max_hops > 0 && td.path_len && td.path_links) {
            /* Pass 1: BFS using only path-disjoint edges */
            int mh = td.max_hops;
            while (qh < qt) {
                int u = queue[qh++];
                for (int v = 0; v < N; v++) {
                    if (visited[v] || !adj[u * N + v]) continue;
                    /* Check if ANY link on path(u,v) is already used */
                    int plen = td.path_len[(size_t)u * N + v];
                    int conflict = 0;
                    for (int h = 0; h < plen; h++) {
                        uint16_t lid = td.path_links[((size_t)u * N + v) * mh + h];
                        if (lid > 0 && lid < max_fl && flink_used[lid] > 0) {
                            conflict = 1; break;
                        }
                    }
                    if (conflict) continue;
                    visited[v] = 1;
                    tree[v] = (int16_t)u;
                    queue[qt++] = v;
                }
            }
            /* Pass 2: fill remaining via any adjacency edge */
            for (int qi = 0; qi < qt; qi++) {
                int u = queue[qi];
                for (int v = 0; v < N; v++) {
                    if (visited[v] || !adj[u * N + v]) continue;
                    visited[v] = 1;
                    tree[v] = (int16_t)u;
                    queue[qt++] = v;
                }
            }
        } else {
            /* Fallback: standard BFS (no path data) */
            while (qh < qt) {
                int u = queue[qh++];
                for (int v = 0; v < N; v++) {
                    if (visited[v] || !adj[u * N + v]) continue;
                    visited[v] = 1;
                    tree[v] = (int16_t)u;
                    queue[qt++] = v;
                }
            }
        }

        /* Bridge disconnected components via latency */
        while (qt < N) {
            int bv = -1, bu = -1;
            float best = 1e30f;
            for (int v = 0; v < N; v++) {
                if (visited[v]) continue;
                for (int u = 0; u < N; u++) {
                    if (!visited[u]) continue;
                    float l = td.lat[(size_t)u * N + v];
                    if (l > 0.0f && l < best) { best = l; bu = u; bv = v; }
                }
            }
            if (bv < 0) break;
            visited[bv] = 1;
            tree[bv] = (int16_t)bu;
            queue[qt++] = bv;
        }

        /* Mark ALL links on this tree's edge paths as used */
        int links_fresh = 0, links_shared = 0;
        for (int i = 0; i < N; i++) {
            if (tree[i] < 0) continue;
            int p = tree[i];
            if (td.max_hops > 0 && td.path_len && td.path_links) {
                int mh = td.max_hops;
                int plen = td.path_len[(size_t)p * N + i];
                for (int h = 0; h < plen; h++) {
                    uint16_t lid = td.path_links[((size_t)p * N + i) * mh + h];
                    if (lid > 0 && lid < max_fl) {
                        if (flink_used[lid] == 0) links_fresh++;
                        else links_shared++;
                        flink_used[lid]++;
                    }
                }
            } else {
                uint16_t fl = td.flink[(size_t)p * N + i];
                if (fl > 0 && fl < max_fl) {
                    if (flink_used[fl] == 0) links_fresh++;
                    else links_shared++;
                    flink_used[fl]++;
                }
            }
        }
        fprintf(stderr, "BBS:   T%d: %d fresh links, %d shared links\n",
                t, links_fresh, links_shared);
    }

    /* Compute depths and stats */
    int max_depth = 0;
    fprintf(stderr, "BBS: tree depths:");
    for (int t = 0; t < tau; t++) {
        int d = tree_depth(parent_arrays + (size_t)t * N, N);
        int rc = 0;
        for (int i = 0; i < N; i++)
            if (parent_arrays[(size_t)t * N + i] == (int16_t)root) rc++;
        fprintf(stderr, " T%d=%d(rc=%d)", t, d, rc);
        if (d > max_depth) max_depth = d;
    }
    fprintf(stderr, "\n");

    /* Count flink collisions between tree pairs */
    int flink_collisions = 0;
    for (int a = 0; a < tau; a++)
        for (int b = a + 1; b < tau; b++) {
            int16_t *ta = parent_arrays + (size_t)a * N;
            int16_t *tb = parent_arrays + (size_t)b * N;
            for (int i = 0; i < N; i++) {
                if (ta[i] < 0) continue;
                uint16_t fl_a = td.flink[(size_t)ta[i] * N + i];
                if (fl_a == 0) continue;
                for (int j = 0; j < N; j++) {
                    if (tb[j] < 0) continue;
                    uint16_t fl_b = td.flink[(size_t)tb[j] * N + j];
                    if (fl_a == fl_b) { flink_collisions++; break; }
                }
            }
        }

    fprintf(stderr, "BBS: tau=%d max_depth=%d baseline=%d flink_collisions=%d\n",
            tau, max_depth, depth0, flink_collisions);

    /* Fallback: if max depth > 2× baseline, reduce tau */
    while (tau > 1 && max_depth > 2 * depth0) {
        tau--;
        fprintf(stderr, "BBS: max_depth %d > 2*baseline %d, reducing to tau=%d\n",
                max_depth, depth0, tau);
        max_depth = 0;
        for (int t = 0; t < tau; t++) {
            int d = tree_depth(parent_arrays + (size_t)t * N, N);
            if (d > max_depth) max_depth = d;
        }
    }

    if (tau == 1) {
        memcpy(parent_arrays, t0_bfs, N * sizeof(int16_t));
        fprintf(stderr, "BBS: fallback to tau=1 (baseline BFS)\n");
    }

    *out_tau = tau;

    free(flink_used); free(t0_bfs);
    free(adj); free(visited); free(queue);
    topo_data_free(&td);
    return 0;
}
