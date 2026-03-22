/*
 * bbs.c — Balanced Broadcast Spanning-trees (BBS) runtime.
 *
 * Included from runner.c — shares topo_data_t, topo_data_load/free,
 * and all MPI headers.
 *
 * This file contains ONLY the .bbs loader and broadcast dispatch.
 * Tree computation and .bbs encoding live in encode.c (included below).
 *
 * =====================================================================
 * PIPELINE DESIGN — WINDOWED IRECV (critical for performance)
 * =====================================================================
 *
 * BBS uses tau=3 spanning trees that share physical edges (max 2 trees
 * per edge).  The k message chunks are assigned round-robin to trees:
 * chunk c travels through tree t = c % tau.
 *
 * THE PROBLEM (old design):
 *   Pre-posting ALL k MPI_Irecv calls at time 0 allows all of the
 *   sender's MPI_Isend flows to match immediately in SimGrid's
 *   rendezvous protocol.  This creates k concurrent flows on every
 *   link, and SimGrid's LMM solver splits bandwidth among all of them.
 *   The pipeline degenerates into a level-by-level broadcast — BBS
 *   time barely changes with k, and BBS is ~2x SLOWER than OBFS.
 *
 *   Compare with OBFS, which uses blocking MPI_Recv per chunk.  The
 *   blocking receive gates the matching send: the parent's Isend can
 *   only start when the child posts its Recv.  Since Recvs are posted
 *   one-by-one, chunks enter the tree at successive T_hop intervals,
 *   creating a proper pipeline.
 *
 * THE FIX (current design):
 *   Post only tau Irecv calls at a time (one per tree), forming a
 *   sliding window.  When a chunk arrives (via Waitany), forward it
 *   and post the NEXT receive for that tree's slot.  This limits
 *   concurrent incoming flows to at most tau per node.
 *
 *   On a 2D mesh, each node's tau parents are typically on DIFFERENT
 *   physical links (degree >= 3), so the tau concurrent flows run at
 *   full bandwidth with zero contention.  The pipeline throughput
 *   becomes tau chunks per T_hop instead of 1.
 *
 *   The root also batches sends in groups of tau (Isend + Waitall per
 *   batch), so chunks enter the tree at the correct pipeline cadence.
 *
 * RESULT:
 *   T_BBS converges to (d + k/tau - 1) * T_hop(S/k), giving up to
 *   tau=3x throughput over a single-tree pipeline.  On shared edges
 *   (max_usage=2), the forwarding step still sees 50% bandwidth, so
 *   the net efficiency is 3 * 50% = 1.5x over OBFS in practice.
 *   Empirically confirmed: BBS/OBFS ratio -> 0.67 (i.e. 1.5x faster).
 * =====================================================================
 */

#define BBS_MAGIC       0xBB50
#define BBS_MAX_TREES   8
#define BBS_MAX_FANOUT  64

/* ---- Per-node routing table (lives on stack, fits in L1) ---- */
typedef struct {
    uint8_t  tau;
    uint8_t  depth;   /* max tree depth (from .bbs header, for auto-chunking) */
    int16_t  parent[BBS_MAX_TREES];
    uint8_t  nchildren[BBS_MAX_TREES];
    int16_t  children[BBS_MAX_TREES][BBS_MAX_FANOUT];
} bbs_route_t;

/* ---- Read only this node's routing from .bbs ---- */
static int bbs_load_route(const char *path, int me, bbs_route_t *rt)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint16_t magic, N, root;
    uint8_t  tau, pad;
    fread(&magic, 2, 1, f);
    fread(&N,     2, 1, f);
    fread(&tau,   1, 1, f);
    fread(&pad,   1, 1, f);
    fread(&root,  2, 1, f);

    if (magic != BBS_MAGIC || tau > BBS_MAX_TREES || me >= (int)N) {
        fclose(f); return -1;
    }

    rt->tau   = tau;
    rt->depth = pad;  /* max tree depth stored in header */

    /* Jump to this node's offset entry, read it, jump to data */
    uint32_t off;
    fseek(f, 8 + (uint32_t)me * 4, SEEK_SET);
    fread(&off, 4, 1, f);
    fseek(f, (long)off, SEEK_SET);

    for (int t = 0; t < tau; t++) {
        int16_t  p;
        uint16_t nc;
        fread(&p,  2, 1, f);
        fread(&nc, 2, 1, f);
        rt->parent[t]    = p;
        rt->nchildren[t] = (uint8_t)nc;
        if (nc > BBS_MAX_FANOUT) { fclose(f); return -1; }
        if (nc > 0) fread(rt->children[t], 2, nc, f);
    }

    fclose(f);
    return 0;
}

/* ---- Encoding / tree generation (only needed when .bbs is missing) ---- */
#include "encode.c"

/* ================================================================
 * run_bbs — the broadcast
 *
 * If .bbs exists: load routing table, broadcast.
 * If .bbs missing: rank 0 computes trees + writes .bbs,
 *   Bcast parent array to all ranks, broadcast.
 * ================================================================ */
static double run_bbs(void *buf, int count, int rank, int size,
                      int root, const char *topo_file, int nchunks)
{
    /* ---- Construct .bbs path in <project_root>/encodings/<topo>/N=<size>/ ----
     * Derive project root from topo_file which is always:
     *   <project_root>/topo/<TOPO>/platform_*.tdat               */
    char bbs_path[512];
    {
        /* Extract topology name = parent directory of topo_file */
        const char *base = strrchr(topo_file, '/');
        const char *topo_name = "unknown";
        char topo_buf[64] = {0};
        if (base) {
            const char *dir_end = base;
            const char *dir_start = topo_file;
            for (const char *p = topo_file; p < dir_end; p++)
                if (*p == '/') dir_start = p + 1;
            int dlen = (int)(dir_end - dir_start);
            if (dlen > 0 && dlen < (int)sizeof(topo_buf)) {
                memcpy(topo_buf, dir_start, dlen);
                topo_buf[dlen] = '\0';
                topo_name = topo_buf;
            }
        }

        /* Find project root: go up from topo/<TOPO>/file to get root.
         * topo_file = "<root>/topo/<TOPO>/<file>" → we need <root>. */
        char proj_root[512] = ".";
        if (base) {
            /* Find the "topo/" component by searching backwards */
            const char *p = base;
            int slashes = 0;
            while (p > topo_file && slashes < 2) {
                p--;
                if (*p == '/') slashes++;
            }
            if (slashes == 2) {
                int rlen = (int)(p - topo_file);
                if (rlen > 0 && rlen < (int)sizeof(proj_root)) {
                    memcpy(proj_root, topo_file, rlen);
                    proj_root[rlen] = '\0';
                }
            }
        }

        /* Ensure directory exists (rank 0 only, harmless if exists) */
        if (rank == 0) {
            char dir[512];
            snprintf(dir, sizeof(dir), "%s/encodings", proj_root);
            mkdir(dir, 0755);
            snprintf(dir, sizeof(dir), "%s/encodings/%s", proj_root, topo_name);
            mkdir(dir, 0755);
            snprintf(dir, sizeof(dir), "%s/encodings/%s/N=%d", proj_root, topo_name, size);
            mkdir(dir, 0755);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        /* Dragonfly uses two tree variants:
         * "s" suffix = shallow (depth≈9, good for small msgs)
         * no suffix  = deep    (depth≈53, good for large msgs)
         * Threshold: 256KB.
         * Shallow tree only supports N=128 (hardcoded offsets). */
        int is_dfly = (strcmp(topo_name, "Dragonfly") == 0);
        int use_shallow = is_dfly && (count <= 262144);
        snprintf(bbs_path, sizeof(bbs_path),
                 "%s/encodings/%s/N=%d/R%d%s.bbs", proj_root, topo_name, size,
                 root, use_shallow ? "s" : "");
    }

    /* ---- Encode if .bbs missing (rank 0 probes and generates) ---- */
    {
        int need_gen = 0;
        if (rank == 0) {
            FILE *probe = fopen(bbs_path, "rb");
            if (probe) { fclose(probe); } else { need_gen = 1; }
        }
        MPI_Bcast(&need_gen, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (need_gen) {
            if (rank == 0) {
                int16_t *trees = (int16_t *)malloc(BBS_MAX_TREES * size * sizeof(int16_t));
                int tau = 1;
                dfly_use_shallow = (strstr(bbs_path, "s.bbs") != NULL &&
                                    strstr(bbs_path, "Dragonfly") != NULL);
                bbs_compute_trees(topo_file, root, size, trees, &tau);
                dfly_use_shallow = 0;
                bbs_write(bbs_path, size, tau, root, trees);
                fprintf(stderr, "BBS: generated %s (tau=%d)\n", bbs_path, tau);
                free(trees);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    /* ---- Load routing strictly from .bbs file ---- */
    bbs_route_t rt;
    if (bbs_load_route(bbs_path, rank, &rt) != 0) {
        fprintf(stderr, "BBS: FATAL: cannot load %s for rank %d\n", bbs_path, rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ---- Setup ---- */
    int tau = rt.tau;
    int k;
    if (nchunks > 0) {
        k = nchunks;
    } else {
        /* Auto-chunk: target 8KB chunks */
        k = count / 8192;
        if (k < 1) k = 1;
    }
    int base_chunk = count / k;
    int leftover   = count % k;
    int is_root    = (rank == root);

    /* Precompute chunk byte-offsets */
    int *coff = (int *)malloc((k + 1) * sizeof(int));
    int *csz  = (int *)malloc(k * sizeof(int));
    coff[0] = 0;
    for (int c = 0; c < k; c++) {
        csz[c]     = base_chunk + (c < leftover ? 1 : 0);
        coff[c + 1] = coff[c] + csz[c];
    }

    /* Allocate send requests */
    int total_sends = 0;
    for (int c = 0; c < k; c++)
        total_sends += rt.nchildren[c % tau];

    MPI_Request *rsend = NULL;
    if (total_sends > 0)
        rsend = (MPI_Request *)malloc(total_sends * sizeof(MPI_Request));

    /* Allocate receive requests */
    MPI_Request *rrecv = (MPI_Request *)malloc(k * sizeof(MPI_Request));

    /* ---- Timed section ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Windowed pipeline: post only tau Irecvs at a time (one per tree).
     * This gates the flow: at most tau concurrent incoming flows per node,
     * each from a different tree parent on a different physical link.
     * Without this gating, pre-posting all k Irecvs allows all flows
     * to start at once, creating massive contention on shared edges. */
    int si = 0;
    if (is_root) {
        /* Root: send tau chunks at a time (one batch per pipeline step).
         * Within a batch, sends to different trees use different outgoing
         * links and proceed in parallel. */
        for (int batch = 0; batch < k; batch += tau) {
            MPI_Request btmp[BBS_MAX_TREES * BBS_MAX_FANOUT];
            int nb = 0;
            for (int off = 0; off < tau && batch + off < k; off++) {
                int c = batch + off;
                int t = c % tau;
                for (int j = 0; j < rt.nchildren[t]; j++)
                    MPI_Isend((char *)buf + coff[c], csz[c], MPI_BYTE,
                              rt.children[t][j], c, MPI_COMM_WORLD,
                              &btmp[nb++]);
            }
            MPI_Waitall(nb, btmp, MPI_STATUSES_IGNORE);
        }
    } else {
        /* Non-root: windowed Irecv — post tau receives initially,
         * then replace each completed one with the next chunk's receive.
         * Waitany returns chunks in arrival order (any tree first).
         *
         * IMPORTANT: use per-tree next-chunk counters so that when
         * tree t's chunk completes, the next posted receive is also
         * for tree t.  A global sequential counter can create window
         * imbalance (2 active receives on one tree, 0 on the other),
         * which allows concurrent flows on the same physical link. */
        int completed = 0;
        int total_recv = 0;

        /* Count how many chunks this node actually receives */
        for (int c = 0; c < k; c++) {
            int t = c % tau;
            if (rt.parent[t] >= 0) total_recv++;
        }

        /* Initialize all request slots to NULL */
        for (int c = 0; c < k; c++)
            rrecv[c] = MPI_REQUEST_NULL;

        /* Per-tree next-chunk counters (start after initial window) */
        int next_tree[BBS_MAX_TREES];
        for (int t = 0; t < tau; t++)
            next_tree[t] = tau + t;   /* tree 0 → 2, tree 1 → 3, etc. */

        /* Post initial window: one receive per tree */
        for (int c = 0; c < tau && c < k; c++) {
            int t = c % tau;
            if (rt.parent[t] >= 0) {
                MPI_Irecv((char *)buf + coff[c], csz[c], MPI_BYTE,
                          MPI_ANY_SOURCE, c, MPI_COMM_WORLD, &rrecv[c]);
            }
        }

        /* Process chunks as they arrive */
        while (completed < total_recv) {
            int idx;
            MPI_Waitany(k, rrecv, &idx, MPI_STATUS_IGNORE);
            if (idx == MPI_UNDEFINED) break;
            completed++;

            /* Forward this chunk to children in its tree */
            int t = idx % tau;
            for (int j = 0; j < rt.nchildren[t]; j++)
                MPI_Isend((char *)buf + coff[idx], csz[idx], MPI_BYTE,
                          rt.children[t][j], idx, MPI_COMM_WORLD,
                          &rsend[si++]);

            /* Post next receive for the SAME TREE that just completed */
            if (next_tree[t] < k && rt.parent[t] >= 0) {
                int nc = next_tree[t];
                MPI_Irecv((char *)buf + coff[nc], csz[nc], MPI_BYTE,
                          MPI_ANY_SOURCE, nc, MPI_COMM_WORLD, &rrecv[nc]);
                next_tree[t] += tau;
            }
        }
    }

    /* Drain all sends */
    if (si > 0)
        MPI_Waitall(si, rsend, MPI_STATUSES_IGNORE);

    double t1 = MPI_Wtime();

    /* Cleanup */
    free(coff); free(csz);
    free(rsend); free(rrecv);

    return t1 - t0;
}
