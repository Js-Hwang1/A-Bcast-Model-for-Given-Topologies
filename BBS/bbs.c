/*
 * bbs.c — Balanced Broadcast Spanning-trees (BBS) runtime.
 *
 * Included from runner.c — shares topo_data_t, topo_data_load/free,
 * and all MPI headers.
 *
 * This file contains ONLY the .bbs loader and broadcast dispatch.
 * Tree computation and .bbs encoding live in encode.c (included below).
 */

#define BBS_MAGIC       0xBB50
#define BBS_MAX_TREES   8
#define BBS_MAX_FANOUT  64

/* ---- Per-node routing table (lives on stack, fits in L1) ---- */
typedef struct {
    uint8_t  tau;
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

    rt->tau = tau;

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
    /* ---- Construct .bbs path ---- */
    char bbs_path[512];
    {
        const char *dot = strrchr(topo_file, '.');
        int plen = dot ? (int)(dot - topo_file) : (int)strlen(topo_file);
        snprintf(bbs_path, sizeof(bbs_path), "%.*s_R%d.bbs", plen, topo_file, root);
    }

    /* ---- Load routing ---- */
    bbs_route_t rt;
    if (bbs_load_route(bbs_path, rank, &rt) != 0) {
        /* .bbs missing — rank 0 generates it, all ranks build
         * route from a shared parent array (no file I/O after).
         * The generation uses only rank-0 compute + one MPI_Bcast
         * to distribute.  The timing barrier afterwards ensures
         * none of this leaks into the broadcast measurement. */
        int16_t *trees = NULL;
        int tau = 1;
        if (rank == 0) {
            trees = (int16_t *)malloc(BBS_MAX_TREES * size * sizeof(int16_t));
            bbs_compute_trees(topo_file, root, size, trees, &tau);
            bbs_write(bbs_path, size, tau, root, trees);
            fprintf(stderr, "BBS: generated %s (tau=%d)\n", bbs_path, tau);
        }
        /* Distribute tree structure */
        MPI_Bcast(&tau, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!trees) trees = (int16_t *)malloc(tau * size * sizeof(int16_t));
        MPI_Bcast(trees, tau * size, MPI_SHORT, 0, MPI_COMM_WORLD);
        /* Build route struct directly from tree arrays */
        rt.tau = (uint8_t)tau;
        for (int t = 0; t < tau; t++) {
            rt.parent[t] = trees[t * size + rank];
            rt.nchildren[t] = 0;
            for (int i = 0; i < size; i++)
                if (trees[t * size + i] == (int16_t)rank)
                    rt.children[t][rt.nchildren[t]++] = (int16_t)i;
        }
        free(trees);
    }

    /* ---- Setup ---- */
    int tau = rt.tau;
    int k   = nchunks > 0 ? nchunks : 1;
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

    /* ---- Timed section ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Pipeline: blocking Recv per chunk, fire-and-forget Isend to children */
    int si = 0;
    for (int c = 0; c < k; c++) {
        int t = c % tau;

        if (!is_root)
            MPI_Recv((char *)buf + coff[c], csz[c], MPI_BYTE,
                     rt.parent[t], c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int j = 0; j < rt.nchildren[t]; j++)
            MPI_Isend((char *)buf + coff[c], csz[c], MPI_BYTE,
                      rt.children[t][j], c, MPI_COMM_WORLD, &rsend[si++]);
    }

    if (si > 0)
        MPI_Waitall(si, rsend, MPI_STATUSES_IGNORE);

    double t1 = MPI_Wtime();

    /* Cleanup */
    free(coff); free(csz);
    free(rsend);

    return t1 - t0;
}
