/*
 * runner.c — Unified, topology-agnostic broadcast runner via SMPI.
 *
 * Algorithms:
 *   "mpi"  — Native MPI_Bcast (binary tree baseline).
 *   "srda" — Manual Scatter + Recursive-Doubling Allgather.
 *             Phase 1: MPI_Scatter distributes N equal pieces.
 *             Phase 2: log2(N) rounds of MPI_Sendrecv, each rank
 *                      exchanges with rank XOR 2^k.
 *             Requires power-of-2 N.
 *             Ref: Thakur et al., "Optimization of Collective Communication
 *                  Operations in MPICH", 2005.
 *   "pipe" — Pipelined chain broadcast (Open MPI style).
 *             Linear chain: root → root+1 → ... → root+N-1 (mod N).
 *             Fan-out 1: each node sends to exactly one successor.
 *             Pipelines nchunks chunks along the chain.
 *             Time ≈ (nchunks + N - 2) × chunk_time.
 *             Ref: Open MPI coll_tuned_bcast_intra_chain;
 *                  Thakur/Rabenseifner/Gropp, IJHPCA 2005.
 *   "bbs"  — (stub) Planned frame-based broadcast; not yet implemented.
 *
 * Usage:
 *   smpirun -np N -platform <xml> -hostfile <hf> \
 *           ./runner <algo> <bytes> [chunks] [root] [out_json]
 *
 * Output:
 *   algorithm : <algo>
 *   nodes     : <N>
 *   msg_bytes : <bytes>
 *   nchunks   : <chunks>
 *   root      : <root>
 *   time_sec  : <seconds>
 *   correct   : yes|NO
 *
 * Build:  smpicc -O2 -o runner runner.c -lm
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================================================================
 * Algorithm 1: Native MPI_Bcast
 * ================================================================ */
static double run_mpi_bcast(void *buf, int count, int root)
{
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    MPI_Bcast(buf, count, MPI_BYTE, root, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    return t1 - t0;
}

/* ================================================================
 * Algorithm 2: SRDA — Scatter + Recursive-Doubling Allgather
 *
 * Phase 1: Root scatters N equal-sized pieces (MPI_Scatter).
 * Phase 2: log2(N) rounds of recursive doubling.
 *          Round k: rank exchanges data with rank XOR 2^k.
 *          After each round, the amount of data doubles.
 *
 * Requires: N is a power of 2.
 *
 * Ref: Thakur, Rabenseifner, Gropp, "Optimization of Collective
 *      Communication Operations in MPICH", IJHPCA 2005.
 * ================================================================ */
static double run_srda(void *buf, int count, int rank, int size, int root)
{
    int log2n = 0;
    {
        int tmp = size;
        while (tmp > 1) { tmp >>= 1; log2n++; }
    }

    /* piece_size: each rank's initial scatter piece */
    int piece_size = count / size;
    int remainder  = count % size;

    /* Scatter buffer: root has full data, others receive their piece */
    char *scatter_recv = malloc(piece_size + 1);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* --- Phase 1: Scatter --- */
    MPI_Scatter(buf, piece_size, MPI_BYTE,
                scatter_recv, piece_size, MPI_BYTE,
                root, MPI_COMM_WORLD);

    /* Copy received piece into correct position in buf */
    memcpy((char *)buf + rank * piece_size, scatter_recv, piece_size);

    /* --- Phase 2: Recursive Doubling Allgather --- */
    /* After scatter, rank r holds piece at offset r*piece_size.
     * Round k (k=0..log2n-1):
     *   partner = rank XOR 2^k
     *   Send what I have, receive what partner has.
     *   After round k, each rank holds 2^(k+1) contiguous pieces. */

    for (int k = 0; k < log2n; k++) {
        int partner = rank ^ (1 << k);

        /* Determine what I currently hold.
         * After round k-1, I hold a block of 2^k pieces starting at
         * a position aligned to 2^k boundaries. */
        int block_size_pieces = (1 << k);
        int my_block_start = (rank >> k) << k;  /* align rank down to 2^k */
        int send_offset = my_block_start * piece_size;
        int send_len    = block_size_pieces * piece_size;

        /* Partner holds an adjacent block of the same size */
        int partner_block_start = (partner >> k) << k;
        int recv_offset = partner_block_start * piece_size;
        int recv_len    = block_size_pieces * piece_size;

        MPI_Sendrecv((char *)buf + send_offset, send_len, MPI_BYTE,
                     partner, k,
                     (char *)buf + recv_offset, recv_len, MPI_BYTE,
                     partner, k,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Handle remainder bytes if count not perfectly divisible */
    if (remainder > 0 && root == 0) {
        /* The last `remainder` bytes weren't scattered.
         * Root broadcasts them with a small bcast. */
        if (rank == root) {
            /* Already in buf at offset size*piece_size */
        }
        MPI_Bcast((char *)buf + size * piece_size, remainder, MPI_BYTE,
                  root, MPI_COMM_WORLD);
    }

    double t1 = MPI_Wtime();

    free(scatter_recv);
    return t1 - t0;
}

/* ================================================================
 * Algorithm 3: Pipelined Chain Broadcast
 *
 * Linear chain: root → root+1 → root+2 → ... → root+N-1  (mod N)
 * Each node has fan-out 1 (exactly one child, except the tail).
 * Chunks are pipelined along the chain.
 *
 * This matches Open MPI's coll_tuned_bcast_intra_chain (chain=1)
 * used for large messages.  Fan-out 1 maximizes steady-state
 * throughput: root sends 1 chunk per chunk_time, vs k×chunk_time
 * for fan-out k (since all children share the outgoing link).
 *
 * Total time ≈ (nchunks + N - 2) × chunk_time.
 *
 * Ref: Open MPI ompi/mca/coll/tuned/coll_tuned_bcast_decision.c
 *      Thakur, Rabenseifner, Gropp, IJHPCA 2005, Section 3.
 * ================================================================ */
static double run_pipe(void *buf, int count, int rank, int size,
                       int root, int nchunks)
{
    /* Position in the chain: root is position 0, root+1 is 1, etc. */
    int pos = (rank - root + size) % size;

    int prev_rank = (pos == 0)        ? -1 : ((pos - 1 + root) % size);
    int next_rank = (pos == size - 1) ? -1 : ((pos + 1 + root) % size);

    int chunk_size = (count + nchunks - 1) / nchunks;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (pos == 0) {
        /* Chain head (root): send each chunk to next, Waitall per chunk
         * to prevent flooding the link with concurrent flows. */
        MPI_Request req;
        for (int c = 0; c < nchunks; c++) {
            int off = c * chunk_size;
            int len = chunk_size;
            if (off + len > count) len = count - off;

            if (next_rank >= 0) {
                MPI_Isend((char *)buf + off, len, MPI_BYTE,
                          next_rank, c, MPI_COMM_WORLD, &req);
                MPI_Wait(&req, MPI_STATUS_IGNORE);
            }
        }
    } else {
        /* Interior / tail: blocking Recv from predecessor paces sends.
         * Accumulate Isend requests, Waitall at end. */
        MPI_Request *send_reqs = NULL;
        int nsend = 0;
        if (next_rank >= 0)
            send_reqs = malloc((size_t)nchunks * sizeof(MPI_Request));

        for (int c = 0; c < nchunks; c++) {
            int off = c * chunk_size;
            int len = chunk_size;
            if (off + len > count) len = count - off;

            MPI_Recv((char *)buf + off, len, MPI_BYTE,
                     prev_rank, c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (next_rank >= 0)
                MPI_Isend((char *)buf + off, len, MPI_BYTE,
                          next_rank, c, MPI_COMM_WORLD,
                          &send_reqs[nsend++]);
        }

        if (nsend > 0)
            MPI_Waitall(nsend, send_reqs, MPI_STATUSES_IGNORE);
        free(send_reqs);
    }

    double t1 = MPI_Wtime();
    return t1 - t0;
}

/* ================================================================
 * Algorithm 4: Topology-aware BFS-tree broadcast
 *
 * Reads a topology config file, builds a BFS spanning tree rooted
 * at `root` that respects the physical adjacency, then broadcasts
 * down that tree.  Each node receives from its parent and forwards
 * to all children sequentially.
 *
 * Topology types and their adjacency definitions:
 *
 *   mesh <rows> <cols>
 *       Grid neighbors (up/down/left/right).  Max degree 4.
 *
 *   butterfly <dim>
 *       Hypercube neighbors (XOR one bit).  Degree = dim.
 *
 *   fattree <hpl>
 *       Layered: intra-leaf clique (same leaf switch) first,
 *       then cross-leaf edges (round-robin via spine).
 *       BFS fills local leaf, then bridges to remote leaves.
 *
 *   dragonfly <G> <C> <R> <P>
 *       Layered: same-router clique first, then same-chassis
 *       (green links), same-group (black), cross-group (blue).
 * ================================================================ */

/* Topology types */
#define TOPO_MESH      1
#define TOPO_BUTTERFLY 2
#define TOPO_FATTREE   3
#define TOPO_DRAGONFLY 4

typedef struct {
    int type;
    union {
        struct { int rows, cols; } mesh;
        struct { int dim; } butterfly;
        struct { int hpl; } fattree;
        struct { int G, C, R, P; } dragonfly;
    };
} topo_cfg_t;

static int parse_topo_cfg(const char *path, topo_cfg_t *cfg)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;

    char type[32];
    if (fscanf(fp, "%31s", type) != 1) { fclose(fp); return -1; }

    if (strcmp(type, "mesh") == 0) {
        cfg->type = TOPO_MESH;
        if (fscanf(fp, "%d %d", &cfg->mesh.rows, &cfg->mesh.cols) != 2)
            { fclose(fp); return -1; }
    } else if (strcmp(type, "butterfly") == 0) {
        cfg->type = TOPO_BUTTERFLY;
        if (fscanf(fp, "%d", &cfg->butterfly.dim) != 1)
            { fclose(fp); return -1; }
    } else if (strcmp(type, "fattree") == 0) {
        cfg->type = TOPO_FATTREE;
        if (fscanf(fp, "%d", &cfg->fattree.hpl) != 1)
            { fclose(fp); return -1; }
    } else if (strcmp(type, "dragonfly") == 0) {
        cfg->type = TOPO_DRAGONFLY;
        if (fscanf(fp, "%d %d %d %d",
                   &cfg->dragonfly.G, &cfg->dragonfly.C,
                   &cfg->dragonfly.R, &cfg->dragonfly.P) != 4)
            { fclose(fp); return -1; }
    } else {
        fclose(fp); return -1;
    }
    fclose(fp);
    return 0;
}

/* ---- Neighbor enumeration per topology ---- */

/* ---- Neighbor enumeration (sorted by topology proximity) ----
 *
 * Returns all neighbors of node u, closest first.
 * For mesh/butterfly: direct graph neighbors (sparse, ≤ 10).
 * For fattree/dragonfly: all other nodes, grouped by hierarchy
 *   level (same-leaf before cross-leaf, same-router before
 *   same-chassis before same-group before cross-group).
 *
 * nbrs[] must hold at least (size-1) entries.
 */
static int get_neighbors_sorted(const topo_cfg_t *cfg, int u, int size,
                                int *nbrs)
{
    int n = 0;
    switch (cfg->type) {
    case TOPO_MESH: {
        int r = u / cfg->mesh.cols, c = u % cfg->mesh.cols;
        if (r > 0)                   nbrs[n++] = (r-1)*cfg->mesh.cols + c;
        if (r < cfg->mesh.rows - 1)  nbrs[n++] = (r+1)*cfg->mesh.cols + c;
        if (c > 0)                   nbrs[n++] = r*cfg->mesh.cols + (c-1);
        if (c < cfg->mesh.cols - 1)  nbrs[n++] = r*cfg->mesh.cols + (c+1);
        break;
    }
    case TOPO_BUTTERFLY:
        for (int d = 0; d < cfg->butterfly.dim; d++)
            nbrs[n++] = u ^ (1 << d);
        break;
    case TOPO_FATTREE: {
        int hpl = cfg->fattree.hpl;
        int my_leaf = u / hpl;
        int base = my_leaf * hpl;
        /* Priority 1: same leaf */
        for (int i = base; i < base + hpl; i++)
            if (i != u) nbrs[n++] = i;
        /* Priority 2: cross-leaf */
        for (int i = 0; i < size; i++)
            if (i / hpl != my_leaf) nbrs[n++] = i;
        break;
    }
    case TOPO_DRAGONFLY: {
        int P = cfg->dragonfly.P, R = cfg->dragonfly.R;
        int C = cfg->dragonfly.C, G = cfg->dragonfly.G;
        int my_rtr = u / P;
        int my_ch  = my_rtr / R;
        int my_grp = my_ch / C;
        /* Priority 1: same router */
        for (int i = my_rtr*P; i < (my_rtr+1)*P; i++)
            if (i != u) nbrs[n++] = i;
        /* Priority 2: same chassis, different router */
        for (int r = my_ch*R; r < (my_ch+1)*R; r++) {
            if (r == my_rtr) continue;
            for (int p = 0; p < P; p++) nbrs[n++] = r*P + p;
        }
        /* Priority 3: same group, different chassis */
        for (int c = my_grp*C; c < (my_grp+1)*C; c++) {
            if (c == my_ch) continue;
            for (int r = c*R; r < (c+1)*R; r++)
                for (int p = 0; p < P; p++) nbrs[n++] = r*P + p;
        }
        /* Priority 4: different group */
        for (int g = 0; g < G; g++) {
            if (g == my_grp) continue;
            for (int c = g*C; c < (g+1)*C; c++)
                for (int r = c*R; r < (c+1)*R; r++)
                    for (int p = 0; p < P; p++) nbrs[n++] = r*P + p;
        }
        break;
    }
    }
    return n;
}

/* ---- Degree-constrained BFS spanning tree ----
 *
 * Unified builder for all topologies.  Each dequeued node claims
 * at most `max_fanout` unvisited neighbors (closest first via
 * get_neighbors_sorted).  Excess neighbors are left for later
 * nodes to claim, producing a deeper but narrower tree.
 *
 * max_fanout = 0  →  unlimited (natural BFS tree).
 */
static void build_bfs_tree(const topo_cfg_t *cfg, int root, int size,
                           int *parent, int max_fanout)
{
    char *visited = calloc(size, 1);
    int *queue = malloc(size * sizeof(int));
    int qh = 0, qt = 0;
    int *nbrs = malloc(size * sizeof(int));

    int limit = (max_fanout > 0) ? max_fanout : size;

    for (int i = 0; i < size; i++) parent[i] = -1;
    visited[root] = 1;
    queue[qt++] = root;

    while (qh < qt) {
        int u = queue[qh++];
        int nn = get_neighbors_sorted(cfg, u, size, nbrs);
        int added = 0;
        for (int i = 0; i < nn && added < limit; i++) {
            int v = nbrs[i];
            if (!visited[v]) {
                visited[v] = 1;
                parent[v] = u;
                queue[qt++] = v;
                added++;
            }
        }
    }

    free(nbrs); free(visited); free(queue);
}

/* ---- Physical link parameters per topology ---- */

/* Effective one-way latency (seconds) for a tree edge u→v,
 * accounting for the number of physical hops through switches. */
static double edge_latency(const topo_cfg_t *cfg, int u, int v)
{
    switch (cfg->type) {
    case TOPO_MESH:      return 100e-9;   /* 1 hop */
    case TOPO_BUTTERFLY: return 100e-9;   /* 1 hop */
    case TOPO_FATTREE: {
        int hpl = cfg->fattree.hpl;
        /* intra-leaf: node→leaf→node = 2 hops × 100 ns */
        if (u / hpl == v / hpl) return 200e-9;
        /* cross-leaf: node→leaf→spine→leaf→node = 4 hops × 100 ns */
        return 400e-9;
    }
    case TOPO_DRAGONFLY: {
        int P = cfg->dragonfly.P, R = cfg->dragonfly.R;
        int C = cfg->dragonfly.C;
        int ru = u / P, rv = v / P;
        if (ru == rv) return 200e-9;              /* same router: 2×100ns */
        int cu = ru / R, cv = rv / R;
        if (cu == cv) return 300e-9;              /* same chassis (green) */
        int gu = cu / C, gv = cv / C;
        if (gu == gv) return 600e-9;              /* same group  (black) */
        return 1000e-9;                            /* cross-group (blue)  */
    }
    }
    return 100e-9;
}

/* Base link latency (seconds) — used as the unit for D_eff. */
static double base_latency(const topo_cfg_t *cfg)
{
    (void)cfg;
    return 100e-9;   /* all topologies use 100 ns as their finest link */
}

/* Link bandwidth (bytes/sec). */
static double link_bandwidth(const topo_cfg_t *cfg)
{
    switch (cfg->type) {
    case TOPO_MESH:      return 50e9;    /* InfiniBand NDR 400 */
    case TOPO_BUTTERFLY: return 12.5e9;  /* InfiniBand EDR     */
    case TOPO_FATTREE:   return 12.5e9;  /* InfiniBand EDR     */
    case TOPO_DRAGONFLY: return 5.25e9;  /* Cray Aries         */
    }
    return 12.5e9;
}

/* ---- Optimal chunk count from tree + topology physics ----
 *
 * Model (pipelined tree broadcast):
 *
 *   T(K) = (K · f_max  +  D_eff − 1) · (L + M/(KB))
 *
 *   K = number of chunks,  M = message size,  B = link bandwidth,
 *   L = base link latency,  f_max = max fan-out,
 *   D_eff = weighted tree depth  (max cumulative edge-latency
 *           from root to any leaf, measured in units of L).
 *
 * Minimising dT/dK = 0  gives:
 *
 *   K* = sqrt( (D_eff − 1) · M  /  (f_max · L · B) )
 *
 * High fan-out → fewer chunks (bottleneck is degree, not depth).
 * Deep tree    → more chunks  (pipeline amortises depth latency).
 */
static int optimal_chunks(const topo_cfg_t *cfg, int size,
                          const int *parent, int count)
{
    double L  = base_latency(cfg);
    double B  = link_bandwidth(cfg);

    /* --- Compute f_max and D_eff from the tree --- */
    int *fanout = calloc(size, sizeof(int));
    double *cum_lat = calloc(size, sizeof(double));

    /* Build children lists so we can traverse in BFS order
     * (needed when root != 0, since parent[v] > v is possible). */
    int *child_of = malloc(size * sizeof(int));  /* flat child array */
    int *cstart   = calloc(size + 1, sizeof(int));

    for (int v = 0; v < size; v++)
        if (parent[v] >= 0) fanout[parent[v]]++;
    for (int v = 0; v < size; v++) cstart[v + 1] = cstart[v] + fanout[v];
    int *pos = calloc(size, sizeof(int));
    for (int v = 0; v < size; v++) {
        if (parent[v] < 0) continue;
        int p = parent[v];
        child_of[cstart[p] + pos[p]++] = v;
    }

    int f_max = 0;
    for (int v = 0; v < size; v++)
        if (fanout[v] > f_max) f_max = fanout[v];

    /* BFS from root to accumulate cum_lat */
    int *queue = malloc(size * sizeof(int));
    int qh = 0, qt = 0;
    int root_node = -1;
    for (int v = 0; v < size; v++)
        if (parent[v] < 0) { root_node = v; break; }
    queue[qt++] = root_node;
    double max_cum_lat = 0.0;

    while (qh < qt) {
        int u = queue[qh++];
        for (int i = cstart[u]; i < cstart[u] + fanout[u]; i++) {
            int v = child_of[i];
            cum_lat[v] = cum_lat[u] + edge_latency(cfg, u, v);
            if (cum_lat[v] > max_cum_lat) max_cum_lat = cum_lat[v];
            queue[qt++] = v;
        }
    }

    free(queue); free(pos); free(child_of); free(cstart);
    free(fanout); free(cum_lat);

    double D_eff = max_cum_lat / L;    /* weighted depth in L-units */
    if (D_eff < 1.0) D_eff = 1.0;
    if (f_max < 1)   f_max = 1;

    /* K* = sqrt( (D_eff - 1) * M / (f_max * L * B) ) */
    double num = (D_eff - 1.0) * (double)count;
    double den = (double)f_max * L * B;
    int K = (int)ceil(sqrt(num / den));

    /* Clamp: at least 1 chunk, chunk size at least 64 bytes */
    if (K < 1) K = 1;
    int max_K = count / 64;
    if (max_K < 1) max_K = 1;
    if (K > max_K) K = max_K;

    return K;
}

static double run_test(void *buf, int count, int rank, int size,
                       int root, const char *topo_file,
                       int max_fanout, int *out_nchunks)
{
    topo_cfg_t cfg;
    if (parse_topo_cfg(topo_file, &cfg) != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: cannot parse topo config: %s\n",
                    topo_file);
        return -1.0;
    }

    /* All ranks compute the same BFS tree independently */
    int *parent = malloc(size * sizeof(int));
    build_bfs_tree(&cfg, root, size, parent, max_fanout);

    /* Determine my children */
    int nchildren = 0;
    int *children = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
        if (parent[i] == rank) children[nchildren++] = i;

    /* Compute optimal pipeline chunk count from tree structure */
    int nchunks = optimal_chunks(&cfg, size, parent, count);
    if (out_nchunks) *out_nchunks = nchunks;
    int chunk_size = (count + nchunks - 1) / nchunks;

    /* Allocate Isend requests: nchunks × nchildren.
     * All sends are non-blocking; the blocking Recv from our parent
     * naturally paces the pipeline.  Waitall only at the very end.
     * This lets SimGrid overlap sends on independent physical links
     * (e.g. different hypercube dimensions) instead of serialising. */
    int total_sends = nchunks * nchildren;
    MPI_Request *send_reqs = NULL;
    if (total_sends > 0)
        send_reqs = malloc((size_t)total_sends * sizeof(MPI_Request));
    int nsend = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int c = 0; c < nchunks; c++) {
        int off = c * chunk_size;
        int len = chunk_size;
        if (off + len > count) len = count - off;

        /* Receive this chunk from parent (root already has it) */
        if (rank != root) {
            MPI_Recv((char *)buf + off, len, MPI_BYTE, parent[rank],
                     c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /* Fire off sends to all children — non-blocking, no wait.
         * Each chunk uses a distinct buffer region so there is no
         * overlap hazard.  SimGrid will share bandwidth on links
         * that carry multiple concurrent flows (same leaf switch)
         * but allow full parallelism on independent links. */
        for (int ch = 0; ch < nchildren; ch++) {
            MPI_Isend((char *)buf + off, len, MPI_BYTE, children[ch],
                      c, MPI_COMM_WORLD, &send_reqs[nsend++]);
        }
    }

    /* Wait for every outstanding send to complete */
    if (nsend > 0)
        MPI_Waitall(nsend, send_reqs, MPI_STATUSES_IGNORE);

    double t1 = MPI_Wtime();

    free(send_reqs);
    free(parent);
    free(children);
    return t1 - t0;
}

/* ================================================================
 * Algorithm 5: BBS — (stub, not yet implemented)
 * ================================================================ */

/* ================================================================
 * Main
 * ================================================================ */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr,
                "Usage: %s <algorithm> <msg_bytes> [nchunks] [root] [out_json] [topo_cfg] [max_fanout]\n"
                "  algorithm  : mpi | srda | pipe | test\n"
                "  nchunks    : pipeline depth (default 64, auto for test)\n"
                "  root       : broadcast root node 0..%d (default 0)\n"
                "  out_json   : output JSON file path (optional, _ = none)\n"
                "  topo_cfg   : topology config file (required for 'test')\n"
                "  max_fanout : max children per node in BFS tree (0 = unlimited)\n",
                argv[0], size - 1);
        MPI_Finalize();
        return 1;
    }

    const char *algo = argv[1];
    int nbytes       = atoi(argv[2]);
    int nchunks      = (argc > 3) ? atoi(argv[3]) : 64;
    int root         = (argc > 4) ? atoi(argv[4]) : 0;
    const char *out_json  = (argc > 5) ? argv[5] : NULL;
    const char *topo_file = (argc > 6) ? argv[6] : NULL;
    int max_fanout   = (argc > 7) ? atoi(argv[7]) : 0;  /* 0 = unlimited */

    /* Treat "_" as no-file placeholder */
    if (out_json  && strcmp(out_json,  "_") == 0) out_json  = NULL;
    if (topo_file && strcmp(topo_file, "_") == 0) topo_file = NULL;

    if (nbytes <= 0) {
        if (rank == 0) fprintf(stderr, "msg_bytes must be > 0\n");
        MPI_Finalize();
        return 1;
    }
    if (root < 0 || root >= size) {
        if (rank == 0)
            fprintf(stderr, "root must be in 0..%d (got %d)\n",
                    size - 1, root);
        MPI_Finalize();
        return 1;
    }

    /* SRDA requires power-of-2 ranks */
    if (strcmp(algo, "srda") == 0) {
        if ((size & (size - 1)) != 0) {
            if (rank == 0)
                fprintf(stderr, "SRDA requires power-of-2 ranks (got %d)\n",
                        size);
            MPI_Finalize();
            return 1;
        }
    }

    /* Allocate buffer; root fills with known pattern */
    char *buf = calloc(nbytes, 1);
    if (rank == root)
        for (int i = 0; i < nbytes; i++)
            buf[i] = (char)(i & 0xFF);

    /* ---- Dispatch ---- */
    double elapsed = 0.0;

    if (strcmp(algo, "mpi") == 0) {
        elapsed = run_mpi_bcast(buf, nbytes, root);

    } else if (strcmp(algo, "srda") == 0) {
        elapsed = run_srda(buf, nbytes, rank, size, root);

    } else if (strcmp(algo, "pipe") == 0) {
        elapsed = run_pipe(buf, nbytes, rank, size, root, nchunks);

    } else if (strcmp(algo, "test") == 0) {
        if (!topo_file) {
            if (rank == 0)
                fprintf(stderr, "test algorithm requires topo_cfg argument\n");
            free(buf);
            MPI_Finalize();
            return 1;
        }
        elapsed = run_test(buf, nbytes, rank, size, root, topo_file,
                          max_fanout, &nchunks);

    } else if (strcmp(algo, "bbs") == 0) {
        if (rank == 0)
            fprintf(stderr, "BBS algorithm is not yet implemented.\n");
        free(buf);
        MPI_Finalize();
        return 1;

    } else {
        if (rank == 0) fprintf(stderr, "Unknown algorithm: %s\n", algo);
        free(buf);
        MPI_Finalize();
        return 1;
    }

    /* ---- Verify ---- */
    int ok = 1;
    for (int i = 0; i < nbytes; i++)
        if (buf[i] != (char)(i & 0xFF)) { ok = 0; break; }

    /* ---- Report ---- */
    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX,
               root, MPI_COMM_WORLD);
    int all_ok;
    MPI_Reduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, root, MPI_COMM_WORLD);

    if (rank == root) {
        printf("algorithm : %s\n", algo);
        printf("nodes     : %d\n", size);
        printf("msg_bytes : %d\n", nbytes);
        printf("nchunks   : %d\n", nchunks);
        printf("root      : %d\n", root);
        printf("time_sec  : %.9f\n", max_time);
        printf("correct   : %s\n", all_ok ? "yes" : "NO");

        /* ---- JSON output ---- */
        if (out_json) {
            FILE *jfp = fopen(out_json, "w");
            if (jfp) {
                fprintf(jfp,
                    "{\n"
                    "  \"algorithm\": \"%s\",\n"
                    "  \"nodes\": %d,\n"
                    "  \"msg_bytes\": %d,\n"
                    "  \"nchunks\": %d,\n"
                    "  \"root\": %d,\n"
                    "  \"time_sec\": %.9f,\n"
                    "  \"correct\": %s\n"
                    "}\n",
                    algo, size, nbytes, nchunks, root,
                    max_time,
                    all_ok ? "true" : "false");
                fclose(jfp);
            } else {
                fprintf(stderr, "Warning: could not open %s for writing\n",
                        out_json);
            }
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
