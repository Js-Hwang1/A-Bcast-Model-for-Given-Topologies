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
 *   "bine" — Negabinary binomial tree (De Sensi et al., SC 2025).
 *             Negabinary XOR partner selection yields ~33% shorter
 *             parent-child distances.  O(log N) steps.
 *             Requires power-of-2 N.
 *   "glf"  — Global-Links-First (Dorier et al., COMHPC 2016).
 *             Hierarchical: multi-level binomial cascade.
 *             Flat: BFS-proximity binomial.
 *
 * Usage:
 *   smpirun -np N -platform <xml> -hostfile <hf> \
 *           ./runner <algo> <bytes> [chunks] [root] [out_json] [topo_cfg]
 *
 *   Sweep mode (smart root sweep with resume):
 *   smpirun -np N -platform <xml> -hostfile <hf> \
 *           ./runner <algo> <bytes> [chunks] sweep <outdir> [topo_cfg]
 *
 *   Sweep checks outdir/N{N}_MSG{M}.json — skips if done.
 *   Otherwise checks per-root outdir/N{N}_MSG{M}_R{r}.json,
 *   runs only missing roots, aggregates, cleans up.
 *   Safe to re-run after crash/OOM — picks up where it left off.
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
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

/* SMPI_SHARED_MALLOC: all N simulated processes share ONE physical
 * allocation, so memory is O(msg) not O(N*msg).
 * Requires --cfg=smpi/shared-malloc-blocksize:SIZE >= msg_bytes
 * to avoid excessive VMA mappings.
 * smpicc's mpi.h already defines this; fall back to malloc otherwise. */
#ifndef SMPI_SHARED_MALLOC
#  define SMPI_SHARED_MALLOC(sz)  malloc(sz)
#  define SMPI_SHARED_FREE(p)     free(p)
#endif

/* Single shared simulation buffer — allocated once, reused by all N
 * SimGrid coroutines (they share one OS address space).
 * Physical: O(msg), not O(N*msg). */
static char  *_sim_buf       = NULL;
static size_t _sim_buf_bytes = 0;

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
    if (remainder > 0) {
        /* The last `remainder` bytes weren't scattered.
         * Root broadcasts them with a small bcast. */
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
 * Algorithm: Bine — Negabinary Binomial Tree (SC 2025)
 *
 * Uses negabinary (base -2) XOR arithmetic for partner selection.
 * At each of the s = log2(N) steps, a rank XOR-flips the bottom
 * (k+1) negabinary bits to find its partner.  Ranks receive when
 * their LSBs are all-same (all 0s or all 1s).
 *
 * This yields ~33% shorter parent-child modular distances compared
 * to standard binomial trees (distance ratio 2/3, see Eq. 2 in
 * the paper).
 *
 * O(log N) steps, blocking Send/Recv per step, no pipelining.
 * Topology-oblivious (works from rank numbering alone).
 * Requires power-of-2 N.
 *
 * Ref: De Sensi et al., "Bine Trees: Enhancing Collective
 *      Operations by Optimizing Communication Locality", SC 2025.
 *      Implementation follows Algorithm 1 and libpico reference
 *      (github.com/HLC-Lab/pico).
 * ================================================================ */

/* Convert a binary integer to its negabinary (base -2) bit pattern.
 * Uses Schroeppel's identity (HAKMEM Item 128, 1972):
 *   negabinary(n) = (0xAAAAAAAA + n) ^ 0xAAAAAAAA               */
static unsigned int binary_to_negabinary(int n)
{
    const unsigned int mask = 0xAAAAAAAAu;
    return (mask + (unsigned int)n) ^ mask;
}

/* Convert a negabinary bit pattern back to a binary integer.
 * Inverse of binary_to_negabinary. */
static int negabinary_to_binary(unsigned int nb)
{
    const unsigned int mask = 0xAAAAAAAAu;
    return (int)((mask ^ nb) - mask);
}

static double run_bine(void *buf, int count, int rank, int size, int root)
{
    int log2n = 0;
    { int tmp = size; while (tmp > 1) { tmp >>= 1; log2n++; } }

    /* Virtual rank: shift so root becomes vrank 0 */
    int vrank = (rank - root + size) % size;
    unsigned int nb = binary_to_negabinary(vrank);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Negabinary binomial broadcast (Algorithm 1 from the paper).
     * s = log2(N) steps, mask from 2^(s-1) down to 1.
     * At each step:
     *   XOR the bottom (k+1) negabinary bits to find the partner.
     *   Receive if LSBs are all-same (all 0s or all 1s) and data
     *   not yet received.
     *   Send if data already received (at this or an earlier step). */
    int recvd = (rank == root);
    unsigned int step_mask = 1u << (log2n - 1);
    int step = 0;

    while (step_mask > 0) {
        unsigned int xor_mask = (step_mask << 1) - 1;

        /* Partner: flip the bottom bits in negabinary space */
        unsigned int partner_nb = nb ^ xor_mask;
        int partner_val = negabinary_to_binary(partner_nb);
        int partner_vrank = ((partner_val % size) + size) % size;
        int partner_rank  = (partner_vrank + root) % size;

        /* Reception criterion: bottom bits are all-same */
        unsigned int lsbs = nb & xor_mask;
        int equal_lsbs = (lsbs == 0 || lsbs == xor_mask);

        if (recvd) {
            MPI_Send(buf, count, MPI_BYTE, partner_rank, step,
                     MPI_COMM_WORLD);
        } else if (equal_lsbs) {
            MPI_Recv(buf, count, MPI_BYTE, partner_rank, step,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recvd = 1;
        }

        step_mask >>= 1;
        step++;
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
#define TOPO_GENERIC   5

typedef struct {
    int type;
    union {
        struct { int rows, cols; } mesh;
        struct { int dim; } butterfly;
        struct { int hpl; } fattree;
        struct { int G, C, R, P; } dragonfly;
    };
} topo_cfg_t;

/* ---- Hierarchy definition per topology ----
 *
 * Returns group sizes from finest to coarsest level.
 *   FatTree:   [hpl]          — 1 level  (leaf switch)
 *   Dragonfly: [P, P*R, P*R*C] — 3 levels (router, chassis, group)
 *   Mesh/Butterfly: (empty)   — 0 levels (flat, use BFS)
 */
static int get_hier_levels(const topo_cfg_t *cfg, int *gsizes)
{
    switch (cfg->type) {
    case TOPO_FATTREE:
        gsizes[0] = cfg->fattree.hpl;
        return 1;
    case TOPO_DRAGONFLY: {
        int P = cfg->dragonfly.P, R = cfg->dragonfly.R;
        int C = cfg->dragonfly.C;
        gsizes[0] = P;
        gsizes[1] = P * R;
        gsizes[2] = P * R * C;
        return 3;
    }
    default: return 0;
    }
}

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
    } else if (strcmp(type, "generic") == 0) {
        cfg->type = TOPO_GENERIC;
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
    case TOPO_GENERIC:
        /* No structural adjacency — return all other nodes.
         * BFS caller will visit in this order (effectively flat). */
        for (int i = 0; i < size; i++)
            if (i != u) nbrs[n++] = i;
        break;
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

    /* Fixup: a fanout-limited BFS may leave orphans when the chain
     * exhausts all fanout slots before reaching some nodes (e.g.
     * fanout=1 on a mesh).  Attach each orphan to its nearest
     * already-visited neighbor, repeating until the tree spans all
     * nodes.  This may raise f_max above max_fanout for a few nodes,
     * which the cost model accounts for automatically. */
    for (;;) {
        int any = 0;
        for (int i = 0; i < size; i++) {
            if (visited[i]) continue;
            int nn = get_neighbors_sorted(cfg, i, size, nbrs);
            for (int j = 0; j < nn; j++) {
                if (visited[nbrs[j]]) {
                    visited[i] = 1;
                    parent[i] = nbrs[j];
                    any = 1;
                    break;
                }
            }
        }
        if (!any) break;
    }

    free(nbrs); free(visited); free(queue);
}

/* ---- Multi-level hierarchical tree for FatTree / Dragonfly ----
 *
 * Chains ambassadors at EVERY hierarchy level, then uses fanout-1
 * BFS within the finest group.  Every chain link carries exactly
 * 1 flow → zero contention.  Each ambassador's children use
 * different link types → full parallelism across levels.
 *
 *   Dragonfly (3 levels):
 *     Level 2: inter-group chain      (blue links,  1 flow)
 *     Level 1: inter-chassis chains   (black links, 1 flow per group)
 *     Level 0: inter-router chains    (green links, 1 flow per chassis)
 *     Finest:  intra-router BFS       (local,       fanout 1)
 *     Depth = 3+3+3+1 = 10  (vs 127 for pipe)
 *
 *   FatTree (1 level):
 *     Level 0: inter-leaf chain       (spine links, 1 flow)
 *     Finest:  intra-leaf BFS         (leaf switch, fanout 1)
 *     Depth = 7+15 = 22  (vs 127 for pipe)
 *
 * For Mesh/Butterfly (0 levels): falls through to standard BFS.
 * Result: a parent[] array compatible with run_test's broadcast loop.
 */
static void build_hierarchical_tree(const topo_cfg_t *cfg, int root, int size,
                                    int *parent, int max_fanout)
{
    int gsizes[8];
    int nlev = get_hier_levels(cfg, gsizes);

    if (nlev == 0) {
        build_bfs_tree(cfg, root, size, parent, max_fanout);
        return;
    }

    for (int i = 0; i < size; i++) parent[i] = -1;
    char *visited = calloc(size, 1);
    visited[root] = 1;

    /* Track ambassador set: each covers a group of 'cov' nodes */
    int *ambs = malloc(size * sizeof(int));
    int *covs = malloc(size * sizeof(int));
    int namb = 1;
    ambs[0] = root;
    covs[0] = size;

    /* Chain ambassadors at every level (coarsest → finest) */
    for (int lev = nlev - 1; lev >= 0; lev--) {
        int sgs = gsizes[lev];   /* sub-group size at this level */
        int *na = malloc(size * sizeof(int));
        int *nc = malloc(size * sizeof(int));
        int nn = 0;

        for (int a = 0; a < namb; a++) {
            int u   = ambs[a];
            int cov = covs[a];
            int nsub = cov / sgs;

            if (nsub <= 1) {
                na[nn] = u; nc[nn] = sgs; nn++;
                continue;
            }

            /* Round-robin sub-groups starting from ambassador's own */
            int base  = (u / cov) * cov;
            int my_si = (u - base) / sgs;
            int prev  = u;

            na[nn] = u; nc[nn] = sgs; nn++;  /* ambassador covers own sub-group */

            for (int s = 1; s < nsub; s++) {
                int si = (my_si + s) % nsub;
                int sa = base + si * sgs;     /* first node of sub-group */
                parent[sa] = prev;
                visited[sa] = 1;
                prev = sa;
                na[nn] = sa; nc[nn] = sgs; nn++;
            }
        }

        free(ambs); free(covs);
        ambs = na; covs = nc; namb = nn;
    }

    /* Finest level: intra-group BFS (fanout from max_fanout parameter;
     * 0 = unlimited).  Fanout 1 gives zero intra-group contention;
     * higher fanout reduces depth at the cost of bandwidth splitting. */
    int fgs = gsizes[0];
    int *queue = malloc(size * sizeof(int));
    int *nbrs  = malloc(size * sizeof(int));
    int local_fo = (max_fanout > 0) ? max_fanout : size;

    for (int a = 0; a < namb; a++) {
        int u   = ambs[a];
        int gid = u / fgs;
        int qh = 0, qt = 0;
        queue[qt++] = u;

        while (qh < qt) {
            int w = queue[qh++];
            int cnt = get_neighbors_sorted(cfg, w, size, nbrs);
            int added = 0;
            for (int j = 0; j < cnt && added < local_fo; j++) {
                int v = nbrs[j];
                if (!visited[v] && v / fgs == gid) {
                    visited[v] = 1;
                    parent[v] = w;
                    queue[qt++] = v;
                    added++;
                }
            }
        }
    }

    free(nbrs); free(queue); free(ambs); free(covs); free(visited);
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
 * Model (pipelined tree broadcast with non-blocking sends):
 *
 *   K* = sqrt( (D_eff − 1) · M  /  (f_max · L · B) )
 *
 *   T(K) = (K · f_eff + D_eff − 1) · (L + O + M / (KB))
 *
 *   K = number of chunks,  M = message size,  B = link bandwidth,
 *   L = base link latency,  O = per-step MPI overhead,
 *   f_max = max fan-out (controls chunk sizing),
 *   f_eff = contention factor (1 if links are independent, f_max
 *           if children share a switch/link — controls T estimate),
 *   D_eff = weighted tree depth  (max cumulative edge-latency
 *           from root to any leaf, measured in units of L).
 *
 * K uses f_max: more children → coarser chunks (amortise overhead).
 * T uses f_eff: only shared-link contention slows the pipeline.
 * O captures fixed per-step cost (MPI_Recv + Isend call overhead),
 * which penalises many small chunks vs fewer large ones.
 */
static int optimal_chunks(const topo_cfg_t *cfg, int size,
                          const int *parent, int count,
                          double *est_time)
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

    /* Contention factor for time estimation:
     *
     *   Flat topologies (Mesh, Butterfly): each tree edge uses a
     *   dedicated physical link (mesh direction, hypercube dimension),
     *   so children send on independent links — no contention.
     *
     *   Hierarchical topologies (FatTree, Dragonfly): children under
     *   the same switch/router share uplink bandwidth.
     *
     * K* uses f_max: more children → coarser chunks (amortise per-send
     * overhead regardless of link independence).
     *
     * T estimate uses f_eff: only shared-link contention slows the
     * pipeline, so independent links appear as f_eff = 1. */
    int gsizes[8];
    int nlev = get_hier_levels(cfg, gsizes);
    int f_eff = (nlev > 0) ? f_max : 1;

    /* K* = sqrt( (D_eff - 1) * M / (f_max * L * B) ) */
    double num = (D_eff - 1.0) * (double)count;
    double den = (double)f_max * L * B;
    int K = (int)ceil(sqrt(num / den));

    /* Clamp: at least 1 chunk, chunk size at least 16 KB.
     * The 16 KB floor prevents over-chunking on deep trees where
     * the formula ignores per-message MPI overhead (~μs). */
    if (K < 1) K = 1;
    int max_K = count / 16384;
    if (max_K < 1) max_K = 1;
    if (max_K > 10) max_K = 10;
    if (K > max_K) K = max_K;

    /* Per-step MPI overhead: each pipeline step involves Recv +
     * Isend calls with fixed setup cost.  Penalises many small
     * chunks, helping the model prefer fewer, larger pipeline steps
     * when the transfer-time difference is marginal. */
    double O = 500e-9;

    if (est_time)
        *est_time = ((double)K * f_eff + D_eff - 1.0) *
                    (L + O + (double)count / ((double)K * B));

    return K;
}

/* ---- Unified tree selection ----
 *
 * Evaluates candidate broadcast trees and selects the one with
 * lowest estimated pipeline time.  Topology-agnostic: the same
 * logic handles all topologies via get_hier_levels() and the
 * pipeline cost model.
 *
 *   Flat topologies (Mesh, Butterfly — 0 hierarchy levels):
 *     Candidates are BFS trees with fanout {unlimited, 1, 2}.
 *
 *   Hierarchical topologies (FatTree, Dragonfly — ≥1 levels):
 *     Adds hierarchical trees (ambassador chains between groups,
 *     BFS within groups) and a flat proximity chain.
 *
 * The cost model T = (K·f_eff + D−1)·(L + O + M/(KB)) naturally
 * favours low-fanout deep trees for large messages (bandwidth-
 * bound) and shallow trees for small messages (latency-bound).
 */
static void build_best_tree(const topo_cfg_t *cfg, int root, int size,
                            int *parent, int count)
{
    int gsizes[8];
    int nlev = get_hier_levels(cfg, gsizes);

    int *tmp = malloc(size * sizeof(int));
    double best = 1e30;

    /* Evaluate hierarchical trees with different intra-group fanouts.
     * For flat topologies (nlev=0) this reduces to plain BFS trees. */
    int fanouts[] = {0, 1, 2};
    for (int i = 0; i < 3; i++) {
        build_hierarchical_tree(cfg, root, size, tmp, fanouts[i]);
        double t;
        optimal_chunks(cfg, size, tmp, count, &t);
        if (t < best) {
            best = t;
            memcpy(parent, tmp, size * sizeof(int));
        }
    }

    /* For hierarchical topologies, also evaluate a flat proximity
     * chain (can win at large messages where pipeline depth
     * matters more than ambassador-chain contention avoidance). */
    if (nlev > 0) {
        build_bfs_tree(cfg, root, size, tmp, 1);
        double t;
        optimal_chunks(cfg, size, tmp, count, &t);
        if (t < best)
            memcpy(parent, tmp, size * sizeof(int));
    }

    free(tmp);
}

/* ================================================================
 * Reusable binomial broadcast on an arbitrary rank subset.
 *
 * members[0..nmembers-1] lists the participating ranks.
 * member_root is the rank (must be in members[]) that holds data.
 * tag_base offsets all MPI tags to prevent collisions when
 * multiple binomials run in sequence on overlapping rank sets.
 * Blocking Send/Recv, O(log nmembers) steps.
 * ================================================================ */
static void binomial_bcast_subset(void *buf, int count, int rank,
                                  const int *members, int nmembers,
                                  int member_root, int tag_base)
{
    /* Find my index in the members array (-1 if not a member) */
    int my_idx = -1, root_idx = -1;
    for (int i = 0; i < nmembers; i++) {
        if (members[i] == rank)       my_idx = i;
        if (members[i] == member_root) root_idx = i;
    }
    if (my_idx < 0) return;  /* not participating */

    /* Virtual rank: shift so root_idx becomes vrank 0 */
    int vrank = (my_idx - root_idx + nmembers) % nmembers;

    /* Compute ceil(log2(nmembers)) */
    int log2n = 0;
    { int tmp = 1; while (tmp < nmembers) { tmp <<= 1; log2n++; } }

    /* Standard binomial broadcast: step k from log2n-1 down to 0.
     * At each step, ranks with bits 0..k all zero are senders;
     * their child at vrank + 2^k is the receiver.  Iterating
     * high-to-low ensures receivers already have data before
     * they become senders at lower steps. */
    for (int k = log2n - 1; k >= 0; k--) {
        unsigned int mask = (1u << (k + 1)) - 1;
        int tag = tag_base + k;

        if (((unsigned int)vrank & mask) == 0) {
            /* Sender at step k */
            int child_vrank = vrank + (1 << k);
            if (child_vrank < nmembers) {
                int child_idx = (child_vrank + root_idx) % nmembers;
                MPI_Send(buf, count, MPI_BYTE, members[child_idx], tag,
                         MPI_COMM_WORLD);
            }
        } else if (((unsigned int)vrank & mask) == (1u << k)) {
            /* Receiver at step k */
            int parent_vrank = vrank - (1 << k);
            int parent_idx = (parent_vrank + root_idx) % nmembers;
            MPI_Recv(buf, count, MPI_BYTE, members[parent_idx], tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

/* ================================================================
 * Algorithm: GLF — Global-Links-First (Dorier et al., COMHPC 2016)
 *
 * Hierarchical topologies (FatTree, Dragonfly):
 *   Multi-level binomial cascade.  At each hierarchy level
 *   (coarsest → finest), current data holders broadcast to one
 *   representative per sub-group via binomial tree.
 *
 *   Dragonfly (3 levels → 4 phases):
 *     Phase 0: inter-group binomial      (global/blue links)
 *     Phase 1: inter-chassis per group   (black links)
 *     Phase 2: inter-router per chassis  (green links)
 *     Phase 3: intra-router binomial     (terminal links)
 *
 *   FatTree (1 level → 2 phases):
 *     Phase 0: inter-leaf binomial       (spine links)
 *     Phase 1: intra-leaf binomial       (leaf switch)
 *
 * Flat topologies (Mesh, Butterfly):
 *   Topology-aware binomial: BFS proximity order from root
 *   defines virtual ranks, standard binomial tree on those.
 *
 * Ref: Dorier et al., "Evaluation of Topology-Aware Broadcast
 *      Algorithms for Dragonfly Networks", COMHPC 2016;
 *      originally "group-first" in Xiang et al.
 * ================================================================ */
static double run_glf(void *buf, int count, int rank, int size,
                      int root, const char *topo_file)
{
    /* Derive .cfg path from .tdat directory: {dir}/topo_{size}.cfg */
    char cfg_path[4096];
    const char *last_slash = strrchr(topo_file, '/');
    if (last_slash) {
        int dirlen = (int)(last_slash - topo_file);
        snprintf(cfg_path, sizeof(cfg_path), "%.*s/topo_%d.cfg", dirlen, topo_file, size);
    } else {
        snprintf(cfg_path, sizeof(cfg_path), "topo_%d.cfg", size);
    }

    topo_cfg_t cfg;
    if (parse_topo_cfg(cfg_path, &cfg) != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: cannot parse topo config: %s\n",
                    cfg_path);
        return -1.0;
    }

    int gsizes[8];
    int nlev = get_hier_levels(&cfg, gsizes);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (nlev > 0) {
        /* ---- Multi-level hierarchical GLF ----
         *
         * For each hierarchy level (coarsest → finest):
         *   Within each super-group, broadcast from the current
         *   leader to one representative per sub-group using
         *   binomial tree.
         *
         * After all inter-level phases, broadcast within the
         * finest-level group to reach every rank. */

        /* Track my current group's data leader.
         * Initially only root has data. */
        int my_leader = root;

        for (int L = nlev - 1; L >= 0; L--) {
            int super_size = (L == nlev - 1) ? size : gsizes[L + 1];
            int sub_size = gsizes[L];
            int nsub = super_size / sub_size;

            if (nsub <= 1) continue;

            /* Identify my super-group */
            int my_super = rank / super_size;
            int super_base = my_super * super_size;

            /* Sub-group containing the leader */
            int leader_sub = (my_leader - super_base) / sub_size;

            /* Build representative array: one per sub-group.
             * Leader's sub-group is represented by the leader;
             * other sub-groups by their first rank. */
            int *members = malloc(nsub * sizeof(int));
            for (int s = 0; s < nsub; s++) {
                if (s == leader_sub)
                    members[s] = my_leader;
                else
                    members[s] = super_base + s * sub_size;
            }

            /* Participate in binomial if I'm a representative */
            int am_member = 0;
            for (int s = 0; s < nsub; s++)
                if (members[s] == rank) { am_member = 1; break; }

            int tag_base = (nlev - 1 - L) * 32;
            if (am_member)
                binomial_bcast_subset(buf, count, rank,
                                      members, nsub, my_leader,
                                      tag_base);

            /* Update my_leader for the next (finer) phase */
            int my_sub = (rank - super_base) / sub_size;
            my_leader = members[my_sub];

            free(members);
        }

        /* Final phase: broadcast within each finest-level group */
        int finest = gsizes[0];
        int finest_base = (rank / finest) * finest;
        int *members = malloc(finest * sizeof(int));
        for (int i = 0; i < finest; i++)
            members[i] = finest_base + i;

        int tag_base = nlev * 32;
        binomial_bcast_subset(buf, count, rank,
                              members, finest, my_leader,
                              tag_base);

        free(members);
    } else {
        /* ---- Flat: BFS-proximity binomial ---- */
        /* Build BFS ordering from root using topology adjacency */
        int *bfs_order = malloc(size * sizeof(int));
        char *visited = calloc(size, 1);
        int *queue = malloc(size * sizeof(int));
        int *nbrs = malloc(size * sizeof(int));
        int qh = 0, qt = 0;

        visited[root] = 1;
        queue[qt++] = root;
        int bfs_n = 0;
        bfs_order[bfs_n++] = root;

        while (qh < qt) {
            int u = queue[qh++];
            int nn = get_neighbors_sorted(&cfg, u, size, nbrs);
            for (int i = 0; i < nn; i++) {
                int v = nbrs[i];
                if (!visited[v]) {
                    visited[v] = 1;
                    queue[qt++] = v;
                    bfs_order[bfs_n++] = v;
                }
            }
        }

        /* bfs_order[] now maps virtual rank → physical rank.
         * Do a standard binomial bcast on these virtual ranks. */
        binomial_bcast_subset(buf, count, rank,
                              bfs_order, size, root, 0);

        free(nbrs);
        free(queue);
        free(visited);
        free(bfs_order);
    }

    double t1 = MPI_Wtime();
    return t1 - t0;
}

/* ================================================================
 * Algorithm 6: FFGB — Furthest-First Greedy Broadcast
 *
 * Discrete timestep simulation (rank 0 only):
 *   Each step, every informed node sends to the furthest unclaimed
 *   uninformed node (by pairwise latency from .tdat).  Senders are
 *   processed in (recv_step ASC, node_id ASC) order; once a target
 *   is claimed in a step, no other sender can pick it.
 *
 * Output: parent[] + child_seq[] arrays, broadcast to all ranks.
 * Execution: simple tree recv-then-Isend, no chunking.
 *
 * Complexity: O(N^2 log N) — fine for N <= 1024.
 * ================================================================ */

static double run_ffgb(void *buf, int count, int rank, int size,
                       int root, const char *topo_file)
{
    /* ---- Load topology data (same struct as run_test) ---- */
    typedef struct {
        int      N;
        double   bw_min, lat_base;
        float    *lat;
        uint16_t *flink;
        int      num_ev;
        double   *eigenvalues, *eigvecs;
    } td_t;

    td_t td;
    {
        FILE *fp = fopen(topo_file, "rb");
        if (!fp) {
            if (rank == 0)
                fprintf(stderr, "Error: cannot open topo data: %s\n",
                        topo_file);
            return -1.0;
        }
        char magic[4];
        uint32_t version, n, num_links;
        if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "TDAT", 4) != 0)
            { fclose(fp); return -1.0; }
        if (fread(&version, 4, 1, fp) != 1 || (version < 1 || version > 3))
            { fclose(fp); return -1.0; }
        if (fread(&n, 4, 1, fp) != 1) { fclose(fp); return -1.0; }
        if (fread(&num_links, 4, 1, fp) != 1) { fclose(fp); return -1.0; }
        if (fread(&td.bw_min, 8, 1, fp) != 1) { fclose(fp); return -1.0; }
        if (fread(&td.lat_base, 8, 1, fp) != 1) { fclose(fp); return -1.0; }

        td.N = (int)n;
        size_t nn = (size_t)n * n;
        td.lat   = malloc(nn * sizeof(float));
        td.flink = malloc(nn * sizeof(uint16_t));
        if (fread(td.lat, sizeof(float), nn, fp) != nn)
            { free(td.lat); free(td.flink); fclose(fp); return -1.0; }
        if (fread(td.flink, sizeof(uint16_t), nn, fp) != nn)
            { free(td.lat); free(td.flink); fclose(fp); return -1.0; }
        td.num_ev = 0; td.eigenvalues = NULL; td.eigvecs = NULL;
        /* skip spectral data — not needed */
        fclose(fp);
    }

    if (td.N != size) {
        if (rank == 0)
            fprintf(stderr,
                "Error: .tdat has %d nodes but MPI size is %d\n",
                td.N, size);
        free(td.lat); free(td.flink);
        return -1.0;
    }

    int N = size;
    int *parent    = malloc(N * sizeof(int));
    int *child_seq = malloc(N * sizeof(int));

    /* ---- Build plan on rank 0 ---- */
    if (rank == 0) {
        int *recv_step   = malloc(N * sizeof(int));
        int *child_count = calloc(N, sizeof(int));
        char *informed   = calloc(N, 1);

        for (int i = 0; i < N; i++) {
            parent[i]    = -1;
            child_seq[i] = 0;
            recv_step[i] = -1;
        }

        informed[root]   = 1;
        recv_step[root]  = 0;
        int n_informed   = 1;
        int step         = 0;

        /* Sender ordering buffer: (recv_step, node_id) — sorted */
        int *senders = malloc(N * sizeof(int));

        while (n_informed < N) {
            step++;

            /* Collect informed nodes, sort by (recv_step ASC, id ASC) */
            int nsend = 0;
            for (int i = 0; i < N; i++)
                if (informed[i]) senders[nsend++] = i;

            /* Insertion sort by (recv_step, id) — both ascending */
            for (int i = 1; i < nsend; i++) {
                int key = senders[i];
                int key_rs = recv_step[key];
                int j = i - 1;
                while (j >= 0 &&
                       (recv_step[senders[j]] > key_rs ||
                        (recv_step[senders[j]] == key_rs &&
                         senders[j] > key))) {
                    senders[j + 1] = senders[j];
                    j--;
                }
                senders[j + 1] = key;
            }

            /* Each sender picks furthest unclaimed uninformed node */
            char *claimed = calloc(N, 1);
            int n_claimed = 0;

            for (int si = 0; si < nsend; si++) {
                int s = senders[si];
                /* Find argmax lat[s*N + j] over uninformed unclaimed j */
                int best_j = -1;
                float best_lat = -1.0f;
                for (int j = 0; j < N; j++) {
                    if (informed[j] || claimed[j]) continue;
                    float l = td.lat[(size_t)s * N + j];
                    if (l > best_lat) {
                        best_lat = l;
                        best_j = j;
                    }
                }
                if (best_j >= 0) {
                    parent[best_j]    = s;
                    child_seq[best_j] = child_count[s]++;
                    recv_step[best_j] = step;
                    claimed[best_j]   = 1;
                    n_claimed++;
                }
            }

            /* Mark all claimed nodes as informed */
            for (int j = 0; j < N; j++)
                if (claimed[j]) informed[j] = 1;
            n_informed += n_claimed;

            free(claimed);

            if (n_claimed == 0) break;  /* safety: shouldn't happen */
        }

        free(senders);
        free(recv_step);
        free(child_count);
        free(informed);
    }

    /* ---- Broadcast plan to all ranks ---- */
    MPI_Bcast(parent,    N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(child_seq, N, MPI_INT, 0, MPI_COMM_WORLD);

    /* ---- Derive children list for this rank ----
     * Scan parent[] for nodes whose parent is this rank,
     * sorted by child_seq[]. */
    int nchildren = 0;
    for (int i = 0; i < N; i++)
        if (parent[i] == rank) nchildren++;

    int *children = NULL;
    if (nchildren > 0) {
        children = malloc(nchildren * sizeof(int));
        int idx = 0;
        for (int i = 0; i < N; i++)
            if (parent[i] == rank)
                children[idx++] = i;
        /* Sort by child_seq */
        for (int i = 1; i < nchildren; i++) {
            int key = children[i];
            int key_seq = child_seq[key];
            int j = i - 1;
            while (j >= 0 && child_seq[children[j]] > key_seq) {
                children[j + 1] = children[j];
                j--;
            }
            children[j + 1] = key;
        }
    }

    /* ---- Execute broadcast ---- */
    MPI_Request *reqs = nchildren > 0
        ? malloc(nchildren * sizeof(MPI_Request)) : NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Recv from parent */
    if (rank != root)
        MPI_Recv(buf, count, MPI_BYTE, parent[rank], 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Isend to children in pre-computed order */
    for (int ch = 0; ch < nchildren; ch++)
        MPI_Isend(buf, count, MPI_BYTE, children[ch], 0,
                  MPI_COMM_WORLD, &reqs[ch]);

    if (nchildren > 0)
        MPI_Waitall(nchildren, reqs, MPI_STATUSES_IGNORE);

    double t1 = MPI_Wtime();

    free(reqs);
    free(children);
    free(parent);
    free(child_seq);
    free(td.lat);
    free(td.flink);

    return t1 - t0;
}

/* ================================================================
 * Algorithm 7: OBFS — Optimal BFS Tree Broadcast
 *
 * Builds a BFS spanning tree from the root on the 1-hop adjacency
 * graph derived from the pairwise latency matrix.  BFS minimises
 * tree depth, and since all tree edges are disjoint links, every
 * level's transfers run at full bandwidth with zero contention.
 *
 * T = depth × T_1hop  (no pipelining, no chunking).
 *
 * For multi-component 1-hop graphs (e.g. FatTree leaf groups),
 * components are chained via the shortest available multi-hop link.
 *
 * Requires: .tdat topology data file.
 * ================================================================ */
static double run_obfs(void *buf, int count, int rank, int size,
                       int root, const char *topo_file, int nchunks)
{
    /* ---- Load topology data ---- */
    typedef struct {
        int      N;
        double   bw_min, lat_base;
        float    *lat;
        uint16_t *flink;
    } td_t;

    td_t td;
    {
        FILE *fp = fopen(topo_file, "rb");
        if (!fp) {
            if (rank == 0)
                fprintf(stderr, "Error: cannot open topo data: %s\n",
                        topo_file);
            return -1.0;
        }
        char magic[4];
        uint32_t version, n, num_links;
        if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "TDAT", 4) != 0)
            { fclose(fp); return -1.0; }
        if (fread(&version, 4, 1, fp) != 1 || (version < 1 || version > 3))
            { fclose(fp); return -1.0; }
        if (fread(&n, 4, 1, fp) != 1) { fclose(fp); return -1.0; }
        if (fread(&num_links, 4, 1, fp) != 1) { fclose(fp); return -1.0; }
        if (fread(&td.bw_min, 8, 1, fp) != 1) { fclose(fp); return -1.0; }
        if (fread(&td.lat_base, 8, 1, fp) != 1) { fclose(fp); return -1.0; }

        td.N = (int)n;
        size_t nn = (size_t)n * n;
        td.lat   = malloc(nn * sizeof(float));
        td.flink = malloc(nn * sizeof(uint16_t));
        if (fread(td.lat, sizeof(float), nn, fp) != nn)
            { free(td.lat); free(td.flink); fclose(fp); return -1.0; }
        if (fread(td.flink, sizeof(uint16_t), nn, fp) != nn)
            { free(td.lat); free(td.flink); fclose(fp); return -1.0; }
        fclose(fp);
    }

    if (td.N != size) {
        if (rank == 0)
            fprintf(stderr,
                "Error: .tdat has %d nodes but MPI size is %d\n",
                td.N, size);
        free(td.lat); free(td.flink);
        return -1.0;
    }

    int N = size;
    int *parent = malloc(N * sizeof(int));

    /* ---- Build BFS tree on rank 0 ---- */
    if (rank == 0) {
        for (int i = 0; i < N; i++) parent[i] = -1;

        /* 1-hop threshold: smallest positive latency × 1.01 */
        float lat_1hop = 1e30f;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                float l = td.lat[(size_t)i * N + j];
                if (l > 0.0f && l < lat_1hop) lat_1hop = l;
            }
        float lat_thresh = lat_1hop * 1.01f;

        char *visited = calloc(N, 1);
        int  *queue   = malloc(N * sizeof(int));
        int  *nbrs    = malloc(N * sizeof(int));
        float *nbr_lat = malloc(N * sizeof(float));

        /* BFS from root on 1-hop graph */
        int qh = 0, qt = 0;
        visited[root] = 1;
        queue[qt++] = root;

        while (qh < qt) {
            int u = queue[qh++];
            /* Collect 1-hop unvisited neighbors */
            int nn = 0;
            for (int v = 0; v < N; v++) {
                if (visited[v]) continue;
                float l = td.lat[(size_t)u * N + v];
                if (l > 0.0f && l <= lat_thresh) {
                    nbrs[nn] = v;
                    nbr_lat[nn] = l;
                    nn++;
                }
            }
            /* Sort neighbors by latency (closest first) */
            for (int i = 1; i < nn; i++) {
                int kn = nbrs[i];
                float kl = nbr_lat[i];
                int j = i - 1;
                while (j >= 0 && nbr_lat[j] > kl) {
                    nbrs[j+1] = nbrs[j];
                    nbr_lat[j+1] = nbr_lat[j];
                    j--;
                }
                nbrs[j+1] = kn;
                nbr_lat[j+1] = kl;
            }
            /* Add all to BFS tree (no fanout limit) */
            for (int i = 0; i < nn; i++) {
                int v = nbrs[i];
                if (visited[v]) continue;  /* may have been added by another */
                visited[v] = 1;
                parent[v] = u;
                queue[qt++] = v;
            }
        }

        /* Handle disconnected 1-hop components:
         * link unvisited nodes via shortest multi-hop path to any
         * visited node, then continue BFS within their component. */
        while (qt < N) {
            /* Find the unvisited node closest to any visited node */
            int best_u = -1, best_v = -1;
            float best_l = 1e30f;
            for (int v = 0; v < N; v++) {
                if (visited[v]) continue;
                for (int u = 0; u < N; u++) {
                    if (!visited[u]) continue;
                    float l = td.lat[(size_t)u * N + v];
                    if (l > 0.0f && l < best_l) {
                        best_l = l;
                        best_u = u;
                        best_v = v;
                    }
                }
            }
            if (best_v < 0) break;  /* shouldn't happen */

            /* Attach and continue BFS from this node */
            visited[best_v] = 1;
            parent[best_v] = best_u;
            queue[qt++] = best_v;

            /* Continue BFS on 1-hop edges from this new component */
            while (qh < qt) {
                int u = queue[qh++];
                int nn = 0;
                for (int v = 0; v < N; v++) {
                    if (visited[v]) continue;
                    float l = td.lat[(size_t)u * N + v];
                    if (l > 0.0f && l <= lat_thresh) {
                        nbrs[nn] = v;
                        nbr_lat[nn] = l;
                        nn++;
                    }
                }
                for (int i = 1; i < nn; i++) {
                    int kn = nbrs[i];
                    float kl = nbr_lat[i];
                    int j = i - 1;
                    while (j >= 0 && nbr_lat[j] > kl) {
                        nbrs[j+1] = nbrs[j];
                        nbr_lat[j+1] = nbr_lat[j];
                        j--;
                    }
                    nbrs[j+1] = kn;
                    nbr_lat[j+1] = kl;
                }
                for (int i = 0; i < nn; i++) {
                    int v = nbrs[i];
                    if (visited[v]) continue;
                    visited[v] = 1;
                    parent[v] = u;
                    queue[qt++] = v;
                }
            }
        }

        free(visited);
        free(queue);
        free(nbrs);
        free(nbr_lat);
    }

    /* ---- Broadcast tree to all ranks ---- */
    MPI_Bcast(parent, N, MPI_INT, 0, MPI_COMM_WORLD);

    /* ---- Derive children list for this rank ---- */
    int nchildren = 0;
    for (int i = 0; i < N; i++)
        if (parent[i] == rank) nchildren++;

    int *children = NULL;
    if (nchildren > 0) {
        children = malloc(nchildren * sizeof(int));
        int idx = 0;
        for (int i = 0; i < N; i++)
            if (parent[i] == rank)
                children[idx++] = i;
    }

    /* ---- Execute pipelined broadcast ---- */
    if (nchunks < 1) nchunks = 1;
    int chunk_sz = count / nchunks;
    if (chunk_sz < 1) { nchunks = count; chunk_sz = 1; }

    /* Fire-and-forget: collect ALL requests, single Waitall at end */
    int max_reqs = nchunks * nchildren;
    MPI_Request *reqs = max_reqs > 0
        ? malloc(max_reqs * sizeof(MPI_Request)) : NULL;
    int nreqs = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int c = 0; c < nchunks; c++) {
        int offset = c * chunk_sz;
        int thiscount = (c == nchunks - 1) ? count - offset : chunk_sz;
        char *ptr = (char *)buf + offset;

        if (rank != root)
            MPI_Recv(ptr, thiscount, MPI_BYTE, parent[rank], c,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int ch = 0; ch < nchildren; ch++)
            MPI_Isend(ptr, thiscount, MPI_BYTE, children[ch], c,
                      MPI_COMM_WORLD, &reqs[nreqs++]);
    }

    if (nreqs > 0)
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    double t1 = MPI_Wtime();

    free(reqs);
    free(children);
    free(parent);
    free(td.lat);
    free(td.flink);

    return t1 - t0;
}

/* ================================================================
 * Data-driven topology structures (replaces topo_cfg_t for run_test)
 *
 * Loaded from .tdat files produced by topo_preprocess.py.
 * Contains pairwise latencies and first-link IDs extracted from
 * the SimGrid XML via Floyd-Warshall on the full network graph
 * (nodes + switches).
 *
 * Key insight: contention detection is automatic.  When two
 * children of a node share the same first outgoing physical link
 * (same flink value), their sends contend — regardless of whether
 * the topology is "flat" or "hierarchical".
 * ================================================================ */

typedef struct {
    int      N;           /* number of compute nodes                */
    double   bw_min;      /* min bottleneck bandwidth (bytes/sec)   */
    double   lat_base;    /* min link latency (seconds)             */
    float    *lat;        /* N*N pairwise latency matrix (seconds)  */
    uint16_t *flink;      /* N*N first-link-ID matrix               */
    int      num_ev;      /* number of eigenvectors (0 if v1)       */
    double   *eigenvalues; /* num_ev eigenvalues (λ₂..λ_{k+1})     */
    double   *eigvecs;    /* num_ev * N eigenvector matrix          */
    int      max_hops;    /* max links per path (0 if no path data) */
    uint8_t  *path_len;   /* N*N path lengths (NULL if no path data)*/
    uint16_t *path_links; /* N*N*max_hops full path link IDs        */
} topo_data_t;

static int topo_data_load(const char *path, topo_data_t *td)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    char magic[4];
    uint32_t version, n, num_links;

    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "TDAT", 4) != 0)
        { fclose(fp); return -1; }
    if (fread(&version, 4, 1, fp) != 1 || (version < 1 || version > 3))
        { fclose(fp); return -1; }
    if (fread(&n, 4, 1, fp) != 1)
        { fclose(fp); return -1; }
    if (fread(&num_links, 4, 1, fp) != 1)
        { fclose(fp); return -1; }
    if (fread(&td->bw_min, 8, 1, fp) != 1)
        { fclose(fp); return -1; }
    if (fread(&td->lat_base, 8, 1, fp) != 1)
        { fclose(fp); return -1; }

    td->N = (int)n;
    size_t nn = (size_t)n * n;
    td->lat   = malloc(nn * sizeof(float));
    td->flink = malloc(nn * sizeof(uint16_t));

    if (fread(td->lat, sizeof(float), nn, fp) != nn)
        { free(td->lat); free(td->flink); fclose(fp); return -1; }
    if (fread(td->flink, sizeof(uint16_t), nn, fp) != nn)
        { free(td->lat); free(td->flink); fclose(fp); return -1; }

    /* Spectral data (v2 only) */
    td->num_ev = 0;
    td->eigenvalues = NULL;
    td->eigvecs = NULL;

    if (version >= 2) {
        uint32_t nev;
        if (fread(&nev, 4, 1, fp) != 1)
            { free(td->lat); free(td->flink); fclose(fp); return -1; }
        td->num_ev = (int)nev;
        if (nev > 0) {
            td->eigenvalues = malloc(nev * sizeof(double));
            td->eigvecs     = malloc((size_t)nev * n * sizeof(double));
            if (fread(td->eigenvalues, sizeof(double), nev, fp) != nev)
                { free(td->lat); free(td->flink);
                  free(td->eigenvalues); free(td->eigvecs);
                  fclose(fp); return -1; }
            if (fread(td->eigvecs, sizeof(double), (size_t)nev * n, fp)
                != (size_t)nev * n)
                { free(td->lat); free(td->flink);
                  free(td->eigenvalues); free(td->eigvecs);
                  fclose(fp); return -1; }
        }
    }

    /* Full path data (v3) */
    td->max_hops   = 0;
    td->path_len   = NULL;
    td->path_links = NULL;

    if (version >= 3) {
        uint8_t mh;
        if (fread(&mh, 1, 1, fp) == 1 && mh > 0) {
            td->max_hops = (int)mh;
            td->path_len   = malloc(nn * sizeof(uint8_t));
            td->path_links = malloc(nn * mh * sizeof(uint16_t));
            if (fread(td->path_len, 1, nn, fp) != nn ||
                fread(td->path_links, sizeof(uint16_t), nn * mh, fp)
                    != nn * mh) {
                free(td->path_len); free(td->path_links);
                td->max_hops = 0;
                td->path_len = NULL;
                td->path_links = NULL;
            }
        }
    }

    fclose(fp);
    return 0;
}

static void topo_data_free(topo_data_t *td)
{
    free(td->lat);
    free(td->flink);
    free(td->eigenvalues);
    free(td->eigvecs);
    free(td->path_len);
    free(td->path_links);
    td->lat         = NULL;
    td->flink       = NULL;
    td->eigenvalues = NULL;
    td->eigvecs     = NULL;
    td->path_len    = NULL;
    td->path_links  = NULL;
}

/* ================================================================
 * Adjacency-Spectral BFS Tree with Data-Driven f_eff
 *
 * One unified model for ALL topologies — no topology-specific code.
 *
 * Tree building:
 *   BFS ordered by latency ASCENDING (physically closest first),
 *   with spectral distance as tie-breaker.  This fills local
 *   physical clusters before crossing to remote ones:
 *     - Flat (Mesh, Butterfly):  BFS expands through direct links
 *     - Hierarchical (FatTree, Dragonfly):  BFS fills same-switch
 *       group first, then crosses to next group
 *
 * Contention detection (f_eff):
 *   For each tree node, count how many children share the same
 *   first physical link (flink).  Max contention group = f_eff.
 *   Detected from data, not topology type:
 *     - Butterfly: 7 distinct first-links per node → f_eff = 1
 *     - FatTree/Dragonfly: 1 first-link per node → f_eff = f_max
 *     - 2D Mesh: 2 first-links → moderate f_eff
 *
 * Pipeline formula:  K* = sqrt((D_eff - 1) * M / (f_max * L * B))
 * Time estimate:     T  = (K * f_eff + D_eff - 1) * (L + M/(K*B))
 * ================================================================ */

/* Comparator state for sorting candidate nodes by latency. */
static const topo_data_t *_sd_td;
static int _sd_ref;

/* Sort by latency ASCENDING (physically closest first). */
static int _adj_spec_cmp(const void *a, const void *b)
{
    int ua = *(const int *)a;
    int ub = *(const int *)b;
    int N  = _sd_td->N;

    float la = _sd_td->lat[(size_t)_sd_ref * N + ua];
    float lb = _sd_td->lat[(size_t)_sd_ref * N + ub];
    if (la < lb) return -1;
    if (la > lb) return  1;
    return 0;
}

/* Build a BFS tree restricted to 1-hop physical adjacency.
 *
 * Only creates tree edges at the minimum non-zero latency level
 * (= direct physical links).  Every tree edge is a single physical
 * hop, avoiding multi-hop path contention.
 *
 * For disconnected 1-hop graphs (FatTree leaf groups, Dragonfly
 * router groups), detects connected components and chains them:
 *
 *   1. Find all connected components via BFS on 1-hop graph
 *   2. Root's component first; chain remaining components as
 *      ambassadors: prev_ambassador → closest_node_in_next_component
 *   3. Within each component: BFS with fanout cap
 *
 * The cross-component chain has fanout 1 (zero contention on
 * inter-group links).  Within-component BFS uses the given fanout.
 *
 * max_fanout = 0 → unlimited (capped by physical degree).
 */
static void build_onehop_bfs(const topo_data_t *td, int root,
                              int size, int *parent, int max_fanout)
{
    int N = td->N;
    int limit = (max_fanout > 0) ? max_fanout : size;

    /* Global 1-hop threshold */
    float lat_1hop = 1e30f;
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++) {
            float l = td->lat[(size_t)i * N + j];
            if (l > 0.0f && l < lat_1hop) lat_1hop = l;
        }
    float lat_thresh = lat_1hop * 1.01f;

    /* ---- Step 1: Find connected components of the 1-hop graph ---- */
    int *comp_id = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) comp_id[i] = -1;

    int *comp_roots = malloc(size * sizeof(int));  /* first node of each component */
    int *comp_sizes = malloc(size * sizeof(int));
    int ncomp = 0;

    int *queue = malloc(size * sizeof(int));
    for (int s = 0; s < size; s++) {
        if (comp_id[s] >= 0) continue;
        int cid = ncomp++;
        comp_roots[cid] = s;
        comp_sizes[cid] = 0;
        int qh = 0, qt = 0;
        comp_id[s] = cid;
        queue[qt++] = s;
        while (qh < qt) {
            int u = queue[qh++];
            comp_sizes[cid]++;
            for (int v = 0; v < size; v++) {
                if (comp_id[v] >= 0) continue;
                float l = td->lat[(size_t)u * N + v];
                if (l > 0.0f && l <= lat_thresh) {
                    comp_id[v] = cid;
                    queue[qt++] = v;
                }
            }
        }
    }

    /* ---- Step 2: Order components by latency from root ---- */
    int root_cid = comp_id[root];

    /* For each component, find the node closest to root (by latency) */
    int *ambassadors = malloc(ncomp * sizeof(int));
    ambassadors[0] = root;  /* slot 0 = root's component */

    /* Sort remaining components by min latency from root */
    int *comp_order = malloc(ncomp * sizeof(int));
    float *comp_dist = malloc(ncomp * sizeof(float));
    int nord = 0;
    for (int c = 0; c < ncomp; c++) {
        if (c == root_cid) continue;
        comp_order[nord] = c;
        /* Find min latency from root to any node in component c */
        float best = 1e30f;
        int best_node = -1;
        for (int v = 0; v < size; v++) {
            if (comp_id[v] != c) continue;
            float l = td->lat[(size_t)root * N + v];
            if (l < best) { best = l; best_node = v; }
        }
        comp_dist[nord] = best;
        ambassadors[c] = best_node;  /* closest to root */
        nord++;
    }
    /* Simple insertion sort by distance */
    for (int i = 1; i < nord; i++) {
        float key_d = comp_dist[i];
        int key_c = comp_order[i];
        int j = i - 1;
        while (j >= 0 && comp_dist[j] > key_d) {
            comp_dist[j+1] = comp_dist[j];
            comp_order[j+1] = comp_order[j];
            j--;
        }
        comp_dist[j+1] = key_d;
        comp_order[j+1] = key_c;
    }

    /* Refine: for each component in order, pick the ambassador
     * closest to the PREVIOUS ambassador (not root) */
    int prev_amb = root;
    for (int i = 0; i < nord; i++) {
        int c = comp_order[i];
        float best = 1e30f;
        int best_node = -1;
        for (int v = 0; v < size; v++) {
            if (comp_id[v] != c) continue;
            float l = td->lat[(size_t)prev_amb * N + v];
            if (l < best) { best = l; best_node = v; }
        }
        ambassadors[c] = best_node;
        prev_amb = best_node;
    }

    /* ---- Step 3: Build tree ---- */
    for (int i = 0; i < size; i++) parent[i] = -1;
    char *visited = calloc(size, 1);
    int *nbrs = malloc(size * sizeof(int));

    /* Binomial tree over ambassadors: depth O(log2(ngroups)) instead
     * of O(ngroups) for a linear chain.  amb_list[0] = root,
     * amb_list[1..nord] = ambassadors in component order.
     * Parent of amb_list[j] = amb_list[j & (j-1)] (clear lowest set bit). */
    {
        int *amb_list = malloc((nord + 1) * sizeof(int));
        amb_list[0] = root;
        for (int i = 0; i < nord; i++)
            amb_list[i + 1] = ambassadors[comp_order[i]];
        for (int j = 1; j <= nord; j++) {
            int p = j & (j - 1);   /* clear lowest set bit = binomial parent */
            parent[amb_list[j]] = amb_list[p];
        }
        free(amb_list);
    }

    /* BFS within each component, starting from its ambassador */
    /* Process root's component first, then others in chain order */
    int *comp_seq = malloc((nord + 1) * sizeof(int));
    comp_seq[0] = root_cid;
    for (int i = 0; i < nord; i++) comp_seq[i+1] = comp_order[i];

    for (int ci = 0; ci < nord + 1; ci++) {
        int c = comp_seq[ci];
        int amb = ambassadors[c];
        int qh = 0, qt = 0;
        visited[amb] = 1;
        queue[qt++] = amb;

        while (qh < qt) {
            int u = queue[qh++];
            int nn = 0;
            for (int v = 0; v < size; v++) {
                if (visited[v] || comp_id[v] != c) continue;
                float l = td->lat[(size_t)u * N + v];
                if (l > 0.0f && l <= lat_thresh)
                    nbrs[nn++] = v;
            }
            if (nn == 0) continue;

            _sd_td  = td;
            _sd_ref = u;
            qsort(nbrs, nn, sizeof(int), _adj_spec_cmp);

            int added = 0;
            for (int i = 0; i < nn && added < limit; i++) {
                int v = nbrs[i];
                visited[v] = 1;
                parent[v] = u;
                queue[qt++] = v;
                added++;
            }
        }
    }

    /* Fixup: any remaining unvisited nodes (shouldn't happen but safe) */
    for (int v = 0; v < size; v++) {
        if (!visited[v]) {
            float best = 1e30f;
            int best_p = root;
            for (int u = 0; u < size; u++) {
                if (!visited[u]) continue;
                float l = td->lat[(size_t)u * N + v];
                if (l < best) { best = l; best_p = u; }
            }
            visited[v] = 1;
            parent[v] = best_p;
        }
    }

    free(comp_seq); free(nbrs); free(visited); free(queue);
    free(comp_id); free(comp_roots); free(comp_sizes);
    free(ambassadors); free(comp_order); free(comp_dist);
}

#include "../BBS/bbs.c"

/* Farthest-Point-First (FPF) Dispersion Tree.
 *
 * Gonzalez (1985) k-center greedy heuristic applied to broadcast tree
 * construction:  at each step, the node farthest from the current
 * informed set is added, with its parent chosen as the nearest
 * available informed node (contention-aware tie-breaking via flink).
 *
 * This maximizes spatial dispersion of informed nodes at each tree
 * level — the first ~K steps place ambassadors across distinct
 * topology groups, then remaining steps fill locally.
 *
 * O(N²) time, rank 0 only.
 *
 * max_fanout = 0 → unlimited.
 */
static void build_fpf_tree(const topo_data_t *td, int root,
                            int size, int *parent, int max_fanout)
{
    int N = td->N;
    int cap = (max_fanout > 0) ? max_fanout : size;

    /* State arrays */
    char *informed = calloc(size, 1);
    float *min_dist = malloc(size * sizeof(float));
    int *fanout = calloc(size, sizeof(int));
    /* Per-parent flink values of assigned children:
     * child_fl[u * size + i] = flink from u to its i-th child. */
    uint16_t *child_fl = malloc((size_t)size * size * sizeof(uint16_t));

    /* Initialize: root is informed */
    for (int j = 0; j < size; j++) {
        parent[j] = -1;
        min_dist[j] = td->lat[(size_t)root * N + j];
    }
    informed[root] = 1;
    min_dist[root] = 0.0f;

    /* Greedy loop: add N-1 nodes */
    for (int step = 0; step < size - 1; step++) {
        /* 1. Find target = argmax min_dist among uninformed */
        int target = -1;
        float best_dist = -1.0f;
        for (int j = 0; j < size; j++) {
            if (informed[j]) continue;
            if (min_dist[j] > best_dist) {
                best_dist = min_dist[j];
                target = j;
            }
        }
        if (target < 0) break;  /* all informed (shouldn't happen) */

        /* 2. Parent selection: nearest eligible informed node,
         *    with contention-aware flink tie-breaking.
         *
         *    a) Find nearest latency among eligible (under fanout cap)
         *    b) 10% threshold catches quantized ties
         *    c) Among candidates within threshold, pick least flink
         *       overlap with existing children
         *    d) Break remaining ties by lowest latency */

        /* a) Nearest eligible latency */
        float nearest_lat = 1e30f;
        for (int u = 0; u < size; u++) {
            if (!informed[u] || fanout[u] >= cap) continue;
            float l = td->lat[(size_t)u * N + target];
            if (l < nearest_lat) nearest_lat = l;
        }

        /* b) Threshold */
        float thresh = nearest_lat * 1.10f;

        /* c-d) Best parent: least overlap, then lowest latency */
        int best_parent = -1;
        int best_overlap = size + 1;
        float best_lat = 1e30f;

        for (int u = 0; u < size; u++) {
            if (!informed[u] || fanout[u] >= cap) continue;
            float l = td->lat[(size_t)u * N + target];
            if (l > thresh) continue;

            /* Count flink overlap with u's existing children */
            uint16_t fl = td->flink[(size_t)u * N + target];
            int overlap = 0;
            for (int ci = 0; ci < fanout[u]; ci++) {
                if (child_fl[(size_t)u * size + ci] == fl)
                    overlap++;
            }

            if (overlap < best_overlap ||
                (overlap == best_overlap && l < best_lat)) {
                best_overlap = overlap;
                best_lat = l;
                best_parent = u;
            }
        }

        /* Fallback: all at cap — pick nearest informed, ignore cap */
        if (best_parent < 0) {
            float bl = 1e30f;
            for (int u = 0; u < size; u++) {
                if (!informed[u]) continue;
                float l = td->lat[(size_t)u * N + target];
                if (l < bl) { bl = l; best_parent = u; }
            }
        }

        /* Assign target to best_parent */
        parent[target] = best_parent;
        child_fl[(size_t)best_parent * size + fanout[best_parent]] =
            td->flink[(size_t)best_parent * N + target];
        fanout[best_parent]++;
        informed[target] = 1;

        /* 3. Update coverage: min_dist to nearest informed node */
        for (int j = 0; j < size; j++) {
            if (informed[j]) continue;
            float d = td->lat[(size_t)target * N + j];
            if (d < min_dist[j]) min_dist[j] = d;
        }
    }

    free(child_fl);
    free(fanout);
    free(min_dist);
    free(informed);
}

/* Estimate pipelined broadcast time for a given tree.
 *
 * Returns (optimal_K, estimated_time) for the tree defined by parent[].
 *
 * Model:  T = (K * f_eff + D_eff - 1) * (L + O + M / (K * B))
 *   K* = sqrt((D_eff - 1) * M / (f_max * L * B))
 *
 * f_eff is detected from the flink matrix:  for each node u with
 * children c1..cf, count how many children share the same first
 * physical link from u.  The max contention group across all nodes
 * gives f_eff.  This is topology-agnostic:
 *   - Butterfly: 7 distinct first-links → f_eff = 1
 *   - FatTree/Dragonfly: 1 first-link per node → f_eff = f_max
 *   - 2D Mesh: 2 first-links → f_eff ≈ f_max/2
 */
static double _estimate_time(const topo_data_t *td, int size,
                              const int *parent, int count,
                              int *out_K)
{
    int N = td->N;
    int *fanout = calloc(size, sizeof(int));
    for (int v = 0; v < size; v++)
        if (parent[v] >= 0) fanout[parent[v]]++;

    int f_max = 0;
    for (int v = 0; v < size; v++)
        if (fanout[v] > f_max) f_max = fanout[v];

    /* Build children lists */
    int *child_of = malloc(size * sizeof(int));
    int *cstart   = calloc(size + 1, sizeof(int));
    for (int v = 0; v < size; v++) cstart[v + 1] = cstart[v] + fanout[v];
    int *pos = calloc(size, sizeof(int));
    for (int v = 0; v < size; v++) {
        if (parent[v] < 0) continue;
        int p = parent[v];
        child_of[cstart[p] + pos[p]++] = v;
    }

    /* BFS: compute D_eff and detect f_eff from flink contention */
    double *cum_lat = calloc(size, sizeof(double));
    int *queue = malloc(size * sizeof(int));
    int qh = 0, qt = 0;
    int root_node = -1;
    for (int v = 0; v < size; v++)
        if (parent[v] < 0) { root_node = v; break; }
    queue[qt++] = root_node;
    double max_cum_lat = 0.0;
    int f_eff = 1;

    while (qh < qt) {
        int u = queue[qh++];
        int nc = fanout[u];

        /* Detect per-node contention from flink:
         * count max children sharing the same first physical link. */
        if (nc > 1) {
            /* Count occurrences of each flink ID among u's children */
            for (int i = cstart[u]; i < cstart[u] + nc; i++) {
                uint16_t fl = td->flink[(size_t)u * N + child_of[i]];
                int cnt = 0;
                for (int j = cstart[u]; j < cstart[u] + nc; j++) {
                    if (td->flink[(size_t)u * N + child_of[j]] == fl)
                        cnt++;
                }
                if (cnt > f_eff) f_eff = cnt;
            }
        }

        for (int i = cstart[u]; i < cstart[u] + nc; i++) {
            int v = child_of[i];
            cum_lat[v] = cum_lat[u]
                       + (double)td->lat[(size_t)u * N + v];
            if (cum_lat[v] > max_cum_lat) max_cum_lat = cum_lat[v];
            queue[qt++] = v;
        }
    }

    free(queue); free(cum_lat); free(pos);
    free(child_of); free(cstart); free(fanout);

    double L = td->lat_base;
    double B = td->bw_min;
    double O = 500e-9;   /* 500 ns MPI overhead per step */
    double D_eff = max_cum_lat / L;
    if (D_eff < 1.0) D_eff = 1.0;
    if (f_max < 1) f_max = 1;

    /* Optimal K: use f_max and raw L to size chunks that amortize
     * per-send overhead.  K* = sqrt((D_eff - 1) * M / (f_max * L * B)) */
    double num = (D_eff - 1.0) * (double)count;
    double den = (double)f_max * L * B;
    int K = (den > 0.0) ? (int)ceil(sqrt(num / den)) : 1;
    if (K < 1) K = 1;
    int max_K = count / 16384;          /* chunks must be >= 16KB */
    if (max_K < 1) max_K = 1;
    if (K > max_K) K = max_K;

    /* Time estimate (using f_eff for contention) */
    double chunk = (double)count / K;
    double T = ((double)K * f_eff + D_eff - 1.0) * (L + O + chunk / B);

    *out_K = K;
    return T;
}

/* Try multiple fanouts on 1-hop BFS, pick the tree with lowest
 * estimated time.
 *
 * Every tree edge is a direct physical link (no contention from
 * multi-hop sharing).  For hierarchical topologies with disconnected
 * 1-hop graphs, component-aware BFS creates ambassador chains.
 */
static void build_spectral_tree(const topo_data_t *td, int root, int size,
                                 int *parent, int count, int *out_K)
{
    int *tmp = malloc(size * sizeof(int));
    double best_T = 1e30;

    /* 1-hop BFS: fanouts {0, 2, 3, 4, 8}
     * fanout=0 is safe here (capped by physical degree, not N) */
    {
        static const int fanouts[] = {0, 1, 2, 3, 4, 8};
        int nf = sizeof(fanouts) / sizeof(fanouts[0]);
        for (int fi = 0; fi < nf; fi++) {
            build_onehop_bfs(td, root, size, tmp, fanouts[fi]);
            int K;
            double T = _estimate_time(td, size, tmp, count, &K);
            if (T < best_T) {
                best_T = T;
                *out_K = K;
                memcpy(parent, tmp, size * sizeof(int));
            }
        }
    }

    /* FPF dispersion tree with multiple fanout caps */
    {
        static const int fpf_fanouts[] = {0, 2, 4, 8};
        int nf = sizeof(fpf_fanouts) / sizeof(fpf_fanouts[0]);
        for (int fi = 0; fi < nf; fi++) {
            build_fpf_tree(td, root, size, tmp, fpf_fanouts[fi]);
            int K;
            double T = _estimate_time(td, size, tmp, count, &K);
            if (T < best_T) {
                best_T = T;
                *out_K = K;
                memcpy(parent, tmp, size * sizeof(int));
            }
        }
    }

    free(tmp);
}

/* Count connected components of the 1-hop physical adjacency graph.
 * Uses the same threshold as build_onehop_bfs. */
static int count_onehop_components(const topo_data_t *td, int size)
{
    int N = td->N;

    /* Global 1-hop threshold */
    float lat_1hop = 1e30f;
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++) {
            float l = td->lat[(size_t)i * N + j];
            if (l > 0.0f && l < lat_1hop) lat_1hop = l;
        }
    float lat_thresh = lat_1hop * 1.01f;

    char *visited = calloc(size, 1);
    int *queue = malloc(size * sizeof(int));
    int ncomp = 0;

    for (int s = 0; s < size; s++) {
        if (visited[s]) continue;
        ncomp++;
        int qh = 0, qt = 0;
        visited[s] = 1;
        queue[qt++] = s;
        while (qh < qt) {
            int u = queue[qh++];
            for (int v = 0; v < size; v++) {
                if (visited[v]) continue;
                float l = td->lat[(size_t)u * N + v];
                if (l > 0.0f && l <= lat_thresh) {
                    visited[v] = 1;
                    queue[qt++] = v;
                }
            }
        }
    }

    free(queue);
    free(visited);
    return ncomp;
}

/* ================================================================
 * Algorithm 4: Pipelined broadcast (run_test)
 *
 * 1. For small-medium messages, uses srda (scatter + recursive-
 *    doubling allgather) which distributes data across all links.
 *    Threshold: flat topology ≤ 2MB, hierarchical ≤ 32KB.
 * 2. Otherwise builds best 1-hop BFS tree (evaluates fanouts,
 *    picks lowest T) and pipelines K chunks.
 * ================================================================ */
static double run_test(void *buf, int count, int rank, int size,
                       int root, const char *topo_file,
                       int *out_nchunks)
{
    topo_data_t td;
    if (topo_data_load(topo_file, &td) != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: cannot load topo data: %s\n",
                    topo_file);
        return -1.0;
    }
    if (td.N != size) {
        if (rank == 0)
            fprintf(stderr,
                "Error: .tdat has %d nodes but MPI size is %d\n",
                td.N, size);
        topo_data_free(&td);
        return -1.0;
    }
    /* 0. Hybrid srda for small-medium messages.
     *    srda's scatter+recursive-doubling-allgather uses all links
     *    and MPI collectives, beating tree broadcast at small sizes.
     *    Threshold depends on topology structure:
     *      Flat (1 component):        count ≤ 2 MB
     *      Hierarchical (>1 comp):    count ≤ 32 KB
     *    Only for power-of-2 N (required by XOR recursive doubling). */
    if ((size & (size - 1)) == 0) {    /* power-of-2 check */
        int ncomp = count_onehop_components(&td, size);
        int srda_thresh = (ncomp <= 1) ? (2 * 1024 * 1024) : (32 * 1024);
        if (count <= srda_thresh) {
            *out_nchunks = 1;
            topo_data_free(&td);
            return run_srda(buf, count, rank, size, root);
        }
    }

    /* 1. Build best tree (evaluates multiple fanouts) */
    int *parent = malloc(size * sizeof(int));
    int nchunks = 1;
    build_spectral_tree(&td, root, size, parent, count, &nchunks);
    *out_nchunks = nchunks;

    /* 2. Determine children */
    int nchildren = 0;
    int *children = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
        if (parent[i] == rank) children[nchildren++] = i;

    /* 3. Pipelined broadcast — single Waitall at the end */
    int chunk_size = (count + nchunks - 1) / nchunks;
    int total_sends = nchunks * nchildren;
    MPI_Request *reqs = total_sends > 0
        ? malloc((size_t)total_sends * sizeof(MPI_Request)) : NULL;
    int nsend = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int c = 0; c < nchunks; c++) {
        int off = c * chunk_size;
        int len = chunk_size;
        if (off + len > count) len = count - off;

        /* Receive chunk from parent */
        if (rank != root)
            MPI_Recv((char *)buf + off, len, MPI_BYTE,
                     parent[rank], c, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        /* Forward to all children */
        for (int ch = 0; ch < nchildren; ch++)
            MPI_Isend((char *)buf + off, len, MPI_BYTE,
                      children[ch], c, MPI_COMM_WORLD,
                      &reqs[nsend++]);
        /* No per-chunk Waitall — let sends overlap across chunks */
    }

    if (nsend > 0)
        MPI_Waitall(nsend, reqs, MPI_STATUSES_IGNORE);

    double t1 = MPI_Wtime();

    free(reqs);
    free(parent);
    free(children);
    topo_data_free(&td);
    return t1 - t0;
}

/* ================================================================
 * Sweep helpers — resume-capable root sweep with per-root checkpoints
 *
 * Usage:  runner <algo> <bytes> <nchunks> sweep <outdir> [topo_cfg]
 *
 * For a given (algo, msg_bytes, N):
 *   1. If bulk  outdir/N{N}_MSG{M}.json exists       → skip, exit 0
 *   2. Check which outdir/N{N}_MSG{M}_R{r}.json exist → skip those roots
 *   3. Run only missing roots, write per-root JSON after each
 *   4. Aggregate all per-root JSONs → bulk JSON
 *   5. Remove per-root files
 * ================================================================ */

static int file_exists(const char *path)
{
    FILE *f = fopen(path, "r");
    if (f) { fclose(f); return 1; }
    return 0;
}

/* Parse time_sec and correct from a per-root JSON file */
static int parse_root_json(const char *path, double *t, int *ok)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    *t = 0.0; *ok = 1;
    while (fgets(line, sizeof(line), f)) {
        char *p;
        if ((p = strstr(line, "\"time_sec\""))) {
            p = strchr(p, ':');
            if (p) sscanf(p + 1, " %lf", t);
        } else if ((p = strstr(line, "\"correct\""))) {
            *ok = strstr(p, "false") ? 0 : 1;
        }
    }
    fclose(f);
    return 0;
}

/* ================================================================
 * Parallel sweep launcher — runs BEFORE MPI_Init.
 *
 * Spawns independent smpirun processes (one per missing root) with
 * fork()/exec(), limited to SWEEP_JOBS concurrency.
 *
 * Required env:
 *   SWEEP_SMPI  — smpirun command prefix, e.g.:
 *       "smpirun -np 128 -platform p.xml -hostfile h.txt --cfg=..."
 *     or with singularity:
 *       "singularity exec --bind /path img.sif smpirun -np 128 ..."
 *
 * Optional env:
 *   SWEEP_JOBS  — max parallel jobs (default 8)
 *
 * Each child runs:
 *   $SWEEP_SMPI <self> <algo> <msg> <nchunks> <root> <outdir>/N_MSG_R<root>.json [topo_cfg]
 * ================================================================ */
static int run_sweep_parallel(int argc, char **argv, const char *smpi_cmd)
{
    const char *algo = argv[1];
    int msg          = atoi(argv[2]);
    int nchunks_arg  = (argc > 3) ? atoi(argv[3]) : 64;
    /* argv[4] = "sweep" */
    const char *outdir   = (argc > 5) ? argv[5] : NULL;
    const char *topo_cfg = (argc > 6) ? argv[6] : NULL;
    if (topo_cfg && strcmp(topo_cfg, "_") == 0) topo_cfg = NULL;

    if (!outdir) {
        fprintf(stderr, "sweep requires: runner <algo> <msg> [nchunks] sweep <outdir> [topo_cfg]\n");
        return 1;
    }

    /* Get binary path for re-invocation */
    const char *bin_path = getenv("SWEEP_BIN");
    if (!bin_path) bin_path = argv[0];

    /* Concurrency limit */
    const char *jobs_str = getenv("SWEEP_JOBS");
    int max_jobs = jobs_str ? atoi(jobs_str) : 8;
    if (max_jobs < 1) max_jobs = 1;

    /* Parse N from the -np argument in SWEEP_SMPI */
    int N = 0;
    {
        const char *np = strstr(smpi_cmd, "-np ");
        if (np) N = atoi(np + 4);
    }
    if (N <= 0) {
        fprintf(stderr, "ERROR: cannot find -np <N> in SWEEP_SMPI\n");
        return 1;
    }

    /* reported nchunks for JSON output */
    int reported_nc = nchunks_arg;
    if (strcmp(algo, "pipe") != 0 && strcmp(algo, "obfs") != 0 &&
        strcmp(algo, "test") != 0)
        reported_nc = 1;

    /* 1. Bulk JSON already exists → skip */
    char bulk_path[4096];
    snprintf(bulk_path, sizeof(bulk_path),
             "%s/N%d_MSG%d.json", outdir, N, msg);
    if (file_exists(bulk_path)) {
        printf("--- %s N=%d MSG=%d already done, skipping ---\n",
               algo, N, msg);
        return 0;
    }

    /* 2. Check which per-root JSONs already exist */
    int *root_done = calloc(N, sizeof(int));
    int n_existing = 0;
    for (int r = 0; r < N; r++) {
        char rp[4096];
        snprintf(rp, sizeof(rp),
                 "%s/N%d_MSG%d_R%d.json", outdir, N, msg, r);
        root_done[r] = file_exists(rp);
        n_existing += root_done[r];
    }

    int n_todo = N - n_existing;
    if (n_todo == 0)
        printf("--- %s N=%d MSG=%d: all %d roots cached, aggregating ---\n",
               algo, N, msg, N);
    else if (n_existing > 0)
        printf("--- %s N=%d MSG=%d: %d/%d cached, running %d (j=%d) ---\n",
               algo, N, msg, n_existing, N, n_todo, max_jobs);
    else
        printf("--- %s N=%d MSG=%d: running all %d roots (j=%d) ---\n",
               algo, N, msg, N, max_jobs);
    fflush(stdout);

    /* 3. Fork parallel smpirun processes for missing roots */
    int running = 0, any_fail = 0;

    for (int r = 0; r < N; r++) {
        if (root_done[r]) continue;

        /* Wait if at concurrency limit */
        while (running >= max_jobs) {
            int status;
            wait(&status);
            running--;
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
                any_fail = 1;
        }

        pid_t pid = fork();
        if (pid == 0) {
            /* Child: exec smpirun for this single root */
            char rp[4096];
            snprintf(rp, sizeof(rp),
                     "%s/N%d_MSG%d_R%d.json", outdir, N, msg, r);
            char cmd[16384];
            if (topo_cfg)
                snprintf(cmd, sizeof(cmd),
                    "%s %s %s %d %d %d %s %s",
                    smpi_cmd, bin_path, algo, msg, nchunks_arg, r, rp, topo_cfg);
            else
                snprintf(cmd, sizeof(cmd),
                    "%s %s %s %d %d %d %s",
                    smpi_cmd, bin_path, algo, msg, nchunks_arg, r, rp);
            execlp("sh", "sh", "-c", cmd, (char *)NULL);
            _exit(127);
        } else if (pid < 0) {
            fprintf(stderr, "ERROR: fork() failed for root %d\n", r);
            any_fail = 1;
        } else {
            running++;
        }
    }

    /* Drain remaining children */
    while (running > 0) {
        int status;
        wait(&status);
        running--;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            any_fail = 1;
    }

    if (any_fail)
        fprintf(stderr,
            "WARNING: some roots failed — will aggregate what exists\n");

    /* 4. Aggregate all per-root JSONs → bulk JSON */
    double *atimes = calloc(N, sizeof(double));
    int correct = 1, any_missing = 0;

    for (int r = 0; r < N; r++) {
        char rp[4096];
        snprintf(rp, sizeof(rp),
                 "%s/N%d_MSG%d_R%d.json", outdir, N, msg, r);
        double t; int ok;
        if (parse_root_json(rp, &t, &ok) != 0) {
            fprintf(stderr, "ERROR: missing root %d (%s) — cannot aggregate\n",
                    r, rp);
            any_missing = 1;
            break;
        }
        atimes[r] = t;
        if (!ok) correct = 0;
    }

    if (!any_missing) {
        double sum = 0, mn = atimes[0], mx = atimes[0];
        for (int i = 0; i < N; i++) {
            sum += atimes[i];
            if (atimes[i] < mn) mn = atimes[i];
            if (atimes[i] > mx) mx = atimes[i];
        }
        double mean = sum / N;
        double sum2 = 0;
        for (int i = 0; i < N; i++)
            sum2 += (atimes[i] - mean) * (atimes[i] - mean);
        double stdev = sqrt(sum2 / N);

        FILE *jfp = fopen(bulk_path, "w");
        if (jfp) {
            fprintf(jfp,
                "{\n"
                "  \"algorithm\": \"%s\",\n"
                "  \"nodes\": %d,\n"
                "  \"msg_bytes\": %d,\n"
                "  \"nchunks\": %d,\n"
                "  \"n_roots\": %d,\n"
                "  \"mean_sec\": %.9e,\n"
                "  \"stdev_sec\": %.9e,\n"
                "  \"min_sec\": %.9e,\n"
                "  \"max_sec\": %.9e,\n"
                "  \"correct\": %s,\n"
                "  \"per_root_sec\": [",
                algo, N, msg, reported_nc,
                N, mean, stdev, mn, mx,
                correct ? "true" : "false");
            for (int i = 0; i < N; i++)
                fprintf(jfp, "%s%.9e", i ? ", " : "", atimes[i]);
            fprintf(jfp, "]\n}\n");
            fclose(jfp);
        }

        /* 5. Cleanup per-root files */
        for (int r = 0; r < N; r++) {
            char rp[4096];
            snprintf(rp, sizeof(rp),
                     "%s/N%d_MSG%d_R%d.json", outdir, N, msg, r);
            remove(rp);
        }

        printf("  %s N=%d MSG=%d: mean=%.6e max=%.6e max/min=%.1fx %s\n",
               algo, N, msg, mean, mx, mx / mn,
               correct ? "OK" : "FAIL");
    } else {
        /* Count how many roots we DO have so user knows progress */
        int have = 0;
        for (int r = 0; r < N; r++) {
            char rp[4096];
            snprintf(rp, sizeof(rp),
                     "%s/N%d_MSG%d_R%d.json", outdir, N, msg, r);
            if (file_exists(rp)) have++;
        }
        fprintf(stderr,
            "  %s N=%d MSG=%d: %d/%d roots completed — re-run to finish\n",
            algo, N, msg, have, N);
    }

    free(atimes);
    free(root_done);
    return any_missing ? 1 : 0;
}

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
                "Usage: %s <algo> <msg_bytes> [nchunks] [root] [out_json] [topo_cfg]\n"
                "       %s <algo> <msg_bytes> [nchunks] sweep <outdir> [topo_cfg]\n"
                "  algorithm  : mpi | srda | pipe | bine | glf | ffgb | obfs | test\n"
                "  nchunks    : pipeline depth (default 64, auto for test)\n"
                "  root       : 0..%d | 'all' | 'sweep' (default 0)\n"
                "  sweep      : smart root sweep — skips existing per-root JSONs,\n"
                "               runs missing roots, aggregates, cleans up\n"
                "  out_json   : output JSON file (single/all root mode)\n"
                "  outdir     : output directory  (sweep mode)\n"
                "  topo_cfg   : topology config file (required for glf/test)\n",
                argv[0], argv[0], size - 1);
        MPI_Finalize();
        return 1;
    }

    const char *algo = argv[1];
    int nbytes       = atoi(argv[2]);
    int nchunks      = (argc > 3) ? atoi(argv[3]) : 64;
    const char *out_json  = (argc > 5) ? argv[5] : NULL;
    const char *topo_file = (argc > 6) ? argv[6] : NULL;
    /* Treat "_" as no-file placeholder */
    if (out_json  && strcmp(out_json,  "_") == 0) out_json  = NULL;
    if (topo_file && strcmp(topo_file, "_") == 0) topo_file = NULL;

    int all_roots  = 0;
    int max_compute_rank = 0;   /* 0 = use size; >0 = limit all-roots to 0..max-1 */
    int sweep_mode = 0;
    const char *sweep_outdir = NULL;
    int root = 0;
    int batch_mode = 0, root_lo_batch = 0, root_hi_batch = 0;
    if (argc > 4) {
        if (strcmp(argv[4], "all") == 0) {
            all_roots = 1;
        } else if (strncmp(argv[4], "all:", 4) == 0) {
            all_roots = 1;
            max_compute_rank = atoi(argv[4] + 4);
        } else if (strcmp(argv[4], "sweep") == 0) {
            sweep_mode = 1;
            sweep_outdir = (argc > 5) ? argv[5] : NULL;
            topo_file    = (argc > 6) ? argv[6] : NULL;
            if (topo_file && strcmp(topo_file, "_") == 0) topo_file = NULL;
            if (!sweep_outdir) {
                if (rank == 0)
                    fprintf(stderr, "sweep mode requires <outdir>\n");
                MPI_Finalize();
                return 1;
            }
        } else {
            char *dash = strchr(argv[4], '-');
            if (dash && dash != argv[4]) {
                batch_mode = 1;
                root_lo_batch = atoi(argv[4]);
                root_hi_batch = atoi(dash + 1);
                const char *colon = strchr(dash, ':');
                if (colon) max_compute_rank = atoi(colon + 1);
            } else {
                root = atoi(argv[4]);
                const char *colon = strchr(argv[4], ':');
                if (colon) max_compute_rank = atoi(colon + 1);
            }
        }
    }

    if (nbytes <= 0) {
        if (rank == 0) fprintf(stderr, "msg_bytes must be > 0\n");
        MPI_Finalize();
        return 1;
    }
    if (!all_roots && !sweep_mode && !batch_mode && (root < 0 || root >= size)) {
        if (rank == 0)
            fprintf(stderr, "root must be in 0..%d or 'all'/'sweep' (got %d)\n",
                    size - 1, root);
        MPI_Finalize();
        return 1;
    }

    /* SRDA and bine require power-of-2 ranks */
    if (strcmp(algo, "srda") == 0 || strcmp(algo, "bine") == 0) {
        if ((size & (size - 1)) != 0) {
            if (rank == 0)
                fprintf(stderr, "%s requires power-of-2 ranks (got %d)\n",
                        algo, size);
            MPI_Finalize();
            return 1;
        }
    }

    /* Validate algo before entering the root loop */
    int algo_id = -1;
    if      (strcmp(algo, "mpi")  == 0) algo_id = 0;
    else if (strcmp(algo, "srda") == 0) algo_id = 1;
    else if (strcmp(algo, "pipe") == 0) algo_id = 2;
    else if (strcmp(algo, "test") == 0) algo_id = 3;
    else if (strcmp(algo, "bine") == 0) algo_id = 4;
    else if (strcmp(algo, "glf")  == 0) algo_id = 5;
    else if (strcmp(algo, "ffgb") == 0) algo_id = 6;
    else if (strcmp(algo, "obfs") == 0) algo_id = 7;
    else if (strcmp(algo, "bbs")  == 0) algo_id = 8;
    else {
        if (rank == 0) fprintf(stderr, "Unknown algorithm: %s\n", algo);
        MPI_Finalize();
        return 1;
    }
    if ((algo_id == 3 || algo_id == 5 || algo_id == 6 || algo_id == 7 || algo_id == 8) && !topo_file) {
        if (rank == 0)
            fprintf(stderr, "%s algorithm requires topo_cfg argument\n", algo);
        MPI_Finalize();
        return 1;
    }

    /* pipe (2) uses nchunks from the command line.
     * test (3) computes its own nchunks internally.
     * For all others, report nchunks=1 (whole message). */
    int reported_nchunks = nchunks;
    if (algo_id != 2 && algo_id != 3 && algo_id != 7 && algo_id != 8)
        reported_nchunks = 1;

    /* ================================================================
     * Sweep mode: smart root sweep with per-root checkpoint/resume
     *
     * Two sub-modes:
     *   SWEEP_SMPI set → rank 0 forks parallel smpirun children (fast)
     *   SWEEP_SMPI unset → sequential root loop within this process
     * ================================================================ */
    if (sweep_mode) {
        /* Parallel sweep: rank 0 forks children, others wait */
        const char *smpi_cmd = getenv("SWEEP_SMPI");
        if (smpi_cmd && strlen(smpi_cmd) > 0) {
            int ret = 0;
            if (rank == 0)
                ret = run_sweep_parallel(argc, argv, smpi_cmd);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            return ret;
        }

        /* Sequential sweep fallback (no SWEEP_SMPI) */
        char bulk_path[4096];
        snprintf(bulk_path, sizeof(bulk_path),
                 "%s/N%d_MSG%d.json", sweep_outdir, size, nbytes);

        /* 1. Already fully aggregated? → done */
        int skip_all = 0;
        if (rank == 0)
            skip_all = file_exists(bulk_path);
        MPI_Bcast(&skip_all, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (skip_all) {
            if (rank == 0)
                printf("--- %s N=%d MSG=%d already done, skipping ---\n",
                       algo, size, nbytes);
            MPI_Finalize();
            return 0;
        }

        /* 2. Which per-root JSONs already exist? */
        int *root_done = calloc(size, sizeof(int));
        int n_existing = 0;
        if (rank == 0) {
            for (int r = 0; r < size; r++) {
                char rp[4096];
                snprintf(rp, sizeof(rp),
                         "%s/N%d_MSG%d_R%d.json",
                         sweep_outdir, size, nbytes, r);
                root_done[r] = file_exists(rp);
                n_existing += root_done[r];
            }
        }
        MPI_Bcast(root_done, size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&n_existing, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int n_todo = size - n_existing;
        if (rank == 0) {
            if (n_existing > 0)
                printf("--- %s N=%d MSG=%d: %d/%d roots cached, running %d ---\n",
                       algo, size, nbytes, n_existing, size, n_todo);
            else
                printf("--- %s N=%d MSG=%d: running all %d roots ---\n",
                       algo, size, nbytes, size);
        }

        /* 3. Run missing roots */
        if ((size_t)nbytes > _sim_buf_bytes) {
            SMPI_SHARED_FREE(_sim_buf);
            _sim_buf = SMPI_SHARED_MALLOC(nbytes);
            _sim_buf_bytes = (size_t)nbytes;
        }
        char *buf = _sim_buf;

        for (int r = 0; r < size; r++) {
            if (root_done[r]) continue;

            /* Only root fills buffer; all coroutines share the same allocation */
            if (rank == r) {
                memset(buf, 0, nbytes);
                for (int i = 0; i < nbytes; i++)
                    buf[i] = (char)(i & 0xFF);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            double elapsed = 0.0;
            switch (algo_id) {
            case 0: elapsed = run_mpi_bcast(buf, nbytes, r);                    break;
            case 1: elapsed = run_srda(buf, nbytes, rank, size, r);             break;
            case 2: elapsed = run_pipe(buf, nbytes, rank, size, r, nchunks);    break;
            case 3: { int nc2 = nchunks;
                      elapsed = run_test(buf, nbytes, rank, size, r, topo_file,
                                         &nc2);                                 break; }
            case 4: elapsed = run_bine(buf, nbytes, rank, size, r);             break;
            case 5: elapsed = run_glf(buf, nbytes, rank, size, r, topo_file);   break;
            case 6: elapsed = run_ffgb(buf, nbytes, rank, size, r, topo_file);  break;
            case 7: elapsed = run_obfs(buf, nbytes, rank, size, r, topo_file,
                                       nchunks);                                break;
            case 8: elapsed = run_bbs(buf, nbytes, rank, size, r, topo_file,
                                      nchunks);                                 break;
            }

            /* Verify */
            int ok = 1;
            for (int i = 0; i < nbytes; i++)
                if (buf[i] != (char)(i & 0xFF)) { ok = 0; break; }

            double max_time;
            MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX,
                       0, MPI_COMM_WORLD);
            int all_ok;
            MPI_Reduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN,
                       0, MPI_COMM_WORLD);

            /* Write per-root JSON immediately (crash recovery checkpoint) */
            if (rank == 0) {
                char rp[4096];
                snprintf(rp, sizeof(rp),
                         "%s/N%d_MSG%d_R%d.json",
                         sweep_outdir, size, nbytes, r);
                FILE *jfp = fopen(rp, "w");
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
                        algo, size, nbytes, reported_nchunks, r, max_time,
                        all_ok ? "true" : "false");
                    fclose(jfp);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        /* 4. Aggregate all per-root JSONs → bulk JSON */
        if (rank == 0) {
            double *atimes = calloc(size, sizeof(double));
            int correct = 1, any_missing = 0;

            for (int r = 0; r < size; r++) {
                char rp[4096];
                snprintf(rp, sizeof(rp),
                         "%s/N%d_MSG%d_R%d.json",
                         sweep_outdir, size, nbytes, r);
                double t; int ok;
                if (parse_root_json(rp, &t, &ok) != 0) {
                    fprintf(stderr,
                        "ERROR: missing %s — cannot aggregate\n", rp);
                    any_missing = 1;
                    break;
                }
                atimes[r] = t;
                if (!ok) correct = 0;
            }

            if (!any_missing) {
                double sum = 0, mn = atimes[0], mx = atimes[0];
                for (int i = 0; i < size; i++) {
                    sum += atimes[i];
                    if (atimes[i] < mn) mn = atimes[i];
                    if (atimes[i] > mx) mx = atimes[i];
                }
                double mean = sum / size;
                double sum2 = 0;
                for (int i = 0; i < size; i++)
                    sum2 += (atimes[i] - mean) * (atimes[i] - mean);
                double stdev = sqrt(sum2 / size);

                FILE *jfp = fopen(bulk_path, "w");
                if (jfp) {
                    fprintf(jfp,
                        "{\n"
                        "  \"algorithm\": \"%s\",\n"
                        "  \"nodes\": %d,\n"
                        "  \"msg_bytes\": %d,\n"
                        "  \"nchunks\": %d,\n"
                        "  \"n_roots\": %d,\n"
                        "  \"mean_sec\": %.9e,\n"
                        "  \"stdev_sec\": %.9e,\n"
                        "  \"min_sec\": %.9e,\n"
                        "  \"max_sec\": %.9e,\n"
                        "  \"correct\": %s,\n"
                        "  \"per_root_sec\": [",
                        algo, size, nbytes, reported_nchunks,
                        size, mean, stdev, mn, mx,
                        correct ? "true" : "false");
                    for (int i = 0; i < size; i++)
                        fprintf(jfp, "%s%.9e",
                                i ? ", " : "", atimes[i]);
                    fprintf(jfp, "]\n}\n");
                    fclose(jfp);
                }

                /* 5. Cleanup per-root files */
                for (int r = 0; r < size; r++) {
                    char rp[4096];
                    snprintf(rp, sizeof(rp),
                             "%s/N%d_MSG%d_R%d.json",
                             sweep_outdir, size, nbytes, r);
                    remove(rp);
                }

                printf("  %s N=%d MSG=%d: mean=%.6e max=%.6e max/min=%.1fx %s\n",
                       algo, size, nbytes, mean, mx, mx / mn,
                       correct ? "OK" : "FAIL");
            }

            free(atimes);
        }

        free(root_done);
        MPI_Finalize();
        return 0;
    }

    /* ---- Root loop (single root / batch / all-roots mode) ---- */
    int root_lo, root_hi;
    if (all_roots) {
        root_lo = 0;
        root_hi = (max_compute_rank > 0 ? max_compute_rank : size) - 1;
    } else if (batch_mode) {
        root_lo = root_lo_batch;
        root_hi = root_hi_batch;
    } else {
        root_lo = root;
        root_hi = root;
    }
    int nroots  = root_hi - root_lo + 1;

    /* In batch mode, check which per-root JSONs already exist and skip them */
    int *root_done = NULL;
    if (batch_mode && out_json) {
        root_done = calloc(nroots, sizeof(int));
        int n_cached = 0;
        if (rank == 0) {
            for (int i = 0; i < nroots; i++) {
                char rp[4096];
                snprintf(rp, sizeof(rp), "%s_R%d.json", out_json, root_lo + i);
                root_done[i] = file_exists(rp);
                n_cached += root_done[i];
            }
            if (n_cached > 0)
                printf("batch %d..%d: %d/%d roots cached, running %d\n",
                       root_lo, root_hi, n_cached, nroots, nroots - n_cached);
        }
        MPI_Bcast(root_done, nroots, MPI_INT, 0, MPI_COMM_WORLD);
    }

    double *times = calloc(nroots, sizeof(double));
    int    *oks   = calloc(nroots, sizeof(int));
    if ((size_t)nbytes > _sim_buf_bytes) {
        SMPI_SHARED_FREE(_sim_buf);
        _sim_buf = SMPI_SHARED_MALLOC(nbytes);
        _sim_buf_bytes = (size_t)nbytes;
    }
    char *buf = _sim_buf;

    for (int r = root_lo; r <= root_hi; r++) {
        /* Skip roots whose JSON already exists (batch mode) */
        if (root_done && root_done[r - root_lo]) continue;
        /* Only root fills buffer; all coroutines share the same allocation */
        if (rank == r) {
            memset(buf, 0, nbytes);
            for (int i = 0; i < nbytes; i++)
                buf[i] = (char)(i & 0xFF);
        }

        /* ---- Dispatch ---- */
        double elapsed = 0.0;
        switch (algo_id) {
        case 0: elapsed = run_mpi_bcast(buf, nbytes, r);                    break;
        case 1: elapsed = run_srda(buf, nbytes, rank, size, r);             break;
        case 2: elapsed = run_pipe(buf, nbytes, rank, size, r, nchunks);    break;
        case 3: elapsed = run_test(buf, nbytes, rank, size, r, topo_file,
                                   &nchunks);
                reported_nchunks = nchunks;                                 break;
        case 4: elapsed = run_bine(buf, nbytes, rank, size, r);              break;
        case 5: elapsed = run_glf(buf, nbytes, rank, size, r, topo_file);    break;
        case 6: elapsed = run_ffgb(buf, nbytes, rank, size, r, topo_file);   break;
        case 7: elapsed = run_obfs(buf, nbytes, rank, size, r, topo_file, nchunks); break;
        case 8: elapsed = run_bbs(buf, nbytes, rank, size, r, topo_file, nchunks);  break;
        }

        /* ---- Verify ---- */
        int ok = 1;
        for (int i = 0; i < nbytes; i++)
            if (buf[i] != (char)(i & 0xFF)) { ok = 0; break; }

        /* ---- Collect on rank 0 ---- */
        double max_time;
        MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);
        int all_ok;
        MPI_Reduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        times[r - root_lo] = max_time;
        oks[r - root_lo]   = all_ok;

        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* ---- Report ---- */
    if (rank == 0) {
        if (all_roots) {
            /* Compute mean, stdev, min, max over all roots */
            double sum = 0, sum2 = 0;
            double mn = times[0], mx = times[0];
            int correct = 1;
            for (int i = 0; i < nroots; i++) {
                sum  += times[i];
                sum2 += times[i] * times[i];
                if (times[i] < mn) mn = times[i];
                if (times[i] > mx) mx = times[i];
                if (!oks[i]) correct = 0;
            }
            double mean  = sum / nroots;
            double stdev = sqrt(sum2 / nroots - mean * mean);

            printf("algorithm : %s\n", algo);
            printf("nodes     : %d\n", max_compute_rank > 0 ? max_compute_rank : size);
            printf("msg_bytes : %d\n", nbytes);
            printf("roots     : all (0..%d)\n", root_hi);
            printf("mean_sec  : %.6e +/- %.2e\n", mean, stdev);
            printf("min_sec   : %.9f\n", mn);
            printf("max_sec   : %.9f\n", mx);
            printf("correct   : %s\n", correct ? "yes" : "NO");

            /* ---- JSON output (all-roots) ---- */
            if (out_json) {
                FILE *jfp = fopen(out_json, "w");
                if (jfp) {
                    fprintf(jfp,
                        "{\n"
                        "  \"algorithm\": \"%s\",\n"
                        "  \"nodes\": %d,\n"
                        "  \"msg_bytes\": %d,\n"
                        "  \"nchunks\": %d,\n"
                        "  \"n_roots\": %d,\n"
                        "  \"mean_sec\": %.9e,\n"
                        "  \"stdev_sec\": %.9e,\n"
                        "  \"min_sec\": %.9e,\n"
                        "  \"max_sec\": %.9e,\n"
                        "  \"correct\": %s,\n"
                        "  \"per_root_sec\": [",
                        algo, max_compute_rank > 0 ? max_compute_rank : size,
                        nbytes, reported_nchunks,
                        nroots, mean, stdev, mn, mx,
                        correct ? "true" : "false");
                    for (int i = 0; i < nroots; i++)
                        fprintf(jfp, "%s%.9e",
                                i ? ", " : "", times[i]);
                    fprintf(jfp, "]\n}\n");
                    fclose(jfp);
                } else {
                    fprintf(stderr,
                        "Warning: could not open %s for writing\n",
                        out_json);
                }
            }
        } else if (batch_mode) {
            /* ---- Batch mode: write per-root JSON files ---- */
            int nodes_val = max_compute_rank > 0 ? max_compute_rank : size;
            printf("algorithm : %s\n", algo);
            printf("nodes     : %d\n", nodes_val);
            printf("msg_bytes : %d\n", nbytes);
            printf("roots     : batch %d..%d\n", root_lo, root_hi);
            for (int i = 0; i < nroots; i++) {
                if (root_done && root_done[i]) continue;
                printf("  root %d : %.9f %s\n", root_lo + i, times[i],
                       oks[i] ? "ok" : "FAIL");
            }

            if (out_json) {
                for (int i = 0; i < nroots; i++) {
                    if (root_done && root_done[i]) continue;
                    int r = root_lo + i;
                    char path[4096];
                    snprintf(path, sizeof(path), "%s_R%d.json", out_json, r);
                    FILE *jfp = fopen(path, "w");
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
                            algo, nodes_val, nbytes, reported_nchunks,
                            r, times[i],
                            oks[i] ? "true" : "false");
                        fclose(jfp);
                    } else {
                        fprintf(stderr,
                            "Warning: could not open %s for writing\n", path);
                    }
                }
            }
        } else {
            printf("algorithm : %s\n", algo);
            printf("nodes     : %d\n", max_compute_rank > 0 ? max_compute_rank : size);
            printf("msg_bytes : %d\n", nbytes);
            printf("nchunks   : %d\n", reported_nchunks);
            printf("root      : %d\n", root);
            printf("time_sec  : %.9f\n", times[0]);
            printf("correct   : %s\n", oks[0] ? "yes" : "NO");

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
                        algo, max_compute_rank > 0 ? max_compute_rank : size,
                        nbytes, reported_nchunks, root,
                        times[0],
                        oks[0] ? "true" : "false");
                    fclose(jfp);
                } else {
                    fprintf(stderr, "Warning: could not open %s for writing\n",
                            out_json);
                }
            }
        }
    }

    free(times); free(oks); free(root_done);
    MPI_Finalize();
    return 0;
}
