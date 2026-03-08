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
#include <stdint.h>

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
 * Raw binary tree broadcast using Recv/Isend/Waitall.
 * Exact same logic as MPI_Bcast's binary tree but using atomic ops.
 * Binary tree: vrank's children are 2*vrank+1 and 2*vrank+2.
 * ================================================================ */
static double run_rawbino(void *buf, int count, int rank, int size,
                          int root)
{
    int vrank = (rank - root + size) % size;

    /* Binary tree: parent = (vrank-1)/2, children = 2*vrank+1, 2*vrank+2 */
    int parent_rank = -1;
    if (vrank != 0)
        parent_rank = ((vrank - 1) / 2 + root) % size;

    int nchildren = 0;
    int children_ranks[2];
    int left  = 2 * vrank + 1;
    int right = 2 * vrank + 2;
    if (left < size)
        children_ranks[nchildren++] = (left + root) % size;
    if (right < size)
        children_ranks[nchildren++] = (right + root) % size;

    MPI_Request *reqs = nchildren > 0
        ? malloc(nchildren * sizeof(MPI_Request)) : NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (vrank != 0)
        MPI_Recv(buf, count, MPI_BYTE, parent_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int ch = 0; ch < nchildren; ch++)
        MPI_Isend(buf, count, MPI_BYTE, children_ranks[ch], 0,
                  MPI_COMM_WORLD, &reqs[ch]);

    if (nchildren > 0)
        MPI_Waitall(nchildren, reqs, MPI_STATUSES_IGNORE);

    double t1 = MPI_Wtime();
    free(reqs);
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
    topo_cfg_t cfg;
    if (parse_topo_cfg(topo_file, &cfg) != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: cannot parse topo config: %s\n",
                    topo_file);
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
} topo_data_t;

static int topo_data_load(const char *path, topo_data_t *td)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    char magic[4];
    uint32_t version, n, num_links;

    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "TDAT", 4) != 0)
        { fclose(fp); return -1; }
    if (fread(&version, 4, 1, fp) != 1 || (version != 1 && version != 2))
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

    fclose(fp);
    return 0;
}

static void topo_data_free(topo_data_t *td)
{
    free(td->lat);
    free(td->flink);
    free(td->eigenvalues);
    free(td->eigvecs);
    td->lat         = NULL;
    td->flink       = NULL;
    td->eigenvalues = NULL;
    td->eigvecs     = NULL;
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
    if (td.num_ev <= 0) {
        if (rank == 0)
            fprintf(stderr,
                "Error: .tdat has no spectral data (need v2 format)\n");
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
 * Spectral broadcast data structures (for run_spec)
 *
 * Loaded from .sdat files produced by spectral_preprocess.py.
 * Contains sparse-Laplacian eigenvectors projected to compute nodes
 * and pairwise latency matrix for bridge selection.
 * ================================================================ */

typedef struct {
    int      N;           /* number of compute nodes                */
    double   bw_min;      /* min bottleneck bandwidth (bytes/sec)   */
    int      num_ev;      /* number of eigenvectors                 */
    double   *eigenvalues; /* num_ev eigenvalues (lambda_2..k+1)    */
    double   *eigvecs;    /* num_ev * N eigenvector matrix          */
    int      has_lat;     /* 1 if lat matrix is present             */
    float    *lat;        /* N*N pairwise latency matrix (seconds)  */
} spec_data_t;

static int spec_data_load(const char *path, spec_data_t *sd)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    char magic[4];
    uint32_t version, n, nev;

    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "SDAT", 4) != 0)
        { fclose(fp); return -1; }
    if (fread(&version, 4, 1, fp) != 1 || (version != 1 && version != 2))
        { fclose(fp); return -1; }
    if (fread(&n, 4, 1, fp) != 1)
        { fclose(fp); return -1; }
    if (fread(&nev, 4, 1, fp) != 1)
        { fclose(fp); return -1; }
    if (fread(&sd->bw_min, 8, 1, fp) != 1)
        { fclose(fp); return -1; }

    sd->N = (int)n;
    sd->num_ev = (int)nev;

    /* Eigenvalues */
    sd->eigenvalues = malloc(nev * sizeof(double));
    if (fread(sd->eigenvalues, sizeof(double), nev, fp) != nev)
        { free(sd->eigenvalues); fclose(fp); return -1; }

    /* Eigenvectors: num_ev * N doubles, row-major */
    size_t ev_size = (size_t)nev * n;
    sd->eigvecs = malloc(ev_size * sizeof(double));
    if (fread(sd->eigvecs, sizeof(double), ev_size, fp) != ev_size)
        { free(sd->eigenvalues); free(sd->eigvecs);
          fclose(fp); return -1; }

    /* Latency matrix: N*N floats */
    size_t nn = (size_t)n * n;
    sd->lat = malloc(nn * sizeof(float));
    if (fread(sd->lat, sizeof(float), nn, fp) != nn) {
        /* lat_matrix is optional — old files might not have it */
        free(sd->lat);
        sd->lat = NULL;
        sd->has_lat = 0;
    } else {
        sd->has_lat = 1;
    }

    fclose(fp);
    return 0;
}

static void spec_data_free(spec_data_t *sd)
{
    free(sd->eigenvalues);
    free(sd->eigvecs);
    free(sd->lat);
    sd->eigenvalues = NULL;
    sd->eigvecs     = NULL;
    sd->lat         = NULL;
}

/* ================================================================
 * Neighbor-BFS Spanning Tree with Fiedler-Ordered Children
 *
 * BFS over physical neighbors — full physical fanout, no restriction.
 * Every unvisited neighbor becomes a child.
 *
 * Fiedler guidance: children are ordered by |Fiedler(child) - Fiedler(u)|
 * descending (most spectrally distant first).  Since Isend is sequential,
 * the first child starts receiving earliest.  Sending to the spectrally
 * farthest neighbor first ensures distant partitions begin propagating
 * independently while closer partitions are still being served.
 *
 * Result:
 *   - Every edge is a 1-hop physical link
 *   - Fanout = physical degree (natural for the topology)
 *   - Depth = BFS depth of the physical graph
 *   - Fiedler ordering maximises parallel diffusion
 * ================================================================ */
static void build_spec_bfs_tree(const spec_data_t *sd, int root,
                                int size, int *parent, int *child_order)
{
    for (int i = 0; i < size; i++) { parent[i] = -1; child_order[i] = 0; }

    char *visited = calloc(size, 1);
    int  *queue   = malloc(size * sizeof(int));
    int   qh = 0, qt = 0;

    visited[root] = 1;
    queue[qt++] = root;

    const double *fiedler = sd->eigvecs;

    while (qh < qt) {
        int u = queue[qh++];
        double fu = fiedler[u];

        if (!sd->has_lat) continue;

        /* Collect ALL unvisited physical neighbors */
        int  nbrs[256];
        double diffs[256];
        int nn = 0;

        for (int v = 0; v < size; v++) {
            if (visited[v]) continue;
            if (sd->lat[(size_t)u * size + v] > 0.0f) {
                nbrs[nn] = v;
                diffs[nn] = fabs(fiedler[v] - fu);
                nn++;
            }
        }

        /* Sort by Fiedler difference descending (selection sort) —
         * most spectrally distant neighbor first */
        for (int i = 0; i < nn; i++) {
            int best = i;
            for (int j = i + 1; j < nn; j++)
                if (diffs[j] > diffs[best]) best = j;
            if (best != i) {
                int tn = nbrs[i]; nbrs[i] = nbrs[best]; nbrs[best] = tn;
                double td = diffs[i]; diffs[i] = diffs[best]; diffs[best] = td;
            }
        }

        /* Adopt all neighbors as children in Fiedler-priority order */
        for (int i = 0; i < nn; i++) {
            int child = nbrs[i];
            parent[child] = u;
            child_order[child] = i;   /* send priority */
            visited[child] = 1;
            queue[qt++] = child;
        }
    }

    free(queue);
    free(visited);
}

/* ================================================================
 * Algorithm 6: Spectral broadcast (run_spec)
 *
 * Two-phase design that fully isolates file I/O from timing:
 *   Phase 1 — spec_setup(): rank 0 loads .sdat, builds tree,
 *             MPI_Bcast parent[] to all ranks.  Called ONCE before
 *             the root loop.
 *   Phase 2 — run_spec():   pure broadcast, no file I/O.
 *             Only Barrier + Recv/Isend/Waitall is timed.
 * ================================================================ */

/* Pre-loaded spec state, populated by spec_setup() */
static spec_data_t _spec_sd;
static int         _spec_loaded = 0;

/* Load .sdat once (rank 0 only), broadcast to all ranks.
 * Returns 0 on success, -1 on error. */
static int spec_setup(const char *sdat_file, int rank, int size)
{
    int ok = 0;

    if (rank == 0) {
        if (spec_data_load(sdat_file, &_spec_sd) != 0) {
            fprintf(stderr, "Error: cannot load spec data: %s\n",
                    sdat_file);
            ok = -1;
        } else if (_spec_sd.N != size) {
            fprintf(stderr,
                "Error: .sdat has %d nodes but MPI size is %d\n",
                _spec_sd.N, size);
            spec_data_free(&_spec_sd);
            ok = -1;
        } else if (_spec_sd.num_ev <= 0) {
            fprintf(stderr, "Error: .sdat has no spectral data\n");
            spec_data_free(&_spec_sd);
            ok = -1;
        }
    }

    /* Broadcast success/failure to all ranks */
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (ok != 0) return -1;

    /* Broadcast the spec data fields from rank 0 to all ranks */
    if (rank != 0) {
        _spec_sd.N = size;
        _spec_sd.num_ev = 0;
        _spec_sd.eigenvalues = NULL;
        _spec_sd.eigvecs = NULL;
        _spec_sd.lat = NULL;
        _spec_sd.has_lat = 0;
    }

    MPI_Bcast(&_spec_sd.bw_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_spec_sd.num_ev, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_spec_sd.has_lat, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int nev = _spec_sd.num_ev;
    if (rank != 0) {
        _spec_sd.eigenvalues = malloc(nev * sizeof(double));
        _spec_sd.eigvecs = malloc((size_t)nev * size * sizeof(double));
    }
    MPI_Bcast(_spec_sd.eigenvalues, nev, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(_spec_sd.eigvecs, nev * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (_spec_sd.has_lat) {
        size_t nn = (size_t)size * size;
        if (rank != 0)
            _spec_sd.lat = malloc(nn * sizeof(float));
        MPI_Bcast(_spec_sd.lat, (int)nn, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    _spec_loaded = 1;
    return 0;
}

static void spec_cleanup(void)
{
    if (_spec_loaded) {
        spec_data_free(&_spec_sd);
        _spec_loaded = 0;
    }
}

/* Pure broadcast — no file I/O.  Tree is built from pre-loaded
 * _spec_sd, then only the Recv/Isend/Waitall is timed.
 *
 * Pipelined: message is split into nchunks pieces.  For each chunk,
 * blocking Recv from parent, then Isend to ALL children (Fiedler-
 * priority order).  Chunks flow through the tree like a pipeline.
 * With 1-hop edges and high fanout, this gives excellent overlap. */
static double run_spec(void *buf, int count, int rank, int size,
                       int root, int nchunks, int *out_nchunks)
{
    int *parent = malloc(size * sizeof(int));
    int *child_order = malloc(size * sizeof(int));
    build_spec_bfs_tree(&_spec_sd, root, size, parent, child_order);

    *out_nchunks = nchunks;

    /* Determine children, sorted by child_order (Fiedler priority) */
    int nchildren = 0;
    int *children = malloc(size * sizeof(int));
    int *order    = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        if (parent[i] == rank) {
            children[nchildren] = i;
            order[nchildren] = child_order[i];
            nchildren++;
        }
    }
    /* Sort children by order (ascending = highest Fiedler diff first) */
    for (int i = 0; i < nchildren; i++) {
        int best = i;
        for (int j = i + 1; j < nchildren; j++)
            if (order[j] < order[best]) best = j;
        if (best != i) {
            int tc = children[i]; children[i] = children[best]; children[best] = tc;
            int to = order[i]; order[i] = order[best]; order[best] = to;
        }
    }

    int chunk_size = (count + nchunks - 1) / nchunks;

    /* Allocate send requests: nchildren per chunk */
    MPI_Request *reqs = (nchildren > 0)
        ? malloc((size_t)nchildren * nchunks * sizeof(MPI_Request)) : NULL;
    int nreqs = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int c = 0; c < nchunks; c++) {
        int off = c * chunk_size;
        int len = chunk_size;
        if (off + len > count) len = count - off;

        /* Blocking Recv from parent — paces the pipeline */
        if (rank != root)
            MPI_Recv((char *)buf + off, len, MPI_BYTE, parent[rank], c,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Non-blocking send to all children (Fiedler order) */
        for (int ch = 0; ch < nchildren; ch++)
            MPI_Isend((char *)buf + off, len, MPI_BYTE, children[ch], c,
                      MPI_COMM_WORLD, &reqs[nreqs++]);
    }

    if (nreqs > 0)
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    double t1 = MPI_Wtime();

    free(reqs);
    free(parent);
    free(child_order);
    free(children);
    free(order);
    return t1 - t0;
}

/* ================================================================
 * Algorithm 5: BBS — (stub, not yet implemented)
 * ================================================================ */

/* ================================================================
 * Algorithm 9: Proto — MWU Arborescence Packing
 *
 * Multiplicative Weights Update (Plotkin-Shmoys-Tardos style)
 * fractional packing of spanning out-arborescences.
 * Oracle: Chu-Liu/Edmonds minimum-cost arborescence.
 *
 * Input: directed 1-hop physical adjacency from .tdat latency data,
 *        with unit edge capacities.
 *
 * Output: selects the tree with lowest estimated pipeline broadcast
 *         time from the packing, then broadcasts on that tree.
 *
 * Ref: idea.py prototype; Edmonds, "Optimum Branchings", 1967;
 *      Plotkin/Shmoys/Tardos, MOR 1995.
 * ================================================================ */

/* ---- Chu-Liu/Edmonds minimum-cost spanning arborescence ----
 *
 * Given a directed graph with n nodes (0..n-1), m edges, and a root,
 * finds the minimum-cost spanning out-arborescence rooted at `root`.
 *
 * Returns number of arborescence edges (n-1) on success, -1 on failure.
 * Fills result[] with edge indices into the caller's edge arrays.
 *
 * Algorithm:
 *   1. For each non-root node, select min-cost incoming edge.
 *   2. If selected edges form no cycle → done (this is the MSA).
 *   3. Contract all cycles into super-nodes, adjust incoming edge
 *      costs by subtracting the cycle edge cost, recurse.
 *   4. Expand: for each cycle, include all cycle edges except the
 *      one replaced by the entering edge from the outer arborescence.
 */
typedef struct {
    int src, dst;
    double cost;
    int orig;   /* original edge index (preserved through contractions) */
} _cle_edge_t;

static int _cle_solve(int n, int m, const _cle_edge_t *E, int root,
                       int *result)
{
    if (n <= 1) return 0;

    /* Step 1: min incoming edge per non-root node */
    int *min_e = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) min_e[i] = -1;

    for (int e = 0; e < m; e++) {
        int v = E[e].dst;
        if (v == root || E[e].src == E[e].dst) continue;
        if (min_e[v] == -1 || E[e].cost < E[min_e[v]].cost)
            min_e[v] = e;
    }
    for (int v = 0; v < n; v++) {
        if (v != root && min_e[v] == -1) {
            free(min_e);
            return -1;   /* unreachable node → no arborescence */
        }
    }

    /* Step 2: detect cycles among selected min-in edges */
    int *cyc = malloc(n * sizeof(int));
    int *vis = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) { cyc[i] = -1; vis[i] = -1; }

    int nc = 0;
    for (int s = 0; s < n; s++) {
        if (s == root) continue;
        int u = s;
        while (u != root && vis[u] == -1 && cyc[u] == -1) {
            vis[u] = s;
            u = E[min_e[u]].src;
        }
        if (u != root && cyc[u] == -1 && vis[u] == s) {
            int v = u;
            do { cyc[v] = nc; v = E[min_e[v]].src; } while (v != u);
            nc++;
        }
    }

    if (nc == 0) {
        /* No cycles: min_e edges form the arborescence directly */
        int cnt = 0;
        for (int v = 0; v < n; v++)
            if (v != root) result[cnt++] = E[min_e[v]].orig;
        free(min_e); free(cyc); free(vis);
        return cnt;
    }

    /* Step 3: contract all cycles into super-nodes */
    int *nmap = malloc(n * sizeof(int));
    int nn = nc;  /* first nc IDs are super-nodes for cycles */
    for (int i = 0; i < n; i++)
        nmap[i] = (cyc[i] >= 0) ? cyc[i] : nn++;

    double *min_cost = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        min_cost[i] = (i != root && min_e[i] >= 0)
                     ? E[min_e[i]].cost : 0.0;

    /* Build contracted edge list:
     *   - Remap endpoints through nmap
     *   - Remove self-loops (both endpoints in same cycle)
     *   - For edges entering a cycle node v: cost -= min_cost[v] */
    _cle_edge_t *NE = malloc(m * sizeof(_cle_edge_t));
    int *enters = malloc(m * sizeof(int));
    int nm = 0;

    for (int e = 0; e < m; e++) {
        int u = nmap[E[e].src], v = nmap[E[e].dst];
        if (u == v) continue;
        double c = E[e].cost;
        if (cyc[E[e].dst] >= 0) c -= min_cost[E[e].dst];
        NE[nm] = (_cle_edge_t){ u, v, c, E[e].orig };
        enters[nm] = E[e].dst;   /* original node this edge enters */
        nm++;
    }

    int new_root = nmap[root];

    /* Recurse on contracted graph */
    int *sub = malloc((nn - 1) * sizeof(int));
    int scnt = _cle_solve(nn, nm, NE, new_root, sub);
    if (scnt < 0) {
        free(min_e); free(cyc); free(vis); free(nmap);
        free(min_cost); free(NE); free(enters); free(sub);
        return -1;
    }

    /* Step 4: expand — find which cycle node is the entry point */
    int *entered = malloc(nc * sizeof(int));
    for (int i = 0; i < nc; i++) entered[i] = -1;

    for (int i = 0; i < scnt; i++) {
        int oi = sub[i];
        for (int e = 0; e < nm; e++) {
            if (NE[e].orig == oi && NE[e].dst < nc) {
                entered[NE[e].dst] = enters[e];
                break;
            }
        }
    }

    /* Collect result:
     *   - All edges from the contracted arborescence (sub[])
     *   - All cycle min-in edges EXCEPT the entry-point node */
    int cnt = 0;
    for (int i = 0; i < scnt; i++)
        result[cnt++] = sub[i];
    for (int v = 0; v < n; v++) {
        if (cyc[v] < 0 || v == root) continue;
        if (entered[cyc[v]] == v) continue;  /* replaced by entering edge */
        result[cnt++] = E[min_e[v]].orig;
    }

    free(min_e); free(cyc); free(vis); free(nmap);
    free(min_cost); free(NE); free(enters); free(sub); free(entered);
    return cnt;
}

/* Public wrapper: find min-cost arborescence rooted at `root`.
 * src[], dst[], cost[] are parallel arrays of m edges on n nodes.
 * Returns n-1 on success, -1 on failure.
 * Fills result_edges[] with indices into src/dst/cost. */
static int edmonds_msa(int n, int m, const int *src, const int *dst,
                        const double *cost, int root, int *result_edges)
{
    _cle_edge_t *E = malloc(m * sizeof(_cle_edge_t));
    for (int i = 0; i < m; i++)
        E[i] = (_cle_edge_t){ src[i], dst[i], cost[i], i };
    int r = _cle_solve(n, m, E, root, result_edges);
    free(E);
    return r;
}

/* ---- MWU arborescence packing ----
 *
 * Iteratively:
 *   1. Set oracle costs = price[e] / cap[e]
 *   2. Find min-cost arborescence via Edmonds
 *   3. Push delta = eta * bottleneck_residual along the tree
 *   4. Update prices: price[e] *= (1 + eps)^(delta / cap[e])
 *
 * Returns number of distinct trees packed.
 * Fills pack_parents[k] with parent array for tree k.
 * Fills pack_weights[k] with weight of tree k.
 * Caller frees pack_parents[k] for each k.
 */
#define PROTO_MAX_ITERS 50

static int mwu_arb_packing(
    int n, int m, const int *src, const int *dst,
    const double *cap, int root,
    double eps, double eta, int max_iters,
    int **pack_parents, double *pack_weights)
{
    double *price = malloc(m * sizeof(double));
    double *used  = calloc(m, sizeof(double));
    double *ocost = malloc(m * sizeof(double));
    int *tree_edges = malloc(n * sizeof(int));

    for (int i = 0; i < m; i++) price[i] = 1.0;

    int ntrees = 0;

    for (int t = 0; t < max_iters; t++) {
        /* Set oracle costs */
        for (int e = 0; e < m; e++) {
            double d = cap[e];
            if (d < 1e-12) d = 1e-12;
            ocost[e] = price[e] / d;
        }

        /* Find min-cost arborescence */
        int ne = edmonds_msa(n, m, src, dst, ocost, root, tree_edges);
        if (ne < 0) break;

        /* Bottleneck residual */
        double bottleneck = 1e30;
        for (int i = 0; i < ne; i++) {
            int e = tree_edges[i];
            double resid = cap[e] - used[e];
            if (resid < bottleneck) bottleneck = resid;
        }
        if (bottleneck <= 1e-12) break;

        double delta = eta * bottleneck;

        /* MWU price update */
        for (int i = 0; i < ne; i++) {
            int e = tree_edges[i];
            used[e] += delta;
            double frac = delta / cap[e];
            price[e] *= pow(1.0 + eps, frac);
        }

        /* Convert edge list to parent array */
        int *par = malloc(n * sizeof(int));
        for (int v = 0; v < n; v++) par[v] = -1;
        for (int i = 0; i < ne; i++) {
            int e = tree_edges[i];
            par[dst[e]] = src[e];
        }

        /* Merge duplicate trees */
        int dup = -1;
        for (int k = 0; k < ntrees; k++) {
            int same = 1;
            for (int v = 0; v < n; v++)
                if (pack_parents[k][v] != par[v]) { same = 0; break; }
            if (same) { dup = k; break; }
        }

        if (dup >= 0) {
            pack_weights[dup] += delta;
            free(par);
        } else {
            pack_parents[ntrees] = par;
            pack_weights[ntrees] = delta;
            ntrees++;
        }
    }

    free(price); free(used); free(ocost); free(tree_edges);
    return ntrees;
}

/* ---- run_proto: MWU packing + pipelined broadcast ----
 *
 * Rank 0 loads .tdat, builds directed 1-hop graph with unit
 * capacities, runs MWU packing, selects the tree with lowest
 * estimated pipeline time, broadcasts parent array to all ranks.
 * All ranks then do pipelined tree broadcast.
 */
static double run_proto(void *buf, int count, int rank, int size,
                        int root, const char *tdat_file,
                        int *out_nchunks)
{
    int *parent = malloc(size * sizeof(int));
    int nchunks = 1;

    if (rank == 0) {
        topo_data_t td;
        int loaded = 0;

        if (topo_data_load(tdat_file, &td) != 0) {
            fprintf(stderr, "Proto: cannot load %s\n", tdat_file);
        } else if (td.N != size) {
            fprintf(stderr, "Proto: .tdat has %d nodes, MPI has %d\n",
                    td.N, size);
            topo_data_free(&td);
        } else {
            loaded = 1;
        }

        if (loaded) {
            int N = td.N;

            /* Build directed 1-hop graph */
            float lat_1hop = 1e30f;
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++) {
                    float l = td.lat[(size_t)i * N + j];
                    if (l > 0.0f && l < lat_1hop) lat_1hop = l;
                }
            float thresh = lat_1hop * 1.01f;

            int m = 0;
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++) {
                    float l = td.lat[(size_t)i * N + j];
                    if (l > 0.0f && l <= thresh) m++;
                }

            int    *src = malloc(m * sizeof(int));
            int    *dst = malloc(m * sizeof(int));
            double *cap = malloc(m * sizeof(double));
            int idx = 0;
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++) {
                    float l = td.lat[(size_t)i * N + j];
                    if (l > 0.0f && l <= thresh) {
                        src[idx] = i;
                        dst[idx] = j;
                        cap[idx] = 1.0;
                        idx++;
                    }
                }

            /* MWU packing */
            int **pack_par = malloc(PROTO_MAX_ITERS * sizeof(int *));
            double *pack_w = malloc(PROTO_MAX_ITERS * sizeof(double));

            int ntrees = mwu_arb_packing(
                size, m, src, dst, cap, root,
                0.25, 0.5, PROTO_MAX_ITERS,
                pack_par, pack_w);

            double total_w = 0;
            for (int k = 0; k < ntrees; k++) total_w += pack_w[k];
            fprintf(stderr, "Proto: packed %d trees, value=%.3f\n",
                    ntrees, total_w);

            /* Select tree with lowest estimated pipeline time */
            if (ntrees > 0) {
                double best_T = 1e30;
                int best_k = 0;
                for (int k = 0; k < ntrees; k++) {
                    int K;
                    double T = _estimate_time(&td, size, pack_par[k],
                                               count, &K);
                    if (T < best_T) {
                        best_T = T;
                        best_k = k;
                        nchunks = K;
                    }
                }
                memcpy(parent, pack_par[best_k], size * sizeof(int));
            } else {
                /* Fallback: 1-hop BFS tree */
                build_onehop_bfs(&td, root, size, parent, 0);
                _estimate_time(&td, size, parent, count, &nchunks);
            }

            for (int k = 0; k < ntrees; k++) free(pack_par[k]);
            free(pack_par); free(pack_w);
            free(src); free(dst); free(cap);
            topo_data_free(&td);
        } else {
            /* Error path: set invalid parent to trigger fallback */
            for (int i = 0; i < size; i++) parent[i] = -1;
        }
    }

    /* Broadcast parent array and nchunks to all ranks */
    MPI_Bcast(parent, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nchunks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    *out_nchunks = nchunks;

    /* Determine children */
    int nchildren = 0;
    int *children = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
        if (parent[i] == rank) children[nchildren++] = i;

    /* Pipelined broadcast */
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

        if (rank != root)
            MPI_Recv((char *)buf + off, len, MPI_BYTE,
                     parent[rank], c, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        for (int ch = 0; ch < nchildren; ch++)
            MPI_Isend((char *)buf + off, len, MPI_BYTE,
                      children[ch], c, MPI_COMM_WORLD,
                      &reqs[nsend++]);
    }

    if (nsend > 0)
        MPI_Waitall(nsend, reqs, MPI_STATUSES_IGNORE);

    double t1 = MPI_Wtime();

    free(reqs); free(parent); free(children);
    return t1 - t0;
}

/* ================================================================
 * Algorithm 8: Spectral Binary Tree broadcast (specbin).
 *
 * Global binary tree built via recursive spectral bisection:
 *   1. Sort all non-root nodes by Fiedler value.
 *   2. Split at median into left/right halves.
 *   3. Pick representative of each half = node closest to spectral
 *      centroid (median Fiedler value of that half).
 *   4. Both representatives become children of the current leader.
 *   5. Recurse: each rep leads its half.
 *
 * Depth = ceil(log2(N)) ≈ 7 for N=128 (same as MPI binary tree).
 * Advantage: partitions follow spectral structure of the physical
 * topology, so children are in spectrally coherent sub-networks.
 *
 * Requires pre-loaded _spec_sd (via spec_setup).
 * ================================================================ */

/* Comparator for qsort: sort node indices by Fiedler value */
static const double *_specbin_fiedler_ptr;   /* set before qsort */

static int _specbin_cmp(const void *a, const void *b)
{
    double fa = _specbin_fiedler_ptr[*(const int *)a];
    double fb = _specbin_fiedler_ptr[*(const int *)b];
    if (fa < fb) return -1;
    if (fa > fb) return  1;
    return 0;
}

/* Pick the node in nodes[0..n-1] whose Fiedler value is closest
 * to the target value.  Returns its INDEX in the nodes array. */
static int _specbin_pick_centroid(const int *nodes, int n,
                                  const double *fiedler, double target)
{
    int best = 0;
    double best_dist = fabs(fiedler[nodes[0]] - target);
    for (int i = 1; i < n; i++) {
        double d = fabs(fiedler[nodes[i]] - target);
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

/* Recursive bisection: assign parent for all nodes in subset. */
static void _specbin_recurse(const double *fiedler,
                             int *nodes, int n,
                             int leader, int *parent)
{
    if (n == 0) return;
    if (n == 1) { parent[nodes[0]] = leader; return; }
    if (n == 2) {
        parent[nodes[0]] = leader;
        parent[nodes[1]] = leader;
        return;
    }

    /* Sort by Fiedler value */
    _specbin_fiedler_ptr = fiedler;
    qsort(nodes, n, sizeof(int), _specbin_cmp);

    int mid = n / 2;
    /* left = nodes[0..mid-1], right = nodes[mid..n-1] */

    /* Compute median Fiedler for each half */
    double left_median  = fiedler[nodes[mid / 2]];
    double right_median = fiedler[nodes[mid + (n - mid) / 2]];

    /* Pick centroids */
    int li = _specbin_pick_centroid(nodes, mid, fiedler, left_median);
    int ri_rel = _specbin_pick_centroid(nodes + mid, n - mid,
                                        fiedler, right_median);
    int ri = mid + ri_rel;

    int left_rep  = nodes[li];
    int right_rep = nodes[ri];

    parent[left_rep]  = leader;
    parent[right_rep] = leader;

    /* Build left subset excluding left_rep */
    int *left = malloc(mid * sizeof(int));
    int ln = 0;
    for (int i = 0; i < mid; i++)
        if (nodes[i] != left_rep) left[ln++] = nodes[i];

    /* Build right subset excluding right_rep */
    int *right = malloc((n - mid) * sizeof(int));
    int rn = 0;
    for (int i = mid; i < n; i++)
        if (nodes[i] != right_rep) right[rn++] = nodes[i];

    _specbin_recurse(fiedler, left,  ln, left_rep,  parent);
    _specbin_recurse(fiedler, right, rn, right_rep, parent);

    free(left);
    free(right);
}

static void build_specbin_tree(const double *fiedler, int root,
                               int size, int *parent)
{
    for (int i = 0; i < size; i++) parent[i] = -1;

    int *subset = malloc((size - 1) * sizeof(int));
    int n = 0;
    for (int i = 0; i < size; i++)
        if (i != root) subset[n++] = i;

    _specbin_recurse(fiedler, subset, n, root, parent);
    free(subset);
}

static double run_specbin(void *buf, int count, int rank, int size,
                          int root, int *out_nchunks)
{
    int *parent = malloc(size * sizeof(int));
    build_specbin_tree(_spec_sd.eigvecs, root, size, parent);

    *out_nchunks = 1;

    /* Collect children */
    int nchildren = 0;
    int children[2];
    for (int i = 0; i < size; i++) {
        if (parent[i] == rank) {
            children[nchildren++] = i;
            if (nchildren == 2) break;   /* binary tree: max 2 */
        }
    }

    MPI_Request reqs[2];

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (rank != root)
        MPI_Recv(buf, count, MPI_BYTE, parent[rank], 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int ch = 0; ch < nchildren; ch++)
        MPI_Isend(buf, count, MPI_BYTE, children[ch], 0,
                  MPI_COMM_WORLD, &reqs[ch]);

    if (nchildren > 0)
        MPI_Waitall(nchildren, reqs, MPI_STATUSES_IGNORE);

    double t1 = MPI_Wtime();
    free(parent);
    return t1 - t0;
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
                "Usage: %s <algorithm> <msg_bytes> [nchunks] [root] [out_json] [topo_cfg]\n"
                "  algorithm  : mpi | srda | pipe | bine | glf | test | spec | proto\n"
                "  nchunks    : pipeline depth (default 64, auto for test)\n"
                "  root       : broadcast root 0..%d or 'all' (default 0)\n"
                "  out_json   : output JSON file path (optional, _ = none)\n"
                "  topo_cfg   : topology config file (required for 'test')\n",
                argv[0], size - 1);
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

    /* root = "all" → sweep every root in a single invocation */
    int all_roots = 0;
    int root = 0;
    if (argc > 4) {
        if (strcmp(argv[4], "all") == 0) {
            all_roots = 1;
        } else {
            root = atoi(argv[4]);
        }
    }

    if (nbytes <= 0) {
        if (rank == 0) fprintf(stderr, "msg_bytes must be > 0\n");
        MPI_Finalize();
        return 1;
    }
    if (!all_roots && (root < 0 || root >= size)) {
        if (rank == 0)
            fprintf(stderr, "root must be in 0..%d or 'all' (got %d)\n",
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
    else if (strcmp(algo, "spec") == 0) algo_id = 6;
    else if (strcmp(algo, "rawbino") == 0) algo_id = 7;
    else if (strcmp(algo, "specbin") == 0) algo_id = 8;
    else if (strcmp(algo, "proto")  == 0) algo_id = 9;
    else {
        if (rank == 0) fprintf(stderr, "Unknown algorithm: %s\n", algo);
        MPI_Finalize();
        return 1;
    }
    if ((algo_id == 3 || algo_id == 5 || algo_id == 6 || algo_id == 8 || algo_id == 9) && !topo_file) {
        if (rank == 0)
            fprintf(stderr, "%s algorithm requires topo_cfg argument\n", algo);
        MPI_Finalize();
        return 1;
    }

    /* pipe (2) and spec (6) use nchunks from the command line.
     * test (3) and specbin (8) compute their own nchunks internally.
     * For all others, report nchunks=1 (whole message). */
    int reported_nchunks = nchunks;
    if (algo_id != 2 && algo_id != 3 && algo_id != 6 && algo_id != 8
        && algo_id != 9)
        reported_nchunks = 1;

    /* ---- Pre-load spec data (once, before root loop) ---- */
    if (algo_id == 6 || algo_id == 8) {
        if (spec_setup(topo_file, rank, size) != 0) {
            MPI_Finalize();
            return 1;
        }
    }

    /* ---- Root loop ---- */
    int root_lo = all_roots ? 0 : root;
    int root_hi = all_roots ? size - 1 : root;
    int nroots  = root_hi - root_lo + 1;

    double *times = calloc(nroots, sizeof(double));
    int    *oks   = calloc(nroots, sizeof(int));
    char   *buf   = calloc(nbytes, 1);

    for (int r = root_lo; r <= root_hi; r++) {
        /* Reset buffer; root fills with known pattern */
        memset(buf, 0, nbytes);
        if (rank == r)
            for (int i = 0; i < nbytes; i++)
                buf[i] = (char)(i & 0xFF);

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
        case 6: elapsed = run_spec(buf, nbytes, rank, size, r,
                                   nchunks, &reported_nchunks);             break;
        case 7: elapsed = run_rawbino(buf, nbytes, rank, size, r);           break;
        case 8: elapsed = run_specbin(buf, nbytes, rank, size, r,
                                      &nchunks);                              break;
        case 9: elapsed = run_proto(buf, nbytes, rank, size, r, topo_file,
                                    &nchunks);
                reported_nchunks = nchunks;                                 break;
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

    /* ---- Cleanup pre-loaded data ---- */
    if (algo_id == 6 || algo_id == 8) spec_cleanup();

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
            printf("nodes     : %d\n", size);
            printf("msg_bytes : %d\n", nbytes);
            printf("roots     : all (0..%d)\n", size - 1);
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
                        algo, size, nbytes, reported_nchunks,
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
        } else {
            printf("algorithm : %s\n", algo);
            printf("nodes     : %d\n", size);
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
                        algo, size, nbytes, reported_nchunks, root,
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

    free(times); free(oks); free(buf);
    MPI_Finalize();
    return 0;
}
