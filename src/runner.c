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
 *   "bbs"  — Frame-based broadcast from a plan file.
 *             Reads frames from ../plans/<Topo>/<N>_root<R>.plan
 *             or from a path passed as 5th argument.
 *             Falls back gracefully if file not found.
 *
 * Usage:
 *   smpirun -np N -platform <xml> -hostfile <hf> \
 *           ./runner <algo> <bytes> [chunks] [root] [plan_file] [out_json]
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
 * Bitmap helpers  (reused from bcast_2dmesh.c)
 * ================================================================ */
#define BM_BYTES(nc)    (((nc) + 7) / 8)
#define BIT_SET(bm, c)  ((bm)[(c)/8] |=  (unsigned char)(1u << ((c)%8)))
#define BIT_GET(bm, c)  (((bm)[(c)/8] >> ((c)%8)) & 1u)

static int bm_first_missing(const unsigned char *src, const unsigned char *dst,
                            int nchunks)
{
    for (int c = 0; c < nchunks; c++)
        if (BIT_GET(src, c) && !BIT_GET(dst, c)) return c;
    return -1;
}

static int bm_full(const unsigned char *bm, int nchunks)
{
    for (int c = 0; c < nchunks; c++)
        if (!BIT_GET(bm, c)) return 0;
    return 1;
}

/* ================================================================
 * Schedule data structure  (reused from bcast_2dmesh.c)
 * ================================================================ */
typedef struct {
    int send_to;
    int send_chunk;
    int recv_from;
    int recv_chunk;
} sched_t;

/* ================================================================
 * Schedule executor  (topology-agnostic version)
 *
 * Replays a round-by-round schedule with MPI_Isend / MPI_Irecv.
 * `size` is the MPI world size (runtime, not compile-time).
 * ================================================================ */
static double exec_schedule(void *buf, int count, int rank, int size,
                            int nchunks, int total_rounds,
                            const sched_t *sched)
{
    const int chunk_size = (count + nchunks - 1) / nchunks;
    char *recv_buf = malloc(chunk_size);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int r = 0; r < total_rounds; r++) {
        const sched_t *s = &sched[r * size + rank];
        MPI_Request reqs[2];
        int nreqs = 0;

        if (s->send_to >= 0) {
            int off = s->send_chunk * chunk_size;
            int len = chunk_size;
            if (off + len > count) len = count - off;
            MPI_Isend((char *)buf + off, len, MPI_BYTE,
                      s->send_to, s->send_chunk, MPI_COMM_WORLD,
                      &reqs[nreqs++]);
        }
        if (s->recv_from >= 0) {
            MPI_Irecv(recv_buf, chunk_size, MPI_BYTE,
                      s->recv_from, s->recv_chunk, MPI_COMM_WORLD,
                      &reqs[nreqs++]);
        }

        if (nreqs > 0)
            MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        if (s->recv_from >= 0 && s->recv_chunk >= 0) {
            int off = s->recv_chunk * chunk_size;
            int len = chunk_size;
            if (off + len > count) len = count - off;
            memcpy((char *)buf + off, recv_buf, len);
        }
    }

    double t1 = MPI_Wtime();
    free(recv_buf);
    return t1 - t0;
}

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
 * Algorithm 4: BBS — Frame-based broadcast from plan file
 *
 * Plan file format:
 *   # BBS frames for <topology> N=<nodes> root=<root>
 *   # nframes: <count>
 *   FRAME 0
 *   0 1
 *   2 3
 *   4 5
 *
 *   FRAME 1
 *   1 2
 *   ...
 *
 * Each "src dst" line is a directed edge in that frame's matching.
 * The planner cycles through frames round-robin, assigning the
 * smallest chunk that src has but dst lacks.
 * ================================================================ */

#define MAX_FRAMES      256
#define MAX_EDGES_FRAME 2048

typedef struct {
    int src;
    int dst;
} frame_edge_t;

static int g_num_frames = 0;
static int g_frame_sizes[MAX_FRAMES];
static frame_edge_t (*g_frames)[MAX_EDGES_FRAME] = NULL;

/* --- Weighted (adaptive) mode --- */
#define MAX_WEDGES 8192

typedef struct {
    int    src;
    int    dst;
    double weight;
} wedge_t;

static wedge_t g_wedges[MAX_WEDGES];
static int     g_num_wedges   = 0;
static int     g_weighted_mode = 0;

/* --- CSR adjacency list (built from weighted edges for Kuhn's matching) --- */
#define MAX_NODES 2048
static int g_adj_dst[MAX_WEDGES];       /* destination nodes */
static int g_adj_off[MAX_NODES + 1];    /* CSR offsets: node u's neighbors at [off[u], off[u+1]) */
static int g_adj_nnodes = 0;            /* number of nodes in the graph */

static int load_plan_file(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;

    g_num_frames   = 0;
    g_num_wedges   = 0;
    g_weighted_mode = 0;

    /* Free previous frame data if any */
    if (g_frames) { free(g_frames); g_frames = NULL; }

    int cur_frame = -1;
    int in_weighted = 0;
    char line[256];

    while (fgets(line, sizeof(line), fp)) {
        /* Skip comments and blank lines */
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;

        /* Detect WEIGHTED keyword */
        if (strncmp(line, "WEIGHTED", 8) == 0) {
            in_weighted = 1;
            g_weighted_mode = 1;
            continue;
        }

        /* Weighted mode: parse "src dst weight" */
        if (in_weighted) {
            int src, dst;
            double weight;
            if (sscanf(line, "%d %d %lf", &src, &dst, &weight) == 3) {
                if (g_num_wedges < MAX_WEDGES) {
                    g_wedges[g_num_wedges].src    = src;
                    g_wedges[g_num_wedges].dst    = dst;
                    g_wedges[g_num_wedges].weight = weight;
                    g_num_wedges++;
                }
            }
            continue;
        }

        /* Frame mode: existing parsing */
        int fnum;
        if (sscanf(line, "FRAME %d", &fnum) == 1) {
            if (!g_frames)
                g_frames = calloc(MAX_FRAMES, sizeof(*g_frames));
            cur_frame = fnum;
            if (cur_frame >= MAX_FRAMES) {
                fclose(fp);
                return -1;
            }
            if (cur_frame >= g_num_frames)
                g_num_frames = cur_frame + 1;
            g_frame_sizes[cur_frame] = 0;
            continue;
        }

        int src, dst;
        if (cur_frame >= 0 && sscanf(line, "%d %d", &src, &dst) == 2) {
            int idx = g_frame_sizes[cur_frame];
            if (idx >= MAX_EDGES_FRAME) continue;
            g_frames[cur_frame][idx].src = src;
            g_frames[cur_frame][idx].dst = dst;
            g_frame_sizes[cur_frame]++;
        }
    }

    fclose(fp);

    if (g_weighted_mode) {
        if (g_num_wedges == 0) return -1;

        /* Build CSR adjacency list from weighted edges.
         * Count outgoing edges per node, then prefix-sum. */
        int max_node = 0;
        for (int i = 0; i < g_num_wedges; i++) {
            if (g_wedges[i].src > max_node) max_node = g_wedges[i].src;
            if (g_wedges[i].dst > max_node) max_node = g_wedges[i].dst;
        }
        g_adj_nnodes = max_node + 1;

        memset(g_adj_off, 0, (g_adj_nnodes + 1) * sizeof(int));

        /* Count outgoing edges per source */
        for (int i = 0; i < g_num_wedges; i++)
            g_adj_off[g_wedges[i].src + 1]++;

        /* Prefix sum */
        for (int i = 1; i <= g_adj_nnodes; i++)
            g_adj_off[i] += g_adj_off[i - 1];

        /* Fill destination array */
        int *tmp_off = malloc(g_adj_nnodes * sizeof(int));
        memcpy(tmp_off, g_adj_off, g_adj_nnodes * sizeof(int));
        for (int i = 0; i < g_num_wedges; i++) {
            int s = g_wedges[i].src;
            g_adj_dst[tmp_off[s]++] = g_wedges[i].dst;
        }
        free(tmp_off);

        return 1;
    }
    return g_num_frames;
}

static int plan_bbs(int nchunks, int root, int size, sched_t **sched_out)
{
    if (g_num_frames == 0) {
        *sched_out = NULL;
        return -1;
    }

    const int bm_sz = BM_BYTES(nchunks);

    unsigned char *bm = calloc(size * bm_sz, 1);
    for (int c = 0; c < nchunks; c++)
        BIT_SET(&bm[root * bm_sz], c);

    int cap    = 256;
    int rounds = 0;
    sched_t *sched = malloc(cap * size * sizeof(sched_t));

    for (;;) {
        int all_done = 1;
        for (int n = 0; n < size; n++) {
            if (!bm_full(&bm[n * bm_sz], nchunks)) { all_done = 0; break; }
        }
        if (all_done) break;

        if (rounds >= cap) {
            cap *= 2;
            sched = realloc(sched, cap * size * sizeof(sched_t));
        }

        sched_t *row = &sched[rounds * size];
        for (int n = 0; n < size; n++) {
            row[n].send_to = row[n].recv_from = -1;
            row[n].send_chunk = row[n].recv_chunk = -1;
        }

        int fidx = rounds % g_num_frames;

        for (int e = 0; e < g_frame_sizes[fidx]; e++) {
            int s = g_frames[fidx][e].src;
            int d = g_frames[fidx][e].dst;

            if (s >= size || d >= size) continue;

            int chunk = bm_first_missing(&bm[s * bm_sz],
                                         &bm[d * bm_sz], nchunks);
            if (chunk < 0) continue;

            row[s].send_to    = d;
            row[s].send_chunk = chunk;
            row[d].recv_from  = s;
            row[d].recv_chunk = chunk;
        }

        /* Update bitmaps AFTER all edges are processed so that a node
         * cannot forward data it receives in the same round (matches
         * the actual MPI_Isend/Irecv + Waitall execution model). */
        for (int n = 0; n < size; n++) {
            if (row[n].recv_chunk >= 0)
                BIT_SET(&bm[n * bm_sz], row[n].recv_chunk);
        }

        rounds++;
    }

    free(bm);
    *sched_out = sched;
    return rounds;
}

/* ================================================================
 * Kuhn's augmenting-path algorithm for maximum bipartite matching
 *
 * Left set: senders (nodes with useful data for some neighbor).
 * Right set: receivers (all neighbors).
 * Edge (u,v) exists iff u has a chunk that v lacks.
 *
 * Per round: builds a maximum matching, achieving C*=1 throughput.
 * Complexity: O(V × E_active) per round — negligible for N≤2048.
 * ================================================================ */
static int    kuhn_match_recv[MAX_NODES];   /* recv v -> matched sender (-1 = free) */
static int    kuhn_visited[MAX_NODES];      /* DFS visited flags per search */
static unsigned char *kuhn_bm = NULL;       /* pointer to current bitmap state */
static int    kuhn_bm_sz  = 0;             /* bytes per node in bitmap */
static int    kuhn_nchunks = 0;            /* total chunks */

static int try_kuhn(int u)
{
    for (int idx = g_adj_off[u]; idx < g_adj_off[u + 1]; idx++) {
        int v = g_adj_dst[idx];
        if (kuhn_visited[v]) continue;
        /* Check if u has any chunk that v lacks */
        if (bm_first_missing(&kuhn_bm[u * kuhn_bm_sz],
                             &kuhn_bm[v * kuhn_bm_sz], kuhn_nchunks) < 0)
            continue;
        kuhn_visited[v] = 1;
        if (kuhn_match_recv[v] < 0 || try_kuhn(kuhn_match_recv[v])) {
            kuhn_match_recv[v] = u;
            return 1;
        }
    }
    return 0;
}

/* ================================================================
 * Algorithm 4b: Adaptive weighted BBS — maximum matching per round
 *
 * Uses Kuhn's augmenting-path algorithm to find a maximum bipartite
 * matching each round.  Achieves C*=1 throughput (each non-root
 * receives 1 chunk/round in steady state).  Expected rounds: K + D.
 * ================================================================ */
static int plan_bbs_adaptive(int nchunks, int root, int size,
                              sched_t **sched_out)
{
    if (g_num_wedges == 0) {
        *sched_out = NULL;
        return -1;
    }

    const int bm_sz = BM_BYTES(nchunks);

    unsigned char *bm = calloc(size * bm_sz, 1);
    for (int c = 0; c < nchunks; c++)
        BIT_SET(&bm[root * bm_sz], c);

    /* Set up Kuhn state pointers */
    kuhn_bm      = bm;
    kuhn_bm_sz   = bm_sz;
    kuhn_nchunks = nchunks;

    int cap    = 256;
    int rounds = 0;
    sched_t *sched = malloc(cap * size * sizeof(sched_t));

    for (;;) {
        int all_done = 1;
        for (int n = 0; n < size; n++) {
            if (!bm_full(&bm[n * bm_sz], nchunks)) { all_done = 0; break; }
        }
        if (all_done) break;

        if (rounds >= cap) {
            cap *= 2;
            sched = realloc(sched, cap * size * sizeof(sched_t));
        }

        sched_t *row = &sched[rounds * size];
        for (int n = 0; n < size; n++) {
            row[n].send_to = row[n].recv_from = -1;
            row[n].send_chunk = row[n].recv_chunk = -1;
        }

        /* Maximum bipartite matching via Kuhn's augmenting paths */
        memset(kuhn_match_recv, -1, size * sizeof(int));

        for (int u = 0; u < size; u++) {
            /* Skip if u has no outgoing edges or no useful data */
            if (g_adj_off[u] >= g_adj_off[u + 1]) continue;
            memset(kuhn_visited, 0, size * sizeof(int));
            try_kuhn(u);
        }

        /* Convert matching to schedule: for each matched receiver v,
         * find the actual chunk to transfer from sender u */
        for (int v = 0; v < size; v++) {
            int u = kuhn_match_recv[v];
            if (u < 0) continue;
            int chunk = bm_first_missing(&bm[u * bm_sz],
                                         &bm[v * bm_sz], nchunks);
            if (chunk < 0) continue;  /* shouldn't happen */
            row[u].send_to    = v;
            row[u].send_chunk = chunk;
            row[v].recv_from  = u;
            row[v].recv_chunk = chunk;
        }

        /* Update bitmaps AFTER all edges (no same-round forwarding) */
        int progress = 0;
        for (int n = 0; n < size; n++) {
            if (row[n].recv_chunk >= 0) {
                BIT_SET(&bm[n * bm_sz], row[n].recv_chunk);
                progress = 1;
            }
        }

        rounds++;

        /* Safety: if no transfers happened, the graph cannot reach all
         * nodes from root — break to avoid infinite loop. */
        if (!progress) {
            free(bm);
            kuhn_bm = NULL;
            free(sched);
            *sched_out = NULL;
            return -1;
        }
    }

    free(bm);
    kuhn_bm = NULL;
    *sched_out = sched;
    return rounds;
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
                "Usage: %s <algorithm> <msg_bytes> [nchunks] [root] [plan_file] [out_json]\n"
                "  algorithm : mpi | srda | pipe | bbs\n"
                "  nchunks   : pipeline depth (default 64)\n"
                "  root      : broadcast root node 0..%d (default 0)\n"
                "  plan_file : BBS plan file path (optional, _ = none)\n"
                "  out_json  : output JSON file path (optional, _ = none)\n",
                argv[0], size - 1);
        MPI_Finalize();
        return 1;
    }

    const char *algo = argv[1];
    int nbytes       = atoi(argv[2]);
    int nchunks      = (argc > 3) ? atoi(argv[3]) : 64;
    int root         = (argc > 4) ? atoi(argv[4]) : 0;
    const char *plan_path = (argc > 5) ? argv[5] : NULL;
    const char *out_json  = (argc > 6) ? argv[6] : NULL;

    /* Treat "_" as no-file placeholder */
    if (plan_path && strcmp(plan_path, "_") == 0) plan_path = NULL;
    if (out_json  && strcmp(out_json,  "_") == 0) out_json  = NULL;

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
    double elapsed      = 0.0;
    int    total_rounds = 0;

    if (strcmp(algo, "mpi") == 0) {
        elapsed = run_mpi_bcast(buf, nbytes, root);

    } else if (strcmp(algo, "srda") == 0) {
        elapsed = run_srda(buf, nbytes, rank, size, root);

    } else if (strcmp(algo, "pipe") == 0) {
        elapsed = run_pipe(buf, nbytes, rank, size, root, nchunks);

    } else if (strcmp(algo, "bbs") == 0) {
        /* Load plan file */
        int loaded = -1;
        if (plan_path) {
            loaded = load_plan_file(plan_path);
        }
        if (loaded <= 0) {
            if (rank == 0)
                fprintf(stderr,
                    "BBS: plan file not found or empty.\n"
                    "  Tried: %s\n"
                    "  Generate plans with LP.py first.\n",
                    plan_path ? plan_path : "(none)");
            free(buf);
            MPI_Finalize();
            return 1;
        }

        sched_t *sched = NULL;
        if (g_weighted_mode)
            total_rounds = plan_bbs_adaptive(nchunks, root, size, &sched);
        else
            total_rounds = plan_bbs(nchunks, root, size, &sched);
        if (total_rounds < 0) {
            if (rank == 0)
                fprintf(stderr, "BBS: planning failed (no frames/edges loaded)\n");
            free(buf);
            MPI_Finalize();
            return 1;
        }
        elapsed = exec_schedule(buf, nbytes, rank, size,
                                nchunks, total_rounds, sched);
        free(sched);

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
        if (strcmp(algo, "bbs") == 0)
            printf("rounds    : %d\n", total_rounds);
        printf("time_sec  : %.9f\n", max_time);
        printf("correct   : %s\n", all_ok ? "yes" : "NO");

        /* ---- JSON output ---- */
        if (out_json) {
            char rounds_buf[32];
            if (strcmp(algo, "bbs") == 0)
                snprintf(rounds_buf, sizeof(rounds_buf), "%d", total_rounds);
            else
                snprintf(rounds_buf, sizeof(rounds_buf), "null");

            FILE *jfp = fopen(out_json, "w");
            if (jfp) {
                fprintf(jfp,
                    "{\n"
                    "  \"algorithm\": \"%s\",\n"
                    "  \"nodes\": %d,\n"
                    "  \"msg_bytes\": %d,\n"
                    "  \"nchunks\": %d,\n"
                    "  \"root\": %d,\n"
                    "  \"rounds\": %s,\n"
                    "  \"time_sec\": %.9f,\n"
                    "  \"correct\": %s\n"
                    "}\n",
                    algo, size, nbytes, nchunks, root,
                    rounds_buf, max_time,
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
