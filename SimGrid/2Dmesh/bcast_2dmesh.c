/*
 * bcast_2dmesh.c — Broadcast experiments on a PxQ 2D mesh via SMPI.
 *
 * Configurable:
 *   - Mesh dimensions: compile-time via -DROWS=P -DCOLS=Q  (default 4x4)
 *   - Broadcast root:  runtime via command-line argument    (default 0)
 *
 * Architecture:
 *   1. PLAN  — compute round-by-round schedule (pure local, not timed).
 *   2. EXECUTE — replay with bare MPI_Isend / MPI_Irecv (timed).
 *
 * Algorithms:
 *   "mpi"    — Native MPI_Bcast (baseline).
 *   "greedy" — Greedy pipelined broadcast (pre-computed schedule).
 *   "bbs"    — BBS frame-based broadcast (pre-computed schedule).
 *   "tree"   — Topology-aware BFS tree pipeline (async, no rounds).
 *
 * Full-duplex: a node may send AND receive in the same round.
 *
 * Build:   smpicc -O2 -DROWS=4 -DCOLS=4 -o bcast_2dmesh bcast_2dmesh.c -lm
 * Run:     smpirun -np 16 -platform platform_2dmesh_4x4.xml \
 *                  -hostfile hostfile \
 *                  ./bcast_2dmesh <algo> <bytes> [chunks] [root]
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Mesh dimensions (override at compile time: -DROWS=P -DCOLS=Q) ---- */
#ifndef ROWS
#define ROWS 4
#endif
#ifndef COLS
#define COLS 4
#endif

#define NUM_NODES   (ROWS * COLS)
#define MAX_DEGREE  4

/* ================================================================
 * Topology — computed at startup from ROWS x COLS
 *
 * Row-major numbering:  node = row * COLS + col
 * Neighbors: left, right, up, down (if in bounds)
 * ================================================================ */
static int g_degree[NUM_NODES];
static int g_adj[NUM_NODES][MAX_DEGREE];

static void init_topology(void)
{
    for (int n = 0; n < NUM_NODES; n++) {
        int r = n / COLS;
        int c = n % COLS;
        int d = 0;
        if (c > 0)        g_adj[n][d++] = n - 1;       /* left  */
        if (c < COLS - 1) g_adj[n][d++] = n + 1;       /* right */
        if (r > 0)        g_adj[n][d++] = n - COLS;     /* up    */
        if (r < ROWS - 1) g_adj[n][d++] = n + COLS;     /* down  */
        g_degree[n] = d;
        for (int i = d; i < MAX_DEGREE; i++)
            g_adj[n][i] = -1;
    }
}

/* ================================================================
 * Bitmap helpers
 * ================================================================ */
#define BM_BYTES(nc)    (((nc) + 7) / 8)
#define BIT_SET(bm, c)  ((bm)[(c)/8] |=  (unsigned char)(1u << ((c)%8)))
#define BIT_GET(bm, c)  (((bm)[(c)/8] >> ((c)%8)) & 1u)

static int bm_has_new(const unsigned char *src, const unsigned char *dst,
                      int bm_sz)
{
    for (int b = 0; b < bm_sz; b++)
        if (src[b] & ~dst[b]) return 1;
    return 0;
}

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
 * Schedule data structure
 *
 * One entry per (round, rank).  send_to == -1 means idle as sender;
 * recv_from == -1 means idle as receiver.
 * ================================================================ */
typedef struct {
    int send_to;
    int send_chunk;
    int recv_from;
    int recv_chunk;
} sched_t;

/* ================================================================
 * Schedule executor  (timed — apple-to-apple with MPI_Bcast)
 *
 * Pure MPI_Isend / MPI_Irecv, no collectives, no state exchange.
 * ================================================================ */
static double exec_schedule(void *buf, int count, int rank,
                            int nchunks, int total_rounds,
                            const sched_t *sched)
{
    const int chunk_size = (count + nchunks - 1) / nchunks;
    char *recv_buf = malloc(chunk_size);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int r = 0; r < total_rounds; r++) {
        const sched_t *s = &sched[r * NUM_NODES + rank];
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
 * Algorithm 2: Greedy — PLANNER
 *
 * Adapted for variable-degree mesh (uses g_degree[], g_adj[]).
 *   - Enumerate all (sender, receiver) neighbor pairs where
 *     sender has a chunk receiver lacks.
 *   - Shuffle deterministically (round-seeded LCG).
 *   - Greedily select non-conflicting pairs (full-duplex).
 *   - Chunk = smallest id sender has that receiver lacks.
 * ================================================================ */
typedef struct { int src; int dst; } pair_t;

static int plan_greedy(int nchunks, int root, sched_t **sched_out)
{
    const int bm_sz = BM_BYTES(nchunks);

    unsigned char *bm = calloc(NUM_NODES * bm_sz, 1);
    for (int c = 0; c < nchunks; c++)
        BIT_SET(&bm[root * bm_sz], c);

    int cap    = 256;
    int rounds = 0;
    sched_t *sched = malloc(cap * NUM_NODES * sizeof(sched_t));

    pair_t *possible = malloc(NUM_NODES * MAX_DEGREE * sizeof(pair_t));

    for (;;) {
        int all_done = 1;
        for (int n = 0; n < NUM_NODES; n++) {
            if (!bm_full(&bm[n * bm_sz], nchunks)) { all_done = 0; break; }
        }
        if (all_done) break;

        if (rounds >= cap) {
            cap *= 2;
            sched = realloc(sched, cap * NUM_NODES * sizeof(sched_t));
        }

        sched_t *row = &sched[rounds * NUM_NODES];
        for (int n = 0; n < NUM_NODES; n++) {
            row[n].send_to = row[n].recv_from = -1;
            row[n].send_chunk = row[n].recv_chunk = -1;
        }

        int nposs = 0;
        for (int s = 0; s < NUM_NODES; s++) {
            const unsigned char *sbm = &bm[s * bm_sz];
            for (int j = 0; j < g_degree[s]; j++) {
                int d = g_adj[s][j];
                const unsigned char *dbm = &bm[d * bm_sz];
                if (bm_has_new(sbm, dbm, bm_sz))
                    possible[nposs++] = (pair_t){s, d};
            }
        }

        /* Deterministic shuffle (Fisher-Yates + LCG) */
        {
            unsigned int seed = (unsigned int)rounds * 2654435761u + 42u;
            for (int i = nposs - 1; i > 0; i--) {
                seed = seed * 1103515245u + 12345u;
                int j = (int)((seed >> 16) % (unsigned)(i + 1));
                pair_t tmp  = possible[i];
                possible[i] = possible[j];
                possible[j] = tmp;
            }
        }

        /* Greedy matching — full-duplex */
        int send_used[NUM_NODES] = {0};
        int recv_used[NUM_NODES] = {0};

        for (int i = 0; i < nposs; i++) {
            int s = possible[i].src;
            int d = possible[i].dst;
            if (send_used[s] || recv_used[d]) continue;

            int chunk = bm_first_missing(&bm[s * bm_sz],
                                         &bm[d * bm_sz], nchunks);
            if (chunk < 0) continue;

            send_used[s] = 1;
            recv_used[d] = 1;

            row[s].send_to    = d;
            row[s].send_chunk = chunk;
            row[d].recv_from  = s;
            row[d].recv_chunk = chunk;

            BIT_SET(&bm[d * bm_sz], chunk);
        }

        rounds++;
    }

    free(bm);
    free(possible);
    *sched_out = sched;
    return rounds;
}

/* ================================================================
 * Algorithm 3: BBS — Frame-based PLANNER
 *
 * BBS uses a fixed set of directed-matching frames.
 * Each frame is a set of (src, dst) pairs respecting full-duplex:
 *   - Each node appears as src at most once  (one send per round)
 *   - Each node appears as dst at most once  (one recv per round)
 *
 * The planner cycles through frames round-robin, and for each
 * active edge (src, dst) in the current frame, assigns the
 * smallest chunk that src has but dst lacks.
 *
 * NOTE: Frame data is topology-specific.  Regenerate frames when
 *       ROWS, COLS, or the root node changes.
 *
 * >>> Fill in BBS_FRAMES[][] with your solver output. <<<
 * ================================================================ */
typedef struct {
    int src;
    int dst;
} frame_edge_t;

/* BBS frames for 4x4 2D mesh (root=0, half-duplex)
 *
 * Generated by extract_bbs_frames.py from the Python BBS pipeline:
 *   1. Limitation matrix (integer weights per directed edge)
 *   2. Wavefront frame generation (8 vertex-disjoint directed matchings)
 *   3. Urgency-based frame ordering
 *   4. Edge-swap optimization
 *
 * Half-duplex: each node appears at most ONCE per frame (as sender
 * OR receiver, never both).  This means zero bandwidth contention —
 * every transfer is pre-resolved, like an FPGA-programmed schedule.
 *
 * 60 total directed edges across 8 frames.
 * Limitation matrix sum = 60 (confirmed).
 */
#define NUM_BBS_FRAMES      8
#define MAX_EDGES_PER_FRAME 9

static int BBS_FRAME_SIZES[NUM_BBS_FRAMES] = {
    8, 6, 8, 8, 6, 8, 8, 8
};

static frame_edge_t BBS_FRAMES[NUM_BBS_FRAMES][MAX_EDGES_PER_FRAME] = {
    /* Frame 0 (8 edges) */
    { {0, 1}, {2, 3}, {4, 5}, {7,11}, {9, 8}, {10, 6}, {12,13}, {15,14} },
    /* Frame 1 (6 edges) */
    { {1, 2}, {4, 8}, {5, 6}, {7,11}, {10, 9}, {13,14} },
    /* Frame 2 (8 edges) */
    { {0, 1}, {2, 3}, {5, 4}, {6, 7}, {8,12}, {11,10}, {13, 9}, {14,15} },
    /* Frame 3 (8 edges) */
    { {0, 4}, {2, 3}, {5, 1}, {7, 6}, {8,12}, {9,13}, {10,11}, {14,15} },
    /* Frame 4 (6 edges) */
    { {1, 2}, {4, 8}, {6, 5}, {9,10}, {11, 7}, {13,14} },
    /* Frame 5 (8 edges) */
    { {0, 4}, {1, 5}, {2, 6}, {7, 3}, {8, 9}, {10,14}, {11,15}, {13,12} },
    /* Frame 6 (8 edges) */
    { {0, 1}, {3, 7}, {4, 8}, {5, 9}, {6, 2}, {11,15}, {12,13}, {14,10} },
    /* Frame 7 (8 edges) */
    { {0, 4}, {1, 2}, {3, 7}, {6,10}, {8,12}, {9, 5}, {14,13}, {15,11} },
};

static int plan_bbs(int nchunks, int root, sched_t **sched_out)
{
    if (BBS_FRAME_SIZES[0] == 0) {
        *sched_out = NULL;
        return -1;
    }

    const int bm_sz = BM_BYTES(nchunks);

    unsigned char *bm = calloc(NUM_NODES * bm_sz, 1);
    for (int c = 0; c < nchunks; c++)
        BIT_SET(&bm[root * bm_sz], c);

    int cap    = 256;
    int rounds = 0;
    sched_t *sched = malloc(cap * NUM_NODES * sizeof(sched_t));

    for (;;) {
        int all_done = 1;
        for (int n = 0; n < NUM_NODES; n++) {
            if (!bm_full(&bm[n * bm_sz], nchunks)) { all_done = 0; break; }
        }
        if (all_done) break;

        if (rounds >= cap) {
            cap *= 2;
            sched = realloc(sched, cap * NUM_NODES * sizeof(sched_t));
        }

        sched_t *row = &sched[rounds * NUM_NODES];
        for (int n = 0; n < NUM_NODES; n++) {
            row[n].send_to = row[n].recv_from = -1;
            row[n].send_chunk = row[n].recv_chunk = -1;
        }

        int fidx = rounds % NUM_BBS_FRAMES;

        for (int e = 0; e < BBS_FRAME_SIZES[fidx]; e++) {
            int s = BBS_FRAMES[fidx][e].src;
            int d = BBS_FRAMES[fidx][e].dst;

            int chunk = bm_first_missing(&bm[s * bm_sz],
                                         &bm[d * bm_sz], nchunks);
            if (chunk < 0) continue;

            row[s].send_to    = d;
            row[s].send_chunk = chunk;
            row[d].recv_from  = s;
            row[d].recv_chunk = chunk;

            BIT_SET(&bm[d * bm_sz], chunk);
        }

        rounds++;
    }

    free(bm);
    *sched_out = sched;
    return rounds;
}

/* ================================================================
 * Algorithm 4: Topology-Aware BFS Tree Pipeline
 *
 * 1. Build a BFS spanning tree from root over the mesh adjacency.
 *    Every tree edge is a direct mesh link (1-hop, zero contention).
 * 2. Pipeline K chunks down the tree with async execution:
 *      Root:     for each chunk: Isend to all children
 *      Interior: for each chunk: blocking Recv from parent → Isend to children
 *      Leaf:     for each chunk: blocking Recv from parent
 * 3. No round synchronization — natural overlap.  Total time ≈
 *    (K + depth - 1) × (chunk_size/bandwidth + latency).
 * ================================================================ */
static void build_bfs_tree(int root,
                           int *parent,    /* [NUM_NODES] */
                           int children[][MAX_DEGREE],
                           int *nchildren) /* [NUM_NODES] */
{
    for (int i = 0; i < NUM_NODES; i++) {
        parent[i] = -1;
        nchildren[i] = 0;
    }
    parent[root] = root;   /* sentinel: root's parent is itself */

    int queue[NUM_NODES];
    int head = 0, tail = 0;
    queue[tail++] = root;

    while (head < tail) {
        int u = queue[head++];
        for (int j = 0; j < g_degree[u]; j++) {
            int v = g_adj[u][j];
            if (parent[v] == -1) {
                parent[v] = u;
                children[u][nchildren[u]++] = v;
                queue[tail++] = v;
            }
        }
    }
}

static double run_tree_pipeline(void *buf, int count, int rank,
                                int root, int nchunks)
{
    /* Build BFS tree (pure local computation, every rank does it) */
    int parent[NUM_NODES];
    int tree_children[NUM_NODES][MAX_DEGREE];
    int nchildren[NUM_NODES];
    build_bfs_tree(root, parent, tree_children, nchildren);

    int my_parent   = parent[rank];
    int my_nkids    = nchildren[rank];
    int chunk_size  = (count + nchunks - 1) / nchunks;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (rank == root) {
        /* Root has no Recv to pace its sends → per-chunk Waitall
         * prevents flooding the links with thousands of concurrent flows. */
        MPI_Request root_reqs[MAX_DEGREE];
        for (int c = 0; c < nchunks; c++) {
            int off = c * chunk_size;
            int len = chunk_size;
            if (off + len > count) len = count - off;

            int nreqs = 0;
            for (int k = 0; k < my_nkids; k++) {
                MPI_Isend((char *)buf + off, len, MPI_BYTE,
                          tree_children[rank][k], c, MPI_COMM_WORLD,
                          &root_reqs[nreqs++]);
            }
            if (nreqs > 0)
                MPI_Waitall(nreqs, root_reqs, MPI_STATUSES_IGNORE);
        }
    } else {
        /* Non-root: Recv naturally paces this node — at most a few
         * outstanding Isends per child link at any time. */
        MPI_Request *send_reqs = NULL;
        int nsend = 0;
        if (my_nkids > 0)
            send_reqs = malloc((size_t)nchunks * my_nkids * sizeof(MPI_Request));

        for (int c = 0; c < nchunks; c++) {
            int off = c * chunk_size;
            int len = chunk_size;
            if (off + len > count) len = count - off;

            MPI_Recv((char *)buf + off, len, MPI_BYTE,
                     my_parent, c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int k = 0; k < my_nkids; k++) {
                MPI_Isend((char *)buf + off, len, MPI_BYTE,
                          tree_children[rank][k], c, MPI_COMM_WORLD,
                          &send_reqs[nsend++]);
            }
        }

        if (nsend > 0)
            MPI_Waitall(nsend, send_reqs, MPI_STATUSES_IGNORE);
        free(send_reqs);
    }

    double t1 = MPI_Wtime();
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

    if (size != NUM_NODES) {
        if (rank == 0)
            fprintf(stderr, "Error: need exactly %d ranks (got %d)\n",
                    NUM_NODES, size);
        MPI_Finalize();
        return 1;
    }

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr,
                "Usage: %s <algorithm> <msg_bytes> [nchunks] [root]\n"
                "  algorithm : mpi | greedy | bbs | tree\n"
                "  nchunks   : pipeline depth (default 64)\n"
                "  root      : broadcast root node 0..%d (default 0)\n",
                argv[0], NUM_NODES - 1);
        MPI_Finalize();
        return 1;
    }

    const char *algo = argv[1];
    int nbytes       = atoi(argv[2]);
    int nchunks      = (argc > 3) ? atoi(argv[3]) : 64;
    int root         = (argc > 4) ? atoi(argv[4]) : 0;

    if (nbytes <= 0) {
        if (rank == 0) fprintf(stderr, "msg_bytes must be > 0\n");
        MPI_Finalize();
        return 1;
    }
    if (root < 0 || root >= NUM_NODES) {
        if (rank == 0)
            fprintf(stderr, "root must be in 0..%d (got %d)\n",
                    NUM_NODES - 1, root);
        MPI_Finalize();
        return 1;
    }

    /* Build adjacency from ROWS x COLS */
    init_topology();

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

    } else if (strcmp(algo, "greedy") == 0) {
        sched_t *sched = NULL;
        total_rounds = plan_greedy(nchunks, root, &sched);
        elapsed = exec_schedule(buf, nbytes, rank,
                                nchunks, total_rounds, sched);
        free(sched);

    } else if (strcmp(algo, "bbs") == 0) {
        sched_t *sched = NULL;
        total_rounds = plan_bbs(nchunks, root, &sched);
        if (total_rounds < 0) {
            if (rank == 0)
                fprintf(stderr,
                    "BBS frames not yet populated.\n"
                    "Fill in BBS_FRAMES[][] and BBS_FRAME_SIZES[] "
                    "in bcast_2dmesh.c\n");
            free(buf);
            MPI_Finalize();
            return 1;
        }
        elapsed = exec_schedule(buf, nbytes, rank,
                                nchunks, total_rounds, sched);
        free(sched);

    } else if (strcmp(algo, "tree") == 0) {
        elapsed = run_tree_pipeline(buf, nbytes, rank, root, nchunks);

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
        printf("topology  : 2Dmesh_%dx%d\n", ROWS, COLS);
        printf("nodes     : %d\n", NUM_NODES);
        printf("root      : %d  (row %d, col %d)\n",
               root, root / COLS, root % COLS);
        printf("msg_bytes : %d\n", nbytes);
        if (strcmp(algo, "mpi") != 0) {
            printf("nchunks   : %d\n", nchunks);
            printf("rounds    : %d\n", total_rounds);
        }
        printf("time_sec  : %.9f\n", max_time);
        printf("correct   : %s\n", all_ok ? "yes" : "NO");
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
