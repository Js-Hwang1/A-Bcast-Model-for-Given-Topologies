/*
 * bcast_16K3.c — Broadcast experiments on the 16K3 topology via SMPI.
 *
 * Architecture:
 *   1. PLAN  — compute a round-by-round broadcast schedule (pure local
 *              computation, no MPI, not timed).
 *   2. EXECUTE — replay the schedule with bare MPI_Isend / MPI_Irecv
 *              (timed).  This is apple-to-apple with MPI_Bcast, which
 *              also executes a pre-determined plan via Send/Recv.
 *
 * Algorithms:
 *   "mpi"    — Native MPI_Bcast (baseline).
 *   "greedy" — Greedy pipelined broadcast (pre-computed schedule).
 *   "bbs"    — Placeholder for BBS / LP-based plans.
 *
 * Full-duplex: a node may send AND receive in the same round.
 *
 * Build:   smpicc -O2 -o bcast_16K3 bcast_16K3.c -lm
 * Run:     smpirun -np 16 -platform platform_16K3.xml \
 *                  -hostfile hostfile ./bcast_16K3 <algo> <bytes> [chunks]
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_NODES   16
#define DEGREE       3
#define ROOT         0

/* ================================================================
 * Topology — 16K3 adjacency (0-indexed)
 * ================================================================ */
static const int ADJ[NUM_NODES][DEGREE] = {
    /* 0 */ { 1,  2,  3},
    /* 1 */ { 0,  6, 10},
    /* 2 */ { 0,  8, 12},
    /* 3 */ { 0,  4, 14},
    /* 4 */ { 3,  5,  9},
    /* 5 */ { 4,  6, 11},
    /* 6 */ { 1,  5,  7},
    /* 7 */ { 6,  8, 13},
    /* 8 */ { 2,  7,  9},
    /* 9 */ { 4,  8, 15},
    /*10 */ { 1, 11, 15},
    /*11 */ { 5, 10, 12},
    /*12 */ { 2, 11, 13},
    /*13 */ { 7, 12, 14},
    /*14 */ { 3, 13, 15},
    /*15 */ { 9, 10, 14},
};

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
 * One entry per (round, rank).  If send_to == -1 the rank is idle
 * as a sender that round; likewise for recv_from.
 * ================================================================ */
typedef struct {
    int send_to;        /* destination rank, or -1    */
    int send_chunk;     /* chunk id to send           */
    int recv_from;      /* source rank, or -1         */
    int recv_chunk;     /* chunk id to receive        */
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

        /* Copy received chunk into place */
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
static double run_mpi_bcast(void *buf, int count)
{
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    MPI_Bcast(buf, count, MPI_BYTE, ROOT, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    return t1 - t0;
}

/* ================================================================
 * Algorithm 2: Greedy — PLANNER
 *
 * Faithful re-implementation of simulate_greedy_pipeline_16k3():
 *   - Enumerate all (sender, receiver) pairs where sender has a
 *     chunk receiver lacks.
 *   - Shuffle deterministically (round-seeded LCG).
 *   - Greedily select non-conflicting pairs.
 *     Full-duplex: send_used[] and recv_used[] are separate.
 *   - Chunk = smallest id sender has that receiver lacks.
 *
 * Runs in pure local memory — no MPI, not timed.
 * Produces a schedule that exec_schedule() replays.
 * ================================================================ */
typedef struct { int src; int dst; } pair_t;

static int plan_greedy(int nchunks, sched_t **sched_out)
{
    const int bm_sz = BM_BYTES(nchunks);

    /* Simulated bitmaps for all nodes */
    unsigned char *bm = calloc(NUM_NODES * bm_sz, 1);
    for (int c = 0; c < nchunks; c++)
        BIT_SET(&bm[ROOT * bm_sz], c);

    /* Dynamic schedule buffer (grows as needed) */
    int cap    = 256;
    int rounds = 0;
    sched_t *sched = malloc(cap * NUM_NODES * sizeof(sched_t));

    /* Workspace */
    pair_t *possible = malloc(NUM_NODES * DEGREE * sizeof(pair_t));

    for (;;) {
        /* Check termination */
        int all_done = 1;
        for (int n = 0; n < NUM_NODES; n++) {
            if (!bm_full(&bm[n * bm_sz], nchunks)) { all_done = 0; break; }
        }
        if (all_done) break;

        /* Grow schedule buffer if needed */
        if (rounds >= cap) {
            cap *= 2;
            sched = realloc(sched, cap * NUM_NODES * sizeof(sched_t));
        }

        /* Initialize this round's schedule: everyone idle */
        sched_t *row = &sched[rounds * NUM_NODES];
        for (int n = 0; n < NUM_NODES; n++) {
            row[n].send_to = row[n].recv_from = -1;
            row[n].send_chunk = row[n].recv_chunk = -1;
        }

        /* Build feasible (sender, receiver) pairs */
        int nposs = 0;
        for (int s = 0; s < NUM_NODES; s++) {
            const unsigned char *sbm = &bm[s * bm_sz];
            for (int j = 0; j < DEGREE; j++) {
                int d = ADJ[s][j];
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

            /* Update simulated state */
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
 * Algorithm 3: BBS — PLANNER (placeholder)
 *
 * Same exec_schedule() executor — only the planner changes.
 * Wire in your LP / frame-based planner here.
 * ================================================================ */
static int plan_bbs(int nchunks, sched_t **sched_out)
{
    (void)nchunks;
    *sched_out = NULL;
    return -1;
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
                "Usage: %s <algorithm> <msg_bytes> [nchunks]\n"
                "  algorithm : mpi | greedy | bbs\n"
                "  nchunks   : pipeline depth (default 64)\n",
                argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *algo = argv[1];
    int nbytes       = atoi(argv[2]);
    int nchunks      = (argc > 3) ? atoi(argv[3]) : 64;

    if (nbytes <= 0) {
        if (rank == 0) fprintf(stderr, "msg_bytes must be > 0\n");
        MPI_Finalize();
        return 1;
    }

    /* Allocate buffer; root fills with known pattern */
    char *buf = calloc(nbytes, 1);
    if (rank == ROOT)
        for (int i = 0; i < nbytes; i++)
            buf[i] = (char)(i & 0xFF);

    /* ---- Dispatch ---- */
    double elapsed = 0.0;
    int    total_rounds = 0;

    if (strcmp(algo, "mpi") == 0) {
        elapsed = run_mpi_bcast(buf, nbytes);

    } else if (strcmp(algo, "greedy") == 0) {
        /* Phase 1: plan (not timed) */
        sched_t *sched = NULL;
        total_rounds = plan_greedy(nchunks, &sched);

        /* Phase 2: execute (timed) */
        elapsed = exec_schedule(buf, nbytes, rank,
                                nchunks, total_rounds, sched);
        free(sched);

    } else if (strcmp(algo, "bbs") == 0) {
        sched_t *sched = NULL;
        total_rounds = plan_bbs(nchunks, &sched);
        if (total_rounds < 0) {
            if (rank == 0)
                fprintf(stderr, "BBS planner not yet implemented.\n");
            free(buf);
            MPI_Finalize();
            return 1;
        }
        elapsed = exec_schedule(buf, nbytes, rank,
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
               ROOT, MPI_COMM_WORLD);
    int all_ok;
    MPI_Reduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        printf("algorithm : %s\n", algo);
        printf("nodes     : %d\n", NUM_NODES);
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
