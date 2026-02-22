/*
 * chain_test.c — Smoke test: can a hand-crafted Send/Recv plan beat MPI_Bcast?
 *
 * Topology: 8-node linear chain  0 — 1 — 2 — 3 — 4 — 5 — 6 — 7
 *
 * MPI_Bcast (binomial tree):
 *   Step 1: 0→1   (1 hop)
 *   Step 2: 0→2   (2 hops!), 1→3 (2 hops!)
 *   Step 3: 0→4   (4 hops!), 1→5, 2→6, 3→7
 *   Multi-hop sends waste bandwidth on intermediate links.
 *
 * Optimal pipeline (topology-aware):
 *   Split message into K chunks.  Each round, every node forwards
 *   the next chunk to its right neighbor.  All transfers are 1-hop.
 *   Completes in (N-1) + (K-1) rounds.
 *   Full pipeline utilization — every link busy at steady state.
 *
 * Build:  smpicc -O2 -o chain_test chain_test.c
 * Run:    smpirun -np 8 -platform platform_chain8.xml \
 *                 -hostfile hostfile ./chain_test <msg_bytes> [nchunks]
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_NODES 8
#define ROOT    0

/* ================================================================
 * Hand-crafted pipeline broadcast for a linear chain.
 *
 * In round r:
 *   Node i SENDS chunk (r - i) to node i+1
 *     valid if:  i < N-1  AND  0 <= (r-i) < K
 *     tag = chunk id = (r - i)
 *
 *   Node i RECEIVES chunk (r - i + 1) from node i-1
 *     (that's what node i-1 sends: chunk r-(i-1) = r-i+1)
 *     valid if:  i > 0  AND  0 <= (r-i+1) < K
 *     tag = chunk id = (r - i + 1)
 *
 * Full-duplex: send and recv happen simultaneously on different chunks.
 * Total rounds = (N-1) + (K-1).
 * ================================================================ */
static double run_pipeline(void *buf, int count, int rank, int nchunks)
{
    int chunk_size   = (count + nchunks - 1) / nchunks;
    int total_rounds = (N_NODES - 1) + (nchunks - 1);
    char *recv_buf   = malloc(chunk_size);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int r = 0; r < total_rounds; r++) {
        MPI_Request reqs[2];
        int nreqs = 0;

        /* SEND: forward chunk (r - rank) to right neighbor */
        int sc = r - rank;                /* chunk id to send */
        if (rank < N_NODES - 1 && sc >= 0 && sc < nchunks) {
            int off = sc * chunk_size;
            int len = chunk_size;
            if (off + len > count) len = count - off;
            MPI_Isend((char *)buf + off, len, MPI_BYTE,
                      rank + 1, sc, MPI_COMM_WORLD, &reqs[nreqs++]);
        }

        /* RECV: receive chunk (r - rank + 1) from left neighbor */
        int rc = r - rank + 1;            /* chunk id to receive */
        if (rank > ROOT && rc >= 0 && rc < nchunks) {
            MPI_Irecv(recv_buf, chunk_size, MPI_BYTE,
                      rank - 1, rc, MPI_COMM_WORLD, &reqs[nreqs++]);
        }

        if (nreqs > 0)
            MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        /* Copy received chunk into place */
        if (rank > ROOT && rc >= 0 && rc < nchunks) {
            int off = rc * chunk_size;
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
 * Pipeline v2: no artificial rounds.
 *
 * Each node loops over chunks:  recv from left, then Isend to right.
 * The Isend overlaps with the next recv — natural pipelining with
 * zero round-synchronization overhead.
 * ================================================================ */
static double run_pipeline_v2(void *buf, int count, int rank, int nchunks)
{
    int chunk_size = (count + nchunks - 1) / nchunks;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Request send_req = MPI_REQUEST_NULL;

    for (int c = 0; c < nchunks; c++) {
        int off = c * chunk_size;
        int len = chunk_size;
        if (off + len > count) len = count - off;

        /* Receive chunk c from left neighbor (blocking) */
        if (rank > ROOT)
            MPI_Recv((char *)buf + off, len, MPI_BYTE,
                     rank - 1, c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Wait for previous send to complete before reusing buffer
         * (not strictly needed here since each chunk has its own
         *  buffer region, but keeps things safe) */
        if (send_req != MPI_REQUEST_NULL)
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);

        /* Forward chunk c to right neighbor (non-blocking) */
        if (rank < N_NODES - 1)
            MPI_Isend((char *)buf + off, len, MPI_BYTE,
                      rank + 1, c, MPI_COMM_WORLD, &send_req);
    }

    /* Wait for last outstanding send */
    if (send_req != MPI_REQUEST_NULL)
        MPI_Wait(&send_req, MPI_STATUS_IGNORE);

    double t1 = MPI_Wtime();
    return t1 - t0;
}

/* ================================================================
 * MPI_Bcast baseline
 * ================================================================ */
static double run_mpi_bcast(void *buf, int count)
{
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    MPI_Bcast(buf, count, MPI_BYTE, ROOT, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    return t1 - t0;
}

/* ================================================================ */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != N_NODES) {
        if (rank == 0)
            fprintf(stderr, "Need exactly %d ranks\n", N_NODES);
        MPI_Finalize();
        return 1;
    }

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <msg_bytes> [nchunks]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int nbytes  = atoi(argv[1]);
    int nchunks = (argc > 2) ? atoi(argv[2]) : 64;

    /* Root fills buffers */
    char *buf_mpi  = calloc(nbytes, 1);
    char *buf_pipe = calloc(nbytes, 1);
    if (rank == ROOT) {
        for (int i = 0; i < nbytes; i++) {
            buf_mpi[i]  = (char)(i & 0xFF);
            buf_pipe[i] = (char)(i & 0xFF);
        }
    }

    /* Third buffer for pipeline v2 */
    char *buf_v2 = calloc(nbytes, 1);
    if (rank == ROOT)
        for (int i = 0; i < nbytes; i++)
            buf_v2[i] = (char)(i & 0xFF);

    /* Run all three */
    double t_mpi  = run_mpi_bcast(buf_mpi, nbytes);
    double t_pipe = run_pipeline(buf_pipe, nbytes, rank, nchunks);
    double t_v2   = run_pipeline_v2(buf_v2, nbytes, rank, nchunks);

    /* Verify all three */
    int ok_mpi = 1, ok_pipe = 1, ok_v2 = 1;
    for (int i = 0; i < nbytes; i++) {
        if (buf_mpi[i]  != (char)(i & 0xFF)) ok_mpi  = 0;
        if (buf_pipe[i] != (char)(i & 0xFF)) ok_pipe = 0;
        if (buf_v2[i]   != (char)(i & 0xFF)) ok_v2   = 0;
    }

    /* Collect max times */
    double max_mpi, max_pipe, max_v2;
    MPI_Reduce(&t_mpi,  &max_mpi,  1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
    MPI_Reduce(&t_pipe, &max_pipe, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
    MPI_Reduce(&t_v2,   &max_v2,   1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
    int all_ok_mpi, all_ok_pipe, all_ok_v2;
    MPI_Reduce(&ok_mpi,  &all_ok_mpi,  1, MPI_INT, MPI_MIN, ROOT, MPI_COMM_WORLD);
    MPI_Reduce(&ok_pipe, &all_ok_pipe, 1, MPI_INT, MPI_MIN, ROOT, MPI_COMM_WORLD);
    MPI_Reduce(&ok_v2,   &all_ok_v2,   1, MPI_INT, MPI_MIN, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        printf("msg_bytes  : %d\n", nbytes);
        printf("nchunks    : %d\n", nchunks);
        printf("\n");
        printf("MPI_Bcast  : %.9f s  (correct: %s)\n",
               max_mpi, all_ok_mpi ? "yes" : "NO");
        printf("Pipeline v1: %.9f s  (correct: %s)  [round-sync]\n",
               max_pipe, all_ok_pipe ? "yes" : "NO");
        printf("Pipeline v2: %.9f s  (correct: %s)  [recv-forward]\n",
               max_v2, all_ok_v2 ? "yes" : "NO");
        printf("\n");

        double s1 = max_mpi / max_pipe;
        double s2 = max_mpi / max_v2;
        if (s2 >= 1.0)
            printf(">>> v2 is %.2fx FASTER than MPI_Bcast <<<\n", s2);
        else
            printf(">>> MPI_Bcast is %.2fx faster than v2 <<<\n", 1.0/s2);
    }

    free(buf_v2);

    free(buf_mpi);
    free(buf_pipe);
    MPI_Finalize();
    return 0;
}
