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
 * Algorithm 4: BBS — (stub, not yet implemented)
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
                "Usage: %s <algorithm> <msg_bytes> [nchunks] [root] [out_json]\n"
                "  algorithm : mpi | srda | pipe | bbs\n"
                "  nchunks   : pipeline depth (default 64)\n"
                "  root      : broadcast root node 0..%d (default 0)\n"
                "  out_json  : output JSON file path (optional, _ = none)\n",
                argv[0], size - 1);
        MPI_Finalize();
        return 1;
    }

    const char *algo = argv[1];
    int nbytes       = atoi(argv[2]);
    int nchunks      = (argc > 3) ? atoi(argv[3]) : 64;
    int root         = (argc > 4) ? atoi(argv[4]) : 0;
    const char *out_json  = (argc > 5) ? argv[5] : NULL;

    /* Treat "_" as no-file placeholder */
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
    double elapsed = 0.0;

    if (strcmp(algo, "mpi") == 0) {
        elapsed = run_mpi_bcast(buf, nbytes, root);

    } else if (strcmp(algo, "srda") == 0) {
        elapsed = run_srda(buf, nbytes, rank, size, root);

    } else if (strcmp(algo, "pipe") == 0) {
        elapsed = run_pipe(buf, nbytes, rank, size, root, nchunks);

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
