/*
 * chain_test.c — Pipelined broadcast on a linear chain.
 *
 * 4 nodes: 0 → 1 → 2 → 3
 * Uses non-blocking sends + double-buffering so each node can
 * receive chunk[i+1] while forwarding chunk[i] (true pipeline).
 *
 * Usage:
 *   smpirun -np 4 -platform chain4.xml -hostfile hostfile_4 \
 *           ./chain_test <total_bytes> <chunk_bytes>
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <total_bytes> <chunk_bytes>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    long total_bytes = atol(argv[1]);
    long chunk_bytes = atol(argv[2]);
    int nchunks = (int)((total_bytes + chunk_bytes - 1) / chunk_bytes);

    /* Double buffer for pipeline overlap */
    char *buf[2];
    buf[0] = calloc(chunk_bytes, 1);
    buf[1] = calloc(chunk_bytes, 1);

    /* Root fills both buffers with pattern */
    if (rank == 0) {
        memset(buf[0], 0xAB, chunk_bytes);
        memset(buf[1], 0xAB, chunk_bytes);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Request send_req = MPI_REQUEST_NULL;

    for (int c = 0; c < nchunks; c++) {
        int cur = c % 2;
        long this_chunk = chunk_bytes;
        if (c == nchunks - 1) {
            long remaining = total_bytes - (long)c * chunk_bytes;
            if (remaining < this_chunk) this_chunk = remaining;
        }

        /* Receive into current buffer */
        if (rank > 0)
            MPI_Recv(buf[cur], (int)this_chunk, MPI_BYTE, rank - 1, c,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Wait for previous send to complete before reusing buffer */
        if (send_req != MPI_REQUEST_NULL)
            MPI_Wait(&send_req, MPI_STATUS_IGNORE);

        /* Non-blocking send: frees this node to receive next chunk */
        if (rank < size - 1)
            MPI_Isend(buf[cur], (int)this_chunk, MPI_BYTE, rank + 1, c,
                      MPI_COMM_WORLD, &send_req);
    }

    /* Wait for final send */
    if (send_req != MPI_REQUEST_NULL)
        MPI_Wait(&send_req, MPI_STATUS_IGNORE);

    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;

    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Verify last node got the data */
    int ok = 1;
    if (rank == size - 1) {
        int last_buf = (nchunks - 1) % 2;
        for (long i = 0; i < chunk_bytes && i < 16; i++)
            if ((unsigned char)buf[last_buf][i] != 0xAB) { ok = 0; break; }
    }

    int all_ok;
    MPI_Reduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("chain_nodes : %d\n", size);
        printf("total_bytes : %ld\n", total_bytes);
        printf("chunk_bytes : %ld\n", chunk_bytes);
        printf("nchunks     : %d\n", nchunks);
        printf("time_sec    : %.9f\n", max_time);
        printf("correct     : %s\n", all_ok ? "yes" : "NO");
    }

    free(buf[0]);
    free(buf[1]);
    MPI_Finalize();
    return 0;
}
