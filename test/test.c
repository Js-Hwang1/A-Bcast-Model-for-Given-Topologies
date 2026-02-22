/*
 * test.c — Measure f(k): time for node 0 to send 1MB to k children
 *          using non-blocking MPI_Isend + MPI_Waitall.
 *
 * For each k = 1..100:
 *   - Rank 0 does k MPI_Isend calls + MPI_Waitall
 *   - Ranks 1..k each do MPI_Irecv + MPI_Wait
 *   - Measure wall-clock time on rank 0
 *
 * Output: "k time_sec" per line (for plotting)
 *
 * Build:  smpicc -O2 -o test test.c
 * Run:    smpirun -np 101 -platform platform_star.xml -hostfile hostfile_101 \
 *                 --cfg=smpi/host-speed:2000Gf --log=root.thres:warning ./test
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MSG_SIZE (1 * 1024 * 1024)   /* 1 MB */
#define MAX_K    100

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < MAX_K + 1) {
        if (rank == 0)
            fprintf(stderr, "Need at least %d ranks (got %d)\n", MAX_K + 1, size);
        MPI_Finalize();
        return 1;
    }

    char *buf = calloc(MSG_SIZE, 1);
    if (rank == 0)
        for (int i = 0; i < MSG_SIZE; i++)
            buf[i] = (char)(i & 0xFF);

    MPI_Request *reqs = malloc(MAX_K * sizeof(MPI_Request));

    /* Warmup: dummy send/recv to trigger SimGrid lazy init */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Isend(buf, MSG_SIZE, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Waitall(1, reqs, MPI_STATUSES_IGNORE);
    } else if (rank == 1) {
        MPI_Request req;
        MPI_Irecv(buf, MSG_SIZE, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        printf("# k  time_sec\n");

    for (int k = 1; k <= MAX_K; k++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        if (rank == 0) {
            /* Send 1MB to ranks 1..k in parallel */
            for (int i = 0; i < k; i++)
                MPI_Isend(buf, MSG_SIZE, MPI_BYTE, i + 1, k,
                          MPI_COMM_WORLD, &reqs[i]);
            MPI_Waitall(k, reqs, MPI_STATUSES_IGNORE);
        } else if (rank <= k) {
            /* Receive 1MB from rank 0 */
            MPI_Request req;
            MPI_Irecv(buf, MSG_SIZE, MPI_BYTE, 0, k,
                       MPI_COMM_WORLD, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }
        /* ranks > k do nothing this iteration */

        double t1 = MPI_Wtime();

        if (rank == 0)
            printf("%d %.9f\n", k, t1 - t0);
    }

    free(reqs);
    free(buf);
    MPI_Finalize();
    return 0;
}
