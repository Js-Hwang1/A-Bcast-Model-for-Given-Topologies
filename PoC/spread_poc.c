/*
 * PoC: sequential binary spread vs leaf-switch overlay
 *
 * 17 MPI ranks:  0..15 = compute nodes on node-0..node-15
 *                16    = leaf switch on leaf-0
 *
 * Mode A ("binary"):
 *   Sequential binary spread among compute ranks.
 *   Depth 0: 0 sends to 1          (wait for completion)
 *   Depth 1: 0 sends to 2, 1→3     (wait for completion)
 *   Depth 2: 0→4,1→5,2→6,3→7      (wait for completion)
 *   Depth 3: 0→8,1→9,2→10,3→11,4→12,5→13,6→14,7→15 (wait)
 *   All sends/recvs per depth are blocking; next depth starts only
 *   after ALL sends at this depth finish.
 *
 * Mode B ("overlay"):
 *   Rank 0 sends to rank 16 (leaf switch).
 *   Rank 16 fans out to ranks 1..15.
 *   All 15 sends from rank 16 happen concurrently (Isend + Waitall).
 *
 * Both modes send the FULL message (not pipelined).
 * Rank 16 (leaf) is idle in mode A.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void mode_binary(int rank, int size, void *buf, int nbytes)
{
    (void)size;
    /* 16 compute ranks: 0..15.  Rank 16 (leaf) participates in barriers only. */

    /* 4 depths for 16 nodes */
    for (int d = 0; d < 4; d++) {
        int stride = 1 << d;          /* 1, 2, 4, 8 */
        int nsenders = 1 << d;        /* 1, 2, 4, 8 */

        /* Am I a sender at this depth? (rank 16 = leaf, skip send/recv) */
        if (rank < 16 && rank < nsenders) {
            int dst = rank + stride;
            if (dst < 16) {
                MPI_Send(buf, nbytes, MPI_BYTE, dst, d, MPI_COMM_WORLD);
            }
        }
        /* Am I a receiver at this depth? */
        if (rank < 16 && rank >= stride && rank < stride * 2) {
            int src = rank - stride;
            MPI_Recv(buf, nbytes, MPI_BYTE, src, d, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

static void mode_overlay(int rank, int size, void *buf, int nbytes)
{
    (void)size;
    if (rank == 0) {
        /* Root sends to leaf switch (rank 16) */
        MPI_Send(buf, nbytes, MPI_BYTE, 16, 0, MPI_COMM_WORLD);
    } else if (rank == 16) {
        /* Leaf switch: receive from root, then fan out to 1..15 */
        MPI_Recv(buf, nbytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        MPI_Request reqs[15];
        for (int i = 1; i <= 15; i++) {
            MPI_Isend(buf, nbytes, MPI_BYTE, i, 1, MPI_COMM_WORLD, &reqs[i-1]);
        }
        MPI_Waitall(15, reqs, MPI_STATUSES_IGNORE);
    } else {
        /* Compute ranks 1..15: receive from leaf switch */
        MPI_Recv(buf, nbytes, MPI_BYTE, 16, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 17) {
        if (rank == 0)
            fprintf(stderr, "Need exactly 17 ranks (16 compute + 1 leaf)\n");
        MPI_Finalize();
        return 1;
    }
    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <binary|overlay> <msg_bytes>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *mode = argv[1];
    int nbytes = atoi(argv[2]);
    void *buf = calloc(1, nbytes);

    /* Warm up */
    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();

    if (strcmp(mode, "binary") == 0)
        mode_binary(rank, size, buf, nbytes);
    else if (strcmp(mode, "overlay") == 0)
        mode_overlay(rank, size, buf, nbytes);
    else {
        if (rank == 0) fprintf(stderr, "Unknown mode: %s\n", mode);
        free(buf);
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("mode      : %s\n", mode);
        printf("msg_bytes : %d\n", nbytes);
        printf("time_sec  : %.9e\n", t1 - t0);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
