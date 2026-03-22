/*
 * PoC: Does fanout from a switch rank use separate links or shared?
 *
 * Uses the real FatTree platform (152 ranks).
 * Rank mapping: 0-127 = compute, 128-135 = leaf, 136-151 = spine.
 *
 * Tests:
 *   single:     leaf-0 (rank 128) sends 1MB to spine-0 (rank 136)
 *   dual:       leaf-0 sends 1MB to spine-0 AND spine-1 concurrently
 *   fanout16:   leaf-0 sends 1MB to ALL 16 spines concurrently
 *   shared:     node-0 AND node-1 both send 1MB to spine-0
 *               (routes share link-leaf0-spine0)
 *   unshared:   node-0 sends to spine-0, node-1 sends to spine-1
 *               (routes use different leaf-spine links)
 *
 * Expected if per-port links are independent:
 *   single ≈ dual ≈ fanout16 (different links, no sharing)
 *   shared ≈ 2× single      (both traverse link-leaf0-spine0)
 *   unshared ≈ single       (disjoint leaf-spine links)
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

    if (size != 152) {
        if (rank == 0) fprintf(stderr, "Need 152 ranks\n");
        MPI_Finalize();
        return 1;
    }
    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <mode> <msg_bytes>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *mode = argv[1];
    int nbytes = atoi(argv[2]);
    void *buf = calloc(1, nbytes);

    /* rank 128 = leaf-0, ranks 136..151 = spine-0..spine-15 */
    const int LEAF0 = 128;
    const int SPINE0 = 136;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (strcmp(mode, "single") == 0) {
        /* leaf-0 → spine-0: 1 link (link-leaf0-spine0) */
        if (rank == LEAF0)
            MPI_Send(buf, nbytes, MPI_BYTE, SPINE0, 0, MPI_COMM_WORLD);
        if (rank == SPINE0)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

    } else if (strcmp(mode, "dual") == 0) {
        /* leaf-0 → spine-0 AND spine-1 concurrently */
        if (rank == LEAF0) {
            MPI_Request reqs[2];
            MPI_Isend(buf, nbytes, MPI_BYTE, SPINE0,   0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(buf, nbytes, MPI_BYTE, SPINE0+1, 0, MPI_COMM_WORLD, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        }
        if (rank == SPINE0)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        if (rank == SPINE0+1)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

    } else if (strcmp(mode, "fanout16") == 0) {
        /* leaf-0 → all 16 spines concurrently */
        if (rank == LEAF0) {
            MPI_Request reqs[16];
            for (int i = 0; i < 16; i++)
                MPI_Isend(buf, nbytes, MPI_BYTE, SPINE0+i, 0,
                          MPI_COMM_WORLD, &reqs[i]);
            MPI_Waitall(16, reqs, MPI_STATUSES_IGNORE);
        }
        for (int i = 0; i < 16; i++) {
            if (rank == SPINE0+i)
                MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
        }

    } else if (strcmp(mode, "shared") == 0) {
        /* node-0 → spine-0 AND node-1 → spine-0 concurrently.
         * Both routes: nodeX → leaf-0 → spine-0.
         * Shared link: link-leaf0-spine0. */
        if (rank == 0)
            MPI_Send(buf, nbytes, MPI_BYTE, SPINE0, 0, MPI_COMM_WORLD);
        if (rank == 1)
            MPI_Send(buf, nbytes, MPI_BYTE, SPINE0, 1, MPI_COMM_WORLD);
        if (rank == SPINE0) {
            MPI_Request reqs[2];
            MPI_Irecv(buf, nbytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(buf, nbytes, MPI_BYTE, 1, 1, MPI_COMM_WORLD, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        }

    } else if (strcmp(mode, "unshared") == 0) {
        /* node-0 → spine-0, node-1 → spine-1 concurrently.
         * Different leaf-spine links: link-leaf0-spine0 vs link-leaf0-spine1. */
        if (rank == 0)
            MPI_Send(buf, nbytes, MPI_BYTE, SPINE0,   0, MPI_COMM_WORLD);
        if (rank == 1)
            MPI_Send(buf, nbytes, MPI_BYTE, SPINE0+1, 0, MPI_COMM_WORLD);
        if (rank == SPINE0)
            MPI_Recv(buf, nbytes, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        if (rank == SPINE0+1)
            MPI_Recv(buf, nbytes, MPI_BYTE, 1, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

    } else {
        if (rank == 0) fprintf(stderr, "Unknown mode: %s\n", mode);
        free(buf);
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("mode: %-12s  bytes: %d  time_sec: %.9e\n",
               mode, nbytes, t1 - t0);

    free(buf);
    MPI_Finalize();
    return 0;
}
