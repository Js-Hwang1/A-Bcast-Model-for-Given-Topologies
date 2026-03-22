/*
 * PoC: Does leaf_rl fanout to 31 children cost the same as fanout to 1?
 *
 * leaf_rl = rank 128 (leaf-0)
 * 16 spines = ranks 136..151 (via link-leaf0-spineX, 16 separate links)
 * 15 local computes = ranks 1..15 (via link-nodeX-leaf0, 15 separate links)
 *
 * Tests:
 *   fan1:   leaf_rl sends to spine-0 only (1 send, 1 link)
 *   fan16:  leaf_rl sends to all 16 spines (16 sends, 16 links)
 *   fan15:  leaf_rl sends to 15 local computes (15 sends, 15 links)
 *   fan31:  leaf_rl sends to all 31 children (31 sends, 31 links)
 *
 * If per-port links are truly independent:
 *   fan1 ≈ fan16 ≈ fan15 ≈ fan31
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
        if (rank == 0) fprintf(stderr, "Usage: %s <mode> <msg_bytes>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *mode = argv[1];
    int nbytes = atoi(argv[2]);
    void *buf = calloc(1, nbytes);

    const int LEAF0 = 128;
    const int SPINE0 = 136;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (strcmp(mode, "fan1") == 0) {
        /* leaf_rl → spine-0 only */
        if (rank == LEAF0)
            MPI_Send(buf, nbytes, MPI_BYTE, SPINE0, 0, MPI_COMM_WORLD);
        if (rank == SPINE0)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

    } else if (strcmp(mode, "fan16") == 0) {
        /* leaf_rl → all 16 spines */
        if (rank == LEAF0) {
            MPI_Request reqs[16];
            for (int i = 0; i < 16; i++)
                MPI_Isend(buf, nbytes, MPI_BYTE, SPINE0 + i, 0,
                          MPI_COMM_WORLD, &reqs[i]);
            MPI_Waitall(16, reqs, MPI_STATUSES_IGNORE);
        }
        for (int i = 0; i < 16; i++)
            if (rank == SPINE0 + i)
                MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

    } else if (strcmp(mode, "fan15") == 0) {
        /* leaf_rl → 15 local computes (ranks 1..15) */
        if (rank == LEAF0) {
            MPI_Request reqs[15];
            for (int i = 0; i < 15; i++)
                MPI_Isend(buf, nbytes, MPI_BYTE, 1 + i, 0,
                          MPI_COMM_WORLD, &reqs[i]);
            MPI_Waitall(15, reqs, MPI_STATUSES_IGNORE);
        }
        if (rank >= 1 && rank <= 15)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

    } else if (strcmp(mode, "fan31") == 0) {
        /* leaf_rl → 16 spines + 15 local computes */
        if (rank == LEAF0) {
            MPI_Request reqs[31];
            for (int i = 0; i < 16; i++)
                MPI_Isend(buf, nbytes, MPI_BYTE, SPINE0 + i, 0,
                          MPI_COMM_WORLD, &reqs[i]);
            for (int i = 0; i < 15; i++)
                MPI_Isend(buf, nbytes, MPI_BYTE, 1 + i, 0,
                          MPI_COMM_WORLD, &reqs[16 + i]);
            MPI_Waitall(31, reqs, MPI_STATUSES_IGNORE);
        }
        for (int i = 0; i < 16; i++)
            if (rank == SPINE0 + i)
                MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
        if (rank >= 1 && rank <= 15)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

    } else {
        if (rank == 0) fprintf(stderr, "Unknown mode: %s\n", mode);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("mode: %-8s  bytes: %d  time_sec: %.9e\n",
               mode, nbytes, t1 - t0);

    free(buf);
    MPI_Finalize();
    return 0;
}
