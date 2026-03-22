/*
 * PoC part 2: Does leaf↔spine edge sharing across trees matter in the
 * actual BBS pipeline?
 *
 * Test: 3 concurrent multi-hop flows from leaf-0 through different spines
 * to different remote leaves. Compare:
 *   A) 3 flows using 3 DIFFERENT spines (no shared leaf-spine links)
 *   B) 3 flows all using the SAME spine (shared leaf-spine link)
 *
 * If edge sharing matters: B >> A.
 * If pipeline stages don't overlap: B ≈ A.
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

    /* Rank mapping:
     * 0-127: compute nodes (16 per leaf)
     * 128-135: leaf-0..leaf-7
     * 136-151: spine-0..spine-15
     *
     * leaf-0 has computes 0-15
     * leaf-1 has computes 16-31
     * leaf-2 has computes 32-47
     * leaf-3 has computes 48-63
     */
    const int LEAF0 = 128;
    const int LEAF1 = 129;
    const int LEAF2 = 130;
    const int LEAF3 = 131;
    const int SPINE0 = 136;
    const int SPINE1 = 137;
    const int SPINE2 = 138;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (strcmp(mode, "baseline") == 0) {
        /* Single flow: leaf-0 → spine-0 → leaf-1 → compute-16
         * 3 links: link-leaf0-spine0 + link-leaf1-spine0 + link-node16-leaf1 */
        if (rank == LEAF0)
            MPI_Send(buf, nbytes, MPI_BYTE, 16, 0, MPI_COMM_WORLD);
        if (rank == 16)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

    } else if (strcmp(mode, "diff_spine") == 0) {
        /* 3 flows from leaf-0 through DIFFERENT spines to different targets:
         * leaf-0 → spine-0 → leaf-1 → node-16
         * leaf-0 → spine-1 → leaf-2 → node-32
         * leaf-0 → spine-2 → leaf-3 → node-48
         * No shared leaf-spine links. */
        if (rank == LEAF0) {
            MPI_Request reqs[3];
            MPI_Isend(buf, nbytes, MPI_BYTE, 16, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(buf, nbytes, MPI_BYTE, 32, 0, MPI_COMM_WORLD, &reqs[1]);
            MPI_Isend(buf, nbytes, MPI_BYTE, 48, 0, MPI_COMM_WORLD, &reqs[2]);
            MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);
        }
        if (rank == 16)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        if (rank == 32)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        if (rank == 48)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

    } else if (strcmp(mode, "same_spine") == 0) {
        /* 3 flows from leaf-0 ALL through spine-0:
         * leaf-0 → spine-0 → leaf-1 → node-16
         * leaf-0 → spine-0 → leaf-2 → node-32
         * leaf-0 → spine-0 → leaf-3 → node-48
         *
         * BUT Floyd only picks ONE spine per src-dst pair.  We can't
         * force spine-0 for all 3.  Instead, test leaf→spine contention
         * directly: leaf-0 sends to 3 DIFFERENT computes under leaf-1.
         * All 3 share link-leaf0-spine? AND link-leaf1-spine?
         */
        if (rank == LEAF0) {
            MPI_Request reqs[3];
            MPI_Isend(buf, nbytes, MPI_BYTE, 16, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(buf, nbytes, MPI_BYTE, 17, 0, MPI_COMM_WORLD, &reqs[1]);
            MPI_Isend(buf, nbytes, MPI_BYTE, 18, 0, MPI_COMM_WORLD, &reqs[2]);
            MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);
        }
        if (rank == 16)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        if (rank == 17)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        if (rank == 18)
            MPI_Recv(buf, nbytes, MPI_BYTE, LEAF0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

    } else {
        if (rank == 0) fprintf(stderr, "Unknown mode: %s\n", mode);
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
