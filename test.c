/*
 * Test: Does pre-posting Irecvs cause SimGrid to split bandwidth
 *       when sends are sequential?
 *
 * Setup: 3 nodes  0 -- 1 -- 2
 *        link 1->2 has 10 GBps bandwidth
 *
 * Test A (sequential sends):
 *   - Node 2 pre-posts K Irecvs
 *   - Node 1 sends K chunks SEQUENTIALLY (blocking MPI_Send)
 *   - If BW is NOT shared: each chunk takes ~msg/BW time
 *   - If BW IS shared:     each chunk takes ~K*msg/BW time
 *
 * Test B (concurrent sends):
 *   - Node 2 pre-posts K Irecvs
 *   - Node 1 fires K Isends ALL AT ONCE (non-blocking)
 *   - BW should be shared K ways -> takes ~K*msg/BW total
 *
 * Test C (baseline - single send):
 *   - One send, one recv. Establishes the single-transfer time.
 *
 * Expected: Test A per-chunk time ≈ Test C time (no sharing)
 *           Test B total time ≈ K * Test C time (sharing)
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K       8           /* number of chunks */
#define MSGSIZE (1024*1024) /* 1 MB per chunk */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0) fprintf(stderr, "Need exactly 3 ranks\n");
        MPI_Finalize();
        return 1;
    }

    char *buf = calloc(K * MSGSIZE, 1);

    /* ================================================================
     * TEST C: Baseline - single 1 MB send from 1 -> 2
     * ================================================================ */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
        double t0 = MPI_Wtime();
        MPI_Send(buf, MSGSIZE, MPI_BYTE, 2, 99, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        printf("[Test C] Baseline single send: %.6f ms\n", (t1 - t0) * 1e3);
    } else if (rank == 2) {
        MPI_Recv(buf, MSGSIZE, MPI_BYTE, 1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ================================================================
     * TEST A: Pre-post K Irecvs, then node 1 sends SEQUENTIALLY
     * ================================================================ */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 2) {
        /* Pre-post ALL K Irecvs before node 1 starts sending */
        MPI_Request reqs[K];
        for (int i = 0; i < K; i++) {
            MPI_Irecv(buf + i * MSGSIZE, MSGSIZE, MPI_BYTE,
                       1, 100 + i, MPI_COMM_WORLD, &reqs[i]);
        }
        /* Signal node 1 that Irecvs are posted */
        MPI_Send(buf, 1, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        /* Wait for all to complete */
        MPI_Waitall(K, reqs, MPI_STATUSES_IGNORE);
    } else if (rank == 1) {
        /* Wait for node 2 to confirm Irecvs are posted */
        MPI_Recv(buf, 1, MPI_BYTE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double t0 = MPI_Wtime();
        for (int i = 0; i < K; i++) {
            double ts = MPI_Wtime();
            MPI_Send(buf + i * MSGSIZE, MSGSIZE, MPI_BYTE,
                     2, 100 + i, MPI_COMM_WORLD);
            double te = MPI_Wtime();
            printf("[Test A] Chunk %d: %.6f ms\n", i, (te - ts) * 1e3);
        }
        double t1 = MPI_Wtime();
        printf("[Test A] TOTAL sequential send (%d chunks): %.6f ms\n",
               K, (t1 - t0) * 1e3);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ================================================================
     * TEST B: Pre-post K Irecvs, then node 1 fires ALL Isends at once
     * ================================================================ */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 2) {
        MPI_Request reqs[K];
        for (int i = 0; i < K; i++) {
            MPI_Irecv(buf + i * MSGSIZE, MSGSIZE, MPI_BYTE,
                       1, 200 + i, MPI_COMM_WORLD, &reqs[i]);
        }
        MPI_Send(buf, 1, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        MPI_Waitall(K, reqs, MPI_STATUSES_IGNORE);
    } else if (rank == 1) {
        MPI_Recv(buf, 1, MPI_BYTE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Request sreqs[K];
        double t0 = MPI_Wtime();
        for (int i = 0; i < K; i++) {
            MPI_Isend(buf + i * MSGSIZE, MSGSIZE, MPI_BYTE,
                      2, 200 + i, MPI_COMM_WORLD, &sreqs[i]);
        }
        MPI_Waitall(K, sreqs, MPI_STATUSES_IGNORE);
        double t1 = MPI_Wtime();
        printf("[Test B] TOTAL concurrent send (%d chunks): %.6f ms\n",
               K, (t1 - t0) * 1e3);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ================================================================
     * Summary (rank 1 prints)
     * ================================================================ */
    if (rank == 1) {
        printf("\n--- Expected behavior ---\n");
        printf("Link BW = 10 GBps, chunk = 1 MB\n");
        printf("Single transfer time ~ 1MB / 10GBps = 0.1 ms\n");
        printf("Test A (seq sends, pre-posted Irecvs): each chunk ~ 0.1 ms, total ~ %.1f ms\n",
               K * 0.1);
        printf("Test B (concurrent sends):             total ~ %.1f ms (BW shared %d ways)\n",
               K * 0.1, K);
        printf("If Test A total ≈ Test B total => pre-posted Irecvs DO cause sharing\n");
        printf("If Test A total ≈ Test C * %d   => NO sharing (sequential = full BW each)\n", K);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
