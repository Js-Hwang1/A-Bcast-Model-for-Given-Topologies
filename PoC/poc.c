/*
 * poc.c -- Router topology PoC experiments for SimGrid SMPI
 *
 * Tests bandwidth sharing, NIC contention, and uplink contention
 * in router-based topologies (star, dual-switch, fat-tree).
 *
 * Usage:
 *   smpirun -np N -platform <xml> -hostfile <hf> \
 *     --cfg=smpi/simulate-computation:no --cfg=smpi/display-timing:yes \
 *     --cfg=network/crosstraffic:0 --cfg=smpi/wos:0 \
 *     --log=root.thres:warning \
 *     ./poc <mode> <msg_bytes>
 *
 * Modes (4-node platforms: star, chain):
 *   p2p         rank 0 -> rank 1                (single flow)
 *   pair        rank 0->1, 2->3                 (independent pairs)
 *   fanout3     rank 0 -> {1,2,3}               (NIC fan-out)
 *   fanin3      rank {1,2,3} -> 0               (NIC fan-in)
 *
 * Modes (8-node platforms: dual_switch, fattree):
 *   p2p         rank 0 -> rank 1                (intra-switch single)
 *   pair        rank 0->1, 2->3                 (intra-switch pairs)
 *   fanout3     rank 0 -> {1,2,3}               (intra-switch fan-out)
 *   cross1      rank 0 -> 4                     (inter-switch single)
 *   cross2      rank 0->4, 1->5                 (inter-switch 2 flows)
 *   cross4      rank 0->4, 1->5, 2->6, 3->7    (inter-switch 4 flows)
 *   mixed       rank 0->1 (intra) + 2->6 (inter)(mixed: no shared links)
 *   fanout_x2   rank 0 -> {4,5}                 (cross-switch fan-out)
 *   fanout_x4   rank 0 -> {4,5,6,7}             (cross-switch fan-out 4)
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void do_send(void *buf, int count, int dest, MPI_Request *req)
{
    MPI_Isend(buf, count, MPI_BYTE, dest, 0, MPI_COMM_WORLD, req);
}

static void do_recv(void *buf, int count, int src, MPI_Request *req)
{
    MPI_Irecv(buf, count, MPI_BYTE, src, 0, MPI_COMM_WORLD, req);
}

/* ---------- mode implementations ---------- */

/* Single point-to-point: rank 0 -> rank 1 */
static void mode_p2p(int rank, void *buf, int n)
{
    MPI_Request req;
    if (rank == 0) {
        do_send(buf, n, 1, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        do_recv(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

/* Independent pairs: 0->1, 2->3 */
static void mode_pair(int rank, void *buf, int n)
{
    MPI_Request req;
    if (rank == 0) {
        do_send(buf, n, 1, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        do_recv(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 2) {
        do_send(buf, n, 3, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 3) {
        do_recv(buf, n, 2, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

/* Fan-out: rank 0 -> {1,2,3} */
static void mode_fanout3(int rank, void *buf, int n)
{
    if (rank == 0) {
        MPI_Request reqs[3];
        do_send(buf, n, 1, &reqs[0]);
        do_send(buf, n, 2, &reqs[1]);
        do_send(buf, n, 3, &reqs[2]);
        MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);
    } else if (rank >= 1 && rank <= 3) {
        MPI_Request req;
        do_recv(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

/* 1 out + 2 in on rank 0: 0->1, 2->0, 3->0 */
static void mode_mix_inout(int rank, void *buf, int n)
{
    if (rank == 0) {
        MPI_Request reqs[3];
        do_send(buf, n, 1, &reqs[0]);   /* out: 0->1 */
        do_recv(buf, n, 2, &reqs[1]);   /* in:  2->0 */
        do_recv(buf, n, 3, &reqs[2]);   /* in:  3->0 */
        MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);
    } else if (rank == 1) {
        MPI_Request req;
        do_recv(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 2) {
        MPI_Request req;
        do_send(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 3) {
        MPI_Request req;
        do_send(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

/* Fan-in: ranks {1,2,3} -> rank 0 */
static void mode_fanin3(int rank, void *buf, int n)
{
    if (rank == 0) {
        MPI_Request reqs[3];
        do_recv(buf, n, 1, &reqs[0]);
        do_recv(buf, n, 2, &reqs[1]);
        do_recv(buf, n, 3, &reqs[2]);
        MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);
    } else if (rank >= 1 && rank <= 3) {
        MPI_Request req;
        do_send(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

/* Cross-switch single: 0 -> 4 */
static void mode_cross1(int rank, void *buf, int n)
{
    MPI_Request req;
    if (rank == 0) {
        do_send(buf, n, 4, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 4) {
        do_recv(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

/* Cross-switch pair: 0->4, 1->5 */
static void mode_cross2(int rank, void *buf, int n)
{
    MPI_Request req;
    if (rank == 0) {
        do_send(buf, n, 4, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        do_send(buf, n, 5, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 4) {
        do_recv(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 5) {
        do_recv(buf, n, 1, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

/* Cross-switch quad: 0->4, 1->5, 2->6, 3->7 */
static void mode_cross4(int rank, void *buf, int n)
{
    MPI_Request req;
    if (rank < 4) {
        do_send(buf, n, rank + 4, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else {
        do_recv(buf, n, rank - 4, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

/* Mixed: rank 0->1 (intra) + rank 2->6 (inter) */
static void mode_mixed(int rank, void *buf, int n)
{
    MPI_Request req;
    if (rank == 0) {
        do_send(buf, n, 1, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
        do_recv(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 2) {
        do_send(buf, n, 6, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else if (rank == 6) {
        do_recv(buf, n, 2, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

/* Fan-out across switches: 0 -> {4, 5} */
static void mode_fanout_x2(int rank, void *buf, int n)
{
    if (rank == 0) {
        MPI_Request reqs[2];
        do_send(buf, n, 4, &reqs[0]);
        do_send(buf, n, 5, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    } else if (rank == 4 || rank == 5) {
        MPI_Request req;
        do_recv(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

/* Fan-out across switches: 0 -> {4, 5, 6, 7} */
static void mode_fanout_x4(int rank, void *buf, int n)
{
    if (rank == 0) {
        MPI_Request reqs[4];
        do_send(buf, n, 4, &reqs[0]);
        do_send(buf, n, 5, &reqs[1]);
        do_send(buf, n, 6, &reqs[2]);
        do_send(buf, n, 7, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    } else if (rank >= 4) {
        MPI_Request req;
        do_recv(buf, n, 0, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

/* ---------- main ---------- */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: poc <mode> <msg_bytes>\n");
        MPI_Finalize();
        return 1;
    }

    const char *mode = argv[1];
    int msg_bytes = atoi(argv[2]);
    void *buf = calloc(1, msg_bytes);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Dispatch */
    if      (strcmp(mode, "p2p") == 0)        mode_p2p(rank, buf, msg_bytes);
    else if (strcmp(mode, "pair") == 0)       mode_pair(rank, buf, msg_bytes);
    else if (strcmp(mode, "fanout3") == 0)    mode_fanout3(rank, buf, msg_bytes);
    else if (strcmp(mode, "fanin3") == 0)     mode_fanin3(rank, buf, msg_bytes);
    else if (strcmp(mode, "mix_inout") == 0)  mode_mix_inout(rank, buf, msg_bytes);
    else if (strcmp(mode, "cross1") == 0)     mode_cross1(rank, buf, msg_bytes);
    else if (strcmp(mode, "cross2") == 0)     mode_cross2(rank, buf, msg_bytes);
    else if (strcmp(mode, "cross4") == 0)     mode_cross4(rank, buf, msg_bytes);
    else if (strcmp(mode, "mixed") == 0)      mode_mixed(rank, buf, msg_bytes);
    else if (strcmp(mode, "fanout_x2") == 0)  mode_fanout_x2(rank, buf, msg_bytes);
    else if (strcmp(mode, "fanout_x4") == 0)  mode_fanout_x4(rank, buf, msg_bytes);
    else {
        if (rank == 0)
            fprintf(stderr, "Unknown mode: %s\n", mode);
        free(buf);
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("%-12s  msg=%-10d  time=%.3f us\n",
               mode, msg_bytes, (t1 - t0) * 1e6);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
