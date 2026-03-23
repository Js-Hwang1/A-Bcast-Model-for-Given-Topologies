#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Isolated test of the group-level two-tree generation.
 * For n entities {ingress, I0, I1, ..., I_{n-3}, egress},
 * we need two anti-correlated trees T0, T1 such that
 * the union degree profile is [2, 4, 4, ..., 4, 2].
 *
 * Tests ngroups = 4, 6, 8, 16.
 */

/* Current implementation (star relay) — known broken for n > 4 */
static void build_current(int ngroups, int root_g, int leaf_g,
                          int *t0_par, int *t1_par)
{
    for (int g = 0; g < ngroups; g++) { t0_par[g] = -1; t1_par[g] = -1; }

    int n_interior = ngroups - 2;
    int interior[64];
    int ni = 0;
    for (int g = 0; g < ngroups; g++)
        if (g != root_g && g != leaf_g) interior[ni++] = g;
    for (int i = 0; i < ni - 1; i++)
        for (int j = i + 1; j < ni; j++)
            if (interior[i] > interior[j])
                { int tmp = interior[i]; interior[i] = interior[j]; interior[j] = tmp; }

    t0_par[interior[0]] = root_g;
    for (int i = 1; i < n_interior; i++)
        t0_par[interior[i]] = interior[0];
    t0_par[leaf_g] = interior[0];

    t1_par[interior[1 % n_interior]] = root_g;
    for (int i = 0; i < n_interior; i++)
        if (i != (1 % n_interior))
            t1_par[interior[i]] = interior[1 % n_interior];
    t1_par[leaf_g] = interior[1 % n_interior];
}

/* Fixed implementation: chain-of-K4 relay.
 *
 * Ordering: ingress, I0, I1, ..., I_{n-3}, egress
 * Let ord[] = [ingress, I0, I1, ..., I_{n-3}, egress]   (n entities)
 *
 * T0 chains forward:  ord[0] → ord[1] → ord[2] → ... → ord[n-1]
 * T1 chains with shift: ord[0] → ord[2] → ord[1] → ord[4] → ord[3] → ... → ord[n-1]
 *   i.e., T1 swaps adjacent interior pairs relative to T0.
 *
 * For n=4: T0: 0→1→2→3, T1: 0→2→1→3
 *   Union degrees: 0:2, 1:4, 2:4, 3:2. Correct!
 *
 * For n=6: T0: 0→1→2→3→4→5, T1: 0→2→1→4→3→5
 *   Union degrees: 0:2, 1:4, 2:4, 3:4, 4:4, 5:2. Correct!
 */
static void build_chain(int ngroups, int root_g, int leaf_g,
                        int *t0_par, int *t1_par)
{
    for (int g = 0; g < ngroups; g++) { t0_par[g] = -1; t1_par[g] = -1; }

    /* Build ordering: [root_g, interior[0], interior[1], ..., leaf_g] */
    int ord[64];
    int n = 0;
    ord[n++] = root_g;
    /* Collect interior groups sorted ascending */
    int interior[64];
    int ni = 0;
    for (int g = 0; g < ngroups; g++)
        if (g != root_g && g != leaf_g) interior[ni++] = g;
    for (int i = 0; i < ni - 1; i++)
        for (int j = i + 1; j < ni; j++)
            if (interior[i] > interior[j])
                { int tmp = interior[i]; interior[i] = interior[j]; interior[j] = tmp; }
    for (int i = 0; i < ni; i++)
        ord[n++] = interior[i];
    ord[n++] = leaf_g;
    /* n == ngroups */

    /* T0: straight chain  ord[0] → ord[1] → ord[2] → ... → ord[n-1] */
    for (int i = 1; i < n; i++)
        t0_par[ord[i]] = ord[i - 1];

    /* T1: swap adjacent interior pairs.
     * Interior positions are 1..n-2.
     * Swap pairs: (1,2), (3,4), (5,6), ...
     * Build permuted order for T1. */
    int perm[64];
    perm[0] = ord[0];           /* root stays */
    perm[n - 1] = ord[n - 1];  /* leaf stays */
    for (int i = 1; i < n - 1; i += 2) {
        if (i + 1 < n - 1) {
            perm[i]     = ord[i + 1];
            perm[i + 1] = ord[i];
        } else {
            perm[i] = ord[i];  /* odd leftover, no swap partner */
        }
    }
    for (int i = 1; i < n; i++)
        t1_par[perm[i]] = perm[i - 1];
}

static void print_and_check(const char *label, int ngroups, int root_g, int leaf_g,
                            int *t0_par, int *t1_par)
{
    printf("\n=== %s (ngroups=%d, root=%d, leaf=%d) ===\n", label, ngroups, root_g, leaf_g);

    printf("T0: ");
    for (int g = 0; g < ngroups; g++) {
        if (t0_par[g] >= 0) printf("%d←%d ", g, t0_par[g]);
    }
    printf("\nT1: ");
    for (int g = 0; g < ngroups; g++) {
        if (t1_par[g] >= 0) printf("%d←%d ", g, t1_par[g]);
    }
    printf("\n");

    /* Compute union degree */
    int deg[64];
    memset(deg, 0, sizeof(deg));
    for (int g = 0; g < ngroups; g++) {
        if (t0_par[g] >= 0) { deg[g]++; deg[t0_par[g]]++; }
        if (t1_par[g] >= 0) { deg[g]++; deg[t1_par[g]]++; }
    }

    printf("Union degrees: ");
    int ok = 1;
    for (int g = 0; g < ngroups; g++) {
        printf("%d:%d ", g, deg[g]);
        if (g == root_g || g == leaf_g) {
            if (deg[g] != 2) ok = 0;
        } else {
            if (deg[g] != 4) ok = 0;
        }
    }
    printf("\nExpected: [2, 4, 4, ..., 4, 2] → %s\n", ok ? "PASS" : "FAIL");

    for (int t = 0; t < 2; t++) {
        int *par = (t == 0) ? t0_par : t1_par;
        int edge_count = 0;
        for (int g = 0; g < ngroups; g++)
            if (par[g] >= 0) edge_count++;
        int reachable = 1;
        for (int g = 0; g < ngroups; g++) {
            if (g == root_g) continue;
            int cur = g, steps = 0;
            while (cur != root_g && steps < ngroups) {
                if (par[cur] < 0) { reachable = 0; break; }
                cur = par[cur];
                steps++;
            }
            if (cur != root_g) reachable = 0;
        }
        printf("T%d: %d edges, reachable=%s\n", t, edge_count,
               reachable ? "yes" : "NO");
    }
}

int main(void)
{
    int sizes[] = {4, 6, 8, 16};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < nsizes; s++) {
        int ng = sizes[s];
        int root_g = 0;
        int leaf_g = ng - 1;
        int t0[64], t1[64];

        build_current(ng, root_g, leaf_g, t0, t1);
        print_and_check("current (star)", ng, root_g, leaf_g, t0, t1);

        build_chain(ng, root_g, leaf_g, t0, t1);
        print_and_check("chain-of-K4", ng, root_g, leaf_g, t0, t1);
    }

    /* Also test with non-zero root */
    printf("\n\n--- Non-zero root tests ---\n");
    {
        int ng = 8, root_g = 3, leaf_g = 7;
        int t0[64], t1[64];
        build_chain(ng, root_g, leaf_g, t0, t1);
        print_and_check("chain-of-K4", ng, root_g, leaf_g, t0, t1);
    }
    {
        int ng = 16, root_g = 7, leaf_g = 15;
        int t0[64], t1[64];
        build_chain(ng, root_g, leaf_g, t0, t1);
        print_and_check("chain-of-K4", ng, root_g, leaf_g, t0, t1);
    }

    return 0;
}
