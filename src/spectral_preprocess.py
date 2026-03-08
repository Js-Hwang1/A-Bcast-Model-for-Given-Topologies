#!/usr/bin/env python3
"""
spectral_preprocess.py — Full-graph sparse Laplacian for spectral broadcast.

Uses the SPARSE adjacency of the FULL graph (compute nodes + switches/routers,
direct physical links only, weight = 1/latency) to compute the Fiedler vector
(eigenvector of lambda_2).  Also builds a compute-node adjacency matrix for
bridge selection in the recursive bisection tree.

Output: .sdat v2 binary consumed by runner.c's run_spec() function.

Usage:
    python3 spectral_preprocess.py <platform.xml> [output.sdat]

    If output.sdat is omitted, writes to <platform>.sdat (replacing .xml).
    Also writes a .sdat.txt human-readable debug summary.
"""

import sys
import os
import struct
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from topo_preprocess import parse_xml, parse_bw

MAGIC = b"SDAT"
VERSION = 2


# -- Build sparse adjacency of the FULL graph --

def build_sparse_adjacency(hosts, links, routes):
    """Build sparse weighted adjacency of the full graph (all hosts including
    switches/routers), using direct physical links only.

    Weight = 1/latency for each link (higher weight = closer/faster).

    Returns:
        W       : scipy.sparse.csr_matrix (H x H) weighted adjacency
        host_idx: dict  host_id -> index
    """
    host_idx = {h: i for i, h in enumerate(hosts)}
    H = len(hosts)

    rows, cols, weights = [], [], []

    for src, dst, link_list in routes:
        if src not in host_idx or dst not in host_idx:
            continue
        si, di = host_idx[src], host_idx[dst]

        total_lat = sum(links[lid][1] for lid in link_list)
        if total_lat <= 0:
            continue

        w = 1.0 / total_lat

        rows.extend([si, di])
        cols.extend([di, si])
        weights.extend([w, w])

    W = csr_matrix((weights, (rows, cols)), shape=(H, H))
    return W, host_idx


# -- Fiedler vector computation --

def compute_fiedler(W, hosts):
    """Compute graph Laplacian Fiedler vector from sparse adjacency and
    project to compute-node indices.

    Returns:
        compute_host_idx : list of ints — full-graph indices of compute nodes
        fiedler_val      : float64 — lambda_2
        fiedler_vec      : (N,) float64 — Fiedler vector projected to compute nodes
    """
    H = W.shape[0]

    # Graph Laplacian: L = D - W
    degrees = np.array(W.sum(axis=1)).flatten()
    D = csr_matrix((degrees, (range(H), range(H))), shape=(H, H))
    L = D - W

    # Identify compute nodes (node-*) sorted by number
    compute_entries = []
    for i, h in enumerate(hosts):
        if h.startswith("node-"):
            num = int(h.split('-')[1])
            compute_entries.append((num, i))
    compute_entries.sort()
    compute_host_idx = [hi for _, hi in compute_entries]

    # Eigendecomposition: 2 smallest eigenvalues (skip lambda_1 ~ 0)
    k = min(2, H - 1)
    print(f"  Sparse eigendecomposition ({H}x{H}, k={k}) ...",
          end="", flush=True)
    t0 = time.time()
    eigvals, eigvecs_cols = eigsh(L.astype(np.float64), k=k, which='SM')
    t1 = time.time()
    print(f" done in {t1 - t0:.1f}s")

    # Sort by eigenvalue
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs_cols = eigvecs_cols[:, order]

    # Fiedler = second smallest (index 1)
    fiedler_val = eigvals[1].astype(np.float64)
    fiedler_full = eigvecs_cols[:, 1]

    # Project to compute nodes
    fiedler_vec = fiedler_full[compute_host_idx].astype(np.float64)

    # Normalize to unit length
    norm = np.linalg.norm(fiedler_vec)
    if norm > 1e-12:
        fiedler_vec /= norm

    return compute_host_idx, fiedler_val, fiedler_vec


# -- Compute-node adjacency matrix --

def build_compute_adjacency(hosts, routes):
    """Build N×N binary adjacency from the XML routes.

    A route in the XML connects two hosts with a list of links.
    If both endpoints are compute nodes (node-*), they are neighbors.

    Returns:
        adj : (N, N) float32 binary adjacency matrix (0/1, zero diagonal)
    """
    # Sorted compute node names
    compute = sorted([h for h in hosts if h.startswith("node-")],
                     key=lambda h: int(h.split('-')[1]))
    N = len(compute)
    idx = {h: i for i, h in enumerate(compute)}

    adj = np.zeros((N, N), dtype=np.float32)

    for src, dst, _ in routes:
        if src in idx and dst in idx:
            adj[idx[src], idx[dst]] = 1.0
            adj[idx[dst], idx[src]] = 1.0

    np.fill_diagonal(adj, 0.0)
    return adj


# -- Binary writer --

def write_sdat(path, N, bw_min, fiedler_val, fiedler_vec, adj_matrix):
    """Write .sdat binary file (v2).

    Layout (little-endian):
        Header:
            char[4]   magic       "SDAT"
            uint32    version     2
            uint32    N           number of compute nodes
            uint32    num_ev      1 (Fiedler only)
            float64   bw_min      min bottleneck bandwidth (B/s)
        Spectral data:
            float64[1]            fiedler eigenvalue (lambda_2)
            float64[N]            fiedler vector
        Adjacency data:
            float32[N*N]          compute-node adjacency matrix
    """
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', N))
        f.write(struct.pack('<I', 1))       # num_ev = 1
        f.write(struct.pack('<d', bw_min))
        f.write(struct.pack('<d', fiedler_val))
        f.write(fiedler_vec.astype('<f8').tobytes())
        f.write(adj_matrix.astype('<f4').tobytes())


def write_debug(path, compute_names, fiedler_val, fiedler_vec, adj_matrix,
                bw_min):
    """Write human-readable debug summary alongside .sdat."""
    N = len(compute_names)

    with open(path, 'w') as f:
        f.write(f"N        = {N}\n")
        f.write(f"bw_min   = {bw_min:.6e} B/s\n")
        f.write(f"lambda_2 = {fiedler_val:.6e}  (Fiedler value)\n\n")

        # Fiedler vector samples
        f.write(f"Fiedler vector (first {min(N, 16)} nodes):\n")
        for i in range(min(N, 16)):
            f.write(f"  node-{i:<4d}: {fiedler_vec[i]:+.6f}\n")

        # Fiedler statistics
        f.write(f"\n  min = {fiedler_vec.min():+.6f}")
        f.write(f"  max = {fiedler_vec.max():+.6f}")
        unique_vals = len(set(f"{v:.6f}" for v in fiedler_vec))
        f.write(f"  unique values = {unique_vals}/{N}\n")

        # Adjacency summary
        nnz = np.count_nonzero(adj_matrix)
        f.write(f"\nAdjacency matrix:\n")
        f.write(f"  nonzeros = {nnz} / {N*N}"
                f"  (density {nnz / (N * N):.4f})\n")
        deg = (adj_matrix > 0).sum(axis=1)
        f.write(f"  degree: min={deg.min()}, max={deg.max()}, "
                f"mean={deg.mean():.1f}\n")

        # Sample adjacency (first 8 nodes)
        n = min(N, 8)
        f.write(f"\nAdjacency weights (first {n} nodes):\n")
        hdr = "        " + "".join(f"{'n-'+str(j):>12s}" for j in range(n))
        f.write(hdr + "\n")
        for i in range(n):
            row = f"  n-{i:<4d}"
            for j in range(n):
                row += f"{adj_matrix[i, j]:12.3e}"
            f.write(row + "\n")


# -- Main --

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 spectral_preprocess.py <platform.xml> "
              "[output.sdat]", file=sys.stderr)
        sys.exit(1)

    xml_path = sys.argv[1]
    if len(sys.argv) > 2:
        sdat_path = sys.argv[2]
    else:
        sdat_path = os.path.splitext(xml_path)[0] + '.sdat'
    debug_path = sdat_path + '.txt'

    print(f"Parsing {xml_path} ...")
    hosts, links, routes = parse_xml(xml_path)
    print(f"  {len(hosts)} hosts, {len(links)} links, "
          f"{len(routes)} routes")

    # Step 1: Build sparse adjacency of the FULL graph
    print("Building sparse adjacency (full graph, direct links only) ...")
    W, host_idx = build_sparse_adjacency(hosts, links, routes)
    nnz = W.nnz
    H = W.shape[0]
    print(f"  {H} nodes, {nnz} nonzeros "
          f"(density {nnz / (H * H):.4f})")

    # Step 2: Fiedler vector
    print("Computing Fiedler vector ...")
    compute_host_idx, fiedler_val, fiedler_vec = \
        compute_fiedler(W, hosts)
    N = len(compute_host_idx)
    print(f"  {N} compute nodes")
    print(f"  lambda_2 = {fiedler_val:.6e}  (Fiedler value)")

    # Step 3: Compute-node adjacency matrix (direct from XML routes)
    print("Building compute-node adjacency matrix ...")
    adj_matrix = build_compute_adjacency(hosts, routes)
    adj_nnz = np.count_nonzero(adj_matrix)
    print(f"  {adj_nnz} nonzeros (density {adj_nnz / (N * N):.4f})")

    # Get bw_min from link data
    bw_min = min(bw for bw, _lat in links.values())
    compute_names = [f"node-{ce[0]}" for ce in
                     sorted((int(h.split('-')[1]), h) for h in hosts
                            if h.startswith("node-"))]

    # Step 4: Write binary
    write_sdat(sdat_path, N, bw_min, fiedler_val, fiedler_vec, adj_matrix)
    sz = os.path.getsize(sdat_path)
    print(f"  Wrote {sdat_path} ({sz:,} bytes)")

    # Step 5: Write debug
    write_debug(debug_path, compute_names, fiedler_val, fiedler_vec,
                adj_matrix, bw_min)
    print(f"  Wrote {debug_path}")


if __name__ == '__main__':
    main()
