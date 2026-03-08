#!/usr/bin/env python3
"""
topo_preprocess.py — Parse SimGrid XML platform file, compute pairwise
latencies and first-link IDs via Floyd-Warshall, write .tdat binary.

The .tdat file is consumed by runner.c's data-driven broadcast tree
builder (run_test with topo_data_t).

Key insight: the flat-vs-hierarchical distinction is a DATA property,
not an algorithm property.  When two children of a node share the same
first outgoing physical link, their sends contend.  The first-link
matrix (flink) encodes this automatically for any topology.

Usage:
    python3 topo_preprocess.py <platform.xml> [output.tdat]

    If output.tdat is omitted, writes to <platform>.tdat (replacing .xml).
    Also writes a .tdat.txt human-readable debug summary.
"""

import sys
import os
import struct
import time
import numpy as np
import xml.etree.ElementTree as ET

MAGIC = b"TDAT"
VERSION = 2


# ── Unit parsers ─────────────────────────────────────────────────────

def parse_bw(s):
    """Parse SimGrid bandwidth string -> bytes/sec."""
    s = s.strip()
    for suffix, mult in [("GBps", 1e9), ("MBps", 1e6), ("kBps", 1e3),
                          ("Bps", 1.0)]:
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


def parse_lat(s):
    """Parse SimGrid latency string -> seconds."""
    s = s.strip()
    for suffix, mult in [("ns", 1e-9), ("us", 1e-6), ("ms", 1e-3),
                          ("s", 1.0)]:
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


# ── XML parser ───────────────────────────────────────────────────────

def parse_xml(path):
    """Extract hosts, links, routes from SimGrid platform XML.

    Returns:
        hosts:  list of host ID strings
        links:  dict  link_id -> (bw_bytes_per_sec, lat_sec)
        routes: list of (src_id, dst_id, [link_ids])
    """
    tree = ET.parse(path)
    root = tree.getroot()

    hosts = []
    links = {}
    routes = []

    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

        if tag == 'host':
            hosts.append(elem.get('id'))
        elif tag == 'link':
            lid = elem.get('id')
            bw = parse_bw(elem.get('bandwidth'))
            lat = parse_lat(elem.get('latency'))
            links[lid] = (bw, lat)
        elif tag == 'route':
            src = elem.get('src')
            dst = elem.get('dst')
            link_ids = []
            for child in elem:
                ctag = child.tag.split('}')[-1] if '}' in child.tag \
                       else child.tag
                if ctag == 'link_ctn':
                    link_ids.append(child.get('id'))
            routes.append((src, dst, link_ids))

    return hosts, links, routes


# ── Floyd-Warshall + first-link tracking ─────────────────────────────

def build_and_solve(hosts, links, routes):
    """Build adjacency on all hosts, run Floyd-Warshall, project to
    compute nodes.

    Returns:
        compute_names : list of compute-node name strings (sorted by index)
        lat_matrix    : N x N float32  pairwise latency (seconds)
        flink_matrix  : N x N uint16   first-link ID per route
        bw_min        : float  minimum link bandwidth (bytes/sec)
        lat_base      : float  minimum link latency (seconds)
        num_links     : int    number of unique links
    """
    # Assign numeric indices to hosts and links
    host_idx = {h: i for i, h in enumerate(hosts)}
    link_names = sorted(links.keys())
    # link IDs are 1-based; 0 means "no link / self"
    link_idx = {name: i + 1 for i, name in enumerate(link_names)}
    H = len(hosts)

    # Initialise distance and first-link matrices
    dist = np.full((H, H), np.inf, dtype=np.float64)
    flink = np.zeros((H, H), dtype=np.int32)
    np.fill_diagonal(dist, 0.0)

    # Fill direct edges from declared routes
    for src, dst, link_list in routes:
        if src not in host_idx or dst not in host_idx:
            continue
        si, di = host_idx[src], host_idx[dst]

        total_lat = sum(links[lid][1] for lid in link_list)
        # First link from src toward dst
        first_lid = link_idx[link_list[0]] if link_list else 0
        # Last link of forward path = first link of reverse direction
        last_lid = link_idx[link_list[-1]] if link_list else 0

        if total_lat < dist[si, di]:
            dist[si, di] = total_lat
            flink[si, di] = first_lid
        if total_lat < dist[di, si]:
            dist[di, si] = total_lat
            flink[di, si] = last_lid

    # ── Floyd-Warshall ──
    print(f"  Floyd-Warshall on {H} hosts ...", end="", flush=True)
    t0 = time.time()

    for k in range(H):
        # dist_through_k[i][j] = dist[i][k] + dist[k][j]
        dk_col = dist[:, k].copy()          # (H,)
        dk_row = dist[k, :].copy()          # (H,)
        new_dist = dk_col.reshape(-1, 1) + dk_row.reshape(1, -1)   # (H,H)

        mask = new_dist < dist
        dist[mask] = new_dist[mask]

        # First-link update: flink[i][j] = flink[i][k]
        fk = flink[:, k].copy()             # (H,)
        rows = np.nonzero(mask)[0]           # row indices of updated cells
        flink[mask] = fk[rows]

    t1 = time.time()
    print(f" done in {t1 - t0:.1f}s")

    # ── Project to compute nodes (those named "node-*") ──
    compute_entries = []
    for i, h in enumerate(hosts):
        if h.startswith("node-"):
            num = int(h.split('-')[1])
            compute_entries.append((num, i))
    compute_entries.sort()                   # sort by node number
    compute_names = [f"node-{num}" for num, _ in compute_entries]
    compute_host_idx = [hi for _, hi in compute_entries]
    N = len(compute_names)

    idx = np.array(compute_host_idx)
    lat_matrix = dist[np.ix_(idx, idx)].astype(np.float32)
    flink_matrix = flink[np.ix_(idx, idx)].astype(np.uint16)

    bw_min = min(bw for bw, _lat in links.values())
    lat_base = min(lat for _bw, lat in links.values())
    num_links = len(links)

    return compute_names, lat_matrix, flink_matrix, bw_min, lat_base, num_links


# ── Spectral analysis ────────────────────────────────────────────────

def spectral_analysis(lat_matrix, lat_base):
    """Build graph Laplacian from latency matrix and compute eigenvectors.

    Returns:
        eigenvalues  : (num_ev,) float64 — λ₂..λ_{k+1} (ascending)
        eigenvectors : (num_ev, N) float64 — corresponding eigenvectors
    """
    N = lat_matrix.shape[0]
    num_ev = int(np.ceil(np.log2(N)))

    # Weighted adjacency: W[i][j] = lat_base / lat[i][j] for i≠j
    # 1-hop neighbors get weight ~1.0, distant nodes get smaller weights
    lat64 = lat_matrix.astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        W = np.where(lat64 > 0, lat_base / lat64, 0.0)
    np.fill_diagonal(W, 0.0)

    # Graph Laplacian: L = D - W
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Eigendecomposition (symmetric → real eigenvalues, sorted ascending)
    print(f"  Eigendecomposition ({N}x{N}) ...", end="", flush=True)
    t0 = time.time()
    eigvals, eigvecs_cols = np.linalg.eigh(L)
    t1 = time.time()
    print(f" done in {t1 - t0:.1f}s")

    # Skip λ₁≈0 (trivial), take next num_ev eigenvectors
    eigenvalues = eigvals[1:num_ev + 1].astype(np.float64)
    # eigvecs_cols[:,k] is the k-th eigenvector (column); store as row-major
    eigenvectors = eigvecs_cols[:, 1:num_ev + 1].T.astype(np.float64)

    return eigenvalues, eigenvectors


# ── Binary writer ────────────────────────────────────────────────────

def write_tdat(path, N, num_links, bw_min, lat_base,
               lat_matrix, flink_matrix,
               eigenvalues=None, eigenvectors=None):
    """Write .tdat binary file (v2 with spectral data).

    Layout (little-endian):
        Header  (32 bytes):
            char[4]  magic    "TDAT"
            uint32   version  2
            uint32   N        number of compute nodes
            uint32   num_links
            float64  bw_min   min bottleneck bandwidth (B/s)
            float64  lat_base min link latency (s)
        Data:
            float32[N*N]  lat    pairwise latency matrix
            uint16[N*N]   flink  first-link ID matrix
        Spectral data (v2):
            uint32            num_ev       number of eigenvectors
            float64[num_ev]   eigenvalues  λ₂..λ_{k+1}
            float64[num_ev*N] eigenvectors row-major: ev[k][i]
    """
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', N))
        f.write(struct.pack('<I', num_links))
        f.write(struct.pack('<d', bw_min))
        f.write(struct.pack('<d', lat_base))
        f.write(lat_matrix.astype('<f4').tobytes())
        f.write(flink_matrix.astype('<u2').tobytes())
        # Spectral data
        num_ev = len(eigenvalues) if eigenvalues is not None else 0
        f.write(struct.pack('<I', num_ev))
        if num_ev > 0:
            f.write(eigenvalues.astype('<f8').tobytes())
            f.write(eigenvectors.astype('<f8').tobytes())


# ── Debug text writer ────────────────────────────────────────────────

def write_debug(path, compute_names, lat_matrix, flink_matrix,
                bw_min, lat_base,
                eigenvalues=None, eigenvectors=None):
    """Write human-readable debug summary alongside .tdat."""
    N = len(compute_names)
    with open(path, 'w') as f:
        f.write(f"N        = {N}\n")
        f.write(f"bw_min   = {bw_min:.6e} B/s\n")
        f.write(f"lat_base = {lat_base:.6e} s\n\n")

        # Sample latencies (first 8 nodes)
        n = min(N, 8)
        f.write("Pairwise latencies (first 8 nodes, seconds):\n")
        hdr = "        " + "".join(f"{'n-'+str(j):>12s}" for j in range(n))
        f.write(hdr + "\n")
        for i in range(n):
            row = f"  n-{i:<4d}"
            for j in range(n):
                row += f"{lat_matrix[i, j]:12.3e}"
            f.write(row + "\n")

        f.write("\nFirst-link IDs (first 8 nodes):\n")
        hdr = "        " + "".join(f"{'n-'+str(j):>8s}" for j in range(n))
        f.write(hdr + "\n")
        for i in range(n):
            row = f"  n-{i:<4d}"
            for j in range(n):
                row += f"{flink_matrix[i, j]:8d}"
            f.write(row + "\n")

        # Contention analysis for node-0
        f.write("\nFirst-link contention from node-0:\n")
        fl0 = flink_matrix[0, 1:]   # skip self
        unique, counts = np.unique(fl0, return_counts=True)
        for lid, cnt in sorted(zip(unique, counts),
                                key=lambda x: -x[1]):
            f.write(f"  link ID {lid:4d}: {cnt:4d} destinations\n")
        if len(counts) > 0:
            f.write(f"  Max contention group = {max(counts)}\n")
            f.write(f"  Num distinct first-links = {len(unique)}\n")
        else:
            f.write("  (no destinations)\n")

        # Spectral analysis summary
        if eigenvalues is not None and len(eigenvalues) > 0:
            f.write(f"\nSpectral analysis ({len(eigenvalues)} eigenvectors):\n")
            f.write(f"  Eigenvalues (λ₂..λ_{len(eigenvalues)+1}):\n")
            for i, ev in enumerate(eigenvalues):
                f.write(f"    λ_{i+2:<3d} = {ev:.6e}\n")
            if len(eigenvalues) >= 2:
                gap_ratio = eigenvalues[1] / eigenvalues[0] \
                    if eigenvalues[0] > 1e-12 else float('inf')
                f.write(f"  Spectral gap ratio (λ₃/λ₂) = {gap_ratio:.4f}\n")
                f.write(f"    (small → sharp communities, "
                        f"large → uniform/flat)\n")
            f.write(f"\n  Fiedler vector (v₂) sample (first {min(N, 16)} nodes):\n")
            fiedler = eigenvectors[0]  # first row = v₂
            for i in range(min(N, 16)):
                f.write(f"    node-{i:<4d}: {fiedler[i]:+.6f}\n")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 topo_preprocess.py <platform.xml> "
              "[output.tdat]", file=sys.stderr)
        sys.exit(1)

    xml_path = sys.argv[1]
    if len(sys.argv) > 2:
        tdat_path = sys.argv[2]
    else:
        tdat_path = os.path.splitext(xml_path)[0] + '.tdat'
    debug_path = tdat_path + '.txt'

    print(f"Parsing {xml_path} ...")
    hosts, links, routes = parse_xml(xml_path)
    print(f"  {len(hosts)} hosts, {len(links)} links, "
          f"{len(routes)} routes")

    compute_names, lat_matrix, flink_matrix, bw_min, lat_base, \
        num_links = build_and_solve(hosts, links, routes)
    N = len(compute_names)
    print(f"  {N} compute nodes")
    print(f"  bw_min   = {bw_min:.6e} B/s")
    print(f"  lat_base = {lat_base:.6e} s")

    eigenvalues, eigenvectors = spectral_analysis(lat_matrix, lat_base)
    print(f"  {len(eigenvalues)} eigenvectors (ceil(log2({N})))")
    print(f"  λ₂ = {eigenvalues[0]:.6e}  (Fiedler value)")
    if len(eigenvalues) >= 2:
        gap = eigenvalues[1] / eigenvalues[0] \
            if eigenvalues[0] > 1e-12 else float('inf')
        print(f"  λ₃/λ₂ = {gap:.4f}  (spectral gap ratio)")

    write_tdat(tdat_path, N, num_links, bw_min, lat_base,
               lat_matrix, flink_matrix, eigenvalues, eigenvectors)
    sz = os.path.getsize(tdat_path)
    print(f"  Wrote {tdat_path} ({sz:,} bytes)")

    write_debug(debug_path, compute_names, lat_matrix, flink_matrix,
                bw_min, lat_base, eigenvalues, eigenvectors)
    print(f"  Wrote {debug_path}")


if __name__ == '__main__':
    main()
