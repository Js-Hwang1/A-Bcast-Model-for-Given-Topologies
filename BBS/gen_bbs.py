#!/usr/bin/env python3
"""
gen_bbs.py — Generate .bbs files from the new tree algorithm.

Usage:
    python3 BBS/gen_bbs.py --rows 8 --cols 16 --roots 0,64,127 \
        --tdat topo/2Dmesh/platform_2dmesh_8x16.tdat
"""

import argparse
import struct
import sys
import os

# Import from visualize_trees.py (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_trees import build_mesh, bbs_compute_trees


def write_bbs(path, N, tau, root, trees):
    """Write a .bbs binary file from parent arrays.

    trees: list of tau parent arrays, each length N (-1 = root).
    """
    # Build children lists
    children = [[[] for _ in range(N)] for _ in range(tau)]
    for t in range(tau):
        for i in range(N):
            p = trees[t][i]
            if p >= 0:
                children[t][p].append(i)

    # Compute node sections
    sections = []
    for i in range(N):
        sec = b''
        for t in range(tau):
            p = trees[t][i] if trees[t][i] >= 0 else -1
            nc = len(children[t][i])
            sec += struct.pack('<h', p)
            sec += struct.pack('<H', nc)
            for c in children[t][i]:
                sec += struct.pack('<h', c)
        sections.append(sec)

    # Header: magic(2) + N(2) + tau(1) + pad(1) + root(2) = 8 bytes
    # Offset table: N * 4 bytes
    header_size = 8 + N * 4
    offsets = []
    cur = header_size
    for sec in sections:
        offsets.append(cur)
        cur += len(sec)

    with open(path, 'wb') as f:
        f.write(struct.pack('<H', 0xBB50))       # magic
        f.write(struct.pack('<H', N))             # N
        f.write(struct.pack('<B', tau))            # tau
        f.write(struct.pack('<B', 0))             # reserved
        f.write(struct.pack('<H', root))           # root
        for off in offsets:
            f.write(struct.pack('<I', off))
        for sec in sections:
            f.write(sec)


def main():
    parser = argparse.ArgumentParser(
        description='Generate .bbs files using the new tree algorithm')
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--cols', type=int, required=True)
    parser.add_argument('--roots', type=str, required=True,
                        help='Comma-separated root list or "all"')
    parser.add_argument('--tdat', type=str, required=True,
                        help='Path to .tdat file (used for output naming)')
    args = parser.parse_args()

    rows, cols = args.rows, args.cols
    N = rows * cols
    tdat_base = args.tdat
    if tdat_base.endswith('.tdat'):
        tdat_base = tdat_base[:-5]

    # Parse roots
    if args.roots == 'all':
        roots = list(range(N))
    else:
        roots = [int(r) for r in args.roots.split(',')]

    print(f"Building {rows}x{cols} mesh ({N} nodes)...")
    mesh_data = build_mesh(rows, cols)
    N, adj, lat, flink, path_len, path_links, max_hops, pos, link_id = mesh_data

    for root in roots:
        assert 0 <= root < N, f"root {root} out of range"
        print(f"\n--- Root {root} ---")
        tau, trees = bbs_compute_trees(N, adj, lat, flink, path_len,
                                        path_links, max_hops, root)
        bbs_path = f"{tdat_base}_R{root}.bbs"
        write_bbs(bbs_path, N, tau, root, trees)
        print(f"Wrote {bbs_path} (tau={tau})")


if __name__ == '__main__':
    main()
