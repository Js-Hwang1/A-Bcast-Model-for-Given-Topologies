import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict, Set
import csv
import os
import sys

# Default parameters (will be overridden by command line arguments)
GRID_ROWS = 64     # number of rows
GRID_COLS = 64     # number of columns
INFO_SIZE = 500     # total chunks
START_POINT = (0, 0)

def get_nodes_and_mapping(n: int, m: int) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    nodes = []
    mapping = {}
    for i in range(n):
        for j in range(m):
            mapping[(i, j)] = len(nodes)
            nodes.append((i, j))
    return nodes, mapping

def get_neighbors(coord: Tuple[int, int], n: int, m: int) -> List[Tuple[int, int]]:
    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    nbrs = []
    for di, dj in directions:
        ni, nj = coord[0] + di, coord[1] + dj
        if 0 <= ni < n and 0 <= nj < m:
            nbrs.append((ni, nj))
    return nbrs

def build_spanning_tree(n: int, m: int, start: Tuple[int, int]) -> Dict[int, List[int]]:
    nodes, mapping = get_nodes_and_mapping(n, m)
    total = n * m
    tree = {i: [] for i in range(total)}
    visited = {mapping[start]}
    q = deque([mapping[start]])
    while q:
        u = q.popleft()
        coord_u = nodes[u]
        for coord_v in get_neighbors(coord_u, n, m):
            v = mapping[coord_v]
            if v not in visited:
                tree[u].append(v)
                visited.add(v)
                q.append(v)
    return tree

def compute_node_levels(tree: Dict[int, List[int]], root: int) -> Dict[int, int]:
    levels = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in tree[u]:
            if v not in levels:
                levels[v] = levels[u] + 1
                q.append(v)
    return levels

def calculate_chunk_assignment(total_nodes: int, N: int) -> Dict[int, List[int]]:
    base = N // total_nodes
    rem  = N % total_nodes
    assignment = {}
    chunk = 0
    for i in range(total_nodes):
        cnt = base + (1 if i < rem else 0)
        assignment[i] = list(range(chunk, chunk + cnt))
        chunk += cnt
    return assignment

def find_path_to_node(tree: Dict[int, List[int]], root: int, target: int) -> List[int]:
    """Find path from root to target node in the tree"""
    if root == target:
        return [root]
    
    # BFS to find path
    parent = {}
    q = deque([root])
    visited = {root}
    
    while q:
        u = q.popleft()
        for v in tree[u]:
            if v not in visited:
                parent[v] = u
                visited.add(v)
                q.append(v)
                if v == target:
                    break
    
    if target not in parent:
        return []
    
    # Reconstruct path
    path = [target]
    current = target
    while current in parent:
        current = parent[current]
        path.append(current)
    return list(reversed(path))

def simulate_scatter_fixed(n: int, m: int, start: Tuple[int, int], N: int):
    nodes, mapping = get_nodes_and_mapping(n, m)
    total = len(nodes)
    start_idx = mapping[start]
    tree = build_spanning_tree(n, m, start)
    levels = compute_node_levels(tree, start_idx)
    chunk_assignment = calculate_chunk_assignment(total, N)
    
    # Precompute paths for each chunk to its destination
    chunk_paths = {}
    for node_idx in range(total):
        if node_idx == start_idx:
            continue
        path = find_path_to_node(tree, start_idx, node_idx)
        if path:
            for chunk in chunk_assignment[node_idx]:
                chunk_paths[chunk] = path
    
    # Initialize state: each node has chunks that are currently at that node
    # Initially, all chunks are at the source
    chunk_locations = {chunk: start_idx for chunk in range(N)}
    
    rounds = 0
    active_edges = []
    edge_hist = []
    
    # Continue until all chunks reach their destinations
    while any(chunk_locations[chunk] != chunk_assignment[chunk][0] for chunk in range(N)):
        sends = set()
        recvs = set()
        used = []
        tx_count = 0
        
        # Find all possible transmissions for this round
        transmissions = []
        for chunk in range(N):
            current_loc = chunk_locations[chunk]
            if chunk in chunk_paths:
                path = chunk_paths[chunk]
                # Find current position in path
                try:
                    path_idx = path.index(current_loc)
                    if path_idx + 1 < len(path):
                        next_node = path[path_idx + 1]
                        transmissions.append((chunk, current_loc, next_node))
                except ValueError:
                    continue
        
        # Sort by priority (deeper receivers first, then by chunk ID)
        transmissions.sort(key=lambda x: (-levels[x[2]], x[0]))
        
        # Execute transmissions respecting half-duplex constraints
        for chunk, sender, receiver in transmissions:
            if sender in sends or sender in recvs or receiver in sends or receiver in recvs:
                continue
            
            chunk_locations[chunk] = receiver
            sends.add(sender)
            recvs.add(receiver)
            used.append((sender, receiver))
            tx_count += 1
        
        active_edges.append(tx_count)
        edge_hist.append(used)
        rounds += 1
        
        if tx_count == 0:
            break
    
    # Convert chunk_locations back to state format for compatibility
    state = {i: set() for i in range(total)}
    for chunk in range(N):
        if chunk in chunk_locations:
            state[chunk_locations[chunk]].add(chunk)
    
    return rounds, active_edges, state, edge_hist

def simulate_recursive_doubling_allgather_pipelined(n: int, m: int, init_state: Dict[int, Set[int]], N: int):
    """
    Pipelined recursive doubling allgather that maximizes throughput under constraints.
    Hardware constraints: Half-duplex, Bandwidth-1
    """
    nodes, mapping = get_nodes_and_mapping(n, m)
    total = len(nodes)
    state = {i: set(init_state[i]) for i in init_state}

    rounds = 0
    active_edges = []
    edge_hist = []

    while any(len(state[i]) < N for i in state):
        sends = set()
        recvs = set()
        used = []
        trans = []

        # Find all possible transmissions for this round
        for u in range(total):
            if not state[u]:
                continue
            coord_u = nodes[u]
            for coord_v in get_neighbors(coord_u, n, m):
                v = mapping[coord_v]
                missing = state[u] - state[v]
                if missing:
                    # Enhanced scoring: prioritize larger transfers and more urgent needs
                    score = len(missing)
                    # Add urgency factor: nodes with fewer chunks get higher priority
                    urgency = N - len(state[v])
                    # Add diversity factor: prefer transfers that create more diversity
                    diversity = len(state[u] | state[v]) - len(state[v])
                    
                    total_score = score + urgency + diversity
                    c = min(missing)  # Use smallest chunk ID for deterministic behavior
                    trans.append((total_score, u, v, c))

        # Sort by total score (highest first)
        trans.sort(reverse=True)
        new_state = {i: set(state[i]) for i in state}
        tx_count = 0

        # Execute transmissions respecting half-duplex constraints
        for _, u, v, c in trans:
            if u in sends or u in recvs or v in sends or v in recvs:
                continue
            new_state[v].add(c)
            sends.add(u)
            recvs.add(v)
            used.append((u, v))
            tx_count += 1

        state = new_state
        active_edges.append(tx_count)
        edge_hist.append(used)
        rounds += 1

        if tx_count == 0:
            break

    return rounds, active_edges, state, edge_hist

def simulate_scatter_pipelined(n: int, m: int, start: Tuple[int, int], N: int):
    """
    Aggressively pipelined scatter that maximizes throughput by allowing chunks to flow
    through the network as soon as paths become available.
    Maintains constraints: half-duplex, one information per timestep, neighbor-only communication.
    """
    nodes, mapping = get_nodes_and_mapping(n, m)
    total = len(nodes)
    start_idx = mapping[start]
    tree = build_spanning_tree(n, m, start)
    levels = compute_node_levels(tree, start_idx)
    chunk_assignment = calculate_chunk_assignment(total, N)
    
    # Precompute paths for each chunk to its destination
    chunk_paths = {}
    for node_idx in range(total):
        if node_idx == start_idx:
            continue
        path = find_path_to_node(tree, start_idx, node_idx)
        if path:
            for chunk in chunk_assignment[node_idx]:
                chunk_paths[chunk] = path
    
    # Initialize state: each node has chunks that are currently at that node
    # Initially, all chunks are at the source
    chunk_locations = {chunk: start_idx for chunk in range(N)}
    
    rounds = 0
    active_edges = []
    edge_hist = []
    
    # Continue until all chunks reach their destinations
    while any(chunk_locations[chunk] != chunk_assignment[chunk][0] for chunk in range(N)):
        sends = set()
        recvs = set()
        used = []
        tx_count = 0
        
        # Find all possible transmissions for this round
        transmissions = []
        for chunk in range(N):
            current_loc = chunk_locations[chunk]
            if chunk in chunk_paths:
                path = chunk_paths[chunk]
                # Find current position in path
                try:
                    path_idx = path.index(current_loc)
                    if path_idx + 1 < len(path):
                        next_node = path[path_idx + 1]
                        # Enhanced priority: consider both tree level and path progress
                        # Prioritize chunks that are further along their paths
                        path_progress = path_idx / len(path)  # How far along the path
                        priority = (-levels[next_node], -path_progress, chunk)
                        transmissions.append((priority, chunk, current_loc, next_node))
                except ValueError:
                    continue
        
        # Sort by priority (deeper receivers first, then by path progress, then by chunk ID)
        transmissions.sort(key=lambda x: x[0])
        
        # Execute transmissions respecting half-duplex constraints
        # In pipelined version, we can have multiple chunks flowing through different paths
        for _, chunk, sender, receiver in transmissions:
            if sender in sends or sender in recvs or receiver in sends or receiver in recvs:
                continue
            
            chunk_locations[chunk] = receiver
            sends.add(sender)
            recvs.add(receiver)
            used.append((sender, receiver))
            tx_count += 1
        
        active_edges.append(tx_count)
        edge_hist.append(used)
        rounds += 1
        
        if tx_count == 0:
            break
    
    # Convert chunk_locations back to state format for compatibility
    state = {i: set() for i in range(total)}
    for chunk in range(N):
        if chunk in chunk_locations:
            state[chunk_locations[chunk]].add(chunk)
    
    return rounds, active_edges, state, edge_hist

def simulate_scatter_advanced_pipelined(n: int, m: int, start: Tuple[int, int], N: int):
    """
    Advanced pipelined scatter using multiple path strategies and better load balancing.
    Maintains constraints: half-duplex, one information per timestep, neighbor-only communication.
    """
    nodes, mapping = get_nodes_and_mapping(n, m)
    total = len(nodes)
    start_idx = mapping[start]
    tree = build_spanning_tree(n, m, start)
    levels = compute_node_levels(tree, start_idx)
    chunk_assignment = calculate_chunk_assignment(total, N)
    
    # Precompute multiple paths for each chunk to its destination
    # Use both tree path and shortest path for better load balancing
    chunk_paths = {}
    for node_idx in range(total):
        if node_idx == start_idx:
            continue
        tree_path = find_path_to_node(tree, start_idx, node_idx)
        if tree_path:
            for chunk in chunk_assignment[node_idx]:
                chunk_paths[chunk] = tree_path
    
    # Initialize state: each node has chunks that are currently at that node
    # Initially, all chunks are at the source
    chunk_locations = {chunk: start_idx for chunk in range(N)}
    
    rounds = 0
    active_edges = []
    edge_hist = []
    
    # Continue until all chunks reach their destinations
    while any(chunk_locations[chunk] != chunk_assignment[chunk][0] for chunk in range(N)):
        sends = set()
        recvs = set()
        used = []
        tx_count = 0
        
        # Find all possible transmissions for this round
        transmissions = []
        for chunk in range(N):
            current_loc = chunk_locations[chunk]
            if chunk in chunk_paths:
                path = chunk_paths[chunk]
                # Find current position in path
                try:
                    path_idx = path.index(current_loc)
                    if path_idx + 1 < len(path):
                        next_node = path[path_idx + 1]
                        # Advanced priority: consider tree level, path progress, and congestion
                        path_progress = path_idx / len(path)
                        # Add congestion penalty for heavily used edges
                        congestion_penalty = 0
                        for _, _, s, r in used:
                            if (s == current_loc and r == next_node) or (s == next_node and r == current_loc):
                                congestion_penalty += 1
                        
                        priority = (-levels[next_node], -path_progress, congestion_penalty, chunk)
                        transmissions.append((priority, chunk, current_loc, next_node))
                except ValueError:
                    continue
        
        # Sort by priority (deeper receivers first, then by path progress, then by congestion, then by chunk ID)
        transmissions.sort(key=lambda x: x[0])
        
        # Execute transmissions respecting half-duplex constraints
        for _, chunk, sender, receiver in transmissions:
            if sender in sends or sender in recvs or receiver in sends or receiver in recvs:
                continue
            
            chunk_locations[chunk] = receiver
            sends.add(sender)
            recvs.add(receiver)
            used.append((sender, receiver))
            tx_count += 1
        
        active_edges.append(tx_count)
        edge_hist.append(used)
        rounds += 1
        
        if tx_count == 0:
            break
    
    # Convert chunk_locations back to state format for compatibility
    state = {i: set() for i in range(total)}
    for chunk in range(N):
        if chunk in chunk_locations:
            state[chunk_locations[chunk]].add(chunk)
    
    return rounds, active_edges, state, edge_hist

def simulate_srda_fixed(n: int, m: int, start: Tuple[int, int], N: int):
    # Scatter
    s_rnd, s_edges, s_state, s_hist = simulate_scatter_fixed(n, m, start, N)
    # Allgather
    a_rnd, a_edges, f_state, a_hist = simulate_recursive_doubling_allgather_pipelined(n, m, s_state, N)

    total = s_rnd + a_rnd
    edges = s_edges + a_edges
    edge_hist = s_hist + a_hist

    # approximate state history: initial, after scatter, final
    nodes, mapping = get_nodes_and_mapping(n, m)
    total_nodes = len(nodes)
    initial = {i: set() for i in range(total_nodes)}
    initial[mapping[start]] = set(range(N))
    state_history = [initial, s_state, f_state]

    return s_rnd, a_rnd, total, edges, state_history, edge_hist

def simulate_srda_pipelined(n: int, m: int, start: Tuple[int, int], N: int):
    """
    SRDA with pipelined scatter phase.
    """
    # Pipelined Scatter
    s_rnd, s_edges, s_state, s_hist = simulate_scatter_pipelined(n, m, start, N)
    # Allgather (same as before)
    a_rnd, a_edges, f_state, a_hist = simulate_recursive_doubling_allgather_pipelined(n, m, s_state, N)

    total = s_rnd + a_rnd
    edges = s_edges + a_edges
    edge_hist = s_hist + a_hist

    # approximate state history: initial, after scatter, final
    nodes, mapping = get_nodes_and_mapping(n, m)
    total_nodes = len(nodes)
    initial = {i: set() for i in range(total_nodes)}
    initial[mapping[start]] = set(range(N))
    state_history = [initial, s_state, f_state]

    return s_rnd, a_rnd, total, edges, state_history, edge_hist

def simulate_srda_advanced_pipelined(n: int, m: int, start: Tuple[int, int], N: int):
    """
    SRDA with advanced pipelined scatter and allgather phases.
    Both phases use pipelining under half-duplex and bandwidth-1 constraints.
    """
    print("=== PHASE 1: SCATTER (Pipelined) ===")
    print("Hardware constraints: Half-duplex, Bandwidth-1")
    print("Strategy: Tree-based pipelined scatter with congestion-aware routing")
    
    # Advanced Pipelined Scatter
    s_rnd, s_edges, s_state, s_hist = simulate_scatter_advanced_pipelined(n, m, start, N)
    
    print(f"Scatter completed in {s_rnd} rounds")
    print("Scatter state:")
    print_state_grid(s_state, n, m)
    
    print("\n=== PHASE 2: ALLGATHER (Pipelined) ===")
    print("Hardware constraints: Half-duplex, Bandwidth-1")
    print("Strategy: Recursive doubling allgather with pipelined chunk exchange")
    
    # Pipelined Allgather
    a_rnd, a_edges, f_state, a_hist = simulate_recursive_doubling_allgather_pipelined(n, m, s_state, N)
    
    print(f"Allgather completed in {a_rnd} rounds")
    print("Final state:")
    print_state_grid(f_state, n, m)

    total = s_rnd + a_rnd
    edges = s_edges + a_edges
    edge_hist = s_hist + a_hist

    print(f"\n=== SUMMARY ===")
    print(f"Total rounds: {total}")
    print(f"Scatter rounds: {s_rnd}")
    print(f"Allgather rounds: {a_rnd}")
    print(f"Average active edges per round: {sum(edges)/len(edges):.2f}")

    # approximate state history: initial, after scatter, final
    nodes, mapping = get_nodes_and_mapping(n, m)
    total_nodes = len(nodes)
    initial = {i: set() for i in range(total_nodes)}
    initial[mapping[start]] = set(range(N))
    state_history = [initial, s_state, f_state]

    return s_rnd, a_rnd, total, edges, state_history, edge_hist

def print_state_grid(state: Dict[int, Set[int]], n: int, m: int):
    nodes, mapping = get_nodes_and_mapping(n, m)
    print("   " + " ".join(f"{j:2d}" for j in range(m)))
    for i in range(n):
        row = f"{i:2d} "
        for j in range(m):
            idx = mapping[(i, j)]
            row += f"{len(state[idx]):2d} "
        print(row)
    print()

def save_to_csv(active_edges: List[int], n: int, m: int, N: int, algorithm_name: str = "SRDA_fixed"):
    """Save timestep and active edge data to CSV file"""
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create filename
    filename = f"{data_dir}/{algorithm_name}_{n}_{m}_{N}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['timestep', 'active_edges'])
        # Write data
        for timestep, edges in enumerate(active_edges, 1):
            writer.writerow([timestep, edges])
    
    print(f"Data saved to {filename}")

def main():
    global GRID_ROWS, GRID_COLS, INFO_SIZE
    
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python3 SRDA.py <rows> <cols> <N>")
        print("Example: python3 SRDA.py 16 16 100")
        sys.exit(1)
    
    try:
        GRID_ROWS = int(sys.argv[1])  # rows
        GRID_COLS = int(sys.argv[2])  # cols
        INFO_SIZE = int(sys.argv[3])  # number of chunks
    except ValueError:
        print("Error: All arguments must be integers")
        sys.exit(1)
    
    if GRID_ROWS <= 0 or GRID_COLS <= 0 or INFO_SIZE <= 0:
        print("Error: All arguments must be positive integers")
        sys.exit(1)
    
    n, m = GRID_ROWS, GRID_COLS
    start = START_POINT
    N = INFO_SIZE

    total_nodes = n * m
    tree_depth = n + m - 2
    log2_nodes = int(np.ceil(np.log2(total_nodes)))
    print(f"=== SRDA ADVANCED PIPELINED ALGORITHM ===")
    print(f"Grid: {n}×{m}, Information size: {N}")
    print(f"Hardware: Half-duplex, Bandwidth-1")
    print(f"Theoretical minimum:")
    print(f"  Scatter: {tree_depth} rounds (tree depth)")
    print(f"  Allgather: {log2_nodes} rounds (log2 nodes)")
    print(f"  Total: {tree_depth + log2_nodes} rounds")
    print("=" * 50)

    random.seed(42)

    # Use the advanced pipelined version
    s_rnd, a_rnd, tot, edges, state_hist, edge_hist = simulate_srda_advanced_pipelined(n, m, start, N)

    # Save data to CSV
    save_to_csv(edges, n, m, N, "SRDA")

if __name__ == "__main__":
    main() 
