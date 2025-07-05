#!/usr/bin/env python3
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Set
import random
import numpy as np
from collections import deque
import csv
import os
import sys

# Default parameters
INFO_SIZE = 1000             # Size of packets
START_NODE = 1              # Starting node for broadcast

def get_16k3_adjacency_list() -> Dict[int, List[int]]:
    """Returns the adjacency list for the 16K3 topology"""
    return {
        1: [2, 3, 4],
        2: [1, 7, 11],
        3: [1, 9, 13],
        4: [1, 5, 15],
        5: [4, 6, 10],
        6: [5, 7, 12],
        7: [2, 6, 8],
        8: [7, 9, 14],
        9: [3, 8, 10],
        10: [5, 9, 16],
        11: [2, 12, 16],
        12: [6, 11, 13],
        13: [3, 12, 14],
        14: [8, 13, 15],
        15: [4, 14, 16],
        16: [10, 11, 15]
    }

def build_spanning_tree(adj_list: Dict[int, List[int]], start: int) -> Dict[int, List[int]]:
    """Build a spanning tree from the start node using BFS"""
    tree = {i: [] for i in adj_list.keys()}
    visited = {start}
    q = deque([start])
    
    while q:
        u = q.popleft()
        for v in adj_list[u]:
            if v not in visited:
                tree[u].append(v)
                visited.add(v)
                q.append(v)
    
    return tree

def compute_node_levels(tree: Dict[int, List[int]], root: int) -> Dict[int, int]:
    """Compute the level of each node in the tree"""
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
    """Calculate which chunks each node should receive during scatter"""
    base = N // total_nodes
    rem = N % total_nodes
    assignment = {}
    chunk = 0
    
    for i in range(1, total_nodes + 1):
        cnt = base + (1 if i <= rem else 0)
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

def simulate_scatter_16k3(start_node: int, N: int) -> Tuple[int, List[int], Dict[int, Set[int]], List[List[Tuple[int, int]]]]:
    """Simulate the scatter phase of SRDA on 16K3 topology"""
    adj_list = get_16k3_adjacency_list()
    total_nodes = len(adj_list)
    tree = build_spanning_tree(adj_list, start_node)
    levels = compute_node_levels(tree, start_node)
    chunk_assignment = calculate_chunk_assignment(total_nodes, N)

    # Build chunk_to_node mapping
    chunk_to_node = {}
    for node_idx in range(1, total_nodes + 1):
        for chunk in chunk_assignment[node_idx]:
            chunk_to_node[chunk] = node_idx

    # Precompute paths for each chunk to its destination
    chunk_paths = {}
    for node_idx in range(1, total_nodes + 1):
        if node_idx == start_node:
            continue
        path = find_path_to_node(tree, start_node, node_idx)
        if path and chunk_assignment[node_idx]:  # Only add if node has chunks assigned
            for chunk in chunk_assignment[node_idx]:
                chunk_paths[chunk] = path

    # Initialize state: each node has chunks that are currently at that node
    # Initially, all chunks are at the source
    chunk_locations = {chunk: start_node for chunk in range(N)}

    rounds = 0
    active_edges = []
    edge_hist = []

    # Continue until all chunks reach their destinations
    while any(chunk_locations[chunk] != chunk_to_node[chunk] for chunk in chunk_paths.keys()):
        sends = set()
        recvs = set()
        used = []
        tx_count = 0

        # Find all possible transmissions for this round
        transmissions = []
        for chunk in chunk_paths.keys():
            current_loc = chunk_locations[chunk]
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
    state = {i: set() for i in range(1, total_nodes + 1)}
    for chunk in range(N):
        if chunk in chunk_locations:
            state[chunk_locations[chunk]].add(chunk)

    return rounds, active_edges, state, edge_hist

def simulate_allgather_16k3(init_state: Dict[int, Set[int]], N: int) -> Tuple[int, List[int], Dict[int, Set[int]], List[List[Tuple[int, int]]]]:
    """Simulate the allgather phase of SRDA on 16K3 topology"""
    adj_list = get_16k3_adjacency_list()
    total_nodes = len(adj_list)
    state = {i: set(init_state[i]) for i in init_state}
    
    rounds = 0
    active_edges = []
    edge_hist = []
    
    while any(len(state[i]) < N for i in state):
        sends = set()
        recvs = set()
        used = []
        trans = []
        
        for u in range(1, total_nodes + 1):
            if not state[u]:
                continue
            for v in adj_list[u]:
                missing = state[u] - state[v]
                if missing:
                    score = len(missing)
                    c = min(missing)
                    trans.append((score, u, v, c))
        
        trans.sort(reverse=True)
        new_state = {i: set(state[i]) for i in state}
        tx_count = 0
        
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

def simulate_srda_16k3(start_node: int, N: int) -> Tuple[int, int, int, List[int], List[Dict[int, Set[int]]], List[List[Tuple[int, int]]]]:
    """Simulate complete SRDA on 16K3 topology"""
    # Scatter phase
    s_rnd, s_edges, s_state, s_hist = simulate_scatter_16k3(start_node, N)
    
    # Allgather phase
    a_rnd, a_edges, f_state, a_hist = simulate_allgather_16k3(s_state, N)
    
    total = s_rnd + a_rnd
    edges = s_edges + a_edges
    edge_hist = s_hist + a_hist
    
    # State history: initial, after scatter, final
    adj_list = get_16k3_adjacency_list()
    total_nodes = len(adj_list)
    initial = {i: set() for i in range(1, total_nodes + 1)}
    initial[start_node] = set(range(N))
    state_history = [initial, s_state, f_state]
    
    return s_rnd, a_rnd, total, edges, state_history, edge_hist

def print_state_grid(state: Dict[int, Set[int]], N: int):
    """Print the current state of all nodes"""
    print("Current Node States:")
    print("-" * 50)
    for node in sorted(state.keys()):
        chunks = len(state[node])
        percentage = chunks / N * 100
        bar_length = int(percentage / 5)
        bar = "█" * bar_length
        print(f"Node {node:2d}: {chunks}/{N} chunks ({percentage:.1f}%) {bar}")
    print("-" * 50)

def save_to_csv(active_edges: List[int], N: int, algorithm_name: str = "SRDA"):
    """Save timestep and active edge data to CSV file"""
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create filename
    filename = f"{data_dir}/{algorithm_name}_16K3_{N}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['timestep', 'active_edges'])
        # Write data
        for timestep, edges in enumerate(active_edges, 1):
            writer.writerow([timestep, edges])
    
    print(f"Data saved to {filename}")

def main():
    """Main function to run SRDA on 16K3 topology"""
    global INFO_SIZE
    
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python3 SRDA.py <N>")
        print("Example: python3 SRDA.py 100")
        sys.exit(1)
    
    try:
        INFO_SIZE = int(sys.argv[1])
    except ValueError:
        print("Error: N must be an integer")
        sys.exit(1)
    
    if INFO_SIZE <= 0:
        print("Error: N must be a positive integer")
        sys.exit(1)
    
    start = START_NODE
    N = INFO_SIZE
    
    print(f"16K3 Topology SRDA Broadcast")
    print(f"Information size: {N}")
    print(f"Start node: {start}")
    
    # Calculate theoretical minimum
    total_nodes = 16
    tree_depth = 4  # Maximum depth in 16K3 spanning tree
    log2_nodes = int(np.ceil(np.log2(total_nodes)))
    print(f"Theoretical min rounds: Scatter={tree_depth}, Allgather={log2_nodes}, Total={tree_depth+log2_nodes}")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run SRDA simulation
    s_rnd, a_rnd, tot, edges, state_hist, edge_hist = simulate_srda_16k3(start, N)
    
    print(f"Scatter rounds:    {s_rnd}")
    print(f"Allgather rounds:  {a_rnd}")
    print(f"Total rounds:      {tot}")
    
    print("\nFinal chunks per node:")
    print_state_grid(state_hist[-1], N)
    
    # Save data to CSV
    save_to_csv(edges, N, "SRDA")

if __name__ == "__main__":
    main()
