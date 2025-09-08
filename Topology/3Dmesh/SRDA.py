#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Dict, Set
from collections import deque
import csv
import os
import math
import sys

# Default parameters
GRID_X = 8      # p (number of rows)
GRID_Y = 6      # q (number of columns)
GRID_Z = 4      # r (number of layers)
INFO_SIZE = 100  # N (size of information)
START_POINT = (0, 0, 0)  # Source at one corner

def get_nodes_and_mapping(p: int, q: int, r: int) -> Tuple[List[Tuple[int,int,int]], Dict[Tuple[int,int,int], int]]:
    """Returns a list of node coordinates and a mapping from coordinate to a unique index."""
    nodes: List[Tuple[int,int,int]] = []
    mapping: Dict[Tuple[int,int,int], int] = {}
    for i in range(p):
        for j in range(q):
            for k in range(r):
                coord = (i, j, k)
                mapping[coord] = len(nodes)
                nodes.append(coord)
    return nodes, mapping

def get_neighbors(coord: Tuple[int,int,int], p: int, q: int, r: int) -> List[Tuple[int,int,int]]:
    """Returns the list of neighbor coordinates for a given node in the 3D grid."""
    neighbors = []
    directions = [(1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)]
    for dx, dy, dz in directions:
        neighbor = (coord[0] + dx, coord[1] + dy, coord[2] + dz)
        if (0 <= neighbor[0] < p and 
            0 <= neighbor[1] < q and 
            0 <= neighbor[2] < r):
            neighbors.append(neighbor)
    return neighbors

def build_spanning_tree(p: int, q: int, r: int, start: Tuple[int,int,int]) -> Dict[int, List[int]]:
    """Builds a spanning tree of the 3D grid starting from the given point."""
    nodes, mapping = get_nodes_and_mapping(p, q, r)
    total = p * q * r
    tree = {i: [] for i in range(total)}
    visited = {mapping[start]}
    queue = deque([mapping[start]])
    
    while queue:
        current = queue.popleft()
        current_coord = nodes[current]
        
        for neighbor_coord in get_neighbors(current_coord, p, q, r):
            neighbor_idx = mapping[neighbor_coord]
            if neighbor_idx not in visited:
                tree[current].append(neighbor_idx)
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)
    
    return tree

def compute_node_levels(tree: Dict[int, List[int]], root: int) -> Dict[int, int]:
    """Computes the level (distance from root) for each node in the tree."""
    levels = {root: 0}
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        level = levels[node]
        
        for child in tree[node]:
            if child not in levels:
                levels[child] = level + 1
                queue.append(child)
    
    return levels

def calculate_chunk_assignment(total_nodes: int, N: int) -> Dict[int, List[int]]:
    """Calculates which chunks each node is responsible for in the scatter phase."""
    base = N // total_nodes
    rem = N % total_nodes
    assignment = {}
    chunk = 0
    
    for i in range(total_nodes):
        cnt = base + (1 if i < rem else 0)
        assignment[i] = list(range(chunk, chunk + cnt))
        chunk += cnt
    
    return assignment

def simulate_scatter(p: int, q: int, r: int, start: Tuple[int,int,int], N: int) -> Tuple[int, List[int], Dict[int, Set[int]], List[List[Tuple[int,int]]]]:
    """Simulates a pipelined, tree-based scatter in 3D mesh (SRDA style)."""
    from collections import deque
    nodes, mapping = get_nodes_and_mapping(p, q, r)
    total = p * q * r
    start_idx = mapping[start]
    tree = build_spanning_tree(p, q, r, start)
    chunk_assignment = calculate_chunk_assignment(total, N)

    # For each chunk, compute the path from source to its assigned node
    chunk_paths = []  # List of lists of node indices
    for node, chunks in chunk_assignment.items():
        for chunk in chunks:
            # BFS to find path from start_idx to node in tree
            parent = {start_idx: None}
            queue = deque([start_idx])
            found = False
            while queue and not found:
                curr = queue.popleft()
                for child in tree[curr]:
                    if child not in parent:
                        parent[child] = curr
                        queue.append(child)
                        if child == node:
                            found = True
                            break
            # Reconstruct path
            path = []
            n = node
            while n is not None:
                path.append(n)
                n = parent[n]
            path = path[::-1]  # from source to node
            chunk_paths.append((chunk, path))

    # For each chunk, track its current position along its path
    chunk_positions = {chunk: 0 for chunk, path in chunk_paths}  # index in path
    chunk_path_map = {chunk: path for chunk, path in chunk_paths}
    chunk_delivered = {chunk: False for chunk, path in chunk_paths}

    # State: which chunks each node has
    state = {i: set() for i in range(total)}
    state[start_idx] = set(range(N))

    rounds = 0
    active_edges = []
    edge_history = []

    while not all(chunk_delivered.values()):
        sending_nodes = set()
        receiving_nodes = set()
        used_edges = []
        # For each chunk, try to move it one hop further along its path
        chunk_moves = []  # (sender, receiver, chunk)
        for chunk, path in chunk_path_map.items():
            if chunk_delivered[chunk]:
                continue
            pos = chunk_positions[chunk]
            if pos >= len(path) - 1:
                # Already at destination
                chunk_delivered[chunk] = True
                continue
            sender = path[pos]
            receiver = path[pos + 1]
            # Only move if sender has the chunk and receiver does not
            if chunk in state[sender] and chunk not in state[receiver]:
                chunk_moves.append((sender, receiver, chunk))
        # To obey half-duplex, only allow one send/receive per node per round
        # Greedily process chunk_moves
        for sender, receiver, chunk in chunk_moves:
            if (sender in sending_nodes or sender in receiving_nodes or
                receiver in sending_nodes or receiver in receiving_nodes):
                continue
            # Perform the transmission
            state[receiver].add(chunk)
            chunk_positions[chunk] += 1
            if chunk_positions[chunk] == len(chunk_path_map[chunk]) - 1:
                chunk_delivered[chunk] = True
            sending_nodes.add(sender)
            receiving_nodes.add(receiver)
            used_edges.append((sender, receiver))
        active_edges.append(len(used_edges))
        edge_history.append(used_edges)
        rounds += 1
        # If no progress, break to avoid infinite loop
        if len(used_edges) == 0:
            print(f"Warning: Scatter stuck at round {rounds}. Chunks delivered: {sum(chunk_delivered.values())}/{N}")
            break
    print(f"Scatter completed in {rounds} rounds. Chunks delivered: {sum(chunk_delivered.values())}/{N}")
    return rounds, active_edges, state, edge_history

def simulate_recursive_doubling_allgather(p: int, q: int, r: int, init_state: Dict[int, Set[int]], N: int) -> Tuple[int, List[int], Dict[int, Set[int]], List[List[Tuple[int,int]]]]:
    """Simulates the recursive doubling allgather phase of SRDA in 3D mesh."""
    nodes, mapping = get_nodes_and_mapping(p, q, r)
    total = p * q * r
    state = {i: init_state[i].copy() for i in init_state}
    
    rounds = 0
    active_edges = []
    edge_history = []
    
    # Continue until all nodes have all chunks
    while any(len(state[i]) < N for i in state):
        sending_nodes = set()
        receiving_nodes = set()
        used_edges = []
        transmissions = []
        
        # Collect possible transmissions
        for sender in range(total):
            if not state[sender]:
                continue
                
            sender_coord = nodes[sender]
            for neighbor_coord in get_neighbors(sender_coord, p, q, r):
                receiver = mapping[neighbor_coord]
                missing = state[sender] - state[receiver]
                
                if missing:
                    # Send exactly 1 chunk per transmission
                    chunk = min(missing)  # Send earliest missing chunk
                    transmissions.append((len(missing), sender, receiver, chunk))
        
        # Sort by score (number of chunks to be gained)
        transmissions.sort(reverse=True)
        
        # Process transmissions with proper conflict resolution
        new_state = {i: state[i].copy() for i in state}
        tx_count = 0
        
        for _, sender, receiver, chunk in transmissions:
            # Skip if sender or receiver already busy (half-duplex constraint)
            if (sender in sending_nodes or sender in receiving_nodes or 
                receiver in sending_nodes or receiver in receiving_nodes):
                continue
                
            new_state[receiver].add(chunk)
            sending_nodes.add(sender)
            receiving_nodes.add(receiver)
            used_edges.append((sender, receiver))
            tx_count += 1
        
        state = new_state
        active_edges.append(tx_count)
        edge_history.append(used_edges)
        rounds += 1
        
        # Exit if no transmissions occurred this round (to prevent infinite loop)
        if tx_count == 0:
            break
    
    return rounds, active_edges, state, edge_history

def simulate_srda_3d(p: int, q: int, r: int, start: Tuple[int,int,int], N: int) -> Tuple[int, int, int, List[int], List[Dict[int, Set[int]]], List[List[Tuple[int,int]]]]:
    """Simulates the complete SRDA algorithm in 3D mesh."""
    # Scatter phase
    scatter_rounds, scatter_edges, scatter_state, scatter_history = simulate_scatter(p, q, r, start, N)
    
    # Allgather phase
    allgather_rounds, allgather_edges, final_state, allgather_history = simulate_recursive_doubling_allgather(
        p, q, r, scatter_state, N)
    
    # Combine results
    total_rounds = scatter_rounds + allgather_rounds
    all_edges = scatter_edges + allgather_edges
    all_history = scatter_history + allgather_history
    
    # Create state history
    nodes, mapping = get_nodes_and_mapping(p, q, r)
    total_nodes = len(nodes)
    initial_state = {i: set() for i in range(total_nodes)}
    initial_state[mapping[start]] = set(range(N))
    state_history = [initial_state, scatter_state, final_state]
    
    return scatter_rounds, allgather_rounds, total_rounds, all_edges, state_history, all_history

def save_to_csv(active_edges: List[int], p: int, q: int, r: int, N: int, algorithm_name: str = "SRDA3D"):
    """Save timestep and active edge data to CSV file"""
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create filename with format: ALGO_p_q_r_N.csv
    filename = f"{data_dir}/{algorithm_name}_{p}_{q}_{r}_{N}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['timestep', 'active_edges'])
        # Write data
        for timestep, edges in enumerate(active_edges, 1):
            writer.writerow([timestep, edges])
    
    # print(f"Data saved to {filename}")

def plot_transmissions(active_edges: List[int], scatter_rounds: int, p: int, q: int, r: int) -> None:
    """Plots the normalized active edges over time, showing scatter and allgather phases."""
    # Disabled plotting
    pass

def get_total_edges_3d_mesh(p: int, q: int, r: int) -> int:
    """Calculate the total number of edges in a 3D mesh."""
    interior_edges = (p-2)*(q-2)*(r-2) * 6
    surface_edges = 6 * ((p-2)*(q-2) + (p-2)*(r-2) + (q-2)*(r-2)) * 5
    edge_edges = 12 * (p-2 + q-2 + r-2) * 4
    corner_edges = 8 * 3
    return (interior_edges + surface_edges + edge_edges + corner_edges) // 2

def parse_command_line_args():
    """Parse command line arguments for p, q, r, N"""
    if len(sys.argv) != 5:
        print("Usage: python3 SRDA3D.py <p> <q> <r> <N>")
        print("Example: python3 SRDA3D.py 8 8 8 100")
        sys.exit(1)
    try:
        p_val = int(sys.argv[1])
        q_val = int(sys.argv[2])
        r_val = int(sys.argv[3])
        N_val = int(sys.argv[4])
        if p_val <= 0 or q_val <= 0 or r_val <= 0 or N_val <= 0:
            raise ValueError("All parameters must be positive integers")
        return p_val, q_val, r_val, N_val
    except ValueError as e:
        print(f"Error: Invalid arguments. {e}")
        print("Usage: python3 SRDA3D.py <p> <q> <r> <N>")
        sys.exit(1)

def main():
    # Parse command line arguments
    p, q, r, N = parse_command_line_args()
    start = (0, 0, 0)
    
    # Calculate theoretical minimum
    tree_depth = p + q + r - 3
    log2_nodes = math.ceil(math.log2(p * q * r))
    # Run simulation
    s_rnd, a_rnd, tot, edges, state_hist, edge_hist = simulate_srda_3d(
        p, q, r, start, N
    )
    # Save to CSV
    save_to_csv(edges, p, q, r, N, "SRDA3D")

if __name__ == "__main__":
    main() 


