#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Dict, Set, Optional
import random
import heapq
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
CHUNKS_PER_TRANSMISSION = 1  # How many chunks can be sent in one transmission
PIPELINE_PRIORITY = True     # Prioritize paths that create efficient pipelines

def get_nodes_and_mapping(p: int, q: int, r: int) -> Tuple[List[Tuple[int,int,int]], Dict[Tuple[int,int,int], int]]:
    """
    Returns a list of node coordinates and a mapping from coordinate to a unique index.
    For a 3D grid, nodes are (i, j, k).
    """
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
    """
    Returns the list of neighbor coordinates for a given node in the 3D grid.
    The order is: +x, +y, +z, -x, -y, -z
    """
    neighbors = []
    directions = [(1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)]
    for dx, dy, dz in directions:
        neighbor = (coord[0] + dx, coord[1] + dy, coord[2] + dz)
        if (0 <= neighbor[0] < p and 
            0 <= neighbor[1] < q and 
            0 <= neighbor[2] < r):
            neighbors.append(neighbor)
    return neighbors

def linearize_position(coord: Tuple[int,int,int], q: int, r: int) -> int:
    """
    Converts a 3D coordinate (i,j,k) into a linearized position.
    """
    i, j, k = coord
    return i * (q * r) + j * r + k

def coord_from_linear(linear_pos: int, q: int, r: int) -> Tuple[int,int,int]:
    """
    Converts a linearized position back to a 3D coordinate (i,j,k).
    """
    i = linear_pos // (q * r)
    remainder = linear_pos % (q * r)
    j = remainder // r
    k = remainder % r
    return (i, j, k)

def manhattan_distance_3d(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> int:
    """
    Calculate Manhattan distance between two 3D coordinates.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def build_optimized_tree_overlay(p: int, q: int, r: int) -> Dict[int, List[int]]:
    """
    Builds an optimized tree overlay considering the 3D mesh proximity.
    This tries to match binary tree parent-child relationships with physical proximity.
    """
    total_nodes = p * q * r
    tree = {}
    
    # Use a BFS-like approach to assign children
    for node_idx in range(total_nodes):
        i, j, k = coord_from_linear(node_idx, q, r)
        
        # Get neighboring coordinates in priority order: +x, +y, +z, diagonal
        neighbors = []
        if i+1 < p:  # +x
            neighbors.append((i+1, j, k))
        if j+1 < q:  # +y
            neighbors.append((i, j+1, k))
        if k+1 < r:  # +z
            neighbors.append((i, j, k+1))
        if i+1 < p and j+1 < q:  # Diagonal xy
            neighbors.append((i+1, j+1, k))
        if i+1 < p and k+1 < r:  # Diagonal xz
            neighbors.append((i+1, j, k+1))
        if j+1 < q and k+1 < r:  # Diagonal yz
            neighbors.append((i, j+1, k+1))
        if i+1 < p and j+1 < q and k+1 < r:  # Diagonal xyz
            neighbors.append((i+1, j+1, k+1))
        
        # Convert neighbors to linear indices
        children = [linearize_position(coord, q, r) for coord in neighbors]
        
        # Filter out nodes that would create cycles (keep only nodes with higher indices)
        children = [c for c in children if c > node_idx]
        
        tree[node_idx] = children
    
    return tree

def find_pipeline_paths(tree: Dict[int, List[int]], p: int, q: int, r: int, total_nodes: int) -> Dict[int, List[int]]:
    """
    Identifies primary pipeline paths through the tree to prioritize during broadcast.
    Returns a dictionary mapping each node to its pipeline children.
    """
    # Start with the optimized tree structure
    pipeline_paths = {i: [] for i in range(total_nodes)}
    
    # Use a breadth-first approach to identify main paths
    visited = set([0])  # Start with the root
    queue = [(0, 0)]  # (node, depth)
    
    while queue:
        node, depth = queue.pop(0)
        
        # Sort children by Manhattan distance to the farthest corner
        far_corner = (p-1, q-1, r-1)
        node_coord = coord_from_linear(node, q, r)
        
        # For each child, calculate distance to far corner
        children_with_distance = []
        for child in tree[node]:
            if child not in visited:
                child_coord = coord_from_linear(child, q, r)
                dist = manhattan_distance_3d(child_coord, far_corner)
                children_with_distance.append((child, dist))
        
        # Sort by distance (ascending, so closest to far corner first)
        children_with_distance.sort(key=lambda x: x[1])
        
        # Select the primary pipeline child (closest to far corner)
        if children_with_distance:
            primary_child = children_with_distance[0][0]
            pipeline_paths[node].append(primary_child)
            
            # Add all children to the regular processing queue
            for child, _ in children_with_distance:
                visited.add(child)
                queue.append((child, depth + 1))
    
    return pipeline_paths

def build_spanning_tree(p: int, q: int, r: int, start: Tuple[int,int,int]) -> Dict[int, List[int]]:
    """
    Builds a spanning tree of the 3D grid starting from the given point.
    This is a breadth-first spanning tree, which optimizes for shortest paths.
    
    Returns a dictionary mapping each node to its children in the tree.
    """
    nodes, mapping = get_nodes_and_mapping(p, q, r)
    total_nodes = p * q * r
    
    # Initialize tree structure
    tree = {i: [] for i in range(total_nodes)}
    
    # BFS to build tree
    visited = set([mapping[start]])
    queue = deque([mapping[start]])
    
    while queue:
        current_idx = queue.popleft()
        current_coord = nodes[current_idx]
        
        # Get physical neighbors in the grid
        for neighbor_coord in get_neighbors(current_coord, p, q, r):
            neighbor_idx = mapping[neighbor_coord]
            
            if neighbor_idx not in visited:
                # Add as child in the tree
                tree[current_idx].append(neighbor_idx)
                
                # Mark as visited and add to queue
                visited.add(neighbor_idx)
                queue.append(neighbor_idx)
    
    return tree

def compute_node_levels(tree: Dict[int, List[int]], root: int) -> Dict[int, int]:
    """
    Computes the level (distance from root) for each node in the tree.
    """
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

def simulate_optimized_bcast(p: int, q: int, r: int, start: Tuple[int,int,int], N: int) -> Tuple[int, List[int], List[Dict[int, Set[int]]], List[List[Tuple[int,int,int]]]]:
    """
    Simulates an optimized broadcast in a 3D mesh with strict constraints:
    - Each node can only send OR receive one chunk per timestep
    - Communication is only between adjacent nodes
    - Information is spread through a spanning tree
    - Pipelining is optimized
    
    Returns:
        rounds: Total number of rounds
        active_edges: Transmissions per round
        state_history: Information state per round
        edge_usage_history: Physical edges used per round
    """
    # Get nodes and mapping
    nodes, mapping = get_nodes_and_mapping(p, q, r)
    total_nodes = len(nodes)
    start_idx = mapping[start]
    
    # Build a spanning tree from the start node
    tree = build_spanning_tree(p, q, r, start)
    
    # Compute level for each node in the tree
    levels = compute_node_levels(tree, start_idx)
    max_level = max(levels.values())
    
    # Initialize state
    state: Dict[int, Set[int]] = {i: set() for i in range(total_nodes)}
    state[start_idx] = set(range(N))  # Source node gets the full message
    
    rounds = 0
    active_edges: List[int] = []
    state_history: List[Dict[int, Set[int]]] = [state.copy()]
    edge_usage_history: List[List[Tuple[int,int,int]]] = []
    
    # Continue until every node is fully informed
    while any(len(chunks) < N for chunks in state.values()):
        transmissions_this_round = 0
        new_state: Dict[int, Set[int]] = {i: state[i].copy() for i in state}
        
        # Track nodes that send or receive in this round
        sending_nodes: Set[int] = set()
        receiving_nodes: Set[int] = set()
        
        # Track edges used in this round
        used_edges = []
        
        # Priority queue to prioritize (highest level first, then latest chunk)
        # Format: (priority, sender, receiver, chunk)
        # Priority is a tuple of (-level, chunk_id)
        transmissions_queue = []
        
        # Generate all possible transmissions for this round
        for sender_idx in range(total_nodes):
            # Skip nodes that have nothing to send
            if not state[sender_idx]:
                continue
                
            for child_idx in tree[sender_idx]:
                # Find missing chunks that child needs
                missing = state[sender_idx] - state[child_idx]
                
                if missing:
                    # For each potential chunk to send, add to priority queue
                    chunk_to_send = min(missing)  # Send the earliest chunk first
                    
                    # Priority: higher level nodes first (negative for max-heap behavior)
                    # then earliest chunk first
                    priority = (-levels[child_idx], chunk_to_send)
                    
                    transmissions_queue.append((priority, sender_idx, child_idx, chunk_to_send))
        
        # Sort by priority
        transmissions_queue.sort()
        
        # Process transmissions in priority order
        for _, sender_idx, receiver_idx, chunk_id in transmissions_queue:
            # Skip if sender or receiver already busy
            if sender_idx in sending_nodes or sender_idx in receiving_nodes:
                continue
            if receiver_idx in sending_nodes or receiver_idx in receiving_nodes:
                continue
            
            # Process the transmission
            new_state[receiver_idx].add(chunk_id)
            sending_nodes.add(sender_idx)
            receiving_nodes.add(receiver_idx)
            used_edges.append((sender_idx, receiver_idx))
            transmissions_this_round += 1
        
        active_edges.append(transmissions_this_round)
        edge_usage_history.append(used_edges)
        state = new_state
        state_history.append(state.copy())
        rounds += 1
        
        # Exit if no transmissions occurred this round (to prevent infinite loop)
        if transmissions_this_round == 0:
            break
    
    return rounds, active_edges, state_history, edge_usage_history

def get_total_edges_3d_mesh(p: int, q: int, r: int) -> int:
    """
    Calculate the total number of edges in a 3D mesh of size p×q×r.
    For a 3D mesh, each interior node has 6 edges, surface nodes have 5 edges,
    edge nodes have 4 edges, and corner nodes have 3 edges.
    """
    # Interior nodes: (p-2)*(q-2)*(r-2) nodes with 6 edges each
    interior_edges = (p-2)*(q-2)*(r-2) * 6
    
    # Surface nodes (excluding edges and corners): 6 faces with (p-2)*(q-2) nodes each
    # Each surface node has 5 edges
    surface_edges = 6 * ((p-2)*(q-2) + (p-2)*(r-2) + (q-2)*(r-2)) * 5
    
    # Edge nodes (excluding corners): 12 edges with (p-2) nodes each
    # Each edge node has 4 edges
    edge_edges = 12 * (p-2 + q-2 + r-2) * 4
    
    # Corner nodes: 8 corners with 3 edges each
    corner_edges = 8 * 3
    
    # Total edges (divide by 2 since each edge is counted twice)
    total_edges = (interior_edges + surface_edges + edge_edges + corner_edges) // 2
    
    return total_edges

def save_to_csv(active_edges: List[int], p: int, q: int, r: int, N: int, algorithm_name: str = "BinTree3D"):
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

def plot_transmissions(active_edges: List[int], p: int, q: int, r: int) -> None:
    """
    Plots the ratio of active edges to total edges in each round.
    Uses the same blue color scheme as other algorithms.
    """
    # Disabled plotting
    pass

def calc_theoretical_min(p: int, q: int, r: int, N: int) -> int:
    """
    Calculate the theoretical minimum rounds with perfect pipelining for 3D mesh.
    For a spanning tree, we need:
    - tree_depth rounds to reach the furthest node
    - (N-1) additional rounds to send the rest of the information
    """
    # In a 3D mesh, the tree depth is at most the Manhattan distance to the furthest corner
    tree_depth = (p-1) + (q-1) + (r-1)  # From (0,0,0) to (p-1,q-1,r-1)
    return tree_depth + N - 1

def parse_command_line_args():
    """Parse command line arguments for p, q, r, N"""
    if len(sys.argv) != 5:
        print("Usage: python3 BinTree.py <p> <q> <r> <N>")
        print("Example: python3 BinTree.py 8 8 8 100")
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
        print("Usage: python3 BinTree.py <p> <q> <r> <N>")
        sys.exit(1)

def main():
    # Parse command line arguments
    p, q, r, N = parse_command_line_args()
    start_coords = (0, 0, 0)
    
    # Calculate theoretical minimum
    theoretical_min = math.ceil(N / CHUNKS_PER_TRANSMISSION)
    # Run simulation
    rounds, active_edges, state_history, edge_usage_history = simulate_optimized_bcast(
        p, q, r, start_coords, N
    )
    # Save to CSV
    save_to_csv(active_edges, p, q, r, N, "BinTree3D")

if __name__ == "__main__":
    main() 
