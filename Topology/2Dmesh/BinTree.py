import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Dict, Set, Optional
import random
import heapq
from collections import deque
import csv
import os
import sys

# Default parameters (will be overridden by command line arguments)
GRID_ROWS = 64     # n (number of rows)
GRID_COLS = 64     # m (number of columns)
INFO_SIZE = 500    # N (size of information)
START_POINT = (0, 0)  # Top-left corner
CHUNKS_PER_TRANSMISSION = 1  # How many chunks can be sent in one transmission
PIPELINE_PRIORITY = True     # Prioritize paths that create efficient pipelines

def get_nodes_and_mapping(n: int, m: int) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    Returns a list of node coordinates and a mapping from coordinate to a unique index.
    For a 2D grid, nodes are (i, j).
    """
    nodes: List[Tuple[int, int]] = []
    mapping: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        for j in range(m):
            coord = (i, j)
            mapping[coord] = len(nodes)
            nodes.append(coord)
    return nodes, mapping

def get_neighbors(coord: Tuple[int, int], n: int, m: int) -> List[Tuple[int, int]]:
    """
    Returns the list of neighbor coordinates for a given node in the grid.
    The order is right, down, left, up.
    """
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for d in directions:
        neighbor = (coord[0] + d[0], coord[1] + d[1])
        if 0 <= neighbor[0] < n and 0 <= neighbor[1] < m:
            neighbors.append(neighbor)
    return neighbors

def linearize_position(coord: Tuple[int, int], m: int) -> int:
    """
    Converts a 2D coordinate (i,j) into a linearized position.
    """
    return coord[0] * m + coord[1]

def coord_from_linear(linear_pos: int, m: int) -> Tuple[int, int]:
    """
    Converts a linearized position back to a 2D coordinate (i,j).
    """
    return (linear_pos // m, linear_pos % m)

def get_binary_tree_parent(pos: int) -> int:
    """
    Returns the parent of a node in a binary tree (0-indexed).
    Root node (0) has no parent, so returns -1.
    """
    if pos == 0:
        return -1
    return (pos - 1) // 2

def get_binary_tree_children(pos: int, total_nodes: int) -> List[int]:
    """
    Returns the children of a node in a binary tree (0-indexed).
    """
    children = []
    left_child = 2 * pos + 1
    right_child = 2 * pos + 2
    
    if left_child < total_nodes:
        children.append(left_child)
    if right_child < total_nodes:
        children.append(right_child)
        
    return children

def build_optimized_tree_overlay(n: int, m: int) -> Dict[int, List[int]]:
    """
    Builds an optimized tree overlay considering the 2D mesh proximity.
    This tries to match binary tree parent-child relationships with physical proximity.
    """
    total_nodes = n * m
    tree = {}
    
    # Use a BFS-like approach to assign children
    for node_idx in range(total_nodes):
        i, j = coord_from_linear(node_idx, m)
        
        # Get neighboring coordinates in priority order: right, down, diagonal
        neighbors = []
        if j+1 < m:  # Right
            neighbors.append((i, j+1))
        if i+1 < n:  # Down
            neighbors.append((i+1, j))
        if i+1 < n and j+1 < m:  # Diagonal down-right (for better connectivity)
            neighbors.append((i+1, j+1))
        
        # Convert neighbors to linear indices
        children = [linearize_position(coord, m) for coord in neighbors]
        
        # Filter out nodes that would create cycles (keep only nodes with higher indices)
        children = [c for c in children if c > node_idx]
        
        tree[node_idx] = children
    
    return tree

def find_pipeline_paths(tree: Dict[int, List[int]], n: int, m: int, total_nodes: int) -> Dict[int, List[int]]:
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
        far_corner = (n-1, m-1)
        node_coord = coord_from_linear(node, m)
        
        # For each child, calculate distance to far corner
        children_with_distance = []
        for child in tree[node]:
            if child not in visited:
                child_coord = coord_from_linear(child, m)
                dist = manhattan_distance(child_coord, far_corner)
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

def sort_transmissions_by_pipeline(transmissions, pipeline_paths):
    """
    Sort possible transmissions to prioritize pipeline paths.
    """
    pipeline_transmissions = []
    other_transmissions = []
    
    for t in transmissions:
        sender, receiver = t[0], t[1]
        if receiver in pipeline_paths.get(sender, []):
            pipeline_transmissions.append(t)
        else:
            other_transmissions.append(t)
    
    # Return pipeline transmissions first, then others
    return pipeline_transmissions + other_transmissions

def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Calculate Manhattan distance between two coordinates.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def direct_route_in_mesh(start: Tuple[int, int], end: Tuple[int, int], n: int, m: int) -> List[Tuple[int, int]]:
    """
    Finds the shortest path from start to end in the 2D mesh using XY routing.
    Returns a list of coordinates forming the path.
    """
    path = [start]
    current = start
    
    # First move in X direction
    while current[1] != end[1]:
        next_j = current[1] + (1 if current[1] < end[1] else -1)
        next_coord = (current[0], next_j)
        path.append(next_coord)
        current = next_coord
    
    # Then move in Y direction
    while current[0] != end[0]:
        next_i = current[0] + (1 if current[0] < end[0] else -1)
        next_coord = (next_i, current[1])
        path.append(next_coord)
        current = next_coord
    
    return path

def build_spanning_tree(n: int, m: int, start: Tuple[int, int]) -> Dict[int, List[int]]:
    """
    Builds a spanning tree of the grid starting from the given point.
    This is a breadth-first spanning tree, which optimizes for shortest paths.
    
    Returns a dictionary mapping each node to its children in the tree.
    """
    nodes, mapping = get_nodes_and_mapping(n, m)
    total_nodes = n * m
    
    # Initialize tree structure
    tree = {i: [] for i in range(total_nodes)}
    
    # BFS to build tree
    visited = set([mapping[start]])
    queue = deque([mapping[start]])
    
    while queue:
        current_idx = queue.popleft()
        current_coord = nodes[current_idx]
        
        # Get physical neighbors in the grid
        for neighbor_coord in get_neighbors(current_coord, n, m):
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

def simulate_optimized_bcast(n: int, m: int, start: Tuple[int, int], N: int) -> Tuple[int, List[int], List[Dict[int, Set[int]]], List[List[Tuple[int, int]]]]:
    """
    Simulates an optimized broadcast in a 2D mesh with strict constraints:
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
    nodes, mapping = get_nodes_and_mapping(n, m)
    total_nodes = len(nodes)
    start_idx = mapping[start]
    
    # Build a spanning tree from the start node
    tree = build_spanning_tree(n, m, start)
    
    # Compute level for each node in the tree
    levels = compute_node_levels(tree, start_idx)
    max_level = max(levels.values())
    
    # Initialize state
    state: Dict[int, Set[int]] = {i: set() for i in range(total_nodes)}
    state[start_idx] = set(range(N))  # Source node gets the full message
    
    rounds = 0
    active_edges: List[int] = []
    state_history: List[Dict[int, Set[int]]] = [state.copy()]
    edge_usage_history: List[List[Tuple[int, int]]] = []
    
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

def print_state_grid(state: Dict[int, Set[int]], n: int, m: int, N: int, nodes: List[Tuple[int, int]], mapping: Dict[Tuple[int, int], int]) -> None:
    """
    Prints a visual representation of the current state of the grid.
    Each cell shows how many chunks that node has out of N.
    """
    print("Current Grid State:")
    print(f"  " + "".join([f"{j:2d}" for j in range(m)]))
    for i in range(n):
        print(f"{i:2d}", end=" ")
        for j in range(m):
            coord = (i, j)
            idx = mapping[coord]
            chunks = len(state[idx])
            print(f"{chunks:2d}", end=" ")
        print()
    print()

def save_to_csv(active_edges: List[int], n: int, m: int, N: int, algorithm_name: str = "BinTreeBcast"):
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

def plot_transmissions(active_edges: List[int]) -> None:
    """
    Plots the number of active transmissions per round.
    """
    # Disabled plotting
    pass

def visualize_broadcast(n: int, m: int, state_history: List[Dict[int, Set[int]]], edge_usage_history: List[List[Tuple[int, int]]]) -> None:
    """
    Visualizes the broadcast process as an animation.
    Saves frames as PNG files.
    """
    # Disabled plotting
    pass

def plot_transmission_heat_map(n: int, m: int, edge_usage_history: List[List[Tuple[int, int]]]) -> None:
    """
    Creates a heat map showing the frequency of transmission through each node.
    """
    # Disabled plotting
    pass

def calc_theoretical_min(n: int, m: int, N: int) -> int:
    """
    Calculate the theoretical minimum rounds with perfect pipelining.
    For a spanning tree, we need:
    - tree_depth rounds to reach the furthest node
    - (N-1) additional rounds to send the rest of the information
    """
    # In a mesh, the tree depth is at most the Manhattan distance to the furthest corner
    tree_depth = (n-1) + (m-1)  # From (0,0) to (n-1,m-1)
    return tree_depth + N - 1

def main() -> None:
    global GRID_ROWS, GRID_COLS, INFO_SIZE
    
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python3 BinTreeBcast.py <rows> <cols> <N>")
        print("Example: python3 BinTreeBcast.py 16 16 100")
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
    
    # Use parameters
    n, m = GRID_ROWS, GRID_COLS
    start = START_POINT
    N = INFO_SIZE

    print(f"2D Grid size: {n}×{m}")
    print(f"Information size: {N}")
    print(f"Start point: {start}")

    # Calculate theoretical minimum
    theoretical_min = calc_theoretical_min(n, m, N)
    print(f"Theoretical minimum rounds: {theoretical_min}")

    # Set random seed for reproducibility
    random.seed(42)

    # Run the optimized broadcast simulation
    print("\nRunning optimized binary tree broadcast...")
    total_rounds, active_edges, state_history, edge_usage_history = simulate_optimized_bcast(n, m, start, N)

    print(f"\nBroadcast completed in {total_rounds} rounds.")
    print(f"Active edges per round: {active_edges[:10]}..." if len(active_edges) > 10 else f"Active edges per round: {active_edges}")

    # Print final state
    print("\nFinal chunks per node:")
    nodes, mapping = get_nodes_and_mapping(n, m)
    print_state_grid(state_history[-1], n, m, N, nodes, mapping)

    # Save data to CSV
    save_to_csv(active_edges, n, m, N, "BinTreeBcast")

    # Disabled plotting calls
    # plot_transmissions(active_edges)
    # visualize_broadcast(n, m, state_history, edge_usage_history)
    # plot_transmission_heat_map(n, m, edge_usage_history)

if __name__ == "__main__":
    main() 
