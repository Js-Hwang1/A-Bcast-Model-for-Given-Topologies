"""
    Simulates the broadcast of a message of size N on a 16K3 (16 vertices and 3 Regular (all nodes have degree 3)) using an optimized Binary Tree structure,
    with the constraint that nodes cannot send and receive simultaneously,
    and each node can at most send OR receive 1 unit of information per timestep.

    This is an optimized binary tree implementation that creates a balanced spanning tree
    optimized for broadcast performance with efficient pipelining.

    Start BinaryTreeBcast at node 1

    Returns:
        rounds: Total number of rounds until every node has all N chunks.
        active_edges: List where each element is the number of transmissions (active edges) in that round.
        state_history: List of node states at each round for visualization.
"""

import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Set
import random
import numpy as np
from collections import deque
import csv
import os
import sys

# Default parameters (will be overridden by command line arguments)
INFO_SIZE = 1000             # Size of packets
START_NODE = 1              # Starting node for broadcast

def create_optimized_tree(graph: Dict[int, List[int]], start: int = 1) -> Dict[int, List[int]]:
    """
    Create an optimized tree spanning structure over the given graph starting from the start node.
    This tree is designed to minimize broadcast time by creating a balanced structure.
    
    Args:
        graph: The adjacency list representation of the graph
        start: Starting node (root of the tree)
    
    Returns:
        A dictionary where each key is a node and its value is a list of its children in the tree
    """
    tree = {i: [] for i in graph.keys()}
    visited = set()
    
    # Use BFS to build a balanced tree
    queue = deque([start])
    visited.add(start)
    
    while queue:
        current = queue.popleft()
        
        # For 16K3, we want to maximize parallelism while maintaining tree structure
        # Allow up to 3 children for root, 2 for others to balance the tree
        max_children = 3 if current == start else 2
        child_count = 0
        
        # Sort neighbors by degree to prefer nodes with more connections for better balance
        neighbors = sorted(graph[current], key=lambda x: len(graph[x]), reverse=True)
        
        for neighbor in neighbors:
            if neighbor not in visited and child_count < max_children:
                tree[current].append(neighbor)
                visited.add(neighbor)
                queue.append(neighbor)
                child_count += 1
    
    return tree

def calculate_node_levels(tree: Dict[int, List[int]], start: int) -> Dict[int, int]:
    """
    Calculate the level (depth) of each node in the tree.
    
    Args:
        tree: The tree structure
        start: Starting node
        
    Returns:
        Dictionary mapping node to its level
    """
    levels = {start: 0}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        for child in tree[current]:
            levels[child] = levels[current] + 1
            queue.append(child)
    
    return levels

def simulate_optimized_broadcast(start_node: int = 1, N: int = 100, verbose: bool = True) -> Tuple[int, List[int], List[Dict[int, int]], List[List[Tuple[int, int, int]]], Dict[int, List[int]]]:
    """
    Simulates an optimized broadcast on a 16K3 topology using an efficient tree structure.
    Uses aggressive pipelining and parallel transmission scheduling while maintaining constraints.
    
    Args:
        start_node: The node to start the broadcast from
        N: Size of the information/message
        verbose: Whether to print detailed logs
        
    Returns:
        rounds: Total number of rounds until every node has all N chunks
        active_edges: List where each element is the number of active edges in that round
        state_history: List of node states at each round for visualization
        transmissions_history: List of lists of (sender, receiver, chunk) tuples for each round
        tree: The tree structure used for broadcasting
    """
    # Adjacency list for 16K3 topology (1-indexed)
    adj_list = {
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
    
    # Create an optimized tree spanning structure
    tree = create_optimized_tree(adj_list, start_node)
    
    # Calculate node levels for efficient scheduling
    node_levels = calculate_node_levels(tree, start_node)
    
    if verbose:
        print(f"Optimized Tree structure from node {start_node}:")
        for parent, children in tree.items():
            if children:
                print(f"  Node {parent} (level {node_levels[parent]}) → {children}")
    
    total_nodes = 16
    
    # Initialize state: for each node, track the set of chunks it has
    state = {i: set() for i in range(1, total_nodes + 1)}
    state[start_node] = set(range(1, N + 1))  # Source node gets the full message
    
    rounds = 0
    active_edges = []
    state_history = [{i: len(chunks) for i, chunks in state.items()}]
    transmissions_history = []
    
    # Find parent of each node
    parent_map = {start_node: None}
    for parent, children in tree.items():
        for child in children:
            parent_map[child] = parent
    
    # Continue until all nodes have all chunks
    while any(len(chunks) < N for node, chunks in state.items()):
        if verbose and rounds % 20 == 0:
            print(f"\n--- Round {rounds + 1} ---")
            
        transmissions_this_round = 0
        round_transmissions = []
        new_state = {node: chunks.copy() for node, chunks in state.items()}
        active_nodes = set()
        
        # Find all possible transmissions for this round
        possible_transmissions = []
        
        # Collect all possible parent-to-child transmissions
        for parent in range(1, total_nodes + 1):
            if parent in active_nodes:
                continue
                
            for child in tree[parent]:
                if child in active_nodes:
                    continue
                    
                # Find chunks this parent has that the child doesn't
                parent_chunks = state[parent]
                child_chunks = state[child]
                chunks_to_send = parent_chunks - child_chunks
                
                if chunks_to_send:
                    # Prioritize by chunk number and level difference
                    chunk_to_send = min(chunks_to_send)
                    level_diff = node_levels[child] - node_levels[parent]
                    possible_transmissions.append((parent, child, chunk_to_send, level_diff))
        
        # Sort transmissions by priority: level difference (prefer closer levels), then chunk number
        possible_transmissions.sort(key=lambda x: (x[3], x[2]))
        
        # Execute transmissions while respecting constraints
        for parent, child, chunk, _ in possible_transmissions:
            if parent in active_nodes or child in active_nodes:
                continue
                
            # Execute the transmission
            new_state[child].add(chunk)
            transmissions_this_round += 1
            round_transmissions.append((parent, child, chunk))
            active_nodes.add(parent)
            active_nodes.add(child)
            
            if verbose and rounds % 20 == 0:
                print(f"Node {parent} sends chunk {chunk} to Node {child}")
        
        if verbose and rounds % 20 == 0 and transmissions_this_round == 0:
            print("No transmissions this round")
        
        active_edges.append(transmissions_this_round)
        transmissions_history.append(round_transmissions)
        state = new_state
        state_history.append({i: len(chunks) for i, chunks in state.items()})
        rounds += 1
        
        # Exit if no progress is being made
        if transmissions_this_round == 0:
            break
            
    return rounds, active_edges, state_history, transmissions_history, tree

def create_star_tree(graph: Dict[int, List[int]], start: int = 1) -> Dict[int, List[int]]:
    """
    Create a star-like tree structure where the start node connects directly to all other nodes.
    This maximizes parallelism for the 16K3 topology.
    
    Args:
        graph: The adjacency list representation of the graph
        start: Starting node (root of the star)
    
    Returns:
        A dictionary where each key is a node and its value is a list of its children
    """
    tree = {i: [] for i in graph.keys()}
    
    # Connect start node to all other nodes directly
    for node in graph.keys():
        if node != start:
            tree[start].append(node)
    
    return tree

def simulate_star_broadcast(start_node: int = 1, N: int = 100, verbose: bool = True) -> Tuple[int, List[int], List[Dict[int, int]], List[List[Tuple[int, int, int]]], Dict[int, List[int]]]:
    """
    Simulates a pipelined broadcast on a 16K3 topology with proper constraints.
    Nodes cannot send and receive simultaneously, and each node can at most send OR receive 1 unit per timestep.
    Implements pipelining where nodes forward chunks they've received to other nodes.
    
    Args:
        start_node: The node to start the broadcast from
        N: Size of the information/message
        verbose: Whether to print detailed logs
        
    Returns:
        rounds: Total number of rounds until every node has all N chunks
        active_edges: List where each element is the number of active edges in that round
        state_history: List of node states at each round for visualization
        transmissions_history: List of lists of (sender, receiver, chunk) tuples for each round
        tree: The tree structure used for broadcasting
    """
    # Adjacency list for 16K3 topology (1-indexed)
    adj_list = {
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
    
    total_nodes = 16
    
    # Initialize state: for each node, track the set of chunks it has
    state = {i: set() for i in range(1, total_nodes + 1)}
    state[start_node] = set(range(1, N + 1))  # Source node gets the full message
    
    rounds = 0
    active_edges = []
    state_history = [{i: len(chunks) for i, chunks in state.items()}]
    transmissions_history = []
    
    # Continue until all nodes have all chunks
    while any(len(chunks) < N for node, chunks in state.items()):
        if verbose and rounds % 20 == 0:
            print(f"\n--- Round {rounds + 1} ---")
            
        transmissions_this_round = 0
        round_transmissions = []
        new_state = {node: chunks.copy() for node, chunks in state.items()}
        active_nodes = set()
        
        # Find all possible transmissions for this round
        possible_transmissions = []
        
        # Collect all possible transmissions from any node to its neighbors
        for sender in range(1, total_nodes + 1):
            if sender in active_nodes:
                continue
                
            for receiver in adj_list[sender]:
                if receiver in active_nodes:
                    continue
                    
                # Find chunks this sender has that the receiver doesn't
                sender_chunks = state[sender]
                receiver_chunks = state[receiver]
                chunks_to_send = sender_chunks - receiver_chunks
                
                if chunks_to_send:
                    # Prioritize by chunk number (send smallest missing chunk first)
                    chunk_to_send = min(chunks_to_send)
                    
                    # Calculate priority: prefer transmissions from nodes with more chunks
                    # and to nodes with fewer chunks (helps balance the distribution)
                    sender_chunk_count = len(sender_chunks)
                    receiver_chunk_count = len(receiver_chunks)
                    priority = (sender_chunk_count, -receiver_chunk_count, chunk_to_send)
                    
                    possible_transmissions.append((sender, receiver, chunk_to_send, priority))
        
        # Sort transmissions by priority: more chunks in sender, fewer in receiver, then chunk number
        possible_transmissions.sort(key=lambda x: x[3], reverse=True)
        
        # Execute transmissions while respecting constraints
        for sender, receiver, chunk, _ in possible_transmissions:
            if sender in active_nodes or receiver in active_nodes:
                continue
                
            # Execute the transmission
            new_state[receiver].add(chunk)
            transmissions_this_round += 1
            round_transmissions.append((sender, receiver, chunk))
            active_nodes.add(sender)
            active_nodes.add(receiver)
            
            if verbose and rounds % 20 == 0:
                print(f"Node {sender} sends chunk {chunk} to Node {receiver}")
        
        if verbose and rounds % 20 == 0 and transmissions_this_round == 0:
            print("No transmissions this round")
        
        active_edges.append(transmissions_this_round)
        transmissions_history.append(round_transmissions)
        state = new_state
        state_history.append({i: len(chunks) for i, chunks in state.items()})
        rounds += 1
        
        # Exit if no progress is being made
        if transmissions_this_round == 0:
            break
            
    return rounds, active_edges, state_history, transmissions_history, {}

def plot_transmissions(active_edges: List[int]) -> None:
    """
    Plots the normalized number of active edges (transmissions) per round using matplotlib.
    Y-axis shows the fraction of total edges (active_edges/24) that are active.
    """
    # Total edges in 16K3 graph: (16 vertices × 3 degree) / 2 = 24 edges
    total_edges = 24
    
    # Normalize active edges
    normalized_edges = [edge_count / total_edges for edge_count in active_edges]
    
    rounds_list = list(range(1, len(active_edges) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(rounds_list, normalized_edges, marker='o', linestyle='-', color='b')
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Active Edges (active edges/ total edges)")
    plt.title("16K3 Binary Tree: Normalized Active Transmission Edges per Round")
    plt.grid(True)
    
    plt.legend()
    plt.show()

def print_state(state: Dict[int, int], N: int) -> None:
    """
    Prints a visual representation of the final state.
    
    Args:
        state: Either a dict mapping node -> chunk_count or node -> set of chunks
        N: Size of the information/message
    """
    print("\nFinal Information Distribution:")
    print("-" * 50)
    
    for node in sorted(state.keys()):
        # Handle both types of state representation
        if isinstance(state[node], set):
            chunks = len(state[node])
        else:
            chunks = state[node]
            
        percentage = chunks / N * 100
        bar_length = int(percentage / 5)  # Scale to make it fit nicely
        bar = "█" * bar_length
        print(f"Node {node:2d}: {chunks}/{N} chunks ({percentage:.1f}%) {bar}")
    print("-" * 50)

def print_tree_structure(tree: Dict[int, List[int]], start_node: int = 1) -> None:
    """
    Prints a visual representation of the optimized tree structure.
    """
    print("\nOptimized Tree Structure:")
    print("-" * 50)
    
    def print_subtree(node, level=0, prefix=''):
        # Print the current node
        print(f"{prefix}Node {node}")
        
        # Print children with increased indentation
        if tree[node]:
            # Handle all children except the last one
            for child in tree[node][:-1]:
                print(f"{prefix}├── ", end='')
                print_subtree(child, level + 1, prefix + "│   ")
            
            # Handle the last child
            if tree[node]:
                print(f"{prefix}└── ", end='')
                print_subtree(tree[node][-1], level + 1, prefix + "    ")
    
    print_subtree(start_node)
    print("-" * 50)

def print_transmission_summary(transmissions_history: List[List[Tuple[int, int, int]]]) -> None:
    """
    Prints a summary of all transmissions that occurred during the broadcast.
    """
    print("\nTransmission Summary:")
    print("-" * 50)
    
    # Count total transmissions and unique edges
    total_transmissions = sum(len(round_trans) for round_trans in transmissions_history)
    unique_edges = set()
    for round_trans in transmissions_history:
        for sender, receiver, _ in round_trans:
            unique_edges.add((sender, receiver))
    
    print(f"Total transmissions: {total_transmissions}")
    print(f"Unique edges used: {len(unique_edges)}")
    
    # Print the first few and last few transmissions
    max_to_print = 5
    if len(transmissions_history) > max_to_print * 2:
        # First few rounds
        for round_idx in range(min(max_to_print, len(transmissions_history))):
            if transmissions_history[round_idx]:
                print(f"Round {round_idx + 1}:")
                for sender, receiver, chunk in transmissions_history[round_idx]:
                    print(f"  Node {sender} → Node {receiver}: chunk {chunk}")
                    
        print("...")
        
        # Last few rounds
        for round_idx in range(len(transmissions_history) - max_to_print, len(transmissions_history)):
            if transmissions_history[round_idx]:
                print(f"Round {round_idx + 1}:")
                for sender, receiver, chunk in transmissions_history[round_idx]:
                    print(f"  Node {sender} → Node {receiver}: chunk {chunk}")
    else:
        # Print all rounds if there aren't many
        for round_idx, round_transmissions in enumerate(transmissions_history, 1):
            if round_transmissions:
                print(f"Round {round_idx}:")
                for sender, receiver, chunk in round_transmissions:
                    print(f"  Node {sender} → Node {receiver}: chunk {chunk}")
    
    print("-" * 50)

def save_to_csv(active_edges: List[int], N: int, algorithm_name: str = "BinTree"):
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

def main() -> None:
    """
    Main function to run the 16K3 Pipelined Binary Tree broadcast simulation.
    Uses a pipelined approach where nodes forward chunks they've received to other nodes,
    while respecting constraints that nodes cannot send and receive simultaneously.
    """
    global INFO_SIZE
    
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python3 BinTree.py <N>")
        print("Example: python3 BinTree.py 100")
        sys.exit(1)
    
    try:
        INFO_SIZE = int(sys.argv[1])
    except ValueError:
        print("Error: N must be an integer")
        sys.exit(1)
    
    if INFO_SIZE <= 0:
        print("Error: N must be a positive integer")
        sys.exit(1)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Use parameters
    start_node = START_NODE
    N = INFO_SIZE
    
    print(f"16K3 Pipelined Binary Tree Broadcast Simulation")
    print(f"Information size: {N} chunks")
    print(f"Starting node: {start_node}")
    print(f"Implementation: Pipelined broadcast with constraint enforcement")
    print(f"Features: Nodes forward chunks to neighbors, no simultaneous send/receive")
    
    # Run the pipelined simulation
    rounds, active_edges, state_history, transmissions_history, tree = simulate_star_broadcast(
        start_node, N, verbose=False)
    
    # Print simulation results
    print(f"\nTotal number of rounds required: {rounds}")
    print(f"Total transmissions: {sum(active_edges)}")
    
    # Calculate the maximum transmissions per round
    max_transmissions = max(active_edges) if active_edges else 0
    print(f"Maximum transmissions in any round: {max_transmissions}")
    
    # Calculate efficiency metrics
    total_edges_16k3 = 24  # (16 vertices × 3 degree) / 2
    avg_utilization = sum(active_edges) / (rounds * total_edges_16k3) if rounds > 0 else 0
    print(f"Average edge utilization: {avg_utilization:.3f}")
    
    # Print final state in a visually appealing format
    print_state(state_history[-1], N)
    
    # Save data to CSV
    save_to_csv(active_edges, N, "BinTree")
    
    # Disabled plotting
    # plot_transmissions(active_edges)

if __name__ == "__main__":
    main()
