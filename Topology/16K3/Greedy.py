import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Set
import random
import numpy as np
from collections import deque
import csv
import os
import sys


INFO_SIZE = 1000             # Size of packets
START_NODE = 1              # Starting node for broadcast

def simulate_vanilla_bfs_16k3(start_node: int = 1, N: int = 100, verbose: bool = True) -> Tuple[int, List[int], List[Dict[int, Set[int]]], List[List[Tuple[int, int, int]]]]:
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
    
    # Initialize state: for each node, track the set of chunks it has
    state = {i: set() for i in range(1, 17)}
    state[start_node] = set(range(1, N + 1))  # Start node has all N pieces of information
    
    bfs_parent = {start_node: None}
    bfs_children = {i: [] for i in range(1, 17)}

    queue = deque([start_node])
    visited = {start_node}
    
    while queue:
        node = queue.popleft()
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                bfs_parent[neighbor] = node
                bfs_children[node].append(neighbor)
    
    rounds = 0
    active_edges = []
    state_history = [state.copy()]
    transmissions_history = []

    while any(len(chunks) < N for node, chunks in state.items()):
        if verbose and rounds % 10 == 0:
            print(f"\n--- Round {rounds + 1} ---")

        active_nodes = set()
        round_transmissions = []
        transmissions_this_round = 0
        
        new_state = {node: chunks.copy() for node, chunks in state.items()}
        
        for level in range(len(visited)):
            level_nodes = [node for node in visited if bfs_level(node, bfs_parent) == level]
            
            for node in level_nodes:
                if node in active_nodes:
                    continue
                
                # Node that has information can send to its BFS children
                if len(state[node]) > 0:
                    for child in bfs_children[node]:
                        # Skip if child already has all N pieces or is active
                        if len(state[child]) >= N or child in active_nodes:
                            continue
                            
                        # Find a chunk that parent has but child doesn't
                        parent_chunks = state[node]
                        child_chunks = state[child]
                        chunks_to_send = parent_chunks - child_chunks
                        
                        if chunks_to_send:
                            # Send the smallest missing chunk
                            chunk_to_send = min(chunks_to_send)
                            new_state[child].add(chunk_to_send)
                            transmissions_this_round += 1
                            round_transmissions.append((node, child, chunk_to_send))
                            active_nodes.add(node)
                            active_nodes.add(child)
                            
                            if verbose and rounds % 10 == 0:
                                print(f"Node {node} sends chunk {chunk_to_send} to Node {child}")
                            
                            # Only one transmission per node
                            break
        
        if transmissions_this_round == 0 and verbose and rounds % 10 == 0:
            print("No transmissions this round")
        
        # Record history for this round
        active_edges.append(transmissions_this_round)
        transmissions_history.append(round_transmissions)
        state = new_state
        state_history.append(state.copy())
        
        rounds += 1
        
        # Stop if no progress can be made
        if transmissions_this_round == 0:
            break
    
    return rounds, active_edges, state_history, transmissions_history

def simulate_greedy_pipeline_16k3(start_node: int = 1, N: int = 100, verbose: bool = True) -> Tuple[int, List[int], List[Dict[int, Set[int]]], List[List[Tuple[int, int, int]]]]:
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
    
    # Initialize state: for each node, track the set of chunks it has
    state = {i: set() for i in range(1, 17)}
    state[start_node] = set(range(1, N + 1))  # Start node has all N pieces of information
    
    rounds = 0
    active_edges = []
    state_history = [state.copy()]
    transmissions_history = []
    
    # Continue until all nodes have all N pieces of information
    while any(len(chunks) < N for node, chunks in state.items()):
        if verbose and rounds % 10 == 0:
            print(f"\n--- Round {rounds + 1} ---")
        
        # Track which nodes are active this round (either sending or receiving)
        active_nodes = set()
        round_transmissions = []
        transmissions_this_round = 0
        
        # Build a list of all possible transmissions for this round
        possible_transmissions = []
        for sender in range(1, 17):
            if len(state[sender]) == 0:
                continue  # Sender has no information to share
            for receiver in adj_list[sender]:
                # Skip if receiver already has all N pieces
                if len(state[receiver]) >= N:
                    continue
                # Sender can share one piece of information with receiver
                possible_transmissions.append((sender, receiver))
        
        # Randomize order to avoid bias from node numbering
        random.shuffle(possible_transmissions)
        
        # Greedily select transmissions to maximize active edges
        for sender, receiver in possible_transmissions:
            # Skip if either node became active in a previous transmission this round
            if sender in active_nodes or receiver in active_nodes:
                continue
            
            # Find a chunk that sender has but receiver doesn't
            sender_chunks = state[sender]
            receiver_chunks = state[receiver]
            chunks_to_send = sender_chunks - receiver_chunks
            
            if chunks_to_send:
                # Send the smallest missing chunk
                chunk_to_send = min(chunks_to_send)
                state[receiver].add(chunk_to_send)
                transmissions_this_round += 1
                round_transmissions.append((sender, receiver, chunk_to_send))
                active_nodes.add(sender)
                active_nodes.add(receiver)
                if verbose and rounds % 10 == 0:
                    print(f"Node {sender} sends chunk {chunk_to_send} to Node {receiver}")
        
        if transmissions_this_round == 0 and verbose and rounds % 10 == 0:
            print("No transmissions this round")
        
        # Record history for this round
        active_edges.append(transmissions_this_round)
        transmissions_history.append(round_transmissions)
        state_history.append(state.copy())
        rounds += 1
        
        # Stop if no progress can be made
        if transmissions_this_round == 0:
            break
    
    return rounds, active_edges, state_history, transmissions_history

def bfs_level(node, parent_map):
    """Calculate BFS level of a node from the root"""
    level = 0
    current = node
    while parent_map.get(current) is not None:
        level += 1
        current = parent_map[current]
    return level

def plot_transmissions(active_edges: List[int], title: str = "16K3: Normalized Active Transmission Edges per Round") -> None:
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
    plt.title(title)
    plt.grid(True)
    
    plt.legend()
    plt.show()

def print_state(state: Dict[int, Set[int]], N: int) -> None:
    """
    Prints a visual representation of the final state.
    """
    print("\nFinal Information Distribution:")
    print("-" * 50)
    for node in sorted(state.keys()):
        pieces = len(state[node])
        percentage = pieces / N * 100
        bar_length = int(percentage / 5)  # Scale to make it fit nicely
        bar = "█" * bar_length
        print(f"Node {node:2d}: {pieces}/{N} pieces ({percentage:.1f}%) {bar}")
    print("-" * 50)

def print_transmission_summary(transmissions_history: List[List[Tuple[int, int, int]]], N: int) -> None:
    """
    Prints a summary of all transmissions that occurred during the broadcast.
    """
    print("\nTransmission Summary:")
    print("-" * 50)
    
    # Count total transmissions
    total_transmissions = sum(len(round_trans) for round_trans in transmissions_history)
    print(f"Total transmissions: {total_transmissions}")
    
    # Calculate how many pieces each node receives
    node_pieces = {}
    for round_idx, round_transmissions in enumerate(transmissions_history, 1):
        for sender, receiver, chunk in round_transmissions:
            if receiver not in node_pieces:
                node_pieces[receiver] = 0
            node_pieces[receiver] += 1
    
    # Print how many rounds it took for each node to receive all pieces
    print(f"\nRounds required for nodes to receive all {N} pieces:")
    
    for node in sorted(node_pieces.keys()):
        if node_pieces[node] == N:
            # Find the round where this node got its last piece
            last_round = 0
            for round_idx, round_transmissions in enumerate(transmissions_history, 1):
                for sender, receiver, chunk in round_transmissions:
                    if receiver == node and node_pieces[node] == N:
                        last_round = round_idx
                        
            print(f"Node {node}: {last_round} rounds")
        else:
            print(f"Node {node}: Incomplete ({node_pieces[node]}/{N} pieces)")
    
    # Print per-round transmission details (first 10 and last 10)
    print("\nRound Transmission Details:")
    for round_idx in range(min(10, len(transmissions_history))):
        if transmissions_history[round_idx]:
            print(f"Round {round_idx + 1}: {len(transmissions_history[round_idx])} transmissions")
            for sender, receiver, chunk in sorted(transmissions_history[round_idx]):
                print(f"  Node {sender} → Node {receiver}: chunk {chunk}")
    
    if len(transmissions_history) > 20:
        print("...")
        
    for round_idx in range(max(10, len(transmissions_history) - 10), len(transmissions_history)):
        if transmissions_history[round_idx]:
            print(f"Round {round_idx + 1}: {len(transmissions_history[round_idx])} transmissions")
            for sender, receiver, chunk in sorted(transmissions_history[round_idx]):
                print(f"  Node {sender} → Node {receiver}: chunk {chunk}")
    
    print("-" * 50)

def compare_algorithms(N: int = 100, start_node: int = 1) -> None:
    """
    Compare vanilla BFS and greedy pipelined algorithms
    """
    print("Comparing broadcast algorithms on 16K3 topology...")
    
    # Run vanilla BFS
    vanilla_rounds, vanilla_edges, vanilla_state, vanilla_trans = simulate_vanilla_bfs_16k3(start_node, N, verbose=False)
    
    # Run greedy pipelined
    greedy_rounds, greedy_edges, greedy_state, greedy_trans = simulate_greedy_pipeline_16k3(start_node, N, verbose=False)
    
    # Print comparison
    print("\n" + "=" * 60)
    print(f"COMPARISON (N={N}, start={start_node})")
    print("=" * 60)
    print(f"Vanilla BFS: {vanilla_rounds} rounds, {sum(vanilla_edges)} total transmissions")
    print(f"Greedy Pipelined: {greedy_rounds} rounds, {sum(greedy_edges)} total transmissions")
    print(f"Improvement: {vanilla_rounds - greedy_rounds} rounds ({((vanilla_rounds - greedy_rounds) / vanilla_rounds) * 100:.2f}%)")
    
    # Plot combined comparison
    plt.figure(figsize=(12, 6))
    
    # Total edges
    total_edges = 24
    
    # Plot vanilla BFS
    rounds_list = list(range(1, len(vanilla_edges) + 1))
    normalized_edges = [edge_count / total_edges for edge_count in vanilla_edges]
    plt.plot(rounds_list, normalized_edges, marker='o', linestyle='-', color='b', label='Vanilla BFS')
    
    # Plot greedy pipelined
    rounds_list = list(range(1, len(greedy_edges) + 1))
    normalized_edges = [edge_count / total_edges for edge_count in greedy_edges]
    plt.plot(rounds_list, normalized_edges, marker='s', linestyle='-', color='r', label='Greedy Pipelined')
    
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Active Edges (active edges / total edges)")
    plt.title(f"16K3 Broadcast: Normalized Active Transmission Edges Comparison (N={N})")
    plt.grid(True)
    plt.legend()
    plt.show()

def save_to_csv(active_edges: List[int], N: int, algorithm_name: str = "Greedy"):
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
    Main function to run the 16K3 broadcast simulations.
    """
    global INFO_SIZE
    
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python3 Greedy.py <N>")
        print("Example: python3 Greedy.py 100")
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
    
    print(f"16K3 Greedy Pipelined Broadcast Simulation")
    print(f"Information size: {N} chunks")
    print(f"Starting node: {start_node}")
    print(f"Implementation: Greedy algorithm with pipelining")
    print(f"Constraints: Each node can only send OR receive once per round")
    
    # Run the simulation
    rounds, active_edges, state_history, transmissions_history = simulate_greedy_pipeline_16k3(start_node, N, verbose=False)
    
    # Print simulation results
    print(f"\nTotal number of rounds required: {rounds}")
    print(f"Total transmissions: {sum(active_edges)}")
    
    # Calculate statistics for active edges
    max_active = max(active_edges) if active_edges else 0
    avg_active = sum(active_edges) / len(active_edges) if active_edges else 0
    print(f"Maximum active edges in any round: {max_active}")
    print(f"Average active edges per round: {avg_active:.2f}")
    
    # Print final state
    print_state(state_history[-1], N)
    
    # Save data to CSV
    save_to_csv(active_edges, N, "Greedy")
    
    # Disabled plotting
    # plot_transmissions(active_edges, "16K3 Greedy Pipelined: Normalized Active Transmission Edges per Round")

if __name__ == "__main__":
    main()
