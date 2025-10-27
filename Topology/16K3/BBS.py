import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Set, Any, Optional
import random
import numpy as np
import csv
import os
import sys
from collections import deque

# Default parameters (will be overridden by command line arguments)
INFO_SIZE = 1000             # Size of information
START_NODE = 1              # Starting node for broadcast

def define_cycles() -> List[List[Tuple[int, int]]]:
    """
    Defines 3 distinct cycles for the 16K3 topology.
    Each cycle is designed to allow exactly 8 nodes to transmit simultaneously (V/2).
    
    Returns:
        List of 3 cycles, where each cycle is a list of (sender, receiver) pairs.
    """
    # The three cycles are designed to cover the entire graph
    # Each cycle activates exactly 8 edges when in steady state
    
    cycle1 = [
        (1,2),(5,4),(7,6),(14,8),(9,10),(11,12),(3,13)
    ]
   
    cycle2 = [
        (1,3),(2,7),(12,6),(5,10),(4,15),(11,16),(9,8),(13,14)

    ]
   
    cycle3 = [
        (1,4),(2,11),(3,9),(7,8),(16,10),(5,6),(13,12),(15,14)
    ]

    cycle4 = [
        (1,2),(4,5),(6,7),(8,14),(10,9),(3,13),(15,16)
    ]
   
    cycle5 = [
        (1,3),(7,2),(6,12),(10,5),(4,15),(16,11),(8,9),(14,13)

    ]
   
    cycle6 = [
        (1,4),(2,11),(9,3),(8,7),(10,16),(6,5),(14,15)
    ]


    
    return [cycle1, cycle2, cycle3, cycle4, cycle5, cycle6]

def bfs_distances(adj: Dict[int, List[int]], root: int) -> Dict[int, int]:
    """Compute BFS distances from root over arbitrary graph (nodes are ints)."""
    dist: Dict[int, int] = {v: -1 for v in adj}
    q = deque([root])
    dist[root] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def verify_cycles_are_matchings(cycles: List[List[Tuple[int, int]]], adj: Dict[int, List[int]]) -> None:
    """Assert each cycle is a directed matching using only adjacency edges (half‑duplex, neighbor‑only)."""
    for idx, cyc in enumerate(cycles):
        used: Set[int] = set()
        for u, v in cyc:
            # neighbor legality
            assert v in adj.get(u, []), f"Cycle {idx+1} has non-adjacent edge ({u},{v})"
            # matching legality: node appears at most once
            assert u not in used and v not in used, f"Cycle {idx+1} violates matching at node {u if u in used else v}"
            used.add(u)
            used.add(v)


def order_cycles_urgent(
    cycles: List[List[Tuple[int, int]]],
    adj: Dict[int, List[int]],
    root: int,
    N: int
) -> List[List[Tuple[int, int]]]:
    """
    Activation-first greedy ordering of cycles, mirroring 2D/3D frame ordering:
      - Score per unused cycle aggregates over deliverable edges u→v:
        urgency (N - recv[v]) * (dist[v] + 1), forward progress, deliverables, activations
      - Smoothing penalizes sharp activation jumps early on
      - If no positive score exists, fall back to max deliverables; else any remaining
    Returns a reordered list of cycles; use this static order in a repeated cycle.
    """
    # Distances for forward progress/urgency
    dist = bfs_distances(adj, root)

    used: Set[int] = set()
    ordered: List[List[Tuple[int, int]]] = []
    # Simulated received-counts per node
    recv = {v: 0 for v in adj}
    recv[root] = N

    last_activations: Optional[int] = None

    while len(used) < len(cycles):
        best_idx = -1
        best_score = -1e18

        for i, cyc in enumerate(cycles):
            if i in used:
                continue
            S_i = 0.0
            deliverables = 0
            activations = 0
            forward_sum = 0.0
            for u, v in cyc:
                if recv[u] > 0 and recv[v] < N:
                    base = (N - recv[v]) * (dist.get(v, 0) + 1)
                    S_i += base
                    deliverables += 1
                    if recv[v] == 0:
                        activations += 1
                    dv = (dist.get(v, 0) - dist.get(u, 0))
                    if dv > 0:
                        forward_sum += dv
                    elif dv == 0:
                        forward_sum -= 0.5
                    else:
                        forward_sum -= 2.0

            smoothing = 0.0
            if last_activations is not None:
                diff = abs(activations - last_activations)
                if len(used) <= 1:
                    smoothing = -6.0 * diff
                else:
                    smoothing = -3.0 * diff

            score = 0.5 * S_i + 0.5 * forward_sum + 8.0 * deliverables + 200.0 * activations + smoothing
            if score > best_score:
                best_score = score
                best_idx = i

        # fallback
        if best_idx == -1 or best_score <= 0:
            best_count = -1
            chosen = -1
            for i, cyc in enumerate(cycles):
                if i in used:
                    continue
                cnt = 0
                for u, v in cyc:
                    if recv[u] > 0 and recv[v] < N:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    chosen = i
            if best_count <= 0:
                for i in range(len(cycles)):
                    if i not in used:
                        chosen = i
                        break
        else:
            chosen = best_idx

        # play chosen cycle in simulation
        a_next = 0
        for u, v in cycles[chosen]:
            if recv[u] > 0 and recv[v] < N:
                if recv[v] == 0:
                    a_next += 1
                recv[v] += 1
        used.add(chosen)
        ordered.append(cycles[chosen])
        last_activations = a_next

    return ordered


def select_chunk_activation_first(
    available_chunks: Set[int],
    sender: int,
    receiver: int,
    adj: Dict[int, List[int]],
    data_state: Dict[int, Set[int]],
    sender_hist: Dict[int, List[int]],
    step_num: int,
) -> int:
    """
    Activation-first greedy selector (distinct messages):
      - Prefer chunks that maximize receiver's ability to forward to its neighbors next step
      - Light smoothing to avoid tight repeats and seed early determinism
    Returns chunk id, or -1 if none available.
    """
    if not available_chunks:
        return -1
    r_nbrs = adj.get(receiver, [])
    recent = set(sender_hist.get(sender, [])[-4:])
    best_chunk = -1
    best_score = -10**9
    for c in available_chunks:
        fwd = 0
        for nb in r_nbrs:
            if c not in data_state[nb]:
                fwd += 1
        score = 100 * fwd
        if len(data_state[receiver]) == 0:
            score += max(0, 200 - c) // 10
        if c not in recent:
            score += 3
        if c not in sender_hist.get(sender, []):
            score += 5
        if score > best_score:
            best_score = score
            best_chunk = c
    return best_chunk


def simulate_bcclp(start_node: int = 1, N: int = 100, verbose: bool = True) -> Tuple[int, List[int], List[Dict[int, Set[int]]], List[List[Tuple[int, int, int]]]]:
    """
    Simulates a BCCLP broadcast on 16K3 topology using 6 cycles.
    Strictly follows the cycle sequence, with the only constraint being that
    we skip an edge if the receiver already has all N chunks.
    
    Args:
        start_node: The node to start the broadcast from
        N: Size of the information/message
        verbose: Whether to print detailed logs
        
    Returns:
        rounds: Total number of rounds until every node has all N chunks
        active_edges: List where each element is the number of active edges in that round
        state_history: List of node states at each round for visualization
        transmissions_history: List of lists of (sender, receiver, chunk) tuples for each round
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
    
    # Get the cycles that define the broadcast pattern
    cycles = define_cycles()
    
    total_nodes = 16
    
    # Initialize state: for each node, track the set of chunks it has (distinct messages)
    state: Dict[int, Set[int]] = {i: set() for i in range(1, total_nodes + 1)}
    state[start_node] = set(range(1, N + 1))  # Source node gets the full message

    # Verify legality (matching + neighbors) and build greedy order once (static cycle)
    verify_cycles_are_matchings(cycles, adj_list)
    cycles = order_cycles_urgent(cycles, adj_list, start_node, N)

    rounds = 0
    active_edges: List[int] = []
    state_history: List[Dict[int, Set[int]]] = [state.copy()]
    transmissions_history: List[List[Tuple[int, int, int]]] = []  # [sender, receiver, chunk]
    
    # Define chunk groups for diversity optimization
    # Group chunks into equal-sized groups to maximize diversity
    group_size = max(1, N // 8)  # 8 groups for 16K3 topology
    chunk_groups = []
    for i in range(0, N, group_size):
        group = list(range(i + 1, min(i + group_size + 1, N + 1)))
        if group:
            chunk_groups.append(group)
    
    # Track what each sender has sent to maintain sender diversity
    sender_transmission_history: Dict[int, List[int]] = {i: [] for i in range(1, total_nodes + 1)}
    
    if verbose:
        print(f"Chunk groups for diversity optimization:")
        for i, group in enumerate(chunk_groups):
            print(f"  Group {i+1}: {group}")
    
    # Continue until every node is fully informed
    while any(len(chunks) < N for node, chunks in state.items()):
        if verbose:
            print(f"\n--- Round {rounds + 1} ---")
        
        # Choose the current cycle (static, greedy-ordered; reuse cyclically)
        current_cycle = cycles[rounds % len(cycles)]
        
        transmissions_this_round = 0
        new_state = {node: chunks.copy() for node, chunks in state.items()}
        round_transmissions = []
        
        # Process each edge in the current cycle strictly (one message per edge)
        for sender, receiver in current_cycle:
            # Only skip if the receiver already has all chunks
            # or if the sender has zero chunks (can't send anything)
            if len(state[receiver]) < N and len(state[sender]) > 0:
                # Find chunks that sender has but receiver doesn't
                sender_chunks = state[sender]
                receiver_chunks = state[receiver]
                chunks_to_send = sender_chunks - receiver_chunks
                
                if chunks_to_send:
                    # Activation-first chunk selection (distinct messages)
                    chunk_to_send = select_chunk_activation_first(
                        chunks_to_send, sender, receiver, adj_list, state, sender_transmission_history, rounds
                    )
                    if chunk_to_send == -1:
                        if verbose:
                            print(f"Skipping ({sender},{receiver}): no eligible chunk to send")
                        continue
                    new_state[receiver].add(chunk_to_send)
                    
                    # Update sender's transmission history
                    sender_transmission_history[sender].append(chunk_to_send)
                    
                    transmissions_this_round += 1
                    round_transmissions.append((sender, receiver, chunk_to_send))
                    
                    if verbose:
                        print(f"Node {sender} sends chunk {chunk_to_send} to Node {receiver}")
                elif verbose:
                    print(f"Skipping ({sender},{receiver}): no new chunks to send")
            elif verbose:
                if len(state[receiver]) >= N:
                    print(f"Skipping ({sender},{receiver}): receiver already has all {N} chunks")
                elif len(state[sender]) == 0:
                    print(f"Skipping ({sender},{receiver}): sender has 0 chunks")
        
        if transmissions_this_round == 0 and verbose:
            print("No transmissions this round")
            
        active_edges.append(transmissions_this_round)
        transmissions_history.append(round_transmissions)
        state = new_state
        state_history.append(state.copy())
        rounds += 1
        
        # Exit if no transmissions occurred (to prevent infinite loop)
        if transmissions_this_round == 0:
            break
    
    return rounds, active_edges, state_history, transmissions_history

def select_optimal_chunk_for_sender(available_chunks: Set[int], sender: int, sender_transmission_history: Dict[int, List[int]], chunk_groups: List[List[int]], round_num: int) -> int:
    """
    Selects the optimal chunk to send to maximize sender diversity.
    
    Strategy:
    1. Calculate diversity score for each available chunk based on sender's history
    2. Prefer chunks from groups that the sender has sent fewer chunks from
    3. Use round number to add some randomization for better distribution
    
    Args:
        available_chunks: Set of chunks the sender can send
        sender: The sender node ID
        sender_transmission_history: Dictionary tracking what each sender has sent
        chunk_groups: List of chunk groups for diversity calculation
        round_num: Current round number for randomization
        
    Returns:
        The optimal chunk to send
    """
    if not available_chunks:
        return min(available_chunks)
    
    # Get sender's transmission history
    sender_history = sender_transmission_history[sender]
    
    # Calculate diversity scores for each available chunk
    chunk_scores = {}
    
    for chunk in available_chunks:
        # Find which group this chunk belongs to
        chunk_group_idx = None
        for i, group in enumerate(chunk_groups):
            if chunk in group:
                chunk_group_idx = i
                break
        
        if chunk_group_idx is None:
            # If chunk doesn't belong to any group, use default scoring
            chunk_scores[chunk] = 1.0
            continue
        
        # Count how many chunks from this group the sender has already sent
        group = chunk_groups[chunk_group_idx]
        sender_chunks_sent_from_group = len(set(sender_history).intersection(set(group)))
        total_chunks_in_group = len(group)
        
        # Calculate diversity score: prefer groups that sender has sent fewer chunks from
        # Lower score = higher priority (we want to minimize this)
        diversity_score = sender_chunks_sent_from_group / total_chunks_in_group
        
        # Add small randomization based on round number to break ties
        # and ensure better distribution across rounds
        randomization = (chunk + round_num) % 1000 / 10000.0
        
        chunk_scores[chunk] = diversity_score + randomization
    
    # Select chunk with lowest score (highest diversity priority)
    optimal_chunk = min(available_chunks, key=lambda c: chunk_scores[c])
    
    return optimal_chunk

def plot_active_edges(active_edges: List[int]) -> None:
    """
    Plots the normalized number of active edges per round for the entire simulation.
    Y-axis shows the fraction of total edges (active_edges/24) that are active.
    """
    # Total edges in 16K3 graph: (16 vertices × 3 degree) / 2 = 24 edges
    total_edges = 24
    
    # Normalize active edges
    normalized_edges = [edge_count / total_edges for edge_count in active_edges]
    
    rounds_list = list(range(1, len(active_edges) + 1))
    
    plt.figure(figsize=(12, 6))
    
    # Plot the normalized active edges
    plt.plot(rounds_list, normalized_edges, marker='o', linestyle='-', color='blue', linewidth=2)

    
    plt.xlabel("Rounds (Time Steps)")
    plt.ylabel("Normalized Active Edges (active edges/ total edges)")
    plt.title("BCCLP 16K3 Broadcast: Normalized Active Edges per Round")
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    plt.ylim(0, max(max(normalized_edges) * 1.1, 0.55))  # Ensure reference lines are visible
    
    plt.legend()
    plt.show()

def print_state(state: Dict[int, Set[int]], N: int) -> None:
    """
    Prints a visual representation of the final state.
    """
    print("\nFinal Information Distribution:")
    print("-" * 50)
    for node in sorted(state.keys()):
        chunks = len(state[node])
        percentage = chunks / N * 100
        bar_length = int(percentage / 5)  # Scale to make it fit nicely
        bar = "█" * bar_length
        print(f"Node {node:2d}: {chunks}/{N} chunks ({percentage:.1f}%) {bar}")
    print("-" * 50)

def print_transmission_summary(transmissions_history: List[List[Tuple[int, int, int]]]) -> None:
    """
    Prints a summary of all transmissions that occurred during the broadcast.
    Includes information about which cycle is being used for each round.
    """
    print("\nTransmission Summary:")
    print("-" * 50)
    
    # Get the cycles for reference
    cycles = define_cycles()
    cycle_names = ["Cycle 1", "Cycle 2", "Cycle 3", "Cycle 4", "Cycle 5", "Cycle 6"]
    
    # Track which nodes are sending in each round to show the pipeline effect
    for round_idx, round_transmissions in enumerate(transmissions_history, 1):
        if not round_transmissions:
            continue
            
        senders = sorted(set(t[0] for t in round_transmissions))
        receivers = sorted(set(t[1] for t in round_transmissions))
        
        # Calculate which cycle was used for this round
        cycle_idx = (round_idx - 1) % len(cycles)
        current_cycle = cycles[cycle_idx]
        current_cycle_name = cycle_names[cycle_idx]
        
        # Collect the actual edges used this round
        edges_used = [(s, r) for s, r, _ in round_transmissions]
        edges_str = ', '.join([f"({s},{r})" for s, r in edges_used])
        
        # Collect the potential edges from the cycle
        cycle_edges_str = ', '.join([f"({s},{r})" for s, r in current_cycle])
        
        print(f"Round {round_idx}:")
        print(f"  Using {current_cycle_name}: [{cycle_edges_str}]")
        print(f"  Active edges: {len(round_transmissions)} - [{edges_str}]")
        print(f"  Sending nodes: {senders}")
        print(f"  Receiving nodes: {receivers}")
        
        if round_idx <= 5 or round_idx >= len(transmissions_history) - 5:
            # Print detailed info for the first 5 and last 5 rounds
            for sender, receiver, chunk in sorted(round_transmissions):
                print(f"  Node {sender} → Node {receiver}: chunk {chunk}")
        
    print("-" * 50)

def save_to_csv(active_edges: List[int], N: int, algorithm_name: str = "BCCLP"):
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
    Main function to run the 16K3 BCCLP broadcast simulation.
    """
    global INFO_SIZE
    
    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: python3 BCCLP.py <N>")
        print("Example: python3 BCCLP.py 100")
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
    
    print(f"16K3 BCCLP Broadcast Simulation (3-Cycle Approach)")
    print(f"Information size: {N} chunks")
    print(f"Starting node: {start_node}")
    print(f"Implementation: Using 3 distinct cycles that cover the 16K3 topology")
    print(f"Design goal: Ramp up to and maintain floor(V/2) = 8 active edges, then ramp down")
    
    # Run the simulation
    rounds, active_edges, state_history, transmissions_history = simulate_bcclp(start_node, N, verbose=False)
    
    # Print simulation results
    print(f"\nTotal number of rounds required: {rounds}")
    print(f"Total transmissions: {sum(active_edges)}")
    
    # Count rounds with maximum (8) active edges
    plateau_rounds = sum(1 for edges in active_edges if edges == 8)
    print(f"Rounds at maximum capacity (8 active edges): {plateau_rounds}")
    
    # Print final state
    print_state(state_history[-1], N)
    
    # Save data to CSV
    save_to_csv(active_edges, N, "BCCLP")
    
    # Disabled plotting
    # plot_active_edges(active_edges)

if __name__ == "__main__":
    main()
