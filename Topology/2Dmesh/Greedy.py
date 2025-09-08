import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Set
import random
import csv
import os
import sys

# Default parameters (will be overridden by command line arguments)
GRID_ROWS = 32     # n (number of rows)
GRID_COLS = 32     # m (number of columns)
INFO_SIZE = 100    # N (size of information)
START_POINT = (0, 0)  # Top-left corner

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
    The order is right, down, left, up (i.e. (0,1), (1,0), (0,-1), (-1,0)).
    """
    neighbors = []
    # Desired order: right, down, left, up.
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for d in directions:
        neighbor = (coord[0] + d[0], coord[1] + d[1])
        if 0 <= neighbor[0] < n and 0 <= neighbor[1] < m:
            neighbors.append(neighbor)
    return neighbors

def build_adjacency_matrix(n: int, m: int, nodes: List[Tuple[int, int]], mapping: Dict[Tuple[int, int], int]) -> List[List[int]]:
    """
    Constructs an explicit adjacency matrix (a list of lists) for the grid.
    If two nodes are neighbors, the corresponding matrix entry is 1.
    """
    N_nodes = len(nodes)
    matrix = [[0] * N_nodes for _ in range(N_nodes)]
    for coord in nodes:
        i = mapping[coord]
        for nb in get_neighbors(coord, n, m):
            j = mapping[nb]
            matrix[i][j] = 1
    return matrix

def build_adjacency_list(matrix: List[List[int]]) -> Dict[int, List[int]]:
    """
    Converts an adjacency matrix into an adjacency list.
    The neighbor order is determined by the order of iteration over columns.
    (Since our matrix was built using get_neighbors, the order is as desired.)
    """
    adj_list = {}
    N = len(matrix)
    for i in range(N):
        neighbors = []
        for j, val in enumerate(matrix[i]):
            if val == 1:
                neighbors.append(j)
        adj_list[i] = neighbors
    return adj_list

def simulate_bfs(n: int, m: int, start: Tuple[int, int], N: int) -> Tuple[int, List[int], List[Dict[int, Set[int]]]]:
    """
    Simulates the broadcast of a message of size N on an n x m 2D grid using BFS,
    with the constraint that nodes cannot send and receive simultaneously,
    and each node can at most send OR receive 1 unit of information per timestep.
    
    Maximizes the number of active edges per timestep by trying all possible
    valid transmissions in a greedy manner.
    
    Returns:
        rounds: Total number of rounds until every node has all N chunks.
        active_edges: List where each element is the number of transmissions (active edges) in that round.
        state_history: List of node states at each round for visualization.
    """
    # Build nodes, mapping, and the explicit adjacency matrix & list.
    nodes, mapping = get_nodes_and_mapping(n, m)
    matrix = build_adjacency_matrix(n, m, nodes, mapping)
    adj_list = build_adjacency_list(matrix)
    
    total_nodes = len(nodes)
    # Initialize state: for each node index, record the set of chunks it has.
    state: Dict[int, Set[int]] = {i: set() for i in range(total_nodes)}
    state[mapping[start]] = set(range(N))  # Source node gets the full message.
    
    rounds = 0
    active_edges: List[int] = []
    state_history: List[Dict[int, Set[int]]] = [state.copy()]
    
    # Continue until every node is fully informed.
    while any(len(chunks) < N for chunks in state.values()):
        transmissions_this_round = 0
        new_state: Dict[int, Set[int]] = {i: state[i].copy() for i in state}
        
        # Track nodes that send or receive in this round
        sending_nodes: Set[int] = set()
        receiving_nodes: Set[int] = set()
        
        # Create all possible valid transmissions for this round
        possible_transmissions = []
        for i in range(total_nodes):
            if not state[i]:  # Skip nodes without any information
                continue
            
            for j in adj_list[i]:
                # Determine if sender has any chunk the receiver lacks
                missing = state[i] - state[j]
                if missing:
                    # Record this as a possible transmission
                    possible_transmissions.append((i, j, min(missing)))
        
        # Randomize order to avoid bias from node numbering
        random.shuffle(possible_transmissions)
        
        # Greedily select transmissions to maximize active edges
        for sender, receiver, chunk in possible_transmissions:
            # Skip if either sender or receiver is already busy this round
            if sender in sending_nodes or sender in receiving_nodes or \
               receiver in sending_nodes or receiver in receiving_nodes:
                continue
            
            # Execute the transmission
            new_state[receiver].add(chunk)
            sending_nodes.add(sender)
            receiving_nodes.add(receiver)
            transmissions_this_round += 1
        
        active_edges.append(transmissions_this_round)
        state = new_state
        state_history.append(state.copy())
        rounds += 1
        
        # Exit if no transmissions occurred this round (to prevent infinite loop)
        if transmissions_this_round == 0:
            break
    
    return rounds, active_edges, state_history

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

def save_to_csv(active_edges: List[int], n: int, m: int, N: int, algorithm_name: str = "Greedy"):
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
    Plots the number of active edges (transmissions) per round using matplotlib.
    The x-axis is the round number; the y-axis is the number of active edges in that round.
    """
    # Disabled plotting
    pass

def main() -> None:
    global GRID_ROWS, GRID_COLS, INFO_SIZE
    
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python3 Greedy.py <rows> <cols> <N>")
        print("Example: python3 Greedy.py 16 16 100")
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
    n = GRID_ROWS
    m = GRID_COLS
    start_coords = START_POINT
    N = INFO_SIZE
    
    print(f"2D Grid size: {n}×{m}")
    print(f"Information size: {N}")
    print(f"Start point: {start_coords}")

    # Set random seed for reproducibility
    random.seed(42)

    # Run the BFS broadcast simulation
    print("\nRunning greedy BFS broadcast...")
    rounds, active_edges, state_history = simulate_bfs(n, m, start_coords, N)

    print(f"\nBroadcast completed in {rounds} rounds.")
    print(f"Active edges per round: {active_edges[:10]}..." if len(active_edges) > 10 else f"Active edges per round: {active_edges}")

    # Print final state
    print("\nFinal chunks per node:")
    nodes, mapping = get_nodes_and_mapping(n, m)
    print_state_grid(state_history[-1], n, m, N, nodes, mapping)

    # Save data to CSV
    save_to_csv(active_edges, n, m, N, "Greedy")

    # Disabled plotting calls
    # plot_transmissions(active_edges)

if __name__ == "__main__":
    main()
