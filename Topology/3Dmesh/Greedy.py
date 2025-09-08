#!/usr/bin/env python3
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Set
import random
import csv
import os
import sys

# Hard-coded parameters for the 3D mesh
GRID_X = 8       # n_x (size in the x-dimension)
GRID_Y = 6       # n_y (size in the y-dimension)
GRID_Z = 4       # n_z (size in the z-dimension)
INFO_SIZE = 100    # N (number of chunks)
START_POINT = (0, 0, 0)  # Source at one corner

def get_nodes_and_mapping(nx: int, ny: int, nz: int) -> Tuple[List[Tuple[int,int,int]], Dict[Tuple[int,int,int], int]]:
    """
    Enumerate all (x,y,z) in the 3D grid and map each coordinate to a unique index.
    """
    nodes: List[Tuple[int,int,int]] = []
    mapping: Dict[Tuple[int,int,int], int] = {}
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                coord = (x, y, z)
                mapping[coord] = len(nodes)
                nodes.append(coord)
    return nodes, mapping

def get_neighbors(coord: Tuple[int,int,int], nx: int, ny: int, nz: int) -> List[Tuple[int,int,int]]:
    """
    6-connected neighbors in the order:
      +x, +y, +z, −x, −y, −z
    """
    x,y,z = coord
    deltas = [(1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)]
    nbrs: List[Tuple[int,int,int]] = []
    for dx,dy,dz in deltas:
        nx_, ny_, nz_ = x+dx, y+dy, z+dz
        if 0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz:
            nbrs.append((nx_, ny_, nz_))
    return nbrs

def build_adjacency_matrix(nodes: List[Tuple[int,int,int]], mapping: Dict[Tuple[int,int,int], int],
                           nx: int, ny: int, nz: int) -> List[List[int]]:
    """
    Build an N_nodes × N_nodes adjacency matrix for the 3D grid.
    """
    N_nodes = len(nodes)
    mat = [[0]*N_nodes for _ in range(N_nodes)]
    for coord in nodes:
        i = mapping[coord]
        for nb in get_neighbors(coord, nx, ny, nz):
            j = mapping[nb]
            mat[i][j] = 1
    return mat

def build_adjacency_list(matrix: List[List[int]]) -> Dict[int, List[int]]:
    """
    Convert adjacency matrix to adjacency list (preserving neighbor order).
    """
    adj: Dict[int, List[int]] = {}
    N = len(matrix)
    for i in range(N):
        adj[i] = [j for j,val in enumerate(matrix[i]) if val]
    return adj

def simulate_bfs_3d(nx: int, ny: int, nz: int,
                    start: Tuple[int,int,int], N: int
                   ) -> Tuple[int, List[int], List[Dict[int, Set[int]]]]:
    """
    Simulate BFS broadcast on the 3D mesh with the standard constraints:
      - Each node can do at most one send OR one receive per round.
      - Greedily maximize number of active edges each round.
    Returns (rounds, active_edges_per_round, state_history).
    """
    nodes, mapping = get_nodes_and_mapping(nx, ny, nz)
    mat = build_adjacency_matrix(nodes, mapping, nx, ny, nz)
    adj = build_adjacency_list(mat)
    total = len(nodes)

    # Initialize each node's received‐chunks set
    state: Dict[int, Set[int]] = {i:set() for i in range(total)}
    state[mapping[start]] = set(range(N))  # Source has all chunks

    rounds = 0
    active_edges: List[int] = []
    history: List[Dict[int, Set[int]]] = [ {i:set(c) for i,c in state.items()} ]

    while any(len(chunks)<N for chunks in state.values()):
        # Prepare for this round
        new_state = {i:set(c) for i,c in state.items()}
        sending: Set[int] = set()
        receiving: Set[int] = set()
        # Collect all possible (sender,receiver,chunk) triples
        poss: List[Tuple[int,int,int]] = []
        for i in range(total):
            if not state[i]:
                continue
            for j in adj[i]:
                missing = state[i] - state[j]
                if missing:
                    poss.append((i, j, min(missing)))
        random.shuffle(poss)  # break index bias

        sent_count = 0
        for i,j,chunk in poss:
            if i in sending or i in receiving or j in sending or j in receiving:
                continue
            new_state[j].add(chunk)
            sending.add(i)
            receiving.add(j)
            sent_count += 1

        # Stop if no progress
        if sent_count==0:
            break

        # Commit this round
        state = new_state
        active_edges.append(sent_count)
        history.append({i:set(c) for i,c in state.items()})
        rounds += 1

    return rounds, active_edges, history

def print_state_grid_3d(state: Dict[int, Set[int]],
                        nx: int, ny: int, nz: int,
                        nodes: List[Tuple[int,int,int]],
                        mapping: Dict[Tuple[int,int,int], int]
                       ) -> None:
    """
    Print each z-slice as a 2D grid of "#chunks held" per node.
    """
    # Disabled printing
    pass

def get_total_edges_3d_mesh(nx: int, ny: int, nz: int) -> int:
    """
    Calculate the total number of edges in a 3D mesh of size nx×ny×nz.
    For a 3D mesh, each interior node has 6 edges, surface nodes have 5 edges,
    edge nodes have 4 edges, and corner nodes have 3 edges.
    """
    # Interior nodes: (nx-2)*(ny-2)*(nz-2) nodes with 6 edges each
    interior_edges = (nx-2)*(ny-2)*(nz-2) * 6
    
    # Surface nodes (excluding edges and corners): 6 faces with (nx-2)*(ny-2) nodes each
    # Each surface node has 5 edges
    surface_edges = 6 * ((nx-2)*(ny-2) + (nx-2)*(nz-2) + (ny-2)*(nz-2)) * 5
    
    # Edge nodes (excluding corners): 12 edges with (nx-2) nodes each
    # Each edge node has 4 edges
    edge_edges = 12 * (nx-2 + ny-2 + nz-2) * 4
    
    # Corner nodes: 8 corners with 3 edges each
    corner_edges = 8 * 3
    
    # Total edges (divide by 2 since each edge is counted twice)
    total_edges = (interior_edges + surface_edges + edge_edges + corner_edges) // 2
    
    return total_edges

def save_to_csv(active_edges: List[int], nx: int, ny: int, nz: int, N: int, algorithm_name: str = "Greedy3D"):
    """Save timestep and active edge data to CSV file"""
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Create filename with format: ALGO_p_q_r_N.csv
    filename = f"{data_dir}/{algorithm_name}_{nx}_{ny}_{nz}_{N}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['timestep', 'active_edges'])
        # Write data
        for timestep, edges in enumerate(active_edges, 1):
            writer.writerow([timestep, edges])
    
    # print(f"Data saved to {filename}")

def plot_transmissions(active_edges: List[int], nx: int, ny: int, nz: int) -> None:
    """
    Plot the ratio of active edges to total edges in each round.
    Uses the same blue color scheme as BCCLP.py.
    """
    # Disabled plotting
    pass

def parse_command_line_args():
    """Parse command line arguments for p, q, r, N"""
    if len(sys.argv) != 5:
        print("Usage: python3 Greedy.py <p> <q> <r> <N>")
        print("Example: python3 Greedy.py 8 8 8 100")
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
        print("Usage: python3 Greedy.py <p> <q> <r> <N>")
        sys.exit(1)

def main():
    # Parse command line arguments
    p, q, r, N = parse_command_line_args()
    start = (0, 0, 0)
    
    # Simulate the broadcast
    rounds, edges_per_round, history = simulate_bfs_3d(
        p, q, r, start, N
    )
    
    # Save to CSV
    save_to_csv(edges_per_round, p, q, r, N, "Greedy3D")

if __name__ == "__main__":
    main()
