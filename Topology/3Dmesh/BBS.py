#!/usr/bin/env python3
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import copy
import logging
from fractions import Fraction
from math import gcd
from functools import reduce
from typing import Dict, List, Tuple, Any, Optional, Set
from networkx.algorithms.matching import max_weight_matching
import sys
import csv
import os

# Disable logging to reduce output
logging.basicConfig(
    level=logging.ERROR,  # Changed from INFO to ERROR to suppress most output
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Optional per-step logger (file-based). Enable with env var BBS_LOG_STEPS=1
step_logger: Optional[logging.Logger] = None

def init_step_logger(p: int, q: int, r: int, N: int) -> None:
    """Initialize a file logger that records per-step state and sends.
    Enabled when env var BBS_LOG_STEPS=1. Logs to 3D/logs/BBS_p_q_r_N.log.
    """
    global step_logger
    if os.environ.get("BBS_LOG_STEPS", "0") != "1":
        step_logger = None
        return
    # Create/replace a dedicated logger to avoid clashing with root logger
    step_logger = logging.getLogger("BBS_STEP")
    step_logger.setLevel(logging.INFO)
    # Clear existing handlers to prevent duplicates across runs
    step_logger.handlers = []
    # Ensure logs directory exists (relative to this file)
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logfile = os.path.join(logs_dir, f"BBS_{p}_{q}_{r}_{N}.log")
    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(message)s'))
    step_logger.addHandler(fh)
    step_logger.info("=== BBS Step Log ===")
    step_logger.info(f"Grid: {p}x{q}x{r}, Chunks per node: N={N}")
    step_logger.info("")

# Default parameters (will be overridden by command line args)
p = 8   # number of rows
q = 8   # number of columns
r = 0   # number of layers (0 for 2D, >0 for 3D)
N = 100  # size of data (chunks to send)
K = 1.0  # limitation matrix scaling factor

def parse_command_line_args():
    """Parse command line arguments for p, q, r, N"""
    if len(sys.argv) != 5:
        print("Usage: python3 BCCLP.py <p> <q> <r> <N>")
        print("Example: python3 BCCLP.py 8 8 8 100")
        sys.exit(1)
    
    try:
        p_val = int(sys.argv[1])
        q_val = int(sys.argv[2])
        r_val = int(sys.argv[3])
        N_val = int(sys.argv[4])
        
        if p_val <= 0 or q_val <= 0 or r_val < 0 or N_val <= 0:
            raise ValueError("All parameters must be positive integers (r can be 0 for 2D)")
        
        return p_val, q_val, r_val, N_val
    except ValueError as e:
        print(f"Error: Invalid arguments. {e}")
        print("Usage: python3 BCCLP.py <p> <q> <r> <N>")
        sys.exit(1)

def is_3d_mesh(adj_list: Dict[Any, List[Any]]) -> Optional[Tuple[int, int, int]]:
    """
    Check if the given adjacency list represents a 3D mesh.
    Returns (rows, cols, depth) if it is a 3D mesh, None otherwise.
    """
    # Check if all nodes are tuples of three integers
    if not all(isinstance(node, tuple) and len(node) == 3 and 
              all(isinstance(x, int) for x in node) for node in adj_list.keys()):
        return None
    
    # Get dimensions
    rows = max(r for r, _, _ in adj_list.keys()) + 1
    cols = max(c for _, c, _ in adj_list.keys()) + 1
    depth = max(d for _, _, d in adj_list.keys()) + 1
    
    # Verify grid structure
    for (r, c, d), neighbors in adj_list.items():
        expected_neighbors = []
        if r > 0: expected_neighbors.append((r-1, c, d))
        if r < rows-1: expected_neighbors.append((r+1, c, d))
        if c > 0: expected_neighbors.append((r, c-1, d))
        if c < cols-1: expected_neighbors.append((r, c+1, d))
        if d > 0: expected_neighbors.append((r, c, d-1))
        if d < depth-1: expected_neighbors.append((r, c, d+1))
        
        # Check if all expected neighbors are present and no extra neighbors
        if set(neighbors) != set(expected_neighbors):
            return None
    
    return rows, cols, depth

def is_boundary(x: int, y: int, z: int, rows: int, cols: int, depth: int) -> bool:
    """Check if a node is on the boundary of the mesh."""
    return x in (0, rows-1) or y in (0, cols-1) or z in (0, depth-1)

def get_neighbors(x: int, y: int, z: int, rows: int, cols: int, depth: int) -> List[Tuple[int, int, int]]:
    """Get valid mesh neighbors for a node."""
    neighbors = []
    for dx, dy, dz in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)):
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < rows and 0 <= ny < cols and 0 <= nz < depth:
            neighbors.append((nx, ny, nz))
    return neighbors

def is_in_line_with_root(x: int, y: int, z: int) -> bool:
    """Check if a node is in line with root (0,0,0)"""
    is_in_line = (x == 0 and y == 0) or (x == 0 and z == 0) or (y == 0 and z == 0)
    print(f"DEBUG is_in_line_with_root({x},{y},{z}): {is_in_line}")
    return is_in_line

def is_straight_line_to_root(x: int, y: int, z: int) -> bool:
    """Check if a node is on a straight line path to root (0,0,0)."""
    return (x == 0 and y == 0) or (x == 0 and z == 0) or (y == 0 and z == 0)

def get_node_type(x: int, y: int, z: int, rows: int, cols: int, depth: int) -> str:
    """Determine the type of node: 'corner', 'edge', 'surface', or 'interior'."""
    is_corner = (x in (0, rows-1) and y in (0, cols-1) and z in (0, depth-1))
    is_edge = (x in (0, rows-1) and y in (0, cols-1)) or \
              (x in (0, rows-1) and z in (0, depth-1)) or \
              (y in (0, cols-1) and z in (0, depth-1))
    is_surface = (x in (0, rows-1)) or (y in (0, cols-1)) or (z in (0, depth-1))
    
    if is_corner:
        return 'corner'
    elif is_edge:
        return 'edge'
    elif is_surface:
        return 'surface'
    else:
        return 'interior'

def is_in_line_with_pqr(x: int, y: int, z: int, p: int, q: int, r: int) -> bool:
    """Check if a node is in line with (P,Q,R)"""
    return (x == p-1 and y == q-1) or (x == p-1 and z == r-1) or (y == q-1 and z == r-1)

def get_neighbor_in_line_with_pqr(x: int, y: int, z: int, p: int, q: int, r: int) -> Optional[Tuple[int, int, int]]:
    """Get the neighbor that is in line with (P,Q,R)"""
    for nx, ny, nz in get_neighbors(x, y, z, p, q, r):
        if is_in_line_with_pqr(nx, ny, nz, p, q, r):
            return (nx, ny, nz)
    return None

def get_neighbor_in_line_with_root(x: int, y: int, z: int, rows: int, cols: int, depth: int) -> Optional[Tuple[int, int, int]]:
    """Get the neighbor that is in line with root (0,0,0)"""
    for nx, ny, nz in get_neighbors(x, y, z, rows, cols, depth):
        if is_in_line_with_root(nx, ny, nz):
            return (nx, ny, nz)
    return None

def get_neighbors_not_in_line(x: int, y: int, z: int, rows: int, cols: int, depth: int) -> list:
    """Get neighbors that are not in line with root"""
    return [(nx, ny, nz) for nx, ny, nz in get_neighbors(x, y, z, rows, cols, depth) 
            if not is_in_line_with_root(nx, ny, nz)]

def is_direction_towards_pqr(x: int, y: int, z: int, nx: int, ny: int, nz: int, p: int, q: int, r: int) -> bool:
    """Check if the edge (x,y,z) -> (nx,ny,nz) is in the direction of (P,Q,R)"""
    # For edges in line with (P,Q,R), we just need to check if we're moving towards the larger coordinates
    return nx > x or ny > y or nz > z

def is_root_prime(x: int, y: int, z: int, rows: int, cols: int, depth: int) -> bool:
    """Check if a node is the Root' corner (the one that takes 8 and gives 6)"""
    # Root' is the corner at (rows-1, cols-1, depth-1)
    is_rp = x == rows-1 and y == cols-1 and z == depth-1
    print(f"DEBUG is_root_prime({x},{y},{z}): {is_rp}")
    return is_rp

def is_in_line_with_root_prime(x: int, y: int, z: int, nx: int, ny: int, nz: int, rows: int, cols: int, depth: int) -> bool:
    """Check if an edge is in line with Root' direction"""
    # Root' is at (rows-1, cols-1, 0), (rows-1, 0, depth-1), or (0, cols-1, depth-1)
    # Check if the edge is moving towards one of these points
    if (rows-1, cols-1, 0) in [(x,y,z), (nx,ny,nz)]:
        return (x,y,z) == (rows-2, cols-1, 0) or (x,y,z) == (rows-1, cols-2, 0) or \
               (nx,ny,nz) == (rows-2, cols-1, 0) or (nx,ny,nz) == (rows-1, cols-2, 0)
    elif (rows-1, 0, depth-1) in [(x,y,z), (nx,ny,nz)]:
        return (x,y,z) == (rows-2, 0, depth-1) or (x,y,z) == (rows-1, 0, depth-2) or \
               (nx,ny,nz) == (rows-2, 0, depth-1) or (nx,ny,nz) == (rows-1, 0, depth-2)
    elif (0, cols-1, depth-1) in [(x,y,z), (nx,ny,nz)]:
        return (x,y,z) == (0, cols-2, depth-1) or (x,y,z) == (0, cols-1, depth-2) or \
               (nx,ny,nz) == (0, cols-2, depth-1) or (nx,ny,nz) == (0, cols-1, depth-2)
    return False

def is_in_line_with_root_b(x: int, y: int, z: int, nx: int, ny: int, nz: int) -> bool:
    """Check if an edge is in line with 1.b direction (straight line to root)"""
    # 1.b direction is towards (0,0,0)
    return (x == 0 and y == 0 and nx == 0 and ny == 0) or \
           (x == 0 and z == 0 and nx == 0 and nz == 0) or \
           (y == 0 and z == 0 and ny == 0 and nz == 0)

def get_corner_coordinates(rows: int, cols: int, depth: int) -> List[Tuple[int, int, int]]:
    """Get list of all corner coordinates in the mesh."""
    return [
        (0, 0, 0),  # Root
        (0, 0, depth-1), (0, cols-1, 0), (0, cols-1, depth-1),
        (rows-1, 0, 0), (rows-1, 0, depth-1), (rows-1, cols-1, 0), (rows-1, cols-1, depth-1)
    ]

def is_edge_type_1(x: int, y: int, z: int, rows: int, cols: int, depth: int) -> bool:
    """Check if an edge node is type 1 (connects corners 1.d and 1.c)"""
    neighbors = get_neighbors(x, y, z, rows, cols, depth)
    has_root_prime = False
    has_other_corner = False
    corner_coords = get_corner_coordinates(rows, cols, depth)
    
    for nx, ny, nz in neighbors:
        if nx == rows-1 and ny == cols-1 and nz == depth-1:  # Root' corner
            has_root_prime = True
        elif (nx, ny, nz) in corner_coords and not (nx == 0 and ny == 0 and nz == 0):  # Other corner (not root)
            has_other_corner = True
    
    return has_root_prime and has_other_corner

def is_facing_root_prime(x: int, y: int, z: int, nx: int, ny: int, nz: int, rows: int, cols: int, depth: int) -> bool:
    """Check if edge (x,y,z) -> (nx,ny,nz) is facing Root' (moving towards Root')"""
    # Root' is at (rows-1, cols-1, depth-1)
    # Check if we're moving towards larger coordinates
    return (nx > x and ny > y) or (nx > x and nz > z) or (ny > y and nz > z)

def is_opposing_root_prime(x: int, y: int, z: int, nx: int, ny: int, nz: int, rows: int, cols: int, depth: int) -> bool:
    """Check if edge (x,y,z) -> (nx,ny,nz) is opposing Root' (moving away from Root')"""
    # Root' is at (rows-1, cols-1, depth-1)
    # Check if we're moving towards smaller coordinates
    return (nx < x and ny < y) or (nx < x and nz < z) or (ny < y and nz < z)

def is_edge_type_2(x: int, y: int, z: int, rows: int, cols: int, depth: int) -> bool:
    """Check if an edge node is type 2 (connects corners 1.d and 1.b)"""
    neighbors = get_neighbors(x, y, z, rows, cols, depth)
    has_root_line = False
    has_other_corner = False
    corner_coords = get_corner_coordinates(rows, cols, depth)
    
    for nx, ny, nz in neighbors:
        if is_in_line_with_root(nx, ny, nz):
            has_root_line = True
        elif (nx, ny, nz) in corner_coords and not is_in_line_with_root(nx, ny, nz):
            has_other_corner = True
    
    return has_root_line and has_other_corner

def compute_3d_mesh_limitation_matrix(rows: int, cols: int, depth: int) -> np.ndarray:
    """
    Compute limitation matrix for a 3D mesh (size of rows, cols, depth) with specific weight patterns.

    Updated Weight Distribution Rules:
    1. Corner Nodes (8 corners of the cube):
       a) Root corner (0,0,0):
          - Gives 14 to all direct neighbors (3 neighbors)
          - Has no incoming edges
       b) Corners with straight line to root (e.g., (rows-1,0,0), (0,cols-1,0), (0,0,depth-1)):
          - Takes 14 from the neighbor one step toward (0,0,0)
          - Takes 5 from the other two neighbors
          - Gives 9 to those same two neighbors
          - No outgoing to the neighbor toward root
       c) "Root'" corner (rows-1, cols-1, depth-1):
          - Takes 8 from each of its 3 neighbors
          - Gives 6 to each of those 3 neighbors
       d) The remaining 3 corners (each has one coordinate = 0 and two = max):
          - Identify the neighbor along the axis pointing toward root' → take 6 from it, give 8 to it
          - Identify the two neighbors along the axes pointing toward a "1b" corner → take 9 from each of those two, give 5 to each of those two

    2. Edge Nodes (12 edges, excluding corners):
       a) Edges on an axis from root (e.g., (i,0,0), (0,j,0), (0,0,k), with 1 ≤ i ≤ rows-2, etc.):
          - Let "prev" be the neighbor one step closer to (0,0,0) → take 14 from it
          - Let "next" be the neighbor one step farther along that same axis → give 14 to it
          - For each of the two "surface" neighbors (off that axis), take 5 from it and give 5 to it
       b) Edges on an axis from a "1.d" corner to root' (e.g., (rows-1, cols-1, k), (rows-1, k, depth-1), (k, cols-1, depth-1)):
          - Among its two neighbors on that same axis, call the one closer to root' "n_rp" and the one closer to the 1.d corner "n_od"
            • From n_rp take 6 and give 8
            • From n_od take 8 and give 6
          - For each of its two "surface" neighbors (off that axis), take 5 and give 5
       c) Edges on an axis from a "1.b" corner to a "1.d" corner:
          - Identify the two axis-neighbors on that same axis:
               • The one with larger coordinate (closer to 1.d) → take 5 from it, give 9 to it
               • The one with smaller coordinate (closer to 1.b) → take 9 from it, give 5 to it
          - For each of its two "surface" neighbors, take 5 and give 5

    3. Surface Nodes (6 faces, excluding edges):
       - For each neighbor:
         • If that neighbor is strictly interior → take 4 from it and give 4 to it
         • Otherwise (corner or edge) → take 5 from it and give 5 to it

    4. Interior Nodes (all nodes not on a face):
       - For each neighbor → take 4 from it and give 4 to it

    In every case, "take" means:
        limitation_matrix[neighbor_index, current_index] = value
    and "give" means:
        limitation_matrix[current_index, neighbor_index] = value

    Finally, zero out the entire column for the root (0,0,0) so it has no incoming edges.
    """
    if rows < 2 or cols < 2 or depth < 2:
        raise ValueError("Each dimension must be at least 2")

    num_nodes = rows * cols * depth
    limitation_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    def get_node_idx(x: int, y: int, z: int) -> int:
        return z * (rows * cols) + y * rows + x

    # Helper: all six orthogonal neighbors (if inside bounds)
    def get_neighbors(x: int, y: int, z: int) -> Tuple[Tuple[int,int,int], ...]:
        nbrs = []
        if x - 1 >= 0:
            nbrs.append((x - 1, y, z))
        if x + 1 < rows:
            nbrs.append((x + 1, y, z))
        if y - 1 >= 0:
            nbrs.append((x, y - 1, z))
        if y + 1 < cols:
            nbrs.append((x, y + 1, z))
        if z - 1 >= 0:
            nbrs.append((x, y, z - 1))
        if z + 1 < depth:
            nbrs.append((x, y, z + 1))
        return tuple(nbrs)

    # Classify corners, edges, surface, interior by boundary-count
    def boundary_count(x: int, y: int, z: int) -> int:
        bc = 0
        if x == 0 or x == rows - 1:
            bc += 1
        if y == 0 or y == cols - 1:
            bc += 1
        if z == 0 or z == depth - 1:
            bc += 1
        return bc

    def is_corner(x: int, y: int, z: int) -> bool:
        return (boundary_count(x, y, z) == 3)

    def is_edge(x: int, y: int, z: int) -> bool:
        return (boundary_count(x, y, z) == 2)

    def is_surface(x: int, y: int, z: int) -> bool:
        return (boundary_count(x, y, z) == 1)

    # Identify special corners:
    root = (0, 0, 0)
    root_idx = get_node_idx(*root)

    corner_1b = {
        (rows - 1, 0, 0),
        (0, cols - 1, 0),
        (0, 0, depth - 1),
    }
    root_prime = (rows - 1, cols - 1, depth - 1)

    # "Other corners" = all corner positions minus {root} minus corner_1b minus {root_prime}
    all_corners = {
        (x, y, z)
        for x in (0, rows - 1)
        for y in (0, cols - 1)
        for z in (0, depth - 1)
    }
    other_corners = all_corners - {root} - corner_1b - {root_prime}

    # 1. Handle all corners first
    for (x, y, z) in all_corners:
        u = get_node_idx(x, y, z)
        if (x, y, z) == root:
            # 1a) Root (0,0,0): give 14 to each neighbor
            for (nx, ny, nz) in get_neighbors(x, y, z):
                limitation_matrix[u, get_node_idx(nx, ny, nz)] = 14
            continue

        if (x, y, z) in corner_1b:
            # 1b) A corner "in line to root": exactly one coordinate = max, the other two = 0
            if x == rows - 1 and y == 0 and z == 0:
                prev = (x - 1, y, z)
                not_in_line = [(x, y + 1, z), (x, y, z + 1)]
            elif x == 0 and y == cols - 1 and z == 0:
                prev = (x, y - 1, z)
                not_in_line = [(x + 1, y, z), (x, y, z + 1)]
            else:  # (x == 0, y == 0, z == depth - 1)
                prev = (x, y, z - 1)
                not_in_line = [(x + 1, y, z), (x, y + 1, z)]
            # Take 14 from "prev"
            limitation_matrix[get_node_idx(*prev), u] = 14
            # Take 5 from each of the other two, and give 9 to each of them
            for (nx, ny, nz) in not_in_line:
                limitation_matrix[get_node_idx(nx, ny, nz), u] = 5
                limitation_matrix[u, get_node_idx(nx, ny, nz)] = 9
            continue

        if (x, y, z) == root_prime:
            # 1c) "Root'" = (rows-1, cols-1, depth-1)
            for (nx, ny, nz) in get_neighbors(x, y, z):
                limitation_matrix[get_node_idx(nx, ny, nz), u] = 8
                limitation_matrix[u, get_node_idx(nx, ny, nz)] = 6
            continue

        # 1d) The "other" three corners
        if (x, y, z) in other_corners:
            if x == rows - 1 and y == cols - 1 and z == 0:
                rp_nb = (x, y, z + 1)
                b_nbs = [(x - 1, y, z), (x, y - 1, z)]
            elif x == rows - 1 and y == 0 and z == depth - 1:
                rp_nb = (x, y + 1, z)
                b_nbs = [(x - 1, y, z), (x, y, z - 1)]
            else:  # (x == 0, y == cols - 1, z == depth - 1)
                rp_nb = (x + 1, y, z)
                b_nbs = [(x, y - 1, z), (x, y, z - 1)]
            # 1d-a) Take 6 from rp_nb, give 8 to rp_nb
            limitation_matrix[get_node_idx(*rp_nb), u] = 6
            limitation_matrix[u, get_node_idx(*rp_nb)] = 8
            # 1d-b) For each "1b-direction" neighbor, take 9 and give 5
            for (nx, ny, nz) in b_nbs:
                limitation_matrix[get_node_idx(nx, ny, nz), u] = 9
                limitation_matrix[u, get_node_idx(nx, ny, nz)] = 5
            continue

    # 2. Handle all edges next (excluding corners and root)
    for z in range(depth):
        for y in range(cols):
            for x in range(rows):
                if is_corner(x, y, z) or (x, y, z) == root:
                    continue
                if not is_edge(x, y, z):
                    continue

                u = get_node_idx(x, y, z)
                x0, xM = (x == 0), (x == rows - 1)
                y0, yM = (y == 0), (y == cols - 1)
                z0, zM = (z == 0), (z == depth - 1)

                # 2a) Edge on an axis from root
                if (x0 and y0 and 0 < z < depth - 1) or \
                   (x0 and z0 and 0 < y < cols - 1) or \
                   (y0 and z0 and 0 < x < rows - 1):
                    if x0 and y0:
                        prev = (x, y, z - 1)
                        next_ = (x, y, z + 1)
                    elif x0 and z0:
                        prev = (x, y - 1, z)
                        next_ = (x, y + 1, z)
                    else:  # y0 and z0
                        prev = (x - 1, y, z)
                        next_ = (x + 1, y, z)
                    # Take 14 from prev
                    limitation_matrix[get_node_idx(*prev), u] = 14
                    # Give 14 to next
                    limitation_matrix[u, get_node_idx(*next_)] = 14
                    # Two surface neighbors
                    for (nx, ny, nz) in get_neighbors(x, y, z):
                        if is_surface(nx, ny, nz):
                            limitation_matrix[get_node_idx(nx, ny, nz), u] = 5
                            limitation_matrix[u, get_node_idx(nx, ny, nz)] = 5
                    continue

                # 2b) Edge on an axis from a "1.d" corner to root'
                if (xM and yM and 0 < z < depth - 1) or \
                   (xM and zM and 0 < y < cols - 1) or \
                   (yM and zM and 0 < x < rows - 1):
                    nbrs = get_neighbors(x, y, z)
                    n_rp = None
                    n_od = None
                    for (nx, ny, nz) in nbrs:
                        if xM and yM:
                            # z varies
                            if nz > z:
                                n_rp = (nx, ny, nz)
                            else:
                                n_od = (nx, ny, nz)
                        elif xM and zM:
                            # y varies
                            if ny > y:
                                n_rp = (nx, ny, nz)
                            else:
                                n_od = (nx, ny, nz)
                        else:  # yM and zM
                            # x varies
                            if nx > x:
                                n_rp = (nx, ny, nz)
                            else:
                                n_od = (nx, ny, nz)
                    # From n_rp take 6, give 8
                    if n_rp is not None:
                        limitation_matrix[get_node_idx(*n_rp), u] = 6
                        limitation_matrix[u, get_node_idx(*n_rp)] = 8
                    # From n_od take 8, give 6
                    if n_od is not None:
                        limitation_matrix[get_node_idx(*n_od), u] = 8
                        limitation_matrix[u, get_node_idx(*n_od)] = 6
                    # All surface neighbors
                    for (nx, ny, nz) in nbrs:
                        if is_surface(nx, ny, nz):
                            limitation_matrix[get_node_idx(nx, ny, nz), u] = 5
                            limitation_matrix[u, get_node_idx(nx, ny, nz)] = 5
                    continue

                # 2c) Edge on an axis from a "1.b" corner to a "1.d" corner
                # Determine the variable coordinate (the one not at 0 or max)
                coords = (x, y, z)
                max_vals = (rows - 1, cols - 1, depth - 1)
                # Find index i where coords[i] not in {0, max_i}
                var_i = next(i for i in range(3) if coords[i] not in {0, max_vals[i]})
                # Construct the two axis-neighbors by ±1 along dimension var_i
                nbrs = get_neighbors(x, y, z)
                neighbor_axis = []
                for (nx, ny, nz) in nbrs:
                    # check if exactly one coordinate differs from (x,y,z) by ±1, same for other two
                    diff_count = (1 if nx != x else 0) + (1 if ny != y else 0) + (1 if nz != z else 0)
                    if diff_count == 1:
                        if ((var_i == 0 and abs(nx - x) == 1 and ny == y and nz == z) or
                            (var_i == 1 and abs(ny - y) == 1 and nx == x and nz == z) or
                            (var_i == 2 and abs(nz - z) == 1 and nx == x and ny == y)):
                            neighbor_axis.append((nx, ny, nz))
                # Sort them so that one with larger var_i coordinate is "toward 1.d"
                # and one with smaller var_i coordinate is "away from 1.d"
                n_low, n_high = sorted(
                    neighbor_axis,
                    key=lambda pt: pt[var_i]
                )  # n_low has smaller coordinate, n_high larger
                # Take 9 from n_low, give 5 to n_low
                limitation_matrix[get_node_idx(*n_low), u] = 9
                limitation_matrix[u, get_node_idx(*n_low)] = 5
                # Take 5 from n_high, give 9 to n_high
                limitation_matrix[get_node_idx(*n_high), u] = 5
                limitation_matrix[u, get_node_idx(*n_high)] = 9
                # Finally, two surface neighbors (the other two in get_neighbors)
                for (nx, ny, nz) in nbrs:
                    if is_surface(nx, ny, nz):
                        limitation_matrix[get_node_idx(nx, ny, nz), u] = 5
                        limitation_matrix[u, get_node_idx(nx, ny, nz)] = 5
                continue

    # 3. Handle surface nodes (boundary_count == 1)
    for z in range(depth):
        for y in range(cols):
            for x in range(rows):
                if is_corner(x, y, z) or is_edge(x, y, z):
                    continue
                if is_surface(x, y, z):
                    u = get_node_idx(x, y, z)
                    for (nx, ny, nz) in get_neighbors(x, y, z):
                        v = get_node_idx(nx, ny, nz)
                        if not (nx in (0, rows - 1) or
                                ny in (0, cols - 1) or
                                nz in (0, depth - 1)):
                            # Neighbor is interior
                            limitation_matrix[v, u] = 4
                            limitation_matrix[u, v] = 4
                        else:
                            # Neighbor is corner or edge
                            limitation_matrix[v, u] = 5
                            limitation_matrix[u, v] = 5

    # 4. Handle interior nodes (boundary_count == 0)
    for z in range(depth):
        for y in range(cols):
            for x in range(rows):
                if is_corner(x, y, z) or is_edge(x, y, z) or is_surface(x, y, z):
                    continue
                # Now boundary_count == 0 → interior
                u = get_node_idx(x, y, z)
                for (nx, ny, nz) in get_neighbors(x, y, z):
                    limitation_matrix[get_node_idx(nx, ny, nz), u] = 4
                    limitation_matrix[u, get_node_idx(nx, ny, nz)] = 4

    # Finally, zero out all incoming into root
    limitation_matrix[:, root_idx] = 0
    return limitation_matrix

def get_node_idx(r: int, c: int, d: int, rows: int, cols: int, depth: int) -> int:
    """Convert 3D coordinates to 1D index using row-major ordering within each z-level."""
    return d * (rows * cols) + c * rows + r

def get_coords(idx: int, rows: int, cols: int, depth: int) -> Tuple[int, int, int]:
    """Convert 1D index to 3D coordinates using row-major ordering within each z-level."""
    d, rem = divmod(idx, rows * cols)
    c, r = divmod(rem, rows)
    return r, c, d

def ComputeLimitationMatrix(
    adj_list: Dict[Any, List[Any]],
    root: Any,
    K: float = 1.0
) -> Tuple[np.ndarray, Dict[Any,int], int]:
    """
    Build and integer-scale the limitation matrix for G → G'.
    For 2D/3D meshes, uses specialized computation rules.
    For other graphs, uses the general K/deg(v) rule.
    
    Returns:
      L_int      – the integer limitation matrix (dtype=int),
      index_map  – mapping node → matrix index,
      scale      – the integer scale factor used (1 for mesh).
    """
    # Check if this is a 2×2×4 mesh first (special case)
    mesh_dims_3d = is_3d_mesh(adj_list)
    if mesh_dims_3d is not None:
        rows, cols, depth = mesh_dims_3d
        # Check if this is a 2×2×4 mesh
        if rows == 2 and cols == 2 and depth == 4:
            # Use specialized 2×2×4 computation
            L_int = compute_2x2x4_mesh_limitation_matrix()
            idx = {v: i for i, v in enumerate(sorted(adj_list.keys()))}
            return L_int, idx, 1  # scale is 1 for 2×2×4 mesh
        # Check if this is a 2×2×r mesh (other than 4)
        elif rows == 2 and cols == 2 and depth >= 2:
            # Use specialized 2×2×r computation
            L_int = compute_2x2xr_mesh_limitation_matrix(depth)
            idx = {v: i for i, v in enumerate(sorted(adj_list.keys()))}
            return L_int, idx, 1  # scale is 1 for 2×2×r mesh
        else:
            # For other 3D meshes, use the general 3D computation
            L_int = compute_3d_mesh_limitation_matrix(rows, cols, depth)
            idx = {v: i for i, v in enumerate(sorted(adj_list.keys()))}
            return L_int, idx, 1  # scale is 1 for 3D mesh
    
    # For non-mesh graphs, use the general computation
    nodes = list(adj_list.keys())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)

    for v, neighs in adj_list.items():
        if v == root:
            continue
        deg = len(neighs)
        if deg == 0:
            continue
        w = K / deg
        j = idx[v]
        for u in neighs:
            i = idx[u]
            L[i, j] = w

    # Compute LCM of denominators
    fracs = [Fraction(val).limit_denominator() for val in L.flatten() if val > 0]
    dens = [f.denominator for f in fracs]
    scale = reduce(lambda a, b: a * b // gcd(a, b), dens, 1) if dens else 1

    # Scale to integer
    L_int = (L * scale).round().astype(int)

    return L_int, idx, scale

def BuildGraph(
    adj_list: Dict[Any, List[Any]],
    root: Any,
    K: float = 1.0
) -> Tuple[nx.Graph, nx.DiGraph, np.ndarray, Dict[Any,int], int]:
    """
    From adj_list and root, build:
      G        – undirected skeleton (nx.Graph),
      Gp_int   – directed integer-weighted graph (nx.DiGraph),
      L_int    – integer limitation matrix,
      idx      – node→index map,
      scale    – integer scale factor.
    """
    # get integer matrix and mapping
    L_int, idx, scale = ComputeLimitationMatrix(adj_list, root, K)

    # build undirected skeleton G
    G = nx.Graph()
    for u, neighs in adj_list.items():
        for v in neighs:
            G.add_edge(u, v)

    # build directed integer-weighted graph Gp_int
    nodes = list(adj_list.keys())
    inv_idx = {i: v for v, i in idx.items()}
    Gp_int = nx.DiGraph()
    Gp_int.add_nodes_from(nodes)
    n = L_int.shape[0]
    for i in range(n):
        for j in range(n):
            w = L_int[i, j]
            if w > 0:
                Gp_int.add_edge(inv_idx[i], inv_idx[j], weight=w)

    return G, Gp_int, L_int, idx, scale

def analyze_node_sums(L_int: np.ndarray, p: int, q: int, r: int) -> None:
    """
    Analyze the sums of rows and columns for each node (except 0) in the limitation matrix.
    Print node indices in (x,y,z) triple notation.
    """
    n_nodes = L_int.shape[0]
    
    # Initialize arrays to store sums for each node
    node_sums = np.zeros(n_nodes)
    col_sums = np.zeros(n_nodes)  # For incoming weights only
    
    # For each node (except 0)
    for node in range(1, n_nodes):
        # Sum of row (outgoing edges) and column (incoming edges)
        row_sum = np.sum(L_int[node, :])
        col_sum = np.sum(L_int[:, node])
        total_sum = row_sum + col_sum
        node_sums[node] = total_sum
        col_sums[node] = col_sum  # Store column sum separately
    
    # Find the maximum sum across all nodes
    max_sum_overall = np.max(node_sums[1:])  # Exclude node 0
    max_sum_node = np.argmax(node_sums[1:]) + 1  # +1 because we excluded node 0
    
    print("\nNode Sum Analysis:")
    print(f"Number of nodes: {n_nodes}")
    print(f"\nMaximum sum found: {max_sum_overall}")
    max_node_coords = get_coords(max_sum_node, p, q, r)
    print(f"Node with maximum sum: {max_node_coords}")
    
    print("\nDetailed node sums (excluding node 0):")
    print("Format: Node (x,y,z): total_sum (incoming_sum)")
    for node in range(1, n_nodes):
        coords = get_coords(node, p, q, r)
        print(f"Node {coords}: {node_sums[node]} (incoming: {col_sums[node]})")
    
    # Print nodes sorted by incoming sum
    print("\nNodes sorted by incoming sum (excluding node 0):")
    sorted_nodes = sorted(range(1, n_nodes), key=lambda x: col_sums[x], reverse=True)
    for node in sorted_nodes:
        coords = get_coords(node, p, q, r)
        print(f"Node {coords}: incoming sum = {col_sums[node]}")

def make_multigraph(
    adj_list: Dict[Any, List[Any]],
    root: Any,
    K: float = 1.0
) -> Tuple[nx.MultiGraph, Dict[Any,int], int]:
    """
    Build an undirected multigraph from the integer limitation matrix
    of a generic graph (given as adj_list and root).
    Returns (G, index_map, scale).
    """
    L_int, idx, scale = ComputeLimitationMatrix(adj_list, root, K)

    inv_idx = {i: v for v, i in idx.items()}

    G = nx.MultiGraph()
    G.add_nodes_from(adj_list.keys())
    n = L_int.shape[0]
    for i in range(n):
        for j in range(n):
            w = L_int[i, j]
            if w > 0:
                u = inv_idx[i]
                v = inv_idx[j]
                for _ in range(int(w)):
                    G.add_edge(u, v)

    return G, idx, scale


def euler_split(G: nx.MultiGraph, Delta: int) -> Tuple[nx.MultiGraph, nx.MultiGraph]:
    """
    Perform the "Euler partition" on G (max-degree Delta, assumed even):
      1. Build H by adding dummy edges so that every vertex has degree exactly Delta.
      2. For each connected component of H, find an Euler tour and alternately
         assign real edges to G1 and G2, discarding dummy edges.
    Returns two MultiGraphs G1, G2 each on the same node set, with
    max_degree(G1), max_degree(G2) ≤ Delta // 2, unless a fallback move is required.
    """
    H = nx.MultiGraph()
    H.add_nodes_from(G.nodes())
    for u, v, key, data in G.edges(keys=True, data=True):
        # Store a marker so we can tell real edges from dummy
        H.add_edge(u, v, key=("real", u, v, key), **data)

    deficits: List[Any] = []
    for v in H.nodes():
        deficit = Delta - H.degree(v)
        for _ in range(deficit):
            deficits.append(v)

    for i in range(0, len(deficits), 2):
        u = deficits[i]
        v = deficits[i + 1]
        H.add_edge(u, v, key=("dummy", i // 2))

    G1 = nx.MultiGraph()
    G2 = nx.MultiGraph()
    G1.add_nodes_from(G.nodes())
    G2.add_nodes_from(G.nodes())

    # To find connected components, use a simple graph view
    H_simple = nx.Graph()
    H_simple.add_nodes_from(H.nodes())
    for u, v, _ in H.edges(keys=True):
        H_simple.add_edge(u, v)

    for comp in nx.connected_components(H_simple):
        Hc = H.subgraph(comp).copy()
        # Eulerian circuit on each component
        tour = list(nx.eulerian_circuit(Hc, source=next(iter(comp)), keys=True))
        take_from_G1 = True

        for u, v, key in tour:
            if key[0] == "dummy":
                take_from_G1 = not take_from_G1
                continue

            _, u0, v0, orig_key = key
            data = G[u0][v0][orig_key]
            if take_from_G1:
                G1.add_edge(u0, v0, key=orig_key, **data)
            else:
                G2.add_edge(u0, v0, key=orig_key, **data)
            take_from_G1 = not take_from_G1

    return G1, G2


def rebalance_edge_colors(G: nx.MultiGraph) -> None:
    """
    Improved rebalancing to minimize the gap between the largest and smallest color-buckets:
      • First, determine the actual number of colors in use.
      • Compute ideal lower = floor(total_edges/num_colors) and upper = ceil(total_edges/num_colors).
      • While there exists any color with count > upper or < lower, attempt to move one edge:
        – Consider every pair (c_big, c_small) where count[c_big] > upper and count[c_small] < lower,
          and attempt to move one edge from c_big to c_small if no adjacency conflict.
      • Repeat until no such move is possible.
    """
    # Determine how many colors are actually in use
    counts: Dict[int,int] = {}
    for _, _, _, data in G.edges(keys=True, data=True):
        c = data["color"]
        counts[c] = counts.get(c, 0) + 1
    if not counts:
        return

    num_colors = max(counts.keys()) + 1
    total_edges = G.number_of_edges()
    lower = total_edges // num_colors
    upper = lower + (1 if total_edges % num_colors != 0 else 0)

    def compute_counts() -> Dict[int, int]:
        ec = {c: 0 for c in range(num_colors)}
        for _, _, _, data in G.edges(keys=True, data=True):
            c = data["color"]
            ec[c] += 1
        return ec

    while True:
        curr = compute_counts()
        overfull = [c for c, cnt in curr.items() if cnt > upper]
        underfull = [c for c, cnt in curr.items() if cnt < lower]
        if not overfull or not underfull:
            break

        moved = False
        # Try every pair (c_big, c_small)
        for c_big in sorted(overfull, key=lambda c: curr[c], reverse=True):
            for c_small in sorted(underfull, key=lambda c: curr[c]):
                # Attempt to move one edge from c_big to c_small
                for u, v, k, data in G.edges(keys=True, data=True):
                    if data["color"] != c_big:
                        continue
                    # Check adjacency conflict at u and v
                    conflict = False
                    for nbr in G[u]:
                        for subk, dat in G[u][nbr].items():
                            if dat["color"] == c_small:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        continue
                    for nbr in G[v]:
                        for subk, dat in G[v][nbr].items():
                            if dat["color"] == c_small:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        continue

                    # Perform move
                    G[u][v][k]["color"] = c_small
                    moved = True
                    break
                if moved:
                    break
            if moved:
                break

        if not moved:
            break


def _color_recursive(G: nx.MultiGraph) -> Tuple[nx.MultiGraph, Dict[Tuple[Any, Any, int], int]]:
    """
    Recursively color edges of G and return a new colored graph Gc
    plus a color_map mapping each (u,v,key) → color.

    Uses Δ-partitioning exactly as in the paper, with a fallback if the split fails
    to reduce the maximum degree.
    """
    Delta = max(d for _, d in G.degree())

    # Base case: Delta ≤ 1
    if Delta <= 1:
        Gc = nx.MultiGraph()
        Gc.add_nodes_from(G.nodes())
        color_map: Dict[Tuple[Any, Any, int], int] = {}
        for u, v, k, data in G.edges(keys=True, data=True):
            Gc.add_edge(u, v, key=k, **data)
            Gc[u][v][k]["color"] = 0
            color_map[(u, v, k)] = 0
        return Gc, color_map

    # If Delta is odd: remove one Δ-matching using Gabow '76 and color those edges 0
    pre_colored: List[Tuple[Any, Any, int]] = []
    if Delta % 2 == 1:
        S = nx.Graph()
        S.add_nodes_from(G.nodes())
        buckets: Dict[Tuple[Any, Any], List[Tuple[Any, Any, int]]] = {}
        for u, v, k in G.edges(keys=True):
            a, b = (u, v) if u <= v else (v, u)
            buckets.setdefault((a, b), []).append((u, v, k))
        S.add_edges_from(buckets.keys())

        weight = {v: (1 if G.degree(v) == Delta else 0) for v in G.nodes()}
        for u, v in S.edges():
            S[u][v]["weight"] = weight[u] + weight[v]

        M = max_weight_matching(S, maxcardinality=False, weight="weight")

        remaining = nx.MultiGraph()
        remaining.add_nodes_from(G.nodes())
        for u, v, k, data in G.edges(keys=True, data=True):
            remaining.add_edge(u, v, key=k, **data)

        for u, v in M:
            a, b = (u, v) if u <= v else (v, u)
            u0, v0, k0 = buckets[(a, b)].pop()
            remaining.remove_edge(u0, v0, k0)
            # Re-add into pre_colored list; we'll assign color 0 later
            pre_colored.append((u0, v0, k0))

        Delta_minus_1 = Delta - 1
        # Recursively color the remaining (now even‐Δ) graph
        G1c, color_map1 = _color_recursive(remaining)

        # Build Gc by merging pre_colored edges (color 0) with G1c (shifted by +1)
        Gc = nx.MultiGraph()
        Gc.add_nodes_from(G.nodes())
        color_map: Dict[Tuple[Any, Any, int], int] = {}

        # Add pre_colored edges with color 0, preserving key and data
        for u0, v0, k0 in pre_colored:
            data = G[u0][v0][k0]
            Gc.add_edge(u0, v0, key=k0, **data)
            Gc[u0][v0][k0]["color"] = 0
            color_map[(u0, v0, k0)] = 0

        # Add edges from G1c with color = original_color + 1
        for u, v, k, data in G1c.edges(keys=True, data=True):
            # Copy full attribute dict, then override color
            attr = dict(data)
            orig_color = attr.get("color", 0)
            new_color = orig_color + 1
            attr["color"] = new_color
            Gc.add_edge(u, v, key=k, **attr)
            color_map[(u, v, k)] = new_color

        return Gc, color_map

    # Delta is even: perform Euler split
    G1, G2 = euler_split(G, Delta)

    # FALLBACK: if split left one half still at full Δ, move one real edge out
    d1 = max((d for _, d in G1.degree()), default=0)
    d2 = max((d for _, d in G2.degree()), default=0)
    if d1 == Delta:
        # Move a single edge from G1 to G2 (preserve key and data)
        for u, v, k in G1.edges(keys=True):
            data = G1[u][v][k]
            G1.remove_edge(u, v, k)
            G2.add_edge(u, v, key=k, **data)
            break
    elif d2 == Delta:
        for u, v, k in G2.edges(keys=True):
            data = G2[u][v][k]
            G2.remove_edge(u, v, k)
            G1.add_edge(u, v, key=k, **data)
            break

    # Recurse on the (now strictly smaller‐max‐degree) subgraphs
    G1c, color_map1 = _color_recursive(G1)
    G2c, color_map2 = _color_recursive(G2)

    # Merge G1c and G2c into Gc, shifting G2c colors by Delta//2
    half = Delta // 2
    Gc = nx.MultiGraph()
    Gc.add_nodes_from(G.nodes())
    color_map: Dict[Tuple[Any, Any, int], int] = {}

    # Copy G1c edges (colors 0..half-1)
    for u, v, k, data in G1c.edges(keys=True, data=True):
        attr = dict(data)
        # color is already correct for G1c
        Gc.add_edge(u, v, key=k, **attr)
        color_map[(u, v, k)] = attr["color"]

    # Copy G2c edges, adding 'half' to their colors
    for u, v, k, data in G2c.edges(keys=True, data=True):
        attr = dict(data)
        orig_color = attr.get("color", 0)
        new_color = orig_color + half
        attr["color"] = new_color
        Gc.add_edge(u, v, key=k, **attr)
        color_map[(u, v, k)] = new_color

    return Gc, color_map


def delta_edge_coloring(G: nx.MultiGraph) -> int:
    """
    Edge-color G in place using exactly Delta colors (for even Delta),
    or Delta colors after removing a matching (for odd Delta).
    Returns the number of colors used.
    """
    # Call the recursive builder to get Gc and the color_map
    Gc, color_map = _color_recursive(G)

    # Rebalance in-place on Gc
    rebalance_edge_colors(Gc)

    # Build a final color_map from Gc (in case rebalancing changed anything)
    final_color_map: Dict[Tuple[Any, Any, int], int] = {}
    for u, v, k, data in Gc.edges(keys=True, data=True):
        final_color_map[(u, v, k)] = data["color"]

    # Copy colors from Gc back into G using the explicit map
    for u, v, k in G.edges(keys=True):
        # It's possible edges were swapped between G1 and G2, but keys remain consistent.
        G[u][v][k]["color"] = final_color_map[(u, v, k)]

    # Number of colors is the max color + 1
    num_colors = max(final_color_map.values()) + 1
    return num_colors

def get_node_coords(node_idx: int, p: int, q: int, r: int) -> Tuple[int, int, int]:
    """Convert 1D node index to 3D coordinates (or 2D if r=0)."""
    x = node_idx % p
    y = (node_idx // p) % q
    z = node_idx // (p * q) if r > 0 else 0
    return (x, y, z)


def compute_manhattan_distances(p: int, q: int, r: int) -> np.ndarray:
    """
    Build an array `dist[v]` = Manhattan distance from the root (0,0,0)
    for each node index v in a p×q×r grid. If r=0, we treat it as 2D (p×q).
    """
    n = p * q * (r if r > 0 else 1)
    dist = np.zeros((n,), dtype=int)
    for v in range(n):
        x, y, z = get_node_coords(v, p, q, r)
        dist[v] = x + y + z
    return dist


def order_frames_urgent(
    frames: List[np.ndarray],
    p: int,
    q: int,
    r: int,
    N: int
) -> List[np.ndarray]:
    """
    Greedy frame ordering that prioritizes node activation (first-time deliveries),
    mirroring the 2D mesh strategy but applied to 3D grids as well.

    Score per unused frame aggregates over deliverable edges u→v:
      - Urgency: (N - data_sim[v]) * (dist[v] + 1)
      - Root proximity: boost if min(dist[u], dist[v]) small (primes wavefront)
      - Forward progress: +dv if dist[v] > dist[u], mild penalties for dv <= 0
      - Deliverables: +10 per deliverable edge (slightly higher than before)
      - Activations: capped linear term to avoid huge spikes early
      - Smoothing: penalize sharp activation jumps early on

    Falls back to max-deliverables if no positive-score frame exists; if none deliverable,
    picks any remaining frame.
    """
    n = frames[0].shape[0]
    used: set = set()
    ordered: List[np.ndarray] = []

    # Simulated chunk counts (root preloaded with N)
    data_sim = np.zeros((n,), dtype=int)
    data_sim[0] = N

    # Manhattan distance from root for prioritization
    dist = compute_manhattan_distances(p, q, r)

    # Precompute per-frame edge lists
    u_list: Dict[int, List[int]] = {}
    v_list: Dict[int, List[int]] = {}
    for i, F in enumerate(frames):
        us, vs = np.nonzero(F)
        u_list[i] = us.tolist()
        v_list[i] = vs.tolist()

    # Heuristic boost for moves along the longest axis/axes toward max coordinate
    dims = (p, q, (r if r > 0 else 1))
    max_len = max(dims)
    longest_axes = {i for i, L in enumerate(dims) if L == max_len}

    last_activations: Optional[int] = None

    # Tunables for smoother ramp-up
    DELIVER_WEIGHT = 10.0
    ACT_WEIGHT = 8.0
    ACT_CAP = 10  # cap activation contribution per frame
    ROOT_BOOST_NEAR = 3  # frames touching nodes with dist <= this get boost
    ROOT_BOOST = 1.2

    while len(used) < len(frames):
        best_idx = -1
        best_score = -1e18

        for i in range(len(frames)):
            if i in used:
                continue

            S_i = 0.0
            deliverables = 0
            activations = 0
            forward_sum = 0.0
            near_root = False

            for u, v in zip(u_list[i], v_list[i]):
                if data_sim[u] > 0 and data_sim[v] < N:
                    base = (N - data_sim[v]) * (dist[v] + 1)

                    # Identify axis of movement and direction
                    ux, uy, uz = get_node_coords(u, p, q, r)
                    vx, vy, vz = get_node_coords(v, p, q, r)
                    dx, dy, dz = vx - ux, vy - uy, vz - uz
                    if dx != 0:
                        axis = 0
                        toward_max = dx > 0
                    elif dy != 0:
                        axis = 1
                        toward_max = dy > 0
                    else:
                        axis = 2
                        toward_max = dz > 0
                    if axis in longest_axes:
                        base *= 1.5
                        if toward_max:
                            base *= 1.2

                    # Root proximity boost (encourage early frames to start at root)
                    if min(dist[u], dist[v]) <= ROOT_BOOST_NEAR:
                        base *= ROOT_BOOST
                        near_root = True
                    S_i += base
                    deliverables += 1
                    if data_sim[v] == 0:
                        activations += 1

                    dv = dist[v] - dist[u]
                    if dv > 0:
                        forward_sum += dv
                    elif dv == 0:
                        forward_sum -= 0.5
                    else:
                        forward_sum -= 2.0

            # Smoothing to damp large activation swings early on
            smoothing = 0.0
            if last_activations is not None:
                diff = abs(activations - last_activations)
                if len(used) <= 3:
                    smoothing = -8.0 * diff
                else:
                    smoothing = -4.0 * diff

            # Activation contribution with cap to avoid huge spikes
            act_contrib = ACT_WEIGHT * min(activations, ACT_CAP)

            # Small bias toward frames near root in the very first selections
            root_bias = 5.0 if (near_root and len(used) < 2) else 0.0

            score = 0.5 * S_i + 0.5 * forward_sum + DELIVER_WEIGHT * deliverables + act_contrib + smoothing + root_bias
            if score > best_score:
                best_score = score
                best_idx = i

        # Fallback: max deliverables if score not positive, else any remaining
        if best_idx == -1 or best_score <= 0:
            best_count = -1
            chosen = -1
            for i in range(len(frames)):
                if i in used:
                    continue
                cnt = 0
                for u, v in zip(u_list[i], v_list[i]):
                    if data_sim[u] > 0 and data_sim[v] < N:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    chosen = i

            if best_count <= 0:
                for i in range(len(frames)):
                    if i not in used:
                        chosen = i
                        break
        else:
            chosen = best_idx

        # Count newly activated receivers (before applying the frame)
        a_next = 0
        for u, v in zip(u_list[chosen], v_list[chosen]):
            if data_sim[u] > 0 and data_sim[v] == 0:
                a_next += 1

        # Apply the chosen frame to the simulation state
        for u, v in zip(u_list[chosen], v_list[chosen]):
            if data_sim[u] > 0 and data_sim[v] < N:
                data_sim[v] += 1

        ordered.append(frames[chosen])
        used.add(chosen)
        last_activations = a_next

    # Optional: rotation to reduce early oscillations (small proxy N)
    def eval_rotation(rot: int) -> float:
        sim_N = max(8, min(32, N))
        sim_state = np.zeros((n,), dtype=int)
        sim_state[0] = sim_N
        seq = ordered[rot:] + ordered[:rot]
        e_us, e_vs = [], []
        for F in seq:
            us, vs = np.nonzero(F)
            e_us.append(us)
            e_vs.append(vs)
        counts: List[int] = []
        for t in range(2 * len(seq)):
            idx_f = t % len(seq)
            us = e_us[idx_f]
            vs = e_vs[idx_f]
            cnt = 0
            for u, v in zip(us, vs):
                if sim_state[u] > 0 and sim_state[v] < sim_N:
                    cnt += 1
            counts.append(cnt)
            for u, v in zip(us, vs):
                if sim_state[u] > 0 and sim_state[v] < sim_N:
                    sim_state[v] += 1
        pen = 0
        for a, b in zip(counts, counts[1:]):
            pen += abs(a - b)
        return pen

    best_rot = 0
    best_pen = float('inf')
    for r_off in range(len(ordered)):
        pen = eval_rotation(r_off)
        if pen < best_pen:
            best_pen = pen
            best_rot = r_off

    if best_rot != 0:
        ordered = ordered[best_rot:] + ordered[:best_rot]

    # --- Additional smoothing: ensure early seeding from root, then interleave and local swaps ---
    def has_root_sender_edge(F: np.ndarray) -> bool:
        # Any oriented edge root(0) -> v
        return np.any(F[0, :] > 0)

    def count_root_sender_edges(F: np.ndarray) -> int:
        # Total oriented edges from root(0)
        return int(np.sum(F[0, :]))
    def frame_load(F: np.ndarray) -> int:
        return int(np.sum(F))

    # Build a prefix (up to 4 frames) that actively send from root to kick-start dissemination
    root_candidates = [(i, count_root_sender_edges(F)) for i, F in enumerate(ordered) if has_root_sender_edge(F)]
    root_candidates.sort(key=lambda x: x[1], reverse=True)
    prefix_pick = [i for i, _ in root_candidates[:4]]
    # If fewer than 4, fill with frames that touch nodes near the root (dist <= 2)
    def is_near_root_frame(F: np.ndarray) -> bool:
        us, vs = np.nonzero(F)
        for u, v in zip(us, vs):
            if min(dist[u], dist[v]) <= 2:
                return True
        return False
    if len(prefix_pick) < 4:
        near_idxs = [i for i, F in enumerate(ordered) if i not in prefix_pick and is_near_root_frame(F)]
        for i in near_idxs:
            prefix_pick.append(i)
            if len(prefix_pick) >= 4:
                break
    rest_idxs = [i for i in range(len(ordered)) if i not in prefix_pick]

    # Interleave remainder via bucketed round-robin to reduce clustering of heavy frames
    sizes = [(i, frame_load(ordered[i])) for i in rest_idxs]
    sizes.sort(key=lambda x: x[1], reverse=True)
    bucket_count = max(3, min(6, len(sizes)//4 if len(sizes) >= 12 else 4))
    buckets: List[List[int]] = [[] for _ in range(bucket_count)]
    for idx, (i, _w) in enumerate(sizes):
        buckets[idx % bucket_count].append(i)
    interleaved_idx: List[int] = []
    # round-robin merge of buckets
    more = True
    while more:
        more = False
        for b in buckets:
            if b:
                interleaved_idx.append(b.pop(0))
                more = True
    # Combine prefix + interleaved remainder
    ordered = [ordered[i] for i in prefix_pick] + [ordered[i] for i in interleaved_idx]

    # Evaluate a sequence by simulating two cycles with small-N proxy and
    # summing absolute step-to-step deltas (lower is smoother).
    def score_sequence(seq: List[np.ndarray]) -> float:
        sim_N = max(8, min(32, N))
        n_local = seq[0].shape[0]
        # Sim state: root preloaded with sim_N
        sim_state = np.zeros((n_local,), dtype=int)
        sim_state[0] = sim_N
        # Precompute edge lists for speed
        e_us: List[np.ndarray] = []
        e_vs: List[np.ndarray] = []
        for F in seq:
            us, vs = np.nonzero(F)
            e_us.append(us)
            e_vs.append(vs)
        counts: List[int] = []
        T = 2 * len(seq)
        for t in range(T):
            idx = t % len(seq)
            us = e_us[idx]
            vs = e_vs[idx]
            cnt = 0
            for u, v in zip(us, vs):
                if sim_state[u] > 0 and sim_state[v] < sim_N:
                    cnt += 1
            counts.append(cnt)
            for u, v in zip(us, vs):
                if sim_state[u] > 0 and sim_state[v] < sim_N:
                    sim_state[v] += 1
        # Delta penalty
        pen = 0.0
        for a, b in zip(counts, counts[1:]):
            pen += abs(a - b)
        # Boundary continuity penalty across cycle wrap
        if len(seq) > 0:
            pen += 0.5 * abs(counts[len(seq)-1] - counts[0])
        # Early zeros penalty: discourage empty early steps
        early_window = min(6, len(counts))
        for t in range(early_window):
            if counts[t] == 0:
                pen += 5.0
        # Mild variance penalty across a single cycle
        cycle = counts[:len(seq)]
        if cycle:
            mu = sum(cycle) / float(len(cycle))
            var = sum((c - mu) ** 2 for c in cycle) / float(len(cycle))
            pen += 0.05 * var
        return pen

    # Greedy adjacent-swap hill-climb with limited passes
    def smooth_adjacent(seq: List[np.ndarray], max_passes: int = 6, frozen_prefix: int = 0) -> List[np.ndarray]:
        cur = list(seq)
        cur_score = score_sequence(cur)
        for _ in range(max_passes):
            improved = False
            for i in range(max(0, frozen_prefix), len(cur) - 1):
                cand = list(cur)
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                s = score_sequence(cand)
                if s + 1e-9 < cur_score:
                    cur, cur_score = cand, s
                    improved = True
            if not improved:
                break
        return cur

    ordered = smooth_adjacent(ordered, max_passes=6, frozen_prefix=len(prefix_pick))

    # Non-adjacent local swaps within a small window
    def smooth_pairs(seq: List[np.ndarray], frozen_prefix: int = 0, passes: int = 2, radius: int = 6) -> List[np.ndarray]:
        cur = list(seq)
        cur_score = score_sequence(cur)
        for _ in range(passes):
            improved = False
            for i in range(max(0, frozen_prefix), len(cur) - 1):
                j_max = min(len(cur), i + 1 + radius)
                best_j = -1
                best_score = cur_score
                for j in range(i + 1, j_max):
                    cand = list(cur)
                    cand[i], cand[j] = cand[j], cand[i]
                    s = score_sequence(cand)
                    if s + 1e-9 < best_score:
                        best_score = s
                        best_j = j
                if best_j != -1:
                    cur[i], cur[best_j] = cur[best_j], cur[i]
                    cur_score = best_score
                    improved = True
            if not improved:
                break
        return cur

    ordered = smooth_pairs(ordered, frozen_prefix=len(prefix_pick), passes=2, radius=6)

    # Greedy resequencing over the remainder (counts model) to avoid stalls throughout
    def deliverables_for_frame(F: np.ndarray, sim_state: np.ndarray, cap: int) -> int:
        us, vs = np.nonzero(F)
        cnt = 0
        for u, v in zip(us, vs):
            if sim_state[u] > 0 and sim_state[v] < cap:
                cnt += 1
        return cnt

    def play_frame_counts(F: np.ndarray, sim_state: np.ndarray, cap: int) -> None:
        us, vs = np.nonzero(F)
        for u, v in zip(us, vs):
            if sim_state[u] > 0 and sim_state[v] < cap:
                sim_state[v] += 1

    if len(ordered) > len(prefix_pick):
        sim_cap = max(8, min(32, N))
        n_local = ordered[0].shape[0]
        sim_state = np.zeros((n_local,), dtype=int)
        sim_state[0] = sim_cap
        # Seed the simulation with the fixed prefix
        for idx in range(len(prefix_pick)):
            play_frame_counts(ordered[idx], sim_state, sim_cap)
        # Greedy selection for the entire remainder
        remaining = list(range(len(prefix_pick), len(ordered)))
        selection: List[int] = []
        while remaining:
            best_j = remaining[0]
            best_cnt = deliverables_for_frame(ordered[best_j], sim_state, sim_cap)
            for j in remaining[1:]:
                cnt = deliverables_for_frame(ordered[j], sim_state, sim_cap)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_j = j
            selection.append(best_j)
            play_frame_counts(ordered[best_j], sim_state, sim_cap)
            remaining.remove(best_j)
        ordered = [ordered[i] for i in range(len(prefix_pick))] + [ordered[i] for i in selection]

    # Final rotation pass after smoothing
    # Final rotation: preserve seeding prefix at the start if it exists
    if len(prefix_pick) == 0:
        best_rot = 0
        best_pen = float('inf')
        for r_off in range(len(ordered)):
            pen = eval_rotation(r_off)
            if pen < best_pen:
                best_pen = pen
                best_rot = r_off
        if best_rot != 0:
            ordered = ordered[best_rot:] + ordered[:best_rot]

    return ordered


def create_directed_frames(
    G_colored: nx.MultiGraph,
    L_int: np.ndarray,
    idx: Dict[Any, int],
    p: int,
    q: int,
    r: int
) -> List[np.ndarray]:
    """
    Build a list of directed-matching frames from the undirected, colored graph G_colored.
    We orient each undirected edge (u–v) by comparing remaining L_int[u,v] vs L_int[v,u].
    Finally we re-order them using order_frames_urgent, which now ensures no multi-step stalls.

    Args:
      G_colored:  undirected MultiGraph where G_colored[u][v][k]["color"] ∈ {0..Δ-1}.
      L_int:      n×n limitation (capacity) matrix.
      idx:        mapping from node key → integer 0..n-1.
      p, q, r:    grid dimensions.

    Returns:
      A list of n×n 0/1 numpy arrays (directed matchings), in "urgent + deliverable" order.
    """
    n = len(idx)
    used_colors = sorted({ data["color"] for _, _, _, data in G_colored.edges(keys=True, data=True) })
    num_colors = len(used_colors)

    # 1) Build empty directed-frame list
    frames: List[np.ndarray] = [np.zeros((n, n), dtype=int) for _ in range(num_colors)]
    reference = L_int.copy()

    # 2) Group all edges by their undirected color
    edges_by_color: Dict[int, List[Tuple[Any,Any,int]]] = {c: [] for c in used_colors}
    for u, v, k in G_colored.edges(keys=True):
        c = G_colored[u][v][k]["color"]
        edges_by_color[c].append((u, v, k))

    # 3) Orient each color-class of edges
    for c in used_colors:
        ci = used_colors.index(c)
        E = edges_by_color[c]

        # Sort by |remaining(u→v) – remaining(v→u)| desc
        def score_edge(e):
            u0, v0, _ = e
            i0, j0 = idx[u0], idx[v0]
            return abs(reference[i0, j0] - reference[j0, i0])

        E_sorted = sorted(E, key=score_edge, reverse=True)

        for u0, v0, k0 in E_sorted:
            i0, j0 = idx[u0], idx[v0]
            fwd = reference[i0, j0]
            bwd = reference[j0, i0]

            if fwd >= bwd and fwd > 0:
                frames[ci][i0, j0] += 1
                reference[i0, j0] -= 1
            elif bwd > 0:
                frames[ci][j0, i0] += 1
                reference[j0, i0] -= 1
            # if both fwd and bwd == 0, that undirected edge had already been fully placed.

    # 4) Infer N from L_int column-sums
    col_sums = np.sum(L_int, axis=0)
    N = int(max(col_sums))  # every non-root node has incoming-sum = N

    # 5) Re-order frames to avoid any multi-step stalls
    ordered = order_frames_urgent(frames, p, q, r, N)
    return ordered


def verify_frames(frames: List[np.ndarray], L_int: np.ndarray) -> bool:
    """
    Verify that the stacked frames exactly equal L_int, and that no frame
    contains both u→v and v→u simultaneously.
    """
    stacked = sum(frames)
    if not np.array_equal(stacked, L_int):
        print("!! Stacked frames ≠ limitation matrix!")
        return False

    n = L_int.shape[0]
    for idx_f, F in enumerate(frames):
        for u in range(n):
            for v in range(n):
                if F[u, v] > 0 and F[v, u] > 0:
                    print(f"!! Frame {idx_f+1} has both {u}->{v} and {v}->{u}!")
                    return False
    return True

def generate_frames() -> List[np.ndarray]:
    """
    Generate frames using the directed Euler coloring method.
    Automatically switches between 2D and 3D based on r value.
    """
    # Determine if we're in 2D or 3D mode
    is_3d = r > 0
    
    if is_3d:
        # Build adjacency list for 3D mesh
        adj_list: Dict[Tuple[int,int,int], List[Tuple[int,int,int]]] = {}
        for i in range(p):
            for j in range(q):
                for k in range(r):
                    nbrs: List[Tuple[int,int,int]] = []
                    if i > 0:       nbrs.append((i-1, j, k))
                    if i < p - 1:   nbrs.append((i+1, j, k))
                    if j > 0:       nbrs.append((i, j-1, k))
                    if j < q - 1:   nbrs.append((i, j+1, k))
                    if k > 0:       nbrs.append((i, j, k-1))
                    if k < r - 1:   nbrs.append((i, j, k+1))
                    adj_list[(i, j, k)] = nbrs
        root = (0, 0, 0)
    else:
        # Build adjacency list for 2D grid
        adj_list: Dict[Tuple[int,int], List[Tuple[int,int]]] = {}
        for i in range(p):
            for j in range(q):
                nbrs: List[Tuple[int,int]] = []
                if i > 0:       nbrs.append((i-1, j))
                if i < p - 1:   nbrs.append((i+1, j))
                if j > 0:       nbrs.append((i, j-1))
                if j < q - 1:   nbrs.append((i, j+1))
                adj_list[(i, j)] = nbrs
        root = (0, 0)
    
    # Get limitation matrix and build colored multigraph
    G, idx, scale = make_multigraph(adj_list, root, K)
    
    # Use special case for 2×2×4 meshes
    if p == 2 and q == 2 and r == 4:
        L_int = compute_2x2x4_mesh_limitation_matrix()
    # Use special case for 2×2×r meshes (other than 4)
    elif p == 2 and q == 2 and r >= 2:
        L_int = compute_2x2xr_mesh_limitation_matrix(r)
    else:
        L_int, _, _ = ComputeLimitationMatrix(adj_list, root, K)
    
    # Color the edges
    num_colors = delta_edge_coloring(G)
    # logger.info(f"Colors used (Δ) = {num_colors}")
    
    # Create directed frames
    frames = create_directed_frames(G, L_int, idx, p, q, r)
    
    # Verify frames match limitation matrix
    stacked = sum(frames)
    if np.array_equal(stacked, L_int):
        # logger.info("✓ Frames correctly match limitation matrix")
        pass
    else:
        # logger.info("✗ Frames do not match limitation matrix!")
        pass
    
    return frames

###############################################################################
# FRAME-BASED BROADCAST FUNCTIONS
###############################################################################

def broadcast_using_frames(frames: List[np.ndarray]) -> Tuple[int, List[int], List[List[List[int]]]]:
    """
    Frame-based broadcast with DISTINCT messages (chunk IDs 1..N).
    - Uses only edges active in each frame (directed matching ensures half-duplex).
    - One message per active edge per step.
    - Operates strictly on mesh neighbors (guaranteed by frames derived from mesh graph).

    Returns
      (total_steps, active_edges_history, data_history)
    where data_history holds per-node chunk-count matrices (diagonal counts) per step for compatibility.
    """
    num_nodes = len(frames[0])

    # State: per-node set of received chunk IDs; root starts with all N chunks
    data_state: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
    data_state[0] = set(range(1, N + 1))

    # Light sender transmission history to diversify choices
    sender_hist: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    # Global chunk popularity (number of holders), initialized with root's holdings
    chunk_popularity: Dict[int, int] = {c: 1 for c in range(1, N + 1)}

    steps = 0
    active_edges_history: List[int] = []

    # For compatibility, snapshot a diagonal-count matrix over time
    def snapshot_counts() -> List[List[int]]:
        M = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            M[i][i] = len(data_state[i])
        return M

    data_history: List[List[List[int]]] = [snapshot_counts()]

    def all_full() -> bool:
        for i in range(num_nodes):
            if len(data_state[i]) < N:
                return False
        return True

    # Neighbor indices in 3D mesh for a linear index
    def neighbor_indices(idx: int) -> List[int]:
        x = idx % p
        y = (idx // p) % q
        z = idx // (p * q) if r > 0 else 0
        nbrs: List[int] = []
        if x - 1 >= 0: nbrs.append(idx - 1)
        if x + 1 < p:  nbrs.append(idx + 1)
        if y - 1 >= 0: nbrs.append(idx - p)
        if y + 1 < q:  nbrs.append(idx + p)
        if r > 0:
            if z - 1 >= 0:        nbrs.append(idx - p * q)
            if z + 1 < r:         nbrs.append(idx + p * q)
        return nbrs

    def select_optimal_chunk_for_sender_3d(
        available: Set[int],
        sender: int,
        receiver: int,
        data: Dict[int, Set[int]],
        step_num: int
    ) -> int:
        """Balanced, activation-aware chunk selection (distinct messages), 3D-aware.
        Goals:
          - Prefer chunks the receiver can forward next (local forward potential).
          - Balance global chunk popularity to avoid synchronization spikes.
          - Keep light diversity to avoid repeats per-sender.
        """
        if not available:
            return -1

        r_nbrs = neighbor_indices(receiver)
        recent = set(sender_hist.get(sender, [])[-4:])
        best_chunk = -1
        best_score = -10**9
        for c in available:
            # Receiver’s forward potential: neighbors that still lack this chunk
            fwd = 0
            for nb in r_nbrs:
                if c not in data[nb]:
                    fwd += 1
            score = 60 * fwd
            # Popularity balancing: prefer globally rarer chunks
            pop = chunk_popularity.get(c, 0)
            score -= 2 * pop
            # Stronger rarity bonus for a brand-new receiver to seed diverse chunks
            if len(data[receiver]) == 0:
                score -= 4 * pop
            # Small diversity bonuses
            if c not in recent:
                score += 3
            if c not in sender_hist.get(sender, []):
                score += 5
            if score > best_score:
                best_score = score
                best_chunk = c
        return best_chunk

    skip_zero = os.environ.get("BBS_SKIP_ZERO_FRAMES", "1") == "1"

    while not all_full():
        progress_made = False
        # Static ordered frame cycle
        for frame_index, frame in enumerate(frames):
            if all_full():
                break
            new_state = {i: set(chunks) for i, chunks in data_state.items()}
            active = 0
            sends_this_step: List[Tuple[int, int, int]] = []

            # Log pre-step state (node -> chunk set)
            if step_logger is not None:
                step_logger.info(f"Step {steps + 1:04d} — Using Frame {frame_index + 1}")
                for i in range(num_nodes):
                    coords = get_node_coords(i, p, q, r)
                    # Sort for stable output
                    chunks_sorted = sorted(list(data_state[i]))
                    step_logger.info(f"  Node {coords}: {chunks_sorted}")
                step_logger.info("  Sends:")

            # Soft cap on first-time activations per step to smooth ramp-up
            dim_sum = p + q + (r if r > 0 else 0)
            base_cap = max(8, int(0.5 * dim_sum))
            # Gradually relax the cap as steps increase
            relax = min(base_cap // 2, steps // 12)
            activation_cap = base_cap + relax
            new_activations_this_step = 0

            for i in range(num_nodes):
                for j in range(num_nodes):
                    if frame[i][j] > 0:
                        if len(data_state[i]) > 0 and len(data_state[j]) < N:
                            # If receiver is brand new and cap reached, defer this activation
                            if len(data_state[j]) == 0 and new_activations_this_step >= activation_cap:
                                continue
                            avail = data_state[i] - data_state[j]
                            if avail:
                                chunk = select_optimal_chunk_for_sender_3d(avail, i, j, data_state, steps)
                                if chunk != -1:
                                    # Count activation if this is receiver's first chunk
                                    was_new = (len(new_state[j]) == 0)
                                    new_state[j].add(chunk)
                                    sender_hist[i].append(chunk)
                                    # Update global popularity when a node first receives a chunk
                                    chunk_popularity[chunk] = chunk_popularity.get(chunk, 0) + 1
                                    if was_new:
                                        new_activations_this_step += 1
                                    active += 1
                                    sends_this_step.append((i, j, chunk))
            data_state = new_state
            # Optionally skip recording pure zero-activity frames (reduces visible stalls)
            if skip_zero and active == 0:
                # Do not advance timestep or record this step; proceed to next frame
                continue
            active_edges_history.append(active)
            steps += 1
            data_history.append(snapshot_counts())
            # Log sends list and short summary
            if step_logger is not None:
                for si, sj, ch in sends_this_step:
                    sx, sy, sz = get_node_coords(si, p, q, r)
                    rx, ry, rz = get_node_coords(sj, p, q, r)
                    step_logger.info(f"    ({sx},{sy},{sz}) -> ({rx},{ry},{rz})  chunk {ch}")
                step_logger.info(f"  Active transmissions this step: {active}")
                step_logger.info("")
            if active > 0:
                progress_made = True
        if not progress_made:
            # No progress over a full cycle; stop to avoid infinite loop
            break

    return steps, active_edges_history, data_history

def is_full(data_matrix: List[List[int]], N: int) -> bool:
    """
    Checks if all nodes have received all N chunks.
    """
    for i in range(len(data_matrix)):
        if data_matrix[i][i] < N:
            return False
    return True

###############################################################################
# PLOTTING
###############################################################################

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

def plot_active_edges_history(history: List[int], p: int, q: int, r: int, title: str = "Normalized Active Edges per Step") -> None:
    """
    Plots the ratio of active edges to total edges in each step.
    """
    # Disabled plotting
    pass

def save_to_csv(active_edges: List[int], p: int, q: int, r: int, N: int, algorithm_name: str = "BBS"):
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
    
    print(f"Data saved to {filename}")

###############################################################################
# MAIN
###############################################################################
def main() -> None:
    """
    Main function to demonstrate using frames for broadcast.
    Automatically switches between 2D and 3D based on r value.
    """
    # Parse command line arguments
    global p, q, r, N
    p, q, r, N = parse_command_line_args()
    
    # Suppress most output
    print(f"Running BCCLP on {p}×{q}×{r} grid with {N} chunks")
    
    # Debug 2×2×4 meshes
    if p == 2 and q == 2 and r == 4:
        debug_2x2x4_limitation_matrix()
    # Debug 2×2×r meshes (other than 4)
    elif p == 2 and q == 2 and r >= 2:
        debug_2x2xr_limitation_matrix(r)
    
    # Initialize optional per-step logger
    init_step_logger(p, q, r, N)

    # 1. Generate frames using directed Euler coloring
    frames = generate_frames()
    
    # 2. Use the frames to simulate the broadcast
    total_steps, active_edges_history, _ = broadcast_using_frames(frames)
    
    # 3. Save results to CSV
    save_to_csv(active_edges_history, p, q, r, N, "BBS")
    
    # 4. Print summary
    print(f"Broadcast completed in {total_steps} steps")
    print(f"Total transmissions: {sum(active_edges_history)}")

def compute_2x2xr_mesh_limitation_matrix(r: int) -> Tuple[np.ndarray]:
    """
    Compute limitation matrix for a 2×2×r mesh using your (A,B,C,D,E,F,G,H,I) = (10,5,1,2,4,1,5,3,1).
    Returns:
      - M: (N×N) numpy array of capacities
      - df: pandas DataFrame listing coords, type, incoming_sum
    """
    if r < 2:
        raise ValueError("r must be at least 2 for a 2×2×r mesh")

    N = 2 * 2 * r

    def idx(x: int, y: int, z: int) -> int:
        return z * 4 + y * 2 + x

    def coords(i: int) -> Tuple[int, int, int]:
        z, rem = divmod(i, 4)
        y, x = divmod(rem, 2)
        return x, y, z

    def node_type(x: int, y: int, z: int) -> str:
        # bottom layer
        if z == 0:
            if (x, y) == (0, 0): return '1.a'
            if (x, y) in [(1, 0), (0, 1)]: return '1.b'
            if (x, y) == (1, 1): return '1.c'
        # top layer
        if z == r - 1:
            if (x, y) == (0, 0): return '1.b'
            if (x, y) in [(1, 0), (0, 1)]: return '1.c'
            if (x, y) == (1, 1): return '1.d'
        # intermediate layers
        if 1 <= z <= r - 2:
            if (x, y) == (0, 0): return '2.a'
            if (x, y) in [(1, 0), (0, 1)]: return '2.b'
            if (x, y) == (1, 1): return '2.c'
        raise ValueError(f"Unclassified node at {(x, y, z)}")

    # capacity table
    # (A,B,C,D,E,F,G,H,I) = (10,5,1,2,4,1,5,3,1)
    A = 16
    B = 9
    C = 4
    D = 6
    E = 8
    F = 4
    G = 8
    H = 5
    I = 3
    WEIGHTS = {
        ('1.a','1.b'): (A,  0),
        ('1.a','2.a'): (A,  0),

        ('1.b','1.c'): ( B,  C),
        ('1.b','2.b'): ( B,  C),

        ('1.c','2.c'): ( E,  D),

        ('2.a','2.b'): ( G,  F),
        ('2.b','2.c'): ( H,  I),
        ('1.d','2.c'): ( D,  E),
        
        # Adjust weights to balance to 12
        ('2.a','2.a'): ( A,  0),  # Vertical connections between 2.a nodes
        ('2.b','2.b'): ( B,  C),  # Vertical connections between 2.b nodes  
        ('2.c','2.c'): ( E,  D),  # Vertical connections between 2.c nodes
        
        # Additional connections to balance specific nodes
        ('2.c','2.b'): (I,  H),  # 2.a to 1.b (top layer)
        ('2.b','1.c'): ( B,  C),  # 2.b to 1.c (top layer)
        ('2.a','1.b'): ( A,  0),  # 2.b to 1.c (top layer)
        ('2.c','1.d'): ( E,  D),  # 2.b to 1.c (top layer)

        ('1.c','1.d'): ( E,  D),  # 2.b to 1.c (top layer),
    }

    def neighbors(x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < 2 and 0 <= ny < 2 and 0 <= nz < r:
                yield nx, ny, nz

    # build the matrix
    M = np.zeros((N, N), dtype=int)
    for u in range(N):
        x, y, z = coords(u)
        tu = node_type(x, y, z)
        for nx, ny, nz in neighbors(x, y, z):
            v = idx(nx, ny, nz)
            if u < v:
                tv = node_type(nx, ny, nz)
                pair = (tu, tv)
                # skip undefined or same-type pairs
                if pair in WEIGHTS:
                    cuv, cvu = WEIGHTS[pair]
                elif (tv, tu) in WEIGHTS:
                    cvu, cuv = WEIGHTS[(tv, tu)]
                else:
                    continue
                M[u, v] = cuv
                M[v, u] = cvu

    # zero out all incoming to the root
    root = idx(0, 0, 0)
    M[:, root] = 0


    return M

def debug_2x2xr_limitation_matrix(r: int) -> None:
    """
    Debug function to print in-node weights for each node in a 2×2×r mesh.
    Shows the limitation matrix and the sum of incoming weights for each node.
    """
    print(f"\n=== DEBUG: 2×2×{r} Limitation Matrix ===")
    
    # Get the limitation matrix
    L_int = compute_2x2xr_mesh_limitation_matrix(r)
    
    def get_node_idx(x: int, y: int, z: int) -> int:
        return z * 4 + y * 2 + x
    
    def get_coords(idx: int) -> Tuple[int, int, int]:
        z = idx // 4
        rem = idx % 4
        y = rem // 2
        x = rem % 2
        return x, y, z
    
    def get_node_type(x: int, y: int, z: int) -> str:
        """Determine node type for debugging"""
        if x == 0 and y == 0 and z == 0:
            return '1.a'
        elif ((x == 1 and y == 0 and z == 0) or 
              (x == 0 and y == 1 and z == 0) or 
              (x == 0 and y == 0 and z == r-1)):
            return '1.b'
        elif ((x == 1 and y == 0 and z == r-1) or
              (x == 0 and y == 1 and z == r-1) or
              (x == 1 and y == 1 and z == 0)):
            return '1.c'
        elif x == 1 and y == 1 and z == r-1:
            return '1.d'
        elif (x == 0 and y == 0 and z >= 1 and z <= r-2):
            return '2.a'
        elif ((x == 1 and y == 0 and z >= 1 and z <= r-2) or 
              (x == 0 and y == 1 and z >= 1 and z <= r-2)):
            return '2.b'
        elif (x == 1 and y == 1 and z >= 1 and z <= r-2):
            return '2.c'
        else:
            return 'unknown'
    
    num_nodes = 2 * 2 * r
    
    print(f"Matrix shape: {L_int.shape}")
    print(f"Number of nodes: {num_nodes}")
    
    # Print the limitation matrix
    print("\nLimitation Matrix:")
    print("Format: L[i,j] = weight from node i to node j")
    print("Rows = sender, Columns = receiver")
    print("     ", end="")
    for j in range(num_nodes):
        coords = get_coords(j)
        node_type = get_node_type(*coords)
        print(f"({coords[0]},{coords[1]},{coords[2]}){node_type}", end=" ")
    print()
    
    for i in range(num_nodes):
        coords = get_coords(i)
        node_type = get_node_type(*coords)
        print(f"({coords[0]},{coords[1]},{coords[2]}){node_type}", end=" ")
        for j in range(num_nodes):
            print(f"{L_int[i,j]:6d}", end=" ")
        print()
    
    # Calculate and print in-node weights (sum of incoming edges)
    print(f"\nIn-node weights (sum of incoming edges):")
    print("Node (p,q,r)type: incoming_weight_sum")
    
    for node_idx in range(num_nodes):
        coords = get_coords(node_idx)
        node_type = get_node_type(*coords)
        incoming_sum = np.sum(L_int[:, node_idx])  # Sum of column
        print(f"Node {coords}{node_type}: {incoming_sum:3d}")
    
    # Check if all non-root nodes have equal incoming weights
    non_root_incoming = [np.sum(L_int[:, i]) for i in range(1, num_nodes)]
    if len(set(non_root_incoming)) == 1:
        print(f"\n✓ All non-root nodes have equal incoming weight: {non_root_incoming[0]}")
    else:
        print(f"\n✗ Non-root nodes have different incoming weights:")
        for i in range(1, num_nodes):
            coords = get_coords(i)
            node_type = get_node_type(*coords)
            print(f"  Node {coords}{node_type}: {non_root_incoming[i-1]}")
    
    # Print out-node weights (sum of outgoing edges)
    print(f"\nOut-node weights (sum of outgoing edges):")
    print("Node (p,q,r)type: outgoing_weight_sum")
    
    for node_idx in range(num_nodes):
        coords = get_coords(node_idx)
        node_type = get_node_type(*coords)
        outgoing_sum = np.sum(L_int[node_idx, :])  # Sum of row
        print(f"Node {coords}{node_type}: {outgoing_sum:3d}")
    
    print("=" * 50)

def debug_limitation_matrix(limitation_matrix, r):
    """Debug function to print limitation matrix and node classifications"""
    print(f"\n=== Limitation Matrix for 2×2×{r} mesh ===")
    print("Node classifications:")
    
    for z in range(r):
        for y in range(2):
            for x in range(2):
                node_idx = get_node_idx(x, y, z)
                
                # Classify nodes according to the exact classification provided
                if x == 0 and y == 0 and z == 0:
                    node_type = "1.a (Root corner)"
                elif ((x == 1 and y == 0 and z == 0) or 
                      (x == 0 and y == 1 and z == 0) or 
                      (x == 0 and y == 0 and z == r-1)):
                    node_type = "1.b (Root in-line corner)"
                elif ((x == 1 and y == 0 and z == r-1) or
                      (x == 0 and y == 1 and z == r-1) or
                      (x == 1 and y == 1 and z == 0)):
                    node_type = "1.c (Corner with two 1.b)"
                elif x == 1 and y == 1 and z == r-1:
                    node_type = "1.d (Root' corner)"
                elif (x == 0 and y == 0 and z >= 1 and z <= r-2):
                    node_type = "2.a (1.a-1.b edge)"
                elif ((x == 1 and y == 0 and z >= 1 and z <= r-2) or 
                      (x == 0 and y == 1 and z >= 1 and z <= r-2)):
                    node_type = "2.b (1.b-1.c edge)"
                elif (x == 1 and y == 1 and z >= 1 and z <= r-2):
                    node_type = "2.c (1.c-1.d edge)"
                else:
                    node_type = "unknown"
                
                print(f"  ({x},{y},{z}) -> {node_type}")
    
    print(f"\nLimitation Matrix ({4*r}×{4*r}):")
    for i in range(4*r):
        row = []
        for j in range(4*r):
            val = limitation_matrix[i, j]
            row.append(f"{val:2d}" if val != 0 else " .")
        print(f"  {i:2d}: {' '.join(row)}")
    
    print(f"\nIncoming weights for each node:")
    for z in range(r):
        for y in range(2):
            for x in range(2):
                node_idx = get_node_idx(x, y, z)
                incoming_weight = np.sum(limitation_matrix[:, node_idx])
                print(f"  ({x},{y},{z}): incoming weight = {incoming_weight}")

def compute_2x2x4_mesh_limitation_matrix() -> np.ndarray:
    """
    Compute limitation matrix for a 2×2×4 mesh using the specific edge weights provided.
    Based on the edge weights:
    (0,0,2) -> (1,0,2) : 1
    (1,0,2) -> (0,0,2) : 1
    (0,1,2) -> (1,1,2) : 1
    (1,1,2) -> (0,1,2) : 1
    
    Every non-root node must have exactly 2 incoming weight.
    
    Returns:
      - M: (16×16) numpy array of capacities
    """
    N = 2 * 2 * 4  # 16 nodes
    M = np.zeros((N, N), dtype=int)
    
    def idx(x: int, y: int, z: int) -> int:
        return z * 4 + y * 2 + x
    
    def coords(i: int) -> Tuple[int, int, int]:
        z = i // 4
        rem = i % 4
        y = rem // 2
        x = rem % 2
        return x, y, z
    
    # Define the specific edge weights for 2×2×4 mesh
    # Format: (x1,y1,z1) -> (x2,y2,z2) : weight
    edge_weights = {
        ((0,0,0), (1,0,0)) : 2,
        ((0,0,0), (0,1,0)) : 1,
        ((1,0,0), (1,1,0)) : 1,
        ((0,1,0), (1,1,0)) : 1,
        ((1,1,0), (0,1,0)) : 1,

        ((0,0,1), (1,0,1)) : 1,
        ((1,0,1), (0,0,1)) : 1,
        ((0,1,1), (1,1,1)) : 1,
        ((1,1,1), (0,1,1)) : 1,

        ((0,0,2), (1,0,2)) : 1,
        ((1,0,2), (0,0,2)) : 1,
        ((0,1,2), (1,1,2)) : 1,
        ((1,1,2), (0,1,2)) : 1,

        ((0,0,3), (1,0,3)) : 1,
        ((1,0,3), (0,0,3)) : 1,
        ((0,1,3), (1,1,3)) : 1,
        ((1,1,3), (0,1,3)) : 1,

        ((0,0,0), (0,0,1)) : 1,
        ((0,0,1), (0,0,2)) : 1,
        ((0,0,2), (0,0,3)) : 1,

        ((1,0,0), (1,0,1)) : 1,
        ((1,0,1), (1,0,2)) : 1,
        ((1,0,2), (1,0,3)) : 1,

        ((0,1,0), (0,1,1)) : 1,
        ((0,1,1), (0,1,2)) : 1,
        ((0,1,2), (0,1,3)) : 1,

        ((1,1,0), (1,1,1)) : 1,
        ((1,1,1), (1,1,2)) : 1,
        ((1,1,2), (1,1,3)) : 1,
    }
    
    # Build the matrix from edge weights
    for (x1, y1, z1), (x2, y2, z2) in edge_weights:
        i = idx(x1, y1, z1)
        j = idx(x2, y2, z2)
        weight = edge_weights[((x1, y1, z1), (x2, y2, z2))]
        M[i, j] = weight
    
    # Zero out all incoming to the root (0,0,0)
    root = idx(0, 0, 0)
    M[:, root] = 0
    
    return M

def debug_2x2x4_limitation_matrix() -> None:
    """
    Debug function to print the limitation matrix for the 2×2×4 mesh.
    Shows the limitation matrix and the sum of incoming weights for each node.
    """
    print(f"\n=== DEBUG: 2×2×4 Limitation Matrix ===")
    
    # Get the limitation matrix
    L_int = compute_2x2x4_mesh_limitation_matrix()
    
    def get_node_idx(x: int, y: int, z: int) -> int:
        return z * 4 + y * 2 + x
    
    def get_coords(idx: int) -> Tuple[int, int, int]:
        z = idx // 4
        rem = idx % 4
        y = rem // 2
        x = rem % 2
        return x, y, z
    
    num_nodes = 2 * 2 * 4  # 16 nodes
    
    print(f"Matrix shape: {L_int.shape}")
    print(f"Number of nodes: {num_nodes}")
    
    # Print the limitation matrix
    print("\nLimitation Matrix:")
    print("Format: L[i,j] = weight from node i to node j")
    print("Rows = sender, Columns = receiver")
    print("     ", end="")
    for j in range(num_nodes):
        coords = get_coords(j)
        print(f"({coords[0]},{coords[1]},{coords[2]})", end=" ")
    print()
    
    for i in range(num_nodes):
        coords = get_coords(i)
        print(f"({coords[0]},{coords[1]},{coords[2]})", end=" ")
        for j in range(num_nodes):
            print(f"{L_int[i,j]:6d}", end=" ")
        print()
    
    # Calculate and print in-node weights (sum of incoming edges)
    print(f"\nIn-node weights (sum of incoming edges):")
    print("Node (x,y,z): incoming_weight_sum")
    
    for node_idx in range(num_nodes):
        coords = get_coords(node_idx)
        incoming_sum = np.sum(L_int[:, node_idx])  # Sum of column
        print(f"Node {coords}: {incoming_sum:3d}")
    
    # Check if all non-root nodes have equal incoming weights
    non_root_incoming = [np.sum(L_int[:, i]) for i in range(1, num_nodes)]
    if len(set(non_root_incoming)) == 1:
        print(f"\n✓ All non-root nodes have equal incoming weight: {non_root_incoming[0]}")
    else:
        print(f"\n✗ Non-root nodes have different incoming weights:")
        for i in range(1, num_nodes):
            coords = get_coords(i)
            print(f"  Node {coords}: {non_root_incoming[i-1]}")
    
    # Print out-node weights (sum of outgoing edges)
    print(f"\nOut-node weights (sum of outgoing edges):")
    print("Node (x,y,z): outgoing_weight_sum")
    
    for node_idx in range(num_nodes):
        coords = get_coords(node_idx)
        outgoing_sum = np.sum(L_int[node_idx, :])  # Sum of row
        print(f"Node {coords}: {outgoing_sum:3d}")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 
