# A-Bcast-Model-for-Given-Topologies

This repository contains the simulation implementations used in our publication for evaluating  broadcast algorithms on several given topologies. The implementations live under the `Topology/` directory and include four algorithms used in the paper:

- BBS
- Greedy
- SRDA
- BinTree

Topologies available:

- `Topology/16K3/` — 16-node MPL topology used in the paper
- `Topology/2Dmesh/` — 2D mesh topology
- `Topology/3Dmesh/` — 3D mesh topology

Each topology folder contains four Python scripts: `BBS.py`, `Greedy.py`, `SRDA.py`, and `BinTree.py`.

## Requirements

- Python 3.8 or newer
- NumPy and matplotlib (used for some statistics/plots). These are optional for running the simulations but recommended when producing figures.

Install recommended packages with pip:

```bash
python3 -m pip install --user numpy matplotlib
```

## How to run

Each algorithm script exposes a simple command-line `main()` that expects a single integer argument: the information size (N) in chunks. The scripts print a short summary to the console and save a CSV file with timestep/active-edge statistics under a `data/` directory created next to the script.

Examples (from the repository root):

Run the 16K3 topology simulations (example N = 100):

```bash
cd Topology/16K3
python3 BBS.py 100
python3 Greedy.py 100
python3 SRDA.py 100
python3 BinTree.py 100
```

Run the 2D mesh (p by q) simulations (example N = 100):

```bash
cd Topology/2Dmesh
python3 BBS.py p q 100
python3 Greedy.py p q 100
python3 SRDA.py p q 100
python3 BinTree.py p q 100
```

Run the 3D mesh (p by q by r) simulations (example N = 100):

```bash
cd Topology/3Dmesh
python3 BBS.py p q r 100
python3 Greedy.py p q r 100
python3 SRDA.py p q r 100
python3 BinTree.py p q r 100
```

## Reproducing results for the paper

1. Choose topology and algorithm as above.
2. Use the same `N` values as reported in the paper and run the script. The script will print summary stats and create the CSV data file.
3. Use the CSV data for plotting or post-processing as required for figures/tables.

## Contact / Notes

For questions and issue, contact the authors listed on the paper.

