# Broadcast Algorithm Simulation on Given Topologies

Simulation framework for evaluating MPI broadcast algorithms across realistic network topologies using [SimGrid](https://simgrid.org/) SMPI. Compares `MPI_Bcast`, SRDA (Scatter + Recursive-Doubling Allgather), and pipelined chain broadcast on four interconnect topologies at scales from 128 to 1,024 nodes.

## Quick Start

All experiments run inside a Docker container (`simgrid/stable`) so you do not need to install SimGrid locally.

```bash
# Build the runner (outputs to bin/)
docker run --rm -v $(pwd):/work -w /work/src simgrid/stable:latest \
    make

# Run a single experiment
docker run --rm -v $(pwd):/work -w /work/src simgrid/stable:latest \
    smpirun -np 128 \
        -platform ../topo/FatTree/platform_fattree_128.xml \
        -hostfile ../topo/FatTree/hostfile_128 \
        --cfg=smpi/host-speed:2000Gf \
        --log=root.thres:warning \
        ../bin/runner mpi 1048576 64 0
```

### Workflow

```bash
# 1. Generate topology files (if not already present)
python3 src/topology_generater.py --all

# 2. Build and sweep all experiments
docker run --rm -v $(pwd):/work -w /work/src simgrid/stable:latest make
./src/sweep.sh --dry-run                          # preview commands
./src/sweep.sh                                    # run all (root=0)
./src/sweep.sh --roots all                        # all roots 0..N-1
./src/sweep.sh --topos FatTree --sizes 128 --algos mpi   # subset

# 3. Aggregate results
python3 src/aggregate.py
```

## Container

The exact environment used to produce all results is the Docker image [`simgrid/stable:latest`](https://hub.docker.com/r/simgrid/stable). A pre-built Singularity image (`bcast.sif`) is included for HPC use.

| Component | Detail |
|-----------|--------|
| Base image | `simgrid/stable:latest` (Debian) |
| SimGrid | 4.1 |
| Compiler | GCC + `smpicc` (SimGrid's MPI C wrapper) |
| Launcher | `smpirun` (simulated MPI execution) |

**Docker** (local):
```bash
docker run --rm -v $(pwd):/work -w /work simgrid/stable:latest <command>
```

**Singularity** (HPC):
```bash
singularity exec --bind $(pwd) bcast.sif <command>
```

Both use the identical SimGrid runtime — results are reproducible across environments.

## Algorithms

| Algorithm | Key | Description |
|-----------|-----|-------------|
| MPI_Bcast | `mpi` | SimGrid's built-in binomial-tree broadcast (baseline) |
| SRDA | `srda` | Scatter + Recursive-Doubling Allgather. `MPI_Scatter` distributes N equal pieces, then log2(N) rounds of `MPI_Sendrecv` (rank XOR 2^k). Requires power-of-2 N. |
| Pipelined | `pipe` | Chain-pipeline broadcast: root sends chunks sequentially along a chain |

## Topologies

All network parameters are derived from published hardware specifications (see [`docs/topo_reference.txt`](docs/topo_reference.txt) for BibTeX citations).

| Topology | Interconnect | Bandwidth | Latency | Modeled After |
|----------|-------------|-----------|---------|---------------|
| 2Dmesh | InfiniBand NDR 400 | 50 GB/s | 100 ns | NVIDIA Eos |
| Butterfly | InfiniBand EDR | 12.5 GB/s | 100 ns | Kim & Dally, ISCA '07 |
| Dragonfly | Cray Aries | 5.25 GB/s | 100/200/400 ns | Theta (ALCF) |
| FatTree | InfiniBand EDR | 12.5 GB/s | 100 ns | Summit (ORNL) |

Pre-generated platform XMLs and hostfiles for N = 128, 256, 512, 1024 are in `topo/`.

## Runner Usage

```
smpirun -np N -platform <xml> -hostfile <hf> \
    bin/runner <algo> <msg_bytes> [nchunks] [root] [out_json]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `algo` | (required) | `mpi`, `srda`, or `pipe` |
| `msg_bytes` | (required) | Message size in bytes |
| `nchunks` | 64 | Number of chunks for pipelining |
| `root` | 0 | Broadcast root rank |
| `out_json` | `_` | Path to write JSON result (`_` = none) |

## Output

Each experiment writes a JSON file to `data/{Topo}/{algo}/N{N}_MSG{M}_R{root}.json`:

```json
{
  "algorithm": "srda",
  "nodes": 512,
  "msg_bytes": 67108864,
  "nchunks": 4096,
  "root": 42,
  "time_sec": 0.018013,
  "correct": true
}
```

Run `python3 src/aggregate.py` to compute mean +/- stdev across all roots and produce `data/summary.csv`.

## Experiment Parameters

| Parameter | Values |
|-----------|--------|
| Topologies | 2Dmesh, Butterfly, Dragonfly, FatTree |
| N | 128, 256, 512, 1024 |
| Message sizes | 256 B, 1 KB, 4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 4 MB, 16 MB, 64 MB |
| Algorithms | MPI_Bcast, SRDA |
| Broadcast root | Every rank 0..N-1 (results averaged) |
| Host speed | 2000 GFlops |
| **Total** | **153,600 experiments** |

## Results

See [`docs/RESULTS.md`](docs/RESULTS.md) for full tables (mean +/- stdev across all root ranks).

Key findings:
- **Butterfly**: SRDA achieves up to **8.7x** speedup over MPI_Bcast (N=512, 16 MB)
- **2Dmesh**: SRDA overtakes MPI at message sizes above ~4 MB (up to 1.8x)
- **Dragonfly / FatTree**: MPI_Bcast's binomial tree remains faster at all sizes due to hierarchical structure

## Project Structure

```
.
├── src/                        # Core source code
│   ├── runner.c                # Broadcast simulator (MPI, SRDA, pipe)
│   ├── Makefile                # Build (outputs to bin/) and run targets
│   ├── topology_generater.py   # Generate platform XMLs + hostfiles
│   ├── aggregate.py            # Aggregate results across roots
│   ├── plot_topology.py        # Topology visualization
│   ├── sweep.sh                # Parallel experiment dispatch
│   ├── backfill.sh             # Re-run missing experiments
│   └── run_experiments.sh      # Sequential experiment sweep
├── bin/                        # Build outputs (gitignored)
├── topo/                       # SimGrid platform XMLs + hostfiles
├── data/                       # Results — JSON (gitignored)
├── docs/                       # Documentation
│   ├── RESULTS.md              # Tabulated results
│   ├── update.md               # Development notes
│   └── topo_reference.txt      # Network parameter citations
├── slurm/                      # HPC job scripts (gitignored)
├── PoC/                        # Proof-of-concept scripts and tests
│   ├── chain_test.c            # Chain topology test
│   └── test/                   # Star-topology tests & plots
├── bcast.def                   # Singularity definition (reference)
└── LEGACY/                     # Original Python discrete-time models
```

## References

- Casanova, H. et al. "Versatile, Scalable, and Accurate Simulation of Distributed Applications and Platforms." *JPDC*, 74(10), 2014.
- Degomme, A. et al. "Simulating MPI Applications: The SMPI Approach." *IEEE TPDS*, 28(8), 2017.
- Thakur, R. et al. "Optimization of Collective Communication Operations in MPICH." *IJHPCA*, 19(1), 2005.
- Kim, J. and Dally, W. "Flattened Butterfly: A Cost-Efficient Topology for High-Radix Networks." *ISCA*, 2007.
- Kim, J. et al. "Technology-Driven, Highly-Scalable Dragonfly Topology." *ISCA*, 2008.
- Leiserson, C. "Fat-Trees: Universal Networks for Hardware-Efficient Supercomputing." *IEEE ToC*, 1985.