# Running Experiments

## Build

```
smpicc -O2 -o src/runner src/runner.c -lm
```

## Run a single experiment

```
smpirun -np <N> -platform <platform.xml> -hostfile <hostfile> \
  --cfg=smpi/simulate-computation:no \
  src/runner <algo> <msg_bytes> <nchunks> <root> <out_json> [topo_data]
```

**Arguments:**
- `algo`: `mpi | srda | pipe | bine | glf | ffgb | test`
- `msg_bytes`: message size in bytes
- `nchunks`: pipeline depth (ignored by `test`, which computes its own)
- `root`: broadcast root rank, or `all` for every root
- `out_json`: output JSON path, or `_` to skip
- `topo_data`: `.tdat` file (required for `test`, `ffgb`, `glf`)

## Examples

### FatTree N=128, 1MB, root=0
```
smpirun -np 128 -platform topo/FatTree/platform_fattree_128.xml \
  -hostfile topo/FatTree/hostfile_128 \
  --cfg=smpi/simulate-computation:no \
  src/runner test 1048576 64 0 _ topo/FatTree/platform_fattree_128.tdat
```

### Butterfly
```
smpirun -np 128 -platform topo/Butterfly/platform_butterfly_128.xml \
  -hostfile topo/Butterfly/hostfile_128 \
  --cfg=smpi/simulate-computation:no \
  src/runner test 1048576 64 0 _ topo/Butterfly/platform_butterfly_128.tdat
```

### Dragonfly
```
smpirun -np 128 -platform topo/Dragonfly/platform_dragonfly_128.xml \
  -hostfile topo/Dragonfly/hostfile_128 \
  --cfg=smpi/simulate-computation:no \
  src/runner test 1048576 64 0 _ topo/Dragonfly/platform_dragonfly_128.tdat
```

### 2D Mesh
```
smpirun -np 128 -platform topo/2Dmesh/platform_2dmesh_8x16.xml \
  -hostfile topo/2Dmesh/hostfile_128 \
  --cfg=smpi/simulate-computation:no \
  src/runner test 1048576 64 0 _ topo/2Dmesh/platform_2dmesh_8x16.tdat
```

## Batch sweep

```
cd src
bash sweep.sh --algos test --sizes 128 --roots 0
```

See `sweep.sh --help` for options (`--topos`, `--msgs`, `--roots all`, `-j 8`, etc.).

## Generating trace profiles

### Step 1: Generate Paje trace

Add these flags to any `smpirun` command.

**MPI operations only** (smaller trace):
```
-trace \
--cfg=tracing/filename:output.trace \
--cfg=tracing/smpi:yes \
--cfg=tracing/smpi/internals:yes
```

**With network bandwidth saturation** (includes per-link bandwidth_used):
```
-trace \
--cfg=tracing/filename:output.trace \
--cfg=tracing/smpi:yes \
--cfg=tracing/smpi/internals:yes \
--cfg=tracing/categorized:yes \
--cfg=tracing/uncategorized:yes
```

Full example (with bandwidth):
```
smpirun -np 128 -platform topo/FatTree/platform_fattree_128.xml \
  -hostfile topo/FatTree/hostfile_128 \
  --cfg=smpi/simulate-computation:no \
  -trace \
  --cfg=tracing/filename:Analysis/fattree_test_1MB.trace \
  --cfg=tracing/smpi:yes \
  --cfg=tracing/smpi/internals:yes \
  --cfg=tracing/categorized:yes \
  --cfg=tracing/uncategorized:yes \
  src/runner test 1048576 64 0 _ topo/FatTree/platform_fattree_128.tdat
```

### Step 2: Convert to Perfetto format

```
python3 src/paje_to_perfetto.py Analysis/fattree_test_1MB.trace
```

This outputs `Analysis/fattree_test_1MB_perfetto.json`.

### Step 3: View

Open **https://ui.perfetto.dev/** and drag the JSON file onto it.

What you'll see:
- **MPI Ranks** (one row per rank) — colored blocks for `PMPI_Recv`, `PMPI_Isend`, `PMPI_Barrier`, `PMPI_Reduce`, etc. with exact timestamps
- **Node-Leaf Links** — bandwidth_used over time from each node to its leaf switch
- **Leaf-Spine Links** — bandwidth_used on uplinks between leaf and spine switches

The bandwidth counters show which links are saturated and when. The `PMPI_Reduce` at the end of rank 0 is from `main()` collecting timing results — it is not part of the broadcast itself.
