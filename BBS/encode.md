# BBS Encoding Format

Balanced Broadcast Spanning-trees (BBS) pre-computes edge-disjoint spanning trees offline and encodes per-node routing tables into a compact binary file (`.bbs`). At runtime, each MPI rank reads only its own section of the file, yielding a direct mapping from incoming messages to outgoing sends with zero computation in the broadcast hot path.

## Encoding overview

A `.bbs` file is generated per (topology, root) pair. It encodes `tau` edge-disjoint spanning trees, all rooted at the broadcast root. Each node's section contains, for every tree, the list of children to which it must forward data. The file is structured for O(1) access: an offset table lets any node jump directly to its section without scanning.

### File layout

```
Offset  Size            Field
──────  ──────────────  ─────────────────────────────────
0       2 bytes         magic       = 0xBB50
2       2 bytes         N           (number of nodes, max 1024)
4       1 byte          tau         (number of trees, max 8)
5       1 byte          (reserved)
6       2 bytes         root        (broadcast root rank)
8       N * 4 bytes     offset[i]   (byte offset to node i's section)

        --- node sections (variable length, packed) ---

For node i, trees 0..tau-1 laid out sequentially:
  2 bytes       parent      (int16; -1 if this node is the root)
  2 bytes       nchildren   (uint16)
  nchildren     children[]  (int16 each)
  * 2 bytes
```

All values are little-endian. Total file size for N=1024, tau=8, average fanout ~2 is approximately 70 KB.

### Per-node section

Each node's section is a flat sequence of `tau` tree entries:

```
[tree 0] parent | nchildren | child_0, child_1, ...
[tree 1] parent | nchildren | child_0, child_1, ...
  ...
[tree tau-1] ...
```

This layout allows a single sequential read from the offset pointer: parse `tau` entries by advancing a pointer through (parent, nchildren, children[]) triples. No random access within the section.

## Runtime protocol

### Setup phase (before timing)

1. Each rank opens the `.bbs` file, reads 8 bytes of header.
2. Seeks to `8 + rank * 4`, reads its 4-byte offset.
3. Seeks to that offset, reads its section into a stack-allocated struct:

```c
typedef struct {
    uint8_t  tau;
    int16_t  parent[MAX_TREES];
    uint8_t  nchildren[MAX_TREES];
    int16_t  children[MAX_TREES][MAX_FANOUT];
} bbs_route_t;
```

This struct fits in L1 cache. The file is closed immediately after loading. No heap allocation. No scanning. Three seeks, three reads.

### Broadcast phase (timed)

The message of size S is split into k chunks. Chunk c is assigned to tree `t = c % tau`. MPI tags equal the chunk index.

**Step 1 -- Pre-post all receives.**
For each chunk c where `parent[c % tau] >= 0`, post a non-blocking receive:

```
MPI_Irecv(buf + offset[c], chunk_size, MPI_BYTE,
          MPI_ANY_SOURCE, tag=c, comm, &recv_req[c])
```

Using `MPI_ANY_SOURCE` eliminates the need to encode or look up the parent at dispatch time. The tag alone is unambiguous: chunk c flows through exactly one tree, and only the node's parent in that tree will send it.

**Step 2 -- Pipeline dispatch.**
Process chunks in order. For each chunk c:

```
Wait(recv_req[c])                       // block until chunk arrives
for j = 0 .. nchildren[c % tau] - 1:
    MPI_Isend(buf + offset[c], chunk_size, MPI_BYTE,
              children[c % tau][j], tag=c, comm, &send_req[si++])
```

The root skips the wait (it already owns the data) and fires sends immediately.

**Step 3 -- Drain.**

```
MPI_Waitall(total_sends, send_reqs, STATUSES_IGNORE)
```

### Why this works without contention

Edge-disjoint trees guarantee that no two trees share a physical link. Within a single tree, each edge is a distinct link (trees are acyclic). Therefore, all concurrent sends across all active pipeline stages operate on disjoint links. No bandwidth sharing occurs. Each send completes in exactly `T_1hop(chunk_size)`.

### Tag uniqueness

Each chunk index c is globally unique. A node receives chunk c from its parent in tree `c % tau`. Since the trees are edge-disjoint, parents differ across trees, so (source, tag) pairs are always distinct even without encoding the source explicitly.

## File naming convention

```
{topology_basename}_R{root}.bbs
```

Example: `platform_2dmesh_8x16_R0.bbs` for root 0 on the 8x16 mesh.

Files are stored in the `encodings/` directory, organized by topology and node count:

```
encodings/
  2Dmesh/N=128/
    platform_2dmesh_8x16_R0.bbs
    platform_2dmesh_8x16_R1.bbs
    ...
  Dragonfly/N=128/
    ...
```

## Offline tree computation

The `.bbs` file is produced by an offline encoder that:

1. Loads the topology graph (from `.tdat` or platform XML).
2. Computes `tau` edge-disjoint spanning trees of minimum balanced depth.
3. Writes the `.bbs` file using the format above.

The tree computation is the hard problem (NP-hard in general for minimizing maximum depth across edge-disjoint spanning trees). The encoding and runtime dispatch are trivial by design: all complexity is pushed to the offline phase.
