#!/usr/bin/env python3
"""Convert SimGrid Paje trace to Chrome Trace Event format for Perfetto.

Usage:
    python3 paje_to_perfetto.py input.trace [output.json]

Generate traces with bandwidth data using:
    smpirun ... -trace \\
      --cfg=tracing/filename:out.trace \\
      --cfg=tracing/smpi:yes \\
      --cfg=tracing/smpi/internals:yes \\
      --cfg=tracing/categorized:yes \\
      --cfg=tracing/uncategorized:yes

Then open https://ui.perfetto.dev/ and drag the JSON file onto it.
"""
import json, sys, re
from collections import defaultdict

def convert(trace_path, out_path):
    val_names = {}     # alias -> name
    var_names = {}     # alias -> name (variable types)
    containers = {}    # id(str) -> {"name": ..., "type": ...}
    events = []

    # Parse all container and definition lines
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in ('%', '#'):
                continue
            parts = line.split()
            etype = parts[0]

            # DefineVariableType: 1 <alias> <type> <name> <color>
            if etype == '1' and len(parts) >= 4:
                var_names[parts[1]] = parts[3].strip('"')

            # DefineEntityValue: 5 <alias> <type> <name> <color>
            if etype == '5' and len(parts) >= 4:
                val_names[parts[1]] = parts[3].strip('"')

            # CreateContainer: 6 <time> <alias> <type> <parent> <name>
            if etype == '6' and len(parts) >= 6:
                alias = parts[2]
                m = re.search(r'"([^"]*)"', line)
                cname = m.group(1) if m else parts[5]
                containers[alias] = {"name": cname, "type": parts[3]}

    # Map container IDs to rank numbers and link names
    rank_map = {}   # container_id -> rank number
    link_map = {}   # container_id -> link name
    for cid, info in containers.items():
        name = info["name"]
        if name.startswith("rank-"):
            rank_map[cid] = int(name.split("-")[1])
        elif any(name.startswith(p) for p in
                 ("leaf-", "spine-", "link-", "link_", "loopback")):
            link_map[cid] = name

    # Second pass: extract state events and bandwidth samples
    bw_samples = defaultdict(list)

    with open(trace_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] in ('%', '#'):
                continue
            etype = parts[0]

            # PushState: 12 <time> <type> <container> <value>
            if etype == '12' and len(parts) >= 5:
                cid = parts[3]
                if cid in rank_map:
                    t_us = float(parts[1]) * 1e6
                    name = val_names.get(parts[4], parts[4])
                    events.append({
                        "name": name, "ph": "B", "ts": t_us,
                        "pid": 0, "tid": rank_map[cid]
                    })

            # PopState: 13 <time> <type> <container>
            elif etype == '13' and len(parts) >= 4:
                cid = parts[3]
                if cid in rank_map:
                    t_us = float(parts[1]) * 1e6
                    events.append({
                        "ph": "E", "ts": t_us,
                        "pid": 0, "tid": rank_map[cid]
                    })

            # AddVariable/SubVariable: 9/10 <time> <var_type> <container> <value>
            elif etype in ('9', '10') and len(parts) >= 5:
                vname = var_names.get(parts[2], parts[2])
                if vname == "bandwidth_used":
                    cid = parts[3]
                    if cid in link_map:
                        t = float(parts[1])
                        val = float(parts[4])
                        bw_samples[link_map[cid]].append(
                            (t, val if etype == '9' else -val))

    # Build bandwidth counters per link
    link_pids = {}
    pid_counter = 1

    for link_name in sorted(bw_samples.keys()):
        samples = bw_samples[link_name]
        if not samples:
            continue

        if link_name.startswith("leaf-"):
            group = "Leaf Switches"
        elif link_name.startswith("spine-"):
            group = "Spine Switches"
        elif "leaf" in link_name:
            group = "Node-Leaf Links"
        elif "spine" in link_name:
            group = "Leaf-Spine Links"
        else:
            group = "Network Links"

        link_pids[link_name] = pid_counter
        events.append({
            "name": "process_name", "ph": "M",
            "pid": pid_counter, "tid": 0,
            "args": {"name": group}
        })
        events.append({
            "name": "thread_name", "ph": "M",
            "pid": pid_counter, "tid": 0,
            "args": {"name": link_name}
        })
        pid_counter += 1

        samples.sort()
        running_bw = 0.0
        for t, delta in samples:
            running_bw += delta
            events.append({
                "name": "bandwidth_used", "ph": "C",
                "ts": t * 1e6,
                "pid": link_pids[link_name], "tid": 0,
                "args": {"bytes_per_sec": max(0, running_bw)}
            })

    # Process name for ranks
    events.append({
        "name": "process_name", "ph": "M",
        "pid": 0, "tid": 0,
        "args": {"name": "MPI Ranks"}
    })

    with open(out_path, 'w') as f:
        json.dump({"traceEvents": events}, f)

    n_rank = sum(1 for e in events if e.get("pid") == 0 and e.get("ph") in ("B", "E"))
    n_bw = sum(1 for e in events if e.get("ph") == "C")
    n_links = len(link_pids)
    print(f"{n_rank} rank events + {n_bw} bandwidth samples ({n_links} links) -> {out_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace('.trace', '_perfetto.json')
    convert(inp, out)
