#!/usr/bin/env python3
"""Generate a star-topology platform XML and hostfile for SimGrid.

node-0 is the hub. node-1 .. node-100 are leaves.
All routes from node-0 to any leaf share node-0's uplink (link-up-0),
so k simultaneous sends from node-0 contend on that single link.

    node-0 --[link-up-0]--> switch --[link-down-i]--> node-i
"""

N_CHILDREN = 100
N_TOTAL = N_CHILDREN + 1
BW = "50GBps"
LAT = "100ns"

# --- Platform XML ---
lines = [
    "<?xml version='1.0'?>",
    '<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">',
    '<platform version="4.1">',
    '  <zone id="star" routing="Full">',
    "",
    f"    <!-- {N_TOTAL} hosts -->",
]

for i in range(N_TOTAL):
    lines.append(f'    <host id="node-{i}" speed="2000Gf"/>')

lines.append("")
lines.append("    <!-- Central switch -->")
lines.append('    <router id="switch"/>')

lines.append("")
lines.append(f"    <!-- Uplink from node-0 to switch -->")
lines.append(f'    <link id="link-up-0" bandwidth="{BW}" latency="{LAT}"/>')

lines.append("")
lines.append(f"    <!-- Downlinks from switch to each leaf -->")
for i in range(1, N_TOTAL):
    lines.append(f'    <link id="link-down-{i}" bandwidth="{BW}" latency="{LAT}"/>')

lines.append("")
lines.append("    <!-- Routes: node-0 <-> node-i go through link-up-0 + link-down-i -->")
for i in range(1, N_TOTAL):
    lines.append(
        f'    <route src="node-0" dst="node-{i}">'
        f' <link_ctn id="link-up-0"/> <link_ctn id="link-down-{i}"/>'
        f" </route>"
    )

lines.append("")
lines.append("    <!-- Routes between leaves: link-down-i + link-down-j (no contention with node-0) -->")
for i in range(1, N_TOTAL):
    for j in range(i + 1, N_TOTAL):
        lines.append(
            f'    <route src="node-{i}" dst="node-{j}">'
            f' <link_ctn id="link-down-{i}"/> <link_ctn id="link-down-{j}"/>'
            f" </route>"
        )

lines.append("")
lines.append("  </zone>")
lines.append("</platform>")

with open("platform_star.xml", "w") as f:
    f.write("\n".join(lines) + "\n")

# --- Hostfile ---
with open("hostfile_101", "w") as f:
    for i in range(N_TOTAL):
        f.write(f"node-{i}\n")

print(f"Generated platform_star.xml ({N_TOTAL} nodes, {N_CHILDREN} children)")
print(f"Generated hostfile_101")
