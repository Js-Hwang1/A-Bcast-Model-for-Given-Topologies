CC      = smpicc
CFLAGS  = -O2 -Wall -Wextra
TARGET  = poc

# Common SMPI flags (no computation, no cross-traffic, W=0 already default)
SMPI_FLAGS = --cfg=smpi/simulate-computation:no \
             --cfg=smpi/display-timing:yes \
             --cfg=network/crosstraffic:0 \
             --log=root.thres:warning

MSG = 1048576

.PHONY: all clean run_all

all: $(TARGET)

$(TARGET): poc.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGET)

# ---- Individual experiment targets ----

# Set A: Star topology (4 nodes, 1 router)
star_%: $(TARGET)
	@smpirun -np 4 -platform platforms/star_4.xml -hostfile hostfile_4 \
		$(SMPI_FLAGS) ./$(TARGET) $* $(MSG)

# Set B: Chain topology (4 nodes, direct connect baseline)
chain_%: $(TARGET)
	@smpirun -np 4 -platform platforms/chain_4.xml -hostfile hostfile_4 \
		$(SMPI_FLAGS) ./$(TARGET) $* $(MSG)

# Set C: Dual switch (8 nodes)
dual_%: $(TARGET)
	@smpirun -np 8 -platform platforms/dual_switch_8.xml -hostfile hostfile_8 \
		$(SMPI_FLAGS) ./$(TARGET) $* $(MSG)

# Set D: Fat-tree 2 spines (8 nodes)
ft2_%: $(TARGET)
	@smpirun -np 8 -platform platforms/fattree_2spine_8.xml -hostfile hostfile_8 \
		$(SMPI_FLAGS) ./$(TARGET) $* $(MSG)

# Set E: Fat-tree 1 spine (8 nodes)
ft1_%: $(TARGET)
	@smpirun -np 8 -platform platforms/fattree_1spine_8.xml -hostfile hostfile_8 \
		$(SMPI_FLAGS) ./$(TARGET) $* $(MSG)

# Set F: Narrow uplink (8 nodes, uplink = 6.25 GBps, half of NIC)
narrow_%: $(TARGET)
	@smpirun -np 8 -platform platforms/narrow_uplink_8.xml -hostfile hostfile_8 \
		$(SMPI_FLAGS) ./$(TARGET) $* $(MSG)

# ---- Run all experiments ----
run_all: $(TARGET)
	@echo "=== Router Topology PoC Experiments ==="
	@echo "Message size: $(MSG) bytes (1 MB)"
	@echo ""
	@echo "--- Set A: Star Topology (4 nodes, 1 router) ---"
	@echo "  Route: node -> nic-link -> router -> nic-link -> node (2 hops)"
	@$(MAKE) --no-print-directory star_p2p
	@$(MAKE) --no-print-directory star_pair
	@$(MAKE) --no-print-directory star_fanout3
	@$(MAKE) --no-print-directory star_fanin3
	@echo ""
	@echo "--- Set B: Chain Topology (4 nodes, direct connect) ---"
	@echo "  Baseline: node -> link -> node (1 hop per step)"
	@$(MAKE) --no-print-directory chain_p2p
	@$(MAKE) --no-print-directory chain_pair
	@$(MAKE) --no-print-directory chain_fanout3
	@$(MAKE) --no-print-directory chain_fanin3
	@echo ""
	@echo "--- Set C: Dual Switch (8 nodes, 2 switches, 1 uplink) ---"
	@echo "  Intra: 2 hops | Inter: 3 hops (through uplink)"
	@$(MAKE) --no-print-directory dual_p2p
	@$(MAKE) --no-print-directory dual_pair
	@$(MAKE) --no-print-directory dual_cross1
	@$(MAKE) --no-print-directory dual_cross2
	@$(MAKE) --no-print-directory dual_cross4
	@$(MAKE) --no-print-directory dual_mixed
	@$(MAKE) --no-print-directory dual_fanout3
	@$(MAKE) --no-print-directory dual_fanout_x2
	@$(MAKE) --no-print-directory dual_fanout_x4
	@echo ""
	@echo "--- Set D: Fat-Tree 2 Spines (8 nodes, 2 leaf, 2 spine) ---"
	@echo "  Intra-leaf: 2 hops | Inter-leaf: 4 hops (node->leaf->spine->leaf->node)"
	@$(MAKE) --no-print-directory ft2_p2p
	@$(MAKE) --no-print-directory ft2_cross1
	@$(MAKE) --no-print-directory ft2_cross2
	@$(MAKE) --no-print-directory ft2_cross4
	@echo ""
	@echo "--- Set E: Fat-Tree 1 Spine (control) ---"
	@echo "  Same structure but only 1 spine path"
	@$(MAKE) --no-print-directory ft1_p2p
	@$(MAKE) --no-print-directory ft1_cross1
	@$(MAKE) --no-print-directory ft1_cross2
	@$(MAKE) --no-print-directory ft1_cross4
	@echo ""
	@echo "--- Set F: Narrow Uplink (6.25 GBps uplink, 12.5 GBps NIC) ---"
	@echo "  Uplink = half of NIC bandwidth"
	@$(MAKE) --no-print-directory narrow_p2p
	@$(MAKE) --no-print-directory narrow_cross1
	@$(MAKE) --no-print-directory narrow_cross2
	@$(MAKE) --no-print-directory narrow_cross4
	@$(MAKE) --no-print-directory narrow_mixed
	@echo ""
	@echo "=== Done ==="
