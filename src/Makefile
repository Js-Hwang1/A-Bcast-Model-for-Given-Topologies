CC       = smpicc
CFLAGS   = -O2 -Wall -Wextra
LDFLAGS  = -lm

TOPO_DIR  = ../topo
PLAN_DIR  = ../plans
RES_DIR   = ../results
OUTDIR    = ../data

# ---- Defaults (override on command line: make run TOPO=FatTree N=256 ...) ----
TOPO      = Butterfly
N         = 128
ALGO      = mpi
MSG       = 1048576
CHUNKS    = 64
ROOT      = 0

TARGET    = runner
OUTFILE   = $(OUTDIR)/$(TOPO)/$(ALGO)/N$(N)_MSG$(MSG)_R$(ROOT).json
PLAN_FILE = _
TRACE     =

.PHONY: all clean gen gen-all run sweep

# ---- Build ----
all: $(TARGET)

$(TARGET): runner.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ---- Generate topologies ----
gen-all:
	python3 topology_generater.py --all

gen:
	python3 topology_generater.py $(TOPO) $(N)

# ---- Resolve platform/hostfile paths ----
# 2Dmesh names its XML with PxQ dimensions, others use N directly
MESH_DIMS = $(shell \
	if [ "$(N)" = "128" ];  then echo 8x16;  \
	elif [ "$(N)" = "256" ];  then echo 16x16; \
	elif [ "$(N)" = "512" ];  then echo 16x32; \
	elif [ "$(N)" = "1024" ]; then echo 32x32; \
	fi)

PLATFORM = $(shell \
	if [ "$(TOPO)" = "2Dmesh" ]; then \
		echo "$(TOPO_DIR)/2Dmesh/platform_2dmesh_$(MESH_DIMS).xml"; \
	elif [ "$(TOPO)" = "Butterfly" ]; then \
		echo "$(TOPO_DIR)/Butterfly/platform_butterfly_$(N).xml"; \
	elif [ "$(TOPO)" = "Dragonfly" ]; then \
		echo "$(TOPO_DIR)/Dragonfly/platform_dragonfly_$(N).xml"; \
	elif [ "$(TOPO)" = "FatTree" ]; then \
		echo "$(TOPO_DIR)/FatTree/platform_fattree_$(N).xml"; \
	fi)

HOSTFILE = $(TOPO_DIR)/$(TOPO)/hostfile_$(N)

# ---- Optional Paje tracing (make run TRACE=1) ----
ifdef TRACE
TRACE_FLAGS = -trace \
  --cfg=tracing/filename:$(OUTDIR)/$(TOPO)/$(ALGO)/N$(N)_MSG$(MSG)_R$(ROOT).trace \
  --cfg=tracing/smpi:yes --cfg=tracing/smpi/internals:yes
endif

# ---- Run ----
run: $(TARGET)
	@mkdir -p $(dir $(OUTFILE))
	smpirun -np $(N) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		--cfg=smpi/host-speed:2000Gf --cfg=smpi/display-timing:yes \
		--log=root.thres:warning $(TRACE_FLAGS) \
		./$(TARGET) $(ALGO) $(MSG) $(CHUNKS) $(ROOT) $(PLAN_FILE) $(OUTFILE)

# ---- Sweep: parallel dispatch of all experiments ----
sweep: $(TARGET)
	./sweep.sh -j $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ---- Clean ----
clean:
	rm -f $(TARGET)
