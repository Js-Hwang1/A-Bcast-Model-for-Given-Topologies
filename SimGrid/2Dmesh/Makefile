CC       = smpicc
CFLAGS   = -O2 -Wall -Wextra -DROWS=$(ROWS) -DCOLS=$(COLS)
LDFLAGS  = -lm

# ---- Mesh dimensions (change here or override: make ROWS=8 COLS=8) ----
ROWS     = 4
COLS     = 4

NP       = $(shell expr $(ROWS) \* $(COLS))
PLATFORM = platform_2dmesh_$(ROWS)x$(COLS).xml
HOSTFILE = hostfile
MSG      = 1048576
CHUNKS   = 64
ROOT     = 0

TARGET   = bcast_2dmesh

.PHONY: all clean gen run-mpi run-greedy run-bbs run-tree run-compare

all: $(TARGET)

$(TARGET): bcast_2dmesh.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ---- Generate platform XML + hostfile for current ROWS x COLS ----
gen:
	python3 gen_platform.py $(ROWS) $(COLS)

# ---- Quick-run targets ----

run-mpi: $(TARGET)
	smpirun -np $(NP) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		./$(TARGET) mpi $(MSG) 0 $(ROOT)

run-greedy: $(TARGET)
	smpirun -np $(NP) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		./$(TARGET) greedy $(MSG) $(CHUNKS) $(ROOT)

run-bbs: $(TARGET)
	smpirun -np $(NP) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		./$(TARGET) bbs $(MSG) $(CHUNKS) $(ROOT)

run-tree: $(TARGET)
	smpirun -np $(NP) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		./$(TARGET) tree $(MSG) $(CHUNKS) $(ROOT)

# ---- Head-to-head comparison ----

run-compare: $(TARGET)
	@echo "=== MPI_Bcast (root=$(ROOT)) ==="
	@smpirun -np $(NP) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		./$(TARGET) mpi $(MSG) 0 $(ROOT)
	@echo ""
	@echo "=== Tree Pipeline (root=$(ROOT)) ==="
	@smpirun -np $(NP) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		./$(TARGET) tree $(MSG) $(CHUNKS) $(ROOT)
	@echo ""
	@echo "=== Greedy (root=$(ROOT)) ==="
	@smpirun -np $(NP) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		./$(TARGET) greedy $(MSG) $(CHUNKS) $(ROOT)

clean:
	rm -f $(TARGET)
