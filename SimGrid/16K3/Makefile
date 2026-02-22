CC       = smpicc
CFLAGS   = -O2 -Wall -Wextra
LDFLAGS  = -lm

PLATFORM = platform_16K3.xml
HOSTFILE = hostfile
NP       = 16
MSG      = 1048576
CHUNKS   = 64

TARGET   = bcast_16K3

.PHONY: all clean run-mpi run-greedy run-compare

all: $(TARGET)

$(TARGET): bcast_16K3.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ---- Quick-run targets ----

run-mpi: $(TARGET)
	smpirun -np $(NP) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		./$(TARGET) mpi $(MSG)

run-greedy: $(TARGET)
	smpirun -np $(NP) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		./$(TARGET) greedy $(MSG) $(CHUNKS)

# ---- Head-to-head: Greedy vs MPI_Bcast ----

run-compare: $(TARGET)
	@echo "=== MPI_Bcast (baseline) ==="
	@smpirun -np $(NP) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		./$(TARGET) mpi $(MSG)
	@echo ""
	@echo "=== Greedy pipelined (full-duplex) ==="
	@smpirun -np $(NP) -platform $(PLATFORM) -hostfile $(HOSTFILE) \
		./$(TARGET) greedy $(MSG) $(CHUNKS)

clean:
	rm -f $(TARGET)
