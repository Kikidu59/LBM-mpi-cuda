# CUDA build for the LBM solver. Compiled with nvcc for the V100 (sm_70).
# Load the cluster modules first, e.g. on Izar:
#   module load gcc cuda
#
# Memory-layout study: the default build uses the SoA layout. Build the AoS
# variant with `make LAYOUT=AOS` (it just defines -DLAYOUT_AOS).

NVCC   ?= nvcc

LAYOUT ?= SOA
ifeq ($(LAYOUT),AOS)
  LAYOUT_FLAG := -DLAYOUT_AOS
endif

# -arch=sm_70 targets the V100. We use C++14 (supported by nvcc) and -O3.
NVCCFLAGS += -O3 -std=c++14 -arch=sm_70 $(LAYOUT_FLAG)

OBJS = main.o lbm.o

all: lbm_cuda

lbm_cuda: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

main.o: main.cc lbm.hh
	$(NVCC) $(NVCCFLAGS) -c -o $@ main.cc

lbm.o: lbm.cu lbm.hh
	$(NVCC) $(NVCCFLAGS) -c -o $@ lbm.cu

clean:
	rm -f lbm_cuda *.o *~

distclean: clean
	rm -f probe.csv

.PHONY: all clean distclean
