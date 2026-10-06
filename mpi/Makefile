# MPI build. Use the MPI C++ compiler wrapper. On Jed, after loading the
# modules (e.g. `module load gcc openmpi`), `mpicxx` is on the PATH.
# No HDF5 dependency here: the MPI version writes only probe.csv (option 1).

# Force the MPI compiler wrapper. We use ':=' (not '?=') on purpose: loading
# the gcc module sets CXX=g++ in the environment, and '?=' would keep that and
# fail to find mpi.h. ':=' always uses mpicxx.
CXX      := mpicxx
CXXFLAGS += -O3 -Wall -Wextra -std=c++11
# We only use the MPI C API: skip the deprecated C++ bindings, which otherwise
# trigger warnings inside some OpenMPI headers.
CXXFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
LDFLAGS  += -lm

OBJS = main.o lbm.o

all: lbm_mpi

lbm_mpi: $(OBJS)
	$(CXX) -o $@ $(OBJS) $(CXXFLAGS) $(LDFLAGS)

%.o: %.cc lbm.hh
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f lbm_mpi *.o *~

distclean: clean
	rm -f probe.csv

.PHONY: all clean distclean
