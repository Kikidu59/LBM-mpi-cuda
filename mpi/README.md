# LBM 2D — MPI version

Distributed-memory parallelization of the serial solver in `../serial/`.

## Idea in one paragraph

The global `nx x ny` grid is split among the ranks with a **1D decomposition
along y**: each rank owns a contiguous block of rows. Because rows are
contiguous in the structure-of-arrays layout, the rows we have to exchange
between neighbours are contiguous too. Each rank keeps **one ghost (halo) row**
on each side. Every step does the usual `collide -> bounce_back`, then a
**halo exchange** (`MPI_Sendrecv`, all 9 directions of the boundary rows), then
`stream -> inlet -> outlet`. Streaming is the only step that reads neighbour
cells, which is why it is the only one that needs communication. The remainder
rows (when `ny % nranks != 0`) are spread over the first ranks for balance.

Output strategy: **probe only** (no HDF5 snapshots), so this build has no HDF5
dependency. The rank that owns the probe row writes `probe.csv`.

## Build (on Jed)

```bash
module load gcc openmpi          # adjust names with `module spider openmpi`
make
```

## Run

Locally / interactively:

```bash
mpirun -np 4 ./lbm_mpi nx=800 ny=400 re=100 steps=60000 every=0
```

Through Slurm (a minimal example):

```bash
srun -n 16 ./lbm_mpi nx=800 ny=400 re=100 steps=60000
```

Useful keys are the same as the serial code (`nx, ny, re, u_in, steps,
cyl_*`, `probe_x`, `probe_y`, `probe`). `every` is accepted but ignored
(no snapshots in the MPI build).

## Validation

The result must match the serial baseline:

```bash
python3 ../viz/strouhal.py probe.csv --u-in 0.05 --diameter 20
# expect St ~ 0.16
```

At the end of the run, rank 0 prints the wall time (slowest rank) and the
global **MLUPS** = `nx * ny * steps / time / 1e6`, comparable to the serial
baseline and across rank counts.
