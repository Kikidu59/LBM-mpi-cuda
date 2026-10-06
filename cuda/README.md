# LBM 2D — CUDA version

Single-GPU (V100) port of the serial solver in `../serial/`. The algorithm is
unchanged; only the data and the time loop move to the GPU.

Design (one thread per cell):
- **One kernel per phase** (`collide`, `bounce_back`, `stream`, `inlet`,
  `outlet`), launched in order in the default stream. Each kernel is the GPU
  version of the matching serial method.
- **SoA layout** (`f[i*N + k]`) by default, so neighbouring threads touch
  neighbouring addresses → coalesced memory access. The build flag
  `LAYOUT=AOS` switches to array-of-structures (`f[k*Q + i]`) for the
  memory-layout study.
- **Double buffering** for streaming: two device buffers `d_f`/`d_ftmp`, the
  pointers are swapped after `stream` (no data copy).
- **Host ↔ device transfers only at the start** (initial condition + solid
  mask) **and once at the end** (the probe time series). The hot loop never
  copies anything.
- The probe `(ux,uy)` is accumulated in a device buffer (one tiny kernel per
  step) and copied back once after the loop, so it does not pollute the timing.

## Build (on Izar)
```bash
module load gcc cuda          # adjust names with `module spider`
make                          # SoA build  -> ./lbm_cuda
make LAYOUT=AOS               # AoS build (for the layout study)
```

## Run
```bash
# reference physics run (validates St ~ 0.16)
srun ./lbm_cuda nx=800 ny=400 re=100 steps=60000 block=256 probe=probe.csv
python3 ../viz/strouhal.py probe.csv --u-in 0.05 --diameter 20

# throughput run (sweep the block size)
srun ./lbm_cuda nx=1024 ny=1024 re=100 steps=4000 block=128
```

Key CLI options: `nx, ny, re, u_in, steps, block` (threads per block),
`probe` (CSV path), `cyl_x/cyl_y/cyl_r`, optional `cyl2_*`.

## Slurm scripts
- `slurm_validate.sh` — reference run + Strouhal check.
- `slurm_study.sh` — block-size sweep, problem-size sweep, SoA vs AoS.

## Output
Probe only (no HDF5), like the MPI version. The vorticity GIF stays the serial
one. The CUDA version is validated through the Strouhal number.
