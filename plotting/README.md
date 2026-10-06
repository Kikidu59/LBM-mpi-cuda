# Plotting

Clean result data (CSV) + matplotlib scripts that turn it into the report/slide
figures. To redo a figure: edit its CSV in `data/`, re-run its script. No number
is hard-coded in a script — every value is read from a CSV (except the GPU
roofline constant, which lives once in `style.py`).

## Layout

```
plotting/
  data/        clean CSVs, one per study
  figures/     output PDFs (vector, for slides/report)
  style.py     shared figure size, fonts, colors, roofline, save helper
  plot_*.py    one script per figure
```

## Regenerate everything

```bash
cd plotting
pip install matplotlib pandas        # once
for s in plot_serial_profile plot_mpi_strong plot_mpi_weak \
         plot_cuda_layout plot_cuda_block plot_cuda_size; do
    python3 $s.py
done
```

Each script prints the path of the PDF it writes into `figures/`.

## Figures

| Script | CSV | Output | What it shows |
|--------|-----|--------|---------------|
| `plot_serial_profile.py` | `serial_profile.csv` | `serial_profile.pdf` | % self-time per function; collide+stream = 95.5% |
| `plot_mpi_strong.py` | `mpi_strong.csv` | `mpi_strong_efficiency.pdf`, `mpi_strong_speedup.pdf` | strong scaling: efficiency E(P) (crossover) + speedup vs ideal/Amdahl |
| `plot_mpi_weak.py` | `mpi_weak.csv` | `mpi_weak.pdf` | weak efficiency 100% -> 39% (1 -> 128 ranks) |
| `plot_cuda_layout.py` | `cuda_layout.csv` | `cuda_layout.pdf` | SoA vs AoS bars, ~3.6x (coalescing), roofline line |
| `plot_cuda_block.py` | `cuda_block.csv` | `cuda_block.pdf` | block-size sweep, plateau 128-512, roofline line |
| `plot_cuda_size.py` | `cuda_size.csv` | `cuda_size.pdf` | problem-size sweep, ~81% roofline at 4096^2 |

`plot_mpi_strong.py` computes speedup = MLUPS(P)/MLUPS(1) and efficiency =
speedup/P from the raw MLUPS, so the CSV only stores the measured throughput.

## Source of the numbers

Every CSV holds measurements from the SCITAS clusters, produced by the Slurm
scripts of each version:

| CSV | Produced by | Machine |
|-----|-------------|---------|
| `serial_profile.csv` | `gprof` flat profile of `../serial/lbm` (reference case) | 1 CPU core |
| `mpi_strong.csv` | `../mpi/slurm_strong.sh` | Jed, 2 nodes × 72 cores |
| `mpi_weak.csv` | `../mpi/slurm_weak.sh` (128 rows per rank) | Jed, up to 2 nodes |
| `cuda_block.csv`, `cuda_size.csv`, `cuda_layout.csv` | `../cuda/slurm_study.sh` | Izar, 1 × V100 |

The roofline ceiling (~3100 MLUPS) is the V100 memory bandwidth (900 GB/s)
divided by the 288 bytes moved per lattice update (9 populations × 8 bytes,
read + write, for collide and stream), as derived in the report.
