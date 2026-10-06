#!/bin/bash
# ===================================================================
# WEAK SCALING of the MPI LBM on Jed.
# The work PER RANK is kept constant: we add ranks and grow the grid
# by the same factor. With our 1D decomposition along y, "constant work
# per rank" means a constant number of rows per rank, so we set
#   ny = ROWS_PER_RANK * P   (nx and steps fixed).
# Ideal weak scaling = constant run time (efficiency 100%). Compare
# against Gustafson's law in the report.
# ===================================================================
#SBATCH --job-name=lbm_weak
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=weak_%j.out

# --- environment + build -------------------------------------------
module load gcc openmpi
make
# -------------------------------------------------------------------

# --- constant work per rank (weak scaling) -------------------------
NX=1024
ROWS_PER_RANK=128          # each rank always owns 128 rows
STEPS=4000
RANKS="1 2 4 8 16 32 64 128"
# -------------------------------------------------------------------

# --- run the sweep -------------------------------------------------
echo "WEAK SCALING  nx=$NX rows/rank=$ROWS_PER_RANK steps=$STEPS"
for P in $RANKS; do
  NY=$(( ROWS_PER_RANK * P ))     # total rows grow with P
  echo "=== ranks=$P  ny=$NY ==="
  srun --ntasks=$P --cpu-bind=cores \
       ./lbm_mpi nx=$NX ny=$NY re=100 steps=$STEPS \
       probe=probe_weak_p${P}.csv
done
# -------------------------------------------------------------------

# Tip: collect the results with
#   grep -E "ranks=|MLUPS" weak_<jobid>.out
