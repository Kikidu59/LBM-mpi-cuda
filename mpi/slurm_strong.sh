#!/bin/bash
# ===================================================================
# STRONG SCALING of the MPI LBM on Jed.
# Fixed problem size, increasing number of MPI ranks. We measure how
# the run time goes down (speedup) as we add ranks. Compare against
# Amdahl's prediction in the report.
#
# One allocation of 2 full nodes; inside it we launch the same problem
# with a growing number of ranks via successive `srun -n $P`.
# ===================================================================
#SBATCH --job-name=lbm_strong
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=strong_%j.out

# --- environment + build -------------------------------------------
module load gcc openmpi
make
# -------------------------------------------------------------------

# --- several representative problem sizes (strong scaling) ---------
# For each fixed size we sweep the ranks.
# steps is reduced for the largest size to keep the total time reasonable;
# that is fine because we measure throughput (MLUPS), not physics here.
SIZES="512 1024 2048"
RANKS="1 2 4 8 16 32 64 128"
steps_for() {            # steps to use for a given square size N
  case "$1" in
    2048) echo 2000 ;;
    *)    echo 4000 ;;
  esac
}
# -------------------------------------------------------------------

# --- run the sweep -------------------------------------------------
for N in $SIZES; do
  STEPS=$(steps_for $N)
  echo "##### STRONG SCALING  size=${N}x${N}  steps=$STEPS #####"
  for P in $RANKS; do
    echo "=== size=$N ranks=$P ==="
    srun --ntasks=$P --cpu-bind=cores \
         ./lbm_mpi nx=$N ny=$N re=100 steps=$STEPS \
         probe=probe_strong_n${N}_p${P}.csv
  done
done
# -------------------------------------------------------------------

# Tip: collect the results with
#   grep -E "size=.* ranks=|MLUPS" strong_<jobid>.out
