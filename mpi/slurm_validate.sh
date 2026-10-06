#!/bin/bash
# ===================================================================
# VALIDATION of the MPI LBM. The physics must not depend on the number
# of ranks: we run the reference case with 1 rank and with 16 ranks and
# check that both give the Karman shedding frequency St ~ 0.16, like the
# serial baseline. This is the run that proves the halo exchange is
# correct (a wrong halo would change or destroy the shedding).
# ===================================================================
#SBATCH --job-name=lbm_validate
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH --output=validate_%j.out

# --- environment + build -------------------------------------------
module load gcc openmpi
make
# -------------------------------------------------------------------

# --- reference case (same as the serial validation run) ------------
NX=800
NY=400
STEPS=60000
# -------------------------------------------------------------------

# --- 1 rank --------------------------------------------------------
echo "=== 1 rank ==="
srun --ntasks=1 ./lbm_mpi nx=$NX ny=$NY re=100 steps=$STEPS probe=probe_p1.csv
# -------------------------------------------------------------------

# --- 16 ranks ------------------------------------------------------
echo "=== 16 ranks ==="
srun --ntasks=16 --cpu-bind=cores ./lbm_mpi nx=$NX ny=$NY re=100 steps=$STEPS probe=probe_p16.csv
# -------------------------------------------------------------------

# --- check both Strouhal numbers -----------------------------------
echo "--- Strouhal, 1 rank ---"
python3 ../viz/strouhal.py probe_p1.csv  --u-in 0.05 --diameter 20
echo "--- Strouhal, 16 ranks ---"
python3 ../viz/strouhal.py probe_p16.csv --u-in 0.05 --diameter 20
# both must print St ~ 0.16
# -------------------------------------------------------------------
