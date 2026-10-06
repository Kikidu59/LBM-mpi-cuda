#!/bin/bash
# ===================================================================
# VALIDATION of the CUDA LBM on a V100 (Izar). We run the reference
# case (same as the serial baseline) and check that the GPU reproduces
# the Karman shedding frequency St ~ 0.16. This proves the kernels are
# physically correct (a wrong streaming/bounce-back would change or
# destroy the shedding).
# ===================================================================
#SBATCH --job-name=lbm_cuda_validate
#SBATCH --account=math-454
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH --output=cuda_validate_%j.out

# --- environment + build -------------------------------------------
module load gcc cuda
make clean && make
# -------------------------------------------------------------------

# --- reference case (same as the serial validation run) ------------
NX=800
NY=400
STEPS=60000
# -------------------------------------------------------------------

echo "=== CUDA reference run (block=256) ==="
srun ./lbm_cuda nx=$NX ny=$NY re=100 steps=$STEPS block=256 probe=probe_cuda.csv

# --- check the Strouhal number -------------------------------------
echo "--- Strouhal (CUDA) ---"
python3 ../viz/strouhal.py probe_cuda.csv --u-in 0.05 --diameter 20
# must print St ~ 0.16, like the serial and MPI versions
# -------------------------------------------------------------------
