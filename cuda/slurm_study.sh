#!/bin/bash
# ===================================================================
# CUDA PERFORMANCE STUDY on a V100 (Izar). Three sweeps, all measuring
# throughput (MLUPS), so we use a short run (steps=4000) and turn the
# physics off-target (the numbers are meaningless physically, we only
# measure memory traffic / throughput, exactly like the MPI scaling).
#
#   1. block size : fixed 1024x1024, block in {32,64,128,256,512,1024}
#   2. problem size: best block, size in {512,1024,2048,4096}^2
#   3. memory layout: SoA vs AoS at 1024x1024 (rebuild with LAYOUT=AOS)
#
# The achieved MLUPS is then compared to the V100 roofline (peak memory
# bandwidth ~900 GB/s) in the report, since LBM is memory-bound.
# ===================================================================
#SBATCH --job-name=lbm_cuda_study
#SBATCH --account=math-454
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=cuda_study_%j.out

module load gcc cuda

STEPS=4000

# ------------------------------------------------------------------
# 1. Block-size sweep (SoA, 1024x1024)
# ------------------------------------------------------------------
echo "######## 1. BLOCK-SIZE SWEEP (SoA, 1024x1024) ########"
make clean && make LAYOUT=SOA
for B in 32 64 128 256 512 1024; do
  echo "==== block=$B ===="
  srun ./lbm_cuda nx=1024 ny=1024 re=100 steps=$STEPS block=$B probe=/dev/null \
    | grep -E "MLUPS|block size"
done

# ------------------------------------------------------------------
# 2. Problem-size sweep (SoA, block=256)
# ------------------------------------------------------------------
echo "######## 2. PROBLEM-SIZE SWEEP (SoA, block=256) ########"
for S in 512 1024 2048 4096; do
  echo "==== size=${S}x${S} ===="
  srun ./lbm_cuda nx=$S ny=$S re=100 steps=$STEPS block=256 probe=/dev/null \
    | grep -E "MLUPS|grid"
done

# ------------------------------------------------------------------
# 3. Memory-layout comparison: SoA vs AoS (1024x1024, block=256)
# ------------------------------------------------------------------
echo "######## 3. LAYOUT SoA vs AoS (1024x1024, block=256) ########"
echo "---- SoA ----"
make clean && make LAYOUT=SOA
srun ./lbm_cuda nx=1024 ny=1024 re=100 steps=$STEPS block=256 probe=/dev/null \
  | grep -E "MLUPS"
echo "---- AoS ----"
make clean && make LAYOUT=AOS
srun ./lbm_cuda nx=1024 ny=1024 re=100 steps=$STEPS block=256 probe=/dev/null \
  | grep -E "MLUPS"

# restore the default SoA build
make clean && make LAYOUT=SOA
