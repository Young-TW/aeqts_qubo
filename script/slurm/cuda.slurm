#!/bin/bash
#SBATCH --job-name=aeqts_qubo_cuda
#SBATCH --output=out_cuda_%j.log
#SBATCH --error=err_cuda_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --account=MST114318
#SBATCH --gpus-per-node=1

module purge

# Load CUDA module
module load cuda
module load cmake

# Configure for CUDA backend in isolated build directory
cmake -S . -B build -DBACKEND=CUDA

# Build the executable
cmake --build build -j

# Run the executable
srun ./build/aeqts_qubo
