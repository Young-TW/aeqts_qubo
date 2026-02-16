#!/bin/bash
#SBATCH --job-name=aeqts_qubo_hip
#SBATCH --output=out_hip_%j.log
#SBATCH --error=err_hip_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --account=MST114318
#SBATCH --gpus-per-node=1

module purge

# Load ROCm module
module load rocm
module load cmake

# Print AMD GPU model
rocm-smi --showproductname

# Configure for HIP backend in isolated build directory
cmake -S . -B build -DBACKEND=HIP

# Build the executable
cmake --build build -j

# Run the executable
srun ./build/aeqts_qubo
