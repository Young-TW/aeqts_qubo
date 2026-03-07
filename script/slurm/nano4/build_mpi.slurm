#!/bin/bash
#SBATCH --job-name=aeqts_qubo_mpi
#SBATCH --output=out_mpi_%j.log
#SBATCH --error=err_mpi_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --partition=dev
#SBATCH --account=GOV109211

module purge

module load gnu14/14.2.0
module load nvhpc/25.9
module load cuda
module load cmake

rm -rf build

export CC=gcc
export CXX=g++

export UCX_MEMTYPE_CACHE=n
export UCX_TLS=rc,cuda_copy,cuda_ipc
export OMPI_MCA_pml=ucx

cmake -S . -B build -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release

cmake --build build -j
