#!/bin/bash
#SBATCH --job-name=aeqts_qubo_mpi
#SBATCH --output=out_mpi_%j.log
#SBATCH --error=err_mpi_%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --account=GOV109211

module purge

module load gnu14/14.2.0
module load nvhpc/25.9
module load cuda
module load cmake

export CC=gcc
export CXX=g++

export UCX_MEMTYPE_CACHE=n
export UCX_TLS=rc,cuda_copy,cuda_ipc
export OMPI_MCA_pml=ucx

mpirun ./build/aeqts_qubo --items 500 --N 50 --iter 1000
