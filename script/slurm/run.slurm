#!/bin/bash
#SBATCH --job-name=aeqts_qubo
#SBATCH --output=out_%j.log
#SBATCH --error=err_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --account=MST114318
#SBATCH --gpus-per-node=1
