#!/bin/bash

#SBATCH --account=def-wperciva
#SBATCH --job-name=HMC_bm
#SBATCH --output=HMC_bm.out
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --mail-user=a2crespi@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load julia/1.10.0

srun julia --project=/home/acrespi/scratch/Flinch -t 32 /home/acrespi/scratch/Flinch/src/HMC_sampler.jl
