#!/bin/bash

#SBATCH --account=rrg-wperciva
#SBATCH --job-name=multichain_analysis
#SBATCH --output=multichain_analysis.out
#SBATCH --time=2:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --mail-user=a2crespi@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

module load julia/1.10.0

srun julia --project=/home/acrespi/scratch/Flinch -t 32 /home/acrespi/scratch/Flinch/src/chain_analysis/multichain_analysis.jl
