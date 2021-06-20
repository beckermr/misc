#!/bin/bash

#SBATCH --job-name=star-mask-truth
#SBATCH --account=metashear
#SBATCH --partition=bdws
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=log_star-mask-truth-%j.oe
#SBATCH --time=1:00:00

source ~/.bashrc
conda activate bebop

python run.py 5000
