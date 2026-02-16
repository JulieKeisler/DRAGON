#!/bin/bash
#SBATCH --ntasks=1                        # Number of tasks (1 for single task)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --partition=gpu                   # GPU


#SBATCH --job-name=nam                    # Job name
#SBATCH --mem=25G                         # Total memory allocated
#SBATCH --output=name_%j.out              # Output file (with job ID)
#SBATCH --error=name_%j.err               # Error file (with job ID)
#SBATCH --time=1:00:00                    # Time limit (40 minutes)

srun python -u run_dragon_sr.py