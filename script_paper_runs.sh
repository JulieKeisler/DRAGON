#!/bin/bash
#SBATCH --ntasks=1                        # Number of tasks (1 for single task)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --partition=cpu_homogen                   # CPU


#SBATCH --job-name=nam                    # Job name
#SBATCH --mem=32G                         # Total memory allocated
#SBATCH --output=name_%j.out              # Output file (with job ID)
#SBATCH --error=name_%j.err               # Error file (with job ID)
#SBATCH --time=7-00:00:00                    # Time limit (7 days)

source /scratch/echikhao/dragon-dev/py311_full/bin/activate

export PATH=$HOME/.juliaup/bin:$PATH
export JULIAUP_CHANNEL=1.11
export PYSR_JULIA_PROJECT=/home/echikhao/julia_pysr_env_111
export PYTHON_JULIAPKG_PROJECT=/home/echikhao/julia_pysr_env_111
export JULIA_PROJECT=/home/echikhao/julia_pysr_env_111
export PYTHON_JULIAPKG_EXE=$HOME/.juliaup/bin/julia

cd /scratch/jkeisler/dragon-dev/leaderboard
srun python -u run_dragonsr_batch.py --plan-file run_plan.txt