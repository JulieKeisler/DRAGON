#!/bin/bash
#SBATCH --job-name=libval
#SBATCH --partition=cpu_homogen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/jkeisler/dragonsr_lib/logs/libval_%A_%a.out
#SBATCH --error=/scratch/jkeisler/dragonsr_lib/logs/libval_%A_%a.err
# Library validation: 6 targets x 5 runs at eps = 0; task t -> target t // 5, run_id t % 5.
TARGETS=(ndvi awei_sh evi2 hubble n4 n11)
T=${TARGETS[$((SLURM_ARRAY_TASK_ID / 5))]}
R=$((SLURM_ARRAY_TASK_ID % 5))
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
export TMPDIR=/scratch/jkeisler/dragonsr_lib/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $TMPDIR
cd /scratch/jkeisler/dragonsr_lib
python run_lib.py $T $R /scratch/jkeisler/dragonsr_lib/results
rm -rf $TMPDIR
