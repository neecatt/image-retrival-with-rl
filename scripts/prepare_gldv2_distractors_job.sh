#!/bin/bash -l
#SBATCH --job-name=prep_gldv2d
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/prep_gldv2d-%j.out
#SBATCH --error=logs/prep_gldv2d-%j.err
#SBATCH --export=NONE
# Track V-D: pack the GLDv2 distractor set into $WORK/datasets/gldv2d (CPU work; Alex is GPU-only).
# Usage: sbatch scripts/prepare_gldv2_distractors_job.sh [args for prepare_gldv2_distractors.py]
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
NCPU="${SLURM_CPUS_ON_NODE:?}"   # nproc would honour OMP_NUM_THREADS=1
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) cpus=$NCPU"
python scripts/prepare_gldv2_distractors.py --workers "$NCPU" "$@"
