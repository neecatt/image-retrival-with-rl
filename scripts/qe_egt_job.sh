#!/bin/bash -l
#SBATCH --job-name=qe_egt
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/qe_egt-%j.out
#SBATCH --error=logs/qe_egt-%j.err
#SBATCH --export=NONE
# EGT-style traversal baseline under the Track V-QE budget, all six descriptors (scripts/analysis/qe_egt_baseline.py).
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)"
Q=/home/vault/b268dc/b268dc21/runs/trackqe
python scripts/analysis/qe_egt_baseline.py --qe_pre $Q/dinov2wise $Q/newresnet_seed42 $Q/newresnet_seed1 $Q/newresnet_seed2 \
  $Q/newresnet_seed3 $Q/newresnet_seed4 --out results/qe/egt_baseline.json
