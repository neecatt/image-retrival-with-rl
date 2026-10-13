#!/bin/bash -l
#SBATCH --job-name=f0_noise
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/f0_noise-%j.out
#SBATCH --error=logs/f0_noise-%j.err
#SBATCH --export=NONE
# Option 2 feasibility (F0): label noise of full GLDv2 vs GLDv2-clean, measured with the verifier.
# Reads single images from the raw train tars on HPCVAULT by offset (no extraction).
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)"
python scripts/gld_noise_check.py --workers "${SLURM_CPUS_ON_NODE:?}" "$@"
