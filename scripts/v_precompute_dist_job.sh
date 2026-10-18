#!/bin/bash -l
#SBATCH --job-name=v_pre_dist
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/v_pre_dist-%j.out
#SBATCH --error=logs/v_pre_dist-%j.err
#SBATCH --export=NONE
# Track V-D: GLDv2 val shortlists over val pool + GLDv2 distractors.
# Usage: sbatch scripts/v_precompute_dist_job.sh MODEL PLAIN_PRE_DIR OUT_DIR TAG
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
M="${1:?MODEL}"; P="${2:?PLAIN_PRE}"; O="${3:?OUT}"; TAG="${4:?TAG}"
[ -e "$M" ] || M=$(ls -t $M | head -1)
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) model=$M"
python scripts/stage_datasets.py gldv2c gldv2d
python scripts/v_precompute_dist.py --data_root "$TMPDIR/data/raw" --model "$M" --plain_pre "$P" --out "$O" --tag "$TAG" 2>&1 | grep -v "it/s\]\|s/it\]"
