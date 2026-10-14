#!/bin/bash -l
#SBATCH --job-name=eval_models
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_models-%j.out
#SBATCH --error=logs/eval_models-%j.err
#SBATCH --export=NONE
# Evaluate model specs (checkpoints, 'imagenet', 'pretrained:<arch>') on a held-out val set + ROxford/RParis.
# Usage: sbatch scripts/eval_models_job.sh OUT_NAME VAL_SET VAL_SIZE MODEL [MODEL ...]
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
NAME="${1:?OUT_NAME}"; VAL="${2:?VAL_SET}"; VSIZE="${3:?VAL_SIZE}"; shift 3
OUT="results/eval/$NAME"; mkdir -p "$OUT"
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) $*" >> "$OUT/jobs.txt"
python scripts/stage_datasets.py "$VAL" roxford5k rparis6k
python scripts/eval_baselines.py --data_root "$TMPDIR/data/raw" --out "$OUT/metrics.json" \
    --val_set "$VAL" --val_size "$VSIZE" --resize long --models "$@"
