#!/bin/bash -l
#SBATCH --job-name=eval_baselines
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=01:15:00
#SBATCH --output=logs/eval_baselines-%j.out
#SBATCH --error=logs/eval_baselines-%j.err
#SBATCH --export=NONE
# Reference numbers under the corrected protocol: off-the-shelf ImageNet R50-GeM, Stage 1, Stage 2 (epoch 5).
# SfM-120k val mAP (selection metric) + ROxford5k/RParis6k official mAP at short/long-side 1024.
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1   # cached DINOv2 weights, no network on nodes
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
OUT="results/eval_baselines_${SLURM_JOB_ID}"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_rev.txt"; git diff > "$OUT/git_diff.patch"
python scripts/stage_datasets.py sfm120k roxford5k rparis6k
python scripts/eval_baselines.py --data_root "$TMPDIR/data/raw" --out "$OUT/metrics.json" \
    --models imagenet checkpoints/stage1_epoch_13_best.pth checkpoints/stage2_epoch_5.pth
