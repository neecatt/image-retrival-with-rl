#!/bin/bash -l
#SBATCH --job-name=smoke_train
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/smoke_train-%j.out
#SBATCH --error=logs/smoke_train-%j.err
#SBATCH --export=NONE
# Smoke test (NOT a research result): Stage 2 RL from the Stage 1 checkpoint for 50 batches,
# then full SfM-120k validation mAP for checkpoint selection. Outputs go to TMPDIR, summaries copied back.
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
OUT="results/smoke_train_${SLURM_JOB_ID}"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_rev.txt"; git diff > "$OUT/git_diff.patch"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python scripts/stage_datasets.py sfm120k
python scripts/make_run_config.py config/config.yaml "$OUT/run_config.yaml" \
    dataset.root_dir="$TMPDIR/data/raw" \
    system.checkpoint_dir="$TMPDIR/ckpt" system.log_dir="$TMPDIR/logs" system.results_dir="$TMPDIR/results" \
    system.num_workers=8 \
    training.stage2.max_batches_per_epoch=50

python train.py --config "$OUT/run_config.yaml" --stage2_only \
    --resume checkpoints/stage1_epoch_13_best.pth --stage2_epochs 1

cp -r "$TMPDIR/results" "$OUT/" 2>/dev/null || true
ls -la "$TMPDIR/ckpt"/* || true
