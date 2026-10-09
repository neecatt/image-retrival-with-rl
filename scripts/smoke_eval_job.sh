#!/bin/bash -l
#SBATCH --job-name=smoke_eval
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=00:25:00
#SBATCH --output=logs/smoke_eval-%j.out
#SBATCH --error=logs/smoke_eval-%j.err
#SBATCH --export=NONE
# Pipeline smoke test (NOT a research result): stage ROxford5k from packed archives into TMPDIR,
# then run the unmodified evaluate.py on the Stage 1 checkpoint. Validates env, GPU, data, loader, checkpoint.
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
module list 2>&1
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda avail", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
git rev-parse HEAD; git status --short | head
python scripts/stage_datasets.py roxford5k --config-in config/config.yaml --config-out "$TMPDIR/run_config.yaml"
python evaluate.py --checkpoint checkpoints/stage1_epoch_13_best.pth --config "$TMPDIR/run_config.yaml" \
    --dataset roxford5k --save_embeddings --output_dir results/smoke_pipeline
