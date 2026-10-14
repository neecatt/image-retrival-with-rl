#!/bin/bash -l
#SBATCH --job-name=train_s1
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=07:00:00
#SBATCH --output=logs/train_s1-%j.out
#SBATCH --error=logs/train_s1-%j.err
#SBATCH --export=NONE
# Full Stage 1 run from ImageNet. Usage: sbatch scripts/train_stage1_job.sh RUN_NAME [key=value overrides...]
# Checkpoints persist in $HPCVAULT/runs/RUN_NAME (rolling last/best); resubmitting the same RUN_NAME resumes.
# Selection on the training set's held-out val only; ROxford/RParis evaluated once at the end on stage1_best.pth.
# Env: TRAIN_SET=sfm120k|gldv2c (default sfm120k; pass dataset.name=gldv2c too), SKIP_FINAL_EVAL=1 for pilots.
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1   # cached DINOv2 weights, no network on nodes
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
: "${HPCVAULT:?HPCVAULT must be set}"
RUN="${1:?usage: sbatch scripts/train_stage1_job.sh RUN_NAME [overrides...]}"; shift
OUT="results/runs/$RUN"; CKPT="$HPCVAULT/runs/$RUN"
mkdir -p "$OUT" "$CKPT"
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)" >> "$OUT/jobs.txt"
git diff > "$OUT/git_diff_${SLURM_JOB_ID}.patch"
pip freeze > "$OUT/pip_freeze.txt"

TRAIN_SET="${TRAIN_SET:-sfm120k}"
if [ "${SKIP_FINAL_EVAL:-0}" = 1 ]; then python scripts/stage_datasets.py "$TRAIN_SET"
else python scripts/stage_datasets.py "$TRAIN_SET" roxford5k rparis6k; fi
python scripts/make_run_config.py config/config.yaml "$OUT/run_config.yaml" \
    dataset.root_dir="$TMPDIR/data/raw" \
    system.checkpoint_dir="$CKPT" system.log_dir="$OUT/tb" system.results_dir="$OUT" \
    system.num_workers=16 system.save_frequency=5 "$@"

RESUME=()
LAST=$(ls -t "$CKPT"/stage1_*/stage1_last.pth 2>/dev/null | head -1 || true)
if [ -n "$LAST" ]; then echo "Resuming from $LAST"; RESUME=(--resume "$LAST"); fi
python train.py --config "$OUT/run_config.yaml" --stage1_only "${RESUME[@]}" 2>&1 \
    | grep -v "it/s\]\|s/it\]"

[ "${SKIP_FINAL_EVAL:-0}" = 1 ] && exit 0
BEST=$(ls -t "$CKPT"/stage1_*/stage1_best.pth | head -1)
echo "Final evaluation of $BEST"
VAL_ARGS=(); [[ "$TRAIN_SET" == gldv2* ]] && VAL_ARGS=(--val_set "$TRAIN_SET" --val_size 512)
python scripts/eval_baselines.py --data_root "$TMPDIR/data/raw" --out "$OUT/final_eval.json" --models "$BEST" "${VAL_ARGS[@]}"
