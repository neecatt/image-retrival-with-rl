#!/bin/bash -l
#SBATCH --job-name=finetune
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err
#SBATCH --export=NONE
# Continue training from a fixed initial checkpoint, for matched comparisons:
#   ARM=ml  continued metric learning (Stage 1 loss, fresh optimizer)
#   ARM=rl  reward tuning (Stage 2)
# Usage: sbatch scripts/finetune_job.sh RUN_NAME ARM INIT_CKPT [key=value overrides...] [--final-eval]
# Selection on the training set's held-out val; with --final-eval, ROxford/RParis are scored once on the selected checkpoint.
# Env: TRAIN_SET=sfm120k|gldv2c (default sfm120k; also pass dataset.name=gldv2c and the 512 px sizes).
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1   # cached DINOv2 weights, no network on nodes
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
: "${HPCVAULT:?HPCVAULT must be set}"
RUN="${1:?RUN_NAME}"; ARM="${2:?ARM ml|rl}"; INIT="${3:?INIT_CKPT}"; shift 3
# INIT may be a glob (e.g. for a Stage 1 run that finishes after submission); resolve it now
INIT=$(ls -t $INIT | head -1); [ -f "$INIT" ] || { echo "init checkpoint not found"; exit 2; }
FINAL=0; OVR=()
for x in "$@"; do if [ "$x" = "--final-eval" ]; then FINAL=1; else OVR+=("$x"); fi; done
case "$ARM" in ml) STAGE=stage1 ;; rl) STAGE=stage2 ;; *) echo "ARM must be ml or rl"; exit 2 ;; esac
OUT="results/runs/$RUN"; CKPT="$HPCVAULT/runs/$RUN"
mkdir -p "$OUT" "$CKPT"
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) arm=$ARM init=$INIT" >> "$OUT/jobs.txt"
git diff > "$OUT/git_diff_${SLURM_JOB_ID}.patch"
sha256sum "$INIT" > "$OUT/init_sha256.txt"

TRAIN_SET="${TRAIN_SET:-sfm120k}"
DATASETS=("$TRAIN_SET" ${EXTRA_STAGE:-}); [ "$FINAL" = 1 ] && DATASETS+=(roxford5k rparis6k)
python scripts/stage_datasets.py "${DATASETS[@]}"
python scripts/make_run_config.py config/config.yaml "$OUT/run_config.yaml" \
    dataset.root_dir="$TMPDIR/data/raw" \
    system.checkpoint_dir="$CKPT" system.log_dir="$OUT/tb" system.results_dir="$OUT" \
    system.num_workers=16 system.save_frequency=1000 "${OVR[@]}"
python train.py --config "$OUT/run_config.yaml" --${STAGE}_only --init_checkpoint "$INIT" < /dev/null 2>&1 \
    | grep -v "it/s\]\|s/it\]"

if [ "$FINAL" = 1 ]; then
    BEST=$(ls -t "$CKPT"/${STAGE}_*/${STAGE}_best.pth | head -1)
    VAL_ARGS=(); [[ "$TRAIN_SET" == gldv2* ]] && VAL_ARGS=(--val_set "$TRAIN_SET" --val_size 512)
    python scripts/eval_baselines.py --data_root "$TMPDIR/data/raw" --out "$OUT/final_eval.json" --models "$BEST" "${VAL_ARGS[@]}"
fi
