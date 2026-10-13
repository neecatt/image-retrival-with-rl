#!/bin/bash -l
#SBATCH --job-name=verified
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/verified-%j.out
#SBATCH --error=logs/verified-%j.err
#SBATCH --export=NONE
# Track N1d: verifier-in-the-loop rounds on the noisy GLDv2 pool.
# Usage: sbatch scripts/train_verified_job.sh RUN ARM INIT [args for train_verified.py] [--final-eval]
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
: "${HPCVAULT:?HPCVAULT must be set}"
RUN="${1:?RUN}"; ARM="${2:?ARM}"; INIT="${3:?INIT}"; shift 3
INIT=$(ls -t $INIT | head -1); [ -f "$INIT" ] || { echo "init not found"; exit 2; }
FINAL=0; ARGS=()
for x in "$@"; do if [ "$x" = "--final-eval" ]; then FINAL=1; else ARGS+=("$x"); fi; done
OUT="results/runs/$RUN"; CKPT="$HPCVAULT/runs/$RUN"
mkdir -p "$OUT" "$CKPT"
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) arm=$ARM init=$INIT" >> "$OUT/jobs.txt"
git diff > "$OUT/git_diff_${SLURM_JOB_ID}.patch"
DATASETS=(gldv2n); [ "$FINAL" = 1 ] && DATASETS+=(roxford5k rparis6k)
python scripts/stage_datasets.py "${DATASETS[@]}"
python scripts/train_verified.py --arm "$ARM" --init "$INIT" --data_root "$TMPDIR/data/raw" \
    --out "$OUT" --ckpt_dir "$CKPT" "${ARGS[@]}" 2>&1 | grep -v "it/s\]\|s/it\]"
if [ "$FINAL" = 1 ]; then
    python scripts/eval_baselines.py --data_root "$TMPDIR/data/raw" --out "$OUT/final_eval.json" \
        --models "$CKPT/verified_best.pth" --val_set gldv2n --val_size 512
fi
