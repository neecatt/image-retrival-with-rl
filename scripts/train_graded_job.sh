#!/bin/bash -l
#SBATCH --job-name=graded
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/graded-%j.out
#SBATCH --error=logs/graded-%j.err
#SBATCH --export=NONE
# Phase 2 graded fine-tuning. Usage:
#   sbatch [-p a100 --gres=gpu:a100:1] scripts/train_graded_job.sh RUN_NAME MODE INIT_CKPT [train_graded.py args...] [--final-eval]
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
: "${HPCVAULT:?HPCVAULT must be set}"
RUN="${1:?RUN_NAME}"; MODE="${2:?rl|approxndcg}"; INIT="${3:?INIT_CKPT}"; shift 3
FINAL=0; ARGS=()
for x in "$@"; do if [ "$x" = "--final-eval" ]; then FINAL=1; else ARGS+=("$x"); fi; done
OUT="results/runs/$RUN"; CKPT="$HPCVAULT/runs/$RUN"; ORACLE="$HPCVAULT/datasets/sfm120k/oracle_v1"
mkdir -p "$OUT" "$CKPT"
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) mode=$MODE init=$INIT partition=$SLURM_JOB_PARTITION" >> "$OUT/jobs.txt"
git diff > "$OUT/git_diff_${SLURM_JOB_ID}.patch"
sha256sum "$INIT" > "$OUT/init_sha256.txt"
DATASETS=(sfm120k); [ "$FINAL" = 1 ] && DATASETS+=(roxford5k rparis6k)
python scripts/stage_datasets.py "${DATASETS[@]}"
python scripts/train_graded.py --mode "$MODE" --init "$INIT" --data_root "$TMPDIR/data/raw" \
    --oracle_dir "$ORACLE" --out "$OUT" --ckpt_dir "$CKPT" "${ARGS[@]}" < /dev/null
if [ "$FINAL" = 1 ]; then
    python scripts/eval_baselines.py --data_root "$TMPDIR/data/raw" --out "$OUT/final_eval.json" --models "$CKPT/graded_best.pth"
fi
