#!/bin/bash -l
#SBATCH --job-name=oracle_label
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/oracle_label-%A_%a.out
#SBATCH --error=logs/oracle_label-%A_%a.err
#SBATCH --export=NONE
# Phase 2.4 graded relevance labelling. Usage:
#   sbatch --array=0-3 scripts/oracle_label_job.sh train 4
#   sbatch --array=0   scripts/oracle_label_job.sh val 1
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
: "${HPCVAULT:?HPCVAULT must be set}"
SPLIT="${1:?train|val}"; NSH="${2:?num_shards}"; SH="${SLURM_ARRAY_TASK_ID:-0}"
OUT="$HPCVAULT/datasets/sfm120k/oracle_v1"
mkdir -p "$OUT"
echo "$SLURM_JOB_ID ${SLURM_ARRAY_JOB_ID:-} $SPLIT shard $SH/$NSH $(date -Is) $(git rev-parse HEAD)" >> "$OUT/jobs.txt"
python scripts/stage_datasets.py sfm120k
python scripts/oracle_label.py --data_root "$TMPDIR/data/raw" --out "$OUT" \
    --split "$SPLIT" --shard "$SH" --num_shards "$NSH" --cpu_workers 24
