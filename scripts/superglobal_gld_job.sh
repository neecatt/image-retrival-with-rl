#!/bin/bash -l
#SBATCH --job-name=supglobgld
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/supglobgld-%j.out
#SBATCH --error=logs/supglobgld-%j.err
#SBATCH --export=NONE
# SuperGlobal descriptors for the GLDv2 val pool + GLDv2 distractors (policy-training data), 3 processes per GPU.
# Usage: sbatch scripts/superglobal_gld_job.sh OUT
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
O="${1:?OUT}"
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)"
python scripts/stage_datasets.py gldv2c gldv2d
for k in 0 1 2; do
  python scripts/superglobal_eval.py --data_root "$TMPDIR/data/raw" --out $O --amp --workers 6 --gld_shard $k/3 > logs/supglobgld-$SLURM_JOB_ID-shard$k.log 2>&1 &
done
wait
grep -h "GLDv2 shard" logs/supglobgld-$SLURM_JOB_ID-shard*.log
python scripts/superglobal_eval.py --data_root "$TMPDIR/data/raw" --out $O --gld_merge
