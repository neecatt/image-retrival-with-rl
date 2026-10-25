#!/bin/bash -l
#SBATCH --job-name=supglob1m
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/supglob1m-%j.out
#SBATCH --error=logs/supglob1m-%j.err
#SBATCH --export=NONE
# SuperGlobal R1M descriptors (fp16 autocast, verified identical to fp32 on ROxf/RPar): 3 extraction processes
# share the GPU (one is latency-bound at ~38 img/s), then merge and evaluate +1M. Usage: sbatch ... OUT
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
O="${1:?OUT}"
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)"
python scripts/stage_datasets.py roxford5k rparis6k
for k in 0 1 2; do
  python scripts/superglobal_eval.py --data_root "$TMPDIR/data/raw" --out $O --amp --workers 6 --r1m_shard $k/3 > logs/supglob1m-$SLURM_JOB_ID-shard$k.log 2>&1 &
done
wait
grep -h "R1M shard" logs/supglob1m-$SLURM_JOB_ID-shard*.log
python scripts/superglobal_eval.py --data_root "$TMPDIR/data/raw" --out $O --amp --r1m
