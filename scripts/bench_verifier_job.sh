#!/bin/bash -l
#SBATCH --job-name=bench_ver
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/bench_ver-%j.out
#SBATCH --error=logs/bench_ver-%j.err
#SBATCH --export=NONE
# Track N1c: batched-verifier throughput and agreement. Usage: sbatch scripts/bench_verifier_job.sh [args]
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)"
python scripts/stage_datasets.py gldv2c
python scripts/bench_verifier.py --data_root "$TMPDIR/data/raw" --dataset gldv2c "$@"
