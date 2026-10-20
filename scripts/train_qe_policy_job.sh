#!/bin/bash -l
#SBATCH --job-name=qe_policy
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/qe_policy-%j.out
#SBATCH --error=logs/qe_policy-%j.err
#SBATCH --export=NONE
# Track V-QE policies. Usage: sbatch scripts/train_qe_policy_job.sh PRE_DIR OUT [args for train_qe_policy.py]
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PRE="${1:?PRE}"; OUT="${2:?OUT}"; shift 2
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)" | tee -a logs/qe_policy_runs.txt
python scripts/train_qe_policy.py --pre "$PRE" --out "$OUT" "$@"
