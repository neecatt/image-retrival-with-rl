#!/bin/bash -l
#SBATCH --job-name=v_precompute
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/v_precompute-%j.out
#SBATCH --error=logs/v_precompute-%j.err
#SBATCH --export=NONE
# Track V step 1: shortlists, shortlist similarities and verifier inliers for a fixed descriptor.
# Usage: sbatch scripts/v_precompute_job.sh MODEL_CKPT OUT_DIR [args]
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
MODEL="${1:?MODEL}"; OUT="${2:?OUT}"; shift 2
MODEL=$(ls -t $MODEL | head -1)
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) model=$MODEL" | tee -a logs/v_precompute_runs.txt
python scripts/stage_datasets.py gldv2c roxford5k rparis6k
python scripts/v_precompute.py --data_root "$TMPDIR/data/raw" --model "$MODEL" --out "$OUT" "$@" 2>&1 | grep -v "it/s\]\|s/it\]"
