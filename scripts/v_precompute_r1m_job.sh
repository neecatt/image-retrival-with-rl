#!/bin/bash -l
#SBATCH --job-name=v_pre_r1m
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/v_pre_r1m-%j.out
#SBATCH --error=logs/v_pre_r1m-%j.err
#SBATCH --export=NONE
# Track V + R1M shortlists. Usage: sbatch scripts/v_precompute_r1m_job.sh MODEL R1M_EMB.npy OUT_DIR
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
M="${1:?MODEL}"; E="${2:?R1M_EMB}"; O="${3:?OUT}"
[ -e "$M" ] || M=$(ls -t $M | head -1)
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) model=$M emb=$E"
python scripts/stage_datasets.py roxford5k rparis6k
python scripts/v_precompute_r1m.py --data_root "$TMPDIR/data/raw" --model "$M" --r1m_emb "$E" --out "$O"
