#!/bin/bash -l
#SBATCH --job-name=qe_headroom
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/qe_headroom-%j.out
#SBATCH --error=logs/qe_headroom-%j.err
#SBATCH --export=NONE
# Track V-QE headroom check (scripts/analysis/qe_headroom.py).
# Usage: sbatch scripts/qe_headroom_job.sh MODEL PLAIN_PRE DIST_PRE R1M_PRE DIST_EMB R1M_EMB OUT [args]
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
M="${1:?MODEL}"; P="${2:?}"; D="${3:?}"; R="${4:?}"; DE="${5:?}"; RE="${6:?}"; O="${7:?OUT}"; shift 7
[ -e "$M" ] || M=$(ls -t $M | head -1)
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) model=$M"
python scripts/stage_datasets.py gldv2c roxford5k rparis6k
python scripts/analysis/qe_headroom.py --data_root "$TMPDIR/data/raw" --model "$M" --plain_pre "$P" --dist_pre "$D" \
  --r1m_pre "$R" --dist_emb "$DE" --r1m_emb "$RE" --out "$O" "$@" 2>&1 | grep -v "it/s\]\|s/it\]"
