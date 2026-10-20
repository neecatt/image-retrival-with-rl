#!/bin/bash -l
#SBATCH --job-name=v_pre_qe
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/v_pre_qe-%j.out
#SBATCH --error=logs/v_pre_qe-%j.err
#SBATCH --export=NONE
# Track V-QE precompute: shortlists extended by query expansion (scripts/v_precompute_qe.py).
# Usage: sbatch scripts/v_precompute_qe_job.sh MODEL PLAIN_PRE DIST_PRE R1M_PRE DIST_EMB R1M_EMB OUT [args]
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
python scripts/stage_datasets.py gldv2c gldv2d roxford5k rparis6k
python scripts/v_precompute_qe.py --data_root "$TMPDIR/data/raw" --model "$M" --plain_pre "$P" --dist_pre "$D" \
  --r1m_pre "$R" --dist_emb "$DE" --r1m_emb "$RE" --out "$O" "$@" 2>&1 | grep -v "it/s\]\|s/it\]"
