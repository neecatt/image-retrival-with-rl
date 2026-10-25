#!/bin/bash -l
#SBATCH --job-name=v_pre_emb
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/v_pre_emb-%j.out
#SBATCH --error=logs/v_pre_emb-%j.err
#SBATCH --export=NONE
# Track V shortlists for precomputed embeddings (scripts/v_precompute_emb.py). Usage: sbatch ... EMB_DIR OUT
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
E="${1:?EMB_DIR}"; O="${2:?OUT}"
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)"
python scripts/stage_datasets.py gldv2c gldv2d roxford5k rparis6k
T=/home/vault/b268dc/b268dc21/runs/trackv; Q=/home/vault/b268dc/b268dc21/runs/trackqe
C=""; for d in $T/*_q10k $T/*_dist $T/*_r1m $Q/newresnet_seed* $Q/dinov2wise; do [ -d $d ] && C="$C $d"; done
python scripts/v_precompute_emb.py --emb_dir $E --data_root "$TMPDIR/data/raw" --queries_from $T/newresnet_seed42_q10k/gldv2c_val.npz \
  --cache_from $C --out $O 2>&1 | grep -v "it/s\]\|s/it\]"
