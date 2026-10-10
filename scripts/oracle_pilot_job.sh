#!/bin/bash -l
#SBATCH --job-name=oracle_pilot
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/oracle_pilot-%j.out
#SBATCH --error=logs/oracle_pilot-%j.err
#SBATCH --export=NONE
# Phase 2.2: 1% pilot of the geometric-verification relevance oracle (ALIKED + LightGlue + MAGSAC).
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
OUT="results/oracle_pilot_${SLURM_JOB_ID}"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_rev.txt"; git diff > "$OUT/git_diff.patch"
pip freeze | grep -i lightglue > "$OUT/lightglue_version.txt"
python scripts/stage_datasets.py sfm120k
CKPT=$(ls /home/vault/b268dc/b268dc21/runs/s1_b32_lr1e-5_seed42/stage1_*/stage1_best.pth)
python scripts/oracle_pilot.py --data_root "$TMPDIR/data/raw" --checkpoint "$CKPT" --out "$OUT"
