#!/bin/bash -l
#SBATCH --job-name=a0_noise
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/a0_noise-%j.out
#SBATCH --error=logs/a0_noise-%j.err
#SBATCH --export=NONE
# A0 label-noise / exploration diagnostic. Usage: sbatch scripts/a0_label_noise_job.sh MODEL_CKPT [args]
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
MODEL="${1:?MODEL_CKPT}"; shift
: "${HPCVAULT:?HPCVAULT must be set}"
OUT="results/a0_label_noise"
mkdir -p "$OUT"
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) model=$MODEL" >> "$OUT/jobs.txt"
python scripts/stage_datasets.py sfm120k
python scripts/a0_label_noise.py --data_root "$TMPDIR/data/raw" --model "$MODEL" --out "$OUT" \
    --emb_file "$HPCVAULT/runs/a0_label_noise/train_embeddings_$(basename $(dirname $(dirname "$MODEL")))_$(basename "$MODEL" .pth).npy" "$@" 2>&1 \
    | grep -v "it/s\]\|s/it\]"
