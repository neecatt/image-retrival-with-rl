#!/bin/bash -l
#SBATCH --job-name=sfm_archive_check
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/sfm_archive_check-%j.out
#SBATCH --error=logs/sfm_archive_check-%j.err
#SBATCH --export=NONE
# CPU-only archive validation (no extraction, no GPU use). Alex has GPU partitions only.
# Submit from the repository root after: mkdir -p logs
set -euo pipefail
unset SLURM_EXPORT_ENV

module load python/3.12-base
module list 2>&1

DS="/home/vault/b268dc/b268dc21/datasets/sfm120k"
OUT="/home/vault/b268dc/b268dc21/datasets/logs"
cd "$SLURM_SUBMIT_DIR"

sha256sum "$DS/ims.tar.gz" > "$OUT/sha256_sfm120k_ims.txt" &
SHA_PID=$!
python3 -I scripts/check_sfm120k_archive.py "$DS/ims.tar.gz" "$DS/retrieval-SfM-120k.pkl" "$OUT/sfm120k_archive_check.json"
wait "$SHA_PID"
cat "$OUT/sha256_sfm120k_ims.txt"
