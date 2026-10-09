#!/bin/bash -l
#SBATCH --job-name=stage_sfm
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/stage_sfm-%j.out
#SBATCH --error=logs/stage_sfm-%j.err
#SBATCH --export=NONE
# Pilot: stage SfM-120k into node-local TMPDIR and verify against the metadata pickle (CPU-only).
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
cd "$SLURM_SUBMIT_DIR"
echo "TMPDIR=${TMPDIR:-UNSET}"; df -h "$TMPDIR" | tail -1
python3 -I scripts/stage_datasets.py sfm120k
python3 -I scripts/verify_staged_sfm120k.py "$TMPDIR/data/raw"
du -sh "$TMPDIR/data/raw/sfm120k/jpg"
cat "$TMPDIR/data/raw/stage_manifest.json"
