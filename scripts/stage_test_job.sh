#!/bin/bash -l
#SBATCH --job-name=stage_test
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/stage_test-%j.out
#SBATCH --error=logs/stage_test-%j.err
#SBATCH --export=NONE
# Pilot: stage ROxford5k + RParis6k into node-local TMPDIR and verify (CPU-only; no GPU use).
# Submit from the repository root after: mkdir -p logs
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
cd "$SLURM_SUBMIT_DIR"
echo "TMPDIR=${TMPDIR:-UNSET}"; df -h "$TMPDIR" | tail -1; df -i "$TMPDIR" | tail -1
python3 -I scripts/stage_datasets.py roxford5k rparis6k --config-in config/config.yaml --config-out "$TMPDIR/run_config.yaml"
python3 -I scripts/verify_staged_revisited.py "$TMPDIR/data/raw" roxford5k rparis6k
grep -n "root_dir" "$TMPDIR/run_config.yaml"
cat "$TMPDIR/data/raw/stage_manifest.json"
