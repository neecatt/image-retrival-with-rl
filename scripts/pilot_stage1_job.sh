#!/bin/bash -l
#SBATCH --job-name=pilot_s1
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/pilot_s1-%j.out
#SBATCH --error=logs/pilot_s1-%j.err
#SBATCH --export=NONE
# Pilot screen for the Stage 1 baseline problem (P1): larger PK batch (32 = 8 landmarks x 4) and lower
# backbone LR, from ImageNet init, 3 short epochs each. Selection metric only: SfM-120k val mAP.
# Benchmarks (ROxford/RParis) are deliberately NOT evaluated here, to keep design choices off the test set.
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
: "${HPCVAULT:?HPCVAULT must be set}"
OUT="results/pilot_s1_${SLURM_JOB_ID}"
CKPT="$HPCVAULT/runs/pilot_s1_${SLURM_JOB_ID}"
mkdir -p "$OUT" "$CKPT"
git rev-parse HEAD > "$OUT/git_rev.txt"; git diff > "$OUT/git_diff.patch"
python scripts/stage_datasets.py sfm120k

# name:learning_rate
for spec in lr1e-4:0.0001 lr1e-5:0.00001 lr1e-6:0.000001; do
    IFS=: read -r name lr <<< "$spec"
    echo "=== RUN $name (batch 32, lr=$lr) ==="
    python scripts/make_run_config.py config/config.yaml "$OUT/$name.yaml" \
        dataset.root_dir="$TMPDIR/data/raw" \
        system.checkpoint_dir="$CKPT/$name" system.log_dir="$TMPDIR/logs_$name" \
        system.results_dir="$OUT/$name" system.num_workers=16 system.seed=42 system.save_frequency=100 \
        training.stage1.batch_size=32 training.stage1.learning_rate="$lr" \
        training.stage1.epochs=3 training.stage1.max_batches_per_epoch=1000
    python train.py --config "$OUT/$name.yaml" --stage1_only 2>&1 \
        | grep -v "it/s\]\|s/it\]" | tee "$OUT/$name.log"
done
