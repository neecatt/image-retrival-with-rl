#!/bin/bash -l
#SBATCH --job-name=pilot_s2_sweep
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=01:15:00
#SBATCH --output=logs/pilot_s2_sweep-%j.out
#SBATCH --error=logs/pilot_s2_sweep-%j.err
#SBATCH --export=NONE
# Pilot screen (one seed, 400 batches each) of Stage 2 noise level, gradient clipping and LR from the
# same Stage 1 checkpoint, with gradient diagnostics. Selection metric: SfM-120k val mAP. Not a final result.
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
OUT="results/pilot_s2_sweep_${SLURM_JOB_ID}"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_rev.txt"; git diff > "$OUT/git_diff.patch"
python scripts/stage_datasets.py sfm120k

# name: sample_std gradient_clip learning_rate
RUNS="a_sigma0.01_clip1:0.01:1.0:0.000002
b_sigma0.05_clip1:0.05:1.0:0.000002
c_sigma0.01_noclip:0.01:0:0.000002
d_sigma0.05_noclip:0.05:0:0.000002
e_sigma0.05_clip1_lr1e-5:0.05:1.0:0.00001"
for spec in $RUNS; do
    IFS=: read -r name std clip lr <<< "$spec"
    echo "=== RUN $name (sample_std=$std gradient_clip=$clip lr=$lr) ==="
    python scripts/make_run_config.py config/config.yaml "$OUT/$name.yaml" \
        dataset.root_dir="$TMPDIR/data/raw" \
        system.checkpoint_dir="$TMPDIR/ckpt_$name" system.log_dir="$TMPDIR/logs_$name" \
        system.results_dir="$OUT/$name" system.num_workers=8 system.seed=42 \
        training.stage2.max_batches_per_epoch=400 training.stage2.warmup_steps=100 training.stage2.diagnostics_every=25 \
        training.stage2.sample_std="$std" training.stage2.gradient_clip="$clip" training.stage2.learning_rate="$lr"
    python train.py --config "$OUT/$name.yaml" --stage2_only \
        --resume checkpoints/stage1_epoch_13_best.pth --stage2_epochs 1 2>&1 \
        | grep -v "it/s\]\|s/it\]" | tee "$OUT/$name.log"
done
