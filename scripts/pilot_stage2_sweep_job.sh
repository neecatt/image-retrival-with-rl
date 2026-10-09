#!/bin/bash -l
#SBATCH --job-name=pilot_s2_sweep
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/pilot_s2_sweep-%j.out
#SBATCH --error=logs/pilot_s2_sweep-%j.err
#SBATCH --export=NONE
# Pilot screen of Stage 2 settings listed in a spec file (scripts/sweeps/*.txt), one seed, from the
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

SPEC="${1:?usage: sbatch pilot_stage2_sweep_job.sh scripts/sweeps/<spec>.txt}"
cp "$SPEC" "$OUT/"
grep -v '^\s*#' "$SPEC" | while read -r name std clip lr cw nb; do
    [ -z "$name" ] && continue
    echo "=== RUN $name (sample_std=$std clip=$clip lr=$lr consistency=$cw batches=$nb) ==="
    python scripts/make_run_config.py config/config.yaml "$OUT/$name.yaml" < /dev/null \
        dataset.root_dir="$TMPDIR/data/raw" \
        system.checkpoint_dir="$TMPDIR/ckpt_$name" system.log_dir="$TMPDIR/logs_$name" \
        system.results_dir="$OUT/$name" system.num_workers=8 system.seed=42 \
        training.stage2.max_batches_per_epoch="$nb" training.stage2.warmup_steps=100 training.stage2.diagnostics_every=25 \
        training.stage2.sample_std="$std" training.stage2.gradient_clip="$clip" \
        training.stage2.learning_rate="$lr" training.stage2.consistency_weight="$cw"
    python train.py --config "$OUT/$name.yaml" --stage2_only \
        --resume checkpoints/stage1_epoch_13_best.pth --stage2_epochs 1 < /dev/null 2>&1 \
        | grep -v "it/s\]\|s/it\]" | tee "$OUT/$name.log"
done
