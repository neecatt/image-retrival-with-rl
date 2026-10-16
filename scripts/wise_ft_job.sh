#!/bin/bash -l
#SBATCH --job-name=wise_ft
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/wise_ft-%j.out
#SBATCH --error=logs/wise_ft-%j.err
#SBATCH --export=NONE
# WiSE-FT: choose the interpolation between pretrained and fine-tuned weights on GLDv2+SfM val.
# Usage: sbatch scripts/wise_ft_job.sh ZERO_CKPT FT_CKPT OUT.pth [--final-eval] [args]
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
Z="${1:?ZERO}"; F="${2:?FT}"; O="${3:?OUT}"; shift 3
F=$(ls -t $F | head -1)
FINAL=0; ARGS=(); for x in "$@"; do if [ "$x" = "--final-eval" ]; then FINAL=1; else ARGS+=("$x"); fi; done
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) zero=$Z ft=$F"
D=(gldv2c sfm120k); [ "$FINAL" = 1 ] && D+=(roxford5k rparis6k)
python scripts/stage_datasets.py "${D[@]}"
python scripts/wise_ft.py --zero "$Z" --ft "$F" --data_root "$TMPDIR/data/raw" --out "$O" "${ARGS[@]}"
if [ "$FINAL" = 1 ]; then
    python scripts/eval_baselines.py --data_root "$TMPDIR/data/raw" --out "${O%.pth}_final_eval.json" --models "$O" \
        --val_set gldv2c --val_size 512 --resize long
fi
