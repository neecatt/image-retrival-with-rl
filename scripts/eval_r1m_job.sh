#!/bin/bash -l
#SBATCH --job-name=eval_r1m
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/eval_r1m-%j.out
#SBATCH --error=logs/eval_r1m-%j.err
#SBATCH --export=NONE
# ROxford/RParis + R1M distractors. Usage: sbatch scripts/eval_r1m_job.sh OUT.json MODEL [MODEL ...]
# R1M is streamed from the verified archives on WORK; embeddings are cached per model on WORK.
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
OUT="${1:?OUT}"; shift
M=(); for x in "$@"; do if [ -e "$x" ] || [[ "$x" == pretrained:* ]] || [ "$x" = imagenet ]; then M+=("$x"); else M+=($(ls -t $x | head -1)); fi; done
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) ${M[*]}"
[ "$(ls /home/atuin/b268dc/b268dc21/datasets/revisitop1m/raw/*.ok | wc -l)" = 100 ] || { echo "R1M not fully verified"; exit 2; }
mkdir -p "$(dirname "$OUT")"
python scripts/stage_datasets.py roxford5k rparis6k
python scripts/eval_r1m.py --data_root "$TMPDIR/data/raw" --out "$OUT" --workers 24 --models "${M[@]}"
