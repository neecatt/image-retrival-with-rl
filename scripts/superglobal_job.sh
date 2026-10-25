#!/bin/bash -l
#SBATCH --job-name=supglob
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/supglob-%j.out
#SBATCH --error=logs/supglob-%j.err
#SBATCH --export=NONE
# SuperGlobal reproduction (scripts/superglobal_eval.py). Usage: sbatch scripts/superglobal_job.sh OUT [args]
# Without args beyond OUT: benchmarks only, fp32 into OUT/fp32 and fp16 autocast into OUT/amp.
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
O="${1:?OUT}"; shift
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)"
python scripts/stage_datasets.py roxford5k rparis6k
if [ $# -eq 0 ]; then
  python scripts/superglobal_eval.py --data_root "$TMPDIR/data/raw" --out $O/fp32
  python scripts/superglobal_eval.py --data_root "$TMPDIR/data/raw" --out $O/amp --amp
else
  python scripts/superglobal_eval.py --data_root "$TMPDIR/data/raw" --out $O "$@"
fi
