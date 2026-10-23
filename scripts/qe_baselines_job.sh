#!/bin/bash -l
#SBATCH --job-name=qe_base
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/qe_base-%j.out
#SBATCH --error=logs/qe_base-%j.err
#SBATCH --export=NONE
# Track V-QE baselines (aQE, verified QE, verify-all with expansion), budget curves and timings for all
# six descriptors in one job (scripts/analysis/qe_baselines.py). Usage: sbatch scripts/qe_baselines_job.sh
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)"
python scripts/stage_datasets.py gldv2c roxford5k rparis6k
T=/home/vault/b268dc/b268dc21/runs/trackv; Q=/home/vault/b268dc/b268dc21/runs/trackqe; E=/home/atuin/b268dc/b268dc21/datasets
declare -A H=([42]=ca8a6dd78f [1]=0d392b07b0 [2]=7f7c864fb3 [3]=dff862b1f7 [4]=fbdbaf6868)
run() { python scripts/analysis/qe_baselines.py --data_root "$TMPDIR/data/raw" --out results/qe/baselines "$@" 2>&1 | grep -v "it/s\]\|s/it\]"; }
run --desc dinov2wise --model /home/vault/b268dc/b268dc21/runs/wise_ft/gd_full_c04_e6.pth --plain_pre $T/dinov2wise_q10k \
    --dist_pre $T/dinov2wise_dist --r1m_pre $T/dinov2wise_r1m --qe_pre $Q/dinov2wise --dist_emb $E/gldv2d/embeddings/dinov2wise_e512.npy \
    --r1m_emb $E/revisitop1m/embeddings/runs_gd_full_c04_e6_89afda7ed7_s1024.npy --timing
for s in 42 1 2 3 4; do
  M=$(ls /home/vault/b268dc/b268dc21/runs/gr_full_c02_e10_seed$s/stage1_*/stage1_best.pth)
  run --desc newresnet_seed$s --model $M --plain_pre $T/newresnet_seed${s}_q10k --dist_pre $T/newresnet_seed${s}_dist \
      --r1m_pre $T/newresnet_seed${s}_r1m --qe_pre $Q/newresnet_seed$s --dist_emb $E/gldv2d/embeddings/newresnet_seed${s}_e512.npy \
      --r1m_emb $E/revisitop1m/embeddings/gr_full_c02_e10_seed${s}_stage1_best_${H[$s]}_s1024.npy
done
