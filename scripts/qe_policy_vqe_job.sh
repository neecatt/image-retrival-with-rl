#!/bin/bash -l
#SBATCH --job-name=qe_pvqe
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/qe_pvqe-%j.out
#SBATCH --error=logs/qe_pvqe-%j.err
#SBATCH --export=NONE
# Verified-QE re-query with verifications chosen by the saved B = 20 policies (scripts/analysis/qe_policy_vqe.py),
# six descriptors; policies = the configs selected on validation in PROGRESS Q5 / Q8.
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)"
python scripts/stage_datasets.py roxford5k rparis6k
Q=/home/vault/b268dc/b268dc21/runs/trackqe
E=/home/atuin/b268dc/b268dc21/datasets/revisitop1m/embeddings
run() { python scripts/analysis/qe_policy_vqe.py --data_root "$TMPDIR/data/raw" --out results/qe/policy_vqe "$@" 2>&1 | grep -v "it/s\]\|s/it\]"; }

run --desc dinov2wise --model /home/vault/b268dc/b268dc21/runs/wise_ft/gd_full_c04_e6.pth --qe_pre $Q/dinov2wise \
  --r1m_emb $E/runs_gd_full_c04_e6_89afda7ed7_s1024.npy --policies \
  classifier=results/qe/v2_sdinov2wise/S_B_lr1e-3_B20_s0/policy_B20_classifier_seed0.pt \
  value=results/qe/v2_sdinov2wise/S_B_lr3e-4_B20_s0/policy_B20_value_seed0.pt \
  rl=results/qe/v2_sdinov2wise/RS_lr3e-4_B20_s0/policy_B20_rl_seed0.pt \
  classifier=results/qe/v2_sdinov2wise/S_B_lr1e-3_B20_s1/policy_B20_classifier_seed1.pt \
  value=results/qe/v2_sdinov2wise/S_B_lr1e-3_B20_s1/policy_B20_value_seed1.pt \
  rl=results/qe/v2_sdinov2wise/RS_lr3e-4_B20_s1/policy_B20_rl_seed1.pt

declare -A H=([42]=ca8a6dd78f [1]=0d392b07b0 [2]=7f7c864fb3 [3]=dff862b1f7 [4]=fbdbaf6868)
declare -A CLS=([42]=S_B_lr3e-4 [1]=S_B_lr3e-4 [2]=S_B_lr3e-4 [3]=S_B_lr3e-4 [4]=S_B_lr1e-3)
declare -A VAL=([42]=S_B_lr1e-3 [1]=S_B_lr3e-4 [2]=S_B_lr1e-3 [3]=S_B_lr1e-3 [4]=S_B_lr1e-3)
for s in 42 1 2 3 4; do
  D=results/qe/v2_s$s; [ $s = 42 ] && D=results/qe/v2
  run --desc newresnet_seed$s --model $(ls /home/vault/b268dc/b268dc21/runs/gr_full_c02_e10_seed$s/stage1_*/stage1_best.pth) \
    --qe_pre $Q/newresnet_seed$s --r1m_emb $E/gr_full_c02_e10_seed${s}_stage1_best_${H[$s]}_s1024.npy --policies \
    classifier=$D/${CLS[$s]}_B20_s0/policy_B20_classifier_seed0.pt value=$D/${VAL[$s]}_B20_s0/policy_B20_value_seed0.pt \
    rl=$D/RS_lr3e-4_B20_s0/policy_B20_rl_seed0.pt
done
