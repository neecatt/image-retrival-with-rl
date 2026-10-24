#!/bin/bash -l
#SBATCH --job-name=qe_rq
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/qe_rq-%j.out
#SBATCH --error=logs/qe_rq-%j.err
#SBATCH --export=NONE
# Track V-QE with the verified-QE re-query in the reward (training/qe_requery.py), one new-ResNet seed, B = 20,
# policy seed 0, configs selected on validation in Q5 (classifier/value DAgger lr 1e-3, RL from scratch lr 3e-4);
# then exact re-query evaluation on the benchmarks (scripts/analysis/qe_policy_vqe.py).
# Usage: sbatch scripts/qe_requery_job.sh SEED R1M_HASH   (e.g. 42 ca8a6dd78f)
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
S="${1:?SEED}"; H="${2:?R1M_HASH}"
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) seed=$S"
python scripts/stage_datasets.py gldv2c roxford5k rparis6k
M=$(ls /home/vault/b268dc/b268dc21/runs/gr_full_c02_e10_seed$S/stage1_*/stage1_best.pth)
P=/home/vault/b268dc/b268dc21/runs/trackqe/newresnet_seed$S; O=results/qe/rq_s$S
DE=/home/atuin/b268dc/b268dc21/datasets/gldv2d/embeddings/newresnet_seed${S}_e512.npy
RQ="--requery_reward --data_root $TMPDIR/data/raw --model $M --dist_emb $DE --bench_pools --budgets 20 --seeds 0"
python scripts/train_qe_policy.py --pre $P --out $O/S_B_lr1e-3_B20_s0 --methods classifier value --lr 1e-3 --sup_iters 12000 --dagger $RQ
python scripts/train_qe_policy.py --pre $P --out $O/RS_lr3e-4_B20_s0 --methods rl --lr 3e-4 --iters 6000 $RQ
python scripts/analysis/qe_policy_vqe.py --data_root "$TMPDIR/data/raw" --desc newresnet_seed${S}_rq --model $M --qe_pre $P \
  --r1m_emb /home/atuin/b268dc/b268dc21/datasets/revisitop1m/embeddings/gr_full_c02_e10_seed${S}_stage1_best_${H}_s1024.npy \
  --out results/qe/policy_vqe --policies classifier=$O/S_B_lr1e-3_B20_s0/policy_B20_classifier_seed0.pt \
  value=$O/S_B_lr1e-3_B20_s0/policy_B20_value_seed0.pt rl=$O/RS_lr3e-4_B20_s0/policy_B20_rl_seed0.pt 2>&1 | grep -v "it/s\]\|s/it\]"
