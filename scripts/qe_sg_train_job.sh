#!/bin/bash -l
#SBATCH --job-name=qe_sg
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/qe_sg-%j.out
#SBATCH --error=logs/qe_sg-%j.err
#SBATCH --export=NONE
# Track V-QE on SuperGlobal descriptors: schedulers trained with the re-query reward (configs as selected for
# the new ResNet, not re-tuned), then exact re-query evaluation with and without SuperGlobal reranking.
# Usage: sbatch scripts/qe_sg_train_job.sh EMB_DIR QE_PRE OUT_TAG
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
E="${1:?EMB_DIR}"; P="${2:?QE_PRE}"; TAG="${3:?TAG}"; O=results/qe/$TAG
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD)"
python scripts/stage_datasets.py gldv2c roxford5k rparis6k
RQ="--requery_reward --data_root $TMPDIR/data/raw --emb_pool $E/gld_pool.npy --dist_emb $E/gldv2d.npy --bench_pools --budgets 20 --seeds 0"
python scripts/train_qe_policy.py --pre $P --out $O/S_B_lr1e-3_B20_s0 --methods classifier value --lr 1e-3 --sup_iters 12000 --dagger $RQ
python scripts/train_qe_policy.py --pre $P --out $O/RS_lr3e-4_B20_s0 --methods rl --lr 3e-4 --iters 6000 $RQ
python scripts/analysis/qe_policy_vqe.py --data_root "$TMPDIR/data/raw" --desc $TAG --emb_dir $E --qe_pre $P --r1m_emb $E/r1m.npy \
  --sg_repo /home/hpc/b268dc/b268dc21/external/SuperGlobal --out results/qe/policy_vqe \
  --policies classifier=$O/S_B_lr1e-3_B20_s0/policy_B20_classifier_seed0.pt value=$O/S_B_lr1e-3_B20_s0/policy_B20_value_seed0.pt \
  rl=$O/RS_lr3e-4_B20_s0/policy_B20_rl_seed0.pt 2>&1 | grep -v "it/s\]\|s/it\]"
