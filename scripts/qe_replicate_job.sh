#!/bin/bash -l
#SBATCH --job-name=qe_rep
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/qe_rep-%j.out
#SBATCH --error=logs/qe_rep-%j.err
#SBATCH --export=NONE
# Track V-QE replication on one descriptor seed in a single job (limited compute): expansion
# precompute, then B = 20 policies with the configs selected on seed-42 validation, one policy seed:
# classifier/value with DAgger at lr 1e-3 and 3e-4 (selection between them on this seed's validation),
# RL from scratch at lr 3e-4. ~3.5 GPU-h.
# Usage: sbatch scripts/qe_replicate_job.sh SEED R1M_HASH [TAG [extra train_qe_policy.py args]]
#   TAG names the output dir results/qe/<TAG>_s<SEED> (default v2); extra args go to every training run.
#   Budget: QE_BUDGET (default 20), passed with sbatch --export=QE_BUDGET=40.
set -euo pipefail
unset SLURM_EXPORT_ENV
module load python/3.12-base
source "$HOME/venvs/rl-env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/home/vault/b268dc/b268dc21/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
S="${1:?SEED}"; H="${2:?R1M_HASH}"; TAG="${3:-v2}"; shift 2; [ $# -gt 0 ] && shift; EXTRA="$*"
T=/home/vault/b268dc/b268dc21/runs/trackv; E=/home/atuin/b268dc/b268dc21/datasets
P=/home/vault/b268dc/b268dc21/runs/trackqe/newresnet_seed$S; O=results/qe/${TAG}_s$S
M=$(ls /home/vault/b268dc/b268dc21/runs/gr_full_c02_e10_seed$S/stage1_*/stage1_best.pth)
echo "$SLURM_JOB_ID $(date -Is) $(git rev-parse HEAD) seed=$S model=$M"
MISSING=""; for p in gldv2c_val gldv2c_val_dist roxford5k rparis6k roxford5k_r1m rparis6k_r1m; do [ -f "$P/$p.npz" ] || MISSING="$MISSING $p"; done
if [ -n "$MISSING" ]; then               # build only the pools a previous run did not finish
  echo "building pools:$MISSING"
  python scripts/stage_datasets.py gldv2c gldv2d roxford5k rparis6k
  python scripts/v_precompute_qe.py --pools $MISSING --data_root "$TMPDIR/data/raw" --model "$M" --plain_pre $T/newresnet_seed${S}_q10k \
    --dist_pre $T/newresnet_seed${S}_dist --r1m_pre $T/newresnet_seed${S}_r1m --dist_emb $E/gldv2d/embeddings/newresnet_seed${S}_e512.npy \
    --r1m_emb $E/revisitop1m/embeddings/gr_full_c02_e10_seed${S}_stage1_best_${H}_s1024.npy --out $P 2>&1 | grep -v "it/s\]\|s/it\]"
fi
B="${QE_BUDGET:-20}"
C="--budgets $B --seeds 0 $EXTRA"
python scripts/train_qe_policy.py --pre $P --out $O/S_B_lr1e-3_B${B}_s0 --methods classifier value --lr 1e-3 --sup_iters 12000 --dagger $C
python scripts/train_qe_policy.py --pre $P --out $O/S_B_lr3e-4_B${B}_s0 --methods classifier value --lr 3e-4 --sup_iters 12000 --dagger $C
python scripts/train_qe_policy.py --pre $P --out $O/RS_lr3e-4_B${B}_s0 --methods rl --lr 3e-4 --iters 6000 $C
