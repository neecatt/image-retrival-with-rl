# Progress log

What was found, what was changed, and the evidence. Newest entries at the top. Milestone status lives in [ROADMAP.md](ROADMAP.md); this file keeps the detail.

## Open problems (as of 2026-10-09)

| # | Problem | Why it matters | Status |
|---|---|---|---|
| P1 | Stage 1 baseline is far below published numbers: ROxford5k Medium 17.72 / Hard 8.60 mAP (official scoring) vs roughly 60+ Medium reported for GeM-ResNet50 trained on SfM-120k | Any reward-tuning gain on top of a weak baseline is hard to interpret, and reviewers will compare against published baselines | Not diagnosed. Suspects: batch size 8 with PK sampling = only **2 landmarks per batch** (few negatives for FastAP), 1024² centre-crop training, no whitening. Unverified |
| P2 | Stage 2 hyperparameters were tuned against a gradient ~20× too small (see F1) | `consistency_weight: 5.0` and `lr: 2e-6` are probably badly balanced now | Must be re-tuned on validation, not on ROxford |
| P3 | Checkpoint selection signal is weak: "validation reward" is mAP inside each batch of 8 (gallery of ~4 images), close to 1.0 | Cannot reliably pick the best epoch | Fixed (F5), verified in smoke run 4515494 |
| P4 | README Stage 1 numbers (16.97 Medium / 3.59 Hard) do not match a rerun of `stage1_epoch_13_best.pth` (18.29 / 9.33 with the old scoring) | Historical results cannot be reproduced as stated | Likely a different checkpoint or older code. The run folders on tinyx have not been copied yet |
| P5 | RL advantage is tiny: reward and baseline differ by <0.01 on average with `sample_std` 0.01, and the policy loss reaches ±135 so `gradient_clip: 1.0` probably clips every step | The update may be mostly clipped noise, and the consistency term is now negligible | Needs gradient-norm logging and a validation sweep of `sample_std`, clip and consistency weight |
| P6 | Gallery embeddings are sampled but receive no policy gradient; memory-bank entries come from older parameters | The objective is "query-only, given a noisy gallery", not full expected AP | Acceptable if stated explicitly in the paper; not changed |

## 2026-10-09: Stage B correctness audit

### Smoke run of train.py (job 4515494, 5 min 43 s, rtxpro6k)
Stage 2 from `stage1_epoch_13_best.pth`, 50 batches, then full SfM validation. Not a research result.
- Everything ran: staging, compiled model + `--resume` (F6), unified RL step (F1, F2), SfM val selection (F5), checkpoint saving.
- SfM-120k val after 50 RL steps: **mAP 64.59**, success@1 93.28 (6,403 queries). The Stage 1 value has not been measured yet, so this is not a gain.
- Observation: SfM val mAP is ~65 while ROxford Medium is ~18, so the model fits SfM-like data but transfers poorly (relevant to P1).
- Train reward falls from ~1.0 to 0.83 during the epoch because the memory bank fills to 4,096 items and the task gets harder, not because the model degrades.
- Speed: ~7.5 it/s at batch 8, 1024² after ~50 s of compile warm-up; 19 GB GPU memory.
- Outputs: `results/smoke_train_4515494/` (run config, git revision and diff, metrics). Checkpoints stayed in TMPDIR on purpose.

### F5. Checkpoint selection now uses full SfM-120k validation mAP (fixed)
- **Problem:** "validation" was mAP or FastAP loss inside random PK batches of 8. The batches are redrawn every epoch without a fixed seed, so the score is noisy from epoch to epoch, and with a gallery of ~4 images the RL reward sits near 1.0.
- **Fix:** `evaluation/sfm_val.py` queries each of the 6,403 val images against all others (self excluded, positives = same 3D-model cluster) with the official trapezoidal AP. Stage 1 and Stage 2 both select on it by default (`training.selection_metric: legacy` restores the old behaviour). Checkpoints store the score.
- **Evidence:** on random data the vectorised mAP matches the per-query official formula to ~1e-8, and queries without positives are excluded.

### F6. Resuming into a compiled model (fixed, verified in job 4515494)
- **Problem:** `train.py` calls `torch.compile` before `--resume`, and the compiled model expects `_orig_mod.`-prefixed keys. The Stage 1 checkpoint has plain keys, so a strict load would fail.
- **Fix:** `utils.helpers.load_checkpoint` strips the prefix and loads into the underlying module.

### Tooling added
- `training.<stage>.max_batches_per_epoch` caps batches per epoch for pilots; leave it unset for real runs.
- `scripts/make_run_config.py` writes a run-specific config with dotted overrides. The canonical configs still contain tinyx output paths, so always override `system.*_dir` and `dataset.root_dir`.

### F1. RL estimator gradient was ~20× too small (fixed)
- **Problem:** in `RLTrainer._sample_on_sphere` the sampled action was not detached inside the log-probability `cos(sample, mean) / σ²`. The score-function (REINFORCE) estimator requires the action to be held fixed, so an extra reparameterisation term entered the gradient.
- **Evidence:** `scripts/audit_estimator.py` compares the estimator with the exact gradient on a tractable problem using the same sampler. At the real setting (d=512, σ=0.01), the old code had direction 0.999 (correct) but magnitude **0.047×** the true gradient. After the fix the real code path gives direction 0.999, magnitude **0.973×**.
- **Effect on old runs:** the policy term was ~20× weaker relative to the consistency term (weight 5.0). Adam rescales the total step, but not this balance. Old Stage 2 runs were effectively dominated by the consistency term.
- **Limit of the test:** single query and a smooth reward. It does not cover the AP reward, the noisy gallery or the memory bank.

### F2. Baseline depended on whether AMP was on (fixed)
- **Problem:** the AMP branch used the configured `moving_average` baseline; the fp32 branch always used an independent second sample. Stage 2 never set `use_amp`, so old runs used the second sample even though the config said `moving_average`.
- **Fix:** one code path for both. The baseline is `sample`, `moving_average` or `none`, and an unknown name raises an error. Sampling and log-probs run in fp32. Configs now say `baseline: "sample"`, matching what the old runs actually did.
- **Evidence:** the baseline is valid. In the tractable test the gradient direction is 0.999 with the second-sample baseline and 0.90 without any baseline.

### F3. Checkpoints were selected on the test benchmark (fixed)
- **Problem:** Stage 2 evaluated ROxford5k every epoch and saved `stage2_best_roxford.pth` at the best ROxford score. The README's headline Stage 2 results use that checkpoint, so their gains are selected on the test set.
- **Fix:** per-epoch ROxford evaluation is off unless `training.stage2.monitor_benchmark: true`, and is labelled diagnostic only. Selection uses SfM-120k validation (but see P3).

### F4. Evaluation AP did not follow the official Revisited protocol (fixed)
- **Problem:** `evaluate.py` used non-interpolated AP. The official revisitop `compute_ap` integrates precision with trapezoids.
- **Evidence:** `scripts/audit_eval_vs_official.py` checks hand cases (e.g. positives at ranks 1 and 3: old 0.8333, official 0.7917) and rescored the saved Stage 1 embeddings (job 4515418): Medium 18.29 → **17.72**, Hard 9.33 → **8.60**. The old scoring was ~0.6–0.7 points high.
- **Fix:** `evaluate.py` now uses the official formula. On the real embeddings the per-query difference against the official code is 0. The official source was fetched from github.com/filipradenovic/revisitop and compared line by line.
- **Unchanged:** Recall@K in `evaluate.py` is "fraction of positives in the top K", not "any positive in the top K". Label it that way in any table.

### Other checks that passed
- Query bounding-box crops are applied; query images are not in the database (Revisited lists keep them separate).
- Queries without positives are skipped, as in the official code. ROxford queries `balliol_000187` and `hertford_000082` have no easy positives but do have hard ones, so they count in both protocols.
- SfM-120k train and val share no images. In RL training, queries and gallery are disjoint halves of the batch.

## 2026-10-08: Moving to Alex (Stage A)

- **Checkpoints:** `stage1_epoch_13_best.pth` and `stage2_epoch_5.pth` copied from tinyx via the user's PC. Byte sizes and sha256 match the originals. Direct SSH from Alex to tinyx is refused.
- **Datasets:** downloaded again as packed archives to `$HPCVAULT/datasets` (linked as `data/archives`), about 45 GB. Byte sizes match the servers and sha256 values are in `datasets/logs/`. Not unpacked on shared storage: WORK has ~470k of 500k group inodes used.
- **Archive contents vs metadata:** all 91,642 train and 6,403 val SfM-120k hashes are in `ims.tar.gz` (job 4504546). All ROxford5k (5,063) and RParis6k (6,392) listed images are in their archives. 20 extra Paris images are unreferenced and ignored.
- **Format mismatch fixed:** the official `gnd_*.pkl` uses `imlist/qimlist`; the loader wants `roxford5k.pkl` with `db_images/query_images`. `scripts/convert_revisited_gnd.py` produces it without changing the data.
- **Staging:** `scripts/stage_datasets.py` extracts into the job's TMPDIR with path checks; Paris is flattened into `jpg/`. Tested for all three datasets (jobs 4504574, 4504625). SfM-120k takes ~160 s and ~39 GB.
- **Environment:** `~/venvs/rl-env` (module `python/3.12-base`, pinned `requirements.txt`, torch 2.9.1+cu128). The `python/3.12-conda` module and conda env used by old job scripts no longer exist.
- **Smoke run:** unmodified `evaluate.py` on the Stage 1 checkpoint ran end to end on one GPU (job 4504653).
- **Cluster notes:** a40/a100 queues were booked out for a day; `rtxpro6k` started jobs within seconds. Alex has no CPU-only partition.
