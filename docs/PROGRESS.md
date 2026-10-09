# Progress log

What was found, what was changed, and the evidence. Newest entries at the top. Milestone status lives in [ROADMAP.md](ROADMAP.md); this file keeps the detail.

## Open problems (as of 2026-10-09, 23:30)

| # | Problem | Why it matters | Status |
|---|---|---|---|
| P1 | Stage 1 baseline is weak: ROxford5k Medium 28.30 / Hard 12.21 (official scoring, long side 1024), only +4.2 Medium over an untrained ImageNet ResNet50-GeM (24.14). Published fine-tuned GeM systems report far higher numbers, though with multi-scale testing and whitening that we do not use | Any reward-tuning gain on top of a weak baseline is hard to interpret | **Largely fixed (C1).** New Stage 1 (3 seeds): ROxford M/H 42.27/19.41, RParis M/H 59.87/32.94, vs 28.30/12.21 and 43.75/19.80 before. Still below published systems that use multi-scale testing and whitening |
| P2 | **Stage 2 RL lowers held-out SfM val mAP in every setting tried** (B2), and the historical `stage2_epoch_5.pth` is worse than Stage 1 on every metric (B1) | The central claim (RL improves retrieval) is currently not supported | Open. Strong anchoring stops the damage but gives no clear gain: best +0.39 (one seed, 1,000 steps, B4), within likely noise. Re-test on the new Stage 1 with longer runs and several seeds; reconsider the reward and gallery design |
| P3 | Checkpoint selection signal is weak: "validation reward" is mAP inside each batch of 8 (gallery of ~4 images), close to 1.0 | Cannot reliably pick the best epoch | Fixed (F5), verified in smoke run 4515494 |
| P4 | README Stage 1 numbers (16.97 Medium / 3.59 Hard) do not match a rerun of `stage1_epoch_13_best.pth` (18.29 / 9.33 with the old scoring) | Historical results cannot be reproduced as stated | Likely a different checkpoint or older code. The run folders on tinyx have not been copied yet |
| P5 | Policy gradient norm is 100–300 vs 0.06–0.08 for the consistency term; `gradient_clip: 1.0` clips 90–100% of steps | The consistency regulariser has no practical effect; updates are clipped to unit norm every step | Measured (B2). Addressed together with P2 |
| P6 | Gallery embeddings are sampled but receive no policy gradient; memory-bank entries come from older parameters | The objective is "query-only, given a noisy gallery", not full expected AP | Acceptable if stated explicitly in the paper; not changed |

## 2026-10-09/10: New Stage 1 baseline and matched comparison

### C1. New Stage 1 baseline, 3 seeds (jobs 4516394, 4516530, 4516531; ~4 h 10 min each on one RTX PRO 6000)
ImageNet ResNet50 + GeM + 512-d head, FastAP, batch 32 (8 landmarks × 4), Adam lr 1e-5, cosine schedule, 20 epochs, 1024² crops. Best epoch chosen on SfM-120k val; benchmarks scored once afterwards (official scoring, longer side 1024).

| Seed | best epoch | SfM val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|---|
| 42 | 19 | 83.03 | 42.38 | 18.85 | 60.60 | 33.70 |
| 1 | 5 | 83.05 | 43.38 | 20.20 | 60.22 | 32.32 |
| 2 | 15 | 83.03 | 41.04 | 19.17 | 58.80 | 32.81 |
| **mean ± std** | | **83.04 ± 0.01** | **42.27 ± 1.17** | **19.41 ± 0.70** | **59.87 ± 0.95** | **32.94 ± 0.70** |
| old Stage 1 (B1) | | 64.59 | 28.30 | 12.21 | 43.75 | 19.80 |

- **+14.0 ROxford Medium and +16.1 RParis Medium** over the old Stage 1, from batch composition alone (2 → 8 landmarks per batch).
- SfM val plateaus at ~83 after ~5 epochs; later epochs change it by <0.5, so best-epoch choice is close to arbitrary between seeds. 10 epochs would likely be enough.
- Checkpoints: `$HPCVAULT/runs/s1_b32_lr1e-5_seed{42,1,2}/` (rolling best/last plus every 5th epoch). Records: `results/runs/s1_b32_lr1e-5_seed*/`.

### C2. Matched comparison design (grid submitted 2026-10-09 23:30, jobs 4518353–4518359)
Question: does reward tuning (RL) beat simply continuing metric learning (ML) from the same Stage 1 model with the same data budget?
- Init: seed-42 Stage 1 best checkpoint. Each arm gets a small validation grid, then the best setting of each is repeated from all three Stage 1 seeds.
- Budget: ~91.6k training images per run (one pass over SfM-120k), split into 2 epochs with SfM val after each. ML: 2 × 1,432 batches of 32. RL: 2 × 5,728 batches of 8, or 2 × 1,432 batches of 32.
- ML grid: constant lr 1e-6, 3e-6, 1e-5 (the smoke run showed the default 1e-4 drops val mAP within 20 steps).
- RL grid (sample_std 0.05, no clipping): consistency weight 5,000 / 500 at lr 2e-6, weight 5 at lr 5e-7, and weight 5,000 at batch 32.
- RL costs more per image (a second sample for the baseline plus the frozen reference forward pass); GPU time is recorded per run.

### C3. Validation grid results (jobs 4518353–4518359; init = seed-42 Stage 1, SfM val 83.03)

| Arm | Setting | SfM val after epoch 1 / 2 | Best Δ | Wall time |
|---|---|---|---|---|
| ML | lr 1e-6 | 82.82 / **82.96** | −0.07 | 18 min |
| ML | lr 3e-6 | 82.67 / 82.75 | −0.28 | 18 min |
| ML | lr 1e-5 | 82.14 / 81.98 | −0.89 | 17 min |
| RL | batch 8, consistency 5,000, lr 2e-6 | **83.15** / 82.98 | +0.12 | 38 min |
| RL | batch 8, consistency 500, lr 2e-6 | 82.51 / 82.41 | −0.52 | 39 min |
| RL | batch 8, consistency 5, lr 5e-7 | 82.75 / 82.56 | −0.28 | 38 min |
| RL | batch 32, consistency 5,000, lr 2e-6 | **83.25** / killed | +0.22 | 34 min (OOM) |

- All differences are within a few tenths of a point from one seed; whether any is real needs the seed repeats.
- Selected on validation: ML lr 1e-6; RL batch 32, consistency 5,000, lr 2e-6.
- **Host-memory OOM:** the batch-32 RL job was killed by the host-RAM limit (~86.7 GB of 92 GB) during validation after epoch 2; the batch-8 RL job peaked at the same level. Likely cause: 16 loader workers on each of three loaders. Seed runs use 8 workers. The epoch-1 checkpoint survived.
- With consistency weight 5,000 the consistency gradient (13–24) is now comparable to the policy gradient (60–120), unlike pilot 1.

### C4. Seed repeats with final evaluation (jobs 4518806–4518811)
Best setting of each arm from each of the three Stage 1 seeds, benchmarks scored once on the validation-selected epoch. Results pending.


### B3. Stage 1 pilot: batch size and learning rate (job 4515831, 50 min)
From ImageNet, batch 32 = 8 landmarks × 4 images (old runs: 8 = 2 × 4), 1024² crops, FastAP, 3 short epochs of 1,000 batches (~96k images, about one pass over SfM-120k). SfM-120k val mAP after each epoch; no benchmark evaluated.

| LR | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| 1e-4 | 71.02 | 73.82 | 78.80 |
| **1e-5** | **76.40** | **78.31** | **79.28** |
| 1e-6 | 66.25 | 70.44 | 71.34 |

- All three beat the old Stage 1 (64.59) on held-out validation within one pass over the data, so the 2-landmark batch is a likely main cause of P1. LR was not the main factor (1e-4 also reaches 78.8).
- Chosen on validation: batch 32, lr 1e-5. Full runs: `scripts/train_stage1_job.sh`, 20 epochs, checkpoints in `$HPCVAULT/runs/<run>`.
- Pilot checkpoints (9.9 GB) are in `$HPCVAULT/runs/pilot_s1_4515831/`; they are disposable once the full runs finish.

### B4. Stage 2 pilot 2: anchoring (job 4515890, 19 min; one seed, 1,000 batches, sample_std 0.05, no clipping)
Starting point: Stage 1 SfM val mAP 64.59.

| Run | consistency weight | LR | SfM val mAP | Δ | consistency grad norm | policy grad norm |
|---|---|---|---|---|---|---|
| f | 5 | 5e-7 | 64.76 | +0.17 | 0.04 | 147 |
| g | 5,000 | 2e-6 | 64.98 | +0.39 | 20.0 | 149 |
| h | 50,000 | 2e-6 | 64.58 | −0.01 | 72.9 | 145 |
| i | 50,000 | 5e-7 | 64.60 | +0.01 | 33.4 | 147 |

- Anchoring (or a low LR) prevents the drop seen in B2, but no setting gives a gain that stands out from single-seed noise.
- Spec: `scripts/sweeps/stage2_pilot2.txt`; logs: `results/pilot_s2_sweep_4515890/`.

### B1. Reference evaluation under the corrected protocol (job 4515706, 13 min)
Official Revisited scoring, single scale. SfM-120k val = leave-one-out mAP over 6,403 images (selection metric). `short`/`long` = shorter/longer image side resized to 1024.

| Model | SfM val mAP | ROxf M (short) | ROxf H (short) | ROxf M (long) | ROxf H (long) | RPar M (long) | RPar H (long) |
|---|---|---|---|---|---|---|---|
| ImageNet ResNet50 + GeM, no training (2048-d) | 59.12 | 18.09 | 4.90 | 24.14 | 5.66 | 44.76 | 21.16 |
| Stage 1 `stage1_epoch_13_best.pth` (512-d) | **64.59** | 17.72 | 8.60 | **28.30** | **12.21** | 43.75 | 19.80 |
| Stage 2 `stage2_epoch_5.pth` (512-d) | 57.73 | 14.30 | 5.46 | 20.40 | 6.01 | 40.22 | 18.81 |

RParis short-side numbers (Medium/Hard): ImageNet 43.14/19.28, Stage 1 40.09/17.82, Stage 2 35.13/17.15.

- **Protocol matters:** resizing by the longer side adds ~10 points to Stage 1 ROxford Medium. `evaluate.py` now does this by default (`dataset.eval_resize: long`, commit 182252c).
- **Stage 1 barely beats no training:** +5.5 SfM val, +4.2 ROxford Medium and −1.0 RParis Medium versus untrained ImageNet features.
- **The historical Stage 2 checkpoint is worse than Stage 1 on all metrics.** The README's Stage 2 gains came from `stage2_best_roxford.pth`, which was selected on ROxford (F3) and has not been copied.
- These are the first reference numbers; they were not used to choose any setting.

### B2. Stage 2 pilot sweep (job 4515798, 16 min; one seed, 400 batches each from Stage 1, warm-up 100 steps)
Starting point: Stage 1 SfM val mAP **64.59**.

| Run | sample_std | clip | LR | SfM val mAP | Δ vs Stage 1 | policy grad norm | consistency grad norm | steps clipped | mean abs advantage |
|---|---|---|---|---|---|---|---|---|---|
| a | 0.01 | 1.0 | 2e-6 | 60.57 | −4.02 | 256 | 0.081 | 90% | 0.026 |
| b | 0.05 | 1.0 | 2e-6 | 62.50 | −2.09 | 162 | 0.064 | 100% | 0.169 |
| c | 0.01 | off | 2e-6 | 62.10 | −2.49 | 292 | 0.085 | 0% | 0.025 |
| d | 0.05 | off | 2e-6 | 63.04 | −1.55 | 158 | 0.070 | 0% | 0.169 |
| e | 0.05 | 1.0 | 1e-5 | 16.99 | −47.60 | 96 | 0.073 | 100% | 0.135 |

- Every setting lowers held-out mAP; a higher LR collapses the model. Larger noise (0.05) gives more informative advantages (91% non-zero vs 58%) and hurts less.
- The consistency term is ~3,000× weaker than the policy gradient, so it does not anchor anything at weight 5.0.
- One seed and 400 steps only: the effect could be an early transient, but nothing here supports running long RL jobs with these settings.
- Logs: `results/pilot_s2_sweep_4515798/`.

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
