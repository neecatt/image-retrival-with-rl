# Progress log

What was found, what was changed, and the evidence. Newest entries at the top. Milestone status lives in [ROADMAP.md](ROADMAP.md); this file keeps the detail.

### D1. DINOv2 ViT-B/14 as second backbone (jobs 4575620–4575624)
Off-the-shelf (no training, no projection, long side 1024):

| Model | GLDv2 val | ROxf M / H | RPar M / H |
|---|---|---|---|
| ImageNet ResNet50-GeM | 29.39 | 24.14 / 5.66 | 44.76 / 21.16 |
| DINOv2 ViT-B/14, GeM over patch tokens | 44.18 | 32.90 / 9.06 | 69.14 / 48.42 |
| **DINOv2 ViT-B/14, CLS token** | 63.33 | **64.51 / 41.81** | **88.29 / 75.24** |
| (GLDv2-trained ResNet50 Stage 1, 5 seeds) | 63.01 | 47.93 / 24.88 | 74.22 / 54.73 |

GLDv2-clean Stage 1 pilot (FastAP, batch 128, 448 px, 3 x 1,000 batches; GLDv2 val after epoch 3): GeM lr 1e-5 73.67, GeM lr 3e-6 73.19, **CLS lr 1e-5 73.92**, CLS lr 3e-6 73.47. ~7.3 min per 1,000 batches, 54–55 GB GPU memory.
- Untrained DINOv2 with the CLS token already beats our trained ResNet50 on both benchmarks. GeM over ViT patch tokens is poor (ViT tokens are signed and GeM clamps them).
- **Full Stage 1 submitted:** CLS, lr 1e-5, 6 epochs (~9 h), job 4575802. Then the matched G10-style comparison (PL RL vs FastAP-bank / Smooth-AP) and Track V on the DINOv2 descriptor.

## 2026-10-15: Track V, RL for budgeted geometric verification

### V1. First results (precompute job 4575648, 22 min; policies job 4575657, 8 min)
Fixed descriptor: GLDv2 FastAP-bank seed 42 (`gcmp_fapb_lr3e-7_seed42`). Top-100 shortlist per query; the method verifies B items one at a time (seeing each outcome), then verified matches (≥ 35 inliers) move to the top by inliers; reward/metric = official AP of the final ranking. Policies are trained on GLDv2-clean val queries of 2/3 of the landmarks (2,041 queries), selected on the other 1/3 (959 queries, "gld_val"), and evaluated once on ROxford/RParis. RL = MLP policy over 10 per-candidate features, REINFORCE with leave-one-out baseline (16 rollouts per query); classifier = same MLP and features trained with BCE to predict a verified match, chosen greedily and adaptively. 3 policy-training seeds each (std ≤ 0.2 except ROxf Hard ≤ 0.7).

| B | method | gld_val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|---|
| – | none (global ranking) | 64.67 | 54.35 | 30.35 | 75.37 | 56.36 |
| 5 | top-B by similarity | 65.08 | 55.17 | 32.49 | 75.41 | 56.37 |
| 5 | classifier | 65.38 | 54.99 | 31.81 | 75.44 | 56.40 |
| 5 | **RL** | **65.84** | **55.26** | 30.04 | **75.57** | 56.43 |
| 10 | top-B | 65.62 | 55.76 | 33.07 | 75.47 | 56.46 |
| 10 | classifier | 66.07 | 56.44 | **33.54** | 75.56 | 56.51 |
| 10 | **RL** | **66.89** | **57.38** | 31.86 | **75.82** | 56.52 |
| 20 | top-B | 66.06 | 56.83 | 33.95 | 75.61 | 56.54 |
| 20 | classifier | 66.89 | 58.22 | **34.76** | 75.78 | 56.58 |
| 20 | **RL** | **67.27** | **60.17** | 33.76 | **76.17** | **56.87** |
| 100 | verify all | 65.05 | 64.85 | 43.10 | 76.89 | 57.71 |

RL − classifier per policy seed: gld_val +0.34 to +0.83 (every seed, every budget); ROxf M +0.15 to +2.08 (every seed); RPar M +0.12 to +0.44 (every seed); **ROxf Hard −0.56 to −1.98 (every seed)**; RPar H −0.01 to +0.33.

- **First setting where RL beats its strongest matched non-RL alternative** on the held-out selection split and on ROxford/RParis Medium, at every budget and in every seed. At B = 20, RL improves ROxford Medium by +5.8 over no verification, +3.3 over the standard top-B re-ranking and +2.0 over the adaptive classifier.
- **But RL is worse on ROxford Hard.** The reward counts every same-landmark image equally, so the policy spends its budget on items whose promotion raises AP the most, which are mostly easy positives (junk under Hard). The classifier, which only predicts "will verify", keeps more hard positives.
- **Verify-all is not an upper bound on GLDv2** (65.05 < budgeted methods): promoting every verified match also promotes false matches and reorders true positives by inliers. On ROxford it is far above every budgeted method (64.85 Medium), so there is a lot of room.
- **Caveats:** the policy-seed std understates the real uncertainty, which comes from the 70 benchmark queries (paired bootstrap over queries still to do). Hyperparameters: a single setting, not tuned. Stronger non-RL baseline still to add: a learned myopic value (regress the one-step AP gain of verifying an item, then choose greedily), which separates "RL optimises the final-AP objective" from "RL plans several steps".

## 2026-10-14: Research review of roadmap and RL plan (no training runs)

- Independently aggregated five `final_eval.json` files per arm from `results/runs/gcmp_fapb_lr3e-7_seed*`, `gcmp_pl_t0.05_lr1e-6_seed*`, and `gcmp_sap_t0.05_lr3e-7_seed*`. G10 means reproduced: FastAP-bank ROxf M/H 52.63/28.78, RPar M/H 76.88/57.72; PL RL 52.15/27.66, 76.19/56.97; Smooth-AP 52.36/28.06, 76.19/56.99. This supports no demonstrated RL advantage on this binary-label setup, not a universal claim that RL cannot help retrieval.
- Track N review recommendation: separate acquisition and optimization. Compare greedy and stochastic candidate acquisition under both a verified-label AP/listwise surrogate and PL reward tuning; include training the strongest surrogate on the exact verified pairs collected by RL. Match unique verifier calls, candidate pool, bank, initialization, selection, and report total compute/cache hits. Improvements from adaptive labels alone should not be attributed to the RL objective.
- Verifier caveat: N0 low inlier counts establish low geometric support, not that all dropped images have wrong landmark labels. Treat failed/unverified matches as uncertain rather than automatic negatives; calibrate strong-match evidence on manually reviewed easy/hard/confusable pairs. The local N1 JSON confirms 87.9 pairs/s and same/different-label AUC 0.824, but label agreement is not independent oracle accuracy.
- Next objective candidate: shortlist utility measured after a fixed verifier/reranker at a fixed verification budget. First test headroom on GLDv2 (B0 only ruled out saturated SfM training); compare against a top-k recall surrogate and verifier-label distillation on identical data. Cached per-pair relevance still permits differentiable supervised controls; non-differentiability alone does not imply an RL advantage.
- Before PPO/reuse: freeze candidate IDs and gallery representations for each rollout update block, save old log-probabilities, and validate a small on-policy RLOO baseline first. Refreshing a bank/index while reusing old rankings can invalidate likelihood-ratio comparisons. Audit query/image ID self-exclusion and duplicates across the bank.
- Literature additions: ELVA (https://arxiv.org/abs/2606.20280, ECCV 2026) already studies ranking-driven RLVR for multimodal retrieval; compare with RELER (https://arxiv.org/abs/2610.07731) and Retrieval-GRPO (https://arxiv.org/abs/2511.13885) before broad RLVR/retrieval novelty claims. These sources do not establish that geometric-verifier-guided landmark tuning is novel.
- Documentation follow-up: the open-problem table below is a dated snapshot; P6 is superseded for the implemented PL path. ROADMAP Stage E remains marked planned despite completed GLDv2 work; RL_RESEARCH_PLAN has stale pending G9b entries and older theory/gates. Preserve historical decisions but label superseded claims and use the latest matched evidence. Oxford/Paris have been inspected across multiple phases: reserve a new locked evaluation (e.g. R1M and held-out landmark identities) for the next claim and report benchmark-overlap checks and paired uncertainty.

These are review recommendations, not completed fixes or authorization to submit new jobs. No model/data changes were made.

## Open problems (dated snapshot, 2026-10-10; later status in the entries below)

| # | Problem | Why it matters | Status |
|---|---|---|---|
| P1 | Stage 1 baseline is weak: ROxford5k Medium 28.30 / Hard 12.21 (official scoring, long side 1024), only +4.2 Medium over an untrained ImageNet ResNet50-GeM (24.14). Published fine-tuned GeM systems report far higher numbers, though with multi-scale testing and whitening that we do not use | Any reward-tuning gain on top of a weak baseline is hard to interpret | **Largely fixed (C1).** New Stage 1 (3 seeds): ROxford M/H 42.27/19.41, RParis M/H 59.87/32.94, vs 28.30/12.21 and 43.75/19.80 before. Still below published systems that use multi-scale testing and whitening |
| P2 | **Stage 2 RL lowers held-out SfM val mAP in every setting tried** (B2), and the historical `stage2_epoch_5.pth` is worse than Stage 1 on every metric (B1) | The central claim (RL improves retrieval) is currently not supported | **Answered for the current method (C4): no measurable gain.** From the new Stage 1, 3 seeds, matched budget: RL vs continued metric learning differs by −0.45 to +0.24 points, all within seed spread. RL is a valid fine-tuning method that does no harm when anchored, but does not beat the control. Improving it needs a method change (Stage D) |
| P3 | Checkpoint selection signal is weak: "validation reward" is mAP inside each batch of 8 (gallery of ~4 images), close to 1.0 | Cannot reliably pick the best epoch | Fixed (F5), verified in smoke run 4515494 |
| P4 | README Stage 1 numbers (16.97 Medium / 3.59 Hard) do not match a rerun of `stage1_epoch_13_best.pth` (18.29 / 9.33 with the old scoring) | Historical results cannot be reproduced as stated | Likely a different checkpoint or older code. The run folders on tinyx have not been copied yet |
| P5 | Policy gradient norm is 100–300 vs 0.06–0.08 for the consistency term; `gradient_clip: 1.0` clips 90–100% of steps | The consistency regulariser has no practical effect; updates are clipped to unit norm every step | Measured (B2). Addressed together with P2 |
| P6 | Gallery embeddings are sampled but receive no policy gradient; memory-bank entries come from older parameters | The objective is "query-only, given a noisy gallery", not full expected AP | Superseded for the PL path (E1): batch gallery items receive gradient through the PL scores; bank entries are still older detached means |

### B0. Track B headroom check before building (CPU, 2026-10-11; `scripts/analysis/trackb_headroom.py` on the A0/A0b pairs)
Track B reward = AP after the verifier re-ranks the global top-20 shortlist. Measured on the same seed-42 Stage 1 shortlists as A0/A0b (binary SfM labels; AP@20 is low in absolute terms because clusters have many more than 20 positives).

| Pool | recall@20 of positives | queries with recall@20 < 1 | AP@20 global → after re-ranking (T = 25) | queries whose AP changes with re-ranking |
|---|---|---|---|---|
| train (2,000 queries) | 97.2% | 9.8% | 12.29 → 12.41 | 8.9% |
| val (6,403 queries) | 90.4% | 34.1% | 45.78 → 46.45 | 28.0% |

- **On the training pool the shortlist is already saturated.** 97% of the top-20 slots that could hold positives do, and re-ranking changes the reward for only 9% of queries. The part of the Track B reward that the global model controls (getting verifiable positives into the shortlist) is almost constant on training data, so RL would get even less signal than with plain AP.
- On unseen images there is more room (recall@20 90%, a third of queries affected), but training happens on the memorised pool.
- **Decision: Track B is not built on SfM-120k.** Like A0/A0b, the limit is the training data, not the algorithm: Stage 1 has fitted SfM-120k train, so any reward computed there is near its ceiling. A reward-tuning gain needs a training pool the model cannot fit, e.g. GLDv2 (≈ 4M images, noisy labels; its "clean" subset was itself built by local-feature verification).

## 2026-10-13: Track N, verifier-in-the-loop RL on noisy full GLDv2

### N3. Track N pilot, seed 42 (jobs 4571069–4571075; full-scale smoke 4571067 passed)
Init: noisy-label Stage 1 (GLDv2 val 53.21). 4 rounds × 10,000 queries × 10 candidates (≈ 400k unique verified pairs per arm), 4 × 716 training steps, bank 4,096, match = ≥ 35 inliers, failed same-label pairs uncertain. Verification ≈ 140 pairs/s; a verifying round takes ~38 min (mostly embedding the 797k pool and verification). GLDv2 val mAP after each round:

| Acquisition × objective | r1 | r2 | r3 | r4 | best | unique verified pairs |
|---|---|---|---|---|---|---|
| greedy × FastAP (re-mining) | 55.05 | 55.74 | **55.99** | 55.78 | 55.99 | 391,164 |
| greedy × PL RL | 55.33 | 55.76 | 55.61 | **55.91** | 55.91 | 391,859 |
| sample × FastAP | 55.10 | 55.60 | 55.97 | **56.17** | **56.17** | 398,154 |
| **sample × PL RL** | 55.35 | 55.75 | 55.97 | **56.07** | 56.07 | 398,201 |
| static × FastAP | 55.28 | 55.78 | **55.92** | 55.81 | 55.92 | 396,292 |
| replay of sample×PL lists × FastAP | 54.91 | 55.57 | 55.73 | **55.84** | 55.84 | 398,201 (reused) |
| *references* | | | | | | |
| noisy-label FastAP-bank (N2, no verifier) | | | | | 55.38 | 0 |
| perfect-cleaning Stage 1 (N2b) | | | | | 56.82 | – |

- **The RL objective ties FastAP again.** With the same sampled acquisition, PL RL 56.07 vs FastAP 56.17 (−0.10); with greedy acquisition 55.91 vs 55.99 (−0.08); FastAP trained on exactly RL's verified lists reaches 55.84. **The Track N gate (RL > sample×FastAP and > replay×FastAP by ≥ 0.3) is not met.**
- **The verifier helps a little.** Every verifier arm beats the noisy-label control (55.38) by +0.5 to +0.8 val, closing roughly a third to a half of the 1.4-point gap to the perfect-cleaning Stage 1. That reference is a from-scratch Stage 1, so this is only a rough scale.
- **Acquisition matters little at this budget.** Sampled ≥ greedy ≥ static by 0.1–0.2, all within single-seed noise. Sampled arms were still rising at round 4.
- No seeds run: the RL-specific gate failed, and a +0.5 verifier gain over noisy labels at this budget is not the claim Track N was built for.

### N2b. Second OOM, root cause fixed; "perfect cleaning" reference (2026-10-14)
- **Jobs 4561438–4561443 OOM-killed again** after ~23 min, this time while verifying round 1 (100k pairs). Root cause in `Verifier._extract`: `ThreadPoolExecutor.map` submitted all ~100k image decodes at once, and decoded images (~2.5 MB each) piled up in host memory faster than the GPU extracted features. The 300-query smoke run (4561437) only loaded ~3k images, so it missed this. Fix: load 256 images at a time and verify in chunks of 5,000 pairs (identical results on a CPU test). The replay job 4561444 was cancelled (dependency never satisfied).
- **Resubmitted:** full-scale smoke run (1 round, 10,000 queries, 100k pairs; job 4570353), then the same pilot chain (4570354–58, replay 4570359).
- **"Perfect cleaning" reference finished** (job 4559935, 5 h 10 min): Stage 1 on only the GLDv2-clean images of the same 20,000 landmarks, matched step count. GLDv2 val **56.82**; ROxf M/H **44.10 / 21.42**; RPar M/H **65.12 / 45.35**. For comparison, the noisy-label Stage 1 reaches 53.21; 35.68 / 16.85; 62.05 / 40.88. So perfect label cleaning is worth +3.6 val, +8.4 / +4.6 ROxford and +3.1 / +4.5 RParis at the Stage 1 level. That is the headroom a verifier-based method can close.

### N2. First full Track N pilot failed (OOM); redesign after the research review (2026-10-14)
- **Jobs 4560131–4560136 (rl / remine / static at two lrs) were OOM-killed** after ~23 min, while embedding the 797k pool in round 1 (MaxRSS 86.7 GB of 92). Cause: 16 RANSAC workers forked after torch/CUDA were loaded, plus 12 loader workers. Fix: the RANSAC pool is forked at program start from a light module (`scripts/ransac_worker.py`), and loader workers are 8. A full-pool smoke run (job 4561437) checks memory before the pilot starts.
- **Noisy-label control finished** (FastAP-bank on the noisy labels, 2 × 1,432 steps, jobs 4560137–38): GLDv2 val 53.21 → **55.38** (lr 1e-6) / **55.32** (lr 3e-7).
- **Research review (top of this file), how it is handled:**
  - *Separate acquisition and optimisation:* done. Arms are now acquisition {greedy, sample, static, replay} × objective {fastap, pl}. The pilot runs greedy×fastap, greedy×pl, sample×fastap, sample×pl (the RL arm), static×fastap, and **replay×fastap = FastAP trained on exactly the lists and verified pairs the RL arm collected**. An RL gain counts only if sample×pl beats sample×fastap and replay×fastap; adaptive labels alone are credited to acquisition.
  - *Failed verification ≠ wrong label:* done. Failed same-label pairs are now uncertain (masked) instead of negatives; only failed pairs with a different label are negatives. Manual calibration of the 35-inlier threshold on reviewed easy/hard/confusable pairs is still open.
  - *Unique verifier calls, cache hits and compute* are logged per round (`budget_used`, minutes); a summary table will be produced with the results.
  - *Locked evaluation:* the next claim should be reported on R1M and on held-out landmark identities, not only on ROxford/RParis, which have been inspected many times. Planned for the final Track N evaluation.
  - *PPO caveats:* not using PPO; noted for later. *Literature:* ELVA (ECCV 2026) added to the novelty check list in PAPER_DRAFT.md.
  - *Documentation follow-ups:* P6 labelled superseded; ROADMAP Stage E status updated; stale G9b entry marked superseded.

### N1d. Verifier-in-the-loop trainer: GPU smoke runs (jobs 4559965–4559967, 7–8 min each)
30,000-image slice, 2 rounds × 400 queries × 10 candidates, 30 training steps per round. All three arms ran end to end. Verifier throughput inside the loop ~140 pairs/s (cache + batching); budgets equal (rl 7,960, remine 7,913, static 7,936 verified pairs). 55–66% of queries have at least one verified match among their 10 candidates. Val unchanged (53.21 → 53.29–53.41), as expected for 60 steps.
- **Full seed-42 pilot submitted** (jobs 4560131–4560138): rl / remine / static at lr 1e-6 and 3e-7 (4 rounds × 10,000 queries × 10 candidates = 400k verified pairs, 4 × 716 steps); noisy-label FastAP-bank control at lr 1e-6 and 3e-7 (2 × 1,432 steps of batch 128).

### N1a–N1c. Noisy pool, noisy-label Stage 1, batched verifier (jobs 4557471, 4557472, 4557490)
- **Noisy pool (N1a, 6 min):** 20,000 random GLDv2-clean training landmarks with all their full-GLDv2 images: 796,692 train images, of which 390,982 are in GLDv2-clean and **405,710 (50.9%) were dropped by the cleaning**; val = the 21,285 GLDv2-clean val images. `$HPCVAULT/datasets/gldv2n` (vault quota now 1,543 / 2,097 GB hard).
- **Stage 1 on noisy labels (N1b, 5 h 4 min):** lr 1e-4, batch 128, 512 px, 10 epochs. Val mAP 43.91 → 53.21. Benchmarks (long side 1024): **ROxf M/H 35.68 / 16.85, RPar M/H 62.05 / 40.88**, far below the GLDv2-clean Stage 1 (47.9 / 24.9 / 74.2 / 54.7, 80k landmarks). Checkpoint `$HPCVAULT/runs/gn_s1_b128_lr1e-4_seed42/`.
- **Batched verifier (N1c, 8 min incl. staging):** 1,024 ALIKED keypoints, LightGlue in batches of 32, MAGSAC. **87.9 pairs/s** on one RTX PRO 6000 (vs 14 per-pair). Same/different-landmark AUC **0.824**; on the same 400 pairs the 2,048-keypoint per-pair oracle reaches 0.829 (Spearman 0.82 between the two). Inliers: same landmark median 39.5 (62% ≥ 25), different landmark median 14 (6.7% ≥ 25, p99 = 34). A match threshold of ~35 inliers keeps different-landmark false matches near 1%.

### N0. Feasibility: how noisy are the images the cleaning dropped? (job 4557280, 20 min)
2,000 random GLDv2-clean training landmarks (val excluded). Per landmark, one *dropped* image (in train.csv, not in train_clean.csv) and one *kept* image, each verified against the same 3 other kept images (ALIKED + LightGlue + MAGSAC, long side 512). Best reference counts.

| Query | verified (≥ 25 inliers) | verified (≥ 100) | median best inliers |
|---|---|---|---|
| kept (GLDv2-clean) | 72.0% | 40.8% | 68 |
| dropped by the cleaning | **14.4%** | **3.1%** | **13** |

- **Gate passed.** Dropped images almost never share geometry with the kept images of their landmark. For these landmarks, full GLDv2 has 51% dropped images, so training on the noisy labels means half the "positives" are unverifiable.
- **Contact sheets** (`results/f0_gld_noise/`): unverified dropped images are a mix of unrelated scenes and other aspects of the place (statues, frescoes, interiors, landscapes, maps). Some are arguably the right landmark but not matchable. ROxford/RParis positives are matchable views of the buildings, so the verifier is a reasonable proxy for benchmark relevance. (The sheets pair each query with its first reference, not necessarily the best-matching one.)
- **Throughput is the bottleneck:** 14 pairs/s on one GPU, reading single images from the raw tars on HPCVAULT and matching pairs one by one. An equal-verifier-budget design with ~10⁵–10⁶ verified pairs needs a faster verifier (batched matching, cached features, packed images on TMPDIR), aiming at ≥ 100 pairs/s per GPU, plus array jobs.
- Built once: index of all 4,132,914 raw train images (`$HPCVAULT/datasets/gldv2/train_index.npz`: tar, offset, size).

## 2026-10-11: Track G, moving training to GLDv2-clean

### G1–G2. Download and packing
- **Why:** SfM-120k train is saturated for Stage 1 (A0, A0b, B0), so no reward computed on it leaves room for RL. GLDv2-clean is the standard modern training set for ROxford/RParis (DELG, CVNet, SuperGlobal).
- **Download (G1):** all 500 official train tars md5-verified (498 GB apparent) plus metadata in `$HPCVAULT/datasets/gldv2` (`fetch_train.sh`, resumable). Images CC-BY (per-image attribution in `train_attribution.csv`), labels CC BY 4.0. Vault charges ~2× apparent size: quota now 1,374 / 1,049 GB soft (hard 2,097). The user decided to keep the raw tars until the GLDv2 work is finished.
- **Packing (G2, job 4536648, a40, 22 min with 16 workers):** all 1,580,470 clean images found, 0 decode failures, resized to long side 512 (JPEG q90), 500 shards, 75 GB apparent, `$HPCVAULT/datasets/gldv2c`. A first attempt (4536631) used 1 worker because `nproc` honours `OMP_NUM_THREADS=1`; it was cancelled and the script now uses `SLURM_CPUS_ON_NODE`.
- **Split (seed 0):** train 1,546,857 images / 79,813 landmarks; val 21,285 images / 1,500 held-out landmarks (≥ 4 images each, at most 30 kept). No landmark is in both splits.
- **Caveat:** GLDv2 may contain Oxford/Paris landmarks; standard practice, to be stated in any write-up.

### G11. RL on top of the best surrogate model (jobs 4555169–4555174, 28–30 min each)
Init: FastAP-bank seed 42 (`gcmp_fapb_lr3e-7_seed42`, GLDv2 val 65.05). Same budget, batches, bank and selection as G10.

| Arm | Setting | ep 1 / ep 2 | best Δ vs init |
|---|---|---|---|
| PL RL | τ 0.05, lr 3e-7 | 65.11 / **65.13** | +0.08 |
| PL RL | τ 0.05, lr 1e-6 | 64.66 / 64.60 | −0.39 |
| PL RL | τ 0.02, lr 3e-7 | 64.68 / 64.58 | −0.37 |
| FastAP-bank continued | lr 1e-7 | 65.00 / 65.04 | −0.01 |
| FastAP-bank continued | lr 3e-7 | 64.96 / **65.07** | +0.02 |
| FastAP-bank continued | lr 1e-6 | 64.75 / 64.99 | −0.06 |

- **No gain.** Best RL 65.13 vs best continued control 65.07 (+0.06, gate ≥ 0.3). From the best surrogate model, neither objective moves validation within this budget. The Pinto recipe gives nothing here, as on SfM-120k (R1). No seeds run.

### G10. PL RL vs AP-surrogate controls on the same bank, 5 seeds (jobs 4553898–4553907, 32–37 min each) — **the G7/G8 claim does not hold**
Smooth-AP (sigmoid τ 0.05, lr 3e-7) and FastAP on the bank (lr 3e-7), selected on the seed-42 pilot (G9), same batches, bank, budget and selection as PL RL.

| Model (5 seeds, mean ± std) | GLDv2 val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| GLDv2 Stage 1 | 63.01 ± 0.16 | 47.93 ± 1.90 | 24.88 ± 2.93 | 74.22 ± 1.44 | 54.73 ± 0.58 |
| + FastAP continuation, no bank (G8 control) | 62.83 ± 0.15 | 48.07 ± 1.59 | 24.64 ± 2.15 | 74.17 ± 1.09 | 54.48 ± 1.06 |
| + softmax, bank | 64.13 ± 0.09 | 48.44 ± 1.84 | 23.97 ± 3.30 | 73.78 ± 1.18 | 54.92 ± 1.08 |
| + Smooth-AP, bank | 64.60 ± 0.06 | 52.36 ± 2.12 | 28.06 ± 2.48 | 76.19 ± 0.99 | 56.99 ± 0.89 |
| **+ FastAP, bank** | **65.10 ± 0.07** | **52.63 ± 2.44** | **28.78 ± 3.49** | **76.88 ± 0.94** | **57.72 ± 0.87** |
| + PL RL (AP@100 reward), bank | 64.77 ± 0.14 | 52.15 ± 1.10 | 27.66 ± 1.56 | 76.19 ± 1.10 | 56.97 ± 1.05 |

Paired differences over 5 seeds, mean ± std (t):

| Comparison | GLDv2 val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| PL − FastAP-bank | −0.33 ± 0.10 (−7.5) | −0.48 ± 1.83 (−0.6) | −1.12 ± 2.40 (−1.0) | −0.68 ± 0.41 (−3.8) | −0.74 ± 0.46 (−3.6) |
| PL − Smooth-AP | +0.17 ± 0.09 (4.1) | −0.21 ± 1.66 (−0.3) | −0.40 ± 1.79 (−0.5) | −0.00 ± 0.25 | −0.01 ± 0.23 |
| FastAP-bank − Stage 1 | +2.09 ± 0.09 | +4.70 ± 0.93 | +3.90 ± 1.29 | +2.65 ± 0.79 | +2.99 ± 0.58 |

- **The G7/G8 gain was a bank effect, not a reward effect.** Once the differentiable AP surrogates use the same bank, they match (Smooth-AP) or beat (FastAP on the bank: better on RParis in every seed, t ≈ −3.7) the PL RL policy. The G8 controls were too weak: the FastAP control had no bank, and the softmax objective transfers poorly to the benchmarks. **The §5 criterion is not met against the strongest matched control.**
- **Long-RL preview** (job 4553037, 4× budget, seed 42): GLDv2 val 64.48; ROxf M/H 52.82 / 27.34, RPar M/H 74.65 / 55.31, vs 64.61 / 51.93 / 26.89 / 74.51 / 55.28 at 1×. No meaningful change.
- **What is established:** (1) GLDv2-clean Stage 1 + a bank-based AP objective is a strong recipe (+4.7 ROxf M, +2.7 RPar M over Stage 1 for FastAP-bank, every seed); (2) PL RL with an AP reward reaches the level of AP surrogates on the same bank, but does not exceed them.

### G9. Stronger differentiable controls and a longer RL run (submitted 2026-10-13)
- **Why:** a reviewer's first question is whether any differentiable AP loss on the same setup matches PL RL. Two AP surrogates were added to `RLTrainer`, using the same all-vs-all batches and 4,096-entry bank:
  - `smoothap_control`: Smooth-AP (Brown et al., ECCV 2020), sigmoid temperature = `pl_tau`. CPU check: at a vanishing temperature it equals exact AP (1 − AP 0.6892 vs 0.6891).
  - `fastap_bank`: FastAP (the Stage 1 loss) over batch + bank, 10 bins.
- **Pilot on seed 42 (jobs 4553027–4553036):** Smooth-AP with sigmoid τ 0.01 / 0.05 × lr 3e-7 / 1e-6 / 3e-6 (6 runs); FastAP-bank at lr 3e-7 / 1e-6 / 3e-6 / 1e-5 (4 runs). PL RL had 6 tuning runs (G6, G6b), so the tuning effort is comparable.
- **Long-RL preview (job 4553037):** PL RL τ 0.05, lr 1e-6, 4 × 2,864 batches (4× the G7/G8 budget, about one GLDv2 epoch), seed 42, with final evaluation. This is a preview only; the decisive budget study will run RL and the strongest control at the same longer budget.
- **Correction:** when the user interrupted the command adding these controls, I said nothing had been written. In fact the file edits had already been applied. They were inspected (`git diff`) and tested before this commit.
- **Pilot result (seed 42, GLDv2 val, ep 1 / ep 2; init 62.89; best PL RL 64.61):**

| Control | lr 3e-7 | lr 1e-6 | lr 3e-6 | lr 1e-5 |
|---|---|---|---|---|
| Smooth-AP, sigmoid τ 0.01 | 62.84 / 63.52 | 61.71 / 57.48 | 54.33 / 50.12 | |
| Smooth-AP, sigmoid τ 0.05 | 64.45 / **64.56** | 64.20 / 64.22 | 57.27 / 59.21 | |
| FastAP on the bank | 64.85 / **65.05** | 64.68 / 65.01 | 59.04 / 59.72 | 47.38 / 52.53 |

- **On validation, FastAP with the bank beats PL RL (65.05 vs 64.61), and Smooth-AP ties (64.56).** The G7/G8 FastAP control had no bank, so part or all of the PL advantage there may come from the bank rather than the reward. Validation gains have not always transferred to the benchmarks (the softmax control gained +1.1 val and nothing on ROxford/RParis), so the benchmarks decide.
- **Submitted:** both controls at their best setting on all 5 seeds with final evaluation (jobs 4553898–4553907; `gcmp_fapb_lr3e-7_seed*`, `gcmp_sap_t0.05_lr3e-7_seed*`).

### G8. GLDv2: PL RL vs controls, 5 seeds (superseded by G10: these controls lacked the bank) (adds seeds 3–4: Stage 1 jobs 4549985 / 4549989, 9 h 57 min / 9 h 26 min; comparisons 4549986–88, 4549990–92, 24–35 min)
Same settings as G7.

| Model (5 seeds, mean ± std) | GLDv2 val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| GLDv2 Stage 1 | 63.01 ± 0.16 | 47.93 ± 1.90 | 24.88 ± 2.93 | 74.22 ± 1.44 | 54.73 ± 0.58 |
| + FastAP continuation | 62.83 ± 0.15 | 48.07 ± 1.59 | 24.64 ± 2.15 | 74.17 ± 1.09 | 54.48 ± 1.06 |
| + softmax control (bank) | 64.13 ± 0.09 | 48.44 ± 1.84 | 23.97 ± 3.30 | 73.78 ± 1.18 | 54.92 ± 1.08 |
| **+ PL RL (AP@100 reward)** | **64.77 ± 0.14** | **52.15 ± 1.10** | **27.66 ± 1.56** | **76.19 ± 1.10** | **56.97 ± 1.05** |

Paired differences over 5 seeds, mean ± std (paired t), and the smallest per-seed difference:

| Comparison | GLDv2 val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| PL − FastAP | +1.94 ± 0.11 | **+4.08 ± 1.74** (t 5.3; min +2.43) | +3.02 ± 1.79 (t 3.8; min +0.70) | **+2.03 ± 0.38** (t 11.9; min +1.55) | +2.49 ± 0.86 (t 6.5; min +1.62) |
| PL − softmax | +0.64 ± 0.11 | **+3.71 ± 0.91** (t 9.1; min +2.31) | +3.69 ± 2.42 (t 3.4; min +1.00) | **+2.42 ± 0.57** (t 9.5; min +1.55) | +2.05 ± 0.53 (t 8.6; min +1.28) |
| PL − Stage 1 | +1.76 ± 0.12 | +4.22 ± 1.64 | +2.78 ± 2.06 (min −0.26) | +1.97 ± 0.63 | +2.24 ± 0.71 |
| softmax − Stage 1 | +1.12 ± 0.17 | +0.51 ± 2.24 | −0.91 ± 2.06 | −0.45 ± 0.98 | +0.19 ± 0.87 |
| FastAP − Stage 1 | −0.18 ± 0.04 | +0.14 ± 0.87 | −0.24 ± 1.79 | −0.06 ± 0.57 | −0.25 ± 0.85 |

- **The G7 result holds on 5 seeds.** PL RL beats both matched controls in all 5 seeds on all four benchmark protocols (every per-seed difference positive; smallest +0.70 on ROxford Hard vs FastAP). The §5 success criterion is met on 5 seeds.
- Neither control moves the benchmarks (softmax improves GLDv2 val only; FastAP continuation is flat), so the gain is specific to the reward.
- Records: `results/runs/gcmp_{pl,sm,ml}_*_seed{42,1,2,3,4}/`, `results/runs/g_s1_b128_lr1e-4_seed*/`.

### G7. GLDv2: PL RL vs controls, 3 seeds with final evaluation (jobs 4543826–4543834, 25–36 min each)
Init: GLDv2 Stage 1 seeds 42 / 1 / 2 (seeds 1–2: jobs 4542597–98, 9 h 43 min each). Settings fixed from the seed-42 val pilot (G6b): PL RL and softmax at τ 0.05, lr 1e-6; FastAP continuation at lr 1e-5; batch 128, 512 px, 2 × 1,432 batches, bank 4,096, no anchor. Selection on GLDv2 val; benchmarks scored once on the selected checkpoint (official scoring, long side 1024).

| Model (3 seeds, mean ± std) | GLDv2 val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| GLDv2 Stage 1 | 63.02 ± 0.21 | 48.08 ± 2.36 | 24.46 ± 3.85 | 73.63 ± 1.24 | 54.45 ± 0.59 |
| + FastAP continuation | 62.83 ± 0.19 | 48.54 ± 1.36 | 24.71 ± 1.80 | 73.74 ± 1.23 | 54.37 ± 1.30 |
| + softmax control (bank) | 64.10 ± 0.03 | 47.27 ± 1.30 | 22.35 ± 3.45 | 73.23 ± 1.26 | 54.66 ± 1.40 |
| **+ PL RL (AP reward)** | **64.74 ± 0.13** | **51.52 ± 0.84** | **27.03 ± 1.61** | **75.86 ± 1.19** | **56.81 ± 1.43** |

Paired differences (same Stage 1 seed), mean ± std; per seed 42 / 1 / 2:

| Comparison | GLDv2 val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| PL − FastAP control | +1.91 ± 0.13 | **+2.98 ± 0.57** (+2.43 / +3.56 / +2.94) | +2.32 ± 1.44 (+0.70 / +2.79 / +3.47) | **+2.13 ± 0.38** (+2.19 / +2.47 / +1.72) | +2.44 ± 0.74 (+2.39 / +3.19 / +1.72) |
| PL − softmax control | +0.64 ± 0.15 | **+4.25 ± 0.47** (+4.15 / +4.76 / +3.84) | +4.67 ± 2.57 | **+2.63 ± 0.42** (+2.52 / +2.28 / +3.10) | +2.15 ± 0.37 |
| PL − Stage 1 | +1.72 ± 0.13 | +3.44 ± 1.58 | +2.56 ± 2.85 (−0.26 / +5.43 / +2.52) | +2.23 ± 0.63 | +2.35 ± 0.98 |
| softmax − Stage 1 | +1.08 ± 0.22 | −0.81 ± 1.27 | −2.11 ± 0.46 | −0.40 ± 1.05 | +0.21 ± 1.09 |

- **First demonstrated reward-tuning gain over matched controls.** Against both controls PL RL is ahead in every column and every seed. The pre-registered criterion (§5: ≥ +1.0 on ROxford Medium and RParis Medium vs the strongest matched control, same sign in every seed, no loss on Hard) is **met over 3 seeds**: +2.98 / +2.13 vs FastAP and +4.25 / +2.63 vs softmax, with Hard also positive in every seed.
- **The softmax control improves GLDv2 val (+1.08) but not the benchmarks** (ROxf H −2.11 in every seed), while PL RL improves both. On saturated SfM-120k the two were indistinguishable (E5); on GLDv2 the expected-AP@100 reward transfers to ROxford/RParis and the first-place likelihood does not.
- **Caveats:** 3 seeds; τ and lr chosen on seed-42 validation only (no test-set selection); one budget (≈ 0.24 of a GLDv2 epoch); single-scale evaluation without R1M distractors; GLDv2 may contain Oxford/Paris landmarks (affects all arms equally).
- **Next:** seeds 3–4 (Stage 1 jobs 4549985, 4549989; dependent comparisons 4549986–88, 4549990–92), then ablations (τ, reward k, budget), R1M distractors, and a second backbone.

### G6b. RL vs controls on GLDv2, pilot 2 (jobs 4543381–4543388, 21–30 min each; init val 62.89)

| Arm | Setting | ep 1 / ep 2 | best Δ vs init |
|---|---|---|---|
| PL RL | τ 0.02, lr 1e-6 | 64.38 / 64.42 | +1.53 |
| PL RL | τ 0.02, lr 3e-7 | 64.51 / 64.30 | +1.62 |
| PL RL | τ 0.05, lr 1e-6 | **64.61** / 64.53 | **+1.72** |
| softmax control | τ 0.02, lr 1e-6 | 63.25 / 61.64 | +0.36 |
| softmax control | τ 0.02, lr 3e-7 | 61.32 / 62.78 | −0.11 |
| softmax control | τ 0.05, lr 1e-6 | 64.13 / 63.90 | +1.24 |
| FastAP continuation | lr 3e-5 | 61.23 / 61.62 | −1.27 |
| FastAP continuation | lr 1e-4 | 55.01 / 57.83 | −5.06 |
| (pilot 1) FastAP continuation | lr 1e-5 | 62.75 / 62.69 | −0.14 |

- **First time RL beats the matched differentiable control at the pilot stage:** best PL 64.61 vs best softmax 64.13 (+0.48, gate ≥ 0.3), and PL is ahead in every matched setting (+1.17, +1.73, +0.48). PL is also stable across epochs while the softmax control drifts.
- Unlike SfM-120k (E5), where both tied, the unsaturated GLDv2 pool lets the difference between the objectives show. RL maximises expected AP over the sampled top-100; softmax pushes each positive to first place. This is one seed; the seeds decide.
- Continued FastAP does not improve on the init at any lr.
- **Seeds submitted with final evaluation** (jobs 4543826–4543834): PL τ 0.05 lr 1e-6, softmax τ 0.05 lr 1e-6 and FastAP lr 1e-5 on GLDv2 Stage 1 seeds 42, 1, 2 (seeds 1–2 start after their Stage 1 runs, jobs 4542597–98, finish). `results/runs/gcmp_*`.

### G6. RL vs controls on GLDv2, pilot 1 (jobs 4543050–4543056, 21–31 min each; init = GLDv2 Stage 1 seed 42, val 62.89)
Batch 128 (32 landmarks × 4), 512 px, 2 × 1,432 batches, bank 4,096, no anchor; GLDv2 val mAP after epoch 1 / 2.

| Arm | Setting | ep 1 / ep 2 |
|---|---|---|
| PL RL | τ 0.01, lr 1e-5 | 50.58 / 38.19 |
| PL RL | τ 0.02, lr 1e-5 | 52.54 / 51.96 |
| PL RL | τ 0.02, lr 3e-6 | 60.73 / 55.96 |
| softmax control | τ 0.01, lr 1e-5 | 0.32 / 0.56 (collapsed) |
| softmax control | τ 0.02, lr 1e-5 | 0.13 / 0.15 (collapsed) |
| softmax control | τ 0.02, lr 3e-6 | 48.73 / 20.73 |
| FastAP continuation | lr 1e-5 | 62.75 / 62.69 |

- **Every listwise run degrades, and the softmax control collapses at lr 1e-5.** Training reward (bank-included AP) rises at first and then falls (softmax τ 0.02, lr 1e-5: 0.96 at step 0 → 0.65 at step 600 → 0.05 by step 1,500). Both objectives use very sharp temperatures (τ 0.01–0.02, i.e. logits × 50–100) with 4,224 candidates per query; on GLDv2 with batch 128 these learning rates are far too high. PL RL degrades more slowly than the softmax control in every matched setting.
- FastAP continuation is stable but flat (the Stage 1 cosine schedule had already decayed to 0).
- **Pilot 2 submitted** (jobs 4543381–4543388): PL and softmax at τ 0.02 with lr 1e-6 / 3e-7 and τ 0.05 with lr 1e-6; FastAP continuation at lr 3e-5 / 1e-4 (the Stage 1 val curve was still rising, so continued metric learning at a real lr is the natural strong control).

### G5. Saturation check on the GLDv2 training pool (job 4542593, 53 min; embedding 1.55M images took 36 min)
Seed-42 GLDv2 Stage 1; 2,000 random train queries, top-20 against the full 1,546,857-image train pool; all 40,000 pairs verified (ALIKED + LightGlue + MAGSAC at 512 px).

| | SfM-120k train (A0) | **GLDv2-clean train** |
|---|---|---|
| top-20 items with the query's label | 97.2% | **38.3%** |
| recall@20 of positives (of min(npos, 20)) | 97.2% | **47.1%** |
| success@1 | 99.3% | **77.3%** |
| queries with recall@20 < 1 | 9.8% | **87.6%** |
| labelled positives with no geometric support (< 25 inliers) | 3.1% | 28.6% |
| labelled negatives verified as matches (≥ 100 inliers), share of negatives | 17.3% | 0.8% |
| label / verifier disagreement, all pairs | 3.5% | **11.4%** |

- **Gate passed by a wide margin: the GLDv2 training pool is not saturated.** Recall@20 is 47% instead of 97%, and almost 9 in 10 queries still miss positives in their top-20, so rewards computed on training batches have real headroom.
- **Labels and verifier disagree more (11.4%, the A0 gate threshold was 10%),** but mostly the other way round from SfM: 29% of same-landmark pairs in the top-20 have no geometric overlap (median 66 inliers for retrieved positives; 40% have ≥ 100). GLDv2 landmarks include interiors, details and distant views of the same place. Hidden matches among negatives are rare (0.8%).
- **Implication:** RL vs controls can now be tested where the training signal is not exhausted. A graded or verification-aware reward has more to work with than on SfM-120k.

### G4b. GLDv2 Stage 1 full run, seed 42 (job 4537099, 9 h 24 min on one RTX PRO 6000)
lr 1e-4, batch 128, 512 px, FastAP, cosine over 10 epochs (~12,100 batches each, ~55 min per epoch).

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| GLDv2 val mAP | 49.11 | 52.09 | 53.24 | 55.10 | 56.41 | 57.79 | 59.57 | 61.09 | 62.33 | **62.89** |

Final evaluation of the epoch-10 checkpoint (official scoring):

| Model | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|
| SfM-120k Stage 1 (5-seed mean, A0a) | 42.33 | 19.46 | 59.17 | 32.44 |
| **GLDv2-clean Stage 1, seed 42** (long side 1024) | **49.93** | **27.15** | **72.27** | **53.79** |
| same, shorter side 1024 | 42.86 | 22.66 | 71.63 | 52.96 |

- **Training on GLDv2-clean gives a much stronger starting model:** +7.6 / +7.7 on ROxford and +13.1 / +21.4 on RParis Medium / Hard over the SfM-120k Stage 1. Val mAP was still rising at epoch 10 (+0.56 in the last epoch), so a longer schedule would likely help further.
- Peak memory 82.6 GiB of 92 with 12 workers; no OOM.
- **Next:** G5 saturation check (job 4542593: 2,000 train queries × top-20 against the full 1.55M train pool, with verification). Seeds 1 and 2 of the same recipe submitted (24 h limit each).

### G4. Stage 1 on GLDv2-clean: pilot (jobs 4536649–4536651, ~22 min each on rtxpro6k)
From ImageNet, batch 128 (32 landmarks × 4), 512 px, FastAP, cosine schedule over 3 short epochs of 1,000 batches (384k images, ~25% of GLDv2-clean). Metric: leave-one-out mAP on the 21,285-image GLDv2 held-out val (1,500 landmarks).

| lr | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| 1e-5 | 40.44 | 42.41 | 43.21 |
| 3e-5 | 42.68 | 45.07 | 46.97 |
| **1e-4** | 40.67 | 43.87 | **48.45** |

- **The new pool is far from saturated:** val mAP ~48 and success@1 ~87%, against 83 / 98.6% on SfM-120k. Rewards computed on GLDv2 should leave room to learn.
- Staging 1.58M files takes ~250 s; 1,000 batches take ~4.5 min (one full epoch of ~12,100 batches ≈ 55 min).
- **Memory:** peak 86.7 GB of the 92 GB per-GPU limit with 12 loader workers, the same peak as the SfM-120k jobs, so it is probably page cache filling to the cap. The full run keeps 12 workers and resumes from `stage1_last.pth` if killed.
- **Full run submitted:** `g_s1_b128_lr1e-4_seed42`, job 4537099, lr 1e-4, 10 epochs (~10 h), time limit 24 h, final ROxford/RParis + GLDv2 val evaluation.

## 2026-10-11: RL tuning on top of the strongest differentiable model (Pinto et al. recipe)

### R1. Pilot from the softmax-control checkpoint (submitted 2026-10-11 ~15:40, jobs 4533986–4533991)
- **Why:** Susano Pinto et al. first train the best maximum-likelihood model, then reward-tune it on the task metric. We always started RL from Stage 1. The softmax control (E5) is our strongest model; it optimises the likelihood of each positive at first place, while AP rewards the whole ranking, so RL may improve the gap.
- **Init:** `cmp_sm_t0.02_seed42` stage2_best (SfM val 84.59 on its own selection run; the init is re-measured at epoch 0 of each run). Same budget (2 × 1,432 batches), bank, batches and selection as E3/E5.
- **Arms (equal tuning effort, 3 each):** PL RL τ 0.01 / 0.02 at lr 2e-6 and τ 0.01 at lr 1e-6; continued softmax control τ 0.02 / 0.01 at lr 2e-6 and τ 0.02 at lr 1e-6.
- **Gate:** best RL arm > best control arm on SfM val by more than the E5 seed spread (≈ 0.3); then 5 seeds with final evaluation.
- **Result (all six completed, 27–28 min each; init SfM val 84.59):**

| Arm | Setting | ep 1 / ep 2 | best Δ vs init |
|---|---|---|---|
| PL RL | τ 0.01, lr 2e-6 | 84.06 / 84.10 | −0.49 |
| PL RL | τ 0.02, lr 2e-6 | 84.23 / 83.72 | −0.36 |
| PL RL | τ 0.01, lr 1e-6 | 84.23 / **84.26** | −0.33 |
| softmax control | τ 0.02, lr 2e-6 | 84.28 / 83.95 | −0.31 |
| softmax control | τ 0.01, lr 2e-6 | 84.18 / 83.95 | −0.41 |
| softmax control | τ 0.02, lr 1e-6 | **84.29** / 84.05 | −0.30 |

- **No arm improves on the init; best RL 84.26 vs best control 84.29.** Gate not met. A second pass of either objective from an already fine-tuned model drifts slightly down, by the same amount for RL and control, so with this data and reward the model is at a plateau.
- The init's 84.59 was itself the best of two epochs of its own run, so part of the drop is regression from a selected peak; this applies to both arms equally.
- **Conclusion:** the Pinto recipe (reward tuning after the best ML model) gives no gain here. With binary SfM-cluster AP as the reward, RL and its ML counterpart are interchangeable from Stage 1 (E5) and from the tuned model (R1).

## 2026-10-11: Revised plan, Track A (verifier-in-the-loop RL)

### A0b. Same diagnostic on held-out SfM-120k val (job 4533274, a40, 65 min; all 6,403 val queries × top-20 = 128,060 pairs, 37 pairs/s with feature caching)

| | ranks 1–5 | 6–10 | 11–20 | all top-20 | (train, A0) |
|---|---|---|---|---|---|
| retrieved items labelled negative | 3.0% | 7.0% | 14.1% | 9.6% | 2.8% |
| … of those, verified match (≥ 100 inliers) | 10.9% | 6.5% | 5.0% | 5.8% (13.7% at ≥ 25) | 17.3% |
| retrieved positives with no support (< 25 inliers) | 1.8% | 2.8% | 4.5% | 3.3% | 3.1% |
| hidden matches (negative label, ≥ 100 inliers) as share of all pairs | | | | 0.55% | 0.49% |
| queries with at least one labelled negative in top-20 | | | | 34.1% | 9.8% |

- **On unseen images the model makes more mistakes** (9.6% of the top-20 are other landmarks; a third of queries have at least one), **but the labels already describe almost all of them correctly**: 94% of retrieved negatives are true non-matches. Hidden matches stay at ~0.5% of retrieved pairs, the same as on train, and label/verifier disagreement is 3.6%.
- **Gate (§4b):** condition 1 barely met (5.8% vs 5%), condition 2 not met (3.6% vs 10%).
- **Conclusion:** within SfM-120k the verifier adds little information beyond the cluster labels, so verifier-in-the-loop RL on SfM-120k is not promising. On an unlabelled pool (GLDv2) the verifier would *be* the supervision, but then the competitor is iterative re-mining with the same verifier, and RL's only extra ingredient is exploration beyond the greedy top-k. After E5 the prior for that winning is low.

### A0a. Stage 1 seeds 3 and 4 finished (jobs 4525721; 4525720 OOM at epoch 6, resumed as 4527173)

| Stage 1 seed | SfM val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| 3 | 82.94 | 43.80 | 18.78 | 57.76 | 31.38 |
| 4 | 82.81 | 41.03 | 20.32 | 58.48 | 32.01 |
| 5-seed mean ± std (42, 1, 2, 3, 4) | 82.97 ± 0.10 | 42.33 ± 1.29 | 19.46 ± 0.74 | 59.17 ± 1.20 | 32.44 ± 0.87 |

- Resume after the OOM worked (one "Resuming" line, 8 loader workers, no further OOM). Checkpoints: `$HPCVAULT/runs/s1_b32_lr1e-5_seed{3,4}`.

### A0. Label-noise / exploration diagnostic (job 4527211, a40, 72 min incl. embedding 91,642 train images)
Seed-42 Stage 1; 2,000 random SfM-120k train queries × top-20 = 40,000 retrieved pairs, all verified (ALIKED + LightGlue + MAGSAC). Output: `results/a0_label_noise/` (pairs, summary, contact sheets); train embeddings cached on `$HPCVAULT/runs/a0_label_noise/`.

| | ranks 1–5 | 6–10 | 11–20 | all top-20 |
|---|---|---|---|---|
| retrieved items labelled negative (other 3D model) | 1.7% | 2.6% | 3.5% | 2.8% |
| … of those, verified match (≥ 100 inliers) | 22.9% | 16.3% | 16.3% | 17.3% (44.0% at ≥ 25) |
| retrieved positives with no support (< 25 inliers) | 1.5% | 2.8% | 4.0% | 3.1% |

- **The verifier finds real label noise.** Contact sheets of the strongest "negatives" show the same landmark every time (Casa Batlló, Powder Tower, Tower Bridge, St Vitus, Colosseum, Torre Agbar, identical fresco ceilings): SfM-120k splits some landmarks into several 3D models, which the cluster labels treat as different.
- **But on the training pool there is little of it to exploit.** The Stage 1 model has fitted its own training images: 97% of the top-20 are already labelled positives. Verified false negatives are 0.5% of retrieved pairs and touch 3.4% of queries; overall label/verifier disagreement is 3.5%.
- **Gate (§4b):** the first condition is met (17% of retrieved negatives verify, threshold 5%); the second is not (disagreement 3.5%, threshold 10%). Formally a pass, but the absolute signal on SfM-120k train is small.
- **Implication:** verifier-in-the-loop RL needs a pool the model has not memorised (held-out or unlabelled images, e.g. a GLDv2 subset), where the top-k is far from saturated and pre-labelling is impossible.
- **Throughput:** 14.9 pairs/s here, because each pair re-extracted features; the labelling job (D5) reached 61 pairs/s by sharing features. A1 must cache features per image.

## 2026-10-10: Phase 3, rankings as actions

### E1. Plackett–Luce ranking policy (implementation and CPU check)
- **Why:** Phases 1–2 used a Gaussian policy on embeddings. Its log-density is a proxy (κ μ·a), and its gradient only comes through random perturbations of a 512-d vector. In every run the model either stayed at the anchor or got worse.
- **What:** `training/pl_policy.py`. The action is a top-k ranking (k = 100) of the batch plus the 4,096-entry bank of means, sampled from a Plackett–Luce model over the scores q·g/τ with the Gumbel top-k trick. The log-probability is exact and differentiable in the scores, so gradients reach both query and gallery embeddings. Every batch item is a query against all others. R = 64 sampled rankings per query with leave-one-out baselines; AP@k reward with positives outside the list counted as missed.
- **CPU checks:** log-probabilities match a brute-force sequential computation to 1e-10, and all top-2 lists sum to probability 1. Gumbel top-k frequencies match PL (first place 0.580 vs 0.579; a specific pair 0.295 vs 0.293). On an enumerable toy problem (6 items, k = 4) the averaged estimate has cosine **0.9999** with the exact gradient of expected AP@k from finite differences. For comparison, the Gaussian policy reached 0.69–0.89 (D1).
- **Pilot (submitted 18:30):** same budget, batches, init (seed-42 Stage 1), learning rate (2e-6) and selection as the C4 RL arm, so the 3-seed C4 control is reused. Grid: τ ∈ {0.01, 0.02, 0.05} without anchor, plus τ = 0.02 with consistency 5,000. Smoke 4525046 failed because YAML reads `2e-6` as a string (learning rate now written `0.000002`); resubmitted as smoke 4525122 (a100mig) and pilots 4525123–26 (rtxpro6k); every full GPU partition is booked until ~01:15–01:30.

### E5. Softmax control vs PL, 3 seeds with final evaluation (jobs 4526124–4526126, 32–35 min each)
Same inits, batches, bank, lr, budget and selection as E3.

| Model (3 seeds, mean ± std) | SfM val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| Stage 1 | 83.04 ± 0.01 | 42.27 ± 1.17 | 19.41 ± 0.70 | 59.87 ± 0.95 | 32.94 ± 0.70 |
| + FastAP control (C4) | 82.95 ± 0.34 | 42.51 ± 1.16 | 20.09 ± 2.46 | 60.25 ± 0.85 | 33.26 ± 0.54 |
| + softmax control, bank, τ 0.02 | **84.52 ± 0.20** | 42.64 ± 1.45 | 19.51 ± 3.41 | **61.51 ± 1.27** | **34.01 ± 0.60** |
| + PL RL τ 0.01 (E3) | 84.24 ± 0.13 | 43.01 ± 0.77 | 20.91 ± 1.11 | 61.33 ± 1.54 | 33.61 ± 1.01 |
| + PL RL τ 0.02 (E3) | 84.14 ± 0.22 | **43.04 ± 0.45** | **21.36 ± 1.34** | 60.77 ± 1.54 | 33.60 ± 1.17 |

Paired PL τ 0.01 − softmax control: SfM val −0.27 ± 0.13, ROxf M +0.37 ± 1.19, ROxf H +1.40 ± 2.44, RPar M −0.17 ± 0.47, RPar H −0.39 ± 1.03 (per seed, ROxf M +1.73 / −0.15 / −0.47).

- **The PL gain in E3 is not an RL effect.** The matched differentiable control on the same bank and batches is as good: better on SfM val in every seed and on RParis on average, worse on ROxford on average (driven by seed 42, which has a weak ROxford control run). No column differs beyond seed noise.
- **Useful by-product:** the bank + all-vs-all listwise objective is a stronger fine-tuning step than FastAP continuation (+1.5 SfM val, +1.6 RPar M, +1.1 RPar H over Stage 1, same sign in all seeds; ROxford unchanged within noise). This is the cross-batch-memory effect known from XBM (Wang et al., CVPR 2020), not a new contribution.
- **Consequence for the RL claim:** with binary AP as reward, a PL policy and its maximum-likelihood counterpart are interchangeable. RL can only add something with a reward the control cannot optimise directly (graded verification relevance, or top-k-only measures).
- Stage 1 seed 3 (job 4525720) was killed by the host-RAM limit at epoch 6 (MaxRSS 86.7 GB of 92 GB, 16 loader workers). Resumed from its last checkpoint with 8 workers.

### E4. Memory-bank control and extra seeds (submitted 2026-10-10 ~20:45)
- **Control:** `training.stage2.policy=softmax_control` (`training/pl_policy.py`), a multi-positive softmax cross-entropy (SupCon) over the same scores q·g/τ, the same all-vs-all batches, the same 4,096 bank of means, lr, budget and selection, run in the same trainer. It is the maximum-likelihood counterpart of the PL policy, so a PL gain over it cannot come from the bank, τ or the batch layout. CPU toy: both modes learn (greedy AP 0.56 / 0.55 after 30 steps).
- **Control τ grid (seed 42; same tuning effort as the PL τ grid):** τ 0.01 / 0.02 / 0.05, jobs 4525724–4525726. The best τ on val then gets 5 seeds with final evaluation.
- **Control τ grid result (seed 42, val mAP ep 1 / ep 2; init 83.03):** τ 0.01 84.22 / 84.13, **τ 0.02 84.59** / 84.12, τ 0.05 83.93 / 83.45. The differentiable control on the same bank matches or beats the PL policy on validation (PL seed 42: τ 0.02 84.23, τ 0.01 84.18). **So most or all of the E3 gain is likely due to the all-vs-all batches plus memory bank with a listwise softmax objective, not to the reward.** 3 seeds of the control at τ 0.02 with final evaluation submitted (`cmp_sm_t0.02_seed{42,1,2}`) to check on the benchmarks.
- **Stage 1 seeds 3 and 4** (jobs 4525720–4525721, ~4 h each, same recipe as C1), so every arm can be run on 5 seeds: Stage 1, FastAP control, softmax control, PL τ 0.01.

### E3. Plackett–Luce 3 seeds with final evaluation (jobs 4525409–4525414, 31–35 min each incl. staging and evaluation)
Best epoch on SfM val per run; benchmarks scored once on it (official scoring, longer side 1024). Stage 1, control and Gaussian RL rows are the C4 runs (same inits, budget, batches, lr and selection).

| Model (3 seeds, mean ± std) | SfM val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| Stage 1 | 83.04 ± 0.01 | 42.27 ± 1.17 | 19.41 ± 0.70 | 59.87 ± 0.95 | 32.94 ± 0.70 |
| + continued metric learning (FastAP) | 82.95 ± 0.34 | 42.51 ± 1.16 | 20.09 ± 2.46 | 60.25 ± 0.85 | 33.26 ± 0.54 |
| + Gaussian RL (C4) | 83.10 ± 0.11 | 42.53 ± 1.39 | 20.34 ± 1.82 | 59.80 ± 0.67 | 33.11 ± 0.66 |
| + PL RL τ 0.02 | 84.14 ± 0.22 | **43.04 ± 0.45** | **21.36 ± 1.34** | 60.77 ± 1.54 | 33.60 ± 1.17 |
| + PL RL τ 0.01 | **84.24 ± 0.13** | 43.01 ± 0.77 | 20.91 ± 1.11 | **61.33 ± 1.54** | **33.61 ± 1.01** |

Paired differences (same Stage 1 seed), mean ± std; per-seed values in brackets (seeds 42 / 1 / 2):

| Comparison | SfM val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| PL τ 0.01 − Stage 1 | +1.21 ± 0.12 | +0.74 ± 0.48 (+0.47 / +0.46 / +1.29) | +1.50 ± 1.33 | +1.46 ± 0.70 (+1.46 / +2.16 / +0.76) | +0.67 ± 1.24 |
| PL τ 0.01 − control | +1.29 ± 0.21 | +0.50 ± 1.27 (+1.68 / +0.66 / −0.83) | +0.82 ± 1.97 | +1.08 ± 0.77 (+1.15 / +1.82 / +0.27) | +0.35 ± 1.35 |
| PL τ 0.02 − Stage 1 | +1.10 ± 0.22 | +0.77 ± 0.73 | +1.95 ± 1.76 | +0.90 ± 1.05 | +0.66 ± 1.62 |
| PL τ 0.02 − control | +1.19 ± 0.22 | +0.53 ± 1.33 | +1.27 ± 2.50 | +0.52 ± 1.07 | +0.34 ± 1.64 |

- **First reproducible gain from reward tuning.** On SfM validation PL RL beats both Stage 1 and the control by +1.1 to +1.3 in every seed (the selection metric, so partly optimistic). On the benchmarks, PL τ 0.01 improves on Stage 1 in all three seeds for ROxford Medium, ROxford Hard and RParis Medium; RParis Hard is +0.67 on average with one seed at −0.34.
- **Against the matched control** the gains are positive on average for every column (ROxf M +0.50, RPar M +1.08) but not yet in every seed: the control has a ±1–2.5 seed spread on 70-query benchmarks. **The pre-registered success criterion (≥ +1.0 on both Medium protocols vs the strongest control, same sign in every seed, no Hard loss) is not met yet**; RParis Medium meets it, ROxford Medium does not.
- The anchored PL run (τ 0.02, consistency 5,000, job 4525126) stayed at the init (83.07): the gain needs the model free to move.
- Both τ values were run and are reported; τ 0.01 is slightly better on val and more consistent on the benchmarks.
- Still open: the memory-bank differentiable control (plan 3.4), more seeds to shrink the spread, R1M distractors.
- Records: `results/runs/cmp_pl_t{0.02,0.01}_cw0_seed{42,1,2}/` (configs, git diffs, init hashes, final_eval.json); checkpoints `$HPCVAULT/runs/cmp_pl_*`.

### E2. Plackett–Luce pilot, seed-42 init (jobs 4525123–4525126, ~28 min each; smoke 4525122 passed)
Same budget, batches, learning rate and selection as the C4 RL arm; init SfM val mAP 83.03. R = 64 rankings, k = 100, bank 4,096.

| Run | SfM val ep 1 / ep 2 | Δ best vs init | train reward sampled / mean ranking |
|---|---|---|---|
| τ 0.02, no anchor | **84.23** / 83.60 | **+1.20** | 0.92 / 0.95 |
| τ 0.01, no anchor | 83.87 / 83.79 | +0.84 | 0.93 / 0.95 |
| τ 0.05, no anchor | 81.48 / 81.15 | −1.55 | 0.87 / 0.95 |
| τ 0.02, consistency 5,000 | 83.05 / 83.07 | +0.04 | (anchor holds it at init, as in Phases 1–2) |

For reference (C3/C4): continued metric learning 82.96 (seed 42), Gaussian RL 83.25 at best; nothing in Phases 1–2 exceeded +0.22.
- **First clear validation gain from reward tuning, without any anchor.** Every Gaussian-policy run without an anchor got worse (D3: 79.22).
- Too much randomness (τ 0.05) hurts. τ 0.01–0.02 keeps the sampled rankings close to the greedy one (reward 0.92–0.93 vs 0.95).
- Epoch 2 is lower than epoch 1 at τ 0.02. Selection on validation takes epoch 1; a shorter schedule or lr decay may help later.
- **Caveats:** one seed, validation only. The control in C4 has no memory bank, so a memory-bank differentiable control (e.g. a softmax/contrastive or Smooth-AP loss on the same bank) is still needed before claiming that RL, rather than the bank, causes the gain.
- **Next (submitted):** 3 seeds × τ ∈ {0.02, 0.01} with final ROxford/RParis evaluation, jobs 4525409–4525414 (`results/runs/cmp_pl_t*_cw0_seed*`).

## 2026-10-10: Phase 1 of the RL research plan ([RL_RESEARCH_PLAN.md](RL_RESEARCH_PLAN.md))

### D1. Multi-sample estimator audit (CPU, `scripts/audit_multisample.py`)
Synthetic problem: 8 queries, 12 gallery items, 4 classes, 16-d; reward = smooth AP so the exact gradient of the expected reward is known (pathwise, 40k samples). Estimators use reward values only. "Single-estimate cos" = median cosine between one gradient estimate and the truth (higher = less variance); "mean cos" = cosine of the average of 100–300 estimates (bias check).

| Estimator (σ = 0.05) | single-estimate cos, query part | single-estimate cos, all params | mean cos | |mean|/|truth| |
|---|---|---|---|---|
| current trainer (1 sample + 1 baseline sample, query only) | 0.12 | 0.10 | 0.75 | 0.81 |
| S = 4, product rollouts, query only | 0.38 | 0.31 | 0.79 | 0.82 |
| S = 16, query only | 0.69 | 0.56 | 0.81 | 0.82 |
| S = 64, query only | 0.89 | 0.72 | 0.81 | 0.80 |
| S = 16, query + gallery | 0.69 | 0.41 | **0.98** | 1.00 |
| S = 64, query + gallery | 0.89 | 0.67 | **0.99** | 1.00 |

- One update of the current trainer is almost pure noise (cos 0.12 with the true direction); S = 64 product rollouts reach 0.89 at no extra network cost.
- Training the gallery is required to follow the full objective: query-only estimators plateau at mean cos ~0.8 because half of the true gradient (gallery part) is missing.
- At σ = 0.15 the direction is unchanged but magnitude is ~0.88× (the vMF-style log-density is only an approximation of the projected-normal sampler at larger σ); σ ≤ 0.05 keeps it unbiased in this test.
- Implementation: `training/rl_estimators.py` (`multi_sample_policy_loss`).

### D8. Noise and label follow-ups (jobs 4522855–4522857; same init, budget and selection as D7)

| Run | nDCG@100 ep 1 / ep 2 | val mAP ep 1 / ep 2 | train reward sampled / mean ranking |
|---|---|---|---|
| RL S16, cw 5000, σ 0.02 | 90.67 / 90.67 | 83.04 / 83.06 | 0.79 / 0.93 |
| RL S16, cw 5000, σ 0.01 | **90.71** / 90.61 | **83.19** / 82.96 | 0.90 / 0.93 |
| ApproxNDCG, binary train labels, lr 1e-6 | 89.11 / 88.90 | 80.40 / 79.82 | – / 0.97 |

Init: 90.67 / 83.03.
- **Lower noise closes the sampling gap but RL still does not move.** At σ = 0.01 the sampled reward (0.90) is close to the mean ranking (0.93), yet the best epoch is +0.04 nDCG@100 and +0.16 mAP, and it is gone one epoch later. That is within noise. The anchored model stays within a consistency MSE of ~4e-5 of Stage 1 in every run, and every weaker anchor (D3, D7) made it worse.
- **The list batches themselves hurt the differentiable control.** Binary labels halve the damage (−1.8 nDCG@100, −3.2 mAP vs −3 / −6 with graded gains), so both the graded gains and the 4-landmark candidate-list batches contribute. For comparison, FastAP on 8-landmark PK batches holds 82.96 (C3).
- **Phase 2 verdict (Gaussian embedding policy, graded nDCG reward): negative.** No arm beats the init on validation, so no 3-seed runs. Across Phases 1–2 the policy either stays at the anchor or degrades.
- **Open concern:** SfM val is close to saturation (success@1 98.6%, mAP 83). C4 showed +0.24 ROxford mAP for the matched control while its SfM val fell by 0.07, so small benchmark changes may be invisible to this selection metric. A hard-positive val metric (only low-overlap level 1–2 positives, like Revisited Hard) is cheap to add from the oracle labels.

### D7. Graded pilots, seed-42 init (jobs 4522500–4522504; 2 epochs × 1,432 batches; RL ~40 min, ApproxNDCG ~32 min)
Init: val graded nDCG@10 89.91, nDCG@100 90.67, mAP 83.03.

| Run | nDCG@100 ep 1 / ep 2 | nDCG@10 ep 2 | val mAP ep 2 | train reward (sampled) |
|---|---|---|---|---|
| RL S16, cw 5000, lr 2e-6 | 90.65 / **90.72** | 89.97 | 83.03 | 0.14 |
| RL S16, cw 500 | 90.36 / 90.45 | 89.94 | 82.49 | 0.17 |
| RL S64, cw 5000 | 90.69 / 90.70 | 89.97 | 83.06 | 0.14 |
| ApproxNDCG lr 1e-6 | 88.53 / 87.44 | 88.20 | 76.97 | (mean ranking 0.93) |
| ApproxNDCG lr 3e-6 | 87.57 / 87.37 | 87.94 | 77.32 | (0.94) |

- **Graded RL is inert again:** the best run is +0.05 nDCG@100 and unchanged mAP, which is within noise. With a weaker anchor (cw 500) it moves, but downward.
- **The graded differentiable control is harmful:** −3 nDCG@100 and −6 mAP after one epoch at either lr. Its selected checkpoint is worse than the init, so the init is the de facto control.
- **Diagnosis for RL:** the sampled policy is far from the deterministic one. The sampled reward is 0.14 while the mean ranking scores ~0.93 on the same batches. With σ = 0.05 the noise vector has norm ≈ 1.1 in 512-d, and it competes with ~4,100 bank negatives, so REINFORCE optimises the reward of a mostly random ranking. Next: σ = 0.02 and 0.01, with a new `reward_mean_policy` log to measure the gap.
- **Diagnosis for the control:** it is unclear whether the drop comes from the graded labels (exponential gains 1/3/7/15 put most weight on near-duplicates) or from the list batches and loss. Next: the same run with every verified pair at level 1 (`--binary_train_grades`).
- Follow-ups: see D8.

### D6. Graded smoke tests (jobs 4522498 rl, 4522499 approxndcg; 30 batches each, ~5 min) — passed
- 31,203 usable train queries over 551 landmarks (≥ 7 verified partners each).
- Seed-42 Stage 1 init on val: graded nDCG@10 **89.91**, nDCG@100 **90.67**, mAP 83.03. This is the bar for the graded pilots.
- After 30 batches both arms are unchanged on val (nDCG@100 90.66 / 90.67), as expected for so few steps. No errors.
- RL training reward (sampled nDCG@10) falls from 0.92 at step 0 to 0.56 averaged over the epoch as the 4,096-entry memory bank of negatives fills; ApproxNDCG's mean-embedding nDCG stays ~0.96. RL grad norm ~80 vs ~3 for ApproxNDCG (σ = 0.05 noise in 512-d has norm ≈ 1.1, so sampled rankings are much noisier than the mean ranking).
- Pilots 4522500–4522504 started after the smokes (2 epochs × 1,432 batches, rtxpro6k).

### D5. Full graded labelling (array job 4522260 train, 4522261 val; ~25–28 min per shard)
ALIKED + LightGlue + MAGSAC on same-landmark pairs only; files in `$HPCVAULT/datasets/sfm120k/oracle_v1/`.

| Split | verified pairs | level 1 (< 25 inliers) | level 2 (25–99) | level 3 (100–299) | level 4 (≥ 300) |
|---|---|---|---|---|---|
| train (57,416 query images, ≤ 40 per landmark × ≤ 25 partners, deduplicated) | 429,672 | 10.8% | 9.1% | 24.6% | 55.5% |
| val (all same-landmark pairs) | 149,371 | 14.8% | 8.1% | 20.8% | 56.2% |

- **Design change before any graded training:** the pilot plan graded low-inlier same-landmark pairs as 0 (irrelevant). That would teach the model to push down other views of the same landmark, which the Revisited benchmarks count as positives (often "hard"). Relevance levels are now 0 = other landmark, 1–4 = same landmark ordered by inliers (commit `fix(graded): keep same-landmark images relevant...`). Verification therefore only orders positives; it never removes them.
- About 11–15% of same-landmark pairs show no geometric overlap; over half overlap strongly (median 344 inliers in train).

### D3. Phase 1 follow-up: does the RL direction help once the model can move? (jobs 4521748/4521749)
Same setup as D2 with S = 64, consistency weight 50 or 0.

| Run | SfM val ep 1 / ep 2 | consistency loss |
|---|---|---|
| consistency 50 | 82.45 / 81.86 | 1.4–2.8e-4 |
| consistency 0 | 81.40 / 79.22 | – |
| control lr 1e-5 (C3), for scale | 82.14 / 81.98 | – |

- **Unconstrained binary-reward RL actively lowers held-out mAP**, faster than continued FastAP at a comparable step size. The well-estimated gradient optimises expected AP of *noisy* embeddings on training landmarks, which does not transfer to deterministic retrieval on new landmarks.
- **Phase 1 closed:** with binary rewards, RL is either inert (anchored) or harmful (free). Effort moves to the graded reward (Phase 2), as the plan anticipated.

### D4. Relevance-oracle pilot (job 4521746, 10 min; 200 queries × 50 pairs, ALIKED + LightGlue + MAGSAC, long side 1024)
| | p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|---|
| inliers, same-landmark pairs | 11 | 27 | 230 | 512 | 750 |
| inliers, hardest other-landmark pairs | 8 | 10 | 12 | 15 | 23 |

- Throughput: 51 images/s (features), 61 pairs/s (matching + RANSAC; RANSAC on one CPU core is ~40% of the time). One degenerate fit in 10,000 pairs.
- 18.6% of same-landmark pairs have < 15 inliers (same landmark, non-overlapping view); 3.5% of other-landmark pairs have ≥ 50 (possible label noise or repeated structures); the other-landmark noise floor is ~10–25 inliers.
- AUC for separating same/other landmark: inliers 0.863 vs Stage 1 similarity 0.898. Within same-landmark pairs, inliers vs Stage 1 similarity Spearman 0.63.
- **Conclusion:** inlier counts are a graded, view-level relevance signal that neither the binary labels nor the current model capture. Planned grades (same-landmark pairs only; other landmarks stay 0): 0 = < 25 inliers, 1 = 25–99, 2 = 100–299, 3 = ≥ 300.
- Full-labelling cost: only same-landmark pairs need verification. ~22k queries (up to 40 per landmark) × ≤ 25 positives ≈ 0.5M pairs ≈ 2–3 GPU-hours at pilot speed, less with parallel RANSAC; split into shards.

### D2. Phase 1 validation pilot (jobs 4521459–4521462, ~34 min each; seed-42 Stage 1, SfM val 83.03)
Batch 32, one pass over SfM-120k (2 × 1,432 batches), σ 0.05, lr 2e-6, no clipping, memory of mean embeddings, binary AP reward.

| Run | samples | consistency | adv. norm. | SfM val ep 1 / ep 2 | policy grad norm | consistency loss |
|---|---|---|---|---|---|---|
| S16 | 16 | 5,000 | no | 83.00 / 83.01 | 12–14 | 2–4e-6 |
| S64 | 64 | 5,000 | no | 83.03 / 83.04 | 6–8 | 1–2e-6 |
| S16, weaker anchor | 16 | 500 | no | 83.02 / **83.09** | 13–15 | 2–4e-5 |
| S16, normalised adv. | 16 | 5,000 | yes | 82.94 / 82.68 | ~283 | 6–8e-5 |

Reference points: control (continued FastAP) 82.96; old RL estimator 83.25 (one epoch, one seed).
- The new estimator gives a much steadier signal (policy grad norm 6–15 instead of 100–300; >99.9% of queries carry signal), but the model **barely moves**: a consistency MSE of ~2e-6 per dimension corresponds to cosine ≈ 0.999 between new and Stage 1 embeddings. With a binary AP reward on a model that FastAP already trained, there is little left to gain.
- Gate 1.5 (≥ control + 0.3 = 83.26) **not met**. Diagnostic follow-up with weak/no anchoring at S = 64 (jobs 4521748/4521749) checks whether the RL direction helps once the model is allowed to move.
- Relevance-oracle pilot job 4521483 crashed on a degenerate MAGSAC fit (OpenCV assertion); fixed and resubmitted as 4521746.

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

### C4. Seed repeats with final evaluation (jobs 4518806–4518811, 29–37 min each)
Best setting of each arm (from C3) started from each of the three Stage 1 seeds, same seed for the fine-tuning run. Benchmarks scored once on the validation-selected epoch (official scoring, longer side 1024).

| Model (3 seeds, mean ± std) | SfM val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| Stage 1 | 83.04 ± 0.01 | 42.27 ± 1.17 | 19.41 ± 0.70 | 59.87 ± 0.95 | 32.94 ± 0.70 |
| + continued metric learning (control) | 82.95 ± 0.34 | 42.51 ± 1.16 | 20.09 ± 2.46 | **60.25 ± 0.85** | **33.26 ± 0.54** |
| + reward tuning (RL) | **83.10 ± 0.11** | **42.53 ± 1.39** | **20.34 ± 1.82** | 59.80 ± 0.67 | 33.11 ± 0.66 |

Paired differences (same Stage 1 seed), mean ± std over 3 seeds:

| Comparison | SfM val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| RL − Stage 1 | +0.06 ± 0.11 | +0.27 ± 0.32 | +0.93 ± 1.12 | −0.07 ± 0.28 | +0.17 ± 0.20 |
| control − Stage 1 | −0.08 ± 0.33 | +0.24 ± 1.71 | +0.69 ± 1.98 | +0.38 ± 0.09 | +0.32 ± 0.32 |
| **RL − control** | +0.14 ± 0.29 | +0.03 ± 1.70 | +0.24 ± 1.42 | −0.45 ± 0.19 | −0.15 ± 0.15 |

- **No demonstrated gain from reward tuning.** RL and the matched control are indistinguishable on ROxford; the control is slightly better on RParis (−0.45 ± 0.19 Medium for RL, consistent over the 3 seeds but small).
- RL changes the model more gently: its seed-to-seed spread around Stage 1 is smaller (ROxford M ±0.32 vs ±1.71 for the control).
- Neither fine-tuning step adds much on top of a well-trained Stage 1 with this budget (one pass over SfM-120k).
- Compute per run including staging and final evaluation: control ~30 min, RL ~36 min on one RTX PRO 6000.
- **Host memory:** all six jobs still peaked at ~84.6–86.7 GB of the 92 GB limit even with 8 workers, so the earlier OOM was not only worker count. Likely page cache from the ~40 GB staged dataset counted against the job. Watch this before larger runs.
- Records: `results/runs/cmp_{ml,rl}_*_seed*/`.


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
