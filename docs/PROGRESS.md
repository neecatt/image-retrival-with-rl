# Progress log

What was found, what was changed, and the evidence. Newest entries at the top. Milestone status lives in [ROADMAP.md](ROADMAP.md); this file keeps the detail.

## Open problems (as of 2026-10-10)

| # | Problem | Why it matters | Status |
|---|---|---|---|
| P1 | Stage 1 baseline is weak: ROxford5k Medium 28.30 / Hard 12.21 (official scoring, long side 1024), only +4.2 Medium over an untrained ImageNet ResNet50-GeM (24.14). Published fine-tuned GeM systems report far higher numbers, though with multi-scale testing and whitening that we do not use | Any reward-tuning gain on top of a weak baseline is hard to interpret | **Largely fixed (C1).** New Stage 1 (3 seeds): ROxford M/H 42.27/19.41, RParis M/H 59.87/32.94, vs 28.30/12.21 and 43.75/19.80 before. Still below published systems that use multi-scale testing and whitening |
| P2 | **Stage 2 RL lowers held-out SfM val mAP in every setting tried** (B2), and the historical `stage2_epoch_5.pth` is worse than Stage 1 on every metric (B1) | The central claim (RL improves retrieval) is currently not supported | **Answered for the current method (C4): no measurable gain.** From the new Stage 1, 3 seeds, matched budget: RL vs continued metric learning differs by −0.45 to +0.24 points, all within seed spread. RL is a valid fine-tuning method that does no harm when anchored, but does not beat the control. Improving it needs a method change (Stage D) |
| P3 | Checkpoint selection signal is weak: "validation reward" is mAP inside each batch of 8 (gallery of ~4 images), close to 1.0 | Cannot reliably pick the best epoch | Fixed (F5), verified in smoke run 4515494 |
| P4 | README Stage 1 numbers (16.97 Medium / 3.59 Hard) do not match a rerun of `stage1_epoch_13_best.pth` (18.29 / 9.33 with the old scoring) | Historical results cannot be reproduced as stated | Likely a different checkpoint or older code. The run folders on tinyx have not been copied yet |
| P5 | Policy gradient norm is 100–300 vs 0.06–0.08 for the consistency term; `gradient_clip: 1.0` clips 90–100% of steps | The consistency regulariser has no practical effect; updates are clipped to unit norm every step | Measured (B2). Addressed together with P2 |
| P6 | Gallery embeddings are sampled but receive no policy gradient; memory-bank entries come from older parameters | The objective is "query-only, given a noisy gallery", not full expected AP | Acceptable if stated explicitly in the paper; not changed |

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
