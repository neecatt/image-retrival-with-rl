# Does reward tuning beat metric learning for landmark retrieval? A controlled study

*Full working draft, 2026-10-14. Every number comes from [PROGRESS.md](PROGRESS.md); the ids in brackets (C4, E5, G10, N3, …) point to the run records with job ids, configs and checkpoints. `[TODO]` marks missing experiments; `[verify]` marks bibliographic details to check against the official source before submission.*

---

## Abstract

Policy-gradient "reward tuning" — fine-tuning a pretrained model with REINFORCE on the task metric — has been reported to improve vision models and, more recently, retrieval systems. Retrieval, however, already has strong differentiable surrogates of its metric (FastAP, Smooth-AP), so a gain from reward tuning is only meaningful against such surrogates trained under identical conditions. We study reward tuning of a global landmark descriptor (ResNet50-GeM) under a protocol that matches initialisation, batches, memory bank, budget and model selection, scores with the official Revisited Oxford/Paris code, selects checkpoints on held-out validation only, and fixes a success criterion before running. Across three training regimes — SfM-120k, GLDv2-clean, and a noisy-label GLDv2 pool with a geometric verifier in the loop — and five policy designs (Gaussian embedding policies with single- and multi-sample estimators, graded verification rewards, and a Plackett–Luce ranking policy), reward tuning never beats the strongest matched surrogate. A ranking policy with an AP@100 reward reaches the level of AP surrogates and appears to beat metric learning by +3 to +4 ROxford mAP when the control lacks the memory bank, but the gain vanishes once the surrogate uses the same bank (5 seeds). With a verifier in the loop, a 2×2 design separating acquisition from objective shows the verified labels, not the RL objective, carry the benefit. Diagnostics explain the regimes: on SfM-120k the pretrained model already retrieves 97% correct top-20 items on its own training images, leaving no reward signal. As by-products we report a strong fine-tuning recipe (GLDv2-clean pretraining plus an AP loss over a cross-batch memory: +4.7 / +3.9 ROxford and +2.7 / +3.0 RParis Medium / Hard over the pretrained model, every seed) and a methodology checklist for evaluating RL in retrieval.

## 1 Introduction

Susano Pinto et al. [1] showed that fine-tuning MLE-pretrained vision models with REINFORCE on a task reward can align them with non-differentiable task metrics. The idea has since been carried to retrieval: embedding-space REINFORCE for text retrieval (RELER [2]), group-relative policy optimisation for dense product retrieval (Retrieval-GRPO [3]), and ranking-driven RL with verifiable rewards for multimodal retrieval (ELVA [4]). Landmark retrieval is an attractive test bed: its metric (mean average precision under the Revisited Oxford/Paris protocol [5]) is defined on rankings, and geometric verification [6, 7] offers a reliable, non-differentiable relevance signal.

But retrieval differs from captioning or detection in one important way: average precision has good differentiable surrogates. FastAP [8] and Smooth-AP [9] optimise AP directly, and cross-batch memories [10] give them large, informative galleries. A reward-tuning gain is therefore only interesting if it survives a surrogate trained on the same data, batches, memory and budget. Several things can also make RL look better than it is: comparing against a weaker loss, selecting checkpoints on the test benchmark, or comparing at different budgets.

This paper asks a narrow question under strict controls: **does reward tuning improve a landmark-retrieval descriptor beyond what a matched differentiable surrogate achieves?** Our contributions:

1. **A controlled protocol** for RL in retrieval (§3): official scoring, validation-only selection, matched initialisation/batches/memory/budget, 3–5 seeds with paired differences, and a pre-registered success criterion.
2. **A negative result across three regimes and five policy designs** (§5): reward tuning never beats the strongest matched surrogate. A Plackett–Luce ranking policy matches AP surrogates on the same memory bank; its apparent +3–4 mAP advantage over metric learning is a memory-bank effect.
3. **Diagnostics that explain why** (§6): training-pool saturation (97% of top-20 correct on SfM-120k train, 47% recall on GLDv2), label/verifier agreement, and a 2×2 acquisition × objective design with a replay control showing that verified labels, not the RL objective, carry the benefit of verifier feedback.
4. **A strong practical recipe** as a by-product (§5.2): GLDv2-clean pretraining plus an AP loss over a cross-batch memory.

## 2 Related work

**Task-reward tuning.** Susano Pinto et al. [1] fine-tune MLE models with plain REINFORCE (one sample, one independent-sample baseline) and report gains across detection, panoptic segmentation, colourisation and captioning. Leave-one-out (RLOO) baselines [11] reduce variance at no extra model cost.

**RL for retrieval and ranking.** PG-Rank [12] trains Plackett–Luce ranking policies with policy gradients; PL-Rank [13, 14] gives efficient low-variance estimators. RELER [2] applies vMF-style embedding-space REINFORCE to text retrieval; Retrieval-GRPO [3] retrieves top-k candidates during training and scores them with a reward model; ELVA [4] studies ranking-driven RLVR for multimodal retrieval `[verify details]`. Our ranking policy follows PG-Rank with RLOO baselines.

**Differentiable AP surrogates and memory.** FastAP [8] (soft histogram binning), Smooth-AP [9] (sigmoid-relaxed ranks), listwise AP [15], ROADMAP [16] and Recall@k surrogates [17] optimise ranking metrics directly. Cross-batch memory (XBM) [10] enlarges the gallery with embeddings from earlier batches. Supervised contrastive losses [18] are the maximum-likelihood counterpart of a softmax ranking policy.

**Landmark retrieval.** Standard training sets are SfM-120k [19] and GLDv2(-clean) [20, 21]; evaluation uses Revisited Oxford/Paris [5]. Strong systems add re-ranking by local-feature verification [6, 7]. We use ALIKED features [22] and LightGlue matching [23] with MAGSAC [24] as the verifier.

## 3 Protocol

- **Scoring.** Official revisitop AP (trapezoidal integration) on ROxford5k and RParis6k, Medium and Hard protocols, query crops applied, longer image side 1024, single scale. An earlier non-official AP overstated mAP by 0.6–0.7 (F4).
- **Selection.** Checkpoints are chosen on held-out validation only: leave-one-out mAP on SfM-120k val (6,403 images) or on a GLDv2-clean validation split of 1,500 held-out landmarks (21,285 images). Benchmarks are scored once on the selected checkpoint. A historical result in this codebase selected on ROxford itself (F3).
- **Matching.** For every comparison: the same initial checkpoint, batch construction, memory bank, number of updates, learning-rate schedule family and selection rule. Hyperparameters of every arm are tuned on seed-42 validation with comparable effort (3–10 runs per arm).
- **Seeds and statistics.** 3 or 5 seeds; each seed has its own pretrained model; we report means ± std and paired differences (same pretrained seed).
- **Pre-registered success criterion** (fixed before the main experiments): against the strongest matched control, ≥ +1.0 mAP on ROxford Medium and RParis Medium averaged over seeds, the same sign in every seed, and no loss on Hard.

## 4 Methods

**Backbone and pretraining.** ResNet50 (ImageNet) + GeM pooling + linear projection to 512-d, L2-normalised (24.6M parameters). Stage 1: FastAP, PK batches (4 images per landmark), Adam, cosine schedule.
- SfM-120k: batch 32 at 1024 px, 20 epochs, lr 1e-5 (C1). An earlier 2-landmark batch gave a much weaker baseline (P1).
- GLDv2-clean: batch 128 at 512 px, 10 epochs, lr 1e-4 (G4b).

**Policies.**
1. *Gaussian embedding policy* (Pinto-style): action = normalise(μ + σε); log-density proxy κ μ·a with the action detached; single-sample (paper style) and multi-sample product rollouts with RLOO baselines and gallery gradients (D1).
2. *Graded verification reward*: exact nDCG@10 with relevance levels 1–4 from verified inlier counts, on candidate-list batches (D5–D8).
3. *Plackett–Luce ranking policy*: each batch item ranks the batch plus a 4,096-entry memory of detached embeddings; scores s = q·g/τ; 64 Gumbel top-100 rankings per query; reward AP@100; RLOO baseline. The log-probability is exact and differentiable in the scores (CPU check: cosine 0.9999 with the exact expected-reward gradient, E1).

**Controls.** FastAP continuation (no memory); softmax/SupCon over the same scores and memory; Smooth-AP over the same memory; FastAP over the same memory ("FastAP-bank"). With a verifier in the loop (§5.3): the same acquisition rule with FastAP instead of the PL objective, a static verified set, and a replay arm that trains FastAP on exactly the pairs verified by the RL arm.

**Verifier.** ALIKED (1,024 keypoints) + LightGlue (batched, 32 pairs) + MAGSAC fundamental-matrix inliers; ~140 pairs/s per GPU with caching. Same/different-landmark AUC 0.824 (0.829 for a 2,048-keypoint per-pair version on the same pairs, N1c). Match: ≥ 35 inliers (≈ 1% different-landmark false matches). Failed same-label pairs are treated as uncertain, not negative.

## 5 Results

### 5.1 SfM-120k: no reward-tuning gain under any policy

Table 1 — SfM-120k, 3 seeds, one pass over the data (2 × 1,432 batches), selection on SfM val (C4, E3, E5).

| Model | SfM val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| Stage 1 | 83.04 | 42.27 ± 1.17 | 19.41 ± 0.70 | 59.87 ± 0.95 | 32.94 ± 0.70 |
| + FastAP continuation | 82.95 | 42.51 ± 1.16 | 20.09 ± 2.46 | 60.25 ± 0.85 | 33.26 ± 0.54 |
| + Gaussian RL (single sample, anchored) | 83.10 | 42.53 ± 1.39 | 20.34 ± 1.82 | 59.80 ± 0.67 | 33.11 ± 0.66 |
| + softmax control, memory | **84.52** | 42.64 ± 1.45 | 19.51 ± 3.41 | **61.51 ± 1.27** | **34.01 ± 0.60** |
| + PL RL, memory, τ 0.01 | 84.24 | 43.01 ± 0.77 | 20.91 ± 1.11 | 61.33 ± 1.54 | 33.61 ± 1.01 |

- Gaussian RL vs FastAP continuation: ROxf M +0.03 ± 1.70, RPar M −0.45 ± 0.19 (C4).
- PL RL vs its softmax counterpart on the same memory: val −0.27 ± 0.13; benchmarks within seed noise (E5).
- Single-seed pilots tell the same story for the other designs:
  - multi-sample Gaussian RL: 83.00–83.09 val vs 83.03 init; worse without an anchor, 81.86 / 79.22 (D2, D3);
  - graded verification reward: +0.05 val nDCG@100; its differentiable ApproxNDCG counterpart degraded (D7, D8);
  - RL on top of the best surrogate model: 84.26 vs 84.29 (R1).

### 5.2 GLDv2-clean: the apparent RL gain is a memory effect

Table 2 — GLDv2-clean, **5 seeds**, 2 × 1,432 batches of 128, selection on GLDv2 val (G8, G10).

| Model | GLDv2 val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| Stage 1 | 63.01 ± 0.16 | 47.93 ± 1.90 | 24.88 ± 2.93 | 74.22 ± 1.44 | 54.73 ± 0.58 |
| + FastAP continuation (no memory) | 62.83 ± 0.15 | 48.07 ± 1.59 | 24.64 ± 2.15 | 74.17 ± 1.09 | 54.48 ± 1.06 |
| + softmax, memory | 64.13 ± 0.09 | 48.44 ± 1.84 | 23.97 ± 3.30 | 73.78 ± 1.18 | 54.92 ± 1.08 |
| + Smooth-AP, memory | 64.60 ± 0.06 | 52.36 ± 2.12 | 28.06 ± 2.48 | 76.19 ± 0.99 | 56.99 ± 0.89 |
| + **FastAP, memory** | **65.10 ± 0.07** | **52.63 ± 2.44** | **28.78 ± 3.49** | **76.88 ± 0.94** | **57.72 ± 0.87** |
| + PL RL, memory, AP@100 reward | 64.77 ± 0.14 | 52.15 ± 1.10 | 27.66 ± 1.56 | 76.19 ± 1.10 | 56.97 ± 1.05 |

Paired differences (5 seeds, mean ± std, t):
- **PL RL − FastAP continuation without memory:** ROxf M **+4.08 ± 1.74** (t 5.3), RPar M **+2.03 ± 0.38** (t 11.9). Every seed and protocol is positive, and this comparison alone would satisfy our success criterion.
- **PL RL − FastAP-bank:** val −0.33 ± 0.10 (t −7.5), ROxf M −0.48 ± 1.83, RPar M **−0.68 ± 0.41** (t −3.8), RPar H −0.74 ± 0.46.
- **PL RL − Smooth-AP:** within ±0.4 everywhere.

The ranking policy improves the pretrained model in every seed, but so do the AP surrogates once they see the same memory. The softmax objective improves validation without improving the benchmarks, which shows that validation gains do not always transfer. A 4× longer RL run (one seed) changes nothing (G9). RL started from the best surrogate model gives +0.06 val over continuing the surrogate (G11).

### 5.3 Noisy labels with a verifier in the loop: labels help, the RL objective does not

Setting (N1, N3):
- 20,000 GLDv2-clean training landmarks with all their full-GLDv2 images: 796,692 images, of which 50.9% were dropped by the official cleaning.
- Dropped images verify against kept images of their landmark in 14% of cases, vs 72% for kept images (N0).
- Stage 1 on these noisy labels: val 53.21. The same recipe on only the kept images ("perfect cleaning"): 56.82 (N1b, N2b).
- Each verifier arm runs 4 rounds × 10,000 queries × 10 candidates (≈ 400k unique verified pairs) and 2,864 updates.

Table 3 — noisy GLDv2 pool, seed 42, GLDv2 val (best round; N3).

| Acquisition (which pairs are verified) | FastAP | PL RL |
|---|---|---|
| greedy top-10 of the current model | 55.99 | 55.91 |
| Plackett–Luce sample from the top-100 | **56.17** | 56.07 |
| initial model, all verified up front | 55.92 | – |
| replay of the PL-RL arm's verified lists | 55.84 | – |
| *no verifier: FastAP-bank on noisy labels* | *55.38* | |
| *perfect-cleaning Stage 1 (reference)* | *56.82* | |

- The objective does not matter: PL RL is −0.08 to −0.10 behind FastAP under each acquisition rule, and FastAP on RL's own verified pairs is close behind.
- Verified labels do matter: every verifier arm is +0.5 to +0.8 above the noisy-label control.
- `[TODO]` seeds and benchmark scores for Table 3, if the section is kept as a main result rather than a pilot.

## 6 Why reward tuning ties

**Same information, more variance.** When the reward is computed from labels that a differentiable loss can also use (cluster labels, cleaned labels, or cached verifier outputs), the policy gradient and a surrogate of the same metric optimise similar objectives from the same information. The surrogate does so with exact, low-variance gradients. Our PL estimator matches the exact expected-reward gradient almost perfectly (E1), so variance is not the bottleneck either; the objectives simply end in the same place.

**Saturated training pools leave no reward signal.** On SfM-120k, the pretrained model retrieves a correct landmark for 97.2% of top-20 slots and 99.3% of first ranks on its own training images (A0); verifier and labels disagree on only 3.5% of pairs (A0, A0b). Re-ranking with the verifier changes the reward for 9% of queries (B0). On GLDv2-clean the same numbers are 47% recall@20 and 77% success@1 (G5). Fine-tuning helps there, but helps surrogates equally.

**Policy-dependent feedback is not enough by itself.** RL's structural advantage is learning from feedback that exists only for what the policy retrieves. But once that feedback is cached, it becomes per-pair labels that a surrogate can consume. Our replay arm makes this concrete: FastAP trained on exactly the pairs the RL arm collected is within 0.25 of it. Exploration in acquisition (sampled vs greedy) helps slightly, but it helps both objectives.

**When could RL still help?** Rewards that are non-decomposable over pairs, too expensive to cache, or defined on pipeline outputs (for example AP after re-ranking with a fixed verifier budget) are not covered by this study.

## 7 Limitations

- One backbone (ResNet50-GeM); no transformer backbone yet `[TODO DINOv2 ViT-B/14]`.
- Single-scale evaluation without learned whitening, without R1M distractors `[TODO R1M]`, and without a held-out landmark-identity test. ROxford/RParis were inspected many times during development; any future claim should be confirmed on a locked evaluation.
- Short fine-tuning budgets (about a quarter of a GLDv2 epoch, or one pass over SfM-120k). One 4× RL run showed no change.
- Hyperparameters were tuned on seed-42 validation only; all arms received comparable tuning.
- GLDv2 may contain Oxford/Paris landmarks; this affects all arms equally, but it limits comparison with methods trained elsewhere.
- Track N (Table 3) is a single-seed pilot.
- The verifier threshold was calibrated on label agreement, not on manually reviewed pairs.

## 8 Conclusion

Under matched controls, policy-gradient reward tuning of a landmark descriptor reaches, but does not exceed, differentiable AP surrogates. This holds on SfM-120k, on GLDv2-clean and with a geometric verifier on noisy GLDv2, and for five policy designs. Apparent gains came from weaker controls, and in our case from the cross-batch memory rather than the reward. We recommend that RL-for-retrieval claims report a surrogate trained on the same data, memory and budget, select on held-out validation, and separate label acquisition from the optimisation objective.

## Compute

About 168 GPU-hours on single NVIDIA RTX PRO 6000 / A40 / A100 GPUs, from 2026-10-08 to 2026-10-14. That includes 83 h of Stage 1 pretraining (15 runs), 52 h of fine-tuning (102 runs), 19 h of verifier-in-the-loop runs and ~6 h of verification and diagnostics. Exact per-job records are in PROGRESS.md and `sacct`.

## References (to verify against official sources)

1. A. Susano Pinto, A. Kolesnikov, Y. Shi, L. Beyer, X. Zhai. Tuning computer vision models with task rewards. ICML 2023. arXiv:2302.08242
2. Q. Liu et al. Learning to Retrieve via Reinforcement Learning in Embedding Space (RELER). arXiv:2610.07731 `[verify authors/venue]`
3. X. Liu et al. Retrieval-GRPO: A Multi-Objective Reinforcement Learning Framework for Dense Retrieval in Taobao Search. arXiv:2511.13885
4. ELVA. arXiv:2606.20280, ECCV 2026 `[verify title/authors]`
5. F. Radenović, A. Iscen, G. Tolias, Y. Avrithis, O. Chum. Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking. CVPR 2018
6. B. Cao, A. Araujo, J. Sim. Unifying Deep Local and Global Features for Image Search (DELG). ECCV 2020
7. S. Lee et al. Correlation Verification for Image Retrieval (CVNet). CVPR 2022
8. F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. Deep Metric Learning to Rank (FastAP). CVPR 2019
9. A. Brown, W. Xie, V. Kalogeiton, A. Zisserman. Smooth-AP. ECCV 2020
10. X. Wang, H. Zhang, W. Huang, M. Scott. Cross-Batch Memory for Embedding Learning. CVPR 2020
11. W. Kool, H. van Hoof, M. Welling. Buy 4 REINFORCE Samples, Get a Baseline for Free! ICLR 2019 workshop
12. A. Singh, T. Joachims. Policy Learning for Fairness in Ranking (PG-Rank). NeurIPS 2019
13. H. Oosterhuis. Computationally Efficient Optimization of Plackett-Luce Ranking Models. SIGIR 2021
14. H. Oosterhuis. Learning-to-Rank at the Speed of Sampling (PL-Rank-3). SIGIR 2022
15. J. Revaud, J. Almazán, R. Rezende, C. de Souza. Learning with Average Precision: Training Image Retrieval with a Listwise Loss. ICCV 2019
16. E. Ramzi et al. Robust and Decomposable Average Precision for Image Retrieval (ROADMAP). NeurIPS 2021
17. Y. Patel, G. Tolias, J. Matas. Recall@k Surrogate Loss with Large Batches and Similarity Mixup. CVPR 2022
18. P. Khosla et al. Supervised Contrastive Learning. NeurIPS 2020
19. F. Radenović, G. Tolias, O. Chum. Fine-tuning CNN Image Retrieval with No Human Annotation. TPAMI 2018
20. T. Weyand, A. Araujo, B. Cao, J. Sim. Google Landmarks Dataset v2. CVPR 2020
21. S. Yokoo et al. Two-stage Discriminative Re-ranking for Large-scale Landmark Retrieval (GLDv2-clean). CVPR Workshops 2020, arXiv:2003.11211
22. X. Zhao et al. ALIKED. IEEE TIM 2023 `[verify]`
23. P. Lindenberger, P.-E. Sarlin, M. Pollefeys. LightGlue. ICCV 2023
24. D. Barath et al. MAGSAC++. CVPR 2020 `[verify which MAGSAC variant OpenCV USAC_MAGSAC implements]`

---

## Before any submission
- `[TODO]` R1M distractors and a locked held-out evaluation for the final numbers; DINOv2 backbone; seeds for Table 3.
- Fix the historical claims in README.md (preliminary, test-selected results).
- Recheck novelty against RELER, ELVA, Retrieval-GRPO and recent RL-for-retrieval papers; read ELVA in full before describing it; check venue deadlines on official sites.
- Regenerate all tables from `results/runs/*/final_eval.json` with one script, rather than copying them from PROGRESS.md.
- Push to a Git remote (none configured yet); the user decides where.
