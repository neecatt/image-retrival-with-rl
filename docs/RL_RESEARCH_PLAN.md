# Making reward tuning work for landmark retrieval: research plan

Written 2026-10-10 after the 3-seed comparison in [PROGRESS.md](PROGRESS.md) (C4) showed no gain from the current RL method. This document explains why, what the literature says, and the path most likely to produce a real, defensible RL result. Status of each step is tracked in [ROADMAP.md](ROADMAP.md).

## 0. Execution status

Updated as work proceeds; evidence goes to [PROGRESS.md](PROGRESS.md).

**Phase 1: competent estimator**
- [x] 1.1 CPU test (2026-10-10, PROGRESS D1): single-update alignment with the true gradient rises from 0.12 (current trainer) to 0.69 (S=16) and 0.89 (S=64); training the gallery removes the bias (mean cos 0.99)
- [x] 1.2 Implemented (commit `feat(rl): use the multi-sample estimator...`): `training.stage2.num_samples`, `memory_store: mean`, `train_gallery`, `normalize_advantage`; CPU-tested with a tiny model
- [ ] 1.3 GPU smoke test: folded into the 1.4 pilot jobs (first minutes of each job)
- [ ] 1.4 Validation pilot **running** (jobs 4521459–4521462, 2026-10-10): seed-42 Stage 1, batch 32, one pass, σ 0.05, lr 2e-6, mean memory; S=16 / S=64 / S=16 with consistency 500 / S=16 with normalised advantages. Bar: control 82.96, old RL 83.25
- [ ] 1.5 Gate: ≥ +0.3 SfM val over control → 3 seeds with final evaluation

**Phase 2 preparation (in parallel)**
- [ ] 2.1 Choose the local feature + matcher, verify licences (default: ALIKED + LightGlue; no push or release decisions made yet)
- [ ] 2.2 1% oracle pilot (~200 queries × 50 candidates): throughput and agreement with SfM landmark labels → full labelling cost estimate
- [ ] 2.3 Implement the graded differentiable control (LambdaLoss-style or graded Smooth-AP)

**Later**
- [ ] Phase 2 main experiment: graded RL vs graded control vs binary control, 3 seeds, Medium and Hard
- [ ] Phase 3 Plackett–Luce ranking actions; Phase 4 DINOv2, R1M, multi-scale + whitening
- [ ] Results freeze target: 2026-10-28

**Open decisions for the user:** non-commercial matcher licences acceptable? Git remote for pushing (nothing pushed; README still has old claims to correct first)?

## 1. Where we stand (measured, not assumed)

| | ROxford M / H | RParis M / H |
|---|---|---|
| New Stage 1, 3 seeds | 42.27 / 19.41 | 59.87 / 32.94 |
| + continued metric learning | 42.51 / 20.09 | 60.25 / 33.26 |
| + current RL | 42.53 / 20.34 | 59.80 / 33.11 |

RL minus control: +0.03 / +0.24 on ROxford, −0.45 / −0.15 on RParis. The diagnostics from the pilots explain a lot of this:

- **Too few samples.** Each query gets one sampled embedding plus one baseline sample. With Gaussian noise on a 512-d embedding, the REINFORCE update is a one-sample, zeroth-order estimate of the gradient of a smoothed AP. Its useful direction is a small part of a 512-d noise vector.
- **Gallery is not part of the policy gradient.** Gallery embeddings are sampled but get no score-function gradient (P6), and the memory bank stores noisy samples from older weights.
- **Binary, saturated reward.** AP against "same landmark or not" is ~0.8–1.0 in most batches, so the advantage is near zero for many queries (B2: 9–42% of advantages exactly zero).
- **Short runs.** Our RL arm ran ~2,900 updates. The source paper ran 30k–100k reward-tuning steps (see §2.1).
- **Strong differentiable competitor.** The control optimizes FastAP, a smooth AP surrogate with exact gradients over the same binary labels. With the same binary labels as reward, RL has little room to add anything.

## 2. What the literature says

### 2.1 The source paper: Susano Pinto et al., ICML 2023 (arXiv 2302.08242)
Read in full (v1 HTML, Sec. 3–6 and App. A).
- MLE pretraining, then plain REINFORCE (Alg. 2): **one sample for the gradient, one independent sample as the baseline**; captioning used 7 other samples for the baseline. No KL or anchoring to the MLE model ("we saw no need"), no MLE loss mixed in.
- **All tasks have discrete token-sequence outputs** (panoptic, detection, colorization, captioning). Sampling an output is natural and the reward is evaluated on a decoded structure.
- Small LR (1e-6 to 3e-7, encoder 10× smaller for captioning), batch 128–512, **30k–100k steps** for panoptic/detection, warm-up then constant LR.
- Why it worked, in their analysis (Sec. 5.1): the MLE model already contains high-reward samples that likelihood ranking cannot pick out; reward tuning moves probability mass to them.
- Gains: panoptic PQ 43.1 → 46.1; detection mAP 39.2 → 54.3; captioning CIDEr 120.0 → 134.5 (ViT-B). No seeds or error bars reported.

**Implication:** the paper does not cover continuous embedding outputs, and its precondition (high-reward outputs exist in the model's own sample distribution and are hard to reach by maximum likelihood) needs to be argued for retrieval rather than assumed.

### 2.2 Closest prior work to this project: RELER (Liu et al., arXiv 2610.07731, 6 Oct 2026, text retrieval)
Read from the arXiv HTML; appendix details beyond A.1 came from a summary and are marked as such in the reading notes.
- **Same idea as our Stage 2:** vMF policies centered on normalized query **and** document embeddings, REINFORCE with RLOO, deterministic retrieval at test time, applied after contrastive pretraining.
- **Key engineering we lack:**
  - **G = 64 samples per query with product rollouts:** G query × G document samples, G² scored pairings, no extra encoder passes.
  - **Documents are sampled and receive gradient.** Ablation: sampling only queries 15.85 vs both 23.38 nDCG@10.
  - **Conditional-mean projection (CMP):** projects each sampled action onto the span that determines the reward. Reported 291–590× lower gradient variance; 21.91 → 23.38 nDCG@10.
  - **Graded reward from teacher relevance labels (nDCG):** graded 22.72 vs binary nDCG 17.45 vs binary MRR 18.90.
- **Matched control reported** (InfoNCE and LambdaLoss, same steps, 3 seeds): RELER beats InfoNCE clearly on a weak backbone (BGE-M3 12.63 → 14.49), modestly on Qwen3-Emb-0.6B (22.64 → 23.38), and within noise on 4B. From a general LLM init there is no advantage (Appendix E).

**Implications for us:**
1. Our null result is consistent with RELER's ablations: one-sided query sampling, few samples, no variance reduction and a binary reward are exactly the configurations that performed worst there.
2. **Novelty:** "vMF embedding-space REINFORCE for retrieval" is now published. Porting it to landmark images would be an adaptation study, not a method contribution. The contribution has to come from something RELER does not do (§4).
3. The biggest lever in RELER is the **graded reward**. Graded relevance is also the thing a binary metric-learning loss cannot use, which is where RL earns its place.

### 2.3 Ranking-policy alternative: Plackett–Luce policy gradients
- PG-Rank / Fair-PG-Rank (Singh & Joachims, NeurIPS 2019; Yadav, Du & Joachims, SIGIR 2021) and PL-Rank / PL-Rank-3 (Oosterhuis, SIGIR 2021 / 2022) treat the **ranking itself** as the stochastic action: a Plackett–Luce distribution over items given model scores, sampled with the Gumbel top-k trick, trained with policy gradients on any ranking metric. PL-Rank-3 reaches unbiased gradient estimates at roughly sorting cost.
- For retrieval this is a natural action space: sampling a ranking needs no extra network forward, so hundreds of samples per query are cheap, and the gradient reaches both query and gallery scores.
- I found **no published application to image/landmark retrieval or deep metric learning** in these searches. The searches were not exhaustive, so treat this as "not found" rather than "novel".

### 2.4 Strong non-RL baselines to keep the comparison honest
- Differentiable AP/recall surrogates: FastAP (current control), Smooth-AP, ROADMAP (Ramzi et al., NeurIPS 2021), blackbox rank differentiation (Rolínek et al., CVPR 2020), Recall@k surrogate with very large batches (Patel et al., CVPR 2022), listwise AP loss with thousands of images per batch (Revaud et al., ICCV 2019).
- Hard negatives across batches: XBM (Wang et al., CVPR 2020) shows that stored embeddings drift slowly enough to serve as a large negative pool. Store **means**, not noisy samples.
- Landmark-specific training (Radenović, Tolias & Chum, TPAMI 2018): GeM, hard positives/negatives from SfM geometry, learned whitening. Current global-descriptor and reranking state of the art includes SuperGlobal (ICCV 2023), CVNet (CVPR 2022) and AMES (ECCV 2024); those use GLDv2 and reranking, so they are context, not direct competitors for an SfM-120k ResNet50 study.
- Geometric verification is non-differentiable. Learned rerankers (CVNet, Reranking Transformers) and distillation approaches exist, but I did not find a method that uses verification output as an RL reward for a global descriptor.

## 3. The argument RL has to make

Score-function RL is a higher-variance optimizer than an exact gradient of the same smooth objective. So "RL optimizes AP directly" is not, on its own, a reason it should beat FastAP on binary labels; our C4 result is what theory predicts. RL becomes the right tool when **the reward contains information or structure that a differentiable surrogate cannot use**:

1. **Graded relevance from a non-differentiable judge.** In landmark retrieval, geometric verification of local features (inlier count after RANSAC) gives a graded, view-aware relevance: same landmark but different facade gets low relevance, while a true overlapping view gets high relevance. It is non-differentiable and expensive, which is exactly the setting where policy gradients are justified. It also matches the Revisited Easy/Hard/Junk semantics better than "same 3D model".
2. **Set-level, top-heavy rewards on large galleries:** truncated AP@100 or a Revisited-style reward with junk ignored, computed against thousands of mined hard negatives.
3. **Rankings as actions** (Plackett–Luce) instead of noisy embeddings, giving cheap, many-sample estimates.

## 4. The path, with decision gates

Each phase has a falsifying experiment and a matched control. Selection is on SfM-120k validation only; ROxford/RParis are scored once at the end of a phase for the validation-selected setting. Seeds: screen with 1, confirm with 3.

### Phase 1: make the existing estimator competent (≈2 days; ~6–10 GPU-h)
Port RELER's variance reduction to the image setting:
- G samples per query and per gallery item (G = 16–64), product rollouts on the precomputed similarity matrix (no extra forward), RLOO baseline, per-query reward standardization.
- Score-function gradient for gallery samples too (fixes P6).
- Memory bank stores **mean** embeddings and is refreshed (XBM-style), not noisy samples.
- Optional: conditional-mean projection.
- Longer runs: 3–5 passes over SfM-120k at lr ~1e-6 to 2e-6 (closer to the source paper's step counts).

**Unit test before GPU runs:** on a small synthetic retrieval problem, compare the estimator against the exact gradient of the smoothed objective (extend `scripts/audit_estimator.py`), and measure gradient variance with G = 1 vs G = 64.
**Gate:** gradient-variance reduction is visible in the diagnostics, and SfM val mAP ≥ control + 0.3 on 1 seed. If not, RL on binary labels is not the contribution; go to Phase 2 anyway, since Phase 2 is the stronger story.

### Phase 2: graded, verification-based reward (the main contribution candidate; ≈5–6 days; ~20–40 GPU-h)
- **Offline relevance oracle:**
  1. Extract local features once for SfM-120k train and val (91k + 6.4k images).
  2. For each query, build a candidate set: same-cluster images plus the top-k hard negatives under the Stage 1 model.
  3. Verify each pair with a local matcher plus RANSAC and store inlier counts as graded relevance (e.g. 0 / 1 / 2 / 3 by inlier bins).

  About 20k queries × 50 candidates ≈ 1M pairs; budget a few GPU-hours, measured on a 1% pilot first. Matcher choice must respect licences (some popular local features are non-commercial only); record it.
- **Reward:** graded nDCG@k or graded AP over the candidate set plus memory negatives.
- **Controls (essential, mirroring RELER):**
  - (a) continued FastAP on binary labels;
  - (b) a **graded differentiable listwise loss** on the same graded labels (e.g. LambdaLoss-style or graded Smooth-AP), so the comparison isolates "RL vs surrogate" rather than "graded vs binary labels";
  - (c) RL with the binary reward (Phase 1).
- **Validation:** SfM val with the same graded labels, plus the binary SfM val mAP.
- **Gate:** RL with the graded reward beats both (a) and (b) on SfM val with 3 seeds (paired, same sign in all seeds), then on ROxford/RParis Medium **and** Hard. The Hard protocol is where view-aware relevance should matter most.

### Phase 3: rankings as actions (≈3–4 days; runs alongside Phase 2 once the reward exists)
- Plackett–Luce policy over the gallery given scores s = q·g/τ, Gumbel top-k sampling (k = 100), 64–256 rankings per query, RLOO, PL-Rank-style estimator. Reward is the same graded nDCG/AP.
- Compare with the Phase 2 vMF policy at a matched budget. Whichever wins on validation goes forward. This is also a second, independent contribution candidate, given no image-retrieval application was found.

### Phase 4: scale and generality (≈4–5 days)
- **Backbone:** DINOv2 ViT-B/14 (Stage C), with the same Stage 1 → control vs RL protocol. RELER saw larger RL gains on weaker starting points, so measure the gain as a function of Stage 1 strength rather than expecting it to grow automatically.
- **Evaluation:** R1M distractors for the final models (large galleries are where ranking rewards should matter); multi-scale testing and learned whitening applied equally to all arms.
- **Optional:** GLDv2-clean subset if time allows.

## 5. Success criteria, decided now

A result counts as "RL helps" only if, against the strongest matched control (graded surrogate when graded labels are used):
- paired gain ≥ +1.0 mAP on ROxford Medium and RParis Medium averaged over 3 seeds, with the same sign in every seed;
- no loss on the Hard protocol;
- compute reported for both arms.

Anything smaller is reported as it is (e.g. "RL matches a tuned surrogate at +X% compute").

## 6. Risks and how they are handled

| Risk | Mitigation |
|---|---|
| Verification oracle is too slow or noisy | 1% pilot first to measure throughput and label agreement with SfM clusters; fall back to graded labels from the Stage 1 teacher's similarity |
| Graded surrogate (control b) matches RL | Still a publishable negative/neutral result with a clear protocol; Phase 3 offers a second mechanism |
| Reward hacking (Pinto Sec. 6) | Validation on binary SfM val and on held-out benchmarks; inspect top-ranked examples |
| Novelty overlap with RELER | Position RELER as closest prior work; claim only what differs (images, verification-graded rewards, PL ranking actions, landmark protocol) after a targeted literature check |
| Deadline (CVPR 2027) | Reverify official dates; Phase 1–2 fit within the sprint ending 2026-10-28, and Phases 3–4 are the stretch |

## 7. Sources

- Susano Pinto et al., Tuning computer vision models with task rewards, ICML 2023: https://arxiv.org/abs/2302.08242
- Liu et al., Learning to Retrieve via Reinforcement Learning in Embedding Space (RELER), 2026: https://arxiv.org/abs/2610.07731 (code: https://github.com/liuqi6777/RELER)
- Kool, van Hoof & Welling, Buy 4 REINFORCE samples, get a baseline for free!, ICLR 2019 workshop: https://mlanthology.org/iclrw/2019/kool2019iclrw-buy
- Ahmadian et al., Back to Basics: Revisiting REINFORCE Style Optimization for RLHF, 2024: https://arxiv.org/abs/2402.14740
- Singh & Joachims, Policy Learning for Fairness in Ranking, NeurIPS 2019: https://arxiv.org/abs/1902.04056
- Yadav, Du & Joachims, Policy-Gradient Training of Fair and Unbiased Ranking Functions, SIGIR 2021: https://arxiv.org/abs/1911.08054
- Oosterhuis, Computationally Efficient Optimization of Plackett-Luce Ranking Models (PL-Rank), SIGIR 2021: https://arxiv.org/abs/2105.00855
- Oosterhuis, Learning-to-Rank at the Speed of Sampling (PL-Rank-3), SIGIR 2022: https://arxiv.org/abs/2204.10872
- Gao et al., Policy-Gradient Training of Language Models for Ranking, 2023: https://arxiv.org/abs/2310.04407
- Wang et al., Cross-Batch Memory for Embedding Learning, CVPR 2020: https://arxiv.org/abs/1912.06798
- Ramzi et al., Robust and Decomposable Average Precision for Image Retrieval (ROADMAP), NeurIPS 2021: https://arxiv.org/abs/2110.01445
- Rolínek et al., Optimizing Rank-based Metrics with Blackbox Differentiation, CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Rolinek_Optimizing_Rank-Based_Metrics_With_Blackbox_Differentiation_CVPR_2020_paper.html
- Patel, Tolias & Matas, Recall@k Surrogate Loss with Large Batches and Similarity Mixup, CVPR 2022: https://arxiv.org/abs/2108.11179
- Revaud et al., Learning with Average Precision: Training Image Retrieval with a Listwise Loss, ICCV 2019: https://arxiv.org/abs/1906.07589
- Radenović, Tolias & Chum, Fine-tuning CNN Image Retrieval with No Human Annotation, TPAMI 2018: https://arxiv.org/abs/1711.02512
- Shao et al., Global Features are All You Need for Image Retrieval and Reranking (SuperGlobal), ICCV 2023: https://arxiv.org/abs/2308.06954
- Lee et al., Correlation Verification for Image Retrieval (CVNet), CVPR 2022: https://arxiv.org/abs/2204.01458
