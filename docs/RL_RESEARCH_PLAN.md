# Making reward tuning work for landmark retrieval: research plan

Written 2026-10-10 after the 3-seed comparison in [PROGRESS.md](PROGRESS.md) (C4) showed no gain from the current RL method. This document explains why, what the literature says, and the path most likely to produce a real, defensible RL result. Status of each step is tracked in [ROADMAP.md](ROADMAP.md).

## 0. Execution status

Updated as work proceeds; evidence goes to [PROGRESS.md](PROGRESS.md).

**Phase 1: competent estimator**
- [x] 1.1 CPU test (2026-10-10, PROGRESS D1): single-update alignment with the true gradient rises from 0.12 (current trainer) to 0.69 (S=16) and 0.89 (S=64); training the gallery removes the bias (mean cos 0.99)
- [x] 1.2 Implemented (commit `feat(rl): use the multi-sample estimator...`): `training.stage2.num_samples`, `memory_store: mean`, `train_gallery`, `normalize_advantage`; CPU-tested with a tiny model
- [x] 1.3 GPU smoke test passed inside the 1.4 jobs (all four completed)
- [x] 1.4 Validation pilot done (PROGRESS D2): 82.68–83.09 vs start 83.03 and control 82.96; signal is far steadier but the anchored model barely moves. Follow-up with weak/no anchoring at S=64 running (jobs 4521748/4521749)
- [x] 1.5 Gate (≥ 83.26 SfM val): **not met**. Weak/no anchoring makes it worse (81.86 / 79.22, PROGRESS D3). Phase 1 closed: binary-reward RL is inert or harmful; Phase 2 is the path

**Phase 2 preparation (in parallel)**
- [x] 2.1 Matcher chosen 2026-10-10: ALIKED (BSD-3) + LightGlue (Apache-2.0, pinned commit eb42fee in `requirements-oracle.txt`); licences checked from the official LICENSE files. Original SuperPoint is non-commercial only, so it is not used
- [x] 2.2 Oracle pilot done (PROGRESS D4): 61 pairs/s; inliers grade same-landmark pairs from 11 (p10) to 750 (p90), noise floor ~12 for other landmarks; grades 0/1/2/3 at 25/100/300 inliers; full labelling ≈ 0.5M same-landmark pairs
- [x] 2.4 Full graded labelling done (PROGRESS D5): 429,672 train and 149,371 val verified pairs. Relevance levels changed to 0 = other landmark, 1–4 = same landmark by inliers, so other views stay positives
- [x] 2.5 Graded pilot done (2026-10-10, PROGRESS D7–D8): **gate not met**. Graded RL is inert at σ 0.05 / 0.02 / 0.01, S 16 / 64 and cw 5000 (best +0.04 val nDCG@100, +0.16 mAP, not sustained), and degrades at cw 500. The ApproxNDCG control degrades (graded −3 / −6, binary labels −1.8 / −3.2 nDCG@100 / mAP). No 3-seed runs. Phase 2 with a Gaussian embedding policy is closed as negative
- [x] 2.3 Graded training stack implemented and CPU-tested (2026-10-10): `losses/graded.py` (exact nDCG@k reward, ApproxNDCG control), `data/graded_lists.py` (candidate-list batches, masks for the query itself and unverified pairs), `training/graded_trainer.py` (modes `rl` / `approxndcg` on identical batches), graded leave-one-out val nDCG in `evaluation/sfm_val.py`, entry `scripts/train_graded.py` + `scripts/train_graded_job.sh`

**Revised path (§4b, 2026-10-11)**
- [x] A0 done (PROGRESS A0): verifier finds real label noise (17% of retrieved "negatives" are verified matches; contact sheets confirm), but the model has memorised SfM-120k train (97% of top-20 already positive), so only 0.5% of retrieved pairs are exploitable. Gate formally passed (condition 1), condition 2 not met. Track A needs an unseen pool (A0b)
- [x] A0b done on val (PROGRESS A0b): unseen images give more retrieval errors (9.6% of top-20, 34% of queries) but the cluster labels are right about 94% of them; hidden matches stay ~0.5% of pairs, disagreement 3.6%. Gate: condition 1 barely met, condition 2 not met. Track A on SfM-120k is not promising; GLDv2 version reduces to exploration vs iterative re-mining (low prior). User chose (2026-10-11) the Pinto recipe next
- [x] R1 done (PROGRESS R1): from the softmax-control model (84.59) every arm drifts down; best RL 84.26 vs best control 84.29. Gate not met; no seeds run
- [x] B0 Track B headroom check (PROGRESS B0): shortlist recall@20 is 97% on the training pool, re-ranking changes the reward for 9% of queries. Track B not built on SfM-120k; the training pool is saturated
- [ ] A1 verifier-in-the-loop RL infrastructure (pair cache, rollout rounds, PPO-clip + GRPO + KL)
- [ ] A2 pilot vs static (C2) and iterative re-mining (C3) controls at equal verifier budget
- [ ] A3 5 seeds with final evaluation; Track B (pipeline reward) if A passes; Track C upgrades for all arms

**Track G: move training to GLDv2-clean (decided 2026-10-11)**
Reason: SfM-120k train is saturated for Stage 1 (A0, A0b, B0), so no reward computed on it leaves room for RL.
- [x] G1 Download done 2026-10-11 (500/500 tars md5-verified, 498 GB apparent; metadata in `$HPCVAULT/datasets/gldv2/metadata`). Clean subset: 1,580,470 images over 81,313 landmarks (median 9 per landmark); images stored at ≤ 800 px. **Vault charges ~2× apparent size, so quota is now 1,217 / 1,049 GB soft (hard 2,097 GB)**; raw tars are re-downloadable (fetch_train.sh) and are candidates for removal after G2, with the user's approval
- [x] G2 Packing done (PROGRESS G1–G2): 1,580,470 images, 0 failures, 75 GB; train 1,546,857 / 79,813 landmarks, val 21,285 / 1,500 landmarks
- [x] G3 Loader (`load_gldv2c`), staging (`stage_datasets.py gldv2c`), `dataset.name=gldv2c` in train.py, GLDv2 val in final evaluation; tested on 2 shards
- [x] G4 GLDv2 Stage 1 seed 42 done (PROGRESS G4b): val 62.89; ROxf M/H 49.93 / 27.15, RPar M/H 72.27 / 53.79 (vs 42.33 / 19.46 / 59.17 / 32.44 for SfM Stage 1). Seeds 1–2 submitted
- [x] G5 Saturation check passed (PROGRESS G5): recall@20 47% and success@1 77% on the GLDv2 train pool (SfM: 97% / 99%); label/verifier disagreement 11.4%
- [x] G6 pilots done (PROGRESS G6, G6b): at lr 1e-6 PL RL 64.61 vs softmax 64.13 vs FastAP 62.75 (init 62.89); PL ahead in every matched setting. **Gate met**
- [x] G7 3 seeds done (PROGRESS G7; superseded by G10): §5 criterion met over 3 seeds against weak controls only. PL RL vs FastAP control ROxf M/H +2.98 / +2.32, RPar M/H +2.13 / +2.44; vs softmax control +4.25 / +4.67 / +2.63 / +2.15; positive in every seed and column
- [x] G8 5 seeds done (PROGRESS G8; superseded by G10, controls lacked the bank): PL RL beats FastAP (no bank) and softmax controls in all 5 seeds on all four protocols (vs FastAP ROxf M/H +4.08 / +3.02, RPar M/H +2.03 / +2.49)
- [x] G9a Stronger-control pilot done (PROGRESS G9): FastAP on the bank 65.05 val (> PL 64.61), Smooth-AP 64.56. Both on 5 seeds with final evaluation submitted (4553898–907); long-RL preview running (4553037)
- [x] G9b 5 seeds vs stronger controls done (PROGRESS G10): **PL RL ties Smooth-AP and loses to FastAP on the bank** (RPar M −0.68, t −3.8). G7/G8 success claim withdrawn; the gain came from the bank. Long-RL 4× preview: no change
- [x] G11 RL on top of the best surrogate (PROGRESS G11): best RL 65.13 vs continued FastAP-bank 65.07 (init 65.05). Gate not met; write-up started in [PAPER_DRAFT.md](PAPER_DRAFT.md)

**Track N: verifier-in-the-loop RL on noisy full GLDv2 (user decision 2026-10-13: a positive RL result matters more than the deadline)**
- Data facts: for the 79,813 GLDv2-clean training landmarks, full GLDv2 has 3,159,948 images, of which 1,613,091 (51%) were dropped by the cleaning.
- Fair controls (decided before building): N-C1 FastAP-bank on noisy labels; N-C2 FastAP-bank on labels cleaned once with the verifier (same verifier budget); **N-C3 iterative re-mining: verify the current model's greedy top-k every round and train FastAP-bank on the verified labels (strongest)**. RL must beat N-C3 to claim a gain; its extra ingredients are only exploration beyond the greedy top-k and the reward-shaped objective, so the odds are uncertain.
- [x] N0 Feasibility passed (PROGRESS N0): dropped images verify 14% (≥ 25 inliers) vs 72% for kept ones; throughput 14 pairs/s is the bottleneck
- [ ] N1 Build: pack the noisy training pool (subset of landmarks), fast batched verifier with feature cache, Stage 1 on noisy labels, verified-pair cache, rounds of RL and of the three controls at equal verifier budget
  - [x] N1a noisy pool done: 796,692 train images, 50.9% dropped by the cleaning
  - [x] N1b noisy-label Stage 1 done: val 53.21; ROxf M/H 35.68 / 16.85, RPar M/H 62.05 / 40.88
  - [x] N1c batched verifier done: 87.9 pairs/s, AUC 0.824 (2,048-kpt oracle 0.829); match threshold ~35 inliers.
  - [ ] N1d round-based trainer written and CPU-tested (`training/verified_rounds.py`, `scripts/train_verified.py`): per round embed the pool, pick k = 10 candidates per query (rl: PL sample from the top-100 at tau 0.05; remine: greedy; static: initial model's greedy, whole budget in round 0), verify uncached pairs, train on verified query lists (match = >= 35 inliers; unknown same-label items masked) with PL policy gradient (rl) or FastAP (controls). Default budget 4 rounds x 10,000 queries x 10 = 400k verified pairs per arm. GPU smoke runs passed (PROGRESS N1d)
  - [x] N2 first pilot (4560131–36) OOM-killed while embedding the pool (16 RANSAC workers forked after CUDA + 12 loaders > 92 GB). Noisy-label FastAP-bank control finished: val 55.38 (lr 1e-6) / 55.32 (lr 3e-7), init 53.21
  - [ ] N2b (second OOM in verification, fixed by chunked loading; resubmitted as smoke 4570353 then 4570354–59; smoke failed on a feature-cache reset bug (KeyError), fixed, resubmitted as smoke 4571067 then 4571069–4571075) redesigned pilot after the 2026-10-14 research review: acquisition {greedy, sample, static, replay} x objective {fastap, pl}; replay = FastAP trained on exactly the pairs the RL arm verified; failed same-label verifications treated as uncertain. Full-pool memory smoke 4561437, then 4561438–41, 4561443 (static), 4561444 (replay, after the RL arm). LR fixed from the GLDv2 pilots: PL 1e-6, FastAP 3e-7
  - [x] N1e "perfect cleaning" reference done: val 56.82, ROxf M/H 44.10 / 21.42, RPar M/H 65.12 / 45.35 (noisy Stage 1: 53.21; 35.68 / 16.85; 62.05 / 40.88)
  - Noisy-label control (FastAP-bank on the noisy labels, same steps) runs through the existing finetune_job.sh path
- [x] ~~G9b Best control vs PL on 5 seeds~~ superseded: done as G10 (see entry above); a matched longer-budget study is still open
- [ ] G9c Ablations (τ, reward k, budget, stronger controls: Smooth-AP / Recall@k surrogate on the same bank), R1M distractors, multi-scale, second backbone (DINOv2)
- Caveat to report: GLDv2 may contain Oxford/Paris landmarks; training on GLDv2-clean for ROxford/RParis is standard practice (DELG, CVNet, SuperGlobal) but must be stated

**Later**
- [ ] ~~Phase 2 main experiment~~ dropped: no arm passed the 2.5 gate
- [ ] 3.0 Hard-positive SfM val metric (level 1–2 positives only) to check that selection can see benchmark-relevant changes
**Phase 3: rankings as actions**
- [x] 3.1 Plackett–Luce policy implemented (2026-10-10, commit `feat(rl): add Plackett-Luce ranking policy...`, `training/pl_policy.py`, `training.stage2.policy=plackett_luce`). CPU check (PROGRESS E1): exact log-prob, Gumbel sampler matches PL, gradient estimate vs exact gradient cosine 0.9999
- [x] 3.2 Pilot done (PROGRESS E2): τ 0.02 without anchor **84.23 SfM val (+1.20 over init)**, τ 0.01 83.87 (+0.84), τ 0.05 81.48 (−1.55)
- [x] 3.3 Gate (> 83.10 SfM val) **met**. 3 seeds × τ {0.02, 0.01} with final evaluation submitted (4525409–14)
- [x] 3.3b 3 seeds done (PROGRESS E3): PL τ 0.01 vs Stage 1 +1.21 SfM val, ROxf M/H +0.74 / +1.50, RPar M/H +1.46 / +0.67; vs FastAP control +0.50 / +0.82 / +1.08 / +0.35 (one seed negative on ROxf M). §5 criterion met on RParis Medium only
- [x] 3.4 Memory-bank control done (PROGRESS E5): over 3 seeds the softmax control equals PL RL (PL − control: SfM val −0.27, ROxf M +0.37 ± 1.19, RPar M −0.17 ± 0.47). **The PL gain is the bank + listwise objective, not RL.** With a binary reward PL RL has no advantage over its ML counterpart
- [ ] 3.6 RL with a reward the control cannot optimise directly: graded verification relevance on the bank + all-vs-all setup, against a graded softmax control
- [ ] 3.5 Five seeds: Stage 1 seeds 3–4 training (4525720–21), then FastAP control, softmax control (best τ) and PL τ 0.01 on seeds 3–4 with final evaluation
- [ ] Phase 4 DINOv2, R1M, multi-scale + whitening
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

## 4b. Revised path after Phase 3 (written 2026-10-11)

### Why a more complex RL algorithm alone will not help
- PPO, GRPO, actor-critic and PL-Rank are better ways to estimate and follow the gradient of the **same** objective (expected reward). Our PL estimator already matches the exact gradient of expected AP@k (cosine 0.9999, PROGRESS E1), so a lower-variance algorithm has nothing left to fix on that objective.
- E5 shows that with a reward computable from per-pair labels known in advance (binary AP from SfM clusters), RL and its maximum-likelihood counterpart (softmax on the same bank) end up in the same place. The bottleneck is **what** is optimised and **what information** the learner gets, not how the gradient is estimated.
- Where RL has a structural advantage, according to the RL-for-ranking and RLVR literature, is when the reward
  1. comes from an **expensive oracle that can only be queried for items the current policy actually retrieves** (bandit or partial feedback, as in online learning to rank and in RL with verifiable rewards for LLMs, where no target exists in advance); or
  2. is a **non-decomposable function of the whole retrieved list or downstream pipeline** (e.g. AP after a non-differentiable re-ranking step), so no per-pair surrogate optimises it exactly.
- Complex algorithms become worth it exactly there: when rollouts are expensive (verifier calls), **PPO-style clipped importance ratios** allow several updates per batch of verified rollouts, and **GRPO-style group normalisation** stabilises learning across queries of very different difficulty.

### Track A (main): verifier-in-the-loop RL ("RLVR for retrieval")
- **Idea:** the policy (PL ranking over the bank, as in Phase 3) retrieves a sampled top-k for each query from the training pool. The geometric verifier (ALIKED + LightGlue + MAGSAC, PROGRESS D4) labels only those retrieved pairs, with results cached. Reward: graded AP/nDCG from verified inliers. Pairs never retrieved are never labelled.
- **Why RL should win here:** a differentiable loss needs labels for the pairs it trains on. Labelling all pairs is infeasible beyond small sets (91k² ≈ 8·10⁹ pairs in SfM-120k alone), so a control can only use pairs chosen in advance. RL gathers labels where the current policy ranks, and explores beyond the greedy top-k by sampling. It also corrects label noise: SfM-120k labels come from 3D-model clusters, so views of the same building in different models count as negatives.
- **Algorithm:** round-based on-policy RL. Each round: embed the pool with the current model, sample PL rankings for N queries, verify new pairs (array job, cached), then run several PPO-clip epochs over these rollouts with GRPO-normalised advantages and a KL penalty to the reference ranking policy (instead of the embedding-MSE anchor). The bank-of-means trick keeps image forward passes to one batch.
- **Fair controls (same verifier budget, same bank and batches):**
  - C1, the softmax control on SfM labels (E5).
  - C2, static pseudo-labels: verify the greedy top-k of the initial model once, then train the softmax control on SfM labels plus the verified pairs.
  - C3, iterative re-mining: like C2, but re-verify the current model's greedy top-k every round. This is the strongest control.

  RL must beat C3 to claim that sampling and the reward add something beyond adaptive pseudo-labelling.
- **Go/no-go diagnostic first (~1 GPU-h):** take Stage 1's top-20 for 2,000 SfM-120k train queries and verify the pairs that the cluster labels call negatives. If almost none are true matches, label noise and exploration offer little, and Track A shrinks to its graded-reward part.
- **Data:** start in SfM-120k (no new data). Scale-up option: GLDv2-clean or unlabelled GLDv2 images as the pool, where pre-labelling is impossible and verifier-in-the-loop is the only supervision. This needs a download and storage check first.

### Track B (extension, same infrastructure): optimise the two-stage pipeline
- Landmark retrieval systems re-rank a global shortlist with local verification (DELG + GV, RRT, CVNet). Reward: AP of the final list **after** verifier re-ranking of the top-K shortlist (and/or after α-QE). Only positives that reach the shortlist and that the verifier can confirm count; the order inside the shortlist is irrelevant. This is a non-decomposable reward of the whole list and the pipeline.
- Evaluate ROxford/RParis with the same re-ranking applied to all arms (standard protocol in the re-ranking literature). Control: the softmax control and a Recall@k surrogate (Patel et al. 2022) on the same bank.

### Track C (equal-footing upgrades, applied to every arm)
- Larger memory bank (16k–64k; XBM), cosine lr decay or a shorter schedule (epoch 2 often falls, E2), EMA of weights, multi-scale testing and learned whitening, DINOv2 backbone (Phase 4). These raise absolute numbers for both RL and controls and are reported as such, not as RL gains.

### Order and gates
1. **A0 label-noise / exploration diagnostic** (≈ half a day). Gate: ≥ 5% of Stage 1 top-20 "negatives" verify as matches (≥ 100 inliers), or verified grades disagree with the cluster labels on ≥ 10% of the top-20.
2. **A1 infrastructure** (≈ 2 days): pair cache, round-based rollout, verification array job, PPO-clip + GRPO + KL in `training/`; CPU tests (ratio = 1 on the first epoch reproduces the E1 gradient).
3. **A2 pilot, seed 42** (≈ 1 day of GPU): RL vs C2 and C3 at an equal verifier budget (e.g. ≤ 300k verified pairs). Gate: RL > C3 on SfM val (graded and binary) by more than the E5 seed spread (≈ 0.3).
4. **A3 5 seeds + final evaluation** of RL, C1, C3 (seeds 3–4 Stage 1 runs exist by then).
5. **B** on top of A if A passes; Track C in parallel throughout.

Honest risk: C3 (iterative re-mining with the same verifier) may capture most of what RL gains. If so, the result is still a clear finding (exploration vs greedy re-mining under a fixed verifier budget), and Track B remains as the second test.

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
- Liu et al., Retrieval-GRPO: multi-objective RL for dense retrieval in Taobao search (online top-k candidates, reward-model feedback), 2025/2026: https://arxiv.org/abs/2511.13885
- Schulman et al., Proximal Policy Optimization Algorithms, 2017: https://arxiv.org/abs/1707.06347
- Shao et al., DeepSeekMath (GRPO), 2024: https://arxiv.org/abs/2402.03300
- Tan et al., Instance-level Image Retrieval using Reranking Transformers, ICCV 2021: https://arxiv.org/abs/2103.12236
- Cao, Araujo & Sim, Unifying Deep Local and Global Features for Image Search (DELG), ECCV 2020: https://arxiv.org/abs/2001.05027
- Iscen et al., Mining on Manifolds, CVPR 2018: https://openaccess.thecvf.com/content_cvpr_2018/papers/Iscen_Mining_on_Manifolds_CVPR_2018_paper.pdf
- Deng et al., InsCLR: Improving Instance Retrieval with Self-Supervision, AAAI 2022: https://arxiv.org/abs/2112.01390
