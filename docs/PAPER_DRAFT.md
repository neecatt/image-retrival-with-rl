# Paper draft: reward tuning for landmark retrieval under matched controls

Working draft, started 2026-10-13. Every number here comes from [PROGRESS.md](PROGRESS.md); section ids (C4, E5, G10, …) point to the run records. Nothing in this file is a result on its own.

## Working title
When does reward tuning help image retrieval? A controlled study of policy-gradient fine-tuning for landmark retrieval

## Claim (as supported today)
1. Reward tuning of a global descriptor with a ranking policy (Plackett–Luce over the gallery, AP@100 reward, leave-one-out baselines) reliably improves a metric-learning model, but **only matches differentiable AP surrogates trained on the same memory bank, batches and budget** (G10: ties Smooth-AP, slightly below FastAP on the bank).
2. Earlier "RL gains" in this project, and plausibly in related work, came from weak controls: no memory bank (G8 vs G10), a test-set-selected checkpoint (F3), or saturated training data (A0, B0).
3. Training-pool saturation decides whether *any* fine-tuning objective can help: on SfM-120k the Stage 1 model retrieves 97% correct top-20 items on its own training images, so rewards are flat (A0, B0); on GLDv2-clean recall@20 is 47% (G5).
4. Practical by-product: GLDv2-clean Stage 1 followed by an AP objective over a 4k memory bank gives +4.7 / +3.9 ROxford and +2.7 / +3.0 RParis (Medium / Hard) over Stage 1 in every seed (G10). The memory-bank effect itself is known (XBM, Wang et al. 2020).

Pending before the claim is final: RL on top of the best surrogate model (pilot running), ablations of the RL policy (τ, k, budget), R1M distractors.

## Outline
1. **Introduction.** Task-reward tuning (Susano Pinto et al. 2023) and recent RL-for-retrieval work (RELER, Retrieval-GRPO) report gains. Retrieval has strong differentiable surrogates of its metric, so the right question is whether RL beats them under matched conditions.
2. **Related work.** Task-reward tuning; policy-gradient learning to rank (PG-Rank, PL-Rank); AP surrogates (FastAP, Smooth-AP, ROADMAP, Recall@k); cross-batch memory (XBM); landmark retrieval training sets (SfM-120k, GLDv2-clean) and protocols (Revisited Oxford/Paris).
3. **Protocol.** Official revisitop scoring; selection on held-out validation only; matched initialisation, budget, batches and memory bank; at least 5 seeds; paired differences; pre-registered success criterion (RL_RESEARCH_PLAN.md §5).
4. **Methods compared.** Gaussian embedding policy (Pinto-style; multi-sample RLOO); graded verification reward; Plackett–Luce ranking policy; controls: FastAP continuation, softmax/SupCon, Smooth-AP and FastAP on the bank.
5. **Results.**
   - 5.1 SfM-120k: every RL variant ties or loses to its matched control (C4, D2–D3, D7–D8, E3–E5, R1).
   - 5.2 Saturation diagnostics (A0, A0b, B0, G5).
   - 5.3 GLDv2-clean: the RL gain over weak controls (G8) disappears against AP surrogates on the same bank (G10).
   - 5.4 Ablations (pending).
6. **Why RL ties.** With a reward computed from labels known in advance, a low-variance differentiable surrogate of the same objective receives the same information. RL's structural advantages (black-box rewards, policy-dependent feedback) need settings where labels are unavailable or the reward is not decomposable. The verifier diagnostics (A0, A0b, G5) quantify how much such feedback adds.
7. **Limitations.** One backbone (ResNet50-GeM); single-scale evaluation without R1M; short fine-tuning budget; GLDv2 may contain Oxford/Paris landmarks; hyperparameters tuned on one seed's validation.

## Main tables (to regenerate from run records before submission)
- T1: SfM-120k, 3–5 seeds: Stage 1, FastAP continuation, Gaussian RL, PL RL, softmax control (C4, E3, E5).
- T2: GLDv2-clean, 5 seeds: Stage 1, FastAP continuation, softmax, Smooth-AP, FastAP-bank, PL RL (G10).
- T3: saturation and verifier diagnostics (A0, A0b, B0, G5).

## Before any submission
- Fix the historical claims in README.md (preliminary, test-selected results).
- Recheck novelty against RELER and recent RL-for-retrieval papers; check venue deadlines on official sites.
- Push to a Git remote (none configured yet); the user decides where.
