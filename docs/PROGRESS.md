# Progress log

What was found, what was changed, and the evidence. Newest entries at the top. Milestone status lives in [ROADMAP.md](ROADMAP.md); this file keeps the detail.

### Q13. Re-query reward over 5 descriptor seeds (2026-10-24)
Jobs 4673832-35 (seeds 1-4, ~2.6 h each, ~10.3 GPU-h) + Q12 (seed 42); same configs, policy seed 0, all three schedulers retrained with the re-query reward; exact full re-query on the benchmarks. Per-query differences averaged over the 5 seeds, bootstrap over queries; seeds+ = seeds with a positive mean (`results/qe/policy_vqe/newresnet_seed*_rq.json`, `scripts/analysis/qe_policy_vqe_summary.py`).

Validation (re-query AP, mean of plain and distractor pools), seeds 42 / 1 / 2 / 3 / 4: classifier 56.01 / 56.45 / 56.40 / 56.23 / 56.32; value 60.20 / 60.52 / 60.51 / 60.41 / 60.52; **RL 60.58 / 61.01 / 61.15 / 60.77 / 60.90** (RL best in 5/5, +0.36 to +0.64 over value).

| mean of 5 seeds, B20 + re-query, M / H | ROxf | ROxf+1M | RPar | RPar+1M |
|---|---|---|---|---|
| no verification | 52.77 / 28.11 | 38.21 / 14.29 | 77.33 / 57.73 | 43.91 / 21.93 |
| classic verified QE (20 most similar) | 63.42 / 38.34 | 51.00 / 24.74 | 83.53 / 66.86 | 53.47 / 28.53 |
| classifier | 66.26 / 42.06 | 53.85 / 27.92 | **85.83 / 69.83** | 58.60 / 33.91 |
| value | 67.62 / 41.73 | 56.49 / 28.92 | 84.57 / 68.18 | **60.30 / 34.31** |
| **RL** | **68.14 / 42.85** | **58.31 / 30.14** | 84.64 / 68.23 | 59.24 / 33.30 |

- RL - classic verified QE: ROxf **+4.72* / +4.52***, ROxf+1M **+7.32* / +5.39***, RPar **+1.12* / +1.37***, RPar+1M **+5.77* / +4.78*** (5/5 seeds except RPar H 4/5).
- RL - value: ROxf+1M M **+1.83* [+0.20, +4.09]** (4/5), ROxf+1M H +1.22 [-0.03, +2.84], ROxf H +1.13 [-0.10, +2.41], ROxf M +0.52 (n.s.); RPar ties; **RPar+1M M -1.06* (0/5), H -1.00* (1/5)**.
- RL - classifier: ROxf M **+1.88***, ROxf+1M M **+4.47*** (5/5), ROxf+1M H **+2.22***; RPar **-1.19* / -1.60***; RPar+1M ties.
- Matches found per query: classic 10.2, classifier 13.1, value 11.3, RL 13.2.
- **Summary:** in the full pipeline (learned scheduling + expansion + verified-QE re-query), all learned schedulers beat classic verified QE by large margins; RL is the best scheduler on validation in every seed and on ROxford+1M (+1.8 over value, +4.5 over the classifier), but loses ~1 mAP to value on RParis+1M and to the classifier on RParis. RL's advantage is dataset-dependent, not uniform.

### Q12. Schedulers retrained with the re-query reward (2026-10-24)
Job 4671621 (2.6 h; `training/qe_requery.py`, `scripts/qe_requery_job.sh`). Reward / selection = verified-QE re-query AP over per-query subsets (top-1000 + positives + candidates; X = 1,037 / 1,098 for the plain / distractor pools); unit tests: subset AP equals the exact full-gallery re-query (8e-8) and the vectorised one-step gains equal brute force (0). New ResNet seed 42, B = 20, policy seed 0, configs as selected in Q5; classifier, value (re-query one-step targets) and RL all retrained; benchmarks with the exact full re-query (`results/qe/policy_vqe/newresnet_seed42_rq.json`).

| seed 42, B20 + re-query, M / H | val (re-query) | ROxf | ROxf+1M | RPar | RPar+1M |
|---|---|---|---|---|---|
| classic verified QE | | 65.31 / 38.48 | 51.46 / 23.10 | 83.53 / 66.75 | 53.44 / 28.54 |
| classifier (retrained) | 56.01 | 69.61 / 44.17 | 56.86 / 30.29 | **87.43 / 71.50** | **61.27 / 37.21** |
| value (retrained) | 60.20 | **69.70** / 42.56 | 57.61 / 29.80 | 84.96 / 68.53 | 59.76 / 33.71 |
| RL (retrained) | **60.58** | 69.18 / 43.15 | **58.63** / 29.97 | 84.45 / 68.02 | 58.73 / 32.60 |
| RL (reused, promote-only reward, Q11) | | 69.03 / 42.32 | 59.01 / 30.42 | 84.14 / 67.73 | 56.78 / 31.56 |

- RL - classic verified QE: +3.9 to +7.2 on ROxford(+1M), +5.3 / +4.1 on RParis+1M (all significant).
- **RL - value: no significant difference** (ROxf M -0.52, ROxf+1M M +1.02, RPar+1M M -1.03; all CIs include 0). **RL - classifier: significantly worse on RParis** (-3.0 / -3.5, +1M -2.5 / -4.6), ties on ROxford.
- Retraining with the true reward did not change RL's benchmark numbers much (ROxf+1M M 59.01 -> 58.63, RPar+1M M 56.78 -> 58.73) and improved the supervised schedulers (value ROxf M 68.22 -> 69.70; classifier RPar+1M 59.05 -> 61.27).
- Validation and benchmarks disagree again for the classifier (lowest on validation, best on RParis), as at B40 (Q7): the number of matches found (classifier 14.4 per query, RL 13.5, value 10.9) drives the re-query on RParis, and GLDv2 validation does not reward it.
- **Conclusion so far:** learned scheduling of verification with expansion + re-query is a large, replicated gain over classic verified QE at the same budget (+4 to +8 mAP); within that pipeline RL is best on validation but not distinguishable from the supervised schedulers on the benchmarks (one seed, one policy seed).

### Q11. Policy-chosen verifications + verified-QE re-query (2026-10-24)
Job 4664543 (45 min; `scripts/analysis/qe_policy_vqe.py`, summary `scripts/analysis/qe_policy_vqe_summary.py`, `results/qe/policy_vqe/<descriptor>.json`). The saved B = 20 policies of Q5 / Q8 (configs selected on validation; trained with the promote-only reward, not retrained) choose the 20 verifications; all verified matches feed the verified-QE re-query of Q10 (alpha = 0, chosen on GLDv2 validation). Sanity: 'shortlist' (the 20 most similar) reproduces classic verified QE (ResNet ROxf M 63.42; DINOv2 79.88 / 60.14 vs 60.11 in Q10, tie-breaking).

New ResNet, mean over 5 descriptor seeds (per-query differences averaged over seeds, bootstrap; seeds+ = seeds with positive mean):

| B20 + re-query, M / H | ROxf | ROxf+1M | RPar | RPar+1M |
|---|---|---|---|---|
| classic verified QE (20 most similar) | 63.42 / 38.34 | 51.00 / 24.74 | 83.53 / 66.86 | 53.47 / 28.53 |
| classifier-chosen | 67.01 / 42.71 | 55.35 / 29.38 | **85.83 / 69.64** | 58.57 / 33.47 |
| value-chosen | 67.24 / 41.95 | 57.10 / 29.50 | 84.36 / 67.75 | **59.66 / 34.12** |
| **RL-chosen** | **68.41 / 42.89** | **58.67 / 30.41** | 84.30 / 67.70 | 59.11 / 33.72 |

- RL - classic verified QE: **ROxf M +5.00* / H +4.55*, ROxf+1M M +7.67* / H +5.66*, RPar+1M M +5.64* / H +5.19*** (all 5/5 seeds); RPar +0.8 (n.s.). Matches found per query: shortlist 10.2, classifier 13.0, value 12.7, RL 13.1.
- RL - value: **ROxf M +1.17* (4/5), ROxf+1M M +1.57* (5/5)**, ROxf H +0.94, ROxf+1M H +0.91 (n.s.); RParis ties (-0.06 to -0.55, n.s.).
- DINOv2 (2 policy seeds): RL-chosen ROxf 81.83 / 62.86, ROxf+1M 74.76 / 53.66, RPar+1M 86.62 / 72.82, vs classic verified QE 79.88 / 60.14, 69.63 / 46.13, 86.83 / 73.13: **ROxf+1M +5.13* / +7.53*, ROxf H +2.73***; RParis ties; RL - value +1.2 to +2.9 on ROxford (n.s., one model).
- **With the re-query, scheduling the 20 checks well pays off strongly** (+5 to +8 mAP over classic verified QE on ROxford+1M and RParis+1M), and RL is the best scheduler on ROxford in both descriptors. DINOv2 + RL scheduling + re-query reaches ROxf+1M 74.76 / 53.66, on par with rEGT (74.9 / 54.1), which verifies the whole database graph offline; still below SuperGlobal / CVNet re-ranking (84.4 / 71.1, 81.9 / 67.4).
- Caveats: the policies were not trained with the re-query in the reward; the re-query step was added after seeing benchmark results of the promote-only pipeline (Q10), with its setting chosen on validation; 70 queries per benchmark. Next: retrain RL with the re-query in the reward (one seed).

### Q10. Classic verified QE beats the QE-environment pipeline (2026-10-24)
Job 4664171 (29 min; `scripts/analysis/qe_baselines.py`, `results/qe/baselines/<descriptor>.json`). Baselines chosen on GLDv2 validation (mean of plain and distractor pools): alpha-QE n = 2-3, alpha = 0; verified QE (verify the B = 20 most similar, matches first by inliers, re-query the whole gallery with q + sum of matches, alpha = 0).

| B = 20, Medium / Hard | ROxf | ROxf+1M | RPar | RPar+1M |
|---|---|---|---|---|
| ResNet (5 seeds): none | 52.77 / 28.11 | 38.21 / 14.29 | 77.33 / 57.73 | 43.91 / 21.93 |
| ResNet: alpha-QE | 55.68 / 31.61 | 42.15 / 18.05 | 79.75 / 61.15 | 48.36 / 24.86 |
| ResNet: **verified QE** | **63.42 / 38.34** | 51.00 / 24.74 | **83.53 / 66.86** | **53.47 / 28.53** |
| ResNet: value (QE env) | 61.60 / 37.58 | 50.66 / 24.97 | 77.93 / 58.16 | 46.83 / 23.50 |
| ResNet: RL (QE env) | 63.34 / 38.05 | **52.25 / 25.26** | 77.94 / 58.16 | 46.82 / 23.36 |
| ResNet: verify all (~250) with expansion | 68.56 / 43.30 | 57.26 / 29.56 | 80.34 / 60.76 | 52.89 / 28.07 |
| DINOv2: none | 68.14 / 48.67 | 58.44 / 36.52 | 91.11 / 80.67 | 83.02 / 66.17 |
| DINOv2: alpha-QE | 71.60 / 51.85 | 61.94 / 39.43 | 92.02 / 82.24 | 84.98 / 69.57 |
| DINOv2: **verified QE** | **79.88 / 60.11** | **69.63 / 46.13** | **92.84 / 84.16** | **86.83 / 73.13** |
| DINOv2: RL (QE env) | 75.09 / 54.31 | 67.68 / 43.81 | 91.23 / 80.69 | 83.22 / 66.30 |
| DINOv2: verify all with expansion | 78.00 / 56.42 | 68.79 / 43.87 | 92.46 / 82.21 | 84.53 / 68.13 |

- **Classic spatially verified QE (Chum et al. 2007) with the same 20 checks matches or beats the RL pipeline almost everywhere**: ties on ResNet ROxf, much better on RParis (+5.6 RPar M, +6.7 RPar+1M M), and on DINOv2 better on every protocol (ROxf M +4.8, ROxf+1M M +2.0, RPar+1M M +3.6); on DINOv2 it even beats verifying all ~250 candidates with our re-ranking rule. Only ResNet ROxf+1M M favours RL (+1.25).
- Cause: the QE environment's rule only moves verified matches to the top and keeps everything else in the global order; verified QE re-queries the whole gallery with the matches, which also lifts relevant images that were never verified.
- **Consequence:** RL's controlled advantage (Q5, Q8) is real but holds inside a weaker re-ranking rule; a paper cannot claim a better system without beating verified QE. Open question: does choosing which images to verify help on top of verified QE (better matches for the re-query)? Cheap test: re-query with the matches found by the saved RL / value / classifier policies.
- Timings (one RTX PRO 6000): verifier ~200 pairs/s (~5 ms per check, ~0.1 s per query at B = 20); policy step ~1 ms per 70 queries (negligible); 1M-image search < 0.1 ms per query.

### Q9. Literature check and EGT-style baseline (2026-10-24)
- Literature (`docs/LITERATURE_QE.md`): no paper found that learns a per-query verification schedule under a fixed budget for landmark retrieval, or couples one with verified neighbour expansion (targeted search, not exhaustive). Closest: rEGT (Chang et al., CVPR 2019), which verifies all n*k database graph edges offline (incl. R1M) plus k = 100 per query and traverses with a hand-set explore/exploit threshold; our setting has no offline database verification and a fixed online budget. Current systems (author-reported): RN101-SuperGlobal + rerank ROxf 90.9 / 80.2, ROxf+1M 84.4 / 71.1; RN101-CVNet + rerank 87.2 / 75.9, 81.9 / 67.4; rEGT (2019) 83.5 / 65.8, 74.9 / 54.1.
- **EGT-style rule under the same online budget** (`scripts/analysis/qe_egt_baseline.py`, job 4664079, 2 min): exploit the unlocked neighbour of the best verified parent if its inliers >= t, else verify the most similar available candidate; t chosen on GLDv2 validation (t = 50 for all six descriptors; t = inf reproduces similarity order exactly). Mean over 5 new-ResNet seeds, B20: ROxf M/H 58.16 / 33.83, ROxf+1M 46.05 / 20.55, RPar+1M 45.42 / 22.61 (RL: 63.34 / 38.05, 52.25 / 25.26, 46.82 / 23.36); B40: ROxf+1M M 49.72 (RL 53.57). DINOv2 B20: ROxf M 72.03, ROxf+1M M 62.31 (RL 75.09, 67.68). The hand-set traversal rule is well below the learned schedulers on ROxford and close to them on RParis; the value policy remains the strongest baseline.
- Remaining baselines (alpha-QE, verified QE at the same budget, verify-all with expansion, budget curves, timings): job 4664006 failed on an out-of-bounds label index for distractor galleries (fixed), resubmitted as 4664171.

### Q8. Track V-QE on the DINOv2 baseline (2026-10-23)
DINOv2 Stage 1 + WiSE-FT (`runs/wise_ft/gd_full_c04_e6.pth`). Jobs 4658686 (expansion precompute + policy seed 0, 3.5 h) and 4658687 (policy seed 1, 2.5 h), ~5.9 GPU-h. B = 20; configs transferred from the new-ResNet seed-42 validation selection (classifier/value DAgger lr 1e-3 vs 3e-4 chosen on DINOv2 validation; RL from scratch lr 3e-4); not re-tuned on DINOv2. Expansion: ROxford 23 candidates per query (55% relevant, Medium), ROxford+1M 29 (43%), RParis(+1M) ~45 (88%); GLDv2 val 5.6 (3.4%), with distractors 15.8 (4.1%). **Sanity check passed:** no verification reproduces ROxf 68.14 / 48.67, ROxf+1M 58.44 / 36.52, RPar 91.11 / 80.67, RPar+1M 83.02 / 66.17 exactly. Means over 2 policy seeds (one descriptor; `seeds+` below counts policy seeds), `results/qe/dinov2_b20.json`.

| B20, DINOv2 | val plain / dist | ROxf M / H | RPar M / H | ROxf+1M M / H | RPar+1M M / H |
|---|---|---|---|---|---|
| no verification | 78.10 / 58.40 | 68.14 / 48.67 | 91.11 / 80.67 | 58.44 / 36.52 | 83.02 / 66.17 |
| top-20 by similarity (expands) | 78.25 / 59.40 | 71.41 / 52.53 | 91.25 / 80.81 | 61.93 / 39.55 | 83.17 / 66.31 |
| classifier | 78.10 / 58.93 | 70.83 / 50.69 | 91.29 / 80.94 | 61.33 / 38.21 | 83.24 / 66.52 |
| value | 79.09 / 61.24 | 73.83 / 54.31 | 91.29 / 80.90 | 64.95 / 42.78 | 83.27 / 66.51 |
| **RL** | **79.28 / 61.54** | **75.09 / 54.31** | 91.23 / 80.69 | **67.68 / 43.81** | 83.22 / 66.30 |

- RL - value: val **+0.19* / +0.31***, **ROxf M +1.27* [+0.32, +2.50]**, **ROxf+1M M +2.72* [+0.85, +4.99]** (both policy seeds positive), ROxf+1M H +1.03 (n.s.), ROxf H tie; RParis small losses (RPar H **-0.21***, others n.s.; RParis is near saturation at 91 / 83).
- RL - classifier: ROxf M/H **+4.26* / +3.62***, ROxf+1M M/H **+6.35* / +5.60***; RParis H -0.2*. With DINOv2 the classifier is below plain top-20 verification on ROxford.
- **The B20 result transfers to a strong descriptor, with a larger margin on ROxford+1M (+2.7 vs +1.6 on the new ResNet)** and much better absolute numbers: ROxford+1M Medium 58.44 (no verification) -> 61.93 (standard top-20) -> 64.95 (value) -> **67.68 (RL)**; ROxford Medium 68.14 -> **75.09**.
- Limits: one DINOv2 model, 2 policy seeds, hyperparameters transferred from ResNet; RParis shows no gain (saturated and RL under-expands there, Q4).

### Q7. Track V-QE replication at B = 40: mixed (2026-10-22)
Same lean protocol as Q5 at a second budget: jobs 4649367-70 (new ResNet seeds 1-4, expansion data reused, ~5.0 h each, ~20 GPU-h); seed 42 from the Q4 grid. Per-seed means in seed order 42, 1, 2, 3, 4 (`results/qe/v2_lean_5seeds_b40.json`).

Mean over 5 seeds, B = 40:

| method | val plain / dist | ROxf M / H | RPar M / H | ROxf+1M M / H | RPar+1M M / H |
|---|---|---|---|---|---|
| top-40 by similarity (expands) | 64.94 / 42.79 | 59.42 / 36.28 | 77.75 / 58.08 | 45.61 / 22.22 | 44.91 / 22.63 |
| classifier | 64.46 / 45.20 | **65.62 / 41.29** | **78.73 / 59.11** | 53.66 / **27.80** | 47.93 / **24.41** |
| value | 66.84 / 46.39 | 64.60 / 39.14 | 78.36 / 58.58 | **53.69** / 26.54 | **48.22** / 24.34 |
| RL | **67.01 / 46.54** | 64.74 / 40.14 | 78.25 / 58.44 | 53.57 / 26.94 | 47.59 / 23.86 |

- RL - value: val **+0.16* / +0.14*** (5/5); **ROxf H +1.01***, **ROxf+1M H +0.40***; ROxford Medium ties; **RParis+1M M -0.63*, H -0.48*** (0/5 seeds positive). 4 significant wins, 2 significant losses in 10.
- RL - classifier: val **+2.55* / +1.33***, but **6 significant benchmark losses** (ROxf M -0.88, ROxf+1M H -0.86, RParis M/H -0.48/-0.68, RParis+1M M/H -0.34/-0.55; 0/5 seeds positive on each).
- At B = 40 the classifier, which is the weakest method on GLDv2 validation, is the best on the benchmarks: validation and benchmarks disagree at large budgets, and the validation-only rule does not pick it. With 40 of ~250 candidates checked, simply verifying likely matches (and expanding through them) is close to enough, so planning matters less.
- **Conclusion across budgets:** RL's controlled advantage over the strongest supervised policy is budget-dependent: clear and replicated at B = 20 on ROxford Medium (Q5), absent at B = 40 on ROxford Medium and negative on RParis+1M. The claim to make is "RL helps when the verification budget is tight relative to the expansion opportunities", not a general win.

### Q6. Oversampling training queries where expansion pays: negative (2026-10-21)
Target: RParis, where RL under-expands (Q4 diagnostic). GLDv2 val landmarks are capped at 30 images, so no training landmark resembles RParis (~250 positives); instead, queries whose expansion candidates include >= 3 relevant images (6.9% of the plain pool, 15.7% with distractors) supply half of every training batch (`--expand_weight 0.5 --expand_min 3`), for all methods alike; validation and benchmarks unchanged. Job 4631705 (3.2 GPU-h; new ResNet seed 42, B = 20, policy seed 0, same configs as Q5), compared with the unweighted Q5 runs of the same seed (`results/qe/v2w_s42`, `results/qe/v2w_s42_selection.json`).

| B20, seed 42 | val plain / dist | ROxf M / H | RPar M / H | ROxf+1M M / H | RPar+1M M / H |
|---|---|---|---|---|---|
| classifier, unweighted | 63.76 / 43.55 | 61.24 / 38.39 | 77.86 / 58.14 | 48.39 / 25.41 | 45.73 / 23.71 |
| classifier, weighted | 63.52 / 43.65 | 64.46 / 39.42 | 78.51 / 58.86 | 50.65 / 25.21 | 47.18 / 24.47 |
| value, unweighted | 66.24 / 46.12 | 62.43 / 37.74 | 78.07 / 58.28 | 50.83 / 24.21 | 47.11 / 23.93 |
| value, weighted | 66.22 / 46.07 | 61.81 / 36.05 | 78.16 / 58.44 | 50.69 / 24.45 | 47.64 / 24.22 |
| RL, unweighted | 66.67 / 46.41 | 64.01 / 37.84 | 78.10 / 58.26 | 53.03 / 25.54 | 47.33 / 23.51 |
| RL, weighted | 66.58 / 46.36 | 64.08 / 37.56 | 78.16 / 58.43 | 52.65 / 25.29 | 47.57 / 23.98 |

- Oversampling raises RParis(+1M) a little for every method, the **classifier most** (RPar+1M M +1.45, ROxf M +3.2); RL changes by +0.2 to +0.5 on RParis+1M and loses 0.4 on ROxf+1M M.
- RL - value with weighting: val +0.37* / +0.29*, ROxf M +2.26*, RParis ties (-0.01 / -0.07): **RL's RParis gap to value is not closed**; RL - classifier becomes significantly negative on ROxf H (-1.86*) and RParis without R1M (-0.35*, -0.43*).
- Validation slightly prefers the unweighted variant for every method (RL 66.67 vs 66.58, classifier 53.66 vs 53.59 mean val), so under the validation-only rule the weighted variant would not be chosen; its RParis gains appear only on the test benchmarks and are not claimed.
- Not adopted. The Q5 result (unweighted) stands; RParis stays a stated limitation. One descriptor seed, one policy seed.

### Q5. Track V-QE replicated over 5 descriptor seeds, B = 20 (2026-10-21)
Lean replication (Q4): jobs 4628705, 4628706, 4628708, 4629382, all completed (~3.4 h each, ~13 GPU-h). New ResNet seeds 42, 1, 2, 3, 4; one policy seed each; configs fixed from seed-42 validation (classifier and value: DAgger lr 1e-3 vs 3e-4, chosen per seed on validation; RL from scratch lr 3e-4). Per-query RL - baseline differences averaged over the 5 seeds, bootstrap over queries; per-seed means in seed order 42, 1, 2, 3, 4 (`scripts/analysis/trackv_seeds.py`, `results/qe/v2_lean_5seeds.json`).

Mean over 5 seeds, B = 20:

| method | val plain / dist | ROxf M / H | RPar M / H | ROxf+1M M / H | RPar+1M M / H |
|---|---|---|---|---|---|
| no verification | 62.71 / 38.16 | 52.77 / 28.11 | 77.33 / 57.73 | 38.21 / 14.29 | 43.91 / 21.93 |
| top-20 by similarity (expands) | 64.58 / 41.57 | 57.00 / 33.73 | 77.51 / 57.86 | 43.62 / 20.71 | 44.31 / 22.22 |
| classifier | 64.01 / 43.47 | 61.09 / 38.42 | 77.93 / 58.31 | 48.52 / 25.82 | 45.59 / 23.09 |
| value | 66.50 / 46.01 | 61.60 / 37.58 | 77.93 / 58.16 | 50.66 / 24.97 | 46.83 / 23.50 |
| **RL** | **66.86 / 46.36** | **63.34** / 38.05 | 77.94 / 58.16 | **52.25** / 25.26 | 46.82 / 23.36 |

RL - value: **val +0.36 [+0.28, +0.44] / +0.35 [+0.28, +0.42]** (5/5 seeds, p < 0.001); **ROxf M +1.74 [+0.77, +2.90]** (5/5, per seed +1.59 +0.72 +2.52 +3.16 +0.72, p = 0.023); **ROxf+1M M +1.59 [+0.72, +2.56]** (5/5, p = 0.027); ROxf H +0.48, ROxf+1M H +0.30, RParis within +-0.15 (all n.s.).
RL - classifier: val +2.9 (5/5); **ROxf M +2.24*, ROxf+1M M +3.73*, RPar+1M M +1.24*** (5/5 on the +1M sets); Hard and RParis without R1M n.s.

- **0 significant losses and 9 significant wins in the 20 RL - baseline comparisons.**
- **First replicated, controlled RL win over the strongest supervised policy:** with verified query expansion, a check's value includes the relevant images it unlocks, which the one-step value policy cannot see; RL beats it on held-out GLDv2 validation in every seed and on ROxford Medium with and without R1M in every seed. In Track V without expansion the same comparison was a tie (L6, L7).
- **Limits:** gains are on ROxford Medium only; Hard and RParis are ties (RParis: RL under-expands, see Q4 diagnostic). One budget (B = 20) was replicated; B10 / B40 showed mixed results on seed 42 (Q4). Benchmarks have 70 queries; the seed-level t-tests use n = 5.

### Q4. Track V-QE results, new ResNet seed 42 (2026-10-21)
63 jobs (all completed, 65 GPU-h, longest 4.5 h), merged per config (`results/qe/v2/merged`); each method's config chosen per budget on mean GLDv2 validation (plain + distractors); paired bootstrap over queries (`results/qe/v2/selection.json`). Selected: RL = from scratch, lr 3e-4 at every budget; value = DAgger lr 1e-3; classifier varies.

| B | method | val plain / dist | ROxf M / H | RPar M / H | ROxf+1M M / H | RPar+1M M / H |
|---|---|---|---|---|---|---|
| 20 | top (expands) | 64.39 / 41.64 | 57.43 / 33.28 | 77.55 / 57.96 | 43.21 / 19.04 | 44.52 / 22.67 |
| 20 | classifier | 63.67 / 43.61 | 61.73 / 38.05 | 78.05 / 58.37 | 49.42 / 25.88 | 46.14 / 23.81 |
| 20 | value | 66.27 / 46.02 | 62.40 / 38.09 | 78.04 / 58.31 | 50.71 / 25.19 | 47.04 / 23.90 |
| 20 | **RL** | **66.66 / 46.41** | **63.93** / 37.83 | 78.08 / 58.30 | **52.75** / 25.50 | **47.13** / 23.61 |

RL - value (* = CI excludes 0):
- B10: val **+0.49* / +0.56***; benchmarks n.s. except **ROxf H -2.31***.
- B20: val **+0.39* / +0.39***, **ROxf M +1.53* [+0.40, +2.86], ROxf+1M M +2.04* [+0.81, +3.54]**; Hard and RParis n.s.
- B40: val **+0.17* / +0.20***; ROxford n.s.; **RParis+1M M -0.48*, H -0.67*, RPar H -0.26***.

RL - classifier: val +1.4 to +4.0 (all *); B20 ROxf M **+2.20***, ROxf+1M M **+3.32***, RPar+1M M **+1.00***; B10 ROxf M **+1.70***, RPar+1M M **+1.34***; B40 RParis (+1M) **-0.56 to -1.03***.

- **First setting where RL beats the strongest supervised policy (value) on held-out validation at every budget** (+0.2 to +0.6, vs <= +0.06 in Track V, L7), as expected when a check's value includes the candidates it unlocks.
- **B20 benchmark win:** ROxford+1M Medium 52.75 for RL vs 50.71 value, 49.42 classifier and 43.21 for standard similarity-ordered verification; Track V's best at B20 was ~48-49.
- **Not uniform:** RL is significantly worse on ROxford Hard at B10 and on RParis at B40. RParis queries have ~250 positives and expansion candidates that are 65-81% relevant, against ~20 positives and ~6% on GLDv2, so the training queries rarely reward heavy expansion; the diagnostic below checks how much budget each policy spends on expansion.
- One descriptor seed only; the benchmark comparisons rest on 70 queries.
- **Diagnostic (selected policies rerun greedily, 3 policy seeds):** RParis+1M at B40: expansion checks classifier 14.0 / value 9.5 / RL 7.7 of 40 (relevant expansions verified 11.7 / 7.8 / 6.2); at B20 5.1 / 5.8 / 5.1. ROxford+1M at B20: 2.7 / 4.7 / 3.9 (relevant 1.5 / 2.1 / 2.2). RL under-expands where expansion is most productive (RParis), consistent with GLDv2 training queries rarely rewarding expansion; on ROxford it expands about as much as value but more precisely. GLDv2 val + distractors (first 300 val queries): relevant images verified through expansion per query: B20 0.4 / 0.6 / 0.7, B40 0.8 / 0.8 / 0.8 (classifier / value / RL), against 6-12 on RParis+1M: the training queries barely reward expansion, so RL's validation lead there comes mostly from better choices inside the shortlist.
- **Replication (2026-10-21):** a first 148-job grid (seeds 1-4 x 4 configs x 3 budgets x 3 policy seeds, ~240 GPU-h) was cancelled at the user's request after ~6 GPU-min (compute is limited). Lean replication instead: one job per descriptor seed (4628705-08, `scripts/qe_replicate_job.sh`, ~3.5 GPU-h each, ~14 GPU-h total): expansion precompute, then B = 20 only, one policy seed, configs fixed from seed-42 validation (classifier/value with DAgger at lr 1e-3 and 3e-4, selected per seed on validation; RL from scratch lr 3e-4). Seed 42 enters the pooled analysis with the same configs and policy seed 0 (`results/qe/v2/*_B20_s0`). Seed 3 (4628707) failed after 50 min on one of the 63 truncated official R1M JPEGs (first time one appeared among verified candidates); the verifier now decodes truncated JPEGs with padding like the R1M embeddings, and the job resumes only the missing pools (resubmitted as 4629382).

### Q3. QE pilot: first signal, two fixes, clean rerun (2026-10-21)
- **First signal (job 4625040, B = 10 only, 3 policy seeds; old batching, valid gradients):** RL fine-tuned from the value policy raises the validation selection score from 55.39 (value) to 55.83 (+0.44; gld_val +0.35, with distractors +0.53). In Track V the same procedure added at most +0.06 (L7). Benchmarks, unpaired means: ROxf M 59.74 -> 60.83, RPar+1M M 45.82 -> 46.21, but ROxf+1M M 47.94 -> 47.13 and ROxf+1M H 24.58 -> 20.67. Expansion itself helps every method (value at B10: ROxf+1M M 47.94 vs ~44 in Track V).
- **Out of memory at B = 20** (4625040): differentiating the whole rollout kept B steps of attention activations (2,048 rollouts x 250 candidates). Fixed: the rollout runs without gradients and records states/actions; the policy gradient is accumulated one step and chunk at a time. Matches the one-graph gradient (relative difference 3.6e-6 with chunking, 0.0 without); a first per-parameter check that looked like a 55% mismatch came from attention biases whose gradient is analytically zero (softmax shift invariance).
- **Speed:** RL from scratch took ~4 h per policy seed at B10, mostly on padding (all queries padded to 250 candidates, GLDv2 queries have ~110-120). Batches now come from one of 8 equal-size groups of queries with similar candidate counts (uniform over queries overall) with padding trimmed; evaluation runs on size-sorted chunks. Per-query APs identical to untrimmed evaluation (max difference 0.0); the position feature keeps a fixed scale.
- **Clean rerun under one code version:** the old grid (4625034-41) is superseded; 63 jobs = 7 configs x budgets 10 / 20 / 40 x policy seeds 0 / 1 / 2, one job each (24 h wall limit), `results/qe/v2/<config>_B<B>_s<seed>`, job ids in `results/qe/v2.jobids` (4625990-4626052); merged per config by `scripts/analysis/qe_merge_seeds.py`.

### Q2. Track V-QE environment and pilot (2026-10-20)
- `scripts/v_precompute_qe.py`: each top-100 shortlist + up to 150 expansion candidates (k = 10 neighbours of every verified match, 2 hops, breadth-first), all new pairs verified offline; one R1M extraction pass for a superset of what both hops can need. Smoke run 4624709 (200 GLDv2 queries + ROxford, 10 min): ROxford 28 expansion candidates per query, 46% verify, 51% relevant (Medium); GLDv2 val 9.4 per query, 7% verify.
- `training/qe_env.py`: candidates available only once unlocked by a verified match; reward = official AP of "verified matches first (by inliers), rest in global order", computed from junk-free global ranks. Matches `official_ap` to < 1e-7 on 300 random rankings (single and batched/padded). End-to-end sanity on the smoke data: no verification ROxf M/H 52.14 / 25.11 and verify-all-100 64.96 / 40.04, identical to Track V.
- `scripts/train_qe_policy.py`: none, top (most similar available, expands), shortlist (Track V top-B, never expands), random, classifier, value (exact one-step AP gain, myopic), RL; all learned methods share the features, including the number of hidden candidates a check would unlock. Cosine lr, selection on GLDv2 val (plain + distractor pools).
- Full precompute 4624710 (new ResNet seed 42): GLDv2 val 8.5 expansion candidates per query (7.5% verify, 6.1% relevant; 85k new pairs), GLDv2 val + distractors 21.3 (8.6% verify, 5.5% relevant; 213k pairs, 1.2% of queries at the cap), plain ROxford done; it then failed on ROxford+1M: the one-pass R1M superset subtracted every query's shortlist from hop 1, so hop-2 neighbours of items that sit in another query's shortlist were never extracted (fixed, plus a missing-file check before verification). Remaining pools rerun as 4625033 (22 min; 7,655 R1M images in one pass): ROxford+1M 37.0 expansion candidates per query (42% verify, 39% relevant, Medium), RParis 48.6 (61% / 81%), RParis+1M 62.4 (57% / 65%). Policy grid resubmitted as 4625034-40 (4624949-55 cancelled); 4625034-38 died with NODE_FAIL on node a2041 after 4.5 min (no Python error) and were resubmitted as 4625137-41 excluding that node; 4625039-40 ran on another node. Pilot grid (budgets 10 / 20 / 40, 3 policy seeds): classifier/value lr 1e-3 / 3e-4 x DAgger off/on, 12,000 updates; RL from scratch lr 3e-4 / 1e-4, 6,000 updates; RL from the value policy, +3,000 updates.

### Q1. Headroom for verification with query expansion (2026-10-20, branch `feat/verified-query-expansion`)
Job 4624689 (9 min), new ResNet seed 42, `scripts/analysis/qe_headroom.py`, `results/qe/headroom_newresnet_seed42/qe_headroom.json`. Every verified shortlist match (>= 35 inliers) unlocks its k nearest gallery neighbours. Oracle = verify-all, then every reachable positive, then the rest (assumes expansion candidates verify perfectly, so it is an upper bound). GLDv2: first 3,000 of the 10k val queries. Sanity: verify-all reproduces Track V all_K (ROxf M 64.96, ROxf+1M M 51.20).

| pool | positives below top-100 | reachable (k = 10 / 20) | global | verify-all 100 | oracle QE k = 10 / 20 |
|---|---|---|---|---|---|
| GLDv2 val | 4.1 | 0.45 / 0.72 | 65.43 | 67.32 | 69.09 / 70.11 |
| GLDv2 val + distractors | 8.7 | 0.99 / 1.51 | 41.69 | 48.26 | 52.20 / 54.10 |
| ROxf M | 51.0 | 9.9 / 14.0 | 52.14 | 64.96 | 69.76 / 71.48 |
| ROxf H | 30.1 | 1.8 / 2.9 | 25.11 | 40.04 | 45.00 / 46.72 |
| ROxf+1M M | 55.2 | 10.5 / 14.1 | 36.38 | 51.20 | 56.94 / 58.99 |
| ROxf+1M H | 32.3 | 1.8 / 2.6 | 10.49 | 26.07 | 31.64 / 33.80 |
| RPar M | 164.9 | 24.6 / 37.0 | 77.37 | 78.65 | 80.76 / 81.99 |
| RPar+1M M | 184.7 | 23.5 / 35.3 | 44.12 | 46.92 | 52.23 / 55.26 |
| RPar+1M H | 132.8 | 10.0 / 16.2 | 22.41 | 24.20 | 28.90 / 32.30 |

- **Expansion opens large headroom beyond verifying the whole shortlist:** +5 to +8 mAP on ROxford(+1M) and RParis+1M at k = 10-20, versus +1 to +3 left by verify-all in Track V. Unlike Track V, a check now has value beyond its immediate AP gain (it unlocks candidates), which a one-step policy cannot see.
- On GLDv2 (policy training data) the signal is weaker: ~1 reachable positive per query with distractors, expansion precision ~6% (ROxford ~50%), because GLDv2 val landmarks have only ~20 images. Training may need the distractor shortlists, where the headroom is +4 to +6.

### L7. Pilot V-E: converged baselines and RL fine-tuned from the value policy (2026-10-20)
Jobs 4621459-68 (2.3-4.1 h), new ResNet seed 42, cosine lr for every method. Validation selection score (mean gld_val plain + distractor), 3 policy seeds:

| B | classifier (48k) | value (48k) | RL from scratch (6k) | RL from value (+3k) |
|---|---|---|---|---|
| 5 | 52.52 | 53.17 | 53.15 | 53.19 |
| 10 | 53.95 | 54.21 | 54.16 | 54.21 |
| 20 | 55.00 | 55.00 | 54.97 | 55.04 |

- **Reward fine-tuning of the converged value policy adds at most +0.06 on validation** (per policy seed +0.00 to +0.06; at B10 with rl_lr 3e-5 the selected RL policy is its unchanged init). Benchmarks after selection (`results/trackv/ve_newresnet_selection.json`): RL - value all n.s. except RPar M +0.06 and RPar+1M M +0.08 at B20; RL - classifier keeps the B5 gains (ROxf M +1.87*, H +1.96*, ROxf+1M H +5.65*) but at B20 the converged classifier ties RL on ROxford.
- **RL from scratch does not beat converged supervised policies** (B20: 54.97 vs 55.00 for both baselines).
- Interpretation: in this formulation the greedy one-step value policy is close to optimal, as expected if the AP gain of verification is close to adaptive submodular (the adaptive greedy policy is then near-optimal; Golovin & Krause 2011), leaving multi-step planning little to add. A clear RL advantage needs a task where myopic choices are provably poor (for example verification that unlocks new candidates, such as verified query expansion, or non-uniform verification costs).

### L6. Track V-D over 5 descriptor seeds, stronger baselines (2026-10-19)
Jobs 4618105-44, all completed. Seed 42: baselines rerun with 12,000 updates (RL keeps 6,000); seeds 1-4 (new ResNet `gr_full_c02_e10_seed{1..4}`): R1M embeddings, plain / distractor / R1M shortlists, 4 configs per method (A lr 1e-3 / 3e-4 / 1e-4, B lr 1e-3; baselines 12,000 updates). Per seed and budget each method's config is chosen on validation; RL - baseline per-query differences averaged over the 5 seeds and bootstrapped over queries; `seeds+` = seeds with a positive mean difference (`scripts/analysis/trackv_seeds.py`, `results/trackv/vd_newresnet_5seeds.json`).

Mean over 5 seeds, B = 20:

| method | val plain / dist | ROxf M / H | ROxf+1M M / H | RPar+1M M / H |
|---|---|---|---|---|
| top-20 | 64.58 / 41.57 | 57.00 / 33.74 | 43.62 / 20.71 | 44.32 / 22.24 |
| classifier | 66.19 / 43.98 | 61.09 / 36.46 | 47.59 / 22.83 | 45.14 / 22.31 |
| value | 66.19 / 43.98 | 61.02 / 36.77 | 47.75 / 23.22 | 45.30 / 22.37 |
| RL | 66.21 / 44.03 | 61.09 / 36.52 | 48.06 / 23.36 | 45.36 / 22.41 |
| verify all 100 | 64.09 / 43.70 | 64.14 / 40.52 | 50.18 / 26.24 | 46.66 / 23.90 |

RL - classifier (pooled, CI95, seeds+):
- **B20: ROxf+1M M +0.47 [+0.10, +0.88] 5/5, H +0.53 [+0.08, +1.01] 5/5, RPar+1M M +0.22 [+0.09, +0.35] 5/5**; dist-val +0.05 5/5; ROxf without R1M ~0.
- **B10: val +0.22 / dist-val +0.28 (5/5), RPar+1M M +0.36 (5/5)**; ROxford n.s. (M -0.59, H -1.29).
- **B5: val +0.62 / +0.63 (5/5), ROxf H +1.83 [+0.38, +3.35] (5/5), RPar+1M M +0.31 (5/5).**

RL - value (myopic one-step AP-gain policy):
- B20: dist-val +0.05 [+0.01, +0.09] 5/5; all benchmarks n.s. (ROxf+1M M +0.31 [-0.01, +0.67] 4/5).
- B10: dist-val +0.05 (4/5); **ROxf H -0.90 [-1.82, -0.13]** (seed-level p = 0.36); others n.s.
- B5: val -0.02, dist-val -0.05 (0/5 seeds positive); RPar+1M M +0.06 and H +0.03 (5/5); ROxford n.s.

- **RL reliably beats the supervised match classifier** (the standard learned re-ranking scheduler), replicated in 5/5 descriptor seeds on validation and on the R1M benchmarks at B20.
- **RL does not beat the myopic value policy:** with 12,000 baseline updates, the value policy ties RL on validation and on the benchmarks. The L5 margins over the value policy (seed 42, +1.70 on ROxf+1M M) shrink to +0.88 on that seed with stronger baselines and to +0.31 (n.s.) pooled. Part of the earlier RL - value margin came from an undertrained value baseline.
- The baselines were still improving at 12,000 updates (best checkpoint at 80-95% of training, +0.05-0.1 in the second half), so the next round uses a cosine lr decay for every method and 48,000 baseline updates.

**Pilot V-E (seed 42, jobs 4621459-68):** cosine lr decay for every method; classifier and value 48,000 updates (lr 1e-3 / 3e-4 x A / B); RL fine-tuned from the trained value policy of the same seed (3,000 updates, rl_lr 1e-4 / 3e-5, A / B; the init is scored first, so the selected RL policy cannot fall below it on validation); RL from scratch, 6,000 updates (lr 3e-4 / 1e-4). If reward tuning improves on the converged value policy, scale to the other seeds and DINOv2.

### L5. Track V-D results: distractor-aware training, matched 6-config grid (2026-10-19)
Data (L4 jobs): distractors packed in 15 min (909,836 images, 0 bad); shortlists over val pool + distractors: new ResNet 97 min, DINOv2 167 min; the fresh val-pool similarities reproduce the plain run (max diff 2.4e-4) and the compact AP tables match `query_tables` on 20/20 queries; 82% (ResNet) / 78% (DINOv2) of the distractor-shortlist items are distractors (ROxf + R1M: 48%). Policies 4615051-62 (2.5-5 h each). Each method's config (lr x variant A/B, 6 per method) chosen per budget on mean gld_val (plain + distractor shortlists); `scripts/analysis/trackv_select.py`, outputs `results/trackv/vd_*_selection.json`.

**New ResNet, B = 20** (RL - baseline, paired bootstrap; * = CI excludes 0):

| method | val (plain / dist) | ROxf M / H | RPar M / H | ROxf+1M M / H | RPar+1M M / H |
|---|---|---|---|---|---|
| none | 62.51 / 38.18 | 52.15 / 25.11 | 77.36 / 57.82 | 36.38 / 10.49 | 44.14 / 22.38 |
| top-20 | 64.39 / 41.64 | 57.45 / 33.28 | 77.54 / 57.96 | 43.20 / 19.04 | 44.54 / 22.65 |
| classifier | 65.91 / 43.87 | 62.25 / 36.54 | 77.74 / 58.13 | 48.53 / 22.41 | 45.45 / 22.80 |
| value | 65.89 / 43.89 | 62.12 / 36.72 | 77.72 / 58.02 | 47.62 / 21.85 | 45.63 / 22.85 |
| **RL** | **65.97 / 44.03** | **62.80 / 36.98** | 77.74 / 58.03 | **49.31 / 23.18** | **45.72** / 22.78 |
| RL - classifier | +0.05 / **+0.16*** | **+0.55* / +0.44*** | -0.01 / -0.11 | **+0.79*** / +0.78 | **+0.27*** / -0.02 |
| RL - value | **+0.07* / +0.14*** | **+0.69*** / +0.25 | +0.01 / +0.01 | **+1.70* / +1.33*** | +0.09 / -0.07 |

Other cells (RL - classifier / RL - value, significant ones only):
- New ResNet B5: vs classifier val **+0.78 / +0.64**, ROxf H **+4.50**, ROxf+1M H **+3.26**, RPar+1M M **+0.34**; vs value all n.s. (ROxf+1M M -0.94, H -2.01, n.s.).
- New ResNet B10: vs classifier val **+0.31 / +0.41**, RPar+1M M **+0.39**; vs value dist-val **+0.15**; benchmarks n.s. (ROxf+1M M vs classifier -1.71 [-4.14, +0.23]; it was **-4.76*** before distractor-aware training, L3).
- DINOv2 WiSE-FT B5: vs classifier val **+0.27 / +0.50**, ROxf H **+1.36**, ROxf+1M H **+2.35**; vs value val **+0.08**.
- DINOv2 B10: vs classifier val **+0.15 / +0.22**; vs value dist-val **+0.12**, RPar M **+0.02**; ROxf H vs value -2.34 [-5.59, +0.00] (n.s.).
- DINOv2 B20 (near the verify-all ceiling, e.g. ROxf+1M H: RL 42.41, verify all 42.22): vs classifier val **+0.08 / +0.07**; vs value dist-val **+0.06**, RPar M **+0.04**; ROxford n.s.

Findings:
- **RL is never significantly worse than either learned baseline in any of the 120 comparisons** (2 descriptors x 3 budgets x 2 baselines x 10 pools: 2 validation + 8 benchmark), and significantly better in 33 of them; it is best on validation in 34 of the 36 (config, budget) cells. Benchmark margins are mostly +0.3 to +1.7, larger on ROxford Hard at B5 (up to +4.5 vs the classifier).
- **Distractor-aware training removes the L3 failure:** ResNet B10 ROxf+1M M, RL - classifier -4.76* -> -1.71 (n.s.).
- **Strongest result: new ResNet B20 with R1M:** RL ROxf+1M M/H 49.31 / 23.18 vs top-20 re-ranking 43.20 / 19.04 and vs the best supervised policy +0.79* / +1.33* (classifier / value on M / H).
- The RL warm start (variant B) did not help: RL from scratch reaches the same validation score and is selected in 5 of 6 cells.
- Baselines peak at 80-97% of their 6,000 updates (RL earlier), so a 12,000-update baseline check is running (jobs 4618105-12, classifier/value, the 4 selected configs, both descriptors).
- **Replication over 4 more descriptor seeds** (new ResNet seeds 1-4): R1M embeddings, plain / distractor / R1M shortlists, then 4 configs per method (A lr 1e-3 / 3e-4 / 1e-4, B lr 1e-3; baselines 12,000 updates, RL 6,000), selection per seed on validation (jobs 4618113-44).

### L4. Storage incident and Track V-D setup (2026-10-18, results pending)
**Vault quota:** vault was at 1,512 GB charged (soft 1,049 GB, grace ending within hours; vault charges ~2x apparent size). The 500 raw GLDv2 train tars (499 GB) moved to WORK `/home/atuin/b268dc/b268dc21/datasets/gldv2/train`, each copied, checked against its official md5 and only then removed from vault (no mismatches). Vault now 472.5 GB; scripts read the tars from WORK.

**Track V-D (fixing the B10 + R1M failure of L3, and strengthening every learned method equally):**
- Distractors for training: all 909,836 full-GLDv2 images of the 121,781 landmarks absent from GLDv2-clean (`scripts/prepare_gldv2_distractors.py`, packed at 512 px to WORK `datasets/gldv2d`; job 4615045). Never relevant to a GLDv2-clean val query; R1M stays test-only.
- Shortlists of the same 10,000 GLDv2 val queries over [val pool, distractors] (`scripts/v_precompute_dist.py`; compact AP tables checked against `query_tables`; inliers reused from the plain run): new ResNet 4615046, DINOv2 WiSE-FT 4615047. DINOv2 + R1M benchmark shortlists: 4614749.
- Policies train on plain + distractor shortlists; selection = mean gld_val over both. Evaluation: ROxf/RPar with and without R1M.
- All learned methods get the new scale-invariant features (`--feat v2`) and the same lr grid {1e-3, 3e-4, 1e-4} x 2 variants = 6 configurations each, chosen per method on validation:
  - A: every method from scratch, 6,000 updates (matched).
  - B: classifier and value with DAgger states (their own policy's rollouts), 6,000 updates; RL warm-started from the selected classifier of the same seed + 3,000 reward-tuning updates (the pretrain-then-tune recipe of the original framework). Validation curves are saved so the baselines' plateau can be checked.
- Jobs: new ResNet 4615051-56, DINOv2 4615057-62 (3 policy seeds, budgets 5/10/20).

### L3. Track V with R1M distractors, Track V on DINOv2 WiSE-FT, DINOv2 R1M (2026-10-18)
**R1M baselines (jobs 4606887-88, ~9 h each at ~31 img/s):**

| model | ROxf M / H | + R1M | RPar M / H | + R1M |
|---|---|---|---|---|
| new ResNet Stage 1 seed 42 (L2) | 52.14 / 25.11 | 36.38 / 10.49 | 77.37 / 57.82 | 44.12 / 22.41 |
| DINOv2 ViT-B/14 CLS, off the shelf | 64.51 / 41.81 | 51.16 / 24.85 | 88.29 / 75.24 | 72.63 / 52.61 |
| **DINOv2 Stage 1 + WiSE-FT α 0.7 (baseline by rule)** | **68.14 / 48.67** | **58.44 / 36.52** | **91.11 / 80.67** | **83.02 / 66.17** |

**Track V + R1M, new ResNet descriptor** (precompute 4608533, 19 min: shortlists over database + R1M, 3,787 R1M images extracted from the archives in 14 min; policies 4608534-35 = the L2 policies, same data and seeds, gld_val numbers identical, re-evaluated on the new shortlists). R1M fills 48% of the ROxford and 33% of the RParis top-100; only 2.4% of the R1M shortlist items reach 35 inliers. lr per method chosen on gld_val (B10 and B20: classifier 1e-3, value 3e-4, RL 3e-4):

| B | method | gld_val | ROxf+1M M | ROxf+1M H | RPar+1M M | RPar+1M H |
|---|---|---|---|---|---|---|
| 0 | none | 62.51 | 36.38 | 10.49 | 44.14 | 22.38 |
| 10 | top-B | 63.61 | 40.99 | 15.81 | 44.31 | 22.52 |
| 10 | classifier | 65.07 | **45.20** | 20.43 | 44.85 | 22.56 |
| 10 | value | 65.31 | 44.15 | **20.70** | 45.11 | 22.52 |
| 10 | RL | **65.41** | 40.45 | 15.75 | **45.17** | **22.61** |
| 20 | top-B | 64.39 | 43.20 | 19.04 | 44.54 | 22.65 |
| 20 | classifier | 65.89 | 48.13 | 22.51 | 45.40 | 22.78 |
| 20 | value | 65.88 | 48.31 | 22.93 | **45.78** | 22.94 |
| 20 | **RL** | **65.96** | **49.02** | **23.02** | 45.76 | **22.97** |
| 100 | verify all | 63.84 | 51.20 | 26.07 | 46.93 | 24.18 |

Paired bootstrap, RL - baseline, each at its selected lr:
- B20: vs classifier ROxf M **+0.89 [+0.30, +1.49]**, RPar M **+0.36 [+0.15, +0.56]**, ROxf H +0.52 [-0.23, +1.38]; vs value ROxf M **+0.71 [+0.05, +1.54]**, others tie.
- **B10: vs classifier ROxf M -4.76 [-8.85, -1.31], ROxf H -4.68 [-9.57, -0.74]; vs value ROxf M -3.70 [-7.44, -0.68], H -4.95 [-9.75, -0.91]** (RL barely above top-10). Without R1M the same B10 policies tie (L2).

Diagnosis (`scripts/analysis/trackv_selection_stats.py`, saved policies rerun greedily, 3 seeds): RL and value learned on GLDv2 to verify deep in the shortlist (mean verified rank 27-31 at B10, half beyond rank 20; classifier 9, top-B 4.5). With R1M the similarity scale of the shortlist shifts (ROxf mean sim at rank 100: 0.62 -> 0.72) and the deep items are often distractors; at B10 RL's hit rate falls to 0.56 (classifier 0.61) and it verifies 33% of the shortlist positives (classifier 43%). The policies never saw distractor-filled shortlists in training (GLDv2 val pool, 21k images); the failure is a train/test shift that the gld_val selection cannot see.

**Track V on the DINOv2 WiSE-FT baseline** (precompute 4608536, 59 min; policies 4608537-38; lr chosen per method on gld_val: B20 classifier 1e-3, value 3e-4, RL 3e-4):

| B | method | gld_val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|---|
| 0 | none | 78.10 | 68.15 | 48.67 | 91.09 | 80.67 |
| 20 | top-B | 78.25 | 71.41 | 52.53 | 91.24 | 80.81 |
| 20 | classifier | 78.82 | **73.89** | 53.57 | **91.27** | 80.80 |
| 20 | value | 78.75 | 73.36 | 53.91 | 91.21 | 80.82 |
| 20 | **RL** | **78.89** | **73.89** | **54.23** | **91.27** | **80.86** |
| 100 | verify all | 75.18 | 75.15 | 54.14 | 91.77 | 81.44 |

Bootstrap at selected lr, B20: vs classifier ROxf M +0.00 [-0.58, +0.43], **ROxf H +0.66 [+0.15, +1.26]**, gld_val +0.07 [+0.00, +0.14]; vs value **ROxf M +0.53 [+0.20, +0.95]**, **gld_val +0.14 [+0.07, +0.22]**. B10: RL ties on ROxf M, ROxf H -1.8 vs both (n.s.). At the same lr the B20 margins vs the classifier are +0.22 / +0.59 (ROxf M, both CIs > 0).
- **At B = 20, RL is never significantly worse than a learned baseline on any descriptor or gallery and is significantly better on at least one ROxford protocol in every setting** (ResNet old/new, DINOv2 off the shelf/WiSE-FT, new ResNet + R1M). Margins are +0.5 to +1.0 on ROxford and +0.1 on gld_val.
- **At B = 10 the result does not hold under R1M:** RL loses by ~4-5 on ROxford+1M because its deep-verification strategy does not transfer to distractor-filled shortlists. Next: train the policies on GLDv2 shortlists that also contain a large distractor set (not R1M, which is part of the test protocol), so the training distribution matches the 1M setting.
- Note: the diagnosis was run on the login node and used ~40 CPU-minutes (2.7 min wall, unbounded threads); future analyses go to Slurm or set OMP_NUM_THREADS.

### L2. First R1M result and Track V on the new ResNet descriptor (2026-10-18)
**R1M (job 4606886, 69 min; R1M embedded in 67 min at 248 img/s):** new ResNet baseline seed 42, ROxford M/H **52.14 / 25.11 -> 36.38 / 10.49 with R1M**, RParis M/H **77.37 / 57.82 -> 44.12 / 22.41 with R1M**. 63 of the 1,001,001 official R1M JPEGs are truncated and were decoded with padding (list next to the embedding cache on WORK). DINOv2 R1M jobs (4606887-88) run at ~31 img/s (about 9 h for R1M).

**Track V on the new ResNet descriptor** (`gr_full_c02_e10_seed42`; precompute 4605732, 50 min; attention policies 4606889-90; 10,000 GLDv2 queries; lr per method chosen on gld_val at B = 20: classifier 1e-3, value 3e-4, RL 3e-4):

| B | method | gld_val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|---|
| 0 | none | 62.51 | 52.15 | 25.11 | 77.36 | 57.82 |
| 20 | top-B | 64.39 | 57.45 | 33.28 | 77.54 | 57.96 |
| 20 | classifier | 65.89 | 61.75 | 36.62 | 77.72 | 58.09 |
| 20 | value | 65.88 | 61.85 | 36.44 | 77.78 | 58.15 |
| 20 | **RL** | **65.96** | **62.76** | **37.03** | **77.80** | 58.13 |
| 100 | verify all | 63.84 | 64.98 | 40.04 | 78.65 | 59.11 |

Paired bootstrap at B = 20 (same lr 3e-4): RL - classifier ROxf M **+1.12 [+0.61, +1.73]** (30 better / 3 worse), gld_val **+0.14 [+0.06, +0.21]**; RL - value ROxf M **+0.91 [+0.24, +1.87]**, ROxf H **+0.59 [+0.09, +1.20]**, gld_val +0.08 [+0.00, +0.16]. At lr 1e-3 the same picture (ROxf M +1.02 and +0.88, both CIs > 0). At B = 10 the methods tie, except RL - value ROxf H -1.91 [-4.24, -0.04] at lr 3e-4. RParis ties (near the verify-all ceiling).
- **The Track V result replicates on a third descriptor:** with 20 verifications RL beats both learned baselines on ROxford Medium by about +1 (significant), +5.3 over standard top-20 re-ranking and +10.6 over no verification. The margins on held-out GLDv2 queries are small (+0.1).

### R3. RL vs surrogates on the new ResNet baseline, 5 seeds with final evaluation (jobs 4606162-76)
All arms lr 1e-6, all-vs-all batches + bank, 2 x 1,432 steps, mean-val selection; init = new ResNet Stage 1 of the same seed.

| model (5 seeds, mean +- std) | GLDv2 val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| new ResNet Stage 1 | 64.15 +- 0.06 | 52.77 +- 1.24 | 28.11 +- 2.87 | 77.33 +- 1.68 | 57.73 +- 1.71 |
| + PL RL | 64.36 +- 0.14 | 55.45 +- 1.48 | 28.47 +- 3.25 | 78.80 +- 1.04 | 59.76 +- 0.96 |
| **+ FastAP-bank** | **64.55 +- 0.12** | **56.06 +- 1.41** | **30.31 +- 2.94** | **79.49 +- 1.02** | **60.70 +- 1.03** |
| + Smooth-AP | 63.96 +- 0.12 | 55.56 +- 1.81 | 29.24 +- 3.00 | 78.81 +- 0.98 | 59.83 +- 0.73 |

Paired over 5 seeds, mean +- std (t):
- **PL RL - FastAP-bank:** val -0.19 (t -10.6), ROxf M -0.60 +- 0.83 (t -1.6), **ROxf H -1.85 +- 0.70 (t -5.9), RPar M -0.69 +- 0.23 (t -6.6), RPar H -0.94 +- 0.27 (t -8.0)**; FastAP-bank is better in every seed on RParis and ROxford Hard.
- PL RL - Smooth-AP: within +-0.1 except ROxf H -0.77 (t -5.2).
- PL RL - Stage 1: ROxf M +2.69 (t 4.4), RPar M +1.47, RPar H +2.03; FastAP-bank - Stage 1: ROxf M +3.29, ROxf H +2.20, RPar M +2.16, RPar H +2.97.
- **Same conclusion on the stronger baseline as in G10:** reward-tuning the descriptor with the PL policy improves the model, but a matched FastAP-bank surrogate improves it more, now significantly on three of four protocols.
- R1M evaluation of the baselines first failed on a truncated official JPEG (fixed: decoded with padding and counted, commit `fix(eval): decode truncated R1M JPEGs...`), resubmitted as jobs 4606886-88. Track V on the new ResNet descriptor: attention policies submitted.

### R2. RL vs surrogates on the new ResNet baseline: seed-42 grid (jobs 4605723-31, 32-34 min each)
Init: new ResNet Stage 1 seed 42 (`gr_full_c02_e10_seed42`, mean GLDv2+SfM val 71.42). All arms: all-vs-all batches of 128 + 4,096 bank, 2 x 1,432 steps, selection on mean val.

| arm | lr 3e-7 | lr 1e-6 | lr 3e-6 |
|---|---|---|---|
| PL RL (AP@100 reward) | 72.39 | **72.53** | 71.73 |
| FastAP-bank | 72.42 | **72.53** | 70.11 |
| Smooth-AP | 72.08 | 72.09 | 70.52 |

- Every arm improves on the init by about +1.1 mean val at lr 1e-6, and **PL RL and FastAP-bank tie exactly (72.53)**; Smooth-AP is 0.4 lower. Too high an lr (3e-6) hurts every arm, the surrogates more than RL.
- **5 seeds with final evaluation submitted** (lr 1e-6 for all three arms, seeds 42/1/2/3/4, `r2cmp_*`).

### L1. R1M distractors for a locked evaluation (2026-10-17)
- **Source:** the official revisitop scripts (github.com/filipradenovic/revisitop): 100 archives `http://ptak.felk.cvut.cz/revisitop/revisitop1m/jpg/revisitop1m.{1..100}.tar.gz` and the list `revisitop1m.txt` (1,001,001 images). No official checksums or licence statement exist; we check each archive's size against the server and `gzip -t`, and record our own sha256 (`*.ok` files).
- **Size and location:** 476 GB, too large for HPCVAULT (double-charged, would exceed the hard limit), so stored packed on WORK (`$WORK/datasets/revisitop1m/raw`; WORK has 10.7 TB and only the file count was tight, which 100 archives do not affect). WORK has no backup; the data is re-downloadable with `fetch.sh` there.
- **No re-encoding:** the images are already at most 1024 px on the long side (checked on archive 1: 510 images, max side 1024, all in the official list), so `scripts/eval_r1m.py` streams the original archives in DataLoader workers and resizes to long side 1024 exactly like the database images. R1M embeddings are cached per model on WORK. `scripts/prepare_r1m.py` (resize/repack) exists but is not used.
- **Protocol:** database = dataset images + all R1M images; positives/junk from the Revisited ground truth; official AP. Results with and without R1M are reported.
- **Queued after the download:** new ResNet baseline (seed 42), DINOv2 baseline (6-epoch run + WiSE-FT 0.7) and off-the-shelf DINOv2.
- **Download incident (2026-10-17 21:15):** I mistook a moment with no active transfers (all four were verifying) for a dead download and started a second copy. Both wrote to archives 27–30; the size check caught it (sizes up to 9.5 GB vs 5 GB expected), the files were deleted and are being re-downloaded, and the duplicate session was killed within minutes. Only archives passing the size + gzip check get an `.ok` marker.
- **Use as a locked evaluation:** R1M results are reported for final models only and are not used for any selection or tuning.

### M9. WiSE-FT on the final DINOv2 candidates (jobs 4602053-55, 2026-10-17)
Mean of GLDv2 val and SfM val, by interpolation weight alpha (0 = off-the-shelf, 1 = fine-tuned):

| alpha | 0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| one-pass rung-2 winner (c04) | 75.16 | 79.03 | 81.75 | 82.61 | 83.18 | 83.47 | **83.54** | 83.43 | 83.16 |
| 6-epoch full run (c04) | 75.16 | 79.53 | 82.52 | 83.42 | 83.95 | **84.12** | 83.92 | 83.36 | 82.45 |

Benchmarks (not used for selection), ROxford M/H, RParis M/H:

| model | mean val | ROxf M / H | RPar M / H |
|---|---|---|---|
| off the shelf (alpha 0) | 75.16 | 64.51 / 41.81 | 88.29 / 75.24 |
| rung-2 c04 raw (alpha 1) | 83.16 | 68.86 / 47.65 | 90.88 / 80.74 |
| rung-2 c04 WiSE-FT alpha 0.8 | 83.54 | 70.14 / 49.39 | 91.25 / 81.57 |
| 6-epoch run raw (alpha 1) | 82.45 | 61.98 / 43.15 | 90.04 / 78.33 |
| **6-epoch run WiSE-FT alpha 0.7 (selected by rule)** | **84.12** | 68.14 / 48.67 | 91.11 / 80.67 |
| (M4) buggy 2-epoch run WiSE-FT alpha 0.7 | 83.41 | 72.14 / 51.38 | 91.85 / 82.94 |

- **WiSE-FT helps both candidates** (+0.4 / +1.7 mean val over the raw fine-tuned models) and almost cancels the damage of long training (6-epoch run: ROxf M 61.98 -> 68.14).
- **By the pre-declared rule (highest mean validation) the DINOv2 baseline is the 6-epoch run with WiSE-FT alpha 0.7** (84.12): ROxf M/H 68.14 / 48.67, RPar M/H 91.11 / 80.67, versus 64.51 / 41.81 and 88.29 / 75.24 off the shelf (+3.6 / +6.9, +2.8 / +5.4).
- **Validation and benchmarks still disagree about the ranking:** the model with the best mean validation (84.12) has lower ROxford than the one-pass interpolation (70.14) and than the earlier buggy-run interpolation (72.14, mean val 83.41). Mean validation, which includes in-domain GLDv2 val, is not a reliable proxy for ROxford Medium among strong DINOv2 candidates. No selection is changed because of this; it is a limitation to report, and the claims on DINOv2 need a locked evaluation.

### M8. Vault cleanup (2026-10-17, at the user's request)
- **Deleted 161 checkpoint directories on `$HPCVAULT/runs` (94.5 GB apparent, about 182 GB charged; quota 1,663 -> 1,481 GB of 2,097 GB hard):** all hyperparameter-search runs except the DINOv2 rung-2 trio, the pilot, smoke and debug runs, the RL tuning grids (p1_*, pl_*, sm_*, g_pl_*, g_sm_*, g_sap_*, g_fapb_*, g_ml_*, fromsm_*, gd0_*, gfromfapb_*, glong_*, grid_*, nv_*, nv2_*), the two DINOv2 Stage 1 runs with the freeze/decay bug, the 20-epoch ResNet run and the A0 embedding caches. Names and sizes are listed in `results/deleted_checkpoints_2026-10-17.txt`. All their metrics, configs and histories remain in `results/runs/` and in this file; only the weights were removed.
- **Kept (final-result artifacts):** all Stage 1 baselines (s1_b32_*, g_s1_b128_lr1e-4_seed*, gr_full_*, gd_full_*, gn_*, gnc_*), the final comparison runs (cmp_*, gcmp_*), Track V data, WiSE-FT outputs, the stage-0 checkpoint and the DINOv2 rung-2 winners.
- **Not touched: datasets (656 GB apparent), dominated by the raw GLDv2 archives (498 GB).** They can be re-downloaded in about two hours but are needed for any repacking at another resolution or for the full-GLDv2 verifier experiments, so deleting them is a separate decision.

### M7. Full-length winner runs finished (2026-10-17; jobs 4596033-38)
**ResNet50, new recipe (AdamW lr 7.5e-5, wd 0.009, warmup 2,000 + cosine, RandomResizedCrop 0.3, 10 epochs), 5 seeds:**

| | GLDv2 val (512 px) | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|
| old GLDv2 Stage 1 (Adam, no warmup, 5 seeds; G8) | 63.01 +- 0.16 | 47.93 +- 1.90 | 24.88 +- 2.93 | 74.22 +- 1.44 | 54.73 +- 0.58 |
| **new recipe (5 seeds)** | **64.15 +- 0.06** | **52.77 +- 1.24** | **28.11 +- 2.87** | **77.33 +- 1.68** | **57.73 +- 1.71** |

Per seed (final): seed 42 52.14 / 25.11, 77.37 / 57.82; seed 1 53.20 / 28.56, 79.23 / 60.02; seed 2 54.32 / 31.83, 76.17 / 56.47; seed 3 53.14 / 29.68, 75.21 / 55.70; seed 4 51.03 / 25.38, 78.70 / 58.62. Mean val (GLDv2+SfM) per epoch rises monotonically to 71.4-71.6 in every seed (best epoch 10, seed 4: 9), no early peak. **The new recipe is a clearly stronger baseline** (+4.8 ROxf M, +3.2 ROxf H, +3.1 RPar M, +3.0 RPar H; every new seed above every old seed on RParis M except one). It replaces the old ResNet Stage 1 as the baseline for all RL comparisons.

**DINOv2 CLS, c04 recipe, 6 epochs x full passes (seed 42):** mean val per epoch 82.20, 82.08, 82.11, 82.21, 82.44, **82.45**; GLDv2 val 77.01 -> 78.56 (rising) while SfM val 87.40 -> 86.33 (falling); train loss 0.131 -> 0.054. Final: GLDv2 val 78.58, **ROxf M/H 61.98 / 43.15, RPar M/H 90.04 / 78.33**. Compared with off the shelf (64.51 / 41.81, 88.29 / 75.24): ROxf Medium is lower, the other three are higher.
- **Long fine-tuning still forgets, in a way the working freeze/decay and AdamW do not prevent:** the validation signals disagree after the first epoch (in-domain GLDv2 val up, SfM val down). The one-pass rung-2 run of the same recipe scored higher on mean val (83.16) than this 6-epoch run (82.45), so on validation a short schedule is better for DINOv2 (ResNet keeps improving with length).
- **Selection caveat (declared):** the mean of GLDv2 val and SfM val is a questionable selection rule for DINOv2 because the two move in opposite directions. Comparing earlier results with the benchmarks shows SfM val tracking ROxford Medium much better than GLDv2 val does (e.g. SfM val 86.3 / 86.9 / 87.6 / 88.5 -> ROxf M 62.0 / 64.5 / 67.2 / 71.4). That observation comes from looking at benchmark numbers, so any change of selection rule based on it is informed by the test set; it must be declared, and the final claims should be confirmed on an evaluation that played no role in any decision (R1M, held-out identities, or the GLDv2 official retrieval set).
- **Next (submitted):** WiSE-FT on the one-pass rung-2 winner (4602053) and on the 6-epoch run (4602054), with final evaluation, and the raw benchmark evaluation of the rung-2 winner (4602055).

### M6. Search rung 2 and full-length winner runs (2026-10-17)
Rung 2 (4 x 3,000 steps, about one GLDv2 pass; mean GLDv2+SfM val, per epoch, best last): DINOv2 **c04 83.16** (81.82 82.49 82.95 83.16), c00 83.12, c10 (lr 4e-6) 82.92; ResNet50 **c02 65.86** (59.00 62.78 65.20 65.86), c11 65.86, c09 (lr 3.8e-5) 65.33. Winners tie again within noise, so the choice is by the pre-set rule (top score, ties by lower lr).
- **Final recipes:** DINOv2 CLS (768-d, no projection), AdamW lr 1.1e-5, weight decay 0.018, layer decay 0.83, 4 frozen blocks, FastAP, default crop, 448 px; ResNet50 + GeM + 512-d projection, AdamW lr 7.5e-5, weight decay 0.009, FastAP, RandomResizedCrop(0.3), 512 px; both batch 128, 2,000 warmup steps then cosine to zero per step, mean-val selection.
- **Full-length runs submitted:** DINOv2 6 epochs x ~12,100 steps (job 4596033, seed 42); ResNet 10 epochs, seeds 42, 1, 2, 3, 4 (jobs 4596034-38). WiSE-FT is applied to the DINOv2 result afterwards. The ResNet seeds are the new Stage 1 baseline for the RL comparisons (replacing the Adam/no-warmup ResNet models), if they beat the old ones on validation.

### M5. Search rung 1 (2026-10-16; 3 x 1,500 steps, mean GLDv2+SfM val, best epoch; jobs 4594530-43)
| DINOv2 config | lr / wd / layer decay / freeze / crop | epochs 1-2-3 | best |
|---|---|---|---|
| c00 | 1.8e-5 / 0.073 / 0.71 / 2 / default | 81.54 82.45 **82.71** | 82.71 |
| c04 | 1.1e-5 / 0.018 / 0.83 / 4 / default | 81.61 82.46 82.69 | 82.69 |
| c03 | 8e-6 / 0.0015 / 0.97 / 6 / RRC | 81.37 82.17 82.54 | 82.54 |
| c07 | 2.6e-5 / 0.005 / 0.64 / 4 / RRC | 81.48 82.24 82.53 | 82.53 |
| c10 | 4e-6 / 0.076 / 0.79 / 2 / default | 81.74 82.32 82.39 | 82.39 |
| c13 | 3e-6 / 0.004 / 0.96 / 2 / RRC | 81.57 82.17 82.29 | 82.29 |
| c02 (ArcFace) | 3e-6 / 0.013 / 0.55 / 0 | 76.09 76.59 76.75 | 76.75 |

| ResNet50 config | lr / wd / crop | epochs 1-2-3 | best |
|---|---|---|---|
| c11 | 1.0e-4 / 1.1e-3 / RRC | 56.57 60.87 **62.38** | 62.38 |
| c02 | 7.5e-5 / 9.0e-3 / RRC | 57.54 61.20 62.20 | 62.20 |
| c12 | 1.5e-4 / 0.036 / RRC | 54.78 60.13 62.08 | 62.08 |
| c09 | 3.8e-5 / 2.3e-4 / default | 59.16 60.99 61.88 | 61.88 |
| c05 | 5.1e-5 / 3.2e-4 / RRC | 58.04 60.81 61.78 | 61.78 |
| c14 | 2.5e-5 / 5.2e-3 / default | 58.43 60.39 60.99 | 60.99 |
| c01 (ArcFace) | 2.2e-5 / 7e-4 / RRC | 33.92 38.36 39.67 | 39.67 |

- **The top FastAP configurations are indistinguishable** (DINOv2 82.3-82.7, ResNet 61.8-62.4, spreads below what one seed can resolve); weight decay, layer decay, freezing depth and crop style show no clear effect in this range. The best epoch is always the last: with warmup + cosine decay to zero inside each rung, the forgetting peak seen in the old runs (best at epoch 2 of 6) does not appear. Longer budgets keep helping (DINOv2 rung 0 -> 1: +0.7).
- **ArcFace improves with budget (DINOv2 75.5 -> 76.8, ResNet 28 -> 39.7) but stays far below FastAP**, so the rung-0 bias does not explain the gap at these budgets; it is dropped.
- **Rung 2** (4 x 3,000 steps = about one GLDv2 pass): DINOv2 c00, c04 and the low-lr hedge c10; ResNet c11, c02 and the low-lr hedge c09 (short runs favour high learning rates, a known bias of successive halving). Jobs below; then a full-length run of each winner, WiSE-FT on the result, and 5 seeds for ResNet.

### M4. Search rung 0 and WiSE-FT results (2026-10-16)
**WiSE-FT** (jobs 4593966–67; α chosen on mean GLDv2+SfM val; α = 0 is the off-the-shelf model, 1 the fine-tuned one; DINOv2 CLS, M2 checkpoints which were trained without working freeze/decay):

| α | 0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| epoch-2 ckpt, mean val | 75.16 | 78.96 | 81.73 | 82.61 | 83.16 | **83.41** | 83.36 | 83.06 | 82.54 |
| epoch-6 ckpt, mean val | 75.16 | 78.85 | 81.69 | 82.64 | 83.24 | **83.47** | 83.32 | 82.74 | 81.60 |

- Interpolating weights beats both end points on validation (+0.9 over the best fine-tuned checkpoint, +8.3 over the off-the-shelf model), and removes the epoch dependence: the over-trained epoch-6 model becomes the best after interpolation. At α = 0.7: GLDv2 val 78.05, SfM val 88.90.
- Benchmarks of the selected interpolations (not used for selection): epoch-2 α 0.7: ROxf M/H 72.14 / 51.38, RPar M/H 91.85 / 82.94; epoch-6 α 0.7: 68.59 / 45.26, 91.46 / 81.83. For reference: off the shelf 64.51 / 41.81, 88.29 / 75.24; fine-tuned epoch 2 67.20 / 45.65, 90.38 / 79.50.
- So the forgetting is largely a drift in weight space that interpolation undoes; the fine-tuned weights hold useful task knowledge but have moved too far from the pretrained solution.

**Rung 0 of the search** (2 × 750 steps; mean val, best epoch; all 32 runs finished in ~2 h):
- *DINOv2:* FastAP configurations 81.7–82.0 (c00 82.02, c04 81.99, c07 81.92, c03 81.87, c10 81.71, c13 81.70), very flat across lr 3e-6–2.6e-5, weight decay 0.0015–0.076, freeze 2–6, layer decay 0.64–0.97; too-small lr (1e-6, 2e-6) 80.5. **Every sub-center ArcFace configuration was worse (best 75.5, down to 15) and most were below the untrained model (75.16).**
- *ResNet50:* FastAP 55.3–58.7 (c11 58.69, c02 58.54, c12 58.49, c09 58.43, c05 58.10, c14 57.63), lr 2.5e-5–1.5e-4 flat; ArcFace 2.7–27.9.
- **Caveat on ArcFace:** the class-centre matrix (79,813 × 3 vectors) starts random and needs many passes; 1,500 steps see each class about once. Successive halving at short rungs is therefore biased against it, so the best ArcFace configuration of each backbone is promoted as a check.
- **Rung 1** (3 × 1,500 steps): the six best FastAP configurations plus the best ArcFace configuration per backbone, jobs 4594530–43.

### M3. Why DINOv2 forgets and peaks at epoch 2: two training bugs/defects found (2026-10-16)
**Diagnosis from the checkpoints** (relative change ‖θ − θ₀‖/‖θ₀‖ and weight-norm ratio vs the off-the-shelf model, per group of blocks):

| run | blocks 0–3 weights (should be frozen) | blocks 4–7 weights | blocks 8–11 weights | LayerNorm, blocks 0–3 |
|---|---|---|---|---|
| pilot, 3k steps | change 0.048, norm ×0.982 | 0.010, ×0.996 | 0.011, ×0.997 | 0.003, ×0.997 |
| full run, epoch 2 (selected) | **0.261, ×0.858** | 0.072, ×0.951 | 0.068, ×0.963 | 0.037, ×0.966 |
| full run, epoch 6 | **0.360, ×0.788** | 0.119, ×0.917 | 0.104, ×0.939 | 0.067, ×0.939 |
| D2 (lr 1e-5), epoch 6 | 0.601, ×0.576 | 0.341, ×0.762 | 0.304, ×0.801 | 0.225, ×0.798 |

1. **Bug: `model.freeze_blocks` and `model.layer_decay` never took effect.** `torch.compile` wraps the model before the optimizer is built, so parameter names start with `_orig_mod.`; `param_groups` (added in M1) matched none of them and put every parameter in one full-lr group. The logs of the M1 grid and M2 run say `param groups: 1`. Consequences: the "4 frozen blocks" moved the most (blocks 0–3: 26% at epoch 2, 36% at epoch 6), and grid configs that differ only in freezing/decay gave identical results (82.07 vs 82.08). The M1 grid conclusions about freezing/decay and the M2 DINOv2 baseline are invalid as tuned results. Unaffected: ResNet runs, Track V, the forgetting test (no freezing requested), all SfM-120k work.
2. **Defect: coupled L2 weight decay with Adam on every parameter.** `optim.Adam(weight_decay=1e-4)` adds 1e-4·θ to the gradient before Adam's normalisation, so wherever the loss gradient is small the decay becomes a full-size update and pulls weights to zero at roughly lr per step, including LayerNorm, biases and LayerScale γ (which gate each block). Weight norms shrink steadily (×0.79 in blocks 0–3 by epoch 6), matching SfM val falling after epoch 2. The standard ViT recipe is AdamW (decoupled decay) with no decay on norms, biases, LayerScale and tokens. The same coupled decay was used for all ResNet Stage 1 runs (BatchNorm included).
3. **Train/test resolution mismatch.** GLDv2 images are stored at long side 512; the default train transform resizes the short side to 492 (≈1.28× upsampling) and random-crops 448², while ROxford/RParis are evaluated at long side 1024.

**Fixes (commit `fix(train): apply ViT freezing and layer decay under torch.compile; ...`)**: names are normalised and a freeze/decay request that matches nothing raises an error (CPU-tested on a compiled model: 9 layer groups, 29.9M frozen parameters, lr 3e-7–3e-6); AdamW with proper no-decay groups; per-step linear warmup + cosine; sub-center ArcFace (79,813 classes × 3 centres, own head lr) as a loss option; loss computed in fp32 under AMP; RandomResizedCrop option; lean checkpoints and a per-run `selection.json`.

**Tuning method from now on (selected only on mean GLDv2+SfM validation):**
- *Successive halving over a scrambled Sobol sample* (`scripts/hp_search.py`): 16 configurations per backbone; rung 0 = 2 × 750 steps, top third promoted to rung 1 = 3 × 1,500 steps, top two to rung 2 = 4 × 3,000 steps, then a full run of the winner. DINOv2 space: lr 1e-6–3e-5 (log), AdamW weight decay 1e-3–1e-1 (log), layer decay 0.55–1.0, freeze {0, 2, 4, 6, 8} blocks, loss {FastAP, sub-center ArcFace}, head lr 3e-4–3e-3, crop {default, RandomResizedCrop 0.3}. ResNet space: lr 2e-5–5e-4, weight decay, loss, head lr, crop.
- *WiSE-FT* (`scripts/wise_ft.py`, Wortsman et al. 2022): interpolate pretrained and fine-tuned weights, α chosen on validation; first applied to the existing M2 checkpoints (jobs 4593966–67).
- Smoke runs of the new path (jobs 4593963–65) **passed** (13, 12 and 10 min): `param groups: 11 layers (frozen layers 0..2: 15,684,096 params, layer decay 0.713)` for DINOv2 freeze 2, `5 layers (frozen 0..8: 58,220,544 params)` for freeze 8 (old runs: `param groups: 1`); sub-center ArcFace with 79,813 classes x 3 centres for both backbones; selection.json and lean checkpoints written.
- **Rung 0 launched 2026-10-16**: 16 DINOv2 + 16 ResNet configurations (`results/hpsearch/{dinov2,resnet}.json`), 2 epochs x 750 steps each, `hp_{space}_r0_cNN`. Collect with `python scripts/hp_search.py collect --configs results/hpsearch/<space>.json --rung 0`.

### M2. Full-length Stage 1 runs with mean-val selection (jobs 4582323, 4581061; finished 2026-10-16)
- **DINOv2 CLS, lr 3e-6, 4 frozen blocks, layer decay 0.75, 6 epochs** (12 h 57 min): mean val per epoch 82.36, **82.54**, 81.89, 81.65, 81.55, 81.60 (GLDv2 val keeps rising to 77.5 while SfM val falls from 88.0 to 85.7: forgetting again, now visible to selection). Selected epoch 2: GLDv2 val 77.44 / SfM val 87.63 (at 448 px). Benchmarks: **ROxf M/H 67.20 / 45.65, RPar M/H 90.38 / 79.50**. The short pilot (mean val 82.08) had 71.38 / 51.31 and 91.26 / 81.34: by the selection rule the full run is the DINOv2 baseline, although it is lower on both benchmarks. Validation differences of ~0.5 do not track 70-query benchmark differences reliably.
- **ResNet50 GLDv2, 20 epochs** (19 h 28 min): mean val rises every epoch to 69.78 (epoch 20). Benchmarks: ROxf M/H 47.72 / 26.16, RPar M/H 74.12 / 53.04, vs 49.93 / 27.15 and 72.27 / 53.79 after 10 epochs (seed 42). No clear gain from doubling the schedule.
- Checked: `stage1_best.pth` of the DINOv2 run holds epoch 2 (selection score 0.8254). The GLDv2 val reported by the final evaluation (77.54) is computed at 512 px, while training validation uses 448 px; this applies to every run.

### M1. DINOv2 Stage 1 tuned honestly; forgetting test; Track V on DINOv2 (2026-10-15)
**Selection rule for all runs below:** mean of GLDv2-clean val and SfM-120k val leave-one-out mAP (both held out; no ROxford/RParis). Off-the-shelf DINOv2 CLS: GLDv2 val 63.33, SfM val 86.93 → mean **75.13**.

*Stage 1 tuning grid* (jobs 4581055–59; FastAP, 768-d CLS output without projection, 448 px, 3 × 1,000 batches; best epoch by mean val):

| lr / freeze / layer decay | GLDv2 val | SfM val | **mean (selection)** | ROxf M / H | RPar M / H |
|---|---|---|---|---|---|
| 1e-5 / 0 / 0.65 | 75.58 | 87.82 | 81.70 | 70.38 / 48.37 | 90.13 / 78.78 |
| 1e-5 / 6 / 1.0 | 75.52 | 87.75 | 81.64 | 70.82 / 49.37 | 90.25 / 79.04 |
| 3e-6 / 0 / 0.75 | 75.67 | 88.47 | 82.07 | 71.35 / 51.20 | 91.24 / 81.29 |
| 1e-6 / 0 / 1.0 | 74.61 | 88.92 | 81.77 | 70.41 / 48.98 | 91.73 / 82.57 |
| **3e-6 / 4 / 0.75** | 75.68 | 88.49 | **82.08** | **71.38 / 51.31** | 91.26 / 81.34 |
| (off the shelf) | 63.33 | 86.93 | 75.13 | 64.51 / 41.81 | 88.29 / 75.24 |

- **Gentle fine-tuning without a random projection beats off-the-shelf DINOv2 on validation and on both benchmarks** (selected config: ROxf +6.9 / +9.5, RPar +3.0 / +6.1). The failed run D2 used a randomly initialised 512-d projection, lr 1e-5 without layer decay, and GLDv2-only selection.
- **Full run of the selected config submitted** (6 epochs, ~9 h, job 4582323). This becomes the DINOv2 baseline.

*Forgetting test* (jobs 4581063–68; from off-the-shelf, all-vs-all batches + 4,096 bank, AMP, 2 × 1,432 batches of 128; best epoch by mean val):

| arm | lr | mean val (selection) | ROxf M / H | RPar M / H |
|---|---|---|---|---|
| PL RL | 1e-6 | 78.50 | 72.15 / 49.49 | 89.32 / 79.78 |
| PL RL | 3e-7 | 79.06 | 71.10 / 48.51 | 87.16 / 74.92 |
| **FastAP-bank** | **3e-7** | **81.20** | **74.29 / 54.85** | 89.70 / 78.96 |
| FastAP-bank | 1e-6 | 79.05 | 75.19 / 54.30 | 87.81 / 77.99 |
| Smooth-AP | 3e-7 | 78.75 | 71.84 / 50.70 | 87.02 / 75.09 |
| Smooth-AP | 1e-6 | 77.77 | 72.45 / 49.11 | 86.75 / 75.04 |

- **RL does not preserve DINOv2's generality better than the surrogates;** FastAP-bank is best on validation (81.20 vs 79.06 for the best RL). The "gentler RL" hypothesis is not supported.
- By validation, the Stage 1 grid winner (82.08) remains the best DINOv2 model.

*Track V on the off-the-shelf DINOv2 descriptor* (precompute 4580901; attention policies 4580902–03; 10,000 GLDv2 queries; lr 3e-4):

| B | method | gld_val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|---|
| 0 | none | 61.08 | 64.50 | 41.79 | 88.29 | 75.24 |
| 20 | top-B | 61.92 | 67.96 | 48.43 | 88.40 | 75.40 |
| 20 | classifier | 63.24 | 71.36 | 49.69 | 88.42 | 75.42 |
| 20 | value | 63.31 | 71.50 | **50.24** | 88.26 | 75.28 |
| 20 | **RL** | **63.43** | **71.60** | 50.16 | 88.36 | 75.39 |
| 100 | verify all | 60.86 | 72.56 | 50.74 | 88.82 | 75.90 |

Bootstrap RL − baseline: gld_val vs value **+0.20** (B10) / **+0.13** (B20), vs classifier **+0.54 / +0.19** (all CIs > 0); ROxf at B20 vs classifier **M +0.24 [+0.11, +0.39], H +0.46 [+0.10, +0.98]**; at B10 ROxf below the classifier (n.s.). RParis ties.
- **The RL verification-policy advantage on held-out queries replicates on a second descriptor,** but on the benchmarks DINOv2 leaves little room: RL at B = 20 is within 1 point of verifying the whole shortlist on ROxford Medium.

### D2. DINOv2 Stage 1 on GLDv2-clean forgets (job 4575802, 9 h 34 min)
CLS token, FastAP, lr 1e-5, batch 128, 448 px, 6 epochs. GLDv2 val 63.33 (off-the-shelf) → 72.65, 72.84, 72.74, 74.08, 75.18, 76.32, **76.51**. Benchmarks of the final checkpoint (long side 1024): **ROxf M/H 53.53 / 26.91, RPar M/H 77.48 / 58.43**, against **64.51 / 41.81 and 88.29 / 75.24 off the shelf**.
- In-domain validation rises by 13 points while ROxford/RParis drop by 11–17: fine-tuning on GLDv2 with FastAP destroys the general features of DINOv2 (forgetting). The selection metric (GLDv2 val) cannot see this.
- **New RL test (submitted):** starting from the off-the-shelf model (`$HPCVAULT/runs/stage0/dinov2_vitb14_cls.pth`, `scripts/make_stage0_checkpoint.py`), fine-tune with PL RL vs FastAP-bank vs Smooth-AP (same batches, bank, AMP, 2 × 1,432 batches, lr 1e-6 / 3e-7), with final benchmark evaluation (first submitted as 4580904–09; cancelled after 5 min and resubmitted with mean GLDv2+SfM val selection, see below). Earlier runs showed RL changing the model more gently (C4), so RL may keep the pretrained generality better.
- **Track V on the off-the-shelf DINOv2 descriptor** submitted: precompute 4580901 (10,000 GLDv2 queries + benchmarks), attention policies 4580902–03.

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

### V4. Attention policy over the whole shortlist (jobs 4576077–78, 2 h 11 min each)
All learned methods (classifier, myopic value, RL) use the same attention network: per-candidate features → 2 self-attention layers over the 100 candidates, with attention logits biased by pairwise descriptor similarity (72k parameters). 10,000-query data (V3); lr 1e-3 or 3e-4 for every method; 3 seeds.

| B | method (lr chosen per method on gld_val) | gld_val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|---|
| 0 | none | 63.41 | 54.35 | 30.35 | 75.37 | 56.36 |
| 10 | top-B | 64.22 | 55.76 | 33.07 | 75.47 | 56.46 |
| 10 | classifier (1e-3) | 65.47 | 57.53 | 34.55 | 75.60 | 56.49 |
| 10 | value (1e-3) | 65.57 | 58.02 | 34.25 | 75.67 | 56.49 |
| 10 | **RL (3e-4)** | **65.77** | **58.85** | **35.91** | 75.61 | 56.42 |
| 20 | top-B | 64.76 | 56.83 | 33.95 | 75.61 | 56.54 |
| 20 | classifier (1e-3) | 66.16 | 60.12 | 36.30 | 75.89 | 56.66 |
| 20 | value (1e-3) | 66.13 | 59.85 | 34.66 | 75.87 | 56.63 |
| 20 | **RL (3e-4)** | **66.25** | **61.33** | **37.28** | **75.90** | **56.67** |
| 100 | verify all | 63.85 | 64.85 | 43.10 | 76.89 | 57.71 |

Paired bootstrap at the same lr (3e-4), RL − baseline:
- gld_val: vs value **+0.17 [+0.08, +0.27]** (B10), **+0.15 [+0.06, +0.24]** (B20); vs classifier **+0.37 / +0.13**.
- ROxf M: vs classifier **+1.19 [+0.51, +1.88]** (B10), **+1.10 [+0.68, +1.57]** (B20); vs value +0.50 [−0.32, +1.47] (B10), **+0.72 [+0.22, +1.33]** (B20).
- ROxf H: vs classifier +1.18 [−0.03, +2.37] (B10), **+0.99 [+0.30, +1.75]** (B20); vs value +1.72 [−0.06, +4.22] (B10), +0.49 [−0.52, +1.88] (B20).
- RParis: ties (RPar M within ±0.05; RPar is near saturation, verify-all is only +1.0 above no verification).

- **With a policy that sees the whole shortlist, RL leads every learned baseline on the held-out split and on ROxford, and the ROxford Hard deficit of V1–V3 disappears.** RL is significantly better than the classifier on ROxford Medium and Hard at B = 20 and than the myopic value policy on ROxford Medium at B = 20.
- At B = 20, RL with attention reaches ROxf M/H **61.33 / 37.28**: +4.5 / +3.3 over standard top-20 re-ranking and +7.0 / +6.9 over no verification, with 20 verifications per query instead of 100.
- The learned baselines gained the most from attention (classifier ROxf M at B20: 58.57 → 60.12). RL's margin over them is modest (+0.1 to +0.4 gld_val, +0.7 to +1.5 ROxford Medium), but it has the same sign in every setting.

### V3. 10,000 training queries and a hard-proxy reward (precompute 4575824, 51 min; policies 4575827–29, 18 min each)
Same descriptor and protocol; 10,000 GLDv2-clean val queries → 6,720 policy-train / 3,280 policy-val (landmark split); 3,000 iterations; reward mixed with a hard-proxy AP (shortlist positives with ≥ 150 inliers ignored) at weight 0 / 0.25 / 0.5.

| B | method (hard mix 0) | gld_val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|---|
| 10 | top-B | 64.22 | 55.76 | 33.07 | 75.47 | 56.46 |
| 10 | classifier | 64.92 | 56.56 | **33.56** | 75.59 | 56.49 |
| 10 | value (myopic) | 64.98 | 56.75 | 32.04 | 75.72 | 56.51 |
| 10 | **RL** | **65.54** | **57.27** | 31.83 | **75.87** | **56.55** |
| 20 | top-B | 64.76 | 56.83 | 33.95 | 75.61 | 56.54 |
| 20 | classifier | 65.77 | 58.57 | **34.87** | 75.83 | 56.59 |
| 20 | value (myopic) | 65.76 | 60.44 | 35.10 | 76.08 | 56.80 |
| 20 | **RL** | **66.04** | **60.58** | 34.63 | **76.18** | **56.88** |

Paired bootstrap, RL − baseline (hard mix 0): gld_val vs value **+0.56 [+0.45, +0.67]** (B10), **+0.28 [+0.18, +0.38]** (B20); vs classifier **+0.62 / +0.27** (both CIs > 0). RPar M vs value **+0.15 [+0.07, +0.23]** (B10), +0.10 [−0.00, +0.23] (B20); vs classifier +0.28 / +0.35 (CIs > 0). ROxf M vs classifier at B20 **+2.01 [+0.18, +3.49]**; vs value +0.15 (n.s.). ROxf H: −1.7 to −0.2 vs the learned baselines, all CIs include 0.

- **RL's advantage over the myopic value policy is confirmed with 3,280 held-out queries,** but it is small: +0.3 to +0.6 mAP. More training data improved the value baseline most (ROxf M at B20: 59.31 → 60.44), which narrowed RL's lead on the benchmarks.
- **The hard-proxy reward does not help ROxford Hard** (all three mixes within noise of each other). The "≥ 150 inliers = easy" proxy apparently does not match the Revisited easy/hard split.
- **Room left:** verifying the whole shortlist reaches ROxf M 64.85 / H 43.10, against ~60.6 / 34.6 at B = 20. Next levers: a policy that looks at the whole shortlist jointly (attention over candidates, for RL and the value baseline alike), larger galleries (R1M) where choosing what to verify matters more, and the DINOv2 descriptor.

### V2. Myopic value baseline and paired bootstrap over queries (job 4575805, 9 min; `scripts/analysis/trackv_bootstrap.py`)
New baseline *value*: the same MLP and features, trained to regress the exact one-step AP gain of verifying each item (targets from the ground truth), chosen greedily. It optimises the same final-AP objective as RL but plans one step ahead.

| B | method | gld_val | ROxf M | ROxf H | RPar M | RPar H |
|---|---|---|---|---|---|---|
| 10 | classifier | 66.07 | 56.44 | 33.54 | 75.56 | 56.51 |
| 10 | value (myopic) | 65.96 | 56.59 | 33.38 | 75.59 | 56.47 |
| 10 | RL | **66.89** | **57.38** | 31.86 | **75.82** | 56.52 |
| 20 | classifier | 66.89 | 58.22 | 34.76 | 75.78 | 56.58 |
| 20 | value (myopic) | 66.97 | 59.31 | **35.67** | 75.87 | 56.58 |
| 20 | RL | **67.27** | **60.17** | 33.76 | **76.17** | **56.87** |

Paired bootstrap over queries (10,000 resamples, 95% CI), RL minus baseline:

| B | vs | gld_val (959 q) | ROxf M (70 q) | ROxf H | RPar M (70 q) | RPar H |
|---|---|---|---|---|---|---|
| 5 | value | **+0.65 [+0.46, +0.85]** | +0.13 [−1.36, +1.30] | −2.16 [−6.00, +0.68] | **+0.14 [+0.06, +0.26]** | +0.03 [−0.01, +0.09] |
| 10 | value | **+0.94 [+0.69, +1.21]** | +0.79 [−1.16, +2.41] | −1.52 [−5.79, +1.50] | **+0.23 [+0.10, +0.44]** | +0.05 [−0.05, +0.13] |
| 20 | value | **+0.30 [+0.11, +0.49]** | +0.85 [−1.11, +2.37] | −1.91 [−5.77, +0.93] | **+0.30 [+0.13, +0.52]** | **+0.29 [+0.11, +0.53]** |
| 20 | classifier | **+0.38 [+0.20, +0.56]** | +1.95 [−0.15, +3.77] | −1.00 [−4.99, +2.18] | **+0.39 [+0.18, +0.66]** | **+0.29 [+0.08, +0.56]** |
| 20 | top-B | **+1.21 [+0.94, +1.51]** | **+3.33 [+0.91, +5.72]** | −0.19 [−4.23, +3.19] | **+0.56 [+0.21, +1.10]** | **+0.33 [+0.09, +0.62]** |

- **RL beats the myopic value policy significantly on the held-out GLDv2 split at every budget**, so the gain comes from multi-step planning, not only from optimising AP. It also beats all baselines on RParis Medium at every budget (CIs exclude 0; e.g. 34 queries better vs 1 worse at B = 10 vs classifier).
- **ROxford is inconclusive with 70 queries:** Medium point estimates +0.1 to +2.0 vs the learned baselines (CIs include 0), Hard −1.0 to −2.2 (CIs include 0). Against the standard top-B re-ranking at B = 20, ROxford Medium +3.33 is significant.
- **Interpretation:** the RL verification policy is a genuine, statistically supported improvement on held-out GLDv2 queries and on RParis. The ROxford Hard drop matches the reward mismatch noted in V1 (the reward does not separate easy from hard positives).

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
