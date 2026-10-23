# Literature check: budgeted verification with query expansion (2026-10-24)

Scope: is a learned (RL) policy that decides which retrieved images to geometrically verify under a
per-query budget, where verified matches unlock their nearest neighbours (verified query expansion),
new for landmark instance retrieval? And where do current systems stand on ROxford/RParis (+R1M)?

Search: web searches (learned/RL verification scheduling, budget-aware re-ranking cascades, learned
query expansion, graph traversal, RL query reformulation) plus reading the closest papers. Not an
exhaustive novelty search; a reviewer may know work we did not find.

## Closest prior work

| work | what it does | verification | learned? | budget / cost | relation to ours |
|---|---|---|---|---|---|
| Total Recall (Chum et al., ICCV 2007) [1]; Total Recall II (CVPR 2011) [2] | spatially verified query expansion: verify top results, re-query with verified images | query-time, fixed top list | no | not modelled | the classic "verified QE" baseline; we add a fixed-budget *choice* of what to verify and a neighbour-unlocking rule |
| EGT / rEGT (Chang et al., CVPR 2019) [3] | traversal of a k-NN graph from the query, explore/exploit with threshold t (Alg. 1); rEGT replaces edge weights by RANSAC inlier counts | **offline for all n·k database edges** (O(nk), incl. R1M) + k = 100 RANSAC per query online (Sec. 3, "Complexity") | no (hand-designed rule, t by cross-validation) | runtime reported (Table 3), no budget | closest mechanism (expansion through verified neighbours). Ours: no offline database verification, a fixed online budget, a learned scheduler. Reported: R-MAC+QE+SV+rEGT ROxf 83.5 / 65.8, ROxf+1M 74.9 / 54.1, RPar 92.8 / 84.6, RPar+1M 87.1 / 75.6 (Table 1) |
| Learning-as-you-go verification ordering (landmark collections) [4] | orders candidate matches by how often visual words verified before | offline collection building | heuristic statistics | efficiency | different task (collection building), not query-time scheduling |
| LAttQE, attention-based QE learning (Gordo et al., ECCV 2020) [5] | learns how to aggregate neighbours into an expanded global query | none | supervised | none | learned QE without verification; complementary |
| CVNet (Lee et al., CVPR 2022) [6], SuperGlobal (Shao et al., ICCV 2023) [7], AMES (Suma et al., ECCV 2024) [8] | learned re-rankers of the top-400 (or more) | learned pairwise model replaces RANSAC | yes (supervised) | CVNet ~24 s per query for top-400, SuperGlobal 0.37 ms (Table 2 of [7]) | the modern alternative to RANSAC verification; our scheduler could schedule any expensive pairwise scorer |
| Budget-aware cascades in text IR (e.g. cost-aware cascade ranking [9]; budgeted re-ranking via active set selection [10]) | decide how many / which documents to re-rank under a cost | expensive re-ranker | learned or adaptive | yes | same problem class in text retrieval; not landmark retrieval, no expansion |
| RL query reformulation (Nogueira & Cho, EMNLP 2017) [11] | RL agent rewrites text queries, reward = recall | none | RL | none | RL for retrieval decisions in text IR |
| Classification with costly features via deep RL [12] | RL decides which costly features to acquire | n/a | RL | yes | analogous "pay for information" setting |

**Assessment.** We found no paper that learns a per-query verification schedule under a fixed budget for
landmark retrieval, nor one that couples it with verified neighbour expansion. The closest mechanism is
rEGT, which avoids query-time choices by verifying the whole database graph offline. The defensible
contribution is therefore: (i) a budgeted online formulation of verified expansion (no offline
database-side verification), (ii) evidence of when sequential decision-making helps (greedy one-step
selection is near-optimal without expansion, Track V; expansion breaks myopia, Track V-QE), and (iii)
a controlled comparison with supervised schedulers. "First to use RL for re-ranking" should not be
claimed (RL for retrieval decisions exists in text IR [11]).

Baselines a reviewer will expect, given this literature:
- verified QE with the same budget (Total Recall style) and alpha-QE without verification;
- an EGT-style traversal rule restricted to the same online budget (priority = best inlier count of a
  verified parent, explore/exploit threshold), since rEGT is the closest expansion method;
- verify-everything with expansion (upper bound of the re-ranking rule);
- a cost/accuracy curve with measured verification time.

## Where current systems stand (author-reported, not reproduced here)

| system | ROxf M / H | ROxf+1M M / H | RPar M / H | RPar+1M M / H | source |
|---|---|---|---|---|---|
| RN101-CVNet, global | 80.2 / 63.1 | 74.0 / 53.7 | 90.3 / 79.1 | 80.6 / 62.2 | [7] Table 1 |
| RN101-CVNet, rerank top-400 | 87.2 / 75.9 | 81.9 / 67.4 | 91.2 / 81.1 | 83.8 / 69.3 | [7] Table 1 |
| RN101-SuperGlobal, no rerank | 85.3 / 72.1 | 78.8 / 61.9 | 92.1 / 83.5 | 83.9 / 69.1 | [7] Table 1 |
| RN101-SuperGlobal, rerank top-400 | 90.9 / 80.2 | 84.4 / 71.1 | 93.3 / 86.7 | 84.9 / 71.4 | [7] Table 1 |
| DINOv2 + AMES (repo README) | 92.7 / 83.7 | n/a | 95.2 / 90.6 | n/a | [8] README, metric not labelled there (likely mAP) |
| R-MAC+QE+SV+rEGT (2019) | 83.5 / 65.8 | 74.9 / 54.1 | 92.8 / 84.6 | 87.1 / 75.6 | [3] Table 1 |
| **ours, DINOv2 + B20 RL with verified expansion** | 75.09 / 54.31 | 67.68 / 43.81 | 91.23 / 80.69 | 83.22 / 66.30 | PROGRESS Q8 |

Public code/checkpoints: CVNet (github.com/sungonce/CVNet, used by SuperGlobal), SuperGlobal
(github.com/ShihaoShao-GH/SuperGlobal), AMES (github.com/pavelsuma/ames), EGT (github.com/layer6ai-labs/egt).
Licences not yet checked.

## References
1. O. Chum, J. Philbin, J. Sivic, M. Isard, A. Zisserman. Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval. ICCV 2007. https://ieeexplore.ieee.org/document/4408891
2. O. Chum, A. Mikulik, M. Perdoch, J. Matas. Total Recall II: Query Expansion Revisited. CVPR 2011. https://mlanthology.org/cvpr/2011/chum2011cvpr-total
3. C. Chang, G. Yu, C. Liu, M. Volkovs. Explore-Exploit Graph Traversal for Image Retrieval. CVPR 2019. https://www.cs.toronto.edu/~mvolkovs/cvpr2019EGT.pdf
4. Improved Geometric Verification for Large Scale Landmark Image Collections. https://www.researchgate.net/publication/267229719_Improved_Geometric_Verification_for_Large_Scale_Landmark_Image_Collections
5. A. Gordo, F. Radenovic, T. Berg. Attention-Based Query Expansion Learning. ECCV 2020. https://arxiv.org/pdf/2007.08019
6. S. Lee, H. Seong, S. Lee, E. Kim. Correlation Verification for Image Retrieval. CVPR 2022. https://arxiv.org/abs/2204.01458
7. S. Shao et al. Global Features are All You Need for Image Retrieval and Reranking. ICCV 2023. https://arxiv.org/html/2308.06954
8. P. Suma, G. Kordopatis-Zilos, A. Iscen, G. Tolias. AMES: Asymmetric and Memory-Efficient Similarity Estimation for Instance-level Retrieval. ECCV 2024. https://arxiv.org/abs/2408.03282, https://github.com/pavelsuma/ames
9. Efficient Cost-Aware Cascade Ranking in Multi-Stage Retrieval. https://www.researchgate.net/publication/318764161_Efficient_Cost-Aware_Cascade_Ranking_in_Multi-Stage_Retrieval
10. BoundaryMORPH: Budgeted Reranking via Active Set Selection for Diffuse Retrieval. https://arxiv.org/pdf/2609.27213
11. R. Nogueira, K. Cho. Task-Oriented Query Reformulation with Reinforcement Learning. EMNLP 2017. https://arxiv.org/abs/1704.04572
12. Classification with Costly Features using Deep Reinforcement Learning. https://arxiv.org/pdf/1711.07364
