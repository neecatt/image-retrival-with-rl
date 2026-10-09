---
name: retrieval-experiment
description: Design, audit, implement, and compare landmark retrieval experiments with metric learning and RL task rewards. Use for benchmark protocols, policy-gradient correctness, backbone comparisons, and experimental ablations in this project.
---

# Retrieval experiments

Produce evidence that separates improvements in representation, additional training, and task-reward optimization. Follow the repository AGENTS.md for all execution.

## Establish the comparison

Inspect relevant configs, trainers, data loaders, evaluators, and actual result artifacts. README scores are historical claims until verified against saved metrics and checkpoints. State the hypothesis and the smallest experiment that can falsify it. Preserve the user's chosen scope and compute budget.

For backbone comparisons, record pretrained checkpoint source, training data, trainable layers, aggregation, descriptor size, input resolution, and inference/postprocessing. Transformer tokens require an explicit aggregation strategy; do not silently treat tokens as CNN feature maps or apply positive-only GeM clamping without checking feature semantics.

From each Stage 1 initialization compare the initial model, continued metric learning, and reward tuning. Match additional compute or document the chosen step/sample budget and measured compute differences. Screen cheaply, then repeat the key comparison across seeds when budget permits. Tune on validation data, not final benchmark results.

Use [experiment-record.md](assets/experiment-record.md) when preparing or recording a run. Fill only relevant fields; keep unknowns explicit. Store records with results and reference the exact config and code revision, including uncommitted changes.

## Audit correctness before scaling

- Verify disjoint training/validation/test data, landmark identities where appropriate, duplicates, query exclusion from galleries, and positive availability after PK query/gallery splitting. Record foundation-model pretraining overlap uncertainty rather than claiming uncontaminated evaluation.
- Use official benchmark scoring and compare it against hand-checkable rankings: junk handling, empty-positive queries, query bounding-box crops, AP integration, and metric denominators. Distinguish fraction-of-positives Recall@K from any-positive success@K. GLDv2 relevance and scoring must follow its retrieval protocol.
- For score-function RL, identify the stochastic action, sampling distribution, matching log density, detached sampled action during density differentiation, detached reward/advantage, and an action-independent baseline. If gallery actions are stochastic, either include their appropriate gradient terms or explicitly define a conditional query-only objective with a fixed/detached gallery. Normalized Gaussian sampling is not automatically von Mises–Fisher sampling.
- Validate a new estimator on a tiny analytically tractable or numerical expected-reward problem before long runs. Check gradient direction and reward gain; a finite nonzero gradient alone does not establish correctness.
- Log reward variance, baseline, advantage spread, gradient norms, positive counts, memory-bank age, and regularization strength where useful. Higher sampled training reward does not establish better deterministic retrieval.

## Interpret results

Keep checkpoint selection fixed using held-out validation. Report benchmark Medium/Hard mAP as absolute percentage points with units, seed variability, and compute. Separate global retrieval from reranking/query expansion; published results with different data, resolution, or descriptor size need explicit caveats.

Choose ablations tied to a mechanism: consistency-only or zero policy weight to isolate RL, baseline strategy to study variance, gallery size/mining/freshness to study scale, and frozen-head versus backbone adaptation to study cost. Avoid an exhaustive grid before pilot evidence.

For large galleries, budget embedding storage and ranking memory. Chunk extraction/search; use exact rankings where the official metric requires them. If approximate search is used, measure and disclose its effect. End with artifact paths, validated findings, unresolved limitations, and the next evidence-driven experiment. Never fabricate missing runs or label proposed changes as novel without literature support.
