# Landmark retrieval research roadmap

Planning snapshot: 2026-10-08. Follow [AGENTS.md](../AGENTS.md) for repository instructions and [HPC.md](HPC.md) for compute operations.

## Project context

- Goal: reconstruct and extend the task-reward tuning framework from "Tuning computer vision models with task rewards" (https://arxiv.org/abs/2302.08242) for landmark instance retrieval, compare stronger backbones and retrieval systems, identify methodological gaps, and test improvements on larger datasets.
- Positioning: the original paper's probabilistic pretraining/reward-tuning framework motivates this work. Metric-learning pretraining and a continuous embedding policy are retrieval adaptations, not a literal reproduction of its task experiments. Distinguish reconstruction, adaptation, and demonstrated new contributions.
- Current pipeline: normalized image embeddings, cosine ranking, Stage 1 FastAP metric learning, followed by Stage 2 REINFORCE-style AP reward tuning with baselines, a gallery memory bank, and consistency regularization against the frozen Stage 1 model.
- Main current model/config: ResNet50 + GeM, 512-dimensional embeddings; SfM-120k training and ROxford5k/RParis6k evaluation. The backbone factory currently supports ResNet and EfficientNet; transformer support remains planned. Inspect actual run configs before assuming default settings.
- Entry points: train.py and evaluate.py. Relevant code: models/, losses/, training/, data/, evaluation/, and config/. Historical artifacts are documented in README.md; verify their presence and provenance on this HPC before using them.
- Historical README results suggest RL gains, but they are preliminary evidence pending estimator/evaluator audits and clean checkpoint selection. Do not present them as newly verified results or as state of the art.
- Compute: user has Alex A100 access with a limited, unspecified allocation. Exact available GPU-hours, GPU memory variant, time limits, transferred datasets/checkpoints, and environment readiness remain to be established. A40 access is a possibility, not confirmed account access.
- Timeline: user requested a three-week experiment sprint starting 2026-10-08, with a CVPR submission ambition. Planning target is CVPR 2027; previously checked registration/paper deadlines were November 10/16, 2026 AoE. Reverify official dates and requirements when preparing submission; publication/submission is a separate authorized action.

## Staged roadmap and planned updates

These are proposed implementation and experiment stages, not completed work. Research stages below are distinct from the model's Stage 1/Stage 2 training. Adapt scope to measured cost and evidence; do not launch an exhaustive grid merely because it appears here.

### Research stage A — HPC readiness and data inventory

Status: largely complete as of 2026-10-08; remaining items listed below.

- Done (evidence in `scripts/` and Slurm logs `logs/*-4504546/4504574/4504625/4504653`): checkpoints `stage1_epoch_13_best.pth` and `stage2_epoch_5.pth` copied from tinyx (byte sizes and sha256 match the tinyx originals, verified 2026-10-09); packed archives for SfM-120k, ROxford5k, RParis6k downloaded to `$HPCVAULT/datasets` (size-verified, sha256 in `datasets/logs/`); archive contents checked against the metadata pickles; node-local TMPDIR staging validated for all three datasets; venv `~/venvs/rl-env` (python/3.12-base, pinned `requirements.txt`) created; unmodified `evaluate.py` ran end to end on 1 GPU (pipeline smoke only, not a research result).
- Open: run folders `stage1_20260405_073610`, `stage2_20260417_231413`, `stage2_20260406_173128` not yet copied; old job scripts (`train_job.sh` etc.) still reference conda `rl-env` and are not usable; train.py not yet smoke-tested; GPU-hour allocation still unknown.

- Inventory transferred checkpoints, manifests, packed datasets, and environments using lightweight reads. Resolve old absolute paths to run-specific Alex paths.
- Adapt job scripts for current Slurm syntax, environment setup, node-local staging, persistent outputs, and resume support. Check quota and account limits before substantial data operations.
- Exit evidence: validated scripts/configs, identified artifacts, and a bounded allocated GPU/data-loading smoke run. Do not mark readiness complete from shell syntax checks alone.

### Research stage B — Correctness and reproducible baseline

Status: initial inspection found issues; fixes and experiments have not been performed.

- Audit normalized-Gaussian sampling versus the current cosine/von-Mises–Fisher-style proxy density, sampled-action detachment for score-function gradients, baseline independence, and stochastic gallery gradient treatment. Validate a consistent estimator on a tractable small problem.
- Replace or reconcile the current precision-at-positive AP calculation with official Revisited scoring; verify crops, junk/empty-positive handling, and metric definitions.
- Replace ROxford-driven checkpoint selection for final generalization claims with held-out training-domain validation. Check identity/duplicate leakage and positive availability after query/gallery splitting.
- Re-evaluate available historical checkpoints under the corrected protocol and establish a ResNet50 baseline. Separate estimator corrections from any proposed research novelty.
- Exit evidence: meaningful correctness checks, documented official protocol, clean checkpoint-selection rule, and reproducible baseline artifacts.

### Research stage C — Strong backbone comparisons

Status: planned; preferred first transformer is DINOv2 ViT-B/14, with DINOv3 ViT-B/16 conditional on access, pilot evidence, and compute.

- Add explicit transformer feature extraction/aggregation and compatible preprocessing; record pretrained weights, resolution, descriptor size, and trainable layers.
- Begin with frozen-backbone head adaptation; test last-block adaptation on promising candidates before expensive full fine-tuning.
- For each selected initialization compare Stage 1, matched-budget continued metric learning, and corrected reward tuning. Screen with one seed and repeat key comparisons across three seeds where budget allows.
- Include accessible published landmark retrieval baselines, distinguishing global retrieval from reranking and data/compute differences. SALAD/place-recognition experiments are an optional additional track.
- Exit evidence: comparable measured gains with compute records and limits, not simply scores from larger pretrained models.

### Research stage D — Focused methodological improvement

Status: hypothesis selection pending pilots and literature review; novelty is not established.

- Investigate whether a gallery representative of difficult large-scale retrieval improves task-reward tuning. Candidate mechanisms include hard-negative selection, memory freshness, and reward/baseline variance reduction.
- Choose one mechanism supported by diagnostics and prior-work analysis. Existing memory banks and consistency regularization alone are not new contributions.
- Run targeted ablations, including a consistency-only/zero-policy-weight control where useful, to isolate the source of improvements.
- Exit evidence: repeatable improvement over strong matched-budget controls, supported mechanism, and explicit failure cases.

### Research stage E — Dataset scale and transfer

Status: planned; data availability/storage feasibility not yet verified.

- Keep SfM-120k as the controlled setting. Add GLDv2-clean training, initially via a fixed documented subset if needed, and scale only after cost estimation.
- Evaluate the strongest models on ROxford/RParis with R1M distractors. Add official GLDv2 retrieval evaluation if preparation fits the budget.
- Use official relevance/metrics and split rules; do not equate GLDv2 retrieval relevance automatically with class equality. Plan chunked extraction/search and descriptor storage, measuring approximation effects if used.
- Exit evidence: verified scale/generalization results with dataset manifests and extraction/ranking cost.

### Research stage F — Results freeze and manuscript preparation

Status: planned. Start method/protocol writing while experiments proceed.

- Consolidate exact configs, revisions, data manifests, seeds, selected checkpoints, metrics, logs, and GPU-hours. Report absolute mAP-point gains, variability, inference/storage cost, and limitations.
- Generate tables/figures from actual artifacts; distinguish local results from cited published results. Verify citations, contribution claims, and submission requirements.
- Preserve final artifacts in backed-up storage within quota. Paper submission or public release requires user authorization; preparing reviewable materials does not.

Sprint scheduling target: week 1 (Oct 8–14) readiness/correctness/baseline and transformer pilots; week 2 (Oct 15–21) primary comparisons and one improvement, beginning scale work; week 3 (Oct 22–28) repeats, scale evaluation, focused ablations, and results freeze. Stages may overlap when independent work is useful.

## Maintaining project state

Update stage statuses only when supported by completed work and link the evidence/artifacts. Keep detailed experiment records with their run results; this roadmap tracks milestones, decisions, and evidence links. Preserve unknowns, unsuccessful runs, and user budget constraints across sessions. Roadmap entries do not independently authorize spending an unknown allocation, bulk deletion, external messages, submission, or publication.

