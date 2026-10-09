# Landmark retrieval with RL

## Project orientation

This project adapts the task-reward tuning framework from [Tuning computer vision models with task rewards](https://arxiv.org/abs/2302.08242) to landmark retrieval. The current pipeline uses ResNet50 + GeM, FastAP pretraining (Stage 1), and REINFORCE-style AP reward fine-tuning (Stage 2), trained on SfM-120k and evaluated on ROxford5k/RParis6k.

Entry points are `train.py` and `evaluate.py`; implementation lives in `models/`, `training/`, `losses/`, `data/`, and `evaluation/`. Inspect actual run configs before assuming defaults. The research objective is to compare stronger backbones, establish reliable reward-tuning gains, and test larger datasets within a limited Alex GPU allocation.

## Essential operating constraints

- Run heavy preprocessing, training, evaluation, GPU work, and substantial builds through Slurm compute allocations. Frontends are for lightweight preparation; never run MPI there.
- Keep bulk image datasets packed on shared storage. Never unpack large small-file datasets into HOME, HPCVAULT, or WORK. Extract only into the allocated job's node-local TMPDIR; fail if it is unavailable.
- Check byte and inode quotas before large operations, including shared group limits. Preserve persistent checkpoints periodically: TMPDIR is temporary and WORK has no backup.
- Respect the user's limited compute budget. Measure a bounded pilot before scaling; verify live scheduler/account limits and environment availability. Roadmap entries alone do not authorize an unspecified compute spend.
- Use cluster modules and isolated project environments; never use sudo or install into system/shared base Python. Resolve paths from the actual environment rather than old-cluster absolute paths.
- Preserve unique data and experiment artifacts. Operate only on relevant task-owned jobs; external messages, submission, and publication need user authorization unless already provided.

## Research correctness

- Distinguish the original framework, this project's retrieval adaptations, and demonstrated new contributions. Historical README gains remain preliminary until verified from artifacts under a validated protocol.
- Validate the estimator and official benchmark scoring before large experiments. Select checkpoints on held-out validation, not final test benchmarks.
- Compare reward tuning against continued metric learning from the same initialization with a documented matched budget. Record configs, code state, manifests, seeds, checkpoints, metrics, and compute; do not fabricate missing results.

## Read documents when relevant

- [README.md](README.md): architecture, usage, and historical artifact references.
- [docs/PROGRESS.md](docs/PROGRESS.md): dated log of problems found, fixes and evidence; open-problem table. Update it after audits or runs.
- [docs/ROADMAP.md](docs/ROADMAP.md): research stages, current status, unresolved questions, schedule, and completion evidence. Read when planning experiments or updating project progress; update supported statuses and evidence after milestone work.
- [docs/HPC.md](docs/HPC.md): detailed storage, staging, environment, Slurm, monitoring, and recovery rules. Read relevant sections before those operations. Existing job scripts have not yet been adapted; do not submit them unchanged for bulk-data runs.
- Keep detailed run records with results. Do not require reading the full roadmap or HPC guide for unrelated small edits.

## Project skills

Read the relevant workflow when the task benefits from it:

- [.agents/skills/retrieval-experiment/SKILL.md](.agents/skills/retrieval-experiment/SKILL.md): experimental design, benchmark correctness, RL audits, and backbone/ablation comparisons.
- [.agents/skills/alex-hpc-experiment/SKILL.md](.agents/skills/alex-hpc-experiment/SKILL.md): preparing, executing, resuming, and diagnosing Alex Slurm experiments.
- [.agents/skills/research-literature/SKILL.md](.agents/skills/research-literature/SKILL.md): sourced paper/implementation comparisons and contribution assessment.

Skills support the requested workflow and do not expand authorization or budgets.
