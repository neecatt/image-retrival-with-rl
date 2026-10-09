---
name: research-literature
description: Review papers and official implementations for landmark retrieval and task-reward learning, build sourced comparison matrices, and assess experimental gaps and contribution claims. Use for research positioning and baseline selection, not ordinary code edits.
---

# Research literature

Turn a research question into an evidence-backed comparison and a feasible experiment. Prioritize original papers, official author repositories, benchmark documentation, and released checkpoints. Search current sources for recent methods or deadlines; do not treat leaderboards or search snippets as verified evidence.

## Read and compare

Read the requested paper's method, objective, experimental setup, limitations, and relevant supplement. For implementation questions inspect official code and checkpoint instructions. Record paper version/date and code revision when differences affect claims. Separate author-reported results, independently reproduced results, and this project's measured results.

For this project, the starting paper is https://arxiv.org/abs/2302.08242. Verify details before describing them: its probabilistic task-reward framework is the source idea, while continuous landmark embedding policies, metric-learning pretraining, and this project's regularization are adaptations to assess independently.

Use [comparison-matrix.md](assets/comparison-matrix.md) for baseline selection or related-work synthesis. Include only relevant methods; unknown implementation/data details remain unknown. Distinguish landmark instance retrieval, place recognition, and semantic retrieval, as well as global-descriptor retrieval and reranking. A stronger backbone is not itself evidence of a stronger learning method.

## Identify a testable gap

For each proposed contribution identify the closest prior work, the actual unresolved limitation, the mechanism proposed to address it, the simplest falsifying experiment, and the fair control. Search both retrieval literature and direct AP/ranking optimization or RL literature before novelty claims. Look for existing memory-bank, mining, variance-reduction, and parameter-efficient adaptation methods rather than relabeling familiar components.

Prefer accessible baselines with runnable code/checkpoints under the user's compute and data budget. Record checkpoint licenses and dataset access terms when they affect use or release; avoid broad speculative compliance reviews. Check current venue deadlines and author requirements using the official conference site when scheduling submission work.

## Deliver evidence

Provide a compact sourced matrix, what is established versus inferred, comparison caveats, and prioritized experiments. Link claims to the supporting paper section or official implementation; maintain BibTeX when a manuscript workflow needs it. Do not invent citations, claim exhaustive novelty searches, promise acceptance, or report unrun experiments. Document failed reproductions and incompatibilities without substituting published scores for local measurements.
