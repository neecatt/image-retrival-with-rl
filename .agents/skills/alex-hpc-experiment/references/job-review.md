# Batch script review

- Shebang and all SBATCH directives precede executable shell statements. Confirm live GPU type/count, partition/account if needed, CPU/memory request, walltime, job name, and unique log paths.
- Submission creates log directories first and uses a known project working directory. Paths with spaces are quoted.
- Environment export follows Alex guidance: --export=NONE and unset SLURM_EXPORT_ENV; required modules and the project environment are explicitly activated. Do not assume interactive-shell state persists.
- Source/config/checkpoint paths remain valid after staging or changing directory. Arguments match the actual Python entry point; do not invent --dataset-path support.
- Data remains packed on shared storage; extraction happens only in allocated TMPDIR after space and archive-path checks. Archives with absolute paths, parent traversal, or escaping links are rejected.
- A unique run config resolves staged input and persistent output paths. Outputs and model caches do not accidentally default into a full HOME directory.
- Resume support preserves state actually required by the trainer. Checkpoints are saved periodically, with atomic replacement when feasible, not solely at job exit.
- Walltime allows staging, initialization, evaluation, and copying results. If signals/traps are used, confirm trainer handling; SIGKILL and node loss cannot be recovered by traps.
- Multi-GPU requests have a matching supported distributed launch and sampler. Worker/thread counts fit CPU allocation.
- Record bash -n result and distinguish syntax validation, scheduler validation, pilot execution, and full completion.
