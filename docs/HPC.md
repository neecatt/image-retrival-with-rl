# Alex HPC operations

Follow the essential constraints in [AGENTS.md](../AGENTS.md). Read the relevant sections here when preparing environments, moving data, submitting jobs, or monitoring experiments.

## Storage and data handling

The user requires compliance with NHR@FAU storage and staging rules.
Reference: https://doc.nhr.fau.de/data/staging/
Reference: https://doc.nhr.fau.de/data/copying/

- Keep source, small metadata, and important compact results in HOME.
- Store bulk datasets as archives or container files (e.g. HDF5 or tar shards) in WORK or HPCVAULT. Never unpack large small-file datasets into HOME, HPCVAULT, or WORK.
- Extract image archives only into the compute job's node-local TMPDIR and train/evaluate from there. Require TMPDIR to be set; do not silently fall back to the repository or shared storage. Validate available local space before extraction.
- TMPDIR is temporary and job-scoped. Save checkpoints periodically to persistent storage, and copy required outputs out before job termination. Do not rely solely on an exit trap to preserve results after a hard timeout or node failure.
- WORK is not backed up. Preserve selected final checkpoints and important results in backed-up HOME/HPCVAULT, subject to quota.
- Check both byte and inode quotas before large transfers, downloads, or extraction. Use shownicerquota.pl; df alone does not show account quota. Respect group quota on WORK and stay below soft limits.
- Resolve storage through HOME, HPCVAULT, WORK, and TMPDIR; do not assume old-cluster paths or change these variables. FASTTMP is not automatically available on Alex.
- Prefer resumable rsync for external archive transfers; use the documented transfer endpoint csnhr.nhr.fau.de. Verify archive integrity and extraction paths. Avoid compression for already-compressed data unless measured useful.
- Avoid repeated recursive scans of large shared-storage image trees. Use manifests and packed formats to reduce metadata load.
- Existing job scripts have not yet been adapted to this policy. Do not submit them unchanged for bulk dataset jobs.

## Frontends and compute jobs

- Frontends are shared: use them for editing, lightweight inspection, input preparation, and submission. Run training, embedding extraction, substantial evaluation/preprocessing, heavy compilation, and GPU work on compute nodes through Slurm.
- Never run MPI jobs on frontends. Use an interactive Slurm allocation for compute-intensive debugging, tests, and builds on the target cluster, especially CUDA extensions.
- Understand commands and every directive before reusing scripts. Never use sudo, apt installation, or system-wide make install. Adapt upstream instructions to the cluster environment.
- Verify current Alex GPU request syntax, partitions, account/QOS access, and time limits before submitting. Slide examples and a typical 24-hour runtime are not guaranteed limits for this account.
- Request explicit GPU type/count, CPUs, memory, and realistic walltime. Begin with a short one-GPU pilot; measure throughput, memory, and scaling before requesting more GPUs or launching many experiments. Respect the user's limited compute allocation.
- Put all #SBATCH directives before the first executable shell statement (including set -euo pipefail). Create Slurm log directories before submission, since Slurm opens output files before executing the script.
- Use a login shell where required, #SBATCH --export=NONE, and unset SLURM_EXPORT_ENV as in the Alex examples. Explicitly load modules and activate the environment inside every job.
- Quote paths, use explicit project/data/output paths, and preserve access to source code when changing into TMPDIR. Do not assume train.py is present in an extracted dataset directory.
- Match DataLoader workers and CPU thread counts to allocated CPUs. Use the documented Slurm launch method for the workload; distributed execution requires actual multi-process training support.
- Include staging and checkpoint copy time in walltime estimates. Support resuming interrupted experiments and stop early on invalid metrics, non-finite losses, or configuration errors.

## Modules and Python environments

- Discover installed versions with module avail (or module avail python); inspect module help/show before choosing dependencies. Use module list to record loaded versions. Module load/unload changes only the current shell.
- Prefer the cluster-provided Python/Conda module and an isolated project environment. Install packages into that environment with its Python (python -m pip), never into system Python or a shared base environment. Do not mix pip --user packages into a Conda environment; use --user only for a deliberately chosen standalone user installation.
- Place large environments and package/model caches in WORK where appropriate, checking inode quotas first. Keep versions reproducible and avoid unnecessary duplicate environments.
- Configure external-network proxies only according to cluster documentation. Never copy credentials into scripts or logs.
- Check PyTorch/CUDA/driver compatibility and verify GPU availability inside an allocated job before a long experiment.

## Monitoring and resource stewardship

- Check submitted jobs using squeue, scontrol show job, and sacct at reasonable intervals; avoid tight polling loops. Record job IDs, configurations, seeds, environment versions, elapsed time, and resource usage with results.
- During active experiment work, inspect logs, metric validity, throughput, CPU/memory use, and GPU utilization. Use nvidia-smi within the allocated job or the documented method for attaching to one's own running job; do not start unallocated work on compute nodes.
- Investigate idle GPUs and data-loading bottlenecks before scaling. Stop one's own failed or wasteful experiment jobs when necessary; never cancel unrelated jobs or touch other users' processes.
- Use available cluster monitoring (ClusterCockpit) where accessible. Do not promise continuous monitoring outside an active session unless an actual monitoring mechanism has been configured.
- Minimize tiny, frequent I/O on shared filesystems; buffer logs and retain useful checkpoints rather than every intermediate artifact. Remove only confirmed disposable task-owned scratch data; preserve unique datasets and experiment results.
- Consult official application examples or experienced cluster support when behavior is unclear. Do not send messages to support or colleagues without user authorization.

## Job and environment references

- Modules: https://doc.nhr.fau.de/environment/modules/
- Python environments: https://doc.nhr.fau.de/environment/python-env/
- Slurm and attaching to a running job: https://doc.nhr.fau.de/batch-processing/batch_system_slurm/
- Monitoring: https://doc.nhr.fau.de/job-monitoring-with-clustercockpit/
- Application examples: https://doc.nhr.fau.de/apps/overview/
- These rules incorporate the user's supplied NHR@FAU slides. Verify live documentation before relying on specific commands, module versions, or scheduler limits.

## Observed storage context (2026-10-08)

- HOME: /home/hpc/b268dc/b268dc21
- HPCVAULT: /home/vault/b268dc/b268dc21
- WORK: /home/atuin/b268dc/b268dc21
- TMPDIR and FASTTMP were unset in the inspected session; resolve TMPDIR inside the scheduled compute job.
- WORK quota is shared by the group. Recheck live usage before each large operation; slide example quotas are not account-specific.

