---
name: alex-hpc-experiment
description: Prepare, run, resume, and diagnose Slurm experiments on the NHR FAU Alex cluster, including dataset staging, quota checks, Python environments, and GPU utilization. Use for this project's HPC execution rather than generic model design.
---

# Alex HPC experiments

Apply the repository AGENTS.md as the source of operating constraints. This skill supplies the experiment workflow, not permission to submit unrelated jobs, consume unspecified allocations, delete unique data, or change system software.

## Prepare a concrete run

Inspect the requested experiment and existing scripts. Resolve live WORK/HPCVAULT paths, byte and inode quotas, available modules, scheduler GPU syntax, account/QOS access, and runtime limits. Use current official NHR documentation when syntax or policy is uncertain; historical slides and observed quotas are not live guarantees.

For script edits alone, prepare and validate without submission. When running the experiment is authorized, proceed within the requested budget; if a material budget is missing, prepare the script and collect read-only facts before asking for the required constraint. Begin with a bounded pilot when appropriate.

Use [job-review.md](references/job-review.md) when creating or modifying a batch script. Do not paste a generic runnable template with guessed partition, module, or environment names. Validate shell syntax with bash -n; this does not validate Slurm directives. Where supported, use sbatch --test-only to check scheduler acceptance without enqueueing a job.

## Stage and execute

Keep shared-storage images packed. Inside the compute allocation, resolve and check job-local TMPDIR, available bytes/inodes, archive integrity, and safe member paths before extraction. Stage only required subsets/shards. Point a run-specific config at staged data and persistent output paths without overwriting the canonical config. Never overwrite HOME, WORK, or TMPDIR environment variables.

Record the actual interpreter, modules, CUDA/PyTorch versions, configuration, data manifest, and Slurm job ID. Verify GPU availability and a small real batch in the allocation; set worker/thread counts within allocated CPUs. Use unique run directories and periodic persistent checkpoints with optimizer, scheduler, scaler, RNG, and other state required by the implemented resume path.

## Monitor and recover

Inspect the user's relevant jobs and logs at moderate intervals during active work. Check throughput, GPU utilization/memory, CPU use, data-loading delays, finite losses, and reward/metric validity. Low GPU use can reflect staging or validation; diagnose the phase before stopping a job. Record why a failed or wasteful task-owned job is stopped, preserving useful artifacts first where feasible.

After completion, inspect sacct exit state and metrics, verify required checkpoints/results exist in persistent storage, and estimate GPU-hours as elapsed allocated hours times GPU count (distinguish this estimate from account billing). Record failure causes instead of silently treating partial output as success. For timeout/OOM, adapt batch size or resume strategy based on evidence. Do not promise monitoring beyond this session without an actual mechanism.

References: https://doc.nhr.fau.de/data/staging/ and https://doc.nhr.fau.de/batch-processing/batch_system_slurm/
