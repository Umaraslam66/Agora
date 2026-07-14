# jobs/ — cluster job files

Rules for anything submitted to a cluster from this directory:

- **Full-node discipline.** Billing is per node: every GPU job must use the
  whole allocated node — multi-GPU training via DDP/`torchrun`, or packed
  parallel runs (multiple independent runs sharing the node). No job may
  idle GPUs on a node it holds.
- **No usernames.** Job files contain no personal usernames, emails, home
  paths, or account strings; use environment variables and cluster-provided
  variables for accounts, scratch paths, and partitions.
- **Cluster-agnostic naming.** Name scripts by what they do
  (`train_lora_sft.sbatch`, `serve_gateway.sbatch`), not by cluster,
  allocation, or person. Cluster-specific values (partition, account, module
  loads) belong in a small per-site env file, not hardcoded in each script.
- Job outputs land under `runs/` with a manifest; any claim made from a run
  must be reproducible from those files.
