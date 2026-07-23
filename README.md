# Force-Aware Whole-Body Control for TALOS

Research project for force-aware whole-body control and learning workflows for
the TALOS humanoid robot.

## Environment

The initial cluster environment has been validated with:

- Isaac Lab `2.3.1`
- Isaac Sim `5.1.0`
- NVIDIA RTX 3080 Ti
- IAS Cluster rootless Podman and SLURM

Shared cluster tooling and reusable infrastructure are pinned as a Git
submodule under `third_party/Shared-IAS`.

## Clone

```bash
git clone --recurse-submodules \
  https://github.com/Hazzzard-LYX/MJ-LAB-WBC-FOR-FORCE-AWARE-TALOS.git
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

## Repository boundaries

This repository is intended for TALOS-specific source code, robot assets,
controllers, task definitions, experiment configurations, and tests.

Datasets, simulator caches, logs, checkpoints, generated USD files, and
training results are intentionally excluded from Git.
