# MJLab container

This image reproduces the locked CUDA-enabled MJLab environment used for the
TALOS tray-transport task. It is built with rootless Podman on the IAS Cluster
and remains compatible with Docker/OCI tooling.

The image contains project source code and Python dependencies. Runtime logs,
checkpoints, and the Warp kernel cache are bind-mounted from the host and are
not stored in image layers.

## Build

Build only from a clean Git commit:

```bash
cd ~/IAS_Workspace/projects/WBC-For-Talos
bash containers/mjlab/build-image.sh
```

The immutable image tag is the first 12 characters of the current Git commit:

```text
localhost/force-aware-talos-mjlab:<git-sha>
```

Image preparation is allowed on the IAS login node, but training and simulation
are not. The Containerfile performs only dependency installation, package
validation, and CPU tests while building.

On a busy login node, build through SLURM to give the dependency resolver a
bounded 32 GiB memory allocation:

```bash
mkdir -p ~/IAS_Workspace/logs
sbatch slurm/mjlab-build-image.sbatch
```

## GPU validation

Create the SLURM output directory before submitting:

```bash
mkdir -p ~/IAS_Workspace/logs
sbatch slurm/mjlab-smoke-test.sbatch
```

The finite smoke job performs:

1. CUDA visibility and exact package-version checks.
2. The CPU unit-test suite.
3. One PPO iteration with 64 environments.
4. One PPO iteration with 1024 environments.

Inspect it with:

```bash
squeue -u "$USER"
ias-job-nvidia-smi JOBID
tail -f ~/IAS_Workspace/logs/mjlab-smoke-JOBID.out
```

The script derives the image tag from the project checkout. Override it only
when intentionally validating another image:

```bash
sbatch --export=ALL,MJLAB_IMAGE=localhost/force-aware-talos-mjlab:TAG \
  slurm/mjlab-smoke-test.sbatch
```
