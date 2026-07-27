# Force-Aware Whole-Body Control for TALOS

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![mjlab](https://img.shields.io/badge/mjlab-1.5.0-76B900.svg)](https://mujocolab.github.io/mjlab/)

Research project for force-aware whole-body control and learning workflows for
the PAL Robotics TALOS humanoid.

The current MJLab implementation trains TALOS to track planar velocity while
carrying a tray with a free payload. The tray is mounted to both wrists and the
payload is governed by contact dynamics rather than a fixed joint. The policy
uses a fixed, hardware-deployable sensor contract and force-aware transport
objectives.

The 153-dimensional actor input is identical across all TALOS tasks. It contains
joint encoder position and velocity, IMU/state-estimator channels, the previous
action and velocity command, 24 instrumented joint-torque measurements, and the
four wrist/ankle six-axis F/T measurements available on the real robot.
Unobservable payload position, velocity, and mass are restricted to the critic.
Actor sensor channels include noise, saturation, and bounded latency during
training.

This repository is derived from
[PAL Robotics' pal_mjlab](https://github.com/pal-robotics/pal_mjlab). The
official repository is retained locally as the `upstream` Git remote.

## Repository layout

- `src/pal_mjlab/`: robots, task definitions, observations, rewards, and scripts
- `tests/`: regression tests for TALOS assets and task configurations
- `third_party/Shared-IAS/`: pinned IAS Cluster documentation and shared tooling
- `uv.lock`: reproducible Python and simulator dependency lock

TALOS-specific source code, robot assets, controllers, experiment
configurations, and tests belong here. Reusable TU Darmstadt IAS Cluster
documentation and utilities belong in
[Shared-IAS](https://github.com/Hazzzard-LYX/Shared-IAS).

Datasets, simulator caches, virtual environments, logs, checkpoints, generated
assets, and training results are intentionally excluded from Git. Important
checkpoints and results must be backed up separately.

## Clone

```bash
git clone --recurse-submodules \
  https://github.com/Hazzzard-LYX/MJ-LAB-WBC-FOR-FORCE-AWARE-TALOS.git
cd MJ-LAB-WBC-FOR-FORCE-AWARE-TALOS
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

## Local MJLab environment

Install `uv`, then create the locked Python 3.12 environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --python 3.12
```

Always run training and playback through `uv run` or this repository's
`.venv/bin`. Reusing an unrelated Conda or pip environment can make
single-environment playback work while batched MuJoCo-Warp simulation fails.

List the registered PAL environments:

```bash
uv run list_envs --keyword pal
```

Run the tray task with a locally stored checkpoint:

```bash
uv run play Mjlab-Velocity-Flat-Pal-Talos-Free-Payload-Tray \
  --checkpoint-file /path/to/model.pt \
  --num-envs 1 \
  --device cuda:0 \
  --viewer native
```

Start a headless training run:

```bash
uv run train Mjlab-Velocity-Flat-Pal-Talos-Free-Payload-Tray \
  --env.scene.num-envs 1024
```

## IAS Cluster

Shared cluster tooling is pinned as the `third_party/Shared-IAS` submodule.
That repository currently documents rootless Podman, SLURM, SSH forwarding,
TensorBoard access, and a validated Isaac Lab 2.3.1 / Isaac Sim 5.1.0 smoke
test. Project-specific MJLab job scripts and experiment configurations remain
in this repository.

All simulation and training on the IAS Cluster must run on a SLURM compute
node. The `mn` login node is only for lightweight repository, file, and job
submission operations.

The reproducible MJLab image and finite GPU validation workflow are documented
in [`containers/mjlab/README.md`](containers/mjlab/README.md).

## Git workflow

- `origin`: this project repository under `Hazzzard-LYX`
- `upstream`: PAL Robotics' official `pal_mjlab`
- `main`: reviewed and reproducible project baseline
- `feat/*`, `fix/*`, `exp/*`: short-lived implementation and experiment branches

Each training run should record the exact Git commit, task name, seed,
environment count, command line, and parent checkpoint. Code and configuration
move through GitHub; large runtime artifacts do not.

## Acknowledgements

This work builds on
[PAL Robotics](https://pal-robotics.com/),
[mjlab](https://github.com/mujocolab/mjlab),
[MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html), and
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html).

See [LICENSE](LICENSE) for licensing details inherited from the upstream
project.
