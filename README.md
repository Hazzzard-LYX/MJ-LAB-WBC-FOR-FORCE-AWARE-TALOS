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

The base 153-dimensional actor input is identical across deployable TALOS tasks.
It contains joint encoder position and velocity, IMU/state-estimator channels,
the previous action and velocity command, 24 instrumented joint-torque
measurements, and the four wrist/ankle six-axis F/T measurements available on
the real robot. Unobservable payload position and velocity are restricted to
the critic. Actor sensor channels include noise, saturation, and bounded
latency during training.

## Payload-mass experiments

The free cube is a standalone six-DoF body. It can slide, tip, and fall from the
tray under contact dynamics. Four registered tasks isolate how payload mass is
provided to the policy:

| Task suffix | Actor payload input | Critic payload input | Joint torque |
| --- | --- | --- | --- |
| `Free-Payload-Tray-Fixed-2p5kg` | none | true state | actor + critic |
| `Tray-Random-Mass-Critic` | none | true state | actor + critic |
| `Tray-Random-Mass-Estimator` | estimated mass | true state | actor + critic + estimator |
| `Tray-Random-Mass-State-Estimator` | estimated mass + position | same estimate | estimator only |
| `Tray-Random-Mass-Oracle` | true mass | true state | actor + critic |

The estimator task uses an eight-step history made only from deployable
proprioception: encoder state, previous action, IMU channels, instrumented joint
torques, and both wrist F/T sensors. An auxiliary supervised loss predicts the
simulator payload mass. The simulator target is not part of the Actor
observation. Its ONNX export has two inputs (`actor_obs` and
`proprioceptive_history`) and returns both `actions` and
`estimated_payload_mass_kg`.

The state-estimator variant makes joint torque private to the eight-step
estimator history. Actor and critic receive neither torque nor simulator
payload state; both are conditioned on the same estimated mass and three-axis
payload COM position in the tray frame. True mass and position are used only by
the supervised estimator objective. Its export additionally returns
`estimated_payload_position_t_m`.

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

Train the three random-mass comparisons:

```bash
uv run train Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-Critic \
  --env.scene.num-envs 1024
uv run train Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-Estimator \
  --env.scene.num-envs 1024
uv run train Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-State-Estimator \
  --env.scene.num-envs 1024
uv run train Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-Oracle \
  --env.scene.num-envs 1024
```

Train TALOS with the tray held only by rigid high-friction hand contacts (no
weld or fixed tray joint).  Start with the fixed 2.5 kg task and its staged
standing/walking command curriculum, then use the random-mass task once
bilateral grasping and walking are stable:

```bash
uv run train Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp \
  --env.scene.num-envs 1024
uv run train Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp-Random-Mass \
  --env.scene.num-envs 1024
uv run train \
  Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp-Oracle-2p5-15kg \
  --env.scene.num-envs 1024
uv run train \
  Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp-Random-Mass-State-Estimator \
  --env.scene.num-envs 1024
```

The `Oracle-2p5-15kg` task keeps the tray completely free and exposes only the
normalized simulator payload mass as one additional actor input.  It is a
privileged behavior teacher for collecting stable wrist-F/T identification
rollouts, not a deployable policy.  Initialize it from the fixed-2.5 kg grasp
checkpoint with `slurm/prepare_contact_grasp_oracle_checkpoint.py`; the new
input is appended with a zero first-layer weight so the initial policy exactly
preserves the learned light-payload behavior.

The actor observes the two real commanded gripper encoders plus the existing
deployable TALOS proprioception and force/torque channels.  Tray pose, handle
slip, contacts, and payload truth are critic-only signals.  The restored
three-finger linkage retains one commanded joint per hand and mechanically
couples all passive finger joints.

The contact-grasp state-estimator variant keeps the tray free and preserves the
same rigid high-friction contacts.  Joint torque is private to an eight-step
hardware-history estimator.  Its predicted payload mass and tray-frame payload
position condition both actor and critic, while payload truth is used only as
the estimator's supervised training target.

To visualize an early checkpoint from a cluster training job, submit
`slurm/mjlab-contact-grasp-livestream.sbatch` with `TRAIN_JOB_ID` and the
12-character `TRAIN_REVISION`.  It waits for `model_100.pt` by default and then
serves four environments through Viser on remote port 18080.

## IAS Cluster

Shared cluster tooling is pinned as the `third_party/Shared-IAS` submodule.
That repository currently documents rootless Podman, SLURM, SSH forwarding,
TensorBoard access, and a validated Isaac Lab 2.3.1 / Isaac Sim 5.1.0 smoke
test. Project-specific MJLab job scripts and experiment configurations remain
in this repository.

All simulation and training on the IAS Cluster must run on a SLURM compute
node. The `mn` login node is only for lightweight repository, file, and job
submission operations. The project SLURM scripts use the validated shared
environment at `~/IAS_Workspace/envs/mjlab-cu128` by default and execute the
checked-out source through `PYTHONPATH`; no OCI import or image rebuild is
required for routine code changes.

Submit a random-mass experiment with:

```bash
sbatch --export=ALL,NUM_ENVS=2816,\
TASK=Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-Critic \
  slurm/mjlab-train.sbatch
```

Set `MJLAB_ENV_ROOT` only when deliberately using a different tested
environment. The container recipe in
[`containers/mjlab/README.md`](containers/mjlab/README.md) remains available
for isolated reproducibility work.

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
