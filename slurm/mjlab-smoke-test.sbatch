#!/usr/bin/env bash
#SBATCH --job-name=mjlab-smoke
#SBATCH --output=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.out
#SBATCH --error=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=stud
#SBATCH --gres=gpu:rtx3080:1

set -euo pipefail

WORKSPACE="${HOME}/IAS_Workspace"
PROJECT_ROOT="${WORKSPACE}/projects/WBC-For-Talos"
MJLAB_ENV_ROOT="${MJLAB_ENV_ROOT:-${WORKSPACE}/envs/mjlab-cu128}"
PYTHON="${MJLAB_ENV_ROOT}/bin/python"
SHORT_REVISION="$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD)"
RUN_ROOT="${WORKSPACE}/logs/mjlab-smoke/${SLURM_JOB_ID}"
CACHE_ROOT="${WORKSPACE}/shared/mjlab-cache"

export TMPDIR="${SLURM_TMPDIR:-${WORKSPACE}/tmp/${SLURM_JOB_ID}}"
export WARP_CACHE_PATH="${CACHE_ROOT}/warp"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p \
  "${TMPDIR}" \
  "${RUN_ROOT}" \
  "${WARP_CACHE_PATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
echo "MJLab environment: ${MJLAB_ENV_ROOT}"
nvidia-smi

test -x "${PYTHON}"
cd "${PROJECT_ROOT}"
"${PYTHON}" containers/mjlab/smoke.py
MUJOCO_GL=disable FORCE_CPU=1 "${PYTHON}" -m pytest -q

for TASK in \
  Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-Critic \
  Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-Estimator \
  Mjlab-Velocity-Flat-Pal-Talos-Tray-Random-Mass-Oracle \
  Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp \
  Mjlab-Velocity-Flat-Pal-Talos-Tray-Contact-Grasp-Random-Mass
do
  TASK_SLUG="${TASK##*-}"
  "${PYTHON}" -m mjlab.scripts.train \
  "${TASK}" \
  --env.scene.num-envs 64 \
  --agent.logger tensorboard \
  --agent.max-iterations 1 \
  --agent.save-interval 1 \
  --agent.run-name "native-smoke-${TASK_SLUG}" \
  --log-root "${RUN_ROOT}/${TASK_SLUG}"
done

echo "MJLAB_GPU_SMOKE_OK"
echo "End: $(date --iso-8601=seconds)"
