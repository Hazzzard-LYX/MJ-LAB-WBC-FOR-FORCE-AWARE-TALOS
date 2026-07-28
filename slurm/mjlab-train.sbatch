#!/usr/bin/env bash
#SBATCH --job-name=mjlab-train
#SBATCH --output=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.out
#SBATCH --error=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=stud
#SBATCH --gres=gpu:rtx3080:1

set -euo pipefail

WORKSPACE="${HOME}/IAS_Workspace"
PROJECT_ROOT="${WORKSPACE}/projects/WBC-For-Talos"
MJLAB_ENV_ROOT="${MJLAB_ENV_ROOT:-${WORKSPACE}/envs/mjlab-cu128}"
PYTHON="${MJLAB_ENV_ROOT}/bin/python"
SHORT_REVISION="$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD)"
TASK="${TASK:-Mjlab-Velocity-Flat-Pal-Talos-Free-Payload-Tray}"
NUM_ENVS="${NUM_ENVS:?Submit with --export=ALL,NUM_ENVS=<count>}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-${SHORT_REVISION}-${NUM_ENVS}env}"
RUN_ROOT="${WORKSPACE}/logs/mjlab-training/${SHORT_REVISION}/${SLURM_JOB_ID}"
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
echo "Task: ${TASK}"
echo "Environments: ${NUM_ENVS}"
echo "Iterations: ${MAX_ITERATIONS}"
echo "Seed: ${SEED}"
nvidia-smi

test -x "${PYTHON}"
cd "${PROJECT_ROOT}"
"${PYTHON}" -m mjlab.scripts.train \
  "${TASK}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --env.seed "${SEED}" \
  --agent.seed "${SEED}" \
  --agent.logger tensorboard \
  --agent.max-iterations "${MAX_ITERATIONS}" \
  --agent.save-interval "${SAVE_INTERVAL}" \
  --agent.run-name "${RUN_NAME}" \
  --log-root "${RUN_ROOT}"

echo "MJLAB_TRAINING_OK"
echo "End: $(date --iso-8601=seconds)"
