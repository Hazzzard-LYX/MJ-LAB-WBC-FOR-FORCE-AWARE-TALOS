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
SHORT_REVISION="$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD)"
IMAGE="${MJLAB_IMAGE:-localhost/force-aware-talos-mjlab:${SHORT_REVISION}}"
IMAGE_ARCHIVE="${MJLAB_IMAGE_ARCHIVE:-${WORKSPACE}/images/force-aware-talos-mjlab-${SHORT_REVISION}.oci}"
TASK="${TASK:-Mjlab-Velocity-Flat-Pal-Talos-Free-Payload-Tray}"
NUM_ENVS="${NUM_ENVS:?Submit with --export=ALL,NUM_ENVS=<count>}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-${SHORT_REVISION}-${NUM_ENVS}env}"
RUN_ROOT="${WORKSPACE}/logs/mjlab-training/${SHORT_REVISION}/${SLURM_JOB_ID}"
CACHE_ROOT="${WORKSPACE}/shared/mjlab-cache"
RUNTIME_BASE="${SLURM_TMPDIR:-/tmp}"

export XDG_RUNTIME_DIR="${RUNTIME_BASE}/podman-${UID}-${SLURM_JOB_ID}"
export TMPDIR="${WORKSPACE}/podman-tmp/${SLURM_JOB_ID}"
mkdir -p \
  "${XDG_RUNTIME_DIR}" \
  "${TMPDIR}" \
  "${RUN_ROOT}" \
  "${CACHE_ROOT}/warp"
chmod 0700 "${XDG_RUNTIME_DIR}"
unset DBUS_SESSION_BUS_ADDRESS

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
echo "Image: ${IMAGE}"
echo "Task: ${TASK}"
echo "Environments: ${NUM_ENVS}"
echo "Iterations: ${MAX_ITERATIONS}"
echo "Seed: ${SEED}"
nvidia-smi

test -f "${IMAGE_ARCHIVE}"
if podman image exists "${IMAGE}"; then
  echo "Using node-local cached image: ${IMAGE}"
else
  podman load --input "${IMAGE_ARCHIVE}"
fi
podman image inspect "${IMAGE}" >/dev/null

podman run \
  --rm \
  --network=host \
  --env MUJOCO_GL=disable \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env WARP_CACHE_PATH=/cache/warp \
  --volume "${CACHE_ROOT}/warp:/cache/warp:rw" \
  --volume "${RUN_ROOT}:/workspace/logs:rw" \
  --workdir /opt/pal_mjlab \
  "${IMAGE}" \
  python -m mjlab.scripts.train \
    "${TASK}" \
    --env.scene.num-envs "${NUM_ENVS}" \
    --env.seed "${SEED}" \
    --agent.seed "${SEED}" \
    --agent.logger tensorboard \
    --agent.max-iterations "${MAX_ITERATIONS}" \
    --agent.save-interval "${SAVE_INTERVAL}" \
    --agent.run-name "${RUN_NAME}" \
    --log-root /workspace/logs

echo "MJLAB_TRAINING_OK"
echo "End: $(date --iso-8601=seconds)"
