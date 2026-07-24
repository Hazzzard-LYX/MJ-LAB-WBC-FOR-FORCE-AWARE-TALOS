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
SHORT_REVISION="$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD)"
IMAGE="${MJLAB_IMAGE:-localhost/force-aware-talos-mjlab:${SHORT_REVISION}}"
IMAGE_ARCHIVE="${MJLAB_IMAGE_ARCHIVE:-${WORKSPACE}/images/force-aware-talos-mjlab-${SHORT_REVISION}.oci}"
RUN_ROOT="${WORKSPACE}/logs/mjlab-smoke/${SLURM_JOB_ID}"
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
echo "Image archive: ${IMAGE_ARCHIVE}"
nvidia-smi
test -f "${IMAGE_ARCHIVE}"
podman load --input "${IMAGE_ARCHIVE}"
podman image inspect "${IMAGE}" >/dev/null

run_container() {
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
    "$@"
}

run_container python /opt/pal_mjlab/containers/mjlab/smoke.py
run_container env MUJOCO_GL=disable FORCE_CPU=1 \
  /opt/pal_mjlab/.venv/bin/python -m pytest -q

run_container python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Pal-Talos-Free-Payload-Tray \
  --env.scene.num-envs 64 \
  --agent.logger tensorboard \
  --agent.max-iterations 1 \
  --agent.save-interval 1 \
  --agent.run-name container-smoke-64env \
  --log-root /workspace/logs/64env

run_container python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Pal-Talos-Free-Payload-Tray \
  --env.scene.num-envs 1024 \
  --agent.logger tensorboard \
  --agent.max-iterations 1 \
  --agent.save-interval 1 \
  --agent.run-name container-smoke-1024env \
  --log-root /workspace/logs/1024env

echo "MJLAB_GPU_SMOKE_OK"
echo "End: $(date --iso-8601=seconds)"
