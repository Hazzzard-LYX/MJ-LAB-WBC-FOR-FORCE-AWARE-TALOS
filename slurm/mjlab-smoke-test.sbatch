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
RUN_ROOT="${WORKSPACE}/logs/mjlab-smoke/${SLURM_JOB_ID}"
CACHE_ROOT="${WORKSPACE}/shared/mjlab-cache"

mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}/warp"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
echo "Image: ${IMAGE}"
nvidia-smi
podman image inspect "${IMAGE}" >/dev/null

run_container() {
  docker run \
    --rm \
    --network=host \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env WARP_CACHE_PATH=/cache/warp \
    --volume "${CACHE_ROOT}/warp:/cache/warp:rw" \
    --volume "${RUN_ROOT}:/workspace/logs:rw" \
    "${IMAGE}" \
    "$@"
}

run_container python /opt/pal_mjlab/containers/mjlab/smoke.py
run_container bash -lc \
  "cd /opt/pal_mjlab && FORCE_CPU=1 python -m pytest -q"

run_container python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Pal-Talos-Free-Payload-Tray \
  --env.scene.num-envs 64 \
  --agent.max-iterations 1 \
  --agent.save-interval 1 \
  --agent.run-name container-smoke-64env \
  --log-root /workspace/logs/64env

run_container python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Pal-Talos-Free-Payload-Tray \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 1 \
  --agent.save-interval 1 \
  --agent.run-name container-smoke-1024env \
  --log-root /workspace/logs/1024env

echo "MJLAB_GPU_SMOKE_OK"
echo "End: $(date --iso-8601=seconds)"
