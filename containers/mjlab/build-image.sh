#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_REPOSITORY="${MJLAB_IMAGE_REPOSITORY:-localhost/force-aware-talos-mjlab}"
FULL_REVISION="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
SHORT_REVISION="$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD)"
IMAGE="${IMAGE_REPOSITORY}:${SHORT_REVISION}"

if ! git -C "${PROJECT_ROOT}" diff --quiet \
  || ! git -C "${PROJECT_ROOT}" diff --cached --quiet; then
  echo "Refusing to build from a dirty Git worktree." >&2
  exit 1
fi

echo "Building ${IMAGE} from ${FULL_REVISION}"
podman build \
  --format docker \
  --build-arg "VCS_REF=${FULL_REVISION}" \
  --file "${SCRIPT_DIR}/Containerfile" \
  --tag "${IMAGE}" \
  "${PROJECT_ROOT}"

podman image inspect "${IMAGE}" >/dev/null
echo "MJLAB_IMAGE=${IMAGE}"
