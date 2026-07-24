#!/usr/bin/env bash
#SBATCH --job-name=mjlab-build
#SBATCH --partition=stud
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.out
#SBATCH --error=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${HOME}/IAS_Workspace/projects/WBC-For-Talos"
RUNTIME_BASE="${SLURM_TMPDIR:-/tmp}"
PODMAN_TMP_ROOT="${HOME}/IAS_Workspace/podman-tmp"

export XDG_RUNTIME_DIR="${RUNTIME_BASE}/podman-${UID}-${SLURM_JOB_ID}"
export TMPDIR="${PODMAN_TMP_ROOT}/${SLURM_JOB_ID}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 0700 "${XDG_RUNTIME_DIR}"
mkdir -p "${TMPDIR}"
unset DBUS_SESSION_BUS_ADDRESS

cd "${PROJECT_DIR}"
bash containers/mjlab/build-image.sh
