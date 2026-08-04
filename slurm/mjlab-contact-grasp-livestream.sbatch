#!/usr/bin/env bash
#SBATCH --job-name=talos-grasp-live
#SBATCH --output=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.out
#SBATCH --error=/home/stud_yuxuan1/IAS_Workspace/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=stud

set -euo pipefail

WORKSPACE="${HOME}/IAS_Workspace"
PROJECT_ROOT="${WORKSPACE}/projects/WBC-For-Talos"
PYTHON="${WORKSPACE}/envs/mjlab-cu128/bin/python"
TRAIN_JOB_ID="${TRAIN_JOB_ID:?Submit with --export=ALL,TRAIN_JOB_ID=<job-id>}"
TRAIN_REVISION="${TRAIN_REVISION:?Submit with --export=ALL,TRAIN_REVISION=<12-char-revision>}"
CHECKPOINT_ITERATION="${CHECKPOINT_ITERATION:-100}"
NUM_ENVS="${NUM_ENVS:-4}"
REMOTE_PORT="${REMOTE_PORT:-18080}"
WAIT_SECONDS="${WAIT_SECONDS:-1800}"
TRAIN_ROOT="${WORKSPACE}/logs/mjlab-training/${TRAIN_REVISION}/${TRAIN_JOB_ID}"

export TMPDIR="${SLURM_TMPDIR:-${WORKSPACE}/tmp/${SLURM_JOB_ID}}"
export WARP_CACHE_PATH="${WORKSPACE}/shared/mjlab-cache/warp"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TMPDIR}" "${WARP_CACHE_PATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Git revision: $(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
echo "Waiting for model_${CHECKPOINT_ITERATION}.pt below ${TRAIN_ROOT}"

DEADLINE="$((SECONDS + WAIT_SECONDS))"
CHECKPOINT=""
while (( SECONDS < DEADLINE )); do
  CHECKPOINT="$(find "${TRAIN_ROOT}" -type f -name "model_${CHECKPOINT_ITERATION}.pt" -print -quit 2>/dev/null || true)"
  if [[ -n "${CHECKPOINT}" ]]; then
    break
  fi
  sleep 10
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "Timed out waiting for the training checkpoint." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" -u slurm/play_contact_grasp_livestream.py \
  --checkpoint "${CHECKPOINT}" \
  --num-envs "${NUM_ENVS}" \
  --forward-velocity 0.2 \
  --port "${REMOTE_PORT}" \
  --device cpu
