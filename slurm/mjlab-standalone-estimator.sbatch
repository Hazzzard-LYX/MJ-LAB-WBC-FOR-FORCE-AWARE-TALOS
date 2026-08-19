#!/usr/bin/env bash
#SBATCH --job-name=talos-estimator
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
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:?Set POLICY_CHECKPOINT to a frozen contact-grasp policy}"
NUM_ENVS="${NUM_ENVS:-1536}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
HISTORY_LENGTH="${HISTORY_LENGTH:-32}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SEED="${SEED:-42}"
SHORT_REVISION="$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD)"
OUTPUT_DIR="${WORKSPACE}/logs/standalone-estimator/${SHORT_REVISION}/${SLURM_JOB_ID}"
CACHE_ROOT="${WORKSPACE}/shared/mjlab-cache"

export TMPDIR="${SLURM_TMPDIR:-${WORKSPACE}/tmp/${SLURM_JOB_ID}}"
export WARP_CACHE_PATH="${CACHE_ROOT}/warp"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TMPDIR}" "${OUTPUT_DIR}" "${WARP_CACHE_PATH}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Git revision: $(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
echo "Policy checkpoint: ${POLICY_CHECKPOINT}"
echo "Environments: ${NUM_ENVS}"
echo "Training steps: ${TRAIN_STEPS}"
echo "Estimator history: ${HISTORY_LENGTH}"
nvidia-smi

test -x "${PYTHON}"
test -f "${POLICY_CHECKPOINT}"
cd "${PROJECT_ROOT}"
"${PYTHON}" -u slurm/train_standalone_payload_estimator.py \
  --policy-checkpoint "${POLICY_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-envs "${NUM_ENVS}" \
  --steps "${TRAIN_STEPS}" \
  --history-length "${HISTORY_LENGTH}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device cuda:0

echo "MJLAB_STANDALONE_ESTIMATOR_OK"
echo "End: $(date --iso-8601=seconds)"
