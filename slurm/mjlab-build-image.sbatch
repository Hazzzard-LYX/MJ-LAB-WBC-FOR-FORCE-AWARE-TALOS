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

cd "${PROJECT_DIR}"
bash containers/mjlab/build-image.sh
