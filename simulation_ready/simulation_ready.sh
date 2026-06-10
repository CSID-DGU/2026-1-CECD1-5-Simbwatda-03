#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${1:-${REACT3D_DATA_DIR:-}}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
REACT3D_ENV="${REACT3D_ENV:-react3d}"
FILTERED_DIR="${REACT3D_SCENE_OUTPUT_FILTERED_DIR:-${DATA_DIR}/scene_output_filtered}"

if [[ -z "${DATA_DIR}" ]]; then
  echo "Usage: $0 <data_dir>" >&2
  echo "Or set REACT3D_DATA_DIR." >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda activation script not found: ${CONDA_SH}" >&2
  exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Data directory not found: ${DATA_DIR}" >&2
  exit 1
fi

scene_name="$(basename "${DATA_DIR}")"

source "${CONDA_SH}"
conda activate "${REACT3D_ENV}"

cd "${SCRIPT_DIR}"

# Create target directories
mkdir -p "${DATA_DIR}/urdf/${scene_name}"
mkdir -p "${DATA_DIR}/launch"
mkdir -p "${DATA_DIR}/rviz"

python generate_urdf_scene_ros.py --folder_path "${FILTERED_DIR}" --output_path "${DATA_DIR}/urdf/${scene_name}"

python generate_launch_file.py --urdf_dir "${DATA_DIR}/urdf/${scene_name}" --output "${DATA_DIR}/launch/${scene_name}.launch"

python generate_rviz_file.py --launch_file "${DATA_DIR}/launch/${scene_name}.launch" --output "${DATA_DIR}/rviz/${scene_name}.rviz"
