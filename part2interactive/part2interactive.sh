#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${1:-${REACT3D_DATA_DIR:-}}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
REACT3D_ENV="${REACT3D_ENV:-react3d}"
MODEL_PATH="${REACT3D_OPDM_MODEL:-${CODE_DIR}/part2interactive/opdm_rgb.pth}"
OUTPUT_DIR="${REACT3D_SCENE_OUTPUT_DIR:-${DATA_DIR}/scene_output}"
FILTERED_DIR="${REACT3D_SCENE_OUTPUT_FILTERED_DIR:-${DATA_DIR}/scene_output_filtered}"
CPU_FLAG=""

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

if [[ "${REACT3D_USE_CPU:-0}" == "1" ]]; then
  CPU_FLAG="--cpu"
fi

source "${CONDA_SH}"
conda activate "${REACT3D_ENV}"

cd "${SCRIPT_DIR}"

python inference.py --scene-folder "${DATA_DIR}" --output "${OUTPUT_DIR}" --model "${MODEL_PATH}" ${CPU_FLAG}

python filter_duplicates.py --input_dir "${OUTPUT_DIR}" --output_dir "${FILTERED_DIR}" --iou_threshold 0.5

python generate_remain_scene.py --scene "${DATA_DIR}/mesh_aligned_0.05.ply" --part_dir "${FILTERED_DIR}"
