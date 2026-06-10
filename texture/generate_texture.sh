#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${1:-${REACT3D_DATA_DIR:-}}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
REACT3D_ENV="${REACT3D_ENV:-react3d}"
FILTERED_DIR="${REACT3D_SCENE_OUTPUT_FILTERED_DIR:-${DATA_DIR}/scene_output_filtered}"
BLENDER_BIN="${BLENDER_BIN:-blender}"

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

source "${CONDA_SH}"
conda activate "${REACT3D_ENV}"

cd "${SCRIPT_DIR}"

# it might take a long time for big scene, change the texture resolution to make it faster
python convert_texture.py --folder "${FILTERED_DIR}" --texture-size 128 --remain-texture-size 512

"${BLENDER_BIN}" --background --python blender_glb_to_dae.py -- --folder "${FILTERED_DIR}"
