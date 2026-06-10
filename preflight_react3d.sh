#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${REACT3D_DATA_DIR:-}}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
REACT3D_ENV="${REACT3D_ENV:-react3d}"
REACT3D_QWEN_ENV="${REACT3D_QWEN_ENV:-react3d_qwen}"

status=0

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    printf '[ok] %s: %s\n' "${label}" "${path}"
  else
    printf '[missing] %s: %s\n' "${label}" "${path}"
    status=1
  fi
}

check_cmd() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    printf '[ok] command: %s\n' "${cmd}"
  else
    printf '[missing] command: %s\n' "${cmd}"
    status=1
  fi
}

echo "Repo root: ${SCRIPT_DIR}"
check_path "conda activation" "${CONDA_SH}"

if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -Fxq "${REACT3D_ENV}"; then
    echo "[ok] conda env: ${REACT3D_ENV}"
  else
    echo "[missing] conda env: ${REACT3D_ENV}"
    status=1
  fi

  if conda env list | awk '{print $1}' | grep -Fxq "${REACT3D_QWEN_ENV}"; then
    echo "[ok] conda env: ${REACT3D_QWEN_ENV}"
  else
    echo "[missing] conda env: ${REACT3D_QWEN_ENV}"
    status=1
  fi
else
  echo "[missing] command: conda"
  status=1
fi

check_path "RAM++ checkpoint" "${SCRIPT_DIR}/ram++/ram_plus_swin_large_14m.pth"
check_path "SAM checkpoint" "${SCRIPT_DIR}/grounded_sam/sam_vit_h_4b8939.pth"
check_path "GroundingDINO checkpoint" "${SCRIPT_DIR}/grounded_sam/groundingdino_swint_ogc.pth"
check_path "OPD checkpoint" "${SCRIPT_DIR}/part2interactive/opdm_rgb.pth"

check_cmd "python"
check_cmd "blender"

if nvidia-smi >/dev/null 2>&1; then
  echo "[ok] nvidia-smi"
else
  echo "[missing] nvidia-smi or usable NVIDIA driver"
  status=1
fi

if [[ -n "${DATA_DIR}" ]]; then
  check_path "data dir" "${DATA_DIR}"
  check_path "images_2" "${DATA_DIR}/images_2"
  check_path "mesh" "${DATA_DIR}/mesh_aligned_0.05.ply"
  check_path "poses" "${DATA_DIR}/pose_intrinsic_imu.json"
  check_path "depth" "${DATA_DIR}/depth"
else
  echo "[info] no data dir provided"
fi

exit "${status}"
