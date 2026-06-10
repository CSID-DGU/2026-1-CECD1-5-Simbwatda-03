#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${REACT3D_DATA_DIR:-}}"
STEP="${2:-all}"

if [[ -z "${DATA_DIR}" ]]; then
  echo "Usage: $0 <data_dir> [scene2part|part2interactive|texture|simulation_ready|all]" >&2
  exit 1
fi

case "${STEP}" in
  scene2part)
    bash "${SCRIPT_DIR}/scene2part/scene2part.sh" "${DATA_DIR}"
    ;;
  part2interactive)
    bash "${SCRIPT_DIR}/part2interactive/part2interactive.sh" "${DATA_DIR}"
    ;;
  texture)
    bash "${SCRIPT_DIR}/texture/generate_texture.sh" "${DATA_DIR}"
    ;;
  simulation_ready)
    bash "${SCRIPT_DIR}/simulation_ready/simulation_ready.sh" "${DATA_DIR}"
    ;;
  all)
    bash "${SCRIPT_DIR}/scene2part/scene2part.sh" "${DATA_DIR}"
    bash "${SCRIPT_DIR}/part2interactive/part2interactive.sh" "${DATA_DIR}"
    bash "${SCRIPT_DIR}/texture/generate_texture.sh" "${DATA_DIR}"
    bash "${SCRIPT_DIR}/simulation_ready/simulation_ready.sh" "${DATA_DIR}"
    ;;
  *)
    echo "Unknown step: ${STEP}" >&2
    echo "Expected one of: scene2part, part2interactive, texture, simulation_ready, all" >&2
    exit 1
    ;;
esac
