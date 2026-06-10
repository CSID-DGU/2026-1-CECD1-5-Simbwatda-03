#!/usr/bin/env bash
# Re-run scene2part from grounded_sam (with new sam_prompt.txt manually set)
# through s2p_5, with container-mask filter applied before s2p_5.
# Skips: ram++, qwen, intrinsic_mvp (already produced), s2p_6 (not needed).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${1:-${REACT3D_DATA_DIR:-}}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
REACT3D_ENV="${REACT3D_ENV:-react3d}"
REACT3D_DEVICE="${REACT3D_DEVICE:-cuda}"
FILTER_SCRIPT="/home/gpuadmin/capstone/USDNet/REACT3D_mesh030_lab/eval/filter_container_masks.py"

if [[ -z "${DATA_DIR}" ]]; then
  echo "Usage: $0 <data_dir>" >&2
  exit 1
fi
[[ -d "${DATA_DIR}" ]] || { echo "missing ${DATA_DIR}" >&2; exit 1; }
[[ -f "${DATA_DIR}/sam_prompt.txt" ]] || { echo "missing sam_prompt.txt" >&2; exit 1; }
[[ -f "${DATA_DIR}/pose_intrinsic_imu_mvp.json" ]] || { echo "missing pose_intrinsic_imu_mvp.json" >&2; exit 1; }
[[ -f "${DATA_DIR}/scene_tags.json" ]] || { echo "missing scene_tags.json" >&2; exit 1; }

source "${CONDA_SH}"
conda activate "${REACT3D_ENV}"

echo "=========================================="
echo "Phase A: grounded_sam (with new sam_prompt)"
echo "  prompt: $(cat ${DATA_DIR}/sam_prompt.txt)"
echo "=========================================="
TEXT_PROMPT="$(cat "${DATA_DIR}/sam_prompt.txt")"
cd "${CODE_DIR}/grounded_sam"
python grounded_sam_detect_doors.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_dir "${DATA_DIR}/images_2/" \
  --output_dir "${DATA_DIR}/grounded_sam" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "${TEXT_PROMPT}" \
  --device "${REACT3D_DEVICE}"

echo "=========================================="
echo "Phase B: s2p_1_fast"
echo "=========================================="
cd "${CODE_DIR}/scene2part"
python s2p_1_fast.py --data_dir "${DATA_DIR}" --image_dir "${DATA_DIR}/images_2" --mesh_path "${DATA_DIR}/mesh_aligned_0.05.ply" --num_max_frames 2000 --num_faces_simplified 80000 --min_cooccurrence_count 3

echo "=========================================="
echo "Phase C: s2p_2 → s2p_3_fast → s2p_4"
echo "=========================================="
python s2p_2.py --data_dir "${DATA_DIR}"
python s2p_3_fast.py --data_dir "${DATA_DIR}" --image_dir "${DATA_DIR}/images_2" --top_k 5
python s2p_4.py --data_dir "${DATA_DIR}"

echo "=========================================="
echo "Phase D: container-mask filter (Stage 1 layer)"
echo "=========================================="
rm -rf "${DATA_DIR}/perception/vis_groups_cleaned_full"  # drop stale backup
python "${FILTER_SCRIPT}" --scene_dir "${DATA_DIR}"

echo "=========================================="
echo "Phase E: s2p_5"
echo "=========================================="
python s2p_5.py \
  --data_dir "${DATA_DIR}" \
  --mesh_path "${DATA_DIR}/mesh_aligned_0.05.ply" \
  --image_dir "${DATA_DIR}/images_2" \
  --depth_dir "${DATA_DIR}/depth" \
  --cluster_mesh_mode best \
  --cluster_iou_threshold 0.6 \
  --cluster_union_max_face_ratio 1.75 \
  --cluster_union_max_extent_ratio 1.25

echo "=========================================="
echo "DONE."
echo "=========================================="
