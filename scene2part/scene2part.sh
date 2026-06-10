#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${1:-${REACT3D_DATA_DIR:-}}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
REACT3D_ENV="${REACT3D_ENV:-react3d}"
REACT3D_QWEN_ENV="${REACT3D_QWEN_ENV:-react3d_qwen}"
REACT3D_DEVICE="${REACT3D_DEVICE:-cuda}"
S2P5_CLUSTER_MESH_MODE="${S2P5_CLUSTER_MESH_MODE:-best}"
S2P5_CLUSTER_IOU_THRESHOLD="${S2P5_CLUSTER_IOU_THRESHOLD:-0.6}"
S2P5_CLUSTER_UNION_MAX_FACE_RATIO="${S2P5_CLUSTER_UNION_MAX_FACE_RATIO:-1.75}"
S2P5_CLUSTER_UNION_MAX_EXTENT_RATIO="${S2P5_CLUSTER_UNION_MAX_EXTENT_RATIO:-1.25}"
S2P6_FACE_KEEP_MODE="${S2P6_FACE_KEEP_MODE:-all_vertices}"

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

# ----------------------ram------------------------
cd "${CODE_DIR}/ram++"

python inference_ram_plus.py --image_dir "${DATA_DIR}/images_2" --pretrained "${CODE_DIR}/ram++/ram_plus_swin_large_14m.pth"

cd "${CODE_DIR}/scene2part"

#------------------------openable tags filter--------------------
if [[ "${REACT3D_SKIP_QWEN:-0}" != "1" ]]; then
  conda activate "${REACT3D_QWEN_ENV}"
  python filter_ram_qwen.py --data_dir "${DATA_DIR}"
else
  echo "Skipping Qwen filter because REACT3D_SKIP_QWEN=1"
fi

PROMPT_FILE="${DATA_DIR}/sam_prompt.txt"
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "Prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi
TEXT_PROMPT="$(cat "${PROMPT_FILE}")"

#-----------------------perception---------------------
conda activate "${REACT3D_ENV}"

python generate_intrinsic_mvp.py "${DATA_DIR}/pose_intrinsic_imu.json"

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

cd "${CODE_DIR}/scene2part"

# high min_cooccurrence_count can give more robust results when you have high RGBD frame count, but maybe too strict for low frame count
# suggestion: 3 for low frame count, 5+ for high frame count
python s2p_1_fast.py --data_dir "${DATA_DIR}" --image_dir "${DATA_DIR}/images_2" --mesh_path "${DATA_DIR}/mesh_aligned_0.05.ply" --num_max_frames 2000 --num_faces_simplified 80000 --min_cooccurrence_count 3

python s2p_2.py --data_dir "${DATA_DIR}"

python s2p_3_fast.py --data_dir "${DATA_DIR}" --image_dir "${DATA_DIR}/images_2" --top_k 5

python s2p_4.py --data_dir "${DATA_DIR}"

python s2p_5.py \
  --data_dir "${DATA_DIR}" \
  --mesh_path "${DATA_DIR}/mesh_aligned_0.05.ply" \
  --image_dir "${DATA_DIR}/images_2" \
  --depth_dir "${DATA_DIR}/depth" \
  --cluster_mesh_mode "${S2P5_CLUSTER_MESH_MODE}" \
  --cluster_iou_threshold "${S2P5_CLUSTER_IOU_THRESHOLD}" \
  --cluster_union_max_face_ratio "${S2P5_CLUSTER_UNION_MAX_FACE_RATIO}" \
  --cluster_union_max_extent_ratio "${S2P5_CLUSTER_UNION_MAX_EXTENT_RATIO}"

python s2p_6.py -d "${DATA_DIR}" --face_keep_mode "${S2P6_FACE_KEEP_MODE}"
