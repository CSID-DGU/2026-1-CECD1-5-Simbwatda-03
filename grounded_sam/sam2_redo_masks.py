"""
SAM2 image predictor를 사용하여 GroundingDINO bbox를 mask로 변환.

기존 grounded_sam/ 의 mask/*.json (GroundingDINO bbox 결과)을 읽고,
SAM2로 더 정확한 mask를 생성. 원본 grounded_sam/mask_data/*.pkl 포맷과
호환되게 grounded_sam2/mask_data/*.pkl에 저장.

Usage (any3dis env):
    PYTHONPATH=/tmp/Any3DIS_unofficial/segmenter2d/segment-anything-2 \
    python sam2_redo_masks.py \
      --scene_dir /home/gpuadmin/capstone/USDNet/REACT3D/data/example_input_scene \
      --sam2_checkpoint /home/gpuadmin/capstone/USDNet/REACT3D/data/checkpoints/sam2_hiera_large.pt \
      --sam2_config sam2_hiera_l.yaml
"""
import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def process_frame(image_path, boxes_xyxy, predictor):
    """Run SAM2 image predictor with boxes prompts. Returns mask tensor [N, 1, H, W]."""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    box_np = np.array(boxes_xyxy, dtype=np.float32)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_np,
        multimask_output=False,
    )
    # predict() returns numpy [N, H, W] when N>1, [1, H, W] when N=1
    # Convert to torch tensor [N, 1, H, W] to match SAM v1 format
    if masks.ndim == 3:
        masks = masks[:, None, :, :]
    elif masks.ndim == 2:
        masks = masks[None, None, :, :]
    return torch.from_numpy(masks.astype(bool))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_config", default="sam2_hiera_l.yaml")
    parser.add_argument("--input_subdir", default="grounded_sam",
                        help="GroundingDINO output dir name (under scene_dir)")
    parser.add_argument("--output_subdir", default="grounded_sam2",
                        help="SAM2 output dir name (under scene_dir)")
    parser.add_argument("--image_dir_name", default="images_2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    input_dir = scene_dir / args.input_subdir
    output_dir = scene_dir / args.output_subdir
    image_dir = scene_dir / args.image_dir_name

    assert (input_dir / "mask").is_dir(), f"missing {input_dir}/mask"
    assert image_dir.is_dir(), f"missing {image_dir}"

    (output_dir / "mask_data").mkdir(parents=True, exist_ok=True)
    (output_dir / "mask").mkdir(parents=True, exist_ok=True)

    # SAM2 init
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(sam2_model)

    json_files = sorted((input_dir / "mask").glob("*.json"))
    print(f"Found {len(json_files)} frames with GroundingDINO output")

    skip_count = 0
    err_count = 0
    for jf in tqdm(json_files):
        frame_name = jf.stem
        out_pkl = output_dir / "mask_data" / f"{frame_name}.pkl"
        if out_pkl.exists():
            continue

        with open(jf) as f:
            json_data = json.load(f)
        boxes = [e["box"] for e in json_data if e.get("value", 0) != 0]
        if len(boxes) == 0:
            skip_count += 1
            continue

        image_path = image_dir / f"{frame_name}.png"
        if not image_path.exists():
            err_count += 1
            continue

        try:
            with torch.no_grad():
                masks = process_frame(image_path, boxes, predictor)
            with open(out_pkl, "wb") as f:
                pickle.dump({"mask": masks}, f)
            # copy json (box) verbatim for downstream consumers
            shutil.copy(jf, output_dir / "mask" / jf.name)
        except Exception as e:
            err_count += 1
            print(f"\n[err] {frame_name}: {e}")

    print(f"Done. skipped(no box)={skip_count}, errors={err_count}")


if __name__ == "__main__":
    main()
