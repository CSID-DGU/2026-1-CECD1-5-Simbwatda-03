"""
Convert REACT3D data format to Any3DIS (ScanNet++ style) format.

REACT3D input structure:
    <scene_dir>/
        images_2/frame_XXXXXX.png         (RGB 960x720, subsampled)
        depth/frame_XXXXXX.png            (depth 256x192, uint16, mm)
        pose_intrinsic_imu.json           (all 6769 frames, pose + intrinsic)
        mesh_aligned_0.05.ply             (scene mesh)

Any3DIS output structure (mimics ScanNet++_2D_5interval):
    <output_dir>/
        Scannetpp_2D_5interval/val/<scene_id>/
            color/<frame_id>.jpg
            depth/<frame_id>.png
            pose/<frame_id>.txt           (4x4 c2w matrix)
            intrinsic/<frame_id>.txt      (3x3 RGB intrinsic, scaled to actual image)
            intrinsic.txt                 (global RGB intrinsic, same as per-frame here)
        Scannetpp_2D_5interval/val/intrinsic_depth.txt   (3x3 depth intrinsic)
        Scannetpp_3D/val/original_ply_files/<scene_id>.ply

Usage:
    python prepare_any3dis_data.py \\
        --react3d_scene /home/gpuadmin/capstone/USDNet/REACT3D/data/example_input_scene \\
        --output_dir /home/gpuadmin/capstone/USDNet/REACT3D/data/any3dis_data \\
        --scene_id react3d_scene
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--react3d_scene", required=True, help="REACT3D scene directory")
    parser.add_argument("--output_dir", required=True, help="Any3DIS data root directory")
    parser.add_argument("--scene_id", required=True, help="Scene ID to use (also subfolder name)")
    args = parser.parse_args()

    scene_dir = Path(args.react3d_scene)
    out_root = Path(args.output_dir)

    # Output paths
    scene2d_dir = out_root / "Scannetpp_2D_5interval" / "val" / args.scene_id
    color_dir = scene2d_dir / "color"
    depth_dir = scene2d_dir / "depth"
    pose_dir = scene2d_dir / "pose"
    intrinsic_dir = scene2d_dir / "intrinsic"
    ply_out_dir = out_root / "Scannetpp_3D" / "val" / "original_ply_files"

    for d in (color_dir, depth_dir, pose_dir, intrinsic_dir, ply_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load pose JSON
    pose_json_path = scene_dir / "pose_intrinsic_imu.json"
    print(f"Loading pose JSON: {pose_json_path}")
    with open(pose_json_path) as f:
        pose_data = json.load(f)

    # Find available RGB frames
    images_dir = scene_dir / "images_2"
    depth_src_dir = scene_dir / "depth"
    rgb_files = sorted(images_dir.glob("frame_*.png"))
    print(f"Found {len(rgb_files)} RGB frames")

    # Determine RGB and depth dimensions
    sample_rgb = cv2.imread(str(rgb_files[0]))
    rgb_h, rgb_w = sample_rgb.shape[:2]
    sample_depth = cv2.imread(str(depth_src_dir / rgb_files[0].name), -1)
    depth_h, depth_w = sample_depth.shape[:2]
    print(f"RGB: {rgb_w}x{rgb_h}, Depth: {depth_w}x{depth_h}")

    # Original intrinsic resolution implied by intrinsic principal point
    sample_K = np.array(pose_data[rgb_files[0].stem]["intrinsic"])
    orig_w = sample_K[0, 2] * 2  # ~1907 → 1920
    orig_h = sample_K[1, 2] * 2  # ~1447 → 1440
    print(f"Original calibration resolution (implied): {orig_w:.0f}x{orig_h:.0f}")

    # Scale factors
    scale_x_rgb = rgb_w / orig_w
    scale_y_rgb = rgb_h / orig_h
    scale_x_depth = depth_w / orig_w
    scale_y_depth = depth_h / orig_h
    print(f"RGB scale: ({scale_x_rgb:.4f}, {scale_y_rgb:.4f})")
    print(f"Depth scale: ({scale_x_depth:.4f}, {scale_y_depth:.4f})")

    # Build scaled intrinsics
    def scale_intrinsic(K, sx, sy):
        K2 = K.copy()
        K2[0, 0] *= sx
        K2[1, 1] *= sy
        K2[0, 2] *= sx
        K2[1, 2] *= sy
        return K2

    rgb_K_global = scale_intrinsic(sample_K, scale_x_rgb, scale_y_rgb)
    depth_K_global = scale_intrinsic(sample_K, scale_x_depth, scale_y_depth)

    print(f"Scaled RGB intrinsic:\n{rgb_K_global}")
    print(f"Scaled depth intrinsic:\n{depth_K_global}")

    # Save global intrinsics
    np.savetxt(scene2d_dir / "intrinsic.txt", rgb_K_global, fmt="%.6f")
    np.savetxt(out_root / "Scannetpp_2D_5interval" / "val" / "intrinsic_depth.txt",
               depth_K_global, fmt="%.6f")
    # Also save inside scene dir for convenience
    np.savetxt(scene2d_dir / "intrinsic_depth.txt", depth_K_global, fmt="%.6f")

    # Per-frame conversion
    skipped = 0
    processed = 0
    for rgb_file in rgb_files:
        frame_stem = rgb_file.stem  # e.g. "frame_000001"

        if frame_stem not in pose_data:
            print(f"  skip {frame_stem}: missing in pose JSON")
            skipped += 1
            continue

        depth_file = depth_src_dir / rgb_file.name
        if not depth_file.exists():
            print(f"  skip {frame_stem}: depth missing")
            skipped += 1
            continue

        # Re-encode frame number compact to match Any3DIS naming (numeric, no leading zeros constraint)
        # but ScanNet++ uses 00000.jpg style. Let's keep frame_XXXXXX names for traceability.
        frame_id = frame_stem  # keep frame_XXXXXX

        # Symlink RGB and depth (no copy, save disk)
        rgb_target = color_dir / f"{frame_id}.jpg"
        depth_target = depth_dir / f"{frame_id}.png"

        # RGB needs to be jpg per scannet_loader. cv2 saves jpg directly.
        if not rgb_target.exists():
            img = cv2.imread(str(rgb_file))
            cv2.imwrite(str(rgb_target), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not depth_target.exists():
            # depth_target should be uint16 png, same as source
            if not depth_target.is_symlink():
                try:
                    os.symlink(depth_file.resolve(), depth_target)
                except OSError:
                    shutil.copy2(depth_file, depth_target)

        # Pose (4x4 c2w in aligned scene coordinates, matching mesh_aligned_0.05.ply)
        pose = np.array(pose_data[frame_stem]["aligned_pose"])
        np.savetxt(pose_dir / f"{frame_id}.txt", pose, fmt="%.6f")

        # Per-frame intrinsic (scaled RGB intrinsic — usually identical across frames in REACT3D)
        K = np.array(pose_data[frame_stem]["intrinsic"])
        K_scaled = scale_intrinsic(K, scale_x_rgb, scale_y_rgb)
        np.savetxt(intrinsic_dir / f"{frame_id}.txt", K_scaled, fmt="%.6f")

        processed += 1
        if processed % 200 == 0:
            print(f"  processed {processed}/{len(rgb_files)}")

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")

    # Copy/symlink scene mesh as point cloud
    mesh_src = scene_dir / "mesh_aligned_0.05.ply"
    mesh_target = ply_out_dir / f"{args.scene_id}.ply"
    if not mesh_target.exists():
        try:
            os.symlink(mesh_src.resolve(), mesh_target)
            print(f"Symlinked mesh: {mesh_target}")
        except OSError:
            shutil.copy2(mesh_src, mesh_target)
            print(f"Copied mesh: {mesh_target}")

    # Write a split file
    split_file = out_root / "react3d_val.txt"
    with open(split_file, "w") as f:
        f.write(f"{args.scene_id}\n")
    print(f"Wrote split file: {split_file}")

    print(f"\nOutput structure ready at: {out_root}")
    print(f"  2D dir: {scene2d_dir}")
    print(f"  Mesh:   {mesh_target}")
    print(f"  Split:  {split_file}")


if __name__ == "__main__":
    main()
