"""
Gravity-snap OBB: rotate part assets so the OBB's near-vertical axis becomes exactly (0,0,1).
Applies the same small rotation to part.ply, base.obj/.ply, inner_box.obj,
and updates obb_world.npy, origin_world.npy, vector_world.npy.

Pivot: OBB centroid.
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh


def compute_snap_rotation(obb_corners: np.ndarray) -> np.ndarray:
    a = obb_corners[1] - obb_corners[0]
    b = obb_corners[3] - obb_corners[0]
    c = obb_corners[4] - obb_corners[0]
    axes = [a, b, c]
    axes_n = [v / (np.linalg.norm(v) + 1e-12) for v in axes]
    z_dots = [abs(v[2]) for v in axes_n]
    pick = int(np.argmax(z_dots))
    v_up = axes_n[pick]
    if v_up[2] < 0:
        v_up = -v_up
    target = np.array([0.0, 0.0, 1.0])
    cos_t = float(np.clip(np.dot(v_up, target), -1.0, 1.0))
    if cos_t > 1.0 - 1e-12:
        return np.eye(3)
    rot_axis = np.cross(v_up, target)
    n = np.linalg.norm(rot_axis)
    if n < 1e-12:
        return np.eye(3)
    rot_axis /= n
    angle = float(np.arccos(cos_t))
    K = np.array([
        [0.0, -rot_axis[2], rot_axis[1]],
        [rot_axis[2], 0.0, -rot_axis[0]],
        [-rot_axis[1], rot_axis[0], 0.0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R


def transform_points(pts: np.ndarray, R: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    return (pts - pivot) @ R.T + pivot


def snap_mesh_file(path: Path, R: np.ndarray, pivot: np.ndarray) -> bool:
    if not path.exists():
        return False
    m = trimesh.load(str(path), force="mesh", process=False)
    if m.vertices is None or len(m.vertices) == 0:
        return False
    m.vertices = transform_points(np.asarray(m.vertices), R, pivot)
    m.export(str(path))
    return True


def process_part(folder: Path) -> dict:
    obb_path = folder / "obb_world.npy"
    if not obb_path.exists():
        return {"folder": str(folder), "skipped": "no obb_world.npy"}

    obb = np.load(obb_path)
    if obb.shape != (8, 3):
        return {"folder": str(folder), "skipped": f"obb shape {obb.shape}"}

    R = compute_snap_rotation(obb)
    pivot = obb.mean(axis=0)
    angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

    # apply rotation
    obb_new = transform_points(obb, R, pivot)
    np.save(obb_path, obb_new)

    origin_path = folder / "origin_world.npy"
    vector_path = folder / "vector_world.npy"
    if origin_path.exists():
        origin = np.load(origin_path).reshape(3)
        np.save(origin_path, transform_points(origin[None, :], R, pivot)[0])
    if vector_path.exists():
        vec = np.load(vector_path).reshape(3)
        np.save(vector_path, R @ vec)  # direction-only, no pivot translation

    cuboid_path = folder / "cuboid_world.npy"
    if cuboid_path.exists():
        cub = np.load(cuboid_path)
        if cub.ndim == 2 and cub.shape[1] == 3:
            np.save(cuboid_path, transform_points(cub, R, pivot))

    mesh_files = ["part.ply", "part.obj", "base.ply", "base.obj", "inner_box.obj"]
    snapped = []
    for fn in mesh_files:
        if snap_mesh_file(folder / fn, R, pivot):
            snapped.append(fn)

    return {
        "folder": folder.name,
        "snap_angle_deg": angle_deg,
        "snapped_files": snapped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="scene_output_filtered/")
    args = ap.parse_args()

    root = Path(args.folder)
    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if not sub.name.startswith("mesh_"):
            continue
        info = process_part(sub)
        rows.append(info)
        ang = info.get("snap_angle_deg", float("nan"))
        n = len(info.get("snapped_files", []))
        print(f"{info['folder']}: rot={ang:.2f}°, mesh files updated={n}")


if __name__ == "__main__":
    main()
