"""
Scene-aware parent body extraction.

For each articulated part, crops the remain_scene mesh around the part's
bounding box to extract the actual door frame / cabinet body as base geometry.
This replaces the OBB shell approach which causes overlap between parts.

Usage:
    python extract_base_from_scene.py --scene_dir <scene_output_filtered>
"""
import argparse
import numpy as np
import trimesh
from pathlib import Path


def extract_base_body(remain_mesh, part_mesh, origin, axis, margin_ratio=0.3):
    """
    Extract base body from remain_scene by cropping near the part's bounding box.

    Args:
        remain_mesh: trimesh of the background scene
        part_mesh: trimesh of the articulated part
        origin: joint origin (3,)
        axis: joint axis (3,)
        margin_ratio: how much to expand the part AABB for cropping

    Returns:
        trimesh of the cropped base body, or None if empty
    """
    part_min, part_max = part_mesh.bounds
    part_size = part_max - part_min
    margin = part_size * margin_ratio

    crop_min = part_min - margin
    crop_max = part_max + margin

    verts = remain_mesh.vertices
    vert_mask = np.all((verts >= crop_min) & (verts <= crop_max), axis=1)

    if vert_mask.sum() == 0:
        return None

    face_mask = vert_mask[remain_mesh.faces].all(axis=1)
    if face_mask.sum() == 0:
        return None

    cropped = remain_mesh.submesh([np.where(face_mask)[0]], append=True)
    return cropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True,
                        help="Path to scene_output_filtered directory")
    parser.add_argument("--margin", type=float, default=0.3,
                        help="AABB expansion ratio (default: 0.3)")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    remain_path = scene_dir / "remain_scene.ply"

    if not remain_path.exists():
        print(f"Error: {remain_path} not found")
        return

    print(f"Loading remain_scene from {remain_path}...")
    remain_mesh = trimesh.load(str(remain_path))
    print(f"  {len(remain_mesh.vertices)} vertices, {len(remain_mesh.faces)} faces")

    mesh_dirs = sorted([
        d for d in scene_dir.iterdir()
        if d.is_dir() and d.name.startswith("mesh_")
    ])

    print(f"Found {len(mesh_dirs)} mesh directories")

    for mesh_dir in mesh_dirs:
        part_path = mesh_dir / "part.ply"
        origin_path = mesh_dir / "origin_world.npy"
        axis_path = mesh_dir / "vector_world.npy"

        if not all(p.exists() for p in [part_path, origin_path, axis_path]):
            print(f"  {mesh_dir.name}: skipped (missing files)")
            continue

        part_mesh = trimesh.load(str(part_path))
        origin = np.load(str(origin_path))
        axis = np.load(str(axis_path))

        base_body = extract_base_body(
            remain_mesh, part_mesh, origin, axis, margin_ratio=args.margin
        )

        if base_body is None or len(base_body.faces) == 0:
            print(f"  {mesh_dir.name}: no nearby remain geometry found")
            continue

        out_path = mesh_dir / "base_scene.obj"
        base_body.export(str(out_path))
        print(f"  {mesh_dir.name}: extracted {len(base_body.faces)} faces -> {out_path.name}")


if __name__ == "__main__":
    main()
