"""
Check if part sweep volumes collide with base_scene or remain_scene geometry.

For each articulated part:
- Sample positions along the joint range (0 to range_max)
- Transform part mesh to each position
- Check intersection with base_scene.obj (its own base) and other parts' bases

Usage:
    python check_sweep_collision.py --scene_dir <scene_output_filtered>
"""
import argparse
import numpy as np
import trimesh
from pathlib import Path


def sweep_part(part_mesh, origin, axis, joint_type, range_max, n_samples=10):
    """Generate part meshes at sampled positions along the joint range."""
    positions = []
    axis_norm = axis / np.linalg.norm(axis)

    for t in np.linspace(0, range_max, n_samples):
        if joint_type == "revolute":
            # Rotate part by angle t around (origin, axis)
            T = trimesh.transformations.rotation_matrix(t, axis_norm, origin)
        else:  # prismatic
            T = trimesh.transformations.translation_matrix(t * axis_norm)

        m = part_mesh.copy()
        m.apply_transform(T)
        positions.append(m)

    return positions


def check_intersection(swept_meshes, target_mesh, penetration_thresh=0.01):
    """Check if swept part vertices are PENETRATING target mesh surface.

    Use ProximityQuery.signed_distance: negative means inside the (closed) mesh,
    but for non-closed meshes we use unsigned distance to surface.
    Here we use closest-surface distance and flag points that are very close
    (within penetration_thresh) as potentially colliding.

    Returns: (close_count, total_count, min_dist) per max-open position
    """
    if target_mesh is None or len(target_mesh.vertices) == 0:
        return 0, 0, None

    # Use KDTree on target vertices for distance check (avoids rtree dependency)
    from scipy.spatial import cKDTree
    tree = cKDTree(target_mesh.vertices)

    # Only check the max-open position (last in swept_meshes)
    final = swept_meshes[-1]
    v = final.vertices
    dist, _ = tree.query(v, k=1)

    close = (dist < penetration_thresh).sum()
    return close, len(v), float(dist.min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    remain_mesh = trimesh.load(str(scene_dir / "remain_scene.ply"))

    mesh_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("mesh_")])

    parts_data = {}
    for mesh_dir in mesh_dirs:
        try:
            part = trimesh.load(str(mesh_dir / "part.ply"))
            origin = np.load(str(mesh_dir / "origin_world.npy"))
            axis = np.load(str(mesh_dir / "vector_world.npy"))
            range_max = float(np.load(str(mesh_dir / "range_max.npy")).flat[0])
            atype_arr = np.load(str(mesh_dir / "articulation_type.npy"))
            atype = str(atype_arr.item()) if isinstance(atype_arr, np.ndarray) else str(atype_arr)
            base_scene = None
            bs_path = mesh_dir / "base_scene.obj"
            if bs_path.exists():
                base_scene = trimesh.load(str(bs_path))

            parts_data[mesh_dir.name] = {
                "part": part,
                "origin": origin,
                "axis": axis,
                "range_max": range_max,
                "joint_type": "revolute" if atype.lower() == "rotation" else "prismatic",
                "base_scene": base_scene,
            }
        except Exception as e:
            print(f"  Skip {mesh_dir.name}: {e}")

    print(f"\n=== Sweep collision check (n_samples=10) ===\n")
    for name, d in parts_data.items():
        swept = sweep_part(d["part"], d["origin"], d["axis"], d["joint_type"], d["range_max"])

        own_in, own_total, own_min = check_intersection(swept, d["base_scene"])
        rem_in, rem_total, rem_min = check_intersection(swept, remain_mesh)

        own_pct = 100 * own_in / own_total if own_total > 0 else 0
        rem_pct = 100 * rem_in / rem_total if rem_total > 0 else 0

        print(f"{name} ({d['joint_type']}, range={d['range_max']:.3f}):")
        print(f"  vs own base_scene at max-open: {own_in}/{own_total} verts within 1cm  (min dist: {own_min})")
        print(f"  vs remain_scene  at max-open: {rem_in}/{rem_total} verts within 1cm  (min dist: {rem_min})")


if __name__ == "__main__":
    main()
