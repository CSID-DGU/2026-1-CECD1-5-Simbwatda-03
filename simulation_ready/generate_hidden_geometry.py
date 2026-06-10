"""
Generate hidden geometry (inner_box) for all articulated parts.

For rotation (cabinet doors): generates 5-face inner box (front/part face removed)
For translation (drawers): generates 4-face inner box (front + top removed) — same as original

Uses OBB corners from obb_world.npy and the same shell-generation logic as
corners_to_mesh() in utilities.py, but as a post-processing step with
scene-level context.

Usage:
    python generate_hidden_geometry.py --scene_dir <scene_output_filtered>
"""
import argparse
import numpy as np
import trimesh
from pathlib import Path


def build_inner_box(corners, pred_type, color=None, thickness_ratio=0.03):
    """
    Build inner_box mesh from OBB corners.

    Same logic as corners_to_mesh() inner_box generation, but:
    - rotation: remove only the part face (left/front)
    - translation: remove part face + top face

    Returns trimesh object.
    """
    corners = np.array(corners)
    if color is None:
        color = np.array([0.5, 0.5, 0.5])

    cuboid_center = np.mean(corners, axis=0)

    faces_def = {
        "front":  [0, 1, 3, 2],
        "back":   [4, 5, 7, 6],
        "right":  [1, 3, 7, 5],
        "top":    [2, 3, 7, 6],
        "bottom": [0, 1, 5, 4],
    }
    # "left" [0,2,6,4] is the part face — always excluded

    # Thickness from shortest edge of left face
    left_indices = [0, 2, 6, 4]
    left_face = corners[left_indices]
    edge_lengths = [np.linalg.norm(left_face[(i+1)%4] - left_face[i]) for i in range(4)]
    min_edge = min(edge_lengths)
    thickness = thickness_ratio * min_edge

    # First pass: compute inner layer for each face (same as base shell generation)
    inner_layer_faces = {}
    for key, indices in faces_def.items():
        outer = corners[indices]
        face_center = np.mean(outer, axis=0)
        normal = np.cross(outer[1] - outer[0], outer[2] - outer[0])
        normal /= np.linalg.norm(normal)
        if np.dot(normal, cuboid_center - face_center) < 0:
            normal = -normal
        inner = outer + thickness * normal
        inner_layer_faces[key] = (outer, inner, normal)

    # Determine which faces to skip
    skip_faces = {"left"}  # part face always skipped (not in faces_def anyway)

    if pred_type == "translation":
        # Also skip top face (highest Z center)
        face_centers = {k: np.mean(corners[idx], axis=0) for k, idx in faces_def.items()}
        top_face_key = max(face_centers, key=lambda k: face_centers[k][2])
        skip_faces.add(top_face_key)

    # Build inner_box mesh
    vertices_list = []
    triangles = []
    vertex_offset = 0

    for key in faces_def:
        if key in skip_faces:
            continue

        outer_face, inner_face, normal = inner_layer_faces[key]
        # Inner box outer = base shell inner layer
        ib_outer = inner_face
        # Inner box inner = further inward
        ib_inner = ib_outer + thickness * normal

        face_verts = np.vstack([ib_outer, ib_inner])  # (8, 3)
        vertices_list.append(face_verts)
        n = ib_outer.shape[0]  # 4

        # Outer face triangles
        triangles.append([vertex_offset, vertex_offset+1, vertex_offset+2])
        triangles.append([vertex_offset, vertex_offset+2, vertex_offset+3])
        # Inner face triangles (flipped)
        triangles.append([vertex_offset+n, vertex_offset+n+2, vertex_offset+n+1])
        triangles.append([vertex_offset+n, vertex_offset+n+3, vertex_offset+n+2])
        # Side walls
        for i in range(n):
            i_next = (i + 1) % n
            v0 = vertex_offset + i
            v1 = vertex_offset + i_next
            v2 = vertex_offset + n + i_next
            v3 = vertex_offset + n + i
            triangles.append([v0, v1, v2])
            triangles.append([v0, v2, v3])

        vertex_offset += 2 * n

    if not vertices_list:
        return None

    vertices = np.vstack(vertices_list)
    triangles = np.array(triangles, dtype=np.int32)
    # Double-sided
    flipped = triangles[:, ::-1]
    all_tris = np.vstack([triangles, flipped])

    colors = np.tile(color, (len(vertices), 1))

    mesh = trimesh.Trimesh(vertices=vertices, faces=all_tris,
                           vertex_colors=np.clip(colors * 255, 0, 255).astype(np.uint8))
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    mesh_dirs = sorted([
        d for d in scene_dir.iterdir()
        if d.is_dir() and d.name.startswith("mesh_")
    ])

    print(f"Found {len(mesh_dirs)} mesh directories")

    for mesh_dir in mesh_dirs:
        obb_path = mesh_dir / "obb_world.npy"
        atype_path = mesh_dir / "articulation_type.npy"
        part_path = mesh_dir / "part.ply"

        if not all(p.exists() for p in [obb_path, atype_path, part_path]):
            print(f"  {mesh_dir.name}: skipped (missing files)")
            continue

        corners = np.load(str(obb_path))
        atype_arr = np.load(str(atype_path))
        atype = str(atype_arr.item()) if isinstance(atype_arr, np.ndarray) else str(atype_arr)
        pred_type = atype.lower()  # "rotation" or "translation"

        # Use part mesh average color for inner_box
        part = trimesh.load(str(part_path))
        if part.visual and hasattr(part.visual, 'vertex_colors') and len(part.visual.vertex_colors) > 0:
            avg_color = np.mean(part.visual.vertex_colors[:, :3], axis=0) / 255.0
        else:
            avg_color = np.array([0.5, 0.5, 0.5])

        inner_box = build_inner_box(corners, pred_type, color=avg_color)

        if inner_box is None:
            print(f"  {mesh_dir.name}: no inner_box generated")
            continue

        out_path = mesh_dir / "inner_box_v2.obj"
        inner_box.export(str(out_path))
        print(f"  {mesh_dir.name} ({pred_type}): {len(inner_box.faces)} faces -> {out_path.name}")


if __name__ == "__main__":
    main()
