"""
Generate thin inner_box.obj by extruding part.ply along its axis direction.

Replaces the cuboid-based inner_box (huge, scene-aware-blind) with a small shell
that lives just behind the part surface. This makes the cavity visible when the
door opens without causing overlap with adjacent parts.

For rotation: extrude opposite to part normal (= direction from origin toward part).
For translation: extrude opposite to the prismatic axis.

Output: inner_box.obj (overwrites the original — original was empty for rotation
in our build anyway, and we want the URDF/vis to pick this up automatically).

Usage:
    python generate_thin_inner_box.py --scene_dir <scene_output_filtered> --depth 0.08
"""
import argparse
import os
from pathlib import Path

import numpy as np
import trimesh


def _principal_thin_axis(part_verts: np.ndarray) -> np.ndarray:
    """The OBB's shortest principal direction = surface normal of a flat panel."""
    c = part_verts.mean(0)
    cov = np.cov((part_verts - c).T)
    vals, vecs = np.linalg.eigh(cov)
    return vecs[:, 0]  # smallest eigenvalue


def build_thin_inner_box(part_mesh: trimesh.Trimesh,
                         origin: np.ndarray,
                         axis: np.ndarray,
                         pred_type: str,
                         depth: float) -> trimesh.Trimesh:
    verts = np.asarray(part_mesh.vertices)
    faces = np.asarray(part_mesh.faces)

    # Determine the "back" direction (into the cabinet)
    if pred_type.lower() == "translation":
        # back = -axis (drawer slides outward along axis; cavity is opposite)
        back_dir = -axis / (np.linalg.norm(axis) + 1e-12)
    else:
        # rotation: door's surface normal points outward from cabinet.
        # Approximate normal as part's thinnest principal direction, oriented away
        # from joint origin (joint sits on cabinet frame, normal points outward).
        n = _principal_thin_axis(verts)
        center_to_origin = origin - verts.mean(0)
        # back = direction from part toward joint origin (deeper into cabinet)
        # Actually: door surface normal points away from cabinet, so "back" = -n
        # Choose sign of n so that it points TOWARD the joint origin's side:
        if np.dot(n, center_to_origin) < 0:
            n = -n
        # `n` now points toward the side of the joint (cabinet interior side).
        back_dir = n / (np.linalg.norm(n) + 1e-12)

    # Translate part mesh backward by depth → back wall of cavity
    back_verts = verts + back_dir * depth

    # Build a thin shell: front face = original part mesh,
    # back face = back_verts (with same triangulation, flipped winding),
    # plus side walls connecting boundary edges.
    n_v = len(verts)
    all_verts = np.vstack([verts, back_verts])
    # Front (original orientation) — we won't render this, it's the part itself.
    # But the inner_box visualization usually wants 5 faces (no front).
    # → Only include: back + side walls.

    # Find boundary edges of part mesh
    edges = part_mesh.edges_sorted  # all edges, each appears once per adjacent face
    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = edges_unique[counts == 1]  # edges with only one adjacent face

    # Back face (flipped winding so it points inward, visible from front)
    back_faces = faces[:, ::-1] + n_v

    # Side walls: for each boundary edge (a, b), make a quad
    # vertices: a, b, b+n_v, a+n_v → two triangles
    side_faces = []
    for a, b in boundary:
        side_faces.append([a, b, b + n_v])
        side_faces.append([a, b + n_v, a + n_v])
    side_faces = np.array(side_faces, dtype=np.int64) if side_faces else np.zeros((0, 3), dtype=np.int64)

    all_faces = np.vstack([back_faces, side_faces])

    inner = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    inner.fix_normals()
    return inner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--depth", type=float, default=0.08,
                        help="Extrusion depth into cabinet (meters, default 8cm)")
    parser.add_argument("--overwrite_original", action="store_true",
                        help="Overwrite inner_box.obj (default: write inner_box_v3.obj)")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    mesh_dirs = sorted(d for d in scene_dir.iterdir()
                       if d.is_dir() and d.name.startswith("mesh_"))
    print(f"Found {len(mesh_dirs)} parts.")

    for md in mesh_dirs:
        part_path = md / "part.ply"
        origin_path = md / "origin_world.npy"
        axis_path = md / "vector_world.npy"
        type_path = md / "articulation_type.npy"
        if not all(p.exists() for p in [part_path, origin_path, axis_path, type_path]):
            print(f"  {md.name}: skip (missing files)")
            continue

        part_mesh = trimesh.load(str(part_path), force="mesh", process=False)
        if len(part_mesh.vertices) < 10:
            print(f"  {md.name}: skip (part mesh too small)")
            continue

        origin = np.load(origin_path).reshape(3)
        axis = np.load(axis_path).reshape(3)
        pred_type = str(np.load(type_path))

        inner = build_thin_inner_box(part_mesh, origin, axis, pred_type, args.depth)
        out_name = "inner_box.obj" if args.overwrite_original else "inner_box_v3.obj"
        out_path = md / out_name
        inner.export(str(out_path))
        print(f"  {md.name}: {len(inner.vertices)} verts, {len(inner.faces)} faces → {out_name}")


if __name__ == "__main__":
    main()
