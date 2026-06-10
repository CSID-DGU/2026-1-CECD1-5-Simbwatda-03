"""
Inner_box from OBB (part-aligned) + axis-direction extrusion.

Why: cuboid_world (OPDM-predicted) can be tilted relative to part surface
(non-parallel, sometimes detached). OBB is fit to part vertices, so its faces are
aligned to the door panel. Building inner_box from OBB guarantees the inner box
is parallel to the door surface.

Procedure per part:
  1. Load obb_world.npy (8 corners) and vector_world.npy (joint axis).
  2. Identify OBB's longest two edges from corner 0 → "front face" (part panel).
  3. Determine the "back" direction:
       - rotation: cross of two longest edges = normal of front face,
         flip sign so it points into the cabinet (away from joint origin).
       - translation: -axis direction (drawer slides outward along axis).
  4. Shift the front face by `depth` along back direction → back face.
  5. Build a 5-face (rotation) or 5-face (translation) shell from the 8 corners
     by reusing corners_to_mesh logic (front face removed in vis since door panel).

Output: inner_box_v3.obj per part.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh


def _largest_two_edges(obb):
    """Return indices and vectors of the two longest OBB edges from corner 0."""
    edges = [(0, 1, obb[1] - obb[0]),
             (0, 3, obb[3] - obb[0]),
             (0, 4, obb[4] - obb[0])]
    edges.sort(key=lambda e: -np.linalg.norm(e[2]))
    return edges[0], edges[1], edges[2]  # longest, second, third (= thinnest)


def _shrink_in_plane(corners_4, factor):
    """Shrink 4 corners toward their centroid (in-plane), keeping plane normal."""
    c = corners_4.mean(0)
    return c + (corners_4 - c) * factor


def build_inner_box_from_obb(obb, axis, pred_type, depth, in_plane_shrink=0.95):
    """
    Build an 8-corner cuboid aligned to OBB front face, extruded back by depth.
    Returns a trimesh shell (5 faces: top, bottom, left, right, back —
    front is the part panel and is removed for inner-cavity visualization).
    """
    obb = np.asarray(obb).reshape(8, 3)

    # Front face = parallelogram spanned by the two longest edges from corner 0
    # corners 0, 1, ?, 3 — need to figure out which 4 of the 8 are the front face
    # The convention from corners_to_mesh in REACT3D: edges from corner 0 are a, b, c
    # The largest two form the front face plane. Front face corners can be:
    #   0, idx_long1, idx_long1 + idx_long2_dir, idx_long2
    # But OBB indexing: 0=base, 1=base+a, 3=base+b, 4=base+c, 5=base+a+c, 7=base+b+c
    # Faces of OBB (from corners_to_mesh in utilities.py):
    #   left = [0,3,7,4]  (b-c plane), front = [0,1,5,4]  (a-c plane), bottom = [0,1,2,3] (a-b plane)
    # The "front face" used as the panel is the one whose normal aligns with thinnest edge.
    a_vec = obb[1] - obb[0]
    b_vec = obb[3] - obb[0]
    c_vec = obb[4] - obb[0]
    lens = [np.linalg.norm(a_vec), np.linalg.norm(b_vec), np.linalg.norm(c_vec)]
    thin_idx = int(np.argmin(lens))  # thinnest axis = door thickness
    # Door panel is in the plane perpendicular to the thinnest axis.
    if thin_idx == 0:  # a is thinnest → front face spans b,c → corners [0,3,7,4]
        front_idx = [0, 3, 7, 4]
    elif thin_idx == 1:  # b is thinnest → front face spans a,c → corners [0,1,5,4]
        front_idx = [0, 1, 5, 4]
    else:  # c is thinnest → front face spans a,b → corners [0,1,2,3]
        front_idx = [0, 1, 2, 3]
    thin_vec = [a_vec, b_vec, c_vec][thin_idx]

    # Back direction = into the cabinet (away from outside).
    if pred_type.lower() == "translation":
        back_dir = -np.asarray(axis) / (np.linalg.norm(axis) + 1e-12)
    else:
        # rotation: normal of front face = thin_vec direction. Flip so it points
        # away from the joint axis direction (axis is the hinge, going up/down;
        # back is horizontal into cabinet).
        front_normal = thin_vec / (np.linalg.norm(thin_vec) + 1e-12)
        # Choose sign that points "into" the cabinet: away from the OBB centroid's
        # neighbor (we'll just pick the sign that makes the back face inside the
        # other corners' centroid).
        obb_centroid = obb.mean(0)
        front_centroid = obb[front_idx].mean(0)
        if np.dot(front_normal, obb_centroid - front_centroid) < 0:
            front_normal = -front_normal
        back_dir = front_normal

    front_corners = obb[front_idx]  # 4 corners on the door panel
    # Slightly shrink in-plane so inner_box edges sit inside the part surface
    front_corners_shrunk = _shrink_in_plane(front_corners, in_plane_shrink)
    back_corners = front_corners_shrunk + back_dir * depth

    # Build mesh: 5 faces (top, bottom, left, right, back). Front omitted.
    # Indexing: 0..3 = front_shrunk, 4..7 = back
    V = np.vstack([front_corners_shrunk, back_corners])
    faces = []
    # Back face (winding so normal points TOWARD front, visible from inside)
    faces.extend([[4, 5, 6], [4, 6, 7]])
    # Side walls (4 of them): for each edge (i, i+1) on front, with back counterparts
    for i in range(4):
        i_next = (i + 1) % 4
        a, b = i, i_next
        c, d = a + 4, b + 4
        # quad a-b-d-c → two triangles, oriented so normal points OUT of the box
        # (so visible from outside; from inside it's back-facing)
        faces.append([a, b, d])
        faces.append([a, d, c])
    faces = np.array(faces, dtype=np.int64)
    out = trimesh.Trimesh(vertices=V, faces=faces, process=False)
    out.fix_normals()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--depth", type=float, default=0.12,
                    help="Extrusion depth into cabinet (m), default 12cm")
    ap.add_argument("--shrink", type=float, default=0.95,
                    help="In-plane shrink of front face (default 0.95)")
    args = ap.parse_args()

    root = Path(args.scene_dir)
    parts = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("mesh_"))
    print(f"Found {len(parts)} parts. depth={args.depth}m, shrink={args.shrink}")

    for md in parts:
        obb_p = md / "obb_world.npy"
        axis_p = md / "vector_world.npy"
        type_p = md / "articulation_type.npy"
        if not all(p.exists() for p in [obb_p, axis_p, type_p]):
            print(f"  {md.name}: skip")
            continue
        obb = np.load(obb_p)
        axis = np.load(axis_p).reshape(3)
        pred_type = str(np.load(type_p))
        try:
            inner = build_inner_box_from_obb(obb, axis, pred_type,
                                             args.depth, args.shrink)
        except Exception as e:
            print(f"  {md.name}: error {e}")
            continue
        inner.export(str(md / "inner_box_v3.obj"))
        print(f"  {md.name}: {len(inner.vertices)} verts, {len(inner.faces)} faces → inner_box_v3.obj")


if __name__ == "__main__":
    main()
