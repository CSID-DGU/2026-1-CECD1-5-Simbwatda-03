"""
Inner_box = original cuboid-shell base.obj shrunk toward its centroid.

Idea: keep the cuboid SHAPE (so the cavity has the box appearance) but make it
slightly smaller than the part surface and the base_scene mesh, so it sits
visibly inside without poking through.

Usage:
    python generate_inner_from_cuboid_shrink.py --scene_dir <dir> --shrink 0.85
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh


def shrink_mesh(mesh: trimesh.Trimesh, shrink: float) -> trimesh.Trimesh:
    c = mesh.vertices.mean(axis=0)
    new_verts = c + (np.asarray(mesh.vertices) - c) * shrink
    out = trimesh.Trimesh(vertices=new_verts, faces=np.asarray(mesh.faces),
                          process=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--shrink", type=float, default=0.85,
                    help="Uniform shrink factor toward centroid (default 0.85)")
    args = ap.parse_args()

    root = Path(args.scene_dir)
    parts = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("mesh_"))
    print(f"Found {len(parts)} parts, shrink={args.shrink}")

    for md in parts:
        base = md / "base.obj"
        if not base.exists():
            print(f"  {md.name}: skip (no base.obj)")
            continue
        m = trimesh.load(str(base), force="mesh", process=False)
        if len(m.vertices) == 0:
            print(f"  {md.name}: skip (empty)")
            continue
        out = shrink_mesh(m, args.shrink)
        out.export(str(md / "inner_box_v3.obj"))
        print(f"  {md.name}: shrunk to {len(out.vertices)} verts → inner_box_v3.obj")


if __name__ == "__main__":
    main()
