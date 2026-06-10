"""
Generate dc_features (deep clustering features) substitute from geometric+color cues.

For each mesh vertex, compute a 12D feature vector:
- normal (3D, unit) — surface orientation
- color (3D, [0,1]) — RGB
- relative_position (3D) — normalized position within scene bbox
- curvature (3D) — local geometric variation (eigenvalues of neighborhood cov)

This serves as a learned-feature substitute for Mask3D/ISBNet features.
Used by Any3DIS to filter superpoints during 2D-to-3D lifting via cosine similarity.

Usage:
    python generate_dc_features.py \\
        --mesh mesh_aligned_0.05.ply \\
        --output dc_feat_path/scene.pth
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from sklearn.neighbors import NearestNeighbors


def compute_curvature(points, k=20):
    """Per-point curvature from local PCA: eigenvalues of neighborhood covariance.
    Returns (N, 3): normalized eigenvalues (smallest first)."""
    print(f"  Computing curvature with k={k} neighbors...")
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, idx = nbrs.kneighbors(points)  # (N, k)

    curvatures = np.zeros((len(points), 3), dtype=np.float32)
    for i in range(len(points)):
        nb = points[idx[i]]
        centered = nb - nb.mean(axis=0)
        cov = centered.T @ centered / k
        eigvals = np.linalg.eigvalsh(cov)  # ascending
        s = eigvals.sum()
        if s > 1e-10:
            eigvals = eigvals / s  # normalize
        curvatures[i] = eigvals
        if i > 0 and i % 100000 == 0:
            print(f"    {i}/{len(points)}")
    return curvatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--curvature_k", type=int, default=20)
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(str(mesh_path), process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces)
    print(f"  vertices: {vertices.shape}")

    # Colors
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == len(vertices):
        colors = np.array(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
    else:
        colors = np.zeros((len(vertices), 3), dtype=np.float32)

    # Normals via Open3D
    print("Computing normals...")
    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(vertices)
    om.triangles = o3d.utility.Vector3iVector(faces)
    om.compute_vertex_normals()
    normals = np.array(om.vertex_normals, dtype=np.float32)

    # Relative position in scene bbox: [0,1]^3
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    extent = maxs - mins
    extent[extent < 1e-6] = 1.0
    rel_pos = ((vertices - mins) / extent).astype(np.float32)

    # Curvature (eigenvalues of local neighborhood covariance)
    curvature = compute_curvature(vertices, k=args.curvature_k).astype(np.float32)

    # Concatenate features
    features = np.concatenate([normals, colors, rel_pos, curvature], axis=1)  # (N, 12)
    print(f"  features shape: {features.shape}")
    print(f"  feature stats: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")

    # Save as torch tensor (float32)
    feat_tensor = torch.from_numpy(features).float()
    torch.save(feat_tensor, str(output_path))
    print(f"Saved dc_features to: {output_path}")


if __name__ == "__main__":
    main()
