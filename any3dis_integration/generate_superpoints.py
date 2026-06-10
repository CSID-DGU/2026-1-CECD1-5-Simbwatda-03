"""
Generate superpoints from a mesh using Felzenszwalb-Huttenlocher graph segmentation.

For each mesh vertex, assign a superpoint ID. Output saved as torch tensor.
Approach:
1. Compute per-vertex normals.
2. Build edge graph from mesh triangles.
3. Edge weight = (1 - dot(normal_i, normal_j)) + color_difference_weight.
4. Felzenszwalb segmentation (Union-Find with threshold k / |component|).

Usage:
    python generate_superpoints.py \\
        --mesh /home/gpuadmin/capstone/USDNet/REACT3D/data/example_input_scene/mesh_aligned_0.05.ply \\
        --output /home/gpuadmin/capstone/USDNet/REACT3D/data/any3dis_data/Scannetpp_3D/val/superpoints/react3d_scene.pth \\
        --k 0.03
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        self.internal_diff = np.zeros(n, dtype=np.float32)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a, b, weight):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.internal_diff[ra] = weight  # max edge weight in component
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def felzenszwalb_mesh_segmentation(vertices, faces, colors, normals, k=0.03, min_segment_size=50):
    """
    Felzenszwalb-Huttenlocher graph segmentation on a mesh.

    Parameters
    ----------
    vertices : (N, 3) float
    faces    : (F, 3) int
    colors   : (N, 3) float in [0,1]
    normals  : (N, 3) float, unit
    k        : threshold parameter (larger = bigger segments)
    min_segment_size : merge components smaller than this with neighbor

    Returns
    -------
    seg : (N,) int  -- superpoint ID per vertex
    """
    n = vertices.shape[0]

    # Build unique edges
    e1 = faces[:, [0, 1]]
    e2 = faces[:, [1, 2]]
    e3 = faces[:, [2, 0]]
    edges = np.vstack([e1, e2, e3])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    print(f"  {len(edges)} unique edges")

    # Compute edge weights
    n_a = normals[edges[:, 0]]
    n_b = normals[edges[:, 1]]
    normal_diff = 1.0 - np.sum(n_a * n_b, axis=1)  # in [0, 2]
    normal_diff = np.clip(normal_diff, 0, 2)

    c_a = colors[edges[:, 0]]
    c_b = colors[edges[:, 1]]
    color_diff = np.linalg.norm(c_a - c_b, axis=1)  # in [0, sqrt(3)]

    weights = normal_diff + 0.5 * color_diff

    # Sort edges by weight ascending
    order = np.argsort(weights)
    edges = edges[order]
    weights = weights[order]

    # Felzenszwalb segmentation
    uf = UnionFind(n)
    # threshold k / |component|: smaller component gets a more generous threshold

    for i in range(len(edges)):
        a, b = edges[i]
        w = weights[i]
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        tau_a = k / uf.size[ra]
        tau_b = k / uf.size[rb]
        threshold = min(uf.internal_diff[ra] + tau_a, uf.internal_diff[rb] + tau_b)
        if w <= threshold:
            uf.union(a, b, w)

    # Post-merge small segments
    for i in range(len(edges)):
        a, b = edges[i]
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        if uf.size[ra] < min_segment_size or uf.size[rb] < min_segment_size:
            uf.union(a, b, weights[i])

    # Build segment IDs (compact 0..K-1)
    roots = np.array([uf.find(i) for i in range(n)])
    unique_roots, seg = np.unique(roots, return_inverse=True)
    return seg.astype(np.int64), len(unique_roots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=float, default=0.03,
                        help="Felzenszwalb threshold. Larger = bigger superpoints. Default 0.03")
    parser.add_argument("--min_size", type=int, default=50,
                        help="Minimum superpoint size (vertices). Default 50")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading mesh: {mesh_path}")
    # Use trimesh for robust mesh I/O with colors
    mesh = trimesh.load(str(mesh_path), process=False)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    print(f"  vertices: {vertices.shape}, faces: {faces.shape}")

    # Colors
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == len(vertices):
        colors = np.array(mesh.visual.vertex_colors)[:, :3] / 255.0
    else:
        print("  warning: no vertex colors, using zeros")
        colors = np.zeros_like(vertices)

    # Normals via Open3D (faster than trimesh)
    print("Computing normals...")
    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(vertices)
    om.triangles = o3d.utility.Vector3iVector(faces)
    om.compute_vertex_normals()
    normals = np.array(om.vertex_normals)
    print(f"  normals: {normals.shape}")

    print(f"Running Felzenszwalb segmentation (k={args.k}, min_size={args.min_size})...")
    seg, num_segments = felzenszwalb_mesh_segmentation(
        vertices, faces, colors, normals, k=args.k, min_segment_size=args.min_size
    )
    print(f"  {num_segments} superpoints created")

    # Statistics
    _, counts = np.unique(seg, return_counts=True)
    print(f"  Segment size — min: {counts.min()}, mean: {counts.mean():.1f}, max: {counts.max()}, median: {np.median(counts):.0f}")

    # Save as torch tensor (Any3DIS expects torch.load-able file)
    seg_tensor = torch.from_numpy(seg).long()
    torch.save(seg_tensor, str(output_path))
    print(f"Saved superpoints to: {output_path}")


if __name__ == "__main__":
    main()
