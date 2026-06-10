"""
Convert Any3DIS output (3D instance proposals) to REACT3D scene_output format.

Any3DIS output:
    <exp>/mask2d_lifted/<scene_id>.pth
        {'ins': [RLE_mask_per_instance], 'mask2d_bank': [...], 'id_bank': [[frame_idx, ...]], 'obj_bank': [...]}

REACT3D scene_output format:
    scene_output/real/mesh_<idx>/
        mask_mesh_refined.ply              # mesh chunk (vertices + triangles + colors)
        top_k_list.pkl                     # list of dicts with rgb_path, depth, extrinsic, mask, mask_idx

For each Any3DIS proposal:
1. Decode 3D vertex mask
2. Extract submesh (vertices + triangles fully inside mask) → save as mask_mesh_refined.ply
3. For each frame in id_bank[i]:
   - Project 3D vertices to 2D using camera pose + intrinsic (with depth occlusion check)
   - Build 2D mask
   - Pack {rgb_path, depth, extrinsic, mask, mask_idx}
4. Save top_k_list.pkl

Usage:
    python any3dis_to_react3d.py \\
        --any3dis_out /home/gpuadmin/capstone/USDNet/REACT3D/data/any3dis_exp/react3d_run1/mask2d_lifted/react3d_scene.pth \\
        --scene_mesh /home/gpuadmin/capstone/USDNet/REACT3D/data/example_input_scene/mesh_aligned_0.05.ply \\
        --react3d_scene /home/gpuadmin/capstone/USDNet/REACT3D/data/example_input_scene \\
        --output_dir /home/gpuadmin/capstone/USDNet/REACT3D/data/example_input_scene/scene_output_any3dis \\
        --min_vertices 200 \\
        --img_w 960 --img_h 720 \\
        --depth_w 256 --depth_h 192
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh


def rle_decode(rle):
    length = rle["length"]
    s = rle["counts"]
    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


def extract_submesh(vertices, faces, colors, vertex_mask):
    """Extract submesh: only keep triangles where all 3 vertices are inside mask."""
    valid_v = vertex_mask.astype(bool)
    # Triangle is kept if all 3 of its vertices are valid
    tri_valid = valid_v[faces[:, 0]] & valid_v[faces[:, 1]] & valid_v[faces[:, 2]]
    kept_faces = faces[tri_valid]
    if len(kept_faces) == 0:
        return None, None, None

    # Re-index vertices: map old indices to new compact indices
    used_v = np.unique(kept_faces.ravel())
    old2new = -np.ones(len(vertices), dtype=np.int64)
    old2new[used_v] = np.arange(len(used_v))
    new_faces = old2new[kept_faces]
    new_vertices = vertices[used_v]
    new_colors = colors[used_v] if colors is not None else None
    return new_vertices, new_faces, new_colors


def project_vertices_to_2d(verts_world, c2w, K, img_w, img_h):
    """Project world vertices to 2D pixel coords using c2w pose and 3x3 intrinsic.

    Returns:
        uv: (N, 2) float pixel coords
        z_cam: (N,) depth in camera (positive = in front)
        valid: (N,) bool inside image bounds AND in front of camera
    """
    n = verts_world.shape[0]
    homog = np.hstack([verts_world, np.ones((n, 1))])  # (N, 4)
    w2c = np.linalg.inv(c2w)
    cam = (w2c @ homog.T).T  # (N, 4)
    z = cam[:, 2]
    x = cam[:, 0]
    y = cam[:, 1]

    uv = np.zeros((n, 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * x / z + K[0, 2]
    uv[:, 1] = K[1, 1] * y / z + K[1, 2]

    in_front = z > 0
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    valid = in_front & in_bounds
    return uv, z, valid


def build_2d_mask_from_projection(verts_world, faces, c2w, K, depth_map, img_w, img_h,
                                   depth_tolerance=0.15, dilate_px=2):
    """Rasterize mesh triangles to 2D mask using depth-aware projection.

    Approach: for each triangle, project its 3 vertices, then fill the triangle
    in the mask if at least 2/3 vertices are visible (depth-consistent).
    """
    uv, z, valid = project_vertices_to_2d(verts_world, c2w, K, img_w, img_h)
    if valid.sum() == 0:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Depth-occlusion check for each vertex
    vis = valid.copy()
    if depth_map is not None:
        # depth_map is at (img_h, img_w) — depth already resized
        valid_idx = np.where(valid)[0]
        uv_int = uv[valid_idx].astype(np.int32)
        u = np.clip(uv_int[:, 0], 0, img_w - 1)
        v = np.clip(uv_int[:, 1], 0, img_h - 1)
        d_at_pixel = depth_map[v, u]
        z_at_valid = z[valid_idx]
        visible_sub = (d_at_pixel == 0) | (np.abs(z_at_valid - d_at_pixel) < depth_tolerance)
        new_vis = np.zeros_like(valid)
        new_vis[valid_idx] = visible_sub
        vis = new_vis

    # Rasterize triangles where at least 2 of 3 vertices are visible
    if faces is None or len(faces) == 0:
        # Fallback: splat vertices
        uv_int = uv[vis].astype(np.int32)
        if len(uv_int) > 0:
            mask[uv_int[:, 1], uv_int[:, 0]] = 1
    else:
        tri_vis = vis[faces].sum(axis=1) >= 2
        valid_faces = faces[tri_vis]
        if len(valid_faces) > 0:
            uv_int = uv.astype(np.int32)
            for tri in valid_faces:
                pts = uv_int[tri].astype(np.int32)
                cv2.fillConvexPoly(mask, pts, 1)

    # Small dilation + closing to clean edges
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--any3dis_out", required=True)
    parser.add_argument("--scene_mesh", required=True)
    parser.add_argument("--react3d_scene", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_vertices", type=int, default=200,
                        help="Skip proposals with fewer vertices than this")
    parser.add_argument("--img_interval", type=int, default=2,
                        help="Must match Any3DIS img_interval (default 2)")
    parser.add_argument("--img_w", type=int, default=960)
    parser.add_argument("--img_h", type=int, default=720)
    parser.add_argument("--depth_w", type=int, default=256)
    parser.add_argument("--depth_h", type=int, default=192)
    parser.add_argument("--top_k", type=int, default=5, help="Top-K best views per part")
    args = parser.parse_args()

    any3dis_path = Path(args.any3dis_out)
    scene_mesh_path = Path(args.scene_mesh)
    react3d_scene = Path(args.react3d_scene)
    output_dir = Path(args.output_dir)

    # Match REACT3D inference.py expected structure:
    #   <output_dir>/perception/vis_groups_final_mesh/real/mesh_XXX/{mask_mesh_refined.ply, top_k_list.pkl}
    out_real = output_dir / "perception" / "vis_groups_final_mesh" / "real"
    out_real.mkdir(parents=True, exist_ok=True)

    print("Loading Any3DIS output...")
    any3dis_data = torch.load(str(any3dis_path), map_location="cpu")
    proposals = any3dis_data["ins"]
    id_bank = any3dis_data["id_bank"]
    obj_bank = any3dis_data["obj_bank"]
    print(f"  {len(proposals)} proposals")

    print("Loading scene mesh...")
    mesh = trimesh.load(str(scene_mesh_path), process=False)
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == len(vertices):
        colors = np.array(mesh.visual.vertex_colors, dtype=np.uint8)[:, :3]
    else:
        colors = np.full((len(vertices), 3), 128, dtype=np.uint8)
    print(f"  {len(vertices)} vertices, {len(faces)} faces")

    print("Loading pose JSON...")
    pose_json = react3d_scene / "pose_intrinsic_imu.json"
    with open(pose_json) as f:
        pose_data = json.load(f)

    # Build subsampled frame ID list matching Any3DIS's loader.
    # Any3DIS loader: sorted list of stems from depth/ dir, sampled every img_interval.
    depth_dir = react3d_scene / "depth"
    sorted_stems = sorted([p.stem for p in depth_dir.glob("frame_*.png")])
    sampled_stems = sorted_stems[::args.img_interval]
    print(f"  {len(sorted_stems)} total frames, {len(sampled_stems)} sampled")

    # Intrinsic at 960x720 (scaled from original 1920x1440)
    # Compute scale once from first frame
    sample_frame = sorted_stems[0]
    K_orig = np.array(pose_data[sample_frame]["intrinsic"])
    orig_w, orig_h = K_orig[0, 2] * 2, K_orig[1, 2] * 2
    sx_rgb, sy_rgb = args.img_w / orig_w, args.img_h / orig_h
    K_rgb = K_orig.copy()
    K_rgb[0, 0] *= sx_rgb
    K_rgb[1, 1] *= sy_rgb
    K_rgb[0, 2] *= sx_rgb
    K_rgb[1, 2] *= sy_rgb
    print(f"  RGB intrinsic (scaled to {args.img_w}x{args.img_h}):\n{K_rgb}")

    images_dir = react3d_scene / "images_2"

    # Process each proposal
    processed = 0
    skipped = 0
    for i, prop in enumerate(proposals):
        vertex_mask = rle_decode(prop)
        nverts_full = int(vertex_mask.sum())

        if nverts_full < args.min_vertices:
            print(f"  [{i}] skip: only {nverts_full} vertices (< {args.min_vertices})")
            skipped += 1
            continue

        # Extract submesh
        sub_verts, sub_faces, sub_colors = extract_submesh(vertices, faces, colors, vertex_mask)
        if sub_verts is None or len(sub_faces) < 20:
            print(f"  [{i}] skip: extracted submesh has {0 if sub_verts is None else len(sub_faces)} faces (too small)")
            skipped += 1
            continue

        # Find largest connected component (mesh might be split)
        sub_mesh_tm = trimesh.Trimesh(vertices=sub_verts, faces=sub_faces,
                                       vertex_colors=sub_colors, process=False)
        components = sub_mesh_tm.split(only_watertight=False)
        if len(components) > 1:
            comp_sizes = [len(c.vertices) for c in components]
            largest = components[np.argmax(comp_sizes)]
            print(f"  [{i}] split into {len(components)} components, keeping largest ({len(largest.vertices)} verts)")
            sub_mesh_tm = largest

        if len(sub_mesh_tm.vertices) < args.min_vertices:
            print(f"  [{i}] skip: largest component too small")
            skipped += 1
            continue

        # Mesh output directory
        mesh_name = f"mesh_{i:03d}"
        mesh_dir = out_real / mesh_name
        mesh_dir.mkdir(parents=True, exist_ok=True)

        # Save mask_mesh_refined.ply
        mesh_out_path = mesh_dir / "mask_mesh_refined.ply"
        # Use Open3D for proper PLY with both vertex colors and triangles
        om = o3d.geometry.TriangleMesh()
        om.vertices = o3d.utility.Vector3dVector(sub_mesh_tm.vertices)
        om.triangles = o3d.utility.Vector3iVector(sub_mesh_tm.faces)
        if hasattr(sub_mesh_tm.visual, "vertex_colors") and sub_mesh_tm.visual.vertex_colors is not None:
            vc = np.array(sub_mesh_tm.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
            om.vertex_colors = o3d.utility.Vector3dVector(vc)
        o3d.io.write_triangle_mesh(str(mesh_out_path), om)

        verts_for_proj = np.array(sub_mesh_tm.vertices)

        # Find top-K best views by projecting to all sampled frames and counting in-bounds vertices.
        # This is robust even if Any3DIS's id_bank used a different coordinate system.
        coverage_scores = []
        for fidx, stem_x in enumerate(sampled_stems):
            if stem_x not in pose_data:
                continue
            c2w_x = np.array(pose_data[stem_x]["aligned_pose"], dtype=np.float64)
            _, _, valid_x = project_vertices_to_2d(verts_for_proj, c2w_x, K_rgb, args.img_w, args.img_h)
            cov = int(valid_x.sum())
            if cov > 30:
                coverage_scores.append((cov, fidx, stem_x))

        # Sort desc, take top-K
        coverage_scores.sort(reverse=True)
        top_k_views = coverage_scores[: args.top_k]

        if len(top_k_views) == 0:
            print(f"  [{i}] skip: no view sees this mesh region")
            skipped += 1
            import shutil
            shutil.rmtree(mesh_dir)
            continue

        top_k_list = []

        for cov, fidx, frame_stem in top_k_views:

            if frame_stem not in pose_data:
                continue

            # Load depth
            depth_path = react3d_scene / "depth" / f"{frame_stem}.png"
            depth_raw = cv2.imread(str(depth_path), -1)
            if depth_raw is None:
                continue
            depth_m = depth_raw.astype(np.float32) / 1000.0  # meters, 256x192

            # Resize depth to RGB for occlusion check
            depth_rgb = cv2.resize(depth_m, (args.img_w, args.img_h), interpolation=cv2.INTER_NEAREST)

            # Camera-to-world (aligned_pose, same coord as scene mesh)
            c2w = np.array(pose_data[frame_stem]["aligned_pose"], dtype=np.float64)

            # Build 2D mask via mesh triangle rasterization
            mask_2d = build_2d_mask_from_projection(
                verts_for_proj, np.array(sub_mesh_tm.faces), c2w, K_rgb, depth_rgb,
                args.img_w, args.img_h, depth_tolerance=0.15, dilate_px=2
            )

            mask_count = int(mask_2d.sum())
            if mask_count < 30:
                continue

            # Resize everything to (depth_w, depth_h) = (256, 192) — REACT3D inference standard
            target_size = (args.depth_w, args.depth_h)
            depth_small = cv2.resize(depth_m, target_size, interpolation=cv2.INTER_NEAREST)
            mask_small = cv2.resize(mask_2d, target_size, interpolation=cv2.INTER_NEAREST)

            # Save resized RGB in mesh subfolder (REACT3D's inference reads o3d.io.read_image from this path)
            rgb_src = images_dir / f"{frame_stem}.png"
            rgb_dst = mesh_dir / f"rgb_{frame_stem}.png"
            if not rgb_dst.exists():
                img = cv2.imread(str(rgb_src))
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(rgb_dst), img)

            frame_entry = {
                "mask_idx": i,
                "rgb_path": str(rgb_dst),
                "depth": depth_small,
                "extrinsic": c2w.astype(np.float32),
                "mask": mask_small.astype(np.uint8),
            }
            top_k_list.append(frame_entry)

        if len(top_k_list) == 0:
            print(f"  [{i}] skip: no usable frames")
            skipped += 1
            # cleanup
            import shutil
            shutil.rmtree(mesh_dir)
            continue

        # Sort by mask area descending (largest visible = best)
        top_k_list.sort(key=lambda x: int(x["mask"].sum()), reverse=True)

        pkl_path = mesh_dir / "top_k_list.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(top_k_list, f)

        print(f"  [{i}] {mesh_name}: {len(sub_mesh_tm.vertices)} verts, {len(sub_mesh_tm.faces)} faces, {len(top_k_list)} views")
        processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")
    print(f"Output: {out_real}")


if __name__ == "__main__":
    main()
