import open3d as o3d
import numpy as np
import os
import pickle


EPS = 1e-8


def find_largest_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Run RANSAC plane segmentation on the point cloud.
    Returns plane_model and inlier indices.
    """
    model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations)
    return model, inliers


def compute_camera_direction(extrinsic_c2w):
    """
    Given a 4x4 extrinsic_c2w matrix (camera->world), compute the camera's viewing direction
    in world coordinates assuming the camera looks along its local +Z axis.
    Also returns camera position in world coords.
    """
    E = np.array(extrinsic_c2w).reshape(4, 4)
    R_c2w = E[:3, :3]
    t_c2w = E[:3, 3]
    camera_dir = R_c2w[:, 2]
    camera_dir = camera_dir / np.linalg.norm(camera_dir)
    camera_pos = t_c2w
    return camera_dir, camera_pos


def find_largest_vertical_plane(ply_path,
                                extrinsic_c2w,
                                distance_threshold=0.01,
                                plane_band=0.02,
                                ransac_n=3,
                                num_iterations=1000,
                                vertical_threshold_deg=10):
    """
    Load PLY, estimate normals, filter for vertical plane candidates,
    fit the largest plane, then split points into plane_cloud and rest_cloud
    depending on the camera direction and positive normal of the plane.

    extrinsic_c2w: 4x4 camera-to-world matrix

    Returns:
    - plane_cloud: open3d.geometry.PointCloud of plane points
    - plane_model: [a, b, c, d] with normal pointing toward camera
    - rest_cloud: open3d.geometry.PointCloud of points beyond the plane along positive normal
    - camera_dir: normalized viewing direction
    - camera_pos: position in world coords
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        raise ValueError(f"No points in {ply_path}")

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)

    vertical = np.array([0, 0, 1])
    cos_max = np.sin(np.deg2rad(vertical_threshold_deg))
    mask = np.abs(normals.dot(vertical)) <= cos_max
    candidate_idx = np.where(mask)[0].tolist()
    if len(candidate_idx) < ransac_n:
        return None, None, None, None, None, None
    candidate_cloud = pcd.select_by_index(candidate_idx)

    model, inliers_sub = find_largest_plane(
        candidate_cloud,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations)
    inliers = [candidate_idx[i] for i in inliers_sub]

    camera_dir, camera_pos = compute_camera_direction(extrinsic_c2w)

    n = np.array(model[:3])
    norm_n = np.linalg.norm(n)
    n_normed = n / norm_n
    d_normed = model[3] / norm_n
    if np.dot(n_normed, camera_dir) < 0:
        n_normed = -n_normed
        d_normed = -d_normed

    pts = np.asarray(pcd.points)
    signed = pts.dot(n_normed) + d_normed

    # Keep only a true band around the fitted plane. The old half-space rule
    # kept everything before the deepest RANSAC inlier, which lets small islands
    # and side/handle faces inflate the downstream raw min/max box.
    plane_idx = np.where(np.abs(signed) <= plane_band)[0].tolist()
    rest_idx = np.where(signed > plane_band)[0].tolist()

    plane_cloud = pcd.select_by_index(plane_idx)
    rest_cloud = pcd.select_by_index(rest_idx)

    plane_model = [*n_normed.tolist(), d_normed]
    return plane_cloud, plane_model, rest_cloud, camera_dir, camera_pos, plane_idx


def _component_summary(mesh):
    if mesh.is_empty() or len(mesh.triangles) == 0:
        return 0, 0.0, 0.0
    clusters, _, areas = mesh.cluster_connected_triangles()
    areas = np.asarray(areas, dtype=np.float64)
    if areas.size == 0:
        return 0, 0.0, 0.0
    total = max(float(areas.sum()), EPS)
    order = np.argsort(-areas)
    largest = float(areas[order[0]] / total)
    second = float(areas[order[1]] / total) if order.size > 1 else 0.0
    return int(areas.size), largest, second


def _mesh_area(mesh):
    if mesh.is_empty() or len(mesh.triangles) == 0:
        return 0.0
    return float(mesh.get_surface_area())


def _rect_stats(mesh):
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.shape[0] < 3:
        return 0.0, 0.0, 0.0
    centered = verts - verts.mean(axis=0)
    cov = np.cov(centered.T)
    _, axes = np.linalg.eigh(cov)
    normal = axes[:, 0]
    axis_u = axes[:, 2]
    axis_v = axes[:, 1]
    u = verts @ axis_u
    v = verts @ axis_v
    n = verts @ normal
    thickness = float(np.percentile(n, 99.5) - np.percentile(n, 0.5))
    return float(u.max() - u.min()), float(v.max() - v.min()), thickness


def _format_stats(label, mesh):
    comps, largest, second = _component_summary(mesh)
    width, height, thickness = _rect_stats(mesh)
    return (
        f"{label}: V={len(mesh.vertices)} F={len(mesh.triangles)} "
        f"area={_mesh_area(mesh):.4f} comps={comps} "
        f"largest={largest * 100:.1f}% second={second * 100:.1f}% "
        f"rect={width:.3f}x{height:.3f} thick={thickness:.3f}"
    )


def _filter_connected_components(mesh, mode="area_ranked", cumulative_area=0.97, min_area_ratio=0.01):
    if mode == "all" or mesh.is_empty() or len(mesh.triangles) == 0:
        return mesh

    clusters, _, areas = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    areas = np.asarray(areas, dtype=np.float64)
    if areas.size <= 1:
        return mesh

    total = max(float(areas.sum()), EPS)
    order = np.argsort(-areas)
    keep_clusters = []
    if mode == "largest":
        keep_clusters = [int(order[0])]
    elif mode == "area_ranked":
        cumulative = 0.0
        for cluster_id in order:
            area_ratio = float(areas[cluster_id] / total)
            if not keep_clusters or area_ratio >= min_area_ratio or cumulative < cumulative_area:
                keep_clusters.append(int(cluster_id))
                cumulative += area_ratio
            if cumulative >= cumulative_area and area_ratio < min_area_ratio:
                break
    else:
        raise ValueError(f"Unknown component keep mode: {mode}")

    triangle_keep = np.isin(clusters, np.asarray(keep_clusters, dtype=clusters.dtype))
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_triangles_by_mask((~triangle_keep).tolist())
    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out


def extract_and_save_plane_mesh(orig_mesh_path,
                                plane_idx,
                                plane_model,
                                plane_band,
                                save_path,
                                face_keep_mode="all_vertices",
                                component_keep="area_ranked",
                                component_area_ratio=0.97,
                                component_min_area_ratio=0.01,
                                debug_stats=False):
    """
    Extract the sub-mesh corresponding to plane_idx vertices from the original triangle mesh (including all triangles where all three vertices are in plane_idx),
    and save it as PLY.
    """
    # 1. Read original mesh
    mesh = o3d.io.read_triangle_mesh(orig_mesh_path)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    has_colors = False
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        has_colors = True

    plane_idx = np.asarray(plane_idx, dtype=np.int64)
    if plane_idx.size == 0:
        print(f"[Warn] No plane vertices selected for {orig_mesh_path}. Skipping save.")
        return None

    # 2. Find triangles that belong to the front plane. The default keeps the
    # strict previous behavior; other modes are useful for diagnosing boundary
    # holes caused by slightly off-plane vertices.
    keep_vertex = np.zeros(len(verts), dtype=bool)
    keep_vertex[plane_idx] = True
    vertex_in_band = keep_vertex[tris]
    if face_keep_mode == "all_vertices":
        mask = vertex_in_band.all(axis=1)
    elif face_keep_mode == "majority_vertices":
        mask = vertex_in_band.sum(axis=1) >= 2
    elif face_keep_mode in {"centroid", "centroid_and_majority"}:
        normal = np.asarray(plane_model[:3], dtype=np.float64)
        offset = float(plane_model[3])
        tri_centroids = verts[tris].mean(axis=1)
        centroid_in_band = np.abs(tri_centroids.dot(normal) + offset) <= plane_band
        if face_keep_mode == "centroid":
            mask = centroid_in_band
        else:
            mask = centroid_in_band & (vertex_in_band.sum(axis=1) >= 2)
    else:
        raise ValueError(f"Unknown face keep mode: {face_keep_mode}")

    tris_sel = tris[mask]
    if tris_sel.shape[0] == 0:
        print(f"[Warn] No plane faces selected for {orig_mesh_path}. Skipping save.")
        return None

    # 3. Renumber only vertices referenced by selected triangles. This lets
    # centroid/majority modes preserve boundary triangles whose one vertex is
    # slightly outside the plane band without carrying unrelated loose vertices.
    used_vertices, inverse = np.unique(tris_sel.reshape(-1), return_inverse=True)
    tris_new = inverse.reshape(-1, 3).astype(np.int32)

    # 4. Build sub-mesh and save
    mesh_plane = o3d.geometry.TriangleMesh()
    mesh_plane.vertices = o3d.utility.Vector3dVector(verts[used_vertices])
    mesh_plane.triangles = o3d.utility.Vector3iVector(tris_new)
    mesh_plane.compute_vertex_normals()
    if has_colors:
        mesh_plane.vertex_colors = o3d.utility.Vector3dVector(colors[used_vertices])

    before_filter = mesh_plane
    mesh_plane = _filter_connected_components(
        mesh_plane,
        mode=component_keep,
        cumulative_area=component_area_ratio,
        min_area_ratio=component_min_area_ratio,
    )

    if debug_stats:
        raw_summary = _format_stats("raw", mesh)
        band_summary = _format_stats("plane_band", before_filter)
        clean_summary = _format_stats("clean", mesh_plane)
        print(f"  {raw_summary}")
        print(f"  {band_summary}")
        print(f"  {clean_summary}")

    o3d.io.write_triangle_mesh(save_path, mesh_plane)
    print(f"Saved plane mesh: {save_path}")
    return mesh_plane

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find and split the largest vertical plane and visualize normal & camera directions.")
    parser.add_argument("--data_dir", "-d", required=True, help="data directory")
    parser.add_argument("--threshold", "-t", type=float, default=0.02,
                        help="Max distance threshold for plane fitting")
    parser.add_argument("--plane_band", type=float, default=0.02,
                        help="Half-band in meters for keeping true plane inliers after RANSAC")
    parser.add_argument("--iterations", "-n", type=int, default=1000,
                        help="Number of RANSAC iterations")
    parser.add_argument("-v", "--vertical_thresh", type=float, default=45,
                        help="Max deviation deg from vertical")
    parser.add_argument("--output_name", default="mask_mesh_refined_clean.ply",
                        help="Output mesh name. Defaults to non-destructive clean candidate.")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite mask_mesh_refined.ply instead of writing output_name.")
    parser.add_argument("--face_keep_mode",
                        choices=["all_vertices", "majority_vertices", "centroid", "centroid_and_majority"],
                        default="all_vertices",
                        help="Triangle selection rule after plane-band fitting")
    parser.add_argument("--component_keep", choices=["all", "largest", "area_ranked"],
                        default="area_ranked",
                        help="Connected-component filtering strategy after plane-band extraction")
    parser.add_argument("--component_area_ratio", type=float, default=0.97,
                        help="For area_ranked, keep components until this cumulative area ratio")
    parser.add_argument("--component_min_area_ratio", type=float, default=0.01,
                        help="For area_ranked, keep any component above this area ratio")
    parser.add_argument("--debug_stats", action="store_true",
                        help="Print before/after component, area, bbox and thickness stats")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional mesh folder names to process, e.g. --only mesh_030 mesh_218")
    args = parser.parse_args()

    data_dir = args.data_dir
    real_dir = os.path.join(data_dir, 'perception', 'vis_groups_final_mesh', 'real')

    for sub_dir in os.listdir(real_dir):
        sub_path = os.path.join(real_dir, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        if args.only is not None and sub_dir not in set(args.only):
            continue
        
        print(f"Processing {sub_path}...")
        ply_path = os.path.join(sub_path, 'mask_mesh.ply')
        top_k_path = os.path.join(sub_path, 'top_k_list.pkl')
        new_ply_name = 'mask_mesh_refined.ply' if args.inplace else args.output_name
        new_ply_path = os.path.join(sub_path, new_ply_name)

        with open(top_k_path, 'rb') as f:
            top_k_list = pickle.load(f)
        extrinsic_c2w = top_k_list[0]["extrinsic"]
        print("frame: ", top_k_list[0]["frame_name"])

        plane_cloud, plane_model, rest_cloud, camera_dir, camera_pos, plane_idx = find_largest_vertical_plane(
            ply_path,
            extrinsic_c2w,
            distance_threshold=args.threshold,
            plane_band=args.plane_band,
            ransac_n=3,
            num_iterations=args.iterations,
            vertical_threshold_deg=args.vertical_thresh
        )
        
        if plane_cloud is None:
            print(f"No valid plane found in {ply_path}. Skipping...")
            continue

        extract_and_save_plane_mesh(
            ply_path,
            plane_idx,
            plane_model,
            args.plane_band,
            new_ply_path,
            face_keep_mode=args.face_keep_mode,
            component_keep=args.component_keep,
            component_area_ratio=args.component_area_ratio,
            component_min_area_ratio=args.component_min_area_ratio,
            debug_stats=args.debug_stats,
        )

        plane_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # red
        rest_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # gray

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Plane normal arrow (red)
        n = np.array(plane_model[:3])
        normal_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005,
            cone_radius=0.01,
            cylinder_height=0.2,
            cone_height=0.05)
        # Align
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, n)
        c = np.dot(z_axis, n)
        v_mat = np.array([[    0, -v[2],  v[1]],
                          [ v[2],     0, -v[0]],
                          [-v[1],  v[0],    0]])
        Rn = np.eye(3) + v_mat + v_mat.dot(v_mat) * ((1 - c) / (np.linalg.norm(v)**2 + 1e-8))
        normal_arrow.rotate(Rn, center=np.zeros(3))
        centroid = np.mean(np.asarray(plane_cloud.points), axis=0)
        normal_arrow.translate(centroid)
        normal_arrow.paint_uniform_color([1.0, 0.0, 0.0])

        # Camera direction arrow (green)
        cam_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005,
            cone_radius=0.01,
            cylinder_height=0.2,
            cone_height=0.05)
        v2 = np.cross(z_axis, camera_dir)
        c2 = np.dot(z_axis, camera_dir)
        v2_mat = np.array([[    0, -v2[2],  v2[1]],
                           [ v2[2],     0, -v2[0]],
                           [-v2[1],  v2[0],    0]])
        Rc = np.eye(3) + v2_mat + v2_mat.dot(v2_mat) * ((1 - c2) / (np.linalg.norm(v2)**2 + 1e-8))
        cam_arrow.rotate(Rc, center=np.zeros(3))
        cam_arrow.translate(camera_pos)
        cam_arrow.paint_uniform_color([0.0, 1.0, 0.0])

        # # Visualize
        # o3d.visualization.draw_geometries([
        #     plane_cloud,
        #     rest_cloud,
        #     axis,
        #     normal_arrow,
        #     cam_arrow
        # ],
        # window_name="Plane & Camera Directions",
        # width=800,
        # height=600)
