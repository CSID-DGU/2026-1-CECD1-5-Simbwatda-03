import os
import json
import numpy as np
import open3d as o3d
import time
import viser
import trimesh
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scene_dir', type=str, required=True, help='Path to the output result dir')
parser.add_argument(
    '--show_part_bases',
    action='store_true',
    help='Show old per-part base.obj/inner_box.obj even when shared_bases manifest exists',
)
parser.add_argument(
    '--objects_only',
    action='store_true',
    help='Hide remain_scene and diagnostic OBB/cuboid axes; show only parts and generated object/base meshes',
)
parser.add_argument(
    '--show_debug_axes',
    action='store_true',
    help='Show world axes, part surface axes, shared box axes, and divider axes',
)
parser.add_argument('--debug_axis_scale', type=float, default=0.18)
args = parser.parse_args()

# ================= Configuration Paths =================
result_dir = args.scene_dir

# ================= Helper Functions =================
def create_arrow_from_vector(origin: np.ndarray,
                             vector: np.ndarray,
                             color: tuple = (1.0, 0.0, 0.0),
                             cylinder_radius: float = 0.01,
                             cone_radius: float = 0.02,
                             cylinder_height_ratio: float = 0.8,
                             cone_height_ratio: float = 0.2) -> o3d.geometry.TriangleMesh:
    length = np.linalg.norm(vector)
    if length == 0:
        raise ValueError("Vector length is zero, cannot create arrow.")
    direction = vector / length

    cylinder_height = length * cylinder_height_ratio
    cone_height = length * cone_height_ratio

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height
    )
    arrow.compute_vertex_normals()

    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        if np.dot(z_axis, direction) < 0:
            rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
        else:
            rot_mat = np.eye(3)
    else:
        axis = axis / axis_len
        angle = np.arccos(np.dot(z_axis, direction))
        rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    arrow.rotate(rot_mat, center=(0, 0, 0))
    arrow.translate(origin)
    return arrow

# --- Added: Extract low-level logic for pose updates ---
def apply_joint_transform(frame_handle, joint_type, origin, axis, val):
    """Apply the transformation matrix to the Viser Frame handle"""
    if joint_type == 'rotation':
        rot = R.from_rotvec(axis * val)
        scipy_quat = rot.as_quat() # [x, y, z, w]
        wxyz = (scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2])
        translation = origin - rot.apply(origin)
        
        frame_handle.position = translation
        frame_handle.wxyz = wxyz
        
    elif joint_type == 'translation':
        translation = axis * val
        frame_handle.position = translation
        frame_handle.wxyz = (1.0, 0.0, 0.0, 0.0)

def make_joint_callback(frame_handle, joint_type, origin, axis):
    def callback(event: viser.GuiEvent):
        val = event.target.value
        # Use the extracted logic
        apply_joint_transform(frame_handle, joint_type, origin, axis, val)
    return callback

def o3d_to_trimesh(o3d_mesh, default_color=(200, 200, 200)):
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    
    if o3d_mesh.has_vertex_colors():
        vertex_colors = (np.asarray(o3d_mesh.vertex_colors) * 255).astype(np.uint8)
    else:
        vertex_colors = np.tile(default_color, (len(vertices), 1)).astype(np.uint8)
        
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)


def unit(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def pca_surface_normal(vertices):
    centered = vertices - vertices.mean(axis=0)
    cov = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(cov)
    return unit(eigenvectors[:, 0])


def surface_frame_from_vertices(vertices, normal_hint=None):
    normal = pca_surface_normal(vertices)
    if normal_hint is not None and np.dot(normal, normal_hint) < 0:
        normal *= -1.0
    vertical = np.array([0.0, 0.0, 1.0])
    vertical = vertical - np.dot(vertical, normal) * normal
    if np.linalg.norm(vertical) < 1e-4:
        vertical = np.array([0.0, 1.0, 0.0])
        vertical = vertical - np.dot(vertical, normal) * normal
    vertical = unit(vertical)
    horizontal = unit(np.cross(vertical, normal))
    return normal, horizontal, vertical


def add_debug_arrow(server, name, origin, vector, color, radius=0.005):
    if np.linalg.norm(vector) < 1e-8:
        return
    arrow = create_arrow_from_vector(
        np.asarray(origin, dtype=float),
        np.asarray(vector, dtype=float),
        color=color,
        cylinder_radius=radius,
        cone_radius=radius * 2.2,
    )
    server.scene.add_mesh_trimesh(
        name=name,
        mesh=o3d_to_trimesh(arrow, default_color=tuple(int(c * 255) for c in color)),
    )


def add_axis_tripod(server, prefix, origin, normal, horizontal, vertical, axis_len):
    add_debug_arrow(server, f"{prefix}/normal_depth_red", origin, np.asarray(normal) * axis_len, (1.0, 0.1, 0.1))
    add_debug_arrow(server, f"{prefix}/horizontal_green", origin, np.asarray(horizontal) * axis_len, (0.1, 0.9, 0.1))
    add_debug_arrow(server, f"{prefix}/vertical_blue", origin, np.asarray(vertical) * axis_len, (0.1, 0.2, 1.0))


def world_from_module_local(module, x, y, z):
    normal = np.asarray(module["normal"], dtype=float)
    horizontal = np.asarray(module["horizontal"], dtype=float)
    vertical = np.asarray(module["vertical"], dtype=float)
    return module["front_offset"] * normal + x * normal + y * horizontal + z * vertical


def load_shared_base_manifest(scene_dir):
    manifest_path = Path(scene_dir) / "shared_bases" / "shared_base_groups.json"
    if not manifest_path.exists():
        return None, {}
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest, manifest.get("part_to_group", {})


# ================= Main Program =================
def main():
    server = viser.ViserServer()
    print("Viser server started! Please open http://localhost:8080 in your browser.")

    if args.show_debug_axes:
        add_debug_arrow(server, "/debug/world/x_red", np.zeros(3), np.array([1.0, 0.0, 0.0]) * 0.5, (1.0, 0.0, 0.0))
        add_debug_arrow(server, "/debug/world/y_green", np.zeros(3), np.array([0.0, 1.0, 0.0]) * 0.5, (0.0, 0.9, 0.0))
        add_debug_arrow(server, "/debug/world/z_blue", np.zeros(3), np.array([0.0, 0.0, 1.0]) * 0.5, (0.0, 0.1, 1.0))
    
    # 1. Render Remain Scene (Background)
    remain_scene_path = os.path.join(result_dir, "remain_scene.ply")
    if not args.objects_only and os.path.exists(remain_scene_path):
        remain_scene = o3d.io.read_triangle_mesh(remain_scene_path)
        if remain_scene.has_vertices():
            remain_trimesh = o3d_to_trimesh(remain_scene)
            server.scene.add_mesh_trimesh(
                name="/scene/remain",
                mesh=remain_trimesh
            )

    # 1.5 Render shared static interiors generated by
    # simulation_ready/generate_shared_base_groups.py. For grouped members we
    # hide the old per-part base by default to avoid duplicated interiors.
    shared_manifest, part_to_shared_group = load_shared_base_manifest(result_dir)
    if shared_manifest is not None:
        for group in shared_manifest.get("groups", []):
            shared_mesh_path = Path(result_dir) / group.get("mesh", "")
            if not shared_mesh_path.exists():
                continue
            shared_mesh = o3d.io.read_triangle_mesh(str(shared_mesh_path))
            if shared_mesh.is_empty():
                continue
            shared_trimesh = o3d_to_trimesh(shared_mesh, default_color=(150, 150, 150))
            server.scene.add_mesh_trimesh(
                name=f"/scene/shared_bases/{group.get('id', shared_mesh_path.stem)}",
                mesh=shared_trimesh,
            )
            if args.show_debug_axes:
                for module in group.get("modules", []):
                    normal = np.asarray(module["normal"], dtype=float)
                    horizontal = np.asarray(module["horizontal"], dtype=float)
                    vertical = np.asarray(module["vertical"], dtype=float)
                    rect = module["rect"]
                    y_mid = (rect[0] + rect[1]) * 0.5
                    z_mid = (rect[2] + rect[3]) * 0.5
                    center = world_from_module_local(module, module["depth"] * 0.5, y_mid, z_mid)
                    axis_len = max(args.debug_axis_scale, min(module["depth"], module["width"], module["height"]) * 0.45)
                    module_prefix = f"/debug/shared_box_axes/{group.get('id', shared_mesh_path.stem)}/{module['id']}"
                    add_axis_tripod(server, module_prefix, center, normal, horizontal, vertical, axis_len)

                    for div_i, y in enumerate(module.get("divider_y", [])):
                        origin = world_from_module_local(module, module["depth"] * 0.5, y, z_mid)
                        prefix = f"/debug/divider_axes/{group.get('id', shared_mesh_path.stem)}/{module['id']}/divider_y_{div_i}"
                        add_debug_arrow(server, f"{prefix}/span_depth_red", origin, normal * axis_len, (1.0, 0.1, 0.1), radius=0.004)
                        add_debug_arrow(server, f"{prefix}/wall_normal_yellow", origin, horizontal * axis_len, (1.0, 0.9, 0.0), radius=0.004)
                        add_debug_arrow(server, f"{prefix}/span_vertical_blue", origin, vertical * axis_len, (0.1, 0.2, 1.0), radius=0.004)

                    for div_i, z in enumerate(module.get("divider_z", [])):
                        origin = world_from_module_local(module, module["depth"] * 0.5, y_mid, z)
                        prefix = f"/debug/divider_axes/{group.get('id', shared_mesh_path.stem)}/{module['id']}/divider_z_{div_i}"
                        add_debug_arrow(server, f"{prefix}/span_depth_red", origin, normal * axis_len, (1.0, 0.1, 0.1), radius=0.004)
                        add_debug_arrow(server, f"{prefix}/span_horizontal_green", origin, horizontal * axis_len, (0.1, 0.9, 0.1), radius=0.004)
                        add_debug_arrow(server, f"{prefix}/wall_normal_yellow", origin, vertical * axis_len, (1.0, 0.9, 0.0), radius=0.004)

    gui_folder = server.gui.add_folder("Joint Controls")
    gui_sliders = []
    
    # --- Added: Save information of all movable parts for animation export ---
    parts_info = []

    # 2. Iterate through all Parts
    for subdir in sorted(os.listdir(result_dir)):
        subdir_path = os.path.join(result_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        ply_file = os.path.join(subdir_path, "part.ply")
        base_obj_file = os.path.join(subdir_path, "base.obj")
        base_ply_file = os.path.join(subdir_path, "base.ply")
        obb_path = os.path.join(subdir_path, "obb_world.npy")
        cuboid_path = os.path.join(subdir_path, "cuboid_world.npy")
        inner_box_v3 = os.path.join(subdir_path, "inner_box_v3.obj")
        inner_box_orig = os.path.join(subdir_path, "inner_box.obj")
        inner_box_file = inner_box_v3 if os.path.exists(inner_box_v3) else inner_box_orig

        joint_type_path = os.path.join(subdir_path, 'articulation_type.npy')
        origin_world_path = os.path.join(subdir_path, 'origin_world.npy')
        vector_world_path = os.path.join(subdir_path, 'vector_world.npy')

        if os.path.exists(ply_file):
            mesh = o3d.io.read_triangle_mesh(ply_file)

            if mesh.is_empty():
                continue

            # --- A. Add Base: original cuboid-shell base.obj (paper approach) ---
            show_static_part_base = args.show_part_bases or (
                not args.objects_only and subdir not in part_to_shared_group
            )
            if show_static_part_base and os.path.exists(base_obj_file):
                base_mesh = o3d.io.read_triangle_mesh(base_obj_file)
                if not base_mesh.is_empty():
                    base_trimesh = o3d_to_trimesh(base_mesh, default_color=(180, 180, 180))
                    server.scene.add_mesh_trimesh(
                        name=f"/scene/{subdir}/base",
                        mesh=base_trimesh
                    )

            # --- A2. Visualize OBB axes (from obb_world.npy corner 0) and
            #        cuboid_world axes (from cuboid_world.npy corner 0).
            #        Goal: verify whether OBB is parallel to part surface and
            #        cuboid frame coincides with OBB. ---
            def _add_axes(obj_path, name_prefix, axis_len_scale=0.6, dim_ratio=False):
                """Draw 3 arrows from corner 0 along the 3 OBB edges.
                Colors: edge a = red, b = green, c = blue."""
                if not os.path.exists(obj_path):
                    return
                corners = np.load(obj_path)
                if corners.shape != (8, 3):
                    return
                origin_pt = corners[0]
                edges = [
                    (corners[1] - corners[0], (1.0, 0.2, 0.2)),  # a — red
                    (corners[3] - corners[0], (0.2, 1.0, 0.2)),  # b — green
                    (corners[4] - corners[0], (0.2, 0.2, 1.0)),  # c — blue
                ]
                for ax_i, (vec, col) in enumerate(edges):
                    L = np.linalg.norm(vec)
                    if L < 1e-6:
                        continue
                    arrow = create_arrow_from_vector(
                        origin_pt, vec * axis_len_scale, color=col,
                        cylinder_radius=0.006, cone_radius=0.014,
                    )
                    arrow_trimesh = o3d_to_trimesh(arrow,
                                                   default_color=tuple(int(c * 255) for c in col))
                    server.scene.add_mesh_trimesh(
                        name=f"/scene/{subdir}/{name_prefix}_axis_{ax_i}",
                        mesh=arrow_trimesh,
                    )
            # if not args.objects_only:
            #     _add_axes(obb_path, "obb")
            #     # cuboid axes drawn slightly shorter so both are visible
            #     _add_axes(cuboid_path, "cuboid", axis_len_scale=0.4)

            # --- B. Create Frame for Part and mount Mesh and Inner Box ---
            part_frame = server.scene.add_frame(
                name=f"/scene/{subdir}/part_frame",
                show_axes=False 
            )

            # part_frame = server.scene.add_frame(
            #     name=f"/my_group",
            #     show_axes=False 
            # )
            
            # B.1 Mount the main Part Mesh
            part_trimesh = o3d_to_trimesh(mesh)
            server.scene.add_mesh_trimesh(
                name=f"/scene/{subdir}/part_frame/mesh",
                mesh=part_trimesh
            )

            if args.show_debug_axes:
                vertices = np.asarray(mesh.vertices)
                normal_hint = None
                if os.path.exists(cuboid_path):
                    cuboid = np.load(cuboid_path)
                    if cuboid.shape == (8, 3):
                        normal_hint = cuboid[1] - cuboid[0]
                normal, horizontal, vertical = surface_frame_from_vertices(vertices, normal_hint)
                center = vertices.mean(axis=0)
                extent = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
                axis_len = max(args.debug_axis_scale * 0.6, min(args.debug_axis_scale, extent * 0.25))
                add_axis_tripod(
                    server,
                    f"/debug/surface_axes/{subdir}",
                    center,
                    normal,
                    horizontal,
                    vertical,
                    axis_len,
                )

            # B.2 Mount Inner Box (if exists) — attach to /scene/{subdir} (BASE level),
            # NOT under part_frame, so it stays fixed when the part rotates/translates.
            if show_static_part_base and os.path.exists(inner_box_file):
                inner_box_mesh = o3d.io.read_triangle_mesh(inner_box_file)
                if not inner_box_mesh.is_empty():
                    inner_box_trimesh = o3d_to_trimesh(inner_box_mesh, default_color=(100, 100, 100))
                    server.scene.add_mesh_trimesh(
                        name=f"/scene/{subdir}/inner_box",
                        mesh=inner_box_trimesh
                    )

            # --- C. Parse joints and establish GUI control interaction ---
            if os.path.isfile(origin_world_path) and os.path.isfile(vector_world_path):
                origin_world = np.load(origin_world_path)
                vec = np.load(vector_world_path)
                norm = np.linalg.norm(vec)
                vector_world = vec / norm if norm > 0 else vec
                joint_type = str(np.load(joint_type_path).item())

                # --- Read Range Max ---
                range_max_path = os.path.join(subdir_path, 'range_max.npy')
                if os.path.isfile(range_max_path):
                    motion_range = float(np.load(range_max_path).item()) if joint_type == 'rotation' else 0.5
                else:
                    motion_range = np.pi / 2 if joint_type == 'rotation' else 0.5

                arrow_color = (0.8, 0.1, 0.1) if joint_type == 'rotation' else (0.0, 0.0, 1.0)
                arrow_mesh = create_arrow_from_vector(origin_world, vector_world, color=arrow_color)
                
                # uncomment this if you want to see the axis of the joint
                # server.scene.add_mesh_simple(
                #     name=f"/scene/{subdir}/arrow",
                #     vertices=np.asarray(arrow_mesh.vertices),
                #     faces=np.asarray(arrow_mesh.triangles),
                #     color=tuple((np.array(arrow_color) * 255).astype(int))
                # )

                with gui_folder:
                    if joint_type == 'rotation':
                        slider = server.gui.add_slider(
                            f"Rot: {subdir}", min=0.0, max=motion_range, step=0.01, initial_value=0.0
                        )
                    else:  
                        slider = server.gui.add_slider(
                            f"Trans: {subdir}", min=0.0, max=motion_range, step=0.01, initial_value=0.0
                        )
                        
                    slider.on_update(make_joint_callback(part_frame, joint_type, origin_world, vector_world))
                    gui_sliders.append(slider)
                    
                    # --- Added: Record physical and animation info for this part ---
                    parts_info.append({
                        'frame': part_frame,
                        'type': joint_type,
                        'origin': origin_world,
                        'axis': vector_world,
                        'max_val': motion_range # Set the upper limit for the animation amplitude
                    })

    # 3. Add Reset Button
    with gui_folder:
        reset_btn = server.gui.add_button("Reset All Joints")
        @reset_btn.on_click
        def _(_):
            for s in gui_sliders:
                s.value = 0.0

        # --- Added: Add export animation button ---
        export_btn = server.gui.add_button("Export Animation (.viser)")
        @export_btn.on_click
        def _(_):
            print("Starting serialization... Generating animation.")
            serializer = server.get_scene_serializer()
            num_frames = 100
            
            # Loop to generate animation frames
            for t in range(num_frames):
                # Use a periodic function for smooth reciprocating animation
                phase = (t / num_frames) * 2 * np.pi
                
                for info in parts_info:
                    # Calculate the joint value for the current frame (moves between 0 and max_val)
                    val = (1.0 - np.cos(phase)) / 2.0 * info['max_val']
                    # Apply low-level transform logic to update this frame
                    apply_joint_transform(info['frame'], info['type'], info['origin'], info['axis'], val)
                
                # Insert sleep to control the exported animation speed (e.g., 30fps)
                serializer.insert_sleep(1.0 / 30.0)

            # Save the generated animation file
            data = serializer.serialize()
            save_path = Path(result_dir) / "recording.viser"
            save_path.write_bytes(data)
            print(f"Animation saved successfully to: {save_path}")
            
            # Restore scene after export
            for s in gui_sliders:
                s.value = 0.0
            for info in parts_info:
                apply_joint_transform(info['frame'], info['type'], info['origin'], info['axis'], 0.0)


    # Keep the main thread running
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
