"""
Generate base body mesh by extruding part surface backward along PCA surface normal.

Instead of OBB shell (corners_to_mesh), this:
1. Computes PCA surface normal from part.ply (most accurate for flat surfaces)
2. Projects part vertices onto the surface plane to get exact 2D boundary
3. Extrudes backward along surface normal using cuboid_world depth
4. Result: base body that is exactly part-sized, perpendicular to surface

Usage:
    python generate_base_from_surface.py --scene_dir <scene_output_filtered>
"""
import argparse
import numpy as np
import trimesh
from pathlib import Path


def compute_surface_normal_pca(vertices):
    """Compute surface normal from PCA (smallest eigenvector = thinnest direction)."""
    centered = vertices - vertices.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # smallest eigenvalue = surface normal direction
    normal = eigenvectors[:, 0]
    return normal / np.linalg.norm(normal)


def get_depth_and_direction_from_cuboid(cuboid_corners, surface_normal, part_vertices):
    """Extract depth AND correct direction from cuboid_world.

    Returns (depth, signed_normal) where signed_normal points behind the surface
    (in the direction cuboid extends away from the part).
    """
    center = part_vertices.mean(axis=0)
    p_proj = (part_vertices - center).dot(surface_normal)
    c_proj = (cuboid_corners - center).dot(surface_normal)

    p_min, p_max = p_proj.min(), p_proj.max()
    c_min, c_max = c_proj.min(), c_proj.max()

    # Cuboid extends from part surface to one side
    extend_pos = c_max - p_max  # how far cuboid extends in +normal beyond part
    extend_neg = p_min - c_min  # how far cuboid extends in -normal beyond part

    if extend_pos > extend_neg:
        # Cuboid extends in +normal → "behind" is +normal direction
        return float(extend_pos), surface_normal
    else:
        # Cuboid extends in -normal → "behind" is -normal direction → flip
        return float(extend_neg), -surface_normal


def build_base_mesh(part_vertices, surface_normal, depth, part_colors=None, thickness=0.003):
    """
    Build base body mesh by extruding part surface boundary backward.

    The mesh consists of:
    - Back wall (flat quad at depth behind surface)
    - Side walls connecting part boundary edges to back wall
    - NO front face (that's where the part/door sits)

    For simplicity and robustness, we use an OBB in the surface plane
    (aligned to PCA axes) rather than the convex hull boundary.
    """
    center = part_vertices.mean(axis=0)
    normal = surface_normal / np.linalg.norm(surface_normal)

    # Build surface-plane axes that are world-aligned
    # Pick the two world axes least aligned with the surface normal
    world_axes = np.eye(3)  # X, Y, Z
    dots = np.abs(world_axes.dot(normal))
    order = np.argsort(dots)  # ascending: most perpendicular first

    # axis1: project the most-perpendicular world axis onto the surface plane
    raw_axis1 = world_axes[order[0]]
    axis1 = raw_axis1 - raw_axis1.dot(normal) * normal
    axis1 = axis1 / np.linalg.norm(axis1)

    # axis2: cross product completes a right-handed frame
    axis2 = np.cross(normal, axis1)
    axis2 = axis2 / np.linalg.norm(axis2)

    # Project part vertices onto surface plane axes to get 2D extents
    centered = part_vertices - center
    proj1 = centered.dot(axis1)
    proj2 = centered.dot(axis2)
    proj_n = centered.dot(normal)

    min1, max1 = proj1.min(), proj1.max()
    min2, max2 = proj2.min(), proj2.max()

    # `normal` is already oriented to point BEHIND the surface (into wall/cabinet).
    # Front face = where part sits (max projection along normal, since normal points back).
    # Back face = front + depth (further along normal direction).
    front_offset = proj_n.max()
    back_offset = front_offset + depth

    # 4 corners of front rectangle (on part surface plane)
    # These are in world coordinates
    front_corners = np.array([
        center + min1 * axis1 + min2 * axis2 + front_offset * normal,  # 0: bottom-left
        center + max1 * axis1 + min2 * axis2 + front_offset * normal,  # 1: bottom-right
        center + max1 * axis1 + max2 * axis2 + front_offset * normal,  # 2: top-right
        center + min1 * axis1 + max2 * axis2 + front_offset * normal,  # 3: top-left
    ])

    # 4 corners of back rectangle
    back_corners = np.array([
        center + min1 * axis1 + min2 * axis2 + back_offset * normal,   # 4
        center + max1 * axis1 + min2 * axis2 + back_offset * normal,   # 5
        center + max1 * axis1 + max2 * axis2 + back_offset * normal,   # 6
        center + min1 * axis1 + max2 * axis2 + back_offset * normal,   # 7
    ])

    vertices = np.vstack([front_corners, back_corners])  # (8, 3)

    # Build faces (no front face - that's where the door/drawer is)
    faces = []

    # Back wall (4,5,6,7)
    faces.append([4, 5, 6])
    faces.append([4, 6, 7])

    # Bottom wall (0,1,5,4)
    faces.append([0, 1, 5])
    faces.append([0, 5, 4])

    # Top wall (3,2,6,7)
    faces.append([3, 2, 6])
    faces.append([3, 6, 7])

    # Left wall (0,3,7,4)
    faces.append([0, 3, 7])
    faces.append([0, 7, 4])

    # Right wall (1,2,6,5)
    faces.append([1, 2, 6])
    faces.append([1, 6, 5])

    faces = np.array(faces, dtype=np.int32)

    # Double-sided rendering
    flipped = faces[:, ::-1]
    all_faces = np.vstack([faces, flipped])

    # Color
    if part_colors is not None and len(part_colors) > 0:
        avg_color = np.mean(part_colors[:, :3], axis=0)
        if avg_color.max() > 1.0:
            avg_color = avg_color / 255.0
    else:
        avg_color = np.array([0.5, 0.5, 0.5])

    vertex_colors = np.tile(avg_color, (len(vertices), 1))

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=all_faces,
        vertex_colors=np.clip(vertex_colors * 255, 0, 255).astype(np.uint8)
    )

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
        part_path = mesh_dir / "part.ply"
        cuboid_path = mesh_dir / "cuboid_world.npy"

        if not part_path.exists():
            print(f"  {mesh_dir.name}: skipped (no part.ply)")
            continue

        # Load part mesh
        part = trimesh.load(str(part_path))
        part_verts = np.array(part.vertices)

        # Compute PCA surface normal (sign-arbitrary)
        surface_normal = compute_surface_normal_pca(part_verts)

        # Get depth AND oriented normal from cuboid_world
        if cuboid_path.exists():
            cuboid_corners = np.load(str(cuboid_path))
            depth, surface_normal = get_depth_and_direction_from_cuboid(
                cuboid_corners, surface_normal, part_verts
            )
        else:
            depth = 0.15
            print(f"  {mesh_dir.name}: no cuboid_world.npy, using default depth={depth}")

        # Get part colors
        part_colors = None
        if part.visual and hasattr(part.visual, 'vertex_colors') and len(part.visual.vertex_colors) > 0:
            part_colors = np.array(part.visual.vertex_colors)

        # Build base mesh
        base_mesh = build_base_mesh(part_verts, surface_normal, depth, part_colors)

        if base_mesh is None:
            print(f"  {mesh_dir.name}: failed to generate base mesh")
            continue

        out_path = mesh_dir / "base_pca.obj"
        base_mesh.export(str(out_path))

        # Print stats for verification
        part_size = part_verts.max(axis=0) - part_verts.min(axis=0)
        base_size = base_mesh.vertices.max(axis=0) - base_mesh.vertices.min(axis=0)
        print(f"  {mesh_dir.name}: depth={depth:.4f}m  part_AABB=[{part_size[0]:.3f},{part_size[1]:.3f},{part_size[2]:.3f}]  base_AABB=[{base_size[0]:.3f},{base_size[1]:.3f},{base_size[2]:.3f}]  faces={len(base_mesh.faces)}")


if __name__ == "__main__":
    main()
