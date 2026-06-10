"""
Generate shared hidden/base geometry for groups of coplanar articulated parts.

This is a post-process over a REACT3D scene_output directory. It does not
rewrite the original per-part outputs. Instead, it writes:

  shared_bases/shared_base_groups.json
  shared_bases/group_XXX_base.obj

The visualizer can then show one static cabinet/drawer interior per group and
hide the old per-part base.obj files for grouped members.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


EPS = 1e-8
FRONT_CORNER_IDS = np.array([0, 2, 4, 6])


@dataclass
class PartInfo:
    name: str
    path: Path
    vertices: np.ndarray
    colors: np.ndarray | None
    center: np.ndarray
    normal: np.ndarray
    depth: float
    front_offset: float
    articulation_type: str
    base_bounds: np.ndarray | None


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if len(g.vertices) > 0]
        if not geometries:
            return trimesh.Trimesh()
        mesh = trimesh.util.concatenate(geometries)
    return mesh


def load_scene_vertices(scene_dir: Path) -> np.ndarray | None:
    remain_path = scene_dir / "remain_scene.ply"
    if not remain_path.exists():
        return None
    mesh = load_mesh(remain_path)
    if len(mesh.vertices) == 0:
        return None
    return np.asarray(mesh.vertices, dtype=np.float64)


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < EPS:
        return vec
    return vec / norm


def load_articulation_type(path: Path) -> str:
    if not path.exists():
        return "unknown"
    value = np.load(str(path), allow_pickle=True)
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def pca_surface_normal(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    cov = np.cov(centered.T)
    _, eigenvectors = np.linalg.eigh(cov)
    return unit(eigenvectors[:, 0])


def bounds_overlap_volume(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    lo = np.maximum(a[0], b[0])
    hi = np.minimum(a[1], b[1])
    ext = np.maximum(0.0, hi - lo)
    return float(np.prod(ext))


def part_color(mesh: trimesh.Trimesh) -> np.ndarray | None:
    visual = getattr(mesh, "visual", None)
    if visual is None or not hasattr(visual, "vertex_colors"):
        return None
    colors = np.asarray(visual.vertex_colors)
    if colors.size == 0:
        return None
    return colors[:, :3]


def load_part(mesh_dir: Path) -> PartInfo | None:
    part_path = mesh_dir / "part.ply"
    if not part_path.exists():
        return None

    mesh = load_mesh(part_path)
    if len(mesh.vertices) == 0:
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = vertices.mean(axis=0)
    colors = part_color(mesh)

    cuboid_path = mesh_dir / "cuboid_world.npy"
    if cuboid_path.exists():
        cuboid = np.load(str(cuboid_path))
        if cuboid.shape == (8, 3):
            normal = unit(cuboid[1] - cuboid[0])
            depth = float(np.linalg.norm(cuboid[1] - cuboid[0]))
            front_offset = float(np.median(cuboid[FRONT_CORNER_IDS] @ normal))
        else:
            normal = pca_surface_normal(vertices)
            depth = 0.18
            front_offset = float(np.median(vertices @ normal))
    else:
        normal = pca_surface_normal(vertices)
        depth = 0.18
        front_offset = float(np.median(vertices @ normal))

    base_bounds = None
    base_path = mesh_dir / "base.obj"
    if base_path.exists():
        base = load_mesh(base_path)
        if len(base.vertices) > 0:
            base_bounds = np.asarray(base.bounds, dtype=np.float64)

    return PartInfo(
        name=mesh_dir.name,
        path=mesh_dir,
        vertices=vertices,
        colors=colors,
        center=center,
        normal=normal,
        depth=max(depth, 0.02),
        front_offset=front_offset,
        articulation_type=load_articulation_type(mesh_dir / "articulation_type.npy"),
        base_bounds=base_bounds,
    )


def aligned_pair_normal(a: PartInfo, b: PartInfo) -> tuple[np.ndarray, float]:
    b_normal = b.normal if float(np.dot(a.normal, b.normal)) >= 0.0 else -b.normal
    dot = float(np.clip(np.dot(a.normal, b_normal), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(dot))
    normal = unit(a.normal + b_normal)
    if np.linalg.norm(normal) < EPS:
        normal = a.normal
    return normal, angle_deg


def group_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = unit(normal)
    vertical = np.array([0.0, 0.0, 1.0])
    vertical = vertical - float(np.dot(vertical, normal)) * normal
    if np.linalg.norm(vertical) < 1e-4:
        vertical = np.array([0.0, 1.0, 0.0])
        vertical = vertical - float(np.dot(vertical, normal)) * normal
    vertical = unit(vertical)
    horizontal = unit(np.cross(vertical, normal))
    return horizontal, vertical


def canonical_group_frame(vertices: np.ndarray, normal_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one parent frame per group, snapped to world vertical.

    Furniture/cabinet compartments should share a parent coordinate frame:
    vertical is world Z, depth is the group front-plane normal flattened onto
    the horizontal plane, and horizontal completes the orthogonal frame.
    """
    normal = pca_surface_normal(vertices)
    if float(np.dot(normal, normal_hint)) < 0.0:
        normal *= -1.0

    world_z = np.array([0.0, 0.0, 1.0])
    normal = normal - float(np.dot(normal, world_z)) * world_z
    if np.linalg.norm(normal) < 1e-4:
        normal = normal_hint - float(np.dot(normal_hint, world_z)) * world_z
    normal = unit(normal)
    vertical = world_z
    horizontal = unit(np.cross(vertical, normal))
    return normal, horizontal, vertical


def fit_module_frame(vertices: np.ndarray, normal_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one canonical orthogonal frame for a module from its surface points."""
    normal = pca_surface_normal(vertices)
    if float(np.dot(normal, normal_hint)) < 0.0:
        normal *= -1.0
    horizontal, vertical = group_frame(normal)
    return normal, horizontal, vertical


def rect_for_vertices(vertices: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    y = vertices @ horizontal
    z = vertices @ vertical
    return np.array([float(y.min()), float(y.max()), float(z.min()), float(z.max())])


def rect_gap(rect_a: np.ndarray, rect_b: np.ndarray) -> tuple[float, float, float]:
    y_gap = max(0.0, max(rect_a[0], rect_b[0]) - min(rect_a[1], rect_b[1]))
    z_gap = max(0.0, max(rect_a[2], rect_b[2]) - min(rect_a[3], rect_b[3]))
    y_overlap = min(rect_a[1], rect_b[1]) - max(rect_a[0], rect_b[0])
    z_overlap = min(rect_a[3], rect_b[3]) - max(rect_a[2], rect_b[2])
    return y_gap, z_gap, max(y_overlap, z_overlap)


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union < EPS:
        return 0.0
    return float(overlap / union)


def interval_size_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    len_a = max(a1 - a0, EPS)
    len_b = max(b1 - b0, EPS)
    return float(min(len_a, len_b) / max(len_a, len_b))


def should_link(
    a: PartInfo,
    b: PartInfo,
    max_normal_deg: float,
    max_plane_gap: float,
    max_inplane_gap: float,
    min_base_overlap: float,
) -> tuple[bool, dict]:
    normal, angle_deg = aligned_pair_normal(a, b)
    plane_gap = abs(float(np.dot(a.center - b.center, normal)))
    horizontal, vertical = group_frame(normal)
    rect_a = rect_for_vertices(a.vertices, horizontal, vertical)
    rect_b = rect_for_vertices(b.vertices, horizontal, vertical)
    y_gap, z_gap, best_overlap = rect_gap(rect_a, rect_b)
    inplane_gap = math.sqrt(y_gap * y_gap + z_gap * z_gap)
    base_overlap = bounds_overlap_volume(a.base_bounds, b.base_bounds)

    normal_ok = angle_deg <= max_normal_deg
    plane_ok = plane_gap <= max_plane_gap
    near_ok = inplane_gap <= max_inplane_gap and best_overlap > -max_inplane_gap
    overlap_ok = base_overlap >= min_base_overlap
    linked = normal_ok and plane_ok and (near_ok or overlap_ok)

    return linked, {
        "angle_deg": float(angle_deg),
        "plane_gap": float(plane_gap),
        "inplane_gap": float(inplane_gap),
        "base_overlap": float(base_overlap),
        "near_ok": bool(near_ok),
        "overlap_ok": bool(overlap_ok),
    }


def connected_components(names: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    parent = {name: name for name in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    return [sorted(group) for group in groups.values()]


def cluster_lines(values: list[float], tolerance: float) -> list[float]:
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - np.mean(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.mean(cluster)) for cluster in clusters]


def transform_local_mesh(
    mesh: trimesh.Trimesh,
    front_offset: float,
    normal: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> trimesh.Trimesh:
    transform = np.eye(4)
    transform[:3, 0] = normal
    transform[:3, 1] = horizontal
    transform[:3, 2] = vertical
    transform[:3, 3] = front_offset * normal
    out = mesh.copy()
    out.apply_transform(transform)
    return out


def box_mesh(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    transform = trimesh.transformations.translation_matrix(center)
    return trimesh.creation.box(extents=extents, transform=transform)


def rects_share_layout_module(
    rect_a: np.ndarray,
    rect_b: np.ndarray,
    max_gap: float,
    min_span_iou: float,
    min_size_ratio: float,
) -> bool:
    y_gap = max(0.0, max(rect_a[0], rect_b[0]) - min(rect_a[1], rect_b[1]))
    z_gap = max(0.0, max(rect_a[2], rect_b[2]) - min(rect_a[3], rect_b[3]))

    y_iou = interval_iou(rect_a[0], rect_a[1], rect_b[0], rect_b[1])
    z_iou = interval_iou(rect_a[2], rect_a[3], rect_b[2], rect_b[3])
    y_ratio = interval_size_ratio(rect_a[0], rect_a[1], rect_b[0], rect_b[1])
    z_ratio = interval_size_ratio(rect_a[2], rect_a[3], rect_b[2], rect_b[3])

    side_by_side = y_gap <= max_gap and z_iou >= min_span_iou and z_ratio >= min_size_ratio
    stacked = z_gap <= max_gap and y_iou >= min_span_iou and y_ratio >= min_size_ratio
    mostly_same_panel = y_iou >= min_span_iou and z_iou >= min_span_iou
    return bool(side_by_side or stacked or mostly_same_panel)


def layout_modules_from_rects(
    panels: list[dict],
    max_gap: float,
    min_span_iou: float,
    min_size_ratio: float,
) -> list[list[int]]:
    names = [str(i) for i in range(len(panels))]
    edges: list[tuple[str, str]] = []
    for i, panel_a in enumerate(panels):
        for j, panel_b in enumerate(panels[i + 1:], start=i + 1):
            if rects_share_layout_module(
                panel_a["rect"],
                panel_b["rect"],
                max_gap=max_gap,
                min_span_iou=min_span_iou,
                min_size_ratio=min_size_ratio,
            ):
                edges.append((str(i), str(j)))
    components = connected_components(names, edges)
    return [[int(idx) for idx in component] for component in components]


def divider_lines_for_module(
    panels: list[dict],
    max_gap: float,
    min_span_iou: float,
    min_size_ratio: float,
    tolerance: float,
) -> tuple[list[float], list[float]]:
    y_lines = []
    z_lines = []
    for i, panel_a in enumerate(panels):
        rect_a = panel_a["rect"]
        center_a = np.array([(rect_a[0] + rect_a[1]) * 0.5, (rect_a[2] + rect_a[3]) * 0.5])
        for panel_b in panels[i + 1:]:
            rect_b = panel_b["rect"]
            center_b = np.array([(rect_b[0] + rect_b[1]) * 0.5, (rect_b[2] + rect_b[3]) * 0.5])

            y_gap = max(0.0, max(rect_a[0], rect_b[0]) - min(rect_a[1], rect_b[1]))
            z_gap = max(0.0, max(rect_a[2], rect_b[2]) - min(rect_a[3], rect_b[3]))
            y_iou = interval_iou(rect_a[0], rect_a[1], rect_b[0], rect_b[1])
            z_iou = interval_iou(rect_a[2], rect_a[3], rect_b[2], rect_b[3])
            y_ratio = interval_size_ratio(rect_a[0], rect_a[1], rect_b[0], rect_b[1])
            z_ratio = interval_size_ratio(rect_a[2], rect_a[3], rect_b[2], rect_b[3])

            side_by_side = y_gap <= max_gap and z_iou >= min_span_iou and z_ratio >= min_size_ratio
            stacked = z_gap <= max_gap and y_iou >= min_span_iou and y_ratio >= min_size_ratio

            if side_by_side and abs(center_a[0] - center_b[0]) > tolerance:
                left, right = (rect_a, rect_b) if center_a[0] < center_b[0] else (rect_b, rect_a)
                y_lines.append(float((left[1] + right[0]) * 0.5))
            if stacked and abs(center_a[1] - center_b[1]) > tolerance:
                lower, upper = (rect_a, rect_b) if center_a[1] < center_b[1] else (rect_b, rect_a)
                z_lines.append(float((lower[3] + upper[2]) * 0.5))

    return cluster_lines(y_lines, tolerance), cluster_lines(z_lines, tolerance)


def estimate_module_depth(
    scene_vertices: np.ndarray | None,
    normal: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    rect: tuple[float, float, float, float],
    front_offset: float,
    fallback_depth: float,
    min_depth: float,
    max_depth: float,
    depth_quantile: float,
    depth_margin: float,
    near_ignore: float,
    min_points: int,
) -> tuple[float, str, int]:
    fallback = float(np.clip(fallback_depth, min_depth, max_depth))
    if scene_vertices is None or len(scene_vertices) == 0:
        return fallback, "cuboid_fallback", 0

    y = scene_vertices @ horizontal
    z = scene_vertices @ vertical
    x = scene_vertices @ normal - front_offset
    y_min, y_max, z_min, z_max = rect
    mask = (
        (y >= y_min)
        & (y <= y_max)
        & (z >= z_min)
        & (z <= z_max)
        & (x > near_ignore)
        & (x < max_depth)
    )
    depths = x[mask]
    if depths.shape[0] < min_points:
        return fallback, "cuboid_fallback", int(depths.shape[0])

    scene_depth = float(np.percentile(depths, depth_quantile) + depth_margin)
    return float(np.clip(scene_depth, min_depth, max_depth)), f"remain_q{depth_quantile:g}", int(depths.shape[0])


def build_open_shell_local(
    depth: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    thickness: float,
    divider_y: list[float] | None = None,
    divider_z: list[float] | None = None,
) -> trimesh.Trimesh:
    width = max(y_max - y_min, thickness * 4.0)
    height = max(z_max - z_min, thickness * 4.0)
    depth = max(depth, thickness * 6.0)
    t = min(thickness, width / 6.0, height / 6.0, depth / 6.0)

    y_mid = (y_min + y_max) * 0.5
    z_mid = (z_min + z_max) * 0.5
    x_mid = depth * 0.5
    parts = [
        box_mesh((t, width, height), (depth - t * 0.5, y_mid, z_mid)),
        box_mesh((depth, t, height), (x_mid, y_min + t * 0.5, z_mid)),
        box_mesh((depth, t, height), (x_mid, y_max - t * 0.5, z_mid)),
        box_mesh((depth, width, t), (x_mid, y_mid, z_min + t * 0.5)),
        box_mesh((depth, width, t), (x_mid, y_mid, z_max - t * 0.5)),
    ]

    for y in divider_y or []:
        if y_min + t < y < y_max - t:
            parts.append(box_mesh((depth, t, height), (x_mid, y, z_mid)))

    for z in divider_z or []:
        if z_min + t < z < z_max - t:
            parts.append(box_mesh((depth, width, t), (x_mid, y_mid, z)))

    return trimesh.util.concatenate(parts)


def build_shared_base_mesh(
    members: list[PartInfo],
    scene_vertices: np.ndarray | None,
    thickness: float,
    margin: float,
    add_dividers: bool,
    module_gap: float,
    module_span_iou: float,
    module_size_ratio: float,
    min_depth: float,
    max_depth: float,
    depth_quantile: float,
    depth_margin: float,
    depth_near_ignore: float,
    depth_min_points: int,
) -> tuple[trimesh.Trimesh, dict]:
    normal = members[0].normal.copy()
    aligned_normals = []
    for part in members:
        aligned_normals.append(part.normal if float(np.dot(normal, part.normal)) >= 0.0 else -part.normal)
    normal_hint = unit(np.mean(aligned_normals, axis=0))
    group_vertices = np.vstack([part.vertices for part in members])
    normal, horizontal, vertical = canonical_group_frame(group_vertices, normal_hint)

    front_offsets = []
    depths = []
    rects = []
    colors = []
    panels = []
    for part in members:
        front_offset = float(np.median(part.vertices @ normal))
        rect = rect_for_vertices(part.vertices, horizontal, vertical)
        front_offsets.append(front_offset)
        depths.append(part.depth)
        rects.append(rect)
        panels.append({
            "member": part.name,
            "vertices": part.vertices,
            "rect": rect,
            "depth": float(part.depth),
            "front_offset": front_offset,
        })
        if part.colors is not None:
            colors.append(part.colors[:, :3])

    rects_np = np.vstack(rects)
    y_min_all = float(rects_np[:, 0].min() - margin)
    y_max_all = float(rects_np[:, 1].max() + margin)
    z_min_all = float(rects_np[:, 2].min() - margin)
    z_max_all = float(rects_np[:, 3].max() + margin)
    width = max(y_max_all - y_min_all, thickness * 4.0)
    height = max(z_max_all - z_min_all, thickness * 4.0)
    depth = max(float(np.percentile(depths, 65)), thickness * 6.0)
    front_offset = float(np.percentile(front_offsets, 50))
    t = min(thickness, width / 6.0, height / 6.0, depth / 6.0)

    module_indices = layout_modules_from_rects(
        panels,
        max_gap=module_gap,
        min_span_iou=module_span_iou,
        min_size_ratio=module_size_ratio,
    )
    module_meshes = []
    module_metadata = []
    for module_idx, indices in enumerate(module_indices):
        module_panels = [panels[i] for i in indices]
        module_frame_panels = []
        for panel in module_panels:
            module_frame_panels.append({
                **panel,
                "rect": rect_for_vertices(panel["vertices"], horizontal, vertical),
                "front_offset": float(np.median(panel["vertices"] @ normal)),
            })

        module_rects = np.vstack([panel["rect"] for panel in module_frame_panels])
        module_depths = [panel["depth"] for panel in module_panels]
        module_front_offsets = [panel["front_offset"] for panel in module_frame_panels]

        y_min = float(module_rects[:, 0].min() - margin)
        y_max = float(module_rects[:, 1].max() + margin)
        z_min = float(module_rects[:, 2].min() - margin)
        z_max = float(module_rects[:, 3].max() + margin)
        module_front_offset = float(np.percentile(module_front_offsets, 50))
        fallback_depth = max(float(np.percentile(module_depths, 65)), thickness * 6.0)
        module_depth, depth_source, depth_points = estimate_module_depth(
            scene_vertices,
            normal,
            horizontal,
            vertical,
            (y_min, y_max, z_min, z_max),
            module_front_offset,
            fallback_depth,
            min_depth,
            max_depth,
            depth_quantile,
            depth_margin,
            depth_near_ignore,
            depth_min_points,
        )
        divider_y, divider_z = ([], [])
        if add_dividers:
            divider_y, divider_z = divider_lines_for_module(
                module_frame_panels,
                max_gap=module_gap,
                min_span_iou=module_span_iou,
                min_size_ratio=module_size_ratio,
                tolerance=max(t * 3.0, 0.01),
            )

        local_mesh = build_open_shell_local(
            module_depth,
            y_min,
            y_max,
            z_min,
            z_max,
            t,
            divider_y=divider_y,
            divider_z=divider_z,
        )
        module_meshes.append(
            transform_local_mesh(
                local_mesh,
                module_front_offset,
                normal,
                horizontal,
                vertical,
            )
        )
        module_metadata.append({
            "id": f"module_{module_idx:03d}",
            "members": [str(panel["member"]) for panel in module_panels],
            "normal": normal.tolist(),
            "horizontal": horizontal.tolist(),
            "vertical": vertical.tolist(),
            "depth": module_depth,
            "fallback_depth": fallback_depth,
            "depth_source": depth_source,
            "depth_points": depth_points,
            "width": float(max(y_max - y_min, t * 4.0)),
            "height": float(max(z_max - z_min, t * 4.0)),
            "front_offset": module_front_offset,
            "rect": [y_min, y_max, z_min, z_max],
            "divider_y": divider_y,
            "divider_z": divider_z,
        })

    mesh = trimesh.util.concatenate(module_meshes)

    if colors:
        avg_color = np.mean(np.vstack(colors), axis=0)
        if avg_color.max() <= 1.0:
            avg_color = avg_color * 255.0
    else:
        avg_color = np.array([160.0, 160.0, 160.0])
    mesh.visual.vertex_colors = np.tile(np.clip(avg_color, 0, 255).astype(np.uint8), (len(mesh.vertices), 1))

    metadata = {
        "normal": normal.tolist(),
        "horizontal": horizontal.tolist(),
        "vertical": vertical.tolist(),
        "front_offset": front_offset,
        "depth": depth,
        "width": width,
        "height": height,
        "thickness": t,
        "layout_mode": "module_clipped_shell",
        "module_count": len(module_metadata),
        "modules": module_metadata,
        "add_dividers": bool(add_dividers),
    }
    return mesh, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True, help="REACT3D scene_output directory")
    parser.add_argument("--max_normal_deg", type=float, default=8.0)
    parser.add_argument("--max_plane_gap", type=float, default=0.08)
    parser.add_argument("--max_inplane_gap", type=float, default=0.12)
    parser.add_argument("--min_base_overlap", type=float, default=1e-4)
    parser.add_argument("--min_group_size", type=int, default=2)
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.015)
    parser.add_argument(
        "--module_gap",
        type=float,
        default=0.06,
        help="Max 2D front-layout gap for merging adjacent panels into one interior module",
    )
    parser.add_argument(
        "--module_span_iou",
        type=float,
        default=0.82,
        help="Required overlap IoU along the shared span when merging layout modules",
    )
    parser.add_argument(
        "--module_size_ratio",
        type=float,
        default=0.85,
        help="Required similar-size ratio along the shared span when merging layout modules",
    )
    parser.add_argument("--min_depth", type=float, default=0.12)
    parser.add_argument("--max_depth", type=float, default=0.60)
    parser.add_argument("--depth_quantile", type=float, default=90.0)
    parser.add_argument("--depth_margin", type=float, default=0.015)
    parser.add_argument("--depth_near_ignore", type=float, default=0.02)
    parser.add_argument("--depth_min_points", type=int, default=500)
    parser.add_argument("--no_dividers", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    scene_vertices = load_scene_vertices(scene_dir)
    mesh_dirs = sorted(d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("mesh_"))
    parts = [p for p in (load_part(d) for d in mesh_dirs) if p is not None]
    part_by_name = {p.name: p for p in parts}
    print(f"Found {len(parts)} articulated parts")

    edges: list[tuple[str, str]] = []
    pair_records = []
    for i, part_a in enumerate(parts):
        for part_b in parts[i + 1:]:
            linked, metrics = should_link(
                part_a,
                part_b,
                args.max_normal_deg,
                args.max_plane_gap,
                args.max_inplane_gap,
                args.min_base_overlap,
            )
            if linked:
                edges.append((part_a.name, part_b.name))
                pair_records.append({"a": part_a.name, "b": part_b.name, **metrics})

    components = connected_components([p.name for p in parts], edges)
    groups = [group for group in components if len(group) >= args.min_group_size]
    print(f"Linked pairs: {len(edges)}")
    for record in pair_records:
        print(
            "  {a} <-> {b}: angle={angle_deg:.2f}deg plane={plane_gap:.3f}m "
            "gap={inplane_gap:.3f}m base_overlap={base_overlap:.6f}".format(**record)
        )
    print(f"Shared groups: {len(groups)}")
    for group in groups:
        print(f"  {', '.join(group)}")

    out_dir = scene_dir / "shared_bases"
    manifest = {
        "version": 1,
        "scene_dir": str(scene_dir),
        "thresholds": {
            "max_normal_deg": args.max_normal_deg,
            "max_plane_gap": args.max_plane_gap,
            "max_inplane_gap": args.max_inplane_gap,
            "min_base_overlap": args.min_base_overlap,
            "min_group_size": args.min_group_size,
            "thickness": args.thickness,
            "margin": args.margin,
            "add_dividers": not args.no_dividers,
            "module_gap": args.module_gap,
            "module_span_iou": args.module_span_iou,
            "module_size_ratio": args.module_size_ratio,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "depth_quantile": args.depth_quantile,
            "depth_margin": args.depth_margin,
            "depth_near_ignore": args.depth_near_ignore,
            "depth_min_points": args.depth_min_points,
            "depth_uses_remain_scene": scene_vertices is not None,
        },
        "groups": [],
        "part_to_group": {},
        "linked_pairs": pair_records,
    }

    if args.dry_run:
        return

    out_dir.mkdir(exist_ok=True)
    for idx, group_names in enumerate(groups):
        group_id = f"group_{idx:03d}"
        members = [part_by_name[name] for name in group_names]
        mesh, metadata = build_shared_base_mesh(
            members,
            scene_vertices=scene_vertices,
            thickness=args.thickness,
            margin=args.margin,
            add_dividers=not args.no_dividers,
            module_gap=args.module_gap,
            module_span_iou=args.module_span_iou,
            module_size_ratio=args.module_size_ratio,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            depth_quantile=args.depth_quantile,
            depth_margin=args.depth_margin,
            depth_near_ignore=args.depth_near_ignore,
            depth_min_points=args.depth_min_points,
        )
        mesh_name = f"{group_id}_base.obj"
        mesh.export(str(out_dir / mesh_name))
        print(f"  {group_id}: {metadata['module_count']} layout modules")
        for module in metadata["modules"]:
            dividers = len(module.get("divider_y", [])) + len(module.get("divider_z", []))
            print(
                f"    {module['id']}: {', '.join(module['members'])} "
                f"depth={module['depth']:.3f} ({module['depth_source']}, n={module['depth_points']}) "
                f"dividers={dividers}"
            )
        for name in group_names:
            manifest["part_to_group"][name] = group_id
        manifest["groups"].append({
            "id": group_id,
            "members": group_names,
            "mesh": f"shared_bases/{mesh_name}",
            **metadata,
        })

    manifest_path = out_dir / "shared_base_groups.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
