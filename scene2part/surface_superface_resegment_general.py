"""General full-scene surface re-segmentation using superface primitives.

This pass is meant for surfaces that were already broken during the initial
3D segmentation/submesh extraction. It does not synthesize patches. Instead it
uses the current clean mesh as a seed, searches the original full-scene mesh for
actual nearby faces, groups those faces into local superfaces, scores each
superface with multi-source evidence, and exports a recovered candidate.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage


EPS = 1e-8


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_lab_root() -> Path:
    return repo_root().parent / "REACT3D_mesh030_lab"


def first_file(patterns: list[Path | str]) -> str:
    for pattern in patterns:
        hits = sorted(glob.glob(str(pattern)))
        if hits:
            return hits[0]
    return ""


def parse_args():
    root = repo_root()
    lab_root = default_lab_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab_root", default=str(lab_root))
    parser.add_argument("--data_dir", default=str(lab_root / "data" / "example_input_scene_surface_lab"))
    parser.add_argument("--full_mesh", default="")
    parser.add_argument("--mesh_ids", nargs="*", default=["mesh_030"])
    parser.add_argument("--all_meshes", action="store_true")
    parser.add_argument("--output_dir", default=str(root / "outputs" / "superface_fullscene_resegment_general_0521"))
    parser.add_argument("--mode", choices=["strict", "balanced", "recall"], default="balanced")
    parser.add_argument("--grid_res", type=float, default=0.015)
    parser.add_argument("--grid_pad", type=float, default=0.045)
    parser.add_argument("--bbox_pad", type=float, default=0.075)
    parser.add_argument("--point_band", type=float, default=0.080)
    parser.add_argument("--seed_band", type=float, default=0.035)
    parser.add_argument("--grow_band", type=float, default=0.070)
    parser.add_argument("--final_band", type=float, default=0.055)
    parser.add_argument("--cell_depth_slack", type=float, default=0.014)
    parser.add_argument("--normal_deg", type=float, default=42.0)
    parser.add_argument("--superface_normal_deg", type=float, default=18.0)
    parser.add_argument("--superface_z_jump", type=float, default=0.020)
    parser.add_argument("--min_points_per_cell", type=int, default=2)
    parser.add_argument("--support_close_iters", type=int, default=1)
    parser.add_argument("--support_dilate_iters", type=int, default=1)
    parser.add_argument("--grow_steps", type=int, default=4)
    parser.add_argument("--secondary_max_superfaces", type=int, default=12)
    parser.add_argument("--secondary_min_cells", type=int, default=5)
    parser.add_argument("--secondary_max_seed_cell_dist", type=float, default=7.0)
    parser.add_argument("--merge_digits", type=int, default=5)
    parser.add_argument("--png_scale", type=int, default=7)
    return parser.parse_args()


def resolve_full_mesh(args) -> str:
    if args.full_mesh:
        return args.full_mesh
    data_dir = Path(args.data_dir)
    return first_existing([
        data_dir / "mesh_aligned_0.05.ply",
        repo_root() / "data" / "example_input_scene" / "mesh_aligned_0.05.ply",
        data_dir / "mesh_aligned_0.02_remeshed.ply",
    ])


def first_existing(paths: list[Path | str]) -> str:
    for path in paths:
        if Path(path).is_file():
            return str(path)
    return ""


def discover_mesh_ids(data_dir: Path) -> list[str]:
    real_dir = data_dir / "perception" / "vis_groups_final_mesh" / "real"
    return sorted(path.name for path in real_dir.glob("mesh_*") if path.is_dir())


def evidence_paths(lab_root: Path, mesh_id: str) -> dict[str, str]:
    return {
        "points_seed0": first_file([
            lab_root / "outputs" / "sam2_maskprop_depthvote_0521" / mesh_id / "*depthvote_points.ply",
            lab_root / "outputs" / "depthvote_labdata_0520" / mesh_id / "*depthvote_points.ply",
            lab_root / "outputs" / "dense_depthvote_labdata_0520" / mesh_id / "*depthvote_points.ply",
        ]),
        "points_seed2": first_file([
            lab_root / "outputs" / "sam2_maskprop_depthvote_seed2_conda118_0521" / mesh_id / "*depthvote_points.ply",
            lab_root / "outputs" / "sam2_maskprop_depthvote_seed2_0521" / mesh_id / "*depthvote_points.ply",
            lab_root / "outputs" / "sam2_maskprop_depthvote_seed2_external_0521" / mesh_id / "*depthvote_points.ply",
        ]),
        "faceco_seed0": first_file([
            lab_root / "outputs" / "sam2_face_cooccurrence_relaxed_evidence_0521" / mesh_id / "*face_cooccurrence_evidence.npz",
            lab_root / "outputs" / "sam2_face_cooccurrence_sweep_mc25_0521" / mesh_id / "*face_cooccurrence_evidence.npz",
        ]),
        "faceco_seed2": first_file([
            lab_root / "outputs" / "sam2_face_cooccurrence_seed2_0521" / mesh_id / "*face_cooccurrence_evidence.npz",
        ]),
        "paper_evidence": first_file([
            lab_root / "outputs" / "paper_region_fill_sam2_faceco_evidence_0521" / mesh_id / "*paper_region_evidence.npz",
            lab_root / "outputs" / "paper_region_fill_labdata_0520" / mesh_id / "*paper_region_evidence.npz",
        ]),
    }


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected mesh at {path}, got {type(mesh)}")
    return mesh


def load_points(path: str | Path) -> np.ndarray:
    if not path or not Path(path).is_file():
        return np.zeros((0, 3), dtype=np.float64)
    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
    if hasattr(obj, "vertices"):
        return np.asarray(obj.vertices, dtype=np.float64)
    return np.zeros((0, 3), dtype=np.float64)


def face_centroids(mesh: trimesh.Trimesh) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices[np.asarray(mesh.faces, dtype=np.int64)].mean(axis=1)


def fit_pca(mesh: trimesh.Trimesh):
    points = np.asarray(mesh.vertices, dtype=np.float64)
    center = points.mean(axis=0)
    cov = np.cov((points - center).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    normal = vecs[:, order[0]]
    u_axis = vecs[:, order[2]]
    v_axis = np.cross(normal, u_axis)
    normal /= max(np.linalg.norm(normal), EPS)
    u_axis /= max(np.linalg.norm(u_axis), EPS)
    v_axis /= max(np.linalg.norm(v_axis), EPS)
    return center, u_axis, v_axis, normal


def local3(points: np.ndarray, center: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return (points - center) @ np.stack([u_axis, v_axis, normal], axis=1)


def cells_from_uv(uv: np.ndarray, uv_min: np.ndarray, shape: np.ndarray, grid_res: float):
    cells = np.floor((uv - uv_min) / max(grid_res, EPS)).astype(np.int64)
    valid = np.all((cells >= 0) & (cells < shape), axis=1)
    return cells, valid


def occupancy_from_uv(uv: np.ndarray, uv_min: np.ndarray, shape: np.ndarray, grid_res: float) -> np.ndarray:
    occ = np.zeros(tuple(shape), dtype=np.int32)
    if uv.size == 0:
        return occ
    cells, valid = cells_from_uv(uv, uv_min, shape, grid_res)
    if np.any(valid):
        c = cells[valid]
        np.add.at(occ, (c[:, 0], c[:, 1]), 1)
    return occ


def filter_points(points: np.ndarray, raw_min: np.ndarray, raw_max: np.ndarray, local_pts: np.ndarray, band: float):
    if points.size == 0:
        return points, local_pts
    in_box = np.all((points >= raw_min) & (points <= raw_max), axis=1)
    near_plane = np.abs(local_pts[:, 2]) <= band
    return points[in_box & near_plane], local_pts[in_box & near_plane]


def mask_from_face_ids(n_faces: int, ids: np.ndarray) -> np.ndarray:
    out = np.zeros(n_faces, dtype=bool)
    ids = np.asarray(ids, dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < n_faces)]
    out[ids] = True
    return out


def load_face_evidence(path: str | Path, n_faces: int):
    empty = {
        "selected": np.zeros(n_faces, dtype=bool),
        "pool": np.zeros(n_faces, dtype=bool),
        "node": np.zeros(n_faces, dtype=bool),
        "hit": np.zeros(n_faces, dtype=np.int64),
    }
    if not path or not Path(path).is_file():
        return empty
    data = np.load(path)
    selected = mask_from_face_ids(n_faces, data["selected_faces"]) if "selected_faces" in data.files else empty["selected"]
    pool_key = "pool_faces" if "pool_faces" in data.files else "pool_face" if "pool_face" in data.files else None
    pool = mask_from_face_ids(n_faces, data[pool_key]) if pool_key else empty["pool"]
    node = np.asarray(data["node_mask"], dtype=bool) if "node_mask" in data.files else empty["node"]
    hit = np.asarray(data["hit_count"], dtype=np.int64) if "hit_count" in data.files else empty["hit"]
    if node.size != n_faces:
        fixed = np.zeros(n_faces, dtype=bool)
        fixed[: min(n_faces, node.size)] = node[: min(n_faces, node.size)]
        node = fixed
    if hit.size != n_faces:
        fixed = np.zeros(n_faces, dtype=np.int64)
        fixed[: min(n_faces, hit.size)] = hit[: min(n_faces, hit.size)]
        hit = fixed
    return {"selected": selected, "pool": pool, "node": node, "hit": hit}


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = int(self.parent[x])
        while p != int(self.parent[p]):
            self.parent[p] = self.parent[self.parent[p]]
            p = int(self.parent[p])
        return p

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def submesh_from_faces(mesh: trimesh.Trimesh, face_ids: np.ndarray) -> trimesh.Trimesh:
    face_ids = np.asarray(sorted(set(map(int, face_ids))), dtype=np.int64)
    if face_ids.size == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
    faces = np.asarray(mesh.faces, dtype=np.int64)[face_ids]
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    out = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64)[used],
        faces=inverse.reshape(-1, 3),
        process=False,
    )
    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(mesh.vertices):
        out.visual.vertex_colors = mesh.visual.vertex_colors[used]
    return out


def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    if len(mesh.vertices) == 0:
        return {"vertices": 0, "faces": 0, "area": 0.0, "extent": [0.0, 0.0, 0.0], "components": 0}
    components = 0
    if len(mesh.faces):
        edges = mesh.edges_sorted.reshape((-1, 2))
        components = len(trimesh.graph.connected_components(edges, min_len=1, engine="scipy"))
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "area": float(mesh.area) if len(mesh.faces) else 0.0,
        "extent": [float(x) for x in mesh.extents],
        "components": int(components),
    }


def prune_by_cell_depth(face_ids: np.ndarray, face_z: np.ndarray, cells: np.ndarray, valid: np.ndarray, final_band: float, slack: float) -> np.ndarray:
    if face_ids.size == 0:
        return face_ids
    valid_ids = face_ids[valid[face_ids]]
    if valid_ids.size == 0:
        return face_ids[np.abs(face_z[face_ids]) <= final_band]
    keep = np.zeros(face_z.shape[0], dtype=bool)
    cell_key = (cells[valid_ids, 0].astype(np.int64) << 32) + cells[valid_ids, 1].astype(np.int64)
    for key in np.unique(cell_key):
        ids = valid_ids[cell_key == key]
        absz = np.abs(face_z[ids])
        best = float(absz.min())
        chosen = ids[(absz <= best + slack) & (absz <= final_band)]
        if chosen.size == 0:
            chosen = ids[[int(np.argmin(absz))]]
        keep[chosen] = True
    return np.where(keep)[0]


def concat_clean_add(clean: trimesh.Trimesh, add: trimesh.Trimesh, merge_digits: int) -> trimesh.Trimesh:
    out = trimesh.util.concatenate([clean, add])
    if merge_digits >= 0 and len(out.vertices):
        out.merge_vertices(digits_vertex=merge_digits)
        out.remove_unreferenced_vertices()
    return out


def label_superfaces(active: np.ndarray, adjacency: np.ndarray, normals: np.ndarray, face_z: np.ndarray, normal_deg: float, z_jump: float):
    active_ids = np.where(active)[0]
    local_index = np.full(active.shape[0], -1, dtype=np.int64)
    local_index[active_ids] = np.arange(active_ids.size, dtype=np.int64)
    uf = UnionFind(active_ids.size)
    if active_ids.size == 0:
        return active_ids, np.zeros(0, dtype=np.int64), []

    dot_thr = math.cos(math.radians(normal_deg))
    adj_active = adjacency[active[adjacency[:, 0]] & active[adjacency[:, 1]]]
    for a, b in adj_active.tolist():
        if normals[a] @ normals[b] >= dot_thr and abs(float(face_z[a] - face_z[b])) <= z_jump:
            uf.union(int(local_index[a]), int(local_index[b]))

    roots = np.array([uf.find(i) for i in range(active_ids.size)], dtype=np.int64)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    superfaces = [active_ids[labels == i] for i in range(unique_roots.size)]
    return active_ids, labels, superfaces


def superface_adjacency(superfaces: list[np.ndarray], face_to_sf: np.ndarray, adjacency: np.ndarray) -> list[set[int]]:
    neighbors = [set() for _ in superfaces]
    sf_a = face_to_sf[adjacency[:, 0]]
    sf_b = face_to_sf[adjacency[:, 1]]
    valid = (sf_a >= 0) & (sf_b >= 0) & (sf_a != sf_b)
    for a, b in zip(sf_a[valid].tolist(), sf_b[valid].tolist()):
        neighbors[a].add(b)
        neighbors[b].add(a)
    return neighbors


def summarize_superfaces(superfaces, seed_face, in_support, strong_evidence, cells, valid, shape, hit0, hit2, paper_mask, score, dist_seed, face_z):
    rows = []
    for idx, faces in enumerate(superfaces):
        ok = valid[faces]
        cell_count = 0
        support_cells = 0
        min_seed_dist = 9999.0
        if np.any(ok):
            cell_mask = np.zeros(tuple(shape), dtype=bool)
            support_mask = np.zeros(tuple(shape), dtype=bool)
            c = cells[faces[ok]]
            cell_mask[c[:, 0], c[:, 1]] = True
            support_mask[c[in_support[faces[ok]], 0], c[in_support[faces[ok]], 1]] = True
            cell_count = int(np.count_nonzero(cell_mask))
            support_cells = int(np.count_nonzero(support_mask))
            min_seed_dist = float(np.min(dist_seed[c[:, 0], c[:, 1]]))
        rows.append({
            "superface": int(idx),
            "faces": int(faces.size),
            "cells": int(cell_count),
            "support_cells": int(support_cells),
            "seed_faces": int(np.count_nonzero(seed_face[faces])),
            "support_ratio": float(np.count_nonzero(in_support[faces]) / max(faces.size, 1)),
            "strong_ratio": float(np.count_nonzero(strong_evidence[faces]) / max(faces.size, 1)),
            "paper_ratio": float(np.count_nonzero(paper_mask[faces]) / max(faces.size, 1)),
            "score_mean": float(np.mean(score[faces])) if faces.size else 0.0,
            "hit0_mean": float(np.mean(hit0[faces])) if faces.size else 0.0,
            "hit2_mean": float(np.mean(hit2[faces])) if faces.size else 0.0,
            "absz_mean": float(np.mean(np.abs(face_z[faces]))) if faces.size else 0.0,
            "z_thickness": float(np.percentile(face_z[faces], 95) - np.percentile(face_z[faces], 5)) if faces.size > 1 else 0.0,
            "min_seed_cell_dist": min_seed_dist,
            "keep": False,
            "reason": "",
        })
    return rows


def choose_superfaces(rows: list[dict], neighbors: list[set[int]], mode: str, grow_steps: int, max_secondary: int, min_cells: int, max_dist: float) -> set[int]:
    keep = set()
    for row in rows:
        if row["seed_faces"] > 0:
            row["keep"] = True
            row["reason"] = "seed-superface"
            keep.add(row["superface"])

    if not keep:
        return keep

    if mode == "strict":
        grow_score, grow_support, grow_strong = 1.25, 0.55, 0.05
    elif mode == "balanced":
        grow_score, grow_support, grow_strong = 1.05, 0.42, 0.03
    else:
        grow_score, grow_support, grow_strong = 0.85, 0.30, 0.01

    frontier = set(keep)
    for _ in range(grow_steps):
        additions = set()
        for sf in frontier:
            for nb in neighbors[sf]:
                if nb in keep:
                    continue
                row = rows[nb]
                if row["cells"] < min_cells:
                    continue
                if row["min_seed_cell_dist"] > max_dist + 2.0:
                    continue
                evidence_ok = (
                    row["score_mean"] >= grow_score
                    or row["support_ratio"] >= grow_support
                    or row["strong_ratio"] >= grow_strong
                    or row["paper_ratio"] >= 0.20
                )
                quality_ok = row["absz_mean"] <= 0.065 and row["z_thickness"] <= 0.060
                if evidence_ok and quality_ok:
                    row["keep"] = True
                    row["reason"] = "grown-neighbor"
                    additions.add(nb)
        if not additions:
            break
        keep.update(additions)
        frontier = additions

    secondary = []
    for row in rows:
        if row["superface"] in keep:
            continue
        if row["cells"] < min_cells or row["min_seed_cell_dist"] > max_dist:
            continue
        if row["absz_mean"] > 0.060 or row["z_thickness"] > 0.055:
            continue
        if row["score_mean"] >= grow_score + 0.15 or row["paper_ratio"] >= 0.25:
            secondary.append(row)
    secondary.sort(key=lambda r: (r["score_mean"], r["support_ratio"], r["cells"], r["faces"]), reverse=True)
    for row in secondary[:max_secondary]:
        row["keep"] = True
        row["reason"] = "nearby-evidence-superface"
        keep.add(row["superface"])
    return keep


def write_rows_csv(path: Path, rows: list[dict]):
    fields = [
        "superface", "keep", "reason", "faces", "cells", "support_cells", "seed_faces",
        "support_ratio", "strong_ratio", "paper_ratio", "score_mean", "hit0_mean",
        "hit2_mean", "absz_mean", "z_thickness", "min_seed_cell_dist",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def paint(draw, origin, title, mask, color, scale):
    ox, oy = origin
    h, w = mask.shape
    draw.text((ox, oy - 18), title, fill=(20, 20, 20))
    draw.rectangle((ox, oy, ox + h * scale, oy + w * scale), outline=(80, 80, 80), width=1)
    xs, ys = np.where(mask)
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.rectangle(
            (ox + x * scale, oy + (w - 1 - y) * scale, ox + (x + 1) * scale - 1, oy + (w - y) * scale - 1),
            fill=color,
        )


def write_map_png(path: Path, panels: list[tuple[str, np.ndarray, tuple[int, int, int]]], scale: int):
    shape = panels[0][1].shape
    panel_w = int(shape[0] * scale)
    panel_h = int(shape[1] * scale)
    cols = 3
    rows = int(math.ceil(len(panels) / cols))
    margin = 34
    gap = 28
    image = Image.new(
        "RGB",
        (margin * 2 + cols * panel_w + (cols - 1) * gap, margin * 2 + rows * (panel_h + 18) + (rows - 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(image)
    for i, panel in enumerate(panels):
        x = margin + (i % cols) * (panel_w + gap)
        y = margin + (i // cols) * (panel_h + gap + 18) + 18
        paint(draw, (x, y), panel[0], panel[1], panel[2], scale)
    image.save(path)


def process_mesh(args, full: trimesh.Trimesh, mesh_id: str) -> dict:
    data_dir = Path(args.data_dir)
    lab_root = Path(args.lab_root)
    mesh_dir = data_dir / "perception" / "vis_groups_final_mesh" / "real" / mesh_id
    raw_path = mesh_dir / "mask_mesh.ply"
    clean_path = mesh_dir / "mask_mesh_refined_clean.ply"
    if not raw_path.is_file() or not clean_path.is_file():
        return {"mesh_id": mesh_id, "status": "missing_raw_or_clean"}

    paths = evidence_paths(lab_root, mesh_id)
    raw = load_mesh(raw_path)
    clean = load_mesh(clean_path)
    if len(clean.vertices) < 3 or len(clean.faces) == 0:
        return {"mesh_id": mesh_id, "status": "empty_clean"}

    p0 = load_points(paths["points_seed0"])
    p2 = load_points(paths["points_seed2"])
    center, u_axis, v_axis, normal = fit_pca(clean)
    raw_vertices = np.asarray(raw.vertices, dtype=np.float64)
    raw_min = raw_vertices.min(axis=0) - args.bbox_pad
    raw_max = raw_vertices.max(axis=0) + args.bbox_pad

    raw_local = local3(face_centroids(raw), center, u_axis, v_axis, normal)
    clean_local = local3(face_centroids(clean), center, u_axis, v_axis, normal)
    p0_local = local3(p0, center, u_axis, v_axis, normal)
    p2_local = local3(p2, center, u_axis, v_axis, normal)
    _, p0_l = filter_points(p0, raw_min, raw_max, p0_local, args.point_band)
    _, p2_l = filter_points(p2, raw_min, raw_max, p2_local, args.point_band)

    uv_sources = [raw_local[:, :2], clean_local[:, :2]]
    if p0_l.size:
        uv_sources.append(p0_l[:, :2])
    if p2_l.size:
        uv_sources.append(p2_l[:, :2])
    all_uv = np.concatenate(uv_sources, axis=0)
    uv_min = all_uv.min(axis=0) - args.grid_pad
    uv_max = all_uv.max(axis=0) + args.grid_pad
    shape = np.maximum(np.ceil((uv_max - uv_min) / max(args.grid_res, EPS)).astype(np.int64) + 1, 1)

    raw_mask = occupancy_from_uv(raw_local[:, :2], uv_min, shape, args.grid_res) > 0
    clean_mask = occupancy_from_uv(clean_local[:, :2], uv_min, shape, args.grid_res) > 0
    seed0_mask = occupancy_from_uv(p0_l[:, :2], uv_min, shape, args.grid_res) >= args.min_points_per_cell
    seed2_mask = occupancy_from_uv(p2_l[:, :2], uv_min, shape, args.grid_res) >= args.min_points_per_cell
    if args.mode == "strict":
        base_support = clean_mask | raw_mask | seed0_mask | (seed2_mask & ndimage.binary_dilation(seed0_mask | raw_mask, iterations=2))
    elif args.mode == "balanced":
        base_support = clean_mask | raw_mask | seed0_mask | (seed2_mask & ndimage.binary_dilation(seed0_mask | raw_mask | clean_mask, iterations=4))
    else:
        base_support = clean_mask | raw_mask | seed0_mask | seed2_mask
    support = base_support.copy()
    if args.support_close_iters > 0:
        support = ndimage.binary_closing(support, structure=np.ones((3, 3), dtype=bool), iterations=args.support_close_iters)
    if args.support_dilate_iters > 0:
        support = ndimage.binary_dilation(support, structure=np.ones((3, 3), dtype=bool), iterations=args.support_dilate_iters)

    full_vertices = np.asarray(full.vertices, dtype=np.float64)
    full_faces = np.asarray(full.faces, dtype=np.int64)
    centroids = full_vertices[full_faces].mean(axis=1)
    face_local = local3(centroids, center, u_axis, v_axis, normal)
    face_cells, valid = cells_from_uv(face_local[:, :2], uv_min, shape, args.grid_res)
    face_normals = np.asarray(full.face_normals, dtype=np.float64)
    n_faces = len(full_faces)

    fc0 = load_face_evidence(paths["faceco_seed0"], n_faces)
    fc2 = load_face_evidence(paths["faceco_seed2"], n_faces)
    paper = load_face_evidence(paths["paper_evidence"], n_faces)
    strong_evidence = (
        fc0["selected"]
        | fc2["selected"]
        | paper["selected"]
        | ((fc0["hit"] >= 2) & paper["pool"])
        | ((fc2["hit"] >= 2) & (paper["pool"] | fc2["node"]))
    )

    valid_ids = np.where(valid)[0]
    vc = face_cells[valid_ids]
    in_support = np.zeros(n_faces, dtype=bool)
    in_seed_cells = np.zeros(n_faces, dtype=bool)
    in_support[valid_ids] = support[vc[:, 0], vc[:, 1]]
    in_seed_cells[valid_ids] = clean_mask[vc[:, 0], vc[:, 1]]

    in_box = np.all((centroids >= raw_min) & (centroids <= raw_max), axis=1)
    normal_ok = np.abs(face_normals @ normal) >= math.cos(math.radians(args.normal_deg))
    seed_face = in_box & valid & normal_ok & (np.abs(face_local[:, 2]) <= args.seed_band) & in_seed_cells
    candidate = in_box & valid & normal_ok & (np.abs(face_local[:, 2]) <= args.grow_band) & (in_support | strong_evidence | seed_face)
    candidate |= seed_face

    adjacency = np.asarray(full.face_adjacency, dtype=np.int64)
    active_ids, labels, superfaces = label_superfaces(
        candidate,
        adjacency,
        face_normals,
        face_local[:, 2],
        args.superface_normal_deg,
        args.superface_z_jump,
    )
    face_to_sf = np.full(n_faces, -1, dtype=np.int64)
    if active_ids.size:
        face_to_sf[active_ids] = labels
    neighbors = superface_adjacency(superfaces, face_to_sf, adjacency)

    dist_seed = ndimage.distance_transform_edt(~clean_mask)
    score = (
        1.0 * in_support.astype(np.float64)
        + 1.5 * strong_evidence.astype(np.float64)
        + np.clip(fc0["hit"], 0, 4) / 4.0
        + np.clip(fc2["hit"], 0, 4) / 4.0
        + 0.4 * paper["pool"].astype(np.float64)
    )
    paper_mask = paper["pool"] | paper["selected"]
    rows = summarize_superfaces(
        superfaces,
        seed_face,
        in_support,
        strong_evidence,
        face_cells,
        valid,
        shape,
        fc0["hit"],
        fc2["hit"],
        paper_mask,
        score,
        dist_seed,
        face_local[:, 2],
    )
    keep = choose_superfaces(
        rows,
        neighbors,
        args.mode,
        args.grow_steps,
        args.secondary_max_superfaces,
        args.secondary_min_cells,
        args.secondary_max_seed_cell_dist,
    )

    selected_raw = np.concatenate([superfaces[idx] for idx in sorted(keep)], axis=0) if keep else np.zeros(0, dtype=np.int64)
    selected = prune_by_cell_depth(selected_raw, face_local[:, 2], face_cells, valid, args.final_band, args.cell_depth_slack)

    clean_like = np.zeros(n_faces, dtype=bool)
    clean_like[valid_ids] = clean_mask[vc[:, 0], vc[:, 1]]
    add_faces = selected[~(clean_like[selected] & (np.abs(face_local[selected, 2]) <= args.final_band))]
    selected_mesh = submesh_from_faces(full, selected)
    add_mesh = submesh_from_faces(full, add_faces)
    final_mesh = concat_clean_add(clean, add_mesh, args.merge_digits)

    final_uv = local3(face_centroids(final_mesh), center, u_axis, v_axis, normal)[:, :2]
    selected_uv = face_local[selected, :2] if selected.size else np.zeros((0, 2), dtype=np.float64)
    add_uv = face_local[add_faces, :2] if add_faces.size else np.zeros((0, 2), dtype=np.float64)
    final_mask = occupancy_from_uv(final_uv, uv_min, shape, args.grid_res) > 0
    selected_mask = occupancy_from_uv(selected_uv, uv_min, shape, args.grid_res) > 0
    add_mask = occupancy_from_uv(add_uv, uv_min, shape, args.grid_res) > 0
    point_union = seed0_mask | seed2_mask
    clean_missing = point_union & ~clean_mask
    final_missing = point_union & ~final_mask
    recovered = clean_missing & final_mask
    lost_vs_clean = clean_mask & ~final_mask

    out_dir = Path(args.output_dir) / mesh_id / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{mesh_id}_superface_fullscene_{args.mode}"
    selected_path = out_dir / f"{stem}_selected_surface.ply"
    add_path = out_dir / f"{stem}_add_faces.ply"
    final_path = out_dir / f"{stem}_union_clean_candidate.ply"
    evidence_path = out_dir / f"{stem}_evidence.npz"
    debug_path = out_dir / f"{stem}_debug.json"
    csv_path = out_dir / f"{stem}_superfaces.csv"
    png_path = out_dir / f"{stem}_evidence_map.png"
    selected_mesh.export(selected_path)
    add_mesh.export(add_path)
    final_mesh.export(final_path)
    rows.sort(key=lambda r: (r["keep"], r["seed_faces"], r["score_mean"], r["cells"], r["faces"]), reverse=True)
    write_rows_csv(csv_path, rows[:400])
    write_map_png(
        png_path,
        [
            ("clean cells", clean_mask, (70, 150, 230)),
            ("seed0 cells", seed0_mask, (45, 95, 230)),
            ("seed2 cells", seed2_mask, (230, 125, 35)),
            ("support cells", support, (100, 100, 100)),
            ("selected full faces", selected_mask, (40, 190, 120)),
            ("added cells", add_mask, (40, 210, 190)),
            ("missing clean", clean_missing, (230, 70, 70)),
            ("missing final", final_missing, (180, 50, 170)),
            ("recovered", recovered, (50, 185, 70)),
        ],
        args.png_scale,
    )

    def count(mask):
        return int(np.count_nonzero(mask))

    def ratio(mask):
        return float(count(mask) / max(count(point_union), 1))

    debug = {
        "mesh_id": mesh_id,
        "status": "ok",
        "inputs": {
            "data_dir": str(data_dir),
            "full_mesh": str(resolve_full_mesh(args)),
            "raw": str(raw_path),
            "clean": str(clean_path),
            **paths,
        },
        "outputs": {
            "selected_surface": str(selected_path),
            "add_faces": str(add_path),
            "union_clean_candidate": str(final_path),
            "evidence": str(evidence_path),
            "debug": str(debug_path),
            "superfaces_csv": str(csv_path),
            "evidence_map_png": str(png_path),
        },
        "parameters": vars(args),
        "counts": {
            "seed0_points_filtered": int(len(p0_l)),
            "seed2_points_filtered": int(len(p2_l)),
            "seed0_cells": count(seed0_mask),
            "seed2_cells": count(seed2_mask),
            "point_union_cells": count(point_union),
            "clean_cells": count(clean_mask),
            "raw_cells": count(raw_mask),
            "support_cells": count(support),
            "seed_faces": int(np.count_nonzero(seed_face)),
            "candidate_faces": int(np.count_nonzero(candidate)),
            "superfaces": int(len(superfaces)),
            "kept_superfaces": int(len(keep)),
            "selected_raw_faces": int(selected_raw.size),
            "selected_pruned_faces": int(selected.size),
            "added_faces": int(add_faces.size),
            "clean_missing_cells": count(clean_missing),
            "final_missing_cells": count(final_missing),
            "recovered_cells": count(recovered),
            "lost_vs_clean_cells": count(lost_vs_clean),
        },
        "ratios_over_point_union_cells": {
            "clean_missing": ratio(clean_missing),
            "final_missing": ratio(final_missing),
            "recovered": ratio(recovered),
            "lost_vs_clean": ratio(lost_vs_clean),
        },
        "stats": {
            "raw": mesh_stats(raw),
            "clean": mesh_stats(clean),
            "selected_surface": mesh_stats(selected_mesh),
            "add_faces": mesh_stats(add_mesh),
            "union_clean_candidate": mesh_stats(final_mesh),
        },
        "kept_superface_summaries": [row for row in rows if row.get("keep")][:120],
    }
    np.savez_compressed(
        evidence_path,
        selected_faces=selected.astype(np.int64),
        add_faces=add_faces.astype(np.int64),
        candidate_faces=np.where(candidate)[0].astype(np.int64),
        seed_faces=np.where(seed_face)[0].astype(np.int64),
        face_to_superface=face_to_sf.astype(np.int64),
        kept_superfaces=np.array(sorted(keep), dtype=np.int64),
        clean_mask=clean_mask,
        raw_mask=raw_mask,
        seed0_mask=seed0_mask,
        seed2_mask=seed2_mask,
        support=support,
        final_mask=final_mask,
        point_union=point_union,
    )
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2)
    return debug


def main():
    args = parse_args()
    full_path = resolve_full_mesh(args)
    if not full_path:
        raise FileNotFoundError("Could not find a full-scene mesh. Pass --full_mesh explicitly.")
    args.full_mesh = full_path
    mesh_ids = discover_mesh_ids(Path(args.data_dir)) if args.all_meshes else args.mesh_ids
    full = load_mesh(full_path)

    summaries = []
    for mesh_id in mesh_ids:
        summary = process_mesh(args, full, mesh_id)
        summaries.append(summary)
        print(json.dumps({
            "mesh_id": mesh_id,
            "status": summary.get("status"),
            "candidate": summary.get("outputs", {}).get("union_clean_candidate"),
            "counts": summary.get("counts"),
            "ratios": summary.get("ratios_over_point_union_cells"),
        }, indent=2))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"superface_fullscene_{args.mode}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(json.dumps({"summary": str(summary_path), "processed": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
