"""Seeded full-scene re-segmentation for broken or incomplete part surfaces.

This recovers actual faces from the full input mesh. It uses the current clean
mesh only as a trusted seed/core, grows over the full-scene face adjacency
graph, and uses plane, normal, depthvote, and face co-occurrence evidence as
gates. It does not synthesize grid patches.
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


def first_existing_glob(patterns: list[Path | str]) -> str:
    for pattern in patterns:
        for hit in sorted(glob.glob(str(pattern))):
            if Path(hit).is_file():
                return hit
    return ""


def first_existing_path(paths: list[Path | str]) -> str:
    for path in paths:
        if Path(path).is_file():
            return str(path)
    return ""


def resolve_default_paths(args):
    lab_root = Path(args.lab_root)
    if args.data_dir is None:
        args.data_dir = str(lab_root / "data" / "example_input_scene_surface_lab")
    data_dir = Path(args.data_dir)
    mesh_id = args.mesh_id

    if args.full_mesh is None:
        args.full_mesh = first_existing_path([
            data_dir / "mesh_aligned_0.05.ply",
            repo_root() / "data" / "example_input_scene" / "mesh_aligned_0.05.ply",
            data_dir / "mesh_aligned_0.02_remeshed.ply",
        ])

    if args.points_seed0 is None:
        args.points_seed0 = first_existing_glob([
            lab_root / "outputs" / "sam2_maskprop_depthvote_0521" / mesh_id / "*depthvote_points.ply",
            lab_root / "outputs" / "depthvote_labdata_0520" / mesh_id / "*depthvote_points.ply",
            lab_root / "outputs" / "dense_depthvote_labdata_0520" / mesh_id / "*depthvote_points.ply",
        ])
    if args.points_seed2 is None:
        args.points_seed2 = first_existing_glob([
            lab_root / "outputs" / "sam2_maskprop_depthvote_seed2_conda118_0521" / mesh_id / "*depthvote_points.ply",
            lab_root / "outputs" / "sam2_maskprop_depthvote_seed2_0521" / mesh_id / "*depthvote_points.ply",
            lab_root / "outputs" / "sam2_maskprop_depthvote_seed2_external_0521" / mesh_id / "*depthvote_points.ply",
        ])
    if args.faceco_seed0 is None:
        args.faceco_seed0 = first_existing_glob([
            lab_root / "outputs" / "sam2_face_cooccurrence_relaxed_evidence_0521" / mesh_id / "*face_cooccurrence_evidence.npz",
            lab_root / "outputs" / "sam2_face_cooccurrence_sweep_mc25_0521" / mesh_id / "*face_cooccurrence_evidence.npz",
        ])
    if args.faceco_seed2 is None:
        args.faceco_seed2 = first_existing_glob([
            lab_root / "outputs" / "sam2_face_cooccurrence_seed2_0521" / mesh_id / "*face_cooccurrence_evidence.npz",
        ])
    if args.paper_evidence is None:
        args.paper_evidence = first_existing_glob([
            lab_root / "outputs" / "paper_region_fill_sam2_faceco_evidence_0521" / mesh_id / "*paper_region_evidence.npz",
            lab_root / "outputs" / "paper_region_fill_labdata_0520" / mesh_id / "*paper_region_evidence.npz",
        ])
    return args


def parse_args():
    root = repo_root()
    lab_root = default_lab_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab_root", default=str(lab_root))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--mesh_id", default="mesh_030")
    parser.add_argument("--full_mesh", default=None)
    parser.add_argument("--points_seed0", default=None)
    parser.add_argument("--points_seed2", default=None)
    parser.add_argument("--faceco_seed0", default=None)
    parser.add_argument("--faceco_seed2", default=None)
    parser.add_argument("--paper_evidence", default=None)
    parser.add_argument("--output_dir", default=str(root / "outputs" / "seeded_fullscene_resegment_general_0521"))
    parser.add_argument("--mode", choices=["strict", "balanced", "recall"], default="balanced")
    parser.add_argument("--grid_res", type=float, default=0.015)
    parser.add_argument("--grid_pad", type=float, default=0.045)
    parser.add_argument("--bbox_pad", type=float, default=0.075)
    parser.add_argument("--point_band", type=float, default=0.080)
    parser.add_argument("--seed_band", type=float, default=0.035)
    parser.add_argument("--grow_band", type=float, default=0.070)
    parser.add_argument("--final_band", type=float, default=0.060)
    parser.add_argument("--normal_deg", type=float, default=42.0)
    parser.add_argument("--min_points_per_cell", type=int, default=2)
    parser.add_argument("--support_close_iters", type=int, default=1)
    parser.add_argument("--support_dilate_iters", type=int, default=1)
    parser.add_argument("--seed_erode_iters", type=int, default=0)
    parser.add_argument("--cell_depth_slack", type=float, default=0.018)
    parser.add_argument("--secondary_max_components", type=int, default=8)
    parser.add_argument("--secondary_min_cells", type=int, default=6)
    parser.add_argument("--secondary_max_seed_cell_dist", type=float, default=7.0)
    parser.add_argument("--merge_digits", type=int, default=5)
    parser.add_argument("--png_scale", type=int, default=7)
    return resolve_default_paths(parser.parse_args())


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh at {path}, got {type(mesh)}")
    return mesh


def load_points(path: str | Path) -> np.ndarray:
    if not path or not Path(path).exists():
        return np.zeros((0, 3), dtype=np.float64)
    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
    if hasattr(obj, "vertices"):
        return np.asarray(obj.vertices, dtype=np.float64)
    raise TypeError(f"Unsupported point source {path}: {type(obj)}")


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


def fit_pca(mesh: trimesh.Trimesh):
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
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


def face_centroids(mesh: trimesh.Trimesh) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return verts[faces].mean(axis=1)


def cells_from_uv(uv: np.ndarray, uv_min: np.ndarray, shape: np.ndarray, grid_res: float):
    cells = np.floor((uv - uv_min) / max(grid_res, EPS)).astype(np.int64)
    valid = np.all((cells >= 0) & (cells < shape), axis=1)
    return cells, valid


def occupancy_from_uv(uv: np.ndarray, uv_min: np.ndarray, shape: np.ndarray, grid_res: float) -> np.ndarray:
    occ = np.zeros(tuple(shape), dtype=np.int32)
    if len(uv) == 0:
        return occ
    cells, valid = cells_from_uv(uv, uv_min, shape, grid_res)
    if np.any(valid):
        c = cells[valid]
        np.add.at(occ, (c[:, 0], c[:, 1]), 1)
    return occ


def filter_points(points: np.ndarray, raw_min: np.ndarray, raw_max: np.ndarray, local_pts: np.ndarray, band: float):
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
    if not path or not Path(path).exists():
        return {
            "selected": np.zeros(n_faces, dtype=bool),
            "pool": np.zeros(n_faces, dtype=bool),
            "node": np.zeros(n_faces, dtype=bool),
            "hit": np.zeros(n_faces, dtype=np.int64),
        }
    data = np.load(path)
    selected = mask_from_face_ids(n_faces, data["selected_faces"]) if "selected_faces" in data.files else np.zeros(n_faces, dtype=bool)
    pool_key = "pool_faces" if "pool_faces" in data.files else "pool_face" if "pool_face" in data.files else None
    pool = mask_from_face_ids(n_faces, data[pool_key]) if pool_key else np.zeros(n_faces, dtype=bool)
    node = np.asarray(data["node_mask"], dtype=bool) if "node_mask" in data.files else np.zeros(n_faces, dtype=bool)
    if node.size != n_faces:
        fixed = np.zeros(n_faces, dtype=bool)
        fixed[: min(n_faces, node.size)] = node[: min(n_faces, node.size)]
        node = fixed
    hit = np.asarray(data["hit_count"], dtype=np.int64) if "hit_count" in data.files else np.zeros(n_faces, dtype=np.int64)
    if hit.size != n_faces:
        fixed = np.zeros(n_faces, dtype=np.int64)
        fixed[: min(n_faces, hit.size)] = hit[: min(n_faces, hit.size)]
        hit = fixed
    return {"selected": selected, "pool": pool, "node": node, "hit": hit}


class UnionFind:
    def __init__(self, n: int, active: np.ndarray):
        self.parent = np.full(n, -1, dtype=np.int64)
        ids = np.where(active)[0]
        self.parent[ids] = ids

    def find(self, x: int) -> int:
        p = self.parent[x]
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return int(p)

    def union(self, a: int, b: int):
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra != rb:
            self.parent[rb] = ra

    def roots_for_active(self, active_ids: np.ndarray) -> np.ndarray:
        roots = np.empty(len(active_ids), dtype=np.int64)
        for i, fid in enumerate(active_ids.tolist()):
            roots[i] = self.find(fid)
        return roots


def component_rows(face_ids: np.ndarray, roots: np.ndarray, seed_face: np.ndarray, cells: np.ndarray, valid: np.ndarray, shape: np.ndarray, score: np.ndarray, hit0: np.ndarray, hit2: np.ndarray, paper: np.ndarray, dist_seed: np.ndarray):
    rows = []
    for rank, root in enumerate(np.unique(roots), start=1):
        comp = face_ids[roots == root]
        ok = valid[comp]
        cell_mask = np.zeros(tuple(shape), dtype=bool)
        min_seed_dist = 9999.0
        if np.any(ok):
            c = cells[comp[ok]]
            cell_mask[c[:, 0], c[:, 1]] = True
            min_seed_dist = float(np.min(dist_seed[c[:, 0], c[:, 1]]))
        rows.append({
            "root": int(root),
            "rank": int(rank),
            "faces": int(comp.size),
            "cells": int(np.count_nonzero(cell_mask)),
            "seed_faces": int(np.count_nonzero(seed_face[comp])),
            "score_mean": float(np.mean(score[comp])) if comp.size else 0.0,
            "hit0_mean": float(np.mean(hit0[comp])) if comp.size else 0.0,
            "hit2_mean": float(np.mean(hit2[comp])) if comp.size else 0.0,
            "paper_ratio": float(np.count_nonzero(paper[comp]) / max(comp.size, 1)),
            "min_seed_cell_dist": min_seed_dist,
        })
    rows.sort(key=lambda r: (r["seed_faces"] > 0, r["cells"], r["score_mean"], r["faces"]), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def prune_by_cell_depth(face_ids: np.ndarray, face_z: np.ndarray, cells: np.ndarray, valid: np.ndarray, final_band: float, slack: float) -> np.ndarray:
    if face_ids.size == 0:
        return face_ids
    selected = np.zeros(face_z.shape[0], dtype=bool)
    selected[face_ids] = True
    valid_ids = face_ids[valid[face_ids]]
    if valid_ids.size == 0:
        return face_ids[np.abs(face_z[face_ids]) <= final_band]
    cell_key = cells[valid_ids, 0].astype(np.int64) << 32
    cell_key = cell_key + cells[valid_ids, 1].astype(np.int64)
    keep = np.zeros(face_z.shape[0], dtype=bool)
    for key in np.unique(cell_key):
        ids = valid_ids[cell_key == key]
        absz = np.abs(face_z[ids])
        best = float(absz.min())
        local_keep = ids[(absz <= best + slack) & (absz <= final_band)]
        if local_keep.size == 0:
            local_keep = ids[[int(np.argmin(absz))]]
        keep[local_keep] = True
    return np.where(keep & selected)[0]


def concat_clean_add(clean: trimesh.Trimesh, add: trimesh.Trimesh, merge_digits: int) -> trimesh.Trimesh:
    out = trimesh.util.concatenate([clean, add])
    if merge_digits >= 0 and len(out.vertices):
        out.merge_vertices(digits_vertex=merge_digits)
        out.remove_unreferenced_vertices()
    return out


def write_components_csv(path: Path, rows: list[dict]):
    fields = ["rank", "root", "keep", "reason", "faces", "cells", "seed_faces", "score_mean", "hit0_mean", "hit2_mean", "paper_ratio", "min_seed_cell_dist"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


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


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    mesh_dir = data_dir / "perception" / "vis_groups_final_mesh" / "real" / args.mesh_id
    full = load_mesh(args.full_mesh)
    raw = load_mesh(mesh_dir / "mask_mesh.ply")
    clean = load_mesh(mesh_dir / "mask_mesh_refined_clean.ply")
    p0 = load_points(args.points_seed0)
    p2 = load_points(args.points_seed2)

    center, u_axis, v_axis, normal = fit_pca(clean)
    raw_vertices = np.asarray(raw.vertices, dtype=np.float64)
    raw_min = raw_vertices.min(axis=0) - args.bbox_pad
    raw_max = raw_vertices.max(axis=0) + args.bbox_pad

    raw_local = local3(face_centroids(raw), center, u_axis, v_axis, normal)
    clean_local = local3(face_centroids(clean), center, u_axis, v_axis, normal)
    p0_local = local3(p0, center, u_axis, v_axis, normal)
    p2_local = local3(p2, center, u_axis, v_axis, normal)
    p0_f, p0_l = filter_points(p0, raw_min, raw_max, p0_local, args.point_band)
    p2_f, p2_l = filter_points(p2, raw_min, raw_max, p2_local, args.point_band)

    uv_sources = [raw_local[:, :2], clean_local[:, :2]]
    if len(p0_l):
        uv_sources.append(p0_l[:, :2])
    if len(p2_l):
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
    seed_cells = clean_mask.copy()
    if args.seed_erode_iters > 0:
        eroded = ndimage.binary_erosion(seed_cells, structure=np.ones((3, 3), dtype=bool), iterations=args.seed_erode_iters)
        if np.any(eroded):
            seed_cells = eroded

    full_vertices = np.asarray(full.vertices, dtype=np.float64)
    full_faces = np.asarray(full.faces, dtype=np.int64)
    centroids = full_vertices[full_faces].mean(axis=1)
    face_local = local3(centroids, center, u_axis, v_axis, normal)
    face_cells, valid = cells_from_uv(face_local[:, :2], uv_min, shape, args.grid_res)
    face_normals = np.asarray(full.face_normals, dtype=np.float64)

    n_faces = len(full_faces)
    fc0 = load_face_evidence(args.faceco_seed0, n_faces)
    fc2 = load_face_evidence(args.faceco_seed2, n_faces)
    paper = load_face_evidence(args.paper_evidence, n_faces)
    strong_evidence = (
        fc0["selected"]
        | fc2["selected"]
        | paper["selected"]
        | ((fc0["hit"] >= 2) & paper["pool"])
        | ((fc2["hit"] >= 2) & (paper["pool"] | fc2["node"]))
    )

    in_box = np.all((centroids >= raw_min) & (centroids <= raw_max), axis=1)
    normal_ok = np.abs(face_normals @ normal) >= math.cos(math.radians(args.normal_deg))
    plane_seed = np.abs(face_local[:, 2]) <= args.seed_band
    plane_grow = np.abs(face_local[:, 2]) <= args.grow_band

    in_support = np.zeros(n_faces, dtype=bool)
    in_seed_cells = np.zeros(n_faces, dtype=bool)
    valid_ids = np.where(valid)[0]
    vc = face_cells[valid_ids]
    in_support[valid_ids] = support[vc[:, 0], vc[:, 1]]
    in_seed_cells[valid_ids] = seed_cells[vc[:, 0], vc[:, 1]]

    seed_face = in_box & valid & normal_ok & plane_seed & in_seed_cells
    candidate_good = in_box & valid & normal_ok & plane_grow & (in_support | strong_evidence)
    candidate_good |= seed_face

    good_ids = np.where(candidate_good)[0]
    uf = UnionFind(n_faces, candidate_good)
    adj = np.asarray(full.face_adjacency, dtype=np.int64)
    adj_good = adj[candidate_good[adj[:, 0]] & candidate_good[adj[:, 1]]]
    for a, b in adj_good.tolist():
        uf.union(a, b)
    roots = uf.roots_for_active(good_ids)

    dist_seed = ndimage.distance_transform_edt(~seed_cells)
    score = (
        1.0 * in_support.astype(np.float64)
        + 1.5 * strong_evidence.astype(np.float64)
        + np.clip(fc0["hit"], 0, 4) / 4.0
        + np.clip(fc2["hit"], 0, 4) / 4.0
        + 0.4 * paper["pool"].astype(np.float64)
    )
    rows = component_rows(good_ids, roots, seed_face, face_cells, valid, shape, score, fc0["hit"], fc2["hit"], paper["pool"] | paper["selected"], dist_seed)

    keep_roots = set()
    for row in rows:
        row["keep"] = False
        row["reason"] = ""
        if row["seed_faces"] > 0:
            row["keep"] = True
            row["reason"] = "seed-connected"
            keep_roots.add(row["root"])

    secondary = []
    if args.mode in {"balanced", "recall"}:
        for row in rows:
            if row["keep"]:
                continue
            if (
                row["cells"] >= args.secondary_min_cells
                and row["min_seed_cell_dist"] <= args.secondary_max_seed_cell_dist
                and (row["score_mean"] >= (1.4 if args.mode == "balanced" else 1.1) or row["paper_ratio"] >= 0.25)
            ):
                secondary.append(row)
        secondary.sort(key=lambda r: (r["score_mean"], r["cells"], r["faces"]), reverse=True)
        for row in secondary[: args.secondary_max_components]:
            row["keep"] = True
            row["reason"] = "nearby-evidence-component"
            keep_roots.add(row["root"])

    keep_root_arr = np.array(sorted(keep_roots), dtype=np.int64)
    selected_raw = good_ids[np.isin(roots, keep_root_arr)] if keep_root_arr.size else np.zeros(0, dtype=np.int64)
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

    out_dir = Path(args.output_dir) / args.mesh_id / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.mesh_id}_seeded_fullscene_{args.mode}"
    selected_path = out_dir / f"{stem}_selected_surface.ply"
    add_path = out_dir / f"{stem}_add_faces.ply"
    final_path = out_dir / f"{stem}_union_clean_candidate.ply"
    debug_path = out_dir / f"{stem}_debug.json"
    evidence_path = out_dir / f"{stem}_evidence.npz"
    csv_path = out_dir / f"{stem}_components.csv"
    png_path = out_dir / f"{stem}_evidence_map.png"
    selected_mesh.export(selected_path)
    add_mesh.export(add_path)
    final_mesh.export(final_path)
    write_components_csv(csv_path, rows[:200])
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
        "inputs": {
            "data_dir": str(data_dir),
            "full_mesh": str(args.full_mesh),
            "raw": str(mesh_dir / "mask_mesh.ply"),
            "clean": str(mesh_dir / "mask_mesh_refined_clean.ply"),
            "points_seed0": str(args.points_seed0),
            "points_seed2": str(args.points_seed2),
            "faceco_seed0": str(args.faceco_seed0),
            "faceco_seed2": str(args.faceco_seed2),
            "paper_evidence": str(args.paper_evidence),
        },
        "outputs": {
            "selected_surface": str(selected_path),
            "add_faces": str(add_path),
            "union_clean_candidate": str(final_path),
            "debug": str(debug_path),
            "evidence": str(evidence_path),
            "components_csv": str(csv_path),
            "evidence_map_png": str(png_path),
        },
        "parameters": vars(args),
        "grid": {"shape": [int(shape[0]), int(shape[1])], "uv_min": uv_min.tolist(), "uv_max": uv_max.tolist()},
        "counts": {
            "seed0_points_filtered": int(len(p0_f)),
            "seed2_points_filtered": int(len(p2_f)),
            "seed0_cells": count(seed0_mask),
            "seed2_cells": count(seed2_mask),
            "point_union_cells": count(point_union),
            "clean_cells": count(clean_mask),
            "raw_cells": count(raw_mask),
            "support_cells": count(support),
            "seed_faces": int(np.count_nonzero(seed_face)),
            "candidate_good_faces": int(np.count_nonzero(candidate_good)),
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
        "component_summaries": rows[:80],
    }
    np.savez_compressed(
        evidence_path,
        selected_faces=selected.astype(np.int64),
        add_faces=add_faces.astype(np.int64),
        candidate_good=np.where(candidate_good)[0].astype(np.int64),
        seed_face=np.where(seed_face)[0].astype(np.int64),
        clean_mask=clean_mask,
        raw_mask=raw_mask,
        seed0_mask=seed0_mask,
        seed2_mask=seed2_mask,
        support=support,
        selected_mask=selected_mask,
        add_mask=add_mask,
        final_mask=final_mask,
        point_union=point_union,
    )
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2)
    print(json.dumps({
        "mode": args.mode,
        "selected_surface": str(selected_path),
        "add_faces": str(add_path),
        "union_clean_candidate": str(final_path),
        "debug": str(debug_path),
        "png": str(png_path),
        "counts": debug["counts"],
        "ratios": debug["ratios_over_point_union_cells"],
        "stats": debug["stats"]["union_clean_candidate"],
        "kept_components": [row for row in rows if row.get("keep")][:12],
    }, indent=2))


if __name__ == "__main__":
    main()
