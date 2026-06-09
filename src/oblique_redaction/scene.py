"""Scene reconstruction: LAS clip → voxel-downsample → 2D Delaunay TIN → Embree raycaster.

Pipeline (top-down):
    build_scene(las_dir, aoi_polygon, voxel_size_m=1.0, buffer_m=200) → SceneMesh

A `SceneMesh` bundles:
    - the trimesh `Trimesh` (vertices in EPSG:3011 / RH2000)
    - a `RayMeshIntersector` ready for batched first-hit queries
    - a `sensitive_face_mask` boolean over faces, True where the face's xy centroid
      lies inside the AOI polygon

Why a max-z voxel downsample
----------------------------
Triangulating the raw LAS (~28 M pts/tile) is infeasible. Voxel-downsampling to a
~1 m grid (keeping the *highest* z per voxel) collapses each xy cell to a single
point representing the visible top surface. The resulting Delaunay TIN over the
downsampled points behaves like a DSM-derived heightfield mesh — no overhangs,
but sloped triangles bridge walls, which is what we need for occlusion testing
on oblique frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import laspy
import numpy as np
import shapely
import trimesh
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

from .timing import log, step


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class SceneMesh:
    mesh: trimesh.Trimesh
    intersector: trimesh.ray.ray_pyembree.RayMeshIntersector
    sensitive_face_mask: np.ndarray   # (F,) bool
    point_count: int
    bbox: tuple[float, float, float, float]   # (xmin, ymin, xmax, ymax) in EPSG:3011

    @property
    def n_faces(self) -> int:
        return len(self.mesh.faces)

    @property
    def n_sensitive_faces(self) -> int:
        return int(self.sensitive_face_mask.sum())


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------

def find_tiles_intersecting_bbox(
    las_dir: Path,
    bbox: tuple[float, float, float, float],
) -> list[Path]:
    """Return LAS files in `las_dir` whose header bbox intersects `bbox` (xmin,ymin,xmax,ymax)."""
    xmin, ymin, xmax, ymax = bbox
    out: list[Path] = []
    for path in sorted(las_dir.glob("*.las")) + sorted(las_dir.glob("*.laz")):
        with laspy.open(path) as fh:
            h = fh.header
            tx0, ty0 = h.mins[0], h.mins[1]
            tx1, ty1 = h.maxs[0], h.maxs[1]
        if tx1 < xmin or tx0 > xmax or ty1 < ymin or ty0 > ymax:
            continue
        out.append(path)
    return out


# ---------------------------------------------------------------------------
# Streaming LAS clip
# ---------------------------------------------------------------------------

CHUNK = 2_000_000


def load_points_in_bbox(
    las_paths: Iterable[Path],
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Read points from each LAS file, filter to bbox, return concatenated (N,3) float64.

    Streams in chunks so memory stays bounded by ~CHUNK points per tile rather than
    the whole file. Uses scaled int coords from laspy (`x`, `y`, `z` properties).
    """
    xmin, ymin, xmax, ymax = bbox
    paths = list(las_paths)
    chunks: list[np.ndarray] = []
    total_in = 0
    total_out = 0

    with step(f"Loading + clipping point cloud ({len(paths)} tiles)") as s:
        for i, path in enumerate(paths):
            with laspy.open(path) as fh:
                n_total = fh.header.point_count
                kept_this_tile = 0
                for chunk in fh.chunk_iterator(CHUNK):
                    x = np.asarray(chunk.x, dtype=np.float64)
                    y = np.asarray(chunk.y, dtype=np.float64)
                    z = np.asarray(chunk.z, dtype=np.float64)
                    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
                    if mask.any():
                        chunks.append(np.column_stack([x[mask], y[mask], z[mask]]))
                        kept_this_tile += int(mask.sum())
                total_in += n_total
                total_out += kept_this_tile
            log.info(
                f"  · tile {i + 1}/{len(paths)} {path.name}: "
                f"kept {kept_this_tile:,}/{n_total:,} pts"
            )
            s.tick(i + 1)

    if not chunks:
        raise RuntimeError(f"no LAS points found inside bbox {bbox}")

    points = np.concatenate(chunks)
    log.info(
        f"  total: kept {total_out:,} of {total_in:,} pts "
        f"({100.0 * total_out / max(total_in, 1):.2f}%)"
    )
    return points


# ---------------------------------------------------------------------------
# Voxel downsample (max-z per xy cell — DSM-like)
# ---------------------------------------------------------------------------

def voxel_downsample_max_z(points: np.ndarray, voxel_xy: float) -> np.ndarray:
    """Keep one point per xy voxel — the one with the largest z.

    voxel_xy is the xy cell size in meters. Z is not bucketed; we keep the actual
    point's z (no quantisation), only its xy is used for cell lookup.
    """
    if voxel_xy <= 0:
        return points
    with step(f"Voxel downsample (xy={voxel_xy} m)"):
        # int keys per cell
        ix = np.floor(points[:, 0] / voxel_xy).astype(np.int64)
        iy = np.floor(points[:, 1] / voxel_xy).astype(np.int64)
        # combine into a single int64 (assume coordinate range fits — Stockholm SWEREF
        # values are < 1e7, /1m = 1e7; both axes fit comfortably in 32 bits each)
        key = (ix.astype(np.int64) << 32) | (iy.astype(np.int64) & 0xFFFFFFFF)
        # sort by (key ascending, z descending) so the first hit per group is max-z
        order = np.lexsort((-points[:, 2], key))
        sorted_keys = key[order]
        first_in_group = np.empty(len(order), dtype=bool)
        first_in_group[0] = True
        first_in_group[1:] = sorted_keys[1:] != sorted_keys[:-1]
        kept = points[order[first_in_group]]
        log.info(f"  · downsampled {len(points):,} → {len(kept):,} points")
    return kept


# ---------------------------------------------------------------------------
# TIN + sensitive face tagging
# ---------------------------------------------------------------------------

def build_tin(
    points: np.ndarray,
    polygon: Polygon,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """2D Delaunay over (x,y); lift each vertex to its z; tag sensitive faces.

    Returns (mesh, sensitive_face_mask).
    """
    with step(f"Delaunay triangulation (n={len(points):,})"):
        tri = Delaunay(points[:, :2])
        faces = tri.simplices  # (F, 3) indices into points
        log.info(f"  · {len(faces):,} triangles")

    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)

    with step(f"Tagging sensitive faces ({len(faces):,})"):
        # centroid xy of each triangle, then vectorised contains_xy from shapely 2.x
        centroids_xy = points[faces, :2].mean(axis=1)   # (F, 2)
        sens = shapely.contains_xy(polygon, centroids_xy[:, 0], centroids_xy[:, 1])
    log.info(f"  · {int(sens.sum()):,} sensitive triangles")
    return mesh, sens


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_scene(
    las_dir: Path,
    polygon: Polygon,
    *,
    buffer_m: float = 200.0,
    voxel_size_m: float = 1.0,
) -> SceneMesh:
    """End-to-end: pick tiles → clip → downsample → triangulate → tag → raycaster."""
    pminx, pminy, pmaxx, pmaxy = polygon.bounds
    bbox = (
        pminx - buffer_m,
        pminy - buffer_m,
        pmaxx + buffer_m,
        pmaxy + buffer_m,
    )
    log.info(f"scene bbox (AOI ⊕ {buffer_m:g} m): {bbox}")

    with step("Selecting LAS tiles"):
        tile_paths = find_tiles_intersecting_bbox(las_dir, bbox)
        log.info(f"  → {len(tile_paths)} tile(s): {[p.name for p in tile_paths]}")
    if not tile_paths:
        raise RuntimeError(f"no LAS tiles in {las_dir} intersect {bbox}")

    points = load_points_in_bbox(tile_paths, bbox)
    points = voxel_downsample_max_z(points, voxel_size_m)
    mesh, sens = build_tin(points, polygon)

    with step("Building Embree raycaster"):
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    return SceneMesh(
        mesh=mesh,
        intersector=intersector,
        sensitive_face_mask=sens,
        point_count=len(points),
        bbox=bbox,
    )
