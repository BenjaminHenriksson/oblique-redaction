"""Camera-model sanity check (HARD GATE).

Projects building footprints from Byggnad.gpkg into the image using each candidate
rotation convention. Saves a downsampled image overlay per convention. We then
inspect them by eye and pick the convention whose outlines line up with the
visible buildings.

Run as a script:
    uv run python tests/test_camera_sanity.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pyogrio.raw import read as ogr_read
from shapely import from_wkb

# allow `python tests/...` invocation without installing in dev mode (we are installed,
# but this keeps the script self-contained anyway)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oblique_redaction.camera import (  # noqa: E402
    ROTATION_CONVENTIONS,
    build_camera,
)
from oblique_redaction.timing import init_logger, log, step  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# Image, EO and building-footprints paths must be provided at runtime via CLI
# flags — they depend on your local data layout.
OUT_DIR = Path("/tmp/oblique_redaction_camera_check")

# Image will be downsampled by this factor for the overlay (full image is 462 MB)
DOWNSAMPLE = 8

# Roof height to add to each footprint's base z when drawing the "rooftop" outline
ROOF_LIFT_M = 20.0

# Search radius around the camera position for buildings to project (square half-side)
SEARCH_RADIUS_M = 2500.0


def read_image_downsampled(tif_path: Path, factor: int) -> np.ndarray:
    """Return an (H', W', 3) uint8 numpy array of the image downsampled by `factor`."""
    with step(f"Reading image (downsampled 1/{factor})"):
        with rasterio.open(tif_path) as src:
            H = src.height
            W = src.width
            out_h = H // factor
            out_w = W // factor
            log.info(f"  source {W}x{H} -> downsampled {out_w}x{out_h}")
            arr = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=rasterio.enums.Resampling.average,
            )
        # rasterio returns (bands, H, W); transpose to (H, W, bands)
        return np.transpose(arr, (1, 2, 0))


def read_building_footprints(gpkg_path: Path, bbox: tuple[float, float, float, float]):
    """Yield (outer_xy_rings, base_z) for each footprint in bbox.

    `outer_xy_rings` is a list of (N,2) numpy arrays — the outer rings of each polygon
    that makes up the (Multi)Polygon. `base_z` is the (constant per footprint) ground z.
    """
    with step(f"Reading Byggnad footprints in bbox {bbox}"):
        meta, fids, geoms_wkb, fields = ogr_read(
            gpkg_path,
            layer="Byggnad_yta",
            bbox=bbox,
            return_fids=False,
        )
    log.info(f"  {len(geoms_wkb)} footprints loaded ({meta.get('crs')})")
    out = []
    for raw in geoms_wkb:
        g = from_wkb(bytes(raw))
        parts = list(g.geoms) if g.geom_type.startswith("Multi") else [g]
        rings = []
        zs = []
        for poly in parts:
            coords = np.asarray(poly.exterior.coords, dtype=np.float64)
            rings.append(coords[:, :2])
            zs.extend(coords[:, 2].tolist())
        if not zs:
            continue
        base_z = float(np.median(zs))
        out.append((rings, base_z))
    return out


def project_polygon_to_image(camera, ring_xy: np.ndarray, z: float) -> np.ndarray | None:
    """Project a closed (N,2) ring at constant z to image pixels.

    Returns (N,2) float32 pixel coords, or None if the ring is entirely behind the
    camera or in front but degenerate. We do NOT clip to image bounds — drawing handles
    out-of-frame vertices.
    """
    pts = np.column_stack([ring_xy, np.full(len(ring_xy), z)])
    uv, depth = camera.world_to_image(pts)
    if not np.any(depth > 0):
        return None
    return uv.astype(np.float32)


def draw_overlay(
    image_np: np.ndarray,
    camera,
    footprints,
    factor: int,
    title: str,
    out_path: Path,
) -> int:
    """Draw projected building outlines on top of the downsampled image.

    Each footprint is drawn at base z (red, thick) and at base+ROOF_LIFT_M (yellow, thick).
    A green dot marks each footprint centroid (base z) to make positions readable even when
    outlines are tangled. Returns count of footprints with vertices in-image.
    """
    base_img = Image.fromarray(image_np).convert("RGBA")
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    H, W = image_np.shape[:2]
    n_in = 0

    for rings, base_z in footprints:
        for ring_xy in rings:
            uv_base = project_polygon_to_image(camera, ring_xy, base_z)
            uv_roof = project_polygon_to_image(camera, ring_xy, base_z + ROOF_LIFT_M)
            any_in = False
            for uv, color in (
                (uv_base, (255, 50, 50, 220)),
                (uv_roof, (255, 235, 50, 220)),
            ):
                if uv is None:
                    continue
                px_ds = uv / factor
                if not np.all(np.isfinite(px_ds)):
                    continue
                px_ds = px_ds.astype(np.int32)
                in_mask = (
                    (px_ds[:, 0] >= 0)
                    & (px_ds[:, 0] < W)
                    & (px_ds[:, 1] >= 0)
                    & (px_ds[:, 1] < H)
                )
                if not in_mask.any():
                    continue
                any_in = True
                draw.line(
                    [tuple(p) for p in px_ds.tolist()],
                    fill=color,
                    width=3,
                )
            # centroid dot
            if any_in and uv_base is not None and np.all(np.isfinite(uv_base)):
                cx, cy = (uv_base.mean(axis=0) / factor).astype(np.int32)
                if 0 <= cx < W and 0 <= cy < H:
                    r = 3
                    draw.ellipse(
                        [cx - r, cy - r, cx + r, cy + r],
                        fill=(80, 255, 80, 230),
                        outline=(0, 0, 0, 255),
                    )
            if any_in:
                n_in += 1

    composed = Image.alpha_composite(base_img, overlay).convert("RGB")
    draw2 = ImageDraw.Draw(composed)
    draw2.rectangle([0, 0, 700, 28], fill=(0, 0, 0))
    draw2.text(
        (6, 6),
        f"{title}  | {n_in} footprints visible",
        fill=(255, 255, 255),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(out_path)
    log.info(f"  wrote {out_path} ({n_in} footprints had vertices in-image)")
    return n_in


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, required=True, help="Source TIFF.")
    ap.add_argument("--eo", type=Path, required=True, help="EO text file.")
    ap.add_argument(
        "--byggnad",
        type=Path,
        required=True,
        help="Building footprints (GeoPackage layer 'Byggnad_yta', or equivalent).",
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--downsample", type=int, default=DOWNSAMPLE)
    ap.add_argument("--radius", type=float, default=SEARCH_RADIUS_M)
    args = ap.parse_args()

    init_logger()
    log.info("=== Camera-model sanity check ===")
    log.info(f"image:    {args.image}")
    log.info(f"eo:       {args.eo}")
    log.info(f"byggnad:  {args.byggnad}")
    log.info(f"out_dir:  {args.out_dir}")

    # Build a default camera once just to get the position; the rotation isn't used
    # for the bbox computation
    cam0 = build_camera(args.image, args.eo, rotation_convention=ROTATION_CONVENTIONS[0])
    cx, cy, _cz = cam0.C_world.tolist()
    bbox = (cx - args.radius, cy - args.radius, cx + args.radius, cy + args.radius)
    log.info(f"search bbox: {bbox}")

    image_np = read_image_downsampled(args.image, args.downsample)
    footprints = read_building_footprints(args.byggnad, bbox)

    summary = []
    for conv in ROTATION_CONVENTIONS:
        with step(f"Convention: {conv}"):
            cam = build_camera(args.image, args.eo, rotation_convention=conv)
            out_path = args.out_dir / f"camera_check_{conv}.png"
            n_in = draw_overlay(
                image_np, cam, footprints, args.downsample, conv, out_path
            )
            summary.append((conv, n_in))

    log.info("=== Summary ===")
    for conv, n_in in summary:
        log.info(f"  {conv:>20s}: {n_in} polygon parts visible")
    log.info(f"Inspect {args.out_dir}/camera_check_*.png and pick the convention whose")
    log.info("outlines line up with visible buildings (red=base, yellow=base+20m).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
