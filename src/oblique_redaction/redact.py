"""Mask computation + pixelate/blur composite + GeoTIFF write.

Top-level entrypoint:
    redact_image(image_path, polygon, eo_path, las_dir, out_path, ...) → None

Pipeline (matches the plan):
    1. build Camera (camera.build_camera)
    2. build SceneMesh (scene.build_scene)
    3. compute the screen-space bbox of the AOI's vertical extrusion
    4. cast one ray per pixel inside that bbox via Embree (chunked, with progress)
    5. mark pixels whose first hit is a sensitive triangle
    6. composite pixelate + Gaussian blur into the source image inside the mask
    7. write the redacted GeoTIFF; side outputs <stem>_mask.tif and <stem>_debug.png

Image I/O uses rasterio so the source GeoTIFF profile (tags, tiling, photometric, etc.)
is preserved. Pixels are processed in a tight bbox window, not the full 10k×14k frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import shapely
from PIL import Image, ImageDraw, ImageFilter
from rasterio.windows import Window
from shapely.geometry import Polygon

from .camera import Camera, build_camera
from .scene import SceneMesh, build_scene
from .timing import log, step


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class RedactionResult:
    out_path: Path
    mask_path: Path
    debug_path: Path
    n_pixels_masked: int
    bbox_uv: tuple[int, int, int, int]   # (u0, v0, u1, v1) in source pixels


# ---------------------------------------------------------------------------
# Image-space bbox of the AOI
# ---------------------------------------------------------------------------

def aoi_screen_bbox(
    camera: Camera,
    polygon: Polygon,
    z_min: float,
    z_max: float,
    pad_px: int = 8,
) -> tuple[int, int, int, int]:
    """Project the AOI polygon vertices at z_min AND z_max to image space; return
    the axis-aligned bbox in source pixel coords clamped to image bounds.

    Returns (u0, v0, u1, v1) inclusive-exclusive.
    """
    ring_xy = np.asarray(polygon.exterior.coords, dtype=np.float64)[:, :2]
    pts = np.vstack(
        [
            np.column_stack([ring_xy, np.full(len(ring_xy), z_min)]),
            np.column_stack([ring_xy, np.full(len(ring_xy), z_max)]),
        ]
    )
    uv, depth = camera.world_to_image(pts)
    valid = depth > 0
    if not valid.any():
        raise RuntimeError("AOI is entirely behind the camera")
    uv = uv[valid]
    u0 = int(np.floor(uv[:, 0].min())) - pad_px
    v0 = int(np.floor(uv[:, 1].min())) - pad_px
    u1 = int(np.ceil(uv[:, 0].max())) + pad_px
    v1 = int(np.ceil(uv[:, 1].max())) + pad_px
    W, H = camera.intrinsics.width, camera.intrinsics.height
    u0 = max(0, min(W, u0))
    v0 = max(0, min(H, v0))
    u1 = max(0, min(W, u1))
    v1 = max(0, min(H, v1))
    if u1 <= u0 or v1 <= v0:
        raise RuntimeError("AOI projects outside image bounds")
    return u0, v0, u1, v1


# ---------------------------------------------------------------------------
# Raycasting → mask
# ---------------------------------------------------------------------------

# Ray batch size — embree handles ~10⁷ rays/s; chunks of 100k keep progress lines visible
RAY_BATCH = 100_000


def compute_mask(
    camera: Camera,
    scene: SceneMesh,
    bbox_uv: tuple[int, int, int, int],
) -> np.ndarray:
    """Cast one ray per pixel in `bbox_uv`. Return a uint8 mask of (H, W) where
    pixels are 1 iff the first ray hit is on a sensitive face.
    """
    u0, v0, u1, v1 = bbox_uv
    W, H = camera.intrinsics.width, camera.intrinsics.height
    bw, bh = u1 - u0, v1 - v0
    n = bw * bh

    # Pixel grid (centre-of-pixel sampling at +0.5)
    us, vs = np.meshgrid(
        np.arange(u0, u1, dtype=np.float64) + 0.5,
        np.arange(v0, v1, dtype=np.float64) + 0.5,
        indexing="xy",
    )
    pixels = np.column_stack([us.ravel(), vs.ravel()])   # (n, 2)

    mask_flat = np.zeros(n, dtype=np.uint8)

    with step("Raycasting", total=n) as s:
        for start in range(0, n, RAY_BATCH):
            end = min(start + RAY_BATCH, n)
            origins, directions = camera.image_to_ray(pixels[start:end])
            face_ids = scene.intersector.intersects_first(origins, directions)
            hit = face_ids >= 0
            sens_hit = np.zeros_like(hit)
            sens_hit[hit] = scene.sensitive_face_mask[face_ids[hit]]
            mask_flat[start:end] = sens_hit.astype(np.uint8)
            s.tick(end)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[v0:v1, u0:u1] = mask_flat.reshape(bh, bw)
    return mask


# ---------------------------------------------------------------------------
# Pixelate + Gaussian blur composite
# ---------------------------------------------------------------------------

def apply_redaction(
    image_np: np.ndarray,
    mask: np.ndarray,
    bbox_uv: tuple[int, int, int, int],
    *,
    pixelate_factor: int = 12,
    blur_sigma: float = 8.0,
) -> np.ndarray:
    """Pixelate + Gaussian blur the masked region inside `bbox_uv`. Composite back.

    Operates only on the bbox slice for speed; everywhere else is unchanged.
    """
    u0, v0, u1, v1 = bbox_uv
    with step("Pixelate + blur composite"):
        crop = image_np[v0:v1, u0:u1]                    # (h, w, 3)
        mask_crop = mask[v0:v1, u0:u1]                   # (h, w) uint8
        if not mask_crop.any():
            log.info("  · empty mask in bbox, returning original")
            return image_np
        h, w = crop.shape[:2]
        # PIL pipeline: pixelate via thumbnail+resize NEAREST, then GaussianBlur
        pil = Image.fromarray(crop)
        small_w = max(1, w // pixelate_factor)
        small_h = max(1, h // pixelate_factor)
        pil_small = pil.resize((small_w, small_h), Image.BOX)
        pil_pix = pil_small.resize((w, h), Image.NEAREST)
        pil_blur = pil_pix.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        redacted = np.asarray(pil_blur)
        # composite — only inside mask
        m3 = mask_crop[..., None].astype(bool)
        out = image_np.copy()
        out[v0:v1, u0:u1] = np.where(m3, redacted, crop)
        log.info(
            f"  · pixelate factor={pixelate_factor}, blur sigma={blur_sigma}, "
            f"replaced {int(mask_crop.sum()):,} px"
        )
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_geotiff(
    src_path: Path,
    out_path: Path,
    redacted: np.ndarray,
) -> None:
    """Copy the source TIFF profile and replace pixel data with `redacted`.

    Source is a multi-band uint8 TIFF (the UltraCam Level-3 product) without GCPs/CRS;
    we preserve `tags` (including ImageDescription) and tiling layout.
    """
    with step(f"Writing GeoTIFF \u2192 {out_path}"):
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            tags_default = src.tags()
            tags_per_band = [src.tags(b + 1) for b in range(src.count)]
        # rasterio expects (bands, H, W)
        if redacted.ndim == 3:
            data = np.transpose(redacted, (2, 0, 1))
        else:
            data = redacted[None, ...]
        profile.update(count=data.shape[0])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
            if tags_default:
                dst.update_tags(**tags_default)
            for b, t in enumerate(tags_per_band, start=1):
                if t:
                    dst.update_tags(b, **t)


def write_mask_tif(
    src_path: Path,
    mask: np.ndarray,
    out_path: Path,
) -> None:
    with step(f"Writing mask TIFF \u2192 {out_path}"):
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
        profile.update(count=1, dtype=rasterio.uint8, photometric="minisblack",
                       nodata=None)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask[None, ...])


def write_debug_overlay(
    image_np: np.ndarray,
    mask: np.ndarray,
    bbox_uv: tuple[int, int, int, int],
    out_path: Path,
    *,
    max_dim: int = 1600,
) -> None:
    """Save a downsampled crop around the bbox with the mask outlined for inspection."""
    with step(f"Writing debug overlay \u2192 {out_path}"):
        u0, v0, u1, v1 = bbox_uv
        # generous padding around bbox so user can see the surrounding context
        pad = max(80, (u1 - u0) // 2, (v1 - v0) // 2)
        H, W = image_np.shape[:2]
        cu0 = max(0, u0 - pad)
        cv0 = max(0, v0 - pad)
        cu1 = min(W, u1 + pad)
        cv1 = min(H, v1 + pad)
        crop = image_np[cv0:cv1, cu0:cu1].copy()
        mask_crop = mask[cv0:cv1, cu0:cu1]
        # tint masked pixels in red, then draw a green outline of the bbox
        red_overlay = crop.copy()
        red_overlay[..., 0] = np.maximum(red_overlay[..., 0], 200)
        red_overlay[..., 1] = red_overlay[..., 1] // 2
        red_overlay[..., 2] = red_overlay[..., 2] // 2
        m3 = mask_crop[..., None].astype(bool)
        crop = np.where(m3, red_overlay, crop)

        pil = Image.fromarray(crop).convert("RGB")
        # downsample if huge
        if max(pil.size) > max_dim:
            scale = max_dim / max(pil.size)
            new_size = (int(pil.size[0] * scale), int(pil.size[1] * scale))
            pil = pil.resize(new_size, Image.BILINEAR)
        draw = ImageDraw.Draw(pil)
        # draw bbox in image-local (after the crop) coords, then scaled
        if max(crop.shape[:2]) > 0:
            scale = pil.size[0] / crop.shape[1]
            local = [
                int((u0 - cu0) * scale),
                int((v0 - cv0) * scale),
                int((u1 - cu0) * scale),
                int((v1 - cv0) * scale),
            ]
            draw.rectangle(local, outline=(0, 255, 0), width=3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_path)


# ---------------------------------------------------------------------------
# Top-level entrypoint
# ---------------------------------------------------------------------------

def redact_image(
    image_path: Path,
    polygon: Polygon,
    eo_path: Path,
    las_dir: Path,
    out_path: Path,
    *,
    voxel_size_m: float = 1.0,
    buffer_m: float = 200.0,
    pixelate_factor: int = 12,
    blur_sigma: float = 8.0,
    rotation_convention: str | None = None,
) -> RedactionResult:
    """End-to-end redaction. `polygon` is in EPSG:3011 (caller does the reprojection)."""
    log.info("=" * 60)
    log.info(f"Aerial redaction: {image_path.name}")
    log.info(f"Polygon bounds (EPSG:3011): {polygon.bounds}")
    log.info("=" * 60)

    cam_kwargs = {} if rotation_convention is None else {"rotation_convention": rotation_convention}
    with step("Build camera"):
        camera = build_camera(image_path, eo_path, **cam_kwargs)

    scene = build_scene(las_dir, polygon, voxel_size_m=voxel_size_m, buffer_m=buffer_m)

    # z range from the LAS-derived mesh; the AOI's screen bbox should cover both
    # base and roof projections
    z_min = float(scene.mesh.vertices[:, 2].min())
    z_max = float(scene.mesh.vertices[:, 2].max())
    log.info(f"scene z range: [{z_min:.1f}, {z_max:.1f}] m (RH2000)")

    with step("Compute image-space bbox"):
        bbox_uv = aoi_screen_bbox(camera, polygon, z_min, z_max, pad_px=16)
        u0, v0, u1, v1 = bbox_uv
        log.info(f"  \u00b7 bbox = u[{u0}..{u1}] v[{v0}..{v1}] = {u1 - u0}\u00d7{v1 - v0} pixels")

    mask = compute_mask(camera, scene, bbox_uv)
    n_masked = int(mask.sum())
    log.info(f"mask: {n_masked:,} pixels marked for redaction")

    with step("Reading source image"):
        with rasterio.open(image_path) as src:
            arr = src.read()
        image_np = np.transpose(arr, (1, 2, 0))   # → (H, W, bands)

    redacted = apply_redaction(
        image_np, mask, bbox_uv,
        pixelate_factor=pixelate_factor,
        blur_sigma=blur_sigma,
    )

    write_geotiff(image_path, out_path, redacted)
    mask_path = out_path.with_name(out_path.stem + "_mask.tif")
    write_mask_tif(image_path, mask, mask_path)
    debug_path = out_path.with_name(out_path.stem + "_debug.png")
    write_debug_overlay(redacted, mask, bbox_uv, debug_path)

    log.info("=" * 60)
    log.info(f"Done. {out_path}")
    log.info(f"     mask:  {mask_path}")
    log.info(f"     debug: {debug_path}")
    log.info("=" * 60)

    return RedactionResult(
        out_path=out_path,
        mask_path=mask_path,
        debug_path=debug_path,
        n_pixels_masked=n_masked,
        bbox_uv=bbox_uv,
    )
