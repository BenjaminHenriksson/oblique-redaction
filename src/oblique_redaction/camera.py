"""Camera model: EO row + TIFF intrinsics → projection / inverse projection.

Coordinate conventions
----------------------
- World frame: SWEREF99 18 00 (EPSG:3011), Z = orthometric height (RH2000).
- Photo frame at (ω, φ, κ) = (0, 0, 0): +x = world East, +y = world North, +z = world Up.
  This matches the standard photogrammetric convention where a nadir camera at zero
  rotation has its optical axis pointing -Z (down).
- Default rotation order: `xyz_intrinsic_T` — the TRANSPOSE of R_x(ω)·R_y(φ)·R_z(κ).
  Terratec labels its rotation "XYZ_R, map-frame to object frame", which by literal
  reading would build R_x·R_y·R_z (intrinsic XYZ) as the world-to-camera matrix.
  Empirically (see `tests/test_camera_sanity.py`), the matching convention is the
  TRANSPOSE — meaning Terratec's "map-frame to object frame" describes the rotation
  of the FRAMES (object basis vectors as columns in the map frame), not of vectors.
  Three other conventions remain exposed in case future EO sources differ.
- For pixel projection we convert photo frame → OpenCV-style camera frame
  (+x right, +y down, +z forward into scene) by `R_photo_to_cv = diag(1, -1, -1)`.

EO output already refers to camera C116 — lever arm + boresight + platform mount are
baked in by Terratec. We do NOT apply them here.

Principal point in TIFF
-----------------------
The TIFF gives PRINCIPAL_POINT_X / PRINCIPAL_POINT_Y in mm relative to the sensor center,
math y up. Pixel principal point:
    u_pp = W/2 + PP_X / pixel_size
    v_pp = H/2 - PP_Y / pixel_size
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .timing import log


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Intrinsics:
    f_px: float        # focal length in pixels
    u_pp: float        # principal point x in pixels
    v_pp: float        # principal point y in pixels
    width: int
    height: int


@dataclass
class EORow:
    image_stem: str
    E: float
    N: float
    h: float           # orthometric height (RH2000), m
    omega: float       # R1, deg
    phi: float         # R2, deg
    kappa: float       # R3, deg


@dataclass
class Camera:
    intrinsics: Intrinsics
    R_world_to_cam: np.ndarray   # (3, 3), OpenCV-style camera frame
    C_world: np.ndarray          # (3,), camera origin in EPSG:3011 / RH2000

    def world_to_image(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to pixel coords.

        Returns
        -------
        uv : (N, 2) float64 — (u, v) pixel coordinates (may be NaN for z<=0)
        depth : (N,) float64 — depth in front of camera; >0 means visible
        """
        xyz = np.atleast_2d(xyz).astype(np.float64)
        rel = xyz - self.C_world
        cam = rel @ self.R_world_to_cam.T   # row-vector form of R @ v_col
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_z = np.where(z > 0, 1.0 / z, np.nan)
            u = self.intrinsics.u_pp + self.intrinsics.f_px * cam[:, 0] * inv_z
            v = self.intrinsics.v_pp + self.intrinsics.f_px * cam[:, 1] * inv_z
        return np.stack([u, v], axis=1), z

    def image_to_ray(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each pixel uv (N, 2), return (origins, directions) in world frame.

        Directions are unit-length. Origins are all the camera position (broadcast).
        """
        uv = np.atleast_2d(uv).astype(np.float64)
        f = self.intrinsics.f_px
        x = (uv[:, 0] - self.intrinsics.u_pp) / f
        y = (uv[:, 1] - self.intrinsics.v_pp) / f
        z = np.ones_like(x)
        cam_dirs = np.stack([x, y, z], axis=1)
        # rotate camera-frame dirs to world frame: v_world_col = R^T @ v_cam_col
        # in row form: cam_dirs @ R
        world_dirs = cam_dirs @ self.R_world_to_cam
        world_dirs /= np.linalg.norm(world_dirs, axis=1, keepdims=True)
        origins = np.broadcast_to(self.C_world, world_dirs.shape).copy()
        return origins, world_dirs


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_intrinsics_from_tiff(tiff_path: Path) -> Intrinsics:
    """Parse the UltraCam ImageDescription tag and return pixel-space intrinsics."""
    with tifffile.TiffFile(tiff_path) as tif:
        page = tif.pages[0]
        H, W = page.shape[:2]
        desc = page.tags["ImageDescription"].value

    def grab(key: str) -> float:
        m = re.search(rf"{re.escape(key)}:\s*([-\d.]+)", desc)
        if not m:
            raise ValueError(f"intrinsic '{key}' not found in TIFF ImageDescription")
        return float(m.group(1))

    f_mm = grab("PRINCIPAL_DISTANCE")
    pp_x_mm = grab("PRINCIPAL_POINT_X")
    pp_y_mm = grab("PRINCIPAL_POINT_Y")
    px_w_mm = grab("PIXEL_SIZE_WIDTH") * 1e-3   # micron → mm
    px_h_mm = grab("PIXEL_SIZE_HEIGHT") * 1e-3
    if abs(px_w_mm - px_h_mm) > 1e-6:
        log.warning(f"non-square pixels: {px_w_mm} × {px_h_mm} mm")
    px_size = (px_w_mm + px_h_mm) / 2.0

    f_px = f_mm / px_size
    u_pp = W / 2.0 + pp_x_mm / px_size
    v_pp = H / 2.0 - pp_y_mm / px_size   # math y up → pixel v down
    return Intrinsics(f_px=f_px, u_pp=u_pp, v_pp=v_pp, width=W, height=H)


def parse_eo_row(eo_path: Path, image_stem: str) -> EORow:
    """Find a row in a Terratec EO file matching the given image stem.

    Columns (per the file header):
        0  Event ID (image stem)
        1  Easting (m)
        2  Northing (m)
        3  Orthometric height, RH2000 (m)
        4  Orthometric height scaled (m)            ← unused
        5  R1 = omega (deg)
        6  R2 = phi (deg)
        7  R3 = kappa (deg)
        ...
    """
    with open(eo_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if not parts or parts[0] != image_stem:
                continue
            return EORow(
                image_stem=parts[0],
                E=float(parts[1]),
                N=float(parts[2]),
                h=float(parts[3]),
                omega=float(parts[5]),
                phi=float(parts[6]),
                kappa=float(parts[7]),
            )
    raise ValueError(f"image stem {image_stem!r} not found in {eo_path}")


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _Rx(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _Ry(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _Rz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# Conventions exposed to the sanity-check loop. Each builds the rotation R such that
# v_photo = R @ v_world (i.e. world-to-photo). See module docstring for the empirical
# choice of default.
ROTATION_CONVENTIONS = (
    "xyz_intrinsic_T",    # ← DEFAULT, transpose of intrinsic XYZ; verified on Cam6L
    "xyz_intrinsic",      # R_x(ω) R_y(φ) R_z(κ)
    "xyz_extrinsic",      # R_z(κ) R_y(φ) R_x(ω)
    "xyz_extrinsic_T",    # transpose of xyz_extrinsic
)

DEFAULT_ROTATION_CONVENTION = "xyz_intrinsic_T"


def world_to_photo_rotation(omega: float, phi: float, kappa: float,
                            convention: str = DEFAULT_ROTATION_CONVENTION) -> np.ndarray:
    Rx, Ry, Rz = _Rx(omega), _Ry(phi), _Rz(kappa)
    if convention == "xyz_intrinsic":
        return Rx @ Ry @ Rz
    if convention == "xyz_extrinsic":
        return Rz @ Ry @ Rx
    if convention == "xyz_intrinsic_T":
        return (Rx @ Ry @ Rz).T
    if convention == "xyz_extrinsic_T":
        return (Rz @ Ry @ Rx).T
    raise ValueError(f"unknown rotation convention {convention!r}")


# Photo (+x E, +y N, +z U) → OpenCV camera (+x right, +y down, +z forward).
# At zero rotation, a downward-looking nadir camera has its optical axis along world -Z;
# in OpenCV camera frame the optical axis is +Z. To get from photo to OpenCV we negate
# both y (up → down) and z (up → -forward), keeping x.
PHOTO_TO_CV = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_camera(tiff_path: Path, eo_path: Path,
                 rotation_convention: str = DEFAULT_ROTATION_CONVENTION) -> Camera:
    intr = parse_intrinsics_from_tiff(tiff_path)
    eo = parse_eo_row(eo_path, tiff_path.stem)

    R_w_to_photo = world_to_photo_rotation(eo.omega, eo.phi, eo.kappa, rotation_convention)
    R_w_to_cv = PHOTO_TO_CV @ R_w_to_photo
    C = np.array([eo.E, eo.N, eo.h], dtype=np.float64)

    log.info(
        f"camera: stem={eo.image_stem} pos=({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f}) "
        f"ω={eo.omega:.4f}° φ={eo.phi:.4f}° κ={eo.kappa:.4f}° "
        f"conv={rotation_convention}"
    )
    log.info(
        f"intrinsics: f={intr.f_px:.1f}px pp=({intr.u_pp:.1f}, {intr.v_pp:.1f}) "
        f"image={intr.width}×{intr.height}"
    )
    return Camera(intrinsics=intr, R_world_to_cam=R_w_to_cv, C_world=C)
