"""CLI entrypoint:

    oblique-redact \
        --image  <path/to/image.tif> \
        --polygon <path/to/aoi.geojson> \
        --eo     <path/to/EO_total.txt> \
        --las-dir <path/to/punktmoln/> \
        --out    <path/to/redacted.tif>

The polygon GeoJSON is read in WGS84 (EPSG:4326) and reprojected to EPSG:3011
(SWEREF99 18 00) before being passed to the rest of the pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import Polygon, shape

from .redact import redact_image
from .timing import init_logger, log, step


def _load_polygon_3011(geojson_path: Path) -> Polygon:
    """Read the first polygon feature from a GeoJSON in WGS84, reproject to EPSG:3011."""
    with step(f"Loading polygon {geojson_path}"):
        data = json.loads(geojson_path.read_text())
        if data.get("type") == "FeatureCollection":
            geom = data["features"][0]["geometry"]
        elif data.get("type") == "Feature":
            geom = data["geometry"]
        else:
            geom = data
        poly_wgs84 = shape(geom)
        if poly_wgs84.geom_type != "Polygon":
            raise ValueError(
                f"expected Polygon, got {poly_wgs84.geom_type} in {geojson_path}"
            )
        to_3011 = Transformer.from_crs("EPSG:4326", "EPSG:3011", always_xy=True)
        xs, ys = to_3011.transform(*poly_wgs84.exterior.xy)
        poly_3011 = Polygon(list(zip(xs, ys)))
        log.info(f"  \u00b7 reprojected to EPSG:3011, bounds {poly_3011.bounds}")
    return poly_3011


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Redact a sensitive site in an aerial image.")
    ap.add_argument("--image", type=Path, required=True, help="Source TIFF (UltraCam Lvl-3).")
    ap.add_argument("--polygon", type=Path, required=True, help="Site polygon GeoJSON in WGS84.")
    ap.add_argument(
        "--eo",
        type=Path,
        required=True,
        help="Terratec EO file (EO_total.txt or per-camera EO_*.txt).",
    )
    ap.add_argument(
        "--las-dir",
        type=Path,
        required=True,
        help="Directory of LAS tiles in EPSG:3011.",
    )
    ap.add_argument("--out", type=Path, required=True, help="Output GeoTIFF path.")
    ap.add_argument("--voxel-size", type=float, default=1.0, help="TIN voxel size [m].")
    ap.add_argument("--buffer", type=float, default=200.0, help="LAS clip buffer around AOI [m].")
    ap.add_argument(
        "--pixelate-factor",
        type=int,
        default=12,
        help="Pixelate downsample factor inside the masked region.",
    )
    ap.add_argument("--blur-sigma", type=float, default=8.0, help="Gaussian blur sigma [px].")
    ap.add_argument(
        "--rotation-convention",
        default=None,
        help="Override the camera rotation convention (default: xyz_intrinsic_T).",
    )
    args = ap.parse_args(argv)

    init_logger()
    polygon = _load_polygon_3011(args.polygon)
    redact_image(
        image_path=args.image,
        polygon=polygon,
        eo_path=args.eo,
        las_dir=args.las_dir,
        out_path=args.out,
        voxel_size_m=args.voxel_size,
        buffer_m=args.buffer,
        pixelate_factor=args.pixelate_factor,
        blur_sigma=args.blur_sigma,
        rotation_convention=args.rotation_convention,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
