"""CLI entrypoint:

    # single image
    oblique-redact \
        --image  <path/to/image.tif> \
        --polygon <path/to/aoi.geojson> \
        --eo     <path/to/EO_total.txt> \
        --las-dir <path/to/punktmoln/> \
        --out    <path/to/redacted.tif>

    # whole directory of images (same EO file + LAS dir), many AOIs
    oblique-redact \
        --image  <path/to/images_dir/> \
        --polygon <path/to/aois.geojson> \
        --eo     <path/to/EO_total.txt> \
        --las-dir <path/to/punktmoln/> \
        --out    <path/to/out_dir/>

`--image` may be a single TIFF or a directory of TIFFs (processed in turn).
`--polygon` may hold one feature or many (FeatureCollection / MultiPolygon); each
becomes an independent AOI — they are NOT merged. `--out` is an output file for a
single image, or an output directory when `--image` is a directory.

The polygon GeoJSON is read in WGS84 (EPSG:4326) and reprojected to EPSG:3011
(SWEREF99 18 00) before being passed to the rest of the pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import Polygon, shape

from .redact import redact_images
from .timing import init_logger, log, step


def _load_polygons_3011(geojson_path: Path) -> list[Polygon]:
    """Read every polygon feature from a GeoJSON in WGS84, reproject each to EPSG:3011.

    Handles a FeatureCollection, a single Feature, or a bare geometry, and splits any
    MultiPolygon into its parts. Each polygon is returned as a separate AOI.
    """
    with step(f"Loading polygons {geojson_path}"):
        data = json.loads(geojson_path.read_text())
        if data.get("type") == "FeatureCollection":
            geoms = [f["geometry"] for f in data["features"]]
        elif data.get("type") == "Feature":
            geoms = [data["geometry"]]
        else:
            geoms = [data]

        to_3011 = Transformer.from_crs("EPSG:4326", "EPSG:3011", always_xy=True)
        polygons: list[Polygon] = []
        for geom in geoms:
            g = shape(geom)
            parts = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
            for poly_wgs84 in parts:
                if poly_wgs84.geom_type != "Polygon":
                    raise ValueError(
                        f"expected Polygon features, got {poly_wgs84.geom_type} in {geojson_path}"
                    )
                xs, ys = to_3011.transform(*poly_wgs84.exterior.xy)
                polygons.append(Polygon(list(zip(xs, ys, strict=True))))
        if not polygons:
            raise ValueError(f"no polygon features found in {geojson_path}")
        log.info(f"  - {len(polygons)} AOI(s) reprojected to EPSG:3011")
    return polygons


def _resolve_images(image_arg: Path, out_arg: Path) -> tuple[list[Path], list[Path]]:
    """Return parallel (image_paths, out_paths) lists for a file or a directory input."""
    if image_arg.is_dir():
        images = sorted(set(image_arg.glob("*.tif")) | set(image_arg.glob("*.tiff")))
        if not images:
            raise SystemExit(f"no .tif/.tiff images found in {image_arg}")
        out_arg.mkdir(parents=True, exist_ok=True)
        out_paths = [out_arg / f"{p.stem}_redacted.tif" for p in images]
        return images, out_paths
    return [image_arg], [out_arg]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Redact sensitive sites in aerial imagery.")
    ap.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Source TIFF, or a directory of TIFFs to process in turn.",
    )
    ap.add_argument(
        "--polygon",
        type=Path,
        required=True,
        help="AOI polygon GeoJSON in WGS84 (one or many features; not merged).",
    )
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
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output GeoTIFF (single image) or output directory (image directory).",
    )
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
    polygons = _load_polygons_3011(args.polygon)
    image_paths, out_paths = _resolve_images(args.image, args.out)
    redact_images(
        image_paths,
        out_paths,
        polygons,
        eo_path=args.eo,
        las_dir=args.las_dir,
        voxel_size_m=args.voxel_size,
        buffer_m=args.buffer,
        pixelate_factor=args.pixelate_factor,
        blur_sigma=args.blur_sigma,
        rotation_convention=args.rotation_convention,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
