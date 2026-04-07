# oblique-redaction

Redact (pixelate + Gaussian blur) sensitive sites in oblique and nadir aerial
imagery, respecting occlusion by buildings, trees and walls.

Given a 2D site polygon (WGS84) and an aerial image with exterior orientation
(EO) data, the pipeline reconstructs the local 3D surface from airborne LiDAR,
projects rays from the camera through every relevant pixel, and masks the
pixels whose first surface hit lies inside the site footprint. The masked
pixels are then pixelated and blurred. Pixels that are physically occluded
(e.g. a building in front of the site) are correctly left untouched.

Built and verified against the Vexcel UltraCam Osprey 4.1 oblique imagery of
Stockholm/Kista (2021), flown by Terratec.

```
   ┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
   │ AOI polygon │    │ aerial TIFF │    │ Terratec EO  │    │ LiDAR LAS   │
   │  (WGS84)    │    │ (UltraCam)  │    │   text file  │    │ tiles (3011)│
   └──────┬──────┘    └──────┬──────┘    └──────┬───────┘    └──────┬──────┘
          │                  └──────┬───────────┘                   │
          │                         ▼                               │
          │                  ┌─────────────┐                        │
          │                  │   Camera    │                        │
          │                  │  (K, R, C)  │                        │
          │                  └──────┬──────┘                        │
          ▼                         │                               ▼
   ┌─────────────┐                  │                  ┌──────────────────┐
   │  Polygon    │                  │                  │  bbox + buffer   │
   │ (EPSG:3011) │                  │                  │  → tile select   │
   └──────┬──────┘                  │                  │  → stream + clip │
          │                         │                  │  → max-z voxel   │
          │                         │                  │  → 2D Delaunay   │
          ├─────────────────────────┼─────────────────►│  → tag sensitive │
          │                         │                  │  → Embree BVH    │
          │                         │                  └────────┬─────────┘
          │                         │                           ▼
          │                         │                    ┌─────────────┐
          │                         └───────────────────►│  raycast    │
          │                                              │  in screen  │
          │                                              │  bbox       │
          │                                              └──────┬──────┘
          │                                                     ▼
          │                                              ┌─────────────┐
          │                                              │  binary     │
          │                                              │  mask       │
          │                                              └──────┬──────┘
          │                                                     ▼
          │                                              ┌─────────────┐
          │                                              │ pixelate +  │
          │                                              │ blur inside │
          │                                              │ mask        │
          │                                              └──────┬──────┘
          │                                                     ▼
          │                                              ┌─────────────┐
          │                                              │ redacted    │
          │                                              │ GeoTIFF     │
          │                                              └─────────────┘
```

## Quick start

```bash
# Install (uv handles everything)
uv sync

# Verify the camera model on your data (HARD GATE — see below)
uv run python tests/test_camera_sanity.py

# Redact one image against one polygon
# (bring your own AOI polygon as a WGS84 GeoJSON; see docs/architecture.md for
#  the expected inputs)
uv run oblique-redact \
  --image   /path/to/image.tif \
  --polygon /path/to/site.geojson \
  --eo      /path/to/EO.txt \
  --las-dir /path/to/las_tiles/ \
  --out     /tmp/redacted.tif
```

Outputs (next to `--out`):

- `redacted.tif` — the redacted GeoTIFF, source profile and tags preserved
- `redacted_mask.tif` — single-band uint8 binary mask
- `redacted_debug.png` — small overlay showing the bbox and the masked region

CLI options:

| Flag | Default | Notes |
|---|---|---|
| `--image` | required | Source TIFF (UltraCam Lvl-3) |
| `--polygon` | required | Site polygon GeoJSON in WGS84 |
| `--eo` | required | Terratec EO file (`EO_total.txt` or per-camera) |
| `--las-dir` | required | Directory of LAS tiles in EPSG:3011 |
| `--out` | required | Output GeoTIFF path |
| `--voxel-size` | `1.0` | TIN xy voxel size in metres |
| `--buffer` | `200.0` | LAS clip buffer around AOI in metres |
| `--pixelate-factor` | `12` | Pixelate downsample factor inside the mask |
| `--blur-sigma` | `8.0` | Gaussian blur sigma in pixels |
| `--rotation-convention` | `xyz_intrinsic_T` | Override the camera rotation convention |

## Verification gate

Before trusting the projection on a new EO source, run

```bash
uv run python tests/test_camera_sanity.py
```

This projects ~100 building footprints from a Stockholm base map onto a
downsampled image with **all four** candidate rotation conventions and saves
side-by-side overlays in `/tmp/oblique_redaction_camera_check/`. Visual
inspection picks the convention whose outlines line up with the visible
buildings. The default in code (`xyz_intrinsic_T`) was chosen this way for
Terratec/Vexcel data — a different EO source may need a different convention.

See [docs/architecture.md](docs/architecture.md#camera-math-the-rotation-convention-is-empirical)
for the full reasoning.

## Results on the Kista example

Three sensitive sites in the same Cam6L image (10560 × 14144 px, 462 MB),
processed end-to-end on CPU:

| | Site 1 | Site 2 | Site 3 |
|---|---:|---:|---:|
| Tiles read | 4 | 2 | 2 |
| Sensitive triangles | 16,774 | 16,700 | 2,120 |
| Image-space bbox (px) | 1767×2361 | 1461×3046 | 989×2048 |
| Mask pixels | 1,351,780 | 1,224,408 | 336,541 |
| Total wall time | 6.52 s | 6.08 s | 4.57 s |

The dominant cost is the per-image scene build (LAS clip + Delaunay), not the
raycasting itself: Embree handles ~6 M rays/s on CPU and the per-AOI ray budget
is well under 5 M. See [docs/scaling-and-future.md](docs/scaling-and-future.md)
for how this scales (and doesn't) at city level.

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — pipeline walkthrough, data
  flow, key algorithms, and the camera math + rotation-convention story.
- **[docs/scaling-and-future.md](docs/scaling-and-future.md)** — what
  generalises to future imagery deliveries, what doesn't, and what to change to
  process the entire Stockholm region instead of single images.

## Layout

```
oblique-redaction/
├── pyproject.toml
├── README.md                              # this file
├── docs/
│   ├── architecture.md
│   └── scaling-and-future.md
├── src/oblique_redaction/
│   ├── __init__.py                        # re-exports
│   ├── timing.py                          # logger, step() ctx, throttled progress
│   ├── camera.py                          # EO + intrinsics → Camera (project + ray)
│   ├── scene.py                           # LAS → TIN + Embree raycaster
│   ├── redact.py                          # mask → pixelate+blur → GeoTIFF I/O
│   └── cli.py                             # `oblique-redact` entrypoint
└── tests/
    └── test_camera_sanity.py              # rotation-convention verification (HARD GATE)
```

## Dependencies

| Package | Why |
|---|---|
| `numpy`, `scipy` | Array math + 2D Delaunay |
| `pyproj` | WGS84 → EPSG:3011 reprojection |
| `shapely` | Polygon ops + vectorised `contains_xy` |
| `rasterio` | Source TIFF read, GeoTIFF write |
| `tifffile` | Parse the UltraCam `ImageDescription` tag |
| `pillow` | Pixelate + Gaussian blur composite |
| `laspy[lazrs]` | Streamed LAS/LAZ reading |
| `trimesh` | Mesh container + ray API |
| `embreex` | Intel Embree CPU raytracing wheel for trimesh |
| `pyogrio` | GeoPackage reading (used by the camera-sanity test only) |

All managed via `uv`. No GDAL CLI / no system PDAL required.

## Status and limits

This is **v1**. It works correctly on Vexcel UltraCam Osprey Cam6L oblique
imagery from Stockholm/Kista 2021, processed with Terratec TerraPos EO. It's
deliberately not over-engineered for cases not yet observed:

- **Single image, single polygon** per CLI invocation. Functions are written so
  batching wraps trivially, but the CLI doesn't loop yet.
- **No lens distortion correction.** UltraCam is a metric camera and the Lvl-3
  product is corrected upstream. Other cameras will likely need a Brown-Conrady
  distortion model added.
- **Rotation convention is empirical** for the Terratec data. New surveyors /
  software → re-run the verification gate before trusting the output.
- **2.5D height-field mesh, no overhangs.** Central Stockholm rarely has them;
  bridges, balconies, and rooftop cantilevers are the failure cases.
- **Buffer-based LAS clip, no line-of-sight cone.** Calibrated for ~1850 m AGL
  cameras; lower-altitude flights may need a cone clip — see
  [docs/scaling-and-future.md](docs/scaling-and-future.md#low-altitude-cameras).
- **Per-query LAS read.** Reads tiles on every invocation. Fine for dozens of
  AOIs, breaks down at city scale — see
  [docs/scaling-and-future.md](docs/scaling-and-future.md#what-changes-at-city-scale)
  for the COPC + caching plan.
