# oblique-redaction

Blur out sensitive sites in aerial photos, correctly, even when buildings or
trees stand in front of them.

You give it the outline of a place that should be hidden (drawn on a map) and an
aerial photo of the area. It works out exactly which pixels in the photo show
that place and blurs only those, leaving everything else untouched. If a building
or tree is *between* the camera and the site, those pixels are left alone: they
aren't part of the site, so they shouldn't be blurred.

It does this by rebuilding the 3D shape of the ground and buildings from laser
scan (LiDAR) data, then tracing, for each pixel, what the camera was actually
looking at.

## Installing

You need [`uv`](https://docs.astral.sh/uv/) (a Python tool). Once it's installed,
from inside this folder run:

```bash
uv sync
```

That downloads everything the tool needs. No other setup is required.

## Using it

Blur one site in one photo:

```bash
uv run oblique-redact \
  --image   /path/to/photo.tif \
  --polygon /path/to/site.geojson \
  --eo      /path/to/EO.txt \
  --las-dir /path/to/laser_scan_tiles/ \
  --out     /tmp/blurred.tif
```

You can also point `--image` at a *folder* of photos and `--polygon` at a file
holding *several* sites. Every photo is processed in turn, and every site that
appears in a photo is blurred. In that case `--out` should be a folder; each
result is saved as `<name>_redacted.tif`.

```bash
uv run oblique-redact \
  --image   /path/to/photos_folder/ \
  --polygon /path/to/sites.geojson \
  --eo      /path/to/EO.txt \
  --las-dir /path/to/laser_scan_tiles/ \
  --out     /path/to/output_folder/
```

### What you provide

| Flag | What it is |
|---|---|
| `--image` | The aerial photo (a TIFF), or a folder of them |
| `--polygon` | The site outline(s), as a GeoJSON map file in standard lat/long (WGS84) |
| `--eo` | The camera orientation text file that came with the imagery |
| `--las-dir` | The folder of LiDAR laser-scan tiles for the area |
| `--out` | Where to save the result: a file (single photo) or a folder (many photos) |

### What you get

Next to your `--out` path:

- `blurred.tif`: the photo with the site(s) blurred, otherwise identical to the original
- `blurred_debug.png`: a small preview image highlighting what was blurred, for a quick visual check

### Optional settings

These have sensible defaults; most people never need to change them.

| Flag | Default | What it controls |
|---|---|---|
| `--pixelate-factor` | `12` | How coarse the blur is (higher is blockier) |
| `--blur-sigma` | `8.0` | How soft the blur edges are |
| `--voxel-size` | `1.0` | Detail of the 3D surface, in metres |
| `--buffer` | `200.0` | How far around the site to look for things that might block the view, in metres |
| `--rotation-convention` | `xyz_intrinsic_T` | Advanced: how camera angles are interpreted (see below) |

## Before trusting a new data source

The tool was calibrated against one specific imagery source. If you bring imagery
from a **different supplier or camera**, run the check first:

```bash
uv run python tests/test_camera_sanity.py
```

This draws known building outlines onto a sample photo and saves preview images
to a temporary folder. Look at them: if the outlines line up with the buildings,
the camera setup is correct and you can trust the results. If they don't, the
imagery uses a different angle convention and needs adjusting before use. See
[docs/architecture.md](docs/architecture.md#4-camera-math-the-rotation-convention-is-empirical)
for the details.

## What it handles, and what it doesn't

Works well on:

- High-altitude aerial imagery (the kind flown for regional mapping surveys)
- Oblique or nadir aerial photos that come with a matching camera orientation file
- Flat-roofed and sloped buildings, trees, walls, and terrain

Not handled yet (would need changes):

- **Overhanging structures**: bridges, balconies, rooftop cantilevers. The 3D
  surface is a "height map", so it can't represent something sticking out over
  empty space beneath it. Rare in typical urban scenes.
- **Other camera types** that aren't pre-corrected for lens distortion (most
  drone and consumer cameras). The supported imagery is already corrected.
- **Very low-altitude flights** (low drone, near-horizontal). The view-blocking
  search is tuned for high-altitude imagery.

## How it works

See **[docs/architecture.md](docs/architecture.md)** for the full technical
walkthrough: the camera maths, the 3D surface reconstruction, and the
pixel-tracing that decides what to blur.
