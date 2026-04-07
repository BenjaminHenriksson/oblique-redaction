# Scaling and future-proofing

Two related questions about taking this from "works on the example" to "works
in production":

1. **What happens when a future imagery delivery uses a different camera, EO
   format, or processing chain?** ([§ Future-proofing across deliveries](#future-proofing-across-deliveries))
2. **What changes when we redact every site across all of Stockholm instead of
   one image at a time?** ([§ What changes at city scale](#what-changes-at-city-scale))

The two are independent but reinforce each other: the things that hold (the
geometric core) are the same; the things that change (data parsing, data
layout) are the same. Engineering effort goes into the **edges of the system**,
not the core.

## Future-proofing across deliveries

Honest assessment, by category. The core idea — *raycast against the measured
surface; the first hit's xy decides the redaction* — is camera-agnostic,
vendor-agnostic, scale-agnostic. The plumbing around it is what needs care.

### Will keep working without changes

- **Different LAS / LAZ tile sets.** Naming, count, sizes, point density are
  all derived from the headers at runtime. Drop new tiles in a directory,
  point `--las-dir` at it, done.
- **Different AOI polygons / different image stems.** All driven by CLI args.
- **Different image dimensions.** Read from the TIFF header.
- **Different cameras *of the same family*** (any UltraCam Lvl-3 — Eagle,
  Falcon, Osprey variants). They all carry the `PRINCIPAL_DISTANCE` /
  `PRINCIPAL_POINT_X/Y` / `PIXEL_SIZE_*` keys in `ImageDescription`.
- **Cam0N (nadir) and Cam4B/5R/6L/7F (other obliques) of the same UltraCam
  Osprey.** Each has its own EO file with its own per-camera platform
  mount/boresight pre-applied to the row. Same code path.

### Will probably need a small adjustment

- **Different EO file format.** `parse_eo_row` is hardcoded to the Terratec
  column layout (event ID at 0; E/N/h at 1/2/3; ω/φ/κ at 5/6/7). New surveyor
  → new parser, but it's ~20 lines.
- **Different camera intrinsics format.** `parse_intrinsics_from_tiff`
  regex-greps UltraCam-specific keys. A Phase One iXM, Leica RCD30, or Vexcel
  UltraMap-via-different-product might use different key names. Same shape,
  different keys → new ~20-line parser per format.

### Will need real engineering

- **Different rotation convention.** The empirical `xyz_intrinsic_T` was chosen
  by visual fit on this specific Cam6L image. If a future delivery uses a
  different processing chain (Inpho, LMP, in-house Vexcel UltraMap, Pix4D,
  custom INS), the convention may differ. **The verification step
  (`tests/test_camera_sanity.py`) is the safety net** — it tries 4 conventions
  and you visually pick. Run it for any new EO source before trusting the
  output.

  A small upgrade: have the verification script auto-pick by computing
  residuals against Byggnad outlines instead of asking a human. Worth doing
  before scaling.

- **Lens distortion.** Ignored because UltraCam is metric and Lvl-3 is
  corrected. Drone cameras (even mid-format ones like the Phase One iXM-RS),
  most consumer cameras, and any *uncorrected* Lvl-1/Lvl-2 product will have
  non-trivial radial distortion. The `Camera` class would need to grow a
  distortion model (Brown-Conrady is the standard) and apply it in both
  `world_to_image` and `image_to_ray`. ~30 lines.

- **Lower-altitude cameras.** See [§ Low-altitude cameras](#low-altitude-cameras)
  below.

- **Multi-image batching.** The current CLI is single-image. A loop wrapping
  `redact_image` is trivial, but to filter "which images see this AOI?" you
  need a *correct* image footprints file. That regeneration is its own task —
  the projection code we have can produce one as a by-product.

- **Genuine 3D scenes (overhangs, bridges, multi-storey atria).** The 2.5 D
  Delaunay can't represent these. Central Stockholm doesn't have many — but
  historic European cores, harbour cranes, some industrial sites do. The fix
  is either Roofer-style explicit building volumes for those structures
  (merged with the LiDAR TIN for everything else) or a true 3D triangulation
  (alpha-shapes / TetGen). Adds significant complexity. Defer until a real
  failure case appears.

- **Higher-density point clouds** (mobile mapping at 1000 pts/m², or dense
  matching from the imagery itself). Voxel downsample handles the density,
  but you might want a smaller voxel (0.25 m → ~16× more triangles, much
  slower Delaunay). At some point you'd switch to a raster DSM rendered as a
  mesh, or to BVH-friendly chunked meshes.

- **Different CRS.** Hardcoded "polygon WGS84 → EPSG:3011" in `cli.py`.
  Externalising this is a 5-line generalisation: take an `--out-crs` argument,
  default to whatever the LAS tiles say (LAS headers carry CRS metadata most
  of the time).

### The thing that holds

Whatever you swap (camera model, EO format, point cloud source, AOI shape),
the inner loop stays:

```python
hits = intersector.intersects_first(origins, directions)
mask = sensitive_face_mask[hits.clip(min=0)] & (hits >= 0)
```

Everything else is plumbing around this. So the architecture is durable; the
parsers are the part that will need maintenance.

## Low-altitude cameras

The current 200 m AOI buffer is calibrated for **high-altitude aerial** —
specifically the ~1850 m above-ground-level Vexcel Osprey flight. At this
altitude, the camera→AOI ray descends through building heights only in the
last ~4% of its length:

| t along camera→AOI ray | z (m) | xy distance from AOI (m) |
|---:|---:|---:|
| 0.50 | 955 | 1287 |
| 0.90 | 215 | 257 |
| 0.95 | 122 | 129 |
| 0.96 | 104 | 103 |
| 0.97 | 85 | 77 |
| 1.00 | 30 | 0 |

Stockholm has nothing taller than ~150 m, so any potential occluder is within
~120 m of the AOI in xy. The 200 m buffer comfortably covers this.

**This is a calibrated default, not a robust principle.** For a 300 m AGL drone
flight, or a near-horizontal helicopter pass, the relevant occluder cone
extends much further from the AOI. The robust fix is a line-of-sight cone
clip:

```python
from shapely.geometry import MultiPoint
cone = MultiPoint([camera_xy, *aoi_xy]).convex_hull.buffer(50)
bbox = cone.bounds                              # for tile selection
mask = shapely.contains_xy(cone, x, y)          # for per-point filter inside chunks
```

That'd be ~10 lines added to `scene.py` and would handle arbitrary camera
altitudes correctly. Deferred because the current default works for the data
we have.

## What changes at city scale

### Why the current approach breaks down

Today, each `redact_image` call:

1. opens every `.las` in `--las-dir` just to read the header → **O(tiles_total)** per query
2. streams the *content* of every tile that intersects the AOI bbox → **O(local_tiles × tile_size)** per query

For Kista: 11 tiles in the directory, 4 read for the AOI, ~1.0 s of LAS work,
~7 M points kept. The header pass is microseconds — that's fine. The streaming
read of 4 tiles dominates.

Scale that up:

| Scenario | Header passes | Tiles streamed | LAS bytes read |
|---|---:|---:|---:|
| Kista, 1 AOI, 1 image (current) | 11 | 4 | ~1.5 GB |
| Kista, 1 AOI, 50 images of it | 11 × 50 = 550 | 4 × 50 = 200 | ~75 GB |
| Stockholm: 50 sites × 50 images each, naive | ~250 K | ~10 K | ~3.5 TB |
| Stockholm: 50 sites × 50 images each, with reuse | ~ once | ~few hundred | ~50 GB |

The naive cost grows with `(sites × images)` because nothing is cached. There
are two completely separate inefficiencies:

1. **LAS data is read repeatedly.** The same tile is opened, decoded, and
   filtered for every query that touches it. LAS is *not* a random-access
   format — even though `laspy` chunks the read, every chunk has to be
   decoded, and the spatial filter is applied client-side after decoding.
2. **The TIN is rebuilt repeatedly.** Two images that look at the same site
   rebuild the same Delaunay mesh from scratch.

Both compound. So before scaling code, the **data layout** needs to change.

### What to change, in order of return-on-effort

#### Tier 1 — cheap code changes that work with the existing data

These are ~50 lines of code each, no data conversion required.

**a. Cache built scenes.** `SceneMesh` is an in-memory object keyed on
`(las_dir, bbox, voxel_size)`. Two queries hitting the same bbox already build
identical meshes. A `functools.lru_cache` on `build_scene` (after rounding the
bbox to grid coordinates) gives free reuse for repeated runs. Doesn't help
across processes, but a single batch script processing many
images-of-the-same-area benefits immediately.

**b. Group by image, not by AOI.** Right now the natural batch is "N AOIs in
1 image processed serially → N scene builds". Instead, build the scene
**once** per image around the union bbox of all AOIs in that image; raycast
once per AOI. The expensive parts (LAS read, voxel downsample, Delaunay, BVH
build) all amortise. The change to `redact.py` is replacing `polygon` with
`polygons: list[Polygon]`, taking the union for the scene, and looping the
sensitive-face tagging + mask compute per polygon. The mesh is shared.

**c. Group by region, not by image.** For images that overlap the same area,
the *same* TIN is valid. Build it once per ~1 km × 1 km tile, query it for
many image+AOI combinations. This changes the abstraction: the scene is now
keyed on a *region*, not on a query.

Combine (b) and (c) and the Stockholm batch goes from "rebuild scene per
(site, image)" to "rebuild scene per region tile, query it for every (site,
image) in the tile". Order-of-magnitude improvement.

#### Tier 2 — convert the data once, change ~100 lines of `scene.py`

The fundamental issue is that `.las` doesn't support spatial indexing, so all
queries pay full decode cost. **Convert the LAS tree to COPC once**, use it
forever:

**COPC (Cloud Optimized Point Cloud)** is LAZ with an octree spatial index
baked into the file. A query for a bbox reads only the octree nodes that
intersect the bbox. PDAL can convert the entire `punktmoln/` tree in one
pipeline:

```bash
pdal pipeline copc-build.json
# inside: { "type": "writers.copc", "filename": "stockholm.copc.laz" }
```

Then in `scene.py`:

```python
import pdal
pipeline = pdal.Reader.copc(filename="stockholm.copc.laz", bounds=str(bbox)).pipeline()
pipeline.execute()
points = pipeline.arrays[0]   # only the points inside bbox
```

PDAL pushes the spatial filter into the COPC reader, so you read **only the
bytes you need**. For our 4-tile, 7 M-point query, this would be ~50 ms
instead of ~1 s. For city-scale queries it's the difference between hours and
minutes.

The trade-off: one-time conversion (a few hours of PDAL run-time for the whole
city, ~20% storage growth from the index) and adding `pdal` as a dep. PDAL is
~150 MB installed.

A lighter alternative: **per-tile spatial caches** — build a small `.npy` or
HDF5 file per LAS tile containing the already-downsampled max-z grid points,
indexed by xy bucket. Reading these caches is ~10× faster than LAS chunked
decode and they're 90% smaller. Less generic than COPC but no PDAL dep.

#### Tier 3 — pre-compute the surface, not just the points

If you're going to redact thousands of images repeatedly (e.g. each new survey
delivery, each policy change to which sites count), the *biggest* win is to
never rebuild the TIN per query at all. Two options:

**a. Pre-built tiled mesh.** Run the voxel-downsample + Delaunay + BVH build
over the whole city *once*, save the output as small spatially indexed mesh
tiles. Each query loads the relevant mesh tile(s) and runs the raycaster
directly. This is exactly what `kista_output.obj` already represents (the
3dfier output we explicitly rejected for v1 because LoD1 flat roofs are too
coarse for oblique imagery). For Stockholm-scale, the trade-off shifts —
accepting slightly coarser building tops in exchange for *zero* per-query
mesh-build time is the right call.

A middle ground: pre-build a **TIN from the LAS via the same code we use
today**, save as `.ply` or quantized mesh tiles per ~250 m grid cell, and
query those at runtime. Best of both worlds — the tight LiDAR-derived surface
plus the speed of pre-computed data.

**b. Pre-computed DSM raster.** A simple GeoTIFF DSM at 1 m resolution is
~36 KB/km² (uint16 metres × 1024² values × 2 bytes / scaled). For all of
Stockholm (~190 km²) that's ~7 GB. Read the relevant window via rasterio,
mesh the heightfield (2 triangles per cell), raycast. **The downside is the
wall-discontinuity problem** — DSM rasters can leak rays through walls on
oblique views. There's a fix (interpret the DSM as a heightfield mesh, not a
raster look-up) but at that point you've reinvented the TIN.

For our use case, **pre-built TIN tiles** is the right answer at city scale.
~10–20 MB per km² of compressed mesh tiles, instant load, no LAS decoding in
the hot path.

#### Tier 4 — bigger architectural changes

These only make sense at very large scale.

**a. Persistent scene server.** If you have a long-running batch of
redactions, run the TIN-build code as a server process that holds Stockholm's
mesh in memory. Queries are RPC calls. The `intersector` is built once and
reused indefinitely. This eliminates per-query setup completely, at the cost
of needing ~5–20 GB resident memory.

**b. GPU rasterisation instead of CPU raytracing.** For very large image
counts (>10 000), porting the raycaster to OpenGL/Vulkan rasterisation
(render the mesh through the camera with a face-id buffer, look up sensitive
faces in the buffer) gives 10–100× throughput. We'd lose the CPU-only
deployment story.

**c. Per-image pre-work.** For each delivered image, store its computed
footprint, the relevant scene tile IDs, and the bbox into the scene mesh.
Then redaction queries become a simple lookup. Effectively a build system for
image + scene assets.

### What I'd actually do for Stockholm-wide deployment

If I had to plan a real production rollout today, in priority order:

1. **Convert the LAS tree to COPC once** (~half a day of PDAL work, ~hours of
   convert time). Single biggest unlock — every future query becomes ~20×
   faster on the LAS read.
2. **Refactor `redact_image` to take many polygons, build the scene once.**
   ~30 lines. Saves the rebuild cost when one image has multiple AOIs.
3. **Add a process-level scene cache keyed on (region tile, voxel size).**
   ~20 lines. Saves the rebuild cost when many images cover the same area.
4. **Group the city into ~250 m × 250 m work units, process all (image, AOI)
   pairs per unit together.** Glue script. Now LAS data is read at most twice
   per region (once for the COPC bbox query, plus the TIN we cache).
5. **Only after all of the above** (and only if performance is still a
   problem) pre-compute the TIN tiles. This is the biggest engineering
   investment but also the biggest payoff if you redact across many image
   deliveries.

The current `scene.py` is about 230 lines. Steps 1–4 add maybe 150 lines and
a one-time data conversion. Throughput goes from "minutes per AOI" to
"seconds per AOI batch", which is the right operating point for city-scale
work.

### What does **not** need to change

The camera math, the projection, the raycast → mask logic, the pixelate/blur
compositor, and the GeoTIFF I/O are all per-query and already fast (the
raycast is sub-second). They scale linearly with the number of images, which
is unavoidable.

**All the scaling work is in how the scene mesh is built and reused.** That's
the leverage point.

## Summary

| Scaling axis | Today | Tier 1 (~50 LOC) | Tier 2 (COPC) | Tier 3 (pre-tiled TIN) |
|---|---|---|---|---|
| 1 image, 1 AOI | 6 s ✓ | 6 s | 5 s | 4 s |
| 1 image, 5 AOIs | 30 s | 8 s | 6 s | 5 s |
| 50 images, same area | 5 min | 30 s | 15 s | 5 s |
| Stockholm (50 × 50) | days | hours | minutes | minutes |

The architecture survives at city scale; the data layout is what needs to
evolve. None of this touches the projection / raycast / compositor code that
took the most care to get right.
