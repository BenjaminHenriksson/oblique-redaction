# Architecture

How the pipeline works, end-to-end: the data flow first, then the key algorithms
and the reasoning behind the design choices.

## The problem

Given a 2D polygon (WGS84) describing a sensitive site and an oblique aerial
image of that area, find the *image pixels* that show the site, and **not** the
pixels showing other things in front of it, then obscure them.

The hardest sub-problem is occlusion. From a steeply oblique camera, a tall
foreground building can hide part of the site, and that building's pixels must
stay untouched. A naive "project the polygon onto the image plane" solution gets
this wrong.

## Inputs and outputs

| Input | Source | What it gives us |
|---|---|---|
| AOI polygon | GeoJSON in WGS84 | The site footprint as a 2D ground polygon |
| TIFF image | the source aerial photo | Pixel data plus intrinsics in the `ImageDescription` tag |
| EO row | the orientation text file | Camera position plus omega/phi/kappa rotation in EPSG:3011 |
| LAS tiles | LiDAR laser scan | The actual measured surface of the world (a dense point cloud) |
| **Output** | redacted TIFF plus debug PNG | Pixelated/blurred site, untouched everything else |

## Pipeline stages

```
  (1) load polygon            (2) build camera
   GeoJSON                     EO row + TIFF tags
      |                              |
      v                              v
   Polygon (EPSG:3011)         Camera (K, R, C)
      |                              |
      +--------------+---------------+
                     |
   (3) build scene   |   <- LAS files
                     v
        [ tile select by header  ]
        [ stream + xy-clip       ]
        [ max-z voxel downsample ]
        [ 2D Delaunay -> TIN     ]
        [ tag sensitive faces    ]
        [ build Embree BVH       ]
                     |
                     v
        SceneMesh {mesh, intersector, sensitive_face_mask}
                     |
   (4) compute screen bbox (project AOI vertices over the height range)
                     |
                     v
        bbox_uv (small window, not the full image)
                     |
   (5) cast one ray per pixel in bbox via Embree
                     |
                     v
        binary mask (full-res, only the bbox region populated)
                     |
   (6) read source image
                     |
                     v
        image array (H, W, 3)
                     |
   (7) pixelate + blur, composite inside the mask
                     |
                     v
   (8) write GeoTIFF preserving profile/tags (plus a debug overlay PNG)
```

## Key ideas

### 1. Reduce redaction to a depth query

The whole problem **becomes**:

> *For each image pixel, what is the (x, y) of the world point its ray hits
> first? If that (x, y) is inside the AOI polygon, redact.*

This single reframing lets us throw away every notion of "buildings" or "objects"
or "semantics". Occlusion becomes a side effect of asking "first hit", which is
what raytracing does for free. No z-sorting code, no special-casing trees vs walls
vs ground. The mask is `sensitive_face_mask[hit_face_id]`, and that is all the
occlusion logic.

### 2. The depth surface comes straight from the LiDAR

Two seductive alternatives were rejected (the `scene.py` module docstring records
this):

- **Clean building polyhedra from a roof-fitting step.** Adds a fitting step that
  *can* mis-segment, and only models buildings, so we would still need a separate
  ground/vegetation surface and have to merge them. More code, more failure modes,
  more dependencies.
- **Rasterized DSM.** Has discontinuities at vertical walls; oblique rays can leak
  *through* a building between a roof cell and the next ground cell. A steep
  oblique view is exactly where this fails.

Instead: **2D Delaunay triangulation of the LAS itself** (xy only, lifted to z).
Buildings, trees, walls and ground all collapse into one continuous triangulated
heightfield. The TIN puts a steeply-sloped triangle along each wall instead of a
discontinuity, which is approximately right for occlusion. No semantic
interpretation step means no semantic failure mode.

### 3. Max-z voxel downsample (the DSM trick)

Triangulating the raw LAS (millions of points) is infeasible. We collapse points
in metre-scale xy cells, **keeping the highest z per cell**:

```python
ix = floor(x / 1.0); iy = floor(y / 1.0)
key = ix << 32 | iy            # one int64 per cell
order = lexsort((-z, key))     # group by cell, max-z first within group
mask = key[order][1:] != key[order][:-1]   # first hit per cell
```

This collapses the cloud by more than an order of magnitude and yields a TIN that
follows the *visible* top surface, the same surface a camera actually sees. No
ground points trapped under building roofs (which would create spurious "tunnels"
the rays could fall into).

### 4. Camera math: the rotation convention is empirical

The orientation file's documentation says:

> *"R1 (omega), R2 (phi), R3 (kappa), map-frame to object frame, Rot. seq.
> XYZ_R"*

A reasonable reading: build `R = R_x(omega) * R_y(phi) * R_z(kappa)` (intrinsic
XYZ) and treat it as world-to-camera. **It is wrong by a transpose.** Empirically,
what they actually mean by "map-frame to object frame" is "the rotation that takes
the *frame's basis vectors* from the map frame to the object frame", which is the
inverse of "the rotation that takes a *vector* from map coordinates to object
coordinates".

So the actual world-to-camera matrix is the transpose,
`(R_x(omega) * R_y(phi) * R_z(kappa)).T`. The 4-way `tests/test_camera_sanity.py`
script projects reference building footprints with all four candidate conventions
and lets visual inspection pick the winner. It is a hard gate before trusting the
projection math, because every downstream step is invalid if the camera is wrong.

The other piece: convert the photogrammetric "+x East, +y North, +z Up" frame to
OpenCV's "+x right, +y down, +z forward" via `diag(1, -1, -1)`, done once at camera
construction time, then forgotten.

The `Camera` dataclass holds the result:

```python
@dataclass
class Camera:
    intrinsics: Intrinsics             # f, principal point, image size (in pixels)
    R_world_to_cam: np.ndarray         # (3, 3), OpenCV-style camera frame
    C_world: np.ndarray                # (3,), camera origin in world coords

    def world_to_image(self, xyz)      # returns (uv, depth)
    def image_to_ray(self, uv)         # returns (origins, unit-length directions)
```

`world_to_image` is the standard pinhole projection in row-vector form:

```python
rel = xyz - C_world                    # (N, 3)
cam = rel @ R_world_to_cam.T           # row-form of R @ v_col
z = cam[:, 2]                          # depth (positive = in front)
u = u_pp + f * cam[:, 0] / z
v = v_pp + f * cam[:, 1] / z
```

`image_to_ray` is the inverse (for raycasting): pixel to unit ray direction in
world coordinates:

```python
x = (u - u_pp) / f; y = (v - v_pp) / f; z = 1
cam_dirs = stack([x, y, z])
world_dirs = cam_dirs @ R_world_to_cam     # row form of R^T @ v_col
world_dirs /= norm                          # unit length
```

### 5. Ray budget control: project the AOI to find a tiny screen bbox

A full frame holds far too many pixels to cast a ray for each. We don't need to:

```python
# Project AOI vertices at BOTH z_min and z_max from the LAS clip
# (a vertical extrusion of the polygon, giving its 3D bounding column)
pts = vstack([ring_xy at z_min, ring_xy at z_max])
uv = camera.world_to_image(pts)
bbox_uv = aabb(uv[depth>0]).clamp(image_bounds)
```

*No possible AOI pixel can lie outside this bbox*: anything inside the AOI's xy
column at any height projects somewhere inside the rectangle bounded by its base
and roof projections. Outside the bbox we don't even bother, which cuts the ray
count from the whole frame to a small window.

### 6. Sensitive faces are tagged once, looked up in O(1) at query time

Per triangle:

```python
centroids_xy = points[faces, :2].mean(axis=1)               # (F, 2)
sensitive_face_mask = shapely.contains_xy(polygon, *centroids_xy.T)   # (F,) bool
```

`shapely.contains_xy` is vectorized through GEOS, so every face centroid is tested
in a single call. Then at raycast time:

```python
hits = intersector.intersects_first(origins, directions)    # (n,) face IDs or -1
mask = hits >= 0
mask[mask] &= sensitive_face_mask[hits[mask]]
```

The mask is a *single bool array indexing*, which is the entire occlusion logic.

### 7. Composite only inside the bbox

Reading and writing the full image is unavoidable (we have to write the unchanged
pixels back). But the *expensive operations*, pixelate and Gaussian blur, only run
inside `image[v0:v1, u0:u1]`. PIL's `resize(BOX)` then `resize(NEAREST)` does the
pixelation; `ImageFilter.GaussianBlur` does the blur. Composite via
`np.where(mask3, blurred, original)`.

## Module-by-module map

### `timing.py`

The logger and progress utilities every other module uses. A `step()` context
manager logs the start and end of each stage with elapsed time, and `s.tick(done)`
logs throttled progress with throughput and ETA during long operations like
raycasting.

### `camera.py`

EO parsing, intrinsics extraction, and the `Camera` dataclass with `world_to_image`
and `image_to_ray`. Default rotation convention is `xyz_intrinsic_T` (the
empirically chosen result); three other conventions are exposed for the
verification loop. Intrinsics missing from a TIFF's `ImageDescription` tag are
recovered from another image of the same camera (see `build_intrinsics_cache`),
with a consistency check that refuses to guess when a camera's images disagree.

Key public symbols:

| Function | Role |
|---|---|
| `parse_intrinsics_from_tiff(path)` | Read intrinsics from the `ImageDescription` tag |
| `parse_eo_row(eo_path, image_stem)` | Parse a column-aligned EO text file |
| `world_to_photo_rotation(omega, phi, kappa, conv)` | Build a 3x3 rotation matrix |
| `build_camera(tiff, eo, conv=...)` | Top-level constructor |

### `scene.py`

Tile selection, then a streamed LAS clip, then max-z voxel downsample, then 2D
Delaunay, then sensitive face tagging, then the Embree BVH. Wraps everything in a
`SceneMesh` dataclass.

```python
@dataclass
class SceneMesh:
    mesh: trimesh.Trimesh
    intersector: trimesh.ray.ray_pyembree.RayMeshIntersector
    sensitive_face_mask: np.ndarray   # (F,) bool
    point_count: int
    bbox: tuple[float, float, float, float]
```

The top-level builder:

```python
def build_scene(las_dir, polygon, *, buffer_m=200, voxel_size_m=1.0) -> SceneMesh
```

### `redact.py`

The conductor. Builds the camera and scene, computes the image-space bbox, batches
rays into Embree, applies pixelate and blur to the masked region, and writes the
output GeoTIFF preserving profile and tags. A downsampled debug overlay PNG is
written alongside for a quick visual check.

### `cli.py`

`oblique-redact --image --polygon --eo --las-dir --out [...]`. Reprojects the
polygon from WGS84 to EPSG:3011 and calls `redact_images`.

### `tests/test_camera_sanity.py`

The hard gate. Reads reference building footprints, projects them into a
downsampled version of the source image with all four candidate rotation
conventions, and saves overlay PNGs to a temporary folder.

Run before trusting any new EO source.

## Things that aren't there

- **No lens distortion.** The source imagery comes from a metric camera and is
  distortion-corrected upstream. The verification overlay would surface any
  residual.
- **No explicit camera frustum culling** for LAS clipping. At high altitude,
  occluders only matter within a short xy distance of the AOI (the camera-to-AOI
  ray stays well above building height until very close to the AOI), so a square
  bbox around the AOI is enough.
- **No overhang handling.** The 2.5D TIN can't model balconies or cantilevers.
  Typical urban scenes have essentially none.
- **No image-to-AOI pre-filtering.** `redact_images` loops a directory of images
  against a list of AOIs (one scene built per AOI, reused across all images, AOIs
  never merged), but it still opens every image and builds its camera to test
  visibility. Fine for dozens to a few hundred images.
- **No semantic understanding of the AOI.** We don't know it is a building or a
  courtyard or empty land, and don't need to: the geometry test handles it
  uniformly.
