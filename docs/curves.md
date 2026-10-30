# Curves

Lava imports USD/Houdini `BasisCurves`, including linear, Bezier, Catmull-Rom and
B-spline bases, nonperiodic and periodic topology, constant/uniform/vertex/varying
widths, material bindings, transforms, visibility edits and animated control points.
The `radius` primvar (or scalar/array `radius` attribute) overrides USD `widths`;
widths are diameters. Nonuniform object scaling uses the largest axis scale for a
conservative round radius. Indexed Hydra control points are expanded on import.
Pinned curves and centripetal Catmull-Rom are currently warned and omitted.
Display color is constant per curves primitive. No strand-specific hair BSDF is
provided: curves use the existing legacy/OpenPBR material and texture subsystem.

## Geometry and traversal

Cubic spans are evaluated into analytic, linearly tapered segments, eight per
span by default. JSON `subdivisions` supports 1–64; linear spans require one
segment. This is a finite spline approximation, not native hardware spline
intersection. Each segment has a padded procedural AABB in a per-object curve BLAS,
instanced alongside polygon geometry in the TLAS. Unchanged curve BLASes are reused
across unrelated geometry edits. Ray-query candidate handling
solves a tapered cone and round endcaps. The same routine handles primary rays,
BSDF continuation, shadows and stochastic opacity. Quadratics are recentered
along the ray to keep thin-strand intersections numerically stable.

`VK_KHR_ray_tracing_position_fetch` retrieves triangle positions; it does not add
native curve primitives. The renderer uses the cross-vendor procedural AABB route
instead of requiring vendor-specific swept-sphere hardware.

Segment records reuse the current 160-byte surface ABI (not CPU hull triangles).
This favors integration over compact hair storage; a dedicated compact curve
record is a future optimization. Materials get stable indices after mesh
materials. Shading reconstructs analytic normals, axial/circumferential UVs and a
surface frame at the hit. Texture loading, BC compression, mipmaps, UDIMs,
normal/bump mapping, transmission, accumulation and GPU presentation are shared.

## Mesh/task visibility

Startup queries and enables `VK_EXT_mesh_shader` only when both mesh and task
features and required limits are present. Raster visibility then uses task groups
of 32 analytic segments. Each group first checks its combined bounds against the
current jittered camera frustum, then compacts surviving segments into its task
payload. Interactive mode also culls diameters below 0.25 pixel; final mode keeps
all strands. This viewport LOD is deliberately approximate and can remove fine
hair at distance. It affects visibility only, never shadow/reflection geometry.

Each mesh workgroup constructs a camera-oriented, circumscribed eight-sided hull
(16 vertices, 28 triangles) including the cap bounds. A dedicated curve fragment
shader intersects the round analytic surface and writes its true depth, material
ID and UVs. The hull itself is not shaded. Polygon rasterization retains its
ordinary fragment shader and early-depth behavior. Raster camera DOF and film
jitter use the same projection as the rest of visibility.

Without mesh/task support a curve scene requested in raster mode uses hardware
ray-query visibility, with an explicit warning. This fallback remains GPU-only.
All GPU paths still require the existing hardware ray-query support.

Curve deformation/topology/radius edits follow the existing asynchronous immutable
BLAS/TLAS replacement policy. OpenPBR material edits update tables without rebuilding
curve geometry. No geometry/material dependency is added to accumulation.

See [visibility performance](curves-visibility-performance.md) for the experimental
polygon-first query product, measured costs, BLAS reuse and conservative Hi-Z design.

## Tests and example

Run `scripts/test_curves.py build/curves.usda` with Houdini's `hython` to generate
a scene and JSON example. GPU tests cover round visibility, varying radius,
AABB-only misses, shadowing, off-camera secondary hits, and final-vs-interactive
subpixel culling. Both ray-query and mesh/task visibility run on supporting GPUs.

JSON scenes add `curves`, for example:

```json
{
  "points": [[0, 0, 0], [0, 1, 0]],
  "counts": [2],
  "basis": "linear",
  "radius": [0.01, 0.03],
  "radius_interpolation": "vertex",
  "color": [0.8, 0.4, 0.1]
}
```

Place this object in the scene's `curves` array. Existing mesh-only JSON remains
compatible. Materials use the same `material` object as polygon meshes.
