# Curve visibility performance

## Implemented

Curve acceleration is now split by USD/Hydra curve object. Each object has an
immutable procedural-AABB BLAS and its own TLAS instance. Instance custom indices
translate object-local primitive indices back to the current shading-record array.
Mesh and curve instances use masks 1 and 2; secondary/shadow queries use mask 255.

A weak, device-specific cache reuses live curve BLASes when their ordered AABBs
and build policy are identical. It does not include material IDs, mesh primitive
offsets or shading parameters in the acceleration key. Thus editing another mesh,
changing materials, or rebuilding another curve object does not rebuild unchanged
curve BLASes. Geometry edits that preserve the exact ordered AABBs can also reuse
acceleration safely: intersection always reads the newly uploaded segment records.
Switching fast-build / fast-trace policies creates a different product. Removed
objects release GPU resources when the last scene snapshot releases them; the
cache does not keep old VRAM allocations alive. Uploading segment records and
rebuilding the scene TLAS remain necessary on a geometry snapshot replacement.
Existing camera/light/material-only update paths still bypass geometry builds.

Immutable mesh/curve shading records reside in device-local VRAM. A reusable
16 MiB host-visible staging buffer uploads them in completed chunks on the builder
thread; the previous scene stays active until the replacement is ready. Staging
is released after upload. AS build-input buffers keep their existing staging path.
See [kapibara measurements](kapibara-profile.md) for the system-memory diagnosis.

The default primary visibility query retains unified traversal. Each newly
committed hit reduces the traversal distance, and the procedural intersection
routine also limits its search to the current committed distance. Do not terminate
at the first curve hit: hardware candidate order does not guarantee nearest hits.

An experimental polygon-first visibility product can be enabled before starting
Houdini or the CLI:

```
LAVA_CURVE_POLYGON_FIRST=1
```

This first traces mask 1, then traces mask 2 with the exact accepted polygon hit as
`tMax`. Both queries use the same current-sample ray, including film/lens sampling,
clipping, and stochastic cutout acceptance. It does not treat a rejected alpha
surface as an occluder. Curves shorten the bound further as they commit hits.
No extra depth image, readback, raster prepass, or synchronization dispatch is
needed. It remains experimental because the additional traversal can outweigh
any saved curve work. The raster visibility path itself is unchanged.

## Measurement

`curve_polygon_first_visibility_matches_unified_and_reports_timings` compares hit
IDs, distances, barycentrics, material IDs and UVs for 4096 segments at 128x128,
with opaque polygon occlusion, exposed curves, and stochastic cutout plus DOF.
It measures only visibility dispatches using Vulkan timestamps after warming both
specialization products. Timing includes a GPU write barrier between repeats. Results below are medians
of nine batches of 16 dispatches with alternating product order.
The staging readback in this test is diagnostic only, not part of viewport output.

RTX 2060 SUPER run with validation enabled (milliseconds per dispatch):

| Scene | Unified | Polygon first |
|---|---:|---:|
| Fully occluded | 0.015 | 0.019 |
| Exposed | 0.107 | 0.113 |
| Cutout and DOF | 0.024 | 0.028 |

These small synthetic scenes do not establish the result for a production groom;
there is no measured Karma comparison. Unified remains the default accordingly.

## Hi-Z design constraints / next experiment

A Hi-Z pyramid is **not implemented**. A depth pyramid alone cannot remove
instances from the immutable shared TLAS; it needs a separate primary-visibility
work scheduler or visible-group lists. Camera-culled groups must remain available
to shadow and continuation rays.

A useful experiment would split visibility into polygon depth, conservative group
classification, and curve tracing. Spatially coherent groups need world bounds,
a projected rectangle and a nearest possible camera-space depth. Reduce polygon
depth with **maximum** depth for conventional increasing depth, with uncovered
pixels represented by far depth. Reject only when the group's nearest possible
depth is farther than every covered pixel's occluder depth. Sampling an average,
minimum depth, or only the center of the projected rectangle is unsafe.

The test must cover all overlapping coarse cells, jitter, lens aperture, near-plane
crossings and orthographic projection. Per-pixel sampled lens rays complicate
screen-space bounds: conservative aperture expansion is required, or disable coarse
rejection for DOF until it is proved correct. Previous-frame depths cannot safely
occlude newly exposed geometry without reprojection/disocclusion handling; start
with current-sample depth. Alpha-tested polygons must use accepted coverage for
the exact sample, rather than being treated as opaque.

For the current hardware-query backend, exact per-ray tMax already lets the GPU
reject whole BVH nodes behind an occluder. The Hi-Z experiment must therefore
measure a net saving after pyramid construction, group tests, list compaction and
scheduling. Record polygon-query time, pyramid/classification time, curve-query
time and surviving group/candidate counts independently. Do not adopt it based
only on fewer analytic intersections.


## Animated polygon meshes with static curves

Interactive fixed-topology mesh deformations use incremental updates. Hydra omits
unchanged curves from mesh-only submissions; Rust shares their immutable CPU
snapshot instead of serializing, parsing or cloning the fur again. The renderer
regenerates only changed mesh records and updates their ranges in the existing
GPU geometry and vertex buffers. Static curve records, AABB inputs and curve
BLAS allocations remain untouched.

Interactive mesh BLAS and shared TLAS builds enable Vulkan `ALLOW_UPDATE`.
Matching-topology deformations refit these structures in place at a renderer
batch boundary, after preceding GPU work has completed. The shared TLAS is
updated to track the mesh's new bounds. Accumulation restarts after both refits
complete, before rendering the changed geometry. These short commits serialize
with rendering; they do not render against partially updated acceleration.

Mesh topology/orientation changes, curve changes, changed mesh count, or changes
in the number of valid lowered triangles use the existing asynchronous immutable
replacement path. Final mode retains full builds favoring traversal quality.
Currently polygon meshes share one BLAS: an edit uploads only changed objects,
but refits the combined polygon BLAS and the shared TLAS. Separate per-mesh BLASes
and background deformation preparation remain possible further optimizations.

Set `LAVA_TRACE=1` to print mesh-only submission sizes and refit commits, including
changed mesh count, uploaded bytes, and zero curve uploads/builds. Native timeline
testing can use `LAVA_TEST_ANIMATION_FRAMES` with `scripts/test_solaris.py`.


### Animated kapibara validation

A native Houdini 20.5 Solaris check used the animated `kapibara_01.hip` snapshot,
a fitted full-kapibara camera, ray-query visibility, 320×180 resolution and one
sample. Frames 2–4 averaged **3.84 s** with the previous renderer and **0.283 s**
with incremental updates. Updated runs enabled Vulkan validation; the original
baseline did not. No GPU test workload ran concurrently with these measurements.
These are complete HUSD frame-change-to-convergence timings, not GPU kernel times
or a full-resolution viewport benchmark.

Per deformation, the JSON payload fell from about 248 MB to 7.8 MB, and the new
path uploaded 8,182,216 bytes of mesh records/positions with zero curve uploads
and zero curve BLAS builds. GPU upload/refit commit wall time was about 10 ms.
The remaining total includes Hydra synchronization, JSON work, CPU mesh lowering,
rendering and host convergence polling. Both compositor outputs used GPU color
and depth. HUSD separately requested a CPU AOV cache Map on each converged frame;
this triggered the existing warning and did not change compositor transport.

The GPU regression compares four deformations against full rebuild images in
both raster and ray-query modes while checking retained buffer and AS handles.
It also rejects topology and curve changes from the refit path. All 36 GPU
regressions passed with Vulkan validation, including asynchronous topology builds
and resize during pending builds. Raw native logs and timing summary are under
`build/kapibara-profile/refit-native-{before-quiet,after}.log` and
`build/kapibara-profile/refit-animation-summary.json`.

For the subsequent mesh-only CPU import optimization and fur-hidden timeline
measurements, see [interactive scene updates](interactive-scene-updates.md).

The shading pass also supports [GPU ray queue reordering](ray-queue-reordering.md),
with measured mixed-material gains and effectively neutral full-fur kapibara time.
