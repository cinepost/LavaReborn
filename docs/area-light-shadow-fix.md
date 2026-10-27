# Rectangular-light banding fix

Reproduced on 2026-10-25 from the saved, unmodified
`/home/max/Documents/usd/lava_kitchen_01.hip`:

- Displayed LOP: `/stage/LAVA_RENDER_SETTINGS`.
- Rectangular light: `/lights/arealight1`, 50×50, intensity 41.18, exposure 1.
- Captured the HIP's saved free-perspective viewport through Houdini's Python API.
  Replayed that view against the same exported geometry/material/light snapshot
  using pre-fix and fixed libraries, 960×720, 512 samples, identical settings.
- Persistent floor/table bands reproduce with the old library and disappear with
  the corrected shadow segments. Comparison images are in
  `build/kitchen-saved-view-before.png` and `build/kitchen-saved-view-after.png`.

The NEE code offset the shadow origin but retained the direction and distance
computed at the original hit point. For a parallel emitter/receiver, the emitter
intersection moves forward by approximately `offset/cos(theta)`. At grazing angles
it lies before the old `distance - 2*offset` cutoff and is counted as an occluder.
Floating-point rounding and varying light/surface positions turn this into bands.

The shared shadow helper now offsets the origin first and reconstructs the finite
segment toward the sampled target, with an endpoint tolerance. Analytic light-BVH
occlusion excludes precisely the sampled emitter. Other rectangles and mesh
occluders remain active. Camera and continuation rays still see all emitters.
Both legacy and OpenPBR NEE use the corrected helper. GPU presentation is unchanged.

A new GPU quadrature regression failed before the fix with measured 0.01213506
versus expected 0.41724974. It now passes for both BSDF models, at the origin and
at kitchen-scale coordinates. A second test verifies separate rectangle and mesh
occlusion are preserved. All seven GPU regressions, fourteen CPU checks and strict
Clippy pass with Vulkan synchronization validation enabled for GPU checks.

The kitchen stage also passes native Houdini 20.5 Solaris rendering at 960×540:
42 GPU color/depth callback deliveries, zero CPU presentation uploads. Houdini's
cache requested one warned CPU map, as before. The release library was atomically
installed into the existing plugin location. The HIP was opened in a new Houdini
window with Lava selected; the file was not modified or saved by this work.
