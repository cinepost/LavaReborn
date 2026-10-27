# Visibility routes and sampling

Select `settings.visibility` in JSON, or the string `lava:visibility` in Hydra/USD
Render Settings. Values are `ray_query` (the compatible default) and `raster`.
CLI values are `--visibility ray-query` and `--visibility raster`. Both choices
are independent of `--mode interactive` / `--mode final`.

The visibility pass owns its hit buffer and optional raster resources. Raster
resources are allocated lazily when selected: two RGBA32UI attachments and one
D32 depth attachment (36 additional bytes/pixel), framebuffer, render pass,
graphics pipeline and compute resolve pipeline. They remain entirely on the GPU.
The material visibility product is reused for stochastic cutouts. Surface BSDF
code is absent from the graphics shaders; accumulation remains material-agnostic.

Ray mode uses hardware primary ray queries. Raster mode pulls triangle positions
from the geometry buffer using `gl_VertexIndex`; perspective-correct barycentrics
and UVs are stored with material/primitive IDs in depth-tested integer attachments.
Attachment writes obey raster/depth ordering, rather than racing fragment SSBO
writes. A GPU resolve writes the shared 32-byte Hit format and overlays analytic
rectangle-emitter intersections using Light BVH. Secondary and shadow rays retain
hardware ray queries in both modes. Raster clipping and ray near/far clipping use
the same camera-space planes. Shared edges may differ within raster subpixel rules.

A 16 KiB device-resident table supplies 4,096 distinct prime bases, enough for
the validated 96-event path limit without wrapping dimensions. A graphics+compute
queue is selected at startup. Raster attachment format features
are checked before allocation. Graphics pipelines share the existing Vulkan driver
cache; all eight shader products use the validated content-addressed SPIR-V cache.
Subpass dependencies synchronize attachment writes/resolve reads/reuse. Material
upload barriers cover vertex and fragment reads as well as compute reads.

## Thin-lens camera

A camera sample has two film-jitter dimensions and two pupil dimensions. These
four samples are coherent across the image **only in raster mode**, allowing a
single hardware projection per sample. Ray-query mode independently scrambles
film and pupil samples per pixel and changes ray origins/directions directly;
it does not modify the projection matrix. The same camera functions reconstruct
the selected primary ray for visibility and initial path construction. Light/path samples are decorrelated
per pixel and sampling dimension with nested digit scrambling.

For a perspective camera, the pupil point translates the camera origin in its XY
plane. An off-axis projection shift/shear makes all pupil views agree on the plane
at `focus_distance`. Film jitter shifts the projection center. The projection uses
Vulkan +Y-down and [0,1] clip depth, with asymmetric aperture offsets preserved.
Orthographic DOF uses a parallel thin-lens shear and displaced origin, also fixing
the selected focus plane. `f_stop = 0` disables DOF. Optical distances and focus
distance must use consistent scene units. The existing native USD camera adapter
converts Houdini optical attributes from tenths of scene units.

## Samplers

Interactive uses an extensible Halton sequence with prime-base radical inverses.
Final uses a fixed-budget Hammersley set: its first coordinate stratifies the known
sample count, and remaining coordinates use radical inverses. Camera dimensions
are fixed and use Cranley–Patterson rotations, keyed per pixel in ray-query mode. Paths begin at dimension four;
per-pixel/per-dimension nested affine digit permutations scramble their prime-base
radical inverses. This avoids reusing camera coordinates for BSDF/NEE choices and
reduces structured high-dimensional projections without replacing the set with
white noise. Cutout acceptance retains a separate deterministic hash RNG.

Hammersley was selected for deterministic low-discrepancy integration of a known
budget. Poisson-disk sampling offers useful blue-noise error structure, but is not
universally a faster-converging integrator. Fixed sets can distribute a known
number of samples better than extensible sequences; see the primary reference
[PBRT Sampling and Integration](https://pbr-book.org/4ed/Sampling_and_Reconstruction/Sampling_and_Integration).
This is a design choice, not a claim of universal superiority over Poisson sampling.

A final frame's total budget is fixed at the start of accumulation. API callers
changing it receive an error requiring reset. Camera/scene/settings updates already
reset history and restart the sequence. Interactive sample prefixes do not depend
on a target sample count. Interactive/final images have matching expected energy,
but intentionally use different sample patterns and need not match bit for bit.

## Verification (2026-10-26)

GPU tests compare raster/ray primitive IDs, barycentrics, UV/material attributes
and depth under perspective/orthographic projection, pinhole/thin-lens DOF,
asymmetric offsets, opacity and analytic rectangles. Tests also check progressive
prefix stability and final-budget reset requirements. Existing OpenPBR/MIS/area
shadow/interactive GPU presentation regressions remain enabled.

The 800×450, 96-sample DOF showcase produced mean absolute RGB differences of
0.00762 / 0.0072 / 0.0068 in 8-bit channel levels between raster/ray paths.
Preview artifacts: `build/dof-raster.png` and `build/dof-ray-query.png`.
On the local RTX GPU, measured render-loop times for that scene were 2.30s raster
and 2.21s ray query; timings are not general performance guarantees. The default
visibility route remains unchanged. Native Houdini 20.5 compositor tests use the
saved kitchen scene with authored `lava:visibility` and active camera DOF.


Final checks: 15 CPU tests, 10 GPU tests and strict Clippy passed. Raster camera
motion delivered 90 GPU images during 2,712 continuous camera requests; ray mode
delivered 94 during 2,625 requests, with zero presentation image readbacks in both.
The native kitchen DOF check delivered GPU color/depth callbacks in both routes
(40 raster / 34 ray), with zero CPU presentation uploads. Houdini's cache requested
one warned CPU map per run, as before. Installed shared libraries were replaced
atomically; restart Houdini to load the new version. User HIP files were not edited.
