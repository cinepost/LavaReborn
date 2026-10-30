# Kapibara GPU profiling

Scene: `test_scenes/kapibara/kapibara_01.hip`, saved frame 1,
displayed Solaris node `/stage/render_settings1`, settings `/Render/Lava`.
The source HIP is read only. Its camera relationship targets a missing camera;
a fitted perspective view was approved for this measurement. All eight world-bound
corners fit inside a 50 mm, 36 x 20.25 aperture camera with a 10% distance margin.

Render parameters: 1920 x 1080, interactive Halton sampling, hardware ray-query
primary visibility, diffuse limit 2, reflection limit 1, refraction limit 0,
roulette start 4, one directional light. Guides are excluded by USD purpose.

## Production staged uploads

The renderer now uploads immutable mesh/curve records into device-local VRAM via
reusable 16 MiB staging. At the same settings, the production path measured:

| Measurement | Original allocation | Production staged upload |
|---|---:|---:|
| Primary visibility | 30.67 ms | 2.86 ms |
| Isolated opaque shadow replay | 20.21 ms | 2.98 ms |
| Original fused shading, including geometry queries | 25.13 ms | 14.06 ms |
| Visibility + original fused shading | 55.81 ms | 16.91 ms |

Output hashes match the original baseline. The new multi-chunk byte-preservation
check and GPU regression suite pass with Vulkan validation. Native Houdini Solaris
viewport delivery is checked separately. Production report:
`build/kapibara-profile/report-staged.json`; validation logs: `upload-gpu.log`,
`upload-husd.log`, `upload-cpu.log`, `upload-clippy.log` in the same directory.
The staging allocation is released after upload. During scene replacement, old
and new geometry records can coexist in VRAM until the new snapshot is activated.

## Initial diagnosis (before staged uploads)

NVIDIA RTX 2060 SUPER, 4 warm-up samples followed by 12 measured samples,
validation disabled for performance runs. Values are medians in milliseconds.
The VRAM diagnostic changes only the geometry-record buffer's storage, not scene
geometry, material parameters, acceleration structures or integrator settings.

| Measurement | Current system-memory records | Device-local VRAM records | Speedup |
|---|---:|---:|---:|
| Primary visibility | 30.67 | 2.86 | 10.7x |
| Isolated opaque shadow replay | 20.21 | 2.97 | 6.8x |
| Shading with geometry-query answers cached | 4.07 | 1.08 | 3.8x |
| Original fused shading including queries, all bounces | 25.13 | 14.17 | 1.8x |
| Primary + original fused shading | 55.81 | 17.14 | 3.3x |
| Isolated continuation replay | 63.06 | 6.18 | 10.2x |

Replay rows are independent experiments, not components to sum. In particular,
the baseline continuation replay is slower than the entire original fused shading
kernel: isolated execution has substantially different cache/occupancy behavior.
Use the original fused shading row for the actual combined rendering cost.

The scene lowers to **41,746 mesh triangles and 3,999,952 analytic curve segments**,
from **499,994 render strands**. The 8,000 guide strands are excluded. Both mesh
and fur use legacy diffuse materials, no textures, and opaque coverage. Approximately
219,659 shadow rays are traced per sample, with 85,845 occluded (39%). The first
secondary wave contains about 192,179 paths and the second about 97,871 paths.

For measured sample 4, primary pixels are 174,404 curves (8.41%), 17,737 mesh
(0.86%), and 1,881,459 background (90.73%). This is a fitted full-object view,
not a close-up with fur covering the entire screen.

The segment/triangle shading-record buffer is **646,671,680 bytes (616.71 MiB)**.
Its baseline Vulkan memory flags are `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`,
without `DEVICE_LOCAL`. The diagnostic copy uses `DEVICE_LOCAL` memory. BLAS/TLAS
storage already resides on the GPU. Geometry access over PCIe is therefore a
confirmed major performance problem for this scene.

Primary visibility SHA-256, first-sample per-stage radiance SHA-256, active path
counts, shadow counts and occlusion counts match between both configurations.
All shadow answers and continuation hits match their captures. The source HIP's
SHA-256 remains `0df6536eb8ba5a5cf44314d211c797edc9b38b28cbf89354b51c51717d19675d`.

Raw reports: `build/kapibara-profile/report.json` and `report-vram.json`.
Validation logs: `smoke-lit.log`, `smoke-vram.log`, `gpu-regressions.log` in that
folder. CPU tests and clippy also pass.

This diagnosis led to staging immutable bulk geometry records into device-local
storage in the production renderer. Uploads now reuse a 16 MiB host-visible staging
buffer, release the submission lock between completed chunks, and publish only
complete scene snapshots. The staging buffer is released after upload. Device-local
allocation uses ordinary VRAM rather than preferring a potentially small host-visible
BAR heap. Camera, light and material-only updates keep their existing paths.

The initial reports above preserve the original baseline and diagnostic comparison.
The profiler's historical VRAM switch is now a no-op when records are already
in device-local memory. See `report-staged.json` for the production-upload rerun.

## Method

The opt-in test in `src/renderer_profile.rs` uses Vulkan GPU timestamps. It measures
primary visibility and each original shading dispatch, including its embedded
shadow/continuation queries. Scene loading, shader compilation, BLAS/TLAS building,
buffer copies, accumulation, filtering, tone mapping and presentation are excluded.

Separate specialization products capture the actual shadow rays and continuation
hits, indexed by pixel for each wavefront stage. They preserve paths and random
sampling. A shadow replay kernel traces precisely those shadow rays, with the same
first-hit termination, analytic curve intersection and material-coverage logic as
production. A continuation replay kernel traces the real queued continuation rays.
A cached-answer shading product performs the shading math and queue compaction
without either geometry query, using the recorded answers. Analytic-light tests
remain part of shading. Neither capture nor replay changes production settings.

Every replayed shadow answer and continuation hit is checked against capture.
Cached-query shading is checked for identical radiance against capture on the
first warm-up sample. Counters and radiance checks are read back outside timed
sections; timed ray records and hit caches stay on the GPU. Statistics collection
is a separate untimed dispatch, so per-ray atomic counters do not inflate query
timings.

Isolated replay and cached shading times are **not additive**: kernel separation,
register pressure, occupancy, memory-cache state and queue scheduling differ from
the original fused kernel. The original shading dispatch time is the actual
combined cost. Report medians, means and ranges after warm-up rather than treating
the difference between two kernels as an exact instruction-level attribution.

## Reproduction

Export with the matching Houdini Python runtime, from `/opt/hfs20.5`:

```sh
source ./houdini_setup_bash -q
./bin/hython /home/max/dev/LavaReborn/scripts/profile_hip_scene.py \
  /home/max/dev/LavaReborn/test_scenes/kapibara/kapibara_01.hip \
  /home/max/dev/LavaReborn/build/kapibara-profile --fit
```

From the project root, build and run the opt-in profiler:

```sh
cargo test --release --lib --no-run
LAVA_PROFILE_SCENE="$PWD/build/kapibara-profile/scene.json" \
LAVA_PROFILE_OUTPUT="$PWD/build/kapibara-profile/report.json" \
LAVA_PROFILE_WIDTH=1920 LAVA_PROFILE_HEIGHT=1080 \
LAVA_PROFILE_WARMUP=4 LAVA_PROFILE_SAMPLES=12 \
cargo test --release --lib profile_scene_visibility_shadows_shading -- --ignored --nocapture
```

Use Vulkan validation for correctness checks, then disable it for reported
performance measurements. Run without competing GPU render workloads. The JSON
report includes each sample and bounce, active path counts, shadow counts,
occlusion counts, material models, geometry storage properties and timing ranges.

To repeat the VRAM counterfactual, add `LAVA_PROFILE_GEOMETRY_VRAM=1` and choose
a different `LAVA_PROFILE_OUTPUT`. The copy is completed before warm-up and its
time is excluded.
