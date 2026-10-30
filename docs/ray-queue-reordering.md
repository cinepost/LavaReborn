# Shading ray queue reordering

The shading pass uses portable GPU software ray reordering after its existing
subgroup compaction. It is enabled by default in interactive and final modes.
Set `LAVA_RAY_REORDERING=0` before starting Houdini or the command-line renderer
to compare with subgroup compaction alone. Restart to change this startup option.
No vendor-specific shader execution reordering extension is required.

## Pipeline and ownership

1. The unchanged shading kernel appends surviving paths with one atomic queue
   reservation per subgroup. Russian roulette and ray depth limits are unchanged.
2. Queue preparation generates separate GPU indirect arguments for shading and
   reordering. Control storage grows from 32 to 48 bytes.
3. A small compute kernel sorts independent blocks of up to 256 rays in place.
   Keys combine outgoing scattering category, a primary-material bucket (128
   buckets), primary mesh/curve class, and outgoing direction octant. Equal keys
   retain local input order. Primary material/class come from visibility metadata;
   these are coherence hints, not the next intersection's material.
4. The next indirect shading dispatch reads the reordered queue. Compute/indirect
   barriers make queue writes visible before consumption.

Sorting only moves key/index pairs during the bitonic network. Each complete
64-byte ray record is loaded/stored once, using about 18 KiB of workgroup shared
memory. The two existing ray queues are retained; no additional full-frame queue,
CPU sorting or queue readback is introduced. Queues below 1,024 rays have zero
reorder workgroups. A partial block with fewer than 16 rays is left unchanged.
Resources and the additional pipeline belong to the shading render pass.

Pixel IDs, random state, throughput, PDFs and all depth counters remain unchanged.
The production shading shader's SPIR-V is byte-for-byte identical to the previous
version. Grouping changes execution order without changing the estimator or
introducing contribution clamps. GPU viewport transport remains unchanged.

This is a block-local software coherence pass, not a global material sort or a
split trace/material-evaluation architecture. Its primary-material hint becomes
less predictive at later bounces. Performance depends on scene and hardware.

## Measurements

RTX 2060 SUPER, interactive mode, three alternating on/off trials per scene.
Each trial renders the same 20 samples, discards the first four from timings, and
compares the resulting image. The table gives the median of the three per-trial
median total sample times. Timings include visibility, all shading bounces,
reordering, accumulation, tone mapping and GPU output allocation; scene load and
AS construction are excluded.

| Scene | Resolution | Off | On | Result |
| --- | --- | ---: | ---: | --- |
| Mixed OpenPBR diffuse/metal/glass, curves, four lights | 1024×576 | 35.75 ms | 33.46 ms | 6.4% lower sample time |
| Frozen full-fur kapibara profiling snapshot | 1920×1080 | 20.83 ms | 21.02 ms | Less than 1% difference; effectively neutral |

These are not measurements of Karma or final-mode render times. Images match
within 1e-5 maximum component error across the paired trials. One explicit
readback per trial/variant is used solely for diagnostic image comparison.
Raw reports are `build/kapibara-profile/reorder-{mixed,kapibara}-report.json` and
`reorder-summary.json`; the mixed input is `reorder-mixed.json` in that folder.

## Validation and diagnostics

A dedicated GPU test feeds synthetic path records through subgroup compaction and
the production reorder kernel. It checks exact payload preservation, survivor
counts, no lost/duplicate paths, dense-block ordering, empty/dead queues, partial
blocks and two-dimensional dispatch indexing. The complete 38-test GPU suite
passes with Vulkan validation, including interactive/final images, normals,
textures, both visibility modes, animated mesh refits and resize.

Use `benchmark_ray_reordering_image_and_total_frame_time` with
`LAVA_REORDER_SCENE`, `LAVA_REORDER_WIDTH`, `LAVA_REORDER_HEIGHT` and
`LAVA_REORDER_OUTPUT` for another scene. The capture/replay stage profiler retains
unsorted baseline dispatches and reports `queue_reordering: false`; use the
full-frame benchmark to include reordering cost. `LAVA_TRACE=1` also prints
pipeline creation timings.
