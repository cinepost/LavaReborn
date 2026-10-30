# Staging upload chunk measurements

The production default is 16 MiB. This experiment compares 4, 16, 32 and 64 MiB
without changing that default. The opt-in harness is
`src/renderer_upload_bench.rs::measure_staging_chunk_sizes`.

## Workload

- NVIDIA RTX 2060 SUPER, 8 GiB VRAM, existing single graphics/compute queue.
- Frozen kapibara export from the preceding profiling work (`build/kapibara-profile/scene.json`).
  The current HIP is not recooked or modified by this benchmark.
- Real kapibara geometry records: 646,671,680 bytes per copy (616.71 MiB).
- Upload volumes: 1x (616.71 MiB), 5x (3.01 GiB), 10x (6.02 GiB).
- Source is a contiguous, prefaulted CPU allocation containing repeated real
  records. Source construction is excluded from upload times.
- New destinations are ordinary device-local memory. The production staging
  implementation allocates/maps/copies/reuses a chunk-sized host-visible buffer,
  submits each chunk, and waits for its fence before reusing staging.
- 1x and 5x destinations are fully resident. To retain headroom for the active
  renderer and desktop, 10x streams through two reused 5x destination windows.
  Thus 10x tests transfer volume, not a complete resident 10x scene.
- Foreground is real 1920x1080 interactive kapibara ray-query rendering with its
  existing depth/light/material settings, one sample per batch, GPU output, and
  no image readback. There is no OpenGL consumer or Houdini UI in this harness.
- Quiet uploads have no competing render submissions. Concurrent uploads compete
  with continuous progressive rendering, including beyond the normal convergence
  budget. A converged stationary viewport is closer to the quiet case.

Three measured trials per scale/chunk/mode; rotate chunk order between trials
and alternate which mode runs first. Warm the foreground for eight frames before
each concurrent run. The baseline is 60 rendered frames after a 12-frame warm-up.
Validate all four sizes separately with Vulkan validation, then disable validation
for performance runs. Verify uploaded prefix and suffix data after every trial.
The preexisting multi-chunk test verifies full data across a chunk boundary.

## Timing interpretation

Upload duration includes staging allocation, CPU memcpy/map/unmap, queue waiting,
recording, GPU DMA, and fence waiting. Destination allocation is recorded
separately; total update wall time includes it. CPU scene extraction, tessellation,
BLAS/TLAS creation, shader compilation and source construction are excluded.

Test-only thread-local instrumentation measures time waiting for the shared
submission mutex and time holding it through command recording / submission /
fence completion. Hold time is a wall-clock exclusion interval for other threads,
not an isolated GPU DMA timestamp. It includes scheduler/driver effects.

Frame times include rendering, GPU-frame allocation and submission contention.
They exclude actual GUI presentation and input latency. Allocation outliers are
kept rather than discarded; steady copy cadence and destination-allocation stalls
must be considered separately. The large buffer is never activated as geometry,
so this experiment does not measure rendering a 5x/10x scene or building its AS.

## Results

Three trials per configuration, 72 measured uploads in total. Upload wall times
below include new destination allocation and staging/copy/fence work; they exclude
CPU source construction and AS construction. Baseline frame median was **21.73 ms**
(p95 25.16 ms over 60 frames). Frame p95 columns pool the actual frames from all
three concurrent trials. Frame counts differ by configuration because shorter
uploads expose fewer frames to contention.

### Practical upload volumes while continuously rendering kapibara

| Chunk | 5x completion (3.01 GiB) | 10x completion (6.02 GiB) | 10x frame median | 10x frame p95 | Upload lock hold p95, 10x |
|---|---:|---:|---:|---:|---:|
| 4 MiB | 17.08 s | 34.53 s | 22.3 ms | 24.1 ms | 2.07 ms |
| 16 MiB | 4.56 s | 9.11 s | 23.3 ms | 25.4 ms | 3.34 ms |
| 32 MiB | 2.46 s | 4.90 s | 24.7 ms | 26.2 ms | 4.82 ms |
| 64 MiB | 1.46 s | 2.80 s | 27.3 ms | 28.9 ms | 7.43 ms |

### Quiet upload completion, no competing render submissions

| Chunk | 1x (616.71 MiB) | 5x (3.01 GiB) | 10x (6.02 GiB) |
|---|---:|---:|---:|
| 4 MiB | 0.171 s | 0.857 s | 1.647 s |
| 16 MiB | 0.138 s | 0.664 s | 1.263 s |
| 32 MiB | 0.137 s | 0.638 s | 1.182 s |
| 64 MiB | 0.140 s | 0.624 s | 1.141 s |

### Interpretation

Under contention, the uploader gets roughly one synchronous chunk submission
between render batches. At 10x, 4 MiB needs about 1542 copies; 64 MiB needs 98
copies across the two windows. Most extra elapsed time is waiting for the render
queue, rather than moving bytes. For example, at 10x/16 MiB, upload-thread queue
waiting totals approximately 7.49 seconds of the 9.11-second completion time.

64 MiB is the fastest bulk-transfer choice here. It reduces 10x concurrent update
completion by about **3.25x** versus 16 MiB, for a frame p95 increase of about
**3.6 ms**. 32 MiB is a compromise: 4.90-second completion and 26.2 ms frame p95.
4 MiB preserves cadence but causes very late geometry activation under sustained
rendering. Quiet 10x completion differs much less: 1.26 seconds at 16 MiB versus
1.14 seconds at 64 MiB.

New 5x destination allocation took approximately 76 ms during concurrent rendering.
The first frame consequently lasted roughly 99–116 ms across configurations; in
those cases it was the maximum frame. p95 alone hides this one-time allocation
stall in long runs. Buffer allocation/reuse is a separate issue from chunk sizing.

The active render workload stays at **1x kapibara** in all cases. A genuinely more
expensive foreground scene can hold the queue longer and worsen contention. Do
not interpret these results as a benchmark of complete resident 5x/10x scenes.
The 10x upload is windowed, and a full 10x scene also needs acceleration structures,
images and old/replacement snapshots beyond its 6.02 GiB of records.

Recommendation: use 64 MiB for bulk geometry transfers when fast scene activation
is the priority; choose 32 MiB for a tighter frame-cadence compromise. The current
16 MiB production default remains unchanged. Select 16, 32 or 64 MiB in Lava
Render Settings → Advanced → Geometry Upload Chunk Size; changes apply to the
next geometry upload without resetting accumulation. Longer term, pipelined
transfer submission and staging reuse should replace the per-chunk synchronous
fence cycle; incremental per-object uploads would also avoid retransferring a whole
large scene for a small geometry edit.

Raw results: `build/kapibara-profile/chunks-report.json`.
Aggregates: `chunks-summary.json` and `chunks-summary.csv` in the same folder.
Vulkan validation passed for the four-size concurrent smoke test and the full
byte-preservation chunk-boundary test. CPU tests and clippy pass. No image readback
was used during foreground rendering.

## Reproduction

Using the exported kapibara JSON from `profile_hip_scene.py`:

```sh
cargo test --release --lib --no-run
LAVA_UPLOAD_SCENE="$PWD/build/kapibara-profile/scene.json" \
LAVA_UPLOAD_OUTPUT="$PWD/build/kapibara-profile/chunks-report.json" \
LAVA_UPLOAD_ROUNDS=3 \
cargo test --release --lib measure_staging_chunk_sizes -- --ignored --nocapture
```

Set `LAVA_UPLOAD_SMOKE=1`, `LAVA_UPLOAD_ROUNDS=1` and
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` for the shorter 1x correctness run.
Raw outputs and logs are under `build/kapibara-profile/chunks-*`.

Frozen source JSON SHA-256: `2e09adefa9b908251740f92a0fbfef6dcef9a88a8628148acbd8b9f4cd6ad46e`.
