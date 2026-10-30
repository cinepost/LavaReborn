# Interactive mesh update latency

Fixed-topology mesh animation refits the polygon BLAS and shared TLAS, retains
static curve resources, and uploads only changed mesh ranges. The worker mailbox
is latest-wins; it does not replay every submitted timeline frame. Camera/light
updates and asynchronous topology replacements retain their existing behavior.

After implementing refits, the main remaining delay in the kapibara scene was
synchronous Hydra mesh import: formatting points, face-varying normals, UVs and
indices into decimal JSON took about 197 ms per mesh update. GPU work alone did
not explain timeline lag.

The mesh adapter now uses a reserved string builder with `to_chars` for scalar
numbers/indices. Float attribute arrays use a compact internal JSON object:
`{"f32le":"0000803f..."}`. This is hex of little-endian IEEE float32 bytes, grouped
by the attribute's component count (three for points/normals, two for UVs). It
avoids formatting and parsing each coordinate as a decimal. World-space points
and normals are converted to their GPU float32 representation before transport.
Mesh validation still rejects non-finite attributes and invalid indexing.

Ordinary nested JSON arrays remain supported on input and remain the serialized
Rust scene format. This transport is internal to the native Hydra adapter; it
requires no render setting. It transports scene attributes, not rendered images.
Vulkan/OpenGL viewport presentation remains unchanged.

## Measurements

Native Houdini 20.5 Solaris, RTX 2060 SUPER, the animated kapibara snapshot with
all curves hidden, a fitted camera, 320×180 pixels and one sample. Frames 2–5 were
measured after initial scene creation. These are first updated-image timings
through the HUSD test harness, not isolated GPU timestamps or a comparison with
Karma. Higher resolutions and render settings can add rendering time.

| Path | Mesh sync | JSON decode | Full frame update |
| --- | ---: | ---: | ---: |
| Previous decimal stream writer | ~198 ms | ~23 ms | ~272 ms |
| Reserved decimal writer | ~60 ms | ~22 ms | ~134 ms |
| Packed float attributes | ~13 ms | ~15 ms | ~78 ms |

The packed run enabled Vulkan validation; the baseline did not. GPU upload and
BLAS/TLAS refit commit time remained roughly 10 ms. CPU mesh lowering still runs
on the renderer worker, and mesh attributes/topology still pass through JSON;
further reductions could use direct typed submission and cached triangle-corner
mappings. Those changes are not part of this update.

Both compositor color and depth used GPU delivery. HUSD separately mapped each
converged AOV into its CPU cache, producing the existing warning. Those cache
reads are separate from viewport presentation and occur in both measured paths.

## Checks

`lava_json_writer_smoke` checks JSON structure, scalar precision, integer bounds
and known packed float bit patterns. Rust transport tests check bit preservation,
legacy array input and rejection of malformed encodings. Native Solaris checks
verify animated mesh updates and expected UDIM red/blue colors with packed UVs.
The 36 GPU regressions pass with Vulkan validation, including mesh refit images,
normal mapping, textures, continuous camera edits, topology updates and resize.

Set `LAVA_TRACE=1` for mesh sync, JSON parse and refit commit timings. Raw results
are in `build/kapibara-profile/latency-{before,fast-after,packed-after}.log`, with
aggregates in `build/kapibara-profile/latency-summary.json`.
