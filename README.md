# Lava Reborn

A Rust + Vulkan renderer prototype with a C++ USD Hydra 1 render delegate for
Houdini 20.5 (USD 24.03). This iteration implements OpenPBR Surface shading, MIS/next-event estimation,
wavefront path tracing with hardware ray queries, progressive accumulation,
filtering, and tone mapping.
It is a foundation for a photorealistic renderer, not a Karma replacement or a
production path tracer.

## Build

Requires Rust 1.88+, a Vulkan loader/driver, and `glslangValidator`, `spirv-val`, and `sha256sum` on PATH.
On Linux, the shader compiler is commonly supplied by `glslang-tools`. The Rust
build compiles six compute and two raster shader products to Vulkan 1.2 SPIR-V; no SDK is needed for
the standalone renderer. Dependencies are pinned in Cargo.lock.

```sh
cargo build --release
cargo run --release -- probe
cargo run --release -- graph
cargo run --release -- validate examples/studio.json
cargo run --release -- render examples/studio.json --mode final \
  --width 640 --height 360 --samples 64 --output frame.ppm
```

`--gpu N` selects an adapter from `probe`. Device initialization checks Vulkan
1.2, a combined graphics/compute queue, the KHR ray query / acceleration structure / deferred host
operation extensions, and rayQuery, accelerationStructure and bufferDeviceAddress
features. CPU Vulkan devices are rejected by default. NVIDIA RTX, AMD RX with RT,
and Intel Arc are target families; support is decided by capabilities, not vendor
names. No application software traversal fallback exists in this iteration.

Output is binary PPM, top row first, with exposure in stops, an ACES-style fitted
curve and an sRGB transfer function. It is a preview output; EXR and full OCIO
integration are future work. `--mode interactive` uses progressive Halton sampling with the low-latency policy;
final frames use a randomized Hammersley set sized to `--samples`.

## Two execution paths

| Policy | Interactive | Final |
|---|---|---|
| Submission size | 1 sample | up to 16 samples |
| Acceleration build | prefer fast build | prefer fast trace |
| Filtering / GPU display | every sample | last sample of the frame |
| Scene changes | background geometry builds; latest ready version activates | synchronous preparation, checked between batches |
| Main objective | frequent fresh viewport images | amortized submission and display costs |

The Hydra adapter publishes each interactive sample without waiting for convergence.
Camera/light edits reset the accumulation pass and reuse the active acceleration
structures. Exposure-only edits reprocess existing floating-point history on the GPU,
including converged images, without tracing new samples or resetting accumulation. Geometry changes prepare a replacement flattened BLAS and TLAS on a
builder thread while the old scene keeps gathering samples, even beyond its
sample target. At a batch boundary the completed latest version is bound and the
accumulation pass clears history/filter outputs and starts at sample zero. Camera
and light edits also apply while geometry is pending. Obsolete builds are discarded.
GPU frames use monotonic serials, so resetting samples never hides the new image. During continuous
camera motion the viewport accepts the latest completed compatible interactive
image, even if a newer request is pending. Completed samples are not discarded
just because another camera update arrived. Convergence and CPU final-result
polling still require the exact requested revision. Camera/light-only updates use
small JSON deltas and retain shared immutable geometry; mesh edits send full
snapshots. Navigation never waits for edits to stop before presenting images.

Viewport presentation uses Vulkan/OpenGL external memory and semaphores. Any
required CPU image fallback prints `Lava [WARNING]` to stdout, including Houdini's
explicit AOV Map requests for its COP/slap-comp cache. Final mode uses synchronous
scene preparation and larger batches, publishing only the completed frame.
A submitted GPU command cannot be preempted; builds and rendering currently share
one serialized graphics/compute queue. These policies are not cross-vendor latency guarantees.

## Houdini / Hydra

Build against the **same** Houdini SDK used by the host. The integration is
C++ because Hydra's public plugin ABI is C++; rendering stays in Rust behind a
small C ABI. CMake does not run Cargo automatically.

```sh
cargo build --release
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHOUDINI_ROOT=/opt/hfs20.5
cmake --build build -j
cmake --install build --prefix "$PWD/build/install"
export PXR_PLUGINPATH_NAME="$PWD/build/install/hdLava/resources${PXR_PLUGINPATH_NAME:+:$PXR_PLUGINPATH_NAME}"
export HOUDINI_PATH="$PWD/build/install/houdini:&"
```

Start Houdini from this environment. Renderer name: **Lava Reborn (Prototype)**;
plugin identifier: `LavaRendererPlugin`. Plugin support is false when no supported
hardware adapter is visible. The installation includes the Rust shared library
next to the delegate. The installation also registers USD discovery through
`houdini/dso/usd_plugins/plugInfo.json`, so adding the staged Houdini directory to
`HOUDINI_PATH` in `houdini.env` works even when USD initializes before
`PXR_PLUGINPATH_NAME` is applied. This build targets Linux; other platforms need library
path / packaging changes and verification.

Settings exposed by the delegate:

- `lava:mode`: string `interactive` or `final`.
- `lava:samples`: integer target sample count (default 64).
- `lava:exposure`: float exposure adjustment in stops (default 0).
- `lava:headlight`: automatic camera-aligned lighting for an interactive scene
  with no supported lights (default true). Final rendering never adds this light.

The color AOV is filtered **scene-linear RGBA float32**, vertically adapted to
Hydra's bottom-left convention. Houdini applies the display transform. The CLI
uses the graph's tone-mapped output instead. The plugin supports mesh Rprims,
camera and sphere/rect/distant light Sprims, and color/depth render-buffer Bprims.
The depth AOV is positive camera-space linear depth; renderer metadata sets
`depthstyle` to `linear`. Solaris 20.5 requires both color and depth for this
compositing path; color-only Hydra tests do not validate viewport presentation.
Sphere lights are point lights; shaping cone parameters make them spot lights.
Rect lights are sampled uniformly by area; distant lights are directional.

## USD command line

With Houdini's Python environment initialized (so `from pxr import Usd` works):

```sh
python scripts/render_usd.py examples/studio.usda --camera /World/Camera \
  --mode final --width 640 --height 360 --samples 64 --output frame.ppm
```

The script resolves a USD frame, bakes world transforms, exports the supported
subset to JSON, then launches `target/release/lava`. `--frame` selects the time
code; `--lava` selects a binary. `--export scene.json` only exports and needs no
GPU. The native Rust CLI reads JSON; it does not claim to parse USD directly.
The Hydra delegate is the in-process Houdini viewport integration.

## Scene subset

See [examples/studio.json](examples/studio.json) and
[examples/studio.usda](examples/studio.usda).

- Perspective and orthographic cameras: camera-to-world transform, focal length,
  horizontal/vertical aperture and offsets, near/far planes; optional thin-lens
  depth of field (`f_stop`, `focus_distance`). Camera looks down local -Z with +Y
  up. JSON optics are in world units; optical ratios may use any consistent unit
  when DOF is disabled. USD attributes are converted from tenths of world units.
- Indexed simple polygon meshes, triangulated by ear clipping in centered double
  precision; unusable faces are omitted with a diagnostic rather than aborting the
  frame. Invalid indices and corrupt topology remain errors; authored shading
  normals are interpolated when present, with geometric normals as fallback and
  for ray offsets. Constant RGB base color is supported. Optional `material` fields select ideal
  metallic/dielectric transport (`metallic`, `transmission`, `ior`). JSON positions
  are world-space.
- Point, spot, rectangular area, and directional lights. Point/spot lighting uses
  inverse-square attenuation; area lights use area/cosine weighting. Intensities
  are prototype radiometric controls, not a complete USD Lux normalization model.
- Background, jittered camera samples, hard point/spot/distant shadows and soft
  area shadows, running-average accumulation, primitive-aware spatial filtering.

Not implemented: general MaterialX material graphs,
subdivision, volumes, instancing, motion blur,
camera shutter effects, lens distortion, arbitrary clipping planes, AOVs beyond
color/depth, render products / EXR, adaptive sampling, or denoising. Hydra collection
masking / render tags, non-default framing/data windows, mesh holes, and full USD
light normalization are not yet honored by the native adapter. The USD export
script does handle mesh holes and imageable purpose/visibility. Camera aspect is
controlled by the supplied apertures; automatic aperture conform is not included.
Constant UsdPreviewSurface and native OpenPBR terminals are supported. MaterialX
graph evaluation remains future work. Scripting of the render graph is planned.

## Verification

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo build
python scripts/smoke.py
PXR_PLUGINPATH_NAME="$PWD/build/resources" build/lava_hydra_smoke
PXR_PLUGINPATH_NAME="$PWD/build/resources" build/lava_usd_smoke examples/unlit.usda /World/Camera
```

The GPU smoke checks interactive/final numerical equivalence, light contribution,
empty scenes, asynchronous updates, and invalid input rejection. The C++ test
loads the actual Hydra plugin, syncs a mesh, camera and light, renders via the
Rust worker, and checks the linear color AOV. Use `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`
for Vulkan validation if the layer is installed.

For **CI testing only**, `LAVA_TEST_ALLOW_CPU_VULKAN=1` permits a CPU Vulkan ICD
that exposes KHR ray queries (e.g. a sufficiently recent lavapipe). This is a
driver implementation of the same Vulkan path, not the future renderer software
fallback, and it does not make the plugin advertise CPU support.

In the implementation environment: eight core tests and strict Clippy passed;
Vulkan smoke checks passed on Mesa llvmpipe with validation enabled. The release
Hydra integration test and a 17-sample render also passed on an NVIDIA RTX 2060
SUPER with Vulkan validation enabled. Houdini 20.5.332 hython discovered and loaded
the installed plugin using the real user houdini.env. Radeon/Arc coverage and
interactive Solaris UI behavior still require testing. See
[docs/architecture.md](docs/architecture.md) for extension points and constraints.

### Kitchen-set regression

`lava_usd_smoke` populates a stage through Houdini's USD imaging adapters and runs
normal Hdx tasks with indexed color buffers, rather than manually syncing prims.
It accepts `scene.usd [cameraPath] [preview.ppm]`; without a camera path it creates
a temporary camera framing the world bounds. It never modifies the input USD.
The unlit example checks the automatic interactive headlight.

The evaluated stage from the local kitchen HIP was tested on the RTX 2060 SUPER:
1,788 mesh prims, 536,903 prepared triangles, 23 omitted unusable faces, and a
nonconstant finite color AOV, with Vulkan validation enabled. This verifies the
missing-image regression, not material parity with Karma. Restart Houdini after
rebuilding/installing the delegate; loaded shared libraries cannot be safely
replaced in a running process. `LAVA_TRACE=1` enables camera/AOV/submission traces.

### Vulkan/OpenGL viewport interop (Linux)

The viewport path uses `VK_KHR_external_memory_fd` and
`VK_KHR_external_semaphore_fd` with `GL_EXT_memory_object_fd` and
`GL_EXT_semaphore_fd`. Capability checks include exportable storage-buffer memory,
exportable binary semaphores, GL extensions, and matching device UUIDs. Exported
allocations are device-local; host-visible staging is allocated only for an
explicit CPU readback. Unsupported combinations keep the CPU AOV path.

Each published GPU frame is immutable. Vulkan releases its packed color/depth
buffer to `VK_QUEUE_FAMILY_EXTERNAL` and signals a semaphore. GL imports the memory
and semaphore, waits on the GPU, and dispatches a compute shader that splits the
planes and flips rows into regular HgiGL color/depth textures. Solaris obtains
these via `HdRenderBuffer::GetResource`; no frame pixels are mapped to CPU on this
path. There is one GPU copy/compute dispatch, not direct image aliasing. GL signals a second external semaphore after its copy, permitting a later
explicit CPU request to acquire Vulkan ownership and copy to staging safely. GPU
fences retain imported resources until the GL copy finishes. The worker still waits for
a Vulkan submission fence before publishing metadata; fully asynchronous rings
and memory reuse are later latency optimizations.

`Resolve` and convergence checks never read image pixels. `Map` explicitly invokes
the readback fallback. Ready placeholder GL textures avoid initial empty-buffer
CPU updates while scene preparation is in progress. On success the console prints:

```text
Lava viewport: Vulkan/OpenGL interop active (color + depth, no image readback)
```

Interop verification (requires an X display and matching GL/Vulkan GPU):

```sh
LAVA_TEST_GL_INTEROP=1 PXR_PLUGINPATH_NAME="$PWD/build/resources" \
  build/lava_usd_smoke examples/unlit.usda /World/Camera
```

This test compares imported GL texture pixels to explicit test-only readback.
`liblava_houdini_smoke.so` additionally runs the actual HUSD imaging/compositor
interface inside hython, recording GPU texture callbacks versus CPU buffer
callbacks; it is not a screenshot or a substitute for live UI testing.

Houdini 20.5 automatically maps converged AOVs for its COP/slap-comp cache, even
when the viewport consumes GPU textures. Each explicit host snapshot requires a
cached staging readback and emits a warning to stdout. It does not switch viewport presentation
back to CPU uploads: the immutable GPU snapshot remains exportable until GL has
copied it. The Solaris integration test distinguishes GPU texture presentation
from this host-requested COP snapshot and reports the readback count explicitly.

With the installed plugin and Houdini environment initialized, the host-level
check is reproducible with:

```sh
hython scripts/test_solaris.py /path/to/evaluated-stage.usdc /cameras/camera1
```

For the local kitchen stage on the RTX 2060 SUPER, the compositor test reports
GPU color/depth callbacks, zero CPU presentation callbacks, and one automatic COP
cache readback. Vulkan validation and the GL/Vulkan pixel-equivalence test pass.
The HIP and source USD are not modified by this test.


The deterministic interactive worker GPU regression holds geometry builds to
verify continued old-scene sampling, superseded build disposal, camera/light
updates during a pending build, accumulation reset against clean images, zero
progressive image readbacks, and final-only publication:

```sh
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation cargo test --lib -- --ignored --nocapture --test-threads=1
```

To exercise progressive kitchen presentation at viewport resolution:

```sh
LAVA_TRACE=1 LAVA_TEST_WIDTH=3840 LAVA_TEST_HEIGHT=2160 LAVA_TEST_SAMPLES=64 \
  hython scripts/test_solaris.py /path/to/evaluated-stage.usdc /cameras/camera1
```

Trace lines report actual imported sample counts before convergence; placeholders
are not counted as imported images. CPU AOV caching remains a host-requested
exception and emits `Lava [WARNING]`.

The Hydra camera-motion regression repeatedly edits the USD camera while executing
Hydra tasks and consuming color/depth GL resources. It requires new GPU imports
while movement is still active and no CPU image readbacks during that interval:

```sh
PXR_PLUGINPATH_NAME="$PWD/build/resources" LAVA_TEST_GL_INTEROP=1 \
  LAVA_TEST_CAMERA_MOTION=1 VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  build/lava_usd_smoke /path/to/scene.usd /path/to/camera
```

The kitchen check reduced each camera update from a 14.5 MB scene snapshot to a
526-byte view delta. This is transport size evidence, not a guaranteed frame rate.


## Light BVH and wavefront path transport

`src/light_bvh.rs` is independent of the render graph. Its shared GPU light/node
buffers and `shaders/light_bvh.glsl` can be bound by any pass. A balanced spatial
hierarchy chooses one light using power and distance, with a positive probability
floor. The same parent-chain PDF is used when a BSDF path hits an area emitter.
Analytic rectangle intersections also provide emitter visibility and occlusion.

Shading uses next-event estimation and power-heuristic MIS between light sampling
and cosine-weighted diffuse BSDF sampling. Point, spot and distant lights are delta
sources, with no competing continuous BSDF PDF. Secondary paths support diffuse,
ideal metallic reflection and ideal dielectric Fresnel reflection/refraction,
including total internal reflection and the radiance eta-squared factor.

Settings in Houdini's Lava render settings:

| Setting | Default | Meaning |
|---|---:|---|
| `lava:diffuseDepth` | 3 | Maximum diffuse scattering events per path |
| `lava:reflectionDepth` | 4 | Maximum mirror/Fresnel reflection events |
| `lava:refractionDepth` | 8 | Maximum dielectric transmission events |
| `lava:rouletteStart` | 3 | Total scattering count at which roulette begins |

Depth limits are independent, each in 0..32; zero disables that lobe. Their sum
bounds the wavefront iterations. Emission/background reached after the last
allowed scattering event is still collected. Roulette survival is the clamped
maximum component of **cumulative throughput**; survivors are divided by their
survival probability. Set roulette start above the sum of limits to disable it.
The corresponding JSON settings are `diffuse_depth`, `reflection_depth`,
`refraction_depth`, `roulette_start` inside the scene's `settings` object. CLI
flags use hyphens, for example `--diffuse-depth 4 --roulette-start 3`.

Between bounces, compute subgroup ballots compact surviving paths into ping-pong
queues. One atomic reservation per subgroup packs active lanes contiguously.
A GPU-only preparation dispatch writes the next indirect workgroup count and
resets the destination counter. No CPU ray-count polling, ray downloads or image
readbacks are used by this loop. Compute subgroup BASIC/BALLOT support is queried
at device creation, with a clear error if absent. Queue storage is 128 bytes per
pixel across both queues, additional to render targets and acceleration memory.

Houdini materials lower constant `UsdPreviewSurface` inputs to OpenPBR, including
base color, metalness, roughness, IOR, emission, and stochastic opacity coverage.
Native `OpenPBRSurface`, `open_pbr_surface`, and
`ND_open_pbr_surface_surfaceshader` terminals use the typed native OpenPBR inputs.
Direct `UsdUVTexture` connections support color and scalar surface inputs. Other
connected graphs warn and use constant fallbacks. Choose Original or Compressed
(BC4/BC7) in Lava Render Settings > Textures; Virtual is inactive.
[Texture loading, caching and limitations](docs/textures.md).
[Material architecture and limitations](docs/materials.md) describe pass products,
cache behavior, supported lobes, and the future MaterialX frontend boundary.

Try the new showcase in Solaris: load `examples/openpbr.usda`, choose
`/World/Camera`, and select Lava. The standalone equivalent is:

```sh
cargo run --release -- render examples/openpbr.json --samples 128 --output openpbr.ppm
```

Try the matching diffuse/metal/glass scenes:

```sh
cargo run --release -- render examples/transport.json --samples 128 --output transport.ppm
# In Solaris, load examples/transport.usda and select /World/Camera.
```

GPU tests compare MIS and roulette means to numerical area-light quadrature, test
mixed delta/area light PDFs, independent lobe limits, ideal mirror/glass energy,
and read 32 bytes of explicit queue diagnostics to verify compaction. Image data
stays on the GPU during progressive presentation. Run all hardware tests with:

```sh
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation cargo test --lib -- --ignored --nocapture --test-threads=1
```

## Visibility and camera sampling

Visibility is selectable through `settings.visibility` (`"ray_query"` or
`"raster"`), the Hydra render property `lava:visibility`, or CLI:

```sh
cargo run --release -- render examples/openpbr.json --visibility raster --samples 128
cargo run --release -- render examples/openpbr.json --visibility ray-query --samples 128
```

Both routes support perspective/orthographic cameras, aperture offsets, and
thin-lens DOF. Set camera `f_stop > 0` and `focus_distance` in scene units.
Raster mode samples the pupil once per image sample and modifies the view and
projection matrix to preserve the focus plane. Interactive Halton prefixes remain
valid when the requested convergence target changes. Final Hammersley sets require
a fixed total sample count; changing it requires an accumulation reset.

The native renderer exposes **Visibility generator (ray_query or raster)**. In a
Solaris Render Settings prim, author a string `lava:visibility = "raster"`.
New controls generated by `scripts/expose_hip_settings.py` include a dropdown;
existing HIP files are preserved. [Implementation and validation](docs/visibility.md).

Native Solaris nodes: [Lava Render Settings and Lava Light Properties](docs/houdini-nodes.md).

Authored mesh normal import and shading: [normal handling](docs/normals.md).
