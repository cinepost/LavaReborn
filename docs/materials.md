# Material subsystem

`src/material` owns the native material IR, validation, GPU record packing,
immutable surface lookup resources, shader compilation/cache, and Vulkan driver
pipeline cache. `scripts/generate_material.py` generates the Rust schema, std430
record definition, reference-input lowering and Hydra parameter import field list
from one field list. The generated `openpbr_defs.usda` plus registry marker are
installed in plugin resources and registered once before USD population; otherwise
Houdini 20.5 drops unknown OpenPBR terminals from Hydra material networks. Regenerate with that script followed by `cargo fmt`.
`generate_openpbr_luts.py` packs the unmodified upstream reference tables.

The frontend accepts native OpenPBR inputs and direct image texture bindings. A future MaterialX
frontend can lower graph outputs to the same IR and generate evaluation code
behind the existing product interfaces. It should use MaterialXGenGlsl or Slang
without embedding a raster lighting loop in a path tracing BSDF. MaterialX graph
execution is not implemented by accepting an OpenPBR terminal ID. Direct
`UsdUVTexture` connections are supported separately; see [textures](textures.md).

## GPU products and ownership

- Visibility uses `materials/visibility.glsl`: material IDs and stochastic coverage,
  with no BSDF or lookup-table code. Triangles store vertex/face-corner UVs. Hits
  carry primitive/barycentrics/depth plus opaque material-ID and interpolated-UV
  metadata. The visibility buffer is 32 bytes per pixel.
- Surface shading uses `materials/surface.glsl`: prepare, emission, lobe evaluation,
  full-mixture PDF and sampling. Only this product includes the reference OpenPBR
  implementation and device-resident lookup tables. The pass interpolates UVs and
  derives an anisotropy tangent from the UV gradients.
- Accumulation has **no material product**. Its shared frame header contains no
  geometry, material, lighting or acceleration-structure declarations. It consumes
  radiance and primitive guide data, keeps history and filters as before.

Material parameters are table data, not shader specialization constants. Each mesh
has a stable table index for its current geometry version, including bit-preserving
32-bit indices in the triangle record. Shading-pass ray queues, depth counters,
roulette and subgroup compaction remain independent of material implementation.
Camera, secondary and shadow traversal share one stochastic coverage definition.

OpenPBR parameter edits upload a replacement GPU table and reset accumulation at a
sample boundary, without rebuilding BLAS/TLAS. Geometry/UV edits continue to use the
existing asynchronous immutable geometry-version path. Final rendering prepares the
scene synchronously. All material tables and 1.08 MiB of reference lookup data live
in device memory after a one-way upload; viewport color/depth interop is unchanged.

## Model and integrator

The model uses Adobe's Apache-2.0 OpenPBR 1.1.1 implementation pinned under
`third_party/openpbr`, with matching evaluation, PDF and sampling functions.
Supported surface inputs include base diffuse/roughness/metalness, rough anisotropic
specular reflection and transmission, coat/color/roughness/darkening, fuzz, thin
film, emission, thin-wall geometry and textured opacity. Colors are linear RGB.
Thin-film thickness uses the reference's micrometers. Texture loading converts
sRGB color inputs to linear RGB. Tangent-space normal and height bump textures
perturb the shading frame; see [texture controls](textures.md).

Explicitly rejected for this surface iteration: nonzero subsurface weight,
transmission absorption depth, volume scattering and spectral dispersion. Nested
medium stacks, separate per-layer normal maps and tangent rotations are future extensions.
Legacy JSON `metallic/transmission/ior` materials retain the old ideal BSDF.
Native PreviewSurface opacity now means coverage; use an OpenPBR transmission
weight for glass. Mesh emission is visible to BSDF paths but is not inserted into
Light BVH NEE; analytic scene lights retain MIS/NEE sampling.

The reference evaluation already includes cosine. NEE uses it without an extra
cosine and compares the light PDF against the same full-mixture BSDF PDF used by
continuation sampling. Independent depth caps mask evaluated lobes; continuous
samples select a surviving diffuse/reflection depth class with probability
compensation while retaining the proposal PDF. Delta events use reference sample
weights and have no continuous MIS competitor. Roulette uses cumulative throughput.

## Compilation and caches

The production backend is portable GLSL compiled with `glslangValidator` to Vulkan
1.2 SPIR-V. `spirv-val` validates both generated and cached blobs. Content keys use
SHA-256 over the product name, ABI/target, compiler backend/version and fully
preprocessed source, including all transitive includes. Temporary files have unique
names and publish atomically; invalid/truncated blobs trigger recompilation.
Cache defaults to `target/material-cache`; override with `LAVA_SHADER_CACHE`.
Material parameter changes do not invalidate shader binaries.

`LAVA_SHADER_COMPILER=slangc` is an experimental compiler hook. The installed Slang
can compile the visibility product but currently rejects a storage-buffer atomic
in the complete wavefront surface pass. It is not the supported build path. The
portable reference and generated interfaces leave room for a native Slang backend.

Vulkan `VkPipelineCache` additionally stores driver-compiled pipelines under
`$XDG_CACHE_HOME/lava-reborn` (or `$HOME/.cache/lava-reborn`). Override the directory
with `LAVA_PIPELINE_CACHE`. Vendor, device, driver version and pipeline-cache UUID
separate driver blobs; Vulkan compatibility headers are checked before use.
Unreadable/incompatible blobs start empty. A cache write failure warns and does not
prevent rendering. Cold OpenPBR driver compilation may still take tens of seconds;
subsequent renderer creation uses the persisted cache.

## Verification

CPU tests cover schema validation, product separation, indexed UV validation and
cache reuse/invalidation/corruption. Ignored RTX tests cover Lambert-limit energy,
roulette, roughness/anisotropy, layered lobes, glass, emission/cutout energy, material
edits without AS builds, reset agreement with a clean render, and camera updates
with GPU-only presentation. `examples/openpbr.json` and `openpbr.usda` provide a
coat/metal/glass/fuzz showcase. Native Hydra tests must enable scene materials.
