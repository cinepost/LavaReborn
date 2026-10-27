# Textures

In Solaris, use **Lava Render Settings > Textures > Texture Mode**. Select the
node's render-settings prim in the viewport. The standalone equivalent is
`--textures original|compressed|virtual`, or `settings.textures` in scene JSON.

- **Original** decodes PNG, JPEG, EXR or HDR without lossy compression and uploads
  floating-point images (R32 for grayscale, RGBA32 for color). HDR values are preserved.
- **Compressed** uses BC4 for grayscale and BC7 for color, loading cached blocks
  on subsequent runs. Both are lossy. Grayscale data with negative values uses
  BC4 SNORM; nonnegative data uses BC4 UNORM. Signed HDR data uses symmetric
  range scaling, preserving zero; unsigned HDR uses positive range scaling.
  Color HDR uses per-channel affine encoding into BC7 UNORM. Shader range
  reconstruction avoids clipping, with lower precision than float originals.
- **Virtual** is inactive: no texture files are loaded or converted, and material
  constants are used. No virtual texture format is assumed yet.

Both active modes generate every mip level down to 1×1, including odd-sized
images, and use trilinear sampling. Color inputs from PNG/JPEG default to sRGB
and are decoded to linear RGB before filtering. Scalar inputs and EXR/HDR default
to raw. Explicit `srgb` or `raw` overrides are supported; alpha stays linear.

## Material bindings

JSON material example (paths resolve relative to the scene JSON):

```json
{
  "textures": {
    "base_color": {
      "file": "textures/wood.png",
      "color_space": "auto",
      "compression": { "quality": "high" },
      "channel": "rgb",
      "multiply": false,
      "wrap_s": "repeat",
      "wrap_t": "repeat",
      "uv_scale": [1, 1],
      "uv_offset": [0, 0],
      "scale": [1, 1, 1, 1],
      "bias": [0, 0, 0, 0]
    }
  }
}
```

Bindings target implemented OpenPBR float/color inputs, including opacity.
Channels are `rgb`, `r`, `g`, `b`, `a`, or `luminance`; wrapping supports `repeat`, `clamp`,
`mirror`, and `black`. Texture values replace the corresponding constant after
scale/bias. Set `multiply: true` to multiply that result by the material constant.
Missing or pending interactive bindings use the constant fallback.

Hydra imports direct `UsdUVTexture` connections into supported OpenPBR and
UsdPreviewSurface inputs, using mesh `st` coordinates, source color space,
channel, scale/bias and wrapping. A custom string input
`inputs:lava:compressionQuality` on the texture shader selects `fast`, `balanced`
(default), or `high` quality per texture. Full MaterialX graph execution, UDIM, UV-transform graphs and multiple UV sets
remain unsupported. JSON
bindings support the UV scale/offset shown above.

## Scheduling and cache

File reads, decoding, mip generation and BC4/BC7 conversion run on a bounded Rayon
pool. Independent files and compression blocks/bands can run in parallel. Override the
worker count with `LAVA_TEXTURE_WORKERS` (1–32; default up to 8).
A dedicated upload worker prepares device-local images with at most two uploads
in flight. GPU submissions share the renderer's queue. Interactive rendering
continues with constants or the previous resident texture until replacement is
ready, then binds it and resets accumulation. Final rendering waits for textures.

Source metadata is checked every 250 ms during interactive rendering. Cache keys
include source path, size, modification/change timestamps, SHA-256 content hash,
color-space handling, compression quality and cache version. Source or settings
changes regenerate compressed textures. Cache entries contain the full pyramid and checksums;
corrupt entries are rebuilt. Writes publish atomically. Old content-addressed
entries are not automatically pruned.

Version 2 `.ltex` BC4/BC7 files live in `LAVA_TEXTURE_CACHE`, or
`$XDG_CACHE_HOME/lava-reborn/textures`, falling back to
`$HOME/.cache/lava-reborn/textures`. Original mode does not use this disk cache. Earlier `.lbc7` cache entries are
ignored and regenerated on demand; old entries are not deleted.

Texture upload is a one-way host-to-GPU transfer; it introduces no rendered-image
readback. Existing Vulkan/OpenGL viewport interop remains in use. An unavoidable
host image fallback emits a warning.

## Current limits

Textures are fully resident, with at most 127 unique texture requests per scene
plus a white fallback descriptor. Startup checks sampled-image descriptor limits
and indexing features; R32/BC4/BC7 uploads check device format support. Mip selection uses
a conservative isotropic camera footprint; secondary rays do not yet carry full
ray differentials or ray cones. Accumulation has no material or texture dependency.

CPU tests cover formats, mip filtering, cache invalidation/corruption and async
reload. GPU tests cover original/BC4/BC7 HDR and mip sampling, opacity in both
visibility paths, inactive virtual mode, and interactive reload without image
readback. Native Hydra tests exercise PreviewSurface/OpenPBR UV bindings and
OpenGL presentation; Solaris tests exercise the installed delegate and HDA.

## Principled Shader and shading frames

Houdini 20.5's auto-preview always exports roughness/metallic texture output `r`,
even when Principled selects another channel. Lava recovers the original authored
Principled inputs through its USD VEX terminal. Base color textures multiply
`basecolor`; roughness textures multiply `rough`; metallic textures multiply
`metallic`. Scalar channel selection follows Luminance/Red/Green/Blue. Luminance
matches Houdini VEX weights (0.299, 0.587, 0.114). Multipliers are folded into the
texture scale once, without multiplying the generated preview scale twice.
Other USD materials retain standard UsdUVTexture scale/bias replacement semantics.

Principled's base **Bump & Normals** controls support tangent-space normal maps,
normal strength and X/Y flips, or height bump maps with bump scale. These alter
the shading frame; triangle geometry and ray offsets still use geometric normals.
The perturbed normal is constrained to the original geometric side and view
hemisphere to prevent light leaks. Tangent handedness follows the triangle UVs.
Height derivatives use filtered central differences and the surface UV metric.
Object/world-space and coat-specific normal maps remain unsupported.

For JSON, bind `normal` or `bump` in `material.textures`. Both are raw data by
default. `normal` uses unsigned RGB (decoded as `2*RGB-1`); set `signed: true` for
already decoded vectors. `strength` defaults to 1 and scales tangent normal XY
or the bump height derivatives. Bump defaults to the red channel; choose
`channel: "luminance"` when appropriate. Normal/bump use the same async loader,
compressed cache and mip pyramids as other textures. Accumulation stays material-agnostic.

## Grayscale storage and sign detection

Single-channel 8/16-bit PNG, grayscale JPEG and single-channel float/half EXR
(including Y or an arbitrary scalar channel name) are supported. EXR data/display
window offsets are respected. RGB/RGBA assets with exactly equal RGB values and
fully opaque alpha are also reduced losslessly to grayscale. Nonopaque alpha
keeps RGBA storage. Grayscale image views sample as `(R,R,R,1)`, so existing color,
scalar, channel-selector and luminance bindings continue to work.

BC4 stores 8 bytes per 4×4 block, half BC7's block size. Each mip has its own
blocks, including 1×1 and odd dimensions. Source minima determine signedness
before mip generation or user scale/bias. Detection runs on decoded pixels in the
async CPU loader, alongside range analysis; the CPU encoder already needs these
pixels, so no additional GPU upload, dispatch, or readback is needed. BC4 blocks
encode in parallel on the loader pool. Fast quality uses min/max endpoints;
Balanced/High additionally search nearby endpoints for lower block error.
