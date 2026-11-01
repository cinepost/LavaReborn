# Solaris nodes

The build installs two native Houdini 20.5 digital assets into the existing Lava
`HOUDINI_PATH` under `houdini/otls`. Restart Houdini after installing the updated
delegate. In `/stage`, press Tab and search for **Lava Render Settings** or
**Lava Light Properties** (Lava submenu).

Connect Lava Render Settings after your scene. Its layout follows Karma's
settings-primitive header, camera/resolution controls, and grouped tabs:

- Sampling: interactive/final mode, ray-query/raster visibility, samples, headlight.
- Adaptive Sampling, Relative Error Threshold and Minimum Samples are in the
  Sampling tab. Samples is the maximum budget; see [adaptive sampling](adaptive-sampling.md).
- Textures: Original, Compressed (BC4/BC7), or Virtual (inactive); see [texture settings](textures.md).
- Limits: diffuse, reflection, refraction limits, Russian roulette start depth,
  and **Reduce Analytic Light Caustics** (off by default).
- Image: **Sample Count Heat Map AOV** (off by default), plus display exposure in stops.
  See [sample count diagnostic](sample-count-aov.md) for the color scale and
  viewport selection. Floating-point exposure edits reprocess the
  existing filtered image on the GPU without clearing history or restarting the
  sample counter, including after convergence.

- Advanced: **Geometry Upload Chunk Size**, with 16, 32 and 64 MiB options
  (default 16 MiB). Changes apply to subsequent geometry uploads and preserve
  accumulation. Larger chunks favor scene activation time at some cost to frame
  cadence during uploads. USD uses `lava:uploadChunkMiB`; JSON scenes use
  `settings.upload_chunk_mib`.

The node authors `/Render/Lava` by default. Select that prim in the Solaris
viewport's Render Settings Prim selector if viewport overrides are active.
Camera optics, focus distance and f-stop remain on the standard Camera LOP.
Only implemented controls and the sample-count diagnostic output are exposed.

**Reduce Analytic Light Caustics** is an opt-in biased firefly control for both
interactive and final rendering. It caps the largest RGB component of reflected
analytic-light sample multipliers at 16, preserving their color ratios, before
applying light intensity. It covers point, spot and directional next-event
samples, rectangular-light next-event samples, and analytic emitter hits after
reflection. Diffuse contributions, transmission, environment and material
emission are not clamped. It can dim sharp highlights and does not guarantee
removal of every firefly. Toggling it resets accumulation without rebuilding
geometry. USD uses `lava:reduceAnalyticCaustics`; JSON scenes use
`settings.reduce_analytic_caustics`.

Connect Lava Light Properties after light creation. Its Primitives field accepts
Solaris selection patterns. It exposes exactly two renderer controls:

Standard Solaris light **Angle**, **Diffuse** and **Specular** controls are also
supported; see [light inputs](light-inputs.md) for their behavior.

| Control | Default | Applies to |
| --- | --- | --- |
| Single Sided | On | Area lights |
| Render Light Geometry | Off | Area lights and meshes with USD MeshLightAPI |

Single Sided restricts emission and analytic intersections to the front side;
disabling it enables both sides with unchanged radiance per side. Render Light
Geometry controls **camera visibility**, in both raster and ray-query modes.
Lighting, BSDF emitter hits and shadowing remain active when it is off, so MIS
continues to use the same emitter PDFs. Ordinary emissive materials are not
hidden unless the mesh is explicitly marked as a geometry light.

Lava currently implements rectangular area sources. The node can author these
properties on disk/cylinder schemas for future support, but does not add renderer
support for those shapes. Mesh lights currently use emissive surface shading and
BSDF hits; mesh-light next-event sampling is not implemented.

Authored attributes are constant boolean primvars `primvars:lava:singleSided`,
`primvars:lava:renderLightGeometry` and (for selected mesh lights)
`primvars:lava:geometryLight`. Non-applicable primitives are not edited.

Build assets with the matching Houdini `hython` using
`scripts/build_houdini_assets.py build/houdini/otls`. CMake builds and installs them
alongside the delegate. `scripts/test_houdini_assets.py <otls-directory>` verifies
the interfaces, default values, USD opinions, edits and saved-HIP reload.
