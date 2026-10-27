# OpenPBR prototype validation — 2026-10-25

Built and atomically installed for local Houdini 20.5 at `/opt/hfs20.5`.
The existing `houdini.env` points to `build/install`; no environment changes were
required. Restart Houdini to load the new shared libraries.

- `cargo test --offline`: 14 CPU tests pass (3 subsystem + 11 integration).
- `cargo clippy --offline --all-targets -- -D warnings`: passes.
- Five hardware tests pass with Vulkan synchronization validation on the local
  NVIDIA RTX GPU. No Vulkan validation errors were reported.
- OpenPBR Lambert mean 1.3372463 versus legacy/quadrature ≈1.3353; roulette mean
  1.3383379. Roughness changes the metal result from 2.4010987 to 1.33681.
  Layered coat/fuzz/thin-film, transmission, emission and stochastic coverage
  checks pass. Material edits match a clean render with zero AS builder calls.
- Continuous-camera regression: 3,866 view updates, 104 GPU images during motion,
  no presentation image readbacks. Native USD/Hydra camera test also passed with
  220 edits and 88 GPU presentations during movement.
- Installed native Hydra showcase: OpenPBR terminals and colored regions pass;
  Vulkan/OpenGL shared color and depth match explicit diagnostic CPU maps with
  zero pixel error.
- Installed native HUSD/Solaris compositor at 960×540: convergence and color/depth
  callbacks pass; 41 GPU color and 41 GPU depth deliveries, zero CPU presentation
  uploads. Houdini requests one warned CPU AOV map for its cache path. This is not
  a claim that host-requested CPU maps perform no readbacks.
- C ABI smoke passes mode equivalence, lights, empty scene, latest revision and
  input validation. Houdini USD Python exports six native materials, four sets of
  UVs and 15,876 triangles; the exported scene validates.

The installed Slang compiler rejects the wavefront storage-buffer atomic in the
full surface product; `LAVA_SHADER_COMPILER=slangc` remains experimental. Production
builds and all GPU checks above use the GLSL/glslang backend. MaterialX graphs,
texture evaluation, volume subsurface/absorption/scattering and dispersion remain
explicitly outside this iteration; see [materials.md](materials.md).

Showcase: `examples/openpbr.usda`, camera `/World/Camera`, renderer Lava Reborn.
Standalone equivalent: `examples/openpbr.json`. Preview: `build/openpbr.png`.
