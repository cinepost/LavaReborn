# Mesh shading normals

Lava imports authored USD mesh `normals` and `primvars:normals`. The primvar takes
precedence when both are authored. Constant, uniform (per face), vertex, varying
and face-varying interpolation are supported, including indexed primvars. Hard
edges retain distinct face-corner values through polygon triangulation.

Meshes marked as subdivision surfaces (including Houdini's Polygons as Subdivision
switches) still use authored normals on their **original polygon cage**. USD
24.03 hides those normals/descriptors in its standard mesh imaging path. Lava
recovers authored normals, interpolation and indices from the original USD
scene delegate, including when Hydra wraps it with scene-index emulation.
No subdivision evaluation, stage edits or global adapter replacement is involved.
Absent normals still use the geometric fallback.

The Hydra adapter and USD-to-JSON exporter bake normals to world space using the
inverse transpose of the object transform and normalize them. Left-handed USD
orientation and mirrored transforms retain the correct geometric orientation.
Malformed native normal data or singular normal transforms warn and fall back to
geometric normals. Missing normals do not trigger automatic smoothing.

The shading pass barycentrically interpolates and normalizes the three corner
normals for every primary and secondary surface hit. Both the legacy BSDF and
OpenPBR use this shading frame, in both ray-query and raster visibility modes.
Zero corner normals and cancelling interpolants use geometric fallback. Shading
normals are aligned to the geometric hemisphere; grazing configurations that
would change entering/exiting classification fall back to the geometric frame.
Geometric normals determine ray offsets and reject reflection/transmission paths
on inconsistent geometric sides, preventing below-surface light leaks.

The accumulation pass remains independent of normals/materials. Editing normals
invalidates the immutable geometry snapshot and resets accumulation when the
replacement is installed; interactive rendering can keep sampling the old scene
while it is prepared.

Standalone JSON meshes accept optional `normals` (world-space float3 array),
`normal_indices` (one normal-table index per polygon corner; omitted means
per-vertex), and `reverse_orientation` (default false). For constant or per-face
normals, supply the appropriate corner index for each polygon corner. Non-finite
or incorrectly indexed JSON normal tables are rejected. Empty arrays preserve
flat shading. The packed std430 triangle record is 160 bytes.

Verification: `scripts/test_normals.py build/normals.usda` under Houdini `hython`
checks all interpolation rates, indexed primvar precedence, transforms and JSON
lowering. `LAVA_TEST_NORMALS=1 build/lava_usd_smoke build/normals.usda /World/Camera`
checks native import and rendered irradiance; GPU tests cover both visibility
modes, OpenPBR/legacy shading, geometric fallback and light-leak rejection.

The normal regression generator also accepts a subdivision scheme, for example
`scripts/test_normals.py build/normals-catmull.usda catmullClark`. The native
regression checks matching irradiance with scheme `none`, `catmullClark`, and
`bilinear`; `LAVA_TEST_SUBD_NORMAL_EDITS=1` additionally tests live scheme toggles
and authored normal edits while asserting unchanged cage topology.
