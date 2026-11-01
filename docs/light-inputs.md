# Light response and distant-light angle

Use the standard Solaris light controls for `inputs:diffuse`, `inputs:specular`
and, on distant lights, `inputs:angle`. No additional Lava node is required.

Diffuse and specular multipliers default to 1, accept nonnegative values above
1, and independently scale the corresponding direct-light lobes. They also
apply when secondary rays reach area lights, finite distant lights, or emissive
meshes marked as geometry lights. Transmission emitter hits use the specular
multiplier. Camera-visible emission is unaffected. These are artistic response
controls and do not preserve energy when changed independently.

Distant-light angle is the full angular diameter in degrees. Zero produces
parallel rays and hard shadows; positive angles sample a cone for soft shadows
and finite reflections, with MIS between light and BSDF sampling. USD's default
is 0.53 degrees. Lava accepts angles from 0 up to, but excluding, 360 degrees.
Extremely small cones whose `1-cos(angular radius)` is at most `1e-12` use the
parallel-ray path for numerical stability.

Lava preserves its existing directional intensity as normal-facing irradiance
when the angle changes. This does not implement USD's optional `inputs:normalize`
control. Legacy JSON directional lights with omitted `angle` retain zero angle;
JSON lights use `diffuse` and `specular` fields with defaults of 1.

Mesh lights read the same response inputs through USD MeshLightAPI. Live input
changes update the material table and reset accumulation without rebuilding
acceleration structures. Mesh lights continue to use emissive surface shading
and BSDF hits; mesh-light next-event sampling is not implemented.

Tests cover all supported analytic light types, independent response channels,
finite-sun energy and penumbrae, and mesh emission. The native Solaris smoke
fixtures animate response values through 0, 1 and 2 to verify live updates.
