# Smooth polygon shadow terminator

Opaque, front-facing polygons with varying authored corner normals use a virtual
lighting point projected from their vertex tangent planes, following
[Hanika, Ray Tracing Gems II, chapter 4](https://jo.dreggn.org/home/2021_terminator.pdf).
Direct lighting and BSDF continuation share that point and its smooth hemisphere;
light selection, area PDFs, MIS and shadow traversal therefore use the same
lighting geometry. Actual visibility hits, UVs, texture footprints and material
evaluation still use the polygon surface.

Missing/constant normals, concave patches, interior hits and transmissive materials
retain their original behavior. No subdivision or geometry rebuild is needed.
This is a biased shading approximation: nearby contact shadows can shift, and
hard normal seams can remain visible. Occluders are still traced normally.

The GPU regression `smooth_polygon_lighting_point_fixes_terminator_without_flat_mesh_leaks`
covers a coarse convex patch, flat/constant-normal controls, a separate blocker,
and transmission, for legacy/OpenPBR and ray-query/raster visibility.
