use anyhow::{ensure, Result};
use glam::{DVec2, DVec3, Mat4, Vec3};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    /// Column-major camera-to-world, looking down local -Z, local +Y up.
    pub transform: [f32; 16],
    pub focal_length: f32,
    pub horizontal_aperture: f32,
    pub vertical_aperture: f32,
    #[serde(default)]
    pub aperture_offset: [f32; 2],
    #[serde(default)]
    pub f_stop: f32,
    #[serde(default = "default_focus")]
    pub focus_distance: f32,
    pub near: f32,
    pub far: f32,
    #[serde(default)]
    pub orthographic: bool,
}
pub use crate::material::Material;
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mesh {
    pub points: Vec<[f32; 3]>,
    pub face_counts: Vec<u32>,
    pub indices: Vec<u32>,
    pub color: [f32; 3],
    #[serde(default)]
    pub material: Material,
    #[serde(default)]
    pub texcoords: Vec<[f32; 2]>,
    /// Optional face-corner indices into texcoords. Otherwise vertex interpolation.
    #[serde(default)]
    pub texcoord_indices: Vec<u32>,
    /// Optional world-space shading normals; empty indices mean vertex interpolation.
    #[serde(default)]
    pub normals: Vec<[f32; 3]>,
    /// Face-corner indices, permitting hard edges and constant/uniform normals.
    #[serde(default)]
    pub normal_indices: Vec<u32>,
    /// Correct geometric orientation after left-handed USD winding / mirrored transforms.
    #[serde(default)]
    pub reverse_orientation: bool,
    /// USD geometry light: only primary visibility is controlled here.
    #[serde(default)]
    pub geometry_light: bool,
    #[serde(default)]
    pub render_light_geometry: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Light {
    Point {
        position: [f32; 3],
        color: [f32; 3],
        intensity: f32,
    },
    Spot {
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        cone_degrees: f32,
        softness: f32,
    },
    Area {
        position: [f32; 3],
        u: [f32; 3],
        v: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        #[serde(default = "default_true")]
        single_sided: bool,
        #[serde(default)]
        render_light_geometry: bool,
    },
    Directional {
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    #[serde(default)]
    pub curves: Vec<crate::curves::Curves>,
    #[serde(default)]
    pub settings: crate::integrator::IntegratorSettings,
    pub camera: Camera,
    pub meshes: Vec<Mesh>,
    pub lights: Vec<Light>,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Triangle {
    pub a: [f32; 4],
    pub b: [f32; 4],
    pub c: [f32; 4],
    pub color: [f32; 4],
    pub material: [f32; 4], // legacy parameters; w is material table index
    pub uv01: [f32; 4],
    pub uv2: [f32; 4],
    pub normal_a: [f32; 4], // w: geometric orientation sign
    pub normal_b: [f32; 4],
    pub normal_c: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    pub position_kind: [f32; 4],
    pub direction_cone: [f32; 4],
    pub color_power: [f32; 4],
    pub u_soft: [f32; 4],
    pub v: [f32; 4],
    pub bvh: [u32; 4],
}
fn default_true() -> bool {
    true
}
fn default_focus() -> f32 {
    5.
}
fn v4(v: [f32; 3], w: f32) -> [f32; 4] {
    [v[0], v[1], v[2], w]
}
#[derive(Debug, Default)]
pub struct GeometryDiagnostics {
    pub skipped_faces: usize,
    pub skipped_triangles: usize,
}
impl Scene {
    pub fn validate_parameters(&self) -> Result<()> {
        self.settings.validate()?;
        let c = &self.camera;
        ensure!(
            c.transform.iter().all(|v| v.is_finite())
                && Mat4::from_cols_array(&c.transform).determinant().abs() > 1e-8,
            "invalid camera transform"
        );
        ensure!(
            [
                c.focal_length,
                c.horizontal_aperture,
                c.vertical_aperture,
                c.near,
                c.far
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0.)
                && c.far > c.near,
            "invalid camera optics/clipping"
        );
        ensure!(
            c.aperture_offset.iter().all(|v| v.is_finite()),
            "invalid aperture offset"
        );
        ensure!(
            c.f_stop.is_finite()
                && c.f_stop >= 0.
                && c.focus_distance.is_finite()
                && c.focus_distance > 0.,
            "invalid depth of field"
        );
        for l in self.gpu_lights() {
            ensure!(
                bytemuck::cast_slice::<GpuLight, f32>(&[l])
                    .iter()
                    .all(|v| v.is_finite()),
                "invalid light"
            );
            ensure!(
                l.color_power.iter().all(|v| *v >= 0.),
                "negative light color or intensity"
            );
            let k = l.position_kind[3] as u32;
            if k == 1 || k == 3 {
                ensure!(
                    Vec3::new(
                        l.direction_cone[0],
                        l.direction_cone[1],
                        l.direction_cone[2]
                    )
                    .length_squared()
                        > 1e-12,
                    "zero light direction"
                );
            }
            if k == 1 {
                ensure!(
                    l.direction_cone[3] > 0.
                        && l.direction_cone[3] < 90.
                        && (0. ..=1.).contains(&l.u_soft[3]),
                    "invalid spotlight cone"
                );
            }
            if k == 2 {
                ensure!(
                    Vec3::from_slice(&l.u_soft[..3])
                        .cross(Vec3::from_slice(&l.v[..3]))
                        .length_squared()
                        > 1e-12,
                    "zero area light"
                );
            }
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<Vec<Triangle>> {
        self.prepare_geometry().map(|(triangles, _)| triangles)
    }
    /// Structural corruption is fatal; zero-area / untriangulatable faces are omitted.
    pub fn prepare_geometry(&self) -> Result<(Vec<Triangle>, GeometryDiagnostics)> {
        self.validate_parameters()?;
        let mut triangles = Vec::new();
        let mut diagnostics = GeometryDiagnostics::default();
        for (material_id, mesh) in self.meshes.iter().enumerate() {
            mesh.material.validate()?;
            ensure!(
                (0. ..=1.).contains(&mesh.material.metallic)
                    && (0. ..=1.).contains(&mesh.material.transmission)
                    && mesh.material.ior.is_finite()
                    && (1. ..=4.).contains(&mesh.material.ior),
                "invalid material"
            );
            ensure!(
                mesh.points
                    .iter()
                    .flatten()
                    .chain(mesh.color.iter())
                    .all(|v| v.is_finite()),
                "non-finite mesh"
            );
            ensure!(mesh.color.iter().all(|v| *v >= 0.), "negative color");
            ensure!(
                mesh.face_counts.iter().map(|&n| n as u64).sum::<u64>()
                    == mesh.indices.len() as u64,
                "face counts do not match indices"
            );
            ensure!(
                mesh.indices
                    .iter()
                    .all(|&i| (i as usize) < mesh.points.len()),
                "mesh index out of bounds"
            );
            ensure!(
                mesh.texcoords.iter().flatten().all(|x| x.is_finite()),
                "non-finite texture coordinates"
            );
            ensure!(
                mesh.texcoords.is_empty()
                    || if mesh.texcoord_indices.is_empty() {
                        mesh.texcoords.len() == mesh.points.len()
                    } else {
                        mesh.texcoord_indices.len() == mesh.indices.len()
                            && mesh
                                .texcoord_indices
                                .iter()
                                .all(|&i| (i as usize) < mesh.texcoords.len())
                    },
                "invalid texture coordinate indexing"
            );
            ensure!(
                mesh.normals.iter().flatten().all(|x| x.is_finite()),
                "non-finite shading normals"
            );
            ensure!(
                if mesh.normals.is_empty() {
                    mesh.normal_indices.is_empty()
                } else if mesh.normal_indices.is_empty() {
                    mesh.normals.len() == mesh.points.len()
                } else {
                    mesh.normal_indices.len() == mesh.indices.len()
                        && mesh
                            .normal_indices
                            .iter()
                            .all(|&i| (i as usize) < mesh.normals.len())
                },
                "invalid normal indexing"
            );
            let mut start = 0;
            for &count in &mesh.face_counts {
                let face = &mesh.indices[start..start + count as usize];
                let face_start = start;
                start += count as usize;
                let Ok(indices) = triangulate(&mesh.points, face) else {
                    diagnostics.skipped_faces += 1;
                    continue;
                };
                for [ia, ib, ic] in indices {
                    let a = mesh.points[ia as usize];
                    let b = mesh.points[ib as usize];
                    let c = mesh.points[ic as usize];
                    if (DVec3::from_array(b.map(f64::from)) - DVec3::from_array(a.map(f64::from)))
                        .cross(
                            DVec3::from_array(c.map(f64::from))
                                - DVec3::from_array(a.map(f64::from)),
                        )
                        .length_squared()
                        == 0.
                    {
                        diagnostics.skipped_triangles += 1;
                        continue;
                    }
                    let uv = |vertex: u32| {
                        if mesh.texcoords.is_empty() {
                            return [0.; 2];
                        }
                        let index = if mesh.texcoord_indices.is_empty() {
                            vertex as usize
                        } else {
                            mesh.texcoord_indices
                                [face_start + face.iter().position(|&i| i == vertex).unwrap()]
                                as usize
                        };
                        mesh.texcoords[index]
                    };
                    let normal = |vertex: u32| {
                        if mesh.normals.is_empty() {
                            return [0.; 3];
                        }
                        let index = if mesh.normal_indices.is_empty() {
                            vertex as usize
                        } else {
                            mesh.normal_indices
                                [face_start + face.iter().position(|&i| i == vertex).unwrap()]
                                as usize
                        };
                        DVec3::from_array(mesh.normals[index].map(f64::from))
                            .try_normalize()
                            .unwrap_or(DVec3::ZERO)
                            .to_array()
                            .map(|x| x as f32)
                    };
                    let (ua, ub, uc) = (uv(ia), uv(ib), uv(ic));
                    triangles.push(Triangle {
                        normal_a: v4(normal(ia), if mesh.reverse_orientation { -1. } else { 1. }),
                        normal_b: v4(normal(ib), 0.),
                        normal_c: v4(normal(ic), 0.),
                        uv01: [ua[0], ua[1], ub[0], ub[1]],
                        uv2: [
                            uc[0],
                            uc[1],
                            if mesh.geometry_light && !mesh.render_light_geometry {
                                1.
                            } else {
                                0.
                            },
                            0.,
                        ],
                        a: v4(a, 0.),
                        b: v4(b, 0.),
                        c: v4(c, 0.),
                        color: v4(mesh.color, 1.),
                        material: [
                            mesh.material.metallic,
                            mesh.material.transmission,
                            mesh.material.ior,
                            f32::from_bits(material_id as u32),
                        ],
                    });
                }
            }
        }
        for (index, curves) in self.curves.iter().enumerate() {
            triangles.extend(curves.segments(self.meshes.len() + index)?);
        }
        ensure!(
            triangles.len() <= u32::MAX as usize / 3,
            "too many triangles"
        );

        Ok((triangles, diagnostics))
    }
    pub fn gpu_lights(&self) -> Vec<GpuLight> {
        self.lights
            .iter()
            .map(|l| {
                let mut g = GpuLight {
                    position_kind: [0.; 4],
                    direction_cone: [0.; 4],
                    color_power: [0.; 4],
                    u_soft: [0.; 4],
                    v: [0.; 4],
                    bvh: [0; 4],
                };
                match *l {
                    Light::Point {
                        position,
                        color,
                        intensity,
                    } => {
                        g.position_kind = v4(position, 0.);
                        g.color_power = v4(color, intensity);
                    }
                    Light::Spot {
                        position,
                        direction,
                        color,
                        intensity,
                        cone_degrees,
                        softness,
                    } => {
                        g.position_kind = v4(position, 1.);
                        g.direction_cone = v4(direction, cone_degrees);
                        g.color_power = v4(color, intensity);
                        g.u_soft[3] = softness;
                    }
                    Light::Area {
                        position,
                        u,
                        v,
                        color,
                        intensity,
                        single_sided,
                        render_light_geometry,
                    } => {
                        g.bvh[1] =
                            u32::from(single_sided) | (u32::from(render_light_geometry) << 1);
                        g.position_kind = v4(position, 2.);
                        g.color_power = v4(color, intensity);
                        g.u_soft = v4(u, 0.);
                        g.v = v4(v, 0.);
                    }
                    Light::Directional {
                        direction,
                        color,
                        intensity,
                    } => {
                        g.position_kind[3] = 3.;
                        g.direction_cone = v4(direction, 0.);
                        g.color_power = v4(color, intensity);
                    }
                }
                g
            })
            .collect()
    }
}

/// Ear clipping in the dominant plane; preserves winding for concave simple polygons.
fn triangulate(points: &[[f32; 3]], face: &[u32]) -> Result<Vec<[u32; 3]>> {
    ensure!(face.len() >= 3, "polygon has fewer than three vertices");
    let origin = DVec3::from_array(points[face[0] as usize].map(f64::from));
    let local: Vec<_> = face
        .iter()
        .map(|&i| DVec3::from_array(points[i as usize].map(f64::from)) - origin)
        .collect();
    let mut normal = DVec3::ZERO;
    for i in 0..face.len() {
        let a = local[i];
        let b = local[(i + 1) % face.len()];
        normal += (a - b).cross(a + b);
    }
    ensure!(normal.length_squared() > 0., "degenerate polygon");
    let axis = normal.abs().max_element();
    let project = |p: DVec3| {
        if normal.x.abs() == axis {
            DVec2::new(p[1], p[2])
        } else if normal.y.abs() == axis {
            DVec2::new(p[0], p[2])
        } else {
            DVec2::new(p[0], p[1])
        }
    };
    let projected: Vec<_> = local.into_iter().map(project).collect();
    let scale = projected
        .iter()
        .map(|p| p.length_squared())
        .fold(0.0f64, f64::max);
    let epsilon = scale * 1e-12;
    let area: f64 = (0..face.len())
        .map(|i| projected[i].perp_dot(projected[(i + 1) % face.len()]))
        .sum();
    ensure!(area.abs() > epsilon, "degenerate projected polygon");
    let sign = area.signum();
    let mut ring: Vec<_> = (0..face.len()).collect();
    let mut result = Vec::new();
    while ring.len() > 3 {
        let mut ear = None;
        for i in 0..ring.len() {
            let a = ring[(i + ring.len() - 1) % ring.len()];
            let b = ring[i];
            let c = ring[(i + 1) % ring.len()];
            let (pa, pb, pc) = (projected[a], projected[b], projected[c]);
            if sign * (pb - pa).perp_dot(pc - pb) <= epsilon {
                continue;
            }
            let contains = ring.iter().any(|&j| {
                j != a && j != b && j != c && {
                    let p = projected[j];
                    sign * (pb - pa).perp_dot(p - pa) >= -epsilon
                        && sign * (pc - pb).perp_dot(p - pb) >= -epsilon
                        && sign * (pa - pc).perp_dot(p - pc) >= -epsilon
                }
            });
            if !contains {
                ear = Some((i, [face[a], face[b], face[c]]));
                break;
            }
        }
        let (index, triangle) = ear.ok_or_else(|| {
            anyhow::anyhow!("cannot triangulate polygon: non-simple or degenerate boundary")
        })?;
        result.push(triangle);
        ring.remove(index);
    }
    result.push([face[ring[0]], face[ring[1]], face[ring[2]]]);
    Ok(result)
}

impl Scene {
    /// OpenPBR tables replace material parameters without affecting AS geometry.
    /// Crossing the legacy/model boundary rebuilds the packed legacy attributes.
    pub(crate) fn same_geometry(&self, other: &Self) -> bool {
        self.curves.len() == other.curves.len()
            && self
                .curves
                .iter()
                .zip(&other.curves)
                .all(|(a, b)| a.same_geometry(b))
            && self.meshes.len() == other.meshes.len()
            && self.meshes.iter().zip(&other.meshes).all(|(a, b)| {
                let surface_tables = a.material.openpbr.is_some() && b.material.openpbr.is_some();
                let attributes_match = surface_tables
                    || (a.material.openpbr.is_none()
                        && b.material.openpbr.is_none()
                        && a.color == b.color
                        && a.material == b.material);
                attributes_match
                    && a.points == b.points
                    && a.face_counts == b.face_counts
                    && a.indices == b.indices
                    && a.normals == b.normals
                    && a.normal_indices == b.normal_indices
                    && a.reverse_orientation == b.reverse_orientation
                    && a.texcoords == b.texcoords
                    && a.texcoord_indices == b.texcoord_indices
                    && a.geometry_light == b.geometry_light
                    && a.render_light_geometry == b.render_light_geometry
            })
    }
}
