//! Curve control points lower to round, tapered analytic segments, never CPU hull triangles.
use crate::{material::Material, scene::Triangle};
use anyhow::{ensure, Result};
use glam::Vec3;
use serde::{Deserialize, Serialize};
fn default_radius() -> Vec<f32> {
    vec![0.01]
}
fn default_color() -> [f32; 3] {
    [0.5; 3]
}
fn default_basis() -> String {
    "linear".into()
}
fn default_rate() -> String {
    "vertex".into()
}
fn default_steps() -> u32 {
    8
}
fn radius<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Vec<f32>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Values {
        One(f32),
        Many(Vec<f32>),
    }
    Ok(match Values::deserialize(d)? {
        Values::One(v) => vec![v],
        Values::Many(v) => v,
    })
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Curves {
    pub points: Vec<[f32; 3]>,
    pub counts: Vec<u32>,
    #[serde(default = "default_basis")]
    pub basis: String,
    #[serde(default)]
    pub periodic: bool,
    #[serde(default = "default_radius", deserialize_with = "radius")]
    pub radius: Vec<f32>,
    #[serde(default = "default_rate")]
    pub radius_interpolation: String,
    #[serde(default = "default_steps")]
    pub subdivisions: u32,
    #[serde(default = "default_color")]
    pub color: [f32; 3],
    #[serde(default)]
    pub material: Material,
}
impl Curves {
    pub(crate) fn same_geometry(&self, other: &Self) -> bool {
        let tables = self.material.openpbr.is_some() && other.material.openpbr.is_some();
        (tables || (self.material == other.material && self.color == other.color))
            && self.points == other.points
            && self.counts == other.counts
            && self.basis == other.basis
            && self.periodic == other.periodic
            && self.radius == other.radius
            && self.radius_interpolation == other.radius_interpolation
            && self.subdivisions == other.subdivisions
    }
    pub fn segments(&self, material_id: usize) -> Result<Vec<Triangle>> {
        self.material.validate()?;
        ensure!(
            (0. ..=1.).contains(&self.material.metallic)
                && (0. ..=1.).contains(&self.material.transmission)
                && self.material.ior.is_finite()
                && (1. ..=4.).contains(&self.material.ior),
            "invalid curve material"
        );
        ensure!(
            self.points
                .iter()
                .flatten()
                .chain(self.radius.iter())
                .chain(self.color.iter())
                .all(|v| v.is_finite()),
            "nonfinite curve data"
        );
        ensure!(
            self.radius.iter().all(|r| *r >= 0.) && !self.radius.is_empty(),
            "curve radii must be nonnegative and nonempty"
        );
        ensure!(
            self.counts.iter().map(|n| *n as usize).sum::<usize>() == self.points.len(),
            "curve control counts do not match points"
        );
        ensure!(
            (1..=64).contains(&self.subdivisions),
            "curve subdivisions must be 1..64"
        );
        ensure!(
            ["linear", "bezier", "catmullRom", "bspline"].contains(&self.basis.as_str()),
            "unsupported curve basis {}",
            self.basis
        );
        ensure!(
            ["vertex", "varying", "uniform", "constant"]
                .contains(&self.radius_interpolation.as_str()),
            "invalid curve radius interpolation"
        );
        let linear = self.basis == "linear";
        let spans = |n: usize| -> Result<usize> {
            ensure!(
                n >= if linear { 2 } else { 4 },
                "too few curve control points"
            );
            if self.basis == "bezier" {
                ensure!(
                    if self.periodic {
                        n.is_multiple_of(3)
                    } else {
                        (n - 1).is_multiple_of(3)
                    },
                    "invalid Bezier control count"
                );
            }
            Ok(if linear {
                if self.periodic {
                    n
                } else {
                    n - 1
                }
            } else if self.basis == "bezier" {
                if self.periodic {
                    n / 3
                } else {
                    (n - 1) / 3
                }
            } else if self.periodic {
                n
            } else {
                n - 3
            })
        };
        let span_counts: Vec<_> = self
            .counts
            .iter()
            .map(|n| spans(*n as usize))
            .collect::<Result<_>>()?;
        let varying: usize = span_counts
            .iter()
            .map(|n| n + usize::from(!self.periodic))
            .sum();
        let expected = match self.radius_interpolation.as_str() {
            "constant" => 1,
            "uniform" => self.counts.len(),
            "varying" => varying,
            _ => self.points.len(),
        };
        ensure!(
            self.radius.len() == 1 || self.radius.len() == expected,
            "curve radius count/interpolation mismatch ({} expected {})",
            self.radius.len(),
            expected
        );
        let mut records = Vec::new();
        let mut point_base = 0;
        let mut varying_base = 0;
        for (curve, (&n, &span_count)) in self.counts.iter().zip(&span_counts).enumerate() {
            let n = n as usize;
            let steps = if linear {
                1
            } else {
                self.subdivisions as usize
            };
            for span in 0..span_count {
                let first = if self.basis == "bezier" {
                    span * 3
                } else {
                    span
                };
                let evaluate = |u: f32| {
                    let u2 = u * u;
                    let u3 = u2 * u;
                    let weights = match self.basis.as_str() {
                        "linear" => [1. - u, u, 0., 0.],
                        "bezier" => [
                            (1. - u).powi(3),
                            3. * u * (1. - u).powi(2),
                            3. * u2 * (1. - u),
                            u3,
                        ],
                        "catmullRom" => [
                            -0.5 * u + u2 - 0.5 * u3,
                            1. - 2.5 * u2 + 1.5 * u3,
                            0.5 * u + 2. * u2 - 1.5 * u3,
                            -0.5 * u2 + 0.5 * u3,
                        ],
                        _ => [
                            (1. - 3. * u + 3. * u2 - u3) / 6.,
                            (4. - 6. * u2 + 3. * u3) / 6.,
                            (1. + 3. * u + 3. * u2 - 3. * u3) / 6.,
                            u3 / 6.,
                        ],
                    };
                    let mut p = Vec3::ZERO;
                    let mut r = 0.;
                    for (j, w) in weights.into_iter().enumerate() {
                        let idx = point_base + (first + j) % n;
                        p += Vec3::from(self.points[idx]) * w;
                        if self.radius.len() > 1 && self.radius_interpolation == "vertex" {
                            r += self.radius[idx] * w;
                        }
                    }
                    r = if self.radius.len() == 1 {
                        self.radius[0]
                    } else {
                        match self.radius_interpolation.as_str() {
                            "constant" => self.radius[0],
                            "uniform" => self.radius[curve],
                            "varying" => {
                                self.radius[varying_base + span] * (1. - u)
                                    + self.radius[varying_base
                                        + (span + 1) % (span_count + usize::from(!self.periodic))]
                                        * u
                            }
                            _ => r,
                        }
                    };
                    (p, r.max(0.))
                };
                for step in 0..steps {
                    let (a, r0) = evaluate(step as f32 / steps as f32);
                    let (b, r1) = evaluate((step + 1) as f32 / steps as f32);
                    ensure!(
                        a.is_finite() && b.is_finite() && r0.is_finite() && r1.is_finite(),
                        "curve evaluation overflow"
                    );
                    if r0.max(r1) == 0. {
                        continue;
                    }
                    let v4 = |p: Vec3, w: f32| [p.x, p.y, p.z, w];
                    records.push(Triangle {
                        a: v4(a, r0),
                        b: v4(b, r1),
                        c: [0.; 4],
                        color: [self.color[0], self.color[1], self.color[2], 1.],
                        material: [
                            self.material.metallic,
                            self.material.transmission,
                            self.material.ior,
                            f32::from_bits(material_id as u32),
                        ],
                        uv01: [
                            (span as f32 + step as f32 / steps as f32) / span_count as f32,
                            0.,
                            (span as f32 + (step + 1) as f32 / steps as f32) / span_count as f32,
                            1.,
                        ],
                        uv2: [0.; 4],
                        normal_a: [0.; 4],
                        normal_b: [0., 0., 0., 1.],
                        normal_c: [0.; 4],
                    });
                }
            }
            point_base += n;
            varying_base += span_count + usize::from(!self.periodic);
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Curves {
        serde_json::from_value(
            serde_json::json!({"points":[[0.,0.,0.],[0.,1.,0.]],"counts":[2],"radius":0.1}),
        )
        .unwrap()
    }
    #[test]
    fn bases_radii_bounds_and_validation() {
        let mut c = fixture();
        let linear = c.segments(7).unwrap();
        assert_eq!(linear.len(), 1);
        assert_eq!(linear[0].a, [0., 0., 0., 0.1]);
        assert_eq!(linear[0].material[3].to_bits(), 7);
        c.points = vec![[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]];
        c.counts = vec![4];
        c.radius = vec![0.1, 0.2];
        c.radius_interpolation = "varying".into();
        for basis in ["bezier", "catmullRom", "bspline"] {
            c.basis = basis.into();
            let result = c.segments(0).unwrap();
            assert_eq!(result.len(), 8);
            assert!((result[0].a[3] - 0.1).abs() < 1e-6);
            assert!((result[7].b[3] - 0.2).abs() < 1e-6);
            let first = if basis == "bezier" { 0. } else { 1. };
            let last = if basis == "bezier" { 3. } else { 2. };
            assert!((result[0].a[0] - first).abs() < 1e-6);
            assert!((result[7].b[0] - last).abs() < 1e-6);
            for pair in result.windows(2) {
                assert_eq!(pair[0].b, pair[1].a);
            }
        }
        c.basis = "linear".into();
        c.periodic = true;
        c.radius = vec![0.1];
        assert_eq!(c.segments(0).unwrap().len(), 4);
        c.radius = vec![0.];
        assert!(c.segments(0).unwrap().is_empty());
        c.radius = vec![-1.];
        assert!(c.segments(0).is_err());
        c.radius = vec![f32::NAN];
        assert!(c.segments(0).is_err());
        c = fixture();
        c.counts = vec![3];
        assert!(c.segments(0).is_err());
    }
}
