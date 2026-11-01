//! Material front-end IR. Parameter data is independent of compiled pass products.
//! Future MaterialX frontends lower graph outputs to this interface (or add a
//! generated graph product); they must not inject material code into accumulation.
mod schema;
use anyhow::{ensure, Result};
pub use schema::{GpuMaterial, OpenPbrSurface};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Material {
    pub metallic: f32,
    pub transmission: f32,
    pub ior: f32,
    /// None preserves the original ideal legacy material for old scenes.
    pub openpbr: Option<OpenPbrSurface>,
    pub textures: std::collections::BTreeMap<String, crate::texture::TextureBinding>,
}
impl Default for Material {
    fn default() -> Self {
        Self {
            metallic: 0.,
            transmission: 0.,
            ior: 1.5,
            openpbr: None,
            textures: Default::default(),
        }
    }
}
impl Material {
    pub fn validate(&self) -> Result<()> {
        for (name, binding) in &self.textures {
            ensure!(
                texture_field(name).is_some(),
                "unsupported material texture input {name}"
            );
            binding.validate()?;
        }
        if let Some(p) = self.openpbr {
            let gpu = p.gpu();
            ensure!(
                gpu.values
                    .iter()
                    .flatten()
                    .all(|v| v.is_finite() && *v >= 0.),
                "OpenPBR inputs must be finite and nonnegative"
            );
            for (name, value) in [
                ("base_weight", p.base_weight),
                ("base_metalness", p.base_metalness),
                ("specular_roughness", p.specular_roughness),
                (
                    "specular_roughness_anisotropy",
                    p.specular_roughness_anisotropy,
                ),
                ("coat_weight", p.coat_weight),
                ("coat_roughness", p.coat_roughness),
                ("coat_roughness_anisotropy", p.coat_roughness_anisotropy),
                ("fuzz_weight", p.fuzz_weight),
                ("fuzz_roughness", p.fuzz_roughness),
                ("transmission_weight", p.transmission_weight),
                ("thin_film_weight", p.thin_film_weight),
                ("geometry_opacity", p.geometry_opacity),
            ] {
                ensure!((0. ..=1.).contains(&value), "OpenPBR {name} outside [0,1]");
            }
            ensure!(
                p.specular_ior >= 1. && p.coat_ior >= 1. && p.thin_film_ior >= 1.,
                "OpenPBR IOR must be >= 1"
            );
            ensure!(p.subsurface_weight==0. && p.transmission_scatter==[0.;3] && p.transmission_dispersion_scale==0. && p.transmission_depth==0., "OpenPBR volume subsurface, absorption/scattering and spectral dispersion are not supported in this surface iteration");
        }
        Ok(())
    }
    pub fn gpu(&self) -> GpuMaterial {
        self.openpbr
            .map(|p| p.gpu())
            .unwrap_or_else(bytemuck::Zeroable::zeroed)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_and_validation() {
        let mut m = Material {
            openpbr: Some(OpenPbrSurface::default()),
            ..Default::default()
        };
        m.validate().unwrap();
        assert_eq!(m.gpu().values[0][0], 1.);
        m.openpbr.as_mut().unwrap().transmission_scatter = [1.; 3];
        assert!(m.validate().is_err());
        assert_eq!(Material::default().gpu().values[0][0], 0.);
    }
    #[test]
    fn accumulator_has_no_material_dependency() {
        for source in [
            include_str!("../../shaders/accumulation.comp"),
            include_str!("../../shaders/common.glsl"),
        ] {
            assert!(!source.contains("Material"));
            assert!(!source.contains("openpbr"));
            assert!(!source.contains("geometry.glsl"));
        }
    }
}

pub mod compiler;

pub(crate) mod pipeline_cache;

/// Pass-specific products are intentionally not interchangeable. No accumulation
/// product exists: that pass only consumes the radiance/guide outputs in the DAG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderProduct {
    Visibility,
    Surface,
}
impl ShaderProduct {
    pub fn interface(self) -> &'static str {
        match self {
            Self::Visibility => include_str!("../../shaders/materials/visibility.glsl"),
            Self::Surface => include_str!("../../shaders/materials/surface.glsl"),
        }
    }
}
pub(crate) fn upload(
    device: std::sync::Arc<crate::vulkan::Device>,
    scene: &crate::scene::Scene,
) -> Result<crate::vulkan::Buffer> {
    let records: Vec<_> = scene
        .meshes
        .iter()
        .map(|m| {
            (
                &m.material,
                m.color,
                if m.geometry_light {
                    m.light_response
                } else {
                    [1.; 2]
                },
            )
        })
        .chain(scene.curves.iter().map(|c| (&c.material, c.color, [1.; 2])))
        .map(|(material, color, response)| {
            let mut record = if material.openpbr.is_none() && !material.textures.is_empty() {
                OpenPbrSurface {
                    base_color: color,
                    base_metalness: material.metallic,
                    transmission_weight: material.transmission,
                    specular_ior: material.ior,
                    ..Default::default()
                }
                .gpu()
            } else {
                material.gpu()
            };
            record.values[0][1] = response[0];
            record.values[0][2] = response[1];
            record
        })
        .collect();
    let bytes = bytemuck::cast_slice(&records);
    if bytes.is_empty() {
        return upload_storage(device, &[0; 16]);
    }
    upload_storage(device, bytes)
}

pub(crate) fn upload_luts(
    device: std::sync::Arc<crate::vulkan::Device>,
) -> Result<crate::vulkan::Buffer> {
    upload_storage(device, include_bytes!("openpbr_luts.bin"))
}
pub(crate) fn upload_storage(
    device: std::sync::Arc<crate::vulkan::Device>,
    bytes: &[u8],
) -> Result<crate::vulkan::Buffer> {
    use ash::vk;
    let staging = crate::vulkan::Buffer::new(
        device.clone(),
        bytes.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        true,
    )?;
    staging.write(bytes)?;
    let gpu = crate::vulkan::Buffer::new(
        device.clone(),
        bytes.len() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        false,
    )?;
    device.submit(|cmd| unsafe {
        device.raw.cmd_copy_buffer(
            cmd,
            staging.raw,
            gpu.raw,
            &[vk::BufferCopy::default().size(bytes.len() as u64)],
        );
        device.raw.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            &[],
            &[],
        );
    })?;
    Ok(gpu)
}

// Shared stable field IDs for texture lowering (visibility uses opacity only).
include!("texture_fields.rs");
