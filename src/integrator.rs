//! Visibility and integrator controls are independent of interactive/final scheduling policy.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisibilityMode {
    #[default]
    RayQuery,
    Raster,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct IntegratorSettings {
    pub visibility: VisibilityMode,
    pub textures: crate::texture::TextureMode,
    pub diffuse_depth: u32,
    pub reflection_depth: u32,
    pub refraction_depth: u32,
    pub roulette_start: u32,
}
impl Default for IntegratorSettings {
    fn default() -> Self {
        Self {
            visibility: VisibilityMode::RayQuery,
            textures: crate::texture::TextureMode::Original,
            diffuse_depth: 3,
            reflection_depth: 4,
            refraction_depth: 8,
            roulette_start: 3,
        }
    }
}
impl IntegratorSettings {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.diffuse_depth <= 32 && self.reflection_depth <= 32 && self.refraction_depth <= 32,
            "ray depths must be 0..32"
        );
        ensure!(self.roulette_start <= 96, "roulette start must be 0..96");
        Ok(())
    }
    pub fn max_scattering_events(self) -> u32 {
        self.diffuse_depth + self.reflection_depth + self.refraction_depth
    }
}
