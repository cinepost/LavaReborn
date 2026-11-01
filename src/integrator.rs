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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct IntegratorSettings {
    pub visibility: VisibilityMode,
    pub adaptive_sampling: bool,
    /// Requested Hydra diagnostic AOV; never enables additional sample gathering.
    pub sample_count_aov: bool,
    pub adaptive_threshold: f32,
    pub adaptive_min_samples: u32,
    pub upload_chunk_mib: u32,
    pub textures: crate::texture::TextureMode,
    pub diffuse_depth: u32,
    pub reflection_depth: u32,
    pub refraction_depth: u32,
    pub roulette_start: u32,
    pub reduce_analytic_caustics: bool,
}
impl Default for IntegratorSettings {
    fn default() -> Self {
        Self {
            visibility: VisibilityMode::RayQuery,
            adaptive_sampling: false,
            sample_count_aov: false,
            adaptive_threshold: 0.03,
            adaptive_min_samples: 16,
            upload_chunk_mib: 16,
            textures: crate::texture::TextureMode::Original,
            diffuse_depth: 3,
            reflection_depth: 4,
            refraction_depth: 8,
            roulette_start: 3,
            reduce_analytic_caustics: false,
        }
    }
}
impl IntegratorSettings {
    pub fn validate(self) -> Result<()> {
        ensure!(
            matches!(self.upload_chunk_mib, 16 | 32 | 64),
            "upload chunk size must be 16, 32 or 64 MiB"
        );
        ensure!(
            self.diffuse_depth <= 32 && self.reflection_depth <= 32 && self.refraction_depth <= 32,
            "ray depths must be 0..32"
        );
        ensure!(
            self.adaptive_threshold.is_finite()
                && self.adaptive_threshold > 0.
                && self.adaptive_threshold <= 1.,
            "adaptive threshold must be in (0, 1]"
        );
        ensure!(
            (2..=1024).contains(&self.adaptive_min_samples),
            "adaptive minimum samples must be 2..1024"
        );
        ensure!(self.roulette_start <= 96, "roulette start must be 0..96");
        Ok(())
    }
    pub(crate) fn same_render_settings(self, other: Self) -> bool {
        let mut value = self;
        value.upload_chunk_mib = other.upload_chunk_mib;
        value.sample_count_aov = other.sample_count_aov;
        value == other
    }
    pub fn max_scattering_events(self) -> u32 {
        self.diffuse_depth + self.reflection_depth + self.refraction_depth
    }
}

#[cfg(test)]
mod upload_policy_tests {
    use super::*;
    #[test]
    fn upload_menu_values_validate_and_preserve_render_settings() {
        let old: IntegratorSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(old.upload_chunk_mib, 16);
        for size in [16, 32, 64] {
            let mut changed = old;
            changed.upload_chunk_mib = size;
            changed.validate().unwrap();
            assert!(old.same_render_settings(changed));
            changed.diffuse_depth += 1;
            assert!(!old.same_render_settings(changed));
        }
        for size in [0, 4, 24, 128] {
            let mut invalid = old;
            invalid.upload_chunk_mib = size;
            assert!(invalid.validate().is_err());
        }
    }
}

#[cfg(test)]
mod adaptive_tests {
    use super::*;
    #[test]
    fn adaptive_defaults_validation_and_reset_key() {
        let defaults: IntegratorSettings = serde_json::from_str("{}").unwrap();
        assert!(!defaults.adaptive_sampling);
        let mut settings = defaults;
        settings.adaptive_sampling = true;
        assert!(!defaults.same_render_settings(settings));
        for threshold in [0., -0.1, 1.1, f32::NAN, f32::INFINITY] {
            settings.adaptive_threshold = threshold;
            assert!(settings.validate().is_err());
        }
        settings.adaptive_threshold = 0.01;
        for minimum in [0, 1, 1025] {
            settings.adaptive_min_samples = minimum;
            assert!(settings.validate().is_err());
        }
        settings.adaptive_min_samples = 32;
        settings.validate().unwrap();
        assert_eq!(
            settings,
            serde_json::from_str(&serde_json::to_string(&settings).unwrap()).unwrap()
        );
    }
}
