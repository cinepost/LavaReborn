//! Driver cache with Vulkan's vendor/device/UUID compatibility header. Cache
//! failure is nonfatal; all pipelines remain reconstructible from cached SPIR-V.
use crate::vulkan::Device;
use ash::vk;
use std::{path::PathBuf, sync::Arc};
pub struct PipelineCache {
    device: Arc<Device>,
    pub raw: vk::PipelineCache,
    path: PathBuf,
}
impl PipelineCache {
    pub fn new(device: Arc<Device>) -> anyhow::Result<Self> {
        let properties = unsafe {
            device
                .instance
                .raw
                .get_physical_device_properties(device.physical)
        };
        let root = std::env::var_os("LAVA_PIPELINE_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("XDG_CACHE_HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| {
                        std::env::var_os("HOME")
                            .map(PathBuf::from)
                            .unwrap_or_else(std::env::temp_dir)
                            .join(".cache")
                    })
                    .join("lava-reborn")
            });
        let uuid = properties
            .pipeline_cache_uuid
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let path = root.join(format!(
            "{}-{}-{}-{uuid}.vkcache",
            properties.vendor_id, properties.device_id, properties.driver_version
        ));
        let bytes = std::fs::read(&path).unwrap_or_default();
        let compatible = bytes.len() >= 32
            && u32::from_ne_bytes(bytes[0..4].try_into().unwrap()) >= 32
            && u32::from_ne_bytes(bytes[4..8].try_into().unwrap()) == 1
            && u32::from_ne_bytes(bytes[8..12].try_into().unwrap()) == properties.vendor_id
            && u32::from_ne_bytes(bytes[12..16].try_into().unwrap()) == properties.device_id
            && bytes[16..32] == properties.pipeline_cache_uuid;
        let initial = if compatible { bytes.as_slice() } else { &[] };
        let raw = unsafe {
            device
                .raw
                .create_pipeline_cache(
                    &vk::PipelineCacheCreateInfo::default().initial_data(initial),
                    None,
                )
                .or_else(|_| {
                    device
                        .raw
                        .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                })?
        };
        Ok(Self { device, raw, path })
    }
    pub fn persist(&self) {
        let result = (|| -> anyhow::Result<()> {
            let bytes = unsafe { self.device.raw.get_pipeline_cache_data(self.raw)? };
            std::fs::create_dir_all(self.path.parent().unwrap())?;
            // Unique per renderer; rename never exposes a partial driver blob.
            let temporary = self.path.with_extension(format!(
                "{}-{}.tmp",
                std::process::id(),
                self.raw.as_raw()
            ));
            std::fs::write(&temporary, bytes)?;
            std::fs::rename(temporary, &self.path)?;
            Ok(())
        })();
        if let Err(e) = result {
            eprintln!("Lava WARNING: cannot persist Vulkan material pipeline cache: {e}");
        }
    }
}
use ash::vk::Handle;
impl Drop for PipelineCache {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_pipeline_cache(self.raw, None);
        }
    }
}
