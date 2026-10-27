//! Shared device-resident prime bases for camera and path low-discrepancy samples.
use crate::vulkan::{Buffer, Device};
use ash::vk;
use std::sync::Arc;
const DIMENSIONS: usize = 4096;
fn primes() -> Vec<u32> {
    let mut composite = vec![false; 65536];
    let mut values = Vec::with_capacity(DIMENSIONS);
    for i in 2..composite.len() {
        if !composite[i] {
            values.push(i as u32);
            if values.len() == DIMENSIONS {
                break;
            }
            if i * i < composite.len() {
                for j in (i * i..composite.len()).step_by(i) {
                    composite[j] = true;
                }
            }
        }
    }
    values
}
pub(crate) fn upload(device: Arc<Device>) -> anyhow::Result<Buffer> {
    let values = primes();
    let bytes = bytemuck::cast_slice(&values);
    let staging = Buffer::new(
        device.clone(),
        bytes.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        true,
    )?;
    staging.write(bytes)?;
    let gpu = Buffer::new(
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
            vk::PipelineStageFlags::VERTEX_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER
                | vk::PipelineStageFlags::COMPUTE_SHADER,
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
#[cfg(test)]
mod tests {
    #[test]
    fn dimensions_have_distinct_prime_bases() {
        let p = super::primes();
        assert_eq!(&p[..4], &[2, 3, 5, 7]);
        assert_eq!(p.len(), 4096);
        assert!(p.windows(2).all(|v| v[0] < v[1]));
        assert!(p
            .iter()
            .all(|&n| (2..=(n as f64).sqrt() as u32).all(|d| n % d != 0)));
        assert!(4 + 96 * 40 < p.len());
    }
}
