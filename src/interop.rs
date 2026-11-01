//! Immutable external-memory frames. GL returns ownership via a second semaphore
//! after its copy. Only explicit CPU Map requests create a staging readback.
use crate::vulkan::{Buffer, Device};
use anyhow::{ensure, Result};
use ash::vk;
use std::sync::{Arc, Mutex};
pub static IMAGE_READBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static GPU_PRESENTATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[derive(Default)]
struct State {
    gl_submitted: bool,
    readback: Option<Vec<f32>>,
}
pub struct GpuFrame {
    pub buffer: Buffer,
    pub ready: vk::Semaphore,
    pub returned: vk::Semaphore,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub sample_count_aov: bool,
    pub serial: u64,
    state: Mutex<State>,
}
impl GpuFrame {
    pub fn new(device: Arc<Device>, width: u32, height: u32, samples: u32) -> Result<Self> {
        Self::with_sample_count(device, width, height, samples, false)
    }
    pub fn with_sample_count(
        device: Arc<Device>,
        width: u32,
        height: u32,
        samples: u32,
        sample_count_aov: bool,
    ) -> Result<Self> {
        unsafe {
            let buffer = Buffer::exported(
                device.clone(),
                width as u64 * height as u64 * if sample_count_aov { 48 } else { 32 },
            )?;
            let mut export = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
            let ready = device.raw.create_semaphore(&info, None)?;
            let returned = match device.raw.create_semaphore(&info, None) {
                Ok(s) => s,
                Err(e) => {
                    device.raw.destroy_semaphore(ready, None);
                    return Err(e.into());
                }
            };
            Ok(Self {
                buffer,
                ready,
                returned,
                width,
                height,
                samples,
                sample_count_aov,
                serial: {
                    static SERIAL: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(1);
                    SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                },
                state: Mutex::default(),
            })
        }
    }
    pub fn export(&self) -> Result<(i32, i32, i32)> {
        unsafe {
            use std::os::fd::{FromRawFd, IntoRawFd};
            let state = self.state.lock().unwrap();
            ensure!(!state.gl_submitted, "GL snapshot already imported");
            let d = &self.buffer.device;
            let memory = ash::khr::external_memory_fd::Device::new(&d.instance.raw, &d.raw);
            let semaphore = ash::khr::external_semaphore_fd::Device::new(&d.instance.raw, &d.raw);
            let m = std::os::fd::OwnedFd::from_raw_fd(
                memory.get_memory_fd(
                    &vk::MemoryGetFdInfoKHR::default()
                        .memory(self.buffer.memory)
                        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
                )?,
            );
            let r = std::os::fd::OwnedFd::from_raw_fd(
                semaphore.get_semaphore_fd(
                    &vk::SemaphoreGetFdInfoKHR::default()
                        .semaphore(self.ready)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
                )?,
            );
            let back = semaphore.get_semaphore_fd(
                &vk::SemaphoreGetFdInfoKHR::default()
                    .semaphore(self.returned)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
            )?;
            Ok((m.into_raw_fd(), r.into_raw_fd(), back))
        }
    }
    pub fn mark_gl_submitted(&self) {
        self.state.lock().unwrap().gl_submitted = true;
        GPU_PRESENTATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn read_packed(&self) -> Result<Vec<f32>> {
        let mut state = self.state.lock().unwrap();
        if let Some(data) = &state.readback {
            return Ok(data.clone());
        }
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            println!("Lava [WARNING]: host requested CPU AOV Map (for example Houdini COP/slap-comp cache); image staging readback to system RAM is required. GPU viewport presentation remains direct.");
        }
        IMAGE_READBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = &self.buffer.device;
        let staging = Buffer::new(
            d.clone(),
            self.buffer.size,
            vk::BufferUsageFlags::TRANSFER_DST,
            true,
        )?;
        let wait = if state.gl_submitted {
            self.returned
        } else {
            self.ready
        };
        let signal = if state.gl_submitted {
            vk::Semaphore::null()
        } else {
            self.ready
        };
        d.submit_sync(wait, signal, |cmd| unsafe {
            let acquire = [vk::BufferMemoryBarrier::default()
                .buffer(self.buffer.raw)
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .dst_queue_family_index(d.queue_family)];
            d.raw.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &acquire,
                &[],
            );
            d.raw.cmd_copy_buffer(
                cmd,
                self.buffer.raw,
                staging.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: self.buffer.size,
                }],
            );
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            d.raw.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
            if !state.gl_submitted {
                let release = [vk::BufferMemoryBarrier::default()
                    .buffer(self.buffer.raw)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .src_queue_family_index(d.queue_family)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)];
                d.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &release,
                    &[],
                );
            }
        })?;
        let data = staging.read_floats((self.buffer.size / 4) as usize)?;
        state.readback = Some(data.clone());
        Ok(data)
    }
    pub fn read_color(&self) -> Result<Vec<f32>> {
        let n = self.width as usize * self.height as usize;
        Ok(self.read_packed()?[..n * 4].to_vec())
    }
    pub fn read_sample_count(&self) -> Result<Vec<f32>> {
        ensure!(self.sample_count_aov, "sampleCount AOV was not requested");
        let n = self.width as usize * self.height as usize;
        Ok(self.read_packed()?[n * 8..n * 12].to_vec())
    }
    pub fn read_depth(&self) -> Result<Vec<f32>> {
        let n = self.width as usize * self.height as usize;
        Ok(self.read_packed()?[n * 4..n * 8]
            .iter()
            .step_by(4)
            .copied()
            .collect())
    }
}
impl Drop for GpuFrame {
    fn drop(&mut self) {
        unsafe {
            self.buffer.device.raw.destroy_semaphore(self.ready, None);
            self.buffer
                .device
                .raw
                .destroy_semaphore(self.returned, None);
        }
    }
}
