//! Vulkan 1.2 compute + KHR ray queries. No vendor-specific ray tracing API.
use anyhow::{anyhow, bail, ensure, Context as _, Result};
use ash::{vk, Entry};
use std::{
    ffi::{CStr, CString},
    sync::Arc,
};
pub struct Instance {
    _entry: Entry,
    pub raw: ash::Instance,
}
impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.raw.destroy_instance(None);
        }
    }
}
#[derive(Debug, Clone)]
pub struct Adapter {
    pub name: String,
    pub ray_query: bool,
    pub hardware: bool,
    pub reason: String,
    pub device: vk::PhysicalDevice,
    pub queue: u32,
}
const EXTENSIONS: [&CStr; 3] = [
    ash::khr::acceleration_structure::NAME,
    ash::khr::ray_query::NAME,
    ash::khr::deferred_host_operations::NAME,
];
impl Instance {
    pub fn new() -> Result<Arc<Self>> {
        unsafe {
            let entry = Entry::load().context("Vulkan loader unavailable")?;
            let name = CString::new("Lava Reborn")?;
            let app = vk::ApplicationInfo::default()
                .application_name(&name)
                .api_version(vk::API_VERSION_1_2);
            let raw = entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )?;
            Ok(Arc::new(Self { _entry: entry, raw }))
        }
    }
    pub fn adapters(&self) -> Result<Vec<Adapter>> {
        unsafe {
            self.raw.enumerate_physical_devices()?.into_iter().map(|device|{
            let props=self.raw.get_physical_device_properties(device);
            let name=CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();
            let extensions=self.raw.enumerate_device_extension_properties(device)?;
            let missing:Vec<_>=EXTENSIONS.iter().filter(|e|!extensions.iter().any(|p|CStr::from_ptr(p.extension_name.as_ptr())==**e)).map(|e|e.to_string_lossy()).collect();
            let mut ray=vk::PhysicalDeviceRayQueryFeaturesKHR::default();
            let mut accel=vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
            let mut address=vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let mut f=vk::PhysicalDeviceFeatures2::default().push_next(&mut ray).push_next(&mut accel).push_next(&mut address);
            self.raw.get_physical_device_features2(device,&mut f);
            let queue=self.raw.get_physical_device_queue_family_properties(device).iter().position(|q|q.queue_count>0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE|vk::QueueFlags::GRAPHICS));
            let ray_query=missing.is_empty() && ray.ray_query!=0 && accel.acceleration_structure!=0 && address.buffer_device_address!=0 && props.api_version>=vk::API_VERSION_1_2 && queue.is_some();
            let reason=if ray_query {"KHR ray query available".into()} else {format!("requires Vulkan 1.2, combined graphics/compute queue, rayQuery, accelerationStructure, bufferDeviceAddress; missing extensions: {missing:?}")};
            Ok(Adapter{name,ray_query,hardware:props.device_type!=vk::PhysicalDeviceType::CPU,reason,device,queue:queue.unwrap_or(0) as u32})
        }).collect()
        }
    }
}
pub struct Device {
    pub instance: Arc<Instance>,
    pub physical: vk::PhysicalDevice,
    pub raw: ash::Device,
    pub accel: ash::khr::acceleration_structure::Device,
    pub queue: vk::Queue,
    pub pool: vk::CommandPool,
    pub scratch_alignment: u64,
    pub queue_family: u32,
    pub interop: bool,
    pub uuid: [u8; 16],
    pub mesh_shader: bool,
    submit_lock: std::sync::Mutex<()>,
}
impl Device {
    pub fn new(instance: Arc<Instance>, index: Option<usize>) -> Result<Arc<Self>> {
        unsafe {
            let adapters = instance.adapters()?;
            let allow_cpu = std::env::var_os("LAVA_TEST_ALLOW_CPU_VULKAN").is_some();
            let selected = if let Some(i) = index {
                adapters
                    .get(i)
                    .ok_or_else(|| anyhow!("GPU index out of range"))?
            } else {
                adapters.iter().find(|a|a.ray_query && (a.hardware || allow_cpu)).ok_or_else(||anyhow!("No hardware ray-query adapter available; software traversal is not implemented"))?
            };
            ensure!(selected.ray_query, "{}: {}", selected.name, selected.reason);
            ensure!(
                selected.hardware || allow_cpu,
                "CPU Vulkan devices are disabled; hardware ray queries required"
            );
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            instance.raw.get_physical_device_properties2(
                selected.device,
                &mut vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup),
            );
            ensure!(
                subgroup
                    .supported_stages
                    .contains(vk::ShaderStageFlags::COMPUTE)
                    && subgroup.supported_operations.contains(
                        vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::BALLOT
                    ),
                "wavefront compaction requires compute subgroup basic/ballot support"
            );
            let priorities = [1.];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(selected.queue)
                .queue_priorities(&priorities)];
            let available = instance
                .raw
                .enumerate_device_extension_properties(selected.device)?;
            let optional = [
                ash::khr::external_memory_fd::NAME,
                ash::khr::external_semaphore_fd::NAME,
            ];
            let mut interop = cfg!(target_os = "linux")
                && optional.iter().all(|name| {
                    available
                        .iter()
                        .any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == *name)
                });
            let mut external = vk::ExternalBufferProperties::default();
            instance.raw.get_physical_device_external_buffer_properties(
                selected.device,
                &vk::PhysicalDeviceExternalBufferInfo::default()
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                    )
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
                &mut external,
            );
            interop &= external
                .external_memory_properties
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE);
            let mut semprops = vk::ExternalSemaphoreProperties::default();
            instance
                .raw
                .get_physical_device_external_semaphore_properties(
                    selected.device,
                    &vk::PhysicalDeviceExternalSemaphoreInfo::default()
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
                    &mut semprops,
                );
            interop &= semprops
                .external_semaphore_features
                .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE);
            let mut names: Vec<_> = EXTENSIONS.iter().map(|x| x.as_ptr()).collect();
            let mut mesh_supported = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
            instance.raw.get_physical_device_features2(
                selected.device,
                &mut vk::PhysicalDeviceFeatures2::default().push_next(&mut mesh_supported),
            );
            let mut mesh_props = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
            instance.raw.get_physical_device_properties2(
                selected.device,
                &mut vk::PhysicalDeviceProperties2::default().push_next(&mut mesh_props),
            );
            let mesh_shader = available
                .iter()
                .any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == ash::ext::mesh_shader::NAME)
                && mesh_supported.mesh_shader != 0
                && mesh_supported.task_shader != 0
                && mesh_props.max_mesh_output_vertices >= 16
                && mesh_props.max_mesh_output_primitives >= 28
                && mesh_props.max_task_work_group_invocations >= 32
                && mesh_props.max_mesh_work_group_invocations >= 32;
            #[cfg(test)]
            let mesh_shader =
                mesh_shader && std::env::var_os("LAVA_TEST_DISABLE_MESH_SHADERS").is_none();
            if mesh_shader {
                names.push(ash::ext::mesh_shader::NAME.as_ptr());
            }
            eprintln!(
                "Lava: {}: hardware ray queries; mesh/task shaders {}",
                selected.name,
                if mesh_shader {
                    "enabled"
                } else {
                    "unavailable (curve visibility uses ray queries)"
                }
            );
            if interop {
                names.extend(optional.iter().map(|n| n.as_ptr()));
            }
            let mut id = vk::PhysicalDeviceIDProperties::default();
            instance.raw.get_physical_device_properties2(
                selected.device,
                &mut vk::PhysicalDeviceProperties2::default().push_next(&mut id),
            );
            let mut indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
            let mut supported = vk::PhysicalDeviceFeatures2::default().push_next(&mut indexing);
            instance
                .raw
                .get_physical_device_features2(selected.device, &mut supported);
            let base_features = supported.features;
            let limits = instance
                .raw
                .get_physical_device_properties(selected.device)
                .limits;
            ensure!(
                indexing.shader_sampled_image_array_non_uniform_indexing != 0,
                "texture sampling requires sampled-image non-uniform descriptor indexing"
            );
            ensure!(
                limits.max_per_stage_descriptor_samplers >= crate::texture::MAX_TEXTURES as u32
                    && limits.max_per_stage_descriptor_sampled_images
                        >= crate::texture::MAX_TEXTURES as u32,
                "GPU has insufficient sampled texture descriptors"
            );
            ensure!(
                limits.max_per_stage_descriptor_storage_buffers >= 16
                    && limits.max_descriptor_set_storage_buffers >= 16,
                "texture tile indirection requires 16 storage buffer descriptors"
            );
            let features = vk::PhysicalDeviceFeatures::default()
                .texture_compression_bc(base_features.texture_compression_bc != 0)
                .shader_sampled_image_array_dynamic_indexing(
                    base_features.shader_sampled_image_array_dynamic_indexing != 0,
                );
            let mut indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
                .shader_sampled_image_array_non_uniform_indexing(true);
            let mut ray = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
            let mut accel_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);
            let mut address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
                .buffer_device_address(true);
            let mut mesh_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
                .mesh_shader(true)
                .task_shader(true);
            let mut info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&names)
                .enabled_features(&features)
                .push_next(&mut indexing)
                .push_next(&mut ray)
                .push_next(&mut accel_features)
                .push_next(&mut address);
            if mesh_shader {
                info = info.push_next(&mut mesh_features);
            }
            let raw = instance
                .raw
                .create_device(selected.device, &info, None)
                .context("creating Vulkan logical device")?;
            let accel = ash::khr::acceleration_structure::Device::new(&instance.raw, &raw);
            let pool = match raw.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(selected.queue)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            ) {
                Ok(p) => p,
                Err(e) => {
                    raw.destroy_device(None);
                    return Err(e.into());
                }
            };
            let mut ap = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut ap);
            instance
                .raw
                .get_physical_device_properties2(selected.device, &mut props);
            Ok(Arc::new(Self {
                physical: selected.device,
                queue_family: selected.queue,
                interop,
                uuid: id.device_uuid,
                mesh_shader,
                queue: raw.get_device_queue(selected.queue, 0),
                raw,
                accel,
                pool,
                instance,
                scratch_alignment: ap.min_acceleration_structure_scratch_offset_alignment as u64,
                submit_lock: std::sync::Mutex::new(()),
            }))
        }
    }
    /// Command-pool and queue access are serialized across submissions.
    pub fn shader_stages(&self) -> vk::ShaderStageFlags {
        let stages = vk::ShaderStageFlags::COMPUTE
            | vk::ShaderStageFlags::VERTEX
            | vk::ShaderStageFlags::FRAGMENT;
        if self.mesh_shader {
            stages | vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT
        } else {
            stages
        }
    }
    pub fn submit(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        self.submit_signal(vk::Semaphore::null(), record)
    }
    pub fn submit_signal(
        &self,
        signal: vk::Semaphore,
        record: impl FnOnce(vk::CommandBuffer),
    ) -> Result<()> {
        self.submit_sync(vk::Semaphore::null(), signal, record)
    }
    pub fn submit_sync(
        &self,
        wait: vk::Semaphore,
        signal: vk::Semaphore,
        record: impl FnOnce(vk::CommandBuffer),
    ) -> Result<()> {
        #[cfg(test)]
        let wait_started = std::time::Instant::now();
        let _guard = self
            .submit_lock
            .lock()
            .map_err(|_| anyhow!("Vulkan submission lock poisoned"))?;
        #[cfg(test)]
        let hold_started = std::time::Instant::now();
        let result = unsafe {
            let cmd = self.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            let result = (|| -> Result<()> {
                self.raw.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                record(cmd);
                self.raw.end_command_buffer(cmd)?;
                let fence = self
                    .raw
                    .create_fence(&vk::FenceCreateInfo::default(), None)?;
                let commands = [cmd];
                let signals = [signal];
                let waits = [wait];
                let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
                let mut info = vk::SubmitInfo::default().command_buffers(&commands);
                if signal != vk::Semaphore::null() {
                    info = info.signal_semaphores(&signals);
                }
                if wait != vk::Semaphore::null() {
                    info = info.wait_semaphores(&waits).wait_dst_stage_mask(&stages);
                }
                let infos = [info];
                let result = self
                    .raw
                    .queue_submit(self.queue, &infos, fence)
                    .and_then(|_| self.raw.wait_for_fences(&[fence], true, u64::MAX));
                self.raw.destroy_fence(fence, None);
                result?;
                Ok(())
            })();
            self.raw.free_command_buffers(self.pool, &[cmd]);
            result
        };
        #[cfg(test)]
        SUBMIT_TIMINGS.with(|timings| {
            if let Some(samples) = timings.borrow_mut().as_mut() {
                samples.push([
                    hold_started.duration_since(wait_started).as_secs_f64() * 1000.,
                    hold_started.elapsed().as_secs_f64() * 1000.,
                ]);
            }
        });
        result
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw.device_wait_idle();
            self.raw.destroy_command_pool(self.pool, None);
            self.raw.destroy_device(None);
        }
    }
}
pub struct Buffer {
    pub device: Arc<Device>,
    pub raw: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    pub allocation_size: u64,
    #[cfg(test)]
    pub(crate) memory_properties: vk::MemoryPropertyFlags,
    host: bool,
}
impl Buffer {
    /// Upload immutable data into ordinary VRAM using a bounded, reusable staging
    /// allocation. Each completed chunk releases the queue lock, allowing the
    /// viewport to continue presenting the previous immutable scene snapshot.
    pub fn uploaded(
        device: Arc<Device>,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Self::uploaded_with_chunk_size(device, bytes, usage, 16 * 1024 * 1024)
    }
    pub(crate) fn uploaded_with_chunk_size(
        device: Arc<Device>,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
        chunk_bytes: usize,
    ) -> Result<Self> {
        let output = Self::new(
            device,
            bytes.len() as u64,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            false,
        )?;
        output.upload_staged_chunks(bytes, chunk_bytes)?;
        Ok(output)
    }
    pub(crate) fn upload_staged_chunks(&self, bytes: &[u8], chunk_bytes: usize) -> Result<()> {
        self.upload_staged_range(bytes, chunk_bytes, 0)
    }
    pub(crate) fn upload_staged_range(
        &self,
        bytes: &[u8],
        chunk_bytes: usize,
        base: u64,
    ) -> Result<()> {
        ensure!(
            chunk_bytes > 0 && base <= self.size && bytes.len() as u64 <= self.size - base,
            "invalid staged upload range"
        );
        let device = &self.device;
        if bytes.is_empty() {
            return Ok(());
        }
        let staging = Self::new(
            device.clone(),
            bytes.len().min(chunk_bytes) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            true,
        )?;
        for (index, chunk) in bytes.chunks(chunk_bytes).enumerate() {
            staging.write(chunk)?;
            device.submit(|cmd| unsafe {
                let offset = base + (index * chunk_bytes) as u64;
                device.raw.cmd_copy_buffer(
                    cmd,
                    staging.raw,
                    self.raw,
                    &[vk::BufferCopy::default()
                        .dst_offset(offset)
                        .size(chunk.len() as u64)],
                );
                let barriers = [vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(self.raw)
                    .offset(offset)
                    .size(chunk.len() as u64)];
                device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &barriers,
                    &[],
                );
            })?;
        }
        Ok(())
    }
    pub fn new(
        device: Arc<Device>,
        size: u64,
        usage: vk::BufferUsageFlags,
        host: bool,
    ) -> Result<Self> {
        Self::allocate(device, size, usage, host, false)
    }
    pub fn exported(device: Arc<Device>, size: u64) -> Result<Self> {
        ensure!(device.interop, "external memory/semaphore FD unavailable");
        Self::allocate(
            device,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            false,
            true,
        )
    }
    fn allocate(
        device: Arc<Device>,
        size: u64,
        usage: vk::BufferUsageFlags,
        host: bool,
        export: bool,
    ) -> Result<Self> {
        unsafe {
            let size = size.max(16);
            if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
                let limits = device
                    .instance
                    .raw
                    .get_physical_device_properties(device.physical)
                    .limits;
                ensure!(
                    size <= limits.max_storage_buffer_range as u64,
                    "storage buffer exceeds device maxStorageBufferRange"
                );
            }
            let mut external = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
            let mut buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            if export {
                buffer_info = buffer_info.push_next(&mut external);
            }
            let raw = device.raw.create_buffer(&buffer_info, None)?;
            let req = device.raw.get_buffer_memory_requirements(raw);
            let props = device
                .instance
                .raw
                .get_physical_device_memory_properties(device.physical);
            let mut flags = if host {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            } else {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            };
            if export {
                flags |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
            }
            let Some(index) = (0..props.memory_type_count)
                .filter(|&i| {
                    req.memory_type_bits & (1 << i) != 0
                        && props.memory_types[i as usize]
                            .property_flags
                            .contains(flags)
                })
                .max_by_key(|&i| {
                    let ty = props.memory_types[i as usize];
                    // A discrete GPU may expose a tiny CPU-visible device-local
                    // BAR heap. Do not put every allocation there just because
                    // it also has DEVICE_LOCAL set. Keep staging in system
                    // memory and GPU-only resources in ordinary VRAM.
                    let preferred = if host {
                        !ty.property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    } else {
                        !ty.property_flags
                            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    };
                    (preferred, props.memory_heaps[ty.heap_index as usize].size)
                })
            else {
                device.raw.destroy_buffer(raw, None);
                bail!("no suitable buffer memory type for {flags:?}");
            };
            let mut alloc_flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut info = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(index);
            if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                info = info.push_next(&mut alloc_flags);
            }
            let mut export_info = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(raw);
            if export {
                info = info.push_next(&mut export_info).push_next(&mut dedicated);
            }
            let memory = match device.raw.allocate_memory(&info, None) {
                Ok(m) => m,
                Err(e) => {
                    device.raw.destroy_buffer(raw, None);
                    let heap = props.memory_types[index as usize].heap_index;
                    bail!("allocating Vulkan buffer: {e}; requested={} bytes, allocation={} bytes, usage={usage:?}, host={host}, export={export}, memory_type={index}, heap={heap}, heap_size={} bytes",
                        size, req.size, props.memory_heaps[heap as usize].size);
                }
            };
            if let Err(e) = device.raw.bind_buffer_memory(raw, memory, 0) {
                device.raw.destroy_buffer(raw, None);
                device.raw.free_memory(memory, None);
                return Err(e.into());
            }
            Ok(Self {
                device,
                raw,
                memory,
                size,
                allocation_size: req.size,
                #[cfg(test)]
                memory_properties: props.memory_types[index as usize].property_flags,
                host,
            })
        }
    }
    pub fn write(&self, bytes: &[u8]) -> Result<()> {
        ensure!(
            self.host && bytes.len() as u64 <= self.size,
            "invalid host buffer write"
        );
        unsafe {
            let ptr = self.device.raw.map_memory(
                self.memory,
                0,
                self.size,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
            self.device.raw.unmap_memory(self.memory);
            Ok(())
        }
    }
    pub fn read_floats(&self, count: usize) -> Result<Vec<f32>> {
        ensure!(
            self.host && count as u64 * 4 <= self.size,
            "invalid buffer read"
        );
        unsafe {
            let ptr = self.device.raw.map_memory(
                self.memory,
                0,
                self.size,
                vk::MemoryMapFlags::empty(),
            )?;
            let result = std::slice::from_raw_parts(ptr.cast::<f32>(), count).to_vec();
            self.device.raw.unmap_memory(self.memory);
            Ok(result)
        }
    }
    pub fn address(&self) -> u64 {
        unsafe {
            self.device
                .raw
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(self.raw))
        }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_buffer(self.raw, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}
pub struct Acceleration {
    pub device: Arc<Device>,
    pub raw: vk::AccelerationStructureKHR,
    _storage: Buffer,
    count: u32,
    top: bool,
    flags: vk::BuildAccelerationStructureFlagsKHR,
}
impl Drop for Acceleration {
    fn drop(&mut self) {
        unsafe {
            self.device
                .accel
                .destroy_acceleration_structure(self.raw, None);
        }
    }
}
impl Acceleration {
    pub fn build(
        device: Arc<Device>,
        geometry: vk::AccelerationStructureGeometryKHR<'_>,
        count: u32,
        top: bool,
        fast_build: bool,
    ) -> Result<Self> {
        unsafe {
            let kind = if top {
                vk::AccelerationStructureTypeKHR::TOP_LEVEL
            } else {
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
            };
            let mut flags = if fast_build {
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD
            } else {
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
            };
            if fast_build {
                flags |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE;
            }
            let geometries = [geometry];
            let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(kind)
                .flags(flags)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geometries);
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            device.accel.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &[count],
                &mut sizes,
            );
            let storage = Buffer::new(
                device.clone(),
                sizes.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                false,
            )?;
            let raw = device.accel.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default()
                    .buffer(storage.raw)
                    .size(sizes.acceleration_structure_size)
                    .ty(kind),
                None,
            )?;
            let result = Self {
                device: device.clone(),
                raw,
                _storage: storage,
                count,
                top,
                flags,
            };
            let alignment = device.scratch_alignment;
            let scratch = Buffer::new(
                device.clone(),
                sizes.build_scratch_size + alignment,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                false,
            )?;
            info = info
                .dst_acceleration_structure(raw)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: (scratch.address() + alignment - 1) & !(alignment - 1),
                });
            let ranges =
                [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count)];
            device.submit(|cmd| {
                device
                    .accel
                    .cmd_build_acceleration_structures(cmd, &[info], &[&ranges]);
                let barriers = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                    .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)];
                device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                        | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &barriers,
                    &[],
                    &[],
                );
            })?;
            Ok(result)
        }
    }
    /// Refit in place only at a renderer batch boundary, after prior submissions complete.
    pub(crate) fn refit(&self, geometry: vk::AccelerationStructureGeometryKHR<'_>) -> Result<()> {
        ensure!(
            self.flags
                .contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE),
            "AS was not built for refitting"
        );
        unsafe {
            let geometries = [geometry];
            let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(if self.top {
                    vk::AccelerationStructureTypeKHR::TOP_LEVEL
                } else {
                    vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
                })
                .flags(self.flags)
                .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                .src_acceleration_structure(self.raw)
                .dst_acceleration_structure(self.raw)
                .geometries(&geometries);
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            self.device.accel.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &[self.count],
                &mut sizes,
            );
            let alignment = self.device.scratch_alignment;
            let scratch = Buffer::new(
                self.device.clone(),
                sizes.update_scratch_size + alignment,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                false,
            )?;
            info = info.scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: (scratch.address() + alignment - 1) & !(alignment - 1),
            });
            let ranges = [
                vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(self.count)
            ];
            self.device.submit(|cmd| {
                let before = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                            | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                    )];
                self.device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                    vk::DependencyFlags::empty(),
                    &before,
                    &[],
                    &[],
                );
                self.device
                    .accel
                    .cmd_build_acceleration_structures(cmd, &[info], &[&ranges]);
                let after = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                    .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)];
                self.device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                        | vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &after,
                    &[],
                    &[],
                );
            })?;
        }
        Ok(())
    }
    pub fn address(&self) -> u64 {
        unsafe {
            self.device.accel.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(self.raw),
            )
        }
    }
}

#[cfg(test)]
mod upload_tests {
    use super::*;
    #[test]
    #[ignore = "requires hardware Vulkan; device-local multi-chunk geometry upload"]
    fn staged_upload_preserves_bytes_across_chunk_boundaries() {
        let device = Device::new(Instance::new().unwrap(), None).unwrap();
        let words: Vec<u32> = (0..(16 * 1024 * 1024 / 4 + 12))
            .map(|i| (i as u32).wrapping_mul(2654435761))
            .collect();
        let uploaded = Buffer::uploaded(
            device.clone(),
            bytemuck::cast_slice(&words),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .unwrap();
        assert!(uploaded
            .memory_properties
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL));
        assert!(!uploaded.host);
        let readback = Buffer::new(
            device.clone(),
            uploaded.size,
            vk::BufferUsageFlags::TRANSFER_DST,
            true,
        )
        .unwrap();
        device
            .submit(|cmd| unsafe {
                device.raw.cmd_copy_buffer(
                    cmd,
                    uploaded.raw,
                    readback.raw,
                    &[vk::BufferCopy::default().size(uploaded.size)],
                );
                device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)],
                    &[],
                    &[],
                );
            })
            .unwrap();
        let actual: Vec<u32> = readback
            .read_floats(words.len())
            .unwrap()
            .iter()
            .map(|f| f.to_bits())
            .collect();
        assert!(actual == words, "staged geometry upload changed bytes");
        let empty = Buffer::uploaded(device, &[], vk::BufferUsageFlags::STORAGE_BUFFER).unwrap();
        assert!(!empty.host);
    }
}

#[cfg(test)]
thread_local! {
    static SUBMIT_TIMINGS: std::cell::RefCell<Option<Vec<[f64; 2]>>> = const { std::cell::RefCell::new(None) };
}
#[cfg(test)]
pub(crate) fn begin_submit_timings() {
    SUBMIT_TIMINGS.with(|t| *t.borrow_mut() = Some(Vec::new()));
}
#[cfg(test)]
pub(crate) fn take_submit_timings() -> Vec<[f64; 2]> {
    SUBMIT_TIMINGS.with(|t| t.borrow_mut().take().unwrap_or_default())
}
