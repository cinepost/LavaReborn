//! Device-local sampled images and material-slot bindings. CPU preparation stays on the loader pool.
use super::*;
use crate::{
    material::Material,
    vulkan::{Buffer, Device},
};
use ash::vk;
use std::collections::HashSet;

pub struct GpuTexture {
    device: Arc<Device>,
    pub image: vk::Image,
    memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub scale: [f32; 4],
    pub bias: [f32; 4],
    pub fingerprint: String,
    pub format: vk::Format,
    pub mip_levels: u32,
}
impl GpuTexture {
    pub fn upload(device: Arc<Device>, asset: &PreparedTexture) -> Result<Self> {
        let mut texture = Self {
            device,
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            sampler: vk::Sampler::null(),
            scale: asset.range_scale,
            bias: asset.range_bias,
            fingerprint: asset.fingerprint.clone(),
            format: match asset.format {
                PixelFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
                PixelFormat::Bc7 => vk::Format::BC7_UNORM_BLOCK,
                PixelFormat::R32Float => vk::Format::R32_SFLOAT,
                PixelFormat::Bc4Unorm => vk::Format::BC4_UNORM_BLOCK,
                PixelFormat::Bc4Snorm => vk::Format::BC4_SNORM_BLOCK,
            },
            mip_levels: asset.levels.len() as u32,
        };
        let first = asset.levels.first().context("empty texture mip chain")?;
        let format = texture.format;
        unsafe {
            let features = texture
                .device
                .instance
                .raw
                .get_physical_device_format_properties(texture.device.physical, format)
                .optimal_tiling_features;
            ensure!(
                features.contains(
                    vk::FormatFeatureFlags::SAMPLED_IMAGE
                        | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
                        | vk::FormatFeatureFlags::TRANSFER_DST
                ),
                "GPU cannot sample texture format {format:?}"
            );
            let limits = texture
                .device
                .instance
                .raw
                .get_physical_device_properties(texture.device.physical)
                .limits;
            ensure!(
                first.width <= limits.max_image_dimension2_d
                    && first.height <= limits.max_image_dimension2_d,
                "texture exceeds GPU dimensions"
            );
            texture.image = texture.device.raw.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: first.width,
                        height: first.height,
                        depth: 1,
                    })
                    .mip_levels(asset.levels.len() as u32)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED),
                None,
            )?;
            let req = texture
                .device
                .raw
                .get_image_memory_requirements(texture.image);
            let properties = texture
                .device
                .instance
                .raw
                .get_physical_device_memory_properties(texture.device.physical);
            let index = (0..properties.memory_type_count)
                .filter(|&i| {
                    req.memory_type_bits & (1 << i) != 0
                        && properties.memory_types[i as usize]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .max_by_key(|&i| {
                    !properties.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                })
                .context("no texture device-local memory")?;
            texture.memory = texture.device.raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(index),
                None,
            )?;
            texture
                .device
                .raw
                .bind_image_memory(texture.image, texture.memory, 0)?;
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(asset.levels.len() as u32)
                .layer_count(1);
            texture.view = texture.device.raw.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(texture.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(
                        if matches!(
                            asset.format,
                            PixelFormat::R32Float | PixelFormat::Bc4Unorm | PixelFormat::Bc4Snorm
                        ) {
                            vk::ComponentMapping {
                                r: vk::ComponentSwizzle::R,
                                g: vk::ComponentSwizzle::R,
                                b: vk::ComponentSwizzle::R,
                                a: vk::ComponentSwizzle::ONE,
                            }
                        } else {
                            vk::ComponentMapping::default()
                        },
                    )
                    .subresource_range(range),
                None,
            )?;
            texture.sampler = texture.device.raw.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .min_lod(0.)
                    .max_lod((asset.levels.len() - 1) as f32),
                None,
            )?;
            let bytes: Vec<u8> = asset
                .levels
                .iter()
                .flat_map(|l| l.bytes.iter().copied())
                .collect();
            let staging = Buffer::new(
                texture.device.clone(),
                bytes.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                true,
            )?;
            staging.write(&bytes)?;
            let mut offset = 0u64;
            let copies: Vec<_> = asset
                .levels
                .iter()
                .enumerate()
                .map(|(level, mip)| {
                    let copy = vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(level as u32)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: mip.width,
                            height: mip.height,
                            depth: 1,
                        });
                    offset += mip.bytes.len() as u64;
                    copy
                })
                .collect();
            texture.device.submit(|cmd| {
                let barrier = [vk::ImageMemoryBarrier::default()
                    .image(texture.image)
                    .subresource_range(range)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
                texture.device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barrier,
                );
                texture.device.raw.cmd_copy_buffer_to_image(
                    cmd,
                    staging.raw,
                    texture.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copies,
                );
                let barrier = [vk::ImageMemoryBarrier::default()
                    .image(texture.image)
                    .subresource_range(range)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
                texture.device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER
                        | vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barrier,
                );
            })?;
        }
        Ok(texture)
    }
    fn descriptor(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
    }
}
impl Drop for GpuTexture {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_sampler(self.sampler, None);
            self.device.raw.destroy_image_view(self.view, None);
            self.device.raw.destroy_image(self.image, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSlot {
    handle: TextureHandle,
    info: [u32; 4],
    scale: [f32; 4],
    bias: [f32; 4],
    uv: [f32; 4],
    parameters: [f32; 4],
}
struct Input {
    material: usize,
    field: usize,
    binding: TextureBinding,
    request: LoadRequest,
}
struct UploadJob {
    request: LoadRequest,
    asset: Arc<PreparedTexture>,
}
struct Uploaded {
    request: LoadRequest,
    result: Result<GpuTexture>,
}
pub struct MaterialTextures {
    device: Arc<Device>,
    manager: TextureManager,
    materials: Vec<Material>,
    mode: TextureMode,
    inputs: Vec<Input>,
    resident: HashMap<LoadRequest, GpuTexture>,
    white: GpuTexture,
    table: Option<Buffer>,
    tile_table: Option<Buffer>,
    catalog: udim::Catalog,
    dirty: bool,
    handles: HashMap<(usize, usize), TextureHandle>,
    upload_sender: Option<mpsc::Sender<UploadJob>>,
    uploaded: mpsc::Receiver<Uploaded>,
    uploading: HashSet<LoadRequest>,
    upload_thread: Option<std::thread::JoinHandle<()>>,
    upload_errors: HashMap<LoadRequest, String>,
}
impl MaterialTextures {
    pub fn new(device: Arc<Device>) -> Result<Self> {
        let white = GpuTexture::upload(
            device.clone(),
            &PreparedTexture {
                format: PixelFormat::Rgba32Float,
                levels: vec![Mip {
                    width: 1,
                    height: 1,
                    bytes: bytemuck::cast_slice(&[[1f32; 4]]).to_vec(),
                }],
                range_scale: [1.; 4],
                range_bias: [0.; 4],
                fingerprint: "white".into(),
                cache_hit: false,
            },
        )?;
        let (upload_sender, jobs) = mpsc::channel::<UploadJob>();
        let (completed, uploaded) = mpsc::channel();
        let gpu = device.clone();
        let upload_thread = std::thread::Builder::new()
            .name("lava-texture-upload".into())
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    let result = GpuTexture::upload(gpu.clone(), &job.asset);
                    if completed
                        .send(Uploaded {
                            request: job.request,
                            result,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })?;
        Ok(Self {
            upload_sender: Some(upload_sender),
            uploaded,
            uploading: HashSet::new(),
            upload_thread: Some(upload_thread),
            upload_errors: HashMap::new(),
            device,
            manager: TextureManager::new(cache_root()),
            materials: Vec::new(),
            mode: TextureMode::Original,
            inputs: Vec::new(),
            resident: HashMap::new(),
            white,
            table: None,
            tile_table: None,
            catalog: udim::Catalog::new(),
            dirty: true,
            handles: HashMap::new(),
        })
    }
    pub fn configure(&mut self, materials: Vec<Material>, mode: TextureMode) -> Result<()> {
        self.materials = materials;
        self.mode = mode;
        self.inputs.clear();
        let mut patterns = HashSet::new();
        for (material, m) in self.materials.iter().enumerate() {
            for (name, binding) in &m.textures {
                let (field, color) =
                    crate::material::texture_field(name).context("unknown texture input")?;
                let request = LoadRequest::new(binding, mode, color);
                if mode != TextureMode::Virtual && udim::is_udim(&request.file) {
                    patterns.insert(request.clone());
                }
                self.inputs.push(Input {
                    material,
                    field,
                    binding: binding.clone(),
                    request,
                });
            }
        }
        self.catalog.configure(&patterns);
        self.reconcile()?;
        self.dirty = true;
        Ok(())
    }
    fn requests(&self) -> HashSet<LoadRequest> {
        if self.mode == TextureMode::Virtual {
            return HashSet::new();
        }
        self.inputs
            .iter()
            .flat_map(|i| {
                if udim::is_udim(&i.request.file) {
                    self.catalog
                        .tiles(&i.request)
                        .iter()
                        .map(|tile| {
                            let mut request = i.request.clone();
                            request.file = tile.file.clone();
                            request
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![i.request.clone()]
                }
            })
            .collect()
    }
    fn reconcile(&mut self) -> Result<()> {
        let unique = self.requests();
        ensure!(
            unique.len() < MAX_TEXTURES,
            "prototype supports at most {} distinct resident texture images, including UDIM tiles",
            MAX_TEXTURES - 1
        );
        let active: Vec<_> = unique.into_iter().collect();
        self.manager.retain(&active);
        self.resident.retain(|r, _| active.contains(r));
        self.upload_errors.retain(|r, _| active.contains(r));
        for request in active {
            self.manager.request(request);
        }
        Ok(())
    }
    pub fn handle(&self, material: usize, field: usize) -> Option<TextureHandle> {
        self.handles.get(&(material, field)).copied()
    }
    pub fn set_mode(&mut self, mode: TextureMode) -> Result<()> {
        if mode != self.mode {
            self.configure(self.materials.clone(), mode)?;
        }
        Ok(())
    }
    pub fn resident_formats(&self) -> Vec<(vk::Format, u32)> {
        self.resident
            .values()
            .map(|r| (r.format, r.mip_levels))
            .collect()
    }
    pub fn pending(&self) -> bool {
        self.catalog.pending()
            || self.manager.pending()
            || self.manager.has_ready()
            || !self.uploading.is_empty()
    }
    pub fn poll(&mut self, wait: bool, set: vk::DescriptorSet) -> Result<bool> {
        loop {
            if self.catalog.poll() {
                self.reconcile()?;
                self.dirty = true;
            }
            let unique = self.requests();
            self.manager.poll_limited(4);
            while let Ok(done) = self.uploaded.try_recv() {
                self.uploading.remove(&done.request);
                if !unique.contains(&done.request) {
                    continue;
                }
                match done.result {
                    Ok(texture) => {
                        self.resident.insert(done.request.clone(), texture);
                        self.upload_errors.remove(&done.request);
                        self.dirty = true;
                    }
                    Err(error) => {
                        let message = format!("{}: {error:#}", done.request.file.display());
                        println!("Lava [WARNING]: texture GPU upload failed; retaining previous texture or material constant: {message}");
                        self.upload_errors.insert(done.request, message);
                    }
                }
            }
            for request in &unique {
                if self.uploading.len() >= 2 {
                    break;
                }
                if self.uploading.contains(request) {
                    continue;
                }
                if let Some(asset) = self.manager.take_ready(request) {
                    if self
                        .resident
                        .get(request)
                        .is_none_or(|r| r.fingerprint != asset.fingerprint)
                    {
                        self.upload_errors.remove(request);
                        self.upload_sender
                            .as_ref()
                            .context("texture upload worker closed")?
                            .send(UploadJob {
                                request: request.clone(),
                                asset,
                            })?;
                        self.uploading.insert(request.clone());
                    }
                }
            }
            if !wait {
                break;
            }
            if !self.catalog.scanning() {
                self.catalog.check_errors()?;
            }
            self.manager.check_errors()?;
            if let Some(error) = self.upload_errors.values().next() {
                bail!("final texture upload failed: {error}");
            }
            if !self.pending() && !self.catalog.scanning() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        if !self.dirty {
            return Ok(false);
        }
        self.bind(set)?;
        self.dirty = false;
        Ok(true)
    }
    fn bind(&mut self, set: vk::DescriptorSet) -> Result<()> {
        use bytemuck::Zeroable;
        let mut slots = vec![GpuSlot::zeroed(); (self.materials.len() * MATERIAL_FIELDS).max(1)];
        let mut descriptors = vec![self.white.descriptor(); MAX_TEXTURES];
        let mut indices = HashMap::new();
        for (i, (request, texture)) in self.resident.iter().enumerate() {
            descriptors[i + 1] = texture.descriptor();
            indices.insert(request, i + 1);
        }
        let mut tiles = Vec::<TextureTile>::new();
        let mut handles = HashMap::<LoadRequest, TextureHandle>::new();
        self.handles.clear();
        for input in &self.inputs {
            let b = &input.binding;
            let handle = *handles.entry(input.request.clone()).or_insert_with(|| {
                let udim = udim::is_udim(&input.request.file);
                let flags = if udim {
                    TextureHandle::UDIM
                } else {
                    TextureHandle::PLAIN
                };
                if self.mode == TextureMode::Virtual {
                    return TextureHandle::placeholder(flags | TextureHandle::VIRTUAL);
                }
                let offset = tiles.len() as u32;
                let entries = if udim {
                    self.catalog
                        .tiles(&input.request)
                        .iter()
                        .map(|tile| {
                            let mut r = input.request.clone();
                            r.file = tile.file.clone();
                            (tile.number, r)
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![(0, input.request.clone())]
                };
                for (number, request) in entries {
                    if let (Some(&index), Some(texture)) =
                        (indices.get(&request), self.resident.get(&request))
                    {
                        tiles.push(TextureTile {
                            info: [number, index as u32, 0, 0],
                            range_scale: texture.scale,
                            range_bias: texture.bias,
                        });
                    }
                }
                let count = tiles.len() as u32 - offset;
                TextureHandle {
                    flags: flags | if count == 0 { TextureHandle::SOLID } else { 0 },
                    table_offset: offset,
                    tile_count: count,
                    reserved: 0,
                }
            });
            self.handles.insert((input.material, input.field), handle);
            let channel = match b.channel.as_str() {
                "r" => 0,
                "g" => 1,
                "b" => 2,
                "a" => 3,
                "luminance" => 5,
                _ => 4,
            };
            let wrap = |w| match w {
                Wrap::Repeat => 0,
                Wrap::Clamp => 1,
                Wrap::Mirror => 2,
                Wrap::Black => 3,
            };
            slots[input.material * MATERIAL_FIELDS + input.field] = GpuSlot {
                handle,
                info: [
                    u32::from(handle.flags & TextureHandle::SOLID == 0),
                    channel,
                    wrap(b.wrap_s),
                    wrap(b.wrap_t),
                ],
                scale: b.scale,
                bias: b.bias,
                uv: [b.uv_scale[0], b.uv_scale[1], b.uv_offset[0], b.uv_offset[1]],
                parameters: [
                    b.multiply as u8 as f32,
                    b.strength,
                    b.signed as u8 as f32,
                    0.,
                ],
            };
        }
        if tiles.is_empty() {
            tiles.push(TextureTile::zeroed());
        }
        let tile_table =
            crate::material::upload_storage(self.device.clone(), bytemuck::cast_slice(&tiles))?;
        let bytes = bytemuck::cast_slice(&slots);
        let table = crate::material::upload_storage(self.device.clone(), bytes)?;
        unsafe {
            let buffer = [vk::DescriptorBufferInfo::default()
                .buffer(table.raw)
                .range(table.size)];
            let tile_buffer = [vk::DescriptorBufferInfo::default()
                .buffer(tile_table.raw)
                .range(tile_table.size)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(20)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&tile_buffer),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(18)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&descriptors),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(19)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer),
            ];
            self.device.raw.update_descriptor_sets(&writes, &[]);
        }
        self.table = Some(table);
        self.tile_table = Some(tile_table);
        Ok(())
    }
}

impl Drop for MaterialTextures {
    fn drop(&mut self) {
        self.upload_sender.take();
        if let Some(thread) = self.upload_thread.take() {
            let _ = thread.join();
        }
    }
}
