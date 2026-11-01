#[path = "renderer_adaptive.rs"]
mod adaptive;
#[cfg(test)]
#[path = "renderer_alpha_tests.rs"]
mod alpha_tests;
#[cfg(test)]
#[path = "renderer_filter_tests.rs"]
mod filter_tests;
#[cfg(test)]
#[path = "renderer_sample_count_tests.rs"]
mod sample_count_tests;
use crate::{
    graph::Graph,
    policy::Mode,
    scene::Scene,
    vulkan::{Acceleration, Buffer, Device, Instance},
};
use anyhow::{ensure, Result};
use ash::vk;
use std::sync::{Arc, Mutex, OnceLock, Weak};
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    transform: [f32; 16],
    optics: [f32; 4],
    clipping_offset: [f32; 4],
    lens: [f32; 4],
    world_to_camera: [f32; 16],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Push {
    width: u32,
    height: u32,
    sample: u32,
    lights: u32,
    exposure: f32,
    radius: u32,
    linear_output: u32,
    filter_phase: u32,
    diffuse_depth: u32,
    reflection_depth: u32,
    refraction_depth: u32,
    roulette_start: u32,
    ray_phase: u32,
    queue_parity: u32,
    max_groups_x: u32,
    sampling_mode: u32,
    target_samples: u32,
    reduce_analytic_caustics: u32,
    mesh_triangles: u32,
    curve_count: u32,
    adaptive_threshold: f32,
    adaptive_min_samples: u32,
}
/// Immutable object acceleration survives replacement of unrelated scene geometry.
/// Weak cache ownership avoids retaining VRAM or creating a Device ownership cycle.
struct CurveBlas {
    acceleration: Acceleration,
    _input: Buffer,
    bounds: Vec<[f32; 6]>,
    fast_build: bool,
}
impl CurveBlas {
    fn cached(device: Arc<Device>, bounds: Vec<[f32; 6]>, fast_build: bool) -> Result<Arc<Self>> {
        static CACHE: OnceLock<Mutex<Vec<Weak<CurveBlas>>>> = OnceLock::new();
        let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
        cache.retain(|entry| entry.strong_count() > 0);
        for entry in cache.iter().filter_map(Weak::upgrade) {
            if Arc::ptr_eq(&entry.acceleration.device, &device)
                && entry.fast_build == fast_build
                && entry.bounds == bounds
            {
                return Ok(entry);
            }
        }
        let usage = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        let input = Buffer::new(device.clone(), (bounds.len() * 24) as u64, usage, true)?;
        input.write(bytemuck::cast_slice(&bounds))?;
        let data = vk::AccelerationStructureGeometryAabbsDataKHR::default()
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: input.address(),
            })
            .stride(24);
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs: data });
        let acceleration =
            Acceleration::build(device, geometry, bounds.len() as u32, false, fast_build)?;
        let entry = Arc::new(Self {
            acceleration,
            _input: input,
            bounds,
            fast_build,
        });
        cache.push(Arc::downgrade(&entry));
        Ok(entry)
    }
}
/// Owns resources produced by this pass. Inputs refer to producers via the compiled DAG.
pub(crate) struct PreparedGeometry {
    tri: Buffer,
    materials: Buffer,
    vertices: Buffer,
    instances: Buffer,
    blas: Acceleration,
    tlas: Acceleration,
    texture_materials: Vec<crate::material::Material>,
    texture_mode: crate::texture::TextureMode,
    curve_blas: Vec<Arc<CurveBlas>>,
    mesh_ranges: Vec<std::ops::Range<usize>>,
    mesh_triangles: u32,
    curve_count: u32,
}
struct GpuPass {
    output: Buffer,
    filtered: Option<Buffer>,
    pipeline: vk::Pipeline,
    raster: Option<crate::raster::RasterVisibility>,
    prepare_pipeline: Option<vk::Pipeline>,
    reorder_pipeline: Option<vk::Pipeline>,
    scratch: Vec<Buffer>,
    samples: u32,
}
impl GpuPass {
    fn reset(&mut self) -> Result<()> {
        self.output.device.submit(|cmd| unsafe {
            let before = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
            self.output.device.raw.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &before,
                &[],
                &[],
            );
            for buffer in std::iter::once(&self.output)
                .chain(self.filtered.iter())
                .chain(self.scratch.iter())
            {
                buffer
                    .device
                    .raw
                    .cmd_fill_buffer(cmd, buffer.raw, 0, vk::WHOLE_SIZE, 0);
            }
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            self.output.device.raw.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
        })?;
        self.samples = 0;
        Ok(())
    }
}
struct AdaptiveResources {
    device: Arc<Device>,
    pipelines: Vec<vk::Pipeline>,
    pixels: Buffer,
    control: Buffer,
    readback: Buffer,
}
impl Drop for AdaptiveResources {
    fn drop(&mut self) {
        unsafe {
            for &p in &self.pipelines {
                self.device.raw.destroy_pipeline(p, None);
            }
        }
    }
}
pub struct Renderer {
    adaptive: Option<AdaptiveResources>,
    sample_count_pipelines: [Option<vk::Pipeline>; 2],
    reorder_rays: bool,
    warned_curve_fallback: bool,
    curve_blas: Vec<Arc<CurveBlas>>,
    mesh_ranges: Vec<std::ops::Range<usize>>,
    mesh_triangles: u32,
    curve_count: u32,
    textures: crate::texture::gpu::MaterialTextures,
    #[cfg(test)]
    coherent_camera_reference: bool,
    #[cfg(test)]
    progressive_sampling_reference: bool,
    device: Arc<Device>,
    pub width: u32,
    pub height: u32,
    pub mode: Mode,
    pub linear_output: bool,
    pub gpu_output: bool,
    pub gpu_frame: Option<Arc<crate::interop::GpuFrame>>,
    pub depth: Vec<f32>,
    pub sample_count_image: Vec<f32>,
    passes: Vec<GpuPass>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    scene_buffers: Vec<Buffer>,
    material_luts: Buffer,
    sample_primes: Buffer,
    tlas: Option<Acceleration>,
    blas: Option<Acceleration>,
    light_count: u32,
    light_bvh: Option<Arc<crate::light_bvh::LightBvh>>,
    settings: crate::integrator::IntegratorSettings,
    max_groups_x: u32,
    plan: crate::graph::Plan,
    sampling_target: u32,
    pipeline_cache: crate::material::pipeline_cache::PipelineCache,
}
impl Renderer {
    fn raster_enabled(&self) -> bool {
        self.settings.visibility == crate::integrator::VisibilityMode::Raster
            && (self.curve_count == 0 || self.device.mesh_shader)
    }
    pub fn new(width: u32, height: u32, mode: Mode, gpu: Option<usize>) -> Result<Self> {
        Self::new_with_ray_reordering(
            width,
            height,
            mode,
            gpu,
            std::env::var("LAVA_RAY_REORDERING").as_deref() != Ok("0"),
        )
    }
    fn new_with_ray_reordering(
        width: u32,
        height: u32,
        mode: Mode,
        gpu: Option<usize>,
        reorder: bool,
    ) -> Result<Self> {
        ensure!(
            width > 0 && height > 0 && width <= 16384 && height <= 16384,
            "resolution must be 1..16384"
        );
        let device = Device::new(Instance::new()?, gpu)?;
        // Initialize handles before fallible creation so every completed allocation has an owner.
        let pipeline_cache = crate::material::pipeline_cache::PipelineCache::new(device.clone())?;
        let material_luts = crate::material::upload_luts(device.clone())?;
        let sample_primes = crate::sampling::upload(device.clone())?;
        let mut r = Self {
            adaptive: None,
            sample_count_pipelines: [None; 2],
            reorder_rays: reorder,
            warned_curve_fallback: false,
            curve_blas: Vec::new(),
            mesh_ranges: Vec::new(),
            mesh_triangles: 0,
            curve_count: 0,
            sample_primes,
            sampling_target: 0,
            material_luts,
            pipeline_cache,
            device: device.clone(),
            width,
            height,
            mode,
            linear_output: false,
            gpu_output: false,
            gpu_frame: None,
            depth: Vec::new(),
            sample_count_image: Vec::new(),
            passes: Vec::new(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            scene_buffers: Vec::new(),
            tlas: None,
            blas: None,
            #[cfg(test)]
            coherent_camera_reference: false,
            #[cfg(test)]
            progressive_sampling_reference: false,
            textures: crate::texture::gpu::MaterialTextures::new(device.clone())?,
            light_count: 0,
            light_bvh: None,
            settings: Default::default(),
            max_groups_x: 0,
            plan: Graph::prototype().compile()?,
        };
        let ray_queues = vec![
            Buffer::new(
                r.device.clone(),
                width as u64 * height as u64 * 64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
            )?,
            Buffer::new(
                r.device.clone(),
                width as u64 * height as u64 * 64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
            )?,
            Buffer::new(
                r.device.clone(),
                48,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                false,
            )?,
        ];
        unsafe {
            r.max_groups_x = r
                .device
                .instance
                .raw
                .get_physical_device_properties(r.device.physical)
                .limits
                .max_compute_work_group_count[0];
            let bindings: Vec<_> = (0..26)
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_count(if binding == 18 {
                            crate::texture::MAX_TEXTURES as u32
                        } else {
                            1
                        })
                        .descriptor_type(match binding {
                            0 => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                            7 => vk::DescriptorType::UNIFORM_BUFFER,
                            15 | 16 => vk::DescriptorType::STORAGE_IMAGE,
                            18 => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            _ => vk::DescriptorType::STORAGE_BUFFER,
                        })
                        .stage_flags(match binding {
                            1 | 7 | 17 => r.device.shader_stages(),
                            13 | 18 | 19 | 20 => {
                                vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT
                            }
                            _ => vk::ShaderStageFlags::COMPUTE,
                        })
                })
                .collect();
            r.set_layout = r.device.raw.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let layouts = [r.set_layout];
            let push = [vk::PushConstantRange::default()
                .stage_flags(r.device.shader_stages())
                .size(std::mem::size_of::<Push>() as u32)];
            r.layout = r.device.raw.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&push),
                None,
            )?;
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: crate::texture::MAX_TEXTURES as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 2,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 21,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                },
            ];
            r.descriptor_pool = r.device.raw.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            r.set = r.device.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(r.descriptor_pool)
                    .set_layouts(&layouts),
            )?[0];
            let shaders: [&[u8]; 6] = [
                include_bytes!(concat!(env!("OUT_DIR"), "/visibility.spv")),
                include_bytes!(concat!(env!("OUT_DIR"), "/shading.spv")),
                include_bytes!(concat!(env!("OUT_DIR"), "/accumulation.spv")),
                include_bytes!(concat!(env!("OUT_DIR"), "/tonemap.spv")),
                include_bytes!(concat!(env!("OUT_DIR"), "/queue_prepare.spv")),
                include_bytes!(concat!(env!("OUT_DIR"), "/queue_reorder.spv")),
            ];
            for (i, shader) in shaders.iter().enumerate() {
                let output = Buffer::new(
                    r.device.clone(),
                    if i >= 3 {
                        16
                    } else {
                        width as u64 * height as u64 * if i == 0 { 32 } else { 16 }
                    },
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                    i == 3,
                )?;
                let filtered = if i == 2 {
                    Some(Buffer::new(
                        r.device.clone(),
                        width as u64 * height as u64 * 16,
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::TRANSFER_SRC,
                        false,
                    )?)
                } else {
                    None
                };
                let code = ash::util::read_spv(&mut std::io::Cursor::new(shader))?;
                let module = r.device.raw.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&code),
                    None,
                )?;
                let polygon_first =
                    u32::from(std::env::var("LAVA_CURVE_POLYGON_FIRST").as_deref() == Ok("1"))
                        .to_ne_bytes();
                let entries = [vk::SpecializationMapEntry::default()
                    .constant_id(50)
                    .offset(0)
                    .size(4)];
                let specialization = vk::SpecializationInfo::default()
                    .map_entries(&entries)
                    .data(&polygon_first);
                let mut stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main");
                if i == 0 {
                    stage = stage.specialization_info(&specialization);
                }
                let creation_started = std::time::Instant::now();
                if std::env::var_os("LAVA_TRACE").is_some() {
                    eprintln!("Lava pipeline: compiling product {i}");
                }
                let result = r.device.raw.create_compute_pipelines(
                    r.pipeline_cache.raw,
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(r.layout)],
                    None,
                );
                r.device.raw.destroy_shader_module(module, None);
                if std::env::var_os("LAVA_TRACE").is_some() {
                    eprintln!(
                        "Lava pipeline: product {i} ready in {:.3}s",
                        creation_started.elapsed().as_secs_f64()
                    );
                }
                let pipeline = match result {
                    Ok(p) => p[0],
                    Err((partial, e)) => {
                        for p in partial {
                            r.device.raw.destroy_pipeline(p, None);
                        }
                        return Err(e.into());
                    }
                };
                if i == 4 {
                    r.passes[1].prepare_pipeline = Some(pipeline);
                } else if i == 5 {
                    r.passes[1].reorder_pipeline = Some(pipeline);
                } else {
                    r.passes.push(GpuPass {
                        output,
                        filtered,
                        pipeline,
                        raster: None,
                        prepare_pipeline: None,
                        reorder_pipeline: None,
                        scratch: Vec::new(),
                        samples: 0,
                    });
                }
            }
            r.passes[1].scratch = ray_queues;
        }
        r.pipeline_cache.persist();
        Ok(r)
    }
    /// Called between completed GPU batches. Scene resources and shader pipelines
    /// survive viewport changes; exported snapshots own their independent storage.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        ensure!(
            width > 0 && height > 0 && width <= 16384 && height <= 16384,
            "resolution must be 1..16384"
        );
        if self.width == width && self.height == height {
            return Ok(());
        }
        let pixels = width as u64 * height as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC;
        // Allocate before committing so allocation failures preserve the old renderer.
        let mut outputs = Vec::new();
        for bytes in [pixels * 32, pixels * 16, pixels * 16, 16] {
            outputs.push(Buffer::new(
                self.device.clone(),
                bytes,
                usage,
                outputs.len() == 3,
            )?);
        }
        let filtered = Buffer::new(self.device.clone(), pixels * 16, usage, false)?;
        let queues = [
            Buffer::new(
                self.device.clone(),
                pixels * 64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
            )?,
            Buffer::new(
                self.device.clone(),
                pixels * 64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
            )?,
        ];
        let adaptive_buffers = if self.adaptive.is_some() {
            Some((
                Buffer::new(self.device.clone(), pixels * 16, usage, false)?,
                Buffer::new(self.device.clone(), pixels * 4, usage, false)?,
            ))
        } else {
            None
        };
        if let Some(raster) = self.passes[0].raster.as_mut() {
            raster.resize(width, height, self.set)?;
        }
        for (pass, output) in self.passes.iter_mut().zip(outputs) {
            pass.output = output;
        }
        self.passes[2].filtered = Some(filtered);
        for (i, queue) in queues.into_iter().enumerate() {
            self.passes[1].scratch[i] = queue;
        }
        if let Some((stats, pixels)) = adaptive_buffers {
            self.passes[2].scratch[0] = stats;
            self.adaptive.as_mut().unwrap().pixels = pixels;
        }
        self.width = width;
        self.height = height;
        unsafe {
            for (binding, buffer) in [
                (3, &self.passes[0].output),
                (4, &self.passes[1].output),
                (5, &self.passes[2].output),
                (6, &self.passes[3].output),
                (8, self.passes[2].filtered.as_ref().unwrap()),
                (10, &self.passes[1].scratch[0]),
                (11, &self.passes[1].scratch[1]),
                (
                    23,
                    self.passes[2]
                        .scratch
                        .first()
                        .unwrap_or(&self.passes[1].scratch[2]),
                ),
                (
                    24,
                    self.adaptive
                        .as_ref()
                        .map_or(&self.passes[1].scratch[2], |a| &a.pixels),
                ),
            ] {
                let info = [vk::DescriptorBufferInfo::default()
                    .buffer(buffer.raw)
                    .range(buffer.size)];
                self.device.raw.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(self.set)
                        .dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info)],
                    &[],
                );
            }
        }
        self.gpu_frame = None;
        self.depth.clear();
        self.reset_accumulation()
    }
    /// CPU triangulation and immutable AS construction may run on a builder thread.
    pub(crate) fn prepare_geometry(
        device: Arc<Device>,
        scene: &Scene,
        mode: Mode,
    ) -> Result<PreparedGeometry> {
        let (triangles, diagnostics) = scene.prepare_geometry()?;
        let mesh_triangles = triangles.iter().take_while(|t| t.normal_b[3] == 0.).count() as u32;
        let curve_count = triangles.len() as u32 - mesh_triangles;
        if diagnostics.skipped_faces > 0 || diagnostics.skipped_triangles > 0 {
            eprintln!("Lava: omitted {} unusable faces and {} zero-area triangles; rendering {} valid triangles", diagnostics.skipped_faces, diagnostics.skipped_triangles, triangles.len());
        }
        let geometry_bytes = bytemuck::cast_slice(&triangles);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        #[cfg(test)]
        let usage = usage | vk::BufferUsageFlags::TRANSFER_SRC;
        // Traversal and shading fetch these records repeatedly. Host-visible
        // memory is only staging; keeping the records there streams over PCIe.
        let tri = Buffer::uploaded_with_chunk_size(
            device.clone(),
            geometry_bytes,
            usage,
            scene.settings.upload_chunk_mib as usize * 1024 * 1024,
        )?;
        // BLAS positions are tightly packed; shading triangles retain std430 layout independently.
        let positions: Vec<[f32; 3]> = triangles
            .iter()
            .take(mesh_triangles as usize)
            .flat_map(|t| {
                [
                    t.a[..3].try_into().unwrap(),
                    t.b[..3].try_into().unwrap(),
                    t.c[..3].try_into().unwrap(),
                ]
            })
            .collect();
        let vertices = Buffer::uploaded_with_chunk_size(
            device.clone(),
            bytemuck::cast_slice(&positions),
            usage,
            scene.settings.upload_chunk_mib as usize * 1024 * 1024,
        )?;
        let data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vertices.address(),
            })
            .vertex_stride(12)
            .max_vertex((positions.len() as u32).saturating_sub(1))
            .index_type(vk::IndexType::NONE_KHR);
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .flags(vk::GeometryFlagsKHR::empty())
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: data });
        let blas = Acceleration::build(
            device.clone(),
            geometry,
            mesh_triangles,
            false,
            mode.policy().fast_build,
        )?;
        let instance = vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 1),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas.address(),
            },
        };
        let mut instance_records = vec![instance];
        // Each curve object owns an independently reusable BLAS. Material IDs and
        // global primitive offsets stay in shading records / TLAS instances, so
        // mesh topology and material edits do not invalidate curve acceleration.
        let mut curve_blas = Vec::new();
        let mut first = mesh_triangles as usize;
        while first < triangles.len() {
            let material = triangles[first].material[3].to_bits();
            let end = triangles[first..]
                .iter()
                .position(|t| t.material[3].to_bits() != material)
                .map_or(triangles.len(), |n| first + n);
            ensure!(
                first < 0x1000000,
                "curve instance primitive offset exceeds 24-bit Vulkan limit"
            );
            let bounds = triangles[first..end]
                .iter()
                .map(|t| {
                    let r = t.a[3].max(t.b[3]);
                    [
                        t.a[0].min(t.b[0]) - r,
                        t.a[1].min(t.b[1]) - r,
                        t.a[2].min(t.b[2]) - r,
                        t.a[0].max(t.b[0]) + r,
                        t.a[1].max(t.b[1]) + r,
                        t.a[2].max(t.b[2]) + r,
                    ]
                })
                .collect();
            let acceleration = CurveBlas::cached(device.clone(), bounds, mode.policy().fast_build)?;
            let mut curve_instance = instance;
            // Masks permit exact polygon-first primary visibility. Secondary
            // rays retain mask 255 and see every geometry class.
            curve_instance.instance_custom_index_and_mask = vk::Packed24_8::new(first as u32, 2);
            curve_instance.acceleration_structure_reference =
                vk::AccelerationStructureReferenceKHR {
                    device_handle: acceleration.acceleration.address(),
                };
            instance_records.push(curve_instance);
            curve_blas.push(acceleration);
            first = end;
        }
        let instances = Buffer::new(
            device.clone(),
            (std::mem::size_of_val(&instance) * instance_records.len()) as u64,
            usage,
            true,
        )?;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                instance_records.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&instance) * instance_records.len(),
            )
        };
        instances.write(bytes)?;
        let data = vk::AccelerationStructureGeometryInstancesDataKHR::default().data(
            vk::DeviceOrHostAddressConstKHR {
                device_address: instances.address(),
            },
        );
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances: data });
        let tlas = Acceleration::build(
            device.clone(),
            geometry,
            instance_records.len() as u32,
            true,
            mode.policy().fast_build,
        )?;
        let materials = crate::material::upload(device.clone(), scene)?;
        let mut offset = 0;
        let mut mesh_ranges = Vec::new();
        for mesh in &scene.meshes {
            let count = Self::mesh_records(scene, mesh)?.len();
            mesh_ranges.push(offset..offset + count);
            offset += count;
        }
        Ok(PreparedGeometry {
            mesh_ranges,
            texture_materials: scene
                .meshes
                .iter()
                .map(|m| m.material.clone())
                .chain(scene.curves.iter().map(|c| c.material.clone()))
                .collect(),
            curve_blas,
            mesh_triangles,
            curve_count,
            texture_mode: scene.settings.textures,
            materials,
            tri,
            vertices,
            instances,
            blas,
            tlas,
        })
    }
    pub(crate) fn device(&self) -> Arc<Device> {
        self.device.clone()
    }
    pub(crate) fn install_geometry(&mut self, geometry: PreparedGeometry) -> Result<()> {
        if geometry.curve_count > 0 && !self.device.mesh_shader && !self.warned_curve_fallback {
            eprintln!("Lava [WARNING]: mesh/task shaders unavailable; curve scenes use hardware ray-query visibility instead of raster visibility.");
            self.warned_curve_fallback = true;
        }
        self.mesh_ranges = geometry.mesh_ranges;
        self.mesh_triangles = geometry.mesh_triangles;
        self.curve_count = geometry.curve_count;
        self.curve_blas = geometry.curve_blas;
        self.textures
            .configure(geometry.texture_materials, geometry.texture_mode)?;
        let placeholder = || {
            Buffer::new(
                self.device.clone(),
                16,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                true,
            )
        };
        let buffers = vec![
            geometry.tri,
            placeholder()?,
            placeholder()?,
            geometry.vertices,
            geometry.instances,
            geometry.materials,
        ];
        self.scene_buffers = buffers;
        self.tlas = Some(geometry.tlas);
        self.blas = Some(geometry.blas);
        Ok(())
    }
    fn mesh_records(
        scene: &Scene,
        mesh: &crate::scene::Mesh,
    ) -> Result<Vec<crate::scene::Triangle>> {
        let local = Scene {
            camera: scene.camera.clone(),
            settings: scene.settings,
            lights: Vec::new(),
            meshes: vec![mesh.clone()],
            curves: Default::default(),
        };
        Ok(local.prepare_geometry()?.0)
    }
    /// Small fixed-topology deformations commit at the boundary between batches.
    /// All CPU validation finishes before any live GPU resource is modified.
    pub(crate) fn try_refit_meshes(&mut self, previous: &Scene, scene: &Scene) -> Result<bool> {
        let preparation_started = std::time::Instant::now();
        if self.mode != Mode::Interactive
            || previous.meshes.len() != scene.meshes.len()
            || (!Arc::ptr_eq(&previous.curves, &scene.curves) && previous.curves != scene.curves)
            || self.mesh_ranges.len() != scene.meshes.len()
        {
            return Ok(false);
        }
        let mut updates = Vec::new();
        for (index, (old, mesh)) in previous.meshes.iter().zip(&scene.meshes).enumerate() {
            if old == mesh {
                continue;
            }
            if old.points.len() != mesh.points.len()
                || old.indices != mesh.indices
                || old.face_counts != mesh.face_counts
                || old.reverse_orientation != mesh.reverse_orientation
            {
                return Ok(false);
            }
            let mut records = Self::mesh_records(scene, mesh)?;
            if records.len() != self.mesh_ranges[index].len() {
                return Ok(false);
            }
            for record in &mut records {
                record.material[3] = f32::from_bits(index as u32);
            }
            let positions: Vec<[f32; 3]> = records
                .iter()
                .flat_map(|t| {
                    [
                        t.a[..3].try_into().unwrap(),
                        t.b[..3].try_into().unwrap(),
                        t.c[..3].try_into().unwrap(),
                    ]
                })
                .collect();
            updates.push((index, records, positions));
        }
        if updates.is_empty() {
            return Ok(false);
        }
        let started = std::time::Instant::now();
        let mut uploaded = 0;
        let chunk = scene.settings.upload_chunk_mib as usize * 1024 * 1024;
        for (index, records, positions) in &updates {
            let start = self.mesh_ranges[*index].start;
            let bytes = bytemuck::cast_slice(records);
            self.scene_buffers[0].upload_staged_range(
                bytes,
                chunk,
                (start * std::mem::size_of::<crate::scene::Triangle>()) as u64,
            )?;
            self.scene_buffers[3].upload_staged_range(
                bytemuck::cast_slice(positions),
                chunk,
                (start * 36) as u64,
            )?;
            uploaded += bytes.len() + positions.len() * 12;
        }
        let data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: self.scene_buffers[3].address(),
            })
            .vertex_stride(12)
            .max_vertex((self.mesh_triangles * 3).saturating_sub(1))
            .index_type(vk::IndexType::NONE_KHR);
        self.blas.as_ref().unwrap().refit(
            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .flags(vk::GeometryFlagsKHR::empty())
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: data }),
        )?;
        let data = vk::AccelerationStructureGeometryInstancesDataKHR::default().data(
            vk::DeviceOrHostAddressConstKHR {
                device_address: self.scene_buffers[4].address(),
            },
        );
        self.tlas.as_ref().unwrap().refit(
            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { instances: data }),
        )?;
        if std::env::var_os("LAVA_TRACE").is_some() {
            eprintln!("Lava mesh refit: meshes={}, uploaded={} bytes, curve uploads=0, curve BLAS builds=0, BLAS=refit TLAS=refit, commit_ms={:.3}, total_ms={:.3}", updates.len(), uploaded, started.elapsed().as_secs_f64()*1000., preparation_started.elapsed().as_secs_f64()*1000.);
        }
        Ok(true)
    }
    pub(crate) fn set_upload_chunk_mib(&mut self, value: u32) -> Result<()> {
        let mut settings = self.settings;
        settings.upload_chunk_mib = value;
        settings.validate()?;
        self.settings.upload_chunk_mib = value;
        Ok(())
    }
    pub(crate) fn update_materials(&mut self, scene: &Scene) -> Result<()> {
        for mesh in &scene.meshes {
            mesh.material.validate()?;
        }
        self.scene_buffers[5] = crate::material::upload(self.device.clone(), scene)?;
        self.textures.configure(
            scene
                .meshes
                .iter()
                .map(|m| m.material.clone())
                .chain(scene.curves.iter().map(|c| c.material.clone()))
                .collect(),
            self.settings.textures,
        )?;
        Ok(())
    }
    pub fn set_scene(&mut self, scene: &Scene) -> Result<()> {
        let geometry = Self::prepare_geometry(self.device.clone(), scene, self.mode)?;
        self.install_geometry(geometry)?;
        self.update_scene_parameters(scene)
    }
    /// Activate camera/light state at a batch boundary after geometry is ready.
    pub(crate) fn update_scene_parameters(&mut self, scene: &Scene) -> Result<()> {
        scene.settings.validate()?;
        if self
            .light_bvh
            .as_ref()
            .is_none_or(|bvh| bvh.source != scene.lights)
        {
            self.light_bvh = Some(crate::light_bvh::LightBvh::upload(
                self.device.clone(),
                scene,
            )?);
        }
        if scene.settings.adaptive_sampling {
            self.ensure_adaptive()?;
        }
        self.settings = scene.settings;
        self.textures.set_mode(scene.settings.textures)?;
        let c = &scene.camera;
        let uniform = CameraUniform {
            transform: c.transform,
            optics: [
                c.horizontal_aperture,
                c.vertical_aperture,
                c.focal_length,
                if c.orthographic { 1. } else { 0. },
            ],
            clipping_offset: [c.near, c.far, c.aperture_offset[0], c.aperture_offset[1]],
            lens: [
                c.f_stop,
                c.focus_distance,
                {
                    let independent = !self.raster_enabled();
                    #[cfg(test)]
                    let independent = independent && !self.coherent_camera_reference;
                    if independent {
                        1.
                    } else {
                        0.
                    }
                },
                0.,
            ],
            world_to_camera: glam::Mat4::from_cols_array(&c.transform)
                .inverse()
                .to_cols_array(),
        };
        let camera = Buffer::new(
            self.device.clone(),
            std::mem::size_of_val(&uniform) as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            true,
        )?;
        camera.write(bytemuck::bytes_of(&uniform))?;
        self.scene_buffers[2] = camera;
        self.light_count = self.light_bvh.as_ref().unwrap().count;
        unsafe {
            let acceleration = [self.tlas.as_ref().unwrap().raw];
            let mut accel = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(&acceleration);
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .push_next(&mut accel)];
            self.device.raw.update_descriptor_sets(&writes, &[]);
            for (binding, buffer) in [
                (1, &self.scene_buffers[0]),
                (2, &self.light_bvh.as_ref().unwrap().lights),
                (3, &self.passes[0].output),
                (4, &self.passes[1].output),
                (5, &self.passes[2].output),
                (6, &self.passes[3].output),
                (7, &self.scene_buffers[2]),
                (8, self.passes[2].filtered.as_ref().unwrap()),
                (9, &self.light_bvh.as_ref().unwrap().nodes),
                (10, &self.passes[1].scratch[0]),
                (11, &self.passes[1].scratch[1]),
                (12, &self.passes[1].scratch[2]),
                (13, &self.scene_buffers[5]),
                (14, &self.material_luts),
                (17, &self.sample_primes),
                // Disabled specialization in production; profiler replaces this binding.
                (21, &self.passes[1].scratch[2]),
                (22, &self.passes[1].scratch[2]),
                (
                    23,
                    self.passes[2]
                        .scratch
                        .first()
                        .unwrap_or(&self.passes[1].scratch[2]),
                ),
                (
                    24,
                    self.adaptive
                        .as_ref()
                        .map_or(&self.passes[1].scratch[2], |a| &a.pixels),
                ),
                (
                    25,
                    self.adaptive
                        .as_ref()
                        .map_or(&self.passes[1].scratch[2], |a| &a.control),
                ),
            ] {
                let infos = [vk::DescriptorBufferInfo::default()
                    .buffer(buffer.raw)
                    .range(buffer.size)];
                let writes = [vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(binding)
                    .descriptor_type(if binding == 7 {
                        vk::DescriptorType::UNIFORM_BUFFER
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    })
                    .buffer_info(&infos)];
                self.device.raw.update_descriptor_sets(&writes, &[]);
            }
        }
        self.reset_accumulation()?;
        Ok(())
    }
    pub fn textures_pending(&self) -> bool {
        self.textures.pending()
    }
    pub fn refresh_textures(&mut self) -> Result<bool> {
        let changed = self.textures.poll(self.mode == Mode::Final, self.set)?;
        if changed {
            self.reset_accumulation()?;
        }
        Ok(changed)
    }
    pub fn samples(&self) -> u32 {
        self.passes[2].samples
    }
    pub fn reset_accumulation(&mut self) -> Result<()> {
        self.passes[2].reset()?;
        self.sampling_target = 0;
        self.gpu_frame = None;
        self.depth.clear();
        self.sample_count_image.clear();
        Ok(())
    }
    /// AOV requests change display products without invalidating radiance history.
    pub(crate) fn set_sample_count_aov(&mut self, enabled: bool) {
        self.settings.sample_count_aov = enabled;
        if !enabled {
            self.sample_count_image.clear();
        }
    }
    fn ensure_sample_count_pipeline(&mut self) -> Result<()> {
        if self.settings.sample_count_aov {
            let index = usize::from(self.settings.adaptive_sampling);
            if self.sample_count_pipelines[index].is_none() {
                let product: &[u8] = if index == 1 {
                    include_bytes!(concat!(env!("OUT_DIR"), "/sample_count_adaptive.spv"))
                } else {
                    include_bytes!(concat!(env!("OUT_DIR"), "/sample_count.spv"))
                };
                let code = ash::util::read_spv(&mut std::io::Cursor::new(product))?;
                unsafe {
                    let module = self.device.raw.create_shader_module(
                        &vk::ShaderModuleCreateInfo::default().code(&code),
                        None,
                    )?;
                    let result = self.device.raw.create_compute_pipelines(
                        self.pipeline_cache.raw,
                        &[vk::ComputePipelineCreateInfo::default()
                            .layout(self.layout)
                            .stage(
                                vk::PipelineShaderStageCreateInfo::default()
                                    .stage(vk::ShaderStageFlags::COMPUTE)
                                    .module(module)
                                    .name(c"main"),
                            )],
                        None,
                    );
                    self.device.raw.destroy_shader_module(module, None);
                    match result {
                        Ok(p) => self.sample_count_pipelines[index] = Some(p[0]),
                        Err((partial, e)) => {
                            for p in partial {
                                self.device.raw.destroy_pipeline(p, None);
                            }
                            return Err(e.into());
                        }
                    }
                }
                self.pipeline_cache.persist();
            }
        }
        Ok(())
    }
    /// Returns one completed batch. Zero remaining reprocesses an existing image only.
    /// Interactive workers check their mailbox after each call.
    pub fn render_batch(&mut self, remaining: u32, exposure: f32) -> Result<Vec<f32>> {
        self.ensure_sample_count_pipeline()?;
        let target_before = self
            .samples()
            .checked_add(remaining)
            .ok_or_else(|| anyhow::anyhow!("sample budget overflow"))?;
        let remaining = if self.refresh_textures()? {
            target_before
        } else {
            remaining
        };
        ensure!(self.tlas.is_some(), "set a scene before rendering");
        ensure!(exposure.is_finite(), "invalid exposure");
        let target = self
            .samples()
            .checked_add(remaining)
            .ok_or_else(|| anyhow::anyhow!("sample budget overflow"))?;
        if remaining > 0 && self.mode == Mode::Final && self.sampling_target != 0 {
            ensure!(self.sampling_target==target,"final sample budget changed; reset accumulation before changing its Hammersley set");
        }
        if remaining > 0 {
            self.sampling_target = target;
        }
        if self.raster_enabled() && self.passes[0].raster.is_none() {
            self.passes[0].raster = Some(crate::raster::RasterVisibility::new(
                self.device.clone(),
                self.width,
                self.height,
                self.layout,
                self.set,
                self.pipeline_cache.raw,
            )?);
            self.pipeline_cache.persist();
        }
        let count = remaining.min(self.mode.policy().batch_samples);
        ensure!(
            self.samples().checked_add(count).is_some()
                && (count > 0
                    || (self.samples() > 0
                        && (self.mode == Mode::Interactive
                            || self.samples() == self.sampling_target))),
            "invalid sample count"
        );
        if self.gpu_output && !self.device.interop {
            println!("Lava [WARNING]: Vulkan external memory/semaphores unavailable; CPU image fallback requires transfer through system RAM.");
            self.gpu_output = false;
        }
        let adaptive = self.settings.adaptive_sampling;
        let sampling_mode = u32::from(self.mode == Mode::Final && !adaptive);
        #[cfg(test)]
        let sampling_mode = if self.progressive_sampling_reference {
            0
        } else {
            sampling_mode
        };
        let output_needed = adaptive || self.mode == Mode::Interactive || count == remaining;
        let gpu = if self.gpu_output && self.device.interop && output_needed {
            match crate::interop::GpuFrame::with_sample_count(
                self.device.clone(),
                self.width,
                self.height,
                self.samples() + count,
                self.settings.sample_count_aov,
            ) {
                Ok(frame) => Some(Arc::new(frame)),
                Err(e) => {
                    self.gpu_output = false;
                    println!("Lava [WARNING]: GPU interop allocation failed; CPU image fallback requires transfer through system RAM: {e:#}");
                    None
                }
            }
        } else {
            None
        };
        // The GPU presentation path does not need a full-sized host-visible
        // output. Allocate it only when CPU output is actually requested.
        let output_size = self.width as u64
            * self.height as u64
            * if self.settings.sample_count_aov {
                48
            } else {
                32
            };
        if gpu.is_none() && output_needed && self.passes[3].output.size != output_size {
            self.passes[3].output = Buffer::new(
                self.device.clone(),
                output_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                true,
            )?;
        }
        let target = gpu
            .as_ref()
            .map(|f| &f.buffer)
            .unwrap_or(&self.passes[3].output);
        let device = &self.device;
        unsafe {
            let infos = [vk::DescriptorBufferInfo::default()
                .buffer(target.raw)
                .range(target.size)];
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos)];
            device.raw.update_descriptor_sets(&writes, &[]);
        }
        device.submit_signal(
            gpu.as_ref()
                .map(|g| g.ready)
                .unwrap_or(vk::Semaphore::null()),
            |cmd| unsafe {
                device.raw.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[self.set],
                    &[],
                );
                for sample in self.samples()..self.samples() + count.max(1) {
                    let mut push = Push {
                        width: self.width,
                        height: self.height,
                        sample: if count == 0 {
                            sample.saturating_sub(1)
                        } else {
                            sample
                        },
                        lights: self.light_count,
                        exposure,
                        radius: self.mode.policy().filter_radius,
                        linear_output: u32::from(self.linear_output),
                        filter_phase: 0,
                        diffuse_depth: self.settings.diffuse_depth,
                        reflection_depth: self.settings.reflection_depth,
                        refraction_depth: self.settings.refraction_depth,
                        roulette_start: self.settings.roulette_start,
                        reduce_analytic_caustics: self.settings.reduce_analytic_caustics as u32,
                        mesh_triangles: self.mesh_triangles,
                        curve_count: self.curve_count,
                        ray_phase: 0,
                        queue_parity: 0,
                        max_groups_x: self.max_groups_x,
                        sampling_mode,
                        target_samples: self.sampling_target,
                        adaptive_threshold: self.settings.adaptive_threshold,
                        adaptive_min_samples: self.settings.adaptive_min_samples,
                    };
                    device.raw.cmd_push_constants(
                        cmd,
                        self.layout,
                        self.device.shader_stages(),
                        0,
                        bytemuck::bytes_of(&push),
                    );
                    if adaptive && count > 0 {
                        self.record_adaptive_prepare(cmd, &push);
                    }
                    for &pass in &self.plan.order {
                        // Re-present existing filtered floating-point history without tracing
                        // or changing accumulation/sample counters.
                        if count == 0 && pass != 3 {
                            continue;
                        }
                        // Final mode filters and transforms only the last sample of the frame.
                        if pass == 3
                            && count > 0
                            && (sample + 1 != self.samples() + count
                                || (!adaptive && self.mode == Mode::Final && count != remaining))
                        {
                            continue;
                        }
                        let raster_visibility = pass == 0 && self.raster_enabled();
                        if raster_visibility {
                            self.passes[0].raster.as_ref().unwrap().record(
                                cmd,
                                self.layout,
                                self.set,
                                self.mesh_triangles * 3,
                                self.curve_count,
                            );
                        }
                        device.raw.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            if raster_visibility {
                                if adaptive {
                                    self.adaptive.as_ref().unwrap().pipelines[3]
                                } else {
                                    self.passes[0].raster.as_ref().unwrap().resolve
                                }
                            } else {
                                if adaptive && pass < 3 {
                                    self.adaptive.as_ref().unwrap().pipelines[pass]
                                } else {
                                    if pass == 3 && self.settings.sample_count_aov {
                                        self.sample_count_pipelines[usize::from(adaptive)].unwrap()
                                    } else {
                                        self.passes[pass].pipeline
                                    }
                                }
                            },
                        );
                        if pass == 1 {
                            let control = &self.passes[1].scratch[2];
                            let before = [vk::MemoryBarrier::default()
                                .src_access_mask(
                                    vk::AccessFlags::SHADER_READ
                                        | vk::AccessFlags::SHADER_WRITE
                                        | vk::AccessFlags::INDIRECT_COMMAND_READ,
                                )
                                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
                            device.raw.cmd_pipeline_barrier(
                                cmd,
                                vk::PipelineStageFlags::ALL_COMMANDS,
                                vk::PipelineStageFlags::TRANSFER,
                                vk::DependencyFlags::empty(),
                                &before,
                                &[],
                                &[],
                            );
                            device.raw.cmd_fill_buffer(cmd, control.raw, 0, 48, 0);
                            let filled = [vk::MemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .dst_access_mask(
                                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                                )];
                            device.raw.cmd_pipeline_barrier(
                                cmd,
                                vk::PipelineStageFlags::TRANSFER,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::DependencyFlags::empty(),
                                &filled,
                                &[],
                                &[],
                            );
                            let groups = (self.width * self.height).div_ceil(64);
                            let x = groups.min(self.max_groups_x);
                            if adaptive {
                                device.raw.cmd_dispatch_indirect(
                                    cmd,
                                    self.adaptive.as_ref().unwrap().control.raw,
                                    16,
                                );
                            } else {
                                device.raw.cmd_dispatch(cmd, x, groups.div_ceil(x), 1);
                            }
                            let wave = [vk::MemoryBarrier::default()
                                .src_access_mask(
                                    vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ,
                                )
                                .dst_access_mask(
                                    vk::AccessFlags::SHADER_READ
                                        | vk::AccessFlags::SHADER_WRITE
                                        | vk::AccessFlags::INDIRECT_COMMAND_READ,
                                )];
                            for bounce in 0..self.settings.max_scattering_events() {
                                device.raw.cmd_pipeline_barrier(
                                    cmd,
                                    vk::PipelineStageFlags::COMPUTE_SHADER
                                        | vk::PipelineStageFlags::DRAW_INDIRECT,
                                    vk::PipelineStageFlags::COMPUTE_SHADER
                                        | vk::PipelineStageFlags::DRAW_INDIRECT,
                                    vk::DependencyFlags::empty(),
                                    &wave,
                                    &[],
                                    &[],
                                );
                                push.ray_phase = 1;
                                push.queue_parity = bounce % 2;
                                device.raw.cmd_push_constants(
                                    cmd,
                                    self.layout,
                                    self.device.shader_stages(),
                                    0,
                                    bytemuck::bytes_of(&push),
                                );
                                device.raw.cmd_bind_pipeline(
                                    cmd,
                                    vk::PipelineBindPoint::COMPUTE,
                                    self.passes[1].prepare_pipeline.unwrap(),
                                );
                                device.raw.cmd_dispatch(cmd, 1, 1, 1);
                                device.raw.cmd_pipeline_barrier(
                                    cmd,
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::PipelineStageFlags::COMPUTE_SHADER
                                        | vk::PipelineStageFlags::DRAW_INDIRECT,
                                    vk::DependencyFlags::empty(),
                                    &wave,
                                    &[],
                                    &[],
                                );
                                if self.reorder_rays {
                                    device.raw.cmd_bind_pipeline(
                                        cmd,
                                        vk::PipelineBindPoint::COMPUTE,
                                        self.passes[1].reorder_pipeline.unwrap(),
                                    );
                                    device.raw.cmd_dispatch_indirect(cmd, control.raw, 32);
                                    device.raw.cmd_pipeline_barrier(
                                        cmd,
                                        vk::PipelineStageFlags::COMPUTE_SHADER,
                                        vk::PipelineStageFlags::COMPUTE_SHADER
                                            | vk::PipelineStageFlags::DRAW_INDIRECT,
                                        vk::DependencyFlags::empty(),
                                        &wave,
                                        &[],
                                        &[],
                                    );
                                }
                                device.raw.cmd_bind_pipeline(
                                    cmd,
                                    vk::PipelineBindPoint::COMPUTE,
                                    if adaptive {
                                        self.adaptive.as_ref().unwrap().pipelines[1]
                                    } else {
                                        self.passes[1].pipeline
                                    },
                                );
                                device.raw.cmd_dispatch_indirect(cmd, control.raw, 16);
                            }
                        } else if adaptive && (pass == 2 || (pass == 0 && !raster_visibility)) {
                            device.raw.cmd_dispatch_indirect(
                                cmd,
                                self.adaptive.as_ref().unwrap().control.raw,
                                16,
                            );
                        } else {
                            device.raw.cmd_dispatch(
                                cmd,
                                self.width.div_ceil(8),
                                self.height.div_ceil(8),
                                1,
                            );
                        }
                        let barriers = [vk::MemoryBarrier::default()
                            .src_access_mask(
                                vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ,
                            )
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )];
                        device.raw.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &barriers,
                            &[],
                            &[],
                        );
                        if pass == 2
                            && sample + 1 == self.samples() + count
                            && (adaptive || self.mode == Mode::Interactive || count == remaining)
                        {
                            // Filtering is the second phase of accumulation; never overwrite history.
                            push.filter_phase = 1;
                            device.raw.cmd_push_constants(
                                cmd,
                                self.layout,
                                self.device.shader_stages(),
                                0,
                                bytemuck::bytes_of(&push),
                            );
                            device.raw.cmd_dispatch(
                                cmd,
                                self.width.div_ceil(8),
                                self.height.div_ceil(8),
                                1,
                            );
                            device.raw.cmd_pipeline_barrier(
                                cmd,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::DependencyFlags::empty(),
                                &barriers,
                                &[],
                                &[],
                            );
                        }
                    }
                }
                let barriers = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &barriers,
                    &[],
                    &[],
                );
                if adaptive && count > 0 {
                    let a = self.adaptive.as_ref().unwrap();
                    let barrier = [vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
                    device.raw.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &barrier,
                        &[],
                        &[],
                    );
                    device.raw.cmd_copy_buffer(
                        cmd,
                        a.control.raw,
                        a.readback.raw,
                        &[vk::BufferCopy::default().size(4)],
                    );
                    let barrier = [vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)];
                    device.raw.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &barrier,
                        &[],
                        &[],
                    );
                }
                if let Some(g) = &gpu {
                    let release = [vk::BufferMemoryBarrier::default()
                        .buffer(g.buffer.raw)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::empty())
                        .src_queue_family_index(device.queue_family)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)];
                    device.raw.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &release,
                        &[],
                    );
                }
            },
        )?;
        self.gpu_frame = gpu;
        self.passes[2].samples += count;
        let complete = adaptive
            && count > 0
            && self.adaptive.as_ref().unwrap().readback.read_floats(1)?[0].to_bits() == 0;
        if complete {
            self.passes[2].samples = self.sampling_target;
        }
        if self.mode == Mode::Final && count != remaining && !complete {
            return Ok(Vec::new());
        }
        if self.gpu_frame.is_some() {
            self.depth.clear();
            return Ok(Vec::new());
        }
        let n = self.width as usize * self.height as usize;
        crate::interop::IMAGE_READBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut packed = self.passes[3].output.read_floats(
            n * if self.settings.sample_count_aov {
                12
            } else {
                8
            },
        )?;
        self.sample_count_image = if self.settings.sample_count_aov {
            packed[n * 8..].to_vec()
        } else {
            Vec::new()
        };
        self.depth = packed[n * 4..n * 8].iter().step_by(4).copied().collect();
        packed.truncate(n * 4);
        Ok(packed)
    }
}
impl Drop for Renderer {
    fn drop(&mut self) {
        if let Some(pass) = self.passes.first_mut() {
            pass.raster.take();
        }
        unsafe {
            for p in self.sample_count_pipelines.iter().flatten() {
                self.device.raw.destroy_pipeline(*p, None);
            }
            for p in &self.passes {
                self.device.raw.destroy_pipeline(p.pipeline, None);
                if let Some(prepare) = p.prepare_pipeline {
                    self.device.raw.destroy_pipeline(prepare, None);
                }
                if let Some(reorder) = p.reorder_pipeline {
                    self.device.raw.destroy_pipeline(reorder, None);
                }
            }
            self.device
                .raw
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.raw.destroy_pipeline_layout(self.layout, None);
            self.device
                .raw
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;
    use crate::{
        integrator::IntegratorSettings,
        scene::{Camera, Light, Material, Mesh},
    };
    pub(super) fn fixture() -> Scene {
        Scene {
            curves: Default::default(),
            settings: IntegratorSettings {
                textures: crate::texture::TextureMode::Original,
                visibility: Default::default(),
                upload_chunk_mib: 16,
                diffuse_depth: 1,
                reflection_depth: 0,
                refraction_depth: 0,
                roulette_start: 96,
                reduce_analytic_caustics: false,
                ..Default::default()
            },
            camera: Camera {
                transform: glam::Mat4::from_translation(glam::Vec3::new(0., 0., 0.5))
                    .to_cols_array(),
                focal_length: 1.,
                horizontal_aperture: 0.0001,
                vertical_aperture: 0.0001,
                aperture_offset: [0.; 2],
                f_stop: 0.,
                focus_distance: 1.,
                near: 0.001,
                far: 100.,
                orthographic: true,
            },
            meshes: vec![Mesh {
                normals: Vec::new(),
                normal_indices: Vec::new(),
                reverse_orientation: false,
                geometry_light: false,
                light_response: [1.; 2],
                render_light_geometry: false,
                texcoords: Vec::new(),
                texcoord_indices: Vec::new(),
                points: vec![
                    [-10., -10., 0.],
                    [10., -10., 0.],
                    [10., 10., 0.],
                    [-10., 10., 0.],
                ],
                face_counts: vec![4],
                indices: vec![0, 1, 2, 3],
                color: [0.8; 3],
                material: Material::default(),
            }],
            lights: vec![Light::Area {
                diffuse: 1.,
                specular: 1.,
                single_sided: true,
                render_light_geometry: true,
                position: [0., 0., 1.],
                u: [2., 0., 0.],
                v: [0., -2., 0.],
                color: [1.; 3],
                intensity: 3.,
            }],
        }
    }
    pub(super) fn mean(renderer: &mut Renderer, scene: &Scene, samples: u32) -> f32 {
        renderer.set_scene(scene).unwrap();
        let mut image = Vec::new();
        while renderer.samples() < samples {
            image = renderer
                .render_batch(samples - renderer.samples(), 0.)
                .unwrap();
        }
        assert!(image.iter().all(|v| v.is_finite() && *v >= 0.));
        image.as_chunks::<4>().0.iter().map(|p| p[0]).sum::<f32>()
            / (renderer.width * renderer.height) as f32
    }
    #[test]
    #[ignore = "requires hardware ray queries; original/BC7/virtual material sampling, mips, HDR and opacity in both visibility routes"]
    fn material_textures_sample_gpu_images_mips_hdr_and_visibility() {
        use crate::integrator::VisibilityMode;
        use crate::texture::{ColorSpace, TextureBinding, TextureMode};
        let root = std::env::temp_dir().join(format!("lava-texture-gpu-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("pattern.png");
        let image = image::RgbaImage::from_fn(8, 8, |x, y| {
            image::Rgba(if (x + y) % 2 == 0 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 0, 255]
            })
        });
        image.save(&path).unwrap();
        let hdr = root.join("hdr.exr");
        image::Rgb32FImage::from_pixel(3, 5, image::Rgb([4., 0.25, 2.]))
            .save(&hdr)
            .unwrap();
        let opacity = root.join("opacity.png");
        image::RgbaImage::from_fn(8, 4, |x, _| {
            image::Rgba(if x < 4 {
                [0u8, 0, 0, 255]
            } else {
                [255, 255, 255, 255]
            })
        })
        .save(&opacity)
        .unwrap();
        for visibility in [VisibilityMode::RayQuery, VisibilityMode::Raster] {
            let mut renderer = Renderer::new(32, 32, Mode::Final, None).unwrap();
            renderer.linear_output = true;
            for mode in [
                TextureMode::Original,
                TextureMode::Compressed,
                TextureMode::Virtual,
            ] {
                let mut scene = fixture();
                scene.settings.visibility = visibility;
                scene.settings.textures = mode;
                scene.settings.diffuse_depth = 0;
                scene.settings.reflection_depth = 0;
                scene.settings.refraction_depth = 0;
                scene.camera.horizontal_aperture = 4.;
                scene.camera.vertical_aperture = 4.;
                scene.lights.clear();
                let mesh = &mut scene.meshes[0];
                mesh.points = vec![[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.], [-2., 2., 0.]];
                mesh.texcoords = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
                mesh.material.openpbr = Some(crate::material::OpenPbrSurface {
                    base_weight: 0.,
                    specular_weight: 0.,
                    emission_luminance: 1.,
                    emission_color: [0.25, 0.5, 0.75],
                    ..Default::default()
                });
                let binding = TextureBinding {
                    file: if mode == TextureMode::Virtual {
                        root.join("not-created.png")
                    } else {
                        path.clone()
                    },
                    color_space: ColorSpace::Raw,
                    uv_scale: [128.; 2],
                    ..Default::default()
                };
                mesh.material
                    .textures
                    .insert("emission_color".into(), binding);
                renderer.set_scene(&scene).unwrap();
                let pixels = renderer.render_batch(8, 0.).unwrap();
                let expected = if mode == TextureMode::Virtual {
                    [0.25, 0.5, 0.75]
                } else {
                    [0.5, 0., 0.]
                };
                for p in pixels.as_chunks::<4>().0 {
                    for c in 0..3 {
                        assert!(
                            (p[c] - expected[c]).abs() < 0.025,
                            "mip sample {p:?} vs {expected:?} {mode:?} {visibility:?}"
                        );
                    }
                }
                let formats = renderer.textures.resident_formats();
                if mode == TextureMode::Virtual {
                    assert!(formats.is_empty());
                    continue;
                }
                assert_eq!(
                    formats,
                    vec![(
                        if mode == TextureMode::Compressed {
                            vk::Format::BC7_UNORM_BLOCK
                        } else {
                            vk::Format::R32G32B32A32_SFLOAT
                        },
                        4
                    )]
                );
                scene.meshes[0]
                    .material
                    .textures
                    .get_mut("emission_color")
                    .unwrap()
                    .file = hdr.clone();
                renderer.set_scene(&scene).unwrap();
                let pixels = renderer.render_batch(8, 0.).unwrap();
                for p in pixels.as_chunks::<4>().0 {
                    for (c, want) in [4., 0.25, 2.].iter().enumerate() {
                        assert!(
                            (p[c] - want).abs() < 0.025,
                            "HDR texture clipped: {p:?} {mode:?}"
                        );
                    }
                }
                scene.meshes[0].material.textures.clear();
                scene.meshes[0]
                    .material
                    .openpbr
                    .as_mut()
                    .unwrap()
                    .emission_color = [1.; 3];
                scene.meshes[0].material.textures.insert(
                    "geometry_opacity".into(),
                    TextureBinding {
                        file: opacity.clone(),
                        color_space: ColorSpace::Raw,
                        channel: "r".into(),
                        ..Default::default()
                    },
                );
                renderer.set_scene(&scene).unwrap();
                let pixels = renderer.render_batch(8, 0.).unwrap();
                let left = pixels[(16 * 32 + 6) * 4];
                let right = pixels[(16 * 32 + 26) * 4];
                assert!(
                    left.abs() < 1e-5 && (right - 1.).abs() < 1e-5,
                    "textured cutout {left}/{right} {visibility:?} {mode:?}"
                );
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    #[ignore = "requires hardware ray queries; sparse UDIM table, per-tile formats/mips/HDR and fallback in both visibility paths"]
    fn udim_handles_resolve_sparse_mixed_tiles_and_virtual_placeholders() {
        use crate::integrator::VisibilityMode;
        use crate::texture::{ColorSpace, TextureBinding, TextureHandle, TextureMode};
        let root = std::env::temp_dir().join(format!("lava-udim-gpu-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let pattern = root.join("asset.<UDIM>.exr");
        image::Rgb32FImage::from_pixel(7, 3, image::Rgb([4., 0.25, 0.5]))
            .save(root.join("asset.1001.exr"))
            .unwrap();
        image::Rgb32FImage::from_pixel(9, 7, image::Rgb([0., 0.125, 2.]))
            .save(root.join("asset.1011.exr"))
            .unwrap();
        for (id, value) in [(1002, -0.5f32), (1013, 0.25)] {
            use exr::prelude::*;
            let channels = SpecificChannels::build()
                .with_channel("Y")
                .with_pixel_fn(|_| (value,));
            Image::from_channels((4, 8), channels)
                .write()
                .to_file(root.join(format!("asset.{id}.exr")))
                .unwrap();
        }
        for visibility in [VisibilityMode::RayQuery, VisibilityMode::Raster] {
            let mut renderer = Renderer::new(48, 32, Mode::Final, None).unwrap();
            renderer.linear_output = true;
            for mode in [
                TextureMode::Original,
                TextureMode::Compressed,
                TextureMode::Virtual,
            ] {
                let mut scene = fixture();
                scene.settings.visibility = visibility;
                scene.settings.textures = mode;
                scene.settings.diffuse_depth = 0;
                scene.settings.reflection_depth = 0;
                scene.settings.refraction_depth = 0;
                scene.lights.clear();
                scene.camera.horizontal_aperture = 6.;
                scene.camera.vertical_aperture = 4.;
                let mesh = &mut scene.meshes[0];
                mesh.points = vec![[-3., -2., 0.], [3., -2., 0.], [3., 2., 0.], [-3., 2., 0.]];
                mesh.texcoords = vec![[0., 0.], [3., 0.], [3., 2.], [0., 2.]];
                mesh.material.openpbr = Some(crate::material::OpenPbrSurface {
                    base_weight: 0.,
                    specular_weight: 0.,
                    emission_luminance: 1.,
                    emission_color: [0.125, 0.25, 0.375],
                    ..Default::default()
                });
                mesh.material.textures.insert(
                    "emission_color".into(),
                    TextureBinding {
                        file: if mode == TextureMode::Virtual {
                            root.join("no_directory/asset.<UDIM>.exr")
                        } else {
                            if mode == TextureMode::Compressed {
                                root.join("asset.%(UDIM)d.exr")
                            } else {
                                pattern.clone()
                            }
                        },
                        color_space: ColorSpace::Raw,
                        bias: [1., 1., 1., 0.],
                        ..Default::default()
                    },
                );
                mesh.material.textures.insert(
                    "base_color".into(),
                    TextureBinding {
                        file: root.join("asset.1001.exr"),
                        color_space: ColorSpace::Raw,
                        ..Default::default()
                    },
                );
                renderer.set_scene(&scene).unwrap();
                let pixels = renderer.render_batch(4, 0.).unwrap();
                let handle = renderer.textures.handle(0, 30).unwrap();
                let plain = renderer.textures.handle(0, 2).unwrap();
                if mode == TextureMode::Virtual {
                    assert_eq!(
                        handle.flags,
                        TextureHandle::UDIM | TextureHandle::VIRTUAL | TextureHandle::SOLID
                    );
                    assert_eq!(
                        plain.flags,
                        TextureHandle::PLAIN | TextureHandle::VIRTUAL | TextureHandle::SOLID
                    );
                    assert!(renderer.textures.resident_formats().is_empty());
                } else {
                    assert_eq!(handle.flags, TextureHandle::UDIM);
                    assert_eq!(handle.tile_count, 4);
                    assert_eq!(plain.flags, TextureHandle::PLAIN);
                    assert_eq!(plain.tile_count, 1);
                    assert_ne!(handle.table_offset, plain.table_offset);
                    let formats = renderer.textures.resident_formats();
                    assert_eq!(formats.len(), 4, "plain/UDIM shared image must deduplicate");
                    if mode == TextureMode::Compressed {
                        assert!(formats.contains(&(vk::Format::BC4_SNORM_BLOCK, 4)));
                        assert!(formats.contains(&(vk::Format::BC4_UNORM_BLOCK, 4)));
                        assert!(formats.contains(&(vk::Format::BC7_UNORM_BLOCK, 3)));
                    }
                }
                for (u, v, want) in [
                    (0, 0, [5., 1.25, 1.5]),
                    (1, 0, [0.5; 3]),
                    (2, 0, [0.125, 0.25, 0.375]),
                    (0, 1, [1., 1.125, 3.]),
                    (1, 1, [0.125, 0.25, 0.375]),
                    (2, 1, [1.25; 3]),
                ] {
                    let want = if mode == TextureMode::Virtual {
                        [0.125, 0.25, 0.375]
                    } else {
                        want
                    };
                    let pixel = &pixels[((24 - v * 16) * 48 + 8 + u * 16) * 4..][..3];
                    assert!(
                        pixel.iter().zip(want).all(|(a, b)| (*a - b).abs() < 0.04),
                        "UDIM ({u},{v}) {pixel:?} vs {want:?} {mode:?} {visibility:?}"
                    );
                }
                for offset in [[-1., -1.], [10., 0.]] {
                    scene.meshes[0]
                        .material
                        .textures
                        .get_mut("emission_color")
                        .unwrap()
                        .uv_offset = offset;
                    renderer.set_scene(&scene).unwrap();
                    let pixels = renderer.render_batch(4, 0.).unwrap();
                    let pixel = &pixels[(24 * 48 + 8) * 4..][..3];
                    assert!(
                        pixel
                            .iter()
                            .zip([0.125, 0.25, 0.375])
                            .all(|(a, b)| (*a - b).abs() < 0.001),
                        "invalid tile coordinates alias another tile"
                    );
                }
                scene.meshes[0].material.textures.clear();
                scene.meshes[0]
                    .material
                    .openpbr
                    .as_mut()
                    .unwrap()
                    .emission_color = [1.; 3];
                scene.meshes[0].material.textures.insert(
                    "geometry_opacity".into(),
                    TextureBinding {
                        file: if mode == TextureMode::Compressed {
                            root.join("asset.%(UDIM)d.exr")
                        } else {
                            pattern.clone()
                        },
                        color_space: ColorSpace::Raw,
                        channel: "r".into(),
                        ..Default::default()
                    },
                );
                renderer.set_scene(&scene).unwrap();
                let pixels = renderer.render_batch(4, 0.).unwrap();
                for (x, expected) in [
                    (8, 1.),
                    (24, if mode == TextureMode::Virtual { 1. } else { 0. }),
                    (40, 1.),
                ] {
                    assert!(
                        (pixels[(24 * 48 + x) * 4] - expected).abs() < 0.001,
                        "UDIM opacity {mode:?} {visibility:?}"
                    );
                }
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    #[ignore = "requires hardware ray queries; R32/BC4 UNORM/SNORM sampling and grayscale swizzles/mips"]
    fn grayscale_texture_gpu_formats_signed_values_and_mips() {
        use crate::integrator::VisibilityMode;
        use crate::texture::{ColorSpace, TextureBinding, TextureMode};
        let root = std::env::temp_dir().join(format!("lava-bc4-gpu-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let png = root.join("gray.png");
        image::GrayImage::from_fn(8, 4, |x, _| image::Luma([if x < 4 { 64u8 } else { 192 }]))
            .save(&png)
            .unwrap();
        let signed = root.join("signed.exr");
        let hdr = root.join("hdr.exr");
        let ramp = root.join("ramp.exr");
        for (path, left, right) in [(&signed, -3., 2.), (&hdr, 0.25, 4.), (&ramp, -3., 3.)] {
            use exr::prelude::*;
            let channels = SpecificChannels::build()
                .with_channel("Y")
                .with_pixel_fn(|p| {
                    (if path == &ramp {
                        p.x() as f32 / 7. * 6. - 3.
                    } else if p.x() < 4 {
                        left
                    } else {
                        right
                    },)
                });
            Image::from_channels((8, 4), channels)
                .write()
                .to_file(path)
                .unwrap();
        }
        for visibility in [VisibilityMode::RayQuery, VisibilityMode::Raster] {
            let mut renderer = Renderer::new(32, 32, Mode::Final, None).unwrap();
            renderer.linear_output = true;
            let mut ramp_reference = Vec::new();
            for mode in [TextureMode::Original, TextureMode::Compressed] {
                for (file, left, right, bias, snorm) in [
                    (&png, 64. / 255., 192. / 255., 0., false),
                    (&signed, -3., 2., 4., true),
                    (&hdr, 0.25, 4., 0., false),
                    (&ramp, -3., 3., 4., true),
                ] {
                    let mut scene = fixture();
                    scene.settings.visibility = visibility;
                    scene.settings.textures = mode;
                    scene.settings.diffuse_depth = 0;
                    scene.settings.reflection_depth = 0;
                    scene.settings.refraction_depth = 0;
                    scene.lights.clear();
                    scene.camera.horizontal_aperture = 4.;
                    scene.camera.vertical_aperture = 4.;
                    let mesh = &mut scene.meshes[0];
                    mesh.points = vec![[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.], [-2., 2., 0.]];
                    mesh.texcoords = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
                    mesh.material.openpbr = Some(crate::material::OpenPbrSurface {
                        base_weight: 0.,
                        specular_weight: 0.,
                        emission_luminance: 1.,
                        emission_color: [1.; 3],
                        ..Default::default()
                    });
                    mesh.material.textures.insert(
                        "emission_color".into(),
                        TextureBinding {
                            file: file.clone(),
                            color_space: ColorSpace::Raw,
                            bias: [bias, bias, bias, 0.],
                            ..Default::default()
                        },
                    );
                    renderer.set_scene(&scene).unwrap();
                    let pixels = renderer.render_batch(4, 0.).unwrap();
                    assert_eq!(
                        renderer.textures.resident_formats(),
                        vec![(
                            if mode == TextureMode::Original {
                                vk::Format::R32_SFLOAT
                            } else if snorm {
                                vk::Format::BC4_SNORM_BLOCK
                            } else {
                                vk::Format::BC4_UNORM_BLOCK
                            },
                            4
                        )]
                    );
                    if file == &ramp {
                        if mode == TextureMode::Original {
                            ramp_reference = pixels.clone();
                        } else {
                            assert!(
                                pixels
                                    .iter()
                                    .zip(&ramp_reference)
                                    .all(|(a, b)| (*a - *b).abs() < 0.45),
                                "hardware BC4 interpolation differs from float ramp"
                            );
                        }
                    } else {
                        for (x, expected) in [(6, left + bias), (26, right + bias)] {
                            let p = &pixels[(16 * 32 + x) * 4..][..3];
                            assert!(
                                p.iter().all(|v| (*v - expected).abs() < 0.04),
                                "gray {file:?} {mode:?} {visibility:?}: {p:?} vs {expected}"
                            );
                        }
                    }
                    scene.meshes[0]
                        .material
                        .textures
                        .get_mut("emission_color")
                        .unwrap()
                        .uv_scale = [128.; 2];
                    renderer.set_scene(&scene).unwrap();
                    let pixels = renderer.render_batch(4, 0.).unwrap();
                    let expected = (left + right) * 0.5 + bias;
                    for p in pixels.as_chunks::<4>().0 {
                        assert!(
                            p[..3].iter().all(|v| (*v - expected).abs() < 0.04),
                            "gray mip {p:?} vs {expected}"
                        );
                    }
                    scene.meshes[0]
                        .material
                        .textures
                        .get_mut("emission_color")
                        .unwrap()
                        .channel = "a".into();
                    scene.meshes[0]
                        .material
                        .textures
                        .get_mut("emission_color")
                        .unwrap()
                        .bias = [0.; 4];
                    renderer.set_scene(&scene).unwrap();
                    let pixels = renderer.render_batch(4, 0.).unwrap();
                    assert!(
                        pixels
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .all(|p| (p[0] - 1.).abs() < 1e-5),
                        "grayscale alpha must remain opaque"
                    );
                }
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    #[ignore = "requires hardware ray queries; transmission amount and RGB tint texture multipliers"]
    fn refraction_textures_match_constant_amount_and_tint() {
        use crate::integrator::VisibilityMode;
        use crate::material::OpenPbrSurface;
        use crate::texture::{ColorSpace, TextureBinding, TextureMode};
        let root = std::env::temp_dir().join(format!("lava-refraction-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let texture = root.join("tint.exr");
        image::Rgb32FImage::from_pixel(4, 4, image::Rgb([0.2, 0.6, 0.9]))
            .save(&texture)
            .unwrap();
        fn rgb(renderer: &mut Renderer, scene: &Scene) -> [f32; 3] {
            renderer.set_scene(scene).unwrap();
            let mut image = Vec::new();
            while renderer.samples() < 256 {
                image = renderer.render_batch(256 - renderer.samples(), 0.).unwrap();
            }
            std::array::from_fn(|c| {
                image.as_chunks::<4>().0.iter().map(|p| p[c]).sum::<f32>() / 64.
            })
        }
        for visibility in [VisibilityMode::RayQuery, VisibilityMode::Raster] {
            let mut renderer = Renderer::new(8, 8, Mode::Final, None).unwrap();
            renderer.linear_output = true;
            for mode in [TextureMode::Original, TextureMode::Compressed] {
                let mut scene = fixture();
                scene.settings.visibility = visibility;
                scene.settings.textures = mode;
                scene.settings.diffuse_depth = 0;
                scene.settings.reflection_depth = 0;
                scene.settings.refraction_depth = 1;
                scene.meshes[0].texcoords = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
                scene.lights = vec![Light::Area {
                    diffuse: 1.,
                    specular: 1.,
                    position: [0., 0., -1.],
                    u: [4., 0., 0.],
                    v: [0., 4., 0.],
                    color: [1.; 3],
                    intensity: 3.,
                    single_sided: true,
                    render_light_geometry: true,
                }];
                let base = OpenPbrSurface {
                    transmission_weight: 0.6,
                    transmission_color: [0.3, 0.5, 0.8],
                    specular_roughness: 0.,
                    ..Default::default()
                };
                let mut equivalent = base;
                equivalent.transmission_weight *= 0.6;
                equivalent.transmission_color = [0.3 * 0.2, 0.5 * 0.6, 0.8 * 0.9];
                scene.meshes[0].material.openpbr = Some(equivalent);
                let expected = rgb(&mut renderer, &scene);
                scene.meshes[0].material.openpbr = Some(base);
                for (field, channel, scale) in [
                    ("transmission_weight", "g", [base.transmission_weight; 4]),
                    ("transmission_color", "rgb", [0.3, 0.5, 0.8, 1.]),
                ] {
                    scene.meshes[0].material.textures.insert(
                        field.into(),
                        TextureBinding {
                            file: texture.clone(),
                            channel: channel.into(),
                            color_space: ColorSpace::Raw,
                            scale,
                            ..Default::default()
                        },
                    );
                }
                let actual = rgb(&mut renderer, &scene);
                for c in 0..3 {
                    assert!(expected[c] > 0.01);
                    assert!(
                        (actual[c] - expected[c]).abs() < 0.008 + expected[c] * 0.05,
                        "refraction {actual:?} expected {expected:?}, {visibility:?} {mode:?}"
                    );
                }
                // A zero authored amount must remain opaque despite a nonzero map.
                scene.meshes[0]
                    .material
                    .openpbr
                    .as_mut()
                    .unwrap()
                    .transmission_weight = 0.;
                scene.meshes[0]
                    .material
                    .textures
                    .get_mut("transmission_weight")
                    .unwrap()
                    .scale = [0.; 4];
                assert_eq!(rgb(&mut renderer, &scene), [0.; 3]);
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    #[ignore = "requires hardware ray queries; texture channels, multipliers and normal/bump shading frames"]
    fn texture_channels_multipliers_and_shading_frames() {
        use crate::integrator::VisibilityMode;
        use crate::texture::{ColorSpace, TextureBinding, TextureMode};
        let root = std::env::temp_dir().join(format!("lava-texture-frame-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let color = root.join("color.exr");
        image::Rgb32FImage::from_pixel(4, 4, image::Rgb([0.2, 0.6, 0.9]))
            .save(&color)
            .unwrap();
        let normal = root.join("normal.exr");
        image::Rgb32FImage::from_pixel(4, 4, image::Rgb([0.6, 0., 0.8]))
            .save(&normal)
            .unwrap();
        let unsigned = root.join("unsigned.exr");
        image::Rgb32FImage::from_pixel(4, 4, image::Rgb([0.8, 0.5, 0.9]))
            .save(&unsigned)
            .unwrap();
        let bump = root.join("bump.exr");
        image::Rgb32FImage::from_fn(64, 4, |x, _| image::Rgb([x as f32 / 63.; 3]))
            .save(&bump)
            .unwrap();
        std::fs::copy(&normal, root.join("normal.1001.exr")).unwrap();
        std::fs::copy(&bump, root.join("bump.1001.exr")).unwrap();
        for visibility in [VisibilityMode::RayQuery, VisibilityMode::Raster] {
            let mut renderer = Renderer::new(16, 16, Mode::Final, None).unwrap();
            renderer.linear_output = true;
            for mode in [TextureMode::Original, TextureMode::Compressed] {
                let mut scene = fixture();
                scene.settings.visibility = visibility;
                scene.settings.textures = mode;
                scene.settings.diffuse_depth = 1;
                scene.settings.reflection_depth = 0;
                scene.settings.refraction_depth = 0;
                scene.meshes[0].texcoords = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
                scene.meshes[0].material.openpbr = Some(crate::material::OpenPbrSurface {
                    base_weight: 0.,
                    specular_weight: 0.,
                    emission_luminance: 0.7,
                    emission_color: [1.; 3],
                    ..Default::default()
                });
                for (channel, expected) in [
                    ("r", 0.2),
                    ("g", 0.6),
                    ("b", 0.9),
                    ("luminance", 0.2 * 0.299 + 0.6 * 0.587 + 0.9 * 0.114),
                ] {
                    scene.meshes[0].material.textures.insert(
                        "emission_luminance".into(),
                        TextureBinding {
                            file: color.clone(),
                            color_space: ColorSpace::Raw,
                            channel: channel.into(),
                            multiply: true,
                            ..Default::default()
                        },
                    );
                    let value = mean(&mut renderer, &scene, 4);
                    assert!(
                        (value - 0.7 * expected).abs() < 0.015,
                        "channel/multiplier {channel}: {value} {mode:?}"
                    );
                }
                scene.meshes[0].material.textures.clear();
                scene.meshes[0].material.openpbr = Some(crate::material::OpenPbrSurface {
                    base_color: [0.8; 3],
                    specular_weight: 0.,
                    ..Default::default()
                });
                scene.lights = vec![Light::Directional {
                    diffuse: 1.,
                    specular: 1.,
                    angle: 0.,
                    direction: [0., 0., -1.],
                    color: [1.; 3],
                    intensity: std::f32::consts::PI,
                }];
                let flat = mean(&mut renderer, &scene, 16);
                for (field, file, strength, signed) in [
                    ("normal", normal.clone(), 1., true),
                    ("normal", unsigned.clone(), 1., false),
                    ("bump", bump.clone(), 15., false),
                    ("normal", root.join("normal.<UDIM>.exr"), 1., true),
                    ("bump", root.join("bump.<UDIM>.exr"), 15., false),
                ] {
                    scene.meshes[0].material.textures.clear();
                    scene.meshes[0].material.textures.insert(
                        field.into(),
                        TextureBinding {
                            file,
                            color_space: ColorSpace::Raw,
                            strength,
                            signed,
                            channel: if field == "bump" { "r" } else { "rgb" }.into(),
                            ..Default::default()
                        },
                    );
                    let tilted = mean(&mut renderer, &scene, 16);
                    assert!(
                        (tilted / flat - 0.8).abs() < 0.04,
                        "{field} frame {tilted}/{flat} {mode:?} {visibility:?}"
                    );
                    scene.meshes[0]
                        .material
                        .textures
                        .get_mut(field)
                        .unwrap()
                        .strength = 0.;
                    let zero = mean(&mut renderer, &scene, 16);
                    assert!(
                        (zero - flat).abs() < 0.01,
                        "zero {field} strength changed frame"
                    );
                }
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    #[ignore = "requires hardware ray queries; bump/normal grazing continuity and back-face frames"]
    fn mapped_normals_remain_continuous_when_orbiting_and_work_from_behind() {
        use crate::integrator::VisibilityMode;
        use crate::material::OpenPbrSurface;
        use crate::texture::{ColorSpace, TextureBinding};
        let root = std::env::temp_dir().join(format!("lava-normal-orbit-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let normal = root.join("normal.exr");
        let bump = root.join("bump.exr");
        image::Rgb32FImage::from_pixel(4, 4, image::Rgb([-0.8, 0., 0.6]))
            .save(&normal)
            .unwrap();
        image::Rgb32FImage::from_fn(64, 4, |x, _| image::Rgb([x as f32 / 63.; 3]))
            .save(&bump)
            .unwrap();
        fn camera(scene: &mut Scene, degrees: f32) {
            let mut transform = glam::Mat4::from_rotation_y(degrees.to_radians());
            transform.w_axis = (transform.z_axis.truncate() * 0.5).extend(1.);
            scene.camera.transform = transform.to_cols_array();
        }
        for visibility in [VisibilityMode::RayQuery, VisibilityMode::Raster] {
            let mut renderer = Renderer::new(8, 8, Mode::Final, None).unwrap();
            renderer.linear_output = true;
            let mut scene = fixture();
            scene.settings.visibility = visibility;
            scene.meshes[0].texcoords = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
            scene.meshes[0].material.openpbr = Some(OpenPbrSurface {
                base_color: [0.8; 3],
                specular_weight: 0.,
                geometry_thin_walled: true,
                ..Default::default()
            });
            scene.lights = vec![Light::Directional {
                diffuse: 1.,
                specular: 1.,
                angle: 0.,
                direction: [0., 0., -1.],
                color: [1.; 3],
                intensity: std::f32::consts::PI,
            }];
            for (field, file, strength, signed) in [
                ("normal", normal.clone(), 1., true),
                ("bump", bump.clone(), 26.25, false),
            ] {
                scene.meshes[0].material.textures.clear();
                scene.meshes[0].material.textures.insert(
                    field.into(),
                    TextureBinding {
                        file,
                        strength,
                        signed,
                        color_space: ColorSpace::Raw,
                        channel: if field == "bump" { "r" } else { "rgb" }.into(),
                        ..Default::default()
                    },
                );
                camera(&mut scene, 36.);
                let a = mean(&mut renderer, &scene, 16);
                camera(&mut scene, 38.);
                let b = mean(&mut renderer, &scene, 16);
                assert!(
                    a > 0.1 && b > 0.1 && (a - b).abs() < 0.04,
                    "orbit discontinuity in {field}: {a} -> {b}, {visibility:?}"
                );
            }
            image::Rgb32FImage::from_pixel(4, 4, image::Rgb([0.6, 0., 0.8]))
                .save(root.join("back.exr"))
                .unwrap();
            scene.meshes[0].material.textures.clear();
            camera(&mut scene, 180.);
            scene.lights = vec![Light::Directional {
                diffuse: 1.,
                specular: 1.,
                angle: 0.,
                direction: [0., 0., 1.],
                color: [1.; 3],
                intensity: std::f32::consts::PI,
            }];
            let flat = mean(&mut renderer, &scene, 16);
            scene.meshes[0].material.textures.insert(
                "normal".into(),
                TextureBinding {
                    file: root.join("back.exr"),
                    signed: true,
                    color_space: ColorSpace::Raw,
                    ..Default::default()
                },
            );
            let mapped = mean(&mut renderer, &scene, 16);
            assert!(
                flat > 0.1 && (mapped / flat - 0.8).abs() < 0.03,
                "back-facing normal map discarded: {mapped}/{flat}, {visibility:?}"
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    #[ignore = "requires hardware ray queries; authored shading normals, fallback and geometric-side rejection"]
    fn authored_shading_normals_change_lighting_in_both_visibility_modes() {
        use crate::integrator::VisibilityMode;
        for visibility in [VisibilityMode::RayQuery, VisibilityMode::Raster] {
            for native in [false, true] {
                let mut renderer = Renderer::new(16, 16, Mode::Final, None).unwrap();
                renderer.linear_output = true;
                let mut scene = fixture();
                scene.settings.visibility = visibility;
                scene.settings.diffuse_depth = 1;
                scene.settings.reflection_depth = 0;
                scene.settings.refraction_depth = 0;
                scene.lights = vec![Light::Directional {
                    diffuse: 1.,
                    specular: 1.,
                    angle: 0.,
                    direction: [0., 0., -1.],
                    color: [1.; 3],
                    intensity: std::f32::consts::PI,
                }];
                if native {
                    scene.meshes[0].material.openpbr = Some(crate::material::OpenPbrSurface {
                        base_color: [0.8; 3],
                        specular_weight: 0.,
                        ..Default::default()
                    });
                }
                let flat = mean(&mut renderer, &scene, 32);
                scene.meshes[0].normals = vec![[0.6, 0., 0.8]; 4];
                let smooth = mean(&mut renderer, &scene, 32);
                assert!(
                    (smooth / flat - 0.8).abs() < 0.025,
                    "normal cosine {smooth}/{flat}, native={native}"
                );
                scene.meshes[0].normals = vec![[0.; 3]; 4];
                let zero = mean(&mut renderer, &scene, 32);
                assert!(
                    (zero - flat).abs() < 1e-6,
                    "zero normals lost geometric fallback"
                );
                // Light below actual geometry must not leak through a tilted shading frame.
                scene.meshes[0].normals = vec![[0.8, 0., 0.6]; 4];
                scene.lights = vec![Light::Directional {
                    diffuse: 1.,
                    specular: 1.,
                    angle: 0.,
                    direction: [-0.9, 0., 0.1],
                    color: [1.; 3],
                    intensity: 100.,
                }];
                let blocked = mean(&mut renderer, &scene, 32);
                assert!(blocked < 0.03, "below-surface light leak: {blocked}");
                scene.meshes[0].normal_indices = vec![0; 4];
                scene.meshes[0].normals = vec![[0.6, 0., 0.8]];
                scene.lights = vec![Light::Directional {
                    diffuse: 1.,
                    specular: 1.,
                    angle: 0.,
                    direction: [0., 0., -1.],
                    color: [1.; 3],
                    intensity: std::f32::consts::PI,
                }];
                let indexed = mean(&mut renderer, &scene, 32);
                assert!((indexed - smooth).abs() < 1e-6);
            }
        }
    }
    #[test]
    #[ignore = "requires hardware ray queries; area light sidedness and primary visibility in both routes"]
    fn light_properties_preserve_illumination_and_control_primary_geometry() {
        for visibility in [
            crate::integrator::VisibilityMode::RayQuery,
            crate::integrator::VisibilityMode::Raster,
        ] {
            let mut renderer = Renderer::new(8, 8, Mode::Final, None).unwrap();
            renderer.linear_output = true;
            let mut scene = fixture();
            scene.settings.visibility = visibility;
            // Receiver sees the reverse side: single sided must be dark.
            if let Light::Area { v, .. } = &mut scene.lights[0] {
                v[1] = 2.;
            }
            let dark = mean(&mut renderer, &scene, 128);
            if let Light::Area { single_sided, .. } = &mut scene.lights[0] {
                *single_sided = false;
            }
            let lit = mean(&mut renderer, &scene, 128);
            assert!(
                dark.abs() < 1e-5 && lit > 0.1,
                "reverse emission {dark} vs {lit}"
            );
            if let Light::Area {
                render_light_geometry,
                ..
            } = &mut scene.lights[0]
            {
                *render_light_geometry = false;
            }
            let hidden = mean(&mut renderer, &scene, 128);
            assert!(
                (hidden - lit).abs() < 1e-6,
                "primary hiding changed illumination {hidden} vs {lit}"
            );
            // Put the source between camera and receiver.
            if let Light::Area {
                position,
                render_light_geometry,
                ..
            } = &mut scene.lights[0]
            {
                position[2] = 0.25;
                *render_light_geometry = true;
            }
            let source = mean(&mut renderer, &scene, 128);
            if let Light::Area {
                render_light_geometry,
                ..
            } = &mut scene.lights[0]
            {
                *render_light_geometry = false;
            }
            let receiver = mean(&mut renderer, &scene, 128);
            assert!(
                (source - 3.).abs() < 0.01 && receiver < source * 0.9,
                "primary emitter {source} vs receiver {receiver}"
            );
            // Ordinary emissive materials stay visible; only marked geometry lights hide.
            scene.lights.clear();
            let mesh = &mut scene.meshes[0];
            mesh.material.openpbr = Some(crate::material::OpenPbrSurface {
                emission_luminance: 2.,
                emission_color: [1.; 3],
                ..Default::default()
            });
            let ordinary = mean(&mut renderer, &scene, 4);
            scene.meshes[0].geometry_light = true;
            let hidden = mean(&mut renderer, &scene, 4);
            scene.meshes[0].render_light_geometry = true;
            let shown = mean(&mut renderer, &scene, 4);
            assert!(ordinary > 1.9 && hidden.abs() < 1e-5 && (ordinary - shown).abs() < 1e-6);
        }
    }
    #[test]
    #[ignore = "requires hardware ray queries; validates statistical MIS/RR energy and lobe depth limits"]
    fn mis_rr_depth_limits_and_gpu_compaction_match_reference_energy() {
        let mut renderer = Renderer::new(16, 16, Mode::Final, None).unwrap();
        renderer.linear_output = true;
        let mut scene = fixture();
        // Deterministic area quadrature of projected solid angle / pi.
        let mut probability = 0f64;
        let n = 256;
        for x in 0..n {
            for y in 0..n {
                let u = -1. + (x as f64 + 0.5) * 2. / n as f64;
                let v = -1. + (y as f64 + 0.5) * 2. / n as f64;
                probability +=
                    4. / ((n * n) as f64 * std::f64::consts::PI * (1. + u * u + v * v).powi(2));
            }
        }
        let expected = 0.8 * (3. * probability) as f32;
        let mis = mean(&mut renderer, &scene, 512);
        assert!(
            (mis - expected).abs() < expected * 0.02,
            "MIS {mis} expected {expected}"
        );
        scene.settings.roulette_start = 0;
        let rr = mean(&mut renderer, &scene, 512);
        assert!(
            (rr - expected).abs() < expected * 0.02,
            "roulette biased energy: {rr} expected {expected}"
        );
        // Only explicit diagnostic metadata (32 bytes), never path/image data.
        let staging = Buffer::new(
            renderer.device.clone(),
            32,
            vk::BufferUsageFlags::TRANSFER_DST,
            true,
        )
        .unwrap();
        renderer
            .device
            .submit(|cmd| unsafe {
                let b = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
                renderer.device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &b,
                    &[],
                    &[],
                );
                renderer.device.raw.cmd_copy_buffer(
                    cmd,
                    renderer.passes[1].scratch[2].raw,
                    staging.raw,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: 32,
                    }],
                );
                let b = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                renderer.device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &b,
                    &[],
                    &[],
                );
            })
            .unwrap();
        let words = staging.read_floats(8).unwrap();
        let peak = words[3].to_bits();
        assert!(
            peak > 0 && peak < 256,
            "ray queue did not compact roulette survivors: {peak}"
        );
        scene.settings.roulette_start = 96;
        scene.lights.push(Light::Point {
            diffuse: 1.,
            specular: 1.,
            position: [0., 0., 1.],
            color: [1.; 3],
            intensity: 2.,
        });
        let mixture = mean(&mut renderer, &scene, 512);
        let mixed_expected = expected + 0.8 * 2. / std::f32::consts::PI;
        assert!(
            (mixture - mixed_expected).abs() < mixed_expected * 0.025,
            "light PMF/delta MIS: {mixture} expected {mixed_expected}"
        );
        scene.lights.pop();
        scene.settings.diffuse_depth = 0;
        assert_eq!(mean(&mut renderer, &scene, 1), 0.);
        scene.meshes[0].material.metallic = 1.;
        scene.settings.reflection_depth = 1;
        let reflected = mean(&mut renderer, &scene, 32);
        assert!(
            (reflected - 2.4).abs() < 1e-4,
            "mirror lobe failed: {reflected}"
        );
        scene.settings.reflection_depth = 0;
        assert_eq!(mean(&mut renderer, &scene, 1), 0.);
        scene.meshes[0].material = Material {
            textures: Default::default(),
            openpbr: None,
            metallic: 0.,
            transmission: 1.,
            ior: 1.5,
        };
        scene.lights = vec![Light::Area {
            diffuse: 1.,
            specular: 1.,
            single_sided: true,
            render_light_geometry: true,
            position: [0., 0., -1.],
            u: [2., 0., 0.],
            v: [0., 2., 0.],
            color: [1.; 3],
            intensity: 3.,
        }];
        scene.settings.refraction_depth = 1;
        let transmitted = mean(&mut renderer, &scene, 512);
        let glass_expected = 0.96 * 0.8 * 3. / (1.5 * 1.5);
        assert!(
            (transmitted - glass_expected).abs() < glass_expected * 0.015,
            "dielectric lobe failed: {transmitted} expected {glass_expected}"
        );
        scene.settings.refraction_depth = 0;
        assert_eq!(mean(&mut renderer, &scene, 1), 0.);
        println!("MIS={mis}, RR={rr}, reference={expected}, mixed={mixture}, mirror={reflected}, glass={transmitted}, compacted peak={peak}/256");
    }
    #[test]
    #[ignore = "requires hardware ray queries; visibility equivalence and isolated GPU timestamp comparison"]
    fn curve_polygon_first_visibility_matches_unified_and_reports_timings() {
        let mut renderer = Renderer::new(128, 128, Mode::Final, None).unwrap();
        let device = renderer.device.clone();
        let code = ash::util::read_spv(&mut std::io::Cursor::new(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/visibility.spv"
        ))))
        .unwrap();
        let (reference, queries) = unsafe {
            let module = device
                .raw
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .unwrap();
            let entries = [vk::SpecializationMapEntry::default()
                .constant_id(50)
                .offset(0)
                .size(4)];
            let enabled = 1u32.to_ne_bytes();
            let specialization = vk::SpecializationInfo::default()
                .map_entries(&entries)
                .data(&enabled);
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .specialization_info(&specialization);
            let pipeline = device
                .raw
                .create_compute_pipelines(
                    renderer.pipeline_cache.raw,
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(renderer.layout)],
                    None,
                )
                .unwrap()[0];
            device.raw.destroy_shader_module(module, None);
            let queries = device
                .raw
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )
                .unwrap();
            (pipeline, queries)
        };
        let props = unsafe {
            device
                .instance
                .raw
                .get_physical_device_properties(device.physical)
        };
        let families = unsafe {
            device
                .instance
                .raw
                .get_physical_device_queue_family_properties(device.physical)
        };
        let bits = families[device.queue_family as usize].timestamp_valid_bits;
        assert!(bits > 0);
        let tick_mask = u64::MAX >> (64 - bits);
        let measure = |renderer: &Renderer, pipeline, repeats| {
            let mut push: Push = bytemuck::Zeroable::zeroed();
            push.width = renderer.width;
            push.height = renderer.height;
            push.mesh_triangles = renderer.mesh_triangles;
            push.curve_count = renderer.curve_count;
            push.sampling_mode = 1;
            push.target_samples = 16;
            device
                .submit(|cmd| unsafe {
                    device.raw.cmd_reset_query_pool(cmd, queries, 0, 2);
                    device.raw.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        renderer.layout,
                        0,
                        &[renderer.set],
                        &[],
                    );
                    device.raw.cmd_push_constants(
                        cmd,
                        renderer.layout,
                        device.shader_stages(),
                        0,
                        bytemuck::bytes_of(&push),
                    );
                    device
                        .raw
                        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                    device.raw.cmd_write_timestamp(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        queries,
                        0,
                    );
                    for _ in 0..repeats {
                        device.raw.cmd_dispatch(
                            cmd,
                            renderer.width.div_ceil(8),
                            renderer.height.div_ceil(8),
                            1,
                        );
                        device.raw.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[vk::MemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)],
                            &[],
                            &[],
                        );
                    }
                    device.raw.cmd_write_timestamp(
                        cmd,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        queries,
                        1,
                    );
                })
                .unwrap();
            let mut ticks = [0u64; 2];
            unsafe {
                device
                    .raw
                    .get_query_pool_results(
                        queries,
                        0,
                        &mut ticks,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .unwrap();
            }
            (ticks[1].wrapping_sub(ticks[0]) & tick_mask) as f64
                * props.limits.timestamp_period as f64
                / (1e6 * repeats as f64)
        };
        // A single object containing a dense, layered bundle of 4096 segments.
        let mut scene = fixture();
        scene.lights.clear();
        scene.camera.transform =
            glam::Mat4::from_translation(glam::Vec3::new(0., 0., 3.)).to_cols_array();
        scene.camera.horizontal_aperture = 2.4;
        scene.camera.vertical_aperture = 2.4;
        let mut points = Vec::new();
        for z in 0..64 {
            for x in 0..64 {
                let px = (x as f32 - 31.5) / 32.;
                let pz = -0.2 - z as f32 * 0.02;
                points.extend([[px, -1., pz], [px, 1., pz]]);
            }
        }
        Arc::make_mut(&mut scene.curves).push(
            serde_json::from_value(serde_json::json!({
                "points":points,"counts":vec![2;4096],"radius":0.012
            }))
            .unwrap(),
        );
        for scenario in ["occluded", "exposed", "cutout_dof"] {
            if scenario == "exposed" {
                for p in &mut scene.meshes[0].points {
                    p[2] = -3.;
                }
            }
            if scenario == "cutout_dof" {
                for p in &mut scene.meshes[0].points {
                    p[2] = 0.;
                }
                scene.meshes[0].material.openpbr = Some(crate::material::OpenPbrSurface {
                    geometry_opacity: 0.35,
                    ..Default::default()
                });
                scene.camera.orthographic = false;
                scene.camera.f_stop = 2.;
                scene.camera.focus_distance = 3.;
            }
            renderer.set_scene(&scene).unwrap();
            renderer.refresh_textures().unwrap();
            // Warm both products before timing and compare exactly the same rays.
            measure(&renderer, renderer.passes[0].pipeline, 1);
            let unified = visibility_snapshot(&renderer);
            measure(&renderer, reference, 1);
            let bounded = visibility_snapshot(&renderer);
            for (a, b) in bounded.iter().zip(&unified) {
                assert_eq!(
                    a[0].to_bits(),
                    b[0].to_bits(),
                    "primitive mismatch {scenario}"
                );
                for k in 1..4 {
                    assert!(
                        (a[k] - b[k]).abs() < 1e-4,
                        "hit mismatch {scenario}: {a:?}/{b:?}"
                    );
                }
                assert_eq!(
                    &a[4..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    &b[4..].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            // Alternate order and take medians to reduce clock/order bias.
            let mut unified_times = Vec::new();
            let mut bounded_times = Vec::new();
            for round in 0..9 {
                if round % 2 == 0 {
                    unified_times.push(measure(&renderer, renderer.passes[0].pipeline, 16));
                    bounded_times.push(measure(&renderer, reference, 16));
                } else {
                    bounded_times.push(measure(&renderer, reference, 16));
                    unified_times.push(measure(&renderer, renderer.passes[0].pipeline, 16));
                }
            }
            unified_times.sort_by(f64::total_cmp);
            bounded_times.sort_by(f64::total_cmp);
            let before = unified_times[4];
            let after = bounded_times[4];
            println!(
                "curve visibility {scenario}: unified={before:.3} ms polygon-first={after:.3} ms"
            );
        }
        unsafe {
            device.raw.destroy_pipeline(reference, None);
            device.raw.destroy_query_pool(queries, None);
        }
    }
    #[test]
    #[ignore = "requires hardware Vulkan; mesh refits retain static curve resources and match rebuilds"]
    fn mesh_deformation_refits_preserve_static_resources_and_match_rebuilds() {
        for visibility in [
            crate::integrator::VisibilityMode::RayQuery,
            crate::integrator::VisibilityMode::Raster,
        ] {
            let mut scene = fixture();
            scene.settings.visibility = visibility;
            scene.camera.horizontal_aperture = 4.;
            scene.camera.vertical_aperture = 3.;
            scene.camera.transform[14] = 5.;
            scene.meshes[0].points =
                vec![[-1., -1., 0.], [0., -1., 0.], [0., 1., 0.], [-1., 1., 0.]];
            let mut other = scene.meshes[0].clone();
            for p in &mut other.points {
                p[0] += 1.5;
            }
            other.color = [0.1, 0.8, 0.2];
            scene.meshes.push(other);
            Arc::make_mut(&mut scene.curves).push(
                serde_json::from_value(serde_json::json!({
                    "points":[[-1.5,-1.,0.5],[-1.5,1.,0.5]],"counts":[2],"radius":0.1
                }))
                .unwrap(),
            );
            let mut renderer = Renderer::new(48, 32, Mode::Interactive, None).unwrap();
            renderer.set_scene(&scene).unwrap();
            let handles = (
                renderer.scene_buffers[0].raw,
                renderer.scene_buffers[3].raw,
                renderer.scene_buffers[4].raw,
                renderer.blas.as_ref().unwrap().raw,
                renderer.tlas.as_ref().unwrap().raw,
            );
            let curves = renderer.curve_blas.clone();
            let mut reference = Renderer::new(48, 32, Mode::Interactive, None).unwrap();
            let mut old_pixels = renderer.render_batch(1, 0.).unwrap();
            for frame in 1..=4 {
                let previous = scene.clone();
                assert!(Arc::ptr_eq(&previous.curves, &scene.curves));
                // Change the second object's bounds, testing the nonzero upload and material-ID offsets.
                for p in &mut scene.meshes[1].points {
                    p[0] -= 0.15;
                    p[2] += 0.1;
                }
                scene.meshes[1].normals = vec![[0., 0.2, 0.98]; 4];
                assert!(renderer.try_refit_meshes(&previous, &scene).unwrap());
                assert_eq!(
                    handles,
                    (
                        renderer.scene_buffers[0].raw,
                        renderer.scene_buffers[3].raw,
                        renderer.scene_buffers[4].raw,
                        renderer.blas.as_ref().unwrap().raw,
                        renderer.tlas.as_ref().unwrap().raw
                    )
                );
                assert!(Arc::ptr_eq(&curves[0], &renderer.curve_blas[0]));
                renderer.update_scene_parameters(&scene).unwrap();
                assert_eq!(renderer.samples(), 0);
                let pixels = renderer.render_batch(1, 0.).unwrap();
                reference.set_scene(&scene).unwrap();
                let expected = reference.render_batch(1, 0.).unwrap();
                let error = pixels
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    error < 0.0001,
                    "{visibility:?} frame {frame}: refit/rebuild error {error}"
                );
                assert_ne!(pixels, old_pixels, "deformation did not change the image");
                old_pixels = pixels;
            }
            let previous = scene.clone();
            scene.meshes[1].face_counts.push(3);
            scene.meshes[1].indices.extend([0, 1, 2]);
            assert!(!renderer.try_refit_meshes(&previous, &scene).unwrap());
            scene = previous.clone();
            Arc::make_mut(&mut scene.curves)[0].points[0][0] += 0.2;
            assert!(!renderer.try_refit_meshes(&previous, &scene).unwrap());
        }
    }
    #[test]
    #[ignore = "requires hardware ray queries; immutable per-object curve BLAS reuse"]
    fn unchanged_curve_objects_reuse_blas_across_scene_rebuilds() {
        let device = Device::new(Instance::new().unwrap(), None).unwrap();
        let mut scene = fixture();
        let curve: crate::curves::Curves = serde_json::from_value(serde_json::json!({
            "points": [[0.,-1.,0.], [0.,1.,0.]], "counts": [2], "radius": 0.1
        }))
        .unwrap();
        scene.curves = vec![curve.clone(), curve].into();
        for p in &mut Arc::make_mut(&mut scene.curves)[1].points {
            p[0] += 2.;
        }
        let first = Renderer::prepare_geometry(device.clone(), &scene, Mode::Interactive).unwrap();
        assert_eq!(first.curve_blas.len(), 2);
        // Changing polygon count also changes all curve primitive/material offsets.
        scene.meshes.push(scene.meshes[0].clone());
        Arc::make_mut(&mut scene.curves)[0].material.metallic = 0.7;
        Arc::make_mut(&mut scene.curves)[0].color = [0.1, 0.2, 0.3];
        let second = Renderer::prepare_geometry(device.clone(), &scene, Mode::Interactive).unwrap();
        assert!(Arc::ptr_eq(&first.curve_blas[0], &second.curve_blas[0]));
        assert!(Arc::ptr_eq(&first.curve_blas[1], &second.curve_blas[1]));
        Arc::make_mut(&mut scene.curves)[0].radius[0] *= 2.;
        let third = Renderer::prepare_geometry(device.clone(), &scene, Mode::Interactive).unwrap();
        assert!(!Arc::ptr_eq(&second.curve_blas[0], &third.curve_blas[0]));
        assert!(Arc::ptr_eq(&second.curve_blas[1], &third.curve_blas[1]));
        let final_geometry = Renderer::prepare_geometry(device, &scene, Mode::Final).unwrap();
        assert!(!Arc::ptr_eq(
            &third.curve_blas[1],
            &final_geometry.curve_blas[1]
        ));
        let released = Arc::downgrade(&third.curve_blas[0]);
        drop(third);
        assert!(
            released.upgrade().is_none(),
            "cache retained unused GPU memory"
        );
    }
    #[test]
    #[ignore = "requires hardware ray queries; polygon-first curve occlusion and cutouts"]
    fn polygon_depth_bounds_curves_without_hiding_front_or_cutout_hits() {
        let mut renderer = Renderer::new(8, 8, Mode::Final, None).unwrap();
        renderer.linear_output = true;
        let mut scene = fixture();
        scene.lights.clear();
        scene.settings.diffuse_depth = 0;
        Arc::make_mut(&mut scene.curves).push(
            serde_json::from_value(serde_json::json!({
                "points": [[0.,-1.,-1.], [0.,1.,-1.]], "counts": [2], "radius": 0.1,
                "material": {"openpbr": {"base_weight":0., "specular_weight":0.,
                    "emission_luminance":2., "emission_color":[1.,1.,1.]}}
            }))
            .unwrap(),
        );
        assert!(
            mean(&mut renderer, &scene, 1) < 0.05,
            "curve behind polygon leaked"
        );
        assert!(visibility_snapshot(&renderer)
            .iter()
            .all(|h| (1..=2).contains(&h[0].to_bits())));
        scene.meshes[0].material.openpbr = Some(crate::material::OpenPbrSurface {
            geometry_opacity: 0.,
            ..Default::default()
        });
        assert!(
            (mean(&mut renderer, &scene, 1) - 2.).abs() < 1e-4,
            "cutout occluded curve"
        );
        scene.meshes[0].material.openpbr = None;
        for p in &mut Arc::make_mut(&mut scene.curves)[0].points {
            p[2] = 0.25;
        }
        assert!(
            (mean(&mut renderer, &scene, 1) - 2.).abs() < 1e-4,
            "front curve was culled"
        );
        scene.camera.near = 0.4;
        assert!(
            mean(&mut renderer, &scene, 1) < 0.05,
            "near-clipped curve survived"
        );
    }
    #[test]
    #[ignore = "requires hardware ray queries and optional EXT mesh shaders; analytic curves in both visibility paths"]
    fn curves_render_round_surfaces_in_query_and_mesh_visibility() {
        let mut scene = fixture();
        scene.meshes.clear();
        scene.lights.clear();
        Arc::make_mut(&mut scene.curves).push(crate::curves::Curves {
            points: vec![[0., -1., 0.], [0., 1., 0.]],
            counts: vec![2],
            basis: "linear".into(),
            periodic: false,
            radius: vec![0.1, 0.2],
            radius_interpolation: "vertex".into(),
            subdivisions: 8,
            color: [0.5; 3],
            material: Material {
                openpbr: Some(crate::material::OpenPbrSurface {
                    base_weight: 0.,
                    specular_weight: 0.,
                    emission_luminance: 2.,
                    emission_color: [1., 0.5, 0.25],
                    ..Default::default()
                }),
                ..Default::default()
            },
        });
        let mut query_dof: Option<f32> = None;
        for visibility in [
            crate::integrator::VisibilityMode::RayQuery,
            crate::integrator::VisibilityMode::Raster,
        ] {
            scene.settings.visibility = visibility;
            let mut renderer = Renderer::new(8, 8, Mode::Final, None).unwrap();
            renderer.linear_output = true;
            renderer.coherent_camera_reference = true;
            let result = mean(&mut renderer, &scene, 4);
            assert!(
                (result - 2.).abs() < 1e-4,
                "curve missing: {result}, {visibility:?}"
            );
            scene.camera.f_stop = 0.5;
            scene.camera.focus_distance = 1.;
            let blurred = mean(&mut renderer, &scene, 16);
            assert!(
                blurred > 0.1 && blurred < 1.9,
                "curve depth of field missing: {blurred}"
            );
            if let Some(reference) = query_dof {
                assert!(
                    (blurred - reference).abs() < 0.01,
                    "curve raster/query DOF differ: {blurred}/{reference}"
                );
            } else {
                query_dof = Some(blurred);
            }
            scene.camera.f_stop = 0.;
            scene.camera.transform =
                glam::Mat4::from_translation(glam::Vec3::new(2., 0., 0.5)).to_cols_array();
            let outside = mean(&mut renderer, &scene, 4);
            assert!(
                outside < 0.02,
                "curve AABB accepted as a surface: {outside}"
            );
            scene.camera.transform =
                glam::Mat4::from_translation(glam::Vec3::new(0., 0., 0.5)).to_cols_array();
        }
    }
    #[test]
    #[ignore = "requires hardware ray queries; curve shadows and secondary analytic intersections"]
    fn curves_cast_shadows_and_remain_visible_to_secondary_rays_outside_camera_frustum() {
        for visibility in [
            crate::integrator::VisibilityMode::RayQuery,
            crate::integrator::VisibilityMode::Raster,
        ] {
            let mut renderer = Renderer::new(8, 8, Mode::Final, None).unwrap();
            renderer.linear_output = true;
            let mut scene = fixture();
            scene.settings.visibility = visibility;
            scene.lights = vec![Light::Point {
                diffuse: 1.,
                specular: 1.,
                position: [0., 0., 2.],
                color: [1.; 3],
                intensity: 4. * std::f32::consts::PI,
            }];
            let clear = mean(&mut renderer, &scene, 16);
            assert!(clear > 0.7);
            let curve:crate::curves::Curves=serde_json::from_value(serde_json::json!({"points":[[0.,-1.,1.],[0.,1.,1.]],"counts":[2],"radius":[0.01,0.2]})).unwrap();
            Arc::make_mut(&mut scene.curves).push(curve);
            let blocked = mean(&mut renderer, &scene, 16);
            assert!(
                blocked < clear * 0.1,
                "procedural curve shadow missing: {blocked}/{clear}"
            );
            for p in &mut Arc::make_mut(&mut scene.curves)[0].points {
                p[0] = 0.15;
            }
            let empty_box = mean(&mut renderer, &scene, 16);
            assert!(
                (empty_box - clear).abs() < 0.025,
                "AABB falsely shadowed tapered curve: {empty_box}/{clear}"
            );
            for p in &mut Arc::make_mut(&mut scene.curves)[0].points {
                p[0] = 0.;
            }
            scene.meshes[0].material.metallic = 1.;
            scene.meshes[0].color = [1.; 3];
            scene.settings.reflection_depth = 1;
            Arc::make_mut(&mut scene.curves)[0].material.openpbr =
                Some(crate::material::OpenPbrSurface {
                    base_weight: 0.,
                    specular_weight: 0.,
                    emission_luminance: 2.,
                    emission_color: [1., 0.5, 0.25],
                    ..Default::default()
                });
            scene.lights.clear();
            let reflected = mean(&mut renderer, &scene, 4);
            assert!(
                (reflected - 2.).abs() < 1e-4,
                "curve behind camera vanished from reflection: {reflected}"
            );
        }
    }
    #[test]
    #[ignore = "requires EXT mesh shaders; subpixel curve culling is viewport-only"]
    fn curve_subpixel_culling_is_interactive_only() {
        let mut scene = fixture();
        scene.meshes.clear();
        scene.lights.clear();
        scene.settings.visibility = crate::integrator::VisibilityMode::Raster;
        Arc::make_mut(&mut scene.curves).push(serde_json::from_value(serde_json::json!({"points":[[0.,-1.,0.],[0.,1.,0.]],"counts":[2],"radius":0.000001,"material":{"openpbr":{"base_weight":0.,"specular_weight":0.,"emission_luminance":2.,"emission_color":[1.,1.,1.]}}})).unwrap());
        let mut viewport = Renderer::new(1, 1, Mode::Interactive, None).unwrap();
        viewport.linear_output = true;
        if !viewport.device.mesh_shader {
            return;
        }
        let culled = mean(&mut viewport, &scene, 256);
        let mut final_frame = Renderer::new(1, 1, Mode::Final, None).unwrap();
        final_frame.linear_output = true;
        let retained = mean(&mut final_frame, &scene, 256);
        assert!(
            culled < 0.02 && retained > culled + 0.01,
            "viewport LOD {culled}, final thin strand {retained}"
        );
    }
    #[test]
    #[ignore = "requires hardware ray queries; analytic reflection clamping and accumulation reset"]
    fn analytic_reflection_clamping_is_opt_in_and_preserves_diffuse() {
        use crate::material::OpenPbrSurface;
        let mut renderer = Renderer::new(8, 8, Mode::Interactive, None).unwrap();
        renderer.linear_output = true;
        let mut scene = fixture();
        scene.settings.reflection_depth = 1;
        scene.meshes[0].material.openpbr = Some(OpenPbrSurface {
            base_weight: 0.,
            specular_roughness: 0.01,
            ..Default::default()
        });
        for light in [
            Light::Directional {
                diffuse: 1.,
                specular: 1.,
                angle: 0.,
                direction: [0., 0., -1.],
                color: [1.; 3],
                intensity: 1.,
            },
            Light::Point {
                diffuse: 1.,
                specular: 1.,
                position: [0., 0., 1.],
                color: [1.; 3],
                intensity: 1.,
            },
            Light::Spot {
                diffuse: 1.,
                specular: 1.,
                position: [0., 0., 1.],
                direction: [0., 0., -1.],
                color: [1.; 3],
                intensity: 1.,
                cone_degrees: 45.,
                softness: 0.,
            },
        ] {
            scene.lights = vec![light];
            scene.settings.reduce_analytic_caustics = false;
            let reference = mean(&mut renderer, &scene, 4);
            let tlas = renderer.tlas.as_ref().unwrap().raw;
            scene.settings.reduce_analytic_caustics = true;
            renderer.update_scene_parameters(&scene).unwrap();
            assert_eq!(renderer.samples(), 0);
            assert_eq!(renderer.tlas.as_ref().unwrap().raw, tlas);
            let mut image = Vec::new();
            while renderer.samples() < 4 {
                image = renderer.render_batch(4 - renderer.samples(), 0.).unwrap();
            }
            let limited = image.as_chunks::<4>().0.iter().map(|p| p[0]).sum::<f32>() / 64.;
            assert!(
                reference > 160.,
                "expected sharp analytic reflection: {reference}"
            );
            assert!(
                limited > 0. && limited <= 16.01 && limited < reference * 0.1,
                "reference={reference}, limited={limited}"
            );
        }
        scene.meshes[0].material.openpbr = Some(OpenPbrSurface {
            specular_weight: 0.,
            ..Default::default()
        });
        scene.settings.reduce_analytic_caustics = false;
        let diffuse = mean(&mut renderer, &scene, 16);
        scene.settings.reduce_analytic_caustics = true;
        assert!((mean(&mut renderer, &scene, 16) - diffuse).abs() < 1e-5);
        scene.lights.clear();
        scene.meshes[0].material.openpbr = Some(OpenPbrSurface {
            emission_luminance: 100.,
            emission_color: [1.; 3],
            ..Default::default()
        });
        scene.settings.reduce_analytic_caustics = false;
        let emission = mean(&mut renderer, &scene, 4);
        scene.settings.reduce_analytic_caustics = true;
        assert!(emission >= 100.);
        assert!((mean(&mut renderer, &scene, 4) - emission).abs() < 1e-4);
        // A bright emitter reflected in an ordinary mirror must not be capped
        // merely because its incident radiance exceeds the multiplier bound.
        scene = fixture();
        scene.settings.reflection_depth = 1;
        scene.meshes[0].material.metallic = 1.;
        scene.meshes[0].color = [1.; 3];
        if let Light::Area { intensity, .. } = &mut scene.lights[0] {
            *intensity = 100.;
        }
        let mirror = mean(&mut renderer, &scene, 4);
        scene.settings.reduce_analytic_caustics = true;
        assert!((mean(&mut renderer, &scene, 4) - mirror).abs() < 1e-4);
        assert!((mirror - 100.).abs() < 1e-4);
    }
    #[test]
    #[ignore = "requires hardware ray queries; OpenPBR energy, lobes, coverage, emission and depth tests"]
    fn openpbr_surface_energy_lobes_and_updates() {
        use crate::material::OpenPbrSurface;
        let mut renderer = Renderer::new(16, 16, Mode::Final, None).unwrap();
        renderer.linear_output = true;
        let mut scene = fixture();
        let legacy = mean(&mut renderer, &scene, 256);
        let mut p = OpenPbrSurface {
            base_color: [0.8; 3],
            specular_weight: 0.,
            ..Default::default()
        };
        scene.meshes[0].material.openpbr = Some(p);
        let diffuse = mean(&mut renderer, &scene, 512);
        assert!(
            (diffuse - legacy).abs() < legacy * 0.025,
            "OpenPBR diffuse MIS {diffuse}, Lambert reference {legacy}"
        );
        scene.settings.roulette_start = 0;
        let rr = mean(&mut renderer, &scene, 512);
        assert!(
            (rr - diffuse).abs() < diffuse * 0.025,
            "OpenPBR RR {rr}, reference {diffuse}"
        );
        scene.settings.roulette_start = 96;
        scene.settings.diffuse_depth = 0;
        assert_eq!(mean(&mut renderer, &scene, 1), 0.);
        scene.settings.reflection_depth = 1;
        p.base_metalness = 1.;
        p.specular_weight = 1.;
        p.specular_roughness = 0.05;
        scene.meshes[0].material.openpbr = Some(p);
        let smooth = mean(&mut renderer, &scene, 256);
        p.specular_roughness = 0.8;
        p.specular_roughness_anisotropy = 0.5;
        scene.meshes[0].material.openpbr = Some(p);
        let rough = mean(&mut renderer, &scene, 512);
        assert!(
            smooth > 0.1 && rough > 0.01 && (smooth - rough).abs() > 0.05,
            "roughness has no effect: {smooth} {rough}"
        );
        p.base_metalness = 0.;
        p.coat_weight = 0.8;
        p.coat_roughness = 0.2;
        p.fuzz_weight = 0.5;
        p.thin_film_weight = 1.;
        p.thin_film_thickness = 0.4;
        scene.settings.diffuse_depth = 1;
        scene.meshes[0].material.openpbr = Some(p);
        let layered = mean(&mut renderer, &scene, 128);
        assert!(layered > 0.01);
        scene.settings.reflection_depth = 0;
        scene.settings.diffuse_depth = 0;
        p = OpenPbrSurface {
            base_color: [1.; 3],
            transmission_weight: 1.,
            specular_roughness: 0.,
            ..Default::default()
        };
        scene.meshes[0].material.openpbr = Some(p);
        scene.lights = vec![Light::Area {
            diffuse: 1.,
            specular: 1.,
            single_sided: true,
            render_light_geometry: true,
            position: [0., 0., -1.],
            u: [2., 0., 0.],
            v: [0., 2., 0.],
            color: [1.; 3],
            intensity: 3.,
        }];
        assert_eq!(mean(&mut renderer, &scene, 1), 0.);
        scene.settings.refraction_depth = 1;
        let glass = mean(&mut renderer, &scene, 512);
        assert!(glass > 0.1, "OpenPBR transmission failed: {glass}");
        scene.settings.refraction_depth = 0;
        p = OpenPbrSurface {
            emission_color: [1., 0.5, 0.25],
            emission_luminance: 2.,
            ..Default::default()
        };
        scene.meshes[0].material.openpbr = Some(p);
        let emission = mean(&mut renderer, &scene, 1);
        assert!((emission - 2.).abs() < 0.001, "emission: {emission}");
        p.geometry_opacity = 0.;
        scene.meshes[0].material.openpbr = Some(p);
        let clear = mean(&mut renderer, &scene, 1);
        assert!(
            (clear - 3.).abs() < 0.0001,
            "coverage zero must reveal the area emitter: {clear}"
        );
        p.geometry_opacity = 0.5;
        scene.meshes[0].material.openpbr = Some(p);
        let cutout = mean(&mut renderer, &scene, 512);
        assert!(
            (cutout - 2.5).abs() < 0.025,
            "stochastic coverage energy: {cutout}"
        );
        println!("OpenPBR Lambert={diffuse}, RR={rr}, smooth={smooth}, rough={rough}, layered={layered}, glass={glass}, emission={emission}, clear={clear}");
    }
    #[test]
    #[ignore = "requires hardware ray queries; smooth polygon shadow terminator and retained occluders"]
    fn smooth_polygon_lighting_point_fixes_terminator_without_flat_mesh_leaks() {
        use crate::integrator::VisibilityMode;
        for visibility in [VisibilityMode::RayQuery, VisibilityMode::Raster] {
            for native in [false, true] {
                let mut renderer = Renderer::new(8, 8, Mode::Final, None).unwrap();
                renderer.linear_output = true;
                let mut scene = fixture();
                scene.settings.visibility = visibility;
                scene.camera.transform =
                    glam::Mat4::from_translation(glam::Vec3::new(0.5, 0., 0.5)).to_cols_array();
                let mesh = &mut scene.meshes[0];
                mesh.points = vec![[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]];
                if native {
                    mesh.material.openpbr = Some(crate::material::OpenPbrSurface {
                        base_color: [0.8; 3],
                        specular_weight: 0.,
                        ..Default::default()
                    });
                }
                scene.lights = vec![Light::Directional {
                    diffuse: 1.,
                    specular: 1.,
                    angle: 0.,
                    direction: [-1., 0., 0.15],
                    color: [1.; 3],
                    intensity: 2. * std::f32::consts::PI,
                }];
                let flat = mean(&mut renderer, &scene, 128);
                assert!(
                    flat < 0.025,
                    "flat geometry leaked light from below: {flat}"
                );
                // A coarse convex cylindrical patch, with smooth vertex normals.
                scene.meshes[0].normals = vec![
                    [-0.6, 0., 0.8],
                    [0.6, 0., 0.8],
                    [0.6, 0., 0.8],
                    [-0.6, 0., 0.8],
                ];
                let smooth = mean(&mut renderer, &scene, 128);
                assert!(smooth>0.25,"coarse smooth polygon still has a hard terminator: {smooth}, {visibility:?}, native={native}");
                let mut blocker = scene.meshes[0].clone();
                blocker.points = vec![
                    [2., -10., -10.],
                    [2., 10., -10.],
                    [2., 10., 10.],
                    [2., -10., 10.],
                ];
                blocker.normals.clear();
                scene.meshes.push(blocker);
                let shadow = mean(&mut renderer, &scene, 128);
                assert!(
                    shadow < smooth * 0.1,
                    "real occluder was lost: {shadow}/{smooth}"
                );
                scene.meshes.pop();
                scene.meshes[0].normals = vec![[0.6, 0., 0.8]; 4];
                let constant = mean(&mut renderer, &scene, 128);
                assert!(
                    constant < 0.025,
                    "constant tilted normals displaced a flat plane: {constant}"
                );
                // Keep true boundary crossing for glass, even with curved normals.
                scene.meshes[0].normals = vec![
                    [-0.6, 0., 0.8],
                    [0.6, 0., 0.8],
                    [0.6, 0., 0.8],
                    [-0.6, 0., 0.8],
                ];
                if native {
                    scene.meshes[0]
                        .material
                        .openpbr
                        .as_mut()
                        .unwrap()
                        .transmission_weight = 1.;
                } else {
                    scene.meshes[0].material.transmission = 1.;
                }
                assert!(
                    mean(&mut renderer, &scene, 128) < 0.025,
                    "transmissive boundary was displaced"
                );
            }
        }
    }
    #[test]
    #[ignore = "requires hardware ray queries; area-light shadow endpoints at grazing angles"]
    fn rectangular_light_grazing_shadow_energy_matches_quadrature() {
        let mut renderer = Renderer::new(16, 16, Mode::Final, None).unwrap();
        renderer.linear_output = true;
        let mut scene = fixture();
        let mut probability = 0f64;
        let n = 256;
        for x in 0..n {
            for y in 0..n {
                let u = 3. + (x as f64 + 0.5) * 2. / n as f64;
                let v = -1. + (y as f64 + 0.5) * 2. / n as f64;
                probability +=
                    4. / ((n * n) as f64 * std::f64::consts::PI * (1. + u * u + v * v).powi(2));
            }
        }
        let expected = 0.8 * (100. * probability) as f32;
        for native in [false, true] {
            if native {
                scene.meshes[0].material.openpbr = Some(crate::material::OpenPbrSurface {
                    base_color: [0.8; 3],
                    specular_weight: 0.,
                    ..Default::default()
                });
            }
            for translation in [glam::Vec3::ZERO, glam::Vec3::new(719., -88., -798.)] {
                scene.camera.transform =
                    glam::Mat4::from_translation(translation + glam::Vec3::new(0., 0., 0.5))
                        .to_cols_array();
                for (p, base) in scene.meshes[0].points.iter_mut().zip([
                    [-10., -10., 0.],
                    [10., -10., 0.],
                    [10., 10., 0.],
                    [-10., 10., 0.],
                ]) {
                    *p = (glam::Vec3::from_array(base) + translation).to_array();
                }
                scene.lights = vec![Light::Area {
                    diffuse: 1.,
                    specular: 1.,
                    single_sided: true,
                    render_light_geometry: true,
                    position: (translation + glam::Vec3::new(4., 0., 1.)).to_array(),
                    u: [2., 0., 0.],
                    v: [0., -2., 0.],
                    color: [1.; 3],
                    intensity: 100.,
                }];
                let actual = mean(&mut renderer, &scene, 512);
                assert!((actual-expected).abs()<expected*0.025,"rectangular emitter self-shadow: native={native}, translation={translation}, actual={actual}, reference={expected}");
            }
        }
    }
    #[test]
    #[ignore = "requires hardware ray queries; sampled emitter exclusion must preserve other shadows"]
    fn rectangular_light_shadow_keeps_other_emitters_and_geometry() {
        let mut renderer = Renderer::new(16, 16, Mode::Final, None).unwrap();
        renderer.linear_output = true;
        for native in [false, true] {
            let mut scene = fixture();
            if native {
                scene.meshes[0].material.openpbr = Some(crate::material::OpenPbrSurface {
                    base_color: [0.8; 3],
                    specular_weight: 0.,
                    ..Default::default()
                });
            }
            scene.lights = vec![Light::Area {
                diffuse: 1.,
                specular: 1.,
                single_sided: true,
                render_light_geometry: true,
                position: [4., 0., 1.],
                u: [2., 0., 0.],
                v: [0., -2., 0.],
                color: [1.; 3],
                intensity: 100.,
            }];
            let unblocked = mean(&mut renderer, &scene, 256);
            assert!(unblocked > 0.3);
            // This separate dark emitter covers every direct sample of the first
            // rectangle, but lies outside the primary camera ray.
            scene.lights.push(Light::Area {
                diffuse: 1.,
                specular: 1.,
                single_sided: true,
                render_light_geometry: true,
                position: [2., 0., 0.5],
                u: [2.5, 0., 0.],
                v: [0., -2., 0.],
                color: [0.; 3],
                intensity: 0.,
            });
            let emitter_blocked = mean(&mut renderer, &scene, 256);
            assert!(
                emitter_blocked < unblocked * 0.1,
                "other emitter shadow lost: {emitter_blocked}/{unblocked}, native={native}"
            );
            scene.lights.pop();
            scene.meshes.push(Mesh {
                normals: Vec::new(),
                normal_indices: Vec::new(),
                reverse_orientation: false,
                geometry_light: false,
                light_response: [1.; 2],
                render_light_geometry: false,
                points: vec![[1., -1., 0.5], [3., -1., 0.5], [3., 1., 0.5], [1., 1., 0.5]],
                face_counts: vec![4],
                indices: vec![0, 1, 2, 3],
                color: [0.; 3],
                material: Material::default(),
                texcoords: Vec::new(),
                texcoord_indices: Vec::new(),
            });
            let mesh_blocked = mean(&mut renderer, &scene, 256);
            assert!(
                mesh_blocked < unblocked * 0.1,
                "mesh shadow lost: {mesh_blocked}/{unblocked}, native={native}"
            );
        }
    }
    fn visibility_snapshot(renderer: &Renderer) -> Vec<[f32; 8]> {
        buffer_snapshot(renderer, &renderer.passes[0].output)
            .as_chunks::<8>()
            .0
            .to_vec()
    }
    pub(super) fn buffer_snapshot(renderer: &Renderer, source: &Buffer) -> Vec<f32> {
        let staging = Buffer::new(
            renderer.device.clone(),
            source.size,
            vk::BufferUsageFlags::TRANSFER_DST,
            true,
        )
        .unwrap();
        renderer
            .device
            .submit(|cmd| unsafe {
                renderer.device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
                    &[],
                    &[],
                );
                renderer.device.raw.cmd_copy_buffer(
                    cmd,
                    source.raw,
                    staging.raw,
                    &[vk::BufferCopy::default().size(source.size)],
                );
                renderer.device.raw.cmd_pipeline_barrier(
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
        staging.read_floats((source.size / 4) as usize).unwrap()
    }
    #[test]
    #[ignore = "requires hardware ray queries; exposure must not change history, filtered image, visibility or sample count"]
    fn exposure_reprocesses_existing_float_image_only() {
        for mode in [Mode::Interactive, Mode::Final] {
            let mut renderer = Renderer::new(16, 16, mode, None).unwrap();
            renderer.linear_output = true;
            let scene = fixture();
            renderer.set_scene(&scene).unwrap();
            let mut original = Vec::new();
            while renderer.samples() < 8 {
                original = renderer.render_batch(8 - renderer.samples(), 0.).unwrap();
            }
            let history = buffer_snapshot(&renderer, &renderer.passes[2].output);
            let filtered =
                buffer_snapshot(&renderer, renderer.passes[2].filtered.as_ref().unwrap());
            let visibility = visibility_snapshot(&renderer);
            for exposure in [1., -1., 0.] {
                let pixels = renderer.render_batch(0, exposure).unwrap();
                assert_eq!(renderer.samples(), 8);
                assert_eq!(
                    history,
                    buffer_snapshot(&renderer, &renderer.passes[2].output)
                );
                assert_eq!(
                    filtered,
                    buffer_snapshot(&renderer, renderer.passes[2].filtered.as_ref().unwrap())
                );
                assert_eq!(visibility, visibility_snapshot(&renderer));
                for (a, b) in original
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(pixels.as_chunks::<4>().0)
                {
                    for c in 0..3 {
                        assert!((a[c] * exposure.exp2() - b[c]).abs() < 1e-6);
                    }
                }
            }
        }
    }
    #[test]
    #[ignore = "requires graphics and ray queries; verify independent per-ray jitter versus coherent raster projection jitter"]
    fn camera_jitter_is_per_ray_only_in_ray_query_mode() {
        use crate::integrator::VisibilityMode;
        for mode in [Mode::Interactive, Mode::Final] {
            for visibility in [VisibilityMode::RayQuery, VisibilityMode::Raster] {
                let mut renderer = Renderer::new(16, 16, mode, None).unwrap();
                let mut scene = fixture();
                scene.settings.visibility = visibility;
                scene.camera.horizontal_aperture = 4.;
                scene.camera.vertical_aperture = 4.;
                scene.lights.clear();
                renderer.set_scene(&scene).unwrap();
                renderer
                    .render_batch(if mode == Mode::Final { 8 } else { 1 }, 0.)
                    .unwrap();
                let hits = visibility_snapshot(&renderer);
                let triangles = scene.validate().unwrap();
                let mut jitters = Vec::new();
                for (i, h) in hits.iter().enumerate() {
                    let primitive = h[0].to_bits();
                    assert!(primitive > 0);
                    let t = triangles[(primitive - 1) as usize];
                    let x = t.a[0] * (1. - h[1] - h[2]) + t.b[0] * h[1] + t.c[0] * h[2];
                    let y = t.a[1] * (1. - h[1] - h[2]) + t.b[1] * h[1] + t.c[1] * h[2];
                    jitters.push([
                        (x / 4. + 0.5) * 16. - (i % 16) as f32,
                        (-y / 4. + 0.5) * 16. - (i / 16) as f32,
                    ]);
                }
                for axis in 0..2 {
                    let lo = jitters
                        .iter()
                        .map(|j| j[axis])
                        .fold(f32::INFINITY, f32::min);
                    let hi = jitters
                        .iter()
                        .map(|j| j[axis])
                        .fold(f32::NEG_INFINITY, f32::max);
                    assert!(
                        lo >= -1e-4 && hi <= 1.0001,
                        "sample outside pixel: {lo}..{hi}"
                    );
                    if visibility == VisibilityMode::RayQuery {
                        assert!(hi - lo > 0.8, "ray jitter correlated: {lo}..{hi}");
                    } else {
                        assert!(
                            hi - lo < 1e-4,
                            "raster projection jitter incoherent: {lo}..{hi}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    #[ignore = "requires graphics and hardware ray queries; per-pixel raster/ray parity including thin-lens DOF"]
    fn raster_visibility_matches_ray_queries_with_depth_of_field() {
        use crate::integrator::VisibilityMode;
        for mode in [Mode::Interactive, Mode::Final] {
            let mut ray = Renderer::new(96, 64, mode, None).unwrap();
            // Projection/ray geometry parity uses a shared sample as a reference;
            // production ray-query sampling is independent per pixel.
            ray.coherent_camera_reference = true;
            let mut raster = Renderer::new(96, 64, mode, None).unwrap();
            for orthographic in [false, true] {
                for f_stop in [0., 0.5] {
                    for opacity in [1., 0.5] {
                        let mut scene = fixture();
                        scene.camera.transform =
                            glam::Mat4::from_translation(glam::Vec3::new(0., 0., 5.))
                                .to_cols_array();
                        scene.camera.orthographic = orthographic;
                        scene.camera.horizontal_aperture = 4.;
                        scene.camera.vertical_aperture = 3.;
                        scene.camera.focal_length = 2.;
                        scene.camera.aperture_offset = [0.2, -0.1];
                        scene.camera.f_stop = f_stop;
                        scene.camera.focus_distance = 3.;
                        scene.camera.near = 0.1;
                        let mut foreground = scene.meshes[0].clone();
                        foreground.points = vec![
                            [-0.7, -0.7, 2.],
                            [0.7, -0.7, 2.],
                            [0.7, 0.7, 2.],
                            [-0.7, 0.7, 2.],
                        ];
                        foreground.material.openpbr = Some(crate::material::OpenPbrSurface {
                            geometry_opacity: opacity,
                            ..Default::default()
                        });
                        foreground.texcoords = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
                        scene.meshes.push(foreground);
                        scene.lights = vec![Light::Area {
                            diffuse: 1.,
                            specular: 1.,
                            single_sided: true,
                            render_light_geometry: true,
                            position: [1.2, 0.2, 3.],
                            u: [0.5, 0., 0.],
                            v: [0., 0.5, 0.],
                            color: [1.; 3],
                            intensity: 1.,
                        }];
                        scene.settings.visibility = VisibilityMode::RayQuery;
                        ray.set_scene(&scene).unwrap();
                        scene.settings.visibility = VisibilityMode::Raster;
                        raster.set_scene(&scene).unwrap();
                        for count in 0..4 {
                            let remaining = 4 - count;
                            ray.render_batch(remaining, 0.).unwrap();
                            raster.render_batch(remaining, 0.).unwrap();
                            let a = visibility_snapshot(&ray);
                            let b = visibility_snapshot(&raster);
                            let mut mismatches = 0;
                            for (a, b) in a.iter().zip(&b) {
                                if a[0].to_bits() != b[0].to_bits() {
                                    mismatches += 1;
                                    continue;
                                }
                                if a[0].to_bits() != 0 {
                                    assert!(
                                        (a[3] - b[3]).abs() < 0.0005,
                                        "depth mismatch {a:?} {b:?}"
                                    );
                                    assert!(
                                        (a[1] - b[1]).abs() < 0.001 && (a[2] - b[2]).abs() < 0.001,
                                        "barycentric mismatch {a:?} {b:?}"
                                    );
                                    assert_eq!(a[4].to_bits(), b[4].to_bits());
                                    for i in 5..7 {
                                        assert!(
                                            (f32::from_bits(a[i].to_bits())
                                                - f32::from_bits(b[i].to_bits()))
                                            .abs()
                                                < 0.001
                                        );
                                    }
                                }
                            }
                            assert!(mismatches<8,"visibility mismatch {mismatches}/{} mode={mode:?}, ortho={orthographic}, fstop={f_stop}, opacity={opacity}",a.len());
                            if mode == Mode::Final {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    #[ignore = "requires hardware ray queries; progressive-prefix stability and fixed final sample budgets"]
    fn sampling_prefix_is_stable_and_final_budget_requires_reset() {
        let scene = fixture();
        let mut short = Renderer::new(16, 16, Mode::Interactive, None).unwrap();
        let mut long = Renderer::new(16, 16, Mode::Interactive, None).unwrap();
        short.set_scene(&scene).unwrap();
        long.set_scene(&scene).unwrap();
        for i in 0..3 {
            let a = short
                .render_batch(if i < 2 { 2 - i } else { 64 }, 0.)
                .unwrap();
            let b = long.render_batch(64 - i, 0.).unwrap();
            assert_eq!(
                a, b,
                "Halton prefix changed with requested convergence target"
            );
        }
        let mut final_frame = Renderer::new(16, 16, Mode::Final, None).unwrap();
        final_frame.set_scene(&scene).unwrap();
        final_frame.render_batch(17, 0.).unwrap();
        assert_eq!(final_frame.samples(), 16);
        assert!(final_frame
            .render_batch(2, 0.)
            .unwrap_err()
            .to_string()
            .contains("final sample budget changed"));
        final_frame.render_batch(1, 0.).unwrap();
        assert_eq!(final_frame.samples(), 17);
        final_frame.reset_accumulation().unwrap();
        final_frame.render_batch(4, 0.).unwrap();
        assert_eq!(final_frame.samples(), 4);
    }
}

#[cfg(test)]
#[path = "renderer_profile.rs"]
mod profiling;

#[cfg(test)]
#[path = "renderer_upload_bench.rs"]
mod upload_benchmark;

#[cfg(test)]
#[path = "renderer_light_tests.rs"]
mod light_tests;
