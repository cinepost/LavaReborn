use crate::{
    graph::Graph,
    policy::Mode,
    scene::Scene,
    vulkan::{Acceleration, Buffer, Device, Instance},
};
use anyhow::{ensure, Result};
use ash::vk;
use std::sync::Arc;
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
}
struct GpuPass {
    output: Buffer,
    filtered: Option<Buffer>,
    pipeline: vk::Pipeline,
    raster: Option<crate::raster::RasterVisibility>,
    prepare_pipeline: Option<vk::Pipeline>,
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
            for buffer in std::iter::once(&self.output).chain(self.filtered.iter()) {
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
pub struct Renderer {
    textures: crate::texture::gpu::MaterialTextures,
    #[cfg(test)]
    coherent_camera_reference: bool,
    device: Arc<Device>,
    pub width: u32,
    pub height: u32,
    pub mode: Mode,
    pub linear_output: bool,
    pub gpu_output: bool,
    pub gpu_frame: Option<Arc<crate::interop::GpuFrame>>,
    pub depth: Vec<f32>,
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
    pub fn new(width: u32, height: u32, mode: Mode, gpu: Option<usize>) -> Result<Self> {
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
                32,
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
            let bindings: Vec<_> = (0..20)
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
                            1 | 7 | 17 => {
                                vk::ShaderStageFlags::COMPUTE
                                    | vk::ShaderStageFlags::VERTEX
                                    | vk::ShaderStageFlags::FRAGMENT
                            }
                            13 | 18 | 19 => {
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
                .stage_flags(
                    vk::ShaderStageFlags::COMPUTE
                        | vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::FRAGMENT,
                )
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
                    descriptor_count: 15,
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
            let shaders: [&[u8]; 5] = [
                include_bytes!(concat!(env!("OUT_DIR"), "/visibility.spv")),
                include_bytes!(concat!(env!("OUT_DIR"), "/shading.spv")),
                include_bytes!(concat!(env!("OUT_DIR"), "/accumulation.spv")),
                include_bytes!(concat!(env!("OUT_DIR"), "/tonemap.spv")),
                include_bytes!(concat!(env!("OUT_DIR"), "/queue_prepare.spv")),
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
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main");
                let result = r.device.raw.create_compute_pipelines(
                    r.pipeline_cache.raw,
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(r.layout)],
                    None,
                );
                r.device.raw.destroy_shader_module(module, None);
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
                } else {
                    r.passes.push(GpuPass {
                        output,
                        filtered,
                        pipeline,
                        raster: None,
                        prepare_pipeline: None,
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
        if diagnostics.skipped_faces > 0 || diagnostics.skipped_triangles > 0 {
            eprintln!("Lava: omitted {} unusable faces and {} zero-area triangles; rendering {} valid triangles", diagnostics.skipped_faces, diagnostics.skipped_triangles, triangles.len());
        }
        let geometry_bytes = bytemuck::cast_slice(&triangles);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        let tri = Buffer::new(device.clone(), geometry_bytes.len() as u64, usage, true)?;
        tri.write(geometry_bytes)?;
        // BLAS positions are tightly packed; shading triangles retain std430 layout independently.
        let positions: Vec<[f32; 3]> = triangles
            .iter()
            .flat_map(|t| {
                [
                    t.a[..3].try_into().unwrap(),
                    t.b[..3].try_into().unwrap(),
                    t.c[..3].try_into().unwrap(),
                ]
            })
            .collect();
        let vertices = Buffer::new(device.clone(), positions.len() as u64 * 12, usage, true)?;
        vertices.write(bytemuck::cast_slice(&positions))?;
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
            triangles.len() as u32,
            false,
            mode.policy().fast_build,
        )?;
        let instance = vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 255),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas.address(),
            },
        };
        let instances = Buffer::new(
            device.clone(),
            std::mem::size_of_val(&instance) as u64,
            usage,
            true,
        )?;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (&instance as *const vk::AccelerationStructureInstanceKHR).cast::<u8>(),
                std::mem::size_of_val(&instance),
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
        let tlas =
            Acceleration::build(device.clone(), geometry, 1, true, mode.policy().fast_build)?;
        let materials = crate::material::upload(device.clone(), scene)?;
        Ok(PreparedGeometry {
            texture_materials: scene.meshes.iter().map(|m| m.material.clone()).collect(),
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
    pub(crate) fn update_materials(&mut self, scene: &Scene) -> Result<()> {
        for mesh in &scene.meshes {
            mesh.material.validate()?;
        }
        self.scene_buffers[5] = crate::material::upload(self.device.clone(), scene)?;
        self.textures.configure(
            scene.meshes.iter().map(|m| m.material.clone()).collect(),
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
                    let independent =
                        self.settings.visibility == crate::integrator::VisibilityMode::RayQuery;
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
        Ok(())
    }
    /// Returns one completed batch. Zero remaining reprocesses an existing image only.
    /// Interactive workers check their mailbox after each call.
    pub fn render_batch(&mut self, remaining: u32, exposure: f32) -> Result<Vec<f32>> {
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
        if self.settings.visibility == crate::integrator::VisibilityMode::Raster
            && self.passes[0].raster.is_none()
        {
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
        let output_needed = self.mode == Mode::Interactive || count == remaining;
        let gpu = if self.gpu_output && self.device.interop && output_needed {
            match crate::interop::GpuFrame::new(
                self.device.clone(),
                self.width,
                self.height,
                self.samples() + count,
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
        let output_size = self.width as u64 * self.height as u64 * 32;
        if gpu.is_none() && output_needed && self.passes[3].output.size < output_size {
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
                        sample,
                        lights: self.light_count,
                        exposure,
                        radius: self.mode.policy().filter_radius,
                        linear_output: u32::from(self.linear_output),
                        filter_phase: 0,
                        diffuse_depth: self.settings.diffuse_depth,
                        reflection_depth: self.settings.reflection_depth,
                        refraction_depth: self.settings.refraction_depth,
                        roulette_start: self.settings.roulette_start,
                        ray_phase: 0,
                        queue_parity: 0,
                        max_groups_x: self.max_groups_x,
                        sampling_mode: u32::from(self.mode == Mode::Final),
                        target_samples: self.sampling_target,
                    };
                    device.raw.cmd_push_constants(
                        cmd,
                        self.layout,
                        vk::ShaderStageFlags::COMPUTE
                            | vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push),
                    );
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
                                || (self.mode == Mode::Final && count != remaining))
                        {
                            continue;
                        }
                        let raster_visibility = pass == 0
                            && self.settings.visibility
                                == crate::integrator::VisibilityMode::Raster;
                        if raster_visibility {
                            self.passes[0].raster.as_ref().unwrap().record(
                                cmd,
                                self.layout,
                                self.set,
                                (self.scene_buffers[0].size
                                    / std::mem::size_of::<crate::scene::Triangle>() as u64
                                    * 3) as u32,
                            );
                        }
                        device.raw.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            if raster_visibility {
                                self.passes[0].raster.as_ref().unwrap().resolve
                            } else {
                                self.passes[pass].pipeline
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
                            device.raw.cmd_fill_buffer(cmd, control.raw, 0, 32, 0);
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
                            device.raw.cmd_dispatch(cmd, x, groups.div_ceil(x), 1);
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
                                    vk::ShaderStageFlags::COMPUTE
                                        | vk::ShaderStageFlags::VERTEX
                                        | vk::ShaderStageFlags::FRAGMENT,
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
                                device.raw.cmd_bind_pipeline(
                                    cmd,
                                    vk::PipelineBindPoint::COMPUTE,
                                    self.passes[1].pipeline,
                                );
                                device.raw.cmd_dispatch_indirect(cmd, control.raw, 16);
                            }
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
                            && (self.mode == Mode::Interactive || count == remaining)
                        {
                            // Filtering is the second phase of accumulation; never overwrite history.
                            push.filter_phase = 1;
                            device.raw.cmd_push_constants(
                                cmd,
                                self.layout,
                                vk::ShaderStageFlags::COMPUTE
                                    | vk::ShaderStageFlags::VERTEX
                                    | vk::ShaderStageFlags::FRAGMENT,
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
        if self.mode == Mode::Final && count != remaining {
            return Ok(Vec::new());
        }
        if self.gpu_frame.is_some() {
            self.depth.clear();
            return Ok(Vec::new());
        }
        let n = self.width as usize * self.height as usize;
        crate::interop::IMAGE_READBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut packed = self.passes[3].output.read_floats(n * 8)?;
        self.depth = packed[n * 4..].iter().step_by(4).copied().collect();
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
            for p in &self.passes {
                self.device.raw.destroy_pipeline(p.pipeline, None);
                if let Some(prepare) = p.prepare_pipeline {
                    self.device.raw.destroy_pipeline(prepare, None);
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
    fn fixture() -> Scene {
        Scene {
            settings: IntegratorSettings {
                textures: crate::texture::TextureMode::Original,
                visibility: Default::default(),
                diffuse_depth: 1,
                reflection_depth: 0,
                refraction_depth: 0,
                roulette_start: 96,
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
    fn mean(renderer: &mut Renderer, scene: &Scene, samples: u32) -> f32 {
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
                    (left - 0.015).abs() < 1e-5 && (right - 1.).abs() < 1e-5,
                    "textured cutout {left}/{right} {visibility:?} {mode:?}"
                );
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
                    direction: [0., 0., -1.],
                    color: [1.; 3],
                    intensity: std::f32::consts::PI,
                }];
                let flat = mean(&mut renderer, &scene, 16);
                for (field, file, strength, signed) in [
                    ("normal", normal.clone(), 1., true),
                    ("normal", unsigned.clone(), 1., false),
                    ("bump", bump.clone(), 15., false),
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
                    direction: [-0.9, 0., 0.1],
                    color: [1.; 3],
                    intensity: 100.,
                }];
                let blocked = mean(&mut renderer, &scene, 32);
                assert!(blocked < 0.03, "below-surface light leak: {blocked}");
                scene.meshes[0].normal_indices = vec![0; 4];
                scene.meshes[0].normals = vec![[0.6, 0., 0.8]];
                scene.lights = vec![Light::Directional {
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
                (dark - 0.012).abs() < 1e-5 && lit > 0.1,
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
            assert!(
                ordinary > 1.9 && (hidden - 0.015).abs() < 1e-5 && (ordinary - shown).abs() < 1e-6
            );
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
        let expected = 0.8 * (3. * probability + 0.015 * (1. - probability)) as f32;
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
        let expected = 0.8 * (100. * probability + 0.015 * (1. - probability)) as f32;
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
    fn buffer_snapshot(renderer: &Renderer, source: &Buffer) -> Vec<f32> {
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
