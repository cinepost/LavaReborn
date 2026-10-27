//! Hardware raster visibility resources, owned by the visibility pass. Color
//! attachments resolve to the same GPU Hit buffer used by ray-query visibility.
use crate::vulkan::Device;
use anyhow::{ensure, Result};
use ash::vk;
use std::sync::Arc;
struct Attachment {
    device: Arc<Device>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}
impl Attachment {
    fn new(device: Arc<Device>, width: u32, height: u32, depth: bool) -> Result<Self> {
        let mut a = Self {
            device,
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
        };
        let format = if depth {
            vk::Format::D32_SFLOAT
        } else {
            vk::Format::R32G32B32A32_UINT
        };
        let usage = if depth {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE
        };
        let features = if depth {
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::STORAGE_IMAGE
        };
        unsafe {
            ensure!(
                a.device
                    .instance
                    .raw
                    .get_physical_device_format_properties(a.device.physical, format)
                    .optimal_tiling_features
                    .contains(features),
                "raster visibility attachment format unsupported: {format:?}"
            );
            a.image = a.device.raw.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let req = a.device.raw.get_image_memory_requirements(a.image);
            let memory = a
                .device
                .instance
                .raw
                .get_physical_device_memory_properties(a.device.physical);
            let index = (0..memory.memory_type_count)
                .filter(|&i| {
                    req.memory_type_bits & (1 << i) != 0
                        && memory.memory_types[i as usize]
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .max_by_key(|&i| {
                    !memory.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                })
                .ok_or_else(|| anyhow::anyhow!("no device-local raster image memory"))?;
            a.memory = a.device.raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(index),
                None,
            )?;
            a.device.raw.bind_image_memory(a.image, a.memory, 0)?;
            a.view = a.device.raw.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(a.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(if depth {
                                vk::ImageAspectFlags::DEPTH
                            } else {
                                vk::ImageAspectFlags::COLOR
                            })
                            .level_count(1)
                            .layer_count(1),
                    ),
                None,
            )?;
        }
        Ok(a)
    }
}
impl Drop for Attachment {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_image_view(self.view, None);
            self.device.raw.destroy_image(self.image, None);
            self.device.raw.free_memory(self.memory, None);
        }
    }
}
pub(crate) struct RasterVisibility {
    device: Arc<Device>,
    attachments: Vec<Attachment>,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    pipeline: vk::Pipeline,
    pub resolve: vk::Pipeline,
    width: u32,
    height: u32,
}
impl RasterVisibility {
    pub fn new(
        device: Arc<Device>,
        width: u32,
        height: u32,
        layout: vk::PipelineLayout,
        set: vk::DescriptorSet,
        cache: vk::PipelineCache,
    ) -> Result<Self> {
        let mut r = Self {
            device,
            attachments: Vec::new(),
            render_pass: vk::RenderPass::null(),
            framebuffer: vk::Framebuffer::null(),
            pipeline: vk::Pipeline::null(),
            resolve: vk::Pipeline::null(),
            width,
            height,
        };
        for depth in [false, false, true] {
            r.attachments
                .push(Attachment::new(r.device.clone(), width, height, depth)?);
        }
        unsafe {
            let descriptions: Vec<_> = (0..3)
                .map(|i| {
                    vk::AttachmentDescription::default()
                        .format(if i == 2 {
                            vk::Format::D32_SFLOAT
                        } else {
                            vk::Format::R32G32B32A32_UINT
                        })
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(if i == 2 {
                            vk::AttachmentStoreOp::DONT_CARE
                        } else {
                            vk::AttachmentStoreOp::STORE
                        })
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(if i == 2 {
                            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                        } else {
                            vk::ImageLayout::GENERAL
                        })
                })
                .collect();
            let colors = [
                vk::AttachmentReference::default()
                    .attachment(0)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                vk::AttachmentReference::default()
                    .attachment(1)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            ];
            let depth = vk::AttachmentReference::default()
                .attachment(2)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            let subpasses = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&colors)
                .depth_stencil_attachment(&depth)];
            let dependencies = [
                vk::SubpassDependency::default()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
                    .dst_stage_mask(
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    )
                    .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    ),
                vk::SubpassDependency::default()
                    .src_subpass(0)
                    .dst_subpass(vk::SUBPASS_EXTERNAL)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ),
            ];
            r.render_pass = r.device.raw.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&descriptions)
                    .subpasses(&subpasses)
                    .dependencies(&dependencies),
                None,
            )?;
            let views: Vec<_> = r.attachments.iter().map(|a| a.view).collect();
            r.framebuffer = r.device.raw.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(r.render_pass)
                    .attachments(&views)
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )?;
            let binaries: [(&[u8], vk::ShaderStageFlags); 2] = [
                (
                    include_bytes!(concat!(env!("OUT_DIR"), "/visibility_raster.vert.spv")),
                    vk::ShaderStageFlags::VERTEX,
                ),
                (
                    include_bytes!(concat!(env!("OUT_DIR"), "/visibility_raster.frag.spv")),
                    vk::ShaderStageFlags::FRAGMENT,
                ),
            ];
            let mut modules = Vec::new();
            let result = (|| -> Result<vk::Pipeline> {
                for (bytes, _) in &binaries {
                    let code = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
                    modules.push(r.device.raw.create_shader_module(
                        &vk::ShaderModuleCreateInfo::default().code(&code),
                        None,
                    )?);
                }
                let stages: Vec<_> = binaries
                    .iter()
                    .zip(&modules)
                    .map(|((_, stage), module)| {
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(*stage)
                            .module(*module)
                            .name(c"main")
                    })
                    .collect();
                let vertex = vk::PipelineVertexInputStateCreateInfo::default();
                let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
                let viewport = [vk::Viewport::default()
                    .width(width as f32)
                    .height(height as f32)
                    .min_depth(0.)
                    .max_depth(1.)];
                let scissors = [vk::Rect2D::default().extent(vk::Extent2D { width, height })];
                let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
                let dynamic =
                    vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
                let vp = vk::PipelineViewportStateCreateInfo::default()
                    .viewports(&viewport)
                    .scissors(&scissors);
                let raster = vk::PipelineRasterizationStateCreateInfo::default()
                    .polygon_mode(vk::PolygonMode::FILL)
                    .cull_mode(vk::CullModeFlags::NONE)
                    .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                    .line_width(1.);
                let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                    .rasterization_samples(vk::SampleCountFlags::TYPE_1);
                let depth = vk::PipelineDepthStencilStateCreateInfo::default()
                    .depth_test_enable(true)
                    .depth_write_enable(true)
                    .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
                let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA);
                    2];
                let blend = vk::PipelineColorBlendStateCreateInfo::default()
                    .attachments(&blend_attachments);
                match r.device.raw.create_graphics_pipelines(
                    cache,
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&stages)
                        .vertex_input_state(&vertex)
                        .input_assembly_state(&assembly)
                        .viewport_state(&vp)
                        .dynamic_state(&dynamic)
                        .rasterization_state(&raster)
                        .multisample_state(&multisample)
                        .depth_stencil_state(&depth)
                        .color_blend_state(&blend)
                        .layout(layout)
                        .render_pass(r.render_pass)],
                    None,
                ) {
                    Ok(p) => Ok(p[0]),
                    Err((partial, e)) => {
                        for p in partial {
                            r.device.raw.destroy_pipeline(p, None);
                        }
                        Err(e.into())
                    }
                }
            })();
            for module in modules {
                r.device.raw.destroy_shader_module(module, None);
            }
            r.pipeline = result?;
            let code = ash::util::read_spv(&mut std::io::Cursor::new(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/visibility_resolve.spv"
            ))))?;
            let module = r
                .device
                .raw
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline = r.device.raw.create_compute_pipelines(
                cache,
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            );
            r.device.raw.destroy_shader_module(module, None);
            r.resolve = match pipeline {
                Ok(p) => p[0],
                Err((partial, e)) => {
                    for p in partial {
                        r.device.raw.destroy_pipeline(p, None);
                    }
                    return Err(e.into());
                }
            };
            for (i, a) in r.attachments[..2].iter().enumerate() {
                let info = [vk::DescriptorImageInfo::default()
                    .image_view(a.view)
                    .image_layout(vk::ImageLayout::GENERAL)];
                r.device.raw.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(15 + i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&info)],
                    &[],
                );
            }
        }
        Ok(r)
    }
    pub fn resize(&mut self, width: u32, height: u32, set: vk::DescriptorSet) -> Result<()> {
        if self.width == width && self.height == height {
            return Ok(());
        }
        let mut attachments = Vec::new();
        for depth in [false, false, true] {
            attachments.push(Attachment::new(self.device.clone(), width, height, depth)?);
        }
        unsafe {
            let views: Vec<_> = attachments.iter().map(|a| a.view).collect();
            let framebuffer = self.device.raw.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(self.render_pass)
                    .attachments(&views)
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )?;
            for (i, a) in attachments[..2].iter().enumerate() {
                let info = [vk::DescriptorImageInfo::default()
                    .image_view(a.view)
                    .image_layout(vk::ImageLayout::GENERAL)];
                self.device.raw.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(15 + i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&info)],
                    &[],
                );
            }
            self.device.raw.destroy_framebuffer(self.framebuffer, None);
            self.framebuffer = framebuffer;
            self.attachments = attachments;
        }
        self.width = width;
        self.height = height;
        Ok(())
    }
    pub unsafe fn record(
        &self,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        set: vk::DescriptorSet,
        vertices: u32,
    ) {
        let clear = [
            vk::ClearValue {
                color: vk::ClearColorValue { uint32: [0; 4] },
            },
            vk::ClearValue {
                color: vk::ClearColorValue { uint32: [0; 4] },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.,
                    stencil: 0,
                },
            },
        ];
        self.device.raw.cmd_set_viewport(
            cmd,
            0,
            &[vk::Viewport::default()
                .width(self.width as f32)
                .height(self.height as f32)
                .min_depth(0.)
                .max_depth(1.)],
        );
        self.device.raw.cmd_set_scissor(
            cmd,
            0,
            &[vk::Rect2D::default().extent(vk::Extent2D {
                width: self.width,
                height: self.height,
            })],
        );
        self.device.raw.cmd_begin_render_pass(
            cmd,
            &vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffer)
                .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                    width: self.width,
                    height: self.height,
                }))
                .clear_values(&clear),
            vk::SubpassContents::INLINE,
        );
        self.device
            .raw
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        self.device.raw.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            layout,
            0,
            &[set],
            &[],
        );
        self.device.raw.cmd_draw(cmd, vertices, 1, 0, 0);
        self.device.raw.cmd_end_render_pass(cmd);
    }
}
impl Drop for RasterVisibility {
    fn drop(&mut self) {
        unsafe {
            self.device.raw.destroy_pipeline(self.pipeline, None);
            self.device.raw.destroy_pipeline(self.resolve, None);
            self.device.raw.destroy_framebuffer(self.framebuffer, None);
            self.device.raw.destroy_render_pass(self.render_pass, None);
        }
    }
}
