//! Opt-in GPU diagnostic harness; never enabled in the viewport renderer.
use super::*;
use serde_json::json;
use sha2::{Digest, Sha256};

struct Probe {
    device: Arc<Device>,
    pool: vk::QueryPool,
    pipelines: Vec<vk::Pipeline>,
    period: f64,
    mask: u64,
}
impl Probe {
    fn new(r: &Renderer) -> Self {
        let d = &r.device;
        let properties = unsafe { d.instance.raw.get_physical_device_properties(d.physical) };
        let queues = unsafe {
            d.instance
                .raw
                .get_physical_device_queue_family_properties(d.physical)
        };
        let bits = queues[d.queue_family as usize].timestamp_valid_bits;
        assert!(bits > 0);
        Self {
            device: d.clone(),
            pool: unsafe {
                d.raw
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2),
                        None,
                    )
                    .unwrap()
            },
            pipelines: Vec::new(),
            period: properties.limits.timestamp_period as f64,
            mask: u64::MAX >> (64 - bits),
        }
    }
    fn pipeline(&mut self, r: &Renderer, bytes: &[u8], constant: u32, value: u32) -> vk::Pipeline {
        println!("Creating diagnostic product constant={constant} value={value}");
        let started = std::time::Instant::now();
        unsafe {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();
            let module = self
                .device
                .raw
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .unwrap();
            let entries = [vk::SpecializationMapEntry::default()
                .constant_id(constant)
                .offset(0)
                .size(4)];
            let data = value.to_ne_bytes();
            let info = vk::SpecializationInfo::default()
                .map_entries(&entries)
                .data(&data);
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .specialization_info(&info);
            let pipeline = self
                .device
                .raw
                .create_compute_pipelines(
                    r.pipeline_cache.raw,
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(r.layout)],
                    None,
                )
                .unwrap()[0];
            self.device.raw.destroy_shader_module(module, None);
            self.pipelines.push(pipeline);
            println!(
                "Diagnostic product ready in {:.2}s",
                started.elapsed().as_secs_f64()
            );
            r.pipeline_cache.persist();
            pipeline
        }
    }
    fn measure(&self, r: &Renderer, push: &Push, pipeline: vk::Pipeline, dispatch: u32) -> f64 {
        self.device
            .submit(|cmd| unsafe {
                barrier(&self.device, cmd);
                self.device.raw.cmd_reset_query_pool(cmd, self.pool, 0, 2);
                bind(r, cmd, push, pipeline);
                self.device.raw.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    self.pool,
                    0,
                );
                if dispatch == 0 {
                    self.device
                        .raw
                        .cmd_dispatch(cmd, r.width.div_ceil(8), r.height.div_ceil(8), 1);
                } else if dispatch == 1 {
                    self.device
                        .raw
                        .cmd_dispatch_indirect(cmd, r.passes[1].scratch[2].raw, 16);
                } else {
                    let n = (r.width * r.height).div_ceil(64);
                    let x = n.min(r.max_groups_x);
                    self.device.raw.cmd_dispatch(cmd, x, n.div_ceil(x), 1);
                }
                self.device.raw.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.pool,
                    1,
                );
                barrier(&self.device, cmd);
            })
            .unwrap();
        let mut ticks = [0u64; 2];
        unsafe {
            self.device
                .raw
                .get_query_pool_results(
                    self.pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .unwrap();
        }
        (ticks[1].wrapping_sub(ticks[0]) & self.mask) as f64 * self.period / 1e6
    }
}
impl Drop for Probe {
    fn drop(&mut self) {
        unsafe {
            for p in &self.pipelines {
                self.device.raw.destroy_pipeline(*p, None);
            }
            self.device.raw.destroy_query_pool(self.pool, None);
        }
    }
}
unsafe fn barrier(d: &Device, cmd: vk::CommandBuffer) {
    d.raw.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)],
        &[],
        &[],
    );
}
unsafe fn bind(r: &Renderer, cmd: vk::CommandBuffer, push: &Push, pipeline: vk::Pipeline) {
    r.device.raw.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        r.layout,
        0,
        &[r.set],
        &[],
    );
    r.device.raw.cmd_push_constants(
        cmd,
        r.layout,
        r.device.shader_stages(),
        0,
        bytemuck::bytes_of(push),
    );
    r.device
        .raw
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
}
fn read_words(d: Arc<Device>, source: &Buffer, offset: u64, count: usize) -> Vec<u32> {
    let staging = Buffer::new(
        d.clone(),
        count as u64 * 4,
        vk::BufferUsageFlags::TRANSFER_DST,
        true,
    )
    .unwrap();
    d.submit(|cmd| unsafe {
        barrier(&d, cmd);
        d.raw.cmd_copy_buffer(
            cmd,
            source.raw,
            staging.raw,
            &[vk::BufferCopy::default()
                .src_offset(offset)
                .size(count as u64 * 4)],
        );
        barrier(&d, cmd);
    })
    .unwrap();
    staging
        .read_floats(count)
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}
fn reset_stage(r: &Renderer, shadow: &Buffer, backup: &Buffer, output_queue: u32, save: bool) {
    let d = &r.device;
    d.submit(|cmd| unsafe {
        barrier(d, cmd);
        d.raw.cmd_fill_buffer(
            cmd,
            r.passes[1].scratch[2].raw,
            output_queue as u64 * 4,
            4,
            0,
        );
        if save {
            d.raw.cmd_fill_buffer(cmd, shadow.raw, 0, vk::WHOLE_SIZE, 0);
            d.raw.cmd_copy_buffer(
                cmd,
                r.passes[1].output.raw,
                backup.raw,
                &[vk::BufferCopy::default().size(backup.size)],
            );
        } else {
            d.raw.cmd_copy_buffer(
                cmd,
                backup.raw,
                r.passes[1].output.raw,
                &[vk::BufferCopy::default().size(backup.size)],
            );
        }
        barrier(d, cmd);
    })
    .unwrap();
}
fn summarize(values: &[f64]) -> serde_json::Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    };
    json!({"median_ms":median,"mean_ms":values.iter().sum::<f64>()/values.len() as f64,"min_ms":sorted[0],"max_ms":sorted[sorted.len()-1]})
}

#[test]
#[ignore = "requires scene JSON in LAVA_PROFILE_SCENE and hardware Vulkan; GPU capture/replay diagnostics"]
fn profile_scene_visibility_shadows_shading() {
    let Ok(path) = std::env::var("LAVA_PROFILE_SCENE") else {
        println!("Set LAVA_PROFILE_SCENE to run the scene profiler");
        return;
    };
    let scene: Scene = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let width = std::env::var("LAVA_PROFILE_WIDTH")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(1920);
    let height = std::env::var("LAVA_PROFILE_HEIGHT")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(1080);
    let warmup = std::env::var("LAVA_PROFILE_WARMUP")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(4u32);
    let samples = std::env::var("LAVA_PROFILE_SAMPLES")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(12u32);
    let mut r = Renderer::new(width, height, Mode::Interactive, None).unwrap();
    // Capture/replay measures individual dispatches. The separate full-frame
    // reorder benchmark includes sorting; this stage profiler retains its baseline.
    r.reorder_rays = false;
    let start = std::time::Instant::now();
    r.set_scene(&scene).unwrap();
    if std::env::var("LAVA_PROFILE_GEOMETRY_VRAM").as_deref() == Ok("1")
        && !r.scene_buffers[0]
            .memory_properties
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
    {
        let source = &r.scene_buffers[0];
        let local = Buffer::new(
            r.device.clone(),
            source.size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            false,
        )
        .unwrap();
        r.device
            .submit(|cmd| unsafe {
                barrier(&r.device, cmd);
                r.device.raw.cmd_copy_buffer(
                    cmd,
                    source.raw,
                    local.raw,
                    &[vk::BufferCopy::default().size(source.size)],
                );
                barrier(&r.device, cmd);
            })
            .unwrap();
        r.scene_buffers[0] = local;
        r.update_scene_parameters(&scene).unwrap();
    }
    // Bind fallback texture descriptors even when this scene has no textures.
    r.refresh_textures().unwrap();
    while r.textures_pending() {
        r.refresh_textures().unwrap();
        std::thread::yield_now();
    }
    println!(
        "Scene ready in {:.2}s: mesh triangles={}, analytic curve segments={}, curve objects={}",
        start.elapsed().as_secs_f64(),
        r.mesh_triangles,
        r.curve_count,
        r.curve_blas.len()
    );
    println!(
        "Segment/shading record memory: {:?}, {} bytes",
        r.scene_buffers[0].memory_properties, r.scene_buffers[0].size
    );
    let opaque = scene
        .meshes
        .iter()
        .map(|m| &m.material)
        .chain(scene.curves.iter().map(|c| &c.material))
        .all(|m| {
            m.openpbr.is_none_or(|p| p.geometry_opacity == 1.)
                && !m.textures.contains_key("geometry_opacity")
        });
    println!(
        "All geometry has opaque coverage: {opaque}; settings={:?}",
        scene.settings
    );
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC;
    let shadow = Buffer::new(
        r.device.clone(),
        16 + width as u64 * height as u64 * 48,
        usage,
        false,
    )
    .unwrap();
    let continuation = Buffer::new(
        r.device.clone(),
        width as u64 * height as u64 * 16,
        usage,
        false,
    )
    .unwrap();
    let backup = Buffer::new(
        r.device.clone(),
        width as u64 * height as u64 * 16,
        usage,
        false,
    )
    .unwrap();
    unsafe {
        let infos = [vk::DescriptorBufferInfo::default()
            .buffer(shadow.raw)
            .range(shadow.size)];
        let continuation_info = [vk::DescriptorBufferInfo::default()
            .buffer(continuation.raw)
            .range(continuation.size)];
        r.device.raw.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(r.set)
                .dst_binding(22)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&continuation_info)],
            &[],
        );
        r.device.raw.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(r.set)
                .dst_binding(21)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos)],
            &[],
        );
    }
    let mut probe = Probe::new(&r);
    let capture = probe.pipeline(
        &r,
        include_bytes!(concat!(env!("OUT_DIR"), "/shading.spv")),
        51,
        1,
    );
    let cached = probe.pipeline(
        &r,
        include_bytes!(concat!(env!("OUT_DIR"), "/shading.spv")),
        51,
        2,
    );
    let shadow_replay = probe.pipeline(
        &r,
        include_bytes!(concat!(env!("OUT_DIR"), "/shadow_profile.spv")),
        52,
        0,
    );
    let statistics = probe.pipeline(
        &r,
        include_bytes!(concat!(env!("OUT_DIR"), "/shadow_profile.spv")),
        52,
        1,
    );
    let continuation_replay = probe.pipeline(
        &r,
        include_bytes!(concat!(env!("OUT_DIR"), "/continuation_profile.spv")),
        52,
        0,
    );
    let mut continuation_times = Vec::new();
    let mut rows = Vec::new();
    let mut visibility_times = Vec::new();
    let mut combined_times = Vec::new();
    let mut inclusive_times = Vec::new();
    let mut replay_times = Vec::new();
    let mut cached_times = Vec::new();
    let mut primary_hits = [0u64; 3];
    let mut radiance_hashes = Vec::new();
    let mut primary_hash = String::new();
    for sample in 0..warmup + samples {
        let mut push = Push {
            width,
            height,
            sample,
            lights: r.light_count,
            exposure: 0.,
            radius: 1,
            linear_output: 1,
            filter_phase: 0,
            diffuse_depth: scene.settings.diffuse_depth,
            reflection_depth: scene.settings.reflection_depth,
            refraction_depth: scene.settings.refraction_depth,
            roulette_start: scene.settings.roulette_start,
            ray_phase: 0,
            queue_parity: 0,
            max_groups_x: r.max_groups_x,
            sampling_mode: 0,
            target_samples: 64,
            reduce_analytic_caustics: scene.settings.reduce_analytic_caustics as u32,
            mesh_triangles: r.mesh_triangles,
            curve_count: r.curve_count,
            adaptive_threshold: scene.settings.adaptive_threshold,
            adaptive_min_samples: scene.settings.adaptive_min_samples,
        };
        let visibility = probe.measure(&r, &push, r.passes[0].pipeline, 0);
        r.device
            .submit(|cmd| unsafe {
                barrier(&r.device, cmd);
                r.device
                    .raw
                    .cmd_fill_buffer(cmd, r.passes[1].scratch[2].raw, 0, 32, 0);
                barrier(&r.device, cmd);
            })
            .unwrap();
        let mut stages = Vec::new();
        let mut inclusive = 0.;
        let mut replay = 0.;
        let mut no_shadow = 0.;
        let mut continuation_time = 0.;
        for phase in 0..=scene.settings.max_scattering_events() {
            push.ray_phase = u32::from(phase > 0);
            push.queue_parity = phase.saturating_sub(1) % 2;
            if phase > 0 {
                r.device
                    .submit(|cmd| unsafe {
                        barrier(&r.device, cmd);
                        bind(&r, cmd, &push, r.passes[1].prepare_pipeline.unwrap());
                        r.device.raw.cmd_dispatch(cmd, 1, 1, 1);
                        barrier(&r.device, cmd);
                    })
                    .unwrap();
            }
            let active = if phase == 0 {
                width * height
            } else {
                read_words(
                    r.device.clone(),
                    &r.passes[1].scratch[2],
                    push.queue_parity as u64 * 4,
                    1,
                )[0]
            };
            if active == 0 {
                break;
            }
            let output = if phase == 0 { 0 } else { 1 - push.queue_parity };
            let dispatch = u32::from(phase > 0); // phase zero uses 64-wide linear dispatch too
            reset_stage(&r, &shadow, &backup, output, true);
            let normal = probe.measure(
                &r,
                &push,
                r.passes[1].pipeline,
                if dispatch == 0 { 2 } else { 1 },
            );
            reset_stage(&r, &shadow, &backup, output, false);
            let instrumented = probe.measure(&r, &push, capture, if dispatch == 0 { 2 } else { 1 });
            let expected = (sample == 0).then(|| {
                read_words(
                    r.device.clone(),
                    &r.passes[1].output,
                    0,
                    (width * height * 4) as usize,
                )
            });
            reset_stage(&r, &shadow, &backup, output, false);
            let shading = probe.measure(&r, &push, cached, if dispatch == 0 { 2 } else { 1 });
            if let Some(expected) = expected {
                let actual = read_words(
                    r.device.clone(),
                    &r.passes[1].output,
                    0,
                    (width * height * 4) as usize,
                );
                radiance_hashes.push(format!(
                    "{:x}",
                    Sha256::digest(bytemuck::cast_slice(&actual))
                ));
                assert_eq!(
                    expected, actual,
                    "cached shading changed radiance in phase {phase}"
                );
            }
            let tracing = probe.measure(&r, &push, shadow_replay, 2);
            let secondary = if phase > 0 {
                probe.measure(&r, &push, continuation_replay, 1)
            } else {
                0.
            };
            continuation_time += secondary;
            probe.measure(&r, &push, statistics, 2);
            let stats = read_words(r.device.clone(), &shadow, 0, 4);
            assert_eq!(
                stats[2], 0,
                "shadow replay differs from captured query at sample {sample} phase {phase}"
            );
            assert_eq!(
                stats[3], 0,
                "continuation replay mismatch at sample {sample} phase {phase}"
            );
            inclusive += normal;
            replay += tracing;
            no_shadow += shading;
            stages.push(json!({"phase":phase,"active_paths":active,"shadow_rays":stats[0],"occluded_shadow_rays":stats[1],"shading_including_queries_ms":normal,"capture_including_queries_ms":instrumented,"shadow_replay_ms":tracing,"continuation_replay_ms":secondary,"shading_cached_queries_ms":shading}));
        }
        if sample == warmup {
            let hit_words = read_words(
                r.device.clone(),
                &r.passes[0].output,
                0,
                width as usize * height as usize * 8,
            );
            primary_hash = format!("{:x}", Sha256::digest(bytemuck::cast_slice(&hit_words)));
            for hit in hit_words.as_chunks::<8>().0 {
                let id = hit[0];
                let kind = if id == 0 {
                    0
                } else if id - 1 < r.mesh_triangles {
                    1
                } else {
                    2
                };
                primary_hits[kind] += 1;
            }
        }
        println!("sample {sample}: visibility={visibility:.3}ms shading+queries={inclusive:.3}ms shadow replay={replay:.3}ms cached-query shading={no_shadow:.3}ms continuation replay={continuation_time:.3}ms");
        if sample >= warmup {
            visibility_times.push(visibility);
            combined_times.push(visibility + inclusive);
            inclusive_times.push(inclusive);
            replay_times.push(replay);
            cached_times.push(no_shadow);
            continuation_times.push(continuation_time);
            rows.push(json!({"sample":sample,"primary_visibility_ms":visibility,"stages":stages}));
        }
    }
    let report = json!({"gpu":unsafe {std::ffi::CStr::from_ptr(r.device.instance.raw.get_physical_device_properties(r.device.physical).device_name.as_ptr())}.to_string_lossy().into_owned(),"scene":path,"resolution":[width,height],"mode":"interactive","queue_reordering":false,"settings":scene.settings,"camera":scene.camera,
        "first_sample_stage_radiance_sha256":radiance_hashes,"primary_visibility_sha256":primary_hash,"warmup_samples":warmup,"measured_samples":samples,"mesh_triangles":r.mesh_triangles,"curve_segments":r.curve_count,"curve_objects":r.curve_blas.len(),
        "geometry_record_memory":{"properties":format!("{:?}",r.scene_buffers[0].memory_properties),"bytes":r.scene_buffers[0].size},"all_coverage_opaque":opaque,"materials":scene.meshes.iter().map(|m|&m.material).chain(scene.curves.iter().map(|c|&c.material)).collect::<Vec<_>>(),"primary_pixels":{"miss":primary_hits[0],"mesh":primary_hits[1],"curve":primary_hits[2]},
        "primary_visibility":summarize(&visibility_times),"primary_plus_fused_shading":summarize(&combined_times),"shading_including_queries":summarize(&inclusive_times),
        "isolated_shadow_replay":summarize(&replay_times),"isolated_continuation_replay":summarize(&continuation_times),"shading_cached_geometry_answers":summarize(&cached_times),"samples":rows,
        "notes":["GPU timestamps exclude scene load, AS build, transfers, accumulation, tone mapping and presentation.","Shadow replay and cached-answer shading are isolated diagnostic kernels; their times are not additive to the original fused kernel.","Cached-answer shading bypasses both shadow and continuation geometry queries; it retains analytic-light tests and queue compaction.","Capture uses the actual per-sample shadow rays and checks every replayed answer."]});
    let output =
        std::env::var("LAVA_PROFILE_OUTPUT").unwrap_or_else(|_| "build/scene-profile.json".into());
    std::fs::write(&output, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    println!("PROFILE REPORT {output}");
}

#[test]
#[ignore = "requires GPU; reordered compaction must preserve every surviving path and sort dense groups"]
fn reordered_compaction_preserves_payloads_and_orders_groups() {
    let scene: Scene = serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
    let mut r = Renderer::new(1, 1, Mode::Interactive, None).unwrap();
    r.set_scene(&scene).unwrap();
    let mut probe = Probe::new(&r);
    let pipeline = probe.pipeline(
        &r,
        include_bytes!(concat!(env!("OUT_DIR"), "/queue_reorder_test.spv")),
        52,
        1,
    );
    let reorder = probe.pipeline(
        &r,
        include_bytes!(concat!(env!("OUT_DIR"), "/queue_reorder.spv")),
        52,
        1,
    );
    for (n, dead) in [
        (0, 0),
        (1, 0),
        (63, 0),
        (64, 0),
        (65, 0),
        (1023, 0),
        (1024, 0),
        (1025, 0),
        (4097, 0),
        (1024, 1),
    ] {
        r.resize(n.max(1), 1).unwrap();
        r.max_groups_x = 2; // Exercise two-dimensional indirect queue indexing too.
        let mut words = Vec::new();
        for i in 0..n {
            let octant = i % 8;
            let mut ray = [0u32; 16];
            for (j, v) in ray.iter_mut().enumerate().take(12) {
                *v = (i as f32 + j as f32 / 16.).to_bits();
            }
            for axis in 0..3 {
                ray[4 + axis] = (if octant & (1 << axis) != 0 {
                    1f32
                } else {
                    -1f32
                })
                .to_bits();
            }
            ray[12] = i;
            ray[13] = 0x123456;
            ray[14] = (((i % 7) | (if i % 2 == 0 { 0 } else { 128 })) << 24) | 0x030201;
            ray[15] = 1 + i % 3;
            words.extend(ray);
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        r.passes[1].scratch[0] =
            Buffer::uploaded(r.device.clone(), bytemuck::cast_slice(&words), usage).unwrap();
        r.passes[1].scratch[1] =
            Buffer::new(r.device.clone(), n.max(1) as u64 * 64, usage, false).unwrap();
        r.passes[1].scratch[2] = Buffer::new(
            r.device.clone(),
            48,
            usage | vk::BufferUsageFlags::INDIRECT_BUFFER,
            true,
        )
        .unwrap();
        r.passes[1].scratch[2]
            .write(bytemuck::cast_slice(&[n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            .unwrap();
        r.update_scene_parameters(&scene).unwrap();
        let mut visibility = vec![0u32; n.max(1) as usize * 8];
        for i in 0..n as usize {
            visibility[i * 8] = if i % 2 == 0 { 1 } else { 2 };
            visibility[i * 8 + 4] = i as u32 % 7;
        }
        r.passes[0]
            .output
            .upload_staged_chunks(bytemuck::cast_slice(&visibility), 16 * 1024 * 1024)
            .unwrap();
        let push = Push {
            width: n.max(1),
            height: 1,
            ray_phase: 1,
            queue_parity: 0,
            filter_phase: dead,
            max_groups_x: 2,
            mesh_triangles: 1,
            ..bytemuck::Zeroable::zeroed()
        };
        probe.measure(&r, &push, pipeline, 2);
        let count = read_words(r.device.clone(), &r.passes[1].scratch[2], 4, 1)[0] as usize;
        let expected: Vec<u32> = (0..n)
            .filter(|i| dead == 0 && i % 3 != 0 && (i / 64 % 3 != 1 || i % 64 < 12))
            .collect();
        assert_eq!(count, expected.len());
        if count == 0 {
            continue;
        }
        let mut sorted_push = push;
        sorted_push.queue_parity = 1;
        probe.measure(&r, &sorted_push, reorder, 2);

        let output = read_words(r.device.clone(), &r.passes[1].scratch[1], 0, count * 16);
        let mut ids = Vec::new();
        let key = |ray: &[u32]| {
            let mut oct = 0;
            for a in 0..3 {
                if f32::from_bits(ray[4 + a]) >= 0. {
                    oct |= 1 << a;
                }
            }
            (ray[15] << 11) | ((ray[14] >> 24) << 3) | oct
        };
        for ray in output.as_chunks::<16>().0 {
            let id = ray[12];
            assert_eq!(ray, &words[id as usize * 16..id as usize * 16 + 16]);
            ids.push(id);
        }
        if count >= 1024 {
            for block in output.chunks(256 * 16) {
                if block.len() / 16 < 16 {
                    continue;
                }
                let mut previous = 0;
                for ray in block.as_chunks::<16>().0 {
                    assert!(key(ray) >= previous);
                    previous = key(ray);
                }
            }
        }
        ids.sort_unstable();
        assert_eq!(ids, expected, "lost/duplicated paths for n={n}");
    }
}

#[test]
#[ignore = "requires GPU; optionally set LAVA_REORDER_SCENE for full-scene reordering comparison"]
fn benchmark_ray_reordering_image_and_total_frame_time() {
    let scene: Scene = if let Ok(path) = std::env::var("LAVA_REORDER_SCENE") {
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    } else {
        serde_json::from_str(include_str!("../examples/studio.json")).unwrap()
    };
    let width = std::env::var("LAVA_REORDER_WIDTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let height = std::env::var("LAVA_REORDER_HEIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(144);
    let mut r =
        Renderer::new_with_ray_reordering(width, height, Mode::Interactive, None, false).unwrap();
    r.gpu_output = true;
    r.linear_output = true;
    r.set_scene(&scene).unwrap();
    while r.textures_pending() {
        r.refresh_textures().unwrap();
        std::thread::yield_now();
    }
    let mut reports = Vec::new();
    let mut reference: Option<Vec<f32>> = None;
    for trial in 0..3 {
        for mode in if trial % 2 == 0 { [0, 1] } else { [1, 0] } {
            r.reorder_rays = mode == 1;
            r.reset_accumulation().unwrap();
            let mut times = Vec::new();
            for sample in 0..20 {
                let started = std::time::Instant::now();
                r.render_batch(20 - sample, 0.).unwrap();
                if sample >= 4 {
                    times.push(started.elapsed().as_secs_f64() * 1000.);
                }
            }
            let image = r.gpu_frame.as_ref().unwrap().read_color().unwrap();
            if let Some(expected) = &reference {
                let error = image
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(error < 1e-5, "reordering changed pixel radiance: {error}");
            } else {
                reference = Some(image);
            }
            reports.push(
                json!({"trial":trial,"reordering":mode==1,"total_frame_wall_ms":summarize(&times)}),
            );
        }
    }

    let report = json!({"resolution":[width,height],"reports":reports,"notes":"Identical pixel RNG/path payloads; total per-sample wall time includes compaction/sorting, all passes and GPU output allocation. One explicit diagnostic readback per trial/variant for image comparison."});
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if let Ok(path) = std::env::var("LAVA_REORDER_OUTPUT") {
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
}
