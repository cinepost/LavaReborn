use super::*;
impl Renderer {
    pub(super) fn ensure_adaptive(&mut self) -> Result<()> {
        if self.adaptive.is_some() {
            return Ok(());
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let mut resources = AdaptiveResources {
            device: self.device.clone(),
            pipelines: Vec::new(),
            pixels: Buffer::new(
                self.device.clone(),
                self.width as u64 * self.height as u64 * 4,
                usage,
                false,
            )?,
            control: Buffer::new(
                self.device.clone(),
                32,
                usage | vk::BufferUsageFlags::INDIRECT_BUFFER,
                false,
            )?,
            readback: Buffer::new(
                self.device.clone(),
                4,
                vk::BufferUsageFlags::TRANSFER_DST,
                true,
            )?,
        };
        let products: [&[u8]; 6] = [
            include_bytes!(concat!(env!("OUT_DIR"), "/visibility_adaptive.spv")),
            include_bytes!(concat!(env!("OUT_DIR"), "/shading_adaptive.spv")),
            include_bytes!(concat!(env!("OUT_DIR"), "/accumulation_adaptive.spv")),
            include_bytes!(concat!(env!("OUT_DIR"), "/visibility_resolve_adaptive.spv")),
            include_bytes!(concat!(env!("OUT_DIR"), "/adaptive_prepare.spv")),
            include_bytes!(concat!(env!("OUT_DIR"), "/adaptive_finalize.spv")),
        ];
        for (i, product) in products.iter().enumerate() {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(product))?;
            unsafe {
                let module = self.device.raw.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&code),
                    None,
                )?;
                let entries = [vk::SpecializationMapEntry::default()
                    .constant_id(50)
                    .size(4)];
                let value =
                    u32::from(std::env::var("LAVA_CURVE_POLYGON_FIRST").as_deref() == Ok("1"))
                        .to_ne_bytes();
                let info = vk::SpecializationInfo::default()
                    .map_entries(&entries)
                    .data(&value);
                let mut stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main");
                if i == 0 {
                    stage = stage.specialization_info(&info);
                }
                let started = std::time::Instant::now();
                if std::env::var_os("LAVA_TRACE").is_some() {
                    eprintln!("Lava pipeline: compiling adaptive product {i}");
                }
                let result = self.device.raw.create_compute_pipelines(
                    self.pipeline_cache.raw,
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(self.layout)],
                    None,
                );
                self.device.raw.destroy_shader_module(module, None);
                match result {
                    Ok(p) => resources.pipelines.push(p[0]),
                    Err((partial, e)) => {
                        for p in partial {
                            self.device.raw.destroy_pipeline(p, None);
                        }
                        return Err(e.into());
                    }
                }
                if std::env::var_os("LAVA_TRACE").is_some() {
                    eprintln!(
                        "Lava pipeline: adaptive product {i} ready in {:.3}s",
                        started.elapsed().as_secs_f64()
                    );
                }
            }
        }
        self.passes[2].scratch.push(Buffer::new(
            self.device.clone(),
            self.width as u64 * self.height as u64 * 16,
            usage,
            false,
        )?);
        self.adaptive = Some(resources);
        self.pipeline_cache.persist();
        Ok(())
    }
    pub(super) unsafe fn record_adaptive_prepare(&self, cmd: vk::CommandBuffer, _push: &Push) {
        let a = self.adaptive.as_ref().unwrap();
        let d = &self.device.raw;
        let before = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &before,
            &[],
            &[],
        );
        d.cmd_fill_buffer(cmd, a.control.raw, 0, 4, 0);
        let filled = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &filled,
            &[],
            &[],
        );
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, a.pipelines[4]);
        let groups = (self.width * self.height).div_ceil(64);
        let x = groups.min(self.max_groups_x);
        d.cmd_dispatch(cmd, x, groups.div_ceil(x), 1);
        let written = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &written,
            &[],
            &[],
        );
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, a.pipelines[5]);
        d.cmd_dispatch(cmd, 1, 1, 1);
        let ready = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::INDIRECT_COMMAND_READ,
            )];
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
            vk::DependencyFlags::empty(),
            &ready,
            &[],
            &[],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::path_tests::fixture;
    use super::*;
    fn snapshot(r: &Renderer, b: &Buffer) -> Vec<f32> {
        let out = Buffer::new(
            r.device.clone(),
            b.size,
            vk::BufferUsageFlags::TRANSFER_DST,
            true,
        )
        .unwrap();
        r.device
            .submit(|cmd| unsafe {
                let ready = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
                r.device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &ready,
                    &[],
                    &[],
                );
                r.device.raw.cmd_copy_buffer(
                    cmd,
                    b.raw,
                    out.raw,
                    &[vk::BufferCopy::default().size(b.size)],
                );
                let ready = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                r.device.raw.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &ready,
                    &[],
                    &[],
                );
            })
            .unwrap();
        out.read_floats((b.size / 4) as usize).unwrap()
    }
    fn run(r: &mut Renderer, target: u32) -> Vec<f32> {
        let mut image = Vec::new();
        while r.samples() < target {
            image = r.render_batch(target - r.samples(), 0.).unwrap();
        }
        image
    }
    #[test]
    #[ignore = "requires Vulkan; adaptive queue pixel mapping, per-pixel moments and reset"]
    fn adaptive_matches_uniform_before_stopping_and_resets() {
        let mut r = Renderer::new(67, 19, Mode::Interactive, None).unwrap();
        r.linear_output = true;
        r.max_groups_x = 2;
        let mut scene = fixture();
        r.set_scene(&scene).unwrap();
        run(&mut r, 16);
        let uniform = snapshot(&r, &r.passes[2].output);
        scene.settings.adaptive_sampling = true;
        scene.settings.adaptive_min_samples = 1024;
        r.set_scene(&scene).unwrap();
        run(&mut r, 16);
        let adaptive = snapshot(&r, &r.passes[2].output);
        assert_eq!(
            uniform, adaptive,
            "pixel mapping or sampling sequence changed"
        );
        let stats = snapshot(&r, &r.passes[2].scratch[0]);
        assert!(stats
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[0].to_bits() == 16 && p[2] >= 0.));
        r.reset_accumulation().unwrap();
        assert!(snapshot(&r, &r.passes[2].scratch[0])
            .iter()
            .all(|v| *v == 0.));
        run(&mut r, 2);
        assert!(snapshot(&r, &r.passes[2].scratch[0])
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[0].to_bits() == 2));
        r.resize(31, 7).unwrap();
        run(&mut r, 3);
        let stats = snapshot(&r, &r.passes[2].scratch[0]);
        assert_eq!(stats.len(), 31 * 7 * 4);
        assert!(stats.as_chunks::<4>().0.iter().all(|p| p[0].to_bits() == 3));
    }
    #[test]
    #[ignore = "requires Vulkan; adaptive early convergence, frozen history, raster and final completion"]
    fn adaptive_converges_and_preserves_frozen_pixels() {
        for visibility in [
            crate::integrator::VisibilityMode::RayQuery,
            crate::integrator::VisibilityMode::Raster,
        ] {
            let mut r = Renderer::new(17, 9, Mode::Final, None).unwrap();
            r.linear_output = true;
            let mut scene = fixture();
            scene.settings.visibility = visibility;
            scene.settings.adaptive_sampling = true;
            // Half the screen is a constant background; the other half has noisy area lighting.
            scene.camera.horizontal_aperture = 2.;
            scene.camera.vertical_aperture = 1.;
            scene.meshes[0].points = vec![
                [-10., -10., 0.],
                [0., -10., 0.],
                [0., 10., 0.],
                [-10., 10., 0.],
            ];
            scene.settings.adaptive_threshold = 0.0001;
            r.set_scene(&scene).unwrap();
            let image = run(&mut r, 64);
            assert!(!image.is_empty() && image.iter().all(|v| v.is_finite()));
            let stats = snapshot(&r, &r.passes[2].scratch[0]);
            let counts: Vec<_> = stats
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[0].to_bits())
                .collect();
            assert!(
                counts.contains(&24) && counts.contains(&64),
                "not independently adaptive: {counts:?}"
            );
            scene.lights.clear();
            scene.meshes[0].material.metallic = 0.;
            scene.settings.adaptive_threshold = 1.;
            r.set_scene(&scene).unwrap();
            r.render_batch(256, 0.).unwrap();
            let image = r.render_batch(256 - r.samples(), 0.).unwrap();
            assert_eq!(r.samples(), 256, "did not signal early completion");
            assert!(!image.is_empty());
            assert!(snapshot(&r, &r.passes[2].scratch[0])
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0].to_bits() == 24));
            let history = snapshot(&r, &r.passes[2].output);
            r.render_batch(0, 1.).unwrap();
            assert_eq!(history, snapshot(&r, &r.passes[2].output));
        }
    }
    #[test]
    #[ignore = "requires Vulkan interop; adaptive completion and exposure use GPU-only presentation"]
    fn adaptive_gpu_output_and_toggle_keep_image_on_device() {
        let mut r = Renderer::new(17, 9, Mode::Interactive, None).unwrap();
        if !r.device.interop {
            return;
        }
        r.gpu_output = true;
        r.linear_output = true;
        let mut scene = fixture();
        scene.lights.clear();
        scene.settings.adaptive_sampling = true;
        r.set_scene(&scene).unwrap();
        let before = crate::interop::IMAGE_READBACKS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(run(&mut r, 128).is_empty());
        assert!(r.gpu_frame.is_some());
        assert_eq!(r.samples(), 128);
        r.render_batch(0, 1.).unwrap();
        scene.settings.adaptive_sampling = false;
        r.update_scene_parameters(&scene).unwrap();
        run(&mut r, 2);
        assert!(r.gpu_frame.is_some());
        assert_eq!(
            before,
            crate::interop::IMAGE_READBACKS.load(std::sync::atomic::Ordering::Relaxed)
        );
    }
    #[test]
    #[ignore = "requires Vulkan; adaptive convergence mask must retain neighbors of noisy pixels"]
    fn adaptive_guard_retains_apparently_clean_neighbors() {
        let mut r = Renderer::new(5, 5, Mode::Interactive, None).unwrap();
        let mut scene = fixture();
        scene.settings.adaptive_sampling = true;
        r.set_scene(&scene).unwrap();
        let push = Push {
            width: 5,
            height: 5,
            sample: 24,
            target_samples: 256,
            max_groups_x: r.max_groups_x,
            ..bytemuck::Zeroable::zeroed()
        };
        let mut stats: Vec<u32> = (0..25)
            .flat_map(|_| [24, 0.02f32.to_bits(), 0, 3])
            .collect();
        for (noisy, expected) in [
            (Some(12usize), vec![6, 7, 8, 11, 12, 13, 16, 17, 18]),
            (Some(0), vec![0, 1, 5, 6]),
            (None, Vec::new()),
        ] {
            for p in stats.as_chunks_mut::<4>().0 {
                p[3] = 3;
            }
            if let Some(i) = noisy {
                stats[i * 4 + 3] = 0;
            }
            r.passes[2].scratch[0]
                .upload_staged_chunks(bytemuck::cast_slice(&stats), 4096)
                .unwrap();
            r.device
                .submit(|cmd| unsafe {
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
                        bytemuck::bytes_of(&push),
                    );
                    r.record_adaptive_prepare(cmd, &push);
                })
                .unwrap();
            let a = r.adaptive.as_ref().unwrap();
            let n = snapshot(&r, &a.control)[0].to_bits() as usize;
            let mut pixels: Vec<_> = snapshot(&r, &a.pixels)[..n]
                .iter()
                .map(|v| v.to_bits())
                .collect();
            pixels.sort_unstable();
            assert_eq!(
                pixels, expected,
                "isolated clean prefix escaped noisy-neighbor guard"
            );
        }
    }
    #[test]
    #[ignore = "requires LAVA_BRIGHTNESS_SCENE; diagnostic adaptive brightness trajectory"]
    fn profile_adaptive_brightness() {
        let Ok(path) = std::env::var("LAVA_BRIGHTNESS_SCENE") else {
            return;
        };
        let mut scene: Scene = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let target = std::env::var("LAVA_BRIGHTNESS_SAMPLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256u32);
        let mut r = Renderer::new(640, 360, Mode::Interactive, None).unwrap();
        r.linear_output = true;
        let mut report = Vec::new();
        let mut reference: Vec<f32> = Vec::new();
        for adaptive in [false, true] {
            scene.settings.adaptive_sampling = adaptive;
            r.set_scene(&scene).unwrap();
            let mut previous = Vec::new();
            let mut previous_stats: Vec<f32> = Vec::new();
            let mut previous_image = Vec::new();
            let mut records = Vec::new();
            let mut step = 0;
            while r.samples() < target {
                step += 1;
                let image = r.render_batch(target - r.samples(), 0.).unwrap();
                let raw = snapshot(&r, &r.passes[2].output);
                let stats = if adaptive {
                    snapshot(&r, &r.passes[2].scratch[0])
                } else {
                    Vec::new()
                };
                let active = if adaptive {
                    r.adaptive
                        .as_ref()
                        .unwrap()
                        .readback
                        .read_floats(1)
                        .unwrap()[0]
                        .to_bits()
                } else {
                    r.width * r.height
                };
                if adaptive && !previous_stats.is_empty() {
                    for i in 0..(r.width * r.height) as usize {
                        if stats[i * 4].to_bits() == previous_stats[i * 4].to_bits() {
                            assert_eq!(
                                &raw[i * 4..i * 4 + 4],
                                &previous[i * 4..i * 4 + 4],
                                "frozen pixel changed at step {step} pixel {i}"
                            );
                        }
                    }
                    if active == 0 {
                        assert_eq!(raw, previous, "completion changed history");
                        assert_eq!(image, previous_image, "completion changed display");
                    }
                }
                let average = |pixels: &[f32]| {
                    let mut sum = 0f64;
                    let mut count = 0;
                    for y in r.height / 4..3 * r.height / 4 {
                        for x in r.width / 4..3 * r.width / 4 {
                            let i = ((y * r.width + x) * 4) as usize;
                            sum += pixels[i] as f64 * 0.2126
                                + pixels[i + 1] as f64 * 0.7152
                                + pixels[i + 2] as f64 * 0.0722;
                            count += 1;
                        }
                    }
                    sum / count as f64
                };
                let raw_mean = average(&raw);
                let display_mean = average(&image);
                if step <= 2 || step % 8 == 0 || r.samples() == target {
                    println!("brightness adaptive={adaptive} step={step} reported={} active={active} raw={raw_mean:.8} display={display_mean:.8}",r.samples());
                }
                records.push(serde_json::json!({"step":step,"reported":r.samples(),"active":active,"raw":raw_mean,"display":display_mean}));
                previous = raw;
                previous_stats = stats;
                previous_image = image;
            }
            image::Rgba32FImage::from_raw(r.width, r.height, previous.clone())
                .unwrap()
                .save(format!("build/adaptive-brightness/raw_{adaptive}.exr"))
                .unwrap();
            let preview: Vec<u8> = previous_image
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    [p[0], p[1], p[2]]
                        .map(|v| (v.max(0.).powf(1. / 2.2).clamp(0., 1.) * 255.).round() as u8)
                })
                .collect();
            image::save_buffer(
                format!("build/adaptive-brightness/display_{adaptive}.png"),
                &preview,
                r.width,
                r.height,
                image::ColorType::Rgb8,
            )
            .unwrap();
            if adaptive {
                let mut dark = 0;
                let mut bright = 0;
                let mut largest = Vec::new();
                for (i, (a, b)) in previous
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(reference.as_chunks::<4>().0.iter())
                    .enumerate()
                {
                    let luma = |v: &[f32; 4]| v[0] * 0.2126 + v[1] * 0.7152 + v[2] * 0.0722;
                    let a = luma(a);
                    let b = luma(b);
                    if b > 0.005 {
                        if a < b * 0.8 {
                            dark += 1;
                        }
                        if a > b * 1.2 {
                            bright += 1;
                        }
                    }
                    if b > 0.005 {
                        largest.push((b - a, i, previous_stats[i * 4].to_bits(), a, b));
                    }
                }
                largest.sort_by(|a, b| b.0.total_cmp(&a.0));
                println!(
                    "brightness local dark20={dark} bright20={bright} worst={:?}",
                    &largest[..20]
                );
            } else {
                reference = previous.clone();
            }
            report.push(serde_json::json!({"adaptive":adaptive,"settings":scene.settings,"trajectory":records}));
        }
        std::fs::write(
            "build/adaptive-brightness/trajectory.json",
            serde_json::to_string_pretty(&report).unwrap(),
        )
        .unwrap();
    }
    #[test]
    #[ignore = "requires LAVA_ADAPTIVE_SCENE and Vulkan; timed adaptive/uniform scene comparison"]
    fn profile_adaptive_scene() {
        let Ok(path) = std::env::var("LAVA_ADAPTIVE_SCENE") else {
            return;
        };
        let mut scene: Scene = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let width = std::env::var("LAVA_ADAPTIVE_WIDTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1280);
        let height = std::env::var("LAVA_ADAPTIVE_HEIGHT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(720);
        let budget = std::env::var("LAVA_ADAPTIVE_SAMPLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256u32);
        let mut r = Renderer::new(width, height, Mode::Final, None).unwrap();
        r.linear_output = true;
        let mut results = Vec::new();
        let mut images = Vec::new();
        let mut raw_images = Vec::new();
        for (label, enabled, threshold, target, minimum) in [
            ("reference", false, 0.03, budget * 4, 32),
            ("uniform", false, 0.03, budget, 32),
            ("uniform_halton", false, 0.03, budget, 32),
            ("adaptive_003", true, 0.03, budget, 32),
            ("adaptive_001", true, 0.01, budget, 32),
            ("adaptive_min128", true, 0.03, budget, 128),
            ("adaptive_no_stop", true, 0.03, budget, 1024),
        ] {
            r.progressive_sampling_reference = label == "uniform_halton";
            scene.settings.adaptive_sampling = enabled;
            scene.settings.adaptive_threshold = threshold;
            scene.settings.adaptive_min_samples = minimum;
            r.set_scene(&scene).unwrap();
            run(&mut r, 4);
            r.reset_accumulation().unwrap();
            let started = std::time::Instant::now();
            let image = run(&mut r, target);
            let seconds = started.elapsed().as_secs_f64();
            let (mean_samples, min_samples, max_samples) = if enabled {
                let stats = snapshot(&r, &r.passes[2].scratch[0]);
                let counts: Vec<_> = stats
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| p[0].to_bits())
                    .collect();
                (
                    counts.iter().map(|&n| n as f64).sum::<f64>() / counts.len() as f64,
                    *counts.iter().min().unwrap(),
                    *counts.iter().max().unwrap(),
                )
            } else {
                (target as f64, target, target)
            };
            println!("Adaptive benchmark: {label} {seconds:.3}s mean_samples={mean_samples:.2} min={min_samples} max={max_samples}");
            results.push(serde_json::json!({"mode":label,"seconds":seconds,"mean_samples":mean_samples,"min_samples":min_samples,"max_samples":max_samples,"budget":target,"minimum_setting":minimum}));
            images.push(image);
            raw_images.push(snapshot(&r, &r.passes[2].output));
        }
        for i in 1..images.len() {
            let mut error = 0f64;
            let mut energy = 0f64;
            for (a, b) in images[i]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(images[0].as_chunks::<4>().0.iter())
            {
                for c in 0..3 {
                    error += (a[c] as f64 - b[c] as f64).powi(2);
                    energy += (b[c] as f64).powi(2);
                }
            }
            results[i]["relative_rmse"] = serde_json::json!((error / energy.max(1e-20)).sqrt());
        }
        for i in 1..raw_images.len() {
            let mut error = 0f64;
            let mut energy = 0f64;
            for (a, b) in raw_images[i]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(raw_images[0].as_chunks::<4>().0.iter())
            {
                for c in 0..3 {
                    error += (a[c] as f64 - b[c] as f64).powi(2);
                    energy += (b[c] as f64).powi(2);
                }
            }
            results[i]["raw_relative_rmse"] = serde_json::json!((error / energy.max(1e-20)).sqrt());
        }
        let report = serde_json::json!({"resolution":[width,height],"settings":scene.settings,"results":results});
        let output = std::env::var("LAVA_ADAPTIVE_REPORT")
            .unwrap_or("build/adaptive-profile/report.json".into());
        std::fs::write(output, serde_json::to_string_pretty(&report).unwrap()).unwrap();
        println!("{report}");
    }
}
