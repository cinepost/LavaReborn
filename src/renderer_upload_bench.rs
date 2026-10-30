//! Opt-in upload/viewport contention measurement. Renderer defaults stay unchanged.
use super::*;
use serde_json::json;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Barrier,
};
use std::time::Instant;

fn stats(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return json!({"count":0});
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let middle = v.len() / 2;
    let median = if v.len().is_multiple_of(2) {
        (v[middle - 1] + v[middle]) * 0.5
    } else {
        v[middle]
    };
    json!({"count":v.len(),"median":median,"mean":v.iter().sum::<f64>()/v.len() as f64,
        "p95":v[(v.len() as f64*0.95).ceil() as usize-1],"max":v[v.len()-1],"min":v[0]})
}
fn gpu_frame(r: &mut Renderer) {
    let image = r.render_batch(1, 0.).unwrap();
    assert!(
        image.is_empty() && r.gpu_output,
        "benchmark requires GPU-only output"
    );
}
fn barrier(d: &Device) {
    d.submit(|cmd| unsafe {
        d.raw.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
            &[],
            &[],
        );
    })
    .unwrap();
}
fn verify(d: Arc<Device>, output: &Buffer, expected: &[u8]) {
    let staging = Buffer::new(d.clone(), 32, vk::BufferUsageFlags::TRANSFER_DST, true).unwrap();
    d.submit(|cmd| unsafe {
        d.raw.cmd_copy_buffer(
            cmd,
            output.raw,
            staging.raw,
            &[
                vk::BufferCopy::default().size(16),
                vk::BufferCopy::default()
                    .src_offset(expected.len() as u64 - 16)
                    .dst_offset(16)
                    .size(16),
            ],
        );
        d.raw.cmd_pipeline_barrier(
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
    let words = staging
        .read_floats(8)
        .unwrap()
        .iter()
        .map(|f| f.to_bits())
        .collect::<Vec<_>>();
    let mut want = expected[..16].to_vec();
    want.extend_from_slice(&expected[expected.len() - 16..]);
    assert!(
        bytemuck::cast_slice::<u32, u8>(&words) == want,
        "uploaded prefix/suffix differs"
    );
}
#[test]
#[ignore = "requires LAVA_UPLOAD_SCENE and hardware Vulkan; chunk-size throughput and viewport contention"]
fn measure_staging_chunk_sizes() {
    let Ok(path) = std::env::var("LAVA_UPLOAD_SCENE") else {
        println!("Set LAVA_UPLOAD_SCENE to run upload measurements");
        return;
    };
    let scene: Scene = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let mut r = Renderer::new(1920, 1080, Mode::Interactive, None).unwrap();
    assert!(
        r.device.interop,
        "GPU interop required for the foreground workload"
    );
    r.gpu_output = true;
    r.set_scene(&scene).unwrap();
    r.refresh_textures().unwrap();
    while r.textures_pending() {
        r.refresh_textures().unwrap();
        std::thread::yield_now();
    }
    let d = r.device.clone();
    for _ in 0..12 {
        gpu_frame(&mut r);
    }
    let baseline: Vec<f64> = (0..60)
        .map(|_| {
            let t = Instant::now();
            gpu_frame(&mut r);
            t.elapsed().as_secs_f64() * 1000.
        })
        .collect();
    let (triangles, _) = scene.prepare_geometry().unwrap();
    let base = bytemuck::cast_slice::<_, u8>(&triangles);
    let base_len = base.len();
    println!(
        "Preparing contiguous 10x source: {} bytes; baseline frame stats {}",
        base_len * 10,
        stats(&baseline)
    );
    let scales: Vec<usize> = if std::env::var("LAVA_UPLOAD_SMOKE").as_deref() == Ok("1") {
        vec![1]
    } else {
        vec![1, 5, 10]
    };
    let source = base.repeat(*scales.iter().max().unwrap());
    drop(triangles);
    let rounds = std::env::var("LAVA_UPLOAD_ROUNDS")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(3usize);
    let mut rows = Vec::new();
    let chunk_sizes = [4usize, 16, 32, 64];
    // Deliberately cap new allocations at 5x: keep old scene / image / AS headroom.
    let resident_limit = base_len * 5;
    let mut renderer = Some(r);
    for scale in scales {
        let bytes = &source[..base_len * scale];
        let capacity = bytes.len().min(resident_limit);
        for round in 0..rounds {
            for slot in 0..chunk_sizes.len() {
                let mib = chunk_sizes[(slot + round) % chunk_sizes.len()];
                for foreground in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let ready = Arc::new(Barrier::new(2));
                    let stop = Arc::new(AtomicBool::new(false));
                    let worker = if foreground {
                        let mut r = renderer.take().unwrap();
                        let ready = ready.clone();
                        let stop = stop.clone();
                        Some(std::thread::spawn(move || {
                            for _ in 0..8 {
                                gpu_frame(&mut r);
                            }
                            ready.wait();
                            crate::vulkan::begin_submit_timings();
                            let mut frames = Vec::new();
                            while !stop.load(Ordering::Acquire) {
                                let t = Instant::now();
                                gpu_frame(&mut r);
                                frames.push(t.elapsed().as_secs_f64() * 1000.);
                            }
                            (r, frames, crate::vulkan::take_submit_timings())
                        }))
                    } else {
                        None
                    };
                    if foreground {
                        ready.wait();
                    }
                    crate::vulkan::begin_submit_timings();
                    let started = Instant::now();
                    let allocation = Instant::now();
                    let output = Buffer::new(
                        d.clone(),
                        capacity as u64,
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
                        false,
                    )
                    .unwrap();
                    let allocation_ms = allocation.elapsed().as_secs_f64() * 1000.;
                    let upload = Instant::now();
                    for (window, part) in bytes.chunks(capacity).enumerate() {
                        if window > 0 {
                            barrier(&d);
                        }
                        output
                            .upload_staged_chunks(part, mib * 1024 * 1024)
                            .unwrap();
                    }
                    let upload_ms = upload.elapsed().as_secs_f64() * 1000.;
                    let wall_ms = started.elapsed().as_secs_f64() * 1000.;
                    let submit = crate::vulkan::take_submit_timings();
                    stop.store(true, Ordering::Release);
                    let (frames, render_submit) = if let Some(worker) = worker {
                        let (r, frames, times) = worker.join().unwrap();
                        renderer = Some(r);
                        (frames, times)
                    } else {
                        (Vec::new(), Vec::new())
                    };
                    verify(d.clone(), &output, bytes.chunks(capacity).last().unwrap());
                    let upload_wait: Vec<_> = submit.iter().map(|s| s[0]).collect();
                    let holds: Vec<_> = submit.iter().map(|s| s[1]).collect();
                    let render_wait: Vec<_> = render_submit.iter().map(|s| s[0]).collect();
                    let row = json!({"scale":scale,"chunk_mib":mib,"round":round,"foreground":foreground,
                        "bytes":bytes.len(),"resident_bytes":capacity,"windowed":bytes.len()>capacity,
                        "allocation_ms":allocation_ms,"upload_ms":upload_ms,"total_ms":wall_ms,
                        "effective_gib_s":bytes.len() as f64/(1024f64.powi(3))/(upload_ms/1000.),
                        "upload_queue_wait_ms":stats(&upload_wait),"upload_lock_hold_ms":stats(&holds),
                        "upload_queue_wait_total_ms":upload_wait.iter().sum::<f64>(),
                        "upload_lock_hold_total_ms":holds.iter().sum::<f64>(),
                        "frame_ms":stats(&frames),"frame_samples_ms":frames,"render_queue_wait_ms":stats(&render_wait)});
                    println!("scale={scale} chunk={mib}MiB round={round} foreground={foreground} upload={upload_ms:.1}ms frame={} hold={}",row["frame_ms"],row["upload_lock_hold_ms"]);
                    rows.push(row);
                }
            }
        }
        let output_path = std::env::var("LAVA_UPLOAD_OUTPUT")
            .unwrap_or_else(|_| "build/upload-sizes.json".into());
        let report = json!({"scene":path,"source_geometry_bytes":base_len,"resident_limit":resident_limit,
            "foreground_resolution":[1920,1080],"foreground_settings":scene.settings,"baseline_frame_ms":stats(&baseline),
            "rounds":rounds,"rows":rows,"notes":["Real contiguous repeated geometry records, one graphics/compute queue and the production staging implementation.",
            "5x is fully resident; 10x is streamed into two 5x resident windows to preserve GPU memory headroom.",
            "Foreground is real 1920x1080 kapibara rendering with GPU output and no OpenGL consumer; no image readback.",
            "Frame elapsed times include CPU submission, GPU execution and contention; they exclude Houdini/GL presentation.",
            "Queue hold timings include command recording, submit and fence wait; they are wall-clock lock residency, not isolated GPU DMA timestamps.",
            "This tests upload volume scaling, not complete 5x/10x scene acceleration structures or render complexity."]});
        std::fs::write(output_path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
}
