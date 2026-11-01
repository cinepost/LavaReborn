use super::path_tests::{buffer_snapshot, fixture};
use super::*;
fn filter(r: &Renderer) -> Vec<f32> {
    let push = Push {
        width: r.width,
        height: r.height,
        radius: 1,
        filter_phase: 1,
        ..bytemuck::Zeroable::zeroed()
    };
    r.device
        .submit(|cmd| unsafe {
            let ready = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            r.device.raw.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &ready,
                &[],
                &[],
            );
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
            let pipeline = if r.settings.adaptive_sampling {
                r.adaptive.as_ref().unwrap().pipelines[2]
            } else {
                r.passes[2].pipeline
            };
            r.device
                .raw
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            r.device
                .raw
                .cmd_dispatch(cmd, r.width.div_ceil(8), r.height.div_ceil(8), 1);
        })
        .unwrap();
    buffer_snapshot(r, r.passes[2].filtered.as_ref().unwrap())
}
#[test]
#[ignore = "requires Vulkan; HDR filter ignores arbitrary stochastic visibility, smooths noise and preserves color edges"]
fn filter_uses_history_without_stochastic_visibility() {
    let mut r = Renderer::new(9, 3, Mode::Final, None).unwrap();
    for adaptive in [false, true] {
        let mut scene = fixture();
        scene.settings.adaptive_sampling = adaptive;
        r.set_scene(&scene).unwrap();
        let mut history = Vec::new();
        for _ in 0..3 {
            for x in 0..9 {
                let color = if x < 3 {
                    let value = [0.97, 1.03, 1.][x];
                    [value, value, value, 1.]
                } else if x < 6 {
                    [4., 0., 0., 1.]
                } else {
                    [0., 4., 0., 1.]
                };
                history.extend(color);
            }
        }
        r.passes[2]
            .output
            .upload_staged_chunks(bytemuck::cast_slice(&history), 4096)
            .unwrap();
        r.passes[0]
            .output
            .upload_staged_chunks(bytemuck::cast_slice(&vec![0u32; 9 * 3 * 8]), 4096)
            .unwrap();
        let original = filter(&r);
        let arbitrary: Vec<u32> = (0..9 * 3 * 8).map(|i| u32::MAX - i as u32).collect();
        r.passes[0]
            .output
            .upload_staged_chunks(bytemuck::cast_slice(&arbitrary), 4096)
            .unwrap();
        assert_eq!(
            original,
            filter(&r),
            "filter depends on stochastic visibility"
        );
        assert_eq!(
            history,
            buffer_snapshot(&r, &r.passes[2].output),
            "filter changed accumulated history"
        );
        assert!(
            (original[(9 + 1) * 4] - 1.).abs() < 0.02,
            "low-contrast noise was not smoothed"
        );
        for x in 3..9 {
            let p = &original[(9 + x) * 4..][..4];
            if x < 6 {
                assert!(
                    (p[0] - 4.).abs() < 1e-5 && p[1].abs() < 1e-5 && p[2].abs() < 1e-5,
                    "red edge bled: {p:?}"
                );
            } else {
                assert!(
                    p[0].abs() < 1e-5 && (p[1] - 4.).abs() < 1e-5 && p[2].abs() < 1e-5,
                    "green edge bled: {p:?}"
                );
            }
        }
        for color in [[0., 0., 0., 1.], [1e30, 2e30, 3e30, 1.]] {
            let constant: Vec<f32> = (0..27).flat_map(|_| color).collect();
            r.passes[2]
                .output
                .upload_staged_chunks(bytemuck::cast_slice(&constant), 4096)
                .unwrap();
            let output = filter(&r);
            assert!(output.iter().all(|v| v.is_finite()));
            for p in output.as_chunks::<4>().0 {
                for c in 0..4 {
                    assert!((p[c] - color[c]).abs() <= color[c].abs() * 1e-6);
                }
            }
        }
        assert_eq!(r.samples(), 0);
    }
}
