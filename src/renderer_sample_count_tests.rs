use super::path_tests::{buffer_snapshot, fixture};
use super::*;
fn heat(count: u32, target: u32) -> [f32; 4] {
    if count == 0 {
        return [0., 0., 0., 1.];
    }
    let t = count as f32 / target as f32;
    let mut result = [0., 0., 0., 1.];
    for (channel, center) in [3., 2., 1.].iter().enumerate() {
        result[channel] = (1.5 - (4. * t - center).abs()).clamp(0., 1.);
    }
    result
}
#[test]
#[ignore = "requires Vulkan; opt-in diagnostic counts, display refresh, resizing and external memory"]
fn sample_count_aov_is_opt_in_and_matches_pixel_counts() {
    let mut r = Renderer::new(17, 9, Mode::Interactive, None).unwrap();
    r.set_scene(&fixture()).unwrap();
    r.render_batch(64, 0.).unwrap();
    assert!(r.sample_count_pipelines.iter().all(Option::is_none));
    assert!(r.adaptive.is_none());
    assert!(r.sample_count_image.is_empty());
    assert_eq!(r.passes[3].output.size, 17 * 9 * 32);
    let history = buffer_snapshot(&r, &r.passes[2].output);
    let samples = r.samples();
    r.set_sample_count_aov(true);
    r.render_batch(0, 0.).unwrap();
    assert_eq!(r.samples(), samples);
    assert_eq!(history, buffer_snapshot(&r, &r.passes[2].output));
    assert!(
        r.adaptive.is_none(),
        "uniform diagnostics must not allocate stats"
    );
    for pixel in r.sample_count_image.as_chunks::<4>().0.iter() {
        assert_eq!(*pixel, heat(samples, 64));
    }
    let image = r.sample_count_image.clone();
    r.render_batch(0, 5.).unwrap();
    assert_eq!(r.sample_count_image, image);
    r.set_sample_count_aov(false);
    r.render_batch(0, 0.).unwrap();
    assert!(r.sample_count_image.is_empty());
    assert_eq!(r.passes[3].output.size, 17 * 9 * 32);
    assert_eq!(history, buffer_snapshot(&r, &r.passes[2].output));
    for mode in [Mode::Interactive, Mode::Final] {
        r.mode = mode;
        r.resize(17, 9).unwrap();
        let mut scene = fixture();
        scene.settings.adaptive_sampling = true;
        scene.settings.sample_count_aov = true;
        r.set_scene(&scene).unwrap();
        while r.samples() < 64 {
            r.render_batch(64 - r.samples(), 0.).unwrap();
        }
        let stats = buffer_snapshot(&r, &r.passes[2].scratch[0]);
        for (pixel, stat) in r
            .sample_count_image
            .as_chunks::<4>()
            .0
            .iter()
            .zip(stats.as_chunks::<4>().0.iter())
        {
            let expected = heat(stat[0].to_bits(), 64);
            for i in 0..4 {
                assert!((pixel[i] - expected[i]).abs() < 1e-6);
            }
        }
        // Synthetic frozen pixels prove the output reads actual per-pixel counts,
        // including zeros and early-stopped pixels, rather than the completion flag.
        let counts = [0u32, 1, 16, 32, 64];
        let stats: Vec<f32> = (0..r.width * r.height)
            .flat_map(|i| {
                [
                    f32::from_bits(counts[i as usize % counts.len()]),
                    0.,
                    0.,
                    f32::from_bits(3),
                ]
            })
            .collect();
        r.passes[2].scratch[0]
            .upload_staged_chunks(bytemuck::cast_slice(&stats), 4096)
            .unwrap();
        r.render_batch(0, 3.).unwrap();
        for (i, pixel) in r.sample_count_image.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(*pixel, heat(counts[i % counts.len()], 64));
        }
        r.resize(7, 5).unwrap();
        while r.samples() < 32 {
            r.render_batch(32 - r.samples(), 0.).unwrap();
        }
        assert_eq!(r.sample_count_image.len(), 7 * 5 * 4);
    }
    r.gpu_output = true;
    r.render_batch(0, 0.).unwrap();
    let gpu = r.gpu_frame.as_ref().unwrap();
    assert!(gpu.sample_count_aov);
    assert_eq!(gpu.buffer.size, 7 * 5 * 48);
    assert_eq!(gpu.read_sample_count().unwrap().len(), 7 * 5 * 4);
    assert_eq!(gpu.read_depth().unwrap().len(), 7 * 5);
    assert_eq!(gpu.read_color().unwrap().len(), 7 * 5 * 4);
    r.set_sample_count_aov(false);
    r.render_batch(0, 0.).unwrap();
    let gpu = r.gpu_frame.as_ref().unwrap();
    assert!(!gpu.sample_count_aov);
    assert_eq!(gpu.buffer.size, 7 * 5 * 32);
    assert!(gpu.read_sample_count().is_err());
}
