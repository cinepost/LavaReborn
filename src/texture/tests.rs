use super::*;
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "lava-texture-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn request(path: PathBuf, mode: TextureMode) -> LoadRequest {
    LoadRequest {
        file: path,
        mode,
        color_space: ColorSpace::Raw,
        compression: Default::default(),
    }
}
fn png(path: &Path, w: u32, h: u32, color: [u8; 4]) {
    image::RgbaImage::from_pixel(w, h, image::Rgba(color))
        .save(path)
        .unwrap();
}
#[test]
fn all_source_formats_keep_hdr_and_complete_mips() {
    let root = Temp::new();
    for ext in ["png", "jpg", "exr", "hdr"] {
        let path = root.path(&format!("source.{ext}"));
        if ext == "png" {
            png(&path, 7, 3, [128, 64, 32, 255]);
        } else if ext == "jpg" {
            image::RgbImage::from_pixel(7, 3, image::Rgb([128u8, 64, 32]))
                .save(&path)
                .unwrap();
        } else {
            image::Rgb32FImage::from_pixel(7, 3, image::Rgb([4f32, 0.25, 2.]))
                .save(&path)
                .unwrap();
        }
        for mode in [TextureMode::Original, TextureMode::Compressed] {
            let asset = prepare(&request(path.clone(), mode), &root.path("cache")).unwrap();
            assert_eq!(
                asset
                    .levels
                    .iter()
                    .map(|l| (l.width, l.height))
                    .collect::<Vec<_>>(),
                [(7, 3), (3, 1), (1, 1)]
            );
            for mip in &asset.levels {
                let expected = if mode == TextureMode::Compressed {
                    mip.width.div_ceil(4) * mip.height.div_ceil(4) * 16
                } else {
                    mip.width * mip.height * 16
                };
                assert_eq!(mip.bytes.len(), expected as usize);
            }
            if ["hdr", "exr"].contains(&ext) {
                if mode == TextureMode::Original {
                    assert!(
                        (bytemuck::cast_slice::<u8, f32>(&asset.levels[0].bytes)[0] - 4.).abs()
                            < 0.01
                    );
                } else {
                    assert!(asset.range_scale[0] >= 4.);
                }
            }
        }
    }
}
#[test]
fn odd_mips_include_every_texel_and_filter_in_linear_space() {
    let root = Temp::new();
    let path = root.path("odd.png");
    let mut pixels = image::RgbaImage::from_pixel(7, 1, image::Rgba([0u8, 0, 0, 255]));
    pixels.put_pixel(6, 0, image::Rgba([255, 0, 0, 255]));
    pixels.save(&path).unwrap();
    let asset = prepare(&request(path, TextureMode::Original), &root.path("cache")).unwrap();
    let mean = bytemuck::cast_slice::<u8, f32>(&asset.levels.last().unwrap().bytes)[0];
    assert!(
        (mean - 1. / 7.).abs() < 1e-6,
        "odd mip lost trailing pixel: {mean}"
    );
    let path = root.path("srgb.png");
    png(&path, 3, 5, [128, 128, 128, 255]);
    let mut request = request(path, TextureMode::Original);
    request.color_space = ColorSpace::Srgb;
    let asset = prepare(&request, &root.path("cache")).unwrap();
    assert!(
        (bytemuck::cast_slice::<u8, f32>(&asset.levels.last().unwrap().bytes)[0] - 0.21586).abs()
            < 1e-4
    );
}
#[test]
fn bc7_cache_tracks_source_content_timestamps_settings_and_corruption() {
    let root = Temp::new();
    let path = root.path("asset.png");
    png(&path, 5, 3, [100, 50, 25, 255]);
    let cache = root.path("cache");
    let mut request = request(path.clone(), TextureMode::Compressed);
    let first = prepare(&request, &cache).unwrap();
    assert!(!first.cache_hit);
    let second = prepare(&request, &cache).unwrap();
    assert!(second.cache_hit);
    assert_eq!(first.fingerprint, second.fingerprint);
    let old_modified = std::fs::metadata(&path).unwrap().modified().unwrap();
    std::fs::File::open(&path)
        .unwrap()
        .set_modified(old_modified + Duration::from_secs(2))
        .unwrap();
    let touched = prepare(&request, &cache).unwrap();
    assert!(!touched.cache_hit);
    assert_ne!(first.fingerprint, touched.fingerprint);
    request.compression.quality = CompressionQuality::High;
    let changed = prepare(&request, &cache).unwrap();
    assert!(!changed.cache_hit);
    assert_ne!(changed.fingerprint, touched.fingerprint);
    assert!(prepare(&request, &cache).unwrap().cache_hit);
    png(&path, 5, 3, [25, 50, 100, 255]);
    std::fs::File::open(&path)
        .unwrap()
        .set_modified(old_modified)
        .unwrap();
    let edited = prepare(&request, &cache).unwrap();
    assert!(!edited.cache_hit);
    assert_ne!(edited.fingerprint, changed.fingerprint);
    std::fs::write(
        cache.join(format!("{}.ltex", edited.fingerprint)),
        b"corrupted blob",
    )
    .unwrap();
    let recovered = prepare(&request, &cache).unwrap();
    assert!(!recovered.cache_hit);
    assert_eq!(edited.fingerprint, recovered.fingerprint);
    assert!(prepare(&request, &cache).unwrap().cache_hit);
}
#[test]
fn background_loader_reloads_files_and_virtual_schedules_nothing() {
    let root = Temp::new();
    let mut manager = TextureManager::new(root.path("cache"));
    manager.request(request(
        root.path("never-created.png"),
        TextureMode::Virtual,
    ));
    assert!(manager.states.is_empty());
    let mut requests = Vec::new();
    for i in 0..8 {
        let path = root.path(&format!("{i}.png"));
        png(&path, 13, 7, [i * 20, 20, 50, 255]);
        let request = request(path, TextureMode::Compressed);
        manager.request(request.clone());
        requests.push(request);
    }
    assert!(manager.pending());
    manager.wait().unwrap();
    for request in &requests {
        assert!(manager.ready(request).is_some());
    }
    let first = manager.ready(&requests[0]).unwrap().fingerprint.clone();
    png(&requests[0].file, 13, 7, [255, 0, 0, 255]);
    manager.last_scan = Instant::now() - Duration::from_secs(1);
    manager.poll();
    assert!(manager.pending());
    manager.wait().unwrap();
    assert_ne!(manager.ready(&requests[0]).unwrap().fingerprint, first);
    manager.request(request(root.path("missing.png"), TextureMode::Original));
    assert!(manager.wait().is_err());
}
#[test]
fn texture_binding_validation_and_virtual_mode_defaults() {
    let binding = TextureBinding {
        file: "test.png".into(),
        ..Default::default()
    };
    binding.validate().unwrap();
    assert_eq!(
        LoadRequest::new(&binding, TextureMode::Original, true).color_space,
        ColorSpace::Srgb
    );
    assert_eq!(
        LoadRequest::new(&binding, TextureMode::Original, false).color_space,
        ColorSpace::Raw
    );
    let mut material = crate::material::Material::default();
    material
        .textures
        .insert("base_color".into(), binding.clone());
    material.validate().unwrap();
    material.textures.insert("unsupported_lobe".into(), binding);
    assert!(material.validate().is_err());
}

#[test]
fn grayscale_formats_bc4_signedness_ranges_mips_and_cache() {
    let root = Temp::new();
    let cache = root.path("cache");
    for name in [
        "gray8.png",
        "gray16.png",
        "gray.jpg",
        "positive.exr",
        "signed.exr",
        "negative.exr",
    ] {
        let file = root.path(name);
        match name {
            "gray8.png" => {
                image::GrayImage::from_fn(7, 3, |x, _| image::Luma([(x * 255 / 6) as u8]))
                    .save(&file)
                    .unwrap()
            }
            "gray16.png" => {
                image::ImageBuffer::from_fn(7, 3, |x, _| image::Luma([(x * 65535 / 6) as u16]))
                    .save(&file)
                    .unwrap()
            }
            "gray.jpg" => image::GrayImage::from_pixel(7, 3, image::Luma([128u8]))
                .save(&file)
                .unwrap(),
            _ => {
                use exr::prelude::*;
                let channels =
                    SpecificChannels::build()
                        .with_channel("Y")
                        .with_pixel_fn(|position| {
                            let x = position.x();
                            (match name {
                                "positive.exr" => x as f32 / 6. * 4.,
                                "signed.exr" => (x as f32 / 6. - 0.5) * 6.,
                                _ => -2.,
                            },)
                        });
                Image::from_channels((7, 3), channels)
                    .write()
                    .to_file(&file)
                    .unwrap();
            }
        }
        let original = prepare(&request(file.clone(), TextureMode::Original), &cache).unwrap();
        assert_eq!(original.format, PixelFormat::R32Float);
        assert_eq!(
            original
                .levels
                .iter()
                .map(|m| (m.width, m.height))
                .collect::<Vec<_>>(),
            [(7, 3), (3, 1), (1, 1)]
        );
        for mip in &original.levels {
            assert_eq!(mip.bytes.len(), (mip.width * mip.height * 4) as usize);
        }
        let req = request(file.clone(), TextureMode::Compressed);
        let compressed = prepare(&req, &cache).unwrap();
        let signed = ["signed.exr", "negative.exr"].contains(&name);
        assert_eq!(
            compressed.format,
            if signed {
                PixelFormat::Bc4Snorm
            } else {
                PixelFormat::Bc4Unorm
            }
        );
        assert_eq!(compressed.range_bias, [0.; 4]);
        for (mip, reference) in compressed.levels.iter().zip(&original.levels) {
            assert_eq!(
                mip.bytes.len(),
                (mip.width.div_ceil(4) * mip.height.div_ceil(4) * 8) as usize
            );
            let values = bytemuck::cast_slice::<u8, f32>(&reference.bytes);
            let bw = mip.width.div_ceil(4);
            for y in 0..mip.height {
                for x in 0..mip.width {
                    let block = ((y / 4 * bw + x / 4) * 8) as usize;
                    let got = bc4::decode(&mip.bytes[block..block + 8], signed)
                        [(y % 4 * 4 + x % 4) as usize]
                        * compressed.range_scale[0];
                    assert!(
                        (got - values[(y * mip.width + x) as usize]).abs()
                            < compressed.range_scale[0] * 0.15,
                        "{name} {x}/{y}: {got}"
                    );
                }
            }
        }
        let hit = prepare(&req, &cache).unwrap();
        assert!(hit.cache_hit);
        assert_eq!(hit.format, compressed.format);
        assert_eq!(hit.levels[0].bytes, compressed.levels[0].bytes);
        std::fs::write(
            cache.join(format!("{}.ltex", hit.fingerprint)),
            b"truncated",
        )
        .unwrap();
        assert!(!prepare(&req, &cache).unwrap().cache_hit);
    }
    let file = root.path("rgb-gray.png");
    png(&file, 3, 5, [100, 100, 100, 255]);
    assert_eq!(
        prepare(&request(file.clone(), TextureMode::Compressed), &cache)
            .unwrap()
            .format,
        PixelFormat::Bc4Unorm
    );
    png(&file, 3, 5, [100, 100, 100, 128]);
    assert_eq!(
        prepare(&request(file, TextureMode::Compressed), &cache)
            .unwrap()
            .format,
        PixelFormat::Bc7
    );
}
