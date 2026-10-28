//! Latest-wins viewport worker with immutable background geometry builds.
use crate::{
    policy::{Mode, Updates},
    renderer::Renderer,
    scene::Scene,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
#[derive(Clone)]
pub struct Request {
    pub scene: Arc<Scene>,
    pub view: Option<Arc<Scene>>,
    pub width: u32,
    pub height: u32,
    pub mode: Mode,
    pub samples: u32,
    pub exposure: f32,
    pub gpu_output: bool,
}
impl Request {
    pub fn parameters(&self) -> &Scene {
        self.view.as_deref().unwrap_or(&self.scene)
    }
}
pub struct Frame {
    pub revision: u64,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub pending_scene: bool,
    pub interactive: bool,
    pub pixels: Vec<f32>,
    pub depth: Vec<f32>,
    pub gpu: Option<Arc<crate::interop::GpuFrame>>,
}
pub struct Worker {
    updates: Arc<Updates<Request>>,
    pub frame: Arc<Mutex<Option<Frame>>>,
    pub error: Arc<Mutex<Option<(u64, String)>>>,
    stop: Arc<AtomicBool>,
    wake: mpsc::SyncSender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Default for Worker {
    fn default() -> Self {
        Self::new()
    }
}
type GeometryBuilder = Arc<
    dyn Fn(
            Arc<crate::vulkan::Device>,
            &Scene,
            Mode,
        ) -> anyhow::Result<crate::renderer::PreparedGeometry>
        + Send
        + Sync,
>;
impl Worker {
    pub fn new() -> Self {
        Self::with_builder(Arc::new(Renderer::prepare_geometry))
    }
    fn with_builder(builder: GeometryBuilder) -> Self {
        let updates = Arc::new(Updates::<Request>::default());
        let frame = Arc::new(Mutex::new(None));
        let error = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let (wake, rx) = mpsc::sync_channel(1);
        let (u, f, e, s) = (updates.clone(), frame.clone(), error.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            let mut renderer: Option<Renderer> = None;
            let mut current: Option<(u64, Arc<Request>)> = None;
            let mut desired: Option<(u64, Arc<Request>)> = None;
            let mut parameters: Option<Arc<Request>> = None;
            let mut building: Option<(
                Arc<Request>,
                std::thread::JoinHandle<anyhow::Result<crate::renderer::PreparedGeometry>>,
            )> = None;
            let mut failed = 0;
            while !s.load(Ordering::Acquire) {
                if let Some(update) = u.take() {
                    desired = Some(update);
                }
                let Some((revision, request)) = desired.clone() else {
                    if rx.recv().is_err() {
                        break;
                    }
                    continue;
                };
                if failed == revision {
                    if rx.recv().is_err() {
                        break;
                    }
                    continue;
                }
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> anyhow::Result<bool> {
                        let r = &request;
                        if renderer.as_ref().map(|v| v.mode != r.mode).unwrap_or(true) {
                            if let Some((_, job)) = building.take() {
                                let _ = job.join();
                            }
                            drop(renderer.take());
                            current = None;
                            parameters = None;
                            renderer = Some(
                                Renderer::new(r.width, r.height, r.mode, None).map_err(|err| {
                                    anyhow::anyhow!(
                                        "creating {}x{} {:?} renderer: {err:#}",
                                        r.width,
                                        r.height,
                                        r.mode
                                    )
                                })?,
                            );
                        }
                        let renderer = renderer.as_mut().unwrap();
                        renderer.resize(r.width, r.height)?;
                        renderer.linear_output = true;
                        // Display-only changes reuse linear history, including a fully converged frame.
                        let display_only = current.as_ref().is_some_and(|(id, previous)| {
                            *id != revision
                                && (Arc::ptr_eq(&previous.scene, &r.scene)
                                    || previous.scene == r.scene)
                                && previous.samples == r.samples
                                && previous.gpu_output == r.gpu_output
                                && parameters.as_ref().is_some_and(|p| {
                                    p.parameters().settings == r.parameters().settings
                                        && p.parameters().camera == r.parameters().camera
                                        && p.parameters().lights == r.parameters().lights
                                })
                        });
                        if display_only {
                            renderer.gpu_output = r.gpu_output;
                            current = Some((revision, request.clone()));
                            parameters = Some(request.clone());
                        } else if current.is_none()
                            || (r.mode == Mode::Final && current.as_ref().unwrap().0 != revision)
                        {
                            renderer.gpu_output = r.gpu_output;
                            renderer.set_scene(&r.scene)?;
                            if r.view.is_some() {
                                renderer.update_scene_parameters(r.parameters())?;
                            }
                            current = Some((revision, request.clone()));
                            parameters = Some(request.clone());
                        } else if current.as_ref().unwrap().0 != revision {
                            if Arc::ptr_eq(&current.as_ref().unwrap().1.scene, &r.scene)
                                || current.as_ref().unwrap().1.scene.same_geometry(&r.scene)
                            {
                                if current.as_ref().unwrap().1.scene.meshes != r.scene.meshes
                                    || current.as_ref().unwrap().1.scene.curves != r.scene.curves
                                {
                                    renderer.update_materials(&r.scene)?;
                                }
                                renderer.gpu_output = r.gpu_output;
                                renderer.update_scene_parameters(r.parameters())?;
                                current = Some((revision, request.clone()));
                                parameters = Some(request.clone());
                            } else {
                                if building.as_ref().is_some_and(|(_, job)| job.is_finished()) {
                                    let (source, job) = building.take().unwrap();
                                    let geometry = job
                                        .join()
                                        .map_err(|_| anyhow::anyhow!("scene builder panicked"))?;
                                    if Arc::ptr_eq(&source.scene, &r.scene)
                                        || source.scene.same_geometry(&r.scene)
                                    {
                                        renderer.install_geometry(geometry?)?;
                                        renderer.update_materials(&r.scene)?;
                                        renderer.gpu_output = r.gpu_output;
                                        renderer.update_scene_parameters(r.parameters())?;
                                        current = Some((revision, request.clone()));
                                        parameters = Some(request.clone());
                                    }
                                    // Obsolete results are dropped without changing the active scene.
                                }
                                if current.as_ref().unwrap().0 != revision && building.is_none() {
                                    let source = request.clone();
                                    let device = renderer.device();
                                    let scene = request.clone();
                                    let build = builder.clone();
                                    building = Some((
                                        source,
                                        std::thread::spawn(move || {
                                            build(device, &scene.scene, Mode::Interactive)
                                        }),
                                    ));
                                }
                            }
                        }
                        // Retire obsolete builds even if a camera/light request restored the active geometry.
                        if current.as_ref().unwrap().0 == revision
                            && building.as_ref().is_some_and(|(_, job)| job.is_finished())
                        {
                            let (_, job) = building.take().unwrap();
                            let _ = job.join();
                        }
                        renderer.refresh_textures()?;
                        let pending_scene =
                            current.as_ref().unwrap().0 != revision || renderer.textures_pending();
                        if pending_scene
                            && parameters.as_ref().is_some_and(|p| {
                                p.parameters().settings != r.parameters().settings
                                    || p.parameters().camera != r.parameters().camera
                                    || p.parameters().lights != r.parameters().lights
                            })
                        {
                            // Camera/light edits remain immediate even while a geometry
                            // replacement is pending; that replacement resets again on commit.
                            renderer.update_scene_parameters(r.parameters())?;
                            parameters = Some(request.clone());
                        }
                        // Exposure is downstream of accumulation; pending AS work must not
                        // turn a display edit into a camera/light reset either.
                        if parameters
                            .as_ref()
                            .is_some_and(|p| p.exposure != r.exposure)
                        {
                            parameters = Some(request.clone());
                        }
                        let active = &current.as_ref().unwrap().1;
                        if renderer.samples() >= active.samples && !pending_scene && !display_only {
                            return Ok(false);
                        }
                        let remaining = if display_only && renderer.samples() >= active.samples {
                            0
                        } else if pending_scene {
                            1
                        } else {
                            active.samples.saturating_sub(renderer.samples())
                        };
                        let pixels = renderer
                            .render_batch(remaining, parameters.as_ref().unwrap().exposure)?;
                        if (r.mode == Mode::Interactive || u.revision() == revision)
                            && (!pixels.is_empty() || renderer.gpu_frame.is_some())
                        {
                            *f.lock().unwrap() = Some(Frame {
                                revision,
                                width: r.width,
                                height: r.height,
                                samples: renderer.samples(),
                                pending_scene,
                                interactive: r.mode == Mode::Interactive,
                                pixels,
                                depth: renderer.depth.clone(),
                                gpu: renderer.gpu_frame.clone(),
                            });
                        }
                        Ok(true)
                    },
                ));
                match result {
                    Ok(Ok(true)) => {}
                    Ok(Ok(false)) => {
                        let _ = rx.recv_timeout(std::time::Duration::from_millis(10));
                    }
                    failure => {
                        let failure = match failure {
                            Ok(Err(err)) => format!("{err:#}"),
                            _ => "render worker panicked".into(),
                        };
                        *e.lock().unwrap() = Some((revision, failure));
                        failed = revision;
                    }
                }
            }
            if let Some((_, job)) = building {
                let _ = job.join();
            }
        });
        Self {
            updates,
            frame,
            error,
            stop,
            wake,
            thread: Some(thread),
        }
    }
    pub fn submit(&self, request: Request) -> anyhow::Result<u64> {
        anyhow::ensure!(request.samples > 0, "samples must be positive");
        anyhow::ensure!(
            request.width > 0
                && request.height > 0
                && request.width <= 16384
                && request.height <= 16384,
            "invalid dimensions"
        );
        anyhow::ensure!(request.exposure.is_finite(), "invalid exposure");
        request.parameters().validate_parameters()?;
        *self.error.lock().unwrap() = None;
        let revision = self.updates.publish(request);
        let _ = self.wake.try_send(());
        Ok(revision)
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = self.wake.try_send(());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn wait(
        worker: &Worker,
        revision: u64,
        pending: bool,
        samples: u32,
    ) -> Arc<crate::interop::GpuFrame> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let error = worker.error.lock().unwrap().clone();
            assert!(error.is_none(), "{error:?}");
            if let Some(frame) = worker.frame.lock().unwrap().as_ref() {
                if frame.revision == revision
                    && frame.pending_scene == pending
                    && frame.samples >= samples
                {
                    return frame.gpu.clone().expect("test requires GPU transport");
                }
            }
            assert!(Instant::now() < deadline, "frame timeout");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn compare_clean(request: &Request, frame: &crate::interop::GpuFrame) {
        let mut clean = Renderer::new(request.width, request.height, request.mode, None).unwrap();
        clean.linear_output = true;
        clean.set_scene(&request.scene).unwrap();
        let mut expected = Vec::new();
        while clean.samples() < request.samples {
            expected = clean
                .render_batch(request.samples - clean.samples(), request.exposure)
                .unwrap();
        }
        let actual = frame.read_color().unwrap();
        assert_eq!(expected.len(), actual.len());
        assert!(
            expected
                .iter()
                .zip(actual)
                .all(|(a, b)| (a - b).abs() < 1e-5),
            "stale scene history survived reset"
        );
    }

    #[test]
    #[ignore = "requires hardware ray queries; asynchronous UDIM tile add/edit/remove in a converged viewport"]
    fn udim_tiles_hot_reload_without_scene_edits_or_image_roundtrip() {
        use crate::texture::{ColorSpace, TextureBinding, TextureMode};
        let root = std::env::temp_dir().join(format!("lava-udim-worker-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let write = |number, color| {
            image::RgbaImage::from_pixel(8, 4, image::Rgba(color))
                .save(root.join(format!("asset.{number}.png")))
                .unwrap();
        };
        write(1001, [255u8, 0, 0, 255]);
        let mut scene: Scene =
            serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
        scene.meshes.truncate(1);
        scene.lights.clear();
        scene.camera.transform =
            glam::Mat4::from_translation(glam::Vec3::new(0., 0., 0.5)).to_cols_array();
        scene.camera.orthographic = true;
        scene.camera.horizontal_aperture = 2.;
        scene.camera.vertical_aperture = 2.;
        scene.settings.textures = TextureMode::Compressed;
        scene.settings.diffuse_depth = 0;
        scene.settings.reflection_depth = 0;
        scene.settings.refraction_depth = 0;
        scene.meshes[0].points = vec![[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.], [-2., 2., 0.]];
        scene.meshes[0].texcoords = vec![[1., 0.], [2., 0.], [2., 1.], [1., 1.]];
        scene.meshes[0].material.openpbr = Some(crate::material::OpenPbrSurface {
            base_weight: 0.,
            specular_weight: 0.,
            emission_luminance: 1.,
            emission_color: [0.2, 0.3, 0.4],
            ..Default::default()
        });
        scene.meshes[0].material.textures.insert(
            "emission_color".into(),
            TextureBinding {
                file: root.join("asset.<UDIM>.png"),
                color_space: ColorSpace::Raw,
                ..Default::default()
            },
        );
        let worker = Worker::new();
        let request = Request {
            scene: Arc::new(scene),
            view: None,
            width: 16,
            height: 16,
            mode: Mode::Interactive,
            samples: 4,
            exposure: 0.,
            gpu_output: true,
        };
        let revision = worker.submit(request).unwrap();
        let mut previous = wait(&worker, revision, false, 4);
        let p = previous.read_color().unwrap();
        assert!(p[..3]
            .iter()
            .zip([0.2, 0.3, 0.4])
            .all(|(a, b)| (*a - b).abs() < 0.01));
        for (step, want) in [(0, [0., 1., 0.]), (1, [0.5; 3]), (2, [0.2, 0.3, 0.4])] {
            let readbacks = crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed);
            match step {
                0 => write(1002, [0, 255, 0, 255]),
                1 => write(1002, [128, 128, 128, 255]),
                _ => std::fs::remove_file(root.join("asset.1002.png")).unwrap(),
            };
            let until = Instant::now() + Duration::from_secs(30);
            let frame = loop {
                assert!(worker.error.lock().unwrap().is_none());
                let f = worker.frame.lock().unwrap();
                if let Some(f) = f.as_ref() {
                    if !f.pending_scene
                        && f.samples >= 4
                        && f.gpu.as_ref().unwrap().serial > previous.serial
                    {
                        break f.gpu.clone().unwrap();
                    }
                }
                drop(f);
                assert!(
                    Instant::now() < until,
                    "UDIM tile change was not discovered"
                );
                std::thread::sleep(Duration::from_millis(5));
            };
            assert_eq!(
                crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed),
                readbacks
            );
            let pixels = frame.read_color().unwrap();
            assert!(
                pixels[..3]
                    .iter()
                    .zip(want)
                    .all(|(a, b)| (*a - b).abs() < 0.015),
                "UDIM update {step} failed: {:?}",
                &pixels[..3]
            );
            previous = frame;
        }
        drop(worker);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    #[ignore = "requires hardware ray queries; resize during blocked scene builds must reuse device and keep presenting"]
    fn resize_keeps_scene_device_and_does_not_join_pending_builds() {
        struct Release(Arc<AtomicBool>);
        impl Drop for Release {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        for visibility in [
            crate::integrator::VisibilityMode::RayQuery,
            crate::integrator::VisibilityMode::Raster,
        ] {
            let gate = Arc::new(AtomicBool::new(false));
            let started = Arc::new(AtomicBool::new(false));
            let (build_gate, build_started) = (gate.clone(), started.clone());
            let worker = Worker::with_builder(Arc::new(move |device, scene, mode| {
                build_started.store(true, Ordering::Release);
                while !build_gate.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Renderer::prepare_geometry(device, scene, mode)
            }));
            // Release before Worker::drop even if an assertion fails.
            let release = Release(gate.clone());
            let mut scene: Scene =
                serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
            scene.settings.visibility = visibility;
            let mut request = Request {
                scene: Arc::new(scene),
                view: None,
                width: 32,
                height: 24,
                mode: Mode::Interactive,
                samples: 4,
                exposure: 0.,
                gpu_output: true,
            };
            let revision = worker.submit(request.clone()).unwrap();
            let first = wait(&worker, revision, false, 4);
            let device = first.buffer.device.clone();
            let mut changed = (*request.scene).clone();
            changed.meshes[0].points[0][0] += 0.1;
            request.scene = Arc::new(changed);
            worker.submit(request.clone()).unwrap();
            let until = Instant::now() + Duration::from_secs(5);
            while !started.load(Ordering::Acquire) {
                assert!(Instant::now() < until);
                std::thread::sleep(Duration::from_millis(1));
            }
            let readbacks = crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed);
            for (width, height) in [(47, 31), (1, 1), (96, 64), (24, 32), (65, 49), (32, 24)] {
                request.width = width;
                request.height = height;
                let revision = worker.submit(request.clone()).unwrap();
                let frame = wait(&worker, revision, true, 1);
                assert_eq!((frame.width, frame.height), (width, height));
                assert!(
                    Arc::ptr_eq(&device, &frame.buffer.device),
                    "resize recreated the Vulkan device"
                );
                assert!(!gate.load(Ordering::Acquire));
            }
            assert_eq!(
                crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed),
                readbacks
            );
            drop(release);
            let revision = worker.submit(request.clone()).unwrap();
            let frame = wait(&worker, revision, false, 4);
            compare_clean(&request, &frame);
            // An immutable pre-resize snapshot remains usable after all target replacements.
            assert_eq!(first.read_color().unwrap().len(), 32 * 24 * 4);
        }
    }
    #[test]
    #[ignore = "requires hardware ray queries; async texture hot reload and per-texture BC7 setting changes"]
    fn textures_reload_without_blocking_or_image_roundtrip() {
        use crate::texture::{ColorSpace, CompressionQuality, TextureBinding, TextureMode};
        let root = std::env::temp_dir().join(format!("lava-worker-texture-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("source.png");
        image::RgbaImage::from_pixel(17, 9, image::Rgba([255u8, 0, 0, 255]))
            .save(&path)
            .unwrap();
        let mut scene: Scene =
            serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
        scene.meshes.truncate(1);
        scene.settings.textures = TextureMode::Compressed;
        scene.camera.transform =
            glam::Mat4::from_translation(glam::Vec3::new(0., 0., 5.)).to_cols_array();
        scene.camera.orthographic = true;
        scene.camera.horizontal_aperture = 2.;
        scene.camera.vertical_aperture = 2.;
        scene.lights.clear();
        scene.meshes[0].points = vec![[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.], [-2., 2., 0.]];
        scene.meshes[0].texcoords = vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
        scene.meshes[0].material.openpbr = Some(crate::material::OpenPbrSurface {
            base_weight: 0.,
            specular_weight: 0.,
            emission_luminance: 1.,
            emission_color: [0.2; 3],
            ..Default::default()
        });
        scene.meshes[0].material.textures.insert(
            "emission_color".into(),
            TextureBinding {
                file: path.clone(),
                color_space: ColorSpace::Raw,
                ..Default::default()
            },
        );
        let worker = Worker::new();
        let mut request = Request {
            scene: Arc::new(scene),
            view: None,
            width: 32,
            height: 24,
            mode: Mode::Interactive,
            samples: 8,
            exposure: 0.,
            gpu_output: true,
        };
        let revision = worker.submit(request.clone()).unwrap();
        let first = wait(&worker, revision, false, 8);
        let pixels = first.read_color().unwrap();
        assert!(pixels[0] > 0.98 && pixels[1] < 0.02);
        let baseline = crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed);
        image::RgbaImage::from_pixel(17, 9, image::Rgba([0u8, 255, 0, 255]))
            .save(&path)
            .unwrap();
        let until = Instant::now() + Duration::from_secs(30);
        let second = loop {
            assert!(worker.error.lock().unwrap().is_none());
            let frame = worker.frame.lock().unwrap();
            if let Some(f) = frame.as_ref() {
                if !f.pending_scene
                    && f.samples >= 8
                    && f.gpu.as_ref().unwrap().serial > first.serial
                {
                    break f.gpu.clone().unwrap();
                }
            }
            drop(frame);
            assert!(
                Instant::now() < until,
                "stationary converged viewport did not notice source edit"
            );
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(
            crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed),
            baseline
        );
        let pixels = second.read_color().unwrap();
        assert!(
            pixels[0] < 0.02 && pixels[1] > 0.98,
            "old texture history survived source reload"
        );
        let mut changed = (*request.scene).clone();
        changed.meshes[0]
            .material
            .textures
            .get_mut("emission_color")
            .unwrap()
            .compression
            .quality = CompressionQuality::High;
        request.scene = Arc::new(changed);
        let revision = worker.submit(request.clone()).unwrap();
        let third = wait(&worker, revision, false, 8);
        assert!(third.serial > second.serial);
        let pixels = third.read_color().unwrap();
        assert!(pixels[0] < 0.02 && pixels[1] > 0.98);
        drop(worker);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    #[ignore = "requires hardware ray queries; exposure re-presents converged GPU frames without restarting samples"]
    fn exposure_only_edits_keep_accumulation_and_gpu_transport() {
        for mode in [Mode::Interactive, Mode::Final] {
            let worker = Worker::new();
            let mut request = Request {
                scene: Arc::new(
                    serde_json::from_str(include_str!("../examples/studio.json")).unwrap(),
                ),
                view: None,
                width: 32,
                height: 24,
                mode,
                samples: 8,
                exposure: 0.,
                gpu_output: true,
            };
            let revision = worker.submit(request.clone()).unwrap();
            let first = wait(&worker, revision, false, 8);
            let original = first.read_color().unwrap();
            let mut previous_serial = first.serial;
            // Native Hydra switches from a full initial scene to a small view delta.
            if mode == Mode::Interactive {
                let mut view = (*request.scene).clone();
                view.meshes.clear();
                request.view = Some(Arc::new(view));
            }
            for exposure in [1., -1., 0.] {
                let baseline_readbacks = crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed);
                request.exposure = exposure;
                let revision = worker.submit(request.clone()).unwrap();
                let frame = wait(&worker, revision, false, 8);
                assert_eq!(worker.frame.lock().unwrap().as_ref().unwrap().samples, 8);
                assert_eq!(
                    frame.serial,
                    previous_serial + 1,
                    "exposure traced new samples instead of re-presenting history"
                );
                previous_serial = frame.serial;
                assert_eq!(
                    crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed),
                    baseline_readbacks
                );
                let pixels = frame.read_color().unwrap();
                for (a, b) in original
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(pixels.as_chunks::<4>().0)
                {
                    for channel in 0..3 {
                        assert!((a[channel] * exposure.exp2() - b[channel]).abs() < 1e-5);
                    }
                }
                // Diagnostic reads above are explicit test requests, not worker presentation.
            }
        }
    }
    #[test]
    #[ignore = "requires hardware ray queries; run with Vulkan validation"]
    fn continuous_camera_requests_present_completed_gpu_frames_while_still_moving() {
        exercise_camera_motion(crate::integrator::VisibilityMode::RayQuery);
    }
    #[test]
    #[ignore = "requires hardware raster and ray queries; raster viewport camera motion without CPU readback"]
    fn raster_camera_requests_present_while_still_moving() {
        exercise_camera_motion(crate::integrator::VisibilityMode::Raster);
    }
    fn exercise_camera_motion(visibility: crate::integrator::VisibilityMode) {
        use crate::ffi::*;
        use std::{
            ffi::CString,
            os::fd::{FromRawFd, OwnedFd},
        };
        unsafe {
            let session = lava_create();
            lava_enable_interop(session, true);
            let mut scene: Scene =
                serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
            scene.settings.visibility = visibility;
            let json = CString::new(serde_json::to_string(&scene).unwrap()).unwrap();
            let initial = lava_submit(session, json.as_ptr(), 1280, 720, false, 2, 0.);
            let deadline = Instant::now() + Duration::from_secs(20);
            while lava_status(session, initial) != 2 {
                assert!(Instant::now() < deadline, "initial frame timeout");
                std::thread::sleep(Duration::from_millis(1));
            }
            let readbacks = lava_image_readback_count();
            let mut view = scene.clone();
            view.meshes.clear();
            let mut serial = 0;
            let mut images_during_motion = 0;
            let mut lagged_images = 0;
            let start = Instant::now();
            let mut requests = 0;
            let mut revision = initial;
            while start.elapsed() < Duration::from_millis(600) {
                view.camera.transform[12] = (requests as f32 * 0.01).sin() * 0.5;
                let json = CString::new(serde_json::to_string(&view).unwrap()).unwrap();
                revision = lava_update_view(session, json.as_ptr(), 1280, 720, 2, 0.);
                assert!(revision > initial, "view submit failed");
                let mut info = std::mem::MaybeUninit::<GpuInfo>::uninit();
                let lease = lava_gpu_frame(session, revision, serial, info.as_mut_ptr());
                if !lease.is_null() {
                    let info = info.assume_init();
                    assert!(info.serial > serial);
                    serial = info.serial;
                    images_during_motion += 1;
                    if lava_status(session, revision) == 0 {
                        lagged_images += 1;
                    }
                    // Export metadata only: no image Map or host readback.
                    drop(OwnedFd::from_raw_fd(info.memory_fd));
                    drop(OwnedFd::from_raw_fd(info.ready_fd));
                    drop(OwnedFd::from_raw_fd(info.returned_fd));
                    lava_release_gpu_frame(lease);
                }
                requests += 1;
                std::thread::yield_now();
            }
            assert!(requests > 50);
            assert!(images_during_motion >= 3, "camera movement starved presentation: {requests} updates, {images_during_motion} images");
            assert!(
                lagged_images > 0,
                "test did not exercise presentation behind the newest camera request"
            );
            let deadline = Instant::now() + Duration::from_secs(20);
            while lava_status(session, revision) != 2 {
                assert!(Instant::now() < deadline, "stopped camera did not converge");
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(lava_image_readback_count(), readbacks);
            println!("{visibility:?}: {requests} continuous camera updates: {images_during_motion} GPU images during motion, {lagged_images} while catching up, zero image readbacks");
            lava_destroy(session);
        }
    }

    #[test]
    #[ignore = "requires hardware ray queries; run with Vulkan validation"]
    fn interactive_build_keeps_sampling_then_resets_and_discards_obsolete_versions() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release = Mutex::new(release_rx);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = calls.clone();
        let worker = Worker::with_builder(Arc::new(move |device, scene, mode| {
            count.fetch_add(1, Ordering::Relaxed);
            started_tx.send(()).unwrap();
            release
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(20))
                .unwrap();
            Renderer::prepare_geometry(device, scene, mode)
        }));
        let scene = serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
        let mut request = Request {
            scene: Arc::new(scene),
            view: None,
            width: 128,
            height: 96,
            mode: Mode::Interactive,
            samples: 128,
            exposure: 0.,
            gpu_output: true,
        };
        let first = worker.submit(request.clone()).unwrap();
        let progressive = wait(&worker, first, false, 1);
        assert!(
            progressive.samples < request.samples,
            "first image waited for convergence"
        );
        let first_frame = wait(&worker, first, false, request.samples);
        request.samples = 2;
        let transfers = crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed);

        Arc::make_mut(&mut request.scene).meshes[0].points[0][0] += 0.2;
        let obsolete = worker.submit(request.clone()).unwrap();
        started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let old = wait(&worker, obsolete, true, 5);
        assert!(old.serial > first_frame.serial);
        // A newer topology version supersedes the held deformation build.
        Arc::make_mut(&mut request.scene).meshes[0]
            .face_counts
            .push(3);
        Arc::make_mut(&mut request.scene).meshes[0]
            .indices
            .extend([0, 1, 2]);
        let newest = worker.submit(request.clone()).unwrap();
        release_tx.send(()).unwrap();
        started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let pending = wait(&worker, newest, true, old.samples + 2);
        assert!(pending.serial > old.serial);
        Arc::make_mut(&mut request.scene).camera.transform[12] += 0.15;
        let newest_camera = worker.submit(request.clone()).unwrap();
        let camera_pending = wait(&worker, newest_camera, true, 1);
        assert!(
            camera_pending.samples < pending.samples,
            "camera edit waited for geometry before resetting"
        );
        release_tx.send(()).unwrap();
        let updated = wait(&worker, newest_camera, false, 2);
        assert_eq!(updated.samples, 2);
        assert!(
            updated.serial > pending.serial,
            "reset sample counter hid the new image"
        );
        assert_eq!(
            crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed),
            transfers,
            "progressive GPU output mapped image data"
        );
        compare_clean(&request, &updated);

        Arc::make_mut(&mut request.scene).camera.transform[12] += 0.4;
        Arc::make_mut(&mut request.scene).lights.clear();
        let camera_light = worker.submit(request.clone()).unwrap();
        let updated = wait(&worker, camera_light, false, 2);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            2,
            "camera/light edit rebuilt geometry"
        );
        compare_clean(&request, &updated);

        // Restore geometry (object position) through the same versioned path.
        Arc::make_mut(&mut request.scene).meshes =
            serde_json::from_str::<Scene>(include_str!("../examples/studio.json"))
                .unwrap()
                .meshes;
        let moved = worker.submit(request.clone()).unwrap();
        started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        wait(&worker, moved, true, 4);
        release_tx.send(()).unwrap();
        compare_clean(&request, &wait(&worker, moved, false, 2));

        let final_worker = Worker::with_builder(Arc::new(|_, _, _| {
            panic!("final mode spawned an interactive builder")
        }));
        request.mode = Mode::Final;
        request.samples = 17;
        let final_revision = final_worker.submit(request.clone()).unwrap();
        let final_frame = wait(&final_worker, final_revision, false, 17);
        assert_eq!(final_frame.samples, 17);
        compare_clean(&request, &final_frame);
    }
    #[test]
    #[ignore = "requires hardware ray queries; OpenPBR table updates without AS rebuild or CPU presentation"]
    fn material_edits_reset_history_without_rebuilding_acceleration() {
        let builds = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = builds.clone();
        let worker = Worker::with_builder(Arc::new(move |device, scene, mode| {
            count.fetch_add(1, Ordering::Relaxed);
            Renderer::prepare_geometry(device, scene, mode)
        }));
        let mut scene: Scene =
            serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
        scene.meshes[0].material.openpbr = Some(crate::material::OpenPbrSurface::default());
        let mut request = Request {
            scene: Arc::new(scene),
            view: None,
            width: 128,
            height: 96,
            mode: Mode::Interactive,
            samples: 4,
            exposure: 0.,
            gpu_output: true,
        };
        let first = worker.submit(request.clone()).unwrap();
        wait(&worker, first, false, 4);
        let readbacks = crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed);
        let material = Arc::make_mut(&mut request.scene).meshes[0]
            .material
            .openpbr
            .as_mut()
            .unwrap();
        material.base_color = [0.1, 0.8, 0.2];
        material.coat_weight = 0.7;
        material.specular_roughness = 0.6;
        // PreviewSurface lowering may also change legacy compatibility fields;
        // these must not trigger AS builds while the native surface table is active.
        Arc::make_mut(&mut request.scene).meshes[0]
            .material
            .metallic = 0.9;
        Arc::make_mut(&mut request.scene).meshes[0].color = [0.3; 3];
        let second = worker.submit(request.clone()).unwrap();
        let frame = wait(&worker, second, false, 4);
        assert_eq!(
            builds.load(Ordering::Relaxed),
            0,
            "material edit rebuilt acceleration"
        );
        assert_eq!(
            crate::interop::IMAGE_READBACKS.load(Ordering::Relaxed),
            readbacks
        );
        compare_clean(&request, &frame);
    }
}
