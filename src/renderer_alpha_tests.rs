use super::path_tests::{buffer_snapshot, fixture};
use super::*;
#[test]
#[ignore = "requires Vulkan; premultiplied cutout alpha, black world and retained explicit lighting"]
fn cutouts_preserve_alpha_without_background_illumination() {
    use crate::{integrator::VisibilityMode, material::OpenPbrSurface};
    for visibility in [VisibilityMode::RayQuery, VisibilityMode::Raster] {
        for adaptive in [false, true] {
            let mut r = Renderer::new(17, 9, Mode::Final, None).unwrap();
            r.linear_output = true;
            let mut scene = fixture();
            scene.settings.visibility = visibility;
            scene.settings.adaptive_sampling = adaptive;
            scene.settings.adaptive_min_samples = 1024;
            scene.settings.diffuse_depth = 2;
            scene.settings.reflection_depth = 2;
            scene.lights.clear();
            // No lights and no emission: secondary misses must not light a surface.
            r.set_scene(&scene).unwrap();
            let render = |r: &mut Renderer, target| {
                let mut pixels = Vec::new();
                while r.samples() < target {
                    pixels = r.render_batch(target - r.samples(), 0.).unwrap();
                }
                pixels
            };
            let image = render(&mut r, 8);
            assert!(image
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [0., 0., 0., 1.]));
            // Cutout samples should average RGB and coverage together, once only.
            scene.meshes[0].material.openpbr = Some(OpenPbrSurface {
                geometry_opacity: 0.5,
                emission_luminance: 2.,
                emission_color: [1., 0.5, 0.25],
                ..Default::default()
            });
            r.set_scene(&scene).unwrap();
            let image = render(&mut r, 256);
            let history = buffer_snapshot(&r, &r.passes[2].output);
            for (pixel, raw) in image
                .as_chunks::<4>()
                .0
                .iter()
                .zip(history.as_chunks::<4>().0)
            {
                assert!((raw[3] - 0.5).abs() < 0.15, "cutout coverage {raw:?}");
                assert_eq!(pixel[3], raw[3], "filter changed coverage");
                for (c, energy) in [2., 1., 0.5].iter().enumerate() {
                    assert!(
                        (pixel[c] - energy * raw[3]).abs() < 1e-5,
                        "RGB not premultiplied: {pixel:?}"
                    );
                }
            }
            if adaptive {
                // RGB alone is perfectly constant here; alpha variance must keep
                // the stochastic cutout active until the budget is exhausted.
                let p = scene.meshes[0].material.openpbr.as_mut().unwrap();
                p.emission_luminance = 0.;
                scene.settings.adaptive_min_samples = 16;
                r.set_scene(&scene).unwrap();
                render(&mut r, 256);
                assert!(
                    buffer_snapshot(&r, &r.passes[2].scratch[0])
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|p| p[0].to_bits() == 256),
                    "dark cutout stopped on RGB-only convergence"
                );
                scene.meshes[0]
                    .material
                    .openpbr
                    .as_mut()
                    .unwrap()
                    .emission_luminance = 2.;
                scene.settings.adaptive_min_samples = 1024;
                r.set_scene(&scene).unwrap();
                render(&mut r, 256);
            }
            // Nonlinear display transforms preserve coverage and zero-alpha holes.
            r.linear_output = false;
            let image = r.render_batch(0, 1.).unwrap();
            assert!(image
                .as_chunks::<4>()
                .0
                .iter()
                .zip(history.as_chunks::<4>().0)
                .all(|(p, h)| p[3] == h[3] && p.iter().all(|v| v.is_finite())));
            scene.meshes[0]
                .material
                .openpbr
                .as_mut()
                .unwrap()
                .geometry_opacity = 0.;
            r.set_scene(&scene).unwrap();
            assert!(render(&mut r, 8)
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [0.; 4]));
            // An explicit emitter behind a cutout still contributes full coverage.
            scene.lights = fixture().lights;
            if let crate::scene::Light::Area { position, u, v, .. } = &mut scene.lights[0] {
                *position = [0., 0., -1.];
                *u = [2., 0., 0.];
                *v = [0., 2., 0.];
            }
            r.linear_output = true;
            r.set_scene(&scene).unwrap();
            for p in render(&mut r, 8).as_chunks::<4>().0 {
                assert!(
                    p[..3].iter().all(|v| (*v - 3.).abs() < 1e-5) && p[3] == 1.,
                    "explicit emitter {visibility:?} adaptive={adaptive}: {p:?}"
                );
            }
        }
    }
}
