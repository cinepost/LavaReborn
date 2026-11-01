use super::path_tests::{fixture, mean};
use super::*;
use crate::scene::Light;
fn response(light: &mut Light, value: f32, spec: bool) {
    let (d, s) = match light {
        Light::Point {
            diffuse, specular, ..
        }
        | Light::Spot {
            diffuse, specular, ..
        }
        | Light::Area {
            diffuse, specular, ..
        }
        | Light::Directional {
            diffuse, specular, ..
        } => (diffuse, specular),
    };
    if spec {
        *s = value
    } else {
        *d = value
    }
}
#[test]
#[ignore = "requires Vulkan; diffuse/specular controls on all analytic lights and both material models"]
fn light_lobe_multipliers_scale_nee_and_emitter_hits() {
    let sources = [
        serde_json::json!({"type":"point","position":[0,0,1],"color":[1,1,1],"intensity":3}),
        serde_json::json!({"type":"spot","position":[0,0,1],"direction":[0,0,-1],"cone_degrees":30,"softness":0.2,"color":[1,1,1],"intensity":3}),
        serde_json::json!({"type":"area","position":[0,0,1],"u":[2,0,0],"v":[0,-2,0],"color":[1,1,1],"intensity":3}),
        serde_json::json!({"type":"directional","direction":[0,0,-1],"angle":20,"color":[1,1,1],"intensity":3}),
    ];
    let mut r = Renderer::new(8, 8, Mode::Final, None).unwrap();
    r.linear_output = true;
    for source in sources {
        for (pbr, spec) in [(false, false), (true, false), (true, true)] {
            let mut scene = fixture();
            scene.settings.diffuse_depth = if spec { 0 } else { 1 };
            scene.settings.reflection_depth = if spec { 1 } else { 0 };
            if pbr {
                scene.meshes[0].material.openpbr = Some(crate::material::OpenPbrSurface {
                    base_weight: if spec { 0. } else { 1. },
                    base_color: [0.8; 3],
                    specular_weight: if spec { 1. } else { 0. },
                    specular_roughness: 0.25,
                    ..Default::default()
                });
            }
            scene.lights = vec![serde_json::from_value(source.clone()).unwrap()];
            let mut values = Vec::new();
            for multiplier in [0., 1., 2.] {
                response(&mut scene.lights[0], multiplier, spec);
                values.push(mean(&mut r, &scene, 256));
            }
            response(&mut scene.lights[0], 1., spec);
            response(&mut scene.lights[0], 0., !spec);
            let other_zero = mean(&mut r, &scene, 256);
            response(&mut scene.lights[0], 2., !spec);
            let other_two = mean(&mut r, &scene, 256);
            assert!(
                (other_zero - other_two).abs() < 0.001,
                "light lobe controls are coupled: {source} {other_zero} {other_two}"
            );
            let contribution = values[1] - values[0];
            assert!(
                contribution > 0.001,
                "no response: {source}, pbr={pbr} spec={spec} {values:?}"
            );
            assert!(
                ((values[2] - values[0]) - 2. * contribution).abs() < 0.002 * contribution.max(1.),
                "nonlinear light response: {source} {values:?}"
            );
        }
    }
    // Perfect legacy mirror: emitter hits, rather than the NEE estimator.
    let mut scene = fixture();
    scene.meshes[0].material.metallic = 1.;
    scene.settings.reflection_depth = 1;
    for source in [
        serde_json::json!({"type":"directional","angle":20,"direction":[0,0,-1],"color":[1,1,1],"intensity":3}),
        serde_json::json!({"type":"area","position":[0,0,1],"u":[2,0,0],"v":[0,-2,0],"color":[1,1,1],"intensity":3}),
    ] {
        scene.lights = vec![serde_json::from_value(source).unwrap()];
        response(&mut scene.lights[0], 0., true);
        let dark = mean(&mut r, &scene, 4);
        response(&mut scene.lights[0], 1., true);
        let one = mean(&mut r, &scene, 4);
        response(&mut scene.lights[0], 2., true);
        let two = mean(&mut r, &scene, 4);
        assert!(one > dark + 1.);
        assert!((two - dark - 2. * (one - dark)).abs() < 0.001);
    }
}
#[test]
#[ignore = "requires Vulkan; finite distant source MIS energy and soft shadow penumbra"]
fn distant_angle_preserves_energy_and_softens_shadows() {
    let mut r = Renderer::new(4, 4, Mode::Final, None).unwrap();
    r.linear_output = true;
    let mut scene = fixture();
    for angle in [0., 1e-18, 0.53, 30., 120., 270.] {
        scene.lights=vec![serde_json::from_value(serde_json::json!({"type":"directional","angle":angle,"direction":[0,0,-1],"color":[1,1,1],"intensity":3})).unwrap()];
        let energy = mean(&mut r, &scene, 1024);
        let expected = 0.8 * 3. / std::f32::consts::PI;
        assert!(
            (energy - expected).abs() < 0.025,
            "angle={angle}: {energy} expected={expected}"
        );
    }
    let mut occluder = scene.meshes[0].clone();
    occluder.points = vec![[-1., -1., 1.], [0.5, -1., 1.], [0.5, 1., 1.], [-1., 1., 1.]];
    occluder.geometry_light = true;
    occluder.render_light_geometry = false;
    scene.meshes.push(occluder);
    scene.camera.transform[14] = 0.5;
    let mut values = Vec::new();
    for (x, angle) in [(0.48, 0.), (0.48, 20.), (0., 20.), (1., 20.)] {
        scene.camera.transform[12] = x;
        if let Light::Directional { angle: a, .. } = &mut scene.lights[0] {
            *a = angle;
        }
        values.push(mean(&mut r, &scene, 2048));
    }
    assert!(
        values[1] > values[0] + 0.05
            && values[1] > values[2] + 0.05
            && values[1] < values[3] - 0.05,
        "no penumbra: {values:?}"
    );
}
#[test]
#[ignore = "requires Vulkan; mesh light emission scales by incoming lobe and preserves camera emission"]
fn mesh_light_response_scales_bounce_emission() {
    let mut r = Renderer::new(4, 4, Mode::Final, None).unwrap();
    r.linear_output = true;
    let mut scene = fixture();
    scene.lights.clear();
    scene.meshes[0].material.metallic = 1.;
    scene.settings.reflection_depth = 1;
    let mut emitter = scene.meshes[0].clone();
    for p in &mut emitter.points {
        p[2] = 1.;
    }
    emitter.geometry_light = true;
    emitter.render_light_geometry = true;
    emitter.material.openpbr = Some(crate::material::OpenPbrSurface {
        base_weight: 0.,
        specular_weight: 0.,
        emission_luminance: 3.,
        emission_color: [1.; 3],
        geometry_thin_walled: true,
        ..Default::default()
    });
    scene.meshes.push(emitter);
    for gain in [0., 1., 2.] {
        scene.meshes[1].light_response = [0., gain];
        let reflected = mean(&mut r, &scene, 4);
        assert!(
            (reflected - 2.4 * gain).abs() < 0.001,
            "mesh specular response {gain}: {reflected}"
        );
    }
    scene.meshes[0].material.metallic = 0.;
    let mut diffuse_values = Vec::new();
    for gain in [0., 1., 2.] {
        scene.meshes[1].light_response = [gain, 0.];
        diffuse_values.push(mean(&mut r, &scene, 256));
    }
    assert!(diffuse_values[1] > diffuse_values[0] + 1.);
    assert!(
        (diffuse_values[2] - diffuse_values[0] - 2. * (diffuse_values[1] - diffuse_values[0]))
            .abs()
            < 0.002
    );
    scene.camera.transform[14] = 1.5;
    scene.meshes[1].light_response = [0., 0.];
    assert!(
        (mean(&mut r, &scene, 4) - 3.).abs() < 0.001,
        "camera emission changed with lobe multipliers"
    );
}
