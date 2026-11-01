use lava_reborn::{
    graph::{Format, Graph, Pass, Resource},
    policy::{Mode, Updates},
    scene::Scene,
};
#[test]
fn graph_orders_and_rejects_feedback() {
    let mut g = Graph::prototype();
    assert_eq!(g.compile().unwrap().order, vec![0, 1, 2, 3]);
    g.passes[0].inputs.push(3);
    assert!(g.compile().is_err());
}
#[test]
fn graph_rejects_unowned_and_missing_outputs() {
    let mut g = Graph::prototype();
    g.resources[0].owner = 2;
    assert!(g.compile().is_err());
    let g = Graph {
        passes: vec![Pass {
            name: "read".into(),
            inputs: vec![0],
            outputs: vec![],
        }],
        resources: vec![Resource {
            name: "orphan".into(),
            format: Format::Hdr,
            owner: 0,
            persistent: false,
        }],
    };
    assert!(g.compile().is_err());
}
#[test]
fn polygon_validation() {
    let mut s: Scene = serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
    assert_eq!(s.validate().unwrap().len(), 3);
    s.meshes[0].indices[0] = 100;
    assert!(s.validate().is_err());
}
#[test]
fn latest_update_wins() {
    let updates = Updates::default();
    updates.publish(1);
    let latest = updates.publish(2);
    let (revision, value) = updates.take().unwrap();
    assert_eq!(revision, latest);
    assert_eq!(*value, 2);
    assert!(updates.take().is_none());
    assert!(Mode::Interactive.policy().batch_samples < Mode::Final.policy().batch_samples);
}
#[test]
fn rejects_invalid_lights_and_camera() {
    let mut s: Scene = serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
    s.camera.near = -1.;
    assert!(s.validate().is_err());
    s.camera.near = 0.01;
    s.camera.transform = [0.; 16];
    assert!(s.validate().is_err());
}
#[test]
fn concave_polygon_preserves_area() {
    let mut s: Scene = serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
    s.meshes = vec![lava_reborn::scene::Mesh {
        normals: Vec::new(),
        normal_indices: Vec::new(),
        reverse_orientation: false,
        geometry_light: false,
        light_response: [1.; 2],
        render_light_geometry: false,
        texcoords: Vec::new(),
        texcoord_indices: Vec::new(),
        points: vec![
            [0., 0., 0.],
            [2., 0., 0.],
            [2., 2., 0.],
            [1., 1., 0.],
            [0., 2., 0.],
        ],
        face_counts: vec![5],
        indices: vec![0, 1, 2, 3, 4],
        color: [1.; 3],
        material: Default::default(),
    }];
    let triangles = s.validate().unwrap();
    assert_eq!(triangles.len(), 3);
    let area: f32 = triangles
        .iter()
        .map(|t| {
            let a = glam::Vec3::from_slice(&t.a[..3]);
            let b = glam::Vec3::from_slice(&t.b[..3]);
            let c = glam::Vec3::from_slice(&t.c[..3]);
            (b - a).cross(c - a).length() * 0.5
        })
        .sum();
    assert!((area - 3.).abs() < 1e-6);
}

#[test]
fn degenerate_faces_do_not_abort_valid_geometry() {
    let mut s: Scene = serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
    s.meshes[1].face_counts.push(3);
    s.meshes[1].indices.extend([0, 0, 0]);
    let (triangles, diagnostics) = s.prepare_geometry().unwrap();
    assert_eq!(triangles.len(), 3);
    assert_eq!(diagnostics.skipped_faces, 1);
    s.meshes[1].indices[0] = u32::MAX;
    assert!(
        s.prepare_geometry().is_err(),
        "invalid indices must still fail"
    );
}

#[test]
fn triangulation_is_stable_far_from_origin() {
    let mut s: Scene = serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
    s.meshes = vec![lava_reborn::scene::Mesh {
        normals: Vec::new(),
        normal_indices: Vec::new(),
        reverse_orientation: false,
        geometry_light: false,
        light_response: [1.; 2],
        render_light_geometry: false,
        texcoords: Vec::new(),
        texcoord_indices: Vec::new(),
        points: vec![
            [10000., 10000., 0.],
            [10000.01, 10000., 0.],
            [10000., 10000.01, 0.],
        ],
        face_counts: vec![3],
        indices: vec![0, 1, 2],
        color: [0.5; 3],
        material: Default::default(),
    }];
    let (triangles, diagnostics) = s.prepare_geometry().unwrap();
    assert_eq!(triangles.len(), 1);
    assert_eq!(diagnostics.skipped_faces, 0);
}

#[test]
fn light_bvh_pdfs_normalize_and_cover_all_emitters() {
    use lava_reborn::{light_bvh::LightTree, scene::Light};
    let mut scene: Scene = serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
    scene.lights = (0..37)
        .map(|i| Light::Point {
            diffuse: 1.,
            specular: 1.,
            position: [i as f32 - 18., (i % 3) as f32, 2.],
            color: [1.; 3],
            intensity: (i + 1) as f32,
        })
        .collect();
    scene.lights.push(Light::Directional {
        diffuse: 1.,
        specular: 1.,
        angle: 0.,
        direction: [0., 0., -1.],
        color: [1.; 3],
        intensity: 2.,
    });
    let tree = LightTree::build(&scene.gpu_lights());
    assert_eq!(tree.nodes.len(), 2 * scene.lights.len() - 1);
    for position in [
        glam::Vec3::ZERO,
        glam::Vec3::new(100., 10., -5.),
        glam::Vec3::new(-18., 0., 2.),
    ] {
        let pdfs: Vec<_> = (0..scene.lights.len())
            .map(|i| tree.pmf(i, position))
            .collect();
        assert!(pdfs.iter().all(|p| p.is_finite() && *p > 0.));
        assert!((pdfs.iter().sum::<f32>() - 1.).abs() < 1e-5);
    }
    assert!(LightTree::build(&[]).nodes.is_empty());
}
#[test]
fn integrator_and_material_limits_are_validated() {
    let mut scene: Scene = serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
    scene.settings.refraction_depth = 33;
    assert!(scene.validate().is_err());
    scene.settings = Default::default();
    scene.meshes[0].material.ior = 0.;
    assert!(scene.validate().is_err());
}

#[test]
fn material_ids_uvs_and_surface_only_parameter_changes() {
    use lava_reborn::material::OpenPbrSurface;
    let mut s: lava_reborn::scene::Scene =
        serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
    let m = &mut s.meshes[0];
    m.texcoords = m.points.iter().map(|p| [p[0], p[2]]).collect();
    m.material.openpbr = Some(OpenPbrSurface::default());
    let triangles = s.validate().unwrap();
    assert_eq!(std::mem::size_of::<lava_reborn::scene::Triangle>(), 160);
    assert_eq!(triangles[0].material[3].to_bits(), 0);
    assert!(triangles.iter().any(|t| t.uv01 != [0.; 4]));
    s.meshes[0].texcoord_indices = vec![u32::MAX; s.meshes[0].indices.len()];
    assert!(s.validate().is_err());
}

#[test]
fn light_visibility_defaults_and_gpu_flags() {
    let mut scene: Scene = serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
    let area = serde_json::json!({"type":"area","position":[0,0,1],"u":[1,0,0],"v":[0,-1,0],"color":[1,1,1],"intensity":1});
    scene.lights = vec![serde_json::from_value(area.clone()).unwrap()];
    assert_eq!(scene.gpu_lights()[0].bvh[1], 1); // Single sided; primary hidden.
    let mut visible = area;
    visible["single_sided"] = false.into();
    visible["render_light_geometry"] = true.into();
    scene.lights[0] = serde_json::from_value(visible).unwrap();
    assert_eq!(scene.gpu_lights()[0].bvh[1], 2);
    let ordinary = scene.validate().unwrap();
    assert!(ordinary.iter().all(|t| t.uv2[2] == 0.));
    for mesh in &mut scene.meshes {
        mesh.geometry_light = true;
    }
    assert!(scene.validate().unwrap().iter().all(|t| t.uv2[2] == 1.));
    for mesh in &mut scene.meshes {
        mesh.render_light_geometry = true;
    }
    assert!(scene.validate().unwrap().iter().all(|t| t.uv2[2] == 0.));
}

#[test]
fn authored_normals_preserve_vertex_and_face_corner_associations() {
    let mut scene: Scene = serde_json::from_str(include_str!("../examples/studio.json")).unwrap();
    scene.meshes.truncate(1);
    scene.meshes[0].normals = vec![[0., 0., 2.], [1., 0., 1.], [0., 1., 1.], [-1., 0., 1.]];
    let triangles = scene.validate().unwrap();
    for t in &triangles {
        for (position, normal) in [(t.a, t.normal_a), (t.b, t.normal_b), (t.c, t.normal_c)] {
            let i = scene.meshes[0]
                .points
                .iter()
                .position(|p| p == &position[..3])
                .unwrap();
            let expected = glam::Vec3::from_array(scene.meshes[0].normals[i]).normalize();
            assert!((glam::Vec3::new(normal[0], normal[1], normal[2]) - expected).length() < 1e-6);
        }
    }
    // Two faces share positions but have independent flat normals (hard edge).
    let mesh = &mut scene.meshes[0];
    mesh.face_counts = vec![3, 3];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    mesh.normals = vec![[0., 1., 0.], [0., 0., 1.]];
    mesh.normal_indices = vec![0, 0, 0, 1, 1, 1];
    mesh.reverse_orientation = true;
    let triangles = scene.validate().unwrap();
    assert_eq!(triangles[0].normal_a, [0., 1., 0., -1.]);
    assert_eq!(triangles[1].normal_a, [0., 0., 1., -1.]);
    scene.meshes[0].normal_indices[0] = 2;
    assert!(scene.validate().is_err());
    scene.meshes[0].normal_indices[0] = 0;
    scene.meshes[0].normals[0][0] = f32::NAN;
    assert!(scene.validate().is_err());
    scene.meshes[0].normals = vec![[0.; 3]; 2];
    assert_eq!(scene.validate().unwrap()[0].normal_b[..3], [0.; 3]);
    scene.meshes[0].normals.clear();
    assert!(scene.validate().is_err()); // Indices without a normal table are corrupt.
}
