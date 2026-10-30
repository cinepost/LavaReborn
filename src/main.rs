use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use lava_reborn::{graph::Graph, policy::Mode, renderer::Renderer, scene::Scene, vulkan::Instance};
#[derive(Parser)]
#[command(about = "Lava Reborn: Vulkan hardware ray-query renderer prototype")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Probe,
    Graph,
    Validate {
        scene: std::path::PathBuf,
    },
    Render {
        scene: std::path::PathBuf,
        #[arg(long, default_value = "frame.ppm")]
        output: std::path::PathBuf,
        #[arg(long, default_value_t = 640)]
        width: u32,
        #[arg(long, default_value_t = 360)]
        height: u32,
        #[arg(long, default_value_t = 64)]
        samples: u32,
        #[arg(long,value_enum,default_value_t=Mode::Final)]
        mode: Mode,
        #[arg(long, default_value_t = 0.)]
        exposure: f32,
        #[arg(long)]
        gpu: Option<usize>,
        #[arg(long)]
        diffuse_depth: Option<u32>,
        #[arg(long, value_enum)]
        visibility: Option<lava_reborn::integrator::VisibilityMode>,
        #[arg(long, value_enum)]
        textures: Option<lava_reborn::texture::TextureMode>,
        #[arg(long)]
        reflection_depth: Option<u32>,
        #[arg(long)]
        refraction_depth: Option<u32>,
        #[arg(long)]
        roulette_start: Option<u32>,
    },
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Probe => {
            for (i, a) in Instance::new()?.adapters()?.iter().enumerate() {
                println!(
                    "[{i}] {}: ray_query={}, hardware={} ({})",
                    a.name, a.ray_query, a.hardware, a.reason
                );
            }
        }
        Command::Graph => {
            let g = Graph::prototype();
            let p = g.compile()?;
            for i in p.order {
                println!(
                    "{}: {:?} -> {:?}",
                    g.passes[i].name, g.passes[i].inputs, g.passes[i].outputs
                );
            }
            println!("barriers: {:?}", p.barriers);
        }
        Command::Validate { scene } => {
            let scene: Scene = serde_json::from_slice(&std::fs::read(scene)?)?;
            println!(
                "{} triangles, {} lights",
                scene.validate()?.len(),
                scene.lights.len()
            );
        }
        Command::Render {
            scene,
            output,
            width,
            height,
            samples,
            mode,
            exposure,
            gpu,
            diffuse_depth,
            visibility,
            textures,
            reflection_depth,
            refraction_depth,
            roulette_start,
        } => {
            ensure!(samples > 0, "samples must be positive");
            let scene_path = scene;
            let mut scene: Scene = serde_json::from_slice(&std::fs::read(&scene_path)?)?;
            for material in scene.meshes.iter_mut().map(|m| &mut m.material).chain(
                std::sync::Arc::make_mut(&mut scene.curves)
                    .iter_mut()
                    .map(|c| &mut c.material),
            ) {
                for texture in material.textures.values_mut() {
                    texture.file = lava_reborn::texture::normalize_path(&texture.file);
                    if texture.file.is_relative() {
                        texture.file = scene_path
                            .parent()
                            .unwrap_or_else(|| std::path::Path::new("."))
                            .join(&texture.file);
                    }
                }
            }
            if let Some(v) = textures {
                scene.settings.textures = v;
            }
            if let Some(v) = visibility {
                scene.settings.visibility = v;
            }
            if let Some(v) = diffuse_depth {
                scene.settings.diffuse_depth = v;
            }
            if let Some(v) = reflection_depth {
                scene.settings.reflection_depth = v;
            }
            if let Some(v) = refraction_depth {
                scene.settings.refraction_depth = v;
            }
            if let Some(v) = roulette_start {
                scene.settings.roulette_start = v;
            }
            scene.validate()?;
            let mut r = Renderer::new(width, height, mode, gpu)?;
            r.set_scene(&scene)?;
            let start = std::time::Instant::now();
            let mut image = Vec::new();
            while r.samples() < samples {
                image = r.render_batch(samples - r.samples(), exposure)?;
                eprintln!("{}/{} samples", r.samples(), samples);
            }
            use std::io::Write;
            let mut file = std::io::BufWriter::new(std::fs::File::create(output)?);
            write!(file, "P6\n{width} {height}\n255\n")?;
            for p in image.as_chunks::<4>().0 {
                file.write_all(
                    &p[..3]
                        .iter()
                        .map(|v| (v.clamp(0., 1.) * 255. + 0.5) as u8)
                        .collect::<Vec<_>>(),
                )?;
            }
            eprintln!("rendered in {:.3}s", start.elapsed().as_secs_f64());
        }
    }
    Ok(())
}
