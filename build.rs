#[path = "src/material/compiler.rs"]
mod compiler;
use std::{env, path::PathBuf};
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=stdc++"); // ISPC compressor wrapper uses the C++ unwind runtime.
    }
    for path in ["shaders", "third_party/openpbr", "src/material/compiler.rs"] {
        println!("cargo:rerun-if-changed={path}");
    }
    for key in ["LAVA_SHADER_COMPILER", "LAVA_SHADER_CACHE"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    for pass in [
        "visibility",
        "shading",
        "accumulation",
        "tonemap",
        "queue_prepare",
        "queue_reorder",
        "queue_reorder_test",
        "shadow_profile",
        "continuation_profile",
        "visibility_resolve",
        "visibility_raster.vert",
        "visibility_raster.frag",
        "curves.task",
        "curves.mesh",
        "curves.frag",
    ] {
        let source = PathBuf::from(if pass.contains('.') {
            format!("shaders/{pass}")
        } else {
            format!("shaders/{pass}.comp")
        });
        let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join(format!("{pass}.spv"));
        let hit = compiler::compile(
            &source,
            pass,
            &output,
            &[
                "shaders".into(),
                "shaders/materials".into(),
                "third_party/openpbr".into(),
            ],
        )
        .unwrap_or_else(|e| panic!("Shader compilation failed: {e}"));
        println!(
            "cargo:warning=Material/pass product {pass}: {}",
            if hit {
                "cache hit"
            } else {
                "compiled and cached"
            }
        );
    }
}
