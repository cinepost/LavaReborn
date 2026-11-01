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
        "sample_count",
        "sample_count_adaptive",
        "adaptive_prepare",
        "adaptive_finalize",
        "visibility_adaptive",
        "shading_adaptive",
        "accumulation_adaptive",
        "visibility_resolve_adaptive",
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
        let base = pass.strip_suffix("_adaptive").unwrap_or(pass);
        let base = if base == "sample_count" {
            "tonemap"
        } else {
            base
        };
        let source = PathBuf::from(if base.contains('.') {
            format!("shaders/{base}")
        } else {
            format!("shaders/{base}.comp")
        });
        let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join(format!("{pass}.spv"));
        let hit = compiler::compile_with_defines(
            &source,
            pass,
            &output,
            &[
                "shaders".into(),
                "shaders/materials".into(),
                "third_party/openpbr".into(),
            ],
            if pass == "sample_count_adaptive" {
                &["LAVA_SAMPLE_COUNT=1", "LAVA_ADAPTIVE=1"]
            } else if pass == "sample_count" {
                &["LAVA_SAMPLE_COUNT=1"]
            } else if pass.ends_with("_adaptive") {
                &["LAVA_ADAPTIVE=1"]
            } else {
                &[]
            },
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
