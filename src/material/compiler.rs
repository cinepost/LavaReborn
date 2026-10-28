//! Content-addressed SPIR-V product cache. Shared by build-time and future graph
//! compilation: no dependency on the renderer, Vulkan device or material values.
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const ABI: &str = "lava-material-products-v1-vulkan1.2";
fn digest(bytes: &[u8]) -> io::Result<String> {
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child.stdin.take().unwrap().write_all(bytes)?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(io::Error::other("sha256sum failed"));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap()
        .into())
}
fn compiler(name: &str) -> Command {
    let mut c = Command::new(name);
    // Houdini/Lava bundle an unrelated libslang.so. Scope this override to slangc.
    if name.ends_with("slangc") {
        c.env("LD_LIBRARY_PATH", "/usr/lib");
    }
    c
}
fn valid(path: &Path) -> bool {
    fs::read(path).is_ok_and(|b| b.len() >= 20 && b.len() % 4 == 0 && b[..4] == [3, 2, 35, 7])
        && Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(path)
            .output()
            .is_ok_and(|o| o.status.success())
}
pub fn compile(
    source: &Path,
    product: &str,
    output: &Path,
    includes: &[PathBuf],
) -> io::Result<bool> {
    let backend =
        std::env::var("LAVA_SHADER_COMPILER").unwrap_or_else(|_| "glslangValidator".into());
    if backend != "glslangValidator" && backend != "slangc" {
        return Err(io::Error::other(
            "LAVA_SHADER_COMPILER must be glslangValidator or slangc",
        ));
    }
    // glslang preprocessing records every transitive include, including vendored LUTs.
    let mut pre = Command::new("glslangValidator");
    pre.arg("-E");
    for include in includes {
        pre.arg(format!("-I{}", include.display()));
    }
    let pre = pre.arg(source).output()?;
    if !pre.status.success() {
        return Err(io::Error::other(
            String::from_utf8_lossy(&pre.stderr).into_owned(),
        ));
    }
    let version = compiler(&backend)
        .arg(if backend == "slangc" {
            "-version"
        } else {
            "--version"
        })
        .output()?;
    if !version.status.success() {
        return Err(io::Error::other("shader compiler version query failed"));
    }
    let mut key = Vec::from(ABI.as_bytes());
    key.extend_from_slice(product.as_bytes());
    key.extend_from_slice(backend.as_bytes());
    key.extend_from_slice(&version.stdout);
    key.extend_from_slice(&version.stderr);
    key.extend_from_slice(&pre.stdout);
    let root = std::env::var_os("LAVA_SHADER_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/material-cache"));
    fs::create_dir_all(&root)?;
    let hash = digest(&key)?;
    let cached = root.join(format!("{product}-{hash}.spv"));
    if valid(&cached) {
        fs::copy(&cached, output)?;
        return Ok(true);
    }
    let temporary = root.join(format!(
        "{product}-{hash}-{}-{}.tmp.spv",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let mut command = compiler(&backend);
    if backend == "slangc" {
        command.args([
            "-lang",
            "glsl",
            "-target",
            "spirv",
            "-profile",
            "glsl_460",
            "-entry",
            "main",
            "-stage",
            match source.extension().and_then(|s| s.to_str()) {
                Some("vert") => "vertex",
                Some("frag") => "fragment",
                Some("mesh") => "mesh",
                Some("task") => "amplification",
                _ => "compute",
            },
        ]);
    } else {
        command.args(["-V", "--target-env", "vulkan1.2"]);
    }
    for include in includes {
        command.arg(format!("-I{}", include.display()));
    }
    let result = command.arg(source).arg("-o").arg(&temporary).output()?;
    if !result.status.success() {
        let _ = fs::remove_file(&temporary);
        return Err(io::Error::other(format!(
            "{product}: {}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    if !valid(&temporary) {
        let _ = fs::remove_file(&temporary);
        return Err(io::Error::other(format!("invalid SPIR-V: {product}")));
    }
    fs::rename(&temporary, &cached)?;
    fs::copy(cached, output)?;
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_tracks_transitive_inputs_and_recovers_from_corruption() {
        let id = format!(
            "cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(&id);
        fs::create_dir_all(&root).unwrap();
        let source = root.join("probe.comp");
        let include = root.join("value.glsl");
        let output = root.join("probe.spv");
        fs::write(&source,"#version 460\n#extension GL_GOOGLE_include_directive : require\n#include \"value.glsl\"\nlayout(local_size_x=1) in;\nlayout(std430,binding=0) buffer Data {uint value;};\nvoid main(){value=VALUE;}\n").unwrap();
        fs::write(&include, "#define VALUE 1u\n").unwrap();
        assert!(!compile(&source, &id, &output, std::slice::from_ref(&root)).unwrap());
        assert!(compile(&source, &id, &output, std::slice::from_ref(&root)).unwrap());
        fs::write(&include, "#define VALUE 2u\n").unwrap();
        assert!(!compile(&source, &id, &output, std::slice::from_ref(&root)).unwrap());
        let cache = std::env::var_os("LAVA_SHADER_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| "target/material-cache".into());
        for entry in fs::read_dir(&cache).unwrap() {
            let path = entry.unwrap().path();
            if path.file_name().unwrap().to_string_lossy().starts_with(&id) {
                fs::write(path, b"truncated blob").unwrap();
            }
        }
        assert!(!compile(&source, &id, &output, std::slice::from_ref(&root)).unwrap());
        assert!(valid(&output));
        for entry in fs::read_dir(cache).unwrap() {
            let path = entry.unwrap().path();
            if path.file_name().unwrap().to_string_lossy().starts_with(&id) {
                fs::remove_file(path).unwrap();
            }
        }
        fs::remove_dir_all(root).unwrap();
    }
}
