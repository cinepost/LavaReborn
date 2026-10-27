//! Parallel image preparation and versioned texture assets, independent of Vulkan/material passes.
mod bc4;
pub mod gpu;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{mpsc, Arc, OnceLock},
    time::{Duration, Instant, UNIX_EPOCH},
};

pub const MAX_TEXTURES: usize = 128;
pub const MATERIAL_FIELDS: usize = 40; // 38 material fields plus normal and bump texture slots.
const CACHE_VERSION: u32 = 2;
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum TextureMode {
    #[default]
    Original,
    Compressed,
    Virtual,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorSpace {
    #[default]
    Auto,
    Srgb,
    Raw,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionQuality {
    Fast,
    #[default]
    Balanced,
    High,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CompressionSettings {
    pub quality: CompressionQuality,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Wrap {
    #[default]
    Repeat,
    Clamp,
    Mirror,
    Black,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextureBinding {
    pub file: PathBuf,
    pub color_space: ColorSpace,
    pub compression: CompressionSettings,
    /// rgb, r, g, b or a; scalar inputs default to r.
    pub channel: String,
    pub multiply: bool,
    pub strength: f32,
    pub signed: bool,
    pub scale: [f32; 4],
    pub bias: [f32; 4],
    pub uv_scale: [f32; 2],
    pub uv_offset: [f32; 2],
    pub wrap_s: Wrap,
    pub wrap_t: Wrap,
}
impl Default for TextureBinding {
    fn default() -> Self {
        Self {
            file: PathBuf::new(),
            color_space: ColorSpace::Auto,
            compression: Default::default(),
            channel: "rgb".into(),
            multiply: false,
            strength: 1.,
            signed: false,
            scale: [1.; 4],
            bias: [0.; 4],
            uv_scale: [1.; 2],
            uv_offset: [0.; 2],
            wrap_s: Wrap::Repeat,
            wrap_t: Wrap::Repeat,
        }
    }
}
impl TextureBinding {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.file.as_os_str().is_empty(), "texture file is empty");
        ensure!(
            ["rgb", "r", "g", "b", "a", "luminance"].contains(&self.channel.as_str()),
            "unsupported texture channel {}",
            self.channel
        );
        ensure!(
            self.scale
                .iter()
                .chain(&self.bias)
                .chain(&self.uv_scale)
                .chain(&self.uv_offset)
                .all(|x| x.is_finite()),
            "non-finite texture transform"
        );
        ensure!(self.strength.is_finite(), "non-finite texture strength");
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LoadRequest {
    pub file: PathBuf,
    pub mode: TextureMode,
    pub color_space: ColorSpace,
    pub compression: CompressionSettings,
}
impl LoadRequest {
    pub fn new(binding: &TextureBinding, mode: TextureMode, color: bool) -> Self {
        let file = std::fs::canonicalize(&binding.file).unwrap_or_else(|_| binding.file.clone());
        let ext = file
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let color_space = if binding.color_space == ColorSpace::Auto {
            if color && ["png", "jpg", "jpeg"].contains(&ext.as_str()) {
                ColorSpace::Srgb
            } else {
                ColorSpace::Raw
            }
        } else {
            binding.color_space
        };
        Self {
            file,
            mode,
            color_space,
            compression: binding.compression.clone(),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceStamp {
    pub size: u64,
    pub modified_ns: u64,
    pub changed_ns: u64,
}
impl SourceStamp {
    pub fn read(path: &Path) -> Result<Self> {
        let meta =
            std::fs::metadata(path).with_context(|| format!("texture {}", path.display()))?;
        let modified_ns = meta
            .modified()?
            .duration_since(UNIX_EPOCH)?
            .as_nanos()
            .try_into()?;
        #[cfg(unix)]
        let changed_ns = {
            use std::os::unix::fs::MetadataExt;
            (meta.ctime().max(0) as u64)
                .saturating_mul(1_000_000_000)
                .saturating_add(meta.ctime_nsec().max(0) as u64)
        };
        #[cfg(not(unix))]
        let changed_ns = modified_ns;
        Ok(Self {
            size: meta.len(),
            modified_ns,
            changed_ns,
        })
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelFormat {
    Rgba32Float,
    R32Float,
    Bc7,
    Bc4Unorm,
    Bc4Snorm,
}
#[derive(Debug, Clone)]
pub struct Mip {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}
#[derive(Debug, Clone)]
pub struct PreparedTexture {
    pub format: PixelFormat,
    pub levels: Vec<Mip>,
    pub range_scale: [f32; 4],
    pub range_bias: [f32; 4],
    pub fingerprint: String,
    pub cache_hit: bool,
}
#[derive(Serialize, Deserialize)]
struct Header {
    version: u32,
    format: PixelFormat,
    request: LoadRequest,
    stamp: SourceStamp,
    source_hash: String,
    range_scale: [f32; 4],
    range_bias: [f32; 4],
    levels: Vec<(u32, u32, usize)>,
    payload_hash: String,
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn cache_root() -> PathBuf {
    std::env::var_os("LAVA_TEXTURE_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    std::env::var_os("HOME")
                        .map(PathBuf::from)
                        .unwrap_or_else(std::env::temp_dir)
                        .join(".cache")
                })
                .join("lava-reborn/textures")
        })
}
fn srgb(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}
/// Area-weighted linear downsampling includes odd trailing rows/columns.
fn downsample(src: &[[f32; 4]], w: u32, h: u32) -> (Vec<[f32; 4]>, u32, u32) {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut dst = vec![[0.; 4]; (nw * nh) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let x0 = x as f64 * w as f64 / nw as f64;
            let x1 = (x + 1) as f64 * w as f64 / nw as f64;
            let y0 = y as f64 * h as f64 / nh as f64;
            let y1 = (y + 1) as f64 * h as f64 / nh as f64;
            let mut sum = [0f64; 4];
            let mut total = 0.;
            for sy in y0.floor() as u32..(y1.ceil() as u32).min(h) {
                for sx in x0.floor() as u32..(x1.ceil() as u32).min(w) {
                    let weight = (x1.min((sx + 1) as f64) - x0.max(sx as f64))
                        * (y1.min((sy + 1) as f64) - y0.max(sy as f64));
                    for c in 0..4 {
                        sum[c] += f64::from(src[(sy * w + sx) as usize][c]) * weight;
                    }
                    total += weight;
                }
            }
            dst[(y * nw + x) as usize] = sum.map(|v| (v / total) as f32);
        }
    }
    (dst, nw, nh)
}
fn encode(
    pixels: &[[f32; 4]],
    w: u32,
    h: u32,
    settings: &CompressionSettings,
    scale: [f32; 4],
    bias: [f32; 4],
) -> Vec<u8> {
    let pw = w.div_ceil(4) * 4;
    let ph = h.div_ceil(4) * 4;
    let mut padded = vec![0u8; (pw * ph * 4) as usize];
    for y in 0..ph {
        for x in 0..pw {
            for c in 0..4 {
                let value =
                    (pixels[(y.min(h - 1) * w + x.min(w - 1)) as usize][c] - bias[c]) / scale[c];
                padded[((y * pw + x) * 4) as usize + c] =
                    (value.clamp(0., 1.) * 255.).round() as u8;
            }
        }
    }
    let options = match settings.quality {
        CompressionQuality::Fast => intel_tex_2::bc7::alpha_fast_settings(),
        CompressionQuality::Balanced => intel_tex_2::bc7::alpha_basic_settings(),
        CompressionQuality::High => intel_tex_2::bc7::alpha_slow_settings(),
    };
    // Independent 4-row block bands share the same bounded Rayon pool. This
    // accelerates a single large texture as well as batches of different files.
    use rayon::prelude::*;
    let stride = (pw * 4) as usize;
    padded
        .par_chunks(stride * 64)
        .map(|band| {
            intel_tex_2::bc7::compress_blocks(
                &options,
                &intel_tex_2::RgbaSurface {
                    data: band,
                    width: pw,
                    height: (band.len() / stride) as u32,
                    stride: pw * 4,
                },
            )
        })
        .collect::<Vec<_>>()
        .concat()
}
fn read_cache(
    path: &Path,
    request: &LoadRequest,
    stamp: SourceStamp,
    hash: &str,
    key: &str,
) -> Result<PreparedTexture> {
    let data = std::fs::read(path)?;
    ensure!(
        data.len() > 12 && &data[..8] == b"LAVATEX\0",
        "bad texture cache magic"
    );
    let n = u32::from_le_bytes(data[8..12].try_into()?) as usize;
    ensure!(
        n <= 1024 * 1024 && n <= data.len() - 12,
        "invalid texture cache header"
    );
    let header: Header = serde_json::from_slice(&data[12..12 + n])?;
    ensure!(
        header.version == CACHE_VERSION
            && &header.request == request
            && header.stamp == stamp
            && header.source_hash == hash,
        "stale texture cache"
    );
    ensure!(
        header.range_scale.iter().all(|x| x.is_finite() && *x > 0.)
            && header.range_bias.iter().all(|x| x.is_finite()),
        "invalid cached range"
    );
    ensure!(
        digest(&data[12 + n..]) == header.payload_hash,
        "corrupt texture cache payload"
    );
    let block_bytes = match header.format {
        PixelFormat::Bc7 => 16,
        PixelFormat::Bc4Unorm | PixelFormat::Bc4Snorm => 8,
        _ => bail!("uncompressed format in compressed texture cache"),
    };
    let mut offset = 12 + n;
    let mut levels = Vec::new();
    let mut expected = None;
    for (w, h, len) in header.levels {
        ensure!(
            w > 0 && h > 0 && w <= 65536 && h <= 65536,
            "invalid cached dimensions"
        );
        if let Some(d) = expected {
            ensure!(d == (w, h), "incomplete cached mip chain");
        }
        let bytes = (w as u64).div_ceil(4) * (h as u64).div_ceil(4) * block_bytes;
        ensure!(
            bytes == len as u64 && len <= data.len() - offset,
            "bad cached mip length"
        );
        levels.push(Mip {
            width: w,
            height: h,
            bytes: data[offset..offset + len].to_vec(),
        });
        offset += len;
        expected = Some(((w / 2).max(1), (h / 2).max(1)));
    }
    ensure!(
        offset == data.len() && levels.last().is_some_and(|l| l.width == 1 && l.height == 1),
        "incomplete cached mip pyramid"
    );
    Ok(PreparedTexture {
        format: header.format,
        levels,
        range_scale: header.range_scale,
        range_bias: header.range_bias,
        fingerprint: key.into(),
        cache_hit: true,
    })
}
fn write_cache(path: &Path, header: &Header, levels: &[Mip]) -> Result<()> {
    let parent = path.parent().context("cache path has no parent")?;
    std::fs::create_dir_all(parent)?;
    let json = serde_json::to_vec(header)?;
    let mut data = b"LAVATEX\0".to_vec();
    data.extend_from_slice(&(json.len() as u32).to_le_bytes());
    data.extend(json);
    for level in levels {
        data.extend_from_slice(&level.bytes);
    }
    static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = path.with_extension(format!(
        "{}-{}.tmp",
        std::process::id(),
        SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let result = (|| {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        file.write_all(&data)?;
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(tmp);
    }
    Ok(result?)
}
/// Source bytes are hashed even on disk-cache hits. mtime/ctime/size and encoder
/// settings are also part of the key; writes are atomic and checked for races.
pub fn prepare(request: &LoadRequest, root: &Path) -> Result<PreparedTexture> {
    ensure!(
        request.mode != TextureMode::Virtual,
        "virtual texture preparation is inactive"
    );
    let stamp = SourceStamp::read(&request.file)?;
    let bytes = std::fs::read(&request.file)?;
    let source_hash = digest(&bytes);
    ensure!(
        SourceStamp::read(&request.file)? == stamp,
        "texture changed while reading"
    );
    let key = digest(&serde_json::to_vec(&(
        CACHE_VERSION,
        request,
        stamp,
        &source_hash,
    ))?);
    let path = root.join(format!("{key}.ltex"));
    if request.mode == TextureMode::Compressed {
        if let Ok(cached) = read_cache(&path, request, stamp, &source_hash, &key) {
            return Ok(cached);
        }
    }
    let reader = image::ImageReader::new(std::io::Cursor::new(&bytes)).with_guessed_format()?;
    let format = reader.format().context("unknown image format")?;
    ensure!(
        [
            image::ImageFormat::Png,
            image::ImageFormat::Jpeg,
            image::ImageFormat::OpenExr,
            image::ImageFormat::Hdr
        ]
        .contains(&format),
        "only PNG, JPEG, EXR and HDR textures are supported"
    );
    let (mut pixels, mut w, mut h, single) = match reader.decode() {
        Ok(image) => {
            let single = image.color().channel_count() == 1;
            let image = image.to_rgba32f();
            let (w, h) = image.dimensions();
            (image.into_raw().as_chunks::<4>().0.to_vec(), w, h, single)
        }
        Err(error) if format == image::ImageFormat::OpenExr => {
            // image's EXR frontend requires RGB; scalar EXR channels need the
            // native dynamic-channel reader. Do not silently discard channels.
            use exr::prelude::*;
            let image = read()
                .no_deep_data()
                .largest_resolution_level()
                .all_channels()
                .first_valid_layer()
                .all_attributes()
                .from_buffered(std::io::Cursor::new(&bytes))
                .with_context(|| format!("decoding {}: {error}", request.file.display()))?;
            let layer = image.layer_data;
            ensure!(
                layer.channel_data.list.len() == 1,
                "EXR has unsupported non-RGB channels"
            );
            let display = image.attributes.display_window;
            let (w, h) = (display.size.width() as u32, display.size.height() as u32);
            let mut pixels = vec![[0., 0., 0., 1.]; (w * h) as usize];
            let offset = layer.attributes.layer_position - display.position;
            for (i, v) in layer.channel_data.list[0]
                .sample_data
                .values_as_f32()
                .enumerate()
            {
                let x = (i % layer.size.width()) as i32 + offset.x();
                let y = (i / layer.size.width()) as i32 + offset.y();
                if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
                    pixels[(y as u32 * w + x as u32) as usize] = [v, v, v, 1.];
                }
            }
            (pixels, w, h, true)
        }
        Err(error) => {
            return Err(error).with_context(|| format!("decoding {}", request.file.display()))
        }
    };
    ensure!(w > 0 && h > 0, "empty texture");
    ensure!(
        pixels.iter().flatten().all(|x| x.is_finite()),
        "texture has non-finite pixels"
    );
    // Also collapse losslessly gray RGB/RGBA assets; preserve nonopaque alpha.
    let single = single
        || pixels
            .iter()
            .all(|p| p[0] == p[1] && p[1] == p[2] && p[3] == 1.);
    if request.color_space == ColorSpace::Srgb {
        for p in &mut pixels {
            for c in &mut p[..3] {
                *c = srgb(*c);
            }
        }
    }
    let mut bias = [0f32; 4];
    let mut upper = [1f32; 4];
    if request.mode == TextureMode::Compressed {
        for p in &pixels {
            for c in 0..4 {
                bias[c] = bias[c].min(p[c]);
                upper[c] = upper[c].max(p[c]);
            }
        }
    }
    let format = match (request.mode, single, bias[0] < 0.) {
        (TextureMode::Compressed, true, true) => PixelFormat::Bc4Snorm,
        (TextureMode::Compressed, true, false) => PixelFormat::Bc4Unorm,
        (TextureMode::Compressed, false, _) => PixelFormat::Bc7,
        (_, true, _) => PixelFormat::R32Float,
        _ => PixelFormat::Rgba32Float,
    };
    let mut scale = std::array::from_fn(|c| upper[c] - bias[c]);
    if single {
        // Symmetric signed scaling keeps zero exact and avoids clipping HDR.
        let range = if format == PixelFormat::Bc4Snorm {
            upper[0].max(-bias[0])
        } else {
            upper[0]
        };
        scale = [range, range, range, 1.];
        bias = [0.; 4];
    }
    ensure!(
        scale.iter().all(|x| x.is_finite() && *x > 0.),
        "texture range cannot be encoded"
    );
    let mut levels = Vec::new();
    loop {
        let bytes = match format {
            PixelFormat::Bc7 => encode(&pixels, w, h, &request.compression, scale, bias),
            PixelFormat::Bc4Unorm | PixelFormat::Bc4Snorm => bc4::encode(
                &pixels,
                w,
                h,
                format == PixelFormat::Bc4Snorm,
                scale[0],
                request.compression.quality,
            ),
            PixelFormat::R32Float => {
                bytemuck::cast_slice(&pixels.iter().map(|p| p[0]).collect::<Vec<_>>()).to_vec()
            }
            PixelFormat::Rgba32Float => bytemuck::cast_slice(&pixels).to_vec(),
        };
        levels.push(Mip {
            width: w,
            height: h,
            bytes,
        });
        if w == 1 && h == 1 {
            break;
        }
        (pixels, w, h) = downsample(&pixels, w, h);
    }
    ensure!(
        SourceStamp::read(&request.file)? == stamp,
        "texture changed while converting"
    );
    if request.mode == TextureMode::Compressed {
        let payload: Vec<u8> = levels
            .iter()
            .flat_map(|l| l.bytes.iter().copied())
            .collect();
        let header = Header {
            version: CACHE_VERSION,
            format,
            request: request.clone(),
            stamp,
            source_hash,
            range_scale: scale,
            range_bias: bias,
            levels: levels
                .iter()
                .map(|l| (l.width, l.height, l.bytes.len()))
                .collect(),
            payload_hash: digest(&payload),
        };
        if let Err(e) = write_cache(&path, &header, &levels) {
            println!(
                "Lava [WARNING]: texture cache write failed for {}: {e:#}",
                request.file.display()
            );
        }
    }
    Ok(PreparedTexture {
        format,
        levels,
        range_scale: scale,
        range_bias: bias,
        fingerprint: key,
        cache_hit: false,
    })
}
fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let default = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2)
            .min(8);
        let count = std::env::var("LAVA_TEXTURE_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(default)
            .clamp(1, 32);
        rayon::ThreadPoolBuilder::new()
            .num_threads(count)
            .thread_name(|i| format!("lava-texture-{i}"))
            .build()
            .expect("texture thread pool")
    })
}
struct State {
    stamp: Option<SourceStamp>,
    generation: u64,
    pending: bool,
    ready: Option<Arc<PreparedTexture>>,
    error: Option<String>,
}
struct Completion {
    request: LoadRequest,
    generation: u64,
    result: Result<PreparedTexture>,
}
/// Per-renderer mailbox; pixel reads, decoding and compression run on the loader pool.
pub struct TextureManager {
    root: PathBuf,
    states: HashMap<LoadRequest, State>,
    sender: mpsc::SyncSender<Completion>,
    receiver: mpsc::Receiver<Completion>,
    last_scan: Instant,
}
impl TextureManager {
    pub fn new(root: PathBuf) -> Self {
        let (sender, receiver) = mpsc::sync_channel(16);
        Self {
            root,
            states: HashMap::new(),
            sender,
            receiver,
            last_scan: Instant::now() - Duration::from_secs(1),
        }
    }
    pub fn request(&mut self, request: LoadRequest) {
        if request.mode == TextureMode::Virtual || self.states.contains_key(&request) {
            return;
        }
        self.states.insert(
            request.clone(),
            State {
                stamp: None,
                generation: 0,
                pending: false,
                ready: None,
                error: None,
            },
        );
        self.schedule(request);
    }
    fn schedule(&mut self, request: LoadRequest) {
        let state = self.states.get_mut(&request).unwrap();
        state.generation += 1;
        state.pending = true;
        state.error = None;
        state.stamp = SourceStamp::read(&request.file).ok();
        let generation = state.generation;
        let sender = self.sender.clone();
        let root = self.root.clone();
        pool().spawn(move || {
            let result = std::panic::catch_unwind(|| prepare(&request, &root))
                .unwrap_or_else(|_| Err(anyhow::anyhow!("texture worker panicked")));
            let _ = sender.send(Completion {
                request,
                generation,
                result,
            });
        });
    }
    pub fn poll(&mut self) {
        self.poll_limited(usize::MAX);
    }
    pub(crate) fn poll_limited(&mut self, limit: usize) {
        if self.last_scan.elapsed() >= Duration::from_millis(250) {
            self.last_scan = Instant::now();
            let changed: Vec<_> = self
                .states
                .iter()
                .filter(|(request, state)| SourceStamp::read(&request.file).ok() != state.stamp)
                .map(|(request, _)| request.clone())
                .collect();
            for request in changed {
                self.schedule(request);
            }
        }
        while self.states.values().filter(|s| s.ready.is_some()).count() < limit {
            let Ok(done) = self.receiver.try_recv() else {
                break;
            };
            let Some(state) = self.states.get_mut(&done.request) else {
                continue;
            };
            if state.generation != done.generation {
                continue;
            }
            state.pending = false;
            match done.result {
                Ok(texture) => {
                    state.ready = Some(Arc::new(texture));
                    state.error = None;
                }
                Err(error) => {
                    let message = format!("{}: {error:#}", done.request.file.display());
                    println!("Lava [WARNING]: texture preparation failed; retaining previous texture or constant material input: {message}");
                    state.error = Some(message);
                }
            }
        }
    }
    pub fn take_ready(&mut self, request: &LoadRequest) -> Option<Arc<PreparedTexture>> {
        self.states.get_mut(request).and_then(|s| s.ready.take())
    }
    pub fn ready(&self, request: &LoadRequest) -> Option<Arc<PreparedTexture>> {
        self.states.get(request).and_then(|s| s.ready.clone())
    }
    pub fn pending(&self) -> bool {
        self.states.values().any(|s| s.pending)
    }
    pub fn has_ready(&self) -> bool {
        self.states.values().any(|s| s.ready.is_some())
    }
    pub fn check_errors(&self) -> Result<()> {
        for state in self.states.values() {
            if let Some(e) = &state.error {
                bail!("final texture preparation failed: {e}");
            }
        }
        Ok(())
    }
    pub fn wait(&mut self) -> Result<()> {
        loop {
            self.poll();
            if !self.pending() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        for state in self.states.values() {
            if let Some(e) = &state.error {
                bail!("final texture preparation failed: {e}");
            }
        }
        Ok(())
    }
    pub fn retain(&mut self, active: &[LoadRequest]) {
        self.states.retain(|key, _| active.contains(key));
    }
}

#[cfg(test)]
mod tests;
