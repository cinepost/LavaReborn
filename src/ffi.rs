//! Stable C boundary: JSON scene snapshots in, tightly-packed RGBA f32 out.
use crate::{policy::Mode, renderer::Renderer, scene::Scene};
use std::ffi::{c_char, CStr, CString};
pub struct Session {
    renderer: Option<Renderer>,
    worker: Option<crate::worker::Worker>,
    error: CString,
    last_request: Option<crate::worker::Request>,
    last_revision: u64,
    interop: bool,
}
impl Session {
    fn error(&mut self, s: String) {
        self.error = CString::new(s.replace('\0', " ")).unwrap();
    }
}
#[no_mangle]
pub extern "C" fn lava_create() -> *mut Session {
    Box::into_raw(Box::new(Session {
        renderer: None,
        worker: None,
        error: CString::default(),
        last_request: None,
        last_revision: 0,
        interop: false,
    }))
}
/// # Safety
/// `session` must be a live handle returned by lava_create, owned by this thread.
#[no_mangle]
pub unsafe extern "C" fn lava_destroy(session: *mut Session) {
    if !session.is_null() {
        drop(Box::from_raw(session));
    }
}
/// # Safety
/// Session must be live; returned string is borrowed until next call or destruction.
#[no_mangle]
pub unsafe extern "C" fn lava_error(session: *const Session) -> *const c_char {
    if session.is_null() {
        return c"null session".as_ptr();
    }
    (*session).error.as_ptr()
}
/// # Safety
/// All pointers must be valid. JSON is NUL terminated; output holds width*height*4 floats.
/// Calls on one session must be serialized. Returns zero on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn lava_render(
    session: *mut Session,
    json: *const c_char,
    width: u32,
    height: u32,
    final_mode: bool,
    samples: u32,
    exposure: f32,
    output: *mut f32,
) -> i32 {
    if session.is_null() || json.is_null() || output.is_null() {
        return -1;
    }
    let s = &mut *session;
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> anyhow::Result<()> {
            anyhow::ensure!(samples > 0, "samples must be positive");
            let scene: Scene = serde_json::from_str(CStr::from_ptr(json).to_str()?)?;
            scene.validate()?;
            let mode = if final_mode {
                Mode::Final
            } else {
                Mode::Interactive
            };
            if s.renderer
                .as_ref()
                .map(|r| r.width != width || r.height != height || r.mode != mode)
                .unwrap_or(true)
            {
                s.renderer = Some(Renderer::new(width, height, mode, None)?);
            }
            let r = s.renderer.as_mut().unwrap();
            r.set_scene(&scene)?;
            let mut image = Vec::new();
            while r.samples() < samples {
                image = r.render_batch(samples - r.samples(), exposure)?;
            }
            std::ptr::copy_nonoverlapping(image.as_ptr(), output, image.len());
            Ok(())
        }));
    match result {
        Ok(Ok(())) => {
            s.error = CString::default();
            0
        }
        Ok(Err(e)) => {
            s.error(format!("{e:#}"));
            -1
        }
        Err(_) => {
            s.error("Rust renderer panicked".into());
            -1
        }
    }
}
/// Queue a scene snapshot and return its revision, or zero on error.
/// # Safety
/// Session must be live and JSON must be NUL terminated. Calls serialized by caller.
#[no_mangle]
pub unsafe extern "C" fn lava_submit(
    session: *mut Session,
    json: *const c_char,
    width: u32,
    height: u32,
    final_mode: bool,
    samples: u32,
    exposure: f32,
) -> u64 {
    if session.is_null() || json.is_null() {
        return 0;
    }
    let s = &mut *session;
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> anyhow::Result<u64> {
            let scene: Scene = serde_json::from_str(CStr::from_ptr(json).to_str()?)?;
            let worker = s.worker.get_or_insert_with(crate::worker::Worker::new);
            let request = crate::worker::Request {
                scene: std::sync::Arc::new(scene),
                view: None,
                width,
                height,
                mode: if final_mode {
                    Mode::Final
                } else {
                    Mode::Interactive
                },
                samples,
                exposure,
                gpu_output: s.interop,
            };
            let revision = worker.submit(request.clone())?;
            s.last_request = Some(request);
            s.last_revision = revision;
            Ok(revision)
        }));
    match result {
        Ok(Ok(revision)) => {
            s.error = CString::default();
            revision
        }
        Ok(Err(e)) => {
            s.error(format!("{e:#}"));
            0
        }
        Err(_) => {
            s.error("submit panicked".into());
            0
        }
    }
}
/// Update only camera/lights/settings while retaining immutable geometry.
/// # Safety
/// Same requirements as lava_submit. Only valid after a successful full submit.
#[no_mangle]
pub unsafe extern "C" fn lava_update_view(
    session: *mut Session,
    json: *const c_char,
    width: u32,
    height: u32,
    samples: u32,
    exposure: f32,
) -> u64 {
    if session.is_null() || json.is_null() {
        return 0;
    }
    let s = &mut *session;
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> anyhow::Result<u64> {
            let view: Scene = serde_json::from_str(CStr::from_ptr(json).to_str()?)?;
            anyhow::ensure!(
                view.meshes.is_empty() && view.curves.is_empty(),
                "view updates must not contain geometry"
            );
            let mut request = s
                .last_request
                .clone()
                .ok_or_else(|| anyhow::anyhow!("full scene required before view updates"))?;
            anyhow::ensure!(
                request.mode == Mode::Interactive,
                "view updates require interactive mode"
            );
            request.view = Some(std::sync::Arc::new(view));
            request.width = width;
            request.height = height;
            request.samples = samples;
            request.exposure = exposure;
            let revision = s.worker.as_ref().unwrap().submit(request.clone())?;
            s.last_request = Some(request);
            s.last_revision = revision;
            Ok(revision)
        }));
    match result {
        Ok(Ok(revision)) => {
            s.error = CString::default();
            revision
        }
        Ok(Err(e)) => {
            s.error(format!("{e:#}"));
            0
        }
        Err(_) => {
            s.error("view update panicked".into());
            0
        }
    }
}

/// Copy latest completed frame for requested revision; returns sample count, zero if pending,
/// -1 on error. Asynchronous output is scene-linear RGBA f32, top row first.
/// # Safety
/// Session must be live. Output holds at least capacity floats; calls serialized by caller.
#[no_mangle]
pub unsafe extern "C" fn lava_poll(
    session: *mut Session,
    revision: u64,
    output: *mut f32,
    capacity: usize,
) -> i64 {
    if session.is_null() || output.is_null() {
        return -1;
    }
    let s = &mut *session;
    let Some(worker) = s.worker.as_ref() else {
        return 0;
    };
    let failure = worker.error.lock().unwrap().clone();
    if let Some((_, e)) = failure.filter(|(r, _)| *r == revision) {
        s.error(e);
        return -1;
    }
    let frame = worker.frame.lock().unwrap();
    let Some(f) = frame
        .as_ref()
        .filter(|f| f.revision == revision && !f.pending_scene)
    else {
        return 0;
    };
    let gpu_pixels;
    let pixels = if let Some(g) = &f.gpu {
        match g.read_color() {
            Ok(p) => {
                gpu_pixels = p;
                &gpu_pixels
            }
            Err(e) => {
                drop(frame);
                s.error(e.to_string());
                return -1;
            }
        }
    } else {
        &f.pixels
    };
    if capacity < pixels.len() {
        drop(frame);
        s.error("output capacity too small".into());
        return -1;
    }
    std::ptr::copy_nonoverlapping(pixels.as_ptr(), output, pixels.len());
    if f.pending_scene {
        0
    } else {
        f.samples as i64
    }
}
#[no_mangle]
pub extern "C" fn lava_is_supported() -> bool {
    std::panic::catch_unwind(|| {
        crate::vulkan::Instance::new()
            .and_then(|i| i.adapters())
            .map(|a| a.iter().any(|a| a.ray_query && a.hardware))
            .unwrap_or(false)
    })
    .unwrap_or(false)
}
/// # Safety
/// Session is live and calls on it are serialized.
#[no_mangle]
pub unsafe extern "C" fn lava_enable_interop(session: *mut Session, enabled: bool) {
    if let Some(s) = session.as_mut() {
        s.interop = enabled;
    }
}
/// Poll progress without mapping any image memory.
/// # Safety
/// Session is live and calls on it are serialized.
#[no_mangle]
pub unsafe extern "C" fn lava_status(session: *mut Session, revision: u64) -> i64 {
    let Some(s) = session.as_mut() else {
        return -1;
    };
    let Some(w) = s.worker.as_ref() else {
        return 0;
    };
    let error = w.error.lock().unwrap().clone();
    if let Some((_, e)) = error.filter(|(r, _)| *r == revision) {
        s.error(e);
        return -1;
    }
    let f = w.frame.lock().unwrap();
    f.as_ref()
        .filter(|f| f.revision == revision)
        .map(|f| if f.pending_scene { 0 } else { f.samples as i64 })
        .unwrap_or(0)
}
/// # Safety
/// Session is live; output has capacity floats; calls are serialized.
#[no_mangle]
pub unsafe extern "C" fn lava_poll_depth(
    session: *mut Session,
    revision: u64,
    output: *mut f32,
    capacity: usize,
) -> i64 {
    if session.is_null() || output.is_null() {
        return -1;
    }
    let s = &mut *session;
    let Some(w) = s.worker.as_ref() else {
        return 0;
    };
    let f = w.frame.lock().unwrap();
    let Some(f) = f.as_ref().filter(|f| f.revision == revision) else {
        return 0;
    };
    let gpu_depth;
    let depth = if let Some(g) = &f.gpu {
        match g.read_depth() {
            Ok(d) => {
                gpu_depth = d;
                &gpu_depth
            }
            Err(_) => return -1,
        }
    } else {
        &f.depth
    };
    if capacity < depth.len() {
        return -1;
    }
    std::ptr::copy_nonoverlapping(depth.as_ptr(), output, depth.len());
    if f.pending_scene {
        0
    } else {
        f.samples as i64
    }
}
#[repr(C)]
pub struct GpuInfo {
    pub memory_fd: i32,
    pub ready_fd: i32,
    pub returned_fd: i32,
    pub allocation_size: u64,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub serial: u64,
    pub uuid: [u8; 16],
}
pub struct GpuLease(std::sync::Arc<crate::interop::GpuFrame>);
/// Export a new immutable GPU snapshot. Returned lease retains its Vulkan storage.
/// GL import consumes both FDs; callers close them if not imported. A semaphore
/// must be waited exactly once per snapshot; share the import across both AOVs.
/// # Safety
/// Session and info are valid, calls serialized. Release a returned lease once.
#[no_mangle]
pub unsafe extern "C" fn lava_gpu_frame(
    session: *mut Session,
    revision: u64,
    after_serial: u64,
    info: *mut GpuInfo,
) -> *mut GpuLease {
    if session.is_null() || info.is_null() {
        return std::ptr::null_mut();
    }
    let s = &mut *session;
    let Some(w) = s.worker.as_ref() else {
        return std::ptr::null_mut();
    };
    let f = w.frame.lock().unwrap();
    let Some(g) = f
        .as_ref()
        .filter(|f| {
            f.revision == revision
                || s.last_request.as_ref().is_some_and(|r| {
                    revision == s.last_revision
                        && r.mode == Mode::Interactive
                        && f.interactive
                        && f.width == r.width
                        && f.height == r.height
                })
        })
        .and_then(|f| f.gpu.clone())
        .filter(|g| g.serial > after_serial)
    else {
        return std::ptr::null_mut();
    };
    let (memory_fd, ready_fd, returned_fd) = match g.export() {
        Ok(fds) => fds,
        Err(_) => return std::ptr::null_mut(),
    };
    *info = GpuInfo {
        memory_fd,
        ready_fd,
        returned_fd,
        allocation_size: g.buffer.allocation_size,
        width: g.width,
        height: g.height,
        samples: g.samples,
        serial: g.serial,
        uuid: g.buffer.device.uuid,
    };
    Box::into_raw(Box::new(GpuLease(g)))
}
/// # Safety
/// Lease is null or a live un-released lease returned by lava_gpu_frame.
#[no_mangle]
pub unsafe extern "C" fn lava_release_gpu_frame(lease: *mut GpuLease) {
    if !lease.is_null() {
        let lease = Box::from_raw(lease);
        drop(lease.0);
    }
}
/// 0 pending, 1 CPU fallback frame, 2 exported Vulkan frame.
/// # Safety
/// Session is live and calls are serialized.
#[no_mangle]
pub unsafe extern "C" fn lava_frame_transport(session: *const Session, revision: u64) -> i32 {
    let Some(s) = session.as_ref() else {
        return 0;
    };
    let Some(w) = s.worker.as_ref() else {
        return 0;
    };
    let frame = w.frame.lock().unwrap();
    frame
        .as_ref()
        .filter(|f| {
            f.revision == revision
                || s.last_request.as_ref().is_some_and(|r| {
                    revision == s.last_revision
                        && r.mode == Mode::Interactive
                        && f.interactive
                        && f.width == r.width
                        && f.height == r.height
                })
        })
        .map(|f| if f.gpu.is_some() { 2 } else { 1 })
        .unwrap_or(0)
}

/// # Safety
/// Lease is live and calls on its session are serialized.
#[no_mangle]
pub unsafe extern "C" fn lava_gpu_frame_gl_submitted(lease: *const GpuLease) {
    if let Some(l) = lease.as_ref() {
        l.0.mark_gl_submitted();
    }
}

/// Diagnostic counter of actual GPU-to-CPU frame transfers in this process.
#[no_mangle]
pub extern "C" fn lava_image_readback_count() -> u64 {
    crate::interop::IMAGE_READBACKS.load(std::sync::atomic::Ordering::Relaxed)
}

#[no_mangle]
pub extern "C" fn lava_gpu_presentation_count() -> u64 {
    crate::interop::GPU_PRESENTATIONS.load(std::sync::atomic::Ordering::Relaxed)
}
