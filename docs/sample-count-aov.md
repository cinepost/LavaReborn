# Sample count diagnostic AOV

In Lava Render Settings, open **Image** and enable **Sample Count Heat Map AOV**.
Select the settings prim (normally `/Render/Lava`) in the Solaris viewport and
choose `sampleCount` in the AOV menu. The option defaults off.

The heat map uses a fixed scale relative to **Samples**, the requested maximum
per-pixel budget. Black means zero samples; blue means few samples, cyan about
one quarter, green one half, yellow three quarters, and red the full budget.
Uniform sampling produces one color across the entire image, including background
pixels. Adaptive sampling displays actual per-pixel counts, including stopped
pixels, instead of the global completion/sample flag. The scale does not change
as the frame refines. Exposure and beauty filtering do not affect it.

The AOV reuses adaptive convergence statistics. Uniform mode derives its value
from the global sample counter and does not allocate per-pixel statistics.
No new counting atomics, reductions, readbacks, or diagnostic dispatches are
introduced. A separate display shader variant writes an additional RGBA32F plane
only when a Hydra `sampleCount` binding is present. Vulkan/OpenGL external memory
carries this plane alongside beauty and depth; a matching GL shader writes its
Hgi texture directly. The ordinary shader variants contain no diagnostic code.
Turning the AOV off omits its plane (16 bytes/pixel) from subsequent snapshots and
releases its GL texture. Cached pipeline objects can be retained for reuse.
Enabling or disabling the output preserves accumulation, including after
convergence.

Explicit CPU AOV Map calls use the existing warned staging fallback. The viewport
presentation path does not transfer image data through system RAM.

For manually authored USD, use a RenderVar with `sourceType = "raw"`,
`sourceName = "sampleCount"`, and `dataType = "color4f"`. Houdini additionally
requires string `driver:parameters:aov:name = "sampleCount"`; reference it from a
RenderProduct in the selected RenderSettings. JSON renderer clients can request
`settings.sample_count_aov = true`; this is an output request, not a sampling
algorithm control. The heat-map output is color encoded, not a scalar EXR count
channel.
