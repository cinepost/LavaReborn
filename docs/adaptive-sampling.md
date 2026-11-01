# Adaptive sampling

In Lava Render Settings → Sampling, enable **Adaptive Sampling**. **Samples**
becomes the maximum per-pixel budget. Defaults are adaptive off, relative error
threshold 0.03 and minimum samples 16. Smaller thresholds retain more samples.
Minimum Samples is a lower bound on convergence testing, not a guarantee that
pixels stop at that exact count; the maximum budget always takes precedence.

Accumulation keeps each pixel's count, running luminance mean, Welford M2 and
consecutive successful check count. Every four samples, after Minimum Samples,
it tests `1.96 * sqrt(M2 / (n * (n - 1)))` against
`threshold * max(abs(mean), 0.01)` in linear scene units. Three consecutive checks
must pass to nominate a pixel as converged. The active mask includes a one-pixel
guard around pixels that fail convergence: apparently clean neighbors keep
sampling beside noisy fur, silhouette coverage and partial shadows. Flat
constant regions can still stop at 24 samples with the default minimum.
Statistics use complete, unfiltered path radiance, before exposure or filtering.
Coverage alpha also must pass an absolute error check:
`1.96 * sqrt(alpha * (1 - alpha) / (n - 1)) <= threshold`. This uses accumulated
alpha and prevents dark stochastic cutouts from converging on constant RGB alone.
This is a practical error estimate, not a statistical guarantee for correlated
low-discrepancy samples. Luminance-only tests can miss chromatic noise and rare
bright events; variance-based stopping can introduce bias.

GPU subgroup compaction builds an active-pixel list. Indirect dispatch traces
primary ray-query visibility and initializes paths only for active pixels.
Secondary compaction and ray reordering retain the existing pixel index in each
path record. Accumulation updates only active pixels and preserves stopped
history. Spatial filtering and display still process the whole image. The
[spatial filter](filtering.md) uses accumulated HDR color instead of stochastic
visibility data. Raster
mode still draws scene geometry, while resolve skips settled pixels and shading
uses the compacted list.

Adaptive mode uses progressive Halton sampling in interactive and final modes,
indexed by each pixel's count for path sampling and ray-query camera rays.
Raster film/lens jitter uses the frame index so shading reconstruction matches
the shared raster projection when a guard pixel resumes after skipped samples.
Uniform final rendering retains Hammersley.
Preprocessor variants remove adaptive statistics and addressing from uniform
visibility, shading and accumulation shader products. Adaptive pipelines are
created on first enable and retained, with existing SPIR-V and Vulkan caches.

Statistics use 16 bytes per pixel and the active list uses another 4 bytes per
pixel, allocated on first enable in device-local memory. No image or per-pixel
statistics are read back in normal viewport rendering. A four-byte active count
is copied to the CPU at the existing completed-batch boundary to signal early
completion. Final batches contain up to 16 iterations, so completion can be
reported after a few empty GPU iterations. Scene/camera/material/light changes
clear statistics with accumulation; floating-point display exposure edits keep
them.

USD attributes: `lava:adaptiveSampling`, `lava:adaptiveThreshold`,
`lava:adaptiveMinSamples`. JSON settings: `adaptive_sampling`,
`adaptive_threshold`, `adaptive_min_samples`. Changing these resets history.

## Reproduce the kapibara comparison

Run `scripts/profile_hip_scene.py` with Houdini's hython to export the HIP without
saving changes, using `--fit` to create a reproducible full-scene camera. Run the
ignored release test `renderer::adaptive::tests::profile_adaptive_scene` with
`LAVA_ADAPTIVE_SCENE` pointing to its `scene.json` export. Optional variables are
`LAVA_ADAPTIVE_WIDTH`, `LAVA_ADAPTIVE_HEIGHT`, `LAVA_ADAPTIVE_SAMPLES` and
`LAVA_ADAPTIVE_REPORT`.

The comparison warms pipelines before timing, excludes scene activation, and
uses identical geometry/camera/light settings. It measures a uniform reference
at four times the budget, uniform Hammersley and Halton controls, adaptive
thresholds 0.03 and 0.01 with minimum 32, a minimum-128 case, and a forced-full-
budget adaptive case. The report records elapsed rendering time, actual sample
counts and relative RGB RMSE against the reference, both before and after
filtering. Diagnostic image and statistics readbacks are explicit test
operations, not viewport transport.

## Initial kapibara measurements (previous filter)

RTX 2060 SUPER, fitted full-kapibara view at frame 1, 1280×720, ray-query
visibility, maximum 256 samples, diffuse depth 5, reflection/refraction disabled.
The HIP contains 499,994 strands and 4,499,946 control points. Pipeline creation
and scene activation are excluded; these are individual warm rendering runs.
The reference uses 1,024 uniform Hammersley samples.

| Configuration | Time | Mean samples | Raw RGB relative RMSE |
| --- | ---: | ---: | ---: |
| Uniform Hammersley | 3.077 s | 256 | 2.008% |
| Uniform Halton control | 3.054 s | 256 | 2.063% |
| Adaptive 0.03, minimum 32 | 2.520 s | 52.24 | 2.162% |
| Adaptive 0.01, minimum 32 | 2.804 s | 61.62 | 2.109% |
| Adaptive 0.03, minimum 128 | 2.834 s | 141.97 | 2.093% |
| Adaptive forced to full budget | 3.270 s | 256 | 2.063% |

Adaptive bookkeeping adds approximately 7.1% against the same Halton sequence
when no pixels stop. Threshold 0.03/minimum 32 saves 18.1% against the existing
uniform final path. Savings are smaller than the reduction in mean samples:
remaining noisy fur pixels still need expensive tracing, while full-image
compaction and display work remain.

The previous primitive-aware spatial filter used the last visibility sample.
Consequently filtered-reference RMSE differs substantially between Hammersley
(2.056%) and uniform Halton (4.802%), even without adaptive stopping. Adaptive
0.03/minimum 32 gives 4.826% filtered RMSE. That sampler/filter interaction motivated replacing visibility-guided weights
with accumulated-color bilateral weights; see [filtering](filtering.md). The
initial comparison above predates that replacement. The 1,024-sample reference is an estimate, not ground truth.

Validation: 45 Vulkan GPU tests, 31 CPU tests, HDA defaults/USD authoring/HIP
reload, exact uniform/adaptive mapping with a queue large enough to enable
reordering, resize/reset/exposure, GPU-only presentation, and native Solaris
animated-light edits with adaptive mode enabled passed. Native pixel inspection
requests CPU AOV maps deliberately; compositor delivery remains GPU color/depth.

## Accumulated-color filter comparison

Replacing the last-sample primitive test with [accumulated-color bilateral
weights](filtering.md) closes the sampler-dependent filtering gap. The same
kapibara export, camera, settings, resolution and sample budgets were rerun:

| Configuration | Previous filtered RMSE | New filtered RMSE | New render time |
| --- | ---: | ---: | ---: |
| Uniform Hammersley | 2.056% | 1.758% | 3.265 s |
| Uniform Halton | 4.802% | 1.796% | 3.265 s |
| Adaptive 0.03, minimum 32 | 4.826% | 1.833% | 2.688 s |
| Adaptive 0.01, minimum 32 | 4.807% | 1.817% | 2.980 s |

Each filtered RMSE uses the 1,024-sample reference processed with the respective
filter; these are filtered-reference errors, not errors against ground truth.
Raw relative RMSE and actual sample counts match the previous run exactly for
every configuration. Thus the filtering change does not alter sampling or
convergence. Adaptive 0.03 remains 17.7% faster than uniform final rendering in
this run. Absolute timings across separate warm runs should not be interpreted
as an isolated measurement of filter cost.

The visibility-independence/HDR/edge-preservation regression, all 46 GPU tests,
31 CPU tests, Clippy, and the installed native Solaris adaptive rendering test
pass. GPU viewport presentation remains direct Vulkan/OpenGL interop.

## Fur and partial-light convergence guard

A short constant sample prefix could nominate a pixel as converged after 24
samples, even when later samples would find lit fur or a narrow lighting path.
This produced localized dark gaps. A three-by-three dilation of the base
unconverged mask now retains a one-pixel guard around noisy pixels. The guard
does not recursively expand across the image. Flat constant regions still
converge early; retained neighbors keep their history and continue sampling.
The raster resolve uses the same eligibility test as queue compaction. Raster
camera film/lens samples use the shared frame index to match the projection if
a guarded pixel resumes after skipping iterations.

The saved kapibara settings were confirmed as interactive ray queries, maximum
256 samples, threshold 0.03, minimum 16, exposure zero. At 640×360 with the fitted
view and a uniform Halton reference at the same budget, pixels more than 20%
below the reference decreased from 69 to zero. The measured central half-width,
half-height region's mean luminance deficit decreased from 0.173% to 0.023%.
These are finite-sample comparisons, not a proof of unbiased adaptive stopping.
Rare isolated paths outside the guard still require an adequate sample budget.

The sample-by-sample diagnostic checks that pixels whose sample count does not
advance retain their exact raw history. A native Solaris test captures the GPU
color texture, then repeats rendering four times after convergence and requires
identical RGBA pixels. The final image stayed unchanged after completion. These
explicit diagnostics read images back; normal viewport transport remains GPU
interop, with the existing warning if Houdini requests a CPU AOV map.

The guard trades some speed for more reliable coverage. A repeat of the 1280×720
comparison with minimum 32 measured uniform final rendering at 3.146 s and
adaptive 0.03 at 2.839 s (9.8% faster). Adaptive mean samples increased from 52.24
to 59.74; raw relative RMSE improved from 2.162% to 2.092%, and filtered RMSE from
1.833% to 1.801%. Timings are warm individual runs. All 48 GPU tests, 31 CPU
tests, Clippy, HDA authoring/reload and native Solaris presentation checks pass.
