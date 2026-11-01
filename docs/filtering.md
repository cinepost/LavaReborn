# Accumulation-guided spatial filtering

The accumulation pass filters its running linear HDR color mean. Filter weights
depend on accumulated colors and pixel distance, with no visibility-buffer,
material, depth or normal input. Uniform and adaptive shader products use the
same filter. Stopped adaptive pixels retain their raw history and can still
participate in the full-image spatial filter. The adaptive sampling guard can
resume a pixel beside an unconverged neighbor, extending its existing history.

The current radius is one pixel. Each neighbor gets a Gaussian spatial weight
multiplied by an RGB bilateral range weight. The range standard deviation is
10% of the pair's largest absolute RGB component, with an absolute minimum of
0.001 in linear scene units. A symmetric pair scale avoids treating bright and
dark centers differently. Colors are normalized before subtracting and squaring
to avoid overflowing on very bright HDR values.

This smooths low-contrast noise while reducing mixing across strong luminance
or chromatic edges. Similar-colored distinct surfaces can mix locally; there
is no geometric boundary guarantee. Filtering remains a biased local smoothing
operation and does not change raw accumulation, adaptive convergence statistics
or sample counts. Exposure is applied afterward on the GPU.

The filter adds no guide buffer, extra rays, CPU image transfers or material
dependencies. Changes to the last stochastic visibility sample cannot alter
the result for a fixed accumulated image. The depth AOV continues to come from
the visibility pass independently of color filtering.

The GPU regression test replaces all visibility data with arbitrary values and
requires identical output, in both uniform and adaptive variants. It also
checks low-contrast smoothing, high-contrast color-edge preservation, unchanged
raw history/sample counters, and finite constant preservation from black to
very bright HDR values. The kapibara sampler comparison reports raw and filtered
errors against a 1,024-sample reference; see [adaptive sampling](adaptive-sampling.md).

## Coverage and transparent background

Uncovered camera samples produce transparent black (RGBA zero), including
stochastically rejected opacity cutouts. Accepted surface and visible emitter
samples have alpha one. Accumulation averages all four channels, yielding
premultiplied linear RGB and coverage alpha. Transmission remains a surface
scattering lobe; it does not reduce coverage alpha.

Filtering preserves the center's accumulated alpha exactly. It filters
unassociated color from covered neighbors, then premultiplies by that alpha.
Zero-coverage pixels remain RGBA zero. This avoids filling cutout holes or
mixing transparent black into surface colors. The filter still uses only
accumulated history, with no last-sample visibility dependency. Nonlinear tone
mapping also unpremultiplies and repremultiplies; linear output retains the
original premultiplied representation.

Adaptive convergence additionally checks the Bernoulli coverage error estimated
from accumulated alpha. A black surface with varying cutout coverage therefore
cannot stop just because its RGB luminance is constant. This uses existing
history and introduces no statistics buffer.

There is no implicit environment illumination or renderer background color.
Camera misses are transparent, and secondary misses contribute zero unless an
explicit analytic light is intersected. Solaris can composite its own viewport
background behind the rendered image. Explicit scene lights, material emission,
and the optional automatic headlight keep their normal behavior. Dome/environment
light support remains outside the prototype's implemented light types.
