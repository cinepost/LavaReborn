#ifndef LAVA_COMMON
#define LAVA_COMMON
// Hit: triangle index + 1 (zero is miss), barycentric coordinates, distance.
struct Hit {
  uint primitive;
  float u;
  float v;
  float distance;
  uvec4 attributes; // opaque producer metadata; consumers may ignore it
};
layout(std430, set = 0, binding = 3) buffer Visibility { Hit hits[]; };
layout(std430, set = 0, binding = 4) buffer Radiance { vec4 radiance[]; };
layout(std430, set = 0, binding = 5) buffer History { vec4 history[]; };
layout(std430, set = 0, binding = 6) buffer Display { vec4 display[]; };
layout(std430, set = 0, binding = 8) buffer Filtered { vec4 filtered[]; };
layout(std140, set = 0, binding = 7) uniform Camera {
  mat4 cameraToWorld;
  vec4 optics; // horizontal/vertical aperture, focal length, orthographic
  vec4 clippingOffset; // near, far, offsets
  vec4 lens;           // f-stop, focus distance, independent per-pixel ray jitter, reserved
  mat4 worldToCamera;
}
camera;
layout(push_constant) uniform Settings {
  uint width;
  uint height;
  uint sampleIndex;
  uint lightCount;
  float exposure;
  uint radius;
  uint linearOutput;
  uint filterPhase;
  uint diffuseDepth;
  uint reflectionDepth;
  uint refractionDepth;
  uint rouletteStart;
  uint rayPhase;
  uint queueParity;
  uint maxGroupsX;
  uint samplingMode;
  uint targetSamples;
}
pc;
uint hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}
float random(inout uint seed) {
  seed += 0x9e3779b9u;
  seed = hash(seed);
  return float(seed >> 8) * (1.0 / 16777216.0);
}
#endif
