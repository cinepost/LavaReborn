#ifndef LAVA_SHADOW_PROFILE
#define LAVA_SHADOW_PROFILE
// Diagnostic products only; storage is not accessed by the production specialization.
struct ShadowRecord {vec4 originDistance;vec4 directionLight;uvec4 result;};
layout(std430,set=0,binding=21) buffer ShadowProfile {uvec4 shadowStats;ShadowRecord shadowRecords[];};
layout(std430,set=0,binding=22) buffer ContinuationProfile {uvec4 continuationRecords[];};
#endif
