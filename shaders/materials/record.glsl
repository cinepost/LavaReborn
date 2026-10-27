// Generated material ABI v1. No BSDF dependency.
#ifndef LAVA_MATERIAL_RECORD
#define LAVA_MATERIAL_RECORD
struct MaterialRecord { vec4 values[38]; };
layout(std430,set=0,binding=13) readonly buffer MaterialTable { MaterialRecord materials[]; };
#endif
