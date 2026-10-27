#ifndef LAVA_MATERIAL_LUTS
#define LAVA_MATERIAL_LUTS
#include "lut_offsets.glsl"
layout(std430,set=0,binding=14) readonly buffer MaterialLuts {vec4 materialLutData[];};
// Reference callback ABI. Storage-buffer lookup avoids embedding megabytes of
// constants in each shader and preserves the reference table interpolation.
vec4 materialLutFetch(int id,ivec3 p) {
 ivec3 size=ivec3(32,id==6?1:32,(id==0||id==3)?32:1);
 p=clamp(p,ivec3(0),size-1);
 return materialLutData[materialLutOffsets[id]+uint((p.z*size.y+p.y)*size.x+p.x)];
}
vec4 materialLutSlice(int id,ivec3 p,vec2 t) {
 return mix(mix(materialLutFetch(id,p),materialLutFetch(id,p+ivec3(1,0,0)),t.x),mix(materialLutFetch(id,p+ivec3(0,1,0)),materialLutFetch(id,p+ivec3(1,1,0)),t.x),t.y);
}
vec4 materialLut3D(int id,vec3 uv) {
 vec3 size=vec3(32,id==6?1:32,(id==0||id==3)?32:1);
 vec3 p=clamp(uv*size-.5,vec3(0),size-1.);ivec3 low=ivec3(floor(p));vec3 t=fract(p);
 return mix(materialLutSlice(id,low,t.xy),materialLutSlice(id,low+ivec3(0,0,1),t.xy),t.z);
}
vec4 materialLut2D(int id,vec2 uv) {return materialLut3D(id,vec3(uv,.5));}
#define OPENPBR_USE_TEXTURE_LUTS 1
#define OPENPBR_SAMPLE_2D_TEXTURE(id,uv) materialLut2D(id,uv)
#define OPENPBR_SAMPLE_3D_TEXTURE(id,uv) materialLut3D(id,uv)
#endif
