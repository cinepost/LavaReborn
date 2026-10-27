#ifndef LAVA_MATERIAL_VISIBILITY
#define LAVA_MATERIAL_VISIBILITY
#include "record.glsl"
#include "textures.glsl"
uint materialId(uint primitive) {return floatBitsToUint(triangles[primitive].material.w);}
float materialCoverage(uint primitive,vec3 origin,vec3 direction) {
 uint id=materialId(primitive);
 if(materials[id].values[0].x==0.)return 1.;
 vec2 uv;float footprint;materialRayCoordinates(primitive,origin,direction,uv,footprint);
 return clamp(materialValue(id,31u,uv,footprint).x,0.,1.);
}
#include "coverage.glsl"
#endif
