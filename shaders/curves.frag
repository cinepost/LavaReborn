#version 460
#extension GL_EXT_ray_query : require
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "geometry.glsl"
#include "camera.glsl"
#include "materials/visibility.glsl"
layout(location=0) in vec3 barycentric;
layout(location=1) flat in uint primitive;
layout(location=0) out uvec4 hitData;
layout(location=1) out uvec4 hitAttributes;
void main(){
 vec3 origin,direction;float plane;
 cameraRay(uvec2(gl_FragCoord.xy),origin,direction,plane);
 {
  float distance;if(!curveIntersect(primitive,origin,direction,camera.clippingOffset.x*plane,camera.clippingOffset.y*plane,distance)||!materialCovered(primitive,origin,direction))discard;
  vec2 bary=vec2(distance,0);Triangle curve=hitTriangle(primitive,bary,origin,direction);
  vec4 clip=cameraProjection()*vec4((camera.worldToCamera*vec4(curve.a.xyz,1)).xyz-cameraLens(),1);
  gl_FragDepth=clip.z/clip.w;
  hitData=uvec4(primitive+1u,floatBitsToUint(distance),0,floatBitsToUint(distance/plane));
  hitAttributes=uvec4(materialId(primitive),floatBitsToUint(curve.uv01.xy),0);return;
 }

}
