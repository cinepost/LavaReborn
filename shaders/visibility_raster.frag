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
 if(triangles[primitive].uv2.z!=0. || !materialCovered(primitive,origin,direction))discard;
 Triangle t=triangles[primitive];vec3 p=t.a.xyz*barycentric.x+t.b.xyz*barycentric.y+t.c.xyz*barycentric.z;
 float depth=-(camera.worldToCamera*vec4(p,1)).z;
 vec2 uv=t.uv01.xy*barycentric.x+t.uv01.zw*barycentric.y+t.uv2.xy*barycentric.z;
 hitData=uvec4(primitive+1u,floatBitsToUint(barycentric.yz),floatBitsToUint(depth));
 hitAttributes=uvec4(materialId(primitive),floatBitsToUint(uv),0);
}
