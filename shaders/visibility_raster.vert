#version 460
#extension GL_EXT_ray_query : require
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "geometry.glsl"
#include "camera.glsl"
layout(location=0) out vec3 barycentric;
layout(location=1) flat out uint primitive;
void main(){
 primitive=uint(gl_VertexIndex)/3u;uint corner=uint(gl_VertexIndex)%3u;
 Triangle t=triangles[primitive];vec3 world=corner==0u?t.a.xyz:corner==1u?t.b.xyz:t.c.xyz;
 barycentric=vec3(corner==0u?1.:0.,corner==1u?1.:0.,corner==2u?1.:0.);
 vec3 local=(camera.worldToCamera*vec4(world,1)).xyz-cameraLens();
 gl_Position=cameraProjection()*vec4(local,1);
}
