#ifndef LAVA_GEOMETRY
#define LAVA_GEOMETRY
struct Triangle {
  vec4 a;
  vec4 b;
  vec4 c;
  vec4 color;
  vec4 material; // metallic, transmission, IOR, table index
  vec4 uv01;
  vec4 uv2;
  vec4 normalA;
  vec4 normalB;
  vec4 normalC;
};
layout(set = 0, binding = 0) uniform accelerationStructureEXT world;
layout(std430, set = 0, binding = 1) readonly buffer Triangles {
  Triangle triangles[];
};
vec3 geometricNormal(Triangle t) {return normalize(cross(t.b.xyz-t.a.xyz,t.c.xyz-t.a.xyz))*t.normalA.w;}
vec3 shadingNormal(Triangle t,vec2 bary,vec3 view) {
 vec3 ng=geometricNormal(t);
 // Zero/absent corner normals fall back locally; cancellation falls back as well.
 vec3 a=dot(t.normalA.xyz,t.normalA.xyz)>1e-20?t.normalA.xyz:ng;
 vec3 b=dot(t.normalB.xyz,t.normalB.xyz)>1e-20?t.normalB.xyz:ng;
 vec3 c=dot(t.normalC.xyz,t.normalC.xyz)>1e-20?t.normalC.xyz:ng;
 vec3 ns=a*(1.-bary.x-bary.y)+b*bary.x+c*bary.y;
 if(dot(ns,ns)<1e-20)return ng;
 ns=normalize(ns);if(dot(ns,ng)<0.)ns=-ns;
 // Shading normals must preserve geometric entering/exiting classification.
 if(dot(ns,view)*dot(ng,view)<=0.)return ng;
 return ns;
}
#endif
