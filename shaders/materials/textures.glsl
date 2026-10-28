#ifndef LAVA_MATERIAL_TEXTURES
#define LAVA_MATERIAL_TEXTURES
#include "../textures.glsl"
#include "record.glsl"
struct TextureSlot {TextureHandle handle;uvec4 info;vec4 scale;vec4 bias;vec4 uv;vec4 parameters;};
layout(std430,set=0,binding=19) readonly buffer TextureSlots {TextureSlot textureSlots[];};
vec4 materialValue(uint id,uint field,vec2 uv,float footprint) {
 TextureSlot slot=textureSlots[id*40u+field];
 vec4 fallback=field<38u?materials[id].values[field]:vec4(0);
 if(slot.info.x==0u)return fallback;
 vec4 value;
 if(!textureSample(slot.handle,uv*slot.uv.xy+slot.uv.zw,footprint*abs(slot.uv.xy),slot.info.zw,value))return fallback;
 value=value*slot.scale+slot.bias;
 if(slot.info.y<4u)value=vec4(value[slot.info.y]);
 if(slot.info.y==5u)value=vec4(dot(value.rgb,vec3(.299,.587,.114)));
 if(field<38u&&slot.parameters.x!=0.)value*=materials[id].values[field];
 return value;
}
// Conservative isotropic pixel footprint on the triangle's UV chart. Compute
// passes cannot use implicit screen derivatives; secondary rays use this camera
// footprint approximation until full ray differentials/cones are introduced.
float materialFootprint(Triangle t,vec3 p) {
 vec2 a=t.uv01.zw-t.uv01.xy,b=t.uv2.xy-t.uv01.xy;
 vec3 e=t.b.xyz-t.a.xyz,f=t.c.xyz-t.a.xyz;
 float det=a.x*b.y-a.y*b.x;
 if(abs(det)<1e-15)return 0.;
 vec3 dpdu=(e*b.y-f*a.y)/det,dpdv=(f*a.x-e*b.x)/det;
 float worldPerUv=max(1e-8,min(length(dpdu),length(dpdv)));
 float depth=abs((camera.worldToCamera*vec4(p,1)).z);
 float pixel=max(camera.optics.x/float(pc.width),camera.optics.y/float(pc.height));
 if(camera.optics.w==0.)pixel*=depth/max(camera.optics.z,1e-8);
 return pixel/worldPerUv;
}
void materialRayCoordinates(uint primitive,vec3 origin,vec3 direction,out vec2 uv,out float footprint) {
 if(isCurve(triangles[primitive])) {
  float distance;curveIntersect(primitive,origin,direction,0.,1e30,distance);
  vec2 bary=vec2(distance,0);Triangle c=hitTriangle(primitive,bary,origin,direction);
  uv=c.uv01.xy;footprint=materialFootprint(c,c.a.xyz);return;
 }
 Triangle t=triangles[primitive];vec3 e=t.b.xyz-t.a.xyz,f=t.c.xyz-t.a.xyz;
 vec3 n=cross(e,f);float denom=dot(n,direction);
 float distance=abs(denom)>1e-20?dot(n,t.a.xyz-origin)/denom:0.;
 vec3 p=origin+distance*direction,v=p-t.a.xyz;
 float ee=dot(e,e),ef=dot(e,f),ff=dot(f,f),det=ee*ff-ef*ef;
 vec2 bary=abs(det)>1e-20?vec2(dot(v,e)*ff-dot(v,f)*ef,dot(v,f)*ee-dot(v,e)*ef)/det:vec2(0);
 uv=t.uv01.xy*(1.-bary.x-bary.y)+t.uv01.zw*bary.x+t.uv2.xy*bary.y;
 footprint=materialFootprint(t,p);
}
#endif
