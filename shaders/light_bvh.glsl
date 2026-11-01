#ifndef LAVA_LIGHT_BVH
#define LAVA_LIGHT_BVH
#include "sampling.glsl"
struct Light {
  vec4 position_kind;
  vec4 direction_cone;
  vec4 color_power;
  vec4 u_soft;
  vec4 v;
  vec4 response;
  uvec4 bvh;
};
layout(std430,set=0,binding=2) readonly buffer Lights { Light lights[]; };
// Read-only subsystem: reusable by any pass with common light/settings bindings.
struct LightNode { vec4 minPower; vec4 maxInfinite; uvec4 links; };
layout(std430,set=0,binding=9) readonly buffer LightNodes { LightNode lightNodes[]; };
const float PI=3.14159265358979323846;
float nodeImportance(uint index,vec3 p) {
  LightNode n=lightNodes[index];
  if(n.maxInfinite.w>0.)return n.minPower.w;
  vec3 delta=max(n.minPower.xyz-p,vec3(0))+max(p-n.maxInfinite.xyz,vec3(0));
  vec3 extent=n.maxInfinite.xyz-n.minPower.xyz;
  return n.minPower.w/(dot(delta,delta)+.25*dot(extent,extent)+1e-4);
}
uint chooseLight(vec3 p,inout uint seed,out float pmf) {
  uint node=0u;pmf=1.;
  while(lightNodes[node].links.z==0xffffffffu) {
    uvec2 children=lightNodes[node].links.xy;
    float a=nodeImportance(children.x,p),b=nodeImportance(children.y,p);
    float probability=clamp(a/(a+b),1e-4,1.-1e-4);
    bool left=pathSample(seed)<probability;
    pmf*=left?probability:1.-probability;
    node=left?children.x:children.y;
  }
  return lightNodes[node].links.z;
}
float lightPmf(uint light,vec3 p) {
  uint node=lights[light].bvh.x;float pdf=1.;
  while(lightNodes[node].links.w!=0xffffffffu) {
    uint parent=lightNodes[node].links.w;
    uvec2 children=lightNodes[parent].links.xy;
    float a=nodeImportance(children.x,p),b=nodeImportance(children.y,p);
    float probability=clamp(a/(a+b),1e-4,1.-1e-4);
    pdf*=node==children.x?probability:1.-probability;node=parent;
  }return pdf;
}
float emitterCosine(Light l,vec3 normal,vec3 direction) {
 float c=dot(normal,-direction);return (l.bvh.y&1u)!=0u?max(0.,c):abs(c);
}
float distantSolidAngle(Light l) {return 2.*PI*l.response.z;}
bool finiteDistant(Light l) {return l.position_kind.w==3.&&l.response.z>1e-12&&l.response.w>0.;}
vec3 sampleDistant(Light l,inout uint seed) {
 vec3 axis=-normalize(l.direction_cone.xyz);
 if(!finiteDistant(l))return axis;
 float d=l.response.z*pathSample(seed),phi=2.*PI*pathSample(seed);
 float sine=sqrt(max(0.,d*(2.-d)));
 vec3 u=normalize(cross(abs(axis.z)<.999?vec3(0,0,1):vec3(0,1,0),axis));
 return normalize(axis*(1.-d)+sine*(u*cos(phi)+cross(axis,u)*sin(phi)));
}
float responseForCategory(vec2 response,uint category) {return category==0u?1.:(category==1u?response.x:response.y);}
float lightResponse(uint id,uint category) {return responseForCategory(lights[id].response.xy,category);}
vec3 distantRadiance(uint id) {Light l=lights[id];return l.color_power.rgb*l.color_power.w/l.response.w;}
float lightPdf(uint index,vec3 origin,vec3 target,vec3 direction) {
  Light l=lights[index];
  if(finiteDistant(l))return lightPmf(index,origin)/distantSolidAngle(l);
  vec3 n=cross(l.u_soft.xyz,l.v.xyz);
  float area=length(n),cosine=emitterCosine(l,n/max(area,1e-20),direction);
  return cosine>0.?lightPmf(index,origin)*dot(target-origin,target-origin)/(area*cosine):0.;
}
bool nodeHit(LightNode n,vec3 origin,vec3 direction,float tmin,float tmax) {
  vec3 d=direction;
  for(int j=0;j<3;j++)if(abs(d[j])<1e-20)d[j]=1e-20;
  vec3 a=(n.minPower.xyz-vec3(1e-5)-origin)/d;
  vec3 b=(n.maxInfinite.xyz+vec3(1e-5)-origin)/d;
  vec3 lo=min(a,b),hi=max(a,b);
  return max(tmin,max(lo.x,max(lo.y,lo.z)))<=min(tmax,min(hi.x,min(hi.y,hi.z)));
}
// Analytic rectangle emitters participate in path hits and occlusion without
// requiring emissive geometry in the mesh BLAS. A bounded balanced BVH stack.
// Only an NEE shadow segment excludes its sampled emitter. Camera and BSDF rays
// still intersect every rectangle, and other emitters can occlude that segment.
bool traceLights(vec3 origin,vec3 direction,float tmin,inout float closest,out uint emitter,uint excludedEmitter,bool primary) {
  emitter=0u;if(pc.lightCount==0u)return false;
  uint stack[64];uint count=1u;stack[0]=0u;bool found=false;
  while(count>0u) {
    LightNode n=lightNodes[stack[--count]];
    if(!nodeHit(n,origin,direction,tmin,closest))continue;
    if(n.links.z==0xffffffffu) {stack[count++]=n.links.x;stack[count++]=n.links.y;continue;}
    if(n.links.z==excludedEmitter)continue;
    Light l=lights[n.links.z];if(l.position_kind.w!=2.)continue;
    if(primary && (l.bvh.y&2u)==0u)continue;
    vec3 u=l.u_soft.xyz,v=l.v.xyz,normal=cross(u,v);
    float denom=dot(normal,direction);if(abs(denom)<1e-12 || ((l.bvh.y&1u)!=0u && denom>=0.))continue;
    float t=dot(normal,l.position_kind.xyz-origin)/denom;
    if(t<tmin||t>=closest)continue;
    vec3 offset=origin+t*direction-l.position_kind.xyz;
    float uu=dot(u,u),uv=dot(u,v),vv=dot(v,v),det=uu*vv-uv*uv;
    if(det<=1e-20)continue;
    vec2 ab=vec2(dot(offset,u)*vv-dot(offset,v)*uv,dot(offset,v)*uu-dot(offset,u)*uv)/det;
    if(any(greaterThan(abs(ab),vec2(.5))))continue;
    closest=t;emitter=n.links.z;found=true;
  }return found;
}
bool traceLights(vec3 origin,vec3 direction,float tmin,inout float closest,out uint emitter,uint excludedEmitter) {
 return traceLights(origin,direction,tmin,closest,emitter,excludedEmitter,false);
}
bool tracePrimaryLights(vec3 origin,vec3 direction,float tmin,inout float closest,out uint emitter) {
 return traceLights(origin,direction,tmin,closest,emitter,0xffffffffu,true);
}
bool traceLights(vec3 origin,vec3 direction,float tmin,inout float closest,out uint emitter) {
  return traceLights(origin,direction,tmin,closest,emitter,0xffffffffu);
}
vec3 emission(uint id) {return lights[id].color_power.rgb*lights[id].color_power.w;}
#endif
