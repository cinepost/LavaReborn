#ifndef LAVA_MATERIAL_SURFACE
#define LAVA_MATERIAL_SURFACE
#include "record.glsl"
#include "luts.glsl"
#include "openpbr.h"
#include "inputs.glsl"
// Stable surface product API: consumers never inspect OpenPBR's internal lobes.
struct MaterialSurface { OpenPBR_PreparedBsdf bsdf; vec3 normal; vec3 geometricNormal; vec3 view; uvec3 depth; bool opaque; };
MaterialSurface materialPrepare(uint id,vec3 normal,vec3 geometricNormal,vec3 view,vec3 throughput,uvec3 depth,vec2 uv,vec3 tangent,vec3 bitangent,float footprint) {
 MaterialSurface s;
 vec3 originalNormal=normal;
 vec3 projected=tangent-normal*dot(normal,tangent);
 vec3 t=length(projected)>1e-10?normalize(projected):normalize(cross(abs(normal.z)<.999?vec3(0,0,1):vec3(0,1,0),normal));
 float handedness=dot(cross(normal,t),bitangent)<0.?-1.:1.;
 vec3 b=cross(normal,t)*handedness;
 TextureSlot ns=textureSlots[id*40u+38u];
 if(ns.info.x!=0u&&textureAvailable(ns.handle,uv*ns.uv.xy+ns.uv.zw)) {
  vec3 mapped=materialValue(id,38u,uv,footprint).xyz;
  if(ns.parameters.z==0.)mapped=2.*mapped-1.;
  mapped.xy*=ns.parameters.y;
  if(dot(mapped,mapped)>1e-12)normal=normalize(t*mapped.x+b*mapped.y+normal*mapped.z);
 }
 TextureSlot bs=textureSlots[id*40u+39u];
 if(bs.info.x!=0u&&textureAvailable(bs.handle,uv*bs.uv.xy+bs.uv.zw)) {
  vec2 size=textureSizeAt(bs.handle,uv*bs.uv.xy+bs.uv.zw);
  vec2 stepUv=max(vec2(max(footprint,1e-6)),1./max(size*abs(bs.uv.xy),vec2(1)));
  float hu=(materialValue(id,39u,uv+vec2(stepUv.x,0),footprint).x-materialValue(id,39u,uv-vec2(stepUv.x,0),footprint).x)/(2.*stepUv.x);
  float hv=(materialValue(id,39u,uv+vec2(0,stepUv.y),footprint).x-materialValue(id,39u,uv-vec2(0,stepUv.y),footprint).x)/(2.*stepUv.y);
  float area=dot(tangent,cross(bitangent,normal));
  if(abs(area)>1e-12)normal=normalize(normal-bs.parameters.y*(hu*cross(bitangent,normal)+hv*cross(normal,tangent))/area);
 }
 // Apply geometric/view constraints once, after all maps. A hard fallback to
 // originalNormal here created popping patches while orbiting and discarded
 // every back-facing map. OpenPBR handles the back face and IOR swap itself.
 if(any(isnan(normal))||any(isinf(normal)))normal=originalNormal;
 normal=viewSafeNormal(normal,geometricNormal,view);
 tangent=t-normal*dot(normal,t);
 if(dot(tangent,tangent)<1e-12)tangent=openpbr_make_basis(normal).t;
 OpenPBR_ResolvedInputs inputs=materialInputs(id,normal,uv,footprint);
 inputs.geometry_basis=openpbr_make_basis(normal,tangent,handedness);
 inputs.geometry_coat_basis=inputs.geometry_basis;
 s.opaque=inputs.transmission_weight<=0.;
 s.bsdf=openpbr_prepare(inputs,throughput,OpenPBR_BaseRgbWavelengths_nm,OpenPBR_VacuumIor,view);
 s.normal=normal;s.geometricNormal=geometricNormal;s.view=view;s.depth=depth;return s;
}
bool materialTransmitted(MaterialSurface s,vec3 wi) {return dot(s.normal,s.view)*dot(s.normal,wi)<0.;}
bool materialValidDirection(MaterialSurface s,vec3 wi) {
 return materialTransmitted(s,wi)==(dot(s.geometricNormal,s.view)*dot(s.geometricNormal,wi)<0.);
}
OpenPBR_DiffuseSpecular materialLobes(MaterialSurface s,vec3 wi) {
 OpenPBR_DiffuseSpecular e=openpbr_eval(s.bsdf,wi);
 if(!materialValidDirection(s,wi)) {e.diffuse=vec3(0);e.specular=vec3(0);return e;}
 if(materialTransmitted(s,wi)) {
  if(s.depth.z>=pc.refractionDepth) {e.diffuse=vec3(0);e.specular=vec3(0);}
 } else {
  if(s.depth.x>=pc.diffuseDepth)e.diffuse=vec3(0);
  if(s.depth.y>=pc.reflectionDepth)e.specular=vec3(0);
 }
 return e;
}
vec3 materialEvaluate(MaterialSurface s,vec3 wi) {OpenPBR_DiffuseSpecular e=materialLobes(s,wi);return e.diffuse+e.specular;}
float materialPdf(MaterialSurface s,vec3 wi) {return openpbr_pdf(s.bsdf,wi);}
vec3 materialEmission(MaterialSurface s) {return s.bsdf.emission;}
float materialEnergy(vec3 x) {return max(0.,dot(x,vec3(.2126,.7152,.0722)));}
bool materialSample(MaterialSurface s,inout uint seed,out vec3 wi,out vec3 weight,out float pdf,out bool delta,out uint category) {
 wi=vec3(0);weight=vec3(0);pdf=0.;delta=true;category=0u;
 OpenPBR_DiffuseSpecular w; uint flags;
 vec3 xi=vec3(pathSample(seed),pathSample(seed),pathSample(seed));
 openpbr_sample(s.bsdf,xi,wi,w,pdf,flags);
 if(!(pdf>0.)||!materialValidDirection(s,wi))return false;
 delta=(flags&OpenPBR_BsdfLobeTypeSpecular)!=0u;
 bool transmitted=materialTransmitted(s,wi);
 if(delta) {
  category=transmitted?2u:1u;
  if((category==2u&&s.depth.z>=pc.refractionDepth)||(category==1u&&s.depth.y>=pc.reflectionDepth))return false;
  weight=w.diffuse+w.specular;return true;
 }
 // The reference samples its full mixture. Re-evaluate permitted lobes, retaining
 // that same proposal PDF for reciprocal MIS, then allocate the depth class.
 OpenPBR_DiffuseSpecular e=materialLobes(s,wi);
 if(transmitted) {category=2u;weight=(e.diffuse+e.specular)/pdf;return materialEnergy(weight)>0.;}
 float d=materialEnergy(e.diffuse),r=materialEnergy(e.specular),total=d+r;
 if(total<=0.)return false;
 float probability=d/total;
 if(pathSample(seed)<probability) {category=0u;weight=e.diffuse/(pdf*probability);}
 else {category=1u;weight=e.specular/(pdf*(1.-probability));}
 return true;
}
#endif
