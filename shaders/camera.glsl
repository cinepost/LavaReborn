#ifndef LAVA_CAMERA
#define LAVA_CAMERA
#include "sampling.glsl"
vec3 cameraLens(vec2 xi) {
 if(camera.lens.x<=0.)return vec3(0);
 float radius=sqrt(xi.x)*camera.optics.z/(2.*camera.lens.x);
 float angle=6.28318530718*xi.y;
 return vec3(cos(angle)*radius,sin(angle)*radius,0);
}
vec3 cameraLens(){return cameraLens(lensSample());}
// Shared by ray-query visibility, raster resolve and path initialization.
void cameraRay(uvec2 p,out vec3 origin,out vec3 direction,out float planeScale) {
#ifdef LAVA_ADAPTIVE
 samplingPixel=p.y*pc.width+p.x;
#endif
 vec2 uv=(vec2(p)+filmSample(p))/vec2(pc.width,pc.height);
 vec2 film=(uv*2.-1.)*vec2(1.,-1.)*camera.optics.xy*.5+camera.clippingOffset.zw;
 vec3 localOrigin=cameraLens(lensSample(p)),localDirection;
 if(camera.optics.w>0.){localOrigin+=vec3(film,0);localDirection=normalize(vec3(film,-camera.lens.y)-localOrigin);}
 else {
  vec3 focus=vec3(film*camera.lens.y/camera.optics.z,-camera.lens.y);
  localDirection=normalize(focus-localOrigin);
 }
 origin=(camera.cameraToWorld*vec4(localOrigin,1)).xyz;
 direction=normalize(mat3(camera.cameraToWorld)*localDirection);
 planeScale=1./max(1e-6,-localDirection.z);
}
// Vulkan clip space: +Y down, depth [0,1]. Lens displacement is accompanied
// by a focus-plane shear; subpixel film jitter is a projection-center shift.
mat4 cameraProjection() {
 vec2 jitter=2.*(filmSample()-.5)/vec2(pc.width,pc.height);
 float near=camera.clippingOffset.x,far=camera.clippingOffset.y;
 if(camera.optics.w>0.) {
  return mat4(vec4(2./camera.optics.x,0,0,0),vec4(0,-2./camera.optics.y,0,0),
    vec4(-2.*cameraLens().x/(camera.lens.y*camera.optics.x),2.*cameraLens().y/(camera.lens.y*camera.optics.y),-1./(far-near),0),
    vec4(-2.*camera.clippingOffset.z/camera.optics.x-jitter.x,
          2.*camera.clippingOffset.w/camera.optics.y-jitter.y,-near/(far-near),1));
 }
 vec2 lens=cameraLens().xy;
 vec2 center=vec2(2.,-2.)*(camera.optics.z*lens/camera.lens.y-camera.clippingOffset.zw)/camera.optics.xy-jitter;
 return mat4(vec4(2.*camera.optics.z/camera.optics.x,0,0,0),
   vec4(0,-2.*camera.optics.z/camera.optics.y,0,0),
   vec4(-center,far/(near-far),-1),vec4(0,0,far*near/(near-far),0));
}
#endif
