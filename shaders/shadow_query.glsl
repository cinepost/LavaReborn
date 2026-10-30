#ifndef LAVA_SHADOW_QUERY
#define LAVA_SHADOW_QUERY
bool traceShadow(vec3 origin,vec3 direction,float distance,uint sampledLight) {
  rayQueryEXT q;
  rayQueryInitializeEXT(q,world,gl_RayFlagsTerminateOnFirstHitEXT,255,origin,1e-5,direction,distance);
  while(rayQueryProceedEXT(q)){
   confirmCandidate(q,1e-5,distance,false);
  }
  if(rayQueryGetIntersectionTypeEXT(q,true)!=gl_RayQueryCommittedIntersectionNoneEXT)return false;
  uint emitter;return !traceLights(origin,direction,1e-5,distance,emitter,sampledLight);
}
#endif
