#ifndef LAVA_QUERY
#define LAVA_QUERY
uint queryPrimitive(rayQueryEXT q){return rayQueryGetIntersectionInstanceCustomIndexEXT(q,true)+rayQueryGetIntersectionPrimitiveIndexEXT(q,true);}
vec2 queryCoordinates(rayQueryEXT q){return rayQueryGetIntersectionTypeEXT(q,true)==gl_RayQueryCommittedIntersectionGeneratedEXT?vec2(rayQueryGetIntersectionTEXT(q,true),0):rayQueryGetIntersectionBarycentricsEXT(q,true);}
void confirmCandidate(rayQueryEXT q,float tmin,float tmax,bool primary){
 uint kind=rayQueryGetIntersectionTypeEXT(q,false);
 uint id=rayQueryGetIntersectionInstanceCustomIndexEXT(q,false)+rayQueryGetIntersectionPrimitiveIndexEXT(q,false);
 vec3 origin=rayQueryGetWorldRayOriginEXT(q),direction=rayQueryGetWorldRayDirectionEXT(q);
 if(kind==gl_RayQueryCandidateIntersectionTriangleEXT){
  if((!primary||triangles[id].uv2.z==0.)&&materialCovered(id,origin,direction))rayQueryConfirmIntersectionEXT(q);
 }else if(kind==gl_RayQueryCandidateIntersectionAABBEXT){
  if(rayQueryGetIntersectionTypeEXT(q,true)!=gl_RayQueryCommittedIntersectionNoneEXT)tmax=min(tmax,rayQueryGetIntersectionTEXT(q,true));
  float distance;if(curveIntersect(id,origin,direction,tmin,tmax,distance)&&materialCovered(id,origin,direction))rayQueryGenerateIntersectionEXT(q,distance);
 }
}
#endif
