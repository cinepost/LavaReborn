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
bool isCurve(Triangle t){return t.normalB.w!=0.;}
// Closed tapered cone with round endpoint hemispheres. Analytic tests run for
// procedural AABB candidates in the existing ray-query loops, including shadows.
bool curveIntersect(uint id,vec3 origin,vec3 direction,float tmin,float tmax,out float closest) {
 Triangle c=triangles[id];vec3 ab=c.b.xyz-c.a.xyz;float lengthAB=length(ab);
 // Recenter along the ray before solving: tiny hair radii must not disappear
 // when subtracting two large, almost equal quadratic discriminant terms.
 float shift=dot((c.a.xyz+c.b.xyz)*.5-origin,direction);
 origin+=direction*shift;tmin-=shift;tmax-=shift;
 vec3 axis=lengthAB>1e-10?ab/lengthAB:vec3(0,0,1);
 vec3 o=origin-c.a.xyz;float os=dot(o,axis),ds=dot(direction,axis);
 vec3 op=o-axis*os,dp=direction-axis*ds;closest=tmax;bool hit=false;
 if(lengthAB>1e-10){
  float slope=(c.b.w-c.a.w)/lengthAB,r=c.a.w+slope*os;
  float a=dot(dp,dp)-slope*slope*ds*ds,b=2.*(dot(op,dp)-r*slope*ds),cc=dot(op,op)-r*r;
  float disc=b*b-4.*a*cc;
  if(disc>=0.){
   vec2 roots;
   if(abs(a)<1e-10){float root=abs(b)>1e-20?-cc/b:tmax;roots=vec2(root);}
   else {float q=-.5*(b+(b<0.?-1.:1.)*sqrt(disc));roots=abs(q)>1e-20?vec2(q/a,cc/q):vec2(-b/(2.*a));}
   for(int i=0;i<2;i++){float d=roots[i],s=os+d*ds;if(d>=tmin&&d<closest&&s>=0.&&s<=lengthAB){closest=d;hit=true;}}
  }
 }
 for(int cap=0;cap<2;cap++){
  vec3 center=cap==0?c.a.xyz:c.b.xyz;float radius=cap==0?c.a.w:c.b.w;
  if(lengthAB<=1e-10)radius=max(c.a.w,c.b.w);
  vec3 co=origin-center;float b=dot(co,direction),disc=b*b-dot(co,co)+radius*radius;
  if(radius<=0.||disc<0.)continue;
  vec2 roots=vec2(-b-sqrt(disc),-b+sqrt(disc));
  for(int i=0;i<2;i++){float d=roots[i],s=os+d*ds;bool end=lengthAB<=1e-10||(cap==0?s<=0.:s>=lengthAB);if(end&&d>=tmin&&d<closest){closest=d;hit=true;}}
 }
 closest+=shift;return hit;
}
void curveFrame(Triangle c,vec3 p,out vec3 normal,out vec3 tangent,out vec2 uv){
 vec3 ab=c.b.xyz-c.a.xyz;float l=length(ab);tangent=l>1e-10?ab/l:vec3(0,0,1);
 float s=dot(p-c.a.xyz,tangent),u=l>1e-10?clamp(s/l,0.,1.):0.;
 vec3 radial=p-mix(c.a.xyz,c.b.xyz,u);
 if(s>0.&&s<l)radial-=tangent*(mix(c.a.w,c.b.w,u)*(c.b.w-c.a.w)/l);
 normal=dot(radial,radial)>1e-20?normalize(radial):vec3(1,0,0);
 vec3 x=normalize(cross(abs(tangent.z)<.999?vec3(0,0,1):vec3(0,1,0),tangent));
 vec3 y=cross(tangent,x);uv=vec2(mix(c.uv01.x,c.uv01.z,u),atan(dot(normal,y),dot(normal,x))*.1591549431+.5);
}
Triangle hitTriangle(uint id,inout vec2 bary,vec3 origin,vec3 direction){
 Triangle t=triangles[id];if(!isCurve(t))return t;
 vec3 p=origin+direction*bary.x,n,axis;vec2 uv;curveFrame(t,p,n,axis,uv);
 float l=max(length(t.b.xyz-t.a.xyz),1e-6)/max(t.uv01.z-t.uv01.x,1e-6),radius=max(max(t.a.w,t.b.w),1e-6);
 vec3 tangent=axis-n*dot(axis,n);if(dot(tangent,tangent)<1e-12)tangent=cross(abs(n.z)<.999?vec3(0,0,1):vec3(0,1,0),n);
 tangent=normalize(tangent);vec3 bitangent=cross(n,tangent);
 t.a=vec4(p,0);t.b=vec4(p+tangent*l,0);t.c=vec4(p+bitangent*6.2831853*radius,0);
 t.normalA=vec4(n,1);t.normalB=vec4(n,1);t.normalC=vec4(n,0);
 t.uv01=vec4(uv,uv+vec2(1,0));t.uv2=vec4(uv+vec2(0,1),0,0);bary=vec2(0);return t;
}
vec3 geometricNormal(Triangle t) {if(isCurve(t))return t.normalA.xyz;return normalize(cross(t.b.xyz-t.a.xyz,t.c.xyz-t.a.xyz))*t.normalA.w;}
vec3 shadingNormal(Triangle t,vec2 bary) {
 vec3 ng=geometricNormal(t);
 // Zero/absent corner normals fall back locally; cancellation falls back as well.
 vec3 a=dot(t.normalA.xyz,t.normalA.xyz)>1e-20?t.normalA.xyz:ng;
 vec3 b=dot(t.normalB.xyz,t.normalB.xyz)>1e-20?t.normalB.xyz:ng;
 vec3 c=dot(t.normalC.xyz,t.normalC.xyz)>1e-20?t.normalC.xyz:ng;
 vec3 ns=a*(1.-bary.x-bary.y)+b*bary.x+c*bary.y;
 if(dot(ns,ns)<1e-20)return ng;
 ns=normalize(ns);if(dot(ns,ng)<0.)ns=-ns;
 return ns;
}
// Hanika, Ray Tracing Gems II ch.4: interpolate projections onto the corner
// tangent planes. Only convex, varying authored normals may move the point.
// Constant tilted normals, missing normals and concave patches remain flat.
vec3 smoothLightingOffset(Triangle t,vec2 bary) {
 if(isCurve(t))return vec3(0);
 vec3 g=geometricNormal(t);
 vec3 a=dot(t.normalA.xyz,t.normalA.xyz)>1e-20?normalize(t.normalA.xyz):g;
 vec3 b=dot(t.normalB.xyz,t.normalB.xyz)>1e-20?normalize(t.normalB.xyz):g;
 vec3 c=dot(t.normalC.xyz,t.normalC.xyz)>1e-20?normalize(t.normalC.xyz):g;
 if(dot(a,g)<0.)a=-a;if(dot(b,g)<0.)b=-b;if(dot(c,g)<0.)c=-c;
 if(max(dot(a-b,a-b),max(dot(a-c,a-c),dot(b-c,b-c)))<1e-8)return vec3(0);
 vec3 w=vec3(1.-bary.x-bary.y,bary);
 vec3 p=t.a.xyz*w.x+t.b.xyz*w.y+t.c.xyz*w.z;
 vec3 offset=-w.x*min(0.,dot(p-t.a.xyz,a))*a
             -w.y*min(0.,dot(p-t.b.xyz,b))*b
             -w.z*min(0.,dot(p-t.c.xyz,c))*c;
 return dot(offset,g)>0.?offset:vec3(0);
}
// Constrain the final normal continuously instead of switching the whole frame
// back to Ng when the camera crosses its tangent plane. Keep outward orientation:
// OpenPBR uses that sign for back faces and IORs. Ordinary valid normals are intact.
vec3 viewSafeNormal(vec3 normal,vec3 ng,vec3 view) {
 float side=dot(ng,view)<0.?-1.:1.;
 vec3 g=ng*side,n=normal*side;
 float z=dot(n,g);
 if(z<1e-5)n=normalize(n+g*(1e-5-z));
 float nv=dot(n,view),limit=max(1e-4,1e-3*abs(dot(view,ng)));
 if(nv<limit)n=normalize(n+view*(limit-nv));
 return n*side;
}
#endif
