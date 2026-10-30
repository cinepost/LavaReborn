#ifndef LAVA_QUEUE_REORDER
#define LAVA_QUEUE_REORDER
// Primary visibility supplies a material/geometry coherence hint, not a predicted next hit.
uint coherenceKey(PathRay ray) {
 uvec3 signs=uvec3(greaterThanEqual(ray.directionDelta.xyz,vec3(0)));
 uint octant=signs.x|(signs.y<<1)|(signs.z<<2);
 Hit source=hits[ray.state.x];
 uint sourceClass=(source.attributes.x&127u)|((source.primitive-1u)>=pc.meshTriangles?128u:0u);
 return (ray.state.w<<11)|(sourceClass<<3)|octant;
}
bool keyLess(uvec2 a,uvec2 b) {return a.x<b.x||(a.x==b.x&&a.y<b.y);}
#endif
