// Stochastic cutout coverage shared by camera, secondary and shadow traversal.
bool materialCovered(uint primitive,vec3 origin,vec3 direction) {
 float coverage=materialCoverage(primitive,origin,direction);if(coverage>=1.)return true;if(coverage<=0.)return false;
 uvec3 a=floatBitsToUint(origin),b=floatBitsToUint(direction);
 uint seed=hash(primitive^pc.sampleIndex^hash(a.x)^hash(a.y+17u)^hash(a.z+31u)^hash(b.x+47u)^hash(b.y+73u)^hash(b.z+97u));
 return random(seed)<coverage;
}
