#ifndef LAVA_SAMPLING
#define LAVA_SAMPLING
// Randomized low-discrepancy samples. Raster samples are frame-coherent;
// ray-query camera samples have independent pixel scrambles.
layout(std430,set=0,binding=17) readonly buffer SamplingPrimes {uint samplePrimes[];};
uint samplingPixel=0u;
float radicalInverse(uint index,uint base) {
 float inverse=1./float(base),weight=inverse,value=0.;
 while(index>0u){value+=float(index%base)*weight;index/=base;weight*=inverse;}
 return min(value,0.99999994);
}
float sequenceSample(uint dimension,uint scramble) {
 float value;
 if(pc.samplingMode==0u)value=radicalInverse(pc.sampleIndex+1u,samplePrimes[dimension]);
 else if(dimension==0u)value=(float(pc.sampleIndex)+.5)/float(max(pc.targetSamples,1u));
 else value=radicalInverse(pc.sampleIndex,samplePrimes[dimension-1u]);
 return fract(value+float(hash(scramble^hash(dimension+0x51633e2du))>>8)*(1./16777216.));
}
// Nested affine digit permutations are bijective in every prime base. Their
// pixel/dimension-specific trees avoid reusing camera coordinates for BSDF/NEE
// samples and decorrelate higher-dimensional projections of Halton/Hammersley.
float scrambledRadicalInverse(uint index,uint base,uint scramble) {
 float inverse=1./float(base),weight=inverse,value=0.;uint prefix=0u;
 while(weight>1e-8) {
  uint digit=index%base,node=hash(scramble^prefix);
  uint multiplier=1u+node%(base-1u),offset=hash(node+0x9e3779b9u)%base;
  value+=float((multiplier*digit+offset)%base)*weight;
  prefix=hash(prefix^digit^0x85ebca6bu);index/=base;weight*=inverse;
 }
 return min(value,0.99999994);
}
float pathSample(inout uint dimension) {
 uint d=dimension++,key=hash(samplingPixel)^hash(d+0x51633e2du);
 uint base=samplePrimes[pc.samplingMode==0u?d:d-1u];
 return scrambledRadicalInverse(pc.sampleIndex+(pc.samplingMode==0u?1u:0u),base,key);
}
vec2 filmSample(){return vec2(sequenceSample(0u,0x71a9u),sequenceSample(1u,0x71a9u));}
vec2 filmSample(uvec2 p){
 uint key=camera.lens.z>0.?hash(p.y*pc.width+p.x)^0x71a9u:0x71a9u;
 return vec2(sequenceSample(0u,key),sequenceSample(1u,key));
}
vec2 lensSample(uvec2 p){
 uint key=camera.lens.z>0.?hash(p.y*pc.width+p.x)^0x71a9u:0x71a9u;
 return vec2(sequenceSample(2u,key),sequenceSample(3u,key));
}
vec2 lensSample(){return vec2(sequenceSample(2u,0x71a9u),sequenceSample(3u,0x71a9u));}
#endif
