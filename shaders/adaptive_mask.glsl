#ifndef LAVA_ADAPTIVE_MASK
#define LAVA_ADAPTIVE_MASK
// Dilate the variance mask by one pixel. A constant prefix in an isolated pixel
// is not reliable evidence of convergence beside noisy fur/partial shadows.
bool adaptiveBaseActive(uint i) {
 return pixelStats[i].count<pc.targetSamples&&pixelStats[i].stable<3u;
}
bool adaptiveEligible(uint i) {
 if(pixelStats[i].count>=pc.targetSamples)return false;
 if(adaptiveBaseActive(i))return true;
 ivec2 p=ivec2(i%pc.width,i/pc.width);
 for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
  ivec2 q=p+ivec2(x,y);
  if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,ivec2(pc.width,pc.height))))continue;
  if(adaptiveBaseActive(uint(q.y)*pc.width+uint(q.x)))return true;
 }
 return false;
}
#endif
