#ifndef LAVA_QUEUE_COMPACT
#define LAVA_QUEUE_COMPACT
// One global reservation per subgroup. Inactive paths never enter the next bounce.
void compactRays(PathRay ray,bool alive) {
 uvec4 mask=subgroupBallot(alive);uint n=subgroupBallotBitCount(mask);
 uint outputQueue=pc.rayPhase==0u?0u:1u-pc.queueParity;
 uint base=0u;
 if(subgroupElect()&&n>0u)base=atomicAdd(counts[outputQueue],n);
 base=subgroupBroadcastFirst(base);
 if(alive) {
  uint destination=base+subgroupBallotExclusiveBitCount(mask);
  if(outputQueue==0u)raysA[destination]=ray;else raysB[destination]=ray;
 }
}
#endif
