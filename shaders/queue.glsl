#ifndef LAVA_QUEUE
#define LAVA_QUEUE
struct PathRay { vec4 originPdf; vec4 directionDelta; vec4 throughput; uvec4 state; };
layout(std430,set=0,binding=10) buffer RayQueueA { PathRay raysA[]; };
layout(std430,set=0,binding=11) buffer RayQueueB { PathRay raysB[]; };
layout(std430,set=0,binding=12) buffer QueueControl { uint counts[4]; uvec4 dispatchArgs; } queueControl;
#define counts queueControl.counts
#define dispatchArgs queueControl.dispatchArgs
#endif
