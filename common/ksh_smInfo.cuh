#include <stdio.h>
#include <stdint.h>
#include <time.h>

static __device__ __inline__ uint32_t __mysmid () {
  uint32_t smid;
  asm volatile  ( "mov.u32 %0, %%smid;" 
                  : "=r"(smid) );
  return smid;
}

static __device__ __inline__ uint32_t __mywarpid () {
  uint32_t warpid;
  asm volatile (  "mov.u32 %0, %%warpid;" 
                  : "=r"(warpid) );
  return warpid;
}

static __device__ __inline__ uint32_t __mylaneid () {
  uint32_t laneid;
  asm volatile (  "mov.u32 %0, %%laneid;" 
                  : "=r"(laneid) );
  return laneid;
}

 struct sminfo {
	uint32_t smid;
	int threadx;
	int thready;
	int blockx;
	int blocky;
	//int index;
	clock_t start;
	clock_t end;
	int streamid;
};