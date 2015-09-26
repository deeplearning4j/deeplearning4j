extern "C"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "deeplearning4j.h"
//an op for the kernel
__device__ float op(float d1,float *params);

__device__ void transform(int n,int idx,float *dy,int incy,float *params,float *result) {
	int totalThreads = gridDim.x * blockDim.x;
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;
	/* equal, positive, non-unit increments. */
	for (; i < n; i += totalThreads) {
		 result[i * incy] = op(dy[i * incy],params);
    }


}
