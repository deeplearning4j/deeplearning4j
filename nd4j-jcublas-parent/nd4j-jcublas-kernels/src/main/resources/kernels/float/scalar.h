extern "C"
#include <stdio.h>
#include <stdlib.h>
#include "deeplearning4j.h"
//scalar and current element
__device__ float op(float d1,float d2,float *params);

__device__ void transform(int n, int idx,float dx,float *dy,int incy,float *params,float *result) {

	int totalThreads = gridDim.x * blockDim.x;
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;

	for (; i < n; i += totalThreads) {
		result[i * incy] = op(dx,dy[i * incy],params);
	}



}


