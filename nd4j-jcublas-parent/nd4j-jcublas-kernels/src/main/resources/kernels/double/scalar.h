extern "C"
#include <stdio.h>
#include <stdlib.h>
#include "deeplearning4j.h"
//scalar and current element
__device__ double op(double d1,double d2,double *params);

__device__ void transform(int n, int idx,double dx,double *dy,int incy,double *params,double *result) {
	int totalThreads = gridDim.x * blockDim.x;
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;

	for (; i < n; i += totalThreads) {
		result[i * incy] = op(dx,dy[i * incy],params);
	}

}


