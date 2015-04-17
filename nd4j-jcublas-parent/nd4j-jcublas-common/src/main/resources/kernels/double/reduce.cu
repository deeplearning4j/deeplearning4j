extern "C"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "deeplearning4j.h"

//referenced: https://github.com/ArchaeaSoftware/cudahandbook/blob/master/reduction/reduction6AnyBlockSize.cuh

//an op for the kernel
__device__ double op(double d1,double *extraParams);

//calculate an update of the reduce operation
__device__ double update(double old,double opOutput,double *extraParams);
//invoked when combining two kernels
__device__ double merge(double f1, double f2,double *extraParams);

//post process result (for things like means etc)
__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result);

/**

Perform a reduction
@param n the number of elements
@param xOffset the starting offset
@param dx the data to perform the reduction on
@param incx the increment on which to perform the reduction
@param extraParams extra parameters used for calculations
@param result where to store the result of the reduction
 */
__device__ void transform(int n, int xOffset,double *dx,int incx,double *extraParams,double *result) {
	extern __shared__ double sPartials[];
	int tid = threadIdx.x;
	int totalThreads = gridDim.x * blockDim.x;
	int start = blockDim.x * blockIdx.x + tid;

	double sum = extraParams[0];

    for ( int i = start; i < n; i += totalThreads) {
             double curr = dx[i * incx];
		     sum = update(sum,op(curr,extraParams),extraParams);
	}

	sPartials[tid] = sum;
	__syncthreads();

	// start the shared memory loop on the next power of 2 less
	// than the block size.  If block size is not a power of 2,
	// accumulate the intermediate sums in the remainder range.
	int floorPow2 = blockDim.x;

	if ( floorPow2 & (floorPow2 - 1) ) {
		while ( floorPow2 & (floorPow2 - 1) ) {
			floorPow2 &= floorPow2 - 1;
		}
		if ( tid >= floorPow2 ) {
			sPartials[tid - floorPow2] = merge(sPartials[tid - floorPow2],sPartials[tid],extraParams);
		}
		__syncthreads();
	}

	for ( int activeThreads = floorPow2 >> 1;	activeThreads;	activeThreads >>= 1 ) {
		if ( tid < activeThreads ) {
			sPartials[tid] = merge(sPartials[tid],sPartials[tid + activeThreads],extraParams);
		}
		__syncthreads();
	}

	if ( tid == 0 ) {
		result[blockIdx.x] = postProcess(sPartials[0],n,xOffset,dx,incx,extraParams,result);
	}

}
