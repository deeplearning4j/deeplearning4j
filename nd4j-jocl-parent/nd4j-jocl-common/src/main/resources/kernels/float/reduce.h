extern "C"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "deeplearning4j.h"

//referenced: https://github.com/ArchaeaSoftware/cudahandbook/blob/master/reduction/reduction6AnyBlockSize.cuh

//an op for the kernel
__global  float op(float d1,float *extraParams);

//calculate an update of the reduce operation
__global  float update(float old,float opOutput,float *extraParams);
//invoked when combining two kernels
__global  float merge(float f1, float f2,float *extraParams);

//post process result (for things like means etc)
__global  float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result);

/**

Perform a reduction
@param n the number of elements
@param xOffset the starting offset
@param dx the data to perform the reduction on
@param incx the increment on which to perform the reduction
@param extraParams extra parameters used for calculations
@param result where to store the result of the reduction
 */
__global  void transform(int n, int xOffset,float *dx,int incx,float *extraParams,float *result) {
	extern _local float sPartials[];
	int tid = get_local_id(0);
	int totalThreads = get_num_groups(0) * get_local_size(0);
	int start = get_local_size(0) * get_group_id(0) + tid;

	float sum = extraParams[0];

    for ( int i = start; i < n; i += totalThreads) {
          float curr = dx[i * incx];
		  sum = update(sum,op(curr,extraParams),extraParams);
    }

	sPartials[tid] = sum;
	barrier(0);

	// start the shared memory loop on the next power of 2 less
	// than the block size.  If block size is not a power of 2,
	// accumulate the intermediate sums in the remainder range.
	int floorPow2 = get_local_size(0);

	if ( floorPow2 & (floorPow2 - 1) ) {
		while ( floorPow2 & (floorPow2 - 1) ) {
			floorPow2 &= floorPow2 - 1;
		}
		if ( tid >= floorPow2 ) {
			sPartials[tid - floorPow2] = merge(sPartials[tid - floorPow2],sPartials[tid],extraParams);
		}
		barrier(0);
	}

	for ( int activeThreads = floorPow2 >> 1;activeThreads;	activeThreads >>= 1 ) {
		if ( tid < activeThreads ) {
			sPartials[tid] = merge(sPartials[tid],sPartials[tid + activeThreads],extraParams);
		}
		barrier(0);
	}

	if ( tid == 0 ) {
		result[get_group_id(0)] = postProcess(sPartials[0],n,xOffset,dx,incx,extraParams,result);
	}

}
