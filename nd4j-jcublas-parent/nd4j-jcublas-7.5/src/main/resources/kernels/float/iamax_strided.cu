extern "C"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
__device__ float merge(float old,float opOutput,float *extraParams) {
      return fmaxf(old,opOutput);
 }


__device__ float update(float old,float opOutput,float *extraParams) {
      return fmaxf(old,opOutput);
 }


__device__ float op(float d1,float *extraParams) {
      return d1;
}


__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
             return reduction;
}



/**

Perform a reduction
@param n the number of elements
@param xOffset the starting offset
@param dx the data to perform the reduction on
@param incx the increment on which to perform the reduction
@param extraParams extra parameters used for calculations
@param result where to store the result of the reduction
 */
__device__ void transform(int n, int xOffset,float *dx,int incx,float *extraParams,float *result) {
	extern __shared__ float sPartials[];
	extern __shared__ int indexes[];
	int tid = threadIdx.x;
	int totalThreads = gridDim.x * blockDim.x;
	int start = blockDim.x * blockIdx.x + tid;

	float sum = extraParams[0];
    int index = start;
    for ( int i = start; i < n; i += totalThreads) {
          float curr = dx[i * incx];
          if(curr > sum) {
               index = i * incx;
          }
		  sum = update(sum,op(curr,extraParams),extraParams);

    }

	sPartials[tid] = sum;
	indexes[tid] = index;

	__syncthreads();

	// start the shared memory loop on the next power of 2 less
	// than the block size.  If block size is not a power of 2,
	// accumulate the intermediate sums in the remainder range.
	int floorPow2 = blockDim.x;

	if (floorPow2 & (floorPow2 - 1)) {
		while ( floorPow2 & (floorPow2 - 1) ) {
			floorPow2 &= floorPow2 - 1;
		}
		if (tid >= floorPow2) {
			float sPartialBack = sPartials[tid - floorPow2];
			float currTid = sPartials[tid];
	    	if(sPartialBack > currTid) {
             	indexes[tid - floorPow2] = indexes[tid];
             }

		  sPartials[tid - floorPow2] = merge(sPartialBack,currTid,extraParams);

		}
		__syncthreads();
	}

	for (int activeThreads = floorPow2 >> 1;activeThreads;	activeThreads >>= 1) {
		if (tid < activeThreads) {
		    if(sPartials[tid] > sPartials[tid + activeThreads]) {
		        indexes[tid] = indexes[tid + activeThreads];
		    }
			sPartials[tid] = merge(sPartials[tid],sPartials[tid + activeThreads],extraParams);

		}
		__syncthreads();
	}

	if ( tid == 0 ) {
		result[blockIdx.x] = postProcess(sPartials[0],n,xOffset,dx,incx,extraParams,result);
		//stride is compounded
		result[blockIdx.x] = indexes[0] / incx;
	}

}

extern "C"
__global__ void iamax_strided_float(int n, int xOffset,float *dx,int incx,float *extraParams,float *result) {
             transform(n,xOffset,dx,incx,extraParams,result);
}


