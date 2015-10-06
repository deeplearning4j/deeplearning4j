extern "C"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
__device__ double merge(double old,double opOutput,double *extraParams) {
      return fmaxf(old,opOutput);
 }


__device__ double update(double old,double opOutput,double *extraParams) {
      return fmaxf(old,opOutput);
 }


__device__ double op(double d1,double *extraParams) {
      return d1;
}


__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
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
__device__ void transform(int n, int xOffset,double *dx,int incx,double *extraParams,double *result) {
	extern __shared__ double sPartials[];
	extern __shared__ int indexes[];
	int tid = threadIdx.x;
	int totalThreads = gridDim.x * blockDim.x;
	int start = blockDim.x * blockIdx.x + tid;

	double sum = extraParams[0];
    int index = start;
    for ( int i = start; i < n; i += totalThreads) {
          double curr = dx[i * incx];
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
			double sPartialBack = sPartials[tid - floorPow2];
			double currTid = sPartials[tid];
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
		double val5 = indexes[0];
		result[blockIdx.x] = indexes[0];
	}

}

extern "C"
__global__ void iamax_strided_double(int n, int xOffset,double *dx,int incx,double *extraParams,double *result) {
             transform(n,xOffset,dx,incx,extraParams,result);
}


