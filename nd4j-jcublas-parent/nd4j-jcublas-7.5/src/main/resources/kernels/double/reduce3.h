extern "C"
#include <math.h>

#define SHARED_MEMORY_LENGTH  128


//an op for the kernel
__device__ double op(double d1,double d2,double *extraParams);

//calculate an update of the reduce operation
__device__ double update(double old,double opOutput,double *extraParams);


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
__device__ void transform_pair(int n, int xOffset,int yOffset,double *dx,double *dy,int incx,int incy,double *extraParams,double *result) {
        extern __shared__ double sPartials[];
        const int tid = threadIdx.x;
        int totalThreads = gridDim.x * blockDim.x;
        int start = blockDim.x * blockIdx.x + tid;

        double sum = result[0];
        for ( size_t i = start; i < n; i += totalThreads) {
             sum = update(sum,op(dx[i * incx],dy[i * incy],extraParams),extraParams);
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
                sPartials[tid - floorPow2] = update(sPartials[tid - floorPow2],sPartials[tid - floorPow2],extraParams);
            }
            __syncthreads();
        }

        for ( int activeThreads = floorPow2 >> 1;
                  activeThreads;
                  activeThreads >>= 1 ) {
            if ( tid < activeThreads ) {
                sPartials[tid] = update(sPartials[tid],sPartials[tid + activeThreads],extraParams);
            }
            __syncthreads();
        }

        if ( tid == 0 ) {
            result[blockIdx.x] = postProcess(sPartials[0],n,xOffset,dx,incx,extraParams,result);
        }

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
        const int tid = threadIdx.x;
        int totalThreads = gridDim.x * blockDim.x;
        int start = blockDim.x * blockIdx.x + tid;

        double sum = result[0];
        for ( size_t i = start; i < n; i += totalThreads) {
             sum = update(sum,op(dx[i * incx],sum,extraParams),extraParams);
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
                sPartials[tid - floorPow2] = update(sPartials[tid - floorPow2],op(sPartials[tid - floorPow2],sPartials[tid],extraParams),extraParams);
            }
            __syncthreads();
        }

        for ( int activeThreads = floorPow2 >> 1;
                  activeThreads;
                  activeThreads >>= 1 ) {
            if ( tid < activeThreads ) {
                sPartials[tid] = update(sPartials[tid],op(sPartials[tid],sPartials[tid + activeThreads],extraParams),extraParams);
            }
            __syncthreads();
        }

        if ( tid == 0 ) {
            result[blockIdx.x] = postProcess(sPartials[0],n,xOffset,dx,incx,extraParams,result);
        }

}
