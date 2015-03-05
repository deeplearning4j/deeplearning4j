extern "C"
#include <reduce.h>


__device__ float update(float old,float opOutput,float *extraParams) {
       return old + opOutput;
 }


/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
*/
__device__ float op(float d1,float d2,float *extraParams) {
      return d1 * d2 * extraParams[0];
}
//an op for the kernel
__device__ float op(float d1,float *extraParams) {
       return  d1 * extraParams[0];

}

//post process result (for things like means etc)
__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
            return reduction;
}


__global__ void cosine_similarity_strided_float(int n, int xOffset,float *dx,int incx,float *extraParams,float *result) {
              transform(n,xOffset,dx,incx,extraParams,result);

 }


