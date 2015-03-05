#include <reduce.h>


__device__ float update(float old,float opOutput,float *extraParams) {
       return opOutput + old;
 }


/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
*/
__device__ float op(float d1,float d2,float *extraParams) {
      return op(d1,extraParams);
}

__device__ float op(float d1,float *extraParams) {
      return powf(d1,2);
}



__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
             return sqrtf(reduction);
}

extern "C"
__global__ void norm2_strided_float(int n, int xOffset,float *dx,int incx,float *extraParams,float *result) {
             transform(n,xOffset,dx,incx,extraParams,result);
}


