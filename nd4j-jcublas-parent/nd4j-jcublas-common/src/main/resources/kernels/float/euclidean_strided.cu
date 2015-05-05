#include "reduce3.h"

__device__ float merge(float old,float opOutput,float *extraParams) {
       return old + opOutput;
 }

__device__ float update(float old,float opOutput,float *extraParams) {
       return old + opOutput;
 }


/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
*/
__device__ float op(float d1,float d2,float *extraParams) {
      return powf(d1 - d2,2.0f);
}


//post process result (for things like means etc)
__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
            return sqrt(reduction);
}

extern "C"
__global__ void euclidean_strided_float(int n, int xOffset,int yOffset,float *dx,float *dy,int incx,int incy,float *extraParams,float *result) {
              transform_pair(n,xOffset,yOffset,dx,dy,incx,incy,extraParams,result);

 }


