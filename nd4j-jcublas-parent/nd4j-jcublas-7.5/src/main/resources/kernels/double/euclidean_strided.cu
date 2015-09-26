#include "reduce3.h"

__device__ double merge(double old,double opOutput,double *extraParams) {
       return old + opOutput;
 }

__device__ double update(double old,double opOutput,double *extraParams) {
       return old + opOutput;
 }


/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
*/
__device__ double op(double d1,double d2,double *extraParams) {
      return pow(d1 - d2,2);
}


//post process result (for things like means etc)
__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
            return sqrt(reduction);
}

extern "C"
__global__ void euclidean_strided_double(int n, int xOffset,int yOffset,double *dx,double *dy,int incx,int incy,double *extraParams,double *result) {
              transform_pair(n,xOffset,yOffset,dx,dy,incx,incy,extraParams,result);

 }


