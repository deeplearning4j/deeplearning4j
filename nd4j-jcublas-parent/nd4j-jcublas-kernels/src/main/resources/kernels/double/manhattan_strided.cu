#include "reduce3.h"


__device__ double update(double old,double opOutput,double *extraParams) {
       return powf(old,2) + opOutput;
 }


/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
*/
__device__ double op(double d1,double d2,double *extraParams) {
      return d1 - d2;
}


//post process result (for things like means etc)
__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
            return reduction / extraParams[0] / extraParams[1];
}

extern "C"
__global__ void manhattan_strided_double(int n, int xOffset,int yOffset,double *dx,double *dy,int incx,int incy,double *extraParams,double *result) {
              transform_pair(n,xOffset,yOffset,dx,dy,incx,incy,extraParams,result);

 }


