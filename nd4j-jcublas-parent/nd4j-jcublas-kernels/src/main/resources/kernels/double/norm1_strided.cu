#include "reduce.h"

__device__ double merge(double old,double opOutput,double *extraParams) {
       return opOutput + old;
 }

__device__ double update(double old,double opOutput,double *extraParams) {
       return opOutput + old;
 }


__device__ double op(double d1,double *extraParams) {
      return abs(d1);
}



__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *params,double *result) {
             return reduction;
}

extern "C"
__global__ void norm1_strided_double(int n, int xOffset,double *dx,int incx,double *params,double *result) {
             transform(n,xOffset,dx,incx,params,result);
}


