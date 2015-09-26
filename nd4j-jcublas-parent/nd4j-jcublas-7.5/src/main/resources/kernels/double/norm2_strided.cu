#include "reduce.h"

__device__ double merge(double old,double opOutput,double *extraParams) {
       return opOutput + old;
 }

__device__ double update(double old,double opOutput,double *extraParams) {
       return opOutput + old;
 }


__device__ double op(double d1,double *extraParams) {
      return pow(d1,2);
}


__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *params,double *result) {
             return sqrtf(reduction);
}
extern "C"
__global__ void norm2_strided_double(int n, int xOffset,double *dx,int incx,double *params,double *result) {
             transform(n,xOffset,dx,incx,params,result);
}


