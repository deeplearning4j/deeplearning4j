#include "reduce.h"



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

extern "C"
__global__ void iamax_strided_float(int n, int xOffset,float *dx,int incx,float *extraParams,float *result) {
             transform(n,xOffset,dx,incx,extraParams,result);
}


