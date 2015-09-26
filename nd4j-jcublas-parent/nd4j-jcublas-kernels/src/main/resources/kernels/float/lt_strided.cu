#include "pairwise_transform.h"

__device__ float op(float d1,float d2,float *params) {
   if(d1 < d2) return 1;
    else return 0;
 }
__device__ float op(float d1,float *params) {
   return d1;
}

extern "C"
__global__ void lt_strided_float(int n,int xOffset,int yOffset, float *dx, float *dy,int incx,int incy,float *params,float *result,int incz) {
    transform(n,xOffset,yOffset,dx,dy,incx,incy,params,result,incz);
}