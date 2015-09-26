#include "scalar.h"
//scalar and current element
__device__ float op(float d1,float d2,float *params) {
   return d1 / d2;
}

extern "C"
__global__ void rdiv_scalar_float(int n, int idx,float dx,float *dy,int incy,float *params,float *result) {
       transform(n,idx,dx,dy,incy,params,result);
 }


