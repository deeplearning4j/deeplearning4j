#include "scalar.h"

__device__ float op(float d1,float d2,float *params) {
   return d2;
}

extern "C"
__global__ void set_scalar_float(int n, int idx,float dx,float *dy,int incy,float *params,float *result,int blockSize) {
       transform(n,idx,dx,dy,incy,params,result,blockSize);
 }


