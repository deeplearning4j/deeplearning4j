#include "scalar.h"

__device__ float op(float d1,float d2,float *params) {
    if(d1 < d2)
      return d2;
   return d1;
}

extern "C"
__global__ void max_scalar_float(int n, int idx,float dx,float *dy,int incy,float *params,float *result,int blockSize) {
       transform(n,idx,dx,dy,incy,params,result,blockSize);
 }


