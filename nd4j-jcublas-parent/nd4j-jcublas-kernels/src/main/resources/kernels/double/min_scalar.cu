#include "scalar.h"

__device__ double op(double d1,double d2,double *params) {
    if(d1 < d2)
      return 1;
   return 0;
}

extern "C"
__global__ void min_scalar_double(int n, int idx,double dx,double *dy,int incy,double *params,double *result,int blockSize) {
       transform(n,idx,dx,dy,incy,params,result,blockSize);
 }


