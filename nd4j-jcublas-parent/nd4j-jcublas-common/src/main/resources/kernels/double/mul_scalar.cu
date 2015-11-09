#include "scalar.h"

__device__ double op(double d1,double d2,double *params) {
   return d2 * d1;
}

extern "C"
__global__ void mul_scalar_double(int n, int idx,double dx,double *dy,int incy,double *params,double *result,int blockSize) {
       transform(n,idx,dx,dy,incy,params,result,blockSize);
 }


