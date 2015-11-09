#include "scalar.h"
//scalar and current element
__device__ float op(float d1,float d2,float *params) {
       if(d2 <= d1) {return 1;}
       return 0;

}

extern "C"
__global__ void lessthanorequal_scalar_float(int n, int idx,float dx,float *dy,int incx,float *params,float *result,int blockSize) {
       transform(n,idx,dx,dy,incx,params,result,blockSize);
 }


