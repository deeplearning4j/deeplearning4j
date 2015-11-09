#include "transform.h"


__device__ double op(double d1,double *params) {
        return pow(d1,params[0]);
}

extern "C"
__global__ void pow_strided_double(int n,int idx,double *dy,int incy,double *params,double *result,int blockSize) {
       transform(n,idx,dy,incy,params,result,blockSize);

 }