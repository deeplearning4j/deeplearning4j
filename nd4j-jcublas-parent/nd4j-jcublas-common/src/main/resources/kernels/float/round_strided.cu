#include "transform.h"


__device__ float op(float d1,float *params) {
        return roundf(d1);
}

extern "C"
__global__ void round_strided_float(int n,int idx,float *dy,int incy,float *params,float *result,int blockSize) {
       transform(n,idx,dy,incy,params,result,blockSize);

 }
