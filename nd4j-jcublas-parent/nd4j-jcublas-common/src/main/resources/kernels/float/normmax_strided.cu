#include "reduce3.h"
__device__ float update(float old,float opOutput,float *extraParams) {
             return fmaxf(fabsf(old),fabsf(opOutput));
 }

__device__ float merge(float old,float opOutput,float *extraParams) {
                    return fmaxf(fabsf(old),fabsf(opOutput));
}
__device__ float op(float d1,float d2,float *extraParams) {
             return fmaxf(fabsf(d1),fabsf(d2));
}



__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
             return fmaxf(fabsf(reduction),fabsf(result[0]));
}

extern "C"
__global__ void normmax_strided_float(int n, int xOffset,float *dx,int incx,float *extraParams,float *result) {
             transform(n,xOffset,dx,incx,extraParams,result);
}


