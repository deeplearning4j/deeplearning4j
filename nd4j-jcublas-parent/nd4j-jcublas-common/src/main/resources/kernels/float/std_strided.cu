#include "reduce.h"


__device__ float update(float old,float opOutput,float *extraParams) {
            //due to standard deviation inheriting from variance
            //the args here are: zero value, bias, mean
            float mean = extraParams[2];
            float curr = (opOutput - mean);
            return old +  powf(curr,2);
 }

__device__ float op(float d1,float d2,float *extraParams) {
       return d1;
}


__device__ float merge(float d1,float d2,float *extraParams) {
       return d1 + d2;
}



__device__ float op(float d1,float *extraParams) {
      return d1;
}


__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
           return sqrtf(reduction);
}


extern "C"
__global__ void std_strided_float(int n, int xOffset,float *dx,int incx,float *extraParams,float *result) {
             transform(n,xOffset,dx,incx,extraParams,result);
}


