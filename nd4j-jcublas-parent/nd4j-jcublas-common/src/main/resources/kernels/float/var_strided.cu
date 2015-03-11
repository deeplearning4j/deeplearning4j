#include <reduce.h>


__device__ float update(float old,float opOutput,float *extraParams) {
       return opOutput + old;
 }


//an op for the kernel
__device__ float op(float d1,float *extraParams) {
       float mean = extraParams[0];
       float curr = (d1 - mean);
       return  powf(curr,2);

}

//post process result (for things like means etc)
__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
             float bias = extraParams[1];
             return  (reduction - (powf(bias,2) / n)) / (n - 1.0);
}

extern "C"
__global__ void var_strided_float(int n, int xOffset,float *dx,int incx,float *extraParams,float *result) {
              transform(n,xOffset,dx,incx,extraParams,result);

 }


