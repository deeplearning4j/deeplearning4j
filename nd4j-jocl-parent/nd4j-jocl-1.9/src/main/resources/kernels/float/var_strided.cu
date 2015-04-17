#include "reduce.h"


__device__ float merge(float f1,float f2,float *extraParams) {
   return f1 + f2;
}

__device__ float update(float old,float opOutput,float *extraParams) {
       float mean = extraParams[2];
       float curr = powf(opOutput - mean,2.0);
       return old + curr;
 }


//an op for the kernel
__device__ float op(float d1,float *extraParams) {
      return d1;

}

//post process result (for things like means etc)
__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
             float bias = extraParams[1];
            return  (reduction - (powf(bias,2.0) / n)) / (float) (n - 1.0);

}

extern "C"
__global__ void var_strided_float(int n, int xOffset,float *dx,int incx,float *extraParams,float *result) {
              transform(n,xOffset,dx,incx,extraParams,result);

 }


