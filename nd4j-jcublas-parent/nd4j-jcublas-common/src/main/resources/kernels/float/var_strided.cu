#include <reduce.h>


__device__ float update(float old,float opOutput,float *extraParams) {
       float mean = extraParams[1];
       float curr = (opOutput - mean);
       return old +  powf(curr,2);
 }


//an op for the kernel
__device__ float op(float d1,float *extraParams) {
      return d1;

}

//post process result (for things like means etc)
__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
             float bias = extraParams[0];
             return  (reduction - (powf(bias,2) / n)) / (float) (n - 1.0);
}

extern "C"
__global__ void var_strided_float(int n, int xOffset,float *dx,int incx,float *extraParams,float *result) {
              transform(n,xOffset,dx,incx,extraParams,result);

 }


