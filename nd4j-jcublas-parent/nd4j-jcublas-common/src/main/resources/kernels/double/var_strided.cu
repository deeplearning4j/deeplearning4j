#include <reduce.h>


__device__ double update(double old,double opOutput,double *extraParams) {
       return opOutput;
 }


/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
*/
__device__ double op(double d1,double d2,double *extraParams) {
      return op(d1,extraParams);
}
//an op for the kernel
__device__ double op(double d1,double *extraParams) {
       double mean = extraParams[0];
       double curr = (d1 - mean);
       return  pow(curr,2);

}

//post process result (for things like means etc)
__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
             double bias = extraParams[1];
             return  (reduction - (pow(bias,2) / n)) / (n - 1.0);
}

extern "C"
__global__ void var_strided_double(int n, int xOffset,double *dx,int incx,double *extraParams,double *result) {
              transform(n,xOffset,dx,incx,extraParams,result);

 }


