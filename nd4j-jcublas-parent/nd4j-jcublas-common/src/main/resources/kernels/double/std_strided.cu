#include <reduce.h>


__device__ double update(double old,double opOutput,double *extraParams) {
       return opOutput + old;
 }


//an op for the kernel
__device__ double op(double d1,double *extraParams) {
       double mean = extraParams[1];
       double curr = (d1 - mean);
       return  pow(curr,2);

}

//post process result (for things like means etc)
__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
             return sqrt(reduction);
}

extern "C"
__global__ void std_strided_double(int n, int xOffset,double *dx,int incx,double *extraParams,double *result) {
              transform(n,xOffset,dx,incx,extraParams,result);

 }


