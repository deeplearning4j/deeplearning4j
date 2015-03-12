#include <reduce.h>


__device__ double update(double old,double opOutput,double *extraParams) {
            double mean = extraParams[1];
            double curr = (opOutput - mean);
            return old +  powf(curr,2);
 }

__device__ double op(double d1,double d2,double *extraParams) {
       return d1 + d2;
}


__device__ double merge(double d1,double d2,double *extraParams) {
       return d1 + d2;
}



__device__ double op(double d1,double *extraParams) {
      return d1;
}


__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
           return sqrtf(reduction);
}


extern "C"
__global__ void std_strided_double(int n, int xOffset,double *dx,int incx,double *extraParams,double *result) {
             transform(n,xOffset,dx,incx,extraParams,result);
}


