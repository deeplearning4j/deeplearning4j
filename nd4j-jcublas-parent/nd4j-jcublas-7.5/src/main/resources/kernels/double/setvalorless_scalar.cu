#include "scalar.h"
//scalar and current element
__device__ double op(double d1,double d2,double *params) {
    if(d2 < d1) {
       return d1;
     }
    return d2;

}

extern "C"
__global__ void setvalorless_scalar_double(int n, int idx,double dx,double *dy,int incx,double *params,double *result) {
       transform(n,idx,dx,dy,incx,params,result);
}


