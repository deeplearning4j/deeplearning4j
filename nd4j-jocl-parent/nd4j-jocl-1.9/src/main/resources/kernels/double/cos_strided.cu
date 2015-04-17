#include "transform.h"


__device__ double op(double d1,double *params) {
        return cos(d1);
}

extern "C"
__global__ void cos_strided_double(int n,int idx,double *dy,int incy,double *params,double *result) {
       transform(n,idx,dy,incy,params,result);

 }
