#include "transform.h"


__device__ float op(float d1,float *params) {
       float min = params[0];
       float max = params[1];
       if(origin >= min && origin <= max)
                return d1;
        if(min == 0 && max == 1) {
               float val = 1 / (1 + expf(-origin));
               return (floorf(val * (max - min)) + min);
         }

         float ret =  (floorf(origin * (max - min)) + min);
         return ret;

}

extern "C"
__global__ void setrange_strided_float(int n,int idx,float *dy,int incy,float *params,float *result) {
       transform(n,idx,dy,incy,params,result);

 }
