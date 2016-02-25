#include <transform.h>


template<> __device__ double op<double>(double d1,double *params) {
        return pow(d1,params[0]);
}


template<> __device__ float op<float>(float d1,float *params) {
        return powf(d1,params[0]);
}


