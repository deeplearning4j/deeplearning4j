#include <scalar.h>
//scalar and current element
template<> __device__ double op<double>(double d1,double d2,double *params) {
    if(d2 >= d1) {return 1;}
    return 0;

}


template<> __device__ float op<float>(float d1,float d2,float *params) {
    if(d2 >= d1) {return 1;}
    return 0;

}





