#include <scalar.h>

template<> __device__ double op<double>(double d1,double d2,double *params) {
   return d2 - d1;
}





template<> __device__ float op<float>(float d1,float d2,float *params) {
   return d2 - d1;
}
