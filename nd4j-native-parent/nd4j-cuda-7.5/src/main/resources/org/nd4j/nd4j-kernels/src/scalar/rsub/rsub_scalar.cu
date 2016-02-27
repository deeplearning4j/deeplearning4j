#include <scalar.h>
//scalar and current element
template<> __device__ double op<double>(double d1,double d2,double *params) {
	return d1 - d2;
}


template<> __device__ float op<float>(float d1,float d2,float *params) {
	return d1 - d2;
}

