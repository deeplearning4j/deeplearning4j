#include <transform.h>


template<> __device__ double op<double>(double d1,double *params) {
	return exp(d1);
}


template<> __device__ float op<float>(float d1,float *params) {
	return expf(d1);
}


