#include <transform.h>


template<> __device__ double op<double>(double d1,double *params) {
	return 1.0 / (1.0 + exp(-d1));
}


template<> __device__ float op<float>(float d1,float *params) {
	return 1.0 / (1.0 + expf(-d1));
}
