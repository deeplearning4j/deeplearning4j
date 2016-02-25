#include <pairwise_transform.h>

template <>  __device__ double op<double>(double d1,double d2,double *params) {
	if(d1 > d2) return 1;
	else return 0;
}

template <> __device__ double op<double>(double d1,double *params) {
	return d1;
}


template <>  __device__ float op<float>(float d1,float d2,float *params) {
	if(d1 > d2) return 1;
	else return 0;
}

template<> __device__ float op<float>(float d1,float *params) {
	return d1;
}


