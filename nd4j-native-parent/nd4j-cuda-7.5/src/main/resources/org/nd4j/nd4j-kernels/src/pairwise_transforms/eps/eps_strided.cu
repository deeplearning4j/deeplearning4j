#include <pairwise_transform.h>

#define MIN 1e-12


template <> __device__ double op<double>(double d1,double d2,double *params) {
	double diff = d1 - d2;
	double absDiff = abs(diff);
	if(absDiff < MIN)
		return 1;
	return 0;
}
template <>  __device__ double op<double>(double d1,double *params) {
	return d1;
}



template <>  __device__ float op<float>(float d1,float d2,float *params) {
	float diff = d1 - d2;
	float absDiff = fabsf(diff);
	if(absDiff < MIN)
		return 1;
	return 0;
}
template <>  __device__ float op<float>(float d1,float *params) {
	return d1;
}

