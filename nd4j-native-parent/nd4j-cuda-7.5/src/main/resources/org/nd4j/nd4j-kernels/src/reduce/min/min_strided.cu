#include <reduce.h>
template<> __device__ double update<double>(double old,double opOutput,double *extraParams) {
	return min(old,opOutput);
}

template<> __device__ double merge<double>(double old,double opOutput,double *extraParams) {
	return min(old,opOutput);
}

/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
 */
template<> __device__ double op<double>(double d1,double d2,double *extraParams) {
	return d1;
}

template<> __device__ double op<double>(double d1,double *extraParams) {
	return d1;
}



template<> __device__ double postProcess<double>(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
	return reduction;
}




template<> __device__ float update<float>(float old,float opOutput,float *extraParams) {
	return fminf(old,opOutput);
}

template<> __device__ float merge<float>(float old,float opOutput,float *extraParams) {
	return fminf(old,opOutput);
}

/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
 */
template<> __device__ float op<float>(float d1,float d2,float *extraParams) {
	return d1;
}

template<> __device__ float op<float>(float d1,float *extraParams) {
	return d1;
}



template<> __device__ float postProcess<float>(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
	return reduction;
}



