#include <reduce3.h>



/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
 */
template<> __device__ double op<double>(double d1,double d2,double *extraParams) {
	return pow(d1 - d2,2);
}


//post process result (for things like means etc)
template<> __device__ double postProcess<double>(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
	return sqrt(reduction);
}


template<> __device__ double merge<double>(double old,double opOutput,double *extraParams) {
	return old + opOutput;
}


template<> __device__ float merge<float>(float old,float opOutput,float *extraParams) {
	return old + opOutput;
}

template<> __device__ float update<float>(float old,float opOutput,float *extraParams) {
	return old + opOutput;
}

template<> __device__ double update<double>(double old,double opOutput,double *extraParams) {
	return old + opOutput;
}


/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
 */
template<> __device__ float op<float>(float d1,float d2,float *extraParams) {
	return powf(d1 - d2,2);
}


//post process result (for things like means etc)
template<> __device__ float postProcess<float>(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
	return sqrtf(reduction);
}



