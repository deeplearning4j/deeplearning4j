#include <reduce.h>


template<> __device__ double merge<double>(double f1,double f2,double *extraParams) {
	return f1 + f2;
}

template<> __device__ double update<double>(double old,double opOutput,double *extraParams) {
	double mean = extraParams[2];
	double curr = pow(opOutput - mean,2.0);
	return old + curr;
}


//an op for the kernel
template<> __device__ double op<double>(double d1,double *extraParams) {
	return d1;

}

//post process result (for things like means etc)
template<> __device__ double postProcess<double>(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
	double bias = extraParams[1];
	return  sqrt((reduction - (pow(bias,2.0) / n)) / (double) (n - 1.0));

}


template<> __device__ float merge<float>(float f1,float f2,float *extraParams) {
	return f1 + f2;
}

template<> __device__ float update<float>(float old,float opOutput,float *extraParams) {
	float mean = extraParams[2];
	float curr = powf(opOutput - mean,2.0);
	return old + curr;
}


//an op for the kernel
template<> __device__ float op<float>(float d1,float *extraParams) {
	return d1;

}

//post process result (for things like means etc)
template<> __device__ float postProcess<float>(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
	float bias = extraParams[1];
	return  sqrtf((reduction - (powf(bias,2.0) / n)) / (float) (n - 1.0));

}








