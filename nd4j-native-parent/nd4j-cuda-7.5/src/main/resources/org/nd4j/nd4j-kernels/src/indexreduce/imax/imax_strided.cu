#include <indexreduce.h>



template<> __device__ IndexValue<double> merge<double>(IndexValue<double> old,IndexValue<double> opOutput,double *extraParams) {
	if(opOutput.value > old.value)
		return opOutput;
	return old;
}


template<> __device__ IndexValue<double> update<double>(IndexValue<double> old,IndexValue<double> opOutput,double *extraParams) {
	if(opOutput.value > old.value)
		return opOutput;
	return old;
}


template<> __device__ IndexValue<double> op<double>(IndexValue<double> d1,double *extraParams) {
	return d1;
}


template<> __device__ IndexValue<double> postProcess<double>(IndexValue<double> reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
	return reduction;
}


template<> __device__ IndexValue<float> merge<float>(IndexValue<float> old,IndexValue<float> opOutput,float *extraParams) {
	if(opOutput.value > old.value)
		return opOutput;
	return old;
}


template<> __device__ IndexValue<float> update<float>(IndexValue<float> old,IndexValue<float> opOutput,float *extraParams) {
	if(opOutput.value > old.value)
		return opOutput;
	return old;
}


template<> __device__ IndexValue<float> op<float>(IndexValue<float> d1,float *extraParams) {
	return d1;
}


template<> __device__ IndexValue<float> postProcess<float>(IndexValue<float> reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
	return reduction;
}




