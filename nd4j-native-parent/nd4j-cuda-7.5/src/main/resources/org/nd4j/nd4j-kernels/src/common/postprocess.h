/*
 * postprocess.h
 *
 *  Created on: Dec 19, 2015
 *      Author: agibsonccc
 */

#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_


//post process result (for things like means etc)
template<typename T>
__device__ T postProcess(T reduction,int n,int xOffset,T *dx,int incx,T *extraParams,T *result);

template <> double postProcess<double>(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result);
template <> float postProcess<float>(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result);


template <typename T>
__device__ void postProcessLoop(int n,int xOffset,T *dx,int incx,T *extraParams,T *result) {
	int tid = threadIdx.x;
	int i = xOffset + (blockIdx.x * blockDim.x + tid);

	for(; i < n; i += gridDim.x * blockDim.x * incx) {
		result[i] = postProcess(result[i],n,xOffset,dx,incx,extraParams,result);
	}
}


extern "C"
__global__ void postProcessLoop_double(int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
	postProcessLoop<double>(n,xOffset,dx,incx,extraParams,result);
}

extern "C"
__global__ void postProcessLoop_float(int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
	postProcessLoop<float>(n,xOffset,dx,incx,extraParams,result);
}





#endif /* POSTPROCESS_H_ */
