#include <stdio.h>
#include <stdlib.h>
//scalar and current element
template<typename T>
__device__ T op(T d1,T d2,T *params);

template<typename T>
__device__ void transform(int n, int idx,T dx,T *dy,int incy,T *params,T *result,int blockSize) {
	int totalThreads = gridDim.x * blockDim.x;
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;

	for (; i < n; i += totalThreads) {
		result[idx + i * incy] = op(dx,dy[idx + i * incy],params);
	}

}

extern "C"
__global__ void printShapeBuffer(int n,int *buff) {
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;
	if(i < n) {
		printf("Buff item %d is %d\n",i,buff[i]);
	}
}


extern "C"
__global__ void transform_double(
		int n,
		int idx,
		double dx,
		double *dy,
		int incx,
		double *params,
		double *result,
		int blockSize) {
       transform<double>(n,idx,dx,dy,incx,params,result,blockSize);
 }



extern "C"
__global__ void transform_float(
		int n,
		int idx,
		float dx,
		float *dy,
		int incx,
		float *params,
		float *result,
		int blockSize) {
       transform<float>(n,idx,dx,dy,incx,params,result,blockSize);
 }
