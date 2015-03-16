#include "reduce.h"


__device__ double update(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}


__device__ double merge(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}

__device__ double op(double d1,double *extraParams) {
	return d1;
}


__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *params,double *result) {
	return reduction;
}


extern "C"
__global__ void sum_strided_double(int n, int xOffset,double *dx,int incx,double *params,double *result) {
	transform(n,xOffset,dx,incx,params,result);
}



int main(void) {
	void *d = NULL;
	int i;
	int numElements = 12;
	size_t size = numElements * sizeof(double);
	printf("[Vector addition of %d elements]\n", numElements);

	// Allocate the host input vector A
	double *h_A = (double *) malloc(size);
	for(i = 0; i < numElements; i++) {
		h_A[i] = i + 1;
		printf("Host %d \n",i);
	}

    double *h_Result = (double*) malloc(size);
    for(int i = 0; i < numElements; i++) {
    	h_Result[i] = 0;
    }


	double *d_A = NULL;
	//allocate memory on device
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_A, size));
	//copy memory from host to device
	CUDA_CHECK_RETURN(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

	double *d_Result = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_Result, size));
	CUDA_CHECK_RETURN(cudaMemcpy(d_Result, h_A, size, cudaMemcpyHostToDevice));



	//dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,dev_partial_c );
	int blocksPerGrid = 128;
	int threadsPerBlock = 256;

	double *extraParams = (double *) malloc(1 * sizeof(double));
	extraParams[0] = 0;
	double *d_extraParams = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&d_extraParams, size));
    CUDA_CHECK_RETURN(cudaMemcpy(d_extraParams, extraParams, 1 * sizeof(double), cudaMemcpyHostToDevice));


	sum_strided_double<<<blocksPerGrid,threadsPerBlock,512 * sizeof(double)>>>(numElements,0,d_A,1,d_extraParams,d_Result);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(h_Result, d_Result, size, cudaMemcpyDeviceToHost));


	printf("Sum %f\n", h_Result[0]);

	CUDA_CHECK_RETURN(cudaFree(d_A));
	CUDA_CHECK_RETURN(cudaFree(d_Result));

	free(h_A);
	free(h_Result);
	CUDA_CHECK_RETURN(cudaDeviceReset());
	CUDA_CHECK_RETURN(cudaFree((void*) d));


	return 0;
}
