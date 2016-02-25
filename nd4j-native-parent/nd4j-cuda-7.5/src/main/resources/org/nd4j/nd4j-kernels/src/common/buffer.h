/*
 * buffer.h
 *
 *  Created on: Dec 24, 2015
 *      Author: agibsonccc
 */

#ifndef BUFFER_H_
#define BUFFER_H_

#include "helper_string.h"
#include "helper_cuda.h"

template <typename T>
struct Buffer {
	int length;
	T *data;
	T *gData;
};

template <>
struct Buffer<double> {
	int length;
	double *data;
	double *gData;
};


template <>
struct Buffer<float> {
	int length;
	float *data;
	float *gData;
};

template <>
struct Buffer<int> {
	int length;
	int *data;
	int *gData;
};


template <typename T>
__device__ __host__ size_t bufferSize(Buffer<T> *buffer) {
	return sizeof(T) * buffer->length;
}

template <typename T>
__host__ void copyDataToGpu(Buffer<T> **buffer) {
	Buffer<T> *bufferRef = *buffer;
	checkCudaErrors(cudaMemcpy(bufferRef->gData,bufferRef->data,bufferSize(bufferRef),cudaMemcpyHostToDevice));
}

template <typename T>
__host__ void copyDataFromGpu(Buffer<T> **buffer) {
	Buffer<T> *bufferRef = *buffer;
	int bufferTotalSize = bufferSize(bufferRef);
	checkCudaErrors(cudaMemcpy(bufferRef->data,bufferRef->gData,bufferTotalSize,cudaMemcpyDeviceToHost));
}
/**
 * Allocate buffer of the given
 * length on the cpu and gpu.
 */
template <typename T>
__host__ void allocBuffer(Buffer<T> **buffer,int length) {
	Buffer<T> *bufferRef = *buffer;
	bufferRef->length = length;
	bufferRef->data = (T *) malloc(sizeof(T) * length);
	checkCudaErrors(cudaMalloc(&bufferRef->gData,sizeof(T) * length));

}


/**
 * Frees the given buffer
 * (gpu and cpu
 */
template <typename T>
__host__ void freeBuffer(Buffer<T> **buffer) {
	Buffer<T> *bufferRef = *buffer;
	delete[] bufferRef->data;
	checkCudaErrors(cudaFree(bufferRef->gData));
}



/**
 * Creates a buffer
 * based on the data
 * and also synchronizes
 * the data on the gpu.
 */
template <typename T>
__host__ Buffer<T> *createBuffer(T *data,int length) {
	Buffer<T> *ret = (Buffer<T> *) malloc(sizeof(Buffer<T>));
	ret->data = data;
	ret->length = length;
	T *gData;
	T** gDataRef = &(gData);
	checkCudaErrors(cudaMalloc((void** ) gDataRef,sizeof(T) * length));
	ret->gData = gData;
	checkCudaErrors(cudaMemcpy(ret->gData,ret->data,sizeof(T) * length,cudaMemcpyHostToDevice));
	return ret;
}


template <typename T>
__host__ void printArr(Buffer<T> *buff) {
	for(int i = 0; i < buff->length; i++) {
		printf("Buffer[%d] was %f\n",i,buff->data[i]);
	}
}


#endif /* BUFFER_H_ */
