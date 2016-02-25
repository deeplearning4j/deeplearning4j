/*
 * array.h
 *
 *  Created on: Dec 24, 2015
 *      Author: agibsonccc
 */

#include "tad.h"
#include "helper_string.h"
#include "helper_cuda.h"


#ifndef ARRAY_H_
#define ARRAY_H_
template <typename T>
struct NDArray {
	T *data;
	T *gData;
	int *shape,*stride;
	int *gShape,*gStride;
	int offset;
	int rank;
	char ordering;
};

template <>
struct NDArray<double> {
	double *data;
	double *gData;
	int *shape,*stride;
	int *gShape,*gStride;
	int offset;
	int rank;
	char ordering;
};


template <>
struct NDArray<float> {
	float *data;
	float *gData;
	int *shape,*stride;
	int *gShape,*gStride;
	int offset;
	int rank;
	char ordering;
};

/**
 * Returns the length of this ndarray
 */
template <typename T>
__device__ __host__ size_t length(NDArray<T> *arr) {
	size_t size = prod(arr->shape,arr->rank);
	return size;
}

/**
 * Returns the length of
 * this ndarray
 * in bytes
 */
template <typename T>
__device__ __host__ size_t lengthInBytes(NDArray<T> *arr) {
	size_t size = prod(arr->shape,arr->rank) * sizeof(T);
	return size;
}

/**
 * Creates an ndarray
 * from the given rank,shape,stride,
 * offset and fills the array
 * with the given default value
 */
template <typename T>
__device__ __host__ NDArray<T> * createFrom(int rank,int *shape,int *stride,int offset,T defaultValue) {
	NDArray<T> *ret = (NDArray<T> *) malloc(sizeof(NDArray<T>));
	ret->rank = rank;
	ret->shape = shape;
	ret->stride = stride;
	ret->offset = offset;
	size_t size = lengthInBytes(ret);
	ret->data = (T*) malloc(size);
	memset(ret->data,defaultValue,size);
	return ret;
}
/**
 * Copy the already allocated host pointers
 * to the gpu.
 *
 * Note that the ndarray must
 * have already been initialized.
 *
 */
template <typename T>
__host__ void allocateNDArrayOnGpu(NDArray<T> **arr) {
	NDArray<T> *arrRef = *arr;
	T *gData;
	size_t size = lengthInBytes(arrRef);
	checkCudaErrors(cudaMalloc(&gData,size));
	checkCudaErrors(cudaMemcpy(gData,arrRef->data,size,cudaMemcpyHostToDevice));
	arrRef->gData = gData;

	size_t intRankSize = arrRef->rank * sizeof(int);
	int *gShape,*gStride;
	checkCudaErrors(cudaMalloc(&gShape,intRankSize));
	checkCudaErrors(cudaMemcpy(gShape,arrRef->shape,intRankSize,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc(&gStride,intRankSize));
	checkCudaErrors(cudaMemcpy(gStride,arrRef->stride,intRankSize,cudaMemcpyHostToDevice));
	arrRef->gShape = gShape;
	arrRef->gStride = gStride;
}

/**
 * Creates an ndarray based on the
 * given parameters
 * and then allocates it on the gpu
 */
template <typename T>
__device__ NDArray<T>* createFromAndAllocateOnGpu(int rank,int *shape,int *stride,int offset,T defaultValue) {
	NDArray<T>* ret = createFrom(rank,shape,stride,offset,defaultValue);
	allocateNDArrayOnGpu(&ret);
	return ret;
}

/**
 * Copies the host data
 * from the gpu
 * to the cpu
 * for the given ndarray
 */
template <typename T>
__host__ void copyFromGpu(NDArray<T> ** arr) {
	NDArray<T> *arrRef = *arr;
	checkCudaErrors(cudaMemcpy(arrRef->data,arrRef->gData,lengthInBytes(arrRef),cudaMemcpyDeviceToHost));
}

template <typename T>
__host__ void freeNDArrayOnGpuAndCpu(NDArray<T> **arr) {
	NDArray<T> *arrRef = *arr;
	delete[] arrRef->data;
	checkCudaErrors(cudaFree(arrRef->gData));
	delete[] arrRef->shape;
	checkCudaErrors(cudaFree(arrRef->gShape));
	delete[] arrRef->stride;
	checkCudaErrors(cudaFree(arrRef->gStride));
}


/**
 * Allocate the data based
 * on the shape information
 */
template <typename T>
__host__ __device__ void allocArrayData(NDArray<T> ** arr) {
	NDArray<T> *arrRef = *arr;
	int dataLength = prod(arrRef->shape,arrRef->rank);
	arrRef->data = malloc(sizeof(T) * dataLength);
}

/**
 * Returns the shape information for this array
 * Note that this allocates memory that should be freed.
 *
 * Note that it will use the pointers directly by reference
 * for shape and stride
 * @return the shape information for the given array
 */
template <typename T>
__host__ __device__ ShapeInformation *shapeInfoForArray(NDArray<T> *arr) {
	ShapeInformation *info = (ShapeInformation *) malloc(sizeof(ShapeInformation));
	info->offset = arr->offset;
	info->order = arr->ordering;
	info->shape = arr->shape;
	info->stride = arr->stride;
	info->rank = arr->rank;
	return info;
}

/**
 * Create based on the given shape information
 * and specified default value.
 * Note that the shape information is assumed to be filled in.
 *
 *
 */
template <typename T>
__host__ __device__ NDArray<T> * createFromShapeInfo(ShapeInformation *info,T defaultValue) {
	return createFrom(info->rank,info->shape,info->stride,info->offset,defaultValue);
}

template <typename T>
__device__ void printArrGpu(NDArray<T> *arr) {
	for(int i = 0; i < length(arr); i++) {
		printf("Arr[%d] is %f\n",arr->gData[i]);
	}
}

template <typename T>
__host__ void printArrHost(NDArray<T> *arr) {
	for(int i = 0; i < length(arr); i++) {
		printf("Arr[%d] is %f\n",i,arr->data[i]);
	}
}

#endif /* ARRAY_H_ */
