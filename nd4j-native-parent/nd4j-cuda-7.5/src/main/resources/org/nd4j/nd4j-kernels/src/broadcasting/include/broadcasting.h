/*
 * broadcasting.h
 *
 *  Created on: Nov 11, 2015
 *      Author: agibsonccc
 */

#ifndef BROADCASTING_H_
#define BROADCASTING_H_

#define MAX_THREADS 512

#include <math.h>
#include <stdio.h>

#include <sharedmem.h>
#include <tad.h>
#include <indexing.h>



template <typename T>
__device__ T op(T d1,T d2);
template <typename T>
__device__ T op(T d1);




template <typename T>
__device__ void transform(
		T *x
		,int *xShapeInfo
		,T *y
		,int *yShapeInfo
		,T *result
		,int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int *gpuInformation) {


	int xElementWiseStride = elementWiseStride(xShapeInfo);
	int xOffset = offset(xShapeInfo);
	int yElementWiseStride = elementWiseStride(yShapeInfo);
	int yOffset = offset(yShapeInfo);



	//length for the tad
	int yLength = length(yShapeInfo);
	//length for the tad
	int xLength  = length(xShapeInfo);

	int resultLength = length(resultShapeInfo);
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			i < resultLength;
			i += blockDim.x * gridDim.x) {
		int yOffset2 = yOffset + ((i / xElementWiseStride)% yLength) * yElementWiseStride;
		if(i < resultLength)
			result[i] = op(x[i],y[yOffset2]);

	}

}

extern "C"
__global__ void transform_double(double *x,int *xShapeInfo,double *y,int *yShapeInfo,double *result,int *resultShapeInfo,int *dimension,
		int dimensionLength,
		int *gpuInformation) {
	transform<double>(x,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,dimension,dimensionLength,gpuInformation);
}


extern "C"
__global__ void transform_float(float *x,int *xShapeInfo,float *y,int *yShapeInfo,float *result,int *resultShapeInfo,int *dimension,
		int dimensionLength,
		int *gpuInformation) {
	transform<float>(x,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,dimension,dimensionLength,gpuInformation);
}



#endif /* BROADCASTING_H_ */
