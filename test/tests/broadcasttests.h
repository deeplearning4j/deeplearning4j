//
// Created by agibsonccc on 1/4/16.
//

#ifndef NATIVEOPERATIONS_BROADCASTSTESTS_H
#define NATIVEOPERATIONS_BROADCASTSTESTS_H
#include <broadcasting.h>
#include <array.h>
#include <shape.h>
#include <buffer.h>
#include "testhelpers.h"
#include <cuda_helpers.h>
static functions::broadcast::BroadcastOpFactory<double> *opFactory3 = 0;

TEST_GROUP(BroadCasting) {
	static int output_method(const char* output, ...) {
		va_list arguments;
		va_start(arguments, output);
		va_end(arguments);
		return 1;
	}
	void setup() {
		opFactory3 = new functions::broadcast::BroadcastOpFactory<double>();

	}
	void teardown() {
		delete opFactory3;
	}
};

#ifdef __CUDACC__
__global__ void tryInstant() {
#if __CUDA_ARCH__ >= 500
	functions::broadcast::ops::Add<double> *op = new functions::broadcast::ops::Add<double>();
	delete op;
#endif
}
#endif

TEST(BroadCasting,Kernel) {
#ifdef __CUDACC__
	tryInstant<<<1,1,1>>>();
	checkCudaErrors(cudaDeviceSynchronize());
#endif
}

TEST(BroadCasting,Addition) {
	functions::broadcast::Broadcast<double> *add = opFactory3->getOp(0);
	int rank = 2;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	int *stride = shape::calcStrides(shape, rank);
	nd4j::array::NDArray<double> *data =
			nd4j::array::NDArrays<double>::createFrom(rank, shape, stride, 0,
					0.0);
	shape::ShapeInformation *shapeInformation =
			nd4j::array::NDArrays<double>::shapeInfoForArray(data);
	int *shapeInfoBuffer = shape::toShapeBuffer(shapeInformation);
	nd4j::buffer::Buffer<int> *shapeInfoBuferBuff = nd4j::buffer::createBuffer(shapeInfoBuffer,shape::shapeInfoLength(shape::rank(shapeInfoBuffer)));
	assertBufferProperties(shapeInfoBuffer);
	int length = nd4j::array::NDArrays<double>::length(data);
	CHECK(length == 4);
	for (int i = 0; i < length; i++)
		data->data->data[i] = i + 1;

	int *vectorShape = (int *) malloc(rank * sizeof(int));
	vectorShape[0] = 1;
	vectorShape[1] = 2;
	int *vectorStride = shape::calcStrides(vectorShape, rank);
	nd4j::array::NDArray<double> *vector =
			nd4j::array::NDArrays<double>::createFrom(rank, vectorShape,
					vectorStride, 0, 0.0);
	for (int i = 0; i < 2; i++)
		vector->data->data[i] = i + 1;
	shape::ShapeInformation *vectorShapeInformation = nd4j::array::NDArrays<double>::shapeInfoForArray(vector);
	int *vectorShapeInfoBuff = shape::toShapeBuffer(vectorShapeInformation);
	nd4j::buffer::Buffer<int> *vectorShapeInfoBuffData =  nd4j::buffer::createBuffer(vectorShapeInfoBuff,shape::shapeInfoLength(shape::rank(vectorShapeInfoBuff)));
	assertBufferProperties(vectorShapeInfoBuff);
	int dimensionLength = 1;
	int *dimension = (int *) malloc(sizeof(int));
	dimension[0] = 1;
	nd4j::buffer::Buffer<int> *dimensionBuff = nd4j::buffer::createBuffer(dimension,1);
	double *extraParams = (double *) malloc(sizeof(double));
	nd4j::buffer::Buffer<double> *extraParamsBuffer = nd4j::buffer::createBuffer(extraParams,1);
	add->exec(data->data->data, shapeInfoBuffer, vector->data->data,
			vectorShapeInfoBuff, data->data->data, shapeInfoBuffer, dimension,
			dimensionLength);

	double *comparison =(double *) malloc(sizeof(double) * 4);
	comparison[0] = 2;
	comparison[1] = 4;
	comparison[2] = 5;
	comparison[3] = 6;
	CHECK(arrsEquals(rank, comparison, data->data->data));

#ifdef __CUDACC__
	for (int i = 0; i < length; i++)
		data->data->data[i] = i + 1;
	int blockSize = 500;
	int gridSize = 256;
	int sMemSize = 20000;

	int *gpuInformation = (int *) malloc(sizeof(int) * 4);
	gpuInformation[0] = blockSize;
	gpuInformation[1] = gridSize;
	gpuInformation[2] = sMemSize;
	gpuInformation[3] = 49152;
	nd4j::buffer::Buffer<int> *gpuInfoBuff = nd4j::buffer::createBuffer<int>(gpuInformation,4);

  /*
   * 	int opNum,
		double *x, int *xShapeInfo,
		double *y, int *yShapeInfo,
		double *result, int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int *gpuInformation
   */

	broadcastDouble<<<blockSize,gridSize,sMemSize>>>(
			5,
			data->data->gData,
			shapeInfoBuferBuff->gData,
			vector->data->gData,
			vectorShapeInfoBuffData->gData,
			data->data->gData,
			shapeInfoBuferBuff->gData,
			dimensionBuff->gData,
			dimensionLength,
			gpuInfoBuff);
	checkCudaErrors(cudaDeviceSynchronize());



	nd4j::buffer::copyDataFromGpu(&data->data);
#endif


	nd4j::array::NDArrays<double>::freeNDArrayOnGpuAndCpu(&data);
	nd4j::buffer::freeBuffer(&extraParamsBuffer);
	free(shapeInformation);
	nd4j::buffer::freeBuffer(&shapeInfoBuferBuff);

	nd4j::array::NDArrays<double>::freeNDArrayOnGpuAndCpu(&vector);
	nd4j::buffer::freeBuffer(&vectorShapeInfoBuffData);
	nd4j::buffer::freeBuffer(&dimensionBuff);

	delete add;

}


#endif //NATIVEOPERATIONS_BROADCASTSTESTS_H
