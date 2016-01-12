//
// Created by agibsonccc on 1/4/16.
//

#ifndef NATIVEOPERATIONS_BROADCASTSTESTS_H
#define NATIVEOPERATIONS_BROADCASTSTESTS_H
#include <broadcasting.h>
#include <array.h>
#include <shape.h>
#include "testhelpers.h"

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
	assertBufferProperties(vectorShapeInfoBuff);
	int dimensionLength = 1;
	int *dimension = (int *) malloc(sizeof(int));
	dimension[0] = 1;

	double *extraParams = (double *) malloc(sizeof(double));

	add->exec(data->data->data, shapeInfoBuffer, vector->data->data,
			vectorShapeInfoBuff, data->data->data, shapeInfoBuffer, dimension,
			dimensionLength);

	double *comparison =(double *) malloc(sizeof(double) * 4);
	comparison[0] = 2;
	comparison[1] = 4;
	comparison[2] = 5;
	comparison[3] = 6;
    for(int i = 0; i < 4; i++) {
       printf("Data[%d] is %f\n",i,data->data->data[i]);
    }
	CHECK(arrsEquals(rank, comparison, data->data->data));
	free(data);
	free(extraParams);
	free(shape);
	free(stride);
	free(shapeInformation);
	free(shapeInfoBuffer);

	free(vector);
	free(vectorShape);
	free(vectorStride);
	free(vectorShapeInformation);
	free(vectorShapeInfoBuff);

	free(dimension);
	delete add;

}


#endif //NATIVEOPERATIONS_BROADCASTSTESTS_H
