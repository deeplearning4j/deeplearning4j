//
// Created by agibsonccc on 1/3/16.
//

#ifndef NATIVEOPERATIONS_TRANSFORMTESTS_H
#define NATIVEOPERATIONS_TRANSFORMTESTS_H

#include <transform.h>
#include <array.h>
#include <templatemath.h>
#include <shape.h>
#include "testhelpers.h"

static functions::transform::TransformOpFactory<double> *opFactory = 0;

TEST_GROUP(Transform) {
	static int output_method(const char* output, ...) {
		va_list arguments;
		va_start(arguments, output);
		va_end(arguments);
		return 1;
	}
	void setup() {
		opFactory = new functions::transform::TransformOpFactory<double>();

	}
	void teardown() {
		delete opFactory;
	}
};

/*TEST(Transform,Log) {
	int rank = 2;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	int *stride = shape::calcStrides(shape, rank);
	nd4j::array::NDArray<double> *data =
			nd4j::array::NDArrays<double>::createFrom(rank, shape, stride, 0,
					0.0);
	int length = nd4j::array::NDArrays<double>::length(data);
	for (int i = 0; i < length; i++)
		data->data->data[i] = i + 1;

	double *extraParams = (double *) malloc(sizeof(double));

	functions::transform::Transform<double> *log = opFactory->getOp(
			"log_strided");
	log->exec(data->data->data, 1, data->data->data, 1, extraParams, length);

	double comparison[4] = { 0., 0.69314718, 1.09861229, 1.38629436 };
	CHECK(arrsEquals(rank, comparison, data->data->data));
	free(data);
	free(extraParams);
	free(shape);
	free(stride);
	delete log;

}*/

TEST(Transform,Sigmoid) {
	int rank = 2;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	int *stride = shape::calcStrides(shape, rank);
	nd4j::array::NDArray<double> *data =
			nd4j::array::NDArrays<double>::createFrom(rank, shape, stride, 0,
					0.0);

	int length = nd4j::array::NDArrays<double>::length(data);
	for (int i = 0; i < length; i++)
		data->data->data[i] = i + 1;
	printf("Initialized data\n");
	double *extraParams = (double *) malloc(sizeof(double));

	functions::transform::Transform<double> *log = opFactory->getOp(10);
	log->exec(data->data->data, 1, data->data->data, 1, extraParams, length);
	double *comparison= (double *)malloc(4 * sizeof(double));
	comparison[0] = 0.7310585786300049;
	comparison[1] = 0.8807970779778823;
	comparison[2] = 0.9525741268224334;
	comparison[3] = 0.9820137900379085;
	CHECK(arrsEquals(rank, comparison, data->data->data));


#ifdef __CUDACC__
	printf("Ran cuda\n");
	setupTransfromFactories();
	for (int i = 0; i < length; i++)
			data->data->data[i] = i + 1;
	nd4j::array::NDArrays<double>::allocateNDArrayOnGpu(&data);
	double *extraParamsData = (double *) malloc(sizeof(double));
	extraParams[0] = 0.0;
	nd4j::buffer::Buffer<double> *extraParamsBuff = nd4j::buffer::createBuffer(extraParamsData,1);
	char *nameOfOp = "sigmoid_strided";
	size_t len = strlen(nameOfOp);
	char *gpuPointer;
	cudaMalloc(&gpuPointer,len * sizeof(char));
	cudaMemcpy(gpuPointer,nameOfOp,len * sizeof(char),cudaMemcpyHostToDevice);
	printf("Copied data\n");
	transformDouble<<<length,length,42000>>>(
			gpuPointer
			,length,
			1,data->data->gData,
			1,extraParamsBuff->gData,
			data->data->gData
			,1);
	checkCudaErrors(cudaDeviceSynchronize());
	nd4j::buffer::freeBuffer(&extraParamsBuff);
	nd4j::buffer::copyDataFromGpu(&data->data);
	CHECK(arrsEquals(rank, comparison, data->data->data));
	cudaFree(gpuPointer);

#endif

	nd4j::array::NDArrays<double>::freeNDArrayOnGpuAndCpu(&data);
	delete log;

}

#endif //NATIVEOPERATIONS_TRANSFORMTESTS_H
