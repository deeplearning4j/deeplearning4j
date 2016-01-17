/*
 * reducetests.h
 *
 *  Created on: Dec 31, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCETESTS_H_
#define REDUCETESTS_H_
#include <array.h>
#include "testhelpers.h"
#include <reduce.h>
#include <helper_cuda.h>
TEST_GROUP(Reduce) {

	static int output_method(const char* output, ...) {
		va_list arguments;
		va_start(arguments, output);
		va_end(arguments);
		return 1;
	}
	void setup() {

	}
	void teardown() {
	}
};

template <typename T>
class ReduceTest : public BaseTest<T> {
public:
	virtual ~ReduceTest() {
		freeOpAndOpFactory();
	}
	ReduceTest() {
		createOperationAndOpFactory();
	}
	ReduceTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
	:  BaseTest<T>(rank,opNum,data,extraParamsLength){
		createOperationAndOpFactory();
	}
	void freeOpAndOpFactory() {
		delete opFactory;
		delete reduce;
	}
	virtual void createOperationAndOpFactory() {
		opFactory = new functions::reduce::ReduceOpFactory<T>();
		reduce = opFactory->create(this->opNum);
	}

	virtual void execCpuKernel() override {
		int *xShapeBuff = shapeBuffer(this->baseData->rank,this->baseData->xShape);
		int *resultShapeBuff = shapeBuffer(this->baseData->resultRank,this->baseData->resultShape);
		assertBufferProperties(xShapeBuff);
		assertBufferProperties(resultShapeBuff);
		reduce->exec(
				this->baseData->data,
				xShapeBuff,
				this->baseData->extraParams,
				this->baseData->result,
				resultShapeBuff,
				this->baseData->dimension,
				this->baseData->dimensionLength);
		free(xShapeBuff);
		free(resultShapeBuff);
	}

	virtual void run () override {
		this->initializeData();
		this->execCpuKernel();
		CHECK(arrsEquals(this->rank, this->assertion, this->baseData->result));


#ifdef __CUDACC__
		this->initializeData();
		nd4j::array::NDArrays<T>::allocateNDArrayOnGpu(&this->data);
		printf("About to exec cuda kernel\n");
		this->executeCudaKernel();
		checkCudaErrors(cudaDeviceSynchronize());
		nd4j::buffer::copyDataFromGpu(&this->result->data);
		CHECK(arrsEquals(this->rank, this->assertion, this->result->data->data));

#endif


	}


protected:
	functions::reduce::ReduceOpFactory<T> *opFactory;
	functions::reduce::ReduceFunction<T> *reduce;
};

class DoubleReduceTest : public  ReduceTest<double> {
public:
	virtual ~DoubleReduceTest() {}
	DoubleReduceTest() {}
	DoubleReduceTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() override {
#ifdef __CUDACC__
		nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->shape);
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->result->rank,this->result->shape->data);

		reduceDouble<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->length,
				this->data->data->gData,
				xShapeBuff->gData,
				extraParamsBuff->gData,
				this->result->data->gData,
				resultShapeBuff->gData,
				gpuInfo->gData,
				dimensionBuffer->gData,
				this->baseData->dimensionLength,
				1
		);

		nd4j::buffer::freeBuffer(&gpuInfo);
		nd4j::buffer::freeBuffer(&dimensionBuffer);
		nd4j::buffer::freeBuffer(&xShapeBuff);
		nd4j::buffer::freeBuffer(&resultShapeBuff);
#endif


	}
};


class FloatReduceTest : public ReduceTest<float> {
public:
	FloatReduceTest() {}
	FloatReduceTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() override {
#ifdef __CUDACC__
		nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->shape);
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->result->rank,this->result->shape->data);

		reduceFloat<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->length,
				this->data->data->gData,
				xShapeBuff->gData,
				extraParamsBuff->gData,
				this->result->data->gData,
				resultShapeBuff->gData,
				gpuInfo->gData,
				dimensionBuffer->gData,
				this->baseData->dimensionLength,
				1
		);

		nd4j::buffer::freeBuffer(&gpuInfo);
		nd4j::buffer::freeBuffer(&dimensionBuffer);
		nd4j::buffer::freeBuffer(&xShapeBuff);
		nd4j::buffer::freeBuffer(&resultShapeBuff);
#endif
	}
};

Data<double> * getData(double *assertion,double startingVal) {
	Data<double> *ret = new Data<double>();

	int rank = 2;
	int length = 4;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 1;
	shape[1] = length;
	ret->xShape = shape;
	ret->rank = 2;
    ret->data = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		ret->data[i] = i + 1;
    double *extraParams = (double *) malloc(sizeof(double));
	extraParams[0] = startingVal;
	ret->extraParams = extraParams;

	ret->assertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 1; i++) {
		printf("Assertion value %f\n",assertion[i]);
		ret->assertion[i] = assertion[i];
	}

	ret->dimension = (int *) malloc(sizeof(int) * 2);
	ret->dimension[0] = shape::MAX_DIMENSION;

	ret->result = (double *) malloc(sizeof(double));
	ret->resultRank = 2;
	ret->resultShape = (int *) malloc(sizeof(int) * 2);
	for(int i = 0; i < 2; i++)
		ret->resultShape[i] = 1;

	return ret;
}

Data<double> * getDataDimension() {
	Data<double> *ret = new Data<double>();
	return ret;
}

TEST(Reduce,ObjectOrientedSum) {
	int opNum = 1;
	double comparison[1] = {10};
	Data<double> *data = getData(comparison,0);
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}

/*
TEST(Reduce, Sum) {
	functions::reduce::ReduceOpFactory<double> *opFactory5 =
			new functions::reduce::ReduceOpFactory<double>();
	functions::reduce::ReduceFunction<double> *sum = opFactory5->create(1);
	CHECK(sum != NULL);
	int length = 4;
	double *data = (double *) malloc(sizeof(double) * length);
	for (int i = 0; i < length; i++) {
		data[i] = i + 1;
	}
	int *resultShapeInfo = shape::createScalarShapeInfo();
	assertBufferProperties(resultShapeInfo);

	shape::ShapeInformation *shapeInfo = (shape::ShapeInformation *) malloc(
			sizeof(shape::ShapeInformation));
	int rank = 2;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 1;
	shape[1] = length;
	int *stride = shape::calcStrides(shape, rank);
	shapeInfo->shape = shape;
	shapeInfo->stride = stride;
	shapeInfo->offset = 0;
	shapeInfo->elementWiseStride = 1;

	int *shapeBuffer = shape::toShapeBuffer(shapeInfo);
	assertBufferProperties(shapeBuffer);
	double *extraParams = (double *) malloc(sizeof(double));
	extraParams[0] = 0.0;

	double *result = (double *) malloc(sizeof(double));
	result[0] = 0.0;
	sum->exec(data, shapeBuffer, extraParams, result, resultShapeInfo);
	double comp = result[0];
	CHECK(10.0 == comp);


	int dimensionLength = 1;
	int *dimension = (int *) malloc(sizeof(int) * dimensionLength);
	dimension[0] = shape::MAX_DIMENSION;


	nd4j::buffer::Buffer<double> *dataBuff = nd4j::buffer::createBuffer(data,length);
	nd4j::buffer::Buffer<int> *xShapeInfoBuff = nd4j::buffer::createBuffer(shapeBuffer,shape::shapeInfoLength(shapeInfo->rank));
	nd4j::buffer::Buffer<double> *extraParamsBuff = nd4j::buffer::createBuffer(extraParams,1);
	nd4j::buffer::Buffer<double> *resultBuffer = nd4j::buffer::createBuffer(result,1);
	nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(dimension,dimensionLength);
	nd4j::buffer::Buffer<int> *resultShapeBuff = nd4j::buffer::createBuffer(resultShapeInfo,shape::shapeInfoLength(2));


#ifdef __CUDACC__

	 * reduceDouble(
		int op,
		int n,
		double *dx,
		int *xShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *gpuInformation,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot)

	int blockSize = 500;
	int gridSize = 256;
	int sMemSize = 20000;
	nd4j::buffer::freeBuffer(&resultBuffer);
	//realloc the buffer
	result = (double *) malloc(sizeof(double) * blockSize);
	resultBuffer = nd4j::buffer::createBuffer(result,blockSize);
	int *gpuInformation = (int *) malloc(sizeof(int) * 4);
	gpuInformation[0] = blockSize;
	gpuInformation[1] = gridSize;
	gpuInformation[2] = sMemSize;
	gpuInformation[3] = 49152;
	nd4j::buffer::Buffer<int> *gpuInfoBuff = nd4j::buffer::createBuffer<int>(gpuInformation,4);

	reduceDouble<<<blockSize,gridSize,sMemSize>>>(
			1,
			length,
			dataBuff->gData,
			xShapeInfoBuff->gData,
			extraParamsBuff->gData,
			resultBuffer->gData,
			resultShapeBuff->gData,
			gpuInfoBuff->gData,
			dimensionBuffer->gData,
			dimensionLength,
			1
	);

	checkCudaErrors(cudaDeviceSynchronize());
	nd4j::buffer::copyDataFromGpu(&resultBuffer);
	double resultFinal = sum->aggregateBuffer(length,result,extraParams);
	CHECK(10.0 == result[0]);
#endif


	nd4j::buffer::freeBuffer(&resultShapeBuff);
	nd4j::buffer::freeBuffer(&dimensionBuffer);
	nd4j::buffer::freeBuffer(&dataBuff);
	nd4j::buffer::freeBuffer(&xShapeInfoBuff);
	nd4j::buffer::freeBuffer(&extraParamsBuff);
	nd4j::buffer::freeBuffer(&resultBuffer);
	delete sum;
	delete opFactory5;

}
TEST(Reduce,DimensionSum) {
	functions::reduce::ReduceOpFactory<double> *opFactory5 =
			new functions::reduce::ReduceOpFactory<double>();
	functions::reduce::ReduceFunction<double> *sum = opFactory5->create(1);
	CHECK(sum != NULL);
	int length = 4;
	double *data = (double *) malloc(sizeof(double) * length);
	for (int i = 0; i < length; i++) {
		data[i] = i + 1;
	}
	int *resultShapeInfo = shape::createScalarShapeInfo();

	shape::ShapeInformation *shapeInfo = (shape::ShapeInformation *) malloc(
			sizeof(shape::ShapeInformation));
	int rank = 2;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	int *stride = shape::calcStrides(shape, rank);
	shapeInfo->shape = shape;
	shapeInfo->stride = stride;
	shapeInfo->offset = 0;
	shapeInfo->elementWiseStride = 1;

	int *shapeBuffer = shape::toShapeBuffer(shapeInfo);
	double *extraParams = (double *) malloc(sizeof(double));
	extraParams[0] = 0.0;

	int resultLength = 2;
	double *result = (double *) malloc(resultLength * sizeof(double));
	for (int i = 0; i < resultLength; i++)
		result[i] = 0.0;
	int dimensionLength = 1;
	int *dimension = (int *) malloc(dimensionLength * sizeof(int));
	dimension[0] = 1;

	sum->exec(data, shapeBuffer, extraParams, result, resultShapeInfo,
			dimension, dimensionLength);
	double *comp = (double *) malloc(sizeof(double) * resultLength);
	comp[0] = 3.0;
	comp[1] = 7.0;
	CHECK(arrsEquals(2, comp, result));
	free(extraParams);
	free(comp);
	free(dimension);
	free(shapeBuffer);
	free(shapeInfo);
	delete sum;
	free(data);
	delete opFactory5;
}*/

#endif /* REDUCETESTS_H_ */
