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
		T *resultData = this->result->data->data;
		reduce->exec(
				this->baseData->data,
				xShapeBuff,
				this->baseData->extraParams,
				resultData,
				resultShapeBuff,
				this->baseData->dimension,
				this->baseData->dimensionLength);
		free(xShapeBuff);
		free(resultShapeBuff);
	}

	virtual void run () override {
		this->initializeData();
		this->execCpuKernel();
		int resultLength = shape::prod(this->baseData->resultShape,this->baseData->rank);
		if(resultLength == 1) {
			if(this->result->data->data[0] != this->baseData->assertion[0]) {
				printf("Compared assertion %f to result %f\n",this->baseData->assertion[0],this->result->data->data[0]);
			}
			DOUBLES_EQUAL(this->baseData->assertion[0],this->result->data->data[0],1e-3);
		}
		else {
			CHECK(arrsEquals(this->rank, this->assertion, this->result->data->data));
		}

#ifdef __CUDACC__
		this->initializeData();
		nd4j::array::NDArrays<T>::allocateNDArrayOnGpu(&this->data);
		printf("About to exec cuda kernel\n");
		this->executeCudaKernel();
		checkCudaErrors(cudaDeviceSynchronize());
		nd4j::buffer::copyDataFromGpu(&this->result->data);
		if(resultLength == 1) {
			if(this->result->data->data[0] != this->baseData->assertion[0]) {
				printf("Compared assertion gpu %f to result %f\n",this->baseData->assertion[0],this->baseData->result[0]);
			}
			DOUBLES_EQUAL(this->baseData->assertion[0],this->result->data->data[0],1e-3);
		}
		else
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
	double *extraParams = (double *) malloc(sizeof(double) * 4);
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

Data<double> * getDataDimension(double *assertion,double startingVal) {
	Data<double> *ret = new Data<double>();

	int rank = 2;
	int length = 4;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	ret->xShape = shape;
	ret->rank = 2;
	ret->data = (double *) malloc(sizeof(double) * length);
	for(int i = 0; i < 4; i++)
		ret->data[i] = i + 1;
	double *extraParams = (double *) malloc(sizeof(double) * 4);
	extraParams[0] = startingVal;
	ret->extraParams = extraParams;

	ret->assertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 2; i++) {
		printf("Assertion value %f\n",assertion[i]);
		ret->assertion[i] = assertion[i];
	}

	ret->dimension = (int *) malloc(sizeof(int) * 2);
	ret->dimension[0] = 1;
	ret->dimensionLength = 1;
	ret->result = (double *) malloc(sizeof(double));
	ret->resultRank = 2;
	ret->resultShape = (int *) malloc(sizeof(int) * 2);
	for(int i = 0; i < 2; i++)
		ret->resultShape[i] = 1;
	ret->resultShape[1] = 2;

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

TEST(Reduce,ObjectOrientedMax) {
	int opNum = 3;
	double comparison[1] = {4};
	Data<double> *data = getData(comparison,0);
	data->extraParams[0] = data->data[0];
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}



TEST(Reduce,ObjectOrientedMin) {
	int opNum = 4;
	double comparison[1] = {1};
	Data<double> *data = getData(comparison,0);
	data->extraParams[0] = data->data[0];
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}


TEST(Reduce,ObjectOrientedNorm1) {
	int opNum = 5;
	double comparison[1] = {10.0};
	Data<double> *data = getData(comparison,0);
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}


TEST(Reduce,ObjectOrientedNorm2) {
	int opNum = 6;
	double comparison[1] = {5.4772255750516612};
	Data<double> *data = getData(comparison,0);
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}


TEST(Reduce,ObjectOrientedMean) {
	int opNum = 0;
	double comparison[1] = {2.5};
	Data<double> *data = getData(comparison,0);
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}



TEST(Reduce,ObjectOrientedProd) {
	int opNum = 8;
	double comparison[1] = {24};
	Data<double> *data = getData(comparison,0);
	data->extraParams[0] = 1.0;
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}









TEST(Reduce,ObjectOrientedDimensionMax) {
	int opNum = 3;
	double comparison[2] = {2.00, 4.00};
	Data<double> *data = getDataDimension(comparison,0);
	data->extraParams[0] = data->data[0];
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}


TEST(Reduce,ObjectOrientedDimensionSum) {
	int opNum = 1;
	double comparison[2] = {3,7};

	Data<double> *data = getDataDimension(comparison,0);
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}

TEST(Reduce,ObjectOrientedDimensionMin) {
	int opNum = 4;
	double comparison[2] = {1.00, 3.00};
	Data<double> *data = getDataDimension(comparison,0);
	data->extraParams[0] = data->data[0];
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}



TEST(Reduce,ObjectOrientedDimensionNorm1) {
	int opNum = 5;
	double comparison[2] = {3.00, 7.00};
	Data<double> *data = getDataDimension(comparison,0);
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}





TEST(Reduce,ObjectOrientedDimensionMean) {
	int opNum = 0;
	double comparison[2] = {1.50, 3.50};
	Data<double> *data = getDataDimension(comparison,0);
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}

TEST(Reduce,ObjectOrientedDimensionProd) {
	int opNum = 8;
	double comparison[2] = {2.00, 12.00};
	Data<double> *data = getDataDimension(comparison,0);
	data->extraParams[0] = 1.0;
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}

TEST(Reduce,ObjectOrientedDimensionNorm2) {
	int opNum = 6;
	double comparison[2] = {2.24, 5.00};
	Data<double> *data = getDataDimension(comparison,0);
	//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
	DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
	test->run();
	delete test;
	delete data;
}





#endif /* REDUCETESTS_H_ */
