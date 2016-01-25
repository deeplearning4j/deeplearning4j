/*
 * summarystatsreducetest.h
 *
 *  Created on: Jan 23, 2016
 *      Author: agibsonccc
 */

#ifndef SUMMARYSTATSREDUCETEST_H_
#define SUMMARYSTATSREDUCETEST_H_

#include <array.h>
#include "testhelpers.h"
#include <summarystatsreduce.h>
#include <helper_cuda.h>
TEST_GROUP(SummaryStatsReduce) {

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
static Data<T> * getDataSummary(T *assertion,T startingVal) {
	Data<T> *ret = new Data<T>();

	int rank = 2;
	int length = 4;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 1;
	shape[1] = length;
	ret->xShape = shape;
	ret->rank = 2;
	ret->data = (T *) malloc(sizeof(T) * 4);
	for(int i = 0; i < 4; i++)
		ret->data[i] = i + 1;
	T *extraParams = (T *) malloc(sizeof(T) * 4);
	extraParams[0] = startingVal;
	ret->extraParams = extraParams;

	ret->assertion = (T *) malloc(sizeof(T) * 4);
	for(int i = 0; i < 1; i++) {
		printf("Assertion value %f\n",assertion[i]);
		ret->assertion[i] = assertion[i];
	}

	ret->dimension = (int *) malloc(sizeof(int) * 2);
	ret->dimension[0] = shape::MAX_DIMENSION;

	ret->result = (T *) malloc(sizeof(T));
	ret->resultRank = 2;
	ret->resultShape = (int *) malloc(sizeof(int) * 2);
	for(int i = 0; i < 2; i++)
		ret->resultShape[i] = 1;

	return ret;
}

template <typename T>
static Data<T> * getDataSummaryDimension(T *assertion,T startingVal) {
	Data<T> *ret = new Data<T>();

	int rank = 2;
	int length = 4;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	ret->xShape = shape;
	ret->rank = 2;
	ret->data = (T *) malloc(sizeof(T) * length);
	for(int i = 0; i < 4; i++)
		ret->data[i] = i + 1;
	T *extraParams = (T *) malloc(sizeof(T) * 4);
	extraParams[0] = startingVal;
	ret->extraParams = extraParams;

	ret->assertion = (T *) malloc(sizeof(T) * 4);
	for(int i = 0; i < 2; i++) {
		printf("Assertion value %f\n",assertion[i]);
		ret->assertion[i] = assertion[i];
	}

	ret->dimension = (int *) malloc(sizeof(int) * 2);
	ret->dimension[0] = 1;
	ret->dimensionLength = 1;
	ret->result = (T *) malloc(sizeof(T));
	ret->resultRank = 2;
	ret->resultShape = (int *) malloc(sizeof(int) * 2);
	for(int i = 0; i < 2; i++)
		ret->resultShape[i] = 1;
	ret->resultShape[1] = 2;

	return ret;
}


template <typename T>
class SummaryStatsReduceTest : public BaseTest<T> {
public:
	SummaryStatsReduceTest() {}
	virtual ~SummaryStatsReduceTest() {
		freeOpAndOpFactory();
	}
	SummaryStatsReduceTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
	:  BaseTest<T>(rank,opNum,data,extraParamsLength){
		createOperationAndOpFactory();
	}
	void freeOpAndOpFactory() {
		delete opFactory;
		delete reduce;
	}

	virtual void createOperationAndOpFactory() {
		opFactory = new functions::summarystats::SummaryStatsReduceOpFactory<T>();
		reduce = opFactory->getOp(this->opNum);
	}

	virtual void execCpuKernel() override {
		int *xShapeBuff = shapeBuffer(this->baseData->rank,this->baseData->xShape);
		int *resultShapeBuff = shapeBuffer(this->baseData->resultRank,this->baseData->resultShape);
		printf("In exec cpu\n");
		reduce->exec(
				this->data->data->data,
				xShapeBuff,
				this->baseData->extraParams,
				this->result->data->data,
				resultShapeBuff,
				this->baseData->dimension,this->baseData->dimensionLength);
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
				for(int i = 0; i < resultLength; i++) {
					printf("Result[%d] is %f\n",i,this->result->data->data[i]);
				}
				printf("Compared assertion gpu %f to result %f\n",this->baseData->assertion[0],this->baseData->result[0]);
			}
			DOUBLES_EQUAL(this->baseData->assertion[0],this->result->data->data[0],1e-3);
		}
		else {
			for(int i = 0; i < resultLength; i++) {
				printf("Result[%d] is %f\n",i,this->result->data->data[i]);
			}
			CHECK(arrsEquals(this->rank, this->assertion, this->result->data->data));

		}

#endif


	}


protected:
	functions::summarystats::SummaryStatsReduceOpFactory<T> *opFactory;
	functions::summarystats::SummaryStatsReduce<T> *reduce;
};

class DoubleSummaryStatsReduceTest : public SummaryStatsReduceTest<double> {
public:
	virtual ~DoubleSummaryStatsReduceTest() {}
	DoubleSummaryStatsReduceTest() {}
	DoubleSummaryStatsReduceTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
	:  SummaryStatsReduceTest<double>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() override {
#ifdef __CUDACC__
		nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->shape);
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->result->rank,this->result->shape->data);

		summaryStatsReduceDouble<<<this->blockSize,this->gridSize,this->sMemSize>>>(
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


class FloatSummaryStatsReduceTest : public SummaryStatsReduceTest<float> {
public:
	FloatSummaryStatsReduceTest() {}
	FloatSummaryStatsReduceTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
	:  SummaryStatsReduceTest<float>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() override {
#ifdef __CUDACC__
		nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->shape);
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->result->rank,this->result->shape->data);

		summaryStatsReduceFloat<<<this->blockSize,this->gridSize,this->sMemSize>>>(
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

TEST(SummaryStatsReduce,ObjectOrientedStandardDeviation) {
	int rank = 2;
	int opNum = 1;
	double assertion[1] = {1.29099440574646};
	Data<double> *data = getDataSummary<double>(assertion,0);
	DoubleSummaryStatsReduceTest *test = new DoubleSummaryStatsReduceTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}




TEST(SummaryStatsReduce,ObjectOrientedVariance) {
	int rank = 2;
	int opNum = 0;
	double assertion[1] = {1.66667};
	Data<double> *data = getDataSummary<double>(assertion,0);
	DoubleSummaryStatsReduceTest *test = new DoubleSummaryStatsReduceTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}




TEST(SummaryStatsReduce,ObjectOrientedDimensionStandardDeviation) {
	int rank = 2;
	int opNum = 1;
	double assertion[2] = { 0.71, 0.71};
	Data<double> *data = getDataSummaryDimension<double>(assertion,0);
	DoubleSummaryStatsReduceTest *test = new DoubleSummaryStatsReduceTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}

TEST(SummaryStatsReduce,ObjectOrientedDimensionVariance) {
	int rank = 2;
	int opNum = 0;
	double assertion[2] = {0.50, 0.50};
	Data<double> *data = getDataSummaryDimension<double>(assertion,0);
	DoubleSummaryStatsReduceTest *test = new DoubleSummaryStatsReduceTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}

TEST(SummaryStatsReduce,ObjectOrientedFloatStandardDeviation) {
	int rank = 2;
	int opNum = 1;
	float assertion[1] = {1.29099440574646};
	Data<float> *data = getDataSummary<float>(assertion,0);
	FloatSummaryStatsReduceTest *test = new FloatSummaryStatsReduceTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}




TEST(SummaryStatsReduce,ObjectOrientedFloatVariance) {
	int rank = 2;
	int opNum = 0;
	float assertion[1] = {1.66667};
	Data<float> *data = getDataSummary<float>(assertion,0);
	FloatSummaryStatsReduceTest *test = new FloatSummaryStatsReduceTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}




TEST(SummaryStatsReduce,ObjectOrientedFloatDimensionStandardDeviation) {
	int rank = 2;
	int opNum = 1;
	float assertion[2] = { 0.71, 0.71};
	Data<float> *data = getDataSummaryDimension<float>(assertion,0);
	FloatSummaryStatsReduceTest *test = new FloatSummaryStatsReduceTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}

TEST(SummaryStatsReduce,ObjectOrientedFloatDimensionVariance) {
	int rank = 2;
	int opNum = 0;
	float assertion[2] = {0.50, 0.50};
	Data<float> *data = getDataSummaryDimension<float>(assertion,0);
	FloatSummaryStatsReduceTest *test = new FloatSummaryStatsReduceTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}



#endif /* SUMMARYSTATSREDUCETEST_H_ */
