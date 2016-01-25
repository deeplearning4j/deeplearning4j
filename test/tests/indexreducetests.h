//
// Created by agibsoncccc on 1/5/16.
//

#ifndef NATIVEOPERATIONS_INDEXREDUCETESTS_H_H
#define NATIVEOPERATIONS_INDEXREDUCETESTS_H_H
#include <array.h>
#include "testhelpers.h"
#include <indexreduce.h>
#include <helper_cuda.h>
TEST_GROUP(IndexReduce) {

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

static Data<double> * getDataIndexReduce(double *assertion,double startingVal) {
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

static Data<double> * getDataIndexReduceDimension(double *assertion,double startingVal) {
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


template <typename T>
class IndexReduceTest : public BaseTest<T> {
public:
	IndexReduceTest() {}
	virtual ~IndexReduceTest() {
		freeOpAndOpFactory();
	}
	IndexReduceTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
	:  BaseTest<T>(rank,opNum,data,extraParamsLength){
		createOperationAndOpFactory();
	}
	void freeOpAndOpFactory() {
		delete opFactory;
		delete reduce;
	}

	virtual void createOperationAndOpFactory() {
		opFactory = new functions::indexreduce::IndexReduceOpFactory<T>();
		reduce = opFactory->getOp(this->opNum);
	}

	virtual void execCpuKernel() override {
		int *xShapeBuff = shapeBuffer(this->baseData->rank,this->baseData->xShape);
		int *resultShapeBuff = shapeBuffer(this->baseData->resultRank,this->baseData->resultShape);
		printf("In exec cpu index reduce \n");
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
				printf("Compared assertion index reduce %f to result %f\n",this->baseData->assertion[0],this->result->data->data[0]);
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
	functions::indexreduce::IndexReduceOpFactory<T> *opFactory;
	functions::indexreduce::IndexReduce<T> *reduce;
};

class DoubleIndexReduceTest : public IndexReduceTest<double> {
public:
	virtual ~DoubleIndexReduceTest() {}
	DoubleIndexReduceTest() {}
	DoubleIndexReduceTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
	:  IndexReduceTest<double>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() override {
#ifdef __CUDACC__
		nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->shape);
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->result->rank,this->result->shape->data);

		indexReduceDouble<<<this->blockSize,this->gridSize,this->sMemSize>>>(
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


class FloatIndexReduceTest : public IndexReduceTest<float> {
public:
	FloatIndexReduceTest() {}
	FloatIndexReduceTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
	:  IndexReduceTest<float>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() override {
#ifdef __CUDACC__
		nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->shape);
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->result->rank,this->result->shape->data);

		indexReduceFloat<<<this->blockSize,this->gridSize,this->sMemSize>>>(
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


TEST(IndexReduce,ObjectOrientedIMax) {
	int rank = 2;
	int opNum = 0;
	double assertion[1] = {3};
	Data<double> *data = getDataIndexReduce(assertion,0);
	DoubleIndexReduceTest *test = new DoubleIndexReduceTest(rank,opNum,data,1);
    test->run();
    delete data;
    delete test;
}

TEST(IndexReduce,ObjectOrientedIMin) {
	int rank = 2;
	int opNum = 1;
	double assertion[1] = {0};
	Data<double> *data = getDataIndexReduce(assertion,0);
	DoubleIndexReduceTest *test = new DoubleIndexReduceTest(rank,opNum,data,1);
    test->run();
    delete data;
    delete test;
}




TEST(IndexReduce,ObjectOrientedDimensionIMax) {
	int rank = 2;
	int opNum = 0;
	double assertion[2] = {1,1};
	Data<double> *data = getDataIndexReduceDimension(assertion,0);
	DoubleIndexReduceTest *test = new DoubleIndexReduceTest(rank,opNum,data,1);
    test->run();
    delete data;
    delete test;
}

TEST(IndexReduce,ObjectOrientedDimensionIMin) {
	int rank = 2;
	int opNum = 1;
	double assertion[2] = {0,0};
	Data<double> *data = getDataIndexReduceDimension(assertion,0);
	DoubleIndexReduceTest *test = new DoubleIndexReduceTest(rank,opNum,data,1);
    test->run();
    delete data;
    delete test;
}

#endif //NATIVEOPERATIONS_INDEXREDUCETESTS_H_H
