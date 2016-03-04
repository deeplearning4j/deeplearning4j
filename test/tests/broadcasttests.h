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


template <typename T>
static Data<T> * getDataBroadcast(int rank,T *comparison) {
	Data<T> *ret = new Data<T>();
	ret->rank = rank;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	ret->xShape = shape;
	int *vectorShape = (int *) malloc(rank * sizeof(int));
	vectorShape[0] = 1;
	vectorShape[1] = 2;
	ret->yRank = 2;
	ret->yShape = vectorShape;
	ret->y = (T *) malloc(sizeof(T) * 2);

	int dimensionLength = 1;
	int *dimension = (int *) malloc(sizeof(int) * dimensionLength);
	dimension[0] = 1;
	ret->dimension = dimension;
	ret->dimensionLength = 1;
	ret->assertion = (T *) malloc(sizeof(T) * 4);
	ret->resultShape = (int *) malloc(sizeof(int) * 2);
	for(int i = 0; i < 2; i++)
		ret->resultShape[i] = 2;
	ret->result = (T *) malloc(sizeof(T) * 4);
	for(int i = 0; i < 4; i++)
		ret->result[i] = comparison[i];
	ret->extraParams = (T *) malloc(sizeof(T));
	return ret;

}






template <typename T>
class BroadcastingTest : public PairWiseTest<T> {

public:
	BroadcastingTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
: PairWiseTest<T>(rank,opNum,data,extraParamsLength) {
		createOperationAndOpFactory();
	}
	BroadcastingTest() {
		createOperationAndOpFactory();
	}
	virtual ~BroadcastingTest() {
		freeOpAndOpFactory();
	}
	void freeOpAndOpFactory() override {
		delete opFactory;
		delete op;
	}

	virtual void createOperationAndOpFactory() override {
		opFactory = new functions::broadcast::BroadcastOpFactory<T>();
		op = opFactory->getOp(this->opNum);
	}

	virtual void execCpuKernel() override {
		int *shapeBuff = shapeBuffer(this->rank,this->baseData->xShape);
		int *yShapeBuff = shapeBuffer(this->yRank,this->baseData->yShape);
		int *resultShapeBuff = shapeBuffer(this->baseData->resultRank,this->baseData->resultShape);
		op->exec(this->data->data->data,
				shapeBuff,
				this->baseData->y,
				yShapeBuff,
				this->baseData->result,
				resultShapeBuff,
				this->baseData->dimension,
				this->baseData->dimensionLength);
		free(shapeBuff);
		free(yShapeBuff);
		free(resultShapeBuff);
	}



	virtual void run () override {
		this->initializeData();
		this->execCpuKernel();
		CHECK(arrsEquals(this->rank, this->assertion, this->result->data->data));


#ifdef __CUDACC__
this->initializeData();
nd4j::array::NDArrays<T>::allocateNDArrayOnGpu(&this->data);
this->executeCudaKernel();
checkCudaErrors(cudaDeviceSynchronize());
nd4j::buffer::copyDataFromGpu(&this->result->data);
CHECK(arrsEquals(this->rank, this->assertion, this->result->data->data));

#endif


	}
protected:
	functions::broadcast::BroadcastOpFactory<T> *opFactory;
	functions::broadcast::Broadcast<T> *op;
};


class FloatBroadcastTranformTest : public BroadcastingTest<float> {
public:
	FloatBroadcastTranformTest() {}
	FloatBroadcastTranformTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
	:  BroadcastingTest<float>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() {
#ifdef __CUDACC__
		nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->baseData->xShape);
		nd4j::buffer::Buffer<int> *yShapeBuff = shapeIntBuffer(this->rank,this->baseData->xShape);
		broadcastFloat<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->data->data->gData,
				xShapeBuff->gData,
				this->yData->data->gData,
				yShapeBuff->gData,
				this->data->data->gData,
				xShapeBuff->gData,
				dimensionBuffer->gData,
				this->baseData->dimensionLength,
				gpuInfo->gData);
		nd4j::buffer::freeBuffer(&dimensionBuffer);
		nd4j::buffer::freeBuffer(&xShapeBuff);
		nd4j::buffer::freeBuffer(&yShapeBuff);
		nd4j::buffer::freeBuffer(&gpuInfo);
#endif
	}
};



class DoubleBroadcastTranformTest : public BroadcastingTest<double> {
public:
	DoubleBroadcastTranformTest() {}
	DoubleBroadcastTranformTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
	:  BroadcastingTest<double>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() {
#ifdef __CUDACC__
		nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->baseData->xShape);
		nd4j::buffer::Buffer<int> *yShapeBuff = shapeIntBuffer(this->rank,this->baseData->xShape);
		broadcastDouble<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->data->data->gData,
				xShapeBuff->gData,
				this->yData->data->gData,
				yShapeBuff->gData,
				this->data->data->gData,
				xShapeBuff->gData,
				dimensionBuffer->gData,
				this->baseData->dimensionLength,
				gpuInfo->gData);
		nd4j::buffer::freeBuffer(&dimensionBuffer);
		nd4j::buffer::freeBuffer(&xShapeBuff);
		nd4j::buffer::freeBuffer(&yShapeBuff);
		nd4j::buffer::freeBuffer(&gpuInfo);
#endif
	}
};


TEST(BroadCasting,ObjectOrientedAddition) {
	int rank = 2;
	int opNum = 0;
	double comparison[4] = {2,4,5,6};
	Data<double> *data = getDataBroadcast<double>(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}


TEST(BroadCasting,ObjectOrientedSubtraction) {
	int rank = 2;
	int opNum = 1;
	double comparison[4] = {0,0,2,2};
	Data<double> *data = getDataBroadcast<double>(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedMultiplication) {
	int rank = 2;
	int opNum = 2;
	double comparison[4] = {1,4,3,8};
	Data<double> *data = getDataBroadcast<double>(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedDivision) {
	int rank = 2;
	int opNum = 3;
	double comparison[4] = {1,1,3,2};
	Data<double> *data = getDataBroadcast<double>(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedReverseDivision) {
	int rank = 2;
	int opNum = 3;
	double comparison[4] = {1,1,0.33333333,0.5};
	Data<double> *data = getDataBroadcast<double>(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}


TEST(BroadCasting,ObjectOrientedReverseSubtraction) {
	int rank = 2;
	int opNum = 5;
	double comparison[4] = {0,0,-2,-2};
	Data<double> *data = getDataBroadcast<double>(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedCopy) {
	int rank = 2;
	int opNum = 6;
	double comparison[4] = {1,2,1,2};
	Data<double> *data = getDataBroadcast<double>(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}


TEST(BroadCasting,ObjectOrientedFloatAddition) {
	int rank = 2;
	int opNum = 0;
	float comparison[4] = {2,4,5,6};
	Data<float> *data = getDataBroadcast<float>(rank,comparison);
	FloatBroadcastTranformTest *test = new FloatBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}


TEST(BroadCasting,ObjectOrientedFloatSubtraction) {
	int rank = 2;
	int opNum = 1;
	float comparison[4] = {0,0,2,2};
	Data<float> *data = getDataBroadcast<float>(rank,comparison);
	FloatBroadcastTranformTest *test = new FloatBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedFloatMultiplication) {
	int rank = 2;
	int opNum = 2;
	float comparison[4] = {1,4,3,8};
	Data<float> *data = getDataBroadcast<float>(rank,comparison);
	FloatBroadcastTranformTest *test = new FloatBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedFloatDivision) {
	int rank = 2;
	int opNum = 3;
	float comparison[4] = {1,1,3,2};
	Data<float> *data = getDataBroadcast<float>(rank,comparison);
	FloatBroadcastTranformTest *test = new FloatBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedFloatReverseDivision) {
	int rank = 2;
	int opNum = 3;
	float comparison[4] = {1,1,0.33333333,0.5};
	Data<float> *data = getDataBroadcast<float>(rank,comparison);
	FloatBroadcastTranformTest *test = new FloatBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}


TEST(BroadCasting,ObjectOrientedFloatReverseSubtraction) {
	int rank = 2;
	int opNum = 5;
	float comparison[4] = {0,0,-2,-2};
	Data<float> *data = getDataBroadcast<float>(rank,comparison);
	FloatBroadcastTranformTest *test = new FloatBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedFloatCopy) {
	int rank = 2;
	int opNum = 6;
	float comparison[4] = {1,2,1,2};
	Data<float> *data = getDataBroadcast<float>(rank,comparison);
	FloatBroadcastTranformTest *test = new FloatBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

#endif //NATIVEOPERATIONS_BROADCASTSTESTS_H
