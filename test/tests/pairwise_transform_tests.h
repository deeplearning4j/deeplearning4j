//
// Created by agibsonccc on 1/3/16.
//

#ifndef NATIVEOPERATIONS_PAIRWISE_TRANSFORM_TESTS_H
#define NATIVEOPERATIONS_PAIRWISE_TRANSFORM_TESTS_H

#include <pairwise_transform.h>
#include <array.h>
#include <shape.h>
#include <buffer.h>
#include "testhelpers.h"

static functions::pairwise_transforms::PairWiseTransformOpFactory<double> *opFactory2 =
		0;

TEST_GROUP(PairWiseTransform) {
	static int output_method(const char* output, ...) {
		va_list arguments;
		va_start(arguments, output);
		va_end(arguments);
		return 1;
	}
	void setup() {
		opFactory2 =
				new functions::pairwise_transforms::PairWiseTransformOpFactory<
				double>();

	}
	void teardown() {
		delete opFactory2;
	}
};

template <typename T>
class PairwiseTransformTest : public PairWiseTest<T> {

public:
	virtual ~PairwiseTransformTest() {
		freeOpAndOpFactory();
	}
	PairwiseTransformTest() {
		createOperationAndOpFactory();
	}
	PairwiseTransformTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
	: PairWiseTest<T>(rank,opNum,data,extraParamsLength) {
		createOperationAndOpFactory();
	}

	virtual void execCpuKernel() override {
		int *shapeBuff = shapeBuffer(this->rank,this->data->shape->data);
		int *yShapeBuff = shapeBuffer(this->yData->rank,this->yData->shape->data);
		int *resultShapeBuff = shapeBuffer(this->result->rank,this->result->shape->data);
		int length = shape::length(shapeBuff);
		assertBufferProperties(shapeBuff);
		assertBufferProperties(yShapeBuff);
		assertBufferProperties(resultShapeBuff);

		int xEleStride = shape::elementWiseStride(shapeBuff);
		int yEleStride = shape::elementWiseStride(yShapeBuff);
		int resultEleStride = shape::elementWiseStride(resultShapeBuff);

		op->exec(
				this->data->data->data,
				xEleStride,
				this->yData->data->data,
				yEleStride,
				this->result->data->data,
				resultEleStride,
				this->extraParams,
				length);

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
		nd4j::buffer::copyDataFromGpu(&this->data->data);
		CHECK(arrsEquals(this->rank, this->assertion, this->result->data->data));

#endif


	}

	virtual void freeOpAndOpFactory() override {
		if(op != NULL)
			delete op;
		if(opFactory != NULL)
			delete opFactory;
	}

	virtual void createOperationAndOpFactory() override {
		opFactory = new functions::pairwise_transforms::PairWiseTransformOpFactory<T>();
		op = opFactory->getOp(this->opNum);
	}

protected:
	functions::pairwise_transforms::PairWiseTransformOpFactory<T> *opFactory;
	functions::pairwise_transforms::PairWiseTransform<T> *op;

};

template <typename T>
Data<T> * getPairwiseData(T *assertion,int opNum) {
	int rank = 2;
	int yRank = 2;
	int resultRank = 2;
	int length = 4;
	Data<T> *data = new Data<T>();
	data->rank = rank;
	data->yRank = yRank;
	data->resultRank = resultRank;
	data->xShape = (int *) malloc(sizeof(int) * rank);
	data->yShape = (int *) malloc(sizeof(int) * yRank);
	data->resultShape = (int *) malloc(sizeof(int) * yRank);

	for(int i = 0; i < 2; i++) {
		data->xShape[i] = 2;
		data->yShape[i] = 2;
		data->resultShape[i] = 2;
	}


	T *extraParams = (T *) malloc(sizeof(T) * 2);
	data->extraParams = extraParams;
	T *y = (T *) malloc(sizeof(T) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}

	T *comparisionAssertion = (T *) malloc(sizeof(T) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = assertion[i];
	data->assertion = comparisionAssertion;

	data->result = (T *) malloc(sizeof(T) * length);

	return data;
}


class DoublePairwiseTranformTest : public PairwiseTransformTest<double> {
public:
	DoublePairwiseTranformTest() {}
	DoublePairwiseTranformTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
	:  PairwiseTransformTest<double>(rank,opNum,data,extraParamsLength){
	}


	virtual void executeCudaKernel() {
#ifdef __CUDACC__
		int *shapeBuff = shapeBuffer(this->rank,this->baseData->xShape);
		int *yShapeBuff = shapeBuffer(this->rank,this->baseData->yShape);
		assertBufferProperties(shapeBuff);
		assertBufferProperties(yShapeBuff);

		nd4j::buffer::Buffer<int> *shapeBuffBuff = nd4j::buffer::createBuffer(shapeBuff,this->rank);
		nd4j::buffer::Buffer<int> *yShapeBuffBuff = nd4j::buffer::createBuffer(yShapeBuff,this->yRank);
		pairWiseTransformDouble<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->length,
				this->data->data->gData,
				this->yData->data->gData,
				this->extraParamsBuff->gData,
				this->data->data->gData,
				shapeBuffBuff->gData,
				yShapeBuffBuff->gData,
				shapeBuffBuff->gData);
		nd4j::buffer::freeBuffer(&shapeBuffBuff);
		nd4j::buffer::freeBuffer(&yShapeBuffBuff);
#endif

	}

};



class FloatPairwiseTranformTest : public PairwiseTransformTest<float> {
public:
	FloatPairwiseTranformTest() {}
	FloatPairwiseTranformTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
	:  PairwiseTransformTest<float>(rank,opNum,data,extraParamsLength) {
	}
	virtual void executeCudaKernel() {
#ifdef __CUDACC__
		int *shapeBuff = shapeBuffer(this->rank,this->baseData->xShape);
		int *yShapeBuff = shapeBuffer(this->rank,this->baseData->yShape);
		assertBufferProperties(shapeBuff);
		assertBufferProperties(yShapeBuff);

		nd4j::buffer::Buffer<int> *shapeBuffBuff = nd4j::buffer::createBuffer(shapeBuff,this->rank);
		nd4j::buffer::Buffer<int> *yShapeBuffBuff = nd4j::buffer::createBuffer(yShapeBuff,this->yRank);
		pairWiseTransformFloat<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->length,
				this->data->data->gData,
				this->yData->data->gData,
				this->extraParamsBuff->gData,
				this->data->data->gData,
				shapeBuffBuff->gData,
				yShapeBuffBuff->gData,
				shapeBuffBuff->gData);
		nd4j::buffer::freeBuffer(&shapeBuffBuff);
		nd4j::buffer::freeBuffer(&yShapeBuffBuff);

#endif

	}
};

TEST(PairWiseTransform,ObjectOrientedAddition) {
	double comparison[4] = {3,5,7,9};

	int rank = 2;
	int opNum = 0;

	Data<double> *data = getPairwiseData<double>(comparison,opNum);
	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedCopy) {
	int rank = 2;
	int opNum = 1;
	double comparison[4] = {2,3,4,5};
	Data<double> *data = getPairwiseData<double>(comparison,opNum);
	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}

TEST(PairWiseTransform,ObjectOrientedDivide) {
	int rank = 2;
	int opNum = 2;
	double comparison[4] = {0.5,  0.66666667,0.75 ,0.8};

	Data<double> *data = getPairwiseData(comparison,opNum);
	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}

TEST(PairWiseTransform,ObjectOrientedEqualTo) {
	int rank = 2;
	int opNum = 3;
	double comparison[4] = {0.0,0.0,0.0,0.0};

	Data<double> *data = getPairwiseData<double>(comparison,opNum);
	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedGreaterThan) {
	int rank = 2;
	int opNum = 4;
	double comparison[4] = {0.0,0.0,0.0,0.0};
	Data<double> *data = getPairwiseData<double>(comparison,opNum);
	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedLessThan) {
	int rank = 2;
	int opNum = 5;
	double comparison[4] = {1.0,1.0,1.0,1.0};
	Data<double> *data = getPairwiseData<double>(comparison,opNum);
	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}



TEST(PairWiseTransform,ObjectOrientedMultiply) {
	int rank = 2;
	int opNum = 6;
	double comparison[4] = {2.,   6.,  12.,  20};
	Data<double> *data = getPairwiseData<double>(comparison,opNum);
	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}

TEST(PairWiseTransform,ObjectOrientedReverseDivide) {
	int rank = 2;
	int opNum = 7;
	double comparison[4] = { 2.,1.5,1.33333333,1.25};
	Data<double> *data = getPairwiseData<double>(comparison,opNum);
	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}



TEST(PairWiseTransform,ObjectOrientedReverseSub) {
	int rank = 2;
	int opNum = 7;
	double comparison[4] = { 2.,1.5,1.33333333,1.25};
	Data<double> *data = getPairwiseData<double>(comparison,opNum);
	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedSubtraction) {
	int rank = 2;
	int opNum = 9;
	double comparison[4] = { -1., -1., -1., -1.};

	Data<double> *data =  getPairwiseData<double>(comparison,opNum);
	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedFloatAddition) {
	float comparison[4] = {3,5,7,9};

	int rank = 2;
	int opNum = 0;
	Data<float> *data = getPairwiseData<float>(comparison,opNum);
	FloatPairwiseTranformTest *test = new FloatPairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedFloatCopy) {
	int rank = 2;
	int opNum = 1;
	float comparison[4] = {2,3,4,5};
	Data<float> *data = getPairwiseData<float>(comparison,opNum);
	FloatPairwiseTranformTest *test = new FloatPairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}

TEST(PairWiseTransform,ObjectOrientedFloatDivide) {
	int rank = 2;
	int opNum = 2;
	float comparison[4] = {0.5,  0.66666667,0.75 ,0.8};

	Data<float> *data = getPairwiseData(comparison,opNum);
	FloatPairwiseTranformTest *test = new FloatPairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}

TEST(PairWiseTransform,ObjectOrientedFloatEqualTo) {
	int rank = 2;
	int opNum = 3;
	float comparison[4] = {0.0,0.0,0.0,0.0};

	Data<float> *data = getPairwiseData<float>(comparison,opNum);
	FloatPairwiseTranformTest *test = new FloatPairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedFloatGreaterThan) {
	int rank = 2;
	int opNum = 4;
	float comparison[4] = {0.0,0.0,0.0,0.0};
	Data<float> *data = getPairwiseData<float>(comparison,opNum);
	FloatPairwiseTranformTest *test = new FloatPairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedFloatLessThan) {
	int rank = 2;
	int opNum = 5;
	float comparison[4] = {1.0,1.0,1.0,1.0};
	Data<float> *data = getPairwiseData<float>(comparison,opNum);
	FloatPairwiseTranformTest *test = new FloatPairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}



TEST(PairWiseTransform,ObjectOrientedFloatMultiply) {
	int rank = 2;
	int opNum = 6;
	float comparison[4] = {2.,   6.,  12.,  20};
	Data<float> *data = getPairwiseData<float>(comparison,opNum);
	FloatPairwiseTranformTest *test = new FloatPairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}

TEST(PairWiseTransform,ObjectOrientedFloatReverseDivide) {
	int rank = 2;
	int opNum = 7;
	float comparison[4] = { 2.,1.5,1.33333333,1.25};
	Data<float> *data = getPairwiseData<float>(comparison,opNum);
	FloatPairwiseTranformTest *test = new FloatPairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}



TEST(PairWiseTransform,ObjectOrientedFloatReverseSub) {
	int rank = 2;
	int opNum = 7;
	float comparison[4] = { 2.,1.5,1.33333333,1.25};
	Data<float> *data = getPairwiseData<float>(comparison,opNum);
	FloatPairwiseTranformTest *test = new FloatPairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedFloatSubtraction) {
	int rank = 2;
	int opNum = 9;
	float comparison[4] = { -1., -1., -1., -1.};
	Data<float> *data =  getPairwiseData<float>(comparison,opNum);
	FloatPairwiseTranformTest *test = new FloatPairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}

#endif //NATIVEOPERATIONS_PAIRWISE_TRANSFORM_TESTS_H
