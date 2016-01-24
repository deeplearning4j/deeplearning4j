//
// Created by agibsonccc on 1/3/16.
//

#ifndef NATIVEOPERATIONS_PAIRWISE_TRANSFORM_TESTS_H
#define NATIVEOPERATIONS_PAIRWISE_TRANSFORM_TESTS_H

#include <pairwise_transform.h>
#include <array.h>
#include <shape.h>
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

class DoublePairwiseTranformTest : public PairwiseTransformTest<double> {
public:
	DoublePairwiseTranformTest() {}
	DoublePairwiseTranformTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
	:  PairwiseTransformTest<double>(rank,opNum,data,extraParamsLength){
	}


	virtual void executeCudaKernel() {
#ifdef __CUDACC__
		int *shapeBuff = shapeBuffer(this->rank,this->shape);
		int *yShapeBuff = shapeBuffer(this->rank,this->yShape);
		assertBufferProperties(shapeBuff);
		assertBufferProperties(yShapeBuff);
		int xOffset = shape::offset(shapeBuff);
		int yOffset = shape::offset(yShapeBuff);
		int xEleStride = shape::elementWiseStride(shapeBuff);
		int yEleStride = shape::elementWiseStride(yShapeBuff);
		pairWiseTransformDouble<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->length,
				xOffset,
				yOffset,
				0,
				this->data->data->gData,
				this->yData->data->gData,
				xEleStride,
				yEleStride,
				this->extraParamsBuff->gData,
				this->data->data->gData,
				1, this->blockSize);
		free(shapeBuff);
		free(yShapeBuff);
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
		int *shapeBuff = shapeBuffer(this->rank,this->shape);
		int *yShapeBuff = shapeBuffer(this->rank,this->yShape);
		assertBufferProperties(shapeBuff);
		assertBufferProperties(yShapeBuff);
		int xOffset = shape::offset(shapeBuff);
		int yOffset = shape::offset(yShapeBuff);
		int xEleStride = shape::elementWiseStride(shapeBuff);
		int yEleStride = shape::elementWiseStride(yShapeBuff);

		pairWiseTransformFloat<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->length,
				xOffset,
				yOffset,
				0,
				this->data->data->gData,
				this->yData->data->gData,
				xEleStride,
				yEleStride,
				this->extraParamsBuff->gData,
				this->data->data->gData,
				1, this->blockSize);
		free(shapeBuff);
		free(yShapeBuff);
#endif
	}
};

TEST(PairWiseTransform,ObjectOrientedAddition) {
	int rank = 2;
	int opNum = 0;
	int yRank = 2;
	int resultRank = 2;
	int length = 4;
	Data<double> *data = new Data<double>();
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


	double *extraParams = (double *) malloc(sizeof(double) * 2);
	data->extraParams = extraParams;
	double *y = (double *) malloc(sizeof(double) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}

	double comparison[4] = {3,5,7,9};
	double *comparisionAssertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = comparison[i];
	data->assertion = comparisionAssertion;

	data->result = (double *) malloc(sizeof(double) * length);

	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedCopy) {
	int rank = 2;
	int opNum = 1;
	int yRank = 2;
	int resultRank = 2;
	int length = 4;
	Data<double> *data = new Data<double>();
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


	double *extraParams = (double *) malloc(sizeof(double) * 2);
	data->extraParams = extraParams;
	double *y = (double *) malloc(sizeof(double) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}

	double comparison[4] = {2,3,4,5};
	double *comparisionAssertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = comparison[i];
	data->assertion = comparisionAssertion;

	data->result = (double *) malloc(sizeof(double) * length);

	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}

TEST(PairWiseTransform,ObjectOrientedDivide) {
	int rank = 2;
	int opNum = 2;
	int yRank = 2;
	int resultRank = 2;
	int length = 4;
	Data<double> *data = new Data<double>();
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


	double *extraParams = (double *) malloc(sizeof(double) * 2);
	data->extraParams = extraParams;
	double *y = (double *) malloc(sizeof(double) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}

	double comparison[4] = {0.5,  0.66666667,0.75 ,0.8};
	double *comparisionAssertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = comparison[i];
	data->assertion = comparisionAssertion;
	data->result = (double *) malloc(sizeof(double) * length);

	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}

TEST(PairWiseTransform,ObjectOrientedEqualTo) {
	int rank = 2;
	int opNum = 3;
	int yRank = 2;
	int resultRank = 2;
	int length = 4;
	Data<double> *data = new Data<double>();
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


	double *extraParams = (double *) malloc(sizeof(double) * 2);
	data->extraParams = extraParams;
	double *y = (double *) malloc(sizeof(double) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}

	double comparison[4] = {0.0,0.0,0.0,0.0};
	double *comparisionAssertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = comparison[i];
	data->assertion = comparisionAssertion;
	data->result = (double *) malloc(sizeof(double) * length);

	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedGreaterThan) {
	int rank = 2;
	int opNum = 4;
	int yRank = 2;
	int resultRank = 2;
	int length = 4;
	Data<double> *data = new Data<double>();
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


	double *extraParams = (double *) malloc(sizeof(double) * 2);
	data->extraParams = extraParams;
	double *y = (double *) malloc(sizeof(double) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}

	double comparison[4] = {0.0,0.0,0.0,0.0};
	double *comparisionAssertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = comparison[i];
	data->assertion = comparisionAssertion;
	data->result = (double *) malloc(sizeof(double) * length);

	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedLessThan) {
	int rank = 2;
	int opNum = 5;
	int yRank = 2;
	int resultRank = 2;
	int length = 4;
	Data<double> *data = new Data<double>();
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


	double *extraParams = (double *) malloc(sizeof(double) * 2);
	data->extraParams = extraParams;
	double *y = (double *) malloc(sizeof(double) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}

	double comparison[4] = {1.0,1.0,1.0,1.0};
	double *comparisionAssertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = comparison[i];
	data->assertion = comparisionAssertion;
	data->result = (double *) malloc(sizeof(double) * length);

	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}



TEST(PairWiseTransform,ObjectOrientedMultiply) {
	int rank = 2;
	int opNum = 6;
	int yRank = 2;
	int resultRank = 2;
	int length = 4;
	Data<double> *data = new Data<double>();
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


	double *extraParams = (double *) malloc(sizeof(double) * 2);
	data->extraParams = extraParams;
	double *y = (double *) malloc(sizeof(double) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}

	double comparison[4] = {2.,   6.,  12.,  20};
	double *comparisionAssertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = comparison[i];
	data->assertion = comparisionAssertion;
	data->result = (double *) malloc(sizeof(double) * length);

	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}

TEST(PairWiseTransform,ObjectOrientedReverseDivide) {
	int rank = 2;
	int opNum = 7;
	int yRank = 2;
	int resultRank = 2;
	int length = 4;
	Data<double> *data = new Data<double>();
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


	double *extraParams = (double *) malloc(sizeof(double) * 2);
	data->extraParams = extraParams;
	double *y = (double *) malloc(sizeof(double) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}

	double comparison[4] = { 2.,1.5,1.33333333,1.25};
	double *comparisionAssertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = comparison[i];
	data->assertion = comparisionAssertion;
	data->result = (double *) malloc(sizeof(double) * length);

	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}



TEST(PairWiseTransform,ObjectOrientedReverseSub) {
	int rank = 2;
	int opNum = 7;
	int yRank = 2;
	int resultRank = 2;
	int length = 4;
	Data<double> *data = new Data<double>();
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


	double *extraParams = (double *) malloc(sizeof(double) * 2);
	data->extraParams = extraParams;
	double *y = (double *) malloc(sizeof(double) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}

	double comparison[4] = { 2.,1.5,1.33333333,1.25};
	double *comparisionAssertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = comparison[i];
	data->assertion = comparisionAssertion;
	data->result = (double *) malloc(sizeof(double) * length);

	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}


TEST(PairWiseTransform,ObjectOrientedSubtraction) {
	int rank = 2;
	int opNum = 9;
	int yRank = 2;
	int resultRank = 2;
	int length = 4;
	Data<double> *data = new Data<double>();
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


	double *extraParams = (double *) malloc(sizeof(double) * 2);
	data->extraParams = extraParams;
	double *y = (double *) malloc(sizeof(double) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}

	double comparison[4] = { -1., -1., -1., -1.};
	double *comparisionAssertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = comparison[i];
	data->assertion = comparisionAssertion;
	data->result = (double *) malloc(sizeof(double) * length);

	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;

}

#endif //NATIVEOPERATIONS_PAIRWISE_TRANSFORM_TESTS_H
