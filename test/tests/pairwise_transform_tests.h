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
	virtual void freeOpAndOpFactory() {
		if(op != NULL)
			delete op;
		if(opFactory != NULL)
			delete opFactory;
	}

	virtual void createOperationAndOpFactory() {
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
		printf("In execute kernel\n");

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
	}
};



class FloatPairwiseTranformTest : public PairwiseTransformTest<float> {
public:
	FloatPairwiseTranformTest() {}
	FloatPairwiseTranformTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
	:  PairwiseTransformTest<float>(rank,opNum,data,extraParamsLength) {
	}
	virtual void executeCudaKernel() {
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
	}
};

TEST(PairWiseTransform,ObjectOrientedAddition) {
	int rank = 2;
	int opNum = 0;
	int yRank = 2;
	int length = 4;
	Data<double> *data = new Data<double>();
	data->rank = 2;
	data->yRank = 2;

	data->xShape = (int *) malloc(sizeof(int) * rank);
	data->yShape = (int *) malloc(sizeof(int) * rank);
	for(int i = 0; i < 2; i++) {
		data->xShape[i] = 2;
		data->yShape[i] = 2;
	}

	double *extraParams = (double *) malloc(sizeof(double) * 2);
	data->extraParams = extraParams;
	printf("Alloced x and y shapes\n");

	double *y = (double *) malloc(sizeof(double) * length);
	data->y = y;
	for(int i = 0; i < length; i++) {
		y[i] = i + 2;
	}
	printf("Alloced y\n");

	double comparison[4] = {3,5,7,9};
	double *comparisionAssertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparisionAssertion[i] = comparison[i];
	data->assertion = comparisionAssertion;
	//	:  PairwiseTransformTest<double>(rank,opNum,data,extraParamsLength){
	DoublePairwiseTranformTest *test = new DoublePairwiseTranformTest(rank,opNum,data,1);
	test->run();

	delete data;
	delete test;

}




#endif //NATIVEOPERATIONS_PAIRWISE_TRANSFORM_TESTS_H
