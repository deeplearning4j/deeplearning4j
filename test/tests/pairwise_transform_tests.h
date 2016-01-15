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
	virtual ~PairwiseTransformTest() {}
	PairwiseTransformTest() {
		createOperationAndOpFactory();
	}
	PairwiseTransformTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
	: PairWiseTest<T>(rank,opNum,data,extraParamsLength) {
		createOperationAndOpFactory();
	}
	virtual void freeOpAndOpFactory() {
		delete op;
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


TEST(PairWiseTransform,Addition) {
	functions::pairwise_transforms::PairWiseTransform<double> *add =
			opFactory2->getOp(0);
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

	add->exec(data->data->data, 1, data->data->data, 1, data->data->data, 1,
			extraParams, length);
	double comparison[4] = { 2, 4, 6, 8 };
	CHECK(arrsEquals(rank, comparison, data->data->data));
	free(data);
	free(extraParams);
	free(shape);
	free(stride);
	delete add;

}

#endif //NATIVEOPERATIONS_PAIRWISE_TRANSFORM_TESTS_H
