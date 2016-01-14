//
// Created by agibsonccc on 1/4/16.
//

#ifndef NATIVEOPERATIONS_SCALARTESTS_H
#define NATIVEOPERATIONS_SCALARTESTS_H
#include <transform.h>
#include <array.h>
#include <shape.h>
#include <scalar.h>
#include "testhelpers.h"

static functions::scalar::ScalarOpFactory<double> *opFactory4 = 0;

TEST_GROUP(ScalarTransform) {
	static int output_method(const char* output, ...) {
		va_list arguments;
		va_start(arguments, output);
		va_end(arguments);
		return 1;
	}
	void setup() {
		opFactory4 = new functions::scalar::ScalarOpFactory<double>();

	}
	void teardown() {
		delete opFactory4;
	}
};

template <typename T>
class ScalarTest : public BaseTest<T> {

public:
	virtual ~ScalarTest() {}

	virtual void freeOpAndOpFactory() {
		delete op;
		delete opFactory;
	}

	virtual void createOperationAndOpFactory() {
		opFactory = new functions::scalar::ScalarOpFactory<T>();
		op = opFactory->getOp(this->opNum);
	}

protected:
	functions::scalar::ScalarOpFactory<T> opFactory;
	functions::scalar::ScalarTransform<T> op;

};


TEST(ScalarTransform,ScalarAdd) {
	functions::scalar::ScalarTransform<double> *add = opFactory4->getOp(0);
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
	add->transform(data->data->data, 1, data->data->data, 1, 1.0, extraParams,
			length);

	double *comparison = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		comparison[i] = i + 2;
	CHECK(arrsEquals(rank, comparison, data->data->data));
	nd4j::array::NDArrays<double>::freeNDArrayOnGpuAndCpu(&data);
	free(extraParams);
	delete add;

}

#endif //NATIVEOPERATIONS_SCALARTESTS_H
