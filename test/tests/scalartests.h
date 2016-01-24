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
	virtual ~ScalarTest() {
		freeOpAndOpFactory();
	}
	ScalarTest() {
		createOperationAndOpFactory();
	}
	ScalarTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
	: BaseTest<T>(rank,opNum,data,extraParamsLength) {
		createOperationAndOpFactory();
	}
	virtual void freeOpAndOpFactory() {
		delete op;
		delete opFactory;
	}

	virtual void createOperationAndOpFactory() {
		opFactory = new functions::scalar::ScalarOpFactory<T>();
		op = opFactory->getOp(this->opNum);
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

	virtual void execCpuKernel() override {
		int *xShapeBuff = shapeBuffer(this->baseData->rank,this->baseData->xShape);
		assertBufferProperties(xShapeBuff);
		int xElementWiseStride = shape::elementWiseStride(xShapeBuff);
		int *resultShapeBuff = shapeBuffer(this->baseData->resultRank,this->baseData->resultShape);
		assertBufferProperties(resultShapeBuff);
		int resultElementWiseStride = shape::elementWiseStride(resultShapeBuff);
		int n = shape::length(xShapeBuff);
		op->transform(this->data->data->data,xElementWiseStride,this->result->data->data,resultElementWiseStride,this->baseData->scalar,this->extraParams,n);
		free(xShapeBuff);
		free(resultShapeBuff);
	}

protected:
	functions::scalar::ScalarOpFactory<T> *opFactory;
	functions::scalar::ScalarTransform<T> *op;

};

class DoubleScalarTest : public ScalarTest<double> {
public:
	DoubleScalarTest() {}
	DoubleScalarTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
	:  ScalarTest<double>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() {
#ifdef __CUDACC__
		int *shapeBuff = shapeBuffer(this->rank,this->shape);
		int eleStride = shape::elementWiseStride(shapeBuff);
		scalarDouble<<<this->blockSize,this->gridSize,
				this->sMemSize>>>(
						this->opNum,
						this->length,
						0,
						this->baseData->scalar,
						this->data->data->gData,
						eleStride,
						this->extraParamsBuff->gData,
						this->result->data->gData,this->blockSize

				);

		free(shapeBuff);
#endif
	}
};


class FloatScalarTest : public ScalarTest<float> {
public:
	FloatScalarTest() {}
	FloatScalarTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
	:  ScalarTest<float>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() {
#ifdef __CUDACC__
		int *shapeBuff = shapeBuffer(this->rank,this->shape);
		int eleStride = shape::elementWiseStride(shapeBuff);
		scalarFloat<<<this->blockSize,this->gridSize,
				this->sMemSize>>>(
						this->opNum,
						this->length,
						1,
						this->baseData->scalar,
						this->data->data->gData,
						eleStride,
						this->extraParamsBuff->gData,
						this->result->data->gData,this->blockSize

				);

		free(shapeBuff);
#endif
	}
};


static Data<double> * getData(double *comparison,int rank) {
	Data<double> *data = new Data<double>();
	data->scalar = 1;
	data->rank = rank;
	data->resultRank = 2;
	data->xShape = (int *) malloc(sizeof(int) * 2);
	data->resultShape = (int *) malloc(sizeof(int) * 2);

	for(int i = 0; i < 2; i++) {
		data->xShape[i] = 2;
		data->resultShape[i] = 2;

	}
	data->result = (double *) malloc(sizeof(double *) * 4);
	data->assertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		data->assertion[i] = comparison[i];
	data->rank = rank;
	data->extraParams = (double *) malloc(sizeof(double) * 2);
	return data;

}

static Data<double> * getData(double *comparison,double scalar,int rank) {
	Data<double> *data = new Data<double>();
	data->scalar = scalar;
	data->rank = rank;
	data->resultRank = 2;
	data->xShape = (int *) malloc(sizeof(int) * 2);
	data->resultShape = (int *) malloc(sizeof(int) * 2);

	for(int i = 0; i < 2; i++) {
		data->xShape[i] = 2;
		data->resultShape[i] = 2;

	}

	data->result = (double *) malloc(sizeof(double *) * 4);
	data->assertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		data->assertion[i] = comparison[i];
	data->rank = rank;
	data->extraParams = (double *) malloc(sizeof(double) * 2);
	return data;

}

TEST(ScalarTransform,ObjectOrientedScalarAdd) {
	int rank = 2;
	int opNum = 0;
	double comparison[4] = { 2,3,4,5 };

	Data<double> *data = getData(comparison,rank);

	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}


TEST(ScalarTransform,ObjectOrientedScalarSub) {
	int rank = 2;
	int opNum = 1;
	double comparison[4] = { 0,1,2,3 };
	Data<double> *data = getData(comparison,rank);
	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}



TEST(ScalarTransform,ObjectOrientedScalarMul) {
	int rank = 2;
	int opNum = 2;
	double comparison[4] = { 2,4,6,8 };
	Data<double> *data = getData(comparison,2,rank);

	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}

TEST(ScalarTransform,ObjectOrientedScalarDiv) {
	int rank = 2;
	int opNum = 3;
	double comparison[4] = { 0.5,1,1.5,2};


	Data<double> *data = getData(comparison,2,rank);

	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}


TEST(ScalarTransform,ObjectOrientedScalarRDiv) {
	int rank = 2;
	int opNum = 4;
	double comparison[4] = { 2,1. ,0.66666667,0.5};
	Data<double> *data = getData(comparison,2,rank);
	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}

TEST(ScalarTransform,ObjectOrientedScalarRSub) {
	int rank = 2;
	int opNum = 5;
	double comparison[4] = {  1.,  0., -1., -2.};

	Data<double> *data = getData(comparison,2,rank);

	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}


TEST(ScalarTransform,ObjectOrientedScalarMax) {
	int rank = 2;
	int opNum = 6;
	double comparison[4] = {  2,2,3,4};

	Data<double> *data = getData(comparison,2,rank);
	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}


TEST(ScalarTransform,ObjectOrientedScalarLessThan) {
	int rank = 2;
	int opNum = 7;
	double comparison[4] = {  1,0,0,0};

	Data<double> *data = getData(comparison,2,rank);
	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}


TEST(ScalarTransform,ObjectOrientedScalarGreaterThan) {
	int rank = 2;
	int opNum = 8;
	double comparison[4] = { 0,0,1,1};
	Data<double> *data =getData(comparison,2,rank);
	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}

TEST(ScalarTransform,ObjectOrientedScalarEquals) {
	int rank = 2;
	int opNum = 9;
	double comparison[4] = {0,1,0,0};
	Data<double> *data = getData(comparison,2,rank);

	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}


TEST(ScalarTransform,ObjectOrientedScalarLessThanOrEqual) {
	int rank = 2;
	int opNum = 10;
	double comparison[4] = {  1,1,0,0};
	Data<double> *data = getData(comparison,2,rank);
	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}


TEST(ScalarTransform,ObjectOrientedScalarNotEqual) {
	int rank = 2;
	int opNum = 11;
	double comparison[4] = {  1,0,1,1};

	Data<double> *data = getData(comparison,2,rank);
	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}

TEST(ScalarTransform,ObjectOrientedScalarMin) {
	int rank = 2;
	int opNum = 12;
	double comparison[4] = { 1,2,2,2};
	Data<double> *data = getData(comparison,2,rank);

	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}



TEST(ScalarTransform,ObjectOrientedScalarSet) {
	int rank = 2;
	int opNum = 13;
	double comparison[4] = {  2,2,2,2};
	Data<double> *data = getData(comparison,2,rank);

	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}




TEST(ScalarTransform,ObjectOrientedScalarMod) {
	int rank = 2;
	int opNum = 14;
	double comparison[4] = {  1.,  0.,  1.,  0.};
	Data<double> *data = getData(comparison,2,rank);

	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}





TEST(ScalarTransform,ObjectOrientedScalarRMod) {
	int rank = 2;
	int opNum = 15;
	double comparison[4] = {  0.,  0.,  2.,  2.};

	Data<double> *data = getData(comparison,2,rank);


	DoubleScalarTest *test = new DoubleScalarTest(rank,opNum,data,1);
	test->run();

	delete test;
	delete data;
}



#endif //NATIVEOPERATIONS_SCALARTESTS_H
