//
// Created by agibsonccc on 1/3/16.
//

#ifndef NATIVEOPERATIONS_TRANSFORMTESTS_H
#define NATIVEOPERATIONS_TRANSFORMTESTS_H

#include <transform.h>
#include <array.h>
#include <templatemath.h>
#include <shape.h>
#include "testhelpers.h"

static functions::transform::TransformOpFactory<double> *opFactory = 0;

TEST_GROUP(Transform) {
	static int output_method(const char* output, ...) {
		va_list arguments;
		va_start(arguments, output);
		va_end(arguments);
		return 1;
	}
	void setup() {
		opFactory = new functions::transform::TransformOpFactory<double>();

	}
	void teardown() {
		delete opFactory;
	}
};

TEST(Transform,Log) {
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

	functions::transform::Transform<double> *log = opFactory->getOp(5);
	log->exec(data->data->data, 1, data->data->data, 1, extraParams, length);

	double comparison[4] = { 0., 0.69314718, 1.09861229, 1.38629436 };
	CHECK(arrsEquals(rank, comparison, data->data->data));
	free(data);
	free(extraParams);
	free(shape);
	free(stride);
	delete log;

}




template <typename T>
class TransformTest : public BaseTest<T> {
protected:
	functions::transform::Transform<T> *op = NULL;
	functions::transform::TransformOpFactory<T> *opFactory = NULL;


public:
	TransformTest() {}
	virtual ~TransformTest() {
		freeOpAndOpFactory();
	}

	TransformTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
	: BaseTest<T>(rank,opNum,data,extraParamsLength) {
		createOperationAndOpFactory();
	}
	virtual void freeOpAndOpFactory() override {
		if(op != NULL)
			delete op;
		if(opFactory != NULL)
			delete opFactory;
	}

	virtual void createOperationAndOpFactory() override {
		opFactory = new functions::transform::TransformOpFactory<T>();
		op = opFactory->getOp(this->opNum);
	}

	void run () {
		this->initializeData();
		op->exec(this->data->data->data, 1, this->data->data->data, 1, this->extraParams, this->length);
		CHECK(arrsEquals(this->rank, this->assertion, this->data->data->data));


#ifdef __CUDACC__
		this->initializeData();
		nd4j::array::NDArrays<T>::allocateNDArrayOnGpu(&this->data);
		this->executeCudaKernel();
		checkCudaErrors(cudaDeviceSynchronize());
		nd4j::buffer::copyDataFromGpu(&this->data->data);
		CHECK(arrsEquals(this->rank, this->assertion, this->data->data->data));

#endif


	}
};

class DoubleTransformTest : public  TransformTest<double> {
public:
	virtual ~DoubleTransformTest() {}
	DoubleTransformTest() {}
	DoubleTransformTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
	:  TransformTest<double>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() override {
		transformDouble<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum
				,this->length,
				1,this->data->data->gData,
				1,this->extraParamsBuff->gData,
				this->data->data->gData
				,1);

	}
};


class FloatTransformTest : public TransformTest<float> {
public:
	virtual ~FloatTransformTest() {}
	FloatTransformTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
	:  TransformTest<float>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() override {
		transformFloat<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum
				,this->length,
				1,this->data->data->gData,
				1,this->extraParamsBuff->gData,
				this->data->data->gData
				,1);
	}
};

class DoubleTwoByTwoTransformTest : public DoubleTransformTest  {
public:
	DoubleTwoByTwoTransformTest() {}
	virtual ~DoubleTwoByTwoTransformTest() {}
	DoubleTwoByTwoTransformTest(int opNum,Data<double> *data,int extraParamsLength):
		DoubleTransformTest(2,opNum,data,extraParamsLength) {
	}

};

TEST(Transform,ObjectOrientedSigmoid) {
	int opNum = 10;
	int rank = 2;
	Data<double> *data = new Data<double>();
	data->xShape = (int *) malloc(sizeof(int) * 2);
	for(int i = 0; i < 2; i++) {
		data->xShape[i] = 2;
	}
	data->assertion = (double *) malloc(sizeof(double) * 4);
	data->assertion [0] = 0.7310585786300049;
	data->assertion [1] = 0.8807970779778823;
	data->assertion [2] = 0.9525741268224334;
	data->assertion [3] = 0.9820137900379085;
	data->rank = rank;
	data->extraParams = (double *) malloc(sizeof(double) * 2);
	DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
	sigmoidTest->run();
	delete sigmoidTest;
	delete data;
}


TEST(Transform,ObjectOrientedLog) {
	int opNum = 10;
	int rank = 2;
	Data<double> *data = new Data<double>();
	data->xShape = (int *) malloc(sizeof(int) * 2);
	for(int i = 0; i < 2; i++) {
		data->xShape[i] = 2;
	}

	double comparison[4] = { 0., 0.69314718, 1.09861229, 1.38629436 };
	data->assertion = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		data->assertion[i] = comparison[i];
	data->rank = rank;
	data->extraParams = (double *) malloc(sizeof(double) * 2);
	DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
	sigmoidTest->run();
	delete sigmoidTest;
	delete data;
}


TEST(Transform,Sigmoid) {
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

	functions::transform::Transform<double> *log = opFactory->getOp(10);
	log->exec(data->data->data, 1, data->data->data, 1, extraParams, length);
	double *comparison= (double *)malloc(4 * sizeof(double));
	comparison[0] = 0.7310585786300049;
	comparison[1] = 0.8807970779778823;
	comparison[2] = 0.9525741268224334;
	comparison[3] = 0.9820137900379085;
	CHECK(arrsEquals(rank, comparison, data->data->data));


#ifdef __CUDACC__
	for (int i = 0; i < length; i++)
		data->data->data[i] = i + 1;
	nd4j::array::NDArrays<double>::allocateNDArrayOnGpu(&data);
	double *extraParamsData = (double *) malloc(sizeof(double));
	extraParams[0] = 0.0;
	nd4j::buffer::Buffer<double> *extraParamsBuff = nd4j::buffer::createBuffer(extraParamsData,1);
	transformDouble<<<length,length,2000>>>(
			10
			,length,
			1,data->data->gData,
			1,extraParamsBuff->gData,
			data->data->gData
			,1);
	checkCudaErrors(cudaDeviceSynchronize());
	nd4j::buffer::freeBuffer(&extraParamsBuff);
	nd4j::buffer::copyDataFromGpu(&data->data);
	CHECK(arrsEquals(rank, comparison, data->data->data));

#endif

	nd4j::array::NDArrays<double>::freeNDArrayOnGpuAndCpu(&data);
	delete log;

}

#endif //NATIVEOPERATIONS_TRANSFORMTESTS_H
