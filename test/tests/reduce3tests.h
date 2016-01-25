//
// Created by agibsonccc on 1/5/16.
//

#ifndef NATIVEOPERATIONS_REDUCE3TESTS_H
#define NATIVEOPERATIONS_REDUCE3TESTS_H
#include <array.h>
#include "testhelpers.h"
#include <reduce3.h>
#include <helper_cuda.h>


TEST_GROUP(Reduce3) {

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
static Data<T> * getDataReduce3(T *assertion,T startingVal) {
	Data<T> *ret = new Data<T>();

	int rank = 2;
	int length = 4;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 1;
	shape[1] = length;
	ret->yShape = shape;
	ret->xShape = shape;
	ret->rank = 2;
	ret->yRank = 2;
	ret->data = (T *) malloc(sizeof(T) * 4);
	ret->y = (T *) malloc(sizeof(T) * 4);

	for(int i = 0; i < 4; i++) {
		ret->data[i] = i + 1;
		ret->y[i] = i + 2;
	}

	T *extraParams = (T *) malloc(sizeof(T) * 4);
	for(int i = 0; i < 4; i++)
		extraParams[i] = startingVal;
	ret->extraParams = extraParams;

	ret->assertion = (T *) malloc(sizeof(T) * 4);
	for(int i = 0; i < 1; i++) {
		printf("Assertion value %f\n",assertion[i]);
		ret->assertion[i] = assertion[i];
	}

	ret->dimension = (int *) malloc(sizeof(int) * 2);
	ret->dimension[0] = shape::MAX_DIMENSION;
	ret->dimensionLength = 1;
	ret->result = (T *) malloc(sizeof(T));
	ret->resultRank = 2;
	ret->resultShape = (int *) malloc(sizeof(int) * 2);
	for(int i = 0; i < 2; i++)
		ret->resultShape[i] = 1;

	return ret;
}

template <typename T>
static Data<T> * getDataReduce3Dimension(T *assertion,T startingVal) {
	Data<T> *ret = new Data<T>();

	int rank = 2;
	int length = 4;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	ret->xShape = shape;
	ret->yShape = shape;
	ret->rank = 2;
	ret->yRank = 2;
	ret->data = (T *) malloc(sizeof(T) * length);
	ret->y = (T *) malloc(sizeof(T) * 4);

	for(int i = 0; i < 4; i++) {
		ret->data[i] = i + 1;
		ret->y[i] = i + 2;
	}

	int numTads = 2;
	int numParams = 4;
	T *extraParams = (T *) malloc(sizeof(T) * numTads * numParams * EXTRA_PARAMS_LENGTH);
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
class Reduce3Test : public PairWiseTest<T> {

public:
	Reduce3Test() {
		createOperationAndOpFactory();
	}
	virtual ~Reduce3Test() {
		freeOpAndOpFactory();
	}
	Reduce3Test(int rank,int opNum,Data<T> *data,int extraParamsLength)
	:  PairWiseTest<T>(rank,opNum,data,extraParamsLength) {
		createOperationAndOpFactory();
	}
	void freeOpAndOpFactory() {
		delete opFactory;
		delete reduce;
	}

	virtual void createOperationAndOpFactory() {
		opFactory = new functions::reduce3::Reduce3OpFactory<T>();
		reduce = opFactory->getOp(this->opNum);
	}

	virtual void execCpuKernel() override {
		int *xShapeBuff = shapeBuffer(this->baseData->rank,this->baseData->xShape);
		int *yShapeBuff = shapeBuffer(this->baseData->yRank,this->baseData->yShape);
		int *resultShapeBuff = shapeBuffer(this->baseData->resultRank,this->baseData->resultShape);
		reduce->exec(
				this->data->data->data,
				xShapeBuff,
				this->baseData->extraParams,
				this->baseData->y,yShapeBuff,
				this->result->data->data,
				resultShapeBuff
				,this->baseData->dimension,
				this->baseData->dimensionLength);
		free(xShapeBuff);
		free(yShapeBuff);
		free(resultShapeBuff);
	}

	virtual void run () override {
		printf("initializing data\n");
		this->initializeData();
		printf("Executing cpu\n");
		int resultLength = shape::prod(this->baseData->resultShape,this->baseData->rank);

		for(int i = 0; i < resultLength; i++) {
			printf("Before executing assertion %d is %f\n",i,this->assertion[i]);
		}
		this->execCpuKernel();

		if(resultLength == 1) {
			if(this->result->data->data[0] != this->baseData->assertion[0]) {
				printf("Compared assertion %f to result %f\n",this->baseData->assertion[0],this->result->data->data[0]);
			}
			DOUBLES_EQUAL(this->baseData->assertion[0],this->result->data->data[0],1e-3);
		}
		else {
			CHECK(arrsEquals(this->rank, this->baseData->assertion, this->result->data->data));
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
	functions::reduce3::Reduce3OpFactory<T> *opFactory;
	functions::reduce3::Reduce3<T> *reduce;
};


class DoubleReduce3Test : public Reduce3Test<double> {
public:
	DoubleReduce3Test() {}
	DoubleReduce3Test(int rank,int opNum,Data<double> *data,int extraParamsLength)
	:  Reduce3Test<double>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() {
#ifdef __CUDACC__
		nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->shape);
		nd4j::buffer::Buffer<int> *yShapeBuff = shapeIntBuffer(this->rank,this->shape);
		nd4j::buffer::Buffer<int> *resultShapeInfo = shapeIntBuffer(this->result->rank,this->result->shape->data);
		reduce3Double<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->length,
				this->data->data->gData,
				xShapeBuff->gData,
				this->yData->data->gData,
				yShapeBuff->gData,
				this->extraParamsBuff->gData,
				this->result->data->gData,
				resultShapeInfo->gData,
				gpuInfo->gData,
				dimensionBuffer->gData,
				this->baseData->dimensionLength,
				1);
		nd4j::buffer::freeBuffer(&dimensionBuffer);
		nd4j::buffer::freeBuffer(&xShapeBuff);
		nd4j::buffer::freeBuffer(&yShapeBuff);
		nd4j::buffer::freeBuffer(&gpuInfo);
		nd4j::buffer::freeBuffer(&resultShapeInfo);
#endif
	}

};

class FloatReduce3Test : public Reduce3Test<float> {
public:
	FloatReduce3Test() {}
	FloatReduce3Test(int rank,int opNum,Data<float> *data,int extraParamsLength)
	:  Reduce3Test<float>(rank,opNum,data,extraParamsLength){
	}
	virtual void executeCudaKernel() {
#ifdef __CUDACC__
		nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->shape);
		nd4j::buffer::Buffer<int> *yShapeBuff = shapeIntBuffer(this->rank,this->shape);
		nd4j::buffer::Buffer<int> *resultShapeInfo = shapeIntBuffer(this->result->rank,this->result->shape->data);
		reduce3Float<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->length,
				this->data->data->gData,
				xShapeBuff->gData,
				this->yData->data->gData,
				yShapeBuff->gData,
				this->extraParamsBuff->gData,
				this->result->data->gData,
				resultShapeInfo->gData,
				gpuInfo->gData,
				dimensionBuffer->gData,
				this->baseData->dimensionLength,
				1);
		nd4j::buffer::freeBuffer(&dimensionBuffer);
		nd4j::buffer::freeBuffer(&xShapeBuff);
		nd4j::buffer::freeBuffer(&yShapeBuff);
		nd4j::buffer::freeBuffer(&gpuInfo);
		nd4j::buffer::freeBuffer(&resultShapeInfo);
#endif

	}
};

TEST(Reduce3,ObjectOrientedDimensionEuclideanDistance) {
	int opNum = 1;
	int rank = 2;
	double assertion[2] = {1.41, 1.41};
	Data<double> *data = getDataReduce3Dimension<double>(assertion,0.0);
	DoubleReduce3Test *test = new DoubleReduce3Test(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}



TEST(Reduce3,ObjectOrientedDimensionManhattanDistance) {
	int opNum = 0;
	int rank = 2;
	double assertion[2] = {2.0,2.0};
	Data<double> *data = getDataReduce3Dimension<double>(assertion,0.0);
	DoubleReduce3Test *test = new DoubleReduce3Test(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}


TEST(Reduce3,ObjectOrientedDimensionCosineSimilarity) {
	int opNum = 2;
	int rank = 2;
	double assertion[2] = {0.9938079488022847,1.0};
	Data<double> *data = getDataReduce3Dimension<double>(assertion,0.0);
	DoubleReduce3Test *test = new DoubleReduce3Test(rank,opNum,data,4);
	test->run();
	delete data;
	delete test;
}



TEST(Reduce3,ObjectOrientedEuclideanDistance) {
	int opNum = 1;
	int rank = 2;
	double assertion[1] = {2.0};
	Data<double> *data = getDataReduce3<double>(assertion,0.0);
	DoubleReduce3Test *test = new DoubleReduce3Test(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}



TEST(Reduce3,ObjectOrientedManhattanDistance) {
	int opNum = 0;
	int rank = 2;
	double assertion[1] = {4.0};
	Data<double> *data = getDataReduce3<double>(assertion,0.0);
	DoubleReduce3Test *test = new DoubleReduce3Test(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}

TEST(Reduce3,ObjectOrientedCosineSimilarity) {
	int opNum = 2;
	int rank = 2;
	double assertion[1] = {0.9938079488022847};
	Data<double> *data = getDataReduce3<double>(assertion,0.0);
	DoubleReduce3Test *test = new DoubleReduce3Test(rank,opNum,data,4);
	test->run();
	delete data;
	delete test;
}


TEST(Reduce3,ObjectOrientedFloatDimensionEuclideanDistance) {
	int opNum = 1;
	int rank = 2;
	float assertion[2] = {1.41, 1.41};
	Data<float> *data = getDataReduce3Dimension<float>(assertion,0.0);
	FloatReduce3Test *test = new FloatReduce3Test(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}



TEST(Reduce3,ObjectOrientedFloatDimensionManhattanDistance) {
	int opNum = 0;
	int rank = 2;
	float assertion[2] = {2.0,2.0};
	Data<float> *data = getDataReduce3Dimension<float>(assertion,0.0);
	FloatReduce3Test *test = new FloatReduce3Test(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}


TEST(Reduce3,ObjectOrientedFloatDimensionCosineSimilarity) {
	int opNum = 2;
	int rank = 2;
	float assertion[2] = {0.9938079488022847,1.0};
	Data<float> *data = getDataReduce3Dimension<float>(assertion,0.0);
	FloatReduce3Test *test = new FloatReduce3Test(rank,opNum,data,4);
	test->run();
	delete data;
	delete test;
}



TEST(Reduce3,ObjectOrientedFloatEuclideanDistance) {
	int opNum = 1;
	int rank = 2;
	float assertion[1] = {2.0};
	Data<float> *data = getDataReduce3<float>(assertion,0.0);
	FloatReduce3Test *test = new FloatReduce3Test(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}



TEST(Reduce3,ObjectOrientedFloatManhattanDistance) {
	int opNum = 0;
	int rank = 2;
	float assertion[1] = {4.0};
	Data<float> *data = getDataReduce3<float>(assertion,0.0);
	FloatReduce3Test *test = new FloatReduce3Test(rank,opNum,data,1);
	test->run();
	delete data;
	delete test;
}

TEST(Reduce3,ObjectOrientedFloatCosineSimilarity) {
	int opNum = 2;
	int rank = 2;
	float assertion[1] = {0.9938079488022847};
	Data<float> *data = getDataReduce3<float>(assertion,0.0);
	FloatReduce3Test *test = new FloatReduce3Test(rank,opNum,data,4);
	test->run();
	delete data;
	delete test;
}




#endif //NATIVEOPERATIONS_REDUCE3TESTS_H
