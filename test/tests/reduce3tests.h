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



Data<double> * getData(double *assertion,double startingVal) {
	Data<double> *ret = new Data<double>();

	int rank = 2;
	int length = 4;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 1;
	shape[1] = length;
	ret->yShape = shape;
	ret->xShape = shape;
	ret->rank = 2;
	ret->yRank = 2;
	ret->data = (double *) malloc(sizeof(double) * 4);
	ret->y = (double *) malloc(sizeof(double) * 4);

	for(int i = 0; i < 4; i++) {
		ret->data[i] = i + 1;
		ret->y[i] = i + 2;
	}

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
    ret->dimensionLength = 1;
	ret->result = (double *) malloc(sizeof(double));
	ret->resultRank = 2;
	ret->resultShape = (int *) malloc(sizeof(int) * 2);
	for(int i = 0; i < 2; i++)
		ret->resultShape[i] = 1;

	return ret;
}

Data<double> * getDataDimension(double *assertion,double startingVal) {
	Data<double> *ret = new Data<double>();

	int rank = 2;
	int length = 4;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	ret->xShape = shape;
	ret->yShape = shape;
	ret->rank = 2;
	ret->yRank = 2;
	ret->data = (double *) malloc(sizeof(double) * length);
	ret->y = (double *) malloc(sizeof(double) * 4);

	for(int i = 0; i < 4; i++) {
		ret->data[i] = i + 1;
		ret->y[i] = i + 1;
	}
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
		this->execCpuKernel();
		int resultLength = shape::prod(this->baseData->resultShape,this->baseData->rank);
		if(resultLength == 1) {
			if(this->result->data->data[0] != this->baseData->assertion[0]) {
				printf("Compared assertion %f to result %f\n",this->baseData->assertion[0],this->result->data->data[0]);
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




TEST(Reduce3,ObjectOrientedEuclideanDistance) {
	int opNum = 1;
	int rank = 2;
	double assertion[1] = {2.0};
	Data<double> *data = getData(assertion,0.0);
	DoubleReduce3Test *test = new DoubleReduce3Test(rank,opNum,data,1);
    test->run();
    delete data;
    delete test;
}



TEST(Reduce3,ObjectOrientedManhattanDistance) {
	int opNum = 0;
	int rank = 2;
	double assertion[1] = {4.0};
	Data<double> *data = getData(assertion,0.0);
	DoubleReduce3Test *test = new DoubleReduce3Test(rank,opNum,data,1);
    test->run();
    delete data;
    delete test;
}

TEST(Reduce3,ObjectOrientedCosineSimilarity) {
	int opNum = 2;
	int rank = 2;
	double assertion[1] = {0.9938079488022847};
	Data<double> *data = getData(assertion,0.0);
	DoubleReduce3Test *test = new DoubleReduce3Test(rank,opNum,data,4);
    test->run();
    delete data;
    delete test;
}



/*TEST(Reduce3,CosineSimilarity) {
	functions::reduce3::Reduce3OpFactory<double> *opFactory6 =
			new functions::reduce3::Reduce3OpFactory<double>();
	functions::reduce3::Reduce3<double> *op = opFactory6->getOp(2);
	int vectorLength = 4;
	shape::ShapeInformation *vecShapeInfo = (shape::ShapeInformation *) malloc(
			sizeof(shape::ShapeInformation));
	int rank = 2;
	int *shape = (int*) malloc(sizeof(int) * rank);
	shape[0] = 1;
	shape[1] = vectorLength;
	int *stride = shape::calcStrides(shape, rank);
	vecShapeInfo->shape = shape;
	vecShapeInfo->stride = stride;
	vecShapeInfo->offset = 0;
	vecShapeInfo->rank = rank;
	vecShapeInfo->elementWiseStride = 1;
	vecShapeInfo->order = 'c';
	int *shapeInfo = shape::toShapeBuffer(vecShapeInfo);
	assertBufferProperties(shapeInfo);
	double *result = (double *) malloc(sizeof(double));
	result[0] = 0.0;
	int *scalarShape = shape::createScalarShapeInfo();
	assertBufferProperties(scalarShape);

	double *vec1 = (double *) malloc(sizeof(double) * vectorLength);
	double *vec2 = (double *) malloc(sizeof(double) * vectorLength);

	for (int i = 0; i < vectorLength; i++) {
		vec1[i] = i + 1;
		vec2[i] = i + 1;
	}

	int extraParamsLength = 3;
	double *extraParams = (double *) malloc(extraParamsLength * sizeof(double));
	extraParams[0] = 0.0;
	extraParams[1] = 5.4772255750516612;
	extraParams[2] = 5.4772255750516612;
	op->exec(vec1, shapeInfo, extraParams, vec2, shapeInfo, result,
			scalarShape);
	CHECK(1.0 == result[0]);

	free(result);
	free(vec1);
	free(vec2);
	free(scalarShape);
	free(vecShapeInfo);
	free(shapeInfo);
	delete (opFactory6);
	delete (op);
}


TEST(Reduce3,EuclideanDistance) {
	functions::reduce3::Reduce3OpFactory<double> *opFactory6 =
			new functions::reduce3::Reduce3OpFactory<double>();
	functions::reduce3::Reduce3<double> *op = opFactory6->getOp(1);
	int vectorLength = 4;
	shape::ShapeInformation *vecShapeInfo = (shape::ShapeInformation *) malloc(
			sizeof(shape::ShapeInformation));
	int rank = 2;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 1;
	shape[1] = vectorLength;
	int *stride = shape::calcStrides(shape, rank);
	vecShapeInfo->shape = shape;
	vecShapeInfo->stride = stride;
	vecShapeInfo->offset = 0;
	vecShapeInfo->rank = rank;
	vecShapeInfo->elementWiseStride = 1;
	vecShapeInfo->order = 'c';
	int *shapeInfo = shape::toShapeBuffer(vecShapeInfo);
	assertBufferProperties(shapeInfo);
	double *result = (double *) malloc(sizeof(double));
	result[0] = 0.0;
	int *scalarShape = shape::createScalarShapeInfo();
	assertBufferProperties(scalarShape);
	double *vec1 = (double *) malloc(sizeof(double) * vectorLength);
	double *vec2 = (double *) malloc(sizeof(double) * vectorLength);

	for (int i = 0; i < vectorLength; i++) {
		vec1[i] = i + 1;
		vec2[i] = vec1[i] + 4;
	}

	double *extraParams = (double *) malloc(sizeof(double));
	extraParams[0] = 0.0;
	op->exec(vec1, shapeInfo, extraParams, vec2, shapeInfo, result,
			scalarShape);
	CHECK(8 == result[0]);

	free(shape);
	free(result);
	free(vec1);
	free(vec2);
	free(scalarShape);
	free(vecShapeInfo);
	free(shapeInfo);
	delete (opFactory6);
	delete (op);
}

TEST(Reduce3,EuclideanDistanceDimension) {
	functions::reduce3::Reduce3OpFactory<double> *opFactory6 =
			new functions::reduce3::Reduce3OpFactory<double>();
	functions::reduce3::Reduce3<double> *op = opFactory6->getOp(1);
	int vectorLength = 4;
	shape::ShapeInformation *vecShapeInfo = (shape::ShapeInformation *) malloc(
			sizeof(shape::ShapeInformation));
	int rank = 2;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	int *stride = shape::calcStrides(shape, rank);
	vecShapeInfo->shape = shape;
	vecShapeInfo->stride = stride;
	vecShapeInfo->offset = 0;
	vecShapeInfo->rank = rank;
	vecShapeInfo->elementWiseStride = 1;
	vecShapeInfo->order = 'c';
	int *shapeInfo = shape::toShapeBuffer(vecShapeInfo);
	assertBufferProperties(shapeInfo);

	nd4j::buffer::Buffer<int> *shapeInfoBuff = nd4j::buffer::createBuffer(shapeInfo,shape::shapeInfoLength(shape::rank(shapeInfo)));



	int resultLength = 2;
	double *result = (double *) malloc(resultLength * sizeof(double));
	for (int i = 0; i < resultLength; i++)
		result[i] = 0.0;

	nd4j::buffer::Buffer<double> *resultBuffer = nd4j::buffer::createBuffer(result,resultLength);

	//change to row vector
	int *scalarShape = shape::createScalarShapeInfo();
	shape::shapeOf(scalarShape)[1] = 2;
	assertBufferProperties(scalarShape);
	nd4j::buffer::Buffer<int> *scalarShapeBuff = nd4j::buffer::createBuffer(scalarShape,shape::shapeInfoLength(shape::rank(scalarShape)));

	double *vec1 = (double *) malloc(sizeof(double) * vectorLength);
	double *vec2 = (double *) malloc(sizeof(double) * vectorLength);


	for (int i = 0; i < vectorLength; i++) {
		vec1[i] = i + 1;
		vec2[i] = vec1[i] + 4;
	}

	nd4j::buffer::Buffer<double> *vec1Buffer = nd4j::buffer::createBuffer(vec1,vectorLength);
	nd4j::buffer::Buffer<double> *vec2Buffer = nd4j::buffer::createBuffer(vec2,vectorLength);

	int dimensionLength = 1;
	int *dimension = (int *) malloc(sizeof(int) * dimensionLength);
	dimension[0] = 1;
	nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(dimension,1);


	double *extraParams = (double *) malloc(sizeof(double));
	extraParams[0] = 0.0;
	nd4j::buffer::Buffer<double> *extraParamsBuff = nd4j::buffer::createBuffer(extraParams,1);
	op->exec(vec1, shapeInfo, extraParams, vec2, shapeInfo, result, scalarShape,
			dimension, dimensionLength);

	double *assertion = (double *) malloc(sizeof(double) * resultLength);
	assertion[0] = 5.656854249492381;
	assertion[1] = 5.6568542494923806;
	CHECK(arrsEquals<double>(2, assertion, result));
#ifdef __CUDACC__

 * reduce3Double(
		int opNum,
		int n, double *dx, int *xShapeInfo,
		double *dy,
		int *yShapeInfo, double *extraParams, double *result,
		int *resultShapeInfo, int *gpuInformation,
		int *dimension,
		int dimensionLength, int postProcessOrNot)


	int blockSize = 500;
	int gridSize = 256;
	int sMemSize = 20000;
	nd4j::buffer::freeBuffer(&resultBuffer);
	//realloc the buffer
	result = (double *) malloc(sizeof(double) * blockSize);
	resultBuffer = nd4j::buffer::createBuffer(result,blockSize);
	int *gpuInformation = (int *) malloc(sizeof(int) * 4);
	gpuInformation[0] = blockSize;
	gpuInformation[1] = gridSize;
	gpuInformation[2] = sMemSize;
	gpuInformation[3] = 49152;
	nd4j::buffer::Buffer<int> *gpuInfoBuff = nd4j::buffer::createBuffer<int>(gpuInformation,4);
	reduce3Double<<<blockSize,gridSize,sMemSize>>>(
			1,
			vectorLength,
			vec1Buffer->gData,
			shapeInfoBuff->gData,
			vec2Buffer->gData,
			shapeInfoBuff->gData,
			extraParamsBuff->gData,
			resultBuffer->gData,
			scalarShapeBuff->gData,
			gpuInfoBuff->gData,
			dimensionBuffer->gData,
			1,
			1
	);
	checkCudaErrors(cudaDeviceSynchronize());
	nd4j::buffer::copyDataFromGpu(&resultBuffer);
	for(int i = 0; i < resultLength; i++) {
		printf("Result[%d] after was %f\n",i,resultBuffer->data[i]);
	}
	CHECK(arrsEquals<double>(2, assertion, result));
#endif


	nd4j::buffer::freeBuffer(&dimensionBuffer);
	nd4j::buffer::freeBuffer(&vec1Buffer);
	nd4j::buffer::freeBuffer(&vec2Buffer);
	free(shape);
	nd4j::buffer::freeBuffer(&resultBuffer);
	nd4j::buffer::freeBuffer(&extraParamsBuff);
	free(assertion);
	nd4j::buffer::freeBuffer(&scalarShapeBuff);
	free(vecShapeInfo);
	free(shapeInfo);
	delete (opFactory6);
	delete (op);
}*/
#endif //NATIVEOPERATIONS_REDUCE3TESTS_H
