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

static Data<double> * getDataBroadcast(int rank,double *comparison) {
	Data<double> *ret = new Data<double>();
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
	ret->y = (double *) malloc(sizeof(double) * 2);

	int dimensionLength = 1;
	int *dimension = (int *) malloc(sizeof(int));
	dimension[0] = 1;
	ret->dimension = dimension;
	ret->dimensionLength = 1;
	ret->assertion = (double *) malloc(sizeof(double) * 4);
	ret->resultShape = (int *) malloc(sizeof(int) * 2);
	for(int i = 0; i < 2; i++)
		ret->resultShape[i] = 2;
	ret->result = (double *) malloc(sizeof(double) * 4);
	for(int i = 0; i < 4; i++)
		ret->result[i] = comparison[i];
	ret->extraParams = (double *) malloc(sizeof(double));
	return ret;

}


TEST(BroadCasting,Addition) {
	functions::broadcast::Broadcast<double> *add = opFactory3->getOp(0);
	int rank = 2;
	int *shape = (int *) malloc(sizeof(int) * rank);
	shape[0] = 2;
	shape[1] = 2;
	int *stride = shape::calcStrides(shape, rank);
	nd4j::array::NDArray<double> *data =
			nd4j::array::NDArrays<double>::createFrom(rank, shape, stride, 0,
					0.0);
	shape::ShapeInformation *shapeInformation =
			nd4j::array::NDArrays<double>::shapeInfoForArray(data);
	int *shapeInfoBuffer = shape::toShapeBuffer(shapeInformation);
	nd4j::buffer::Buffer<int> *shapeInfoBuferBuff = nd4j::buffer::createBuffer(shapeInfoBuffer,shape::shapeInfoLength(shape::rank(shapeInfoBuffer)));
	assertBufferProperties(shapeInfoBuffer);
	int length = nd4j::array::NDArrays<double>::length(data);
	CHECK(length == 4);
	for (int i = 0; i < length; i++)
		data->data->data[i] = i + 1;

	int *vectorShape = (int *) malloc(rank * sizeof(int));
	vectorShape[0] = 1;
	vectorShape[1] = 2;
	int *vectorStride = shape::calcStrides(vectorShape, rank);
	nd4j::array::NDArray<double> *vector =
			nd4j::array::NDArrays<double>::createFrom(rank, vectorShape,
					vectorStride, 0, 0.0);
	for (int i = 0; i < 2; i++)
		vector->data->data[i] = i + 1;
	shape::ShapeInformation *vectorShapeInformation = nd4j::array::NDArrays<double>::shapeInfoForArray(vector);
	int *vectorShapeInfoBuff = shape::toShapeBuffer(vectorShapeInformation);
	nd4j::buffer::Buffer<int> *vectorShapeInfoBuffData =  nd4j::buffer::createBuffer(vectorShapeInfoBuff,shape::shapeInfoLength(shape::rank(vectorShapeInfoBuff)));
	assertBufferProperties(vectorShapeInfoBuff);
	int dimensionLength = 1;
	int *dimension = (int *) malloc(sizeof(int));
	dimension[0] = 1;
	nd4j::buffer::Buffer<int> *dimensionBuff = nd4j::buffer::createBuffer(dimension,1);
	double *extraParams = (double *) malloc(sizeof(double));
	nd4j::buffer::Buffer<double> *extraParamsBuffer = nd4j::buffer::createBuffer(extraParams,1);
	add->exec(data->data->data, shapeInfoBuffer, vector->data->data,
			vectorShapeInfoBuff, data->data->data, shapeInfoBuffer, dimension,
			dimensionLength);

	double *comparison =(double *) malloc(sizeof(double) * 4);
	comparison[0] = 2;
	comparison[1] = 4;
	comparison[2] = 5;
	comparison[3] = 6;
	CHECK(arrsEquals(rank, comparison, data->data->data));

#ifdef __CUDACC__
	for (int i = 0; i < length; i++)
		data->data->data[i] = i + 1;
	int blockSize = 500;
	int gridSize = 256;
	int sMemSize = 20000;

	int *gpuInformation = (int *) malloc(sizeof(int) * 4);
	gpuInformation[0] = blockSize;
	gpuInformation[1] = gridSize;
	gpuInformation[2] = sMemSize;
	gpuInformation[3] = 49152;
	nd4j::array::NDArrays<double>::allocateNDArrayOnGpu(&data);
	nd4j::array::NDArrays<double>::allocateNDArrayOnGpu(&vector);
	nd4j::array::NDArrays<double>::copyFromGpu(&data);

	nd4j::buffer::Buffer<int> *gpuInfoBuff = nd4j::buffer::createBuffer<int>(gpuInformation,4);
	for(int i = 0; i < length; i++) {
		printf("Data[%d] for broadcast before was %f\n",i,data->data->data[i]);
	}
	broadcastDouble<<<blockSize,gridSize,sMemSize>>>(
			0,
			data->data->gData,
			shapeInfoBuferBuff->gData,
			vector->data->gData,
			vectorShapeInfoBuffData->gData,
			data->data->gData,
			shapeInfoBuferBuff->gData,
			dimensionBuff->gData,
			dimensionLength,
			gpuInfoBuff->gData);
	checkCudaErrors(cudaDeviceSynchronize());
	nd4j::buffer::copyDataFromGpu(&data->data);
	for(int i = 0; i < length; i++) {
		printf("Data[%d] for broadcast was %f\n",i,data->data->data[i]);
	}
	CHECK(arrsEquals(rank, comparison, data->data->data));
#endif


	nd4j::array::NDArrays<double>::freeNDArrayOnGpuAndCpu(&data);
	nd4j::buffer::freeBuffer(&extraParamsBuffer);
	free(shapeInformation);
	nd4j::buffer::freeBuffer(&shapeInfoBuferBuff);

	nd4j::array::NDArrays<double>::freeNDArrayOnGpuAndCpu(&vector);
	nd4j::buffer::freeBuffer(&vectorShapeInfoBuffData);
	nd4j::buffer::freeBuffer(&dimensionBuff);

	delete add;

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
		int *shapeBuff = shapeBuffer(this->rank,this->shape);
		int *yShapeBuff = shapeBuffer(this->yRank,this->yShape);
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
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->shape);
		nd4j::buffer::Buffer<int> *yShapeBuff = shapeIntBuffer(this->rank,this->shape);
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
	Data<double> *data = getDataBroadcast(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}


TEST(BroadCasting,ObjectOrientedSubtraction) {
	int rank = 2;
	int opNum = 1;
	double comparison[4] = {0,0,2,2};
	Data<double> *data = getDataBroadcast(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedMultiplication) {
	int rank = 2;
	int opNum = 2;
	double comparison[4] = {1,4,3,8};
	Data<double> *data = getDataBroadcast(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedDivision) {
	int rank = 2;
	int opNum = 3;
	double comparison[4] = {1,1,3,2};
	Data<double> *data = getDataBroadcast(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedReverseDivision) {
	int rank = 2;
	int opNum = 3;
	double comparison[4] = {1,1,0.33333333,0.5};
	Data<double> *data = getDataBroadcast(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}


TEST(BroadCasting,ObjectOrientedReverseSubtraction) {
	int rank = 2;
	int opNum = 5;
	double comparison[4] = {0,0,-2,-2};
	Data<double> *data = getDataBroadcast(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}

TEST(BroadCasting,ObjectOrientedCopy) {
	int rank = 2;
	int opNum = 6;
	double comparison[4] = {1,2,1,2};
	Data<double> *data = getDataBroadcast(rank,comparison);
	DoubleBroadcastTranformTest *test = new DoubleBroadcastTranformTest(rank,opNum,data,1);
	delete data;
	delete test;

}




#endif //NATIVEOPERATIONS_BROADCASTSTESTS_H
