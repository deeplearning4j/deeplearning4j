/*
 * testhelpers.h
 *
 *  Created on: Jan 1, 2016
 *      Author: agibsonccc
 */

#ifndef TESTHELPERS_H_
#define TESTHELPERS_H_
#include <CppUTest/TestHarness.h>
#include <templatemath.h>
int arrsEquals(int rank, int *comp1, int *comp2);

template<typename T>
int arrsEquals(int rank, T *comp1, T *comp2) {
	int ret = 1;
	for (int i = 0; i < rank; i++) {
		T eps = nd4j::math::nd4j_abs(comp1[i] - comp2[i]);
		ret = ret && (eps < 1e-3);
	}

	return ret;
}

/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
int *shapeBuffer(int rank, int *shape) {
	int *stride = shape::calcStrides(shape, rank);
	shape::ShapeInformation * shapeInfo = (shape::ShapeInformation *) malloc(
			sizeof(shape::ShapeInformation));
	shapeInfo->shape = shape;
	shapeInfo->stride = stride;
	shapeInfo->offset = 0;
	shapeInfo->rank = rank;
	int elementWiseStride = shape::computeElementWiseStride(rank, shape, stride,
			0);
	shapeInfo->elementWiseStride = elementWiseStride;
	int *shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
	free(shapeInfo);
	return shapeInfoBuffer;
}

void assertBufferProperties(int *shapeBuffer) {
	CHECK(shape::rank(shapeBuffer) >= 2);
	CHECK(shape::length(shapeBuffer) >= 1);
	CHECK(shape::elementWiseStride(shapeBuffer) >= 1);
}





template <typename T>
class BaseTest {


public:

	BaseTest(int rank,int opNum,int extraParamsLength) {
		this->rank = rank;
		this->opNum = opNum;
		this->extraParams = extraParamsLength;
		init();
	}



	virtual ~BaseTest() {
		nd4j::array::NDArrays<T>::freeNDArrayOnGpuAndCpu(&data);
		nd4j::array::NDArrays<T>::freeNDArrayOnGpuAndCpu(&extraParamsBuff);
		freeOpAndOpFactory();
		freeAssertion();
	}





protected:
	int rank;
	int *shape;
	int *stride;
	nd4j::array::NDArray<T> *data;
	T *assertion;
	T *extraParams;
	int blockSize = 500;
	int gridSize = 256;
	int sMemSize = 20000;
	nd4j::buffer::Buffer<T> *extraParamsBuff;
	int length;
	int opNum;
	int extraParamsLength;
	virtual T *getAssertion() = 0;
	virtual void freeAssertion() = 0;
	virtual void freeOpAndOpFactory() = 0;
	virtual void createOperationAndOpFactory() = 0;
	virtual void run() = 0;
	virtual void init() {
		shape = (int *) malloc(sizeof(int) * rank);
		stride = shape::calcStrides(shape, rank);
		data = nd4j::array::NDArrays<T>::createFrom(rank, shape, stride, 0,
				0.0);
		initializeData();
		length = nd4j::array::NDArrays<T>::length(data);
		extraParams = (T *) malloc(sizeof(T) * extraParamsLength);
		extraParamsBuff = nd4j::buffer::createBuffer(extraParams,extraParamsLength);
		createOperationAndOpFactory();
		assertion = getAssertion();
	}


	virtual void initializeData() {
		for (int i = 0; i < length; i++)
			data->data->data[i] = i + 1;
	}

};

template <typename T>
class PairWiseTest : public BaseTest<T> {
protected:
	int yRank;
	int *yShape;
	int *yStride;
	typedef BaseTest<T> super;
	nd4j::array::NDArray<T> *yData;

public:
	virtual ~PairWiseTest() {}
	virtual void init() override {
		super::init();
		yShape = (int *) malloc(sizeof(int) * yRank);
		yStride = shape::calcStrides(yShape,yRank);
		yData = nd4j::array::NDArrays<T>::createFrom(yRank, yShape, yStride, 0,
						0.0);
	}


};

template <typename T>
class DimensionTest : public BaseTest<T> {
public:
	DimensionTest(int rank,int opNum,int extraParamsLength,int dimensionLength) : BaseTest<T>(rank,opNum,extraParamsLength,dimensionLength){
		createDimension();
		initDimension();
	}

	virtual ~DimensionTest() {}
	void createDimension() {
		dimension = (int *) malloc(sizeof(dimension) * dimensionLength);
	}

protected:
	virtual void initDimension() = 0;
    typedef BaseTest<T> super;



protected:
	int dimensionLength = 1;
	int *dimension;
};



#endif /* TESTHELPERS_H_ */
