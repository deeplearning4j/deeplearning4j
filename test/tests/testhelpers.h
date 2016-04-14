/*
 * testhelpers.h
 *
 *  Created on: Jan 1, 2016
 *      Author: agibsonccc
 */

#ifndef TESTHELPERS_H_
#define TESTHELPERS_H_
#include <CppUTest/TestHarness.h>
#include <CppUTest/CommandLineTestRunner.h>
#include <templatemath.h>
#include <buffer.h>
#include <array.h>

#include "optype.h"

template<typename T>
inline int arrsEquals(int rank, const T comp1[], const T comp2[]) {
    for (int i = 0; i < rank; i++) {
        DOUBLES_EQUAL(comp1[i],comp2[i],1e-1);
    }

    return 1;
}

template <typename T>
class Data {
public:
    ~Data() {
        //NOTE: delete nullptr; is well defined!
        // but CppUnit does seem to mind...

        if(xShape)
            delete []xShape;

        if(yShape && yShape != xShape)
            delete []yShape;

        if(resultShape)
            delete []resultShape;

        if(data)
            delete []data;

        if(dimension)
            delete []dimension;

        if(assertion)
            delete []assertion;

        if(y)
            delete []y;

        if(result)
            delete []result;

        if(extraParams)
            delete []extraParams;
    }

    T scalar;
    T *data = nullptr;
    T *y = nullptr;
    T *result = nullptr;
    T *extraParams = nullptr;
    T *assertion = nullptr;
    int *xShape = nullptr;
    int *yShape = nullptr;
    int *resultShape = nullptr;
    int rank;
    int yRank;
    int resultRank;
    int *dimension = nullptr;
    int dimensionLength;

};

template <typename T>
void freeData(Data<T> *data);


/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
inline int *shapeBuffer(int rank, int *shape) {
    int *stride = shape::calcStrides(shape, rank);
    shape::ShapeInformation * shapeInfo = new shape::ShapeInformation(shape, stride, 0, rank);
    int elementWiseStride = shape::computeElementWiseStride(rank, shape, stride,
                                                            0);
    if(elementWiseStride < 1)
        elementWiseStride = 1;
    shapeInfo->elementWiseStride = elementWiseStride;
    int *shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
    return shapeInfoBuffer;
}

inline void assertBufferProperties(int *shapeBuffer) {
    CHECK(shape::rank(shapeBuffer) >= 2);
    CHECK(shape::length(shapeBuffer) >= 1);
    CHECK(shape::elementWiseStride(shapeBuffer) >= 1);
}


inline nd4j::buffer::Buffer<int>* shapeIntBuffer(int rank, int *shape) {
    int *shapeBuffRet = shapeBuffer(rank,shape);
    nd4j::buffer::Buffer<int> *ret = nd4j::buffer::createBuffer(shapeBuffRet,shape::shapeInfoLength(rank));
    return ret;
}

inline nd4j::buffer::Buffer<int>* gpuInformationBuffer(int blockSize, int gridSize, int sharedMemorySize) {
    int *ret = (int *) malloc(sizeof(int) * 4);
    ret[0] = blockSize;
    ret[1] = gridSize;
    ret[2] = sharedMemorySize;
    ret[3] = sharedMemorySize;
    nd4j::buffer::Buffer<int> *ret2 = nd4j::buffer::createBuffer(ret,4);
    return ret2;
}


template <typename T>
class BaseTest {


public:
    BaseTest() {
    }
    BaseTest(int rank_, int opNum_,Data<T> *data, int extraParamsLength_)
        : rank(rank_), baseData(data), opNum(opNum_), extraParamsLength(extraParamsLength_) {
        init();
    }

    virtual ~BaseTest() {
        if(data != NULL)
            nd4j::array::NDArrays<T>::freeNDArrayOnGpuAndCpu(data);

        if(extraParamsBuff != NULL)
            nd4j::buffer::freeBuffer(extraParamsBuff);

        delete []result;
    }

    virtual nd4j::buffer::Buffer<int> * gpuInformationBuffer() {
        int *ret = (int *) malloc(sizeof(int) * 4);
        ret[0] = blockSize;
        ret[1] = gridSize;
        ret[2] = sMemSize;
        ret[3] = sMemSize;
        nd4j::buffer::Buffer<int> *ret2 = nd4j::buffer::createBuffer(ret,4);
        return ret2;
    }

protected:
    int rank;
    Data<T> *baseData;
    nd4j::array::NDArray<T> *data = NULL;
    nd4j::array::NDArray<T> *result = NULL;
    T *assertion = NULL;
    T *extraParams = NULL;
    int blockSize = 32;
    int gridSize = 64;
    int sMemSize = 3000;
    nd4j::buffer::Buffer<T> *extraParamsBuff = NULL;
    int length;
    int opNum;
    int extraParamsLength;
    virtual void executeCudaKernel() = 0;
    virtual void execCpuKernel() = 0;
    virtual void freeOpAndOpFactory() = 0;
    virtual void createOperationAndOpFactory() = 0;
    virtual void run() = 0;

    virtual void init() {
        rank = this->baseData->rank;
        int *stride = shape::calcStrides(this->baseData->xShape, rank);
        data = nd4j::array::NDArrays<T>::createFrom(rank, this->baseData->xShape, stride, 0,
                                                    0.0);
        length = nd4j::array::NDArrays<T>::length(data);
        extraParams = this->baseData->extraParams;

        extraParamsBuff = nd4j::buffer::createBuffer(extraParams,extraParamsLength);
        assertion = this->baseData->assertion;
        int resultLength = shape::prod(this->baseData->resultShape,this->baseData->resultRank);


        int *resultStride = shape::calcStrides(this->baseData->resultShape,this->baseData->resultRank);
        result = nd4j::array::NDArrays<T>::createFrom(
                this->baseData->resultRank
                , this->baseData->resultShape,
                resultStride, 0,
                0.0);

        for(int i = 0; i < resultLength; i++) {
            result->data->data[i] = baseData->result[i];
        }

        delete []resultStride;
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
    nd4j::array::NDArray<T> *yData;

public:
    PairWiseTest() {
    }
    //BaseTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
    PairWiseTest(int rank, int opNum, Data<T> *data, int extraParamsLength)
            : BaseTest<T>(rank,opNum,data,extraParamsLength)  {
        yRank = this->baseData->yRank;
        yShape = this->baseData->yShape;
        yStride = shape::calcStrides(yShape,yRank);
        yData = nd4j::array::NDArrays<T>::createFrom(this->baseData->y,yRank, yShape, yStride, 0);
    }

    virtual ~PairWiseTest() {
        delete yData;
        delete []yStride;
    }
};



template <typename T>
class TwoByTwoTest : public BaseTest<T> {
public:
    virtual ~TwoByTwoTest() {}
    TwoByTwoTest(int rank, int opNum,Data<T> *data,int extraParamsLength) : BaseTest<T>(rank,opNum,data,extraParamsLength) {}
    TwoByTwoTest(int opNum,Data<T> *data,int extraParamsLength) : BaseTest<T>(2,opNum,data,extraParamsLength) {}
    virtual void initShape() override {
        for(int i = 0; i < 2; i++) {
            this->shape[i] = 2;
        }
    }
protected:
    typedef BaseTest<T> super;
};






#endif /* TESTHELPERS_H_ */
