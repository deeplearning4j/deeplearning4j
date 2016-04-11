/*
 * reducetests.h
 *
 *  Created on: Dec 31, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCETESTS_H_
#define REDUCETESTS_H_
#include <array.h>
#include "testhelpers.h"
#include <reduce.h>
#include <helper_cuda.h>

TEST_GROUP(Reduce) {

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
class ReduceTest : public BaseTest<T> {
public:
    virtual ~ReduceTest() {
        freeOpAndOpFactory();
    }
    ReduceTest() {
        createOperationAndOpFactory();
    }
    ReduceTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
            :  BaseTest<T>(rank,opNum,data,extraParamsLength){
        createOperationAndOpFactory();
    }
    void freeOpAndOpFactory() {
        delete opFactory;
        delete reduce;
    }
    virtual void createOperationAndOpFactory() {
        opFactory = new functions::reduce::ReduceOpFactory<T>();
        reduce = opFactory->create(this->opNum);
    }

    virtual void execCpuKernel() override {
        int *xShapeBuff = shapeBuffer(this->baseData->rank,this->baseData->xShape);
        int *resultShapeBuff = shapeBuffer(this->baseData->resultRank,this->baseData->resultShape);
        assertBufferProperties(xShapeBuff);
        assertBufferProperties(resultShapeBuff);
        T *resultData = this->result->data->data;
        reduce->exec(
                this->baseData->data,
                xShapeBuff,
                this->baseData->extraParams,
                resultData,
                resultShapeBuff,
                this->baseData->dimension,
                this->baseData->dimensionLength);
        free(xShapeBuff);
        free(resultShapeBuff);
    }

    virtual void run () override {
        this->initializeData();
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
    functions::reduce::ReduceOpFactory<T> *opFactory;
    functions::reduce::ReduceFunction<T> *reduce;
};

class DoubleReduceTest : public  ReduceTest<double> {
public:
    virtual ~DoubleReduceTest() {}
    DoubleReduceTest() {}
    DoubleReduceTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
            :  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    }
    virtual void executeCudaKernel() override {
#ifdef __CUDACC__
        nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->baseData->xShape);
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->result->rank,this->result->shape->data);

		reduceDouble<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->length,
				this->data->data->gData,
				xShapeBuff->gData,
				extraParamsBuff->gData,
				this->result->data->gData,
				resultShapeBuff->gData,
				gpuInfo->gData,
				dimensionBuffer->gData,
				this->baseData->dimensionLength,
				1
		);

		nd4j::buffer::freeBuffer(&gpuInfo);
		nd4j::buffer::freeBuffer(&dimensionBuffer);
		nd4j::buffer::freeBuffer(&xShapeBuff);
		nd4j::buffer::freeBuffer(&resultShapeBuff);
#endif


    }
};


class FloatReduceTest : public ReduceTest<float> {
public:
    FloatReduceTest() {}
    FloatReduceTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
            :  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    }
    virtual void executeCudaKernel() override {
#ifdef __CUDACC__
        nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->baseData->xShape);
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->result->rank,this->result->shape->data);

		reduceFloat<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum,
				this->length,
				this->data->data->gData,
				xShapeBuff->gData,
				extraParamsBuff->gData,
				this->result->data->gData,
				resultShapeBuff->gData,
				gpuInfo->gData,
				dimensionBuffer->gData,
				this->baseData->dimensionLength,
				1
		);

		nd4j::buffer::freeBuffer(&gpuInfo);
		nd4j::buffer::freeBuffer(&dimensionBuffer);
		nd4j::buffer::freeBuffer(&xShapeBuff);
		nd4j::buffer::freeBuffer(&resultShapeBuff);
#endif
    }
};


template <typename T>
static Data<T> * getDataReduceOneFiftyByFour(T *assertion,T startingVal) {
    Data<T> *ret = new Data<T>();

    int rank = 2;
    int length = 600;
    int resultLength = 4;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 150;
    shape[1] = 4;
    ret->xShape = shape;
    ret->rank = 2;
    ret->data = (T *) malloc(sizeof(T) * length);
    for(int i = 0; i < length; i++)
        ret->data[i] = i + 1;
    T *extraParams = (T *) malloc(sizeof(T) * 4);
    for(int i = 0; i < 4; i++)
        extraParams[i] = startingVal;
    ret->extraParams = extraParams;

    ret->assertion = (T *) malloc(sizeof(T) * resultLength);
    for(int i = 0; i < resultLength; i++) {
        ret->assertion[i] = assertion[i];
    }

    ret->dimension = (int *) malloc(sizeof(int) * 2);
    ret->dimension[0] = 0;
    ret->dimensionLength = 1;

    ret->result = (T *) malloc(sizeof(T) * resultLength);
    ret->resultRank = 2;
    ret->resultShape = (int *) malloc(sizeof(int) * 2);
    ret->resultShape[0] = 1;
    ret->resultShape[1] = 4;

    return ret;
}


template <typename T>
static Data<T> * getDataReduceTwoByThree(T *assertion,T startingVal) {
    Data<T> *ret = new Data<T>();

    int rank = 2;
    int length = 6;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 3;
    ret->xShape = shape;
    ret->rank = 2;
    ret->data = (T *) malloc(sizeof(T) * length);
    for(int i = 0; i < length; i++)
        ret->data[i] = i + 1;
    T *extraParams = (T *) malloc(sizeof(T) * 4);
    for(int i = 0; i < 4; i++)
        extraParams[i] = startingVal;
    ret->extraParams = extraParams;

    ret->assertion = (T *) malloc(sizeof(T) * 4);
    for(int i = 0; i < 3; i++) {
        ret->assertion[i] = assertion[i];
    }

    ret->dimension = (int *) malloc(sizeof(int) * 2);
    ret->dimension[0] = 0;
    ret->dimensionLength = 1;

    ret->result = (T *) malloc(sizeof(T));
    ret->resultRank = 2;
    ret->resultShape = (int *) malloc(sizeof(int) * 2);
    ret->resultShape[0] = 1;
    ret->resultShape[1] = 3;
    return ret;
}

template <typename T>
static Data<T> * getDataReduce(T *assertion,T startingVal) {
    Data<T> *ret = new Data<T>();

    int rank = 2;
    int length = 4;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 1;
    shape[1] = length;
    ret->xShape = shape;
    ret->rank = 2;
    ret->data = (T *) malloc(sizeof(T) * 4);
    for(int i = 0; i < 4; i++)
        ret->data[i] = i + 1;
    T *extraParams = (T *) malloc(sizeof(T) * 4);
    for(int i = 0; i < 4; i++)
        extraParams[i] = startingVal;
    ret->extraParams = extraParams;

    ret->assertion = (T *) malloc(sizeof(T) * 4);
    for(int i = 0; i < 1; i++) {
        ret->assertion[i] = assertion[i];
    }

    ret->dimension = (int *) malloc(sizeof(int) * 2);
    ret->dimension[0] = MAX_DIMENSION;

    ret->result = (T *) malloc(sizeof(T));
    ret->resultRank = 2;
    ret->resultShape = (int *) malloc(sizeof(int) * 2);
    for(int i = 0; i < 2; i++)
        ret->resultShape[i] = 1;

    return ret;
}

template <typename T>
static Data<T> * getDataReduceDimension(T *assertion,T startingVal) {
    Data<T> *ret = new Data<T>();

    int rank = 2;
    int length = 4;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    ret->xShape = shape;
    ret->rank = 2;
    ret->data = (T *) malloc(sizeof(T) * length);
    for(int i = 0; i < 4; i++)
        ret->data[i] = i + 1;
    T *extraParams = (T *) malloc(sizeof(T) * 4);
    extraParams[0] = startingVal;
    ret->extraParams = extraParams;

    ret->assertion = (T *) malloc(sizeof(T) * 4);
    for(int i = 0; i < 2; i++) {
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
static Data<T> * getDataReduceDimensionMulti(T *assertion,T startingVal) {
    Data<T> *ret = new Data<T>();

    int rank = 3;
    int resultRank = 2;
    int length = 12;
    int resultLength = 3;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    shape[2] = 3;
    ret->xShape = shape;
    ret->rank = rank;
    ret->data = (T *) malloc(sizeof(T) * length);
    for(int i = 0; i < length; i++)
        ret->data[i] = i + 1;
    T *extraParams = (T *) malloc(sizeof(T) * 4);
    extraParams[0] = startingVal;
    ret->extraParams = extraParams;

    ret->assertion = (T *) malloc(sizeof(T) * resultLength);
    for(int i = 0; i < resultLength; i++) {
        ret->assertion[i] = assertion[i];
    }

    ret->dimension = (int *) malloc(sizeof(int) * 2);
    ret->dimension[0] = 0;
    ret->dimension[1] = 1;
    ret->dimensionLength = 2;
    ret->result = (T *) malloc(sizeof(T) * resultLength);
    ret->resultRank = 2;
    ret->resultShape = (int *) malloc(sizeof(int) * resultRank);
    ret->resultShape[0] = 1;
    ret->resultShape[1] = 3;

    return ret;
}

TEST(Reduce,ObjectOrientedSum) {
    int opNum = 1;
    double comparison[1] = {10};

    Data<double> *data = getDataReduce<double>(comparison,0);
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedMax) {
    int opNum = 3;
    double comparison[1] = {4};
    Data<double> *data = getDataReduce<double>(comparison,0);
    data->extraParams[0] = data->data[0];
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}



TEST(Reduce,ObjectOrientedMin) {
    int opNum = 4;
    double comparison[1] = {1};
    Data<double> *data = getDataReduce<double>(comparison,0);
    data->extraParams[0] = data->data[0];
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}


TEST(Reduce,ObjectOrientedNorm1) {
    int opNum = 5;
    double comparison[1] = {10.0};
    Data<double> *data = getDataReduce<double>(comparison,0);
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}


TEST(Reduce,ObjectOrientedNorm2) {
    int opNum = 6;
    double comparison[1] = {5.4772255750516612};
    Data<double> *data = getDataReduce<double>(comparison,0);
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}


TEST(Reduce,ObjectOrientedMean) {
    int opNum = 0;
    double comparison[1] = {2.5};
    Data<double> *data = getDataReduce<double>(comparison,0);
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}



TEST(Reduce,ObjectOrientedProd) {
    int opNum = 8;
    double comparison[1] = {24};
    Data<double> *data = getDataReduce<double>(comparison,0);
    data->extraParams[0] = 1.0;
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}









TEST(Reduce,ObjectOrientedDimensionMax) {
    int opNum = 3;
    double comparison[2] = {2.00, 4.00};
    Data<double> *data = getDataReduceDimension<double>(comparison,0);
    data->extraParams[0] = data->data[0];
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}


TEST(Reduce,ObjectOrientedDimensionSum) {
    int opNum = 1;
    double comparison[2] = {3,7};

    Data<double> *data = getDataReduceDimension<double>(comparison,0);
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedDimensionMin) {
    int opNum = 4;
    double comparison[2] = {1.00, 3.00};
    Data<double> *data = getDataReduceDimension<double>(comparison,0);
    data->extraParams[0] = data->data[0];
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}



TEST(Reduce,ObjectOrientedDimensionNorm1) {
    int opNum = 5;
    double comparison[2] = {3.00, 7.00};
    Data<double> *data = getDataReduceDimension<double>(comparison,0);
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}





TEST(Reduce,ObjectOrientedDimensionMean) {
    int opNum = 0;
    double comparison[2] = {1.50, 3.50};
    Data<double> *data = getDataReduceDimension<double>(comparison,0);
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedDimensionProd) {
    int opNum = 8;
    double comparison[2] = {2.00, 12.00};
    Data<double> *data = getDataReduceDimension<double>(comparison,0);
    data->extraParams[0] = 1.0;
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedDimensionNorm2) {
    int opNum = 6;
    double comparison[2] = {2.24, 5.00};
    Data<double> *data = getDataReduceDimension<double>(comparison,0);
//	:  ReduceTest<double>(rank,opNum,data,extraParamsLength){
    DoubleReduceTest *test = new DoubleReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}



TEST(Reduce,ObjectOrientedFloatSum) {
    int opNum = 1;
    float comparison[1] = {10};

    Data<float> *data = getDataReduce<float>(comparison,0);
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedFloatMax) {
    int opNum = 3;
    float comparison[1] = {4};
    Data<float> *data = getDataReduce<float>(comparison,0);
    data->extraParams[0] = data->data[0];
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}



TEST(Reduce,ObjectOrientedFloatMin) {
    int opNum = 4;
    float comparison[1] = {1};
    Data<float> *data = getDataReduce<float>(comparison,0);
    data->extraParams[0] = data->data[0];
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}


TEST(Reduce,ObjectOrientedFloatNorm1) {
    int opNum = 5;
    float comparison[1] = {10.0};
    Data<float> *data = getDataReduce<float>(comparison,0);
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}


TEST(Reduce,ObjectOrientedFloatNorm2) {
    int opNum = 6;
    float comparison[1] = {5.4772255750516612};
    Data<float> *data = getDataReduce<float>(comparison,0);
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}


TEST(Reduce,ObjectOrientedFloatMean) {
    int opNum = 0;
    float comparison[1] = {2.5};
    Data<float> *data = getDataReduce<float>(comparison,0);
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}



TEST(Reduce,ObjectOrientedFloatProd) {
    int opNum = 8;
    float comparison[1] = {24};
    Data<float> *data = getDataReduce<float>(comparison,0);
    data->extraParams[0] = 1.0;
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}









TEST(Reduce,ObjectOrientedFloatDimensionMax) {
    int opNum = 3;
    float comparison[2] = {2.00, 4.00};
    Data<float> *data = getDataReduceDimension<float>(comparison,0);
    data->extraParams[0] = data->data[0];
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}


TEST(Reduce,ObjectOrientedFloatDimensionSum) {
    int opNum = 1;
    float comparison[2] = {3,7};

    Data<float> *data = getDataReduceDimension<float>(comparison,0);
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedFloatDimensionMin) {
    int opNum = 4;
    float comparison[2] = {1.00, 3.00};
    Data<float> *data = getDataReduceDimension<float>(comparison,0);
    data->extraParams[0] = data->data[0];
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}



TEST(Reduce,ObjectOrientedFloatDimensionNorm1) {
    int opNum = 5;
    float comparison[2] = {3.00, 7.00};
    Data<float> *data = getDataReduceDimension<float>(comparison,0);
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}





TEST(Reduce,ObjectOrientedFloatDimensionMean) {
    int opNum = 0;
    float comparison[2] = {1.50, 3.50};
    Data<float> *data = getDataReduceDimension<float>(comparison,0);
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedFloatDimensionProd) {
    int opNum = 8;
    float comparison[2] = {2.00, 12.00};
    Data<float> *data = getDataReduceDimension<float>(comparison,0);
    data->extraParams[0] = 1.0;
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedFloatDimensionNorm2) {
    int opNum = 6;
    float comparison[2] = {2.24, 5.00};
    Data<float> *data = getDataReduceDimension<float>(comparison,0);
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedSumDimensionMulti) {
    int opNum = 1;
    float comparison[3] = {22.00, 26.00, 30.00};
    Data<float> *data = getDataReduceDimensionMulti<float>(comparison,0);
    data->extraParams[0] = 1.0;
    FloatReduceTest *test = new FloatReduceTest(3,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedMeanDimensionMulti) {
    int opNum = 0;
    float comparison[3] = {5.5,  6.5,  7.5};
    Data<float> *data = getDataReduceDimensionMulti<float>(comparison,0);
    data->extraParams[0] = 1.0;
    FloatReduceTest *test = new FloatReduceTest(3,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedFloatDimensionSumOneFiftyByFour) {
    int opNum = 1;
    float comparison[4] = {44850.0f, 45000.0f, 45150.0f, 45300.0f};

    Data<float> *data = getDataReduceOneFiftyByFour<float>(comparison,0);
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}

TEST(Reduce,ObjectOrientedFloatDimensionSumTwoByThree) {
    printf("Running target test\n");
    int opNum = 1;
    float comparison[3] = {5.,  7.,  9.};

    Data<float> *data = getDataReduceTwoByThree<float>(comparison,0);
//	:  ReduceTest<float>(rank,opNum,data,extraParamsLength){
    FloatReduceTest *test = new FloatReduceTest(2,opNum,data,1);
    test->run();
    delete test;
    delete data;
}


#endif /* REDUCETESTS_H_ */
