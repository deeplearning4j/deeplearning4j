//
// Created by agibsoncccc on 1/5/16.
//

#ifndef NATIVEOPERATIONS_INDEXREDUCETESTS_H_H
#define NATIVEOPERATIONS_INDEXREDUCETESTS_H_H
#include <array.h>
#include "testhelpers.h"
#include <indexreduce.h>
#include <helper_cuda.h>
TEST_GROUP(IndexReduce) {

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
static Data<T> * getDataIndexReduce(T *assertion,T startingVal) {
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
    extraParams[0] = startingVal;
    ret->extraParams = extraParams;

    ret->assertion = (T *) malloc(sizeof(T) * 4);
    for(int i = 0; i < 1; i++) {
        ret->assertion[i] = assertion[i];
    }

    ret->dimension = (int *) malloc(sizeof(int) * 2);
    ret->dimension[0] = shape::MAX_DIMENSION;

    ret->result = (T *) malloc(sizeof(T));
    ret->resultRank = 2;
    ret->resultShape = (int *) malloc(sizeof(int) * 2);
    for(int i = 0; i < 2; i++)
        ret->resultShape[i] = 1;

    return ret;
}

template <typename T>
static Data<T> * getDataIndexReduceDimension(T *assertion,T startingVal) {
    Data<T> *ret = new Data<T>();

    int rank = 2;
    int length = 4;
    int resultLength = 2;
    ret->xShape = (int *) malloc(sizeof(int) * rank);
    ret->xShape[0] = 2;
    ret->xShape [1] = 2;
    ret->rank = 2;
    ret->data = (T *) malloc(sizeof(T) * length);
    for(int i = 0; i < length; i++)
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
    ret->result = (T *) malloc(sizeof(T) * resultLength);
    ret->resultRank = 2;
    ret->resultShape = (int *) malloc(sizeof(int) * rank);
    ret->resultShape[0] = 1;
    ret->resultShape[1] = 2;

    return ret;
}


template <typename T>
static Data<T> * getDataIndexReduceDimensionMulti(T *assertion,T startingVal) {
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

template <typename T>
class IndexReduceTest : public BaseTest<T> {
public:
    IndexReduceTest() {}
    virtual ~IndexReduceTest() {
        freeOpAndOpFactory();
    }
    IndexReduceTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
            :  BaseTest<T>(rank,opNum,data,extraParamsLength){
        createOperationAndOpFactory();
    }
    void freeOpAndOpFactory() {
        delete opFactory;
        delete reduce;
    }

    virtual void createOperationAndOpFactory() {
        opFactory = new functions::indexreduce::IndexReduceOpFactory<T>();
        reduce = opFactory->getOp(this->opNum);
    }

    virtual void execCpuKernel() override {
        int *xShapeBuff = shapeBuffer(this->baseData->rank,this->baseData->xShape);
        int *resultShapeBuff = shapeBuffer(this->baseData->resultRank,this->baseData->resultShape);
        printf("In exec cpu index reduce \n");
        reduce->exec(
                this->data->data->data,
                xShapeBuff,
                this->baseData->extraParams,
                this->result->data->data,
                resultShapeBuff,
                this->baseData->dimension,this->baseData->dimensionLength);
        free(xShapeBuff);
        free(resultShapeBuff);

    }

    virtual void run () override {
        this->initializeData();
        this->execCpuKernel();
        int resultLength = shape::prod(this->baseData->resultShape,this->baseData->rank);
        if(resultLength == 1) {
            if(this->result->data->data[0] != this->baseData->assertion[0]) {
                printf("Compared assertion index reduce %f to result %f\n",this->baseData->assertion[0],this->result->data->data[0]);
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
		printf("Executed cuda kernel\n");
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
    functions::indexreduce::IndexReduceOpFactory<T> *opFactory;
    functions::indexreduce::IndexReduce<T> *reduce;
};

class DoubleIndexReduceTest : public IndexReduceTest<double> {
public:
    virtual ~DoubleIndexReduceTest() {}
    DoubleIndexReduceTest() {}
    DoubleIndexReduceTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
            :  IndexReduceTest<double>(rank,opNum,data,extraParamsLength){
    }
    virtual void executeCudaKernel() override {
#ifdef __CUDACC__
        printf("Creating gpu information buffer\n");
        nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
        printf("Created gpu information buffer\n");
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		printf("Created dimension buffer\n");
		printf("Creating shape buffer for x\n");
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->baseData->xShape);
		printf("Created shape buffer for x\n");
        printf("Creating result shape buffer\n");
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->result->rank,this->result->shape->data);
        printf("Created result shape buffer\n");
		indexReduceDouble<<<this->blockSize,this->gridSize,this->sMemSize>>>(
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


class FloatIndexReduceTest : public IndexReduceTest<float> {
public:
    FloatIndexReduceTest() {}
    FloatIndexReduceTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
            :  IndexReduceTest<float>(rank,opNum,data,extraParamsLength){
    }
    virtual void executeCudaKernel() override {
#ifdef __CUDACC__
        nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->baseData->xShape);
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->baseData->resultRank,this->baseData->resultShape);
        indexReduceFloat<<<this->blockSize,this->gridSize,this->sMemSize>>>(
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

        printf("Ran cuda kernel\n");
		nd4j::buffer::freeBuffer(&gpuInfo);
		printf("Freed gpu info\n");
		nd4j::buffer::freeBuffer(&dimensionBuffer);
		printf("Freed dimension buffer\n");
		nd4j::buffer::freeBuffer(&xShapeBuff);
		printf("Freed x shape buffer\n");
		nd4j::buffer::freeBuffer(&resultShapeBuff);
        printf("Freed result shape buffer\n");
#endif
    }
};


TEST(IndexReduce,ObjectOrientedIMax) {
    printf("Running double imax\n");
    int rank = 2;
    int opNum = 0;
    double assertion[1] = {3};
    Data<double> *data = getDataIndexReduce<double>(assertion,0);
    DoubleIndexReduceTest *test = new DoubleIndexReduceTest(rank,opNum,data,1);
    test->run();
    delete data;
    delete test;
    printf("Ran double imax\n");
}

TEST(IndexReduce,ObjectOrientedIMin) {
    printf("Running double imin\n");
    int rank = 2;
    int opNum = 1;
    double assertion[1] = {0};
    Data<double> *data = getDataIndexReduce<double>(assertion,0);
    DoubleIndexReduceTest *test = new DoubleIndexReduceTest(rank,opNum,data,1);
    test->run();
    delete data;
    delete test;
    printf("Ran double imin\n");
}




TEST(IndexReduce,ObjectOrientedDimensionIMax) {
    printf("Running dimension double imax\n");
    int rank = 2;
    int opNum = 0;
    double assertion[2] = {1,1};
    Data<double> *data = getDataIndexReduceDimension<double>(assertion,0);
    DoubleIndexReduceTest *test = new DoubleIndexReduceTest(rank,opNum,data,1);
    test->run();
    delete data;
    delete test;
    printf("Ran dimension imax double\n");
}

TEST(IndexReduce,ObjectOrientedDimensionIMin) {
    printf("Running double dimension imin\n");
    int rank = 2;
    int opNum = 1;
    double assertion[2] = {0,0};
    Data<double> *data = getDataIndexReduceDimension<double>(assertion,0);
    DoubleIndexReduceTest *test = new DoubleIndexReduceTest(rank,opNum,data,1);
    test->run();
    delete data;
    delete test;
    printf("Ran double dimension imin");
}




TEST(IndexReduce,ObjectOrientedFloatIMax) {
    printf("Running float imax\n");
    int rank = 2;
    int opNum = 0;
    float assertion[1] = {3};
    Data<float> *data = getDataIndexReduce<float>(assertion,0);
    FloatIndexReduceTest *test = new FloatIndexReduceTest(rank,opNum,data,1);
    test->run();
    delete data;
    delete test;
}

TEST(IndexReduce,ObjectOrientedFloatIMin) {
    int rank = 2;
    int opNum = 1;
    float assertion[1] = {0};

    Data<float> *data = getDataIndexReduce<float>(assertion,0);
    FloatIndexReduceTest *test = new FloatIndexReduceTest(rank,opNum,data,1);
    test->run();

    delete data;
    delete test;
}

TEST(IndexReduce,ObjectOrientedFloatDimensionIMax) {
    int rank = 2;
    int opNum = 0;
    float assertion[2] = {1,1};

    Data<float> *data = getDataIndexReduceDimension<float>(assertion,0);
    FloatIndexReduceTest *test = new FloatIndexReduceTest(rank,opNum,data,1);
    test->run();

    delete data;
    delete test;
}

TEST(IndexReduce,ObjectOrientedFloatDimensionIMin) {
    int rank = 2;
    int opNum = 1;
    float assertion[2] = {0,0};

    Data<float> *data = getDataIndexReduceDimension<float>(assertion,0);
    FloatIndexReduceTest *test = new FloatIndexReduceTest(rank,opNum,data,1);
    test->run();

    delete data;
    delete test;
}


TEST(IndexReduce,ObjectOrientedFloatDimensionIMinMulti) {
    int rank = 2;
    int opNum = 1;
    float assertion[3] = {0,0,0};
    Data<float> *data = getDataIndexReduceDimensionMulti<float>(assertion,0);
    FloatIndexReduceTest *test = new FloatIndexReduceTest(rank,opNum,data,1);
    test->run();

    delete data;
    delete test;
}


TEST(IndexReduce,ObjectOrientedFloatDimensionIMaxMulti) {
    int rank = 2;
    int opNum = 0;
    float assertion[3] = {3,3,3};

    Data<float> *data = getDataIndexReduceDimensionMulti<float>(assertion,0);
    FloatIndexReduceTest *test = new FloatIndexReduceTest(rank,opNum,data,1);
    test->run();

    delete data;
    delete test;
}
#endif //NATIVEOPERATIONS_INDEXREDUCETESTS_H_H
