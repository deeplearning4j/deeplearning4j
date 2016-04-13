/*
 * summarystatsreducetest.h
 *
 *  Created on: Jan 23, 2016
 *      Author: agibsonccc
 */

#ifndef SUMMARYSTATSREDUCETEST_H_
#define SUMMARYSTATSREDUCETEST_H_

#include <array.h>
#include "testhelpers.h"
#include <summarystatsreduce.h>
#include <helper_cuda.h>

TEST_GROUP(SummaryStatsReduce) {

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
static Data<T> * getDataSummary(const T assertion[2],T startingVal) {
    Data<T> *ret = new Data<T>();

    int rank = 2;
    int length = 4;
    int *shape = new int[rank];
    shape[0] = 1;
    shape[1] = length;
    ret->xShape = shape;
    ret->rank = 2;
    ret->data = new T[4];
    for(int i = 0; i < 4; i++)
        ret->data[i] = i + 1;
    T *extraParams = new T[4];
    extraParams[0] = startingVal;
    ret->extraParams = extraParams;

    ret->assertion = new T[4];
    for(int i = 0; i < 1; i++) {
        ret->assertion[i] = assertion[i];
    }

    ret->dimension = new int[2];
    ret->dimension[0] = MAX_DIMENSION;

    ret->result = new T;
    ret->resultRank = 2;
    ret->resultShape = new int[2];
    for(int i = 0; i < 2; i++)
        ret->resultShape[i] = 1;

    return ret;
}

template <typename T>
inline Data<T> * getDataSummaryDimension(const T assertion[2],T startingVal) {
    Data<T> *ret = new Data<T>();

    int rank = 2;
    int length = 4;
    int *shape = new int[rank];
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
inline Data<T>* getDataSummaryDimensionMulti(const T assertion[3], T startingVal) {
    auto ret = new Data<T>;

    constexpr int rank = 3;
    constexpr int resultRank = 2;
    constexpr int length = 12;
    constexpr int resultLength = 3;

    int *shape = new int[rank];
    shape[0] = 2;
    shape[1] = 2;
    shape[2] = 3;

    ret->xShape = shape;
    ret->rank = rank;
    ret->data = new T[length];

    for(int i = 0; i < length; i++)
        ret->data[i] = i + 1;

    T *extraParams = new T[4];
    extraParams[0] = startingVal;
    ret->extraParams = extraParams;

    ret->assertion = new T[resultLength];
    memcpy(ret->assertion, assertion, sizeof(T)*resultLength);

    ret->dimension = new int[resultRank];
    ret->dimension[0] = 0;
    ret->dimension[1] = 1;
    ret->dimensionLength = 2;

    ret->result = new T[resultLength];
    ret->resultRank = 2;

    ret->resultShape = new int [resultRank];
    ret->resultShape[0] = 1;
    ret->resultShape[1] = 3;

    return ret;
}


template <typename T>
class SummaryStatsReduceTest : public BaseTest<T> {
public:
    SummaryStatsReduceTest() {
        createOperationAndOpFactory();
    }
    virtual ~SummaryStatsReduceTest() {
        freeOpAndOpFactory();
        this->blockSize = 32;
        this-> gridSize = 64;
        this->sMemSize = 20000;

    }
    SummaryStatsReduceTest(int rank, int opNum, Data<T> *data,int extraParamsLength)
            :  BaseTest<T>(rank,opNum,data,extraParamsLength){
        createOperationAndOpFactory();

    }
    void freeOpAndOpFactory() {
        delete opFactory;
        delete reduce;
    }

    virtual void createOperationAndOpFactory() {
        opFactory = new functions::summarystats::SummaryStatsReduceOpFactory<T>();
        reduce = opFactory->getOp(this->opNum);
    }

    virtual void execCpuKernel() override {
        int *xShapeBuff = shapeBuffer(this->baseData->rank,this->baseData->xShape);
        int *resultShapeBuff = shapeBuffer(this->baseData->resultRank,this->baseData->resultShape);
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
				for(int i = 0; i < resultLength; i++) {
					printf("Result[%d] is %f\n",i,this->result->data->data[i]);
				}
				printf("Compared assertion gpu %f to result %f\n",this->baseData->assertion[0],this->baseData->result[0]);
			}
			DOUBLES_EQUAL(this->baseData->assertion[0],this->result->data->data[0],1e-3);
		}
		else {
			CHECK(arrsEquals(this->rank, this->assertion, this->result->data->data));


		}

#endif


    }


protected:
    functions::summarystats::SummaryStatsReduceOpFactory<T> *opFactory;
    functions::summarystats::SummaryStatsReduce<T> *reduce;
};

class DoubleSummaryStatsReduceTest : public SummaryStatsReduceTest<double> {
public:
    virtual ~DoubleSummaryStatsReduceTest() {}
    DoubleSummaryStatsReduceTest() {}
    DoubleSummaryStatsReduceTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
            :  SummaryStatsReduceTest<double>(rank,opNum,data,extraParamsLength){
    }
    virtual void executeCudaKernel() override {
#ifdef __CUDACC__
        nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->baseData->xShape);
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->result->rank,this->result->shape->data);

		summaryStatsReduceDouble<<<this->blockSize,this->gridSize,this->sMemSize>>>(
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


class FloatSummaryStatsReduceTest : public SummaryStatsReduceTest<float> {
public:
    FloatSummaryStatsReduceTest() {}
    FloatSummaryStatsReduceTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
            :  SummaryStatsReduceTest<float>(rank,opNum,data,extraParamsLength){
    }
    virtual void executeCudaKernel() override {
#ifdef __CUDACC__
        nd4j::buffer::Buffer<int> *gpuInfo = this->gpuInformationBuffer();
		nd4j::buffer::Buffer<int> *dimensionBuffer = nd4j::buffer::createBuffer(this->baseData->dimension,this->baseData->dimensionLength);
		nd4j::buffer::Buffer<int> *xShapeBuff = shapeIntBuffer(this->rank,this->baseData->xShape);
		nd4j::buffer::Buffer<int> *resultShapeBuff = shapeIntBuffer(this->result->rank,this->result->shape->data);

		summaryStatsReduceFloat<<<this->blockSize,this->gridSize,this->sMemSize>>>(
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

TEST(SummaryStatsReduce,ObjectOrientedStandardDeviation) {
    constexpr int rank = 2;
    constexpr double assertion[] = {1.29099440574646};

    Data<double> *data = getDataSummary<double>(assertion,0);
    DoubleSummaryStatsReduceTest test(rank, op_type::StandardDeviation, data, 1);

    test.run();

    delete data;
}

TEST(SummaryStatsReduce,ObjectOrientedVariance) {
    constexpr int rank = 2;
    constexpr double assertion[] = {1.66667};

    Data<double> *data = getDataSummary<double>(assertion,0);
    DoubleSummaryStatsReduceTest test(rank, op_type::Variance, data, 1);

    test.run();

    delete data;
}

TEST(SummaryStatsReduce,ObjectOrientedDimensionStandardDeviation) {
    constexpr int rank = 2;
    constexpr double assertion[] = { 0.71, 0.71};

    Data<double> *data = getDataSummaryDimension<double>(assertion,0);
    DoubleSummaryStatsReduceTest test(rank, op_type::StandardDeviation,data,1);

    test.run();

    delete data;
}

TEST(SummaryStatsReduce,ObjectOrientedDimensionVariance) {
    constexpr int rank = 2;
    double assertion[] = {0.50, 0.50};
    Data<double> *data = getDataSummaryDimension<double>(assertion,0);
    DoubleSummaryStatsReduceTest test(rank, op_type::Variance,data,1);

    test.run();

    delete data;
}

TEST(SummaryStatsReduce,ObjectOrientedFloatStandardDeviation) {
    constexpr int rank = 2;
    float assertion[] = {1.29099440574646};

    Data<float> *data = getDataSummary<float>(assertion,0);
    FloatSummaryStatsReduceTest test(rank, op_type::StandardDeviation, data, 1);

    test.run();

    delete data;
}

TEST(SummaryStatsReduce,ObjectOrientedFloatVariance) {
    constexpr int rank = 2;
    float assertion[] = {1.66667};

    Data<float> *data = getDataSummary<float>(assertion,0);
    FloatSummaryStatsReduceTest test(rank, op_type::Variance, data, 1);

    test.run();

    delete data;
}

TEST(SummaryStatsReduce,ObjectOrientedFloatDimensionStandardDeviation) {
    constexpr int rank = 2;
    constexpr float assertion[] = { 0.71, 0.71};

    Data<float> *data = getDataSummaryDimension<float>(assertion,0);
    FloatSummaryStatsReduceTest test(rank, op_type::StandardDeviation, data, 1);

    test.run();

    delete data;
}

TEST(SummaryStatsReduce,ObjectOrientedFloatDimensionVariance) {
    constexpr int rank = 2;
    constexpr float assertion[2] = {0.50, 0.50};

    Data<float> *data = getDataSummaryDimension<float>(assertion,0);
    FloatSummaryStatsReduceTest test(rank, op_type::Variance,data,1);

    test.run();

    delete data;
}

TEST(SummaryStatsReduce,ObjectOrientedFloatDimensionStandardDeviationMulti) {
    constexpr int rank = 3;
    constexpr float comparison[] = {15.0,15.0,15.0};

    auto data = getDataSummaryDimensionMulti<float>(comparison, 0);
    data->extraParams[0] = 1.0;
    FloatSummaryStatsReduceTest test(rank, op_type::Variance, data, 1);

    test.run();

    delete data;
}

#endif /* SUMMARYSTATSREDUCETEST_H_ */
