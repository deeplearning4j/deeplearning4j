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




template <typename T>
class TransformTest : public BaseTest<T> {
protected:
    functions::transform::Transform<T> *op = nullptr;
    functions::transform::TransformOpFactory<T> *opFactory = nullptr;


public:

    int useIndexes = 0;

    TransformTest() {}




    virtual ~TransformTest() {
        freeOpAndOpFactory();
    }

    TransformTest(int rank,int opNum,Data<T> *data,int extraParamsLength)
            : BaseTest<T>(rank,opNum,data,extraParamsLength) {
        createOperationAndOpFactory();
    }
    virtual void freeOpAndOpFactory() override {
        if(op != nullptr)
            delete op;
        if(opFactory != nullptr)
            delete opFactory;
    }

    virtual void createOperationAndOpFactory() override {
        opFactory = new functions::transform::TransformOpFactory<T>();
        op = opFactory->getOp(this->opNum);
    }


    virtual void run () override {
        this->initializeData();
        this->execCpuKernel();
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

    virtual void execCpuKernel() override {
        int *xShapeBuffer = shapeBuffer(this->baseData->rank,this->baseData->xShape);
        int *resultShapeBuffer = shapeBuffer(this->baseData->resultRank,this->baseData->resultShape);
        Nd4jIndex *indexes  = shape::computeIndices(xShapeBuffer);

        if(useIndexes) {
            op->exec(this->data->data->data,
                     xShapeBuffer,
                     this->data->data->data,
                     resultShapeBuffer,
                     this->extraParams,
                     indexes);
        } else {
            op->exec(this->data->data->data,
                     xShapeBuffer,
                     this->data->data->data,
                     resultShapeBuffer,
                     this->extraParams);
        }

        delete []xShapeBuffer;
        delete []resultShapeBuffer;
        delete []indexes;
    }

};




class DoubleTransformTest : public  TransformTest<double> {
public:
    virtual ~DoubleTransformTest() {}
    DoubleTransformTest() {}
    DoubleTransformTest(int rank,int opNum,Data<double> *data,int extraParamsLength)
            :  TransformTest<double>(rank,opNum,data,extraParamsLength) {
    }
    virtual void executeCudaKernel() override {
#ifdef __CUDACC__
        transformDouble<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum
				,this->length,
				this->data->data->gData,
				1,
				this->extraParamsBuff->gData,
				this->data->data->gData);


#endif
    }
};



class FloatTransformTest : public TransformTest<float> {
public:
    FloatTransformTest() {}
    virtual ~FloatTransformTest() {}
    FloatTransformTest(int rank,int opNum,Data<float> *data,int extraParamsLength)
            :  TransformTest<float>(rank,opNum,data,extraParamsLength){
    }
    virtual void executeCudaKernel() override {
#ifdef __CUDACC__
        transformFloat<<<this->blockSize,this->gridSize,this->sMemSize>>>(
				this->opNum
				,this->length,
				this->data->data->gData,
				1,this->extraParamsBuff->gData,
				this->data->data->gData
				);
#endif
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

class FloatTwoByTwoTransformTest : public FloatTransformTest  {
public:
    FloatTwoByTwoTransformTest() {}
    virtual ~FloatTwoByTwoTransformTest() {}
    FloatTwoByTwoTransformTest(int opNum,Data<float> *data,int extraParamsLength):
            FloatTransformTest(2,opNum,data,extraParamsLength) {
    }

};

template <typename T>
static Data<T> * getDataTransform(T *assertion,int rank) {
    Data<T> *data = new Data<T>();
    data->resultRank = 2;
    data->xShape = new int[data->resultRank];
    data->resultShape = new int[data->resultRank];

    for(int i = 0; i < data->resultRank; i++) {
        data->xShape[i] = 2;
        data->resultShape[i] = 2;
    }
    data->assertion = new T[4];
    for(int i = 0; i < 4; i++) {
        data->assertion[i] = assertion[i];
    }

    data->rank = rank;
    data->extraParams = new T[2];
    data->result = new T[4];

    return data;

}


TEST(Transform,ObjectOrientedSigmoid) {
    int opNum = 10;
    int rank = 2;
    double comparision[4] = {0.7310585786300049,0.8807970779778823
            ,0.9525741268224334,0.9820137900379085};
    Data<double> *data = getDataTransform<double>(comparision,rank);

    DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}


TEST(Transform,ObjectOrientedLog) {
    int opNum = 5;
    int rank = 2;
    double comparison[4] = { 0., 0.69314718, 1.09861229, 1.38629436 };

    Data<double> *data = getDataTransform<double>(comparison,rank);
    DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}

TEST(Transform,ObjectOrientedTanh) {
    int opNum = 15;
    int rank = 2;
    double comparison[4] = {  0.76159416,  0.96402758,  0.99505475,  0.9993293 };

    Data<double> *data = getDataTransform<double>(comparison,rank);
    DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}


TEST(Transform,ObjectOrientedSin) {
    int opNum = 12;
    int rank = 2;
    double comparison[4] = {  0.84147098,  0.90929743,  0.14112001, -0.7568025 };

    Data<double> *data = getDataTransform<double>(comparison,rank);
    DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}

TEST(Transform,ObjectOrientedCoSine) {
    int opNum = 2;
    int rank = 2;
    double comparison[4] = { 0.54030231, -0.41614684, -0.9899925 , -0.65364362 };
    Data<double> *data = getDataTransform<double>(comparison,rank);
    DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}


TEST(Transform,ObjectOrientedNeg) {
    int opNum = 6;
    int rank = 2;
    double comparison[4] = { -1,-2,-3,-4 };

    Data<double> *data = getDataTransform<double>(comparison,rank);
    DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}

TEST(Transform,ObjectOrientedExp) {
    int opNum = 3;
    int rank = 2;
    double comparison[4] = {  2.71828183,   7.3890561 ,  20.08553692,  54.59815003 };
    Data<double> *data = getDataTransform<double>(comparison,rank);
    DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}

TEST(Transform,ObjectOrientedAbs) {
    int opNum = 0;
    int rank = 2;
    double comparison[4] = {  1,2,3,4};
    Data<double> *data = getDataTransform<double>(comparison,rank);
    DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}






TEST(Transform,ObjectOrientedSqrt) {
    int opNum = 14;
    int rank = 2;
    double comparison[4] = {  1.        ,  1.41421356,  1.73205081,  2.};
    Data<double> *data = getDataTransform<double>(comparison,rank);
    DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}


TEST(Transform,ObjectOrientedATan) {
    int opNum = 18;
    int rank = 2;
    double comparison[4] = {  0.78539816,  1.10714872,  1.24904577,  1.32581766};
    Data<double> *data = getDataTransform<double>(comparison,rank);
    DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}


TEST(Transform,ObjectOrientedPow) {
    int opNum = 7;
    int rank = 2;
    double comparison[4] = {  1,4,9,16};
    Data<double> *data = getDataTransform<double>(comparison,rank);
    data->extraParams[0] = 2.0;
    DoubleTwoByTwoTransformTest *sigmoidTest = new DoubleTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}


TEST(Transform,ObjectOrientedFloatSigmoid) {
    int opNum = 10;
    int rank = 2;
    float comparision[4] = {0.7310585786300049,0.8807970779778823
            ,0.9525741268224334,0.9820137900379085};
    Data<float> *data = getDataTransform<float>(comparision,rank);

    FloatTwoByTwoTransformTest *sigmoidTest = new FloatTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}


TEST(Transform,ObjectOrientedFloatLog) {
    int opNum = 5;
    int rank = 2;
    float comparison[4] = { 0., 0.69314718, 1.09861229, 1.38629436 };

    Data<float> *data = getDataTransform<float>(comparison,rank);
    FloatTwoByTwoTransformTest *sigmoidTest = new FloatTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}

TEST(Transform,ObjectOrientedFloatTanh) {
    int opNum = 15;
    int rank = 2;
    float comparison[4] = {  0.76159416,  0.96402758,  0.99505475,  0.9993293 };

    Data<float> *data = getDataTransform<float>(comparison,rank);
    FloatTwoByTwoTransformTest *sigmoidTest = new FloatTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}


TEST(Transform,ObjectOrientedFloatSin) {
    int opNum = 12;
    int rank = 2;
    float comparison[4] = {  0.84147098,  0.90929743,  0.14112001, -0.7568025 };

    Data<float> *data = getDataTransform<float>(comparison,rank);
    FloatTwoByTwoTransformTest *sigmoidTest = new FloatTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}

TEST(Transform,ObjectOrientedFloatCoSine) {
    int opNum = 2;
    int rank = 2;
    float comparison[4] = { 0.54030231, -0.41614684, -0.9899925 , -0.65364362 };
    Data<float> *data = getDataTransform<float>(comparison,rank);
    FloatTwoByTwoTransformTest *sigmoidTest = new FloatTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}


TEST(Transform,ObjectOrientedFloatNeg) {
    int opNum = 6;
    int rank = 2;
    float comparison[4] = { -1,-2,-3,-4 };

    Data<float> *data = getDataTransform<float>(comparison,rank);
    FloatTwoByTwoTransformTest *sigmoidTest = new FloatTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}

TEST(Transform,ObjectOrientedFloatExp) {
    int opNum = 3;
    int rank = 2;
    float comparison[4] = {  2.71828183,   7.3890561 ,  20.08553692,  54.59815003 };
    Data<float> *data = getDataTransform<float>(comparison,rank);
    FloatTwoByTwoTransformTest *sigmoidTest = new FloatTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}

TEST(Transform,ObjectOrientedFloatAbs) {
    int opNum = 0;
    int rank = 2;
    float comparison[4] = {  1,2,3,4};
    Data<float> *data = getDataTransform<float>(comparison,rank);
    FloatTwoByTwoTransformTest *sigmoidTest = new FloatTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}






TEST(Transform,ObjectOrientedFloatSqrt) {
    int opNum = 14;
    int rank = 2;
    float comparison[4] = {  1.,1.41421356,  1.73205081,  2.};
    Data<float> *data = getDataTransform<float>(comparison,rank);
    FloatTwoByTwoTransformTest *sigmoidTest = new FloatTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}


TEST(Transform,ObjectOrientedFloatATan) {
    int opNum = 18;
    int rank = 2;
    float comparison[4] = {  0.78539816,  1.10714872,  1.24904577,  1.32581766};
    Data<float> *data = getDataTransform<float>(comparison,rank);
    FloatTwoByTwoTransformTest *sigmoidTest = new FloatTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}


TEST(Transform,ObjectOrientedFloatPow) {
    int opNum = 7;
    int rank = 2;
    float comparison[4] = { 1,4,9,16};
    Data<float> *data = getDataTransform<float>(comparison,rank);
    data->extraParams[0] = 2.0;
    FloatTwoByTwoTransformTest *sigmoidTest = new FloatTwoByTwoTransformTest(opNum,data,rank);
    sigmoidTest->run();
    delete sigmoidTest;
    delete data;
}









#endif //NATIVEOPERATIONS_TRANSFORMTESTS_H
