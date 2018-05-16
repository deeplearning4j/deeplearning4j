//
// @author raver119@gmail.com
//

#ifndef LIBND4J_TADTESTS_H
#define LIBND4J_TADTESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <NDArrayFactory.h>

using namespace nd4j;

class TadTests : public testing::Test {
public:
    int numLoops = 100000000;

    int extLoops = 1000;
    int intLoops = 1000;
};

TEST_F(TadTests, Test4DTad1) {
    std::unique_ptr<NDArray<float>> arraySource(nd4j::NDArrayFactory<float>::linspace(1.0f, 10000.0f, 10000));

    std::unique_ptr<NDArray<float>> arrayExp(new NDArray<float>('c', {2, 1, 4, 4}));
    std::unique_ptr<NDArray<float>> arrayBad(new NDArray<float>('c', {2, 1, 4, 4}));

    arrayExp->setBuffer(arraySource->getBuffer());
    //arrayExp->printShapeInfo("Exp shapeBuffer: ");


    std::vector<int> badShape({4, 2, 1, 4, 4, 80, 16, 4, 1, 0, -1, 99});

    arrayBad->setBuffer(arraySource->getBuffer());
    arrayBad->setShapeInfo(badShape.data());
    //arrayBad->printShapeInfo("Bad shapeBuffer: ");


    int dim = 1;
    shape::TAD tad(arrayBad->getShapeInfo(), &dim, 1);
    tad.createTadOnlyShapeInfo();
    tad.createOffsets();

    std::vector<int> exp({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95});
    for (int e = 0; e < 32; e++) {
        ASSERT_EQ((int) tad.tadOffsets[e],  exp.data()[e]);
    }
}

TEST_F(TadTests, TestNumTads1) {
    NDArray<float> x('c', {2, 3});
    NDArray<float> y('c', {2, 2});

    std::vector<int> dim({0});

    Nd4jIndex tadLengthX = shape::tadLength(x.getShapeInfo(), dim.data(), dim.size());
    Nd4jIndex numTadsX = x.lengthOf() / tadLengthX;

    Nd4jIndex tadLengthY = shape::tadLength(y.getShapeInfo(), dim.data(), dim.size());
    Nd4jIndex numTadsY = y.lengthOf() / tadLengthY;

    ASSERT_EQ(2, tadLengthX);
    ASSERT_EQ(3, numTadsX);

    ASSERT_EQ(2, tadLengthY);
    ASSERT_EQ(2, numTadsY);
}

TEST_F(TadTests, TestShapeTad_1) {

    float buff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};    
    int shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};   

    NDArray<float> input(buff, shapeInfo);
    
    std::vector<int> dimensions = {0,1,2};
    Nd4jIndex tadLength = shape::tadLength(input.getShapeInfo(), dimensions.data(), dimensions.size());
    Nd4jIndex numTads = input.lengthOf() / tadLength;
    
    shape::TAD tad(input.getShapeInfo(), dimensions.data(), dimensions.size());
    tad.createTadOnlyShapeInfo();
    tad.createOffsets();

    int tadShapeInfo[shape::shapeInfoLength(tad.tadOnlyShapeInfo[0])];
    std::memcpy(tadShapeInfo, tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));

    float* tadBuff = input.getBuffer() + tad.tadOffsets[0];
    NDArray<float> tadArr(tadBuff, tadShapeInfo);
   
    ASSERT_TRUE(numTads==1);
    ASSERT_TRUE(input.isSameShapeStrict(&tadArr));
    ASSERT_TRUE(input.equalsTo(&tadArr));
    
}

TEST_F(TadTests, TadNoAxis_1) {
    NDArray<float> array('c', {2, 3});

    shape::TAD tad(array.shapeInfo(), nullptr, 0);
    tad.createTadOnlyShapeInfo();
    tad.createOffsets();

    ASSERT_TRUE(tad.wholeThing);

    ASSERT_TRUE(shape::equalsStrict(tad.tadOnlyShapeInfo, array.shapeInfo()));
}

TEST_F(TadTests, TadEdgeCase_1) {
    NDArray<float> array('c', {5, 4, 1});
    NDArray<float> exp('c', {5, 4});
    NDArrayFactory<float>::linspace(1, array);

    auto tad = array.tensorAlongDimension(0, {0, 1});

    ASSERT_TRUE(exp.isSameShape(tad));

    delete tad;
}

TEST_F(TadTests, TestEdgeCase_2) {
    NDArray<float> array('f', {2, 3, 1}, {1, 4, 2, 5, 3, 6});

    auto tad1 = array.tensorAlongDimension(1, {2});

    for (int e = 0 ; e < array.lengthOf(); e++) {
        auto tad = array.tensorAlongDimension(e, {2});

        ASSERT_NEAR(tad->getScalar(0), array.getIndexedScalar(e), 1e-5);

        delete tad;
    }

    delete tad1;
}

TEST_F(TadTests, TadEdgeCase_2) {
    NDArray<float> array('c', {2, 3, 4});

    auto tad = array.tensorAlongDimension(0, {1});

    tad->printShapeInfo("TAD shape");
    ASSERT_EQ(3, tad->lengthOf());

    delete tad;
}

/*
 // FIXME: we want this test passing eventually
TEST_F(TadTests, Tad_1D_1) {
    NDArray<float> x('c', {5, 4});

    std::vector<int> dims({1});
    shape::TAD tad(x.shapeInfo(), dims.data(), dims.size());
    tad.createTadOnlyShapeInfo();
    tad.createOffsets();

    auto shapeRank = shape::rank(tad.tadOnlyShapeInfo);
    auto shapeLength = shape::length(tad.tadOnlyShapeInfo);

    ASSERT_EQ(1, shapeRank);   
}
*/

///////////////////////////////////////////////////////////////////
/*
TEST_F(TadTests, TestShapeTad_2) {
        
    NDArray<float> input('c', {2,1,4,1});
    
    NDArrayFactory<float>::linspace(1, input);

    NDArray<float> expected('c', {4,1});    

    std::vector<int> dimensions = {2,3};
    Nd4jIndex tadLength = shape::tadLength(input.getShapeInfo(), dimensions.data(), dimensions.size());
    Nd4jIndex numTads = input.lengthOf() / tadLength;
    shape::TAD tad(input.getShapeInfo(), dimensions.data(), dimensions.size());
    tad.createTadOnlyShapeInfo();    
    
    for(int i=0; i<8; i++)
        std::cout<<tad.tadOnlyShapeInfo[i]<<" ";
    std::cout<<std::endl;

    ASSERT_TRUE(shape::shapeEquals(expected.getShapeInfo(), tad.tadOnlyShapeInfo));

}
*/

#endif //LIBND4J_TADTESTS_H
