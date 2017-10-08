//
// @author raver119@gmail.com
//

#ifndef LIBND4J_TADTESTS_H
#define LIBND4J_TADTESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <NDArrayFactory.h>

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
    arrayExp->printShapeInfo("Exp shapeBuffer: ");


    std::vector<int> badShape({4, 2, 1, 4, 4, 80, 16, 4, 1, 0, -1, 99});

    arrayBad->setBuffer(arraySource->getBuffer());
    arrayBad->setShapeInfo(badShape.data());
    arrayBad->printShapeInfo("Bad shapeBuffer: ");


    int dim = 1;
    shape::TAD tad(arrayBad->getShapeInfo(), &dim, 1);
    tad.createTadOnlyShapeInfo();
    tad.createOffsets();

    std::unique_ptr<int> exp(new int[32] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95});
    for (int e = 0; e < 32; e++) {
        ASSERT_EQ((int) tad.tadOffsets[e],  exp.get()[e]);
    }
}

TEST_F(TadTests, TestNumTads1) {
    NDArray<float> x(2, 3, 'c');
    NDArray<float> y(2, 2, 'c');

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

#endif //LIBND4J_TADTESTS_H
