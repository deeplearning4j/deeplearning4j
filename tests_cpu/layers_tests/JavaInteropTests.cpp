//
// @author raver119@gmail.com
//

#include <NativeOps.h>
#include <NDArray.h>
#include <ops/declarable/CustomOperations.h>
#include "testlayers.h"

using namespace nd4j;

class JavaInteropTests : public testing::Test {
public:

};


TEST_F(JavaInteropTests, TestShapeExposure1) {
    NDArray<float> input('c', {1, 2, 5, 4});
    NDArray<float> weights('c', {3, 2, 2, 2});
    NDArray<float> exp('c', {1, 3, 5, 4});


    NativeOps nativeOps;

    nd4j::ops::conv2d<float> op;

    std::vector<float> tArgs({});
    std::vector<int> iArgs({2, 2, 1, 1, 0, 0, 1, 1, 1});


    Nd4jPointer ptrs[] = {(Nd4jPointer) input.getShapeInfo(), (Nd4jPointer) weights.getShapeInfo()};

    auto shapeList = nativeOps.calculateOutputShapesFloat(nullptr, op.getOpHash(), ptrs, 2, tArgs.data(), tArgs.size(), iArgs.data(), iArgs.size());

    //ASSERT_EQ(1, shapeList->size());

    ASSERT_EQ(exp.rankOf(), shape::rank((int *)shapeList[0]));
    ASSERT_EQ(exp.sizeAt(0), shape::shapeOf((int *)shapeList[0])[0]);
    ASSERT_EQ(exp.sizeAt(1), shape::shapeOf((int *)shapeList[0])[1]);
    ASSERT_EQ(exp.sizeAt(2), shape::shapeOf((int *)shapeList[0])[2]);
    ASSERT_EQ(exp.sizeAt(3), shape::shapeOf((int *)shapeList[0])[3]);
}
