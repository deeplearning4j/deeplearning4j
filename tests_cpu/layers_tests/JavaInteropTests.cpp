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


TEST_F(JavaInteropTests, TestSconv2d_1) {
    NDArray<float> input('c', {3, 8, 8, 8});
    NDArray<float> weightsD('c', {1, 3, 1, 1});
    NDArray<float> weightsP('c', {2, 3, 1, 1});
    NDArray<float> bias('c', {1, 2});
    NDArray<float> output('c', {3, 2, 8, 8});
    output.assign(0.0);

    NDArrayFactory<float>::linspace(1, input);
    NDArrayFactory<float>::linspace(1, weightsD);
    NDArrayFactory<float>::linspace(1, weightsP);
    NDArrayFactory<float>::linspace(1, bias);

    NDArray<float> expOutput('c', {3, 2, 8, 8});

    nd4j::ops::sconv2d<float> op;


    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) input.getBuffer(), (Nd4jPointer) weightsD.getBuffer(), (Nd4jPointer) weightsP.getBuffer(), (Nd4jPointer) bias.getBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) input.getShapeInfo(), (Nd4jPointer) weightsD.getShapeInfo(), (Nd4jPointer) weightsP.getShapeInfo(), (Nd4jPointer) bias.getShapeInfo()};


    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) output.getBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) output.getShapeInfo()};

    NativeOps nativeOps;

    int exp[] = {1, 1, 1, 1, 0, 0, 1, 1, 0};

    nativeOps.execCustomOpFloat(nullptr, op.getOpHash(), ptrsInBuffer, ptrsInShapes, 4, ptrsOutBuffers, ptrsOutShapes, 1,
                                nullptr, 0, exp, 9, false);

    output.printBuffer("output");

    ASSERT_NEAR(17551, output.getScalar(0), 1e-5);
}
