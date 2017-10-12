//
// Created by raver119 on 11.10.2017.
//

#include "testlayers.h"
#include <NDArray.h>
#include <ops/declarable/OpTuple.h>

using namespace nd4j;
using namespace nd4j::ops;

class OpTupleTests : public testing::Test {
    public:
};

TEST_F(OpTupleTests, DirectConstructorTest1) {
    NDArray<float> *alpha = new NDArray<float>('c', {1, 2});
    NDArray<float> *beta = new NDArray<float>('c', {1, 2});
    OpTuple tuple("dummy", {alpha, beta}, {12.0f}, {1,2, 3});

    ASSERT_EQ("dummy", tuple._opName);
    ASSERT_EQ(2, tuple._inputs.size());
    ASSERT_EQ(0, tuple._outputs.size());
    ASSERT_EQ(1, tuple._tArgs.size());
    ASSERT_EQ(3, tuple._iArgs.size());
}

TEST_F(OpTupleTests, BuilderTest1) {
    NDArray<float> *alpha = new NDArray<float>('c', {1, 2});
    NDArray<float> *beta = new NDArray<float>('c', {1, 2});
    OpTuple tuple("dummy");
    tuple.addInput(alpha)
            ->addInput(beta)
            ->setTArgs({12.0f})
            ->setIArgs({1, 2, 3});


    ASSERT_EQ("dummy", tuple._opName);
    ASSERT_EQ(2, tuple._inputs.size());
    ASSERT_EQ(0, tuple._outputs.size());
    ASSERT_EQ(1, tuple._tArgs.size());
    ASSERT_EQ(3, tuple._iArgs.size());
}