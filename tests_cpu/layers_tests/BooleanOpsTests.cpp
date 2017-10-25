//
// Created by raver119 on 13.10.2017.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class BooleanOpsTests : public testing::Test {
public:

};


TEST_F(BooleanOpsTests, LtTest_1) {
    auto x = NDArrayFactory<float>::scalar(1.0f);
    auto y = NDArrayFactory<float>::scalar(2.0f);

    nd4j::ops::lt_scalar<float> op;


    ASSERT_TRUE(op.evaluate({x, y}));

    delete x;
    delete y;
}

TEST_F(BooleanOpsTests, LtTest_2) {
    auto x = NDArrayFactory<float>::scalar(2.0f);
    auto y = NDArrayFactory<float>::scalar(1.0f);

    nd4j::ops::lt_scalar<float> op;


    ASSERT_FALSE(op.evaluate({x, y}));

    delete x;
    delete y;
}
