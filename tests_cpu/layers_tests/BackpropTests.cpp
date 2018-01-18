//
// Created by raver119 on 13.01.2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class BackpropTests : public testing::Test {
public:

};

TEST_F(BackpropTests, Test_Add_1) {
    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> y('c', {3, 4});
    NDArray<float> e('c', {2, 3, 4});

    nd4j::ops::add_bp<float> op;
    auto result = op.execute({&x, &y, &e}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());

    auto eps = result->at(0);
    auto grad = result->at(1);

    ASSERT_TRUE(x.isSameShape(eps));
    ASSERT_TRUE(y.isSameShape(grad));

    delete result;
}