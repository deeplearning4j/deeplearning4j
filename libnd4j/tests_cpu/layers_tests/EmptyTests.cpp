//
// Created by raver on 6/18/2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
// #include <array/NDArrayList.h>

using namespace nd4j;


class EmptyTests : public testing::Test {
public:

    EmptyTests() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(EmptyTests, Test_Create_Empty) {
    auto empty = NDArray<float>::createEmpty();
    ASSERT_TRUE(empty->isEmpty());

    ASSERT_TRUE(empty->shapeInfo() == nullptr);
    ASSERT_TRUE(empty->buffer() == nullptr);

    delete empty;
}

TEST_F(EmptyTests, Test_Concat_1) {
    auto empty = NDArray<float>::createEmpty();
    auto scalar = new NDArray<float>(1.0f);

    nd4j::ops::concat<float> op;
    auto result = op.execute({empty, scalar}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(scalar->isSameShape(z));
    ASSERT_EQ(*scalar, *z);

    delete empty;
    delete scalar;
    delete result;
}

