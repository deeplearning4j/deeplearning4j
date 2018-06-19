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

    ASSERT_EQ(0, empty->lengthOf());
    ASSERT_TRUE(empty->buffer() == nullptr);

    delete empty;
}

TEST_F(EmptyTests, Test_Concat_1) {
    auto empty = NDArray<float>::createEmpty();
    auto vector = new NDArray<float>('c', {1}, {1.0f});

    ASSERT_TRUE(empty->isEmpty());

    nd4j::ops::concat<float> op;
    auto result = op.execute({empty, vector}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

//    z->printShapeInfo("z shape");
//    z->printIndexedBuffer("z buffr");

    ASSERT_EQ(*vector, *z);

    delete empty;
    delete vector;
    delete result;
}


TEST_F(EmptyTests, Test_Concat_2) {
    auto empty = NDArray<float>::createEmpty();
    auto scalar1 = new NDArray<float>(1.0f);
    auto scalar2 = new NDArray<float>(2.0f);
    NDArray<float> exp('c', {2}, {1.f, 2.f});

    ASSERT_TRUE(empty->isEmpty());

    nd4j::ops::concat<float> op;
    auto result = op.execute({empty, scalar1, scalar2}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

//    z->printShapeInfo("z shape");
//    z->printIndexedBuffer("z buffr");

    ASSERT_EQ(exp, *z);

    delete empty;
    delete scalar1;
    delete scalar2;
    delete result;
}

TEST_F(EmptyTests, Test_Reshape_1) {
    NDArray<float> vector('c', {1}, {119.0f});
    NDArray<float> exp(119.0f);
    auto empty = NDArray<float>::createEmpty();

    nd4j::ops::reshape<float> op;
    auto result = op.execute({&vector, empty}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(exp, *result->at(0));

    delete empty;
    delete result;
}

TEST_F(EmptyTests, Test_Reshape_2) {
    NDArray<float> vector('c', {1}, {119.0f});
    NDArray<float> exp(119.0f);
    auto empty = NDArray<float>::createEmpty();

    nd4j::ops::reshape<float> op;
    auto result = op.execute({&vector, empty}, {}, {}, true);

    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(exp, *result->at(0));
    ASSERT_EQ(exp, vector);

    delete empty;
    delete result;
}