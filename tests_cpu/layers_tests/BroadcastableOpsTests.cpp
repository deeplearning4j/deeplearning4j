//
// Created by raver119 on 23.11.17.
//


#include "testlayers.h"
#include <Graph.h>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::graph;

class BroadcastableOpsTests : public testing::Test {
public:

};

TEST_F(BroadcastableOpsTests, Test_Add_1) {
    NDArray<float> x('c', {5, 5});
    NDArray<float> y('c', {1, 5});
    NDArray<float> exp('c', {5, 5});
    NDArrayFactory<float>::linspace(1, x);
    NDArrayFactory<float>::linspace(1, y);
    NDArrayFactory<float>::linspace(1, exp);

    exp.template applyBroadcast<simdOps::Add<float>>({1}, &y);


    nd4j::ops::add<float> op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_Multiply_1) {
    NDArray<float> x('c', {5, 5});
    NDArray<float> y('c', {1, 5});
    NDArray<float> exp('c', {5, 5});
    NDArrayFactory<float>::linspace(1, x);
    NDArrayFactory<float>::linspace(1, y);
    NDArrayFactory<float>::linspace(1, exp);

    exp.template applyBroadcast<simdOps::Multiply<float>>({1}, &y);


    nd4j::ops::multiply<float> op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_SquaredSubtract_1) {
    NDArray<float> x('c', {5, 5});
    NDArray<float> y('c', {1, 5});
    NDArray<float> exp('c', {5, 5});
    NDArrayFactory<float>::linspace(1, x);
    NDArrayFactory<float>::linspace(1, y);
    NDArrayFactory<float>::linspace(1, exp);

    exp.template applyBroadcast<simdOps::SquaredSubtract<float>>({1}, &y);


    nd4j::ops::squaredsubtract<float> op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_ScalarBroadcast_1) {
    NDArray<float> x('c', {1, 1}, {1});
    NDArray<float> y('c', {1, 3}, {0, 1, 2});
    NDArray<float> exp('c', {1,3}, {1, 0, -1});

    nd4j::ops::subtract<float> op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_ScalarBroadcast_2) {
    NDArray<float> x('c', {1, 1}, {1});
    NDArray<float> y('c', {1, 3}, {0, 1, 2});
    NDArray<float> exp('c', {1,3}, {1, 2, 3});

    nd4j::ops::add<float> op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_Maximum_1) {
    NDArray<float> x('c', {2, 3}, {1, 2, 1, 2, 3, 2});
    NDArray<float> row('c', {1, 3}, {2, 2, 2});
    NDArray<float> exp('c', {2, 3}, {2, 2, 2, 2, 3, 2});

    nd4j::ops::maximum<float> op;
    auto result = op.execute({&x, &row}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_Minimum_1) {
    NDArray<float> x('c', {2, 3}, {1, 2, 1, 2, 3, 2});
    NDArray<float> col('c', {2, 1}, {2, 1});
    NDArray<float> exp('c', {2, 3}, {1, 2, 1, 1, 1, 1});

    nd4j::ops::minimum<float> op;
    auto result = op.execute({&x, &col}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}