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