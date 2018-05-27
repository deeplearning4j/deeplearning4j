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


TEST_F(BroadcastableOpsTests, Test_Shape_1) {
    nd4j::ops::minimum<float> op;

    Nd4jLong shapeX[] = {2, 2, 5, 5, 1, 0, 1, 99};
    Nd4jLong shapeY[] = {2, 2, 5, 5, 1, 0, 1, 99};
    ShapeList inputShape({shapeX, shapeY});
    VariableSpace<float> vs;
    Context<float> ctx(1, &vs, false);

    auto shapes = op.calculateOutputShape(&inputShape, ctx);

    auto shapeZ = shapes->at(0);
    ASSERT_TRUE(shape::shapeEquals(shapeX, shapeZ));

    shapes->destroy();
    delete shapes;
}

TEST_F(BroadcastableOpsTests, Test_Shape_2) {
    nd4j::ops::minimum<float> op;

    Nd4jLong shapeX[] = {2, 1, 1, 1, 1, 0, 1, 99};
    Nd4jLong shapeY[] = {2, 2, 5, 5, 1, 0, 1, 99};
    ShapeList inputShape({shapeX, shapeY});
    VariableSpace<float> vs;
    Context<float> ctx(1, &vs, false);

    auto shapes = op.calculateOutputShape(&inputShape, ctx);

    auto shapeZ = shapes->at(0);
    ASSERT_TRUE(shape::shapeEquals(shapeY, shapeZ));

    shapes->destroy();
    delete shapes;
}


TEST_F(BroadcastableOpsTests, Test_Shape_3) {
    nd4j::ops::minimum<float> op;

    Nd4jLong shapeX[] = {2, 5, 3, 1, 1, 0, 1, 99};
    Nd4jLong shapeY[] = {2, 1, 3, 3, 1, 0, 1, 99};
    ShapeList inputShape({shapeX, shapeY});
    VariableSpace<float> vs;
    Context<float> ctx(1, &vs, false);

    auto shapes = op.calculateOutputShape(&inputShape, ctx);

    auto shapeZ = shapes->at(0);
    ASSERT_TRUE(shape::shapeEquals(shapeX, shapeZ));

    shapes->destroy();
    delete shapes;
}


TEST_F(BroadcastableOpsTests, Test_Shape_4) {
    nd4j::ops::minimum<float> op;

    Nd4jLong shapeX[] = {2, 5, 3, 1, 1, 0, 1, 99};
    Nd4jLong shapeY[] = {2, 5, 1, 1, 1, 0, 1, 99};
    ShapeList inputShape({shapeX, shapeY});
    VariableSpace<float> vs;
    Context<float> ctx(1, &vs, false);

    auto shapes = op.calculateOutputShape(&inputShape, ctx);

    auto shapeZ = shapes->at(0);
    ASSERT_TRUE(shape::shapeEquals(shapeX, shapeZ));

    shapes->destroy();
    delete shapes;
}

// (2,1,3) + (4,3) = (2,4,3)

TEST_F(BroadcastableOpsTests, Test_Shape_5) {
    nd4j::ops::minimum<float> op;

    Nd4jLong shapeX[] = {3, 2, 1, 3, 3, 3, 1, 0, 1, 99};
    Nd4jLong shapeY[] = {2, 4, 3, 3, 1, 0, 1, 99};
    Nd4jLong shapeE[] = {3, 2, 4, 3, 12, 3, 1, 0, 1, 99};
    ShapeList inputShape({shapeX, shapeY});
    VariableSpace<float> vs;
    Context<float> ctx(1, &vs, false);

    auto shapes = op.calculateOutputShape(&inputShape, ctx);

    auto shapeZ = shapes->at(0);
    ASSERT_TRUE(shape::shapeEquals(shapeE, shapeZ));

    shapes->destroy();
    delete shapes;
}

TEST_F(BroadcastableOpsTests, Test_Scalar_Add_1) {
    NDArray<float> x('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> y(2.0f);
    NDArray<float> exp('c', {2, 2}, {3, 4, 5, 6});

    nd4j::ops::add<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_Inplace_Output_1) {
    NDArray<float> x('c', {2, 1, 3});
    NDArray<float> y('c', {4, 3});
    NDArray<float> o('c', {2, 4, 3});
    NDArray<float> e('c', {2, 4, 3});
    float *buffO1 = o.buffer();
    y.assign(1.0f);
    e.assign(1.0f);

    nd4j::ops::add<float> op;
    auto result = op.execute({&x, &y}, {&o}, {}, {});
    ASSERT_EQ(Status::OK(), result);

    float *buffO2 = o.buffer();

    ASSERT_TRUE(e.isSameShape(o));
    ASSERT_TRUE(e.equalsTo(o));

    ASSERT_TRUE(buffO1 == buffO2);
}

TEST_F(BroadcastableOpsTests, Test_Subtract_1) {
    NDArray<float> x(1.0f);
    NDArray<float> y('c', {2}, {0.0f, 1.0f});
    NDArray<float> e('c', {2}, {1.0f, 0.0f});

    auto z = x - y;

    ASSERT_TRUE(e.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Subtract_2) {
    NDArray<float> x(1.0f);
    NDArray<float> y('c', {2}, {0.0f, 1.0f});
    NDArray<float> e('c', {2}, {1.0f, 0.0f});

    nd4j::ops::subtract<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    auto z = result->at(0);

    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}