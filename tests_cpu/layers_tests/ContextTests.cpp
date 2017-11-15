//
// Created by raver119 on 30.10.2017.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class ContextTests : public testing::Test {
public:

};


TEST_F(ContextTests, Basic_Test_1) {
    VariableSpace<float> variableSpace;

    auto _20 = new NDArray<float>('c', {2, 2});
    auto _21 = new NDArray<float>('c', {2, 2});

    _20->assign(1.0f);
    _21->assign(2.0f);

    variableSpace.putVariable(2, 0, _20);
    variableSpace.putVariable(2, 1, _21);

    Context<float> block(1, &variableSpace);

    block.pickInput(2, 0);
    block.pickInput(2, 1);

    ASSERT_EQ(2, block.inputs()->size());
    ASSERT_EQ(2, block.width());

    ASSERT_TRUE(variableSpace.hasVariable(2, 0));
    ASSERT_TRUE(variableSpace.hasVariable(2, 1));

    ASSERT_NEAR(1.0f, block.variable(0)->getNDArray()->meanNumber(), 1e-5);
    ASSERT_NEAR(2.0f, block.variable(1)->getNDArray()->meanNumber(), 1e-5);
}


TEST_F(ContextTests, Basic_Test_2) {
    VariableSpace<float> variableSpace;

    auto _20 = new NDArray<float>('c', {2, 2});
    auto _21 = new NDArray<float>('c', {2, 2});

    _20->assign(1.0f);
    _21->assign(2.0f);

    variableSpace.putVariable(-1, _20);
    variableSpace.putVariable(-2, _21);

    Context<float> block(1, &variableSpace);

    block.pickInput(-1);
    block.pickInput(-2);

    ASSERT_EQ(2, block.inputs()->size());
    ASSERT_EQ(2, block.width());

    ASSERT_TRUE(variableSpace.hasVariable(-1));
    ASSERT_TRUE(variableSpace.hasVariable(-2));

    ASSERT_NEAR(1.0f, block.variable(0)->getNDArray()->meanNumber(), 1e-5);
    ASSERT_NEAR(2.0f, block.variable(1)->getNDArray()->meanNumber(), 1e-5);
}


TEST_F(ContextTests, Basic_Test_3) {
    VariableSpace<float> variableSpace;

    Context<float> ctx(1, &variableSpace);

    auto _20 = new NDArray<float>('c', {2, 2});

    ctx.pushNDArrayToVariableSpace(1, 1, _20);

    ASSERT_TRUE(variableSpace.hasVariable(1, 1));
}


TEST_F(ContextTests, Basic_Test_4) {
    VariableSpace<float> variableSpace;

    Context<float> ctx(1, &variableSpace);

    auto _20 = new NDArray<float>('c', {2, 2});
    NDArrayFactory<float>::linspace(1, *_20);

    auto _21 = new NDArray<float>('c', {2, 2});
    NDArrayFactory<float>::linspace(10, *_21);

    ctx.pushNDArrayToVariableSpace(1, 1, _20);

    ASSERT_TRUE(variableSpace.hasVariable(1, 1));

    ctx.pushNDArrayToVariableSpace(1, 1, _21);

    auto vA = ctx.variable(1, 1);

    ASSERT_TRUE(vA->getNDArray()->equalsTo(_21));
}

TEST_F(ContextTests, Basic_Test_5) {
    VariableSpace<float> variableSpace;

    Context<float> ctx(1, &variableSpace);

    auto _20 = new NDArray<float>('c', {2, 2});
    NDArrayFactory<float>::linspace(1, *_20);

    auto exp = _20->dup();

    ctx.pushNDArrayToVariableSpace(1, 1, _20);

    ASSERT_TRUE(variableSpace.hasVariable(1, 1));

    ctx.pushNDArrayToVariableSpace(1, 1, _20);

    auto vA = ctx.variable(1, 1);

    ASSERT_TRUE(vA->getNDArray() == _20);

    ASSERT_TRUE(vA->getNDArray()->equalsTo(exp));

    delete exp;
}


TEST_F(ContextTests, Basic_Test_6) {
    VariableSpace<float> variableSpace;

    Context<float> ctx(1, &variableSpace);

    auto v0 = ctx.ensureVariable();
    auto v1 = ctx.ensureVariable(1);

    ASSERT_TRUE(variableSpace.hasVariable(1, 0));
    ASSERT_TRUE(variableSpace.hasVariable(1, 1));

    auto var0 = variableSpace.getVariable(1, 0);
    auto var1 = variableSpace.getVariable(1, 1);

    ASSERT_TRUE(v0 == var0);
    ASSERT_TRUE(v1 == var1);
}


TEST_F(ContextTests, Basic_Test_7) {
    VariableSpace<float> variableSpace;

    Context<float> ctx(1, &variableSpace);

    auto v0 = ctx.ensureVariable();
    auto v1 = ctx.ensureVariable(1);

    ASSERT_TRUE(variableSpace.hasVariable(1, 0));
    ASSERT_TRUE(variableSpace.hasVariable(1, 1));

    auto var0 = variableSpace.getVariable(1, 0);
    auto var1 = variableSpace.getVariable(1, 1);

    ASSERT_TRUE(v0 == var0);
    ASSERT_TRUE(v1 == var1);


    auto _10 = new NDArray<float>('c', {2, 2});
    NDArrayFactory<float>::linspace(1, *_10);

    auto _11 = new NDArray<float>('c', {2, 2});
    NDArrayFactory<float>::linspace(10, *_11);

    ctx.pushNDArrayToVariableSpace(1, 0, _10);
    ctx.pushNDArrayToVariableSpace(1, 1, _11);

    auto z0 = variableSpace.getVariable(1, 0);
    auto z1 = variableSpace.getVariable(1, 1);

    ASSERT_TRUE(v0 == z0);
    ASSERT_TRUE(v1 == z1);
}

TEST_F(ContextTests, Basic_Test_8) {
    VariableSpace<float> variableSpace;

    Context<float> ctx(1, &variableSpace);

    auto _10 = new NDArray<float>('c', {2, 2});
    NDArrayFactory<float>::linspace(1, *_10);

    auto _11 = new NDArray<float>('c', {2, 2});
    NDArrayFactory<float>::linspace(10, *_11);

    ctx.pushNDArrayToVariableSpace(1, 0, _10);
    ctx.pushNDArrayToVariableSpace(1, 1, _11);

    auto z0 = variableSpace.getVariable(1, 0);
    auto z1 = variableSpace.getVariable(1, 1);

    auto v0 = ctx.ensureVariable();
    auto v1 = ctx.ensureVariable(1);

    ASSERT_TRUE(v0 == z0);
    ASSERT_TRUE(v1 == z1);
}


TEST_F(ContextTests, Basic_Test_9) {
    VariableSpace<float> variableSpace;

    NDArray<float> in('c', {5, 5});

    Context<float> ctx(1, &variableSpace, true);
    ctx.pushNDArrayToVariableSpace(1, 1, &in, false);
}

TEST_F(ContextTests, Basic_Test_10) {
    VariableSpace<float> variableSpace;

    Context<float> ctx(119, &variableSpace);
}