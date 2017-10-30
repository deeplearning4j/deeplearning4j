//
// Created by raver119 on 30.10.2017.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class BlockTests : public testing::Test {
public:

};


TEST_F(BlockTests, Basic_Test_1) {
    VariableSpace<float> variableSpace;

    auto _20 = new NDArray<float>('c', {2, 2});
    auto _21 = new NDArray<float>('c', {2, 2});

    _20->assign(1.0f);
    _21->assign(2.0f);

    variableSpace.putVariable(2, 0, _20);
    variableSpace.putVariable(2, 1, _21);

    Block<float> block(1, &variableSpace);

    block.pickInput(2, 0);
    block.pickInput(2, 1);

    ASSERT_EQ(2, block.inputs()->size());
    ASSERT_EQ(2, block.width());

    ASSERT_TRUE(variableSpace.hasVariable(2, 0));
    ASSERT_TRUE(variableSpace.hasVariable(2, 1));

    ASSERT_NEAR(1.0f, block.variable(0)->getNDArray()->meanNumber(), 1e-5);
    ASSERT_NEAR(2.0f, block.variable(1)->getNDArray()->meanNumber(), 1e-5);
}


TEST_F(BlockTests, Basic_Test_2) {
    VariableSpace<float> variableSpace;

    auto _20 = new NDArray<float>('c', {2, 2});
    auto _21 = new NDArray<float>('c', {2, 2});

    _20->assign(1.0f);
    _21->assign(2.0f);

    variableSpace.putVariable(-1, _20);
    variableSpace.putVariable(-2, _21);

    Block<float> block(1, &variableSpace);

    block.pickInput(-1);
    block.pickInput(-2);

    ASSERT_EQ(2, block.inputs()->size());
    ASSERT_EQ(2, block.width());

    ASSERT_TRUE(variableSpace.hasVariable(-1));
    ASSERT_TRUE(variableSpace.hasVariable(-2));

    ASSERT_NEAR(1.0f, block.variable(0)->getNDArray()->meanNumber(), 1e-5);
    ASSERT_NEAR(2.0f, block.variable(1)->getNDArray()->meanNumber(), 1e-5);
}