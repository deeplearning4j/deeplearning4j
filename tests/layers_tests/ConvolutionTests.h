//
// @author raver119@gmail.com
//

#ifndef LIBND4J_CONVOLUTIONTESTS_H
#define LIBND4J_CONVOLUTIONTESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <Block.h>
#include <Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/declarable_ops.h>
#include <ops/declarable/generic/convo/convo_ops.h>

using namespace nd4j::graph;

class ConvolutionTests : public testing::Test {
public:

};

TEST_F(ConvolutionTests, TestConv2D_1) {
    double _expB[]{664.0, 700.0, 736.0, 344.0, 808.0, 844.0, 880.0, 408.0, 952.0, 988.0, 1024.0, 472.0, 1096.0, 1132.0, 1168.0, 536.0, 466.0, 480.0, 494.0, 220.0, 1528.0, 1628.0, 1728.0, 856.0, 1928.0, 2028.0, 2128.0, 1048.0, 2328.0, 2428.0, 2528.0, 1240.0, 2728.0, 2828.0, 2928.0, 1432.0, 1346.0, 1392.0, 1438.0, 700.0, 2392.0, 2556.0, 2720.0, 1368.0, 3048.0, 3212.0, 3376.0, 1688.0, 3704.0, 3868.0, 4032.0, 2008.0, 4360.0, 4524.0, 4688.0, 2328.0, 2226.0, 2304.0, 2382.0, 1180.0};
    int _expS[]{4, 1, 3, 5, 4, 60, 20, 4, 1, 0, 1, 99};
    auto input = new NDArray<double>('c', {1, 2, 5, 4});
    auto weights = new NDArray<double> ('c', {3, 2, 2, 2});

    for (int e = 0; e < input->lengthOf(); e++)
        input->putScalar(e, e + 1);

    for (int e = 0; e < weights->lengthOf(); e++)
        weights->putScalar(e, e + 1);

    auto exp = new NDArray<double>(_expB, _expS);
    exp->triggerAllocationFlag(false, false);

    auto variableSpace = new VariableSpace<double>();
    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, weights);

    auto block = new Block<double>(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1, -2});

    // 5,5 kernel
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(2);

    // 1,1 stride
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(1);

    // 0,0 padding
    block->getIArguments()->push_back(0);
    block->getIArguments()->push_back(0);

    // 1,1 dilation
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(1);

    // same mode
    block->getIArguments()->push_back(1);

    nd4j::ops::conv2d<double> op;

    Nd4jStatus status = op.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto res = variableSpace->getVariable(1)->getNDArray();


    // checking output shape
    ASSERT_EQ(1, res->sizeAt(0));
    ASSERT_EQ(3, res->sizeAt(1));
    ASSERT_EQ(5, res->sizeAt(2));
    ASSERT_EQ(4, res->sizeAt(3));

    // basically the same as above
    ASSERT_TRUE(res->isSameShape(exp));

    // just for visual validation
    exp->printBuffer("Expected");
    res->printBuffer("Actual  ");
    res->printShapeInfo("Result shape");

    // final check
    ASSERT_TRUE(res->equalsTo(exp));

    delete block;
    delete variableSpace;
}


TEST_F(ConvolutionTests, TestAvgFF1) {
    NDArray<float> input('c', {4, 2, 1, 11, 11});

    input.assign(451.0);

    NDArray<float> output('c', {4, 2, 1, 10, 10});


    std::pair<int, int> pair0(1,0);
    std::pair<int, int> pair1(1,1);


    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);

    variableSpace->putVariable(pair0, &output);

    Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1});

    // kernel params
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(2);

    // stride
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(1);

    // padding
    block->getIArguments()->push_back(0);
    block->getIArguments()->push_back(0);
    block->getIArguments()->push_back(0);

    // ceiling
    block->getIArguments()->push_back(1);

    // padding count
    block->getIArguments()->push_back(1);



    nd4j::ops::avgpool3d<float> avgpool3d;

    Nd4jStatus result = avgpool3d.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, result);

    //output.printBuffer("Result");

    ASSERT_NEAR(451.0f, output.template reduceNumber<simdOps::Mean<float>>(), 1e-5);

}


TEST_F(ConvolutionTests, TestFullConv3D_1) {
    NDArray<float> input('c', {4, 3, 3, 56, 56});
    NDArray<float> weights('f', {2, 3, 3, 5, 5});
    NDArray<float> bias('c', {1, 2});

    input.assign(1.0);
    weights.assign(2.0);
    bias.putScalar(0, 1.0f);
    bias.putScalar(1, 1.0f);

    NDArray<float> output('c', {4, 2, 1, 11, 11});

    VariableSpace<float>* variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, &input);
    variableSpace->putVariable(-2, &weights);
    variableSpace->putVariable(-3, &bias);

    variableSpace->putVariable(1, &output);
    Block<float>* block = new Block<float>(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1, -2, -3});

    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(5);
    block->getIArguments()->push_back(5);
    block->getIArguments()->push_back(0);
    block->getIArguments()->push_back(0);
    block->getIArguments()->push_back(0);

    nd4j::ops::conv3d<float> conv3d;
}

#endif //LIBND4J_CONVOLUTIONTESTS_H
