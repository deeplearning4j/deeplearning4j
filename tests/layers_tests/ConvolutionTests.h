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


TEST_F(ConvolutionTests, SeparableConv2D_FF_NoBias_1) {
    float _expB[] = {10025.f,   38775.f,   10350.f,   40350.f,   10675.f,   41925.f,   11000.f,   43500.f,     11325.f,   45075.f,   11650.f,   46650.f,   13275.f,   54525.f,   13600.f,   56100.f,     13925.f,   57675.f,   14250.f,   59250.f,   14575.f,   60825.f,   14900.f,   62400.f,     16525.f,   70275.f,   16850.f,   71850.f,   17175.f,   73425.f,   17500.f,   75000.f,     17825.f,   76575.f,   18150.f,   78150.f,   19775.f,   86025.f,   20100.f,   87600.f,     20425.f,   89175.f,   20750.f,   90750.f,   21075.f,   92325.f,   21400.f,   93900.f,     23025.f,  101775.f,   23350.f,  103350.f,   23675.f,  104925.f,   24000.f,  106500.f,     24325.f,  108075.f,   24650.f,  109650.f,   26275.f,  117525.f,   26600.f,  119100.f,     26925.f,  120675.f,   27250.f,  122250.f,   27575.f,  123825.f,   27900.f,  125400.f,     75025.f,  353775.f,   75350.f,  355350.f,   75675.f,  356925.f,   76000.f,  358500.f,     76325.f,  360075.f,   76650.f,  361650.f,   78275.f,  369525.f,   78600.f,  371100.f,     78925.f,  372675.f,   79250.f,  374250.f,   79575.f,  375825.f,   79900.f,  377400.f,     81525.f,  385275.f,   81850.f,  386850.f,   82175.f,  388425.f,   82500.f,  390000.f,     82825.f,  391575.f,   83150.f,  393150.f,   84775.f,  401025.f,   85100.f,  402600.f,     85425.f,  404175.f,   85750.f,  405750.f,   86075.f,  407325.f,   86400.f,  408900.f,     88025.f,  416775.f,   88350.f,  418350.f,   88675.f,  419925.f,   89000.f,  421500.f,     89325.f,  423075.f,   89650.f,  424650.f,   91275.f,  432525.f,   91600.f,  434100.f,     91925.f,  435675.f,   92250.f,  437250.f,   92575.f,  438825.f,   92900.f,  440400.f,     119400.f,  273150.f,  120350.f,  275350.f,  121300.f,  277550.f,  122250.f,  279750.f,     123200.f,  281950.f,  124150.f,  284150.f,  128900.f,  295150.f,  129850.f,  297350.f,     130800.f,  299550.f,  131750.f,  301750.f,  132700.f,  303950.f,  133650.f,  306150.f,     138400.f,  317150.f,  139350.f,  319350.f,  140300.f,  321550.f,  141250.f,  323750.f,     142200.f,  325950.f,  143150.f,  328150.f,  147900.f,  339150.f,  148850.f,  341350.f,     149800.f,  343550.f,  150750.f,  345750.f,  151700.f,  347950.f,  152650.f,  350150.f,     157400.f,  361150.f,  158350.f,  363350.f,  159300.f,  365550.f,  160250.f,  367750.f,     161200.f,  369950.f,  162150.f,  372150.f,  166900.f,  383150.f,  167850.f,  385350.f,     168800.f,  387550.f,  169750.f,  389750.f,  170700.f,  391950.f,  171650.f,  394150.f,     309400.f,  713150.f,  310350.f,  715350.f,  311300.f,  717550.f,  312250.f,  719750.f,     313200.f,  721950.f,  314150.f,  724150.f,  318900.f,  735150.f,  319850.f,  737350.f,     320800.f,  739550.f,  321750.f,  741750.f,  322700.f,  743950.f,  323650.f,  746150.f,     328400.f,  757150.f,  329350.f,  759350.f,  330300.f,  761550.f,  331250.f,  763750.f,     332200.f,  765950.f,  333150.f,  768150.f,  337900.f,  779150.f,  338850.f,  781350.f,     339800.f,  783550.f,  340750.f,  785750.f,  341700.f,  787950.f,  342650.f,  790150.f,     347400.f,  801150.f,  348350.f,  803350.f,  349300.f,  805550.f,  350250.f,  807750.f,     351200.f,  809950.f,  352150.f,  812150.f,  356900.f,  823150.f,  357850.f,  825350.f,     358800.f,  827550.f,  359750.f,  829750.f,  360700.f,  831950.f,  361650.f,  834150.f};
    int _expS[] = {4, 2, 4, 6, 6, 144, 36, 6, 1, 0, 1, 99};
    NDArray<float> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);

    int sY = 1;
    int sX = 1;
    int pY = 0;
    int pX = 0;
    int iC = 2;
    int oC = 2;
    int kY = 5;
    int kX = 5;
    int iY = 10;
    int iX = 10;
    int B = 2;

    auto input = new NDArray<float>('c', {B, iC, iY, iX});
    for (int e = 0; e < input->lengthOf(); e++)
        input->putScalar(e, e+1);

    auto weights = new NDArray<float> ('c', {oC, iC, kY, kX});
    for (int e = 0; e < weights->lengthOf(); e++)
        weights->putScalar(e, e+1);

    auto variableSpace = new VariableSpace<float>();
    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, weights);

    auto block = new Block<float>(1, variableSpace, false);
    block->fillInputs({-1, -2});

    block->getIArguments()->push_back(kY);
    block->getIArguments()->push_back(kX);

    block->getIArguments()->push_back(sY);
    block->getIArguments()->push_back(sX);

    block->getIArguments()->push_back(pY);
    block->getIArguments()->push_back(pX);

    // dilation
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(1);

    // NOT same mode
    block->getIArguments()->push_back(0);

    nd4j::ops::sconv2d<float> op;

    Nd4jStatus status = op.execute(block);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    NDArray<float>* output = variableSpace->getVariable(1)->getNDArray();

    output->printShapeInfo("Result shape");
    ASSERT_TRUE(exp.isSameShape(output));

        exp.printBuffer("Expctd buffer", 50);
    output->printBuffer("Result buffer", 50);
    ASSERT_TRUE(exp.equalsTo(output));
}

#endif //LIBND4J_CONVOLUTIONTESTS_H
