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
#include <NDArrayFactory.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
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

    exp.printShapeInfo("Expected shape");
    output->printShapeInfo("Result shape");
    ASSERT_TRUE(exp.isSameShape(output));

        //exp.printBuffer("Expctd buffer");
    //output->printBuffer("Result buffer");
    ASSERT_TRUE(exp.equalsTo(output));
}

TEST_F(ConvolutionTests, SeparableConv2D_BP_NoBias_1) {
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
    
    //double _expWB[] = {10658316.0,  10892892.0,  11127468.0,  11362044.0,  11596620.0,  11831196.0,     12065772.0,  12300348.0,  12534924.0,  12769500.0,  13004076.0,  13238652.0,     13473228.0,  13707804.0,  13942380.0,  14176956.0,  14411532.0,  14646108.0,     14880684.0,  15115260.0,  15349836.0,  15584412.0,  15818988.0,  16053564.0,     16288140.0,  25949820.0,  26371020.0,  26792220.0,  27213420.0,  27634620.0,     28055820.0,  28477020.0,  28898220.0,  29319420.0,  29740620.0,  30161820.0,     30583020.0,  31004220.0,  31425420.0,  31846620.0,  32267820.0,  32689020.0,     33110220.0,  33531420.0,  33952620.0,  34373820.0,  34795020.0,  35216220.0,     35637420.0,  36058620.0,  13039068.0,  13366956.0,  13694844.0,  14022732.0,     14350620.0,  14678508.0,  15006396.0,  15334284.0,  15662172.0,  15990060.0,     16317948.0,  16645836.0,  16973724.0,  17301612.0,  17629500.0,  17957388.0,     18285276.0,  18613164.0,  18941052.0,  19268940.0,  19596828.0,  19924716.0,     20252604.0,  20580492.0,  20908380.0,  30663372.0,  31177884.0,  31692396.0,     32206908.0,  32721420.0,  33235932.0,  33750444.0,  34264956.0,  34779468.0,     35293980.0,  35808492.0,  36323004.0,  36837516.0,  37352028.0,  37866540.0,     38381052.0,  38895564.0,  39410076.0,  39924588.0,  40439100.0,  40953612.0,     41468124.0,  41982636.0,  42497148.0,  43011660.0};
    double _expWB[] = {10658316.0,  10892892.0,  11127468.0,  11362044.0,  11596620.0,  11831196.0,     12065772.0,  12300348.0,  12534924.0,  12769500.0,  13004076.0,  13238652.0,     13473228.0,  13707804.0,  13942380.0,  14176956.0,  14411532.0,  14646108.0,     14880684.0,  15115260.0,  15349836.0,  15584412.0,  15818988.0,  16053564.0,     16288140.0,  25949820.0,  26371020.0,  26792220.0,  27213420.0,  27634620.0,     28055820.0,  28477020.0,  28898220.0,  29319420.0,  29740620.0,  30161820.0,     30583020.0,  31004220.0,  31425420.0,  31846620.0,  32267820.0,  32689020.0,     33110220.0,  33531420.0,  33952620.0,  34373820.0,  34795020.0,  35216220.0,     35637420.0,  36058620.0,  13039068.0,  13366956.0,  13694844.0,  14022732.0,     14350620.0,  14678508.0,  15006396.0,  15334284.0,  15662172.0,  15990060.0,     16317948.0,  16645836.0,  16973724.0,  17301612.0,  17629500.0,  17957388.0,     18285276.0,  18613164.0,  18941052.0,  19268940.0,  19596828.0,  19924716.0,     20252604.0,  20580492.0,  20908380.0,  30663372.0,  31177884.0,  31692396.0,     32206908.0,  32721420.0,  33235932.0,  33750444.0,  34264956.0,  34779468.0,     35293980.0,  35808492.0,  36323004.0,  36837516.0,  37352028.0,  37866540.0,     38381052.0,  38895564.0,  39410076.0,  39924588.0,  40439100.0,  40953612.0,     41468124.0,  41982636.0,  42497148.0,  43011660.0,};
    int _expWS[] = {4, 2, 2, 5, 5, 50, 25 ,5, 1,  0, 1, 99};
    
    double _expEB[] = {1888.0,    3866.0,    5936.0,    8100.0,   10360.0,   10640.0,    8720.0,    6698.0,    4572.0,    2340.0,    4278.0,    8758.0,   13444.0,   18340.0,   23450.0,   24060.0,    19708.0,   15130.0,   10322.0,    5280.0,    7230.0,   14796.0,   22704.0,   30960.0,    39570.0,   40560.0,   33204.0,   25476.0,   17370.0,    8880.0,   10804.0,   22100.0,    33896.0,   46200.0,   59020.0,   60440.0,   49448.0,   37916.0,   25836.0,   13200.0,    15060.0,   30790.0,   47200.0,   64300.0,   82100.0,   84000.0,   68680.0,   52630.0,    35840.0,   18300.0,   17220.0,   35170.0,   53860.0,   73300.0,   93500.0,   95400.0,    77920.0,   59650.0,   40580.0,   20700.0,   15620.0,   31868.0,   48752.0,   66280.0,    84460.0,   86080.0,   70232.0,   53708.0,   36500.0,   18600.0,   13158.0,   26820.0,    40992.0,   55680.0,   70890.0,   72180.0,   58836.0,   44952.0,   30522.0,   15540.0,    9774.0,   19906.0,   30400.0,   41260.0,   52490.0,   53400.0,   43492.0,   33202.0,    22526.0,   11460.0,    5408.0,   11006.0,   16796.0,   22780.0,   28960.0,   29440.0,    23960.0,   18278.0,   12392.0,    6300.0,   10182.0,   20648.0,   31400.0,   42440.0,    53770.0,   54300.0,   44036.0,   33476.0,   22618.0,   11460.0,   21886.0,   44362.0,    67432.0,   91100.0,  115370.0,  116480.0,   94420.0,   71746.0,   48454.0,   24540.0,    35172.0,   71262.0,  108276.0,  146220.0,  185100.0,  186840.0,  151392.0,  114990.0,    77628.0,   39300.0,   50100.0,  101468.0,  154112.0,  208040.0,  263260.0,  265680.0,    215192.0,  163388.0,  110260.0,   55800.0,   66730.0,  135100.0,  205120.0,  276800.0,    350150.0,  353300.0,  286060.0,  217120.0,  146470.0,   74100.0,   70390.0,  142480.0,    216280.0,  291800.0,  369050.0,  372200.0,  301300.0,  228640.0,  154210.0,   78000.0,    60196.0,  121796.0,  184808.0,  249240.0,  315100.0,  317720.0,  257096.0,  195020.0,    131484.0,   66480.0,   48120.0,   97326.0,  147624.0,  199020.0,  251520.0,  253560.0,    205104.0,  155526.0,  104820.0,   52980.0,   34102.0,   68950.0,  104548.0,  140900.0,    178010.0,  179420.0,  145084.0,  109978.0,   74098.0,   37440.0,   18082.0,   36548.0,    55400.0,   74640.0,   94270.0,   95000.0,   76796.0,   58196.0,   39198.0,   19800.0,    9376.0,   19130.0,   29264.0,   39780.0,   50680.0,   50960.0,   41552.0,   31754.0,    21564.0,   10980.0,   20694.0,   42166.0,   64420.0,   87460.0,  111290.0,  111900.0,    91132.0,   69562.0,   47186.0,   24000.0,   34014.0,   69228.0,  105648.0,  143280.0,    182130.0,  183120.0,  148980.0,  113604.0,   76986.0,   39120.0,   49396.0,  100436.0,    153128.0,  207480.0,  263500.0,  264920.0,  215336.0,  164060.0,  111084.0,   56400.0,    66900.0,  135910.0,  207040.0,  280300.0,  355700.0,  357600.0,  290440.0,  221110.0,    149600.0,   75900.0,   69060.0,  140290.0,  213700.0,  289300.0,  367100.0,  369000.0,    299680.0,  228130.0,  154340.0,   78300.0,   59972.0,  121724.0,  185264.0,  250600.0,    317740.0,  319360.0,  259160.0,  197132.0,  133268.0,   67560.0,   48582.0,   98532.0,    149856.0,  202560.0,  256650.0,  257940.0,  209172.0,  159000.0,  107418.0,   54420.0,    34830.0,   70594.0,  107296.0,  144940.0,  183530.0,  184440.0,  149476.0,  113554.0,    76670.0,   38820.0,   18656.0,   37790.0,   57404.0,   77500.0,   98080.0,   98560.0,    79832.0,   60614.0,   40904.0,   20700.0,   24870.0,   50312.0,   76328.0,  102920.0,    130090.0,  130620.0,  105668.0,   80132.0,   54010.0,   27300.0,   52702.0,  106570.0,    161608.0,  217820.0,  275210.0,  276320.0,  223444.0,  169378.0,  114118.0,   57660.0,    83556.0,  168894.0,  256020.0,  344940.0,  435660.0,  437400.0,  353568.0,  267918.0,    180444.0,   91140.0,  117492.0,  237404.0,  359744.0,  484520.0,  611740.0,  614160.0,    496280.0,  375932.0,  253108.0,  127800.0,  154570.0,  312220.0,  472960.0,  636800.0,    803750.0,  806900.0,  651820.0,  493600.0,  332230.0,  167700.0,  158230.0,  319600.0,    484120.0,  651800.0,  822650.0,  825800.0,  667060.0,  505120.0,  339970.0,  171600.0,    133348.0,  269252.0,  407720.0,  548760.0,  692380.0,  695000.0,  561224.0,  424844.0,    285852.0,  144240.0,  105144.0,  212238.0,  321288.0,  432300.0,  545280.0,  547320.0,    441840.0,  334374.0,  224916.0,  113460.0,   73558.0,  148438.0,  224644.0,  302180.0,    381050.0,  382460.0,  308668.0,  233530.0,  157042.0,   79200.0,   38530.0,   77732.0,    117608.0,  158160.0,  199390.0,  200120.0,  161468.0,  122132.0,   82110.0,   41400.0,};
    int _expES[] = {4, 2, 2, 10, 10, 200, 100, 10, 1, 0, 1, 99};

    NDArray<double> expW(_expWB, _expWS);
    expW.triggerAllocationFlag(false, false);

    NDArray<double> expE(_expEB, _expES);
    expE.triggerAllocationFlag(false, false);

    auto input = new NDArray<double>('c', {B, iC, iY, iX});
    for (int e = 0; e < input->lengthOf(); e++)
        input->putScalar(e, e+1);

    auto weights = new NDArray<double> ('c', {oC, iC, kY, kX});
    for (int e = 0; e < weights->lengthOf(); e++)
        weights->putScalar(e, e+1);


    auto epsilonNext = new NDArray<double>('c', {B, iC * oC, 6, 6});
    for (int e = 0; e < epsilonNext->lengthOf(); e++)
        epsilonNext->putScalar(e, e+1);

    auto col = new NDArray<double>('c', {B, iC, kY, kX, 6, 6});
    for (int e = 0; e < col->lengthOf(); e++)
        col->putScalar(e, e+1);


    auto variableSpace = new VariableSpace<double>();
    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, weights);
    variableSpace->putVariable(-3, epsilonNext);

    auto block = new Block<double>(1, variableSpace, false);
    block->fillInputs({-1, -2, -3});

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


    variableSpace->getStash()->storeArray(1, "im2col", col);


    nd4j::ops::sconv2d_bp<double> op;

    Nd4jStatus status = op.execute(block);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    std::pair<int, int> pe(1, 0);
    std::pair<int, int> pgw(1, 1);
    auto epsilon = variableSpace->getVariable(pe)->getNDArray();
    auto gradW = variableSpace->getVariable(pgw)->getNDArray();

    ASSERT_TRUE(expW.isSameShape(gradW));

    //expW.printBuffer("Expctd buffer");
    //gradW->printBuffer("Result buffer");
    ASSERT_TRUE(expW.equalsTo(gradW));


    ASSERT_TRUE(expE.isSameShape(epsilon));

    //    expE.printBuffer("Expctd buffer");
    //epsilon->printBuffer("Result buffer");
    ASSERT_TRUE(expE.equalsTo(epsilon));

    delete variableSpace;
    delete block;
}

TEST_F(ConvolutionTests, deconv2D_FF_NoBias_1) {
    int _expS[] = {4, 2, 3, 8, 8, 192, 64, 8, 1, 0, 1, 99};
    double _expB[] = {6276.0,   12831.0,   19668.0,   26790.0,   27012.0,   20703.0,   14100.0,    7200.0,    13719.0,   28023.0,   42918.0,   58410.0,   58902.0,   45105.0,   30693.0,   15660.0,    22389.0,   45696.0,   69930.0,   95100.0,   95910.0,   73386.0,   49899.0,   25440.0,    32346.0,   65970.0,  100884.0,  137100.0,  138276.0,  105726.0,   71838.0,   36600.0,    33726.0,   68790.0,  105204.0,  142980.0,  144156.0,  110226.0,   74898.0,   38160.0,    27555.0,   56154.0,   85806.0,  116520.0,  117474.0,   89748.0,   60933.0,   31020.0,    19917.0,   40557.0,   61926.0,   84030.0,   84714.0,   64671.0,   43875.0,   22320.0,    10752.0,   21879.0,   33384.0,   45270.0,   45636.0,   34815.0,   23604.0,   12000.0,    7551.0,   15456.0,   23718.0,   32340.0,   32562.0,   24978.0,   17025.0,    8700.0,    16569.0,   33873.0,   51918.0,   70710.0,   71202.0,   54555.0,   37143.0,   18960.0,    27114.0,   55371.0,   84780.0,  115350.0,  116160.0,   88911.0,   60474.0,   30840.0,    39246.0,   80070.0,  122484.0,  166500.0,  167676.0,  128226.0,   87138.0,   44400.0,    40626.0,   82890.0,  126804.0,  172380.0,  173556.0,  132726.0,   90198.0,   45960.0,    33180.0,   67629.0,  103356.0,  140370.0,  141324.0,  107973.0,   73308.0,   37320.0,    23967.0,   48807.0,   74526.0,  101130.0,  101814.0,   77721.0,   52725.0,   26820.0,    12927.0,   26304.0,   40134.0,   54420.0,   54786.0,   41790.0,   28329.0,   14400.0,    8826.0,   18081.0,   27768.0,   37890.0,   38112.0,   29253.0,   19950.0,   10200.0,    19419.0,   39723.0,   60918.0,   83010.0,   83502.0,   64005.0,   43593.0,   22260.0,    31839.0,   65046.0,   99630.0,  135600.0,  136410.0,  104436.0,   71049.0,   36240.0,    46146.0,   94170.0,  144084.0,  195900.0,  197076.0,  150726.0,  102438.0,   52200.0,    47526.0,   96990.0,  148404.0,  201780.0,  202956.0,  155226.0,  105498.0,   53760.0,    38805.0,   79104.0,  120906.0,  164220.0,  165174.0,  126198.0,   85683.0,   43620.0,    28017.0,   57057.0,   87126.0,  118230.0,  118914.0,   90771.0,   61575.0,   31320.0,    15102.0,   30729.0,   46884.0,   63570.0,   63936.0,   48765.0,   33054.0,   16800.0,    17220.0,   34863.0,   52932.0,   71430.0,   72228.0,   54831.0,   36996.0,   18720.0,    36327.0,   73527.0,  111606.0,  150570.0,  152214.0,  115521.0,   77925.0,   39420.0,    57381.0,  116112.0,  176202.0,  237660.0,  240198.0,  182250.0,  122907.0,   62160.0,    80442.0,  162738.0,  246900.0,  332940.0,  336420.0,  255198.0,  172062.0,   87000.0,    84702.0,  171318.0,  259860.0,  350340.0,  353820.0,  268338.0,  180882.0,   91440.0,    66867.0,  135210.0,  205038.0,  276360.0,  279042.0,  211572.0,  142581.0,   72060.0,    46845.0,   94701.0,  143574.0,  193470.0,  195306.0,  148047.0,   99747.0,   50400.0,    24576.0,   49671.0,   75288.0,  101430.0,  102372.0,   77583.0,   52260.0,   26400.0,    22095.0,   44688.0,   67782.0,   91380.0,   92178.0,   69906.0,   47121.0,   23820.0,    46377.0,   93777.0,  142206.0,  191670.0,  193314.0,  146571.0,   98775.0,   49920.0,    72906.0,  147387.0,  223452.0,  301110.0,  303648.0,  230175.0,  155082.0,   78360.0,    101742.0,  205638.0,  311700.0,  419940.0,  423420.0,  320898.0,  216162.0,  109200.0,    106002.0,  214218.0,  324660.0,  437340.0,  440820.0,  334038.0,  224982.0,  113640.0,    83292.0,  168285.0,  254988.0,  343410.0,  346092.0,  262197.0,  176556.0,   89160.0,    58095.0,  117351.0,  177774.0,  239370.0,  241206.0,  182697.0,  122997.0,   62100.0,    30351.0,   61296.0,   92838.0,  124980.0,  125922.0,   95358.0,   64185.0,   32400.0,    26970.0,   54513.0,   82632.0,  111330.0,  112128.0,   84981.0,   57246.0,   28920.0,    56427.0,  114027.0,  172806.0,  232770.0,  234414.0,  177621.0,  119625.0,   60420.0,    88431.0,  178662.0,  270702.0,  364560.0,  367098.0,  278100.0,  187257.0,   94560.0,    123042.0,  248538.0,  376500.0,  506940.0,  510420.0,  386598.0,  260262.0,  131400.0,    127302.0,  257118.0,  389460.0,  524340.0,  527820.0,  399738.0,  269082.0,  135840.0,    99717.0,  201360.0,  304938.0,  410460.0,  413142.0,  312822.0,  210531.0,  106260.0,    69345.0,  140001.0,  211974.0,  285270.0,  287106.0,  217347.0,  146247.0,   73800.0,    36126.0,   72921.0,  110388.0,  148530.0,  149472.0,  113133.0,   76110.0,   38400.0,};
    NDArray<double> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);


    auto input = new NDArray<double>('c', {2, 3, 4, 4});
    auto weights = new NDArray<double>('c', {3, 3, 5, 5});

    nd4j::NDArrayFactory<double>::linspace(1, *input);
    nd4j::NDArrayFactory<double>::linspace(1, *weights);

    auto variableSpace = new VariableSpace<double>();
    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, weights);

    auto block = new Block<double>(1, variableSpace, false);
    block->fillInputs({-1, -2});

    block->getIArguments()->push_back(5);
    block->getIArguments()->push_back(5);

    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(1);

    block->getIArguments()->push_back(0);
    block->getIArguments()->push_back(0);

    // dilation
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(1);

    // NOT same mode
    block->getIArguments()->push_back(0);

    nd4j::ops::deconv2d<double> op;

    Nd4jStatus status = op.execute(block);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    NDArray<double>* output = variableSpace->getVariable(1)->getNDArray();

    ASSERT_TRUE(exp.isSameShape(output));

        exp.printBuffer("Expctd buffer");
    output->printBuffer("Result buffer");
    ASSERT_TRUE(exp.equalsTo(output));
}

#endif //LIBND4J_CONVOLUTIONTESTS_H
