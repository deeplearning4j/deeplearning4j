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
    float _expB[] = {10025.0f,    10350.0f,    10675.0f,    11000.0f,    11325.0f,    11650.0f,    13275.0f,    13600.0f,    13925.0f,    14250.0f,    14575.0f,    14900.0f,    16525.0f,    16850.0f,    17175.0f,    17500.0f,    17825.0f,    18150.0f,    19775.0f,    20100.0f,    20425.0f,    20750.0f,    21075.0f,    21400.0f,    23025.0f,    23350.0f,    23675.0f,    24000.0f,    24325.0f,    24650.0f,    26275.0f,    26600.0f,    26925.0f,    27250.0f,    27575.0f,    27900.0f,    38775.0f,    40350.0f,    41925.0f,    43500.0f,    45075.0f,    46650.0f,    54525.0f,    56100.0f,    57675.0f,    59250.0f,    60825.0f,    62400.0f,    70275.0f,    71850.0f,    73425.0f,    75000.0f,    76575.0f,    78150.0f,    86025.0f,    87600.0f,    89175.0f,    90750.0f,    92325.0f,    93900.0f,   101775.0f,   103350.0f,   104925.0f,    106500.0f,   108075.0f,   109650.0f,   117525.0f,   119100.0f,   120675.0f,   122250.0f,    123825.0f,   125400.0f,    67525.0f,    70350.0f,    73175.0f,    76000.0f,    78825.0f,    81650.0f,    95775.0f,    98600.0f,   101425.0f,   104250.0f,   107075.0f,   109900.0f,    124025.0f,   126850.0f,   129675.0f,   132500.0f,   135325.0f,   138150.0f,   152275.0f,    155100.0f,   157925.0f,   160750.0f,   163575.0f,   166400.0f,   180525.0f,   183350.0f,    186175.0f,   189000.0f,   191825.0f,   194650.0f,   208775.0f,   211600.0f,   214425.0f,    217250.0f,   220075.0f,   222900.0f,   119400.0f,   120350.0f,   121300.0f,   122250.0f,    123200.0f,   124150.0f,   128900.0f,   129850.0f,   130800.0f,   131750.0f,   132700.0f,    133650.0f,   138400.0f,   139350.0f,   140300.0f,   141250.0f,   142200.0f,   143150.0f,    147900.0f,   148850.0f,   149800.0f,   150750.0f,   151700.0f,   152650.0f,   157400.0f,    158350.0f,   159300.0f,   160250.0f,   161200.0f,   162150.0f,   166900.0f,   167850.0f,    168800.0f,   169750.0f,   170700.0f,   171650.0f,   273150.0f,   275350.0f,   277550.0f,    279750.0f,   281950.0f,   284150.0f,   295150.0f,   297350.0f,   299550.0f,   301750.0f,    303950.0f,   306150.0f,   317150.0f,   319350.0f,   321550.0f,   323750.0f,   325950.0f,    328150.0f,   339150.0f,   341350.0f,   343550.0f,   345750.0f,   347950.0f,   350150.0f,    361150.0f,   363350.0f,   365550.0f,   367750.0f,   369950.0f,   372150.0f,   383150.0f,    385350.0f,   387550.0f,   389750.0f,   391950.0f,   394150.0f,   426900.0f,   430350.0f,    433800.0f,   437250.0f,   440700.0f,   444150.0f,   461400.0f,   464850.0f,   468300.0f,    471750.0f,   475200.0f,   478650.0f,   495900.0f,   499350.0f,   502800.0f,   506250.0f,    509700.0f,   513150.0f,   530400.0f,   533850.0f,   537300.0f,   540750.0f,   544200.0f,    547650.0f,   564900.0f,   568350.0f,   571800.0f,   575250.0f,   578700.0f,   582150.0f,    599400.0f,   602850.0f,   606300.0f,   609750.0f,   613200.0f,   616650.0f,    75025.0f,    75350.0f,    75675.0f,    76000.0f,    76325.0f,    76650.0f,    78275.0f,    78600.0f,    78925.0f,    79250.0f,    79575.0f,    79900.0f,    81525.0f,    81850.0f,    82175.0f,    82500.0f,    82825.0f,    83150.0f,    84775.0f,    85100.0f,    85425.0f,    85750.0f,    86075.0f,    86400.0f,    88025.0f,    88350.0f,    88675.0f,    89000.0f,    89325.0f,    89650.0f,    91275.0f,    91600.0f,    91925.0f,    92250.0f,    92575.0f,    92900.0f,    353775.0f,   355350.0f,   356925.0f,   358500.0f,   360075.0f,   361650.0f,   369525.0f,    371100.0f,   372675.0f,   374250.0f,   375825.0f,   377400.0f,   385275.0f,   386850.0f,    388425.0f,   390000.0f,   391575.0f,   393150.0f,   401025.0f,   402600.0f,   404175.0f,    405750.0f,   407325.0f,   408900.0f,   416775.0f,   418350.0f,   419925.0f,   421500.0f,    423075.0f,   424650.0f,   432525.0f,   434100.0f,   435675.0f,   437250.0f,   438825.0f,    440400.0f,   632525.0f,   635350.0f,   638175.0f,   641000.0f,   643825.0f,   646650.0f,    660775.0f,   663600.0f,   666425.0f,   669250.0f,   672075.0f,   674900.0f,   689025.0f,    691850.0f,   694675.0f,   697500.0f,   700325.0f,   703150.0f,   717275.0f,   720100.0f,    722925.0f,   725750.0f,   728575.0f,   731400.0f,   745525.0f,   748350.0f,   751175.0f,    754000.0f,   756825.0f,   759650.0f,   773775.0f,   776600.0f,   779425.0f,   782250.0f,    785075.0f,   787900.0f,   309400.0f,   310350.0f,   311300.0f,   312250.0f,   313200.0f,    314150.0f,   318900.0f,   319850.0f,   320800.0f,   321750.0f,   322700.0f,   323650.0f,    328400.0f,   329350.0f,   330300.0f,   331250.0f,   332200.0f,   333150.0f,   337900.0f,    338850.0f,   339800.0f,   340750.0f,   341700.0f,   342650.0f,   347400.0f,   348350.0f,    349300.0f,   350250.0f,   351200.0f,   352150.0f,   356900.0f,   357850.0f,   358800.0f,    359750.0f,   360700.0f,   361650.0f,   713150.0f,   715350.0f,   717550.0f,   719750.0f,    721950.0f,   724150.0f,   735150.0f,   737350.0f,   739550.0f,   741750.0f,   743950.0f,    746150.0f,   757150.0f,   759350.0f,   761550.0f,   763750.0f,   765950.0f,   768150.0f,    779150.0f,   781350.0f,   783550.0f,   785750.0f,   787950.0f,   790150.0f,   801150.0f,    803350.0f,   805550.0f,   807750.0f,   809950.0f,   812150.0f,   823150.0f,   825350.0f,    827550.0f,   829750.0f,   831950.0f,   834150.0f,  1116900.0f,  1120350.0f,  1123800.0f,    1127250.0f,  1130700.0f,  1134150.0f,  1151400.0f,  1154850.0f,  1158300.0f,  1161750.0f,    1165200.0f,  1168650.0f,  1185900.0f,  1189350.0f,  1192800.0f,  1196250.0f,  1199700.0f,    1203150.0f,  1220400.0f,  1223850.0f,  1227300.0f,  1230750.0f,  1234200.0f,  1237650.0f,    1254900.0f,  1258350.0f,  1261800.0f,  1265250.0f,  1268700.0f,  1272150.0f,  1289400.0f,    1292850.0f,  1296300.0f,  1299750.0f,  1303200.0f,  1306650.0f,};
    int _expS[] = {4, 2, 6, 6, 6, 144, 36, 6, 1, 0, 1, 99};
    NDArray<float> exp(_expB, _expS);
    exp.triggerAllocationFlag(false, false);

    int sY = 1;
    int sX = 1;
    int pY = 0;
    int pX = 0;
    int iC = 2;
    int oC = 3;
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

        exp.printBuffer("Expctd buffer");
    output->printBuffer("Result buffer");
    ASSERT_TRUE(exp.equalsTo(output));
}

TEST_F(ConvolutionTests, SeparableConv2D_BP_NoBias_1) {
    int sY = 1;
    int sX = 1;
    int pY = 0;
    int pX = 0;
    int iC = 2;
    int oC = 3;
    int kY = 5;
    int kX = 5;
    int iY = 10;
    int iX = 10;
    int B = 2;
    
    //double _expWB[] = {10658316.0,  10892892.0,  11127468.0,  11362044.0,  11596620.0,  11831196.0,     12065772.0,  12300348.0,  12534924.0,  12769500.0,  13004076.0,  13238652.0,     13473228.0,  13707804.0,  13942380.0,  14176956.0,  14411532.0,  14646108.0,     14880684.0,  15115260.0,  15349836.0,  15584412.0,  15818988.0,  16053564.0,     16288140.0,  25949820.0,  26371020.0,  26792220.0,  27213420.0,  27634620.0,     28055820.0,  28477020.0,  28898220.0,  29319420.0,  29740620.0,  30161820.0,     30583020.0,  31004220.0,  31425420.0,  31846620.0,  32267820.0,  32689020.0,     33110220.0,  33531420.0,  33952620.0,  34373820.0,  34795020.0,  35216220.0,     35637420.0,  36058620.0,  13039068.0,  13366956.0,  13694844.0,  14022732.0,     14350620.0,  14678508.0,  15006396.0,  15334284.0,  15662172.0,  15990060.0,     16317948.0,  16645836.0,  16973724.0,  17301612.0,  17629500.0,  17957388.0,     18285276.0,  18613164.0,  18941052.0,  19268940.0,  19596828.0,  19924716.0,     20252604.0,  20580492.0,  20908380.0,  30663372.0,  31177884.0,  31692396.0,     32206908.0,  32721420.0,  33235932.0,  33750444.0,  34264956.0,  34779468.0,     35293980.0,  35808492.0,  36323004.0,  36837516.0,  37352028.0,  37866540.0,     38381052.0,  38895564.0,  39410076.0,  39924588.0,  40439100.0,  40953612.0,     41468124.0,  41982636.0,  42497148.0,  43011660.0};
    double _expWB[] = {15371868.0f,  15699756.0f,  16027644.0f,  16355532.0f,  16683420.0f,  17011308.0f,    17339196.0f,  17667084.0f,  17994972.0f,  18322860.0f,  18650748.0f,  18978636.0f,    19306524.0f,  19634412.0f,  19962300.0f,  20290188.0f,  20618076.0f,  20945964.0f,    21273852.0f,  21601740.0f,  21929628.0f,  22257516.0f,  22585404.0f,  22913292.0f,    23241180.0f,  37709724.0f,  38317548.0f,  38925372.0f,  39533196.0f,  40141020.0f,    40748844.0f,  41356668.0f,  41964492.0f,  42572316.0f,  43180140.0f,  43787964.0f,    44395788.0f,  45003612.0f,  45611436.0f,  46219260.0f,  46827084.0f,  47434908.0f,    48042732.0f,  48650556.0f,  49258380.0f,  49866204.0f,  50474028.0f,  51081852.0f,    51689676.0f,  52297500.0f,  17752620.0f,  18173820.0f,  18595020.0f,  19016220.0f,    19437420.0f,  19858620.0f,  20279820.0f,  20701020.0f,  21122220.0f,  21543420.0f,    21964620.0f,  22385820.0f,  22807020.0f,  23228220.0f,  23649420.0f,  24070620.0f,    24491820.0f,  24913020.0f,  25334220.0f,  25755420.0f,  26176620.0f,  26597820.0f,    27019020.0f,  27440220.0f,  27861420.0f,  42423276.0f,  43124412.0f,  43825548.0f,    44526684.0f,  45227820.0f,  45928956.0f,  46630092.0f,  47331228.0f,  48032364.0f,    48733500.0f,  49434636.0f,  50135772.0f,  50836908.0f,  51538044.0f,  52239180.0f,    52940316.0f,  53641452.0f,  54342588.0f,  55043724.0f,  55744860.0f,  56445996.0f,    57147132.0f,  57848268.0f,  58549404.0f,  59250540.0f,  20133372.0f,  20647884.0f,    21162396.0f,  21676908.0f,  22191420.0f,  22705932.0f,  23220444.0f,  23734956.0f,    24249468.0f,  24763980.0f,  25278492.0f,  25793004.0f,  26307516.0f,  26822028.0f,    27336540.0f,  27851052.0f,  28365564.0f,  28880076.0f,  29394588.0f,  29909100.0f,    30423612.0f,  30938124.0f,  31452636.0f,  31967148.0f,  32481660.0f,  47136828.0f,    47931276.0f,  48725724.0f,  49520172.0f,  50314620.0f,  51109068.0f,  51903516.0f,    52697964.0f,  53492412.0f,  54286860.0f,  55081308.0f,  55875756.0f,  56670204.0f,    57464652.0f,  58259100.0f,  59053548.0f,  59847996.0f,  60642444.0f,  61436892.0f,    62231340.0f,  63025788.0f,  63820236.0f,  64614684.0f,  65409132.0f,  66203580.0f,};
    int _expWS[] = {4, 3, 2, 5, 5, 50, 25 ,5, 1,  0, 1, 99};
    
    double _expEB[] = {9261.0f,    18786.0f,    28578.0f,    38640.0f,    48975.0f,    49770.0f,    40386.0f,    30720.0f,    20769.0f,    10530.0f,    19995.0f,    40551.0f,    61674.0f,    83370.0f,    105645.0f,   107310.0f,    87054.0f,    66201.0f,    44745.0f,    22680.0f,    32292.0f,    65475.0f,    99558.0f,   134550.0f,   170460.0f,   173070.0f,   140364.0f,   106713.0f,    72108.0f,    36540.0f,    46242.0f,    93738.0f,   142500.0f,   192540.0f,   243870.0f,    247500.0f,   200676.0f,   152526.0f,   103038.0f,    52200.0f,    61935.0f,   125520.0f,    190770.0f,   257700.0f,   326325.0f,   331050.0f,   268350.0f,   203910.0f,   137715.0f,    69750.0f,    67425.0f,   136590.0f,   207510.0f,   280200.0f,   354675.0f,   359400.0f,    291210.0f,   221190.0f,   149325.0f,    75600.0f,    58146.0f,   117750.0f,   178824.0f,    241380.0f,   305430.0f,   309360.0f,   250572.0f,   190254.0f,   128394.0f,    64980.0f,    46854.0f,    94851.0f,   144000.0f,   194310.0f,   245790.0f,   248850.0f,   201492.0f,    152937.0f,   103176.0f,    52200.0f,    33459.0f,    67713.0f,   102768.0f,   138630.0f,    175305.0f,   177420.0f,   143610.0f,   108969.0f,    73491.0f,    37170.0f,    17871.0f,    36156.0f,    54858.0f,    73980.0f,    93525.0f,    94620.0f,    76566.0f,    58080.0f,    39159.0f,    19800.0f,    36660.0f,    73983.0f,   111972.0f,   150630.0f,   189960.0f,    191130.0f,   154272.0f,   116733.0f,    78510.0f,    39600.0f,    76863.0f,   155085.0f,    234672.0f,   315630.0f,   397965.0f,   400380.0f,   323106.0f,   244437.0f,   164367.0f,    82890.0f,   120699.0f,   243486.0f,   368370.0f,   495360.0f,   624465.0f,   628200.0f,    506862.0f,   383382.0f,   257751.0f,   129960.0f,   168258.0f,   339366.0f,   513336.0f,    690180.0f,   869910.0f,   875040.0f,   705900.0f,   533838.0f,   358842.0f,   180900.0f,    219630.0f,   442905.0f,   669840.0f,   900450.0f,  1134750.0f,  1141350.0f,   920580.0f,    696075.0f,   467820.0f,   235800.0f,   227370.0f,   458475.0f,   693330.0f,   931950.0f,    1174350.0f,  1180950.0f,   952440.0f,   720105.0f,   483930.0f,   243900.0f,   190242.0f,    383538.0f,   579900.0f,   779340.0f,   981870.0f,   987300.0f,   796116.0f,   601806.0f,    404358.0f,   203760.0f,   149031.0f,   300402.0f,   454122.0f,   610200.0f,   768645.0f,    772830.0f,   623070.0f,   470916.0f,   316359.0f,   159390.0f,   103647.0f,   208887.0f,    315726.0f,   424170.0f,   534225.0f,   537090.0f,   432942.0f,   327165.0f,   219753.0f,    110700.0f,    54000.0f,   108813.0f,   164442.0f,   220890.0f,   278160.0f,   279630.0f,    225372.0f,   170283.0f,   114360.0f,    57600.0f,    42309.0f,    85530.0f,   129666.0f,    174720.0f,   220695.0f,   221490.0f,   179058.0f,   135696.0f,    91401.0f,    46170.0f,    89331.0f,   180519.0f,   273570.0f,   368490.0f,   465285.0f,   466950.0f,   377358.0f,    285873.0f,   192489.0f,    97200.0f,   141156.0f,   285147.0f,   431982.0f,   581670.0f,    734220.0f,   736830.0f,   595260.0f,   450801.0f,   303444.0f,   153180.0f,   197874.0f,    399594.0f,   605172.0f,   814620.0f,  1027950.0f,  1031580.0f,   833124.0f,   630750.0f,    424446.0f,   214200.0f,   259575.0f,   524040.0f,   793410.0f,  1067700.0f,  1346925.0f,    1351650.0f,  1091310.0f,   825990.0f,   555675.0f,   280350.0f,   265065.0f,   535110.0f,    810150.0f,  1090200.0f,  1375275.0f,  1380000.0f,  1114170.0f,   843270.0f,   567285.0f,    286200.0f,   222738.0f,   449526.0f,   680376.0f,   915300.0f,  1154310.0f,  1158240.0f,    934860.0f,   707358.0f,   475722.0f,   239940.0f,   175158.0f,   353403.0f,   534744.0f,    719190.0f,   906750.0f,   909810.0f,   734148.0f,   555345.0f,   373392.0f,   188280.0f,    122235.0f,   246561.0f,   372984.0f,   501510.0f,   632145.0f,   634260.0f,   511674.0f,    386961.0f,   260115.0f,   131130.0f,    63879.0f,   128820.0f,   194826.0f,   261900.0f,    330045.0f,   331140.0f,   267078.0f,   201936.0f,   135711.0f,    68400.0f,    85908.0f,    173127.0f,   261660.0f,   351510.0f,   442680.0f,   443850.0f,   357744.0f,   270309.0f,    181542.0f,    91440.0f,   178599.0f,   359853.0f,   543768.0f,   730350.0f,   919605.0f,    922020.0f,   743010.0f,   561309.0f,   376911.0f,   189810.0f,   278163.0f,   560358.0f,    846594.0f,  1136880.0f,  1431225.0f,  1434960.0f,  1156158.0f,   873270.0f,   586287.0f,    295200.0f,   384690.0f,   774822.0f,  1170408.0f,  1571460.0f,  1977990.0f,  1983120.0f,    1597548.0f,  1206462.0f,   809850.0f,   407700.0f,   498270.0f,  1003425.0f,  1515480.0f,    2034450.0f,  2560350.0f,  2566950.0f,  2067540.0f,  1561155.0f,  1047780.0f,   527400.0f,    506010.0f,  1018995.0f,  1538970.0f,  2065950.0f,  2599950.0f,  2606550.0f,  2099400.0f,    1585185.0f,  1063890.0f,   535500.0f,   419634.0f,   844914.0f,  1275852.0f,  1712460.0f,    2154750.0f,  2160180.0f,  1739604.0f,  1313310.0f,   881286.0f,   443520.0f,   325935.0f,    656154.0f,   990666.0f,  1329480.0f,  1672605.0f,  1676790.0f,  1350126.0f,  1019124.0f,    683775.0f,   344070.0f,   224823.0f,   452535.0f,   683142.0f,   916650.0f,  1153065.0f,    1155930.0f,   930606.0f,   702357.0f,   471177.0f,   237060.0f,   116208.0f,   233877.0f,    353010.0f,   473610.0f,   595680.0f,   597150.0f,   480684.0f,   362739.0f,   243312.0f,    122400.0f,};
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

    //    exp.printBuffer("Expctd buffer");
    //output->printBuffer("Result buffer");
    ASSERT_TRUE(exp.equalsTo(output));
}

TEST_F(ConvolutionTests, conv2D_BP_Bias_1) {
    printf("\n");
    double _expWGradB[] = {9312.0, 12580.0, 9528.0, 13168.0, 17712.0, 13360.0, 9960.0, 13348.0, 10032.0, 13344.0, 18148.0, 13848.0, 19312.0, 26160.0, 19888.0, 15144.0, 20452.0, 15504.0};
    int _expWGradS[] = {4, 2, 1, 3, 3, 9, 9, 3, 1, 0, 1, 99};
    NDArray<double> expWGrad(_expWGradB, _expWGradS);
    expWGrad.triggerAllocationFlag(false, false);

    double _expBGradB[] = {784.0, 1296.0};
    int _expBGradS[] = {2, 2, 1, 1, 1, 0, 1, 99};

    NDArray<double> expBGrad(_expBGradB, _expBGradS);
    expBGrad.triggerAllocationFlag(false, false);

    NDArray<double> input('c', {2, 1, 4, 4});
    NDArray<double> weights('c', {2, 1, 3, 3});
    NDArray<double> bias('c', {2, 1});
    NDArray<double> epsilonNext('c', {2, 2, 4, 4});


    double _expEpsB[] = {952.0, 1540.0, 1636.0, 1180.0, 1791.0, 2886.0, 3057.0, 2193.0, 2223.0, 3570.0, 3741.0, 2673.0, 1900.0, 3028.0, 3160.0, 2240.0, 2872.0, 4612.0, 4708.0, 3356.0, 5247.0, 8358.0, 8529.0, 6033.0, 5679.0, 9042.0, 9213.0, 6513.0, 4588.0, 7252.0, 7384.0, 5184.0};
    NDArray<double> expEps(_expEpsB, input.getShapeInfo());

    nd4j::NDArrayFactory<double>::linspace(1, input);
    nd4j::NDArrayFactory<double>::linspace(1, weights);
    nd4j::NDArrayFactory<double>::linspace(1, epsilonNext);

    nd4j::ops::conv2d_bp<double> op;

    auto results = op.execute({&input, &weights, &bias, &epsilonNext}, {},  {3, 3, 1, 1, 0, 0, 1, 1, 1});

    ASSERT_TRUE(results->size() == 3);

    auto epsilon = results->at(0);
    auto gradW = results->at(1);
    auto gradB = results->at(2);

    ASSERT_TRUE(expWGrad.isSameShape(gradW));

    //expWGrad.printBuffer("Expctd buffer");
    //  gradW->printBuffer("Result buffer");
    ASSERT_TRUE(expWGrad.equalsTo(gradW));


    ASSERT_TRUE(input.isSameShape(epsilon));

    //  expEps.printBuffer("Expctd buffer");
    //epsilon->printBuffer("Result buffer");
    ASSERT_TRUE(expEps.equalsTo(epsilon));

    ASSERT_TRUE(expBGrad.isSameShape(gradB));

    ASSERT_TRUE(expBGrad.equalsTo(gradB));
}


TEST_F(ConvolutionTests, conv2D_BP_NoBias_1) {
    printf("\n");
    double _expWGradB[] = {9312.0, 12580.0, 9528.0, 13168.0, 17712.0, 13360.0, 9960.0, 13348.0, 10032.0, 13344.0, 18148.0, 13848.0, 19312.0, 26160.0, 19888.0, 15144.0, 20452.0, 15504.0};
    int _expWGradS[] = {4, 2, 1, 3, 3, 9, 9, 3, 1, 0, 1, 99};
    NDArray<double> expWGrad(_expWGradB, _expWGradS);
    expWGrad.triggerAllocationFlag(false, false);

    NDArray<double> input('c', {2, 1, 4, 4});
    NDArray<double> weights('c', {2, 1, 3, 3});
    NDArray<double> epsilonNext('c', {2, 2, 4, 4});


    double _expEpsB[] = {952.0, 1540.0, 1636.0, 1180.0, 1791.0, 2886.0, 3057.0, 2193.0, 2223.0, 3570.0, 3741.0, 2673.0, 1900.0, 3028.0, 3160.0, 2240.0, 2872.0, 4612.0, 4708.0, 3356.0, 5247.0, 8358.0, 8529.0, 6033.0, 5679.0, 9042.0, 9213.0, 6513.0, 4588.0, 7252.0, 7384.0, 5184.0};
    NDArray<double> expEps(_expEpsB, input.getShapeInfo());

    nd4j::NDArrayFactory<double>::linspace(1, input);
    nd4j::NDArrayFactory<double>::linspace(1, weights);
    nd4j::NDArrayFactory<double>::linspace(1, epsilonNext);

    nd4j::ops::conv2d_bp<double> op;

    auto results = op.execute({&input, &weights, &epsilonNext}, {},  {3, 3, 1, 1, 0, 0, 1, 1, 1});

    ASSERT_TRUE(results->size() == 2);

    auto epsilon = results->at(0);
    auto gradW = results->at(1);

    ASSERT_TRUE(expWGrad.isSameShape(gradW));

    //expWGrad.printBuffer("Expctd buffer");
    //  gradW->printBuffer("Result buffer");
    ASSERT_TRUE(expWGrad.equalsTo(gradW));


    ASSERT_TRUE(input.isSameShape(epsilon));

    //  expEps.printBuffer("Expctd buffer");
    //epsilon->printBuffer("Result buffer");
    ASSERT_TRUE(expEps.equalsTo(epsilon));
}

#endif //LIBND4J_CONVOLUTIONTESTS_H
