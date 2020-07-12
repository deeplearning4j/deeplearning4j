/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#ifndef LIBND4J_CONVOLUTIONTESTS1_H
#define LIBND4J_CONVOLUTIONTESTS1_H

#include "testlayers.h"
#include <array/NDArray.h>
#include <graph/Context.h>
#include <graph/Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/col2im.h>
#include <helpers/PointersManager.h>
#include <helpers/GradCheck.h>

#ifdef HAVE_MKLDNN
#include <ops/declarable/platform/mkldnn/mkldnnUtils.h>
#endif

using namespace sd;
using namespace sd::graph;

class ConvolutionTests1 : public testing::Test {
public:

};

template <typename T>
class TypedConvolutionTests1 : public testing::Test {
public:

};

typedef ::testing::Types<double, float> TestingTypes;
TYPED_TEST_CASE(TypedConvolutionTests1, TestingTypes);

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv2d_1) {

    int bS=1, iH=5,iW=4,  iC=2,oC=3,  kH=2,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    TypeParam _expB[]{664.0, 700.0, 736.0, 344.0, 808.0, 844.0, 880.0, 408.0, 952.0, 988.0, 1024.0, 472.0, 1096.0, 1132.0, 1168.0, 536.0, 466.0, 480.0, 494.0, 220.0, 1528.0, 1628.0, 1728.0, 856.0, 1928.0, 2028.0, 2128.0, 1048.0, 2328.0, 2428.0, 2528.0, 1240.0, 2728.0, 2828.0, 2928.0, 1432.0, 1346.0, 1392.0, 1438.0, 700.0, 2392.0, 2556.0, 2720.0, 1368.0, 3048.0, 3212.0, 3376.0, 1688.0, 3704.0, 3868.0, 4032.0, 2008.0, 4360.0, 4524.0, 4688.0, 2328.0, 2226.0, 2304.0, 2382.0, 1180.0};
    Nd4jLong _expS[]{4, 1, 3, 5, 4, 60, 20, 4, 1, typeid(TypeParam) == typeid(float) ? 8192 : 16384, 1, 99};
    auto input = NDArrayFactory::create_<TypeParam>('c', {bS, iC, iH, iW});
    auto weights = NDArrayFactory::create_<TypeParam>('c', {oC, iC, kH, kW});
    for (int e = 0; e < input->lengthOf(); e++)
        input->p(e, e + 1);

    for (int e = 0; e < weights->lengthOf(); e++)
        weights->p(e, e + 1);
    weights->permutei({2,3,1,0});

    // weights->printShapeInfo("weights");

    ArrayOptions::setDataType(_expS, input->dataType());
    auto exp = new NDArray(_expB, _expS);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, weights);

    auto block = new Context(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1, -2});
    // 5,5 kernel
    block->getIArguments()->push_back(kH);
    block->getIArguments()->push_back(kW);

    // 1,1 stride
    block->getIArguments()->push_back(sH);
    block->getIArguments()->push_back(sW);

    // 0,0 padding
    block->getIArguments()->push_back(pH);
    block->getIArguments()->push_back(pW);

    // 1,1 dilation
    block->getIArguments()->push_back(dH);
    block->getIArguments()->push_back(dW);

    // same mode
    block->getIArguments()->push_back(1);

    // is NHWC
    block->getIArguments()->push_back(0);

    sd::ops::conv2d op;

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
    // exp->printIndexedBuffer("Expected");
    // res->printIndexedBuffer("Actual  ");
    // res->printShapeInfo("Result shape");
    // final check
    ASSERT_TRUE(res->equalsTo(exp));

    delete block;
    delete variableSpace;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv2d_2) {
    auto input = NDArrayFactory::create<TypeParam>('c', {1, 1, 1, 4});
    auto weights = NDArrayFactory::create<TypeParam>('c', {1, 1, 1, 4});
    auto exp = NDArrayFactory::create<TypeParam>('c', {1, 4, 1, 4}, {2.f, 4.f, 6.f, 8.f, 2.f, 4.f, 6.f, 8.f, 2.f, 4.f, 6.f, 8.f, 2.f, 4.f, 6.f, 8.f});

    weights.assign(2.0);
    input.linspace(1);

    sd::ops::conv2d op;
    auto result = op.evaluate({&input, &weights}, {}, {1, 1, 1, 1, 0, 0, 1, 1, 0, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv2d_3) {

    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=4,oW=3;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    auto input = NDArrayFactory::create<TypeParam>('c', {bS, iH, iW, iC});
    auto weights = NDArrayFactory::create<TypeParam>('c', {kH, kW, iC, oC});
    auto bias = NDArrayFactory::create<TypeParam>('c', {oC}, {1.f, 2.f, 3.f});


    auto expOutput = NDArrayFactory::create<TypeParam>('c', {bS, oH, oW, oC},{ 152.f,  155.2f,  158.4f, 152.f,  155.2f,  158.4f,  66.4f,   68.f,   69.6f, 170.4f,  175.2f,  180.f, 170.4f,  175.2f,  180.f,  70.8f,   73.2f,   75.6f,
                                                      170.4f,  175.2f,  180.f, 170.4f,  175.2f,  180.f,  70.8f,   73.2f,   75.6f,  75.2f,   78.4f,   81.6f,  75.2f,   78.4f,   81.6f,  28.f,   29.6f,   31.2f,
                                                      152.f,  155.2f,  158.4f, 152.f,  155.2f,  158.4f,  66.4f,   68.f,   69.6f, 170.4f,  175.2f,  180.f, 170.4f,  175.2f,  180.f,  70.8f,   73.2f,   75.6f,
                                                      170.4f,  175.2f,  180.f, 170.4f,  175.2f,  180.f,  70.8f,   73.2f,   75.6f,  75.2f,   78.4f,   81.6f,  75.2f,   78.4f,   81.6f,  28.f,   29.6f,   31.2f});
    input = 2.;
    weights.linspace(0.1, 0.1);

    sd::ops::conv2d op;
    auto results = op.evaluate({&input, &weights}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));


}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv2d_4) {

    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    auto input = NDArrayFactory::create<TypeParam>('c', {bS, iH, iW, iC});
    auto weights = NDArrayFactory::create<TypeParam>('c', {kH, kW, iC, oC});
    auto bias = NDArrayFactory::create<TypeParam>('c', {oC}, {1,2,3});

    auto expOutput = NDArrayFactory::create<TypeParam>('c', {bS, oH, oW, oC},{ 170.4f,175.20001f,180.f,170.4f,175.20001f,180.f,170.4f,175.20001f,180.f,170.4f,175.20001f,180.f,170.4f,175.20001f,180.f,170.4f,175.20001f,180.f,170.4f,175.20001f,180.f,170.4f,175.20001f,180.f});

    input = 2.;
    weights.linspace(0.1, 0.1);

    sd::ops::conv2d op;
    auto results = op.evaluate({&input, &weights}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv2d_5) {

    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iC, iH, iW});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {oC, iC, kH, kW});
    auto bias     = NDArrayFactory::create<TypeParam>('c', {oC}, {1,2,3});

    auto expOutput = NDArrayFactory::create<TypeParam>('c', {bS, oC, oH, oW}, {61.f,   61.f,  61.f,   61.f, 177.2f,  177.2f, 177.2f,  177.2f, 293.4f,  293.4f, 293.4f,  293.4f,  61.f,   61.f,  61.f,   61.f, 177.2f,  177.2f, 177.2f,  177.2f, 293.4f,  293.4f, 293.4f,  293.4f});

    input = 2.;
    weights.linspace(0.1, 0.1);
    weights.permutei({2,3,1,0});

    sd::ops::conv2d op;
    auto results = op.evaluate({&input, &weights, &bias}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto output = results.at(0);

    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv2d_6) {
    auto input = NDArrayFactory::create<TypeParam>('c', {54, 1, 12, 12});
    auto weights = NDArrayFactory::create<TypeParam>('c', {1, 2, 12, 2});

    sd::ops::conv2d op;
    auto result = op.evaluate({&input, &weights}, {}, {-1,-1,  1,1,  0,0,  1,1,  1,1});
    ASSERT_EQ(Status::OK(), result.status());
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv2d_7) {

    int bS=1, iH=256,iW=256,  iC=1,oC=1,  kH=4,kW=3,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    // int       oH=256,oW=256;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iC, iH, iW});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {kH, kW, iC, oC});

    input = 5.;
    weights = 3.;

    sd::ops::conv2d op;
    auto results = op.evaluate({&input, &weights}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());


}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv2d_8) {

    int bS=1, iH=6,iW=8,  iC=2,oC=2,  kH=2,kW=1,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=6,oW=8;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iC, iH, iW}, {0.679350, 0.355087, 0.842789, 0.200313, 0.701499, 0.310693, 0.447940, 0.938010, 0.326674, 0.151873, 0.383318, 0.782123, 0.198807,
        0.798564, 0.163263, 0.146968, 0.260897, 0.135058, 0.756209, 0.275454, 0.369088, 0.092826, 0.836492, 0.268413, 0.095062, 0.312795, 0.135918, 0.517544, 0.328703,
        0.061736, 0.396431, 0.248016, 0.548959, 0.115046, 0.814362, 0.721564, 0.404494, 0.299089, 0.403884, 0.988311, 0.022296, 0.927782, 0.318416, 0.068546, 0.284533,
        0.232720, 0.352142, 0.058909, 0.711221, 0.674457, 0.196946, 0.699497, 0.074322, 0.420425, 0.584263, 0.149574, 0.446406, 0.723072, 0.064481, 0.483078, 0.875996,
        0.569819, 0.445863, 0.527755, 0.016646, 0.753678, 0.140636, 0.754129, 0.161932, 0.775037, 0.332645, 0.117394, 0.017711, 0.608476, 0.525152, 0.917194, 0.849891,
        0.589423, 0.852278, 0.390636, 0.889683, 0.669445, 0.698873, 0.961480, 0.157401, 0.157364, 0.493520, 0.569937, 0.126832, 0.115728, 0.786368, 0.737939, 0.490079, 0.608414, 0.956500, 0.390098});

    NDArray weights('c', {kH, kW, iC, oC}, {0.07581716775894165, 0.8706002235412598, 0.29345420002937317, 0.5281786322593689, 0.10540834069252014, 0.3663792014122009, 0.17209206521511078, 0.6257694959640503});
    NDArray bias('c', {1, oC}, {0.7414038777351379, 0.8980839848518372});

    NDArray expOutput('c', {bS, oC, oH, oW}, {1.112878, 1.106691, 0.914598, 1.127438, 0.988108, 1.070572, 1.040759, 0.962728, 0.927537, 1.109045, 0.893301, 1.101278, 1.080314,
        1.112327, 1.030041, 0.955914, 0.779137, 1.110499, 0.944709, 1.195986, 0.997814, 1.083822, 1.090898, 0.889572, 0.964781, 1.071012, 1.111928, 1.291319, 1.085454, 0.977661,
        1.149068, 1.077099, 1.068283, 1.064290, 1.177125, 1.212480, 0.932593, 0.939493, 1.118576, 1.056927, 0.780314, 0.845707, 0.996308, 0.963152, 0.906792, 0.937590, 1.048791,
        0.860346, 2.264212, 2.071576, 1.916629, 2.030785, 2.169075, 2.039786, 1.935480, 2.177816, 1.524273, 1.933327, 1.630923, 2.406983, 1.770406, 2.413284, 1.790349, 1.476586,
        1.179925, 1.909109, 2.009143, 2.299778, 1.957207, 1.779718, 2.480604, 1.529086, 1.748063, 1.952856, 2.029487, 2.699131, 1.879842, 1.471205, 2.150177, 2.039078, 1.933456,
        1.764169, 2.584944, 2.521004, 1.744296, 1.707578, 2.237938, 2.325231, 0.984485, 1.766936, 1.590640, 1.347524, 1.404648, 1.422042, 1.709862, 1.155412});

    sd::ops::conv2d op;
    auto results = op.evaluate({&input, &weights, &bias}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto output = results.at(0);

    // output->printBuffer();

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv2d_9) {

    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW
    int wFormat     = 1;             // 0-[kH, kW, iC, oC], 1-[oC, iC, kH, kW], 2-[oC, kH, kW, iC]

    NDArray input('c', {bS, iC, iH, iW}, sd::DataType::FLOAT32);
    NDArray weights('c', {oC, iC, kH, kW}, {-3., -1.8, -0.6, 0.6, 1.8, 3., -2.7, -1.5, -0.3, 0.9, 2.1, 3.3, -2.4, -1.2, 0., 1.2, 2.4, 3.6, -2.1, -0.9, 0.3, 1.5,
                                2.7, 3.9, -2.9, -1.7, -0.5, 0.7, 1.9, 3.1, -2.6, -1.4, -0.2, 1., 2.2, 3.4, -2.3, -1.1, 0.1, 1.3, 2.5, 3.7, -2., -0.8, 0.4, 1.6,
                                2.8, 4., -2.8, -1.6, -0.4, 0.8, 2., 3.2, -2.5, -1.3, -0.1, 1.1, 2.3, 3.5, -2.2, -1., 0.2, 1.4, 2.6, 3.8, -1.9, -0.7, 0.5, 1.7, 2.9, 4.1}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, {-1,2,0.5}, sd::DataType::FLOAT32);

    NDArray expOutput('c', {bS, oC, oH, oW}, {37.699997, 32.300041, 21.499989, 16.100004, 74.900024, 68.300003, 55.100006, 48.499969, 107.599983, 99.799988,
                                            84.200005, 76.400009, -221.5, -226.899994, -237.699997, -243.099991, -241.899994, -248.5, -261.700012, -268.299988,
                                            -266.799988, -274.600006, -290.200012, -298.}, sd::DataType::FLOAT32);

    input.linspace(25,-0.5);

    sd::ops::conv2d op;
    auto results = op.evaluate({&input, &weights, &bias}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat, wFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv2d_10) {

    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=4,oW=3;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW
    int wFormat     = 2;             // 0-[kH, kW, iC, oC], 1-[oC, iC, kH, kW], 2-[oC, kH, kW, iC]

    NDArray input('c', {bS, iH, iW, iC}, sd::DataType::FLOAT32);
    NDArray weights('c', {oC, kH, kW, iC}, {-3., -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3., 3.3,
                                            3.6, 3.9, -2.9, -2.6, -2.3, -2., -1.7, -1.4, -1.1, -0.8, -0.5, -0.2, 0.1, 0.4, 0.7, 1., 1.3, 1.6, 1.9, 2.2, 2.5, 2.8,
                                            3.1, 3.4, 3.7, 4., -2.8, -2.5, -2.2, -1.9, -1.6, -1.3, -1., -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2., 2.3, 2.6,
                                            2.9, 3.2, 3.5, 3.8, 4.1}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, {-1,2,0.5}, sd::DataType::FLOAT32);

    NDArray expOutput('c', {bS, oH, oW, oC}, {463.400055, 498.800018, 529.700012, 410.600006, 442.799988, 470.500031, 113.600006, 130.400009, 142.699982,
        -63.999958, -19.600082, 20.300007, -85.600052, -45.999939, -10.899940, -144.100021, -124., -108.399994, -128.799988, -98.799973, -73.300011,
        -150.400009, -125.200012, -104.500008, -133.300003, -120.399994, -112.000008, -170.199997, -154., -142.299988, -146.200012, -133.199997, -124.699997,
        -88.000008, -80.800003, -78.099991, -170.200012, -173.199997, -180.699982, -223., -229.199997, -239.900009, -88., -90.400002, -97.300003, -323.200012,
        -336.399994, -354.100037, -344.800018, -362.799988, -385.299957, -100.900002, -109.600006, -122.800003, -388.000031, -415.599976, -447.700012, -409.599976,
        -442., -478.900024, -90.099991, -105.999992, -126.399994, 117.800003, 95.599991, 68.899994, 141.799988, 116.399994, 86.5, 171.200012, 159.200012, 142.699997}, sd::DataType::FLOAT32);

    input.linspace(25,-0.5);

    sd::ops::conv2d op;
    auto results = op.evaluate({&input, &weights, &bias}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat, wFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, sconv2d_1) {
    float _expB[] = {10025.0f,    10350.0f,    10675.0f,    11000.0f,    11325.0f,    11650.0f,    13275.0f,    13600.0f,    13925.0f,    14250.0f,    14575.0f,    14900.0f,    16525.0f,    16850.0f,    17175.0f,    17500.0f,    17825.0f,    18150.0f,    19775.0f,    20100.0f,    20425.0f,    20750.0f,    21075.0f,    21400.0f,    23025.0f,    23350.0f,    23675.0f,    24000.0f,    24325.0f,    24650.0f,    26275.0f,    26600.0f,    26925.0f,    27250.0f,    27575.0f,    27900.0f,    38775.0f,    40350.0f,    41925.0f,    43500.0f,    45075.0f,    46650.0f,    54525.0f,    56100.0f,    57675.0f,    59250.0f,    60825.0f,    62400.0f,    70275.0f,    71850.0f,    73425.0f,    75000.0f,    76575.0f,    78150.0f,    86025.0f,    87600.0f,    89175.0f,    90750.0f,    92325.0f,    93900.0f,   101775.0f,   103350.0f,   104925.0f,    106500.0f,   108075.0f,   109650.0f,   117525.0f,   119100.0f,   120675.0f,   122250.0f,    123825.0f,   125400.0f,    67525.0f,    70350.0f,    73175.0f,    76000.0f,    78825.0f,    81650.0f,    95775.0f,    98600.0f,   101425.0f,   104250.0f,   107075.0f,   109900.0f,    124025.0f,   126850.0f,   129675.0f,   132500.0f,   135325.0f,   138150.0f,   152275.0f,    155100.0f,   157925.0f,   160750.0f,   163575.0f,   166400.0f,   180525.0f,   183350.0f,    186175.0f,   189000.0f,   191825.0f,   194650.0f,   208775.0f,   211600.0f,   214425.0f,    217250.0f,   220075.0f,   222900.0f,   119400.0f,   120350.0f,   121300.0f,   122250.0f,    123200.0f,   124150.0f,   128900.0f,   129850.0f,   130800.0f,   131750.0f,   132700.0f,    133650.0f,   138400.0f,   139350.0f,   140300.0f,   141250.0f,   142200.0f,   143150.0f,    147900.0f,   148850.0f,   149800.0f,   150750.0f,   151700.0f,   152650.0f,   157400.0f,    158350.0f,   159300.0f,   160250.0f,   161200.0f,   162150.0f,   166900.0f,   167850.0f,    168800.0f,   169750.0f,   170700.0f,   171650.0f,   273150.0f,   275350.0f,   277550.0f,    279750.0f,   281950.0f,   284150.0f,   295150.0f,   297350.0f,   299550.0f,   301750.0f,    303950.0f,   306150.0f,   317150.0f,   319350.0f,   321550.0f,   323750.0f,   325950.0f,    328150.0f,   339150.0f,   341350.0f,   343550.0f,   345750.0f,   347950.0f,   350150.0f,    361150.0f,   363350.0f,   365550.0f,   367750.0f,   369950.0f,   372150.0f,   383150.0f,    385350.0f,   387550.0f,   389750.0f,   391950.0f,   394150.0f,   426900.0f,   430350.0f,    433800.0f,   437250.0f,   440700.0f,   444150.0f,   461400.0f,   464850.0f,   468300.0f,    471750.0f,   475200.0f,   478650.0f,   495900.0f,   499350.0f,   502800.0f,   506250.0f,    509700.0f,   513150.0f,   530400.0f,   533850.0f,   537300.0f,   540750.0f,   544200.0f,    547650.0f,   564900.0f,   568350.0f,   571800.0f,   575250.0f,   578700.0f,   582150.0f,    599400.0f,   602850.0f,   606300.0f,   609750.0f,   613200.0f,   616650.0f,    75025.0f,    75350.0f,    75675.0f,    76000.0f,    76325.0f,    76650.0f,    78275.0f,    78600.0f,    78925.0f,    79250.0f,    79575.0f,    79900.0f,    81525.0f,    81850.0f,    82175.0f,    82500.0f,    82825.0f,    83150.0f,    84775.0f,    85100.0f,    85425.0f,    85750.0f,    86075.0f,    86400.0f,    88025.0f,    88350.0f,    88675.0f,    89000.0f,    89325.0f,    89650.0f,    91275.0f,    91600.0f,    91925.0f,    92250.0f,    92575.0f,    92900.0f,    353775.0f,   355350.0f,   356925.0f,   358500.0f,   360075.0f,   361650.0f,   369525.0f,    371100.0f,   372675.0f,   374250.0f,   375825.0f,   377400.0f,   385275.0f,   386850.0f,    388425.0f,   390000.0f,   391575.0f,   393150.0f,   401025.0f,   402600.0f,   404175.0f,    405750.0f,   407325.0f,   408900.0f,   416775.0f,   418350.0f,   419925.0f,   421500.0f,    423075.0f,   424650.0f,   432525.0f,   434100.0f,   435675.0f,   437250.0f,   438825.0f,    440400.0f,   632525.0f,   635350.0f,   638175.0f,   641000.0f,   643825.0f,   646650.0f,    660775.0f,   663600.0f,   666425.0f,   669250.0f,   672075.0f,   674900.0f,   689025.0f,    691850.0f,   694675.0f,   697500.0f,   700325.0f,   703150.0f,   717275.0f,   720100.0f,    722925.0f,   725750.0f,   728575.0f,   731400.0f,   745525.0f,   748350.0f,   751175.0f,    754000.0f,   756825.0f,   759650.0f,   773775.0f,   776600.0f,   779425.0f,   782250.0f,    785075.0f,   787900.0f,   309400.0f,   310350.0f,   311300.0f,   312250.0f,   313200.0f,    314150.0f,   318900.0f,   319850.0f,   320800.0f,   321750.0f,   322700.0f,   323650.0f,    328400.0f,   329350.0f,   330300.0f,   331250.0f,   332200.0f,   333150.0f,   337900.0f,    338850.0f,   339800.0f,   340750.0f,   341700.0f,   342650.0f,   347400.0f,   348350.0f,    349300.0f,   350250.0f,   351200.0f,   352150.0f,   356900.0f,   357850.0f,   358800.0f,    359750.0f,   360700.0f,   361650.0f,   713150.0f,   715350.0f,   717550.0f,   719750.0f,    721950.0f,   724150.0f,   735150.0f,   737350.0f,   739550.0f,   741750.0f,   743950.0f,    746150.0f,   757150.0f,   759350.0f,   761550.0f,   763750.0f,   765950.0f,   768150.0f,    779150.0f,   781350.0f,   783550.0f,   785750.0f,   787950.0f,   790150.0f,   801150.0f,    803350.0f,   805550.0f,   807750.0f,   809950.0f,   812150.0f,   823150.0f,   825350.0f,    827550.0f,   829750.0f,   831950.0f,   834150.0f,  1116900.0f,  1120350.0f,  1123800.0f,    1127250.0f,  1130700.0f,  1134150.0f,  1151400.0f,  1154850.0f,  1158300.0f,  1161750.0f,    1165200.0f,  1168650.0f,  1185900.0f,  1189350.0f,  1192800.0f,  1196250.0f,  1199700.0f,    1203150.0f,  1220400.0f,  1223850.0f,  1227300.0f,  1230750.0f,  1234200.0f,  1237650.0f,    1254900.0f,  1258350.0f,  1261800.0f,  1265250.0f,  1268700.0f,  1272150.0f,  1289400.0f,    1292850.0f,  1296300.0f,  1299750.0f,  1303200.0f,  1306650.0f,};
    Nd4jLong _expS[] = {4, 2, 6, 6, 6, 144, 36, 6, 1, 8192, 1, 99};
    NDArray exp(_expB, _expS);

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

    auto input = NDArrayFactory::create_<float>('c', {B, iC, iY, iX});
    for (int e = 0; e < input->lengthOf(); e++)
        input->p(e, e+1);

    auto weights = NDArrayFactory::create_<float>('c', {oC, iC, kY, kX});
    for (int e = 0; e < weights->lengthOf(); e++)
        weights->p(e, e+1);
    weights->permutei({2,3,1,0});

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, weights);

    auto block = new Context(1, variableSpace, false);
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

    sd::ops::sconv2d op;

    Nd4jStatus status = op.execute(block);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    auto output = variableSpace->getVariable(1)->getNDArray();

    //exp.printShapeInfo("Expected shape");
    //output->printShapeInfo("Result shape");
    ASSERT_TRUE(exp.isSameShape(output));

        //exp.printBuffer("Expctd buffer");
    //output->printBuffer("Result buffer");
    ASSERT_TRUE(exp.equalsTo(output));

    delete block;
    delete variableSpace;
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, sconv2d_2) {
    TypeParam _expBFF[] = {108.9405008f,  109.5920008f,  110.2435008f,  110.8950008f,   111.5465008f,  112.1980008f,  115.4555008f,  116.1070008f,   116.7585008f,  117.410000f,   118.061500f,   118.7130009f,   121.9705009f,  122.6220009f,  123.2735009f,  123.9250009f,   124.5765009f,  125.2280009f,  128.4855009f,  129.1370009f,   129.7885009f,  130.4400009f,  131.09150f,    131.74300f,   135.0005010f,  135.6520010f,  136.3035010f,  136.9550010f,   137.6065010f,  138.2580010f,  141.5155010f,  142.1670010f,   142.8185010f,  143.4700010f,  144.1215010f,  144.7730010f,   248.9617514f,  250.670751f,   252.3797515f,  254.0887515f,   255.7977515f,  257.5067515f,  266.0517515f,  267.7607515f,   269.469751f,   271.1787516f,  272.8877516f,  274.5967516f,   283.1417516f,  284.8507516f,
                           286.5597516f,  288.268751f,   289.9777517f,  291.6867517f,  300.2317517f,  301.9407517f,   303.6497517f,  305.3587517f,  307.067751f,   308.7767518f,   317.3217518f,  319.0307518f,  320.7397518f,  322.4487518f,   324.157751f,   325.866751f,   334.4117519f,  336.1207519f,   337.8297519f,  339.5387519f,  341.2477519f,  342.95675f,   388.9829964f,  391.7494964f,  394.5159964f,  397.2824964f,   400.048996f,   402.8154963f,  416.647996f,   419.4144962f,   422.1809962f,  424.9474962f,  427.7139962f,  430.4804962f,   444.3129961f,  447.0794961f,  449.8459961f,  452.6124960f,   455.3789960f,  458.1454960f,  471.9779959f,  474.7444959f,   477.5109959f,  480.2774959f,  483.0439959f,  485.8104958f,   499.6429958f,  502.4094957f,  505.1759957f,  507.9424957f,
                           510.7089957f,  513.4754957f,  527.3079956f,  530.0744956f,   532.8409956f,  535.607495f,   538.3739955f,  541.1404955f,   529.0042487f,  532.8282487f,  536.6522487f,  540.4762487f,   544.3002487f,  548.1242487f,  567.2442487f,  571.068248f,   574.892248f,   578.716248f,   582.540248f,   586.3642486f,   605.4842486f,  609.3082486f,  613.1322486f,  616.9562486f,   620.7802486f,  624.6042486f,  643.7242486f,  647.5482486f,   651.3722486f,  655.1962486f,  659.0202486f,  662.8442486f,   681.9642486f,  685.7882486f,  689.6122486f,  693.4362486f,   697.2602486f,  701.0842486f,  720.2042486f,  724.0282486f,   727.852248f,   731.676248f,   735.500248f,   739.324248f,   669.0255044f,  673.9070044f,  678.7885044f,  683.6700044f,   688.5515044f,  693.4330044f,
                           717.8405044f,  722.7220044f,   727.6035044f,  732.4850044f,  737.3665044f,  742.2480044f,   766.6555043f,  771.5370043f,  776.4185043f,  781.3000043f,   786.1815043f,  791.0630043f,  815.4705043f,  820.3520043f,   825.2335043f,  830.1150043f,  834.9965043f,  839.8780043f,   864.2855042f,  869.1670042f,  874.0485042f,  878.9300042f,   883.8115042f,  888.6930042f,  913.1005042f,  917.9820042f,   922.8635042f,  927.7450042f,  932.6265042f,  937.5080042f,   809.0467424f,  814.9857424f,  820.9247424f,  826.8637423f,   832.8027423f,  838.7417423f,  868.4367421f,  874.3757421f,   880.3147420f,  886.2537420f,  892.1927420f,  898.13174f,   927.8267418f,  933.7657418f,  939.7047417f,  945.6437417f,   951.5827417f,  957.5217416f,  987.2167415f,  993.155741f,
                           999.0947414f, 1005.0337414f, 1010.972741f,  1016.9117413f,   1046.6067412f, 1052.5457411f, 1058.4847411f, 1064.4237411f,   1070.3627410f, 1076.3017410f, 1105.996740f,  1111.9357408f,   1117.8747408f, 1123.8137408f, 1129.7527407f, 1135.6917407f,   949.0679815f,  956.0644814f,  963.060981f,   970.0574813f,   977.0539812f,  984.0504811f, 1019.0329807f, 1026.0294807f,   1033.0259806f, 1040.0224805f, 1047.0189804f, 1054.0154804f,   1088.9979800f, 1095.9944799f, 1102.9909798f, 1109.987479f,   1116.9839797f, 1123.9804796f, 1158.9629792f, 1165.9594791f,   1172.9559791f, 1179.9524790f, 1186.9489789f, 1193.9454788f,   1228.9279785f, 1235.9244784f, 1242.9209783f, 1249.9174782f,   1256.913978f,  1263.9104781f, 1298.8929777f, 1305.8894776f,   1312.8859775f, 1319.8824775f, 1326.8789774f, 1333.8754773f,   1089.0892560f, 1097.1432561f, 1105.1972562f, 1113.251256f,   1121.3052563f, 1129.3592564f, 1169.6292568f, 1177.6832568f,   1185.7372569f, 1193.7912570f, 1201.845257f,  1209.8992571f,   1250.1692575f, 1258.2232576f, 1266.2772576f, 1274.3312577f,   1282.3852578f, 1290.4392579f, 1330.7092582f, 1338.7632583f,   1346.8172584f, 1354.8712584f, 1362.9252585f, 1370.9792586f,   1411.24925f,   1419.3032590f, 1427.3572591f, 1435.4112592f,   1443.465259f,  1451.5192593f, 1491.7892597f, 1499.8432598f,   1507.8972598f, 1515.9512599f, 1524.0052600f, 1532.059260f,   1229.1105073f, 1238.2220073f, 1247.3335073f, 1256.4450073f,   1265.5565073f, 1274.668007f,  1320.2255074f, 1329.3370074f,   1338.4485074f, 1347.5600075f, 1356.6715075f, 1365.7830075f,   1411.340507f,  1420.4520076f, 1429.5635076f, 1438.6750076f,   1447.7865076f, 1456.8980076f, 1502.4555077f, 1511.5670077f,   1520.6785077f, 1529.7900077f, 1538.9015077f, 1548.013007f,   1593.5705078f, 1602.6820078f, 1611.793507f,  1620.9050079f,   1630.0165079f, 1639.1280079f, 1684.6855080f, 1693.7970080f,   1702.9085080f, 1712.0200080f, 1721.1315080f, 1730.2430080f,   1369.1317613f, 1379.3007614f, 1389.4697614f, 1399.6387615f,   1409.8077615f, 1419.976761f,  1470.8217618f, 1480.9907618f,   1491.159761f,  1501.3287619f, 1511.4977619f, 1521.6667620f,   1572.5117622f, 1582.6807622f, 1592.8497623f, 1603.0187623f,   1613.1877624f, 1623.3567624f, 1674.2017626f, 1684.3707627f,   1694.5397627f, 1704.7087628f, 1714.8777628f, 1725.046762f,   1775.8917631f, 1786.0607631f, 1796.229763f,  1806.3987632f,   1816.5677632f, 1826.7367633f, 1877.5817635f, 1887.7507635f,   1897.9197636f, 1908.0887636f, 1918.2577637f, 1928.4267637f,   304.3905022f,  305.0420022f,  305.6935022f,  306.3450022f,   306.9965022f,  307.6480022f,  310.9055022f,  311.5570022f,   312.208502f,   312.860002f,   313.5115023f,  314.1630023f,   317.4205023f,  318.0720023f,  318.7235023f,  319.3750023f,   320.0265023f,  320.6780023f,  323.9355023f,  324.5870023f,   325.2385023f,  325.8900023f,  326.541502f,   327.193002f,   330.4505024f,  331.1020024f,  331.7535024f,  332.4050024f,   333.0565024f,  333.7080024f,  336.9655024f,  337.6170024f,   338.2685024f,  338.9200024f,  339.5715024f,  340.223002f,   761.6617542f,  763.3707542f,  765.0797542f,  766.7887542f,   768.4977542f,  770.206754f,   778.7517543f,  780.4607543f,   782.1697543f,  783.8787543f,  785.5877543f,  787.2967543f,   795.8417544f,  797.5507544f,  799.2597544f,  800.9687544f,   802.6777544f,  804.3867544f,  812.9317545f,  814.6407545f,   816.3497545f,  818.0587545f,  819.7677545f,  821.4767545f,   830.0217546f,  831.7307546f,  833.4397546f,  835.1487546f,   836.8577546f,  838.5667546f,  847.1117547f,  848.8207547f,   850.5297547f,  852.2387547f,  853.9477547f,  855.6567547f,   1218.9329915f, 1221.6994915f, 1224.4659915f, 1227.232491f,   1229.9989914f, 1232.7654914f, 1246.5979913f, 1249.3644913f,   1252.1309913f, 1254.8974913f, 1257.6639913f, 1260.430491f,   1274.2629912f, 1277.029491f,  1279.7959911f, 1282.5624911f,   1285.3289911f, 1288.0954911f, 1301.9279910f, 1304.6944910f,   1307.4609910f, 1310.22749f,   1312.9939909f, 1315.7604909f,   1329.5929908f, 1332.3594908f, 1335.1259908f, 1337.8924908f,   1340.6589908f, 1343.4254908f, 1357.2579907f,
                           1360.0244907f,   1362.7909906f, 1365.5574906f, 1368.3239906f, 1371.0904906f,   1676.2042479f, 1680.0282479f, 1683.8522479f, 1687.6762479f,   1691.5002479f, 1695.3242479f, 1714.4442479f, 1718.2682479f,   1722.0922479f, 1725.9162479f, 1729.7402479f, 1733.5642479f,   1752.6842479f, 1756.5082479f, 1760.3322479f, 1764.1562479f,   1767.9802479f, 1771.8042479f, 1790.9242479f, 1794.7482479f,   1798.5722479f, 1802.3962479f, 1806.2202479f, 1810.044247f,   1829.1642478f, 1832.9882478f, 1836.8122478f, 1840.6362478f,   1844.4602478f, 1848.2842478f, 1867.4042478f, 1871.2282478f,   1875.0522478f, 1878.8762478f, 1882.7002478f, 1886.5242478f,   2133.4755029f, 2138.3570029f, 2143.2385029f, 2148.1200029f,   2153.0015029f, 2157.8830029f, 2182.2905028f, 2187.1720028f,   2192.0535028f, 2196.9350028f, 2201.8165028f, 2206.6980028f,   2231.1055028f, 2235.9870028f, 2240.8685028f, 2245.7500028f,   2250.6315028f, 2255.5130028f, 2279.9205027f, 2284.8020027f,   2289.6835027f, 2294.5650027f, 2299.4465027f, 2304.3280027f,   2328.7355027f, 2333.6170027f, 2338.4985027f, 2343.3800027f,   2348.2615027f, 2353.1430027f, 2377.5505026f, 2382.4320026f,   2387.3135026f, 2392.1950026f, 2397.0765026f, 2401.9580026f,   2590.7467330f, 2596.6857330f, 2602.6247329f, 2608.5637329f,   2614.5027329f, 2620.441732f,  2650.1367327f, 2656.0757327f,   2662.0147326f, 2667.9537326f, 2673.8927326f, 2679.8317325f,   2709.5267324f, 2715.465732f,  2721.4047323f, 2727.3437323f,   2733.282732f,  2739.2217322f, 2768.9167321f, 2774.8557320f,   2780.7947320f, 2786.7337320f, 2792.6727319f, 2798.6117319f,   2828.306731f,  2834.2457317f, 2840.1847317f, 2846.1237317f,   2852.0627316f, 2858.0017316f, 2887.6967314f, 2893.6357314f,   2899.5747314f, 2905.5137313f, 2911.4527313f, 2917.3917313f,   3048.0179587f, 3055.0144586f, 3062.0109585f, 3069.0074584f,   3076.0039584f, 3083.0004583f, 3117.9829579f, 3124.9794578f,   3131.9759578f, 3138.9724577f, 3145.9689576f, 3152.9654575f,   3187.947957f,  3194.9444571f, 3201.9409570f, 3208.9374569f,   3215.933956f,  3222.9304568f, 3257.9129564f, 3264.9094563f,   3271.9059562f, 3278.9024562f, 3285.8989561f,
                           3292.8954560f,   3327.8779556f, 3334.874455f,  3341.8709555f, 3348.8674554f,   3355.8639553f, 3362.860455f,  3397.8429549f, 3404.8394548f,   3411.8359547f, 3418.8324546f, 3425.8289546f, 3432.8254545f,   3505.28927f,   3513.3432780f, 3521.3972781f, 3529.4512782f,   3537.5052782f, 3545.5592783f, 3585.8292787f, 3593.8832788f,   3601.9372788f, 3609.9912789f, 3618.0452790f, 3626.099279f,
                           3666.3692794f, 3674.4232795f, 3682.4772796f, 3690.5312796f,   3698.5852797f, 3706.6392798f, 3746.9092801f, 3754.9632802f,   3763.0172803f, 3771.0712804f, 3779.1252804f, 3787.1792805f,   3827.4492809f, 3835.50328f,   3843.5572810f, 3851.6112811f,   3859.6652812f, 3867.7192812f, 3907.9892816f, 3916.0432817f,   3924.097281f,
                           3932.1512818f, 3940.2052819f, 3948.2592820f,   3962.5605113f, 3971.6720113f, 3980.783511f,  3989.8950114f,   3999.0065114f, 4008.1180114f, 4053.6755115f, 4062.7870115f,   4071.8985115f, 4081.0100115f, 4090.1215115f, 4099.2330115f,   4144.7905116f, 4153.9020116f, 4163.0135116f, 4172.1250116f,
                           4181.236511f,  4190.3480117f, 4235.9055117f, 4245.0170117f,   4254.128511f,  4263.2400118f, 4272.3515118f, 4281.4630118f,   4327.0205119f, 4336.1320119f, 4345.2435119f, 4354.3550119f,   4363.4665119f, 4372.5780119f, 4418.1355120f, 4427.2470120f,   4436.3585120f, 4445.4700120f, 4454.581512f,  4463.6930121f,   4419.8317743f, 4430.0007744f, 4440.1697744f, 4450.338774f,   4460.5077745f, 4470.6767745f, 4521.521774f,  4531.6907748f,
                           4541.8597748f, 4552.0287749f, 4562.1977749f, 4572.3667750f,   4623.2117752f, 4633.3807752f, 4643.5497753f, 4653.7187753f,   4663.8877754f, 4674.0567754f, 4724.9017756f, 4735.0707757f,   4745.2397757f, 4755.4087757f, 4765.5777758f, 4775.7467758f,   4826.591776f,  4836.7607761f, 4846.9297761f, 4857.0987762f,   4867.2677762f, 4877.4367763f, 4928.2817765f, 4938.4507765f,   4948.6197766f, 4958.7887766f, 4968.957776f,  4979.12677675f};
    Nd4jLong _expSFF[] = {4, 2, 10, 6, 6, 360, 36, 6, 1, typeid(TypeParam) == typeid(float) ? 8192 : 16384, 1, 99,};
    NDArray expFF(_expBFF, _expSFF);

    auto input = NDArrayFactory::create<TypeParam>('c', {2, 3, 10, 10});
    auto weightsD = NDArrayFactory::create<TypeParam>('c', {5, 3, 5, 5});
    auto weightsP = NDArrayFactory::create<TypeParam>('c', {10, 15, 1, 1});

    input.linspace(1);
    weightsD.linspace(1);
    weightsP.linspace(1);
    weightsD.permutei({2,3,1,0});
    weightsP.permutei({2,3,1,0});

    input.applyScalar(scalar::Divide, 100.0, input);
    weightsD.applyScalar(scalar::Divide, 100.0, weightsD);
    weightsP.applyScalar(scalar::Divide, 100.0, weightsP);

    sd::ops::sconv2d op;

    auto resultFF = op.evaluate({&input, &weightsD, &weightsP},  {5, 5, 1, 1, 0, 0, 1, 1, 0, 0});

    auto z = resultFF.at(0);
    //z->printShapeInfo("FF shape");


    ASSERT_TRUE(z->isSameShape(&expFF));

    //expFF.printBuffer("e");
    //z->printBuffer("z");
    ASSERT_TRUE(z->equalsTo(&expFF, 1e-3));
}

TYPED_TEST(TypedConvolutionTests1, sconv2d_3) {
    auto input = NDArrayFactory::create<TypeParam>('c', {3, 3, 8, 8});
    auto weightsD = NDArrayFactory::create<TypeParam>('c', {1, 3, 1, 1});
    auto weightsP = NDArrayFactory::create<TypeParam>('c', {2, 3, 1, 1});
    auto bias = NDArrayFactory::create<TypeParam>('c', {2});
    auto output = NDArrayFactory::create<TypeParam>('c', {3, 2, 8, 8});
    output.assign(0.0);

    input.linspace(1);
    weightsD.linspace(1);
    weightsP.linspace(1);
    bias.linspace(1);
    weightsD.permutei({2,3,1,0});
    weightsP.permutei({2,3,1,0});

    auto expOutput = NDArrayFactory::create<TypeParam>('c', {3, 2, 8, 8});

    sd::ops::sconv2d op;
    Nd4jStatus status = op.execute({&input, &weightsD, &weightsP, &bias}, {&output}, {1, 1, 1, 1, 0, 0, 1, 1, 0});
    auto result = op.evaluate({&input, &weightsD, &weightsP, &bias}, {1, 1, 1, 1, 0, 0, 1, 1, 0});

    auto z = result.at(0);

    //printf("\n");
    //output.printBuffer("output");
    //z->printBuffer("z");


    //ASSERT_TRUE(expOutput.isSameShape(z));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, sconv2d_4) {

    int bS=1, iH=6,iW=6,  iC=3,oC=2,mC=3,  kH=1,kW=1,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=6,oW=6;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iC, iH, iW}, {0.679350, 0.355087, 0.842789, 0.200313, 0.701499, 0.310693, 0.447940, 0.938010, 0.326674, 0.151873, 0.383318, 0.782123, 0.198807,
        0.798564, 0.163263, 0.146968, 0.260897, 0.135058, 0.756209, 0.275454, 0.369088, 0.092826, 0.836492, 0.268413, 0.095062, 0.312795, 0.135918, 0.517544, 0.328703,
        0.061736, 0.396431, 0.248016, 0.548959, 0.115046, 0.814362, 0.721564, 0.404494, 0.299089, 0.403884, 0.988311, 0.022296, 0.927782, 0.318416, 0.068546, 0.284533,
        0.232720, 0.352142, 0.058909, 0.711221, 0.674457, 0.196946, 0.699497, 0.074322, 0.420425, 0.584263, 0.149574, 0.446406, 0.723072, 0.064481, 0.483078, 0.875996,
        0.569819, 0.445863, 0.527755, 0.016646, 0.753678, 0.140636, 0.754129, 0.161932, 0.775037, 0.332645, 0.117394, 0.017711, 0.608476, 0.525152, 0.917194, 0.849891,
        0.589423, 0.852278, 0.390636, 0.889683, 0.669445, 0.698873, 0.961480, 0.157401, 0.157364, 0.493520, 0.569937, 0.126832, 0.115728, 0.786368, 0.737939, 0.490079,
        0.608414, 0.956500, 0.390098, 0.147305, 0.850645, 0.497650, 0.071866, 0.082150, 0.035314, 0.732041, 0.369934, 0.840666, 0.273894, 0.431796, 0.133231});
    NDArray weightsD('c', {kH, kW, iC, mC}, {0.5340641736984253, 0.8257383108139038, 0.3279532492160797, 0.27217748761177063, 0.05432872101664543, 0.31322699785232544, 0.6599581837654114, 0.35526034235954285, 0.5765137672424316});
    NDArray weightsP('c', {1, 1, iC*mC, oC}, {0.4442146420478821, 0.3362849950790405, 0.5215804576873779, 0.5305071473121643, 0.7323054075241089, 0.5168435573577881, 0.8601323962211609, 0.2587810158729553, 0.9473239779472351, 0.39540114998817444, 0.04835261031985283, 0.8724213242530823, 0.8607604503631592, 0.8382210731506348, 0.8573186993598938, 0.6496091485023499, 0.8864102959632874, 0.14267340302467346});
    NDArray biases('c', {1,oC}, {0.8807470202445984, 0.6262521147727966});

    NDArray expOutput('c', {bS, oC, oH, oW}, {1.643804, 2.135067, 2.494167, 2.628944, 2.700440, 2.257452, 2.562539, 2.293667, 2.493985, 2.014933, 2.301736, 2.939066, 1.492952,
        2.026476, 1.771098, 2.013162, 1.315507, 1.289951, 2.831223, 2.196924, 2.028261, 2.024326, 2.983223, 1.809527, 1.434322, 2.513157, 1.826834, 1.608869, 1.297912, 1.212318,
        2.295934, 1.844615, 2.591148, 1.597267, 2.317755, 1.755642, 1.324064, 1.542060, 1.892052, 1.939339, 1.922781, 1.720199, 1.833396, 1.728024, 1.757968, 1.410675, 1.661960,
        2.096277, 1.178815, 1.637460, 1.254187, 1.491076, 0.968625, 0.986342, 2.116042, 1.536920, 1.504321, 1.490398, 2.136795, 1.351860, 1.148578, 1.817408, 1.327139, 1.288620,
        0.962232, 0.980667, 1.623775, 1.417320, 1.845710, 1.237095, 1.762792, 1.352515});

    sd::ops::sconv2d op;
    auto results = op.evaluate({&input, &weightsD, &weightsP, &biases}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

TYPED_TEST(TypedConvolutionTests1, conv2D_BP_Bias_1) {
    TypeParam _expWGradB[] = {9312.0, 12580.0, 9528.0, 13168.0, 17712.0, 13360.0, 9960.0, 13348.0, 10032.0, 13344.0, 18148.0, 13848.0, 19312.0, 26160.0, 19888.0, 15144.0, 20452.0, 15504.0};
    Nd4jLong _expWGradS[] = {4, 2, 1, 3, 3, 9, 9, 3, 1, typeid(TypeParam) == typeid(float) ? 8192 : 16384, 1, 99};
    NDArray expWGrad(_expWGradB, _expWGradS);
    expWGrad.permutei({2,3,1,0});

    TypeParam _expBGradB[] = {784.0, 1296.0};
    Nd4jLong _expBGradS[] = {2, 2, 1, 1, 1, typeid(TypeParam) == typeid(float) ? 8192 : 16384, 1, 99};

    NDArray expBGrad(_expBGradB, _expBGradS);

    auto input = NDArrayFactory::create<TypeParam>('c', {2, 1, 4, 4});
    auto weights = NDArrayFactory::create<TypeParam>('c', {2, 1, 3, 3});
    auto bias = NDArrayFactory::create<TypeParam>('c', {2, 1});
    auto epsilonNext = NDArrayFactory::create<TypeParam>('c', {2, 2, 4, 4});


    TypeParam _expEpsB[] = {952.0, 1540.0, 1636.0, 1180.0, 1791.0, 2886.0, 3057.0, 2193.0, 2223.0, 3570.0, 3741.0, 2673.0, 1900.0, 3028.0, 3160.0, 2240.0, 2872.0, 4612.0, 4708.0, 3356.0, 5247.0, 8358.0, 8529.0, 6033.0, 5679.0, 9042.0, 9213.0, 6513.0, 4588.0, 7252.0, 7384.0, 5184.0};
    NDArray expEps(_expEpsB, input.shapeInfo());

    input.linspace(1);
    weights.linspace(1);
    epsilonNext.linspace(1);
    weights.permutei({2,3,1,0});

    sd::ops::conv2d_bp op;

    auto results = op.evaluate({&input, &weights, &bias, &epsilonNext}, {},  {3, 3, 1, 1, 0, 0, 1, 1, 1}, {});

    ASSERT_TRUE(results.size() == 3);

    auto epsilon = results.at(0);
    auto gradW = results.at(1);
    auto gradB = results.at(2);

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


TYPED_TEST(TypedConvolutionTests1, conv2D_BP_NoBias_1) {
    TypeParam _expWGradB[] = {9312.0, 12580.0, 9528.0, 13168.0, 17712.0, 13360.0, 9960.0, 13348.0, 10032.0, 13344.0, 18148.0, 13848.0, 19312.0, 26160.0, 19888.0, 15144.0, 20452.0, 15504.0};
    Nd4jLong _expWGradS[] = {4, 2, 1, 3, 3, 9, 9, 3, 1, typeid(TypeParam) == typeid(float) ? 8192 : 16384, 1, 99};
    NDArray expWGrad(_expWGradB, _expWGradS);
    expWGrad.permutei({2,3,1,0});

    auto input = NDArrayFactory::create<TypeParam>('c', {2, 1, 4, 4});
    auto weights = NDArrayFactory::create<TypeParam>('c', {2, 1, 3, 3});
    auto epsilonNext = NDArrayFactory::create<TypeParam>('c', {2, 2, 4, 4});


    TypeParam _expEpsB[] = {952.0, 1540.0, 1636.0, 1180.0, 1791.0, 2886.0, 3057.0, 2193.0, 2223.0, 3570.0, 3741.0, 2673.0, 1900.0, 3028.0, 3160.0, 2240.0, 2872.0, 4612.0, 4708.0, 3356.0, 5247.0, 8358.0, 8529.0, 6033.0, 5679.0, 9042.0, 9213.0, 6513.0, 4588.0, 7252.0, 7384.0, 5184.0};
    NDArray expEps(_expEpsB, input.shapeInfo());

    input.linspace(1);
    weights.linspace(1);
    epsilonNext.linspace(1);
    weights.permutei({2,3,1,0});

    sd::ops::conv2d_bp op;

    auto results = op.evaluate({&input, &weights, &epsilonNext}, {},  {3, 3, 1, 1, 0, 0, 1, 1, 1}, {});

    ASSERT_TRUE(results.size() == 2);

    auto epsilon = results.at(0);
    auto gradW = results.at(1);

    ASSERT_TRUE(expWGrad.isSameShape(gradW));

    //expWGrad.printBuffer("Expctd buffer");
    //  gradW->printBuffer("Result buffer");
    ASSERT_TRUE(expWGrad.equalsTo(gradW));


    ASSERT_TRUE(input.isSameShape(epsilon));

    //  expEps.printBuffer("Expctd buffer");
    //epsilon->printBuffer("Result buffer");
    ASSERT_TRUE(expEps.equalsTo(epsilon));


}

TYPED_TEST(TypedConvolutionTests1, sconv2d_conv2d_1) {

    auto input = NDArrayFactory::create<TypeParam>('c', {2, 3, 10, 10});
    auto weightsD = NDArrayFactory::create<TypeParam>('c', {5, 5, 3, 2}, {1.f, 76.f, 26.f, 101.f, 51.f, 126.f, 2.f, 77.f, 27.f, 102.f, 52.f, 127.f, 3.f, 78.f, 28.f, 103.f, 53.f, 128.f, 4.f, 79.f, 29.f, 104.f, 54.f, 129.f, 5.f, 80.f, 30.f, 105.f, 55.f, 130.f,
                                        6.f, 81.f, 31.f, 106.f, 56.f, 131.f, 7.f, 82.f, 32.f, 107.f, 57.f, 132.f, 8.f, 83.f, 33.f, 108.f, 58.f, 133.f, 9.f, 84.f, 34.f, 109.f, 59.f, 134.f, 10.f, 85.f, 35.f, 110.f, 60.f, 135.f,
                                        11.f, 86.f, 36.f, 111.f, 61.f, 136.f, 12.f, 87.f, 37.f, 112.f, 62.f, 137.f, 13.f, 88.f, 38.f, 113.f, 63.f, 138.f, 14.f, 89.f, 39.f, 114.f, 64.f, 139.f, 15.f, 90.f, 40.f, 115.f, 65.f, 140.f,
                                        16.f, 91.f, 41.f, 116.f, 66.f, 141.f, 17.f, 92.f, 42.f, 117.f, 67.f, 142.f, 18.f, 93.f, 43.f, 118.f, 68.f, 143.f, 19.f, 94.f, 44.f, 119.f, 69.f, 144.f, 20.f, 95.f, 45.f, 120.f, 70.f, 145.f,
                                        21.f, 96.f, 46.f, 121.f, 71.f, 146.f, 22.f, 97.f, 47.f, 122.f, 72.f, 147.f, 23.f, 98.f, 48.f, 123.f, 73.f, 148.f, 24.f, 99.f, 49.f, 124.f, 74.f, 149.f, 25.f, 100.f, 50.f, 125.f, 75.f, 150.f});
    auto weightsP = NDArrayFactory::create<TypeParam>('c', {1, 1, 6, 10}, {0.0001f, 0.0007f, 0.0013f, 0.0019f, 0.0025f, 0.0031f, 0.0037f, 0.0043f, 0.0049f, 0.0055f,0.0002f, 0.0008f, 0.0014f, 0.0020f, 0.0026f, 0.0032f, 0.0038f, 0.0044f, 0.0050f, 0.0056f,
                                        0.0003f, 0.0009f, 0.0015f, 0.0021f, 0.0027f, 0.0033f, 0.0039f, 0.0045f, 0.0051f, 0.0057f,0.0004f, 0.0010f, 0.0016f, 0.0022f, 0.0028f, 0.0034f, 0.0040f, 0.0046f, 0.0052f, 0.0058f,
                                        0.0005f, 0.0011f, 0.0017f, 0.0023f, 0.0029f, 0.0035f, 0.0041f, 0.0047f, 0.0053f, 0.0059f,0.0006f, 0.0012f, 0.0018f, 0.0024f, 0.0030f, 0.0036f, 0.0042f, 0.0048f, 0.0054f, 0.0060f});

    auto expFF = NDArrayFactory::create<TypeParam>('c', {2, 6, 6, 6}, {10025.0f,10350.0f,10675.0f,11000.0f,11325.0f,11650.0f,13275.0f,13600.0f,13925.0f,14250.0f,14575.0f,14900.0f,16525.0f,16850.0f,
                                        17175.0f,17500.0f,17825.0f,18150.0f,19775.0f,20100.0f,20425.0f,20750.0f,21075.0f,21400.0f,23025.0f,23350.0f,23675.0f,24000.0f,
                                        24325.0f,24650.0f,26275.0f,26600.0f,26925.0f,27250.0f,27575.0f,27900.0f,53150.0f,55350.0f,57550.0f,59750.0f,61950.0f,64150.0f,
                                        75150.0f,77350.0f,79550.0f,81750.0f,83950.0f,86150.0f,97150.0f,99350.0f,101550.0f,103750.0f,105950.0f,108150.0f,119150.0f,
                                        121350.0f,123550.0f,125750.0f,127950.0f,130150.0f,141150.0f,143350.0f,145550.0f,147750.0f,149950.0f,152150.0f,163150.0f,
                                        165350.0f,167550.0f,169750.0f,171950.0f,174150.0f,119400.0f,120350.0f,121300.0f,122250.0f,123200.0f,124150.0f,128900.0f,
                                        129850.0f,130800.0f,131750.0f,132700.0f,133650.0f,138400.0f,139350.0f,140300.0f,141250.0f,142200.0f,143150.0f,147900.0f,
                                        148850.0f,149800.0f,150750.0f,151700.0f,152650.0f,157400.0f,158350.0f,159300.0f,160250.0f,161200.0f,162150.0f,166900.0f,
                                        167850.0f,168800.0f,169750.0f,170700.0f,171650.0f,350025.0f,352850.0f,355675.0f,358500.0f,361325.0f,364150.0f,378275.0f,
                                        381100.0f,383925.0f,386750.0f,389575.0f,392400.0f,406525.0f,409350.0f,412175.0f,415000.0f,417825.0f,420650.0f,434775.0f,
                                        437600.0f,440425.0f,443250.0f,446075.0f,448900.0f,463025.0f,465850.0f,468675.0f,471500.0f,474325.0f,477150.0f,491275.0f,
                                        494100.0f,496925.0f,499750.0f,502575.0f,505400.0f,353775.0f,355350.0f,356925.0f,358500.0f,360075.0f,361650.0f,369525.0f,
                                        371100.0f,372675.0f,374250.0f,375825.0f,377400.0f,385275.0f,386850.0f,388425.0f,390000.0f,391575.0f,393150.0f,401025.0f,
                                        402600.0f,404175.0f,405750.0f,407325.0f,408900.0f,416775.0f,418350.0f,419925.0f,421500.0f,423075.0f,424650.0f,432525.0f,
                                        434100.0f,435675.0f,437250.0f,438825.0f,440400.0f,771900.0f,775350.0f,778800.0f,782250.0f,785700.0f,789150.0f,806400.0f,
                                        809850.0f,813300.0f,816750.0f,820200.0f,823650.0f,840900.0f,844350.0f,847800.0f,851250.0f,854700.0f,858150.0f,875400.0f,
                                        878850.0f,882300.0f,885750.0f,889200.0f,892650.0f,909900.0f,913350.0f,916800.0f,920250.0f,923700.0f,927150.0f,944400.0f,
                                        947850.0f,951300.0f,954750.0f,958200.0f,961650.0f,107525.0f,107850.0f,108175.0f,108500.0f,108825.0f,109150.0f,110775.0f,
                                        111100.0f,111425.0f,111750.0f,112075.0f,112400.0f,114025.0f,114350.0f,114675.0f,115000.0f,115325.0f,115650.0f,117275.0f,
                                        117600.0f,117925.0f,118250.0f,118575.0f,118900.0f,120525.0f,120850.0f,121175.0f,121500.0f,121825.0f,122150.0f,123775.0f,
                                        124100.0f,124425.0f,124750.0f,125075.0f,125400.0f,713150.0f,715350.0f,717550.0f,719750.0f,721950.0f,724150.0f,735150.0f,
                                        737350.0f,739550.0f,741750.0f,743950.0f,746150.0f,757150.0f,759350.0f,761550.0f,763750.0f,765950.0f,768150.0f,779150.0f,
                                        781350.0f,783550.0f,785750.0f,787950.0f,790150.0f,801150.0f,803350.0f,805550.0f,807750.0f,809950.0f,812150.0f,823150.0f,
                                        825350.0f,827550.0f,829750.0f,831950.0f,834150.0f,404400.0f,405350.0f,406300.0f,407250.0f,408200.0f,409150.0f,413900.0f,
                                        414850.0f,415800.0f,416750.0f,417700.0f,418650.0f,423400.0f,424350.0f,425300.0f,426250.0f,427200.0f,428150.0f,432900.0f,433850.0f,434800.0f,435750.0f,436700.0f,437650.0f,442400.0f,443350.0f,444300.0f,445250.0f,446200.0f,447150.0f,451900.0f,452850.0f,453800.0f,454750.0f,455700.0f,456650.0f,1197525.0f,1200350.0f,1203175.0f,1206000.0f,1208825.0f,1211650.0f,1225775.0f,1228600.0f,1231425.0f,1234250.0f,1237075.0f,1239900.0f,1254025.0f,1256850.0f,1259675.0f,1262500.0f,1265325.0f,1268150.0f,1282275.0f,1285100.0f,1287925.0f,1290750.0f,1293575.0f,1296400.0f,1310525.0f,1313350.0f,1316175.0f,1319000.0f,1321825.0f,1324650.0f,1338775.0f,1341600.0f,1344425.0f,1347250.0f,1350075.0f,1352900.0f,826275.0f,827850.0f,829425.0f,831000.0f,832575.0f,834150.0f,842025.0f,843600.0f,845175.0f,846750.0f,848325.0f,849900.0f,857775.0f,859350.0f,860925.0f,862500.0f,864075.0f,865650.0f,873525.0f,875100.0f,876675.0f,878250.0f,879825.0f,881400.0f,889275.0f,890850.0f,892425.0f,894000.0f,895575.0f,897150.0f,905025.0f,906600.0f,908175.0f,909750.0f,911325.0f,912900.0f,1806900.0f,1810350.0f,1813800.0f,1817250.0f,1820700.0f,1824150.0f,1841400.0f,1844850.0f,1848300.0f,1851750.0f,1855200.0f,1858650.0f,1875900.0f,1879350.0f,1882800.0f,1886250.0f,1889700.0f,1893150.0f,1910400.0f,1913850.0f,1917300.0f,1920750.0f,1924200.0f,1927650.0f,1944900.0f,1948350.0f,1951800.0f,1955250.0f,1958700.0f,1962150.0f,1979400.0f,1982850.0f,1986300.0f,1989750.0f,1993200.0f,1996650.f});
    auto exp2FF = NDArrayFactory::create<TypeParam>('c', {2, 10, 6, 6}, {827.4900282f,832.2350283f,836.9800284f,841.725028f,846.4700287f,851.2150288f,874.9400293f,879.6850294f,884.4300295f,889.1750296f,893.9200297f,898.665029f,
                                        922.3900304f,927.1350305f,931.8800306f,936.6250307f,941.3700308f,946.1150309f,969.8400315f,974.5850316f,979.3300317f,984.0750318f,988.8200319f,993.5650320f,
                                        1017.2900326f,1022.0350327f,1026.7800328f,1031.5250329f,1036.2700330f,1041.0150331f,1064.7400337f,1069.4850338f,1074.2300339f,1078.9750340f,1083.7200341f,
                                        1088.4650342f,1822.4550553f,1833.995055f,1845.5350558f,1857.075056f,1868.6150563f,1880.1550566f,1937.8550578f,1949.3950581f,1960.9350583f,1972.4750586f,
                                        1984.015058f,1995.5550591f,2053.2550604f,2064.7950606f,2076.3350609f,2087.8750611f,2099.4150614f,2110.955061f,2168.6550629f,2180.1950632f,2191.7350634f,
                                        2203.2750637f,2214.8150639f,2226.3550642f,2284.0550655f,2295.5950657f,2307.1350660f,2318.6750662f,2330.2150665f,2341.7550667f,2399.4550680f,2410.9950683f,
                                        2422.5350685f,2434.0750688f,2445.6150690f,2457.1550693f,2817.419968f,2835.7549686f,2854.0899683f,2872.4249680f,2890.7599677f,2909.0949674f,3000.7699660f,
                                        3019.104965f,3037.4399655f,3055.7749652f,3074.1099649f,3092.4449646f,3184.1199632f,3202.4549629f,3220.789962f,3239.1249624f,3257.4599621f,3275.7949618f,
                                        3367.4699604f,3385.8049601f,3404.1399598f,3422.474959f,3440.8099593f,3459.1449590f,3550.8199576f,3569.1549573f,3587.4899570f,3605.8249567f,3624.1599565f,
                                        3642.4949562f,3734.1699548f,3752.5049545f,3770.8399542f,3789.1749539f,3807.5099536f,3825.8449534f,3812.385098f,3837.5150988f,3862.6450994f,3887.7751000f,
                                        3912.9051006f,3938.0351012f,4063.6851041f,4088.8151047f,4113.9451053f,4139.0751059f,4164.2051065f,4189.3351071f,4314.9851100f,4340.1151106f,4365.2451112f,
                                        4390.3751118f,4415.5051124f,4440.6351130f,4566.2851159f,4591.4151165f,4616.5451171f,4641.6751177f,4666.805118f,4691.9351188f,4817.5851218f,4842.7151224f,
                                        4867.8451230f,4892.975123f,4918.1051241f,4943.2351247f,5068.8851277f,5094.0151283f,5119.1451288f,5144.2751294f,5169.4051300f,5194.5351306f,4807.3499803f,
                                        4839.2749801f,4871.1999799f,4903.1249797f,4935.0499795f,4966.9749793f,5126.5999784f,5158.5249782f,5190.4499780f,5222.3749778f,5254.2999777f,5286.2249775f,
                                        5445.8499765f,5477.774976f,5509.6999762f,5541.6249760f,5573.5499758f,5605.4749756f,5765.0999747f,5797.0249745f,5828.9499743f,5860.8749741f,5892.7999739f,
                                        5924.724973f,6084.3499728f,6116.2749726f,6148.1999724f,6180.1249723f,6212.0499721f,6243.9749719f,6403.59997f,6435.5249708f,6467.4499706f,6499.3749704f,
                                        6531.2999702f,6563.2249700f,5802.3150007f,5841.0350006f,5879.7550005f,5918.4750004f,5957.195000f,5995.9150003f,6189.5149999f,6228.2349998f,6266.9549997f,
                                        6305.6749996f,6344.3949995f,6383.114999f,6576.7149990f,6615.4349990f,6654.1549989f,6692.8749988f,6731.5949987f,6770.3149986f,6963.9149982f,7002.6349981f,
                                        7041.3549981f,7080.0749980f,7118.7949979f,7157.5149978f,7351.1149974f,7389.8349973f,7428.5549972f,7467.2749972f,7505.9949971f,7544.7149970f,7738.3149966f,7777.0349965f,7815.7549964f,7854.4749963f,7893.1949963f,7931.9149962f,6797.2799488f,6842.794948f,6888.3099489f,6933.8249490f,6979.3399491f,7024.8549492f,7252.4299497f,7297.9449498f,7343.4599499f,7388.9749500f,7434.489950f,7480.0049501f,7707.5799506f,7753.0949507f,7798.6099508f,7844.1249509f,7889.6399510f,7935.1549511f,8162.7299515f,8208.2449516f,8253.7599517f,8299.2749518f,8344.7899519f,8390.3049520f,8617.8799525f,8663.394952f,8708.9099526f,8754.4249527f,8799.9399528f,8845.4549529f,9073.0299534f,9118.5449535f,9164.0599536f,9209.5749537f,9255.089953f,9300.604953f,7792.2451647f,7844.5551655f,7896.8651663f,7949.1751671f,8001.4851679f,8053.7951686f,8315.3451725f,8367.6551733f,8419.9651741f,8472.2751749f,8524.585175f,8576.8951764f,8838.4451803f,8890.7551811f,8943.0651819f,8995.3751827f,9047.6851834f,9099.9951842f,9361.5451881f,9413.8551889f,9466.1651897f,9518.475190f,9570.7851912f,9623.0951920f,9884.6451959f,9936.9551967f,9989.2651975f,10041.5751982f,10093.8851990f,10146.1951998f,10407.7452037f,10460.0552045f,10512.3652053f,10564.6752060f,10616.9852068f,10669.2952076f,8787.210074f,8846.3150748f,8905.4200750f,8964.5250752f,9023.6300755f,9082.7350757f,9378.2600768f,9437.3650770f,9496.4700773f,9555.5750775f,9614.6800777f,9673.7850779f,9969.3100791f,10028.4150793f,10087.5200795f,10146.625079f,10205.7300800f,10264.8350802f,10560.3600813f,10619.465081f,10678.5700818f,10737.6750820f,10796.7800822f,10855.8850825f,11151.4100836f,11210.5150838f,11269.6200840f,11328.7250843f,11387.8300845f,11446.9350847f,11742.4600858f,11801.5650861f,11860.6700863f,11919.7750865f,11978.880086f,12037.9850870f,9782.1750935f,9848.0750935f,9913.9750934f,9979.8750934f,10045.7750934f,10111.6750933f,10441.1750931f,10507.0750931f,10572.9750931f,10638.8750930f,10704.7750930f,10770.6750930f,11100.1750928f,11166.0750927f,11231.9750927f,11297.8750927f,11363.7750926f,11429.6750926f,11759.1750924f,11825.0750924f,11890.9750923f,11956.8750923f,12022.7750923f,12088.6750922f,12418.175092f,12484.0750920f,12549.9750920f,12615.8750919f,12681.7750919f,12747.6750919f,13077.1750917f,13143.0750916f,13208.9750916f,13274.8750916f,13340.7750915f,13406.6750915f,2250.990060f,2255.7350610f,2260.4800611f,2265.2250612f,2269.9700613f,2274.7150614f,2298.4400619f,2303.185062f,2307.9300622f,2312.6750623f,2317.4200624f,2322.1650625f,2345.8900630f,2350.6350631f,2355.380063f,2360.1250634f,2364.8700635f,2369.6150636f,2393.3400641f,2398.0850642f,2402.8300643f,2407.5750644f,2412.320064f,2417.0650647f,2440.7900652f,2445.5350653f,2450.2800654f,2455.0250655f,2459.7700656f,2464.515065f,2488.2400663f,2492.9850664f,2497.7300665f,2502.4750666f,2507.2200667f,2511.9650668f,5284.4551315f,5295.9951318f,5307.535132f,5319.0751323f,5330.6151326f,5342.1551328f,5399.8551341f,5411.3951343f,5422.9351346f,5434.475134f,5446.0151351f,5457.5551354f,5515.2551366f,5526.7951369f,5538.3351371f,5549.8751374f,5561.4151376f,5572.9551379f,5630.6551392f,5642.1951394f,5653.7351397f,5665.2751399f,5676.8151402f,5688.3551404f,5746.0551417f,5757.5951420f,5769.1351422f,5780.6751425f,5792.2151427f,5803.7551430f,5861.455144f,5872.9951445f,5884.5351448f,5896.0751450f,5907.6151453f,5919.1551455f,8317.919884f,8336.2548841f,8354.5898838f,8372.9248835f,8391.2598832f,8409.59488f,8501.2698815f,8519.6048813f,8537.9398810f,8556.2748807f,8574.6098804f,8592.9448801f,8684.6198787f,8702.9548784f,8721.2898782f,8739.6248779f,8757.9598776f,8776.2948773f,8867.9698759f,8886.3048756f,8904.6398753f,8922.9748751f,8941.3098748f,8959.6448745f,9051.3198731f,9069.6548728f,9087.9898725f,9106.3248722f,9124.6598720f,9142.9948717f,9234.6698703f,9253.0048700f,9271.3398697f,9289.6748694f,9308.0098691f,9326.3448689f,11351.3852747f,11376.5152753f,11401.6452759f,11426.7752765f,11451.9052771f,11477.0352777f,11602.6852806f,11627.8152812f,11652.9452818f,11678.0752824f,11703.2052830f,11728.335283f,11853.9852865f,11879.1152871f,11904.2452877f,11929.3752883f,11954.505288f,11979.6352894f,12105.2852924f,12130.4152930f,12155.545293f,12180.6752941f,12205.8052947f,12230.9352953f,12356.5852983f,12381.715298f,12406.8452994f,12431.9753000f,12457.1053006f,12482.2353012f,12607.8853041f,12633.0153047f,12658.1453053f,12683.2753059f,12708.4053065f,12733.5353071f,14384.8499244f,14416.7749242f,14448.6999240f,14480.6249238f,14512.549923f,14544.4749235f,14704.0999225f,14736.024922f,14767.9499222f,14799.8749220f,14831.7999218f,14863.7249216f,15023.3499207f,15055.2749205f,15087.1999203f,15119.1249201f,15151.0499199f,15182.9749197f,15342.5999188f,15374.5249186f,15406.4499184f,15438.374918f,15470.2999181f,15502.2249179f,15661.84991f,15693.7749168f,15725.6999166f,15757.6249164f,15789.5499162f,15821.4749160f,15981.0999151f,16013.0249149f,16044.9499147f,16076.8749145f,16108.7999143f,16140.7249142f,17418.314976f,17457.0349761f,17495.7549760f,17534.4749759f,17573.1949758f,17611.9149757f,17805.5149753f,17844.234975f,17882.9549752f,17921.6749751f,17960.3949750f,17999.1149749f,18192.7149745f,18231.4349744f,18270.154974f,18308.8749743f,18347.5949742f,18386.3149741f,18579.9149737f,18618.6349736f,18657.3549735f,18696.074973f,18734.7949734f,18773.5149733f,18967.1149729f,19005.8349728f,19044.5549727f,19083.2749726f,19121.994972f,19160.7149725f,19354.3149721f,19393.0349720f,19431.7549719f,19470.4749718f,19509.1949717f,19547.914971f,20451.7799765f,20497.2949766f,20542.8099767f,20588.3249768f,20633.8399769f,20679.3549770f,20906.929977f,20952.4449775f,20997.9599776f,21043.4749777f,21088.9899778f,21134.5049779f,21362.0799784f,21407.5949785f,21453.1099786f,21498.624978f,21544.139978f,21589.6549788f,21817.2299793f,21862.7449794f,21908.2599795f,21953.7749796f,21999.2899797f,22044.8049798f,22272.3799802f,22317.8949803f,22363.4099804f,22408.9249805f,22454.4399806f,22499.9549807f,22727.529981f,22773.044981f,22818.5599813f,22864.0749814f,22909.5899815f,22955.1049816f,23485.2453985f,23537.555399f,23589.8654000f,23642.1754008f,23694.4854016f,23746.7954024f,24008.3454063f,24060.655407f,24112.9654078f,24165.2754086f,24217.5854094f,24269.8954102f,24531.4454141f,24583.7554148f,24636.0654156f,24688.3754164f,24740.6854172f,24792.99541f,25054.545421f,25106.8554226f,25159.1654234f,25211.4754242f,25263.7854250f,25316.0954257f,25577.6454296f,25629.9554304f,25682.2654312f,25734.5754320f,25786.8854328f,25839.1954335f,26100.7454374f,26153.0554382f,26205.3654390f,26257.6754398f,26309.985440f,26362.2954413f,26518.7101423f,26577.8151425f,26636.920142f,26696.0251430f,26755.1301432f,26814.2351434f,27109.7601446f,27168.8651448f,27227.9701450f,27287.0751452f,27346.1801455f,27405.2851457f,27700.8101468f,27759.9151470f,27819.0201473f,27878.1251475f,27937.2301477f,27996.33514f,28291.8601491f,28350.9651493f,28410.0701495f,28469.175149f,28528.2801500f,28587.3851502f,28882.9101513f,28942.0151516f,29001.1201518f,29060.2251520f,29119.3301522f,29178.4351525f,29473.9601536f,29533.0651538f,29592.1701540f,29651.2751543f,29710.3801545f,29769.4851547f,29552.1750826f,29618.0750825f,29683.9750825f,29749.8750825f,29815.7750824f,29881.6750824f,30211.1750822f,30277.0750822f,30342.9750821f,30408.8750821f,30474.7750821f,30540.6750820f,30870.175081f,30936.0750818f,31001.9750818f,31067.8750817f,31133.7750817f,31199.6750817f,31529.1750815f,31595.075081f,31660.9750814f,31726.8750814f,31792.7750813f,31858.6750813f,32188.1750811f,32254.0750811f,32319.975081f,32385.8750810f,32451.7750810f,32517.6750809f,32847.1750808f,32913.0750807f,32978.9750807f,33044.875080f,33110.7750806f,33176.67508062f});

    input.linspace(1);

    sd::ops::sconv2d op;
    auto resultFF = op.evaluate({&input, &weightsD}, {}, {5, 5, 1, 1, 0, 0, 1, 1, 0}, {});

    auto z = resultFF.at(0);

    ASSERT_TRUE(z->isSameShape(&expFF));
    ASSERT_TRUE(z->equalsTo(&expFF, 1));


    sd::ops::conv2d op2d;
    // weightsP.printShapeInfo();
    auto result2D = op2d.evaluate({z, &weightsP}, {}, {1, 1, 1, 1, 0, 0, 1, 1, 0, 0}, {});

    auto z2d = result2D.at(0);
    // z2d->printBuffer();

    ASSERT_TRUE(z2d->isSameShape(&exp2FF));
    ASSERT_TRUE(z2d->equalsTo(&exp2FF));
}

TEST_F(ConvolutionTests1, deconv2d_bp_1) {

    int bS=3, iH=4,iW=4,  iC=3,oC=2,  kH=1,kW=1,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=4,oW=4;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iC, iH, iW}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, sd::DataType::FLOAT32);
    NDArray weights('c',{kH,kW,oC,iC}, {1,3,5,2,4,6}, sd::DataType::FLOAT32);
    NDArray gradO('c', {bS, oC, oH, oW},sd::DataType::FLOAT32);

    NDArray expGradI('c', {bS, iC, iH, iW}, {35.f,   38.f,   41.f,   44.f,   47.f,   50.f,   53.f,   56.f,   59.f,   62.f,   65.f,    68.f,   71.f,   74.f,
                77.f,   80.f,   71.f,   78.f,   85.f,   92.f,   99.f,  106.f,    113.f,  120.f,  127.f,  134.f,  141.f,  148.f,  155.f,  162.f,  169.f,
                176.f,  107.f,    118.f,  129.f,  140.f,  151.f,  162.f,  173.f,  184.f,  195.f,  206.f,  217.f,  228.f,    239.f,  250.f,  261.f,  272.f,
                131.f,  134.f,  137.f,  140.f,  143.f,  146.f,  149.f,    152.f,  155.f,  158.f,  161.f,  164.f,  167.f,  170.f,  173.f,  176.f,  295.f,
                302.f,    309.f,  316.f,  323.f,  330.f,  337.f,  344.f,  351.f,  358.f,  365.f,  372.f,  379.f,    386.f,  393.f,  400.f,  459.f,  470.f,
                481.f,  492.f,  503.f,  514.f,  525.f,  536.f,    547.f,  558.f,  569.f,  580.f,  591.f,  602.f,  613.f,  624.f,  227.f,  230.f,  233.f,
                236.f,  239.f,  242.f,  245.f,  248.f,  251.f,  254.f,  257.f,  260.f,  263.f,  266.f,    269.f,  272.f,  519.f,  526.f,  533.f,  540.f,
                547.f,  554.f,  561.f,  568.f,  575.f,    582.f,  589.f,  596.f,  603.f,  610.f,  617.f,  624.f,  811.f,  822.f,  833.f,  844.f,    855.f,
                866.f,  877.f,  888.f,  899.f,  910.f,  921.f,  932.f,  943.f,  954.f,  965.f,    976.f}, sd::DataType::FLOAT32);
    NDArray expGradW('c', {kH, kW, oC, iC}, {160008., 191112., 222216., 203400., 246792., 290184.f}, sd::DataType::FLOAT32);
    NDArray expGradB('c', {oC}, {1944.f,  2712.f}, sd::DataType::FLOAT32);

    input.linspace(1);
    bias.linspace(1);
    gradO.linspace(1);


    sd::ops::deconv2d_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto gradI = results.at(0);
    auto gradW = results.at(1);
    auto gradB = results.at(2);

    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));

    ASSERT_TRUE(expGradB.isSameShape(gradB));
    ASSERT_TRUE(expGradB.equalsTo(gradB));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, deconv2d_bp_2) {

    int bS=3, iH=4,iW=4,  iC=3,oC=2,  kH=2,kW=1,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=4,oW=4;    // 5,4
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW
    int wFormat     = 1;             // 0 - [kH, kW, oC, iC], 1 - [iC, oC, kH, kW], 2 - [iC, kH, kW, oC]

    NDArray input('c', {bS, iC, iH, iW}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, {-0.1, 0.2}, sd::DataType::FLOAT32);
    NDArray weights('c',{iC, oC, kH, kW}, {1., 7., 2., 10., 3., 8., 4., 11., 5., 9., 6., 12.}, sd::DataType::FLOAT32);
    NDArray gradO('c', {bS, oC, oH, oW},sd::DataType::FLOAT32);

    NDArray expGradI('c', {bS, iC, iH, iW}, {-77.400002, -77.199997, -77., -76.800003, -76.599998, -76.400002, -76.200005, -76., -75.800003, -75.599998, -75.399994,
        -75.199997, -11.32, -11.29, -11.26, -11.23, -100.839996, -100.580002, -100.32, -100.059998, -99.800003, -99.540001, -99.279999, -99.019997, -98.760002, -98.50,
        -98.240005, -97.979996, -26.52, -26.450001, -26.380001, -26.309999, -124.279999, -123.959991, -123.639999, -123.32, -123., -122.68, -122.360001, -122.040001,
         -121.720001, -121.400009, -121.080002, -120.759995, -41.720001, -41.610001, -41.50, -41.389999, -71., -70.800003, -70.599998, -70.399994, -70.199997, -70., -69.800003, -69.600006, -69.400002, -69.199997, -69., -68.799995, -10.360001, -10.33, -10.30, -10.27, -92.519997, -92.260002, -92., -91.740005, -91.479996, -91.220001, -90.960007, -90.700005, -90.440002, -90.18, -89.919998, -89.660004, -24.280001, -24.209999, -24.139999, -24.07, -114.040001, -113.720001, -113.400009, -113.080002, -112.759995, -112.440002, -112.120003, -111.800003, -111.480003, -111.159996, -110.839996, -110.520004, -38.200001, -38.09, -37.980003, -37.869999, -64.599998, -64.400002, -64.199997, -64., -63.799995, -63.599998, -63.400002, -63.199997, -63., -62.799995, -62.599998, -62.400002, -9.40, -9.37, -9.34, -9.309999, -84.200005, -83.940002, -83.68, -83.419998, -83.160004, -82.900002, -82.639999, -82.379997, -82.119995, -81.860001, -81.600006, -81.339996, -22.040001, -21.970001, -21.90, -21.83, -103.800003, -103.480003, -103.159996, -102.839996, -102.520004, -102.200005, -101.879997, -101.559998, -101.239998, -100.919998, -100.599998, -100.279999, -34.68, -34.57, -34.459999, -34.349998}, sd::DataType::FLOAT32);

    NDArray expGradW('c', {iC, oC, kH, kW}, {-3010.799805, -2502.420410, -2899.439209, -2407.380615, -242.159332, -437.460510, -253.680466, -434.580048, 2526.479980, 1627.500000, 2392.079834, 1538.220093}, sd::DataType::FLOAT32);
    NDArray expGradB('c', {oC}, {-173.040009, -165.360016}, sd::DataType::FLOAT32);

    input.linspace(70., -1);
    gradO.linspace(-4, 0.01);

    sd::ops::deconv2d_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat, wFormat});

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto gradI = results.at(0);
    auto gradW = results.at(1);
    auto gradB = results.at(2);

    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));

    ASSERT_TRUE(expGradB.isSameShape(gradB));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, deconv2d_bp_3) {

    int bS=3, iH=4,iW=4,  iC=3,oC=2,  kH=2,kW=1,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=5,oW=4;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW
    int wFormat     = 2;             // 0 - [kH, kW, oC, iC], 1 - [iC, oC, kH, kW], 2 - [iC, kH, kW, oC]

    NDArray input('c', {bS, iH, iW, iC}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, {-0.1, 0.2}, sd::DataType::FLOAT32);
    NDArray weights('c',{iC, kH, kW, oC}, {1., 4., 7., 10., 2., 5., 8., 11., 3., 6., 9., 12.}, sd::DataType::FLOAT32);
    NDArray gradO('c', {bS, oH, oW, oC}, sd::DataType::FLOAT32);

    NDArray expGradI('c', {bS, iH, iW, iC}, {-86.5, -102.320007, -118.139999, -86.060005, -101.800003, -117.540001, -85.619995, -101.279999, -116.940002, -85.18,
        -100.759995, -116.339996, -84.740005, -100.239998, -115.739998, -84.300003, -99.720001, -115.139999, -83.860001, -99.199997, -114.539993, -83.419998, -98.68,
        -113.939995, -82.979996, -98.160004, -113.339996, -82.539993, -97.639999, -112.739998, -82.099998, -97.120003, -112.139999, -81.660004, -96.600006, -111.539993,
        -81.220001, -96.080002, -110.939995, -80.779999, -95.559998, -110.340012, -80.340004, -95.040001, -109.740005, -79.900002, -94.519997, -109.139992, -77.699997,
        -91.919998, -106.139999, -77.260002, -91.400002, -105.540001, -76.820007, -90.880005, -104.940002, -76.380005, -90.360001, -104.339996, -75.940002, -89.839996, -103.740005, -75.5, -89.320007, -103.139999, -75.060005, -88.800003, -102.540001, -74.619995, -88.279999, -101.940002, -74.18, -87.759995, -101.339996, -73.740005, -87.239998, -100.739998, -73.300003, -86.720001, -100.139999, -72.860001, -86.199997, -99.539993, -72.419998, -85.68, -98.939995, -71.979996, -85.160004, -98.339996, -71.539993, -84.639999, -97.740005, -71.099998, -84.120003, -97.139999, -68.899994, -81.519997, -94.139999, -68.459999, -81.00, -93.539993, -68.019997, -80.479996, -92.940002, -67.580002, -79.959999, -92.339996, -67.139999, -79.440002, -91.740005, -66.699997, -78.919998, -91.139999, -66.260002, -78.399994, -90.540001, -65.820007, -77.880005, -89.940002, -65.380005, -77.360001, -89.339996, -64.940002, -76.839996, -88.740005, -64.5, -76.320007, -88.139999, -64.060005, -75.800003, -87.540001, -63.619995, -75.279999, -86.940002, -63.18, -74.759995, -86.339996, -62.739998, -74.239998, -85.739998, -62.299999, -73.720001, -85.139999}, sd::DataType::FLOAT32);

    NDArray expGradW('c', {iC, kH, kW, oC}, {-592.800110, -593.039917, -594.719116, -594.960266, -427.199890, -427.919617, -432.959900, -433.679993, -261.600281, -262.799591, -271.200317, -272.399536}, sd::DataType::FLOAT32);
    NDArray expGradB('c', {oC}, {-204.600006, -204.}, sd::DataType::FLOAT32);

    input.linspace(70., -1);
    gradO.linspace(-4, 0.01);

    sd::ops::deconv2d_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat, wFormat});

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto gradI = results.at(0);
    auto gradW = results.at(1);
    auto gradB = results.at(2);

    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));

    ASSERT_TRUE(expGradB.isSameShape(gradB));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
}

TYPED_TEST(TypedConvolutionTests1, Test_Conv1D_ff_1) {
    auto input = NDArrayFactory::create<TypeParam>('c', {2, 2, 6});
    auto weights = NDArrayFactory::create<TypeParam>('c', {2, 2, 3}, {1,5,9,3,7,11,2,6,10,4,8,12});
    auto bias = NDArrayFactory::create<TypeParam>('c', {3});
    auto expFF = NDArrayFactory::create<TypeParam>('c', {2, 3, 5}, {59.0f, 69.0f, 79.0f, 89.0f, 99.0f, 132.0f, 158.0f, 184.0f, 210.0f, 236.0f, 205.0f, 247.0f, 289.0f, 331.0f, 373.0f, 179.0f, 189.0f, 199.0f, 209.0f, 219.0f, 444.0f, 470.0f, 496.0f, 522.0f, 548.0f, 709.0f, 751.0f, 793.0f, 835.0f, 877.0f});
    auto expEps = NDArrayFactory::create<TypeParam>('c', {2, 2, 6}, {130.0f, 293.0f, 326.0f, 359.0f, 392.0f, 220.0f, 166.0f, 371.0f, 416.0f, 461.0f, 506.0f, 280.0f, 355.0f, 788.0f, 821.0f, 854.0f, 887.0f, 490.0f, 481.0f, 1046.0f, 1091.0f, 1136.0f, 1181.0f, 640.0f});
    auto expGW = NDArrayFactory::create<TypeParam>('c', {3, 2, 2}, {1415.0f, 1520.0f, 2045.0f, 2150.0f, 1865.0f, 2020.0f, 2795.0f, 2950.0f, 2315.0f, 2520.0f, 3545.0f, 3750.0f});
    auto expGB = NDArrayFactory::create<TypeParam>('c', {3}, {105.0f, 155.0f, 205.0f});

    expGW.permutei({2,1,0});
    input.linspace(1);
    bias.linspace(1);

    sd::ops::conv1d op;
    auto result_FF = op.evaluate({&input, &weights, &bias}, {}, {2, 1, 0, 1, 0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result_FF.status());

    auto z = result_FF.at(0);

    ASSERT_TRUE(expFF.isSameShape(z));
    ASSERT_TRUE(expFF.equalsTo(z));

    sd::ops::conv1d_bp op_bp;

    auto epsilonNxt = new NDArray(z->dup());
    epsilonNxt->linspace(1);

    auto result_BP = op_bp.evaluate({&input, &weights, &bias, epsilonNxt}, {}, {2, 1, 0, 1, 0, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result_BP.status());

    auto eps = result_BP.at(0);
    auto gradW = result_BP.at(1);
    auto gradB = result_BP.at(2);

    ASSERT_TRUE(expEps.isSameShape(eps));
    ASSERT_TRUE(expGW.isSameShape(gradW));
    ASSERT_TRUE(expGB.isSameShape(gradB));

    ASSERT_TRUE(expEps.equalsTo(eps));
    ASSERT_TRUE(expGW.equalsTo(gradW));
    ASSERT_TRUE(expGB.equalsTo(gradB));

    delete epsilonNxt;
}


TYPED_TEST(TypedConvolutionTests1, Test_Conv1D_ff_2) {
    auto input = NDArrayFactory::create<TypeParam>('c', {2, 2, 6});
    auto weights = NDArrayFactory::create<TypeParam>('c', {2, 2, 3}, {1.f, 5.f, 9.f, 3.f, 7.f, 11.f, 2.f, 6.f, 10.f, 4.f, 8.f, 12.f});

    input.linspace(1);

    sd::ops::conv1d op;
    auto result = op.evaluate({&input, &weights}, {}, {2, 1, 0, 1, 1,0});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv1d_causal_1) {

    int bS=2, iW=3,  iC=4,oC=3,  kW=2,  sW=1,  pW=0,  dW=1;
    int oW = (iW-1)/sW + 1;
    int paddingMode = 2;             // CAUSAL
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iW, iC});
    NDArray weights('c', {kW, iC, oC});
    NDArray bias('c', {oC}, {-1,-2,-3});

    NDArray expOutput('c', {bS, oW, oC}, {18. ,  18. ,  18. , 53. ,  55.6,  58.2, 89.8,  95.6, 101.4, 102. , 106.8, 111.6, 163.4, 175.6, 187.8, 200.2, 215.6, 231.});

    input.linspace(1., 1.);
    weights.linspace(0.1, 0.1);

    sd::ops::conv1d op;
    auto results = op.evaluate({&input, &weights, &bias}, {kW, sW, pW, dW,  paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv1d_causal_2) {

    int bS=2, iW=16,  iC=3,oC=4,  kW=2,  sW=2,  pW=0,  dW=1;
    int oW = (iW-1)/sW + 1;
    int paddingMode = 2;             // CAUSAL
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iW, iC});
    NDArray weights('c', {kW, iC, oC});
    NDArray bias('c', {oC}, {-1,-2,-3,-4});

    NDArray expOutput('c', {bS, oW, oC}, { 10. ,   9.6,   9.2,   8.8, 48.9,  51.8,  54.7,  57.6, 88.5,  95. , 101.5, 108. , 128.1, 138.2, 148.3, 158.4,
                                           167.7, 181.4, 195.1, 208.8, 207.3, 224.6, 241.9, 259.2, 246.9, 267.8, 288.7, 309.6, 286.5, 311. , 335.5, 360. ,
                                           254.8, 268.8, 282.8, 296.8, 365.7, 397.4, 429.1, 460.8, 405.3, 440.6, 475.9, 511.2, 444.9, 483.8, 522.7, 561.6,
                                           484.5, 527. , 569.5, 612. , 524.1, 570.2, 616.3, 662.4, 563.7, 613.4, 663.1, 712.8, 603.3, 656.6, 709.9, 763.2});

    input.linspace(1., 1.);
    weights.linspace(0.1, 0.1);

    sd::ops::conv1d op;
    auto results = op.evaluate({&input, &weights, &bias}, {kW, sW, pW, dW, paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv1d_causal_3) {

    int bS=2, iW=16,  iC=3,oC=4,  kW=3,  sW=3,  pW=0,  dW=1;
    int oW = (iW-1)/sW + 1;
    int paddingMode = 2;             // CAUSAL
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iW, iC});
    NDArray weights('c', {kW, iC, oC});
    NDArray bias('c', {oC}, {-1,-2,-3,-4});

    NDArray expOutput('c', {bS, oW, oC}, {17.2,   16.8,   16.4,   16.,145.4,  151.6,  157.8,  164.,283.1,  297.4,  311.7,  326.,  420.8,  443.2,  465.6,  488.,
                                558.5,  589.,  619.5,  650.,696.2001,  734.8,  773.4,  812.,  434.8,  448.8,  462.8,  476.8,   879.8,  929.2,  978.6, 1028.,
                                1017.5, 1075., 1132.5, 1190.,1155.2001, 1220.8, 1286.4, 1352.,1292.8999, 1366.6, 1440.3, 1514.,  1430.6001, 1512.4, 1594.2, 1676.});

    input.linspace(1., 1.);
    weights.linspace(0.1, 0.1);

    sd::ops::conv1d op;
    auto results = op.evaluate({&input, &weights, &bias}, {kW, sW, pW, dW, paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv1d_causal_4) {

    int bS=2, iW=8,  iC=3,oC=4,  kW=3,  sW=1,  pW=0,  dW=3;
    int oW = (iW-1)/sW + 1;
    int paddingMode = 2;             // CAUSAL
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iW, iC});
    NDArray weights('c', {kW, iC, oC});
    NDArray bias('c', {oC}, {-1,-2,-3,-4});

    NDArray expOutput('c', {bS, oW, oC}, {17.2,  16.8,  16.4,  16. ,43.3,  43.8,  44.3,  44.8,69.4,  70.8,  72.2,  73.6,106.5, 109.4, 112.3, 115.2,147.9, 152.6, 157.3, 162. ,189.3, 195.8, 202.3,
                                        208.8,234.5, 243.4, 252.3, 261.2,280.4, 292. , 303.6, 315.2, 226. , 232.8, 239.6, 246.4,  252.1, 259.8, 267.5, 275.2,278.2, 286.8, 295.4, 304. ,437.7,
                                        455. , 472.3, 489.6,479.1, 498.2, 517.3, 536.4,520.5, 541.4, 562.3, 583.2,  601.7, 632.2, 662.7, 693.2, 647.6, 680.8, 714. , 747.2});

    input.linspace(1., 1.);
    weights.linspace(0.1, 0.1);

    sd::ops::conv1d op;
    auto results = op.evaluate({&input, &weights, &bias}, {kW, sW, pW, dW, paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv1d_causal_5) {

    int bS=2, iW=8,  iC=3,oC=4,  kW=3,  sW=1,  pW=0,  dW=3;
    int oW = (iW-1)/sW + 1;
    int paddingMode = 2;             // CAUSAL
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iC, iW});
    NDArray weights('c', {kW, iC, oC});
    NDArray bias('c', {oC}, {-1,-2,-3,-4});

    NDArray expOutput('c', {bS, oC, oW}, { 83.7,  92.4, 101.1, 162.1, 175.9, 189.7, 223.4, 238.7,85.4,  94.4, 103.4, 167.4, 181.8, 196.2, 233.2, 249.4,87.1,  96.4, 105.7, 172.7, 187.7, 202.7, 243. , 260.1,
                         88.8,  98.4, 108. , 178. , 193.6, 209.2, 252.8, 270.8, 292.5, 301.2, 309.9, 493.3, 507.1, 520.9, 590.6, 605.9,  301.4, 310.4, 319.4, 513. , 527.4, 541.8, 622. , 638.2,
                        310.3, 319.6, 328.9, 532.7, 547.7, 562.7, 653.4, 670.5,  319.2, 328.8, 338.4, 552.4, 568. , 583.6, 684.8, 702.8});

    input.linspace(1., 1.);
    weights.linspace(0.1, 0.1);

    sd::ops::conv1d op;
    auto results = op.evaluate({&input, &weights, &bias}, {kW, sW, pW, dW, paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv1d_causal_6) {

    int bS=2, iW=16,  iC=3,oC=4,  kW=3,  sW=3,  pW=0,  dW=1;
    int oW = (iW-1)/sW + 1;
    int paddingMode = 2;             // CAUSAL
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iC, iW});
    NDArray weights('c', {kW, iC, oC});
    NDArray bias('c', {oC}, {-1,-2,-3,-4});

    NDArray expOutput('c', {bS, oC, oW}, {159.7,335.3,381.2,427.1,473. ,518.9,163.8,351.4,400. ,448.6,497.2,545.8,167.9,367.5,418.8,470.1,521.4,572.7,172. ,383.6,437.6,491.6,545.6,599.6,
                                577.3, 1069.7, 1115.6, 1161.5, 1207.4, 1253.3,595.8, 1129. , 1177.6, 1226.2, 1274.8, 1323.4,614.3, 1188.3, 1239.6, 1290.9, 1342.2, 1393.5,
                                632.8, 1247.6, 1301.6, 1355.6, 1409.6, 1463.6});

    input.linspace(1., 1.);
    weights.linspace(0.1, 0.1);

    sd::ops::conv1d op;
    auto results = op.evaluate({&input, &weights, &bias}, {kW, sW, pW, dW, paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv1d_causal_7) {

    int bS=2, iW=8,  iC=3,oC=4,  kW=2,  sW=1,  pW=0,  dW=1;
    int oW = (iW-1)/sW + 1;
    int paddingMode = 2;             // CAUSAL
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iW, iC}, sd::DataType::FLOAT32);
    NDArray weights('c', {kW, iC, oC}, sd::DataType::FLOAT32);

    NDArray expOutput('c', {bS, oW, oC}, {11.000000, 11.600000, 12.200000, 12.800000, 30.099998, 32.200001, 34.299999, 36.400002, 49.899998, 53.800003, 57.699997,
        61.599998, 69.699997, 75.400002, 81.099998, 86.800003, 89.500000, 97.000000, 104.500000, 112.000000, 109.300003, 118.600006, 127.899994, 137.199997, 129.100006,
        140.199997, 151.300003, 162.399994, 148.899994, 161.800003, 174.699997, 187.600006, 133.399994, 141.200012, 149.000000, 156.800003, 188.500000, 205.000000,
        221.500000, 238.000000, 208.299988, 226.600006, 244.899994, 263.200012, 228.100006, 248.200012, 268.299988, 288.399994, 247.899994, 269.799988, 291.700012,
        313.600006, 267.700012, 291.399994, 315.100006, 338.799988, 287.500000, 313.000000, 338.500000, 364.000000, 307.299988, 334.600006, 361.899994, 389.200012}, sd::DataType::FLOAT32);

    input.linspace(1., 1.);
    weights.linspace(0.1, 0.1);

    sd::ops::conv1d op;
    auto results = op.evaluate({&input, &weights}, {kW, sW, pW, dW,  paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv1d_causal_8) {

    int bS=2, iW=8,  iC=3,oC=4,  kW=2,  sW=1,  pW=0,  dW=2;
    int oW = (iW-1)/sW + 1;
    int paddingMode = 2;             // CAUSAL
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iW, iC}, sd::DataType::FLOAT32);
    NDArray weights('c', {kW, iC, oC}, sd::DataType::FLOAT32);

    NDArray expOutput('c', {bS, oW, oC}, {11.000000, 11.600000, 12.200000, 12.800000, 26.299999, 27.799999, 29.299999, 30.799999, 45.399998, 48.399998,
        51.400002, 54.400005, 65.199997, 70.000000, 74.800003, 79.600006, 85.000000, 91.600006, 98.199997, 104.800003, 104.799995, 113.199997, 121.600006,
        130.000000, 124.599998, 134.800003, 145.000000, 155.200012, 144.399994, 156.399994, 168.399994, 180.400009, 133.400009, 141.199997, 149.000000,
        156.800003, 148.699997, 157.400009, 166.099991, 174.800003, 203.800003, 221.200012, 238.599991, 256.000000, 223.599991, 242.799988, 262.000000,
        281.200012, 243.399994, 264.399994, 285.399994, 306.399994, 263.199982, 286.000000, 308.799988, 331.600006, 283.000000, 307.600006, 332.200012,
        356.800018, 302.799988, 329.199982, 355.600006, 382.000000}, sd::DataType::FLOAT32);

    input.linspace(1., 1.);
    weights.linspace(0.1, 0.1);

    sd::ops::conv1d op;
    auto results = op.evaluate({&input, &weights}, {kW, sW, pW, dW,  paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv1d_causal_bp_1) {

    int bS=2, iW=3,  iC=4,oC=3,  kW=2,  sW=1,  pW=0,  dW=1;
    int oW = (iW-1)/sW + 1;
    int paddingMode = 2;             // CAUSAL
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iW, iC});
    NDArray weights('c', {kW, iC, oC});
    NDArray bias('c', {oC}, {-1,-2,-3});
    NDArray gradO('c', {bS, oW, oC});

    input.linspace(1., 1.);
    weights.linspace(0.1, 0.1);
    gradO.linspace(-1.5, 0.1);

    const OpArgsHolder argsHolderFF({&input, &weights, &bias}, {}, {kW, sW, pW, dW,  paddingMode, dataFormat});
    const OpArgsHolder argsHolderBP({&input, &weights, &bias, &gradO}, {}, {kW, sW, pW, dW,  paddingMode, dataFormat});

    sd::ops::conv1d opFF;
    sd::ops::conv1d_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

TEST_F(ConvolutionTests1, Test_Dilation2D_1) {
    auto input = NDArrayFactory::create<double>('c', {2, 6, 6, 3});
    auto weights = NDArrayFactory::create<double>('c', {3, 2, 3});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 3, 3}, {77,   79,   81,   83,   85,   87,   80,   82,   84,  113,  115,  117, 119,  121,  123,  116,  118,  120,  107,  109,  111,  113,  115,  117, 110,  112,  114,  185,  187,  189,  191,  193,  195,  188,  190,  192, 221,  223,  225,  227,  229,  231,  224,  226,  228,  215,  217,  219, 221,  223,  225,  218,  220,  222,});

    input.linspace(1);
    weights.linspace(1);

    sd::ops::dilation2d op;
    auto result = op.evaluate({&input, &weights}, {1, 1,2,2,1, 1,2,2,1});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(ConvolutionTests1, Test_Dilation2D_2) {
    auto input = NDArrayFactory::create<double>('c', {2, 6, 6, 3});
    auto weights = NDArrayFactory::create<double>('c', {3, 2, 3});
    auto exp = NDArrayFactory::create<double>('c', {2, 1, 2, 3}, {95, 97, 99, 101, 103, 105, 203, 205, 207, 209, 211, 213});

    input.linspace(1);
    weights.linspace(1);

    sd::ops::dilation2d op;
    auto result = op.evaluate({&input, &weights}, {0, 1,2,2,1, 1,2,2,1});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv2d_bp_test1) {

    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=4,oW=3;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    auto input = NDArrayFactory::create<TypeParam>('c', {bS, iH, iW, iC});
    auto weights = NDArrayFactory::create<TypeParam>('c', {kH, kW, iC, oC});
    auto bias = NDArrayFactory::create<TypeParam>('c', {oC}, {1,2,3});
    auto gradO = NDArrayFactory::create<TypeParam>('c', {bS, oH, oW, oC});

    auto expGradI = NDArrayFactory::create<TypeParam>('c', {bS, iH, iW, iC},{ 0.226f,  0.343f,  0.46f,  0.577f, 1.172f,  1.46f,  1.748f,  2.036f, 1.892f,  2.288f,  2.684f,  3.08f, 1.284f,  1.581f,  1.878f,  2.175f, 4.458f,  5.133f,  5.808f,  6.483f, 6.186f,  7.023f,  7.86f,  8.697f,
                                                     3.39f,  3.93f,  4.47f,  5.01f, 9.642f, 10.803f, 11.964f, 13.125f,11.37f, 12.693f, 14.016f, 15.339f, 5.266f,  5.707f,  6.148f,  6.589f,12.98f, 13.916f, 14.852f, 15.788f,14.564f, 15.608f, 16.652f, 17.696f,
                                                     3.25f,  4.015f,  4.78f,  5.545f, 9.812f, 11.396f, 12.98f, 14.564f,10.532f, 12.224f, 13.916f, 15.608f, 9.708f, 10.977f, 12.246f, 13.515f,25.194f, 27.813f, 30.432f, 33.051f,26.922f, 29.703f, 32.484f, 35.265f,
                                                    11.814f, 13.326f, 14.838f, 16.35f,30.378f, 33.483f, 36.588f, 39.693f,32.106f, 35.373f, 38.64f, 41.907f,13.474f, 14.563f, 15.652f, 16.741f,31.988f, 34.22f, 36.452f, 38.684f,33.572f, 35.912f, 38.252f, 40.592f});

    auto expGradW = NDArrayFactory::create<TypeParam>('c', {kH, kW, iC, oC},{14.4f, 14.76f, 15.12f,14.4f, 14.76f, 15.12f,14.4f, 14.76f, 15.12f,14.4f, 14.76f, 15.12f, 9.24f,  9.48f,  9.72f, 9.24f,  9.48f,  9.72f, 9.24f,  9.48f,  9.72f, 9.24f,  9.48f,  9.72f,
                                                    17.04f, 17.52f, 18.f,17.04f, 17.52f, 18.f, 17.04f, 17.52f, 18.f, 17.04f, 17.52f, 18.f,10.88f, 11.2f, 11.52f,10.88f, 11.2f, 11.52f,10.88f, 11.2f, 11.52f,10.88f, 11.2f, 11.52f,
                                                    11.16f, 11.52f, 11.88f,11.16f, 11.52f, 11.88f,11.16f, 11.52f, 11.88f,11.16f, 11.52f, 11.88f, 7.08f,  7.32f,  7.56f, 7.08f,  7.32f,  7.56f, 7.08f,  7.32f,  7.56f, 7.08f,  7.32f,  7.56f});
    // auto expGradB('c', {oC},{});

    input = 2.;
    weights.linspace(0.1, 0.1);
    gradO.linspace(0.01, 0.01);

    sd::ops::conv2d_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto gradI = results.at(0);
    auto gradW = results.at(1);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));

}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv2d_bp_test2) {

    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    auto input = NDArrayFactory::create<TypeParam>('c', {bS, iH, iW, iC});
    auto weights = NDArrayFactory::create<TypeParam>('c', {kH, kW, iC, oC});
    auto bias = NDArrayFactory::create<TypeParam>('c', {oC}, {1,2,3});
    auto gradO = NDArrayFactory::create<TypeParam>('c', {bS, oH, oW, oC});

    auto expGradI = NDArrayFactory::create<TypeParam>('c', {bS, iH, iW, iC},{ 0.014f, 0.032f, 0.05f, 0.068f,0.118f,0.181f, 0.244f, 0.307f,0.212f,0.257f, 0.302f, 0.347f,0.208f,0.298f, 0.388f, 0.478f,1.028f,1.262f, 1.496f, 1.73f,1.036f,1.18f, 1.324f, 1.468f,
                                                     0.928f,1.018f, 1.108f, 1.198f,2.9f,3.134f, 3.368f, 3.602f,2.188f,2.332f, 2.476f, 2.62f, 1.202f,1.274f, 1.346f, 1.418f,3.142f,3.313f, 3.484f, 3.655f,2.048f,2.147f, 2.246f, 2.345f,
                                                     0.086f,0.212f, 0.338f, 0.464f,0.694f,0.973f, 1.252f, 1.531f,0.716f,0.869f, 1.022f, 1.175f,1.216f,1.522f, 1.828f, 2.134f,3.908f,4.574f, 5.24f, 5.906f,2.908f,3.268f, 3.628f, 3.988f,
                                                     3.664f,3.97f, 4.276f, 4.582f,9.236f,9.902f,10.568f,11.234f,5.788f,6.148f, 6.508f, 6.868f,3.002f,3.182f, 3.362f, 3.542f,7.174f,7.561f, 7.948f, 8.335f,4.28f,4.487f, 4.694f, 4.901f});

    auto expGradW = NDArrayFactory::create<TypeParam>('c', {kH, kW, iC, oC},{1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,
                                                    1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,
                                                    1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f,1.84f, 2.f, 2.16f});
    // auto expGradB('c', {oC},{});

    input = 2.;
    weights.linspace(0.1, 0.1);
    gradO.linspace(0.01, 0.01);

    sd::ops::conv2d_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto gradI = results.at(0);
    auto gradW = results.at(1);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv2d_bp_test3) {
    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    auto input = NDArrayFactory::create<TypeParam>('c', {bS, iC, iH, iW});
    auto weights = NDArrayFactory::create<TypeParam>('c', {oC, iC, kH, kW});
    auto bias = NDArrayFactory::create<TypeParam>('c', {oC}, {1,2,3});
    auto gradO = NDArrayFactory::create<TypeParam>('c', {bS, oC, oH, oW});

    auto expGradI = NDArrayFactory::create<TypeParam>('c', {bS, iC, iH, iW},{ 0.567f, 1.224f, 0.66f, 1.314f, 2.82f, 1.512f, 1.386f, 2.976f, 1.596f, 0.801f, 1.71f, 0.912f, 0.657f, 1.422f, 0.768f, 1.53f, 3.288f, 1.764f, 1.602f, 3.444f, 1.848f, 0.927f, 1.98f, 1.056f,
                                                     0.747f, 1.62f, 0.876f, 1.746f, 3.756f, 2.016f, 1.818f, 3.912f, 2.1f, 1.053f, 2.25f, 1.2f, 0.837f, 1.818f, 0.984f, 1.962f, 4.224f, 2.268f, 2.034f, 4.38f, 2.352f, 1.179f, 2.52f, 1.344f,
                                                     1.467f, 3.06f, 1.596f, 3.186f, 6.636f, 3.456f, 3.402f, 7.08f, 3.684f, 1.845f, 3.834f, 1.992f, 1.773f, 3.69f, 1.92f, 3.834f, 7.968f, 4.14f, 4.05f, 8.412f, 4.368f, 2.187f, 4.536f, 2.352f,
                                                     2.079f, 4.32f, 2.244f, 4.482f, 9.3f, 4.824f, 4.698f, 9.744f, 5.052f, 2.529f, 5.238f, 2.712f, 2.385f, 4.95f, 2.568f, 5.13f, 10.632f, 5.508f, 5.346f, 11.076f, 5.736f, 2.871f, 5.94f, 3.072f});

    auto expGradW = NDArrayFactory::create<TypeParam>('c', {oC, iC, kH, kW},{1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f,
                                                    1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 1.3600e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f,
                                                    2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.0000e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f,
                                                    2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f, 2.6400e+00f});
    auto expGradB = NDArrayFactory::create<TypeParam>('c', {oC},{0.68f, 1.f, 1.32f});

    input = 2.;
    weights.linspace(0.1, 0.1);
    gradO.linspace(0.01, 0.01);
    weights.permutei({2,3,1,0});
    expGradW.permutei({2,3,1,0});

    sd::ops::conv2d_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto gradI = results.at(0);
    auto gradW = results.at(1);
    auto gradB = results.at(2);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));

    ASSERT_TRUE(expGradB.isSameShape(gradB));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv2d_bp_4) {

    int bS=1, iH=7,iW=1,  iC=2,oC=3,  kH=2,kW=1,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=7,oW=1;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iC, iH, iW}, sd::DataType::FLOAT32);
    NDArray weights('c', {kH, kW, iC, oC}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, {1,2,3}, sd::DataType::FLOAT32);
    NDArray gradO('c', {bS, oC, oH, oW}, sd::DataType::FLOAT32);

    NDArray gradI('c', {bS, iC, iH, iW}, sd::DataType::FLOAT32);
    NDArray gradW('c', {kH, kW, iC, oC}, sd::DataType::FLOAT32);
    NDArray gradB('c', {oC}, sd::DataType::FLOAT32);


    input = 2.;
    weights.linspace(0.1, 0.1);
    gradO.linspace(0.01, 0.01);

    sd::ops::conv2d_bp op;
    auto status = op.execute({&input, &weights, &bias, &gradO}, {&gradI, &gradW, &gradB}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat}, {});

    ASSERT_EQ(Status::OK(), status);
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv2d_bp_5) {

    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW
    int wFormat     = 1;             // 0-[kH, kW, iC, oC], 1-[oC, iC, kH, kW], 2-[oC, kH, kW, iC]

    NDArray input('c', {bS, iC, iH, iW}, sd::DataType::FLOAT32);
    NDArray weights('c', {oC, iC, kH, kW}, {3.6, 2.4, 1.2, 0.0, -1.2, -2.4, 3.3, 2.1, 0.9, -0.3, -1.5, -2.7, 3.0, 1.8, 0.6, -0.6, -1.8, -3.0, 2.7, 1.5, 0.3, -0.9, -2.1, -3.3, 3.5, 2.3, 1.1, -0.1, -1.3, -2.5, 3.2, 2.0, 0.8, -0.4, -1.6, -2.8, 2.9, 1.7, 0.5, -0.7, -1.9, -3.1, 2.6, 1.4, 0.2, -1.0, -2.2, -3.4, 3.4, 2.2, 1.0, -0.2, -1.4, -2.6, 3.1, 1.9, 0.7, -0.5, -1.7, -2.9, 2.8, 1.6, 0.4, -0.8, -2.0, -3.2, 2.5, 1.3, 0.1, -1.1, -2.3, -3.5}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, {1,-0.5, 0.1}, sd::DataType::FLOAT32);
    NDArray gradO('c', {bS, oC, oH, oW}, sd::DataType::FLOAT32);

    NDArray expGradI('c', {bS, iC, iH, iW},{0.517, 0.959, 0.406, 0.884, 1.474, 0.518, 0.020, -0.398, -0.490, -0.281, -0.853, -0.608, 0.472, 0.860, 0.352, 0.776, 1.240,
        0.392, -0.088, -0.632, -0.616, -0.344, -0.988, -0.680, 0.427, 0.761, 0.298, 0.668, 1.006, 0.266, -0.196, -0.866, -0.742, -0.407, -1.123, -0.752, 0.382, 0.662,
        0.244, 0.560, 0.772, 0.140, -0.304, -1.100, -0.868, -0.470, -1.258, -0.824, 1.777, 3.047, 1.234, 2.540, 3.922, 1.310, -0.052, -1.406, -1.426, -0.749, -2.221,
        -1.508, 1.624, 2.732, 1.072, 2.216, 3.256, 0.968, -0.376, -2.072, -1.768, -0.920, -2.572, -1.688, 1.471, 2.417, 0.910, 1.892, 2.590, 0.626, -0.700, -2.738, -2.110,
        -1.091, -2.923, -1.868, 1.318, 2.102, 0.748, 1.568, 1.924, 0.284, -1.024, -3.404, -2.452, -1.262, -3.274, -2.048}, sd::DataType::FLOAT32);

    NDArray expGradW('c', {oC, iC, kH, kW},{-3.3, -2.62, -1.26, -0.58, 0.78, 1.46, 4.86, 5.54, 6.9, 7.58, 8.940001, 9.619999, 13.02, 13.700001, 15.06, 15.74, 17.1,
        17.780001, 21.18, 21.860001, 23.219999, 23.900002, 25.259998, 25.940001, -10.340001, -9.34, -7.339999, -6.34, -4.339999, -3.339999, 1.66, 2.66, 4.660001,
        5.660001, 7.66, 8.66, 13.66, 14.660001, 16.66, 17.66, 19.66, 20.66, 25.66, 26.66, 28.66, 29.66, 31.66, 32.66, -17.380001, -16.059999, -13.420003, -12.099999,
        -9.46, -8.139999, -1.540001, -0.219999, 2.419999, 3.739999, 6.379999, 7.7, 14.299999, 15.62, 18.26, 19.58, 22.219999, 23.539999, 30.139999, 31.459999, 34.099998,
        35.419998, 38.060001, 39.380001}, sd::DataType::FLOAT32);

    NDArray expGradB('c', {oC}, {0.68, 1., 1.32}, sd::DataType::FLOAT32);

    input.linspace(-48, 1);
    // weights.linspace(3.6, -0.1);
    gradO.linspace(0.01, 0.01);

    sd::ops::conv2d_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat, wFormat});
    auto gradI = results.at(0);
    auto gradW = results.at(1);
    auto gradB = results.at(2);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));

    ASSERT_TRUE(expGradB.isSameShape(gradB));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv2d_bp_6) {

    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=4,oW=3;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW
    int wFormat     = 2;             // 0-[kH, kW, iC, oC], 1-[oC, iC, kH, kW], 2-[oC, kH, kW, iC]

    NDArray input('c', {bS, iH, iW, iC}, sd::DataType::FLOAT32);
    NDArray weights('c', {oC, kH, kW, iC}, {3.6, 0.0, 3.3, -0.3, 3.0, -0.6, 2.7, -0.9, 3.5, -0.1, 3.2, -0.4, 2.9, -0.7, 2.6, -1.0, 3.4, -0.2, 3.1, -0.5, 2.8, -0.8, 2.5, -1.1, 2.4, -1.2, 2.1, -1.5, 1.8, -1.8, 1.5, -2.1, 2.3, -1.3, 2.0, -1.6, 1.7, -1.9, 1.4, -2.2, 2.2, -1.4, 1.9, -1.7, 1.6, -2.0, 1.3, -2.3, 1.2, -2.4, 0.9, -2.7, 0.6, -3.0, 0.3, -3.3, 1.1, -2.5, 0.8, -2.8, 0.5, -3.1, 0.2, -3.4, 1.0, -2.6, 0.7, -2.9, 0.4, -3.2, 0.1, -3.5}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, {1,-0.5, 0.1}, sd::DataType::FLOAT32);
    NDArray gradO('c', {bS, oH, oW, oC}, sd::DataType::FLOAT32);

    NDArray expGradI('c', {bS, iH, iW, iC}, {0.882, -0.522, 0.765, -0.639, 1.953, -1.503, 1.665, -1.791, 2.691, -2.061, 2.295, -2.457, 2.259, -1.305, 1.962, -1.602, 4.545,
        -3.555, 3.870, -4.230, 5.625, -4.419, 4.788, -5.256001, 4.122, -2.358, 3.582, -2.898, 7.785, -6.147, 6.624, -7.308, 8.865, -7.011, 7.541999, -8.334, 3.273, -2.019,
        2.832, -2.460, 6.069, -5.163, 5.133, -6.099, 6.771, -5.757, 5.727, -6.801, 5.958, -3.222, 5.193, -3.987, 10.809, -8.198999, 9.225, -9.783, 11.547, -8.757, 9.855,
        -10.448999, 9.711, -5.517, 8.441999, -6.786, 17.505001, -13.922999, 14.886, -16.542, 18.585001, -14.787001, 15.804001, -17.568001, 11.574, -6.570, 10.062, -8.082,
        20.745001, -16.514999, 17.639999, -19.619999, 21.825001, -17.379002, 18.558001, -20.646, 8.133, -4.935, 7.044, -6.024, 14.492998, -12.291, 12.261, -14.523001, 15.195001, -12.885, 12.855, -15.225}, sd::DataType::FLOAT32);

    NDArray expGradW('c', {oC, kH, kW, iC},{34.559998, 41.760010, 48.959999, 56.160004, 33.119999, 37.739998, 42.360001, 46.979996, 120.960007, 129.480011, 138.0, 146.519989,
        91.200005, 96.639999, 102.079994, 107.520004, 114.479996, 120.059998, 125.639999, 131.220001, 82.080002, 85.620003, 89.160004, 92.699997, 33.120003, 40.499996,
        47.879993, 55.260002, 32.399998, 37.139996, 41.880001, 46.620003, 120.479988, 129.240005, 137.999985, 146.759995, 91.199997, 96.799995, 102.399994, 108.0, 115.199989,
        120.959999, 126.720001, 132.479996, 82.799995, 86.460007, 90.119995, 93.779999, 31.679998, 39.239994, 46.800003, 54.359997, 31.680000, 36.540001, 41.400002, 46.260002,
        120.0, 129.0, 138.0, 147.0, 91.200005, 96.960007, 102.720001, 108.480003, 115.919998, 121.860001, 127.799988, 133.740005, 83.520004, 87.300003, 91.080002, 94.860001}, sd::DataType::FLOAT32);

    NDArray expGradB('c', {oC}, {8.520, 8.760, 9.}, sd::DataType::FLOAT32);

    input.linspace(-48, 1);
    gradO.linspace(0.01, 0.01);

    sd::ops::conv2d_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat, wFormat});
    auto gradI = results.at(0);
    auto gradW = results.at(1);
    auto gradB = results.at(2);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));

    ASSERT_TRUE(expGradB.isSameShape(gradB));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
}

////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_bp_test1) {

    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=3,oH=4,oW=3;
    int paddingMode = 1;             // 1-SAME,  0-VALID;
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iD, iH, iW, iC});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC});
    auto bias     = NDArrayFactory::create<TypeParam>('c', {oC}, {1,2,3});
    auto gradO    = NDArrayFactory::create<TypeParam>('c', {bS, oD, oH, oW, oC});

    auto expGradI = NDArrayFactory::create<TypeParam>('c', {bS, iD, iH, iW, iC},{0.226f, 0.343f, 0.46f, 0.577f, 1.172f, 1.46f, 1.748f, 2.036f, 1.892f, 2.288f, 2.684f, 3.08f, 1.284f, 1.581f, 1.878f, 2.175f, 4.458f, 5.133f, 5.808f, 6.483f, 6.186f, 7.023f, 7.86f, 8.697f, 3.39f, 3.93f, 4.47f, 5.01f, 9.642f, 10.803f, 11.964f, 13.125f, 11.37f, 12.693f, 14.016f, 15.339f,
                                                        5.266f, 5.707f, 6.148f, 6.589f, 12.98f, 13.916f, 14.852f, 15.788f, 14.564f, 15.608f, 16.652f, 17.696f, 6.284f, 7.166f, 8.048f, 8.93f, 17.896f, 19.768f, 21.64f, 23.512f, 21.928f, 24.016f, 26.104f, 28.192f, 18.12f, 19.686f, 21.252f, 22.818f, 45.852f, 49.146f, 52.44f, 55.734f, 53.196f, 56.814f, 60.432f, 64.05f,
                                                       28.164f, 30.216f, 32.268f, 34.32f, 67.884f, 72.15f, 76.416f, 80.682f, 75.228f, 79.818f, 84.408f, 88.998f, 29.324f, 30.854f, 32.384f, 33.914f, 67.432f, 70.6f, 73.768f, 76.936f, 73.192f, 76.576f, 79.96f, 83.344f, 27.884f, 30.062f, 32.24f, 34.418f, 66.28f, 70.744f, 75.208f, 79.672f, 70.312f, 74.992f, 79.672f, 84.352f,
                                                       58.296f, 61.806f, 65.316f, 68.826f, 133.98f, 141.162f, 148.344f, 155.526f, 141.324f, 148.83f, 156.336f, 163.842f, 68.34f, 72.336f, 76.332f, 80.328f, 156.012f, 164.166f, 172.32f, 180.474f, 163.356f, 171.834f, 180.312f, 188.79f, 61.292f, 64.118f, 66.944f, 69.77f, 136.552f, 142.312f, 148.072f, 153.832f, 142.312f, 148.288f, 154.264f, 160.24f,
                                                        9.298f, 11.359f, 13.42f, 15.481f, 27.092f, 31.268f, 35.444f, 39.62f, 27.812f, 32.096f, 36.38f, 40.664f, 26.556f, 29.769f, 32.982f, 36.195f, 66.666f, 73.173f, 79.68f, 86.187f, 68.394f, 75.063f, 81.732f, 88.401f, 28.662f, 32.118f, 35.574f, 39.03f, 71.85f, 78.843f, 85.836f, 92.829f, 73.578f, 80.733f, 87.888f, 95.043f,
                                                       29.89f, 32.275f, 34.66f, 37.045f, 70.004f, 74.828f, 79.652f, 84.476f, 71.588f, 76.52f, 81.452f, 86.384f, 71.084f, 75.854f, 80.624f, 85.394f, 163.048f, 172.696f, 182.344f, 191.992f, 167.08f, 176.944f, 186.808f, 196.672f, 138.648f, 146.046f, 153.444f, 160.842f, 310.236f, 325.194f, 340.152f, 355.11f, 317.58f, 332.862f, 348.144f, 363.426f,
                                                      148.692f, 156.576f, 164.46f, 172.344f, 332.268f, 348.198f, 364.128f, 380.058f, 339.612f, 355.866f, 372.12f, 388.374f, 125.228f, 130.646f, 136.064f, 141.482f, 274.792f, 285.736f, 296.68f, 307.624f, 280.552f, 291.712f, 302.872f, 314.032f, 92.684f, 98.75f, 104.816f, 110.882f, 211.432f, 223.672f, 235.912f, 248.152f, 215.464f, 227.92f, 240.376f, 252.832f,
                                                      178.824f, 188.166f, 197.508f, 206.85f, 398.364f, 417.21f, 436.056f, 454.902f, 405.708f, 424.878f, 444.048f, 463.218f, 188.868f, 198.696f, 208.524f, 218.352f, 420.396f, 440.214f, 460.032f, 479.85f, 427.74f, 447.882f, 468.024f, 488.166f, 157.196f, 163.91f, 170.624f, 177.338f, 343.912f, 357.448f, 370.984f, 384.52f, 349.672f, 363.424f, 377.176f, 390.928f});

    auto expGradW = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC},{120.96f, 122.04f, 123.12f, 120.96f, 122.04f, 123.12f, 120.96f, 122.04f, 123.12f, 120.96f, 122.04f, 123.12f, 79.56f, 80.28f, 81.f, 79.56f, 80.28f, 81.f, 79.56f, 80.28f, 81.f, 79.56f, 80.28f, 81.f,
                                                        154.8f, 156.24f, 157.68f, 154.8f, 156.24f, 157.68f, 154.8f, 156.24f, 157.68f, 154.8f, 156.24f, 157.68f, 101.76f, 102.72f, 103.68f, 101.76f, 102.72f, 103.68f, 101.76f, 102.72f, 103.68f, 101.76f, 102.72f, 103.68f,
                                                        111.24f, 112.32f, 113.4f, 111.24f, 112.32f, 113.4f, 111.24f, 112.32f, 113.4f, 111.24f, 112.32f, 113.4f, 73.08f, 73.8f, 74.52f, 73.08f, 73.8f, 74.52f, 73.08f, 73.8f, 74.52f, 73.08f, 73.8f, 74.52f,
                                                         67.68f, 68.4f, 69.12f, 67.68f, 68.4f, 69.12f, 67.68f, 68.4f, 69.12f, 67.68f, 68.4f, 69.12f, 44.4f, 44.88f, 45.36f, 44.4f, 44.88f, 45.36f, 44.4f, 44.88f, 45.36f, 44.4f, 44.88f, 45.36f,
                                                         85.92f, 86.88f, 87.84f, 85.92f, 86.88f, 87.84f, 85.92f, 86.88f, 87.84f, 85.92f, 86.88f, 87.84f, 56.32f, 56.96f, 57.6f, 56.32f, 56.96f, 57.6f, 56.32f, 56.96f, 57.6f, 56.32f, 56.96f, 57.6f,
                                                         61.2f, 61.92f, 62.64f, 61.2f, 61.92f, 62.64f, 61.2f, 61.92f, 62.64f, 61.2f, 61.92f, 62.64f, 40.08f, 40.56f, 41.04f, 40.08f, 40.56f, 41.04f, 40.08f, 40.56f, 41.04f, 40.08f, 40.56f, 41.04f});
    // auto expGradB('c', {oC},{});

    input = 2.;
    weights.linspace(0.1, 0.1);
    gradO.linspace(0.01, 0.01);

    sd::ops::conv3dnew_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto gradI = results.at(0);
    auto gradW = results.at(1);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));


}


////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_bp_test2) {

    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME,  0-VALID;
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iD, iH, iW, iC});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC});
    auto bias     = NDArrayFactory::create<TypeParam>('c', {oC}, {1,2,3});
    auto gradO    = NDArrayFactory::create<TypeParam>('c', {bS, oD, oH, oW, oC});

    auto expGradI = NDArrayFactory::create<TypeParam>('c', {bS, iD, iH, iW, iC},{ 0.014f, 0.032f, 0.05f, 0.068f, 0.118f, 0.181f, 0.244f, 0.307f, 0.212f, 0.257f, 0.302f, 0.347f, 0.208f, 0.298f, 0.388f, 0.478f, 1.028f, 1.262f, 1.496f, 1.73f, 1.036f, 1.18f, 1.324f, 1.468f, 0.928f, 1.018f, 1.108f, 1.198f, 2.9f, 3.134f, 3.368f, 3.602f, 2.188f, 2.332f, 2.476f, 2.62f,
                                                         1.202f, 1.274f, 1.346f, 1.418f, 3.142f, 3.313f, 3.484f, 3.655f, 2.048f, 2.147f, 2.246f, 2.345f, 0.532f, 0.676f, 0.82f, 0.964f, 2.324f, 2.666f, 3.008f, 3.35f, 2.008f, 2.206f, 2.404f, 2.602f, 3.584f, 3.98f, 4.376f, 4.772f, 10.552f, 11.452f, 12.352f, 13.252f, 7.4f, 7.904f, 8.408f, 8.912f,
                                                         6.752f, 7.148f, 7.544f, 7.94f, 17.752f, 18.652f, 19.552f, 20.452f, 11.432f, 11.936f, 12.44f, 12.944f, 5.932f, 6.184f, 6.436f, 6.688f, 14.42f, 14.978f, 15.536f, 16.094f, 8.704f, 9.01f, 9.316f, 9.622f, 3.11f, 3.236f, 3.362f, 3.488f, 7.39f, 7.669f, 7.948f, 8.227f, 4.388f, 4.541f, 4.694f, 4.847f,
                                                         8.56f, 8.866f, 9.172f, 9.478f, 19.892f, 20.558f, 21.224f, 21.89f, 11.548f, 11.908f, 12.268f, 12.628f, 11.008f, 11.314f, 11.62f, 11.926f, 25.22f, 25.886f, 26.552f, 27.218f, 14.428f, 14.788f, 15.148f, 15.508f, 7.322f, 7.502f, 7.682f, 7.862f, 16.462f, 16.849f, 17.236f, 17.623f, 9.248f, 9.455f, 9.662f, 9.869f,
                                                         0.158f, 0.392f, 0.626f, 0.86f, 1.27f, 1.765f, 2.26f, 2.755f, 1.22f, 1.481f, 1.742f, 2.003f, 2.224f, 2.746f, 3.268f, 3.79f, 6.788f, 7.886f, 8.984f, 10.082f, 4.78f, 5.356f, 5.932f, 6.508f, 6.4f, 6.922f, 7.444f, 7.966f, 15.572f, 16.67f, 17.768f, 18.866f, 9.388f, 9.964f, 10.54f, 11.116f,
                                                         4.802f, 5.09f, 5.378f, 5.666f, 11.206f, 11.809f, 12.412f, 13.015f, 6.512f, 6.827f, 7.142f, 7.457f, 6.004f, 6.58f, 7.156f, 7.732f, 14.996f, 16.202f, 17.408f, 18.614f, 9.208f, 9.838f, 10.468f, 11.098f, 17.984f, 19.244f, 20.504f, 21.764f, 42.808f, 45.436f, 48.064f, 50.692f, 25.256f, 26.624f, 27.992f, 29.36f,
                                                        28.064f, 29.324f, 30.584f, 31.844f, 63.832f, 66.46f, 69.088f, 71.716f, 36.2f, 37.568f, 38.936f, 40.304f, 18.316f, 19.f, 19.684f, 20.368f, 40.916f, 42.338f, 43.76f, 45.182f, 22.816f, 23.554f, 24.292f, 25.03f, 8.438f, 8.78f, 9.122f, 9.464f, 18.91f, 19.621f, 20.332f, 21.043f, 10.58f, 10.949f, 11.318f, 11.687f,
                                                        20.944f, 21.682f, 22.42f, 23.158f, 46.388f, 47.918f, 49.448f, 50.978f, 25.66f, 26.452f, 27.244f, 28.036f, 26.848f, 27.586f, 28.324f, 29.062f, 58.628f, 60.158f, 61.688f, 63.218f, 31.996f, 32.788f, 33.58f, 34.372f, 16.106f, 16.502f, 16.898f, 17.294f, 34.894f, 35.713f, 36.532f, 37.351f, 18.896f, 19.319f, 19.742f, 20.165f});

    auto expGradW = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC},{7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f,
                                                        7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f,
                                                        7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f,
                                                        7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f, 7.52f, 7.84f, 8.16f});
    // auto expGradB('c', {oC},{});

    input = 2.;
    weights.linspace(0.1, 0.1);
    gradO.linspace(0.01, 0.01);

    sd::ops::conv3dnew_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto gradI = results.at(0);
    auto gradW = results.at(1);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));


}

////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_bp_test3) {

    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME,  0-VALID;
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iC, iD, iH, iW});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {oC, iC, kD, kH, kW});
    auto bias     = NDArrayFactory::create<TypeParam>('c', {oC}, {1,2,3});
    auto gradO    = NDArrayFactory::create<TypeParam>('c', {bS, oC, oD, oH, oW});

    auto expGradI = NDArrayFactory::create<TypeParam>('c', {bS, iC, iD, iH, iW},{2.091f, 4.356f, 2.268f, 4.53f, 9.42f, 4.896f, 4.65f, 9.672f, 5.028f, 2.517f, 5.226f, 2.712f, 4.932f, 10.242f, 5.316f, 10.62f, 22.02f, 11.412f, 10.908f, 22.62f, 11.724f, 5.868f, 12.15f, 6.288f, 2.913f, 6.03f, 3.12f, 6.234f, 12.888f, 6.66f, 6.402f, 13.236f, 6.84f, 3.423f, 7.068f, 3.648f,
                                                        2.415f, 5.04f, 2.628f, 5.25f, 10.932f, 5.688f, 5.37f, 11.184f, 5.82f, 2.913f, 6.054f, 3.144f, 5.724f, 11.898f, 6.18f, 12.348f, 25.62f, 13.284f, 12.636f, 26.22f, 13.596f, 6.804f, 14.094f, 7.296f, 3.381f, 7.002f, 3.624f, 7.242f, 14.976f, 7.74f, 7.41f, 15.324f, 7.92f, 3.963f, 8.184f, 4.224f,
                                                        2.739f, 5.724f, 2.988f, 5.97f, 12.444f, 6.48f, 6.09f, 12.696f, 6.612f, 3.309f, 6.882f, 3.576f, 6.516f, 13.554f, 7.044f, 14.076f, 29.22f, 15.156f, 14.364f, 29.82f, 15.468f, 7.74f, 16.038f, 8.304f, 3.849f, 7.974f, 4.128f, 8.25f, 17.064f, 8.82f, 8.418f, 17.412f, 9.f, 4.503f, 9.3f, 4.8f,
                                                        3.063f, 6.408f, 3.348f, 6.69f, 13.956f, 7.272f, 6.81f, 14.208f, 7.404f, 3.705f, 7.71f, 4.008f, 7.308f, 15.21f, 7.908f, 15.804f, 32.82f, 17.028f, 16.092f, 33.42f, 17.34f, 8.676f, 17.982f, 9.312f, 4.317f, 8.946f, 4.632f, 9.258f, 19.152f, 9.9f, 9.426f, 19.5f, 10.08f, 5.043f, 10.416f, 5.376f,
                                                        5.619f, 11.484f, 5.868f, 11.73f, 23.964f, 12.24f, 12.138f, 24.792f, 12.66f, 6.333f, 12.93f, 6.6f, 12.42f, 25.362f, 12.948f, 25.884f, 52.836f, 26.964f, 26.748f, 54.588f, 27.852f, 13.932f, 28.422f, 14.496f, 6.873f, 14.022f, 7.152f, 14.298f, 29.16f, 14.868f, 14.754f, 30.084f, 15.336f, 7.671f, 15.636f, 7.968f,
                                                        6.807f, 13.896f, 7.092f, 14.178f, 28.932f, 14.76f, 14.586f, 29.76f, 15.18f, 7.593f, 15.486f, 7.896f, 14.94f, 30.474f, 15.54f, 31.068f, 63.348f, 32.292f, 31.932f, 65.1f, 33.18f, 16.596f, 33.822f, 17.232f, 8.205f, 16.722f, 8.52f, 17.034f, 34.704f, 17.676f, 17.49f, 35.628f, 18.144f, 9.075f, 18.48f, 9.408f,
                                                        7.995f, 16.308f, 8.316f, 16.626f, 33.9f, 17.28f, 17.034f, 34.728f, 17.7f, 8.853f, 18.042f, 9.192f, 17.46f, 35.586f, 18.132f, 36.252f, 73.86f, 37.62f, 37.116f, 75.612f, 38.508f, 19.26f, 39.222f, 19.968f, 9.537f, 19.422f, 9.888f, 19.77f, 40.248f, 20.484f, 20.226f, 41.172f, 20.952f, 10.479f, 21.324f, 10.848f,
                                                        9.183f, 18.72f, 9.54f, 19.074f, 38.868f, 19.8f, 19.482f, 39.696f, 20.22f, 10.113f, 20.598f, 10.488f, 19.98f, 40.698f, 20.724f, 41.436f, 84.372f, 42.948f, 42.3f, 86.124f, 43.836f, 21.924f, 44.622f, 22.704f, 10.869f, 22.122f, 11.256f, 22.506f, 45.792f, 23.292f, 22.962f, 46.716f, 23.76f, 11.883f, 24.168f, 12.288f});

    auto expGradW = NDArrayFactory::create<TypeParam>('c', {oC, iC, kD, kH, kW},{5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f,
                                                        5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f, 5.28f,
                                                        7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f,
                                                        7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f, 7.84f,
                                                        10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f,
                                                        10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f, 10.4f});

    auto expGradB = NDArrayFactory::create<TypeParam>('c', {oC},{2.64f, 3.92f, 5.2f});

    input = 2.;
    weights.linspace(0.1, 0.1);
    gradO.linspace(0.01, 0.01);
    weights.permutei({2, 3, 4, 1, 0});
    expGradW.permutei({2, 3, 4, 1, 0});

    sd::ops::conv3dnew_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto* gradI = results.at(0);
    auto* gradW = results.at(1);
    auto* gradB = results.at(2);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));

    ASSERT_TRUE(expGradB.isSameShape(gradB));
    ASSERT_TRUE(expGradB.equalsTo(gradB));


}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv3d_bp_test4) {

    int bS=2, iD=4,iH=3,iW=3,  iC=4,oC=3,  kD=3,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW
    int wFormat     = 1;             // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

    NDArray input('c', {bS, iC, iD, iH, iW}, sd::DataType::FLOAT32);
    NDArray weights('c', {oC, iC, kD, kH, kW}, {7., 5.8, 4.6, 3.4, 2.2, 1., -0.2, -1.4, -2.6, -3.8, -5., -6.2, 6.7, 5.5, 4.3, 3.1, 1.9, 0.7, -0.5, -1.7, -2.9, -4.1,
        -5.3, -6.5, 6.4, 5.2, 4., 2.8, 1.6, 0.4, -0.8, -2., -3.2, -4.4, -5.6, -6.8, 6.1, 4.9, 3.7, 2.5, 1.3, 0.1, -1.1, -2.3, -3.5, -4.7, -5.9, -7.1, 6.9, 5.7, 4.5,
        3.3, 2.1, 0.9, -0.3, -1.5, -2.7, -3.9, -5.1, -6.3, 6.6, 5.4, 4.2, 3., 1.8, 0.6, -0.6, -1.8, -3., -4.2, -5.4, -6.6, 6.3, 5.1, 3.9, 2.7, 1.5, 0.3, -0.9, -2.1,
        -3.3, -4.5, -5.7, -6.9, 6., 4.8, 3.6, 2.4, 1.2, 0., -1.2, -2.4, -3.6, -4.8, -6., -7.2, 6.8, 5.6, 4.4, 3.2, 2., 0.8, -0.4, -1.6, -2.8, -4., -5.2, -6.4, 6.5, 5.3, 4.1, 2.9, 1.7, 0.5, -0.7, -1.9, -3.1, -4.3, -5.5, -6.7, 6.2, 5., 3.8, 2.6, 1.4, 0.2, -1., -2.2, -3.4, -4.6, -5.8, -7., 5.9, 4.7, 3.5, 2.3, 1.1, -0.1, -1.3, -2.5, -3.7, -4.9, -6.1, -7.3}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, {1,-0.5, 0.1}, sd::DataType::FLOAT32);
    NDArray gradO('c', {bS, oC, oD, oH, oW}, sd::DataType::FLOAT32);

    NDArray expGradI('c', {bS, iC, iD, iH, iW},{1.847, 3.577, 1.694, 3.460, 6.542, 3.010, 1.469, 2.677, 1.172, 3.226, 5.929999, 2.632, 5.408, 9.483999, 3.932, 1.894,
        2.978, 1.012, 0.058, -0.694, -0.824, -1.504, -4.916, -3.556, -1.850, -4.798, -3.020, -1.069, -2.687, -1.654, -3.236, -7.714, -4.550, -2.311, -5.315, -3.040,
        1.766, 3.406, 1.604, 3.280, 6.164, 2.812, 1.370, 2.470, 1.064, 3.028, 5.516, 2.416, 4.976, 8.584001, 3.464, 1.660, 2.492, 0.760, -0.140, -1.108, -1.040, -1.936,
         -5.816, -4.024, -2.084, -5.284, -3.272, -1.186, -2.930, -1.780, -3.488, -8.236, -4.820, -2.446, -5.594, -3.184, 1.685, 3.235, 1.514, 3.100, 5.786, 2.614, 1.271,
         2.263, 0.956, 2.830, 5.102, 2.200, 4.544001, 7.683999, 2.996, 1.426, 2.006, 0.508, -0.338, -1.522, -1.256, -2.368, -6.716, -4.492, -2.318, -5.770, -3.524, -1.303,
         -3.173, -1.906, -3.740, -8.757999, -5.090, -2.581, -5.873, -3.328, 1.604, 3.064, 1.424, 2.920, 5.408, 2.416, 1.172, 2.056, 0.848, 2.632, 4.688, 1.984, 4.112, 6.784, 2.528, 1.192, 1.520, 0.256, -0.536, -1.936, -1.472, -2.800, -7.616, -4.960, -2.552, -6.256, -3.776, -1.420, -3.416, -2.032, -3.992, -9.280001, -5.360, -2.716, -6.152, -3.472, 6.815001, 12.649, 5.798, 11.668, 21.230, 9.490, 4.709, 8.292999, 3.548, 9.706, 17.162001, 7.384, 14.912001, 25.036001, 9.980001, 4.918, 7.298, 2.308, -0.374, -3.286, -2.984, -5.824, -17.012001, -11.332001, -5.738, -14.302, -8.636, -3.013, -7.439, -4.462, -8.852, -20.674, -11.894, -5.983, -13.523, -7.576, 6.518, 12.046, 5.492, 11.056, 19.988001, 8.860001, 4.394, 7.654, 3.224, 9.075999, 15.883999, 6.736001, 13.616, 22.407999, 8.648, 4.252, 5.947999, 1.624, -1.004, -4.564, -3.632, -7.120, -19.639999, -12.664001, -6.404, -15.652, -9.320, -3.346, -8.114, -4.804, -9.536, -22.059999, -12.596, -6.334, -14.233999, -7.936, 6.221, 11.443, 5.186, 10.444, 18.746, 8.230, 4.079, 7.015, 2.900, 8.446, 14.606001, 6.088, 12.320, 19.779999, 7.316, 3.586, 4.598001, 0.940, -1.634, -5.842, -4.280, -8.416, -22.268002, -13.996, -7.070001, -17.001999, -10.004001, -3.679, -8.789, -5.146, -10.220, -23.445999, -13.298, -6.684999, -14.945, -8.296, 5.924, 10.840, 4.880, 9.832001, 17.504, 7.600, 3.764, 6.376, 2.576, 7.816, 13.328, 5.440001, 11.024, 17.152, 5.983999, 2.920, 3.247999, 0.256, -2.264, -7.120, -4.928, -9.712, -24.896, -15.328, -7.736, -18.352001, -10.688, -4.012, -9.464, -5.488, -10.903999, -24.832001, -14.000, -7.035999, -15.656, -8.655999}, sd::DataType::FLOAT32);

    NDArray expGradW('c', {oC, iC, kD, kH, kW},{-24.399998, -23.080000, -20.440001, -19.119999, -12.519999, -11.199998, -8.560001, -7.240002, -0.639999, 0.679999,
        3.320001, 4.640001, 23.119999, 24.439999, 27.080002, 28.400002, 35.000000, 36.320000, 38.959999, 40.279999, 46.879997, 48.200005, 50.839996, 52.160004,
        70.639999, 71.959999, 74.599998, 75.919998, 82.520004, 83.840004, 86.479996, 87.800003, 94.399994, 95.719994, 98.360001, 99.680008, 118.160004, 119.479996,
        122.120003, 123.440010, 130.040009, 131.360001, 134.000000, 135.319992, 141.919998, 143.239990, 145.879990, 147.200012, -70.159996, -68.200005, -64.279999,
        -62.319996, -52.519993, -50.559994, -46.640003, -44.680000, -34.880001, -32.919998, -29.000002, -27.040005, 0.400004, 2.359996, 6.279998, 8.240004, 18.040001,
         20.000000, 23.920002, 25.879999, 35.680000, 37.639996, 41.560001, 43.520000, 70.959999, 72.919998, 76.840004, 78.799995, 88.599998, 90.560005, 94.479996, 96.440002, 106.240005, 108.199997, 112.120003, 114.080002, 141.519989, 143.479996, 147.400009, 149.360001, 159.159988, 161.119995, 165.040009, 167.000000, 176.800003, 178.760010, 182.679993, 184.639999, -115.920006, -113.320000, -108.120003, -105.520012, -92.520004, -89.919991, -84.720001, -82.119995, -69.120010, -66.520004, -61.320000, -58.719994, -22.320000, -19.719999, -14.520001, -11.920001, 1.079997, 3.679997, 8.879997, 11.480003, 24.480001, 27.079998, 32.280003, 34.880001, 71.279999, 73.880005, 79.080002, 81.680000, 94.679993, 97.280006, 102.479996, 105.080002, 118.080002, 120.679993, 125.879997, 128.479996, 164.880005, 167.479996, 172.679993, 175.279999, 188.279984, 190.880005, 196.080002, 198.679993, 211.680008, 214.280014, 219.479996, 222.079987}, sd::DataType::FLOAT32);

    NDArray expGradB('c', {oC}, {2.64, 3.92, 5.2}, sd::DataType::FLOAT32);

    input.linspace(-75, 0.5);
    gradO.linspace(0.01, 0.01);

    sd::ops::conv3dnew_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat, wFormat});
    auto gradI = results.at(0);
    auto gradW = results.at(1);
    auto gradB = results.at(2);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));

    ASSERT_TRUE(expGradB.isSameShape(gradB));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv3d_bp_test5) {

    int bS=2, iD=4,iH=3,iW=3,  iC=4,oC=3,  kD=3,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=4,oH=3,oW=3;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW
    int wFormat     = 2;             // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

    NDArray input('c', {bS, iD, iH, iW, iC}, sd::DataType::FLOAT32);
    NDArray weights('c', {oC, kD, kH, kW, iC}, {15., 14.7, 14.4, 14.1, 13.8, 13.5, 13.2, 12.9, 12.6, 12.3, 12., 11.7, 11.4, 11.1, 10.8, 10.5, 10.2, 9.9, 9.6, 9.3, 9.,
        8.7, 8.4, 8.1, 7.8, 7.5, 7.2, 6.9, 6.6, 6.3, 6., 5.7, 5.4, 5.1, 4.8, 4.5, 4.2, 3.9, 3.6, 3.3, 3., 2.7, 2.4, 2.1, 1.8, 1.5, 1.2, 0.9, 14.9, 14.6, 14.3, 14.,
        13.7, 13.4, 13.1, 12.8, 12.5, 12.2, 11.9, 11.6, 11.3, 11., 10.7, 10.4, 10.1, 9.8, 9.5, 9.2, 8.9, 8.6, 8.3, 8., 7.7, 7.4, 7.1, 6.8, 6.5, 6.2, 5.9, 5.6, 5.3, 5.,
         4.7, 4.4, 4.1, 3.8, 3.5, 3.2, 2.9, 2.6, 2.3, 2., 1.7, 1.4, 1.1, 0.8, 14.8, 14.5, 14.2, 13.9, 13.6, 13.3, 13., 12.7, 12.4, 12.1, 11.8, 11.5, 11.2, 10.9, 10.6,
         10.3, 10., 9.7, 9.4, 9.1, 8.8, 8.5, 8.2, 7.9, 7.6, 7.3, 7., 6.7, 6.4, 6.1, 5.8, 5.5, 5.2, 4.9, 4.6, 4.3, 4., 3.7, 3.4, 3.1, 2.8, 2.5, 2.2, 1.9, 1.6, 1.3, 1., 0.7}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, {1,-0.5, 0.1}, sd::DataType::FLOAT32);
    NDArray gradO('c', {bS, oD, oH, oW, oC}, sd::DataType::FLOAT32);

    NDArray expGradI('c', {bS, iD, iH, iW, iC}, {13.565001, 13.286001, 13.007000, 12.728001, 28.264000, 27.652000, 27.040001, 26.427999, 32.547997, 31.827999, 31.108002,
     30.388000, 31.647999, 30.927998, 30.208000, 29.487999, 64.484001, 62.935997, 61.387997, 59.839996, 72.188004, 70.424004, 68.660004, 66.896004, 43.852001, 42.807999,
      41.764000, 40.719997, 87.596001, 85.400002, 83.204002, 81.007996, 95.299988, 92.887993, 90.475998, 88.063995, 34.130997, 33.348000, 32.564999, 31.782001, 67.856995,
      66.210007, 64.563004, 62.916000, 72.987000, 71.178001, 69.369003, 67.559998, 70.179001, 68.369995, 66.561005, 64.751999, 137.927994, 134.147995, 130.367996, 126.587997,
       146.891998, 142.787994, 138.683990, 134.580017, 84.597000, 82.302002, 80.007004, 77.711998, 164.820007, 160.067993, 155.316010, 150.563995, 173.783997, 168.707993,
       163.631989, 158.556000, 58.674000, 57.162003, 55.649994, 54.138000, 114.027008, 110.921997, 107.816994, 104.711990, 119.156998, 115.889999, 112.623001, 109.355995, 113.433006, 110.166000, 106.899002, 103.632004, 218.603989, 211.908020, 205.211975, 198.515991, 227.568008, 220.547974, 213.528015, 206.507996, 127.850998, 124.098000, 120.345001, 116.591995, 245.496002, 237.828018, 230.159988, 222.492004, 254.459991, 246.468002, 238.475998, 230.483994, 34.049000, 32.797997, 31.547001, 30.295998, 64.479996, 61.924000, 59.368004, 56.812000, 67.035995, 64.372002, 61.707996, 59.044003, 62.248001, 59.584003, 56.919998, 54.256001, 116.180000, 110.744003, 105.307999, 99.872002, 120.428001, 114.776001, 109.124001, 103.472000, 69.268005, 66.279999, 63.292000, 60.304001, 128.923996, 122.839996, 116.755997, 110.671997, 133.171997, 126.872002, 120.571991, 114.271996, 94.565002, 92.342010, 90.118996, 87.896004, 182.488007, 177.988007, 173.488007, 168.988007, 186.772003, 182.164001, 177.556000, 172.947998, 178.095993, 173.488007, 168.880005, 164.272003, 341.828003, 332.504028, 323.180023, 313.856018, 349.532013, 339.992004, 330.451996, 320.911987, 190.299988, 185.368011, 180.436005, 175.503998, 364.940002, 354.967987, 344.996002, 335.024017, 372.644012, 362.455994, 352.268005, 342.080017, 132.303009, 128.604004, 124.904999, 121.206001, 252.536987, 245.057999, 237.578979, 230.100006, 257.666992, 250.026001, 242.385010, 234.744019, 243.195007, 235.554001, 227.912994, 220.272003, 460.631958, 445.188019, 429.744019, 414.299988, 469.595947, 453.827972, 438.059998, 422.291992, 257.613007, 249.486008, 241.358994, 233.232010, 487.523987, 471.108032, 454.691986, 438.276001, 496.488037, 479.748016, 463.007996, 446.268005, 156.846008, 152.417999, 147.989990, 143.561996, 298.707001, 289.769989, 280.833008, 271.895996, 303.837006, 294.737976, 285.638977, 276.540009, 286.449005, 277.350006, 268.250977, 259.151978, 541.307983, 522.947998, 504.587982, 486.227997, 550.271973, 531.588013, 512.903992, 494.220032, 300.867004, 291.281982, 281.696991, 272.112000, 568.200012, 548.868042, 529.535950, 510.204010, 577.164062, 557.507935, 537.851990, 518.196045, 83.944992, 80.750000, 77.555000, 74.360001, 156.496002, 150.052002, 143.608002, 137.164001, 159.052002, 152.500000, 145.947998, 139.395996, 146.488007, 139.936005, 133.384003, 126.832001, 269.107971, 255.895996, 242.684006, 229.471985, 273.356018, 259.927979, 246.500000, 233.071991, 153.507996, 146.632004, 139.755997, 132.880005, 281.851990, 267.992004, 254.132004, 240.272003, 286.100006, 272.023987, 257.947998, 243.872009}, sd::DataType::FLOAT32);

    NDArray expGradW('c', {oC, kD, kH, kW, iC}, {396.899872, 429.570007, 462.240234, 494.910156, 313.739960, 335.250000, 356.760071, 378.270020, 403.379944, 424.350006,
        445.320007, 466.289978, 299.520020, 313.319977, 327.119995, 340.920013, 1556.280029, 1594.979980, 1633.679932, 1672.379883, 1090.080078, 1115.520020, 1140.959961,
        1166.400024, 1183.679932, 1208.400024, 1233.119995, 1257.840088, 821.279907, 837.519897, 853.760010, 870.000000, 1500.119873, 1525.500122, 1550.880005, 1576.260010,
        1029.780029, 1046.429932, 1063.080078, 1079.729980, 1080.539917, 1096.650024, 1112.760010, 1128.869995, 738.000000, 748.560059, 759.119995, 769.679993, 389.880005,
        422.819946, 455.759979, 488.699951, 309.420013, 331.109985, 352.799988, 374.490051, 399.780029, 420.930023, 442.080017, 463.230011, 297.359985, 311.280029, 325.200012, 339.120056, 1553.400146, 1592.459961, 1631.520020, 1670.579956, 1088.640015, 1114.320068, 1140.000000, 1165.679932, 1183.199951, 1208.160034, 1233.119995, 1258.079956, 821.280029, 837.680054, 854.079956, 870.479980, 1502.819946, 1528.469971, 1554.119995, 1579.770020, 1031.939941, 1048.770020, 1065.599976, 1082.429932, 1083.420044, 1099.709961, 1116.000000, 1132.290039, 740.159973, 750.840027, 761.519958, 772.199951, 382.859924, 416.070099, 449.279968, 482.489990, 305.099976, 326.970062, 348.840027, 370.709991, 396.179962, 417.510010, 438.839966, 460.169952, 295.200012, 309.239990, 323.279968, 337.320007, 1550.519775, 1589.939941, 1629.359985, 1668.779907, 1087.200073, 1113.119995, 1139.039917, 1164.959961, 1182.719971, 1207.920044, 1233.119995, 1258.320190, 821.279968, 837.840027, 854.400024, 870.959961, 1505.520142, 1531.439819, 1557.359985, 1583.279907, 1034.100098, 1051.110107, 1068.120117, 1085.130005, 1086.299927, 1102.770020, 1119.239990, 1135.710083, 742.319946, 753.119995, 763.919983, 774.720032}, sd::DataType::FLOAT32);

    NDArray expGradB('c', {oC}, {77.400002, 78.119995, 78.840004}, sd::DataType::FLOAT32);

    input.linspace(-75, 0.5);
    gradO.linspace(0.01, 0.01);

    sd::ops::conv3dnew_bp op;
    auto results = op.evaluate({&input, &weights, &bias, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat, wFormat});
    auto gradI = results.at(0);
    auto gradW = results.at(1);
    auto gradB = results.at(2);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

    ASSERT_TRUE(expGradW.isSameShape(gradW));
    ASSERT_TRUE(expGradW.equalsTo(gradW));

    ASSERT_TRUE(expGradB.isSameShape(gradB));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_test1) {

    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 1;             // 1-SAME,  0-VALID;
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iD, iH, iW, iC});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC});
    auto expected = NDArrayFactory::create<TypeParam>('c', {2, 3, 4, 3, 3}, {534.4f, 540.8f, 547.2f, 534.4f, 540.8f, 547.2f, 248.f, 251.2f, 254.4f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 314.4f, 319.2f, 324.f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 314.4f, 319.2f, 324.f,
                                                   380.8f, 387.2f, 393.6f, 380.8f, 387.2f, 393.6f, 171.2f, 174.4f, 177.6f, 534.4f, 540.8f, 547.2f, 534.4f, 540.8f, 547.2f, 248.f, 251.2f, 254.4f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 314.4f, 319.2f, 324.f,
                                                   686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 314.4f, 319.2f, 324.f, 380.8f, 387.2f, 393.6f, 380.8f, 387.2f, 393.6f, 171.2f, 174.4f, 177.6f, 152.f, 155.2f, 158.4f, 152.f, 155.2f, 158.4f, 66.4f, 68.f, 69.6f,
                                                   170.4f, 175.2f, 180.f, 170.4f, 175.2f, 180.f, 70.8f, 73.2f, 75.6f, 170.4f, 175.2f, 180.f, 170.4f, 175.2f, 180.f, 70.8f, 73.2f, 75.6f, 75.2f, 78.4f, 81.6f, 75.2f, 78.4f, 81.6f, 28.f, 29.6f, 31.2f,
                                                   534.4f, 540.8f, 547.2f, 534.4f, 540.8f, 547.2f, 248.f, 251.2f, 254.4f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 314.4f, 319.2f, 324.f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 314.4f, 319.2f, 324.f,
                                                   380.8f, 387.2f, 393.6f, 380.8f, 387.2f, 393.6f, 171.2f, 174.4f, 177.6f, 534.4f, 540.8f, 547.2f, 534.4f, 540.8f, 547.2f, 248.f, 251.2f, 254.4f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 314.4f, 319.2f, 324.f,
                                                   686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 314.4f, 319.2f, 324.f, 380.8f, 387.2f, 393.6f, 380.8f, 387.2f, 393.6f, 171.2f, 174.4f, 177.6f, 152.f, 155.2f, 158.4f, 152.f, 155.2f, 158.4f, 66.4f, 68.f, 69.6f,
                                                   170.4f, 175.2f, 180.f, 170.4f, 175.2f, 180.f, 70.8f, 73.2f, 75.6f, 170.4f, 175.2f, 180.f, 170.4f, 175.2f, 180.f, 70.8f, 73.2f, 75.6f, 75.2f, 78.4f, 81.6f, 75.2f, 78.4f, 81.6f, 28.f, 29.6f, 31.2f});
    input = 2.;
    weights.linspace(0.1, 0.1);

    sd::ops::conv3dnew op;
    auto results = op.evaluate({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_test2) {

    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 0;             // 1-SAME,  0-VALID;
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iD, iH, iW, iC});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC});
    auto expected = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2, 3},  {686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f,
                                                    686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f,
                                                    686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f,
                                                    686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f, 686.4f, 696.f, 705.6f});
    input = 2.;
    weights.linspace(0.1, 0.1);

    sd::ops::conv3dnew op;
    auto results = op.evaluate({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

}


//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_test3) {

    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 0;             // 1-SAME,  0-VALID;
    int dataFormat =  0;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iC, iD, iH, iW});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC});
    auto expected = NDArrayFactory::create<TypeParam>('c', {2, 3, 2, 2, 2});
    input = 2.;
    weights = 0.5;
    expected = 48.;

    sd::ops::conv3dnew op;
    auto results = op.evaluate({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

}

////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_test4) {

    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 0;             // 1-SAME,  0-VALID;
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iC, iD, iH, iW});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC});
    auto bias     = NDArrayFactory::create<TypeParam>('c', {oC});
    auto expected = NDArrayFactory::create<TypeParam>('c', {2, 3, 2, 2, 2});

    input = 2.;
    weights = 0.5;
    expected = 49.;
    bias = 1.;

    sd::ops::conv3dnew op;
    auto results = op.evaluate({&input, &weights, &bias}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto* output = results.at(0);

    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

}

////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_test5) {

    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 0;             // 1-SAME,  0-VALID;
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iC, iD, iH, iW});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC});
    auto bias     = NDArrayFactory::create<TypeParam>('c', {oC},{1.f, 2.f, 3.f});
    auto expected = NDArrayFactory::create<TypeParam>('c', {2, 3, 2, 2, 2},{49.f,  49.f, 49.f,  49.f,  49.f,  49.f, 49.f,  49.f,  50.f,  50.f, 50.f,  50.f,  50.f,  50.f, 50.f,  50.f,
                                                  51.f,  51.f, 51.f,  51.f,  51.f,  51.f, 51.f,  51.f,  49.f,  49.f, 49.f,  49.f,  49.f,  49.f, 49.f,  49.f,
                                                  50.f,  50.f, 50.f,  50.f,  50.f,  50.f, 50.f,  50.f,  51.f,  51.f, 51.f,  51.f,  51.f,  51.f, 51.f,  51.f});
    input = 2.;
    weights = 0.5;

    sd::ops::conv3dnew op;
    auto results = op.evaluate({&input, &weights, &bias}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto* output = results.at(0);

    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

}

////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_test6) {

    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 0;             // 1-SAME,  0-VALID;
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iC, iD, iH, iW});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {oC, iC, kD, kH, kW});
    auto bias     = NDArrayFactory::create<TypeParam>('c', {oC},{1.f, 2.f, 3.f});
    auto expected = NDArrayFactory::create<TypeParam>('c', {2, 3, 2, 2, 2},{236.2f, 236.2f, 236.2f, 236.2f, 236.2f, 236.2f, 236.2f, 236.2f, 698.f, 698.f, 698.f, 698.f,
                                                  698.f, 698.f, 698.f, 698.f, 1159.8f, 1159.8f, 1159.8f, 1159.8f, 1159.8f, 1159.8f, 1159.8f, 1159.8f,
                                                  236.2f, 236.2f, 236.2f, 236.2f, 236.2f, 236.2f, 236.2f, 236.2f, 698.f, 698.f, 698.f, 698.f,
                                                  698.f, 698.f, 698.f, 698.f, 1159.8f, 1159.8f, 1159.8f, 1159.8f, 1159.8f, 1159.8f, 1159.8f, 1159.8f});
    input = 2.;
    weights.linspace(0.1, 0.1);
    weights.permutei({2, 3, 4, 1, 0});

    sd::ops::conv3dnew op;
    auto results = op.evaluate({&input, &weights, &bias}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto* output = results.at(0);

    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_test7) {

    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 0;             // 1-SAME,  0-VALID;
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iC, iD, iH, iW});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {oC, iC, kD, kH, kW});
    auto expected = NDArrayFactory::create<TypeParam>('c', {2, 3, 2, 2, 2},{235.2f, 235.2f, 235.2f, 235.2f, 235.2f, 235.2f, 235.2f, 235.2f, 696.f, 696.f, 696.f, 696.f, 696.f, 696.f, 696.f, 696.f,
                                                  1156.8f, 1156.8f, 1156.8f, 1156.8f, 1156.8f, 1156.8f, 1156.8f, 1156.8f, 235.2f, 235.2f, 235.2f, 235.2f, 235.2f, 235.2f, 235.2f, 235.2f,
                                                  696.f, 696.f, 696.f, 696.f, 696.f, 696.f, 696.f, 696.f, 1156.8f, 1156.8f, 1156.8f, 1156.8f, 1156.8f, 1156.8f, 1156.8f, 1156.8f});
    input = 2.;
    weights.linspace(0.1, 0.1);
    weights.permutei({2, 3, 4, 1, 0});

    sd::ops::conv3dnew op;
    auto results = op.evaluate({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_test8) {
    auto x = NDArrayFactory::create<TypeParam>('c', {4, 2, 28, 28, 3});
    auto y = NDArrayFactory::create<TypeParam>('c', {2, 5, 5, 3, 4});
    auto e = NDArrayFactory::create<TypeParam>('c', {4, 1, 7, 10, 4});

    sd::ops::conv3dnew op;
    auto result = op.evaluate({&x, &y}, {}, {2,5,5, 5,4,3, 0,0,0, 1,1,1, 1,1});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

    ASSERT_TRUE(e.isSameShape(z));
}

TYPED_TEST(TypedConvolutionTests1, conv3d_test9) {
    auto x = NDArrayFactory::create<TypeParam>('c', {4, 2, 28, 28, 3});
    auto w = NDArrayFactory::create<TypeParam>('c', {2, 5, 5, 3, 4});
    auto exp = NDArrayFactory::create<TypeParam>('c', {4, 1, 7, 10, 4});

    sd::ops::conv3dnew op;
    auto result = op.evaluate({&x, &w}, {}, {2,5,5, 5,4,3, 0,0,0, 1,1,1, 1,1});
    ASSERT_EQ(Status::OK(), result.status());

    ShapeList shapeList({x.shapeInfo(), w.shapeInfo()});
    ContextPrototype proto;
    Context ctx(1);
    ctx.getIArguments()->push_back(2);
    ctx.getIArguments()->push_back(5);
    ctx.getIArguments()->push_back(5);

    ctx.getIArguments()->push_back(5);
    ctx.getIArguments()->push_back(4);
    ctx.getIArguments()->push_back(3);

    ctx.getIArguments()->push_back(0);
    ctx.getIArguments()->push_back(0);
    ctx.getIArguments()->push_back(0);

    ctx.getIArguments()->push_back(1);
    ctx.getIArguments()->push_back(1);
    ctx.getIArguments()->push_back(1);

    ctx.getIArguments()->push_back(0);
    ctx.getIArguments()->push_back(1);  // previous variant was "ctx.getIArguments()->push_back(0)" and this caused fail

    auto shapes = op.calculateOutputShape(&shapeList, ctx);
    ASSERT_EQ(1, shapes->size());

    auto s = shapes->at(0);

    auto z = result.at(0);
    // z->printShapeInfo("z shape");

    ASSERT_TRUE(exp.isSameShape(z));

    delete shapes;
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_test10) {

    int bS=1, iD=2,iH=2,iW=2,  iC=1,oC=1,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 1;             // 1-SAME,  0-VALID;
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iC, iD, iH, iW});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC});

    input = 2.;
    weights = 1.;

    sd::ops::conv3dnew op;
    auto results = op.evaluate({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, conv3d_test11) {

    int bS=5, iD=4,iH=14,iW=14,  iC=1,oC=1,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=3,oH=13,oW=13;
    int paddingMode = 0;             // 1-SAME,  0-VALID;
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iC, iD, iH, iW});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC});
    auto expected = NDArrayFactory::create<TypeParam>('c', {bS, oC, oD, oH, oW});

    input = 2.;
    weights = 1.;

    sd::ops::conv3dnew op;
    auto results = op.evaluate({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(output->isSameShape(&expected));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv3d_test12) {

    int bS=2, iD=4,iH=3,iW=3,  iC=4,oC=3,  kD=3,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW
    int wFormat     = 1;             // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

    NDArray input('c', {bS, iC, iD, iH, iW}, sd::DataType::FLOAT32);
    NDArray weights('c', {oC, iC, kD, kH, kW}, {-14.4, -13.2, -12.0, -10.8, -9.6, -8.4, -7.2, -6.0, -4.8, -3.6, -2.4, -1.2, -14.1, -12.9, -11.7, -10.5, -9.3, -8.1,
        -6.9, -5.7, -4.5, -3.3, -2.1, -0.9, -13.8, -12.6, -11.4, -10.2, -9.0, -7.8, -6.6, -5.4, -4.2, -3.0, -1.8, -0.6, -13.5, -12.3, -11.1, -9.9, -8.7, -7.5, -6.3,
        -5.1, -3.9, -2.7, -1.5, -0.3, -14.3, -13.1, -11.9, -10.7, -9.5, -8.3, -7.1, -5.9, -4.7, -3.5, -2.3, -1.1, -14.0, -12.8, -11.6, -10.4, -9.2, -8.0, -6.8, -5.6,
        -4.4, -3.2, -2.0, -0.8, -13.7, -12.5, -11.3, -10.1, -8.9, -7.7, -6.5, -5.3, -4.1, -2.9, -1.7, -0.5, -13.4, -12.2, -11.0, -9.8, -8.6, -7.4, -6.2, -5.0, -3.8, -2.6, -1.4, -0.2, -14.2, -13.0, -11.8, -10.6, -9.4, -8.2, -7.0, -5.8, -4.6, -3.4, -2.2, -1.0, -13.9, -12.7, -11.5, -10.3, -9.1, -7.9, -6.7, -5.5, -4.3, -3.1, -1.9, -0.7, -13.6, -12.4, -11.2, -10.0, -8.8, -7.6, -6.4, -5.2, -4.0, -2.8, -1.6, -0.4, -13.3, -12.1, -10.9, -9.7, -8.5, -7.3, -6.1, -4.9, -3.7, -2.5, -1.3, -0.1}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, {-1,2,0.5}, sd::DataType::FLOAT32);

    NDArray expOutput('c', {bS, oC, oD, oH, oW}, {-42520.597656, -42344.199219, -41991.402344, -41814.996094, -40932.992188, -40756.597656, -40403.800781, -40227.406250,
            -41953.601562, -41779.601562, -41431.597656, -41257.601562, -40387.601562, -40213.597656, -39865.601562, -39691.597656, -41391.105469, -41219.492188,
            -40876.300781, -40704.699219, -39846.707031, -39675.097656, -39331.898438, -39160.300781, -17119.001953, -16942.599609, -16589.798828, -16413.400391,
            -15531.399414, -15355.000000, -15002.199219, -14825.800781, -16897.597656, -16723.597656, -16375.599609, -16201.599609, -15331.599609, -15157.600586,
            -14809.601562, -14635.598633, -16680.703125, -16509.099609, -16165.900391, -15994.300781, -15136.300781, -14964.700195, -14621.500000, -14449.900391}, sd::DataType::FLOAT32);

    input.linspace(150,-0.5);

    sd::ops::conv3dnew op;
    auto results = op.evaluate({&input, &weights, &bias}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat, wFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv3d_test13) {

    int bS=2, iD=4,iH=3,iW=3,  iC=4,oC=3,  kD=3,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=4,oH=3,oW=3;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW
    int wFormat     = 2;             // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

    NDArray input('c', {bS, iD, iH, iW, iC}, sd::DataType::FLOAT32);
    NDArray weights('c', {oC, kD, kH, kW, iC}, {-7., -6.7, -6.4, -6.1, -5.8, -5.5, -5.2, -4.9, -4.6, -4.3, -4., -3.7, -3.4, -3.1, -2.8, -2.5, -2.2, -1.9, -1.6, -1.3,
        -1., -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2., 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7, 5., 5.3, 5.6, 5.9, 6.2, 6.5, 6.8, 7.1, -6.9, -6.6, -6.3,
         -6., -5.7, -5.4, -5.1, -4.8, -4.5, -4.2, -3.9, -3.6, -3.3, -3., -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1,
         2.4, 2.7, 3., 3.3, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6., 6.3, 6.6, 6.9, 7.2, -6.8, -6.5, -6.2, -5.9, -5.6, -5.3, -5., -4.7, -4.4, -4.1, -3.8, -3.5, -3.2,
         -2.9, -2.6, -2.3, -2., -1.7, -1.4, -1.1, -0.8, -0.5, -0.2, 0.1, 0.4, 0.7, 1., 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4., 4.3, 4.6, 4.9, 5.2, 5.5, 5.8, 6.1, 6.4, 6.7, 7., 7.3}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, {-1,2,0.5}, sd::DataType::FLOAT32);

    NDArray expOutput('c', {bS, oD, oH, oW, oC}, {3969.399658, 4168.399902, 4362.899414, 3812.600586, 4005.200195, 4193.299805, 1317.000000, 1413.199829, 1504.899902,
        3498.999756, 3678.800049, 3854.100098, 3342.200195, 3515.599854, 3684.500244, 1139.400024, 1226.000000, 1308.099976, 685.799927, 772.400024, 854.500000,
        645.800049, 729.200073, 808.099976, 80.799995, 123.200012, 161.100006, -2851.000732, -2597.199707, -2347.899414, -2855.799805, -2611.600098, -2371.900879,
        -2124.399414, -2003.199951, -1886.500244, -2865.399902, -2640.400146, -2419.899902, -2870.199951, -2654.800049, -2443.899902, -2045.200073, -1938.399902,
        -1836.100220, -2596.000244, -2489.199707, -2386.900146, -2540.799561, -2438.800049, -2341.300049, -1539.699951, -1488.400024, -1441.599854, -2894.200195,
        -2726.800049, -2563.899902, -2899.000488, -2741.199707, -2587.899658, -1886.800171, -1808.800049, -1735.300171, -2908.599121, -2770.000488, -2635.900146, -2913.400146, -2784.399658, -2659.899902, -1807.599976, -1743.999878, -1684.900146, -2099.199951, -2035.599976, -1976.500366, -2044.000244, -1985.199707, -1930.900024, -1161.699951, -1132.000122, -1106.800171, -2731.399902, -2647.599609, -2568.300293, -2580.999756, -2503.600098, -2430.699951, -1457.400024, -1418.800049, -1384.700073, -2280.200195, -2215.600098, -2155.500732, -2129.799561, -2071.600098, -2017.899780, -1174.200073, -1145.200195, -1120.699829, -1282.200073, -1253.199951, -1228.699951, -1168.599976, -1142.799927, -1121.500122, -615.199951, -601.600037, -592.500000, -1675.399658, -1706.800049, -1742.700073, -1832.200073, -1870.000000, -1912.299561, -814.199951, -833.200012, -856.699951, -2145.800049, -2196.399902, -2251.500244, -2302.600342, -2359.599854, -2421.100098, -991.800049, -1020.400024, -1053.500000, -754.199951, -782.800049, -815.900085, -794.199951, -825.999939, -862.299988, -293.600006, -308.800018, -328.500000, -3023.800293, -3115.600098, -3211.900391, -3028.599121, -3130.000244, -3235.899902, -1173.999878, -1225.600098, -1281.699951, -3038.200195, -3158.799805, -3283.899902, -3043.000000, -3173.199707, -3307.900391, -1094.800049, -1160.800049, -1231.300049, -608.799988, -674.799988, -745.300049, -553.599976, -624.400024, -699.700012, -27.700012, -62.799988, -102.400009, -3066.999512, -3245.199707, -3427.900391, -3071.800293, -3259.599854, -3451.900146, -936.400085, -1031.199951, -1130.500000, -3081.400146, -3288.400635, -3499.899414, -3086.200439, -3302.799805, -3523.899902, -857.199951, -966.400024, -1080.099976, -111.999969, -221.199936, -334.900024, -56.800079, -170.799988, -289.299927, 350.299927, 293.600037, 232.399979, 2683.000244, 2536.400146, 2385.300049, 2833.399658, 2680.400391, 2522.900391, 1940.999878, 1864.399902, 1783.300049, 3134.200195, 2968.399414, 2798.100098, 3284.600098, 3112.400391, 2935.699707, 2224.199707, 2138.000244, 2047.300049, 2807.399658, 2721.200195, 2630.500000, 2921.000000, 2831.599854, 2737.699707, 1775.200195, 1731.199951, 1682.699829}, sd::DataType::FLOAT32);

    input.linspace(75,-0.5);

    sd::ops::conv3dnew op;
    auto results = op.evaluate({&input, &weights, &bias}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat, wFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, pointwise_conv2d_test1) {

    int bS=2, iH=4,iW=3,  iC=4,oC=3;

    int dataFormat = 1;           // 1-NHWC, 0-NCHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iH, iW, iC});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {1,   1, iC, oC});
    auto bias     = NDArrayFactory::create<TypeParam>('c', {oC});


    auto expOutput = NDArrayFactory::create<TypeParam>('c', {bS, iH, iW, oC},{ 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f,
                                                      7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f,
                                                      6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f,
                                                      5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f, 5.4f, 6.2f, 7.0f});
    input = 2.;
    weights.linspace(0.1, 0.1);
    bias = 1.;

    sd::ops::pointwise_conv2d op;
    auto results = op.evaluate({&input, &weights, &bias}, {}, {dataFormat});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, vol2col_test1) {

    int bS=2, iD=2,iH=3,iW=2,  iC=3,oC=2,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=2,oH=3,oW=2;

    NDArray volume('c', {bS, iC, iD, iH, iW}, sd::DataType::FLOAT32);
    NDArray columns('c', {bS, iC, kD, kH, kW, oD, oH, oW}, sd::DataType::FLOAT32);

    columns = -1.;
    volume.linspace(1);

    NDArray columnsExpected('c', {bS, iC, kD, kH, kW, oD, oH, oW}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 2., 0., 4., 0., 6.,0., 8., 0., 10., 0., 12., 0., 3., 4., 5., 6., 0., 0., 9., 10., 11., 12., 0., 0., 4., 0., 6., 0., 0., 0., 10., 0., 12., 0., 0., 0., 5., 6.,
0., 0., 0., 0., 11., 12., 0., 0., 0., 0., 6., 0., 0., 0., 0., 0., 12., 0., 0., 0., 0., 0., 7., 8., 9., 10., 11., 12., 0., 0., 0., 0., 0., 0., 8., 0., 10., 0., 12., 0., 0., 0., 0., 0., 0., 0., 9., 10., 11., 12., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 12., 0., 0., 0., 0., 0.,
0., 0., 0., 0., 11., 12., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 12., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 13., 14., 15., 16., 17.,18., 19., 20., 21., 22., 23., 24., 14., 0., 16., 0., 18., 0., 20., 0., 22., 0., 24., 0., 15., 16., 17., 18., 0., 0., 21., 22., 23., 24., 0.,
0., 16., 0., 18., 0., 0., 0., 22., 0., 24., 0., 0., 0., 17., 18., 0., 0., 0., 0., 23., 24., 0., 0., 0., 0., 18., 0., 0., 0., 0., 0., 24., 0., 0., 0., 0., 0., 19., 20., 21., 22., 23., 24., 0., 0., 0., 0., 0., 0., 20., 0., 22., 0., 24., 0., 0., 0., 0., 0., 0., 0., 21., 22., 23.,
24., 0., 0., 0., 0., 0., 0., 0., 0., 22., 0., 24., 0., 0., 0., 0., 0., 0., 0., 0., 0., 23., 24., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 24.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 26., 0., 28., 0., 30., 0., 32., 0.,
34., 0., 36., 0., 27., 28., 29., 30., 0., 0., 33., 34., 35., 36., 0., 0., 28., 0., 30., 0., 0., 0., 34., 0., 36., 0., 0., 0., 29., 30., 0., 0., 0., 0., 35., 36., 0., 0., 0., 0., 30., 0., 0., 0., 0., 0., 36., 0., 0., 0., 0., 0., 31., 32., 33., 34., 35., 36., 0., 0., 0., 0., 0.,
0., 32., 0., 34., 0., 36., 0., 0., 0., 0., 0., 0., 0., 33., 34., 35., 36., 0., 0., 0., 0., 0., 0., 0., 0., 34., 0., 36., 0., 0., 0., 0., 0., 0., 0., 0., 0., 35., 36., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 36., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 37., 38., 39., 40.,
41., 42., 43., 44., 45., 46., 47., 48., 38., 0., 40., 0., 42., 0., 44., 0., 46., 0., 48., 0., 39., 40., 41., 42., 0., 0., 45., 46., 47., 48., 0., 0., 40., 0., 42., 0., 0., 0., 46., 0., 48., 0., 0., 0., 41., 42., 0., 0., 0., 0., 47., 48., 0., 0., 0., 0., 42., 0., 0., 0., 0.,
0., 48., 0., 0., 0., 0., 0., 43., 44., 45., 46., 47., 48., 0., 0., 0., 0., 0., 0., 44., 0., 46., 0., 48., 0., 0., 0., 0., 0., 0., 0., 45., 46., 47., 48., 0., 0., 0., 0., 0., 0., 0., 0., 46., 0., 48., 0., 0., 0., 0., 0., 0., 0., 0., 0., 47., 48., 0., 0., 0., 0., 0., 0., 0., 0.,
0., 0., 48., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 50., 0., 52., 0., 54.,0., 56., 0., 58., 0., 60., 0., 51., 52., 53., 54., 0., 0., 57., 58., 59., 60., 0., 0., 52., 0., 54., 0., 0., 0., 58., 0., 60., 0., 0., 0.,
53., 54., 0., 0., 0., 0., 59., 60., 0., 0., 0., 0., 54., 0., 0., 0., 0., 0., 60., 0., 0., 0., 0., 0., 55., 56., 57., 58., 59., 60., 0., 0.,0., 0., 0., 0., 56., 0., 58., 0., 60., 0., 0., 0., 0., 0., 0., 0., 57., 58., 59., 60., 0., 0., 0., 0., 0., 0., 0., 0., 58., 0., 60., 0.,
0., 0., 0., 0., 0., 0., 0., 0., 59., 60., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 60., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 62., 0., 64., 0., 66., 0., 68., 0., 70., 0., 72., 0., 63., 64., 65., 66., 0., 0., 69.,
70., 71., 72., 0., 0., 64., 0., 66., 0., 0., 0., 70., 0., 72., 0., 0., 0., 65., 66., 0., 0., 0., 0., 71., 72., 0., 0., 0., 0., 66., 0., 0., 0., 0., 0., 72., 0., 0., 0., 0., 0., 67., 68., 69., 70., 71., 72., 0., 0., 0., 0., 0., 0., 68., 0., 70., 0., 72., 0., 0., 0., 0., 0., 0.,
0., 69., 70., 71., 72., 0., 0., 0., 0., 0., 0., 0., 0., 70., 0., 72., 0., 0., 0., 0., 0., 0., 0., 0., 0., 71., 72., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 72., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}, sd::DataType::FLOAT32);

    graph::Context context(1);
    sd::ops::ConvolutionUtils::vol2col(context, volume, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW);
    // columns.printBuffer();

    ASSERT_TRUE(columns.equalsTo(columnsExpected));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, vol2col_test2) {

    int bS=2, iD=2,iH=3,iW=2,  iC=3,oC=2,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=2,oH=3,oW=2;

    auto volume  = NDArrayFactory::create<float>('c', {iD, bS, iH, iC, iW});
    volume.permutei({1, 3, 0, 2, 4});
    volume.linspace(1);

    auto columns = NDArrayFactory::create<float>('c', {kD, iC, kH, oW, kW, bS, oD, oH});
    columns.permutei({5, 1, 0, 2, 4, 6, 7, 3});
    columns = -1.;
    auto columnsExpected = NDArrayFactory::create<float>('c', {bS, iC, kD, kH, kW, oD, oH, oW}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f,
10.f, 11.f, 12.f, 2.f, 0.f, 4.f, 0.f, 6.f, 0.f, 8.f, 0.f, 10.f, 0.f, 12.f, 0.f, 3.f, 4.f, 5.f, 6.f, 0.f, 0.f, 9.f, 10.f, 11.f, 12.f, 0.f, 0.f, 4.f, 0.f, 6.f, 0.f, 0.f, 0.f, 10.f, 0.f, 12.f, 0.f, 0.f, 0.f, 5.f, 6.f, 0.f, 0.f, 0.f, 0.f, 11.f, 12.f, 0.f, 0.f, 0.f, 0.f, 6.f, 0.f, 0.f, 0.f, 0.f, 0.f, 12.f, 0.f, 0.f, 0.f, 0.f, 0.f, 7.f, 8.f,
9.f, 10.f, 11.f, 12.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 8.f, 0.f, 10.f, 0.f, 12.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 9.f, 10.f, 11.f, 12.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 10.f, 0.f, 12.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 11.f, 12.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 12.f, 0.f, 0.f, 0.f, 0.f, 0.f,
0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 14.f, 0.f, 16.f, 0.f, 18.f, 0.f, 20.f, 0.f, 22.f, 0.f, 24.f, 0.f, 15.f, 16.f, 17.f, 18.f, 0.f, 0.f, 21.f, 22.f, 23.f, 24.f, 0.f, 0.f, 16.f, 0.f, 18.f, 0.f, 0.f, 0.f, 22.f, 0.f, 24.f, 0.f, 0.f, 0.f, 17.f, 18.f, 0.f, 0.f, 0.f, 0.f,
23.f, 24.f, 0.f, 0.f, 0.f, 0.f, 18.f, 0.f, 0.f, 0.f, 0.f, 0.f, 24.f, 0.f, 0.f, 0.f, 0.f, 0.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 20.f, 0.f, 22.f, 0.f, 24.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 21.f, 22.f, 23.f, 24.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 22.f, 0.f, 24.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
0.f, 0.f, 0.f, 23.f, 24.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 24.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 26.f, 0.f, 28.f, 0.f, 30.f, 0.f, 32.f, 0.f, 34.f, 0.f, 36.f, 0.f, 27.f, 28.f, 29.f, 30.f, 0.f, 0.f, 33.f, 34.f, 35.f, 36.f,
0.f, 0.f, 28.f, 0.f, 30.f, 0.f, 0.f, 0.f, 34.f, 0.f, 36.f, 0.f, 0.f, 0.f, 29.f, 30.f, 0.f, 0.f, 0.f, 0.f, 35.f, 36.f, 0.f, 0.f, 0.f, 0.f, 30.f, 0.f, 0.f, 0.f, 0.f, 0.f, 36.f, 0.f, 0.f, 0.f, 0.f, 0.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 32.f, 0.f, 34.f, 0.f, 36.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 33.f,
34.f, 35.f, 36.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 34.f, 0.f, 36.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 35.f, 36.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 36.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 38.f, 0.f, 40.f,
0.f, 42.f, 0.f, 44.f, 0.f, 46.f, 0.f, 48.f, 0.f, 39.f, 40.f, 41.f, 42.f, 0.f, 0.f, 45.f, 46.f, 47.f, 48.f, 0.f, 0.f, 40.f, 0.f, 42.f, 0.f, 0.f, 0.f, 46.f, 0.f, 48.f, 0.f, 0.f, 0.f, 41.f, 42.f, 0.f, 0.f, 0.f, 0.f, 47.f, 48.f, 0.f, 0.f, 0.f, 0.f, 42.f, 0.f, 0.f, 0.f, 0.f, 0.f, 48.f, 0.f, 0.f, 0.f, 0.f, 0.f, 43.f, 44.f, 45.f, 46.f, 47.f,
48.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 44.f, 0.f, 46.f, 0.f, 48.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 45.f, 46.f, 47.f, 48.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 46.f, 0.f, 48.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 47.f, 48.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 48.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
0.f, 0.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 50.f, 0.f, 52.f, 0.f, 54.f, 0.f, 56.f, 0.f, 58.f, 0.f, 60.f, 0.f, 51.f, 52.f, 53.f, 54.f, 0.f, 0.f, 57.f, 58.f, 59.f, 60.f, 0.f, 0.f, 52.f, 0.f, 54.f, 0.f, 0.f, 0.f, 58.f, 0.f, 60.f, 0.f, 0.f, 0.f, 53.f, 54.f, 0.f, 0.f, 0.f, 0.f, 59.f, 60.f, 0.f, 0.f,
0.f, 0.f, 54.f, 0.f, 0.f, 0.f, 0.f, 0.f, 60.f, 0.f, 0.f, 0.f, 0.f, 0.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 56.f, 0.f, 58.f, 0.f, 60.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 57.f, 58.f, 59.f, 60.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 58.f, 0.f, 60.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 59.f, 60.f,
0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 60.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 67.f, 68.f, 69.f, 70.f, 71.f, 72.f, 62.f, 0.f, 64.f, 0.f, 66.f, 0.f, 68.f, 0.f, 70.f, 0.f, 72.f, 0.f, 63.f, 64.f, 65.f, 66.f, 0.f, 0.f, 69.f, 70.f, 71.f, 72.f, 0.f, 0.f, 64.f, 0.f, 66.f,
0.f, 0.f, 0.f, 70.f, 0.f, 72.f, 0.f, 0.f, 0.f, 65.f, 66.f, 0.f, 0.f, 0.f, 0.f, 71.f, 72.f, 0.f, 0.f, 0.f, 0.f, 66.f, 0.f, 0.f, 0.f, 0.f, 0.f, 72.f, 0.f, 0.f, 0.f, 0.f, 0.f, 67.f, 68.f, 69.f, 70.f, 71.f, 72.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 68.f, 0.f, 70.f, 0.f, 72.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 69.f, 70.f, 71.f, 72.f, 0.f, 0.f,
0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 70.f, 0.f, 72.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 71.f, 72.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 72.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

    graph::Context context(1);
    sd::ops::ConvolutionUtils::vol2col(context, volume, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW);
        // columns.printBuffer();

    ASSERT_TRUE(columns.equalsTo(columnsExpected));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, col2im_test1) {

    int bS=2, iH=2,iW=2,  iC=2,   kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oH=2,oW=2;

    auto image  = NDArrayFactory::create<float>('c', {bS, iC, iH, iW});
    image = -2.;

    auto columns = NDArrayFactory::create<float>('c', {bS, iC, kH, kW, oH, oW});
    columns.linspace(1);

    auto imageExpected = NDArrayFactory::create<float>('c', {bS, iC, iH, iW}, {1.f,  7.f,  12.f,  34.f,  17.f,  39.f,  44.f,  98.f,  33.f,  71.f,  76.f,  162.f,  49.f,  103.f,  108.f,  226.f});


    sd::ops::col2im op;
    auto status = op.execute({&columns}, {&image}, {sH, sW, pH, pW, iH, iW, dH, dW, 0});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(image.equalsTo(imageExpected));
}


//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, upsampling2d_test1) {

    const int bS=3,  iH=2,iW=2,  iC=3;
    const int factorH=2, factorW=3;
    const int isNCHW = 0;                    // data format, default is NCHW

    auto input  = NDArrayFactory::create<float>('c', {bS, iH, iW, iC});
    input.linspace(1);

    auto expOutput = NDArrayFactory::create<float>('c', {bS, iH*factorH, iW*factorW, iC}, {1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 4.f, 5.f, 6.f, 4.f, 5.f, 6.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 4.f, 5.f, 6.f, 4.f, 5.f, 6.f,
                                         7.f, 8.f, 9.f, 7.f, 8.f, 9.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 10.f, 11.f, 12.f, 10.f, 11.f, 12.f, 7.f, 8.f, 9.f, 7.f, 8.f, 9.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 10.f, 11.f, 12.f, 10.f, 11.f, 12.f,
                                        13.f, 14.f, 15.f, 13.f, 14.f, 15.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 16.f, 17.f, 18.f, 16.f, 17.f, 18.f, 13.f, 14.f, 15.f, 13.f, 14.f, 15.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 16.f, 17.f, 18.f, 16.f, 17.f, 18.f,
                                        19.f, 20.f, 21.f, 19.f, 20.f, 21.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 22.f, 23.f, 24.f, 22.f, 23.f, 24.f, 19.f, 20.f, 21.f, 19.f, 20.f, 21.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 22.f, 23.f, 24.f, 22.f, 23.f, 24.f,
                                        25.f, 26.f, 27.f, 25.f, 26.f, 27.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 28.f, 29.f, 30.f, 28.f, 29.f, 30.f, 25.f, 26.f, 27.f, 25.f, 26.f, 27.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 28.f, 29.f, 30.f, 28.f, 29.f, 30.f,
                                        31.f, 32.f, 33.f, 31.f, 32.f, 33.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 34.f, 35.f, 36.f, 34.f, 35.f, 36.f, 31.f, 32.f, 33.f, 31.f, 32.f, 33.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 34.f, 35.f, 36.f, 34.f, 35.f, 36.f});

    sd::ops::upsampling2d op;
    auto results = op.evaluate({&input}, {factorH, factorW, isNCHW});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, upsampling2d_test2) {

    const int bS=3,  iH=2,iW=2,  iC=3;
    const int factorH=2, factorW=3;
    const int isNCHW = 1;                    // data format, default is NCHW

    auto input  = NDArrayFactory::create<float>('c', {bS, iC, iH, iW});
    input.linspace(1);

    auto expOutput = NDArrayFactory::create<float>('c', {bS, iC, iH*factorH, iW*factorW}, {1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 4.f, 4.f, 4.f, 3.f, 3.f, 3.f, 4.f, 4.f, 4.f,
                                 5.f, 5.f, 5.f, 6.f, 6.f, 6.f, 5.f, 5.f, 5.f, 6.f, 6.f, 6.f, 7.f, 7.f, 7.f, 8.f, 8.f, 8.f, 7.f, 7.f, 7.f, 8.f, 8.f, 8.f, 9.f, 9.f, 9.f, 10.f, 10.f, 10.f, 9.f, 9.f, 9.f, 10.f, 10.f, 10.f, 11.f, 11.f, 11.f, 12.f, 12.f, 12.f, 11.f, 11.f, 11.f, 12.f, 12.f, 12.f,
                                13.f, 13.f, 13.f, 14.f, 14.f, 14.f, 13.f, 13.f, 13.f, 14.f, 14.f, 14.f, 15.f, 15.f, 15.f, 16.f, 16.f, 16.f, 15.f, 15.f, 15.f, 16.f, 16.f, 16.f, 17.f, 17.f, 17.f, 18.f, 18.f, 18.f, 17.f, 17.f, 17.f, 18.f, 18.f, 18.f, 19.f, 19.f, 19.f, 20.f, 20.f, 20.f, 19.f, 19.f, 19.f, 20.f, 20.f, 20.f,
                                21.f, 21.f, 21.f, 22.f, 22.f, 22.f, 21.f, 21.f, 21.f, 22.f, 22.f, 22.f, 23.f, 23.f, 23.f, 24.f, 24.f, 24.f, 23.f, 23.f, 23.f, 24.f, 24.f, 24.f, 25.f, 25.f, 25.f, 26.f, 26.f, 26.f, 25.f, 25.f, 25.f, 26.f, 26.f, 26.f, 27.f, 27.f, 27.f, 28.f, 28.f, 28.f, 27.f, 27.f, 27.f, 28.f, 28.f, 28.f,
                                29.f, 29.f, 29.f, 30.f, 30.f, 30.f, 29.f, 29.f, 29.f, 30.f, 30.f, 30.f, 31.f, 31.f, 31.f, 32.f, 32.f, 32.f, 31.f, 31.f, 31.f, 32.f, 32.f, 32.f,
                                33.f, 33.f, 33.f, 34.f, 34.f, 34.f, 33.f, 33.f, 33.f, 34.f, 34.f, 34.f, 35.f, 35.f, 35.f, 36.f, 36.f, 36.f, 35.f, 35.f, 35.f, 36.f, 36.f, 36.f});

    sd::ops::upsampling2d op;
    auto results = op.evaluate({&input}, {factorH, factorW, isNCHW});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, upsampling3d_test1) {

    const int bS=3,  iD=2,iH=2,iW=2,  iC=3;
    const int factorD=2,factorH=3,factorW=2;
    const int isNCDHW = 0;                    // data format, default is NCHW

    auto input  = NDArrayFactory::create<float>('c', {bS, iD, iH, iW, iC});
    input.linspace(1);

    auto expOutput = NDArrayFactory::create<float>('c', {bS, iD*factorD, iH*factorH, iW*factorW, iC}, {1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 4.f, 5.f, 6.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 4.f, 5.f, 6.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 10.f, 11.f, 12.f, 7.f, 8.f, 9.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 10.f, 11.f, 12.f,
             7.f, 8.f, 9.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 10.f, 11.f, 12.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 4.f, 5.f, 6.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 4.f, 5.f, 6.f, 1.f, 2.f, 3.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 10.f, 11.f, 12.f,
             7.f, 8.f, 9.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 10.f, 11.f, 12.f, 7.f, 8.f, 9.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 16.f, 17.f, 18.f, 13.f, 14.f, 15.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 16.f, 17.f, 18.f, 13.f, 14.f, 15.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 16.f, 17.f, 18.f,
            19.f, 20.f, 21.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 22.f, 23.f, 24.f, 19.f, 20.f, 21.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 22.f, 23.f, 24.f, 19.f, 20.f, 21.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 22.f, 23.f, 24.f, 13.f, 14.f, 15.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 16.f, 17.f, 18.f, 13.f, 14.f, 15.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 16.f, 17.f, 18.f,
            13.f, 14.f, 15.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 22.f, 23.f, 24.f, 19.f, 20.f, 21.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 22.f, 23.f, 24.f, 19.f, 20.f, 21.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 28.f, 29.f, 30.f,
            25.f, 26.f, 27.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 28.f, 29.f, 30.f, 25.f, 26.f, 27.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 28.f, 29.f, 30.f, 31.f, 32.f, 33.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 34.f, 35.f, 36.f, 31.f, 32.f, 33.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 34.f, 35.f, 36.f, 31.f, 32.f, 33.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 34.f, 35.f, 36.f,
            25.f, 26.f, 27.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 28.f, 29.f, 30.f, 25.f, 26.f, 27.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 28.f, 29.f, 30.f, 25.f, 26.f, 27.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 28.f, 29.f, 30.f, 31.f, 32.f, 33.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 34.f, 35.f, 36.f, 31.f, 32.f, 33.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 34.f, 35.f, 36.f,
            31.f, 32.f, 33.f, 31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 40.f, 41.f, 42.f, 37.f, 38.f, 39.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 40.f, 41.f, 42.f, 37.f, 38.f, 39.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 40.f, 41.f, 42.f, 43.f, 44.f, 45.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 46.f, 47.f, 48.f,
            43.f, 44.f, 45.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 46.f, 47.f, 48.f, 43.f, 44.f, 45.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 46.f, 47.f, 48.f, 37.f, 38.f, 39.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 40.f, 41.f, 42.f, 37.f, 38.f, 39.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 40.f, 41.f, 42.f, 37.f, 38.f, 39.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 40.f, 41.f, 42.f,
            43.f, 44.f, 45.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 46.f, 47.f, 48.f, 43.f, 44.f, 45.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 46.f, 47.f, 48.f, 43.f, 44.f, 45.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 46.f, 47.f, 48.f, 49.f, 50.f, 51.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 52.f, 53.f, 54.f, 49.f, 50.f, 51.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 52.f, 53.f, 54.f,
            49.f, 50.f, 51.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 52.f, 53.f, 54.f, 55.f, 56.f, 57.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 58.f, 59.f, 60.f, 55.f, 56.f, 57.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 58.f, 59.f, 60.f, 55.f, 56.f, 57.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 58.f, 59.f, 60.f, 49.f, 50.f, 51.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 52.f, 53.f, 54.f,
            49.f, 50.f, 51.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 52.f, 53.f, 54.f, 49.f, 50.f, 51.f, 49.f, 50.f, 51.f, 52.f, 53.f, 54.f, 52.f, 53.f, 54.f, 55.f, 56.f, 57.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 58.f, 59.f, 60.f, 55.f, 56.f, 57.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 58.f, 59.f, 60.f, 55.f, 56.f, 57.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 58.f, 59.f, 60.f,
            61.f, 62.f, 63.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 64.f, 65.f, 66.f, 61.f, 62.f, 63.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 64.f, 65.f, 66.f, 61.f, 62.f, 63.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 64.f, 65.f, 66.f, 67.f, 68.f, 69.f, 67.f, 68.f, 69.f, 70.f, 71.f, 72.f, 70.f, 71.f, 72.f, 67.f, 68.f, 69.f, 67.f, 68.f, 69.f, 70.f, 71.f, 72.f, 70.f, 71.f, 72.f,
            67.f, 68.f, 69.f, 67.f, 68.f, 69.f, 70.f, 71.f, 72.f, 70.f, 71.f, 72.f, 61.f, 62.f, 63.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 64.f, 65.f, 66.f, 61.f, 62.f, 63.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 64.f, 65.f, 66.f, 61.f, 62.f, 63.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 64.f, 65.f, 66.f, 67.f, 68.f, 69.f, 67.f, 68.f, 69.f, 70.f, 71.f, 72.f, 70.f, 71.f, 72.f,
            67.f, 68.f, 69.f, 67.f, 68.f, 69.f, 70.f, 71.f, 72.f, 70.f, 71.f, 72.f, 67.f, 68.f, 69.f, 67.f, 68.f, 69.f, 70.f, 71.f, 72.f, 70.f, 71.f, 72.f});

    sd::ops::upsampling3d op;
    auto results = op.evaluate({&input}, {factorD, factorH, factorW, isNCDHW});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, upsampling3d_test2) {

    const int bS=3,  iD=2,iH=2,iW=2,  iC=3;
    const int factorD=2,factorH=3,factorW=2;
    const int isNCDHW = 1;                    // data format, default is NCHW

    auto input = NDArrayFactory::create<float>('c', {bS, iC, iD, iH, iW});
    input.linspace(1);

    auto expOutput = NDArrayFactory::create<float>('c', {bS, iC, iD*factorD, iH*factorH, iW*factorW}, { 1.f, 1.f, 2.f, 2.f, 1.f, 1.f, 2.f, 2.f, 1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f, 3.f, 3.f, 4.f, 4.f, 3.f, 3.f, 4.f, 4.f, 1.f, 1.f, 2.f, 2.f, 1.f, 1.f, 2.f, 2.f, 1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f, 3.f, 3.f, 4.f, 4.f, 3.f, 3.f, 4.f, 4.f, 5.f, 5.f, 6.f, 6.f, 5.f, 5.f, 6.f, 6.f, 5.f, 5.f, 6.f, 6.f, 7.f, 7.f, 8.f, 8.f, 7.f, 7.f, 8.f, 8.f, 7.f, 7.f, 8.f, 8.f,
             5.f, 5.f, 6.f, 6.f, 5.f, 5.f, 6.f, 6.f, 5.f, 5.f, 6.f, 6.f, 7.f, 7.f, 8.f, 8.f, 7.f, 7.f, 8.f, 8.f, 7.f, 7.f, 8.f, 8.f, 9.f, 9.f, 10.f, 10.f, 9.f, 9.f, 10.f, 10.f, 9.f, 9.f, 10.f, 10.f, 11.f, 11.f, 12.f, 12.f, 11.f, 11.f, 12.f, 12.f, 11.f, 11.f, 12.f, 12.f, 9.f, 9.f, 10.f, 10.f, 9.f, 9.f, 10.f, 10.f, 9.f, 9.f, 10.f, 10.f, 11.f, 11.f, 12.f, 12.f, 11.f, 11.f, 12.f, 12.f, 11.f, 11.f, 12.f, 12.f,
            13.f, 13.f, 14.f, 14.f, 13.f, 13.f, 14.f, 14.f, 13.f, 13.f, 14.f, 14.f, 15.f, 15.f, 16.f, 16.f, 15.f, 15.f, 16.f, 16.f, 15.f, 15.f, 16.f, 16.f, 13.f, 13.f, 14.f, 14.f, 13.f, 13.f, 14.f, 14.f, 13.f, 13.f, 14.f, 14.f, 15.f, 15.f, 16.f, 16.f, 15.f, 15.f, 16.f, 16.f, 15.f, 15.f, 16.f, 16.f, 17.f, 17.f, 18.f, 18.f, 17.f, 17.f, 18.f, 18.f, 17.f, 17.f, 18.f, 18.f, 19.f, 19.f, 20.f, 20.f, 19.f, 19.f, 20.f, 20.f, 19.f, 19.f, 20.f, 20.f,
            17.f, 17.f, 18.f, 18.f, 17.f, 17.f, 18.f, 18.f, 17.f, 17.f, 18.f, 18.f, 19.f, 19.f, 20.f, 20.f, 19.f, 19.f, 20.f, 20.f, 19.f, 19.f, 20.f, 20.f, 21.f, 21.f, 22.f, 22.f, 21.f, 21.f, 22.f, 22.f, 21.f, 21.f, 22.f, 22.f, 23.f, 23.f, 24.f, 24.f, 23.f, 23.f, 24.f, 24.f, 23.f, 23.f, 24.f, 24.f, 21.f, 21.f, 22.f, 22.f, 21.f, 21.f, 22.f, 22.f, 21.f, 21.f, 22.f, 22.f, 23.f, 23.f, 24.f, 24.f, 23.f, 23.f, 24.f, 24.f, 23.f, 23.f, 24.f, 24.f,
            25.f, 25.f, 26.f, 26.f, 25.f, 25.f, 26.f, 26.f, 25.f, 25.f, 26.f, 26.f, 27.f, 27.f, 28.f, 28.f, 27.f, 27.f, 28.f, 28.f, 27.f, 27.f, 28.f, 28.f, 25.f, 25.f, 26.f, 26.f, 25.f, 25.f, 26.f, 26.f, 25.f, 25.f, 26.f, 26.f, 27.f, 27.f, 28.f, 28.f, 27.f, 27.f, 28.f, 28.f, 27.f, 27.f, 28.f, 28.f, 29.f, 29.f, 30.f, 30.f, 29.f, 29.f, 30.f, 30.f, 29.f, 29.f, 30.f, 30.f, 31.f, 31.f, 32.f, 32.f, 31.f, 31.f, 32.f, 32.f, 31.f, 31.f, 32.f, 32.f,
            29.f, 29.f, 30.f, 30.f, 29.f, 29.f, 30.f, 30.f, 29.f, 29.f, 30.f, 30.f, 31.f, 31.f, 32.f, 32.f, 31.f, 31.f, 32.f, 32.f, 31.f, 31.f, 32.f, 32.f, 33.f, 33.f, 34.f, 34.f, 33.f, 33.f, 34.f, 34.f, 33.f, 33.f, 34.f, 34.f, 35.f, 35.f, 36.f, 36.f, 35.f, 35.f, 36.f, 36.f, 35.f, 35.f, 36.f, 36.f, 33.f, 33.f, 34.f, 34.f, 33.f, 33.f, 34.f, 34.f, 33.f, 33.f, 34.f, 34.f, 35.f, 35.f, 36.f, 36.f, 35.f, 35.f, 36.f, 36.f, 35.f, 35.f, 36.f, 36.f,
            37.f, 37.f, 38.f, 38.f, 37.f, 37.f, 38.f, 38.f, 37.f, 37.f, 38.f, 38.f, 39.f, 39.f, 40.f, 40.f, 39.f, 39.f, 40.f, 40.f, 39.f, 39.f, 40.f, 40.f, 37.f, 37.f, 38.f, 38.f, 37.f, 37.f, 38.f, 38.f, 37.f, 37.f, 38.f, 38.f, 39.f, 39.f, 40.f, 40.f, 39.f, 39.f, 40.f, 40.f, 39.f, 39.f, 40.f, 40.f, 41.f, 41.f, 42.f, 42.f, 41.f, 41.f, 42.f, 42.f, 41.f, 41.f, 42.f, 42.f, 43.f, 43.f, 44.f, 44.f, 43.f, 43.f, 44.f, 44.f, 43.f, 43.f, 44.f, 44.f,
            41.f, 41.f, 42.f, 42.f, 41.f, 41.f, 42.f, 42.f, 41.f, 41.f, 42.f, 42.f, 43.f, 43.f, 44.f, 44.f, 43.f, 43.f, 44.f, 44.f, 43.f, 43.f, 44.f, 44.f, 45.f, 45.f, 46.f, 46.f, 45.f, 45.f, 46.f, 46.f, 45.f, 45.f, 46.f, 46.f, 47.f, 47.f, 48.f, 48.f, 47.f, 47.f, 48.f, 48.f, 47.f, 47.f, 48.f, 48.f, 45.f, 45.f, 46.f, 46.f, 45.f, 45.f, 46.f, 46.f, 45.f, 45.f, 46.f, 46.f, 47.f, 47.f, 48.f, 48.f, 47.f, 47.f, 48.f, 48.f, 47.f, 47.f, 48.f, 48.f,
            49.f, 49.f, 50.f, 50.f, 49.f, 49.f, 50.f, 50.f, 49.f, 49.f, 50.f, 50.f, 51.f, 51.f, 52.f, 52.f, 51.f, 51.f, 52.f, 52.f, 51.f, 51.f, 52.f, 52.f, 49.f, 49.f, 50.f, 50.f, 49.f, 49.f, 50.f, 50.f, 49.f, 49.f, 50.f, 50.f, 51.f, 51.f, 52.f, 52.f, 51.f, 51.f, 52.f, 52.f, 51.f, 51.f, 52.f, 52.f, 53.f, 53.f, 54.f, 54.f, 53.f, 53.f, 54.f, 54.f, 53.f, 53.f, 54.f, 54.f, 55.f, 55.f, 56.f, 56.f, 55.f, 55.f, 56.f, 56.f, 55.f, 55.f, 56.f, 56.f,
            53.f, 53.f, 54.f, 54.f, 53.f, 53.f, 54.f, 54.f, 53.f, 53.f, 54.f, 54.f, 55.f, 55.f, 56.f, 56.f, 55.f, 55.f, 56.f, 56.f, 55.f, 55.f, 56.f, 56.f, 57.f, 57.f, 58.f, 58.f, 57.f, 57.f, 58.f, 58.f, 57.f, 57.f, 58.f, 58.f, 59.f, 59.f, 60.f, 60.f, 59.f, 59.f, 60.f, 60.f, 59.f, 59.f, 60.f, 60.f, 57.f, 57.f, 58.f, 58.f, 57.f, 57.f, 58.f, 58.f, 57.f, 57.f, 58.f, 58.f, 59.f, 59.f, 60.f, 60.f, 59.f, 59.f, 60.f, 60.f, 59.f, 59.f, 60.f, 60.f,
            61.f, 61.f, 62.f, 62.f, 61.f, 61.f, 62.f, 62.f, 61.f, 61.f, 62.f, 62.f, 63.f, 63.f, 64.f, 64.f, 63.f, 63.f, 64.f, 64.f, 63.f, 63.f, 64.f, 64.f, 61.f, 61.f, 62.f, 62.f, 61.f, 61.f, 62.f, 62.f, 61.f, 61.f, 62.f, 62.f, 63.f, 63.f, 64.f, 64.f, 63.f, 63.f, 64.f, 64.f, 63.f, 63.f, 64.f, 64.f, 65.f, 65.f, 66.f, 66.f, 65.f, 65.f, 66.f, 66.f, 65.f, 65.f, 66.f, 66.f, 67.f, 67.f, 68.f, 68.f, 67.f, 67.f, 68.f, 68.f, 67.f, 67.f, 68.f, 68.f,
            65.f, 65.f, 66.f, 66.f, 65.f, 65.f, 66.f, 66.f, 65.f, 65.f, 66.f, 66.f, 67.f, 67.f, 68.f, 68.f, 67.f, 67.f, 68.f, 68.f, 67.f, 67.f, 68.f, 68.f, 69.f, 69.f, 70.f, 70.f, 69.f, 69.f, 70.f, 70.f, 69.f, 69.f, 70.f, 70.f, 71.f, 71.f, 72.f, 72.f, 71.f, 71.f, 72.f, 72.f, 71.f, 71.f, 72.f, 72.f, 69.f, 69.f, 70.f, 70.f, 69.f, 69.f, 70.f, 70.f, 69.f, 69.f, 70.f, 70.f, 71.f, 71.f, 72.f, 72.f, 71.f, 71.f, 72.f, 72.f, 71.f, 71.f, 72.f, 72.f});

    sd::ops::upsampling3d op;
    auto results = op.evaluate({&input}, {factorD, factorH, factorW, isNCDHW});
    auto* output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));

}


//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, upsampling3d_bp_test1) {

    const int bS=1,  iD=2,iH=2,iW=2,  iC=1;
    const int factorD=2, factorH=2, factorW=2;
    const int isNCDHW = 1;                    // data format, default is NCHW

    auto input  = NDArrayFactory::create<float>('c', {bS, iC, iD, iH, iW});
    auto gradO  = NDArrayFactory::create<float>('c', {bS, iC, iD*factorD, iH*factorH, iW*factorW});
    gradO = 1.;

    auto expGradI = NDArrayFactory::create<float>('c', {bS, iC, iD, iH, iW});
    expGradI = 8.;

    sd::ops::upsampling3d_bp op;
    auto results = op.evaluate({&input, &gradO}, {isNCDHW});
    auto* gradI = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));
}

TYPED_TEST(TypedConvolutionTests1, conv2D_input_BP_test1) {

    auto inputShape = NDArrayFactory::create<TypeParam>('c', {4}, {2, 1, 4, 4});
    auto weights = NDArrayFactory::create<TypeParam>('c', {2, 1, 3, 3});
    auto epsilonNext = NDArrayFactory::create<TypeParam>('c', {2, 2, 4, 4});
    auto shapeArr = NDArrayFactory::create<TypeParam>('c', {2, 1, 4, 4});


    TypeParam _expEpsB[] = {952.0, 1540.0, 1636.0, 1180.0, 1791.0, 2886.0, 3057.0, 2193.0, 2223.0, 3570.0, 3741.0, 2673.0, 1900.0, 3028.0, 3160.0, 2240.0, 2872.0, 4612.0, 4708.0, 3356.0, 5247.0, 8358.0, 8529.0, 6033.0, 5679.0, 9042.0, 9213.0, 6513.0, 4588.0, 7252.0, 7384.0, 5184.0};
    NDArray expEps(_expEpsB, shapeArr.shapeInfo());

    weights.linspace(1);
    epsilonNext.linspace(1);
    weights.permutei({2,3,1,0});

    sd::ops::conv2d_input_bp op;

    auto results = op.evaluate({&inputShape, &weights, &epsilonNext}, {},  {3, 3, 1, 1, 0, 0, 1, 1, 1});

    ASSERT_TRUE(results.size() == 1);

    auto epsilon = results.at(0);

    ASSERT_TRUE(shapeArr.isSameShape(epsilon));
    ASSERT_TRUE(expEps.equalsTo(epsilon));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, upsampling3d_bp_test3) {

    const int bS=1,  iD=3,iH=3,iW=3,  iC=2;
    const int factorD=2, factorH=2, factorW=2;
    const int isNCDHW = 1;                    // data format, default is NCHW

    NDArray input('c', {bS, iC, iD, iH, iW}, sd::DataType::FLOAT32);
    NDArray gradO('c', {bS, iC, iD*factorD, iH*factorH, iW*factorW}, {0.6793504, 0.35508695, 0.84278935, 0.20031333, 0.7014987, 0.31069338,
        0.44793984, 0.93800974, 0.32667395, 0.15187258, 0.38331753, 0.78212297, 0.1988072, 0.7985636, 0.1632634, 0.14696825, 0.26089668,
        0.13505761, 0.7562093, 0.27545404, 0.36908787, 0.09282647, 0.83649176, 0.26841334, 0.09506222, 0.31279507, 0.13591796, 0.5175439,
        0.32870287, 0.061735712, 0.39643127, 0.248016, 0.5489592, 0.115046196, 0.8143622, 0.7215636, 0.40449402, 0.29908907, 0.4038839,
        0.9883108, 0.022296403, 0.927782, 0.3184157, 0.0685462, 0.28453344, 0.23272, 0.35214192, 0.058909304, 0.7112212, 0.6744568, 0.19694561,
        0.6994972, 0.0743224, 0.42042503, 0.5842631, 0.14957358, 0.44640633, 0.72307247, 0.06448108, 0.48307765, 0.8759956, 0.5698191, 0.4458631,
        0.5277549, 0.016646361, 0.753678, 0.14063567, 0.7541292, 0.16193217, 0.7750374, 0.3326449, 0.11739397, 0.017710684, 0.60847557, 0.52515227,
        0.9171938, 0.84989065, 0.5894228, 0.85227835, 0.39063585, 0.88968325, 0.6694452, 0.698873, 0.96147966, 0.15740126, 0.15736352, 0.49352047,
        0.5699365, 0.12683152, 0.11572781, 0.7863682, 0.737939, 0.49007934, 0.6084143, 0.9564999, 0.3900982, 0.14730452, 0.8506447, 0.49765033,
        0.07186628, 0.08214969, 0.035314173, 0.7320408, 0.36993408, 0.8406658, 0.27389422, 0.43179566, 0.13323106, 0.19297548, 0.24689731, 0.38641843,
        0.51154125, 0.19903564, 0.1416313, 0.69769853, 0.25363067, 0.78221816, 0.9300991, 0.3355119, 0.5588076, 0.6643576, 0.018850708, 0.63755876,
        0.2904297, 0.43490165, 0.84251267, 0.46609768, 0.38139546, 0.52318525, 0.9901826, 0.9257676, 0.6434591, 0.016828254, 0.9187561, 0.22897908,
        0.0063138064, 0.66597503, 0.19036093, 0.59552056, 0.69888055, 0.22146936, 0.9124342, 0.8708221, 0.7273687, 0.52397245, 0.66288394, 0.2188415,
        0.3354802, 0.03566524, 0.5101009, 0.5017283, 0.75122046, 0.1884508, 0.7407126, 0.6253045, 0.47145858, 0.5369367, 0.19884548, 0.99008304,
        0.08256686, 0.91884845, 0.02360027, 0.98895234, 0.3751719, 0.91783875, 0.4338776, 0.6783008, 0.6667967, 0.46720362, 0.7508773, 0.52304846,
        0.76631916, 0.4187526, 0.7653719, 0.5159193, 0.42730415, 0.49462363, 0.2731735, 0.8862948, 0.043214794, 0.3197591, 0.040378205, 0.5427239,
        0.9228089, 0.045940384, 0.70047987, 0.8419288, 0.53966296, 0.009444186, 0.038044546, 0.03158029, 0.43485752, 0.9204235, 0.5478789, 0.8290083,
        0.11868837, 0.0229866, 0.6639305, 0.8757367, 0.8279557, 0.76270294, 0.43242732, 0.4713431, 0.2569212, 0.30575937, 0.44395888, 0.99384075,
        0.6127142, 0.44844577, 0.6347944, 0.098358564, 0.34233716, 0.9329664, 0.65776783, 0.108565055, 0.2052629, 0.46441218, 0.041791342, 0.89369565,
        0.7000381, 0.2106213, 0.51152664, 0.44200692, 0.8293282, 0.20901772, 0.6387249, 0.8016979, 0.11178707, 0.109545894, 0.19654618, 0.060582615,
        0.08239174, 0.64630795, 0.32862368, 0.60225064, 0.8328141, 0.5484566, 0.8120276, 0.38822946, 0.6742381, 0.34913155, 0.42887798, 0.45344824,
        0.73956585, 0.9714739, 0.42937812, 0.45185348, 0.84535813, 0.046436775, 0.8802151, 0.8676222, 0.42625394, 0.4985318, 0.42399272, 0.122144565,
        0.0060101906, 0.47253844, 0.18123977, 0.86316174, 0.5863874, 0.3852012, 0.9785553, 0.0054711984, 0.88500834, 0.020897374, 0.27467912, 0.3852802,
        0.0766939, 0.94622654, 0.38687763, 0.3308602, 0.7770494, 0.9052543, 0.22258204, 0.42207044, 0.18050623, 0.21057767, 0.012561422, 0.7977821,
        0.61251044, 0.7203693, 0.6028265, 0.6036933, 0.1446382, 0.6712341, 0.76634467, 0.4854034, 0.26634562, 0.76523924, 0.16348523, 0.2663676,
        0.96846986, 0.8273284, 0.10700377, 0.7600526, 0.6771002, 0.47963092, 0.21264452, 0.56934077, 0.5514792, 0.85725874, 0.99090636, 0.54562527,
        0.93597686, 0.21142527, 0.4628326, 0.35011524, 0.31464386, 0.31164807, 0.65928996, 0.94418925, 0.39666295, 0.9496393, 0.103756346, 0.482158,
        0.49171793, 0.4108867, 0.22594318, 0.97093135, 0.5974685, 0.34632966, 0.54835194, 0.10499302, 0.9767778, 0.55008715, 0.54379046, 0.3583731,
        0.33369112, 0.04279039, 0.24939054, 0.23943715, 0.06775989, 0.7750291, 0.24329625, 0.4327169, 0.86916673, 0.80322117, 0.049972698, 0.47177452,
        0.37419558, 0.15303156, 0.121425234, 0.75884604, 0.8191354, 0.48554084, 0.053899214, 0.7858246, 0.39219773, 0.77579063, 0.34507045, 0.46070176,
        0.14496958, 0.47706795, 0.50678796, 0.64902323, 0.3277943, 0.0017530271, 0.6536156, 0.8582253, 0.95703506, 0.9963951, 0.8239163, 0.305142,
        0.012419582, 0.9498972, 0.1595827, 0.47947606, 0.5071124, 0.78227425, 0.2066719, 0.5217094, 0.7841406, 0.5260441, 0.49798164, 0.10975622,
        0.8633349, 0.76298475, 0.14295428, 0.6131504, 0.43794408, 0.50339264, 0.4504877, 0.19235311, 0.6678411, 0.80769485, 0.67495126, 0.96461457,
        0.10535406, 0.66438645, 0.4372345, 0.93851465, 0.8635335, 0.3405871, 0.45652762, 0.3636232, 0.52931345, 0.20154329, 0.07698499, 0.6125804,
        0.3583082, 0.3894796, 0.32601944, 0.5237369, 0.66683626, 0.08541841, 0.4815708, 0.11897489, 0.97555137, 0.3602705, 0.9620871, 0.6361821,
        0.71167386, 0.5134439, 0.57761437, 0.58598644, 0.39387667, 0.6966405, 0.46841687, 0.85788506, 0.9957087, 0.051309288, 0.24846801, 0.55938333,
        0.10230542, 0.9370694, 0.57527155, 0.54656035, 0.28896323, 0.51303476, 0.8865, 0.38641605, 0.9836358}, sd::DataType::FLOAT32);

    NDArray expGradI('c', {bS, iC, iD, iH, iW}, {3.510932, 3.4310975, 3.538762, 4.148549, 2.8380678, 2.5431657, 3.3928843, 3.228055, 3.1467278,
        3.2603023, 5.611751, 4.334653, 3.3697734, 4.603307, 4.4357986, 4.32991, 3.0532732, 3.1370173, 4.181534, 2.9965065, 2.8553872, 5.2719016,
        4.5671935, 3.7027276, 3.3517184, 5.2544537, 3.5107024, 4.1496124, 3.9333878, 3.1798909, 3.1446428, 3.0932689, 3.9730802, 3.0466917,
        4.9675374, 4.769673, 3.766952, 3.6375027, 3.6492167, 4.9440994, 3.8379507, 3.467589, 4.719474, 3.1295977, 4.5177174, 4.2760015, 2.8443856,
        4.225355, 4.377341, 4.4398847, 4.710785, 4.4199953, 3.928307, 4.8769503}, sd::DataType::FLOAT32);

    sd::ops::upsampling3d_bp op;
    auto results = op.evaluate({&input, &gradO}, {isNCDHW});
    auto* gradI = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expGradI.isSameShape(gradI));
    ASSERT_TRUE(expGradI.equalsTo(gradI));

}


//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, deconv2d_test1) {

    int bS=2, oH=4,oW=4,  oC=5,iC=10,  kH=2,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       iH=3,iW=3;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    auto input    = NDArrayFactory::create<float>('c', {bS, iH, iW, iC});
    auto weights  = NDArrayFactory::create<float>('c', {kH, kW, oC, iC});
    auto exp = NDArrayFactory::create<float>('c', {bS, oH, oW, oC}, {  2.75f, 7.75f, 12.75f, 17.75f, 22.75f, 30.5f, 40.5f, 50.5f, 60.5f, 70.5f, 30.5f, 40.5f, 50.5f, 60.5f, 70.5f, 27.75f, 32.75f, 37.75f, 42.75f, 47.75f,
                                                  55.5f, 65.5f, 75.5f, 85.5f, 95.5f, 161.f, 181.f, 201.f, 221.f, 241.f, 161.f, 181.f, 201.f, 221.f, 241.f, 105.5f, 115.5f, 125.5f, 135.5f, 145.5f,
                                                  55.5f, 65.5f, 75.5f, 85.5f, 95.5f, 161.f, 181.f, 201.f, 221.f, 241.f, 161.f, 181.f, 201.f, 221.f, 241.f, 105.5f, 115.5f, 125.5f, 135.5f, 145.5f,
                                                  52.75f, 57.75f, 62.75f, 67.75f, 72.75f, 130.5f, 140.5f, 150.5f, 160.5f, 170.5f, 130.5f, 140.5f, 150.5f, 160.5f, 170.5f, 77.75f, 82.75f, 87.75f, 92.75f, 97.75f,
                                                   2.75f, 7.75f, 12.75f, 17.75f, 22.75f, 30.5f, 40.5f, 50.5f, 60.5f, 70.5f, 30.5f, 40.5f, 50.5f, 60.5f, 70.5f, 27.75f, 32.75f, 37.75f, 42.75f, 47.75f,
                                                  55.5f, 65.5f, 75.5f, 85.5f, 95.5f, 161.f, 181.f, 201.f, 221.f, 241.f, 161.f, 181.f, 201.f, 221.f, 241.f, 105.5f, 115.5f, 125.5f, 135.5f, 145.5f,
                                                  55.5f, 65.5f, 75.5f, 85.5f, 95.5f, 161.f, 181.f, 201.f, 221.f, 241.f, 161.f, 181.f, 201.f, 221.f, 241.f, 105.5f, 115.5f, 125.5f, 135.5f, 145.5f,
                                                  52.75f, 57.75f, 62.75f, 67.75f, 72.75f, 130.5f, 140.5f, 150.5f, 160.5f, 170.5f, 130.5f, 140.5f, 150.5f, 160.5f, 170.5f, 77.75f, 82.75f, 87.75f, 92.75f, 97.75f});
    input = 0.5;
    weights.linspace(0.1, 0.1);

    sd::ops::deconv2d op;
    auto results = op.evaluate({&input, &weights}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, deconv2d_test2) {

    int bS=2, iH=4,iW=4,  iC=5,oC=10,  kH=2,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=4,oW=4;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    auto input    = NDArrayFactory::create<float>('c', {bS, oH, oW, oC});
    auto weights  = NDArrayFactory::create<float>('c', {kH, kW, iC, oC});
    auto exp = NDArrayFactory::create<float>('c', {bS, iH, iW, iC}, {2.75f,    7.75f,   12.75f,   17.75f,   22.75f,  30.5f,   40.5f,   50.5f,   60.5f,   70.5f,  30.5f,   40.5f,   50.5f,   60.5f,   70.5f,  30.5f,   40.5f,   50.5f,   60.5f,   70.5f,
                                                55.5f,   65.5f,   75.5f,   85.5f,   95.5f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f,
                                                55.5f,   65.5f,   75.5f,   85.5f,   95.5f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f,
                                                55.5f,   65.5f,   75.5f,   85.5f,   95.5f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f,
                                                 2.75f,    7.75f,   12.75f,   17.75f,   22.75f,  30.5f,   40.5f,   50.5f,   60.5f,   70.5f,  30.5f,   40.5f,   50.5f,   60.5f,   70.5f,  30.5f,   40.5f,   50.5f,   60.5f,   70.5f,
                                                55.5f,   65.5f,   75.5f,   85.5f,   95.5f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f,
                                                55.5f,   65.5f,   75.5f,   85.5f,   95.5f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f,
                                                55.5f,   65.5f,   75.5f,   85.5f,   95.5f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f  });
    input = 0.5;
    weights.linspace(0.1, 0.1);

    sd::ops::deconv2d op;
    auto results = op.evaluate({&input, &weights}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, deconv2d_test3) {

    int bS=1, oH=5,oW=5,  oC=3,iC=2,  kH=2,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=2,dW=2;
    int       iH=3,iW=3;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    auto input    = NDArrayFactory::create<float>('c', {bS, iH, iW, iC});
    auto weights  = NDArrayFactory::create<float>('c', {kH, kW, oC, iC});
    auto bias     = NDArrayFactory::create<float>('c', {oC});

    auto exp = NDArrayFactory::create<float>('c', {bS, oH, oW, oC}, {-2.9f, -6.8f, -10.7f, -2.6f, -6.1f, -9.6f, -16.9f, -23.9f, -30.9f, -13.1f, -16.6f, -20.1f, -11.6f, -14.7f, -17.8f, -2.0f, -4.7f, -7.4f, -1.7f, -4.0f, -6.3f, -11.5f, -16.1f,
                                -20.7f, -8.6f, -10.9f, -13.2f, -7.1f, -9.0f, -10.9f, -27.4f, -32.8f, -38.2f, -24.4f, -29.0f, -33.6f, -65.0f, -74.2f, -83.4f, -38.2f, -42.8f, -47.4f,
                                -32.8f, -36.6f, -40.4f, -18.2f, -20.9f, -23.6f, -15.5f, -17.8f, -20.1f, -39.1f, -43.7f, -48.3f, -22.4f, -24.7f, -27.0f, -18.5f, -20.4f, -22.3f, -10.1f, -11.6f, -13.1f,
                                -7.4f, -8.5f, -9.6f, -19.3f, -21.5f, -23.7f, -10.7f, -11.8f, -12.9f, -6.8f, -7.5f, -8.2f});

    input.linspace(-10, 0.5);
    weights.linspace(0.1, 0.1);
    bias = 0.2;

    sd::ops::deconv2d op;
    auto results = op.evaluate({&input, &weights}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, deconv2d_test4) {

    NDArray input('c', {2, 3, 4, 4}, sd::DataType::FLOAT32);
    NDArray weights('c', {3, 3, 5, 5}, sd::DataType::FLOAT32);
    NDArray exp('c', {2,3,8,8}, {6276.0,12831.0,19668.0,26790.0,27012.0,20703.0,14100.0,7200.0,13719.0,28023.0,42918.0,58410.0,58902.0,45105.0,30693.0,15660.0,22389.0,45696.0,69930.0,95100.0,95910.0,73386.0,49899.0,25440.0,32346.0,65970.0,
                                100884.0,137100.0,138276.0,105726.0,71838.0,36600.0,33726.0,68790.0,105204.0,142980.0,144156.0,110226.0,74898.0,38160.0,27555.0,56154.0,85806.0,116520.0,117474.0,89748.0,60933.0,31020.0,19917.0,40557.0,61926.0,
                                84030.0,84714.0,64671.0,43875.0,22320.0,10752.0,21879.0,33384.0,45270.0,45636.0,34815.0,23604.0,12000.0,7551.0,15456.0,23718.0,32340.0,32562.0,24978.0,17025.0,8700.0,16569.0,33873.0,51918.0,70710.0,71202.0,
                                54555.0,37143.0,18960.0,27114.0,55371.0,84780.0,115350.0,116160.0,88911.0,60474.0,30840.0,39246.0,80070.0,122484.0,166500.0,167676.0,128226.0,87138.0,44400.0,40626.0,82890.0,126804.0,172380.0,173556.0,132726.0,
                                90198.0,45960.0,33180.0,67629.0,103356.0,140370.0,141324.0,107973.0,73308.0,37320.0,23967.0,48807.0,74526.0,101130.0,101814.0,77721.0,52725.0,26820.0,12927.0,26304.0,40134.0,54420.0,54786.0,41790.0,28329.0,14400.0,
                                8826.0,18081.0,27768.0,37890.0,38112.0,29253.0,19950.0,10200.0,19419.0,39723.0,60918.0,83010.0,83502.0,64005.0,43593.0,22260.0,31839.0,65046.0,99630.0,135600.0,136410.0,104436.0,71049.0,36240.0,46146.0,94170.0,
                                144084.0,195900.0,197076.0,150726.0,102438.0,52200.0,47526.0,96990.0,148404.0,201780.0,202956.0,155226.0,105498.0,53760.0,38805.0,79104.0,120906.0,164220.0,165174.0,126198.0,85683.0,43620.0,28017.0,57057.0,87126.0,
                                118230.0,118914.0,90771.0,61575.0,31320.0,15102.0,30729.0,46884.0,63570.0,63936.0,48765.0,33054.0,16800.0,17220.0,34863.0,52932.0,71430.0,72228.0,54831.0,36996.0,18720.0,36327.0,73527.0,111606.0,150570.0,152214.0,
                                115521.0,77925.0,39420.0,57381.0,116112.0,176202.0,237660.0,240198.0,182250.0,122907.0,62160.0,80442.0,162738.0,246900.0,332940.0,336420.0,255198.0,172062.0,87000.0,84702.0,171318.0,259860.0,350340.0,353820.0,
                                268338.0,180882.0,91440.0,66867.0,135210.0,205038.0,276360.0,279042.0,211572.0,142581.0,72060.0,46845.0,94701.0,143574.0,193470.0,195306.0,148047.0,99747.0,50400.0,24576.0,49671.0,75288.0,101430.0,102372.0,77583.0,
                                52260.0,26400.0,22095.0,44688.0,67782.0,91380.0,92178.0,69906.0,47121.0,23820.0,46377.0,93777.0,142206.0,191670.0,193314.0,146571.0,98775.0,49920.0,72906.0,147387.0,223452.0,301110.0,303648.0,230175.0,155082.0,
                                78360.0,101742.0,205638.0,311700.0,419940.0,423420.0,320898.0,216162.0,109200.0,106002.0,214218.0,324660.0,437340.0,440820.0,334038.0,224982.0,113640.0,83292.0,168285.0,254988.0,343410.0,346092.0,262197.0,176556.0,
                                89160.0,58095.0,117351.0,177774.0,239370.0,241206.0,182697.0,122997.0,62100.0,30351.0,61296.0,92838.0,124980.0,125922.0,95358.0,64185.0,32400.0,26970.0,54513.0,82632.0,111330.0,112128.0,84981.0,57246.0,28920.0,56427.0,114027.0,172806.0,232770.0,234414.0,177621.0,119625.0,60420.0,88431.0,178662.0,270702.0,364560.0,367098.0,278100.0,187257.0,94560.0,123042.0,248538.0,376500.0,506940.0,510420.0,386598.0,260262.0,131400.0,127302.0,257118.0,389460.0,524340.0,527820.0,399738.0,269082.0,135840.0,99717.0,201360.0,304938.0,410460.0,413142.0,312822.0,210531.0,106260.0,69345.0,140001.0,211974.0,285270.0,287106.0,217347.0,146247.0,73800.0,36126.0,72921.0,110388.0,148530.0,149472.0,113133.0,76110.0,38400.0}, sd::DataType::FLOAT32);

    input.linspace(1);
    weights.linspace(1);
    weights.permutei({2,3,1,0});

    sd::ops::deconv2d op;
    auto result = op.evaluate({&input, &weights}, {5, 5, 1, 1, 0, 0, 1, 1, 0, 0});

    auto z = result.at(0);
    // z->printShapeInfo();
    // z->printBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, deconv2d_test5) {
    Nd4jLong _expS[] = {4, 2, 3, 8, 8, 192, 64, 8, 1, 16384, 1, 99};
    double _expB[] = {6276.0,12831.0,19668.0,26790.0,27012.0,20703.0,14100.0,7200.0,13719.0,28023.0,42918.0,58410.0,58902.0,45105.0,30693.0,15660.0,22389.0,45696.0,69930.0,95100.0,95910.0,73386.0,49899.0,25440.0,32346.0,65970.0,100884.0,137100.0,138276.0,105726.0,71838.0,36600.0,33726.0,68790.0,105204.0,142980.0,144156.0,110226.0,74898.0,38160.0,27555.0,56154.0,85806.0,116520.0,117474.0,89748.0,60933.0,31020.0,19917.0,40557.0,61926.0,84030.0,84714.0,64671.0,43875.0,22320.0,10752.0,21879.0,33384.0,45270.0,45636.0,34815.0,23604.0,12000.0,7551.0,15456.0,23718.0,32340.0,32562.0,24978.0,17025.0,8700.0,16569.0,33873.0,51918.0,70710.0,71202.0,54555.0,37143.0,18960.0,27114.0,55371.0,84780.0,115350.0,116160.0,88911.0,60474.0,30840.0,39246.0,80070.0,122484.0,166500.0,167676.0,128226.0,87138.0,44400.0,40626.0,82890.0,126804.0,172380.0,173556.0,132726.0,90198.0,45960.0,33180.0,67629.0,103356.0,140370.0,141324.0,107973.0,73308.0,37320.0,23967.0,48807.0,74526.0,101130.0,101814.0,77721.0,52725.0,26820.0,12927.0,26304.0,40134.0,54420.0,54786.0,41790.0,28329.0,14400.0,8826.0,18081.0,27768.0,37890.0,38112.0,29253.0,19950.0,10200.0,19419.0,39723.0,60918.0,83010.0,83502.0,64005.0,43593.0,22260.0,31839.0,65046.0,99630.0,135600.0,136410.0,104436.0,71049.0,36240.0,46146.0,94170.0,144084.0,195900.0,197076.0,150726.0,102438.0,52200.0,47526.0,96990.0,148404.0,201780.0,202956.0,155226.0,105498.0,53760.0,38805.0,79104.0,120906.0,164220.0,165174.0,126198.0,85683.0,43620.0,28017.0,57057.0,87126.0,118230.0,118914.0,90771.0,61575.0,31320.0,15102.0,30729.0,46884.0,63570.0,63936.0,48765.0,33054.0,16800.0,17220.0,34863.0,52932.0,71430.0,72228.0,54831.0,36996.0,18720.0,36327.0,73527.0,111606.0,150570.0,152214.0,115521.0,77925.0,39420.0,57381.0,116112.0,176202.0,237660.0,240198.0,182250.0,122907.0,62160.0,80442.0,162738.0,246900.0,332940.0,336420.0,255198.0,172062.0,87000.0,84702.0,171318.0,259860.0,350340.0,353820.0,268338.0,180882.0,91440.0,66867.0,135210.0,205038.0,276360.0,279042.0,211572.0,142581.0,72060.0,46845.0,94701.0,143574.0,193470.0,195306.0,148047.0,99747.0,50400.0,24576.0,49671.0,75288.0,101430.0,102372.0,77583.0,52260.0,26400.0,22095.0,44688.0,67782.0,91380.0,92178.0,69906.0,47121.0,23820.0,46377.0,93777.0,142206.0,191670.0,193314.0,146571.0,98775.0,49920.0,72906.0,147387.0,223452.0,301110.0,303648.0,230175.0,155082.0,78360.0,101742.0,205638.0,311700.0,419940.0,423420.0,320898.0,216162.0,109200.0,106002.0,214218.0,324660.0,437340.0,440820.0,334038.0,224982.0,113640.0,83292.0,168285.0,254988.0,343410.0,346092.0,262197.0,176556.0,89160.0,58095.0,117351.0,177774.0,239370.0,241206.0,182697.0,122997.0,62100.0,30351.0,61296.0,92838.0,124980.0,125922.0,95358.0,64185.0,32400.0,26970.0,54513.0,82632.0,111330.0,112128.0,84981.0,57246.0,28920.0,56427.0,114027.0,172806.0,232770.0,234414.0,177621.0,119625.0,60420.0,88431.0,178662.0,270702.0,364560.0,367098.0,278100.0,187257.0,94560.0,123042.0,248538.0,376500.0,506940.0,510420.0,386598.0,260262.0,131400.0,127302.0,257118.0,389460.0,524340.0,527820.0,399738.0,269082.0,135840.0,99717.0,201360.0,304938.0,410460.0,413142.0,312822.0,210531.0,106260.0,69345.0,140001.0,211974.0,285270.0,287106.0,217347.0,146247.0,73800.0,36126.0,72921.0,110388.0,148530.0,149472.0,113133.0,76110.0,38400.0,};
    NDArray exp(_expB, _expS);

    auto input = NDArrayFactory::create<double>('c', {2, 3, 4, 4});
    auto weights = NDArrayFactory::create<double>('c', {3, 3, 5, 5});
    auto z = NDArrayFactory::create<double>('c', {2, 3, 8, 8});

    input.linspace(1);
    weights.linspace(1);
    weights.permutei({2,3,1,0});

    sd::ops::deconv2d op;
    auto result = op.execute({&input, &weights}, {&z}, {5, 5, 1, 1, 0, 0, 1, 1, 0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result);

    ASSERT_TRUE(exp.isSameShape(&z));
    ASSERT_TRUE(exp.equalsTo(&z));
}

TYPED_TEST(TypedConvolutionTests1, deconv2d_test6) {

    int bS=2, iH=4,iW=4,  iC=3,oC=3,  kH=5,kW=5,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=8,oW=8;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    auto input = NDArrayFactory::create<TypeParam>('c', {bS, iC, iH, iW});
    auto weights = NDArrayFactory::create<TypeParam>('c', {kH, kW, oC, iC}, {1.f, 76.f, 151.f, 26.f, 101.f, 176.f, 51.f, 126.f, 201.f, 2.f, 77.f, 152.f, 27.f, 102.f, 177.f, 52.f, 127.f, 202.f, 3.f, 78.f, 153.f, 28.f, 103.f, 178.f, 53.f, 128.f, 203.f,
                                    4.f, 79.f, 154.f, 29.f, 104.f, 179.f, 54.f, 129.f, 204.f, 5.f, 80.f, 155.f, 30.f, 105.f, 180.f, 55.f, 130.f, 205.f, 6.f, 81.f, 156.f, 31.f, 106.f, 181.f, 56.f, 131.f, 206.f,
                                    7.f, 82.f, 157.f, 32.f, 107.f, 182.f, 57.f, 132.f, 207.f, 8.f, 83.f, 158.f, 33.f, 108.f, 183.f, 58.f, 133.f, 208.f, 9.f, 84.f, 159.f, 34.f, 109.f, 184.f, 59.f, 134.f, 209.f,
                                    10.f, 85.f, 160.f, 35.f, 110.f, 185.f, 60.f, 135.f, 210.f, 11.f, 86.f, 161.f, 36.f, 111.f, 186.f, 61.f, 136.f, 211.f, 12.f, 87.f, 162.f, 37.f, 112.f, 187.f, 62.f, 137.f, 212.f,
                                    13.f, 88.f, 163.f, 38.f, 113.f, 188.f, 63.f, 138.f, 213.f, 14.f, 89.f, 164.f, 39.f, 114.f, 189.f, 64.f, 139.f, 214.f, 15.f, 90.f, 165.f, 40.f, 115.f, 190.f, 65.f, 140.f, 215.f,
                                    16.f, 91.f, 166.f, 41.f, 116.f, 191.f, 66.f, 141.f, 216.f, 17.f, 92.f, 167.f, 42.f, 117.f, 192.f, 67.f, 142.f, 217.f, 18.f, 93.f, 168.f, 43.f, 118.f, 193.f, 68.f, 143.f, 218.f,
                                    19.f, 94.f, 169.f, 44.f, 119.f, 194.f, 69.f, 144.f, 219.f, 20.f, 95.f, 170.f, 45.f, 120.f, 195.f, 70.f, 145.f, 220.f, 21.f, 96.f, 171.f, 46.f, 121.f, 196.f, 71.f, 146.f, 221.f,
                                    22.f, 97.f, 172.f, 47.f, 122.f, 197.f, 72.f, 147.f, 222.f, 23.f, 98.f, 173.f, 48.f, 123.f, 198.f, 73.f, 148.f, 223.f, 24.f, 99.f, 174.f, 49.f, 124.f, 199.f, 74.f, 149.f, 224.f,
                                    25.f, 100.f, 175.f,50.f, 125.f, 200.f,75.f, 150.f, 225.f});

    auto exp = NDArrayFactory::create<TypeParam>('c', {bS, oC, oH, oW}, {6276.0f,   12831.0f,   19668.0f,   26790.0f,   27012.0f,   20703.0f,   14100.0f,    7200.0f,    13719.0f,   28023.0f,   42918.0f,   58410.0f,   58902.0f,   45105.0f,   30693.0f,   15660.0f,    22389.0f,   45696.0f,   69930.0f,   95100.0f,   95910.0f,   73386.0f,   49899.0f,   25440.0f,    32346.0f,   65970.0f,  100884.0f,  137100.0f,  138276.0f,  105726.0f,   71838.0f,   36600.0f,    33726.0f,   68790.0f,  105204.0f,  142980.0f,  144156.0f,  110226.0f,   74898.0f,   38160.0f,    27555.0f,   56154.0f,   85806.0f,  116520.0f,  117474.0f,   89748.0f,   60933.0f,   31020.0f,    19917.0f,   40557.0f,   61926.0f,   84030.0f,   84714.0f,   64671.0f,   43875.0f,   22320.0f,    10752.0f,   21879.0f,   33384.0f,   45270.0f,   45636.0f,   34815.0f,   23604.0f,   12000.0f,    7551.0f,   15456.0f,   23718.0f,   32340.0f,   32562.0f,   24978.0f,   17025.0f,    8700.0f,    16569.0f,   33873.0f,   51918.0f,   70710.0f,   71202.0f,   54555.0f,   37143.0f,   18960.0f,    27114.0f,   55371.0f,   84780.0f,  115350.0f,  116160.0f,   88911.0f,   60474.0f,   30840.0f,    39246.0f,   80070.0f,  122484.0f,  166500.0f,  167676.0f,  128226.0f,   87138.0f,   44400.0f,    40626.0f,   82890.0f,  126804.0f,  172380.0f,  173556.0f,  132726.0f,   90198.0f,   45960.0f,    33180.0f,   67629.0f,  103356.0f,  140370.0f,  141324.0f,  107973.0f,   73308.0f,   37320.0f,    23967.0f,   48807.0f,   74526.0f,  101130.0f,  101814.0f,   77721.0f,   52725.0f,   26820.0f,    12927.0f,   26304.0f,   40134.0f,   54420.0f,   54786.0f,   41790.0f,   28329.0f,   14400.0f,    8826.0f,   18081.0f,   27768.0f,   37890.0f,   38112.0f,   29253.0f,   19950.0f,   10200.0f,    19419.0f,   39723.0f,   60918.0f,   83010.0f,   83502.0f,   64005.0f,   43593.0f,   22260.0f,    31839.0f,   65046.0f,   99630.0f,  135600.0f,  136410.0f,  104436.0f,   71049.0f,   36240.0f,    46146.0f,   94170.0f,  144084.0f,  195900.0f,  197076.0f,  150726.0f,  102438.0f,   52200.0f,    47526.0f,   96990.0f,  148404.0f,  201780.0f,  202956.0f,  155226.0f,  105498.0f,   53760.0f,    38805.0f,   79104.0f,  120906.0f,  164220.0f,  165174.0f,  126198.0f,   85683.0f,   43620.0f,    28017.0f,   57057.0f,   87126.0f,  118230.0f,  118914.0f,   90771.0f,   61575.0f,   31320.0f,    15102.0f,   30729.0f,   46884.0f,   63570.0f,   63936.0f,   48765.0f,   33054.0f,   16800.0f,    17220.0f,   34863.0f,   52932.0f,   71430.0f,   72228.0f,   54831.0f,   36996.0f,   18720.0f,    36327.0f,   73527.0f,  111606.0f,  150570.0f,  152214.0f,  115521.0f,   77925.0f,   39420.0f,    57381.0f,  116112.0f,  176202.0f,  237660.0f,  240198.0f,  182250.0f,  122907.0f,   62160.0f,    80442.0f,  162738.0f,  246900.0f,  332940.0f,  336420.0f,  255198.0f,  172062.0f,   87000.0f,    84702.0f,  171318.0f,  259860.0f,  350340.0f,  353820.0f,  268338.0f,  180882.0f,   91440.0f,    66867.0f,  135210.0f,  205038.0f,  276360.0f,  279042.0f,  211572.0f,  142581.0f,   72060.0f,    46845.0f,   94701.0f,  143574.0f,  193470.0f,  195306.0f,  148047.0f,   99747.0f,   50400.0f,    24576.0f,   49671.0f,   75288.0f,  101430.0f,  102372.0f,   77583.0f,   52260.0f,   26400.0f,    22095.0f,   44688.0f,   67782.0f,   91380.0f,   92178.0f,   69906.0f,   47121.0f,   23820.0f,    46377.0f,   93777.0f,  142206.0f,  191670.0f,  193314.0f,  146571.0f,   98775.0f,   49920.0f,    72906.0f,  147387.0f,  223452.0f,  301110.0f,  303648.0f,  230175.0f,  155082.0f,   78360.0f,    101742.0f,  205638.0f,  311700.0f,  419940.0f,  423420.0f,  320898.0f,  216162.0f,  109200.0f,    106002.0f,  214218.0f,  324660.0f,  437340.0f,  440820.0f,  334038.0f,  224982.0f,  113640.0f,    83292.0f,  168285.0f,  254988.0f,  343410.0f,  346092.0f,  262197.0f,  176556.0f,   89160.0f,    58095.0f,  117351.0f,  177774.0f,  239370.0f,  241206.0f,  182697.0f,  122997.0f,   62100.0f,    30351.0f,   61296.0f,   92838.0f,  124980.0f,  125922.0f,   95358.0f,   64185.0f,   32400.0f,    26970.0f,   54513.0f,   82632.0f,  111330.0f,  112128.0f,   84981.0f,   57246.0f,   28920.0f,    56427.0f,  114027.0f,  172806.0f,  232770.0f,  234414.0f,  177621.0f,  119625.0f,   60420.0f,    88431.0f,  178662.0f,  270702.0f,  364560.0f,  367098.0f,  278100.0f,  187257.0f,   94560.0f,    123042.0f,  248538.0f,  376500.0f,  506940.0f,  510420.0f,  386598.0f,  260262.0f,  131400.0f,    127302.0f,  257118.0f,  389460.0f,  524340.0f,  527820.0f,  399738.0f,  269082.0f,  135840.0f,    99717.0f,  201360.0f,  304938.0f,  410460.0f,  413142.0f,  312822.0f,  210531.0f,  106260.0f,    69345.0f,  140001.0f,  211974.0f,  285270.0f,  287106.0f,  217347.0f,  146247.0f,   73800.0f,    36126.0f,   72921.0f,  110388.0f,  148530.0f,  149472.0f,  113133.0f,   76110.0f,   38400.0f});

    input.linspace(1);

    sd::ops::deconv2d op;
    auto results = op.evaluate({&input, &weights}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});

    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

}

TEST_F(ConvolutionTests1, deconv2d_test7) {

    NDArray exp('c', {3, 2, 4, 4}, {218., 227., 236., 245., 254., 263., 272., 281., 290., 299.,  308., 317., 326., 335., 344., 353., 270., 282., 294., 306.,  318., 330., 342., 354., 366., 378., 390., 402., 414., 426.,  438., 450., 650., 659., 668., 677., 686., 695., 704., 713.,  722., 731., 740., 749., 758., 767., 776., 785., 846., 858.,  870., 882., 894., 906., 918., 930., 942., 954., 966., 978.,  990., 1002., 1014., 1026., 1082., 1091., 1100., 1109., 1118., 1127.,  1136., 1145., 1154., 1163., 1172., 1181., 1190., 1199., 1208., 1217.,  1422., 1434., 1446., 1458., 1470., 1482., 1494., 1506., 1518., 1530.,  1542., 1554., 1566., 1578., 1590., 1602.});

    auto input = NDArrayFactory::create<double>('c', {3, 3, 4, 4});
    auto weights = NDArrayFactory::create<double>('c',{1, 1, 2, 3}, {1,3,5,2,4,6});
    auto bias = NDArrayFactory::create<double>('c', {2});

    input.linspace(1);
    bias.linspace(1);

    sd::ops::deconv2d op;

    auto result = op.evaluate({&input, &weights, &bias}, {1, 1, 1, 1, 0, 0, 1, 1, 1, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto output = result.at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, deconv2d_test8) {

    int bS=1, iH=7,iW=7,  iC=3,oC=2,  kH=1,kW=1,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=7,oW=7;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    NDArray input('c', {bS, iC, iH, iW}, {0.679350, 0.355087, 0.842789, 0.200313, 0.701499, 0.310693, 0.447940, 0.938010, 0.326674, 0.151873, 0.383318, 0.782123, 0.198807,
        0.798564, 0.163263, 0.146968, 0.260897, 0.135058, 0.756209, 0.275454, 0.369088, 0.092826, 0.836492, 0.268413, 0.095062, 0.312795, 0.135918, 0.517544, 0.328703,
        0.061736, 0.396431, 0.248016, 0.548959, 0.115046, 0.814362, 0.721564, 0.404494, 0.299089, 0.403884, 0.988311, 0.022296, 0.927782, 0.318416, 0.068546, 0.284533,
        0.232720, 0.352142, 0.058909, 0.711221, 0.674457, 0.196946, 0.699497, 0.074322, 0.420425, 0.584263, 0.149574, 0.446406, 0.723072, 0.064481, 0.483078, 0.875996,
        0.569819, 0.445863, 0.527755, 0.016646, 0.753678, 0.140636, 0.754129, 0.161932, 0.775037, 0.332645, 0.117394, 0.017711, 0.608476, 0.525152, 0.917194, 0.849891,
        0.589423, 0.852278, 0.390636, 0.889683, 0.669445, 0.698873, 0.961480, 0.157401, 0.157364, 0.493520, 0.569937, 0.126832, 0.115728, 0.786368, 0.737939, 0.490079,
        0.608414, 0.956500, 0.390098, 0.147305, 0.850645, 0.497650, 0.071866, 0.082150, 0.035314, 0.732041, 0.369934, 0.840666, 0.273894, 0.431796, 0.133231, 0.192975,
        0.246897, 0.386418, 0.511541, 0.199036, 0.141631, 0.697699, 0.253631, 0.782218, 0.930099, 0.335512, 0.558808, 0.664358, 0.018851, 0.637559, 0.290430, 0.434902,
        0.842513, 0.466098, 0.381395, 0.523185, 0.990183, 0.925768, 0.643459, 0.016828, 0.918756, 0.228979, 0.006314, 0.665975, 0.190361, 0.595521, 0.698881, 0.221469,
        0.912434, 0.870822, 0.727369, 0.523972, 0.662884, 0.218841});

    NDArray weights('c', {kH, kW, oC, iC}, {0.4195024073123932, 0.22738978266716003, 0.10093523561954498, 0.25008103251457214, 0.3183899223804474, 0.5976081490516663});
    NDArray bias('c', {1, oC}, {0.3596062958240509, 0.6866418123245239});

    NDArray exp('c', {bS, oC, oH, oW}, {0.848190, 0.560603, 0.880509, 0.464103, 0.823376, 0.660138, 0.666382, 0.882257, 0.704650, 0.451427, 0.649734, 0.911822, 0.611581,
        0.847623, 0.568191, 0.439341, 0.710854, 0.473843, 0.927273, 0.605861, 0.724540, 0.530591, 0.804268, 0.478136, 0.602198, 0.639553, 0.669082, 0.855013, 0.678572,
        0.617800, 0.667545, 0.765899, 0.835564, 0.631733, 0.921562, 0.790830, 0.588187, 0.597934, 0.725855, 0.822259, 0.455384, 0.998167, 0.683336, 0.591897, 0.705213,
        0.748148, 0.648922, 0.484723, 0.873482, 1.368675, 0.881096, 1.169214, 0.781504, 1.433406, 1.171439, 1.348675, 1.227033, 1.256600, 0.824772, 1.051633, 1.308692,
        1.148711, 1.334007, 1.014448, 0.813336, 1.408801, 0.916766, 1.583323, 1.362920, 1.226212, 1.149715, 1.330235, 0.770671, 1.285158, 1.105632, 1.272558, 1.590159,
        1.235054, 1.201363, 1.222816, 1.623673, 1.590317, 1.322463, 1.206481, 1.466262, 0.974741, 0.922343, 1.367100, 1.087943, 1.084952, 1.586691, 1.133576, 1.405098,
        1.471922, 1.484062, 1.212039, 1.144419, 1.266123});

    sd::ops::deconv2d op;
    auto results = op.evaluate({&input, &weights, &bias}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});

    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, deconv2d_test9) {

    int bS=2, oH=4,oW=4,  oC=5,iC=10,  kH=2,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       iH=3,iW=3;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW
    int wFormat     = 1;             // 0-[kH, kW, oC, iC], [iC, oC, kH, kW], [iC, kH, kW, oC]

    NDArray input('c', {bS, iH, iW, iC}, sd::DataType::FLOAT32);
    NDArray weights('c', {iC, oC, kH, kW}, {100.000000, 75.000000, 50.000000, 25.000000, 95.000000, 70.000000, 45.000000, 20.000000, 90.000000, 65.000000, 40.000000,
        15.000000, 85.000000, 60.000000, 35.000000, 10.000000, 80.000000, 55.000000, 30.000000, 5.000000, 99.500000, 74.500000, 49.500000, 24.500000, 94.500000, 69.500000,
        44.500000, 19.500000, 89.500000, 64.500000, 39.500000, 14.500000, 84.500000, 59.500000, 34.500000, 9.500000, 79.500000, 54.500000, 29.500000, 4.500000, 99.000000,
        74.000000, 49.000000, 24.000000, 94.000000, 69.000000, 44.000000, 19.000000, 89.000000, 64.000000, 39.000000, 14.000000, 84.000000, 59.000000, 34.000000, 9.000000,
        79.000000, 54.000000, 29.000000, 4.000000, 98.500000, 73.500000, 48.500000, 23.500000, 93.500000, 68.500000, 43.500000, 18.500000, 88.500000, 63.500000, 38.500000,
        13.500000, 83.500000, 58.500000, 33.500000, 8.500000, 78.500000, 53.500000, 28.500000, 3.500000, 98.000000, 73.000000, 48.000000, 23.000000, 93.000000, 68.000000,
        43.000000, 18.000000, 88.000000, 63.000000, 38.000000, 13.000000, 83.000000, 58.000000, 33.000000, 8.000000, 78.000000, 53.000000, 28.000000, 3.000000, 97.500000, 72.500000, 47.500000, 22.500000, 92.500000, 67.500000, 42.500000, 17.500000, 87.500000, 62.500000, 37.500000, 12.500000, 82.500000, 57.500000, 32.500000, 7.500000, 77.500000, 52.500000, 27.500000, 2.500000, 97.000000, 72.000000, 47.000000, 22.000000, 92.000000, 67.000000, 42.000000, 17.000000, 87.000000, 62.000000, 37.000000, 12.000000, 82.000000, 57.000000, 32.000000, 7.000000, 77.000000, 52.000000, 27.000000, 2.000000, 96.500000, 71.500000, 46.500000, 21.500000, 91.500000, 66.500000, 41.500000, 16.500000, 86.500000, 61.500000, 36.500000, 11.500000, 81.500000, 56.500000, 31.500000, 6.500000, 76.500000, 51.500000, 26.500000, 1.500000, 96.000000, 71.000000, 46.000000, 21.000000, 91.000000, 66.000000, 41.000000, 16.000000, 86.000000, 61.000000, 36.000000, 11.000000, 81.000000, 56.000000, 31.000000, 6.000000, 76.000000, 51.000000, 26.000000, 1.000000, 95.500000, 70.500000, 45.500000, 20.500000, 90.500000, 65.500000, 40.500000, 15.500000, 85.500000, 60.500000, 35.500000, 10.500000, 80.500000, 55.500000, 30.500000, 5.500000, 75.500000, 50.500000, 25.500000, 0.500000}, sd::DataType::FLOAT32);
    NDArray expOutput('c', {bS, oH, oW, oC}, {-30844.250000, -29266.750000, -27689.250000, -26111.750000, -24534.250000, -52823.500000, -49718.500000, -46613.500000, -43508.500000, -40403.500000, -51118.500000,
        -48113.500000, -45108.500000, -42103.500000, -39098.500000, -21501.750000, -20024.250000, -18546.750000, -17069.250000, -15591.750000, -42981.000000, -39976.000000, -36971.000000, -33966.000000, -30961.000000,
        -69482.000000, -63572.000000, -57662.000000, -51752.000000, -45842.000000, -67072.000000, -61362.000000, -55652.000000, -49942.000000, -44232.000000, -26046.000000, -23241.000000, -20436.000000, -17631.000000,
        -14826.000000, -38616.000000, -35911.000000, -33206.000000, -30501.000000, -27796.000000, -62252.000000, -56942.000000, -51632.000000, -46322.000000, -41012.000000, -59842.000000, -54732.000000, -49622.000000,
        -44512.000000, -39402.000000, -23181.000000, -20676.000000, -18171.000000, -15666.000000, -13161.000000, -12204.250000, -10926.750000, -9649.250000, -8371.750000, -7094.250000, -17543.500000, -15038.500000,
        -12533.500000, -10028.500000, -7523.500000, -16838.500000, -14433.499023, -12028.500000, -9623.500000, -7218.500000, -5361.750000, -4184.250000, -3006.750000, -1829.250000, -651.750000, -22046.750000, -20919.250000,
        -19791.750000, -18664.250000, -17536.750000, -37478.500000, -35273.500000, -33068.500000, -30863.500000, -28658.500000, -35773.500000, -33668.500000, -31563.500000, -29458.500000, -27353.500000, -14954.250000,
        -13926.750000, -12899.250000, -11871.750000, -10844.250000, -29886.000000, -27781.000000, -25676.000000, -23571.000000, -21466.000000, -47792.000000, -43682.000000, -39572.000000, -35462.000000, -31352.000000,
        -45382.000000, -41472.000000, -37562.000000, -33652.000000, -29742.000000, -17451.000000, -15546.000000, -13641.000000, -11736.000000, -9831.000000, -25521.000000, -23716.000000, -21911.000000, -20106.000000, -18301.000000, -40562.000000, -37052.000000, -33542.000000, -30032.000000, -26522.000000, -38152.000000, -34842.000000, -31532.000000, -28222.000000, -24912.000000, -14586.000000, -12981.000000, -11376.000000, -9771.000000, -8166.000000, -7906.750000, -7079.250000, -6251.750000, -5424.250000, -4596.750000, -11198.500000, -9593.500000, -7988.500000, -6383.500000, -4778.500000, -10493.500000, -8988.500000, -7483.500000, -5978.500000, -4473.500000, -3314.250000, -2586.750000, -1859.250000, -1131.750000, -404.250000}, sd::DataType::FLOAT32);

    input.linspace(-32, 0.1);

    sd::ops::deconv2d op;
    auto results = op.evaluate({&input, &weights}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat, wFormat});
    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, deconv2d_test10) {

    int bS=2, oH=4,oW=4,  iC=5,oC=10,  kH=2,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       iH=4,iW=4;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW
    int wFormat     = 2;             // 0-[kH, kW, oC, iC], [iC, oC, kH, kW], [iC, kH, kW, oC]

    NDArray input('c', {bS, iC, iH, iW}, sd::DataType::FLOAT32);
    NDArray weights('c', {iC, kH, kW, oC}, {100., 95., 90., 85., 80., 75., 70., 65., 60., 55., 50., 45., 40., 35., 30., 25., 20., 15., 10., 5., 0., -5., -10., -15.,
        -20., -25., -30., -35., -40., -45., -50., -55., -60., -65., -70., -75., -80., -85., -90., -95., 99., 94., 89., 84., 79., 74., 69., 64., 59., 54., 49., 44.,
        39., 34., 29., 24., 19., 14., 9., 4., -1., -6., -11., -16., -21., -26., -31., -36., -41., -46., -51., -56., -61., -66., -71., -76., -81., -86., -91., -96.,
        98., 93., 88., 83., 78., 73., 68., 63., 58., 53., 48., 43., 38., 33., 28., 23., 18., 13., 8., 3., -2., -7., -12., -17., -22., -27., -32., -37., -42., -47.,
        -52., -57., -62., -67., -72., -77., -82., -87., -92., -97., 97., 92., 87., 82., 77., 72., 67., 62., 57., 52., 47., 42., 37., 32., 27., 22., 17., 12., 7., 2.,
        -3., -8., -13., -18., -23., -28., -33., -38., -43., -48., -53., -58., -63., -68., -73., -78., -83., -88., -93., -98., 96., 91., 86., 81., 76., 71., 66., 61.,
        56., 51., 46., 41., 36., 31., 26., 21., 16., 11., 6., 1., -4., -9., -14., -19., -24., -29., -34., -39., -44., -49., -54., -59., -64., -69., -74., -79., -84., -89., -94., -99.}, sd::DataType::FLOAT32);
    NDArray expOutput('c', {bS, oC, oH, oW}, {-14128., -21007., -20934., -20861., -13660., -12972., -12926.000977, -12880., -13468., -12788., -12742., -12696.000977,
        -13276., -12604., -12558., -12512., -13408., -19569.5, -19501.5, -19433.5, -12230., -10117., -10081.000977, -10045., -12058., -9973., -9937., -9901.000977,
        -11886., -9829., -9793., -9757., -12688., -18132., -18069., -18006., -10800., -7262., -7236., -7210., -10648., -7157.999512, -7132., -7106., -10496., -7054.,
        -7027.999512, -7002., -11968., -16694.5, -16636.5, -16578.5, -9370., -4406.999023, -4391., -4375., -9238., -4343., -4326.999023, -4311., -9106., -4279., -4263.,
        -4246.999023, -11247.999023, -15257., -15204., -15151., -7940., -1551.999023, -1546., -1540., -7828., -1528.000977, -1521.999023, -1516., -7716., -1504.,
        -1498.000977, -1491.999023, -10527.999023, -13819.5, -13771.5, -13723.5, -6510., 1303.000977, 1299., 1295., -6418., 1286.999023, 1283.000977, 1279., -6326.,
        1271., 1266.999023, 1263.000977, -9807.999023, -12382., -12339., -12296., -5080., 4158.000977, 4144., 4130., -5008., 4101.999023, 4088., 4074., -4936., 4046., 4031.999023, 4018., -9088., -10944.5, -10906.5, -10868.5, -3650., 7013., 6989., 6965., -3598., 6917., 6893., 6869., -3546., 6821., 6797., 6773., -8368., -9507., -9474., -9441., -2220., 9868., 9834., 9800., -2187.999512, 9732., 9698., 9664., -2156., 9596., 9562., 9528., -7648., -8069.5, -8041.5, -8013.499512, -790.000488, 12723., 12679., 12635., -777.999512, 12547., 12503., 12459., -766., 12371., 12327., 12283., -10208., -15167., -15094., -15021., -9820., -9292., -9246., -9200., -9628., -9108., -9062., -9016., -9436., -8924., -8878., -8832., -9687.999023, -14129.5, -14061.5, -13993.5, -8790., -7236.999023, -7201., -7164.999512, -8618., -7093., -7057., -7021., -8446., -6949., -6913., -6877., -9168., -13092., -13029., -12966., -7760., -5182., -5156., -5129.999512, -7608., -5078., -5052., -5026., -7456., -4974., -4948., -4922., -8648., -12054.5, -11996.5, -11938.5, -6730., -3127., -3111., -3095., -6598., -3063., -3047., -3031., -6465.999512, -2999., -2983.000488, -2967., -8128., -11017., -10964., -10911., -5700.000488, -1072., -1066., -1060., -5587.999512, -1048.000488, -1042., -1036., -5476., -1023.999512, -1018.000488, -1012., -7608., -9979.5, -9931.5, -9883.5, -4670.000488, 983., 979., 975., -4577.999512, 966.999512, 963., 959., -4486., 951.000488, 946.999512, 943., -7088., -8942., -8899., -8856., -3640.000488, 3038., 3024., 3010., -3567.999512, 2981.999512, 2968., 2954., -3496., 2926.000488, 2911.999512, 2898., -6568., -7904.5, -7866.5, -7828.499512, -2610.000488, 5093., 5069., 5045., -2557.999512, 4996.999512, 4973., 4949., -2506., 4901.000488, 4877., 4853., -6048., -6867., -6834., -6800.999512, -1580., 7148., 7114., 7080., -1547.999512, 7012., 6978., 6944., -1516., 6876.000488, 6842., 6808., -5528., -5829.5, -5801.5, -5773.499512, -550., 9203., 9159., 9115., -537.999512, 9027., 8983., 8939., -526., 8851., 8807., 8763.}, sd::DataType::FLOAT32);

    input.linspace(-32, 0.1);

    sd::ops::deconv2d op;
    auto results = op.evaluate({&input, &weights}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat, wFormat});
    ASSERT_EQ(Status::OK(), results.status());

    auto output = results.at(0);

    ASSERT_TRUE(expOutput.isSameShape(output));
    ASSERT_TRUE(expOutput.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedConvolutionTests1, deconv2d_tf_test1) {

    int bS=2, iH=4,iW=4,  iC=5,oC=10,  kH=2,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=3,oW=3;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS, oH, oW, oC});
    auto weights  = NDArrayFactory::create<TypeParam>('c', {kH, kW, iC, oC});
    auto outShape = NDArrayFactory::create<TypeParam>('c', {4}, {static_cast<TypeParam>(bS), static_cast<TypeParam>(iH), static_cast<TypeParam>(iW), static_cast<TypeParam>(iC)});
    auto exp = NDArrayFactory::create<TypeParam>('c', {bS, iH, iW, iC}, {  2.75f,    7.75f,   12.75f,   17.75f,   22.75f,  30.5f,   40.5f,   50.5f,   60.5f,   70.5f,  30.5f,   40.5f,   50.5f,   60.5f,   70.5f,  27.75f,   32.75f,   37.75f,   42.75f,   47.75f,
                                                  55.5f,   65.5f,   75.5f,   85.5f,   95.5f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f, 105.5f,  115.5f,  125.5f,  135.5f,  145.5f,
                                                  55.5f,   65.5f,   75.5f,   85.5f,   95.5f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f, 105.5f,  115.5f,  125.5f,  135.5f,  145.5f,
                                                  52.75f,   57.75f,   62.75f,   67.75f,   72.75f, 130.5f,  140.5f,  150.5f,  160.5f,  170.5f, 130.5f,  140.5f,  150.5f,  160.5f,  170.5f,  77.75f,   82.75f,   87.75f,   92.75f,   97.75f,
                                                   2.75f,    7.75f,   12.75f,   17.75f,   22.75f,  30.5f,   40.5f,   50.5f,   60.5f,   70.5f,  30.5f,   40.5f,   50.5f,   60.5f,   70.5f,  27.75f,   32.75f,   37.75f,   42.75f,   47.75f,
                                                  55.5f,   65.5f,   75.5f,   85.5f,   95.5f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f, 105.5f,  115.5f,  125.5f,  135.5f,  145.5f,
                                                  55.5f,   65.5f,   75.5f,   85.5f,   95.5f, 161.f,  181.f,  201.f,  221.f,  241.f, 161.f,  181.f,  201.f,  221.f,  241.f, 105.5f,  115.5f,  125.5f,  135.5f,  145.5f,
                                                  52.75f,   57.75f,   62.75f,   67.75f,   72.75f, 130.5f,  140.5f,  150.5f,  160.5f,  170.5f, 130.5f,  140.5f,  150.5f,  160.5f,  170.5f,  77.75f,   82.75f,   87.75f,   92.75f,   97.75f});
    input = 0.5;
    weights.linspace(0.1, 0.1);

    sd::ops::deconv2d_tf op;
    auto results = op.evaluate({&outShape, &weights, &input}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto output = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv2d_bp_7) {
  int bS=2, iH=12,iW=12,  iC=3,oC=3,  kH=3,kW=3,  sH=2,sW=2,  pH=0,pW=0,  dH=1,dW=1;
  int       oH=6,oW=6;
  int paddingMode = 1;             // 1-SAME, 0-VALID;
  int dataFormat  = 0;             // 1-NHWC, 0-NCHW

  NDArray input('c', {bS, iC, iH, iW}, sd::DataType::FLOAT32);
  NDArray weights('c', {kH, kW, iC, oC}, sd::DataType::FLOAT32);
  NDArray bias('c', {oC}, {1,2,3}, sd::DataType::FLOAT32);
  NDArray gradO('c', {bS, oC, oH, oW}, sd::DataType::FLOAT32);
  NDArray gradI('c', {bS, iC, iH, iW}, sd::DataType::FLOAT32);
  NDArray gradW('c', {kH, kW, iC, oC}, sd::DataType::FLOAT32);
  NDArray gradB('c', {oC}, sd::DataType::FLOAT32);
  input = 2.;
  weights.linspace(0.1, 0.1);
  gradO.linspace(0.01, 0.01);

  sd::ops::conv2d_bp op;
  auto status = op.execute({&input, &weights, &bias, &gradO}, {&gradI, &gradW, &gradB}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat}, {});
  ASSERT_EQ(Status::OK(), status);
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv2d_ff_119_1) {
  auto i = NDArrayFactory::create<float>('c', {2, 3, 13, 13});
  auto w = NDArrayFactory::create<float>('c', {3, 3, 3, 3});
  auto b = NDArrayFactory::create<float>('c', {3});
  auto o = NDArrayFactory::create<float>('c', {2, 3, 6, 6});

  sd::ops::conv2d op_ff;
  auto status = op_ff.execute({&i, &w, &b}, {&o}, {3,3,  2,2,  0,0,  1,1,  0,  0,  1});
  ASSERT_EQ(Status::OK(), status);

  auto gi = i.ulike();
  auto gw = w.ulike();

  sd::ops::conv2d_bp op_bp;
  status = op_bp.execute({&i, &w, &b, &o}, {&gi, &gw}, {3,3,  2,2,  0,0,  1,1,  0,  0,  1});
  ASSERT_EQ(Status::OK(), status);
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConvolutionTests1, conv2d_ff_119_2) {
  auto i = NDArrayFactory::create<float>('c', {2, 3, 17, 17});
  auto w = NDArrayFactory::create<float>('c', {3, 3, 3, 3});
  auto b = NDArrayFactory::create<float>('c', {3});
  auto o = NDArrayFactory::create<float>('c', {2, 3, 8, 8});

  sd::ops::conv2d op_ff;
  auto status = op_ff.execute({&i, &w, &b}, {&o}, {3,3,  2,2,  0,0,  1,1,  0,  0,  1});
  ASSERT_EQ(Status::OK(), status);

  auto gi = i.ulike();
  auto gw = w.ulike();

  sd::ops::conv2d_bp op_bp;
  status = op_bp.execute({&i, &w, &b, &o}, {&gi, &gw}, {3,3,  2,2,  0,0,  1,1,  0,  0,  1});
  ASSERT_EQ(Status::OK(), status);
}

#endif //LIBND4J_CONVOLUTIONTESTS1_H

