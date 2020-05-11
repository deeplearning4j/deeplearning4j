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

#include "testlayers.h"
#include <graph/Context.h>
#include <iomanip>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <array/NDArray.h>
#include <array/NDArrayList.h>
#include <legacy/NativeOps.h>
#include <ops/gemm.h>
#include <helpers/PointersManager.h>

using namespace sd;
using namespace sd::graph;

class DeclarableOpsTests1 : public testing::Test {
public:

    const int bS = 2;       // batch size
    const int iD = 1;       // input depth (number of picture channels, for example rgb=3)
    const int iH = 28;      // picture height in pixels
    const int iW = 28;      // picture width in pixels
    const int oD = 3;       // output depth (= N for dense layer)
    const int kH = 5;       // kernel height in pixels
    const int kW = 5;       // kernel width in pixels
    const int sH = 1;       // stride step in horizontal direction
    const int sW = 1;       // stride step in vertical direction
    const int pH = 0;       // padding height
    const int pW = 0;       // padding width
    const int dH = 2;       // dilation height
    const int dW = 2;       // dilation width
    const int oH = (iH - kH - (kH - 1) * (dH - 1) + 2 * pH) / sH + 1;     // output height
    const int oW = (iW - kW - (kW - 1) * (dW - 1) + 2 * pW) / sW + 1;     // output width

    DeclarableOpsTests1() {
        sd::memory::MemoryTracker::getInstance()->reset();
    }

    ~DeclarableOpsTests1() {
        sd::memory::MemoryTracker::getInstance()->summarize();
    }
};

template <typename T>
class TypedDeclarableOpsTests1 : public testing::Test {
public:

    const int bS = 2;       // batch size
    const int iD = 1;       // input depth (number of picture channels, for example rgb=3)
    const int iH = 28;      // picture height in pixels
    const int iW = 28;      // picture width in pixels
    const int oD = 3;       // output depth (= N for dense layer)
    const int kH = 5;       // kernel height in pixels
    const int kW = 5;       // kernel width in pixels
    const int sH = 1;       // stride step in horizontal direction
    const int sW = 1;       // stride step in vertical direction
    const int pH = 0;       // padding height
    const int pW = 0;       // padding width
    const int dH = 2;       // dilation height
    const int dW = 2;       // dilation width
    const int oH = (iH - kH - (kH - 1) * (dH - 1) + 2 * pH) / sH + 1;     // output height
    const int oW = (iW - kW - (kW - 1) * (dW - 1) + 2 * pW) / sW + 1;     // output width

    TypedDeclarableOpsTests1() {
        printf("\n");
    }
};

typedef ::testing::Types<double, float> TestingTypes;
TYPED_TEST_CASE(TypedDeclarableOpsTests1, TestingTypes);

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, BasicInitialization1) {
    auto concat = new sd::ops::concat();
    std::string expName("concat");
    ASSERT_EQ(expName, *(concat->getOpName()));

    auto x0 = NDArrayFactory::create_<float>('c', { 1, 5 });
    auto x1 = NDArrayFactory::create_<float>('c', { 1, 5 });
    auto x2 = NDArrayFactory::create_<float>('c', { 1, 5 });
    auto x3 = NDArrayFactory::create_<float>('c', { 1, 5 });
    auto x4 = NDArrayFactory::create_<float>('c', { 1, 5 });

    x0->assign(1.0f);
    x1->assign(1.0f);
    x2->assign(1.0f);
    x3->assign(1.0f);
    x4->assign(1.0f);

    auto variableSpace = new VariableSpace();

    variableSpace->putVariable(-1, x0);
    variableSpace->putVariable(-2, x1);
    variableSpace->putVariable(-3, x2);
    variableSpace->putVariable(-4, x3);
    variableSpace->putVariable(-5, x4);

    auto nodeVar = new Variable();

    variableSpace->putVariable(1, nodeVar);

    Context block(1, variableSpace);
    block.getIArguments()->push_back(1);
    block.fillInputs({ -1, -2, -3, -4, -5 });

    ASSERT_FALSE(nodeVar->hasNDArray());

    Nd4jStatus result = concat->execute(&block);

    ASSERT_TRUE(nodeVar->hasNDArray());

    ASSERT_EQ(25, nodeVar->getNDArray()->lengthOf());

    ASSERT_NEAR(25.0, nodeVar->getNDArray()->reduceNumber(reduce::Sum).e<double>(0), 1e-5);

    ASSERT_EQ(ND4J_STATUS_OK, result);


    delete variableSpace;
    delete concat;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, BasicInitialization2) {
    auto op = sd::ops::OpRegistrator::getInstance()->getOperation("concat");

    ASSERT_TRUE(op != nullptr);
    std::string expName("concat");
    ASSERT_EQ(expName, *(op->getOpName()));

    ASSERT_EQ(-1, op->getOpDescriptor()->getNumberOfInputs());
    ASSERT_EQ(1, op->getOpDescriptor()->getNumberOfOutputs());
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ApplyGradientDescent_1) {
    auto x = NDArrayFactory::create<double>('c', { 3,4 }, { 1,2,3,4,5,6,7,8,9,10,11,12 });
    auto y = NDArrayFactory::create<double>('c', { 3,4 }, { 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2 });
    auto exp = NDArrayFactory::create<double>('c', { 3,4 });
    exp.linspace(0.9, 0.9);
    sd::ops::apply_sgd op;
    auto result = op.evaluate({ &x, &y }, { 1. }, {});
    ASSERT_EQ(result.status(), ND4J_STATUS_OK);
    auto z = result.at(0);

    ASSERT_TRUE(z->equalsTo(exp));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, AssignBroadcastTest_1) {
    auto x = NDArrayFactory::create<double>('c', { 3,4 }, { 1,2,3,4,5,6,7,8,9,10,11,12 });
    auto y = NDArrayFactory::create<double>('c', { 1,4 }, { 0.1,0.2,0.3,0.4 });
    auto exp = NDArrayFactory::create<double>('c', { 3,4 }, { 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4 });
    sd::ops::assign op;
    auto result = op.evaluate({ &x, &y });
    ASSERT_EQ(result.status(), ND4J_STATUS_OK);
    auto z = result.at(0);

    ASSERT_TRUE(z->equalsTo(exp));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, AssignBroadcastTest_2) {
    auto x = NDArrayFactory::create<double>('c', { 3,4 }, { 1,2,3,4,5,6,7,8,9,10,11,12 });
    auto y = NDArrayFactory::create<double>('c', { 1,4 }, { 0.1,0.2,0.3,0.4 });
    auto eps = NDArrayFactory::create<double>('c', { 3,4 }, { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 });
    auto exp1 = NDArrayFactory::create<double>('c', { 3,4 }); // zero
    auto exp2 = NDArrayFactory::create<double>('c', { 1,4 }, { 3, 6, 9, 12 });
    sd::ops::assign_bp op;
    auto result = op.evaluate({ &x, &y, &eps });
    ASSERT_EQ(result.status(), ND4J_STATUS_OK);
    auto z1 = result.at(0);
    auto z2 = result.at(1);

    ASSERT_TRUE(z1->equalsTo(exp1));
    ASSERT_TRUE(z2->equalsTo(exp2));

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, AXpY_Test_1) {
    auto x = NDArrayFactory::create<double>('c', { 3,4 }, { 1,2,3,4,5,6,7,8,9,10,11,12 });
    auto y = NDArrayFactory::create<double>('c', { 3,4 }, { 1,2,3,4,5,6,7,8,9,10,11,12 });
    auto exp = NDArrayFactory::create<double>('c', { 3,4 });
    exp.linspace(3, 3);
    sd::ops::axpy op;
    auto result = op.evaluate({ &x, &y }, { 2. });
    ASSERT_EQ(result.status(), ND4J_STATUS_OK);
    auto z = result.at(0);

    ASSERT_TRUE(z->equalsTo(exp));

}

TEST_F(DeclarableOpsTests1, BasicInitialization3) {
    auto op1 = sd::ops::OpRegistrator::getInstance()->getOperation("concat");
    std::string expName("concat");
    auto hash = sd::ops::HashHelper::getInstance()->getLongHash(expName);

    auto op2 = sd::ops::OpRegistrator::getInstance()->getOperation(hash);

    ASSERT_TRUE(op1 == op2);
}


TEST_F(DeclarableOpsTests1, SynonymInitialization2) {
    auto op = sd::ops::OpRegistrator::getInstance()->getOperation("Mul");
    auto op2 = sd::ops::OpRegistrator::getInstance()->getOperation("multiply");

    ASSERT_TRUE(op != nullptr);
    std::string expName("multiply");
    ASSERT_EQ(expName, *(op->getOpName()));
    ASSERT_TRUE(op == op2);
}


TEST_F(DeclarableOpsTests1, TestTensorMmul1) {

    NDArray x('c', { 2, 3, 4 }, sd::DataType::FLOAT32);
    NDArray y('c', { 2, 3, 4 }, sd::DataType::FLOAT32);

    x.linspace(1);
    y.linspace(1);

    NDArray exp('c', { 2, 2 }, { 650.0, 1586.0, 1586.0, 4250.0 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 2,1,2,2,1,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* out = results.at(0);

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));


}

TEST_F(DeclarableOpsTests1, TestTensorDot2) {

    NDArray x('f', { 2, 3, 4 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24. }, sd::DataType::FLOAT32);
    NDArray y('f', { 2, 3, 4 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24. }, sd::DataType::FLOAT32);

    NDArray exp('c', { 2, 2 }, { 2300.0, 2444.0, 2444.0, 2600.0 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 2,1,2,2,1,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* out = results.at(0);

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));


}

TEST_F(DeclarableOpsTests1, TestTensorDot3) {

    NDArray x('c', { 2, 3, 4 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24. }, sd::DataType::FLOAT32);
    NDArray y('f', { 2, 3, 4 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24. }, sd::DataType::FLOAT32);

    NDArray exp('f', { 2, 2 }, { 1090.0, 2818.0, 1168.0, 3040.0 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 2,1,2,2,1,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* out = results.at(0);

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));


}

TEST_F(DeclarableOpsTests1, TestTensorDot4) {

    NDArray x('f', { 2, 3, 4 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24. }, sd::DataType::FLOAT32);
    NDArray y('c', { 2, 3, 4 }, { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24. }, sd::DataType::FLOAT32);

    NDArray exp('f', { 2, 2 }, { 1090.0, 1168.0, 2818.0, 3040.0 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 2,1,2,2,1,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* out = results.at(0);

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));


}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot5) {

    auto x = NDArrayFactory::create<double>('c', { 2,3,4 }, { 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15 });
    auto y = NDArrayFactory::create<double>('c', { 2,4,3 }, { 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16 });
    auto expected = NDArrayFactory::create<double>('c', { 2,4,2,4 }, { 44,110,160, 66,132, 38, 88,154, 68,170,224,102,204, 82,136,238, 92,230,288,138,276,126,184,322, 116,290,352,174,348,170,232,406, 76,190,160,114,228,182,152,266, 100,250,224,150,300,226,200,350, 124,310,288,186,372,270,248,434, 148,370,352,222,444,314,296,518 });

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 1,1,1,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));



}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot6) {

    auto x = NDArrayFactory::create<double>('c', { 2,3,4 }, { 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15 });
    auto y = NDArrayFactory::create<double>('f', { 2,4,3 }, { 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16 });
    auto expected = NDArrayFactory::create<double>('c', { 2,4,2,4 }, { 22, 66,110,154, 44, 88,132,176, 34,102,170,238, 68,136,204,272, 46,138,230,322, 92,184,276,368, 58,174,290,406,116,232,348,464, 38,114,190,266, 76,152,228,304, 50,150,250,350,100,200,300,400, 62,186,310,434,124,248,372,496, 74,222,370,518,148,296,444,592 });

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 1,1,1,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));



}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot7) {

    auto x = NDArrayFactory::create<double>('f', { 2,3,4 }, { 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15 });
    auto y = NDArrayFactory::create<double>('c', { 2,4,3 }, { 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16 });
    auto expected = NDArrayFactory::create<double>('c', { 2,4,2,4 }, { 76,166,112,106,196, 62,136,226, 60,174,208, 98,212,230,136,250, 76,214,336,122,260,174,168,306, 124,286,240,178,340,150,232,394, 100,226,176,142,268,106,184,310, 84,234,272,134,284,274,184,334, 100,274,400,158,332,218,216,390, 148,346,304,214,412,194,280,478 });

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 1,1,1,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));



}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot8) {

    auto x = NDArrayFactory::create<double>('f', { 2,3,4 }, { 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15 });
    auto y = NDArrayFactory::create<double>('f', { 2,4,3 }, { 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16 });
    auto expected = NDArrayFactory::create<double>('c', { 2,4,2,4 }, { 30, 90,150,210, 60,120,180,240, 38,114,190,266, 76,152,228,304, 46,138,230,322, 92,184,276,368, 54,162,270,378,108,216,324,432, 42,126,210,294, 84,168,252,336, 50,150,250,350,100,200,300,400, 58,174,290,406,116,232,348,464, 66,198,330,462,132,264,396,528 });

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 1,1,1,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));



}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot9) {

    // NDArray z('f',{2,2,3}, sd::DataType::DOUBLE);
    // z.linspace(1);
    // z.printShapeInfo();
    // z.printIndexedBuffer();
    // z.reshapei('c', {4,3});
    // z.printShapeInfo();
    // z.printIndexedBuffer();

    auto x = NDArrayFactory::create<double>('f', { 2,3,4 }, { 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15 });
    auto y = NDArrayFactory::create<double>('f', { 2,4,3 }, { 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16 });
    auto expected = NDArrayFactory::create<double>('c', { 3,4,4,3 }, { 14, 14, 14, 30, 30, 30, 46, 46, 46, 62, 62, 62, 86, 86, 86,198,198,198,310,310,310,422,422,422, 62, 62, 62,142,142,142,222,222,222,302,302,302, 38, 38, 38, 86, 86, 86,134,134,134,182,182,182, 38, 38, 38, 86, 86, 86,134,134,134,182,182,182, 14, 14, 14, 30, 30, 30, 46, 46, 46, 62, 62, 62, 86, 86, 86,198,198,198,310,310,310,422,422,422, 62, 62, 62,142,142,142,222,222,222,302,302,302, 62, 62, 62,142,142,142,222,222,222,302,302,302, 38, 38, 38, 86, 86, 86,134,134,134,182,182,182, 14, 14, 14, 30, 30, 30, 46, 46, 46, 62, 62, 62, 86, 86, 86,198,198,198,310,310,310,422,422,422 });

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 1,0,1,0 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));


}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot10) {

    auto x = NDArrayFactory::create<double>('f', { 2,3,4 }, { 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15 });
    auto y = NDArrayFactory::create<double>('f', { 2,4,3 }, { 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16 });
    auto expected = NDArrayFactory::create<double>('c', { 4,4 }, { 114,258,402,546, 138,314,490,666, 162,370,578,786, 186,426,666,906 });

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 2,0,1, 2,0,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));



}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot11) {

    auto x = NDArrayFactory::create<double>('c', { 2,3,4 }, { 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15 });
    auto y = NDArrayFactory::create<double>('f', { 2,4,3 }, { 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16 });
    auto expected = NDArrayFactory::create<double>('c', { 4,4 }, { 98,218,338,458, 134,302,470,638, 170,386,602,818, 206,470,734,998 });

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 2,0,1, 2,0,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));



}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot12) {

    auto x = NDArrayFactory::create<double>('c', { 2,3,4 }, { 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15 });
    auto y = NDArrayFactory::create<double>('c', { 2,4,3 }, { 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16 });
    auto expected = NDArrayFactory::create<double>('c', { 4,4 }, { 272,292,312,332, 368,396,424,452, 464,500,536,572, 560,604,648,692 });

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 2,0,1, 2,0,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));



}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot13) {

    auto x = NDArrayFactory::create<double>('c', { 2,3,4 }, { 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15 });
    auto y = NDArrayFactory::create<double>('c', { 4,2,3 }, { 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16 });
    auto expected = NDArrayFactory::create<double>('c', { 3,3 }, { 640,560,640, 576,624,576, 640,560,640 });

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 2,0,2, 2,1,0 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));



}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot14) {

    auto x = NDArrayFactory::create<double>('f', { 2,3,4 }, { 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15 });
    auto y = NDArrayFactory::create<double>('c', { 4,2,3 }, { 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16 });
    auto expected = NDArrayFactory::create<double>('c', { 3,3 }, { 648,600,520, 648,536,648, 520,600,648 });

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 2,0,2, 2,1,0 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));



}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot15) {

    auto x = NDArrayFactory::create<double>('f', { 2,3,4 }, { 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15 });
    auto y = NDArrayFactory::create<double>('f', { 4,2,3 }, { 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16 });
    auto expected = NDArrayFactory::create<double>('c', { 3,3 }, { 624,624,624, 656,656,656, 624,624,624 });

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 2,0,2, 2,1,0 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));



}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot16) {

    NDArray x('c', { 1 }, std::vector<double>{2}, sd::DataType::FLOAT32);
    NDArray y('c', { 2,1,2 }, { 1,2,3,4 }, sd::DataType::FLOAT32);
    NDArray exp('c', { 2,2 }, { 2,4,6,8 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul op;
    auto results = op.evaluate({ &x, &y }, {}, { 1,0, 1,1 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto* result = results.at(0);

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));


}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestTensorDot17) {

    NDArray x('f', { 16,16 }, sd::DataType::FLOAT32);
    NDArray y('f', { 1000,16 }, sd::DataType::FLOAT32);
    NDArray z('c', { 16,1000 }, sd::DataType::FLOAT32);

    sd::ops::tensormmul op;
    auto status = op.execute({ &x, &y }, { &z }, {}, { 1,1, 1,1 }, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, DivergentCheck1) {
    auto op = sd::ops::OpRegistrator::getInstance()->getOperation("switch");

    ASSERT_TRUE(op != nullptr);
    std::string expName("Switch");
    ASSERT_EQ(expName, *(op->getOpName()));
    ASSERT_TRUE(op->getOpDescriptor()->isDivergent());
    ASSERT_EQ(2, op->getOpDescriptor()->getNumberOfOutputs());
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, AddMatrices1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto exp = NDArrayFactory::create_<float>('c', { 5, 3 });
    x->assign(2);
    y->assign(1);
    exp->assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::add addOp;

    addOp.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete exp;
    delete block;
    delete variableSpace;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, AddVectorVector1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto exp = NDArrayFactory::create_<float>('c', { 1, 15 });
    x->assign(2);
    y->assign(1);
    exp->assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::add addOp;

    addOp.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete exp;
    delete block;
    delete variableSpace;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, AddMatrixScalar1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto exp = NDArrayFactory::create<float>('c', { 5, 3 });
    x->assign(2);
    y->assign(1);
    exp.assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::add addOp;

    addOp.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, AddScalarScalar1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto exp = NDArrayFactory::create<float>('c', { 1, 1 });
    x->assign(2);
    y->assign(1);
    exp.assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::add addOp;

    addOp.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, SubtractMatrices1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto exp = NDArrayFactory::create<float>('c', { 5, 3 });
    x->assign(3);
    y->assign(1);
    exp.assign(2);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::subtract subOp;

    subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));


    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, SubtractTest_1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 6 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 6 });
    auto exp = NDArrayFactory::create<float>('c', { 1, 6 });
    x->assign(3);
    y->assign(1);
    exp.assign(2);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::subtract subOp;

    subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));


    delete variableSpace;
    delete block;
}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, SubtractTest_2) {

    auto x = NDArrayFactory::create<float>('c', { 3, 4, 5, 1 });
    auto y = NDArrayFactory::create<float>('c', { 1, 6 });
    //    auto y({6}, {1,1,1,1,1,1});
    auto exp = NDArrayFactory::create<float>('c', { 3, 4, 5, 6 });
    x.assign(3);
    y.assign(1);
    exp.assign(2);


    sd::ops::subtract subOp;

    auto res = subOp.evaluate({ &x, &y });

    ASSERT_TRUE(res.status() == ND4J_STATUS_OK);

    ASSERT_TRUE(res.at(0)->equalsTo(&exp));


}

TEST_F(DeclarableOpsTests1, TestRng1) {
    /*
        Nd4jLong *buffer = new Nd4jLong[100000];

        sd::random::RandomBuffer *rng = (sd::random::RandomBuffer *) initRandom(nullptr, 123, 100000, (Nd4jPointer) buffer);

        if (rng == nullptr)
            throw std::runtime_error("RNG initialization failed");

        auto x = NDArrayFactory::create_<float>('c', {5, 3});
        auto variableSpace = new VariableSpace();
        variableSpace->putVariable(-1, x);
        auto block = new Context(1, variableSpace, true);
        block->fillInputs({-1});
        block->setRNG(rng);
        block->getTArguments()->push_back(0.0f);
        block->getTArguments()->push_back(1.0f);

        sd::ops::randomuniform uniform;

        Nd4jStatus status = uniform.execute(block);

        ASSERT_EQ(ND4J_STATUS_OK, status);

        ASSERT_TRUE(x->sumNumber() > 0.0);

        destroyRandom((Nd4jPointer) rng);
        delete[] buffer;

        delete variableSpace;
        delete block;
        */
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, MergeSumTest1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 5 });
    auto y = NDArrayFactory::create_<float>('c', { 5, 5 });
    auto z = NDArrayFactory::create_<float>('c', { 5, 5 });
    auto exp = NDArrayFactory::create<float>('c', { 5, 5 });
    x->assign(3);
    y->assign(1);
    z->assign(2);
    exp.assign(6);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    variableSpace->putVariable(-3, z);
    variableSpace->putVariable(1, new Variable(NDArrayFactory::create_<float>('c', { 5, 5 })));
    auto block = new Context(1, variableSpace, false);
    block->fillInputs({ -1, -2, -3 });

    sd::ops::mergeadd merge;

    merge.execute(block);

    auto res = variableSpace->getVariable(1)->getNDArray();

    ASSERT_TRUE(res->equalsTo(&exp));

    delete variableSpace;
    delete block;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ClipByValue1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 5 });
    auto exp = NDArrayFactory::create<float>('c', { 5, 5 });
    x->assign(4);
    x->p(0, -1);
    x->p(1, 2);
    exp.assign(3);
    exp.p(0, 0);
    exp.p(1, 2);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(1, new Variable());
    auto block = new Context(1, variableSpace, true);
    block->getTArguments()->push_back(0.0f);
    block->getTArguments()->push_back(3.0f);
    block->fillInputs({ -1 });

    sd::ops::clipbyvalue clip;

    clip.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));


    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, MergeAvgTest1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 5 });
    auto y = NDArrayFactory::create_<float>('c', { 5, 5 });
    auto z = NDArrayFactory::create_<float>('c', { 5, 5 });
    auto exp = NDArrayFactory::create<float>('c', { 5, 5 });
    x->assign(3);
    y->assign(1);
    z->assign(2);
    exp.assign(2);

    auto zu = NDArrayFactory::create<float>('c', { 5, 5 });

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    variableSpace->putVariable(-3, z);
    variableSpace->putVariable(1, new Variable(NDArrayFactory::create_<float>('c', { 5, 5 })));
    auto block = new Context(1, variableSpace, false);
    block->fillInputs({ -1, -2, -3 });

    sd::ops::mergeavg merge;

    merge.execute(block);

    auto res = variableSpace->getVariable(1)->getNDArray();

    ASSERT_TRUE(res->equalsTo(&exp));

    delete block;
    delete variableSpace;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, SubtractVectorVector1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto exp = NDArrayFactory::create<float>('c', { 1, 15 });
    x->assign(3);
    y->assign(1);
    exp.assign(2);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::subtract subOp;

    subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete block;
    delete variableSpace;

}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, SubtractMatrixScalar1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto exp = NDArrayFactory::create<float>('c', { 5, 3 });
    x->assign(3);
    y->assign(1);
    exp.assign(2);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::subtract subOp;

    subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete block;
    delete variableSpace;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, SubtractScalarScalar1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto exp = NDArrayFactory::create<float>('c', { 1, 1 });
    x->assign(3);
    y->assign(1);
    exp.assign(2);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::subtract subOp;

    subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete block;
    delete variableSpace;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseSubtractMatrices1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto exp = NDArrayFactory::create<float>('c', { 5, 3 });
    x->assign(3.f);
    y->assign(1.f);
    exp.assign(-2.f);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::reversesubtract subOp;

    subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseSubtractTest_1) {

    auto x = NDArrayFactory::create<float>('c', { 1, 6 });
    auto y = NDArrayFactory::create<float>('c', { 1, 6 });
    auto exp = NDArrayFactory::create<float>('c', { 1, 6 });
    x.assign(3.f);
    y.assign(1.f);
    exp.assign(-2.f);

    sd::ops::reversesubtract subOp;

    auto res = subOp.evaluate({ &x, &y });

    ASSERT_TRUE(res.status() == ND4J_STATUS_OK);
    ASSERT_TRUE(res.at(0)->equalsTo(&exp));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseSubtractTest_2) {

    //    auto x('c', {1, 6});
    auto x = NDArrayFactory::create<float>('c', { 1, 6 });
    auto y = NDArrayFactory::create<float>('c', { 3, 4, 5, 1 });
    auto exp = NDArrayFactory::create<float>('c', { 3, 4, 5, 6 });
    auto z(exp);
    x.assign(3.f);
    y.assign(1.f);
    exp.assign(-2.f);
    x.applyTrueBroadcast(BROADCAST(ReverseSubtract), y, z, true);

    ASSERT_TRUE(exp.equalsTo(&z));

    sd::ops::reversesubtract subOp;

    auto res = subOp.evaluate({ &x, &y });

    ASSERT_TRUE(res.status() == ND4J_STATUS_OK);
    ASSERT_TRUE(res.at(0)->equalsTo(&exp));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseSubtractTest_3) {

    //    auto x('c', {1, 6});
    auto x = NDArrayFactory::create<float>('c', { 6 });
    auto y = NDArrayFactory::create<float>('c', { 3, 4, 5, 1 });
    auto exp = NDArrayFactory::create<float>('c', { 3, 4, 5, 6 });
    auto z(exp);
    x.assign(1);
    y.assign(3);
    exp.assign(2);
    x.applyTrueBroadcast(BROADCAST(ReverseSubtract), y, z, true);
    ASSERT_TRUE(z.equalsTo(&exp));
    sd::ops::reversesubtract subOp;

    auto res = subOp.evaluate({ &x, &y });
    ASSERT_TRUE(res.status() == ND4J_STATUS_OK);
    ASSERT_TRUE(res.at(0)->equalsTo(&exp));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseModTest_1) {

    //    auto x('c', {1, 6});
    auto x = NDArrayFactory::create<double>('c', { 6 });
    auto y = NDArrayFactory::create<double>('c', { 3, 4, 5, 1 });
    auto exp = NDArrayFactory::create<double>('c', { 3, 4, 5, 6 });
    auto z(exp);
    x.assign(2.);
    y.assign(9.f);
    exp.assign(1.f);
    y.applyTrueBroadcast(BROADCAST(Mod), x, z, true);
    ASSERT_TRUE(exp.equalsTo(&z));

    x.applyTrueBroadcast(BROADCAST(ReverseMod), y, exp, true);
    ASSERT_TRUE(exp.equalsTo(&z));

    sd::ops::reversemod subOp;

    auto res = subOp.evaluate({ &x, &y });

    ASSERT_TRUE(res.status() == ND4J_STATUS_OK);
    ASSERT_TRUE(res.at(0)->equalsTo(&exp));
    ASSERT_TRUE(exp.equalsTo(&z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseModTest_2) {

    //    auto x('c', {1, 6});
    auto x = NDArrayFactory::create<float>('c', { 3, 4, 5 });
    auto y = NDArrayFactory::create<float>('c', { 3, 4, 5 });
    auto exp = NDArrayFactory::create<float>('c', { 3, 4, 5 });
    auto z(exp);
    x.assign(2.f);
    y.assign(9.f);
    exp.assign(1.f);
    x.applyTrueBroadcast(BROADCAST(ReverseMod), y, z, true);
    ASSERT_TRUE(z.equalsTo(&exp));
    x.applyTrueBroadcast(BROADCAST(ReverseMod), y, exp, true);
    ASSERT_TRUE(z.equalsTo(&exp));

    sd::ops::reversemod subOp;

    auto res = subOp.evaluate({ &x, &y });

    ASSERT_TRUE(res.status() == ND4J_STATUS_OK);
    ASSERT_TRUE(res.at(0)->equalsTo(&exp));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseSubtractVectorVector1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto exp = NDArrayFactory::create_<float>('c', { 1, 15 });
    x->assign(3);
    y->assign(1);
    exp->assign(-2);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::reversesubtract subOp;

    subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseSubtractMatrixScalar1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto exp = NDArrayFactory::create_<float>('c', { 5, 3 });
    x->assign(3);
    y->assign(1);
    exp->assign(-2);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::reversesubtract subOp;

    subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseSubtractScalarScalar1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto exp = NDArrayFactory::create_<float>('c', { 1, 1 });
    x->assign(3);
    y->assign(1);
    exp->assign(-2);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::reversesubtract subOp;

    subOp.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, MultiplyMatrices1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto exp = NDArrayFactory::create_<float>('c', { 5, 3 });
    x->assign(2);
    y->assign(3);
    exp->assign(6);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::multiply mul;

    mul.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, MultiplyVectorVector1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto exp = NDArrayFactory::create_<float>('c', { 1, 15 });
    x->assign(2);
    y->assign(3);
    exp->assign(6);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::multiply mul;

    mul.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, MultiplyMatrixScalar) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto exp = NDArrayFactory::create_<float>('c', { 5, 3 });
    x->assign(2);
    y->assign(3);
    exp->assign(6);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::multiply mul;

    mul.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, MultiplyScalarScalar1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto exp = NDArrayFactory::create_<float>('c', { 1, 1 });
    x->assign(2);
    y->assign(3);
    exp->assign(6);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::multiply mul;

    mul.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete block;
    delete variableSpace;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestSoftMax_bp_1) {

    auto input = NDArrayFactory::create_<double>('c', { 2, 2 });
    for (int e = 0; e < input->lengthOf(); e++)
        input->p(e, e + 1);

    auto epsilon = NDArrayFactory::create_<double>('c', { 2, 2 });
    epsilon->p(0, 0.1f);
    epsilon->p(1, 0.2f);
    epsilon->p(2, 0.3f);
    epsilon->p(3, 0.4f);

    auto output = NDArrayFactory::create_<double>('c', { 2, 2 });
    output->assign(1.0f);

    auto exp = NDArrayFactory::create_<double>('c', { 2, 2 });
    exp->p(0, -0.019661194f);
    exp->p(1, 0.019661194f);
    exp->p(2, -0.019661194f);
    exp->p(3, 0.019661194f);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, epsilon);
    variableSpace->putVariable(1, output);
    //variableSpace->putVariable(42, exp);

    auto block = new Context(1, variableSpace, false);
    block->fillInputs({ -1, -2 });

    sd::ops::softmax_bp op;

    Nd4jStatus status = op.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(output->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, BroadcastDivideTest_1) {

    auto  x = NDArrayFactory::create<float>('c', { 3, 4, 5, 1 });
    auto  y = NDArrayFactory::create<float>('c', { 1, 6 });
    auto  exp = NDArrayFactory::create<float>('c', { 3, 4, 5, 6 });
    x.assign(6);
    y.assign(2);
    exp.assign(3);

    sd::ops::divide div;

    auto res = div.evaluate({ &x, &y });

    ASSERT_EQ(res.status(), ND4J_STATUS_OK);
    ASSERT_TRUE(res.at(0)->equalsTo(exp));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, BroadcastDivideTest_2) {

    auto  x = NDArrayFactory::create<float>('c', { 3, 4, 5, 1 });
    auto  y = NDArrayFactory::create<float>('c', { 1, 6 });
    auto  exp = NDArrayFactory::create<float>('c', { 3, 4, 5, 6 });
    x.assign(6);
    y.assign(2);
    exp.assign(3);

    sd::ops::divide_no_nan div;
    auto res = div.evaluate({ &x, &y });

    ASSERT_EQ(res.status(), ND4J_STATUS_OK);
    ASSERT_TRUE(res.at(0)->equalsTo(exp));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, BroadcastDivideTest_3) {

    auto  x = NDArrayFactory::create<float>({ 6,6,6,6,6 });
    auto  y = NDArrayFactory::create<float>({ 3,3,0,3,3 });
    auto  exp = NDArrayFactory::create<float>({ 2, 2, 0, 2, 2 });

    sd::ops::divide_no_nan div;
    auto res = div.evaluate({ &x, &y });

    ASSERT_EQ(res.status(), ND4J_STATUS_OK);
    ASSERT_TRUE(res.at(0)->equalsTo(exp));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, BroadcastReverseDivideTest_1) {

    auto  x = NDArrayFactory::create<float>('c', { 3, 4, 5, 1 });
    auto  y = NDArrayFactory::create<float>('c', { 1, 6 });
    auto  exp = NDArrayFactory::create<float>('c', { 3, 4, 5, 6 });
    x.assign(3.f);
    y.assign(6.f);
    exp.assign(2.f);

    sd::ops::reversedivide div;

    auto res = div.evaluate({ &x, &y });

    ASSERT_EQ(res.status(), ND4J_STATUS_OK);

    ASSERT_TRUE(res.at(0)->equalsTo(exp));
    auto z(exp);
    x.applyTrueBroadcast(BROADCAST(ReverseDivide), y, z, true);
    y.applyTrueBroadcast(BROADCAST(Divide), x, exp, true);

    ASSERT_TRUE(z.equalsTo(&exp));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, DivideMatrices1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto exp = NDArrayFactory::create_<float>('c', { 5, 3 });
    x->assign(6);
    y->assign(2);
    exp->assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::divide div;

    div.execute(block);

    ASSERT_TRUE(x->equalsTo(exp));

    delete variableSpace;
    delete block;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, DivideVectorVector1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto exp = NDArrayFactory::create<float>('c', { 1, 15 });
    x->assign(6);
    y->assign(2);
    exp.assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::divide div;

    div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, DivideMatrixScalar1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto exp = NDArrayFactory::create<float>('c', { 5, 3 });
    x->assign(6);
    y->assign(2);
    exp.assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::divide div;

    div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete block;
    delete variableSpace;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, DivideScalarScalar1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 1 });
    auto y = NDArrayFactory::create_<float>('c', { 5, 1 });
    auto exp = NDArrayFactory::create<float>('c', { 5, 1 });
    x->assign(6);
    y->assign(2);
    exp.assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::divide div;

    div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseDivideMatrices1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto exp = NDArrayFactory::create<float>('c', { 5, 3 });
    x->assign(2);
    y->assign(6);
    exp.assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::reversedivide div;

    div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseDivideVectorVector1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 15 });
    auto exp = NDArrayFactory::create<float>('c', { 1, 15 });
    x->assign(2);
    y->assign(6);
    exp.assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::reversedivide div;

    div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseDivideMatrixScalar1) {

    auto x = NDArrayFactory::create_<float>('c', { 5, 3 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto exp = NDArrayFactory::create<float>('c', { 5, 3 });
    x->assign(2);
    y->assign(6);
    exp.assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::reversedivide div;

    div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, ReverseDivideScalarScalar1) {

    auto x = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto y = NDArrayFactory::create_<float>('c', { 1, 1 });
    auto exp = NDArrayFactory::create<float>('c', { 1, 1 });
    x->assign(2);
    y->assign(6);
    exp.assign(3);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    auto block = new Context(1, variableSpace, true);
    block->fillInputs({ -1, -2 });

    sd::ops::reversedivide div;

    div.execute(block);

    ASSERT_TRUE(x->equalsTo(&exp));

    delete variableSpace;
    delete block;
}

TEST_F(DeclarableOpsTests1, Test_Cast_1) {
    // TODO: right now there's no real cast implementation, but genera idea should be the same: arrays equality to be expected
    auto x = NDArrayFactory::create<float>('c', { 5, 5 });
    auto yExp = NDArrayFactory::create<float16>('c', { 5, 5 });
    x.linspace(1);
    yExp.linspace(1);
    sd::ops::cast op;

    auto result = op.evaluate({ &x }, {}, { 3 });
    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);
    ASSERT_TRUE(yExp.equalsTo(z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestRegistrator1) {
    auto res = sd::ops::OpRegistrator::getInstance()->getAllCustomOperations();
}

// //////////////////////////////////////////////////////////////////////
// TEST_F(DeclarableOpsTests1, TestLegacyExecution1) {
//     auto x = NDArrayFactory::create_<float>('c', {10, 10});
//     x->assign(1.0f);

//     auto y = NDArrayFactory::create_<float>('c', {10, 10});
//     y->assign(2.0f);

//     auto z = NDArrayFactory::create_<float>('c', {10, 10});

//     auto exp = NDArrayFactory::create_<float>('c', {10, 10});
//     exp->assign(3.0f);
//     z->assign(120.0f);
//     std::string opName("add");

//     auto hash = sd::ops::HashHelper::getInstance()->getInstance()->getLongHash(opName);

//     auto inputBuffers = new Nd4jPointer[2];
//     auto inputShapes = new Nd4jPointer[2];

//     inputBuffers[0] = (Nd4jPointer) x->buffer();
//     inputBuffers[1] = (Nd4jPointer) y->buffer();

//     inputShapes[0] = (Nd4jPointer) x->shapeInfo();
//     inputShapes[1] = (Nd4jPointer) y->shapeInfo();

//     auto outputBuffers = new Nd4jPointer[1];
//     auto outputShapes = new Nd4jPointer[1];

//     outputBuffers[0] = (Nd4jPointer) z->buffer();
//     outputShapes[0] = (Nd4jPointer) z->shapeInfo();


//     //auto status = execCustomOp(nullptr, hash, inputBuffers, inputShapes, 2, outputBuffers, outputShapes, 1, nullptr, 0, nullptr, 0, false);
//     auto status = execCustomOp(nullptr, hash, inputBuffers, inputShapes, 2, outputBuffers, outputShapes, 1, nullptr, 0, nullptr, 0, nullptr, 0, false);
//     ASSERT_EQ(ND4J_STATUS_OK, status);
//     ASSERT_NEAR(2.0f, y->meanNumber().e<float>(0), 1e-5);
//     ASSERT_NEAR(1.0f, x->meanNumber().e<float>(0), 1e-5);
//     ASSERT_NEAR(3.0f, z->meanNumber().e<float>(0), 1e-5);

//     delete x;
//     delete y;
//     delete z;
//     delete exp;
//     delete[] inputBuffers;
//     delete[] inputShapes;
//     delete[] outputBuffers;
//     delete[] outputShapes;
// }

// //////////////////////////////////////////////////////////////////////
// TEST_F(DeclarableOpsTests1, TestLegacyExecution2) {
//     auto x = NDArrayFactory::create_<float>('c', {10, 10});
//     x->assign(1.0f);

//     auto y = NDArrayFactory::create_<float>('c', {10, 10});
//     y->assign(2.0f);

//     auto z = NDArrayFactory::create_<float>('c', {10, 10});

//     auto exp = NDArrayFactory::create_<float>('c', {10, 10});
//     exp->assign(3.0);

//     std::string opName("add");

//     auto hash = sd::ops::HashHelper::getInstance()->getInstance()->getLongHash(opName);

//     auto inputBuffers = new Nd4jPointer[2];
//     auto inputShapes = new Nd4jPointer[2];

//     inputBuffers[0] = (Nd4jPointer) x->buffer();
//     inputBuffers[1] = (Nd4jPointer) y->buffer();

//     inputShapes[0] = (Nd4jPointer) x->shapeInfo();
//     inputShapes[1] = (Nd4jPointer) y->shapeInfo();

//     auto outputBuffers = new Nd4jPointer[1];
//     auto outputShapes = new Nd4jPointer[1];

//     execCustomOp(nullptr, hash, inputBuffers, inputShapes, 2, outputBuffers, outputShapes, 1, nullptr, 0, nullptr, 0, nullptr, 0, true);

//     ASSERT_NEAR(2.0, y->meanNumber().e<float>(0), 1e-5);
//     ASSERT_NEAR(3.0, x->meanNumber().e<float>(0), 1e-5);


//     delete x;
//     delete y;
//     delete z;
//     delete exp;
//     delete[] inputBuffers;
//     delete[] inputShapes;
//     delete[] outputBuffers;
//     delete[] outputShapes;
// }

#ifndef __CUDABLAS__
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestGemv1) {
    /*
    auto xBuffer = new float[15]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
    auto xShape = new Nd4jLong[8] {2, 5, 3, 3, 1, 0, 1, 99};
    ArrayOptions::setDataType(xShape, sd::DataType::FLOAT32);
    auto x = new NDArray(xBuffer, xShape);

    auto yBuffer = new float[3]{2.f, 4.f, 6.f};
    auto yShape = new Nd4jLong[8] {2, 3, 1, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(yShape, sd::DataType::FLOAT32);

    auto y = new NDArray(yBuffer, yShape);

    auto z = NDArrayFactory::create_<float>('f', {5, 1});

    auto expBuffer = new float[5]{28.00f,64.00f,100.00f,136.00f,172.00f};
    auto exp = new NDArray(expBuffer, z->shapeInfo());

     sd::blas::GEMV<float, float, float>::op('f',  x->rows(), x->columns(), 1.0f, x->buffer(), y->rows(), y->buffer(), 1, 0.0, z->buffer(), 1);

    ASSERT_TRUE(z->equalsTo(exp));

    delete []xBuffer; delete []xShape; delete x; delete []yBuffer; delete []yShape; delete y; delete z; delete []expBuffer; delete exp;
     */
}

#endif


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Transpose1) {
    auto x = NDArrayFactory::create_<float>('c', { 3,5,2 });
    auto exp = NDArrayFactory::create_<float>('c', { 2,5,3 });

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(1, new Variable());

    auto block = new Context(1, variableSpace, false);  // not-in-place
    block->fillInputs({ -1 });
    sd::ops::transpose transpose;

    Nd4jStatus  status = transpose.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto result = variableSpace->getVariable(block->getNodeId())->getNDArray();

    ASSERT_TRUE(exp->isSameShape(result));
    ASSERT_TRUE(exp->dataType() == result->dataType());
    ASSERT_TRUE(exp->ordering() == result->ordering());

    delete exp;
    delete block;
    delete variableSpace;
}

//////////////////////////////////////////////////////////////////////
// not-in-place
TEST_F(DeclarableOpsTests1, Permute1) {

    Nd4jLong shapeX[] = { 3, 5,10,15,  150,15,1,  0,1,99 };
    Nd4jLong shapeExp[] = { 3, 15,5,10,  50,10,1,  0,1,99 };
    const std::vector<int> perm = { 2, 0, 1 };

    ArrayOptions::setDataType(shapeX, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(shapeExp, sd::DataType::FLOAT32);

    auto x = new NDArray(shapeX, true);
    auto exp = new NDArray(shapeExp, true);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(1, new Variable());

    auto block = new Context(1, variableSpace, false);  // not-in-place
    block->fillInputs({ -1 });
    auto arguments = block->getIArguments();
    *arguments = perm;      // set dimensions to be permuted

    sd::ops::permute permute;
    Nd4jStatus status = permute.execute(block);
    auto result = variableSpace->getVariable(block->getNodeId())->getNDArray();

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(result->isSameShapeStrict(*exp));

    delete block;
    delete variableSpace;
    delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestArgumentsValidation1) {
    Nd4jLong shapeX[] = { 3, 5, 10, 15, 150, 15, 1, 0, 1, 99 };
    Nd4jLong shapeExp[] = { 3, 15, 5, 10, 1, 150, 15, 0, -1, 99 };

    ArrayOptions::setDataType(shapeX, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(shapeExp, sd::DataType::FLOAT32);

    const std::vector<int> perm = { 2, 0, 1 };
    auto x = new NDArray(shapeX);
    auto exp = new NDArray(shapeExp);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(1, new Variable());

    auto block = new Context(1, variableSpace, false);  // not-in-place
    block->fillInputs({ -1 });

    sd::ops::im2col permute;
    Nd4jStatus status = permute.execute(block);

    ASSERT_TRUE(status != 0);

    delete exp;
    delete block;
    delete variableSpace;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestReductionShape1) {
    auto input = NDArrayFactory::create_<float>('c', { 4, 5, 5, 10, 10 });

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, input);

    auto block = new Context(1, variableSpace, false);  // not-in-place
    block->fillInputs({ -1 });

    // kernel params
    block->getIArguments()->push_back(MAX_INT);

    sd::ops::testreduction testop;

    auto inP = new Nd4jLong[shape::shapeInfoLength(input->shapeInfo())];
    memcpy(inP, input->shapeInfo(), shape::shapeInfoByteLength(input->rankOf()));

    auto inshape = new ShapeList(inP);

    auto shapes = testop.calculateOutputShape(inshape, *block);

    ASSERT_EQ(1, shapes->size());
    ASSERT_EQ(0, shapes->at(0)[0]); // scalar shape has rank 0
    ASSERT_EQ(8192, shapes->at(0)[1]);
    ASSERT_EQ(1, shapes->at(0)[2]);

    delete[] inP;
    delete variableSpace;
    delete block;
    delete inshape;
    delete shapes;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestReductionShape2) {
    auto input = NDArrayFactory::create_<float>('c', { 4, 5, 5, 10, 10 });

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, input);

    auto block = new Context(1, variableSpace, false);  // not-in-place
    block->fillInputs({ -1 });

    // kernel params
    //block->getIArguments()->push_back(4);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(2);
    block->getIArguments()->push_back(3);
    block->getIArguments()->push_back(4);

    sd::ops::testreduction testop;

    auto inshapes = new ShapeList(input->shapeInfo());
    auto shapes = testop.calculateOutputShape(inshapes, *block);
    ASSERT_EQ(1, shapes->size());
    ASSERT_EQ(1, shapes->at(0)[0]);
    ASSERT_EQ(4, shapes->at(0)[1]);
    ASSERT_EQ(1, shapes->at(0)[2]);

    delete variableSpace;
    delete block;
    delete shapes;
    delete inshapes;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, TestCustomShape1) {
    auto input = NDArrayFactory::create_<float>('c', { 2, 3, 4 });

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, input);

    auto block = new Context(1, variableSpace, false);  // not-in-place
    block->fillInputs({ -1 });

    sd::ops::testcustom test;

    auto inshapes = new ShapeList(input->shapeInfo());
    auto shapes = test.calculateOutputShape(inshapes, *block);


    ASSERT_EQ(input->shapeInfo()[0], shapes->at(0)[0]);
    ASSERT_EQ(input->shapeInfo()[1] * 2, shapes->at(0)[1]);
    ASSERT_EQ(input->shapeInfo()[2] * 2, shapes->at(0)[2]);
    ASSERT_EQ(input->shapeInfo()[3] * 2, shapes->at(0)[3]);

    delete variableSpace;
    delete block;
    delete shapes;
    delete inshapes;
}


//////////////////////////////////////////////////////////////////////
/*
TEST_F(DeclarableOpsTests1, Sum1) {

    float xBuff[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int xShape[]  = {2, 4, 2, 2, 1, 0, 1, 99};
    float expBuff[] = {16, 20};
    int expShape[]  = {2, 1, 2, 2, 1, 0, 1, 99};

    const std::vector<int> dimensions = {1,0};

    auto x = NDArrayFactory::create_<float>(xBuff, xShape);
    auto z = NDArrayFactory::create_<float>(1, 2, 'c');
    auto exp(expBuff, expShape);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(1, z);

    auto block = new Context(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1});
    std::vector<int>* arguments = block->getIArguments();
    *arguments = dimensions;

    sd::ops::sum<float> sum;
    Nd4jStatus status = sum.execute(block);

    auto result = variableSpace->getVariable(block->getNodeId())->getNDArray();

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(result->equalsTo(&exp));

    delete block;
    delete variableSpace;
}
*/

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Pnormpool2d1) {

    auto x = NDArrayFactory::create_<float>('c', { bS,iD,iH,iW });
    auto exp = NDArrayFactory::create<float>('c', { bS,iD,oH,oW });
    // auto z('c',{bS,iD,oH,oW});

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    // variableSpace->putVariable(1, &z);

    auto block = new Context(1, variableSpace, false);
    block->fillInputs({ -1 });
    std::vector<int>* argI = block->getIArguments();
    *argI = { kH,kW, sH,sW, pH,pW, dW,dH, 0, 1, 0 };  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode; 9 - extraParam0 for pnorm case;

    sd::ops::pnormpool2d pooling;
    Nd4jStatus status = pooling.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto result = variableSpace->getVariable(block->getNodeId())->getNDArray();
    ASSERT_TRUE(exp.isSameShape(result));

    delete variableSpace;
    delete block;
}

/*/////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, IsMax1) {

    float xBuff[]   = {1,2,3,4,5,6,7,8,9};
    Nd4jLong xShape[]    = {2,3,3,3,1,0,1,99};
    bool expBuff[] = {0,0,1,0,0,1,0,0,1};
    ArrayOptions::setDataType(xShape, sd::DataType::BOOL);

    auto x = new NDArray(xBuff, xShape);
    NDArray exp(expBuff, xShape);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);

    auto block = new Context(1, variableSpace, false);
    block->fillInputs({-1});
    std::vector<int>* argI = block->getIArguments();
//    *argI = {1};                                        // dimensions
    argI->push_back(1); // = {1};                                        // dimensions

    sd::ops::ismax ismaxOp;
    Nd4jStatus status = ismaxOp.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto result = variableSpace->getVariable(block->getNodeId())->getNDArray();
    result->printIndexedBuffer("IS_MAX");
    ASSERT_TRUE(exp.equalsTo(result));

    delete variableSpace;
    delete block;
}
*/

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, IsMax1) {
    NDArray x('c', { 3, 3 }, sd::DataType::FLOAT32);
    //    NDArray exp('c', {3, 3}, sd::DataType::BOOL);
    NDArray exp('c', { 3, 3 }, sd::DataType::FLOAT32);
    x.linspace(1);
    exp.p<bool>(0, 2, true);
    exp.p<bool>(1, 2, true);
    exp.p<bool>(2, 2, true);

    sd::ops::ismax ismaxOp;
    auto result = ismaxOp.evaluate({ &x }, {}, { 1 });

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto res = result.at(0);
    //res->printIndexedBuffer("IS_MAX");
    ASSERT_TRUE(exp.equalsTo(res));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, IsMax2) {
    NDArray x('c', { 3, 3 }, sd::DataType::FLOAT32);
    //    NDArray exp('c', {3, 3}, sd::DataType::BOOL);
    NDArray exp('c', { 3, 3 }, sd::DataType::FLOAT32);
    x.linspace(1);
    //exp.p<bool>(0, 2, true);
    //exp.p<bool>(1, 2, true);
    exp.p<bool>(2, 2, true);

    sd::ops::ismax ismaxOp;
    auto result = ismaxOp.evaluate({ &x }, {}, { 0, 1 });

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto res = result.at(0);
    //res->printIndexedBuffer("IS_MAX");
    ASSERT_TRUE(exp.equalsTo(res));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, IsMax3) {
    NDArray x = NDArrayFactory::create<float>(120.f); //('c', {3, 3}, sd::DataType::FLOAT32);
//    NDArray exp('c', {3, 3}, sd::DataType::BOOL);
    NDArray exp = NDArrayFactory::create<float>(1.f);//, sd::DataType::FLOAT32); //'c', {3, 3}, sd::DataType::FLOAT32);
    x.linspace(1);
    //exp.p<bool>(0, 2, true);
    //exp.p<bool>(1, 2, true);
    //exp.p<bool>(2, 2, true);

    sd::ops::ismax ismaxOp;
    auto result = ismaxOp.evaluate({ &x }, {}, { 0 });

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto res = result.at(0);
    //res->printIndexedBuffer("IS_MAX");
    ASSERT_TRUE(exp.equalsTo(res));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, IsMax4) {
    auto x = NDArrayFactory::create<double>('c', { 6 }, { 0, 0, 0, 2, 2, 0 });
    auto z = NDArrayFactory::create<bool>('c', { 6 });
    auto e = NDArrayFactory::create<bool>('c', { 6 }, { false, false, false, true, false, false });

    sd::ops::ismax op;
    auto result = op.execute({ &x }, { &z });
    ASSERT_EQ(Status::OK(), result);

    ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////
// TEST_F(DeclarableOpsTests1, sru_old_test1) {

//     const int bS = 2;
//     const int K = 3;
//     const int N = 4;

//     NDArray input('c', {bS,K,N}, sd::DataType::DOUBLE);
//     NDArray weights('c', {3*K,K}, sd::DataType::DOUBLE);
//     NDArray bias('c', {1,2*K}, sd::DataType::DOUBLE);
//     NDArray init('c', {bS,K}, sd::DataType::DOUBLE);
//     NDArray mask('c', {bS,K}, sd::DataType::DOUBLE);
//     NDArray expState('c', {bS,K,N}, {0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715}, sd::DataType::DOUBLE);
//     NDArray expOut('c', {bS,K,N}, {1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656}, sd::DataType::DOUBLE);

//     input.assign(1.5);
//     weights.assign(0.5);
//     bias.assign(0.3) ;
//     init.assign(1.);
//     mask.assign(1.);

//     sd::ops::sru_old op;
//     auto  results = op.execute({&input, &weights, &bias, &init, &mask}, {}, {});
//     ASSERT_TRUE(results.size() == 2);

//     auto state  = results.at(0);
//     auto output = results.at(1);
//     // state->printBuffer();
//     // expState.printIndexedBuffer("EXP STATE");
//     // state->printIndexedBuffer("OUT STATE");
//     ASSERT_TRUE(expState.equalsTo(state));
//     ASSERT_TRUE(expOut.equalsTo(output));

//
// }

//////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, sru_test1) {

    const int bS = 2;
    const int K = 3;
    const int N = 4;

    NDArray input('c', { bS,K,N }, sd::DataType::DOUBLE);
    NDArray weights('c', { 3 * K,K }, sd::DataType::DOUBLE);
    NDArray bias('c', { 2 * K }, sd::DataType::DOUBLE);
    NDArray init('c', { bS,K }, sd::DataType::DOUBLE);
    NDArray mask('c', { bS,K }, sd::DataType::DOUBLE);
    NDArray expState('c', { bS,K,N }, { 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656, 1.090533, 1.174509, 1.252403, 1.324656 }, sd::DataType::DOUBLE);
    NDArray expOut('c', { bS,K,N }, { 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715 }, sd::DataType::DOUBLE);

    input.assign(1.5);
    weights.assign(0.5);
    bias.assign(0.3);
    init.assign(1.);
    mask.assign(1.);

    sd::ops::sru op;
    auto results = op.evaluate({ &input, &weights, &bias, &init, &mask });
    ASSERT_TRUE(results.size() == 2);

    auto output = results.at(0);
    auto state = results.at(1);

    ASSERT_TRUE(expState.equalsTo(state));
    ASSERT_TRUE(expOut.equalsTo(output));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, sru_bp) {

    const int bS = 2;
    const int K = 3;
    const int N = 4;
    std::vector<double> expGradXBuff = { -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165, -0.0259303, -0.03869125, -0.0302272, -0.02299165 };
    std::vector<double> expGradWBuff = { 0.42526005,0.42526005,0.42526005, 0.42526005,0.42526005,0.42526005, 0.42526005,0.42526005,0.42526005, -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, 0.42526005,0.42526005,0.42526005, 0.42526005,0.42526005,0.42526005, 0.42526005,0.42526005,0.42526005, -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.5282811 , -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215, -0.15967215 };
    std::vector<double> expGradBBuff = { -0.7043748, -0.7043748, -0.7043748, -0.2128962, -0.2128962, -0.2128962 };
    std::vector<double> expGradInitBuff = { 1.1421, 1.1421, 1.1421, 1.1421, 1.1421, 1.1421 };
    std::vector<double> stateBuff = { 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715, 0.847983, 0.874549, 0.896109, 0.913715 };

    auto input = NDArrayFactory::create<double>('c', { bS,K,N });
    auto weights = NDArrayFactory::create<double>('c', { 3 * K,K });
    auto bias = NDArrayFactory::create<double>('c', { 1,2 * K });
    auto init = NDArrayFactory::create<double>('c', { bS,K });
    auto mask = NDArrayFactory::create<double>('c', { bS,K });
    auto state = NDArrayFactory::create<double>('c', { bS,K,N }, stateBuff);
    auto inGradCt = NDArrayFactory::create<double>('c', { bS,K });
    auto inGradH = NDArrayFactory::create<double>('c', { bS,K,N });

    auto expGradX = NDArrayFactory::create<double>('c', { bS,K,N }, expGradXBuff);
    auto expGradW = NDArrayFactory::create<double>('c', { bS,3 * K,K }, expGradWBuff);
    auto expGradB = NDArrayFactory::create<double>('c', { 1,2 * K }, expGradBBuff);
    auto expGradInit = NDArrayFactory::create<double>('c', { bS,K }, expGradInitBuff);

    input.assign(1.5);
    weights.assign(0.5);
    bias.assign(0.3);
    mask.assign(1.);
    init.assign(1.);
    inGradCt.assign(0.5);
    inGradH.assign(0.5);

    sd::ops::sru_bp bp;
    auto resultsBP = bp.evaluate({ &input, &weights, &bias, &init, &state, &inGradCt, &inGradH, &mask }, {}, {});
    ASSERT_TRUE(resultsBP.size() == 4);

    auto gradX = resultsBP.at(0);
    auto gradW = resultsBP.at(1);
    auto gradB = resultsBP.at(2);
    auto gradInit = resultsBP.at(3);
    // expGradX.printBuffer("Exp GRAD");
    // gradX->printBuffer("Res GRAD");
    ASSERT_TRUE(expGradX.equalsTo(gradX, 1e-4));
    ASSERT_TRUE(expGradW.equalsTo(gradW));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
    ASSERT_TRUE(expGradInit.equalsTo(gradInit));


}

//////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, sru_bi_1) {

    const int bS = 2;
    const int K = 3;
    const int N = 4;

    NDArray input('c', { N,bS,2 * K }, sd::DataType::DOUBLE);
    NDArray weights('c', { 2 * K,6 * K }, sd::DataType::DOUBLE);
    NDArray bias('c', { 4 * K }, sd::DataType::DOUBLE);
    NDArray init('c', { bS,2 * K }, sd::DataType::DOUBLE);
    NDArray mask('c', { bS,2 * K }, sd::DataType::DOUBLE);
    NDArray expState('c', { N,bS,2 * K }, { 1.02857, 1.02857, 1.02857, 1.11288, 1.11288, 1.11288, 1.02857, 1.02857, 1.02857, 1.11288, 1.11288, 1.11288, 1.0569, 1.0569, 1.0569, 1.08501, 1.08501, 1.08501, 1.0569, 1.0569, 1.0569, 1.08501, 1.08501, 1.08501, 1.08501, 1.08501, 1.08501, 1.0569, 1.0569, 1.0569, 1.08501, 1.08501, 1.08501, 1.0569, 1.0569, 1.0569, 1.11288, 1.11288, 1.11288, 1.02857, 1.02857, 1.02857, 1.11288, 1.11288, 1.11288, 1.02857, 1.02857, 1.02857 });
    NDArray expOut('c', { N,bS,2 * K }, { 0.779265, 0.779265, 0.779265, 0.810752, 0.810752, 0.810752, 0.779265, 0.779265, 0.779265, 0.810752, 0.810752, 0.810752, 0.790317, 0.790317, 0.790317, 0.800804, 0.800804, 0.800804, 0.790317, 0.790317, 0.790317, 0.800804, 0.800804, 0.800804, 0.800804, 0.800804, 0.800804, 0.790317, 0.790317, 0.790317, 0.800804, 0.800804, 0.800804, 0.790317, 0.790317, 0.790317, 0.810752, 0.810752, 0.810752, 0.779265, 0.779265, 0.779265, 0.810752, 0.810752, 0.810752, 0.779265, 0.779265, 0.779265 });

    input.assign(1.5);
    weights.assign(0.5);
    bias.assign(0.3);
    init.assign(1.);
    mask.assign(1.);

    sd::ops::sru_bi op;
    auto results = op.evaluate({ &input, &weights, &bias, &init, &mask }, {}, {});
    ASSERT_TRUE(results.size() == 2);

    auto output = results.at(0);
    auto state = results.at(1);
    // state->printBuffer();
    // output->printBuffer();

    ASSERT_TRUE(expState.equalsTo(state));
    ASSERT_TRUE(expOut.equalsTo(output));


}

TEST_F(DeclarableOpsTests1, sru_bi_bp_1) {

    const int bS = 2;
    const int K = 3;
    const int N = 3;
    std::vector<double> expGradXBuff = { 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129, 0.00408129 };
    std::vector<double> expGradInitBuff = { 1.05121, 1.05121, 1.05121, 1.02676, 1.02676, 1.02676, 1.05121, 1.05121, 1.05121, 1.02676, 1.02676, 1.02676 };
    std::vector<double> expGradWBuff = { 0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02595354,-0.090096 ,-0.00882456,0.02595354,-0.090096 ,-0.0088245, 0.02595354,-0.090096 ,-0.00882456,0.01651665,-0.0559437,-0.0084390, 0.01651665,-0.0559437,-0.00843906,0.01651665,-0.0559437,-0.00843906, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.02124567,-0.0731508,-0.00868926,0.02124567,-0.0731508,-0.0086892, 0.02124567,-0.0731508,-0.00868926,0.02084955,-0.0712011,-0.0085608, 0.02084955,-0.0712011,-0.00856086,0.02084955,-0.0712011,-0.00856086, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926, 0.01671156,-0.0570699,-0.00856086,0.01671156,-0.0570699,-0.0085608, 0.01671156,-0.0570699,-0.00856086,0.02534988,-0.0880002,-0.0086892, 0.02534988,-0.0880002,-0.00868926,0.02534988,-0.0880002,-0.00868926 };
    std::vector<double> expGradBBuff = { -0.0734389, -0.0734389, -0.0734389, -0.0717151, -0.0717151, -0.0717151, -0.0734389, -0.0734389, -0.0734389, -0.0717151, -0.0717151, -0.0717151, -0.00869156, -0.00869156, -0.00869156, -0.00856306, -0.00856306, -0.00856306, -0.00869156, -0.00869156, -0.00869156, -0.00856306, -0.00856306, -0.00856306 };
    std::vector<double> stateBuff = { 1.028569, 1.028569, 1.028569, 1.112884, 1.112884, 1.112884, 1.028569, 1.028569, 1.028569, 1.112884,1.112884, 1.112884, 1.056905, 1.056905, 1.056905, 1.085009, 1.085009, 1.085009, 1.056905, 1.056905,1.056905, 1.085009, 1.085009, 1.085009, 1.085009, 1.085009, 1.085009, 1.056905, 1.056905, 1.056905,1.085009, 1.085009, 1.085009, 1.056905, 1.056905, 1.056905 };

    auto input = NDArrayFactory::create<double>('c', { N,bS,2 * K });
    auto weights = NDArrayFactory::create<double>('c', { 2 * K,6 * K });
    auto bias = NDArrayFactory::create<double>('c', { 4 * K });
    auto init = NDArrayFactory::create<double>('c', { bS,2 * K });
    auto mask = NDArrayFactory::create<double>('c', { bS,2 * K });
    NDArray state('c', { N,bS,2 * K }, stateBuff);
    auto inGradCt = NDArrayFactory::create<double>('c', { bS,2 * K });
    auto inGradH = NDArrayFactory::create<double>('c', { N,bS,2 * K });

    NDArray gradBias('c', { bS,4 * K }, expGradBBuff);

    NDArray expGradX('c', { N,bS,2 * K }, expGradXBuff);
    NDArray expGradW('c', { N,2 * K,6 * K }, expGradWBuff);
    auto expGradB = NDArrayFactory::create<double>('c', { 4 * K });
    gradBias.reduceAlongDimension(reduce::Sum, expGradB, { 0 });    // [bS, 4K] -> [4K]
    NDArray expGradInit('c', { bS,2 * K }, expGradInitBuff);

    input.assign(1.5);
    weights.assign(0.5);
    bias.assign(0.3);
    mask.assign(1.);
    init.assign(1.);
    inGradCt.assign(0.5);
    inGradH.assign(0.5);

    sd::ops::sru_bi_bp bp;
    auto resultsBP = bp.evaluate({ &input, &weights, &bias, &init, &state, &inGradCt, &inGradH, &mask }, {}, {});
    ASSERT_TRUE(resultsBP.size() == 4);

    auto gradX = resultsBP.at(0);
    auto gradW = resultsBP.at(1);
    auto gradB = resultsBP.at(2);
    auto gradInit = resultsBP.at(3);

    ASSERT_TRUE(expGradX.equalsTo(gradX));
    ASSERT_TRUE(expGradW.equalsTo(gradW));
    ASSERT_TRUE(expGradB.equalsTo(gradB));
    ASSERT_TRUE(expGradInit.equalsTo(gradInit));


}

TEST_F(DeclarableOpsTests1, ArgMax1) {
    auto x = NDArrayFactory::create<float>('c', { 3, 5 });
    x.linspace(1);
    auto exp = NDArrayFactory::create<Nd4jLong>('c', { 3 });
    exp.assign(4);

    sd::ops::argmax op;

    auto result = op.evaluate({ &x }, {}, { 1 });

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


}


TEST_F(DeclarableOpsTests1, ArgMax2) {
    auto x = NDArrayFactory::create<float>('c', { 3, 5 });
    x.linspace(1);
    auto exp = NDArrayFactory::create<Nd4jLong>('c', { 5 });
    exp.assign(2);

    sd::ops::argmax op;

    auto result = op.evaluate({ &x }, {}, { 0 });

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


}


TEST_F(DeclarableOpsTests1, ArgMax3) {
    auto x = NDArrayFactory::create<float>('c', { 3, 5 });
    auto dim = NDArrayFactory::create<float>('c', { 1, 1 }, { 0. });
    x.linspace(1);
    auto exp = NDArrayFactory::create<Nd4jLong>('c', { 5 });
    exp.assign(2);

    sd::ops::argmax op;

    auto result = op.evaluate({ &x, &dim }, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


}

TEST_F(DeclarableOpsTests1, ArgMax4) {
    auto x = NDArrayFactory::create<float>('c', { 3, 5 });
    auto dim = NDArrayFactory::create<float>('c', { 1, 1 }, { 1 });
    x.linspace(1);
    auto exp = NDArrayFactory::create<Nd4jLong>('c', { 3 });
    exp.assign(4);

    sd::ops::argmax op;

    auto result = op.evaluate({ &x, &dim }, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


}


TEST_F(DeclarableOpsTests1, ArgMax5) {
    auto x = NDArrayFactory::create<float>('c', { 3, 5 });
    auto dim = NDArrayFactory::create<float>('c', { 1, 2 }, { 0, 1 });
    x.linspace(1);
    auto exp = NDArrayFactory::create<Nd4jLong>(14);


    sd::ops::argmax op;

    auto result = op.evaluate({ &x, &dim }, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


}

TEST_F(DeclarableOpsTests1, ArgMax6) {
    auto x = NDArrayFactory::create<float>('c', { 3, 4, 5 });
    auto dim = NDArrayFactory::create<float>(-1.f);
    x.linspace(1);


    sd::ops::argmax op;

    auto expected = op.evaluate({ &x }, {}, { 2 });
    ASSERT_EQ(Status::OK(), expected.status());
    auto exp = expected.at(0);


    auto result = op.evaluate({ &x, &dim }, {}, {});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

    ASSERT_EQ(*exp, *z);
}


TEST_F(DeclarableOpsTests1, ArgMin1) {
    auto x = NDArrayFactory::create<float>('c', { 3, 5 });
    x.linspace(1);
    //    auto exp('c', {3, 1});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', { 3 });
    exp.assign(0.0f);

    sd::ops::argmin op;

    auto result = op.evaluate({ &x }, {}, { 1 });

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


}


TEST_F(DeclarableOpsTests1, SquareTests1) {
    auto x = NDArrayFactory::create<float>('c', { 3, 5 });
    x.linspace(1);

    auto exp = NDArrayFactory::create<float>('c', { 3, 5 });
    exp.linspace(1);
    exp *= exp;

    sd::ops::square op;

    auto result = op.evaluate({ &x }, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.equalsTo(z));


}

TEST_F(DeclarableOpsTests1, OneHotTests_1) {

    auto indices = NDArrayFactory::create<float>('c', { 1, 4 }, { 0.0f, 2.0f, -1.0f, 1.0f });

    auto exp = NDArrayFactory::create<float>('c', { 1, 4, 3 }, { 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f });

    sd::ops::onehot op;

    auto result = op.evaluate({ &indices }, { 1.0f, 0.0f }, { -1, 3 });
    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);
    // z->printBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


}

TEST_F(DeclarableOpsTests1, OneHotTests_2) {
    auto indices = NDArrayFactory::create<float>('c', { 2, 2 }, { 0.f, 2.f, 1.f, -1.f });

    auto exp = NDArrayFactory::create<float>('c', { 2, 2, 3 }, { 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f });

    sd::ops::onehot op;
    auto result = op.evaluate({ &indices }, { 1.0f, 0.0f }, { -1, 3 });

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));


}

TEST_F(DeclarableOpsTests1, OneHotTests_3) {
    auto indices = NDArrayFactory::create<float>('c', { 4 }, { 0.0f, 2.0f, -1.0f, 1.0f });

    auto exp = NDArrayFactory::create<float>('c', { 4, 3 }, { 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f });

    sd::ops::onehot op;

    auto result = op.evaluate({ &indices }, { 1.0f, 0.0f }, { -1, 3 });
    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    // z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


}

TEST_F(DeclarableOpsTests1, OneHotTests_4) {
    auto indices = NDArrayFactory::create<float>('c', { 4 }, { 0.0f, 2.0f, -1.0f, 1.0f });
    auto depth = NDArrayFactory::create<float>(3.0f);

    auto exp = NDArrayFactory::create<float>('c', { 4, 3 }, { 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f });

    sd::ops::onehot op;

    auto result = op.evaluate({ &indices, &depth }, { 1.0f, 0.0f }, {});
    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


}

TEST_F(DeclarableOpsTests1, OneHotTests_5) {
    auto indices = NDArrayFactory::create<float>('c', { 4 }, { 0.0f, 2.0f, -1.0f, 1.0f });
    auto depth = NDArrayFactory::create<float>(3.0f);
    auto on = NDArrayFactory::create<float>(1.0f);
    auto off = NDArrayFactory::create<float>(0.0f);

    auto exp = NDArrayFactory::create<float>('c', { 4, 3 }, { 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f });

    sd::ops::onehot op;

    auto result = op.evaluate({ &indices, &depth, &on, &off }, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


}

TEST_F(DeclarableOpsTests1, OneHotTests_6) {
    auto indices = NDArrayFactory::create<float>('c', { 3 }, { 0.f, 1.f, 2.f });
    auto e = NDArrayFactory::create<float>('c', { 3, 3 }, { 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f });

    sd::ops::onehot op;
    auto result = op.evaluate({ &indices }, { 1.0, 0.0 }, { 0, 3 });
    auto z = result.at(0);

    ASSERT_EQ(e, *z);


}

TEST_F(DeclarableOpsTests1, OneHotTests_7) {
    auto indices = NDArrayFactory::create<int>('c', { 3 }, { 0, 1, 2 });
    auto e = NDArrayFactory::create<float16>('c', { 3, 3 }, { 1., 0., 0., 0., 1., 0., 0., 0., 1. });

    sd::ops::onehot op;
    auto result = op.evaluate({ &indices }, { 1.0, 0.0 }, { 0, 3 }, {}, { sd::DataType::HALF }, false);
    auto z = result.at(0);

    ASSERT_EQ(e, *z);


}

TEST_F(DeclarableOpsTests1, FillAs_1) {
    auto x = NDArrayFactory::create<float>('c', { 2, 2 });
    x.assign(117);

    float scalar = 119.f;

    sd::ops::fill_as op;
    auto result = op.evaluate({ &x }, { scalar }, {});

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    ASSERT_TRUE(x.isSameShape(result.at(0)));

    ASSERT_NEAR(scalar, result.at(0)->meanNumber().e<float>(0), 1e-5f);


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, LRN1) {
    sd::ops::lrn lrn;

    lrn.getOpName();
}

TEST_F(DeclarableOpsTests1, Test_Range_Integer_1) {
    auto exp = NDArrayFactory::create<int>('c', { 4 });
    exp.linspace(1);

    sd::ops::range op;

    auto result = op.evaluate({}, {}, { 1, 5, 1 });
    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    ASSERT_EQ(1, result.size());

    auto array = result.at(0);
    // array->printIndexedBuffer("Range integer 1");
    ASSERT_TRUE(exp.isSameShape(array));
    ASSERT_TRUE(exp.equalsTo(array));


}


TEST_F(DeclarableOpsTests1, Test_Range_Integer_2) {
    auto exp = NDArrayFactory::create<float>('c', { 4 });
    exp.linspace(1);

    auto start = NDArrayFactory::create<float>('c', { 1, 1 });
    auto stop = NDArrayFactory::create<float>('c', { 1, 1 });
    auto step = NDArrayFactory::create<float>('c', { 1, 1 });
    start.p(0, 1.f);
    stop.p(0, 5.f);
    step.p(0, 1.f);

    sd::ops::range op;

    auto result = op.evaluate({ &start, &stop, &step }, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    ASSERT_EQ(1, result.size());

    auto array = result.at(0);

    ASSERT_TRUE(exp.isSameShape(array));
    ASSERT_TRUE(exp.equalsTo(array));


}


TEST_F(DeclarableOpsTests1, Test_Range_Integer_3) {
    auto exp = NDArrayFactory::create<float>('c', { 4 });
    exp.linspace(1);

    sd::ops::range op;

    auto result = op.evaluate({}, { 1.f, 5.f, 1.f }, {});
    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    ASSERT_EQ(1, result.size());

    auto array = result.at(0);

    ASSERT_TRUE(exp.isSameShape(array));
    ASSERT_TRUE(exp.equalsTo(array));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test1) {

    NDArray input('c', { 3, 3 }, { -1.f, 1.f, -2.f, 2.f, -3.f, 3.f, -4.f, 4.f, 5.f }, sd::DataType::FLOAT32);

    NDArray expOutput('c', { 3, 3 }, { 1.14195199e-01, 8.43794734e-01, 4.20100661e-02, 2.68454951e-01, 1.80883523e-03, 7.29736214e-01, 9.02116571e-05, 2.68917160e-01, 7.30992629e-01 }, sd::DataType::FLOAT32);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, {}, {});
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test2) {
    NDArray input('c', { 3, 3, 3 }, { -1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14 }, sd::DataType::FLOAT32);
    NDArray expOutput('c', { 3, 3, 3 }, { 4.73142e-02,4.73847e-02,6.69062e-03, 9.50330e-01,8.67881e-04,9.92976e-01, 2.35563e-03,9.51747e-01,3.33106e-04, 4.74259e-02,2.26032e-06,4.74259e-02, 2.91395e-07,9.99998e-01,3.94360e-08, 9.52574e-01,1.12535e-07,9.52574e-01, 7.58256e-10,4.74259e-02,1.22325e-11, 1.00000e+00,1.32293e-11,1.19203e-01, 3.77513e-11,9.52574e-01,8.80797e-01 }, sd::DataType::FLOAT32);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, { 1 }, {});
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test3) {
    NDArray input('c', { 3, 3, 3 }, { -1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14 }, sd::DataType::FLOAT32);
    NDArray expOutput('c', { 3, 3, 3 }, { 2.47262e-03,1.23395e-04,3.35350e-04, 1.23395e-04,4.53979e-05,1.23395e-04, 6.14417e-06,1.23395e-04,5.56530e-09, 9.97527e-01,1.12521e-07,9.99665e-01, 1.52281e-08,9.99955e-01,2.06090e-09, 9.99994e-01,2.78912e-10,6.69285e-03, 3.05146e-07,9.99876e-01,4.13855e-08, 9.99877e-01,5.60254e-09,9.99877e-01, 7.58251e-10,9.99877e-01,9.93307e-01 }, sd::DataType::FLOAT32);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, { 0 }, {});
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test4) {
    NDArray input('c', { 1, 5 }, { -1, 1, -2, 2, 3 }, sd::DataType::FLOAT32);
    NDArray expOutput('c', { 1, 5 }, { 0.01198,0.08855,0.00441,0.24072,0.65434 }, sd::DataType::FLOAT32);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, { 1 }, {});
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test5) {
    NDArray input('c', { 1, 5 }, { -1, 1, -2, 2, 3 }, sd::DataType::FLOAT32);
    NDArray expOutput('c', { 1, 5 }, { 1,1,1,1,1 }, sd::DataType::FLOAT32);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, { 0 });
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test6) {
    NDArray input('c', { 5, 1 }, { -1, 1, -2, 2, 3 }, sd::DataType::FLOAT32);
    NDArray expOutput('c', { 5, 1 }, { 0.01198,0.08855,0.00441,0.24072,0.65434 }, sd::DataType::FLOAT32);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, { 0 }, {});
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test7) {
    NDArray input('c', { 5, 1 }, { -1, 1, -2, 2, 3 }, sd::DataType::FLOAT32);
    NDArray expOutput('c', { 5, 1 }, { 1,1,1,1,1 }, sd::DataType::FLOAT32);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, { 1 }, {});
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test8) {
    NDArray input('c', { 5 }, { -1, 1, -2, 2, 3 }, sd::DataType::FLOAT32);
    NDArray expOutput('c', { 5 }, { 0.01198,0.08855,0.00441,0.24072,0.65434 }, sd::DataType::FLOAT32);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, {}, {});
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test9) {
    NDArray input('c', { 2, 2, 2, 2 }, { -1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8 }, sd::DataType::FLOAT32);
    NDArray expOutput('c', { 2, 2, 2, 2 }, { 0.731059, 0.268941, 0.268941, 0.731059, 0.731059, 0.268941, 0.268941, 0.731059, 0.731059, 0.268941, 0.268941, 0.731059, 0.731059, 0.268941, 0.268941, 0.731059 }, sd::DataType::FLOAT32);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, { 2 }, {});
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test10) {
    NDArray input('c', { 2, 2, 2, 2, 2 }, { -1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14, -14, 15, -15, 16,-16 }, sd::DataType::FLOAT32);
    NDArray expOutput('c', { 2, 2, 2, 2, 2 }, { 0.119203, 0.880797, 0.017986, 0.982014, 0.002473, 0.997527, 0.000335, 0.999665, 0.000045, 0.999955, 0.000006, 0.999994, 0.000001, 0.999999, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.00000 }, sd::DataType::FLOAT32);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, { 4 }, {});
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test11) {
    NDArray input('c', { 2, 2, 2, 2, 2, 2 }, { -1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14, -14, 15, -15, 16,-16, -2.1, 2.1, -2.2, 2.2, -2.3, 2.3, -2.4, 2.4, -2.5,2.5 ,-2.6,2.6, -2.7,2.7, -2.8,2.8, -2.9,2.9, -3.0,3.0, -3.1,3.1, -3.2,3.2, -3.3,3.3, 3.4, -3.4, 3.5, -3.5, 3.6,-3.6 }, sd::DataType::FLOAT32);
    NDArray expOutput('c', { 2, 2, 2, 2, 2, 2 }, { 0.731059, 0.268941, 0.268941, 0.731059, 0.731059, 0.268941, 0.268941, 0.731059, 0.731059, 0.268941, 0.268941, 0.731059, 0.731059, 0.268941, 0.268941, 0.731059, 0.731059, 0.268941, 0.268941, 0.731059, 0.731059, 0.268941, 0.268941, 0.731059, 0.000000, 1.000000, 1.000000, 0.000000, 0.268941, 0.731059, 0.731059, 0.268941, 0.524979, 0.475021, 0.475021, 0.524979, 0.524979, 0.475021, 0.475021, 0.524979, 0.524979, 0.475021, 0.475021, 0.524979, 0.524979, 0.475021, 0.475021, 0.524979, 0.524979, 0.475021, 0.475021, 0.524979, 0.524979, 0.475021, 0.475021, 0.524979, 0.001229, 0.998771, 0.998771, 0.001229, 0.475021, 0.524979, 0.524979, 0.475021 }, sd::DataType::FLOAT32);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, { 4 }, {});
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, softmax_test12) {
    NDArray input('f', { 2, 2, 2, 2, 2, 2 }, { -1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14, -14, 15, -15, 16,-16, -2.1, 2.1, -2.2, 2.2, -2.3, 2.3, -2.4, 2.4, -2.5,2.5 ,-2.6,2.6, -2.7,2.7, -2.8,2.8, -2.9,2.9, -3.0,3.0, -3.1,3.1, -3.2,3.2, -3.3,3.3, 3.4, -3.4, 3.5, -3.5, 3.6,-3.6 }, sd::DataType::FLOAT32);
    NDArray exp('c', { 2, 2, 2, 2, 2, 2 }, { 0.982014, 0.598688, 0.982014, 0.598688, 0.017986, 0.401312, 0.017986, 0.401312, 0.982014, 0.598688, 0.000000, 0.001359, 0.017986, 0.401312, 1.000000, 0.998641, 0.982014, 0.598688, 0.000000, 0.001659, 0.017986, 0.401312, 1.000000, 0.998341, 0.982014, 0.598688, 0.000000, 0.001113, 0.017986, 0.401312, 1.000000, 0.998887, 0.017986, 0.401312, 0.017986, 0.401312, 0.982014, 0.598688, 0.982014, 0.598688, 0.017986, 0.401312, 1.000000, 0.998641, 0.982014, 0.598688, 0.000000, 0.001359, 0.017986, 0.401312, 1.000000, 0.998341, 0.982014, 0.598688, 0.000000, 0.001659, 0.017986, 0.401312, 1.000000, 0.998887, 0.982014, 0.598688, 0.000000, 0.001113 }, sd::DataType::FLOAT32);

    auto expOutput = NDArray('f', { 2, 2, 2, 2, 2, 2 }, sd::DataType::FLOAT32);
    expOutput.assign(exp);

    sd::ops::softmax op;
    auto results = op.evaluate({ &input }, {}, { 3 }, {});
    auto z = results.at(0);

    ASSERT_EQ(Status::OK(), results.status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));


}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_1) {

    float inBuff[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    float expBuff[] = { 24., 23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1. };
    Nd4jLong shapeInfo[] = { 3, 2, 3, 4, 12, 4, 1, 0, 1, 99 };
    ArrayOptions::setDataType(shapeInfo, sd::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, { 0,1,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_2) {

    float inBuff[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    float expBuff[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    Nd4jLong shapeInfo[] = { 3, 2, 3, 4, 12, 4, 1, 0, 1, 99 };
    ArrayOptions::setDataType(shapeInfo, sd::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, {}, {}, {}, true);

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(input));
    ASSERT_TRUE(expected.equalsTo(&input));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_3) {

    float inBuff[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    float expBuff[] = { 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 24., 23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13. };
    Nd4jLong shapeInfo[] = { 3, 2, 3, 4, 12, 4, 1, 0, 1, 99 };
    ArrayOptions::setDataType(shapeInfo, sd::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, { 1,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_4) {

    float inBuff[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    float expBuff[] = { 16,15,14,13,20,19,18,17,24,23,22,21,4,3,2,1,8,7,6,5,12,11,10,9, };
    Nd4jLong shapeInfo[] = { 3, 2, 3, 4, 12, 4, 1, 0, 1, 99 };
    ArrayOptions::setDataType(shapeInfo, sd::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, { 0,2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_5) {

    float inBuff[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    float expBuff[] = { 21., 22., 23., 24., 17., 18., 19., 20., 13., 14., 15., 16., 9., 10., 11., 12., 5., 6., 7., 8., 1., 2., 3., 4. };
    Nd4jLong shapeInfo[] = { 3, 2, 3, 4, 12, 4, 1, 0, 1, 99 };
    ArrayOptions::setDataType(shapeInfo, sd::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, { 0,1 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));


}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_6) {

    float inBuff[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    float expBuff[] = { 4., 3., 2., 1., 8., 7., 6., 5., 12., 11., 10., 9., 16., 15., 14., 13., 20., 19., 18., 17., 24., 23., 22., 21. };
    Nd4jLong shapeInfo[] = { 3, 2, 3, 4, 12, 4, 1, 0, 1, 99 };
    ArrayOptions::setDataType(shapeInfo, sd::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, { 2 }, {}, {}, true);

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(input));
    ASSERT_TRUE(expected.equalsTo(&input));


}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_7) {

    float inBuff[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    float expBuff[] = { 9., 10., 11., 12., 5., 6., 7., 8., 1., 2., 3., 4., 21., 22., 23., 24., 17., 18., 19., 20., 13., 14., 15., 16. };
    Nd4jLong shapeInfo[] = { 3, 2, 3, 4, 12, 4, 1, 0, 1, 99 };
    ArrayOptions::setDataType(shapeInfo, sd::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, { 1 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);
    //expected.printIndexedBuffer("E");
    //result->printIndexedBuffer("R");

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));


}



//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_8) {

    float inBuff[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    float expBuff[] = { 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 24., 23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13. };
    Nd4jLong shapeInfo[] = { 3, 2, 3, 4, 12, 4, 1, 0, 1, 99 };
    ArrayOptions::setDataType(shapeInfo, sd::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, { 2,1 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));


}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_9) {

    float inBuff[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
    float expBuff[] = { 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12. };
    Nd4jLong shapeInfo[] = { 3, 2, 3, 4, 12, 4, 1, 0, 1, 99 };
    ArrayOptions::setDataType(shapeInfo, sd::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, { 0 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));


}

TEST_F(DeclarableOpsTests1, Reverse_10) {
    auto x = NDArrayFactory::create<double>('c', { 4, 3 }, { 1.5375735, 0.1592365, 0.09966054, 0.677872, 1.144433, -1.0355669, 0.48456487, -0.67863184, 0.85020787, 0.13950661, 0.20998026, -1.1660044 });
    auto i = NDArrayFactory::create<int>('c', { 1 }, { -1 });
    auto e = NDArrayFactory::create<double>('c', { 4, 3 }, { 0.09966054, 0.1592365, 1.5375735,  -1.0355669, 1.144433, 0.677872,0.85020787, -0.67863184, 0.48456487,  -1.1660044, 0.20998026, 0.13950661 });

    sd::ops::reverse op;
    auto result = op.evaluate({ &x, &i }, {}, {}, {});

    auto z = result.at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_11) {


    auto input = NDArrayFactory::create<float>('c', { 2,3,4 });
    auto expected = NDArrayFactory::create<float>('c', { 2,3,4 }, { 24.f, 23.f, 22.f, 21.f, 20.f, 19.f, 18.f, 17.f, 16.f,
                                                                 15.f, 14.f, 13.f, 12.f, 11.f, 10.f,  9.f,  8.f,  7.f,
                                                                  6.f,  5.f,  4.f,  3.f,  2.f,  1.f });

    input.linspace(1);
    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, { 0, 1, 2 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_12) {


    auto input = NDArrayFactory::create<float>({ 0.f, 1.f, 2.f, 3.f, 4.f });
    auto expected = NDArrayFactory::create<float>({ 4.f, 3.f, 2.f, 1.f, 0.f });

    //input.linspace(1);
    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, { 0 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);
    //result->printIndexedBuffer("Result reverse");
    //expected.printIndexedBuffer("Expected reverse");
    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_13) {


    auto input = NDArrayFactory::create<float>({ 0.f, 1.f, 2.f, 3.f, 4.f });
    auto expected = NDArrayFactory::create<float>({ 4.f, 3.f, 2.f, 1.f, 0.f });

    //input.linspace(1);
    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, { -1 });

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));


}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests1, Reverse_14) {


    auto input = NDArrayFactory::create<double>({ 0.f, 1.f, 2.f, 3.f, 4.f });
    auto expected = NDArrayFactory::create<double>({ 0.f, 1.f, 2.f, 3.f, 4.f });

    //input.linspace(1);
    sd::ops::reverse op;
    auto results = op.evaluate({ &input }, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results.status());

    auto result = results.at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));


}

TEST_F(DeclarableOpsTests1, Test_Expose_1) {
    auto input0 = NDArrayFactory::create<float>('c', { 2, 3 }, { 1, 2, 3, 6, 5, 4 });
    auto input1 = NDArrayFactory::create<float>('c', { 2, 3 }, { 3, 2, 1, 4, 5, 6 });

    sd::ops::expose op;

    auto result = op.evaluate({ &input0, &input1 });

    ASSERT_EQ(ND4J_STATUS_OK, result.status());

    auto z0 = result.at(0);
    auto z1 = result.at(1);

    ASSERT_TRUE(input0.equalsTo(z0));
    ASSERT_TRUE(input1.equalsTo(z1));


}

TEST_F(DeclarableOpsTests1, Test_Expose_2) {
    auto list = new NDArrayList(0, true);

    auto var = new Variable(nullptr, "arraylist", -1, 0);
    var->setNDArrayList(list);

    VariableSpace variableSpace;
    variableSpace.putVariable(-1, var);
    variableSpace.trackList(list);

    Context block(1, &variableSpace);
    block.pickInput(-1);

    sd::ops::expose op;
    auto result = op.execute(&block);

    ASSERT_EQ(ND4J_STATUS_OK, result);
    ASSERT_TRUE(variableSpace.hasVariable(1));

    auto var1 = variableSpace.getVariable(1);

    ASSERT_EQ(var->variableType(), var1->variableType());

    auto list1 = var1->getNDArrayList();

    ASSERT_TRUE(list == list1);

}

TEST_F(DeclarableOpsTests1, Test_Release) {
    auto x = NDArrayFactory::create<float>('c', { 8, 8 });
    // x.printShapeInfo("x shape");
}
