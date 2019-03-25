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
#include <Context.h>
#include <iomanip>
#include <Variable.h>
#include <VariableSpace.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>
#include <NativeOps.h>
#include <ops/gemm.h>

using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests16 : public testing::Test {
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
    const int oH = (iH - kH - (kH-1)*(dH-1) + 2*pH)/sH + 1;     // output height
    const int oW = (iW - kW - (kW-1)*(dW-1) + 2*pW)/sW + 1;     // output width

    DeclarableOpsTests16() {
        printf("\n");
    }
};

template <typename T>
class TypedDeclarableOpsTests16 : public testing::Test {
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
    const int oH = (iH - kH - (kH-1)*(dH-1) + 2*pH)/sH + 1;     // output height
    const int oW = (iW - kW - (kW-1)*(dW-1) + 2*pW)/sW + 1;     // output width

    TypedDeclarableOpsTests16() {
        printf("\n");
    }
};

typedef ::testing::Types<double, float> TestingTypes;
TYPED_TEST_CASE(TypedDeclarableOpsTests16, TestingTypes);


//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests16, Maxpool2d_bp2) {

    int bS=2, iD=1, iH=4,iW=4, oD=3, kH=2,kW=2, sH=1,sW=1, pH=0,pW=0, dH=1,dW=1;
    int oH = (iH - kH - (kH-1)*(dH-1) + 2*pH)/sH + 1;
    int oW = (iW - kW - (kW-1)*(dW-1) + 2*pW)/sW + 1;

    TypeParam epsilonBuff[]  = {6., 7., 8., 10., 11., 12., 14., 15., 16., 22., 23., 24., 26., 27., 28., 30., 31., 32.};
    TypeParam expectedBuff[] = {0., 0., 0., 0.,0., 6., 7., 8.,0.,10.,11.,12.,0.,14.,15.,16.,0., 0., 0., 0.,0.,22.,23.,24.,0.,26.,27.,28.,0.,30.,31.,32.};

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS,iD,iH,iW});
    auto epsilon  = NDArrayFactory::create<TypeParam>('c', {bS,iD,oH,oW});
    auto expected = NDArrayFactory::create<TypeParam>('c', {bS,iD,iH,iW});


    input.linspace(1.);
    epsilon.setBuffer(epsilonBuff);
    expected.setBuffer(expectedBuff);

    std::initializer_list<Nd4jLong> argI = {kH,kW, sH,sW, pH,pW, dW,dH, 0, 0, 0};   // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;

    nd4j::ops::maxpool2d_bp op;
    auto results = op.execute({&input, &epsilon}, {}, argI);
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests16, Avgpool2d_bp2) {

    int bS=2, iD=1, iH=4,iW=4, oD=3, kH=2,kW=2, sH=1,sW=1, pH=0,pW=0, dH=1,dW=1;
    int oH = (iH - kH - (kH-1)*(dH-1) + 2*pH)/sH + 1;
    int oW = (iW - kW - (kW-1)*(dW-1) + 2*pW)/sW + 1;

    TypeParam epsilonBuff[] = {3.5 , 4.5 , 5.5, 7.5 , 8.5 , 9.5, 11.5, 12.5, 13.5, 19.5, 20.5, 21.5, 23.5, 24.5, 25.5, 27.5, 28.5, 29.5};
    TypeParam expectedBuff[] = {0.875, 2., 2.5,  1.375, 2.75 , 6., 7.,  3.75, 4.75 ,10., 11., 5.75, 2.875, 6., 6.5, 3.375, 4.875, 10.,10.5, 5.375, 10.75, 22.,23., 11.75, 12.75, 26.,27., 13.75, 6.875, 14.,14.5, 7.375};

    auto input    = NDArrayFactory::create<TypeParam>('c', {bS,iD,iH,iW});
    auto epsilon  = NDArrayFactory::create<TypeParam>('c', {bS,iD,oH,oW});
    auto expected = NDArrayFactory::create<TypeParam>('c', {bS,iD,iH,iW});


    input.linspace(1.);
    epsilon.setBuffer(epsilonBuff);
    expected.setBuffer(expectedBuff);

    std::initializer_list<Nd4jLong> argI = {kH,kW, sH,sW, pH,pW, dW,dH, 1, 1, 0};

    nd4j::ops::avgpool2d_bp op;
    auto results = op.execute({&input, &epsilon}, {}, argI);
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

TEST_F(DeclarableOpsTests16, ArgMax1) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});
    x.linspace(1);
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {3});
    exp.assign(4);

    nd4j::ops::argmax op;

    auto result = op.execute({&x}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests16, ArgMax2) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});
    x.linspace(1);
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {5});
    exp.assign(2);

    nd4j::ops::argmax op;

    auto result = op.execute({&x}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests16, ArgMax3) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});
    auto dim = NDArrayFactory::create<float>('c', {1, 1}, {0.});
    x.linspace(1);
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {5});
    exp.assign(2);

    nd4j::ops::argmax op;

    auto result = op.execute({&x, &dim}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests16, ArgMax4) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});
    auto dim = NDArrayFactory::create<float>('c', {1, 1}, {1});
    x.linspace(1);
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {3});
    exp.assign(4);

    nd4j::ops::argmax op;

    auto result = op.execute({&x, &dim}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests16, ArgMax5) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});
    auto dim = NDArrayFactory::create<float>('c', {1, 2}, {0, 1});
    x.linspace(1);
    auto exp = NDArrayFactory::create<Nd4jLong>(14);


    nd4j::ops::argmax op;

    auto result = op.execute({&x, &dim}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests16, ArgMax6) {
    auto x = NDArrayFactory::create<float>('c', {3, 4, 5});
    auto dim = NDArrayFactory::create<float>(-1.f);
    x.linspace(1);


    nd4j::ops::argmax op;

    auto expected = op.execute({&x}, {}, {2});
    ASSERT_EQ(Status::OK(), expected->status());
    auto exp = expected->at(0);


    auto result = op.execute({&x, &dim}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(*exp, *z);

    delete result;
    delete expected;
}


TEST_F(DeclarableOpsTests16, ArgMin1) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});
    x.linspace(1);
//    auto exp('c', {3, 1});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {3});
    exp.assign(0.0f);

    nd4j::ops::argmin op;

    auto result = op.execute({&x}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests16, SquareTests1) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});
    x.linspace(1);

    auto exp = NDArrayFactory::create<float>('c', {3, 5});
    exp.linspace(1);
    exp *= exp;

    nd4j::ops::square op;

    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests16, OneHotTests_1) {
    auto indices = NDArrayFactory::create<float>('c', {1, 4}, {0.0f, 2.0f, -1.0f, 1.0f});

    auto exp = NDArrayFactory::create<float>('c', {4, 3}, {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f});

    nd4j::ops::onehot op;

    auto result = op.execute({&indices}, {1.0f, 0.0f}, {-1, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests16, OneHotTests_2) {
    auto indices = NDArrayFactory::create<float>('c', {2, 2}, {0.f, 2.f, 1.f, -1.f});

    auto exp = NDArrayFactory::create<float>('c', {2, 2, 3}, {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f});

    nd4j::ops::onehot op;
    auto result = op.execute({&indices}, {1.0f, 0.0f}, {-1, 3});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests16, OneHotTests_3) {
    auto indices = NDArrayFactory::create<float>('c', {4}, {0.0f, 2.0f, -1.0f, 1.0f});

    auto exp = NDArrayFactory::create<float>('c', {4, 3}, {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f});

    nd4j::ops::onehot op;

    auto result = op.execute({&indices}, {1.0f, 0.0f}, {-1, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests16, OneHotTests_4) {
    auto indices = NDArrayFactory::create<float>('c', {4}, {0.0f, 2.0f, -1.0f, 1.0f});
    auto depth = NDArrayFactory::create<float>(3.0f);

    auto exp = NDArrayFactory::create<float>('c', {4, 3}, {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f});

    nd4j::ops::onehot op;

    auto result = op.execute({&indices, &depth}, {1.0f, 0.0f}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests16, OneHotTests_5) {
    auto indices = NDArrayFactory::create<float>('c', {4}, {0.0f, 2.0f, -1.0f, 1.0f});
    auto depth = NDArrayFactory::create<float>(3.0f);
    auto on = NDArrayFactory::create<float>(1.0f);
    auto off = NDArrayFactory::create<float>(0.0f);

    auto exp = NDArrayFactory::create<float>('c', {4, 3}, {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f});

    nd4j::ops::onehot op;

    auto result = op.execute({&indices, &depth, &on, &off}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests16, FillAs_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2});
    x.assign(117);

    float scalar = 119.f;

    nd4j::ops::fill_as op;
    auto result = op.execute({&x}, {scalar}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(x.isSameShape(result->at(0)));

    ASSERT_NEAR(scalar, result->at(0)->meanNumber().e<float>(0), 1e-5f);

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, LRN1) {
    nd4j::ops::lrn lrn;

    lrn.getOpName();
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Stack_1) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    Nd4jLong shape1[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.execute({&input1, &input2}, {}, {0});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Stack_2) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1,  2,  3,  4, 13, 14, 16, 16, 5,  6,  7,  8, 17, 18, 19, 20, 9, 10, 11, 12, 21, 22, 23, 24};
    Nd4jLong shape1[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 3, 2, 4, 8, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.execute({&input1, &input2}, {}, {1});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Stack_3) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    Nd4jLong shape1[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 2, 1, 12, 12, 12, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.execute({&input1, &input2}, {}, {0});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Stack_4) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    Nd4jLong shape1[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 1, 2, 12, 24, 12, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.execute({&input1, &input2}, {}, {1});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Stack_5) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    Nd4jLong shape1[]    = {2, 12, 1, 1,  1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 12, 1, 1,  1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 2, 12, 1, 12, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.execute({&input1, &input2}, {}, {0});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Stack_6) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1 ,13 ,2 ,14 ,3 ,16 ,4 ,16 ,5 ,17 ,6 ,18 ,7 ,19 ,8 ,20 ,9 ,21 ,10 ,22 ,11 ,23 ,12 ,24};
    Nd4jLong shape1[]    = {2, 12, 1, 1, 12, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 12, 1, 1, 12, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 12, 2, 1, 2, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.execute({&input1, &input2}, {}, {1});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Stack_7) {

    float buff1[]   = {1};
    float expBuff[] = {1, 1, 1};
    Nd4jLong shape1[]    = {2, 1, 1, 1, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 3, 1, 1, 1, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.execute({&input1, &input1, &input1}, {}, {0});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Stack_8) {

    float buff1[]   = {1};
    float expBuff[] = {1, 1, 1};
    Nd4jLong shape1[]    = {1, 1, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {2, 3, 1, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.execute({&input1, &input1, &input1}, {}, {0});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Stack_9) {

    float buff1[]   = {1};
    float expBuff[] = {1, 1, 1};
    Nd4jLong shape1[]    = {2, 1, 1, 1, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 1, 3, 1, 3, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.execute({&input1, &input1, &input1}, {}, {1});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Stack_10) {

    float buff1[]   = {1};
    float expBuff[] = {1, 1, 1};
    Nd4jLong shape1[]    = {1, 1, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {2, 1, 3, 3, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.execute({&input1, &input1, &input1}, {}, {1});
    auto output = results->at(0);

    //expected.printShapeInfo("exp");
    //output->printShapeInfo("out");

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

TEST_F(DeclarableOpsTests16, Stack_11) {

    float buff1[]   = {1};
    float expBuff[] = {1, 1, 1};
    Nd4jLong shape1[]    = {1, 1, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {2, 3, 1, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, nd4j::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, nd4j::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray expected(expBuff, expShape);

    nd4j::ops::stack op;
    auto results = op.execute({&input1, &input1, &input1}, {}, {});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


TEST_F(DeclarableOpsTests16, Test_Range_Integer_1) {
    auto exp = NDArrayFactory::create<int>('c', {4});
    exp.linspace(1);

    nd4j::ops::range op;

    auto result = op.execute({}, {}, {1, 5, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(1, result->size());

    auto array = result->at(0);
    array->printIndexedBuffer("Range integer 1");
    ASSERT_TRUE(exp.isSameShape(array));
    ASSERT_TRUE(exp.equalsTo(array));

    delete result;
}


TEST_F(DeclarableOpsTests16, Test_Range_Integer_2) {
    auto exp = NDArrayFactory::create<float>('c', {4});
    exp.linspace(1);

    auto start = NDArrayFactory::create<float>('c', {1, 1});
    auto stop = NDArrayFactory::create<float>('c', {1, 1});
    auto step = NDArrayFactory::create<float>('c', {1, 1});
    start.p(0, 1.f);
    stop.p(0, 5.f);
    step.p(0, 1.f);

    nd4j::ops::range op;

    auto result = op.execute({&start, &stop, &step}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(1, result->size());

    auto array = result->at(0);

    ASSERT_TRUE(exp.isSameShape(array));
    ASSERT_TRUE(exp.equalsTo(array));

    delete result;
}


TEST_F(DeclarableOpsTests16, Test_Range_Integer_3) {
    auto exp = NDArrayFactory::create<float>('c', {4});
    exp.linspace(1);

    nd4j::ops::range op;

    auto result = op.execute({}, {1.f, 5.f, 1.f}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(1, result->size());

    auto array = result->at(0);

    ASSERT_TRUE(exp.isSameShape(array));
    ASSERT_TRUE(exp.equalsTo(array));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, softmax_test1) {
    auto input = NDArrayFactory::create<double>('c', {3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, 5});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3}, {1.14195199e-01, 8.43794734e-01, 4.20100661e-02, 2.68454951e-01, 1.80883523e-03, 7.29736214e-01, 9.02116571e-05, 2.68917160e-01, 7.30992629e-01});

    nd4j::ops::softmax op;
    auto results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, softmax_test2) {
    auto input = NDArrayFactory::create<double>('c', {3, 3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3, 3}, {4.73142e-02,   4.73847e-02,   6.69062e-03, 9.50330e-01,   8.67881e-04,   9.92976e-01, 2.35563e-03,   9.51747e-01,   3.33106e-04, 4.74259e-02,   2.26032e-06,   4.74259e-02, 2.91395e-07,   9.99998e-01,   3.94360e-08, 9.52574e-01,   1.12535e-07,   9.52574e-01, 7.58256e-10,   4.74259e-02,   1.22325e-11, 1.00000e+00,   1.32293e-11,   1.19203e-01, 3.77513e-11,   9.52574e-01,   8.80797e-01});

    nd4j::ops::softmax op;
    auto results = op.execute({&input}, {}, {1}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, softmax_test3) {
    auto input = NDArrayFactory::create<double>('c', {3, 3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3, 3}, {2.47262e-03,   1.23395e-04,   3.35350e-04, 1.23395e-04,   4.53979e-05,   1.23395e-04, 6.14417e-06,   1.23395e-04,   5.56530e-09, 9.97527e-01,   1.12521e-07,   9.99665e-01, 1.52281e-08,   9.99955e-01,   2.06090e-09, 9.99994e-01,   2.78912e-10,   6.69285e-03, 3.05146e-07,   9.99876e-01,   4.13855e-08, 9.99877e-01,   5.60254e-09,   9.99877e-01, 7.58251e-10,   9.99877e-01,   9.93307e-01});

    nd4j::ops::softmax op;
    auto results = op.execute({&input}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, softmax_test4) {
    auto input = NDArrayFactory::create<double>('c', {1, 5}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {1, 5}, {0.01198,  0.08855,  0.00441,  0.24072,  0.65434});

    nd4j::ops::softmax op;
    auto results = op.execute({&input}, {}, {1}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, softmax_test5) {
    auto input = NDArrayFactory::create<double>('c', {1, 5}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {1, 5}, {1,  1,  1,  1,  1});

    nd4j::ops::softmax op;
    auto results = op.execute({&input}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, softmax_test6) {
    auto input = NDArrayFactory::create<double>('c', {5, 1}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {5, 1}, {0.01198,  0.08855,  0.00441,  0.24072,  0.65434});

    nd4j::ops::softmax op;
    auto results = op.execute({&input}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, softmax_test7) {
    auto input = NDArrayFactory::create<double>('c', {5, 1}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {5, 1}, {1,  1,  1,  1,  1});

    nd4j::ops::softmax op;
    auto results = op.execute({&input}, {}, {1}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, softmax_test8) {
    auto input = NDArrayFactory::create<double>('c', {5}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {5}, {0.01198,  0.08855,  0.00441,  0.24072,  0.65434});

    nd4j::ops::softmax op;
    auto results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Test_Stack_Edge_1) {
    float inBuff[]  = {1.0f, 2.0f, 3.0f};
    float expBuff[] = {1.0f, 2.0f, 3.0f};

    auto input = NDArrayFactory::create<float>(inBuff, 'c', {1, 3});

    auto exp = NDArrayFactory::create<float>(expBuff, 'c', {1, 1, 3});

    nd4j::ops::stack op;

    auto result = op.execute({&input}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Test_Stack_Edge_2) {
    float inBuff[]  = {1.0f, 2.0f, 3.0f};
    float expBuff[] = {1.0f, 2.0f, 3.0f};

    auto input = NDArrayFactory::create<float>(inBuff, 'c', {1, 1, 3});

    auto exp = NDArrayFactory::create<float>(expBuff, 'c', {1, 1, 1, 3});

    nd4j::ops::stack op;

    auto result = op.execute({&input}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Test_Stack_Edge_3) {
    float inBuff[]  = {1.0f, 2.0f, 3.0f};
    float expBuff[] = {1.0f, 2.0f, 3.0f};

    auto input = NDArrayFactory::create<float>(inBuff, 'c', {1, 3});

    auto exp = NDArrayFactory::create<float>(expBuff, 'c', {1, 1, 3});

    nd4j::ops::stack op;

    auto result = op.execute({&input}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printShapeInfo();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_1 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {24., 23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.};
    Nd4jLong shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shapeInfo, nd4j::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {0,1,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_2 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    Nd4jLong shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shapeInfo, nd4j::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {}, {}, true);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(&input));
    ASSERT_TRUE(expected.equalsTo(&input));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_3 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 24., 23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13.};
    Nd4jLong shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shapeInfo, nd4j::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {1,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_4 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {16,15,14,13,    20,19,18,17,       24,23,22,21,    4,3,2,1,    8,7,6,5,      12,11,10,9,};
    Nd4jLong shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shapeInfo, nd4j::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_5 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {21., 22., 23., 24., 17., 18., 19., 20., 13., 14., 15., 16., 9., 10., 11., 12., 5., 6., 7., 8., 1., 2., 3., 4.};
    Nd4jLong shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shapeInfo, nd4j::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {0,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_6 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {4., 3., 2., 1., 8., 7., 6., 5., 12., 11., 10., 9., 16., 15., 14., 13., 20., 19., 18., 17., 24., 23., 22., 21.};
    Nd4jLong shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shapeInfo, nd4j::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {2}, {}, true);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(&input));
    ASSERT_TRUE(expected.equalsTo(&input));

    delete results;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_7 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {9., 10., 11., 12., 5., 6., 7., 8., 1., 2., 3., 4., 21., 22., 23., 24., 17., 18., 19., 20., 13., 14., 15., 16.};
    Nd4jLong shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shapeInfo, nd4j::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}



//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_8 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 24., 23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13.};
    Nd4jLong shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shapeInfo, nd4j::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {2,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_9 ) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    Nd4jLong shapeInfo[] = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shapeInfo, nd4j::DataType::FLOAT32);

    NDArray input(inBuff, shapeInfo);
    NDArray expected(expBuff, shapeInfo);
    NDArray output(shapeInfo);

    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

TEST_F(DeclarableOpsTests16, Reverse_10 ) {
    auto x = NDArrayFactory::create<double>('c', {4, 3}, {1.5375735, 0.1592365, 0.09966054, 0.677872, 1.144433, -1.0355669, 0.48456487, -0.67863184, 0.85020787, 0.13950661, 0.20998026, -1.1660044});
    auto i = NDArrayFactory::create<int>('c', {1}, {-1});
    auto e = NDArrayFactory::create<double>('c', {4, 3}, {0.09966054, 0.1592365, 1.5375735,  -1.0355669, 1.144433, 0.677872,   0.85020787, -0.67863184, 0.48456487,  -1.1660044, 0.20998026, 0.13950661});

    nd4j::ops::reverse op;
    auto result = op.execute({&x, &i}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_11 ) {


    auto input = NDArrayFactory::create<float>('c', {2,3,4});
    auto expected = NDArrayFactory::create<float>('c', {2,3,4}, {24., 23., 22., 21., 20., 19., 18., 17., 16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.});

    input.linspace(1);
    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {0, 1, 2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_12 ) {


    auto input = NDArrayFactory::create<float>({0.f, 1.f, 2.f, 3.f, 4.f});
    auto expected = NDArrayFactory::create<float>({4.f, 3.f, 2.f, 1.f, 0.f});

    //input.linspace(1);
    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    //result->printIndexedBuffer("Result reverse");
    //expected.printIndexedBuffer("Expected reverse");
    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_13 ) {


    auto input = NDArrayFactory::create<float>({0.f, 1.f, 2.f, 3.f, 4.f});
    auto expected = NDArrayFactory::create<float>({4.f, 3.f, 2.f, 1.f, 0.f});

    //input.linspace(1);
    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {-1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests16, Reverse_14 ) {


    auto input = NDArrayFactory::create<double>({0.f, 1.f, 2.f, 3.f, 4.f});
    auto expected = NDArrayFactory::create<double>({0.f, 1.f, 2.f, 3.f, 4.f});

    //input.linspace(1);
    nd4j::ops::reverse op;
    auto results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
// CONSTANT mode 2D
TEST_F(DeclarableOpsTests16, Pad_1) {

    float inBuff[]  = {1,2,3,4,5,6};
    int padBuff[] = {1,1,2,2};
    float expBuff[] = {0,0,0,0,0,0,0, 0,0,1,2,3,0,0, 0,0,4,5,6,0,0, 0,0,0,0,0,0,0};

    auto input    = NDArrayFactory::create<float>(inBuff,  'c', {2,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {2,2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4,7});

    nd4j::ops::pad op;
    auto results = op.execute({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// REFLECT mode 2D
TEST_F(DeclarableOpsTests16, Pad_2) {

    float inBuff[]  = {1,2,3,4,5,6};
    int padBuff[] = {1,1,2,2};
    float expBuff[] = {6,5,4,5,6,5,4, 3,2,1,2,3,2,1, 6,5,4,5,6,5,4, 3,2,1,2,3,2,1};

    auto input    = NDArrayFactory::create<float>(inBuff,  'c', {2,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {2,2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4,7});

    nd4j::ops::pad op;
    auto results = op.execute({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// SYMMETRIC mode 2D
TEST_F(DeclarableOpsTests16, Pad_3) {

    float inBuff[]  = {1,2,3,4,5,6};
    int padBuff[] = {1,1,2,2};
    float expBuff[] = {2,1,1,2,3,3,2, 2,1,1,2,3,3,2, 5,4,4,5,6,6,5, 5,4,4,5,6,6,5};

    auto input    = NDArrayFactory::create<float>(inBuff,  'c', {2,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {2,2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4,7});

    nd4j::ops::pad op;
    auto results = op.execute({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// CONSTANT mode 3D
TEST_F(DeclarableOpsTests16, Pad_4) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    int padBuff[] = {1,1,2,2,2,2};
    float expBuff[] = {0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 1, 2, 3,0,0,0,0, 4, 5, 6,0,0,0,0, 7, 8, 9,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0,10,11,12,0,0,0,0,13,14,15,0,0,0,0,16,17,18,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0};

    auto input    = NDArrayFactory::create<float>(inBuff,  'c', {2,3,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {3,2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4,7,7});

    nd4j::ops::pad op;
    auto results = op.execute({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}



////////////////////////////////////////////////////////////////////
// REFLECT mode 3D
TEST_F(DeclarableOpsTests16, Pad_5) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    int padBuff[] = {1,1,2,2,2,2};
    float expBuff[] = {18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 15,14,13,14,15,14,13, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 15,14,13,14,15,14,13, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1};
    auto input    = NDArrayFactory::create<float>(inBuff,  'c', {2,3,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {3,2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4,7,7});

    nd4j::ops::pad op;
    auto results = op.execute({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// SYMMETRIC mode 3D
TEST_F(DeclarableOpsTests16, Pad_6) {

    float inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    int padBuff[] = {1,1,2,2,2,2};
    float expBuff[] = {5, 4, 4, 5, 6, 6, 5, 2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2, 5, 4, 4, 5, 6, 6, 5, 8, 7, 7, 8, 9, 9, 8, 8, 7, 7, 8, 9, 9, 8, 5, 4, 4, 5, 6, 6, 5, 5, 4, 4, 5, 6, 6, 5, 2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2, 5, 4, 4, 5, 6, 6, 5, 8, 7, 7, 8, 9, 9, 8, 8, 7, 7, 8, 9, 9, 8, 5, 4, 4, 5, 6, 6, 5, 14,13,13,14,15,15,14, 11,10,10,11,12,12,11, 11,10,10,11,12,12,11, 14,13,13,14,15,15,14, 17,16,16,17,18,18,17, 17,16,16,17,18,18,17, 14,13,13,14,15,15,14, 14,13,13,14,15,15,14, 11,10,10,11,12,12,11, 11,10,10,11,12,12,11, 14,13,13,14,15,15,14, 17,16,16,17,18,18,17, 17,16,16,17,18,18,17, 14,13,13,14,15,15,14};

    auto input    = NDArrayFactory::create<float>(inBuff,  'c', {2,3,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {3,2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4,7,7});

    nd4j::ops::pad op;
    auto results = op.execute({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
// CONSTANT mode 4D
TEST_F(DeclarableOpsTests16, Pad_7)
{

    float inBuff[] =  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    float expBuff[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 0, 0, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto input = NDArrayFactory::create<float>(inBuff, 'c', {2, 2, 2, 2});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {4, 2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad op;
    auto results = op.execute({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
// REFLECT mode 4D
TEST_F(DeclarableOpsTests16, Pad_8)
{

    float inBuff[] =  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    float expBuff[] = {16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1};
    auto input = NDArrayFactory::create<float>(inBuff, 'c', {2, 2, 2, 2});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {4, 2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad op;
    auto results = op.execute({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////
// SYMMETRIC mode 4D
TEST_F(DeclarableOpsTests16, Pad_9)
{

    float inBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    float expBuff[] = {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16};
    auto input = NDArrayFactory::create<float>(inBuff, 'c', {2, 2, 2, 2});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {4, 2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad op;
    auto results = op.execute({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

TEST_F(DeclarableOpsTests16, Test_Expose_1) {
    auto input0 = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 3, 6, 5, 4});
    auto input1 = NDArrayFactory::create<float>('c', {2, 3}, {3, 2, 1, 4, 5, 6});

    nd4j::ops::expose op;

    auto result = op.execute({&input0, &input1}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z0 = result->at(0);
    auto z1 = result->at(1);

    ASSERT_TRUE(input0.equalsTo(z0));
    ASSERT_TRUE(input1.equalsTo(z1));

    delete result;
}

TEST_F(DeclarableOpsTests16, Test_Expose_2) {
    auto list = new NDArrayList(0, true);

    auto var = new Variable(nullptr, "arraylist", -1, 0);
    var->setNDArrayList(list);

    VariableSpace variableSpace;
    variableSpace.putVariable(-1, var);
    variableSpace.trackList(list);

    Context block(1, &variableSpace);
    block.pickInput(-1);

    nd4j::ops::expose op;
    auto result = op.execute(&block);

    ASSERT_EQ(ND4J_STATUS_OK, result);
    ASSERT_TRUE(variableSpace.hasVariable(1));

    auto var1 = variableSpace.getVariable(1);

    ASSERT_EQ(var->variableType(), var1->variableType());

    auto list1 = var1->getNDArrayList();

    ASSERT_TRUE(list == list1);

}


