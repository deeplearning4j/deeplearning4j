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
// Created by raver119 on 09.02.18.
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests6 : public testing::Test {
public:

    DeclarableOpsTests6() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests6, Test_StridedSlice_Once_Again_1) {
    auto matrix = NDArrayFactory::create<double>('c', {5, 2});
    auto b = NDArrayFactory::create<double>('c', {1}, {0.});
    auto e = NDArrayFactory::create<double>('c', {1}, {1});
    auto s = NDArrayFactory::create<double>('c', {1}, {1});

    auto exp = NDArrayFactory::create<double>('c', {2}, {1.0f, 2.0f});

    matrix.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix, &b, &e, &s}, {}, {0, 0, 0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_StridedSlice_Once_Again_2) {
    auto matrix = NDArrayFactory::create<double>('c', {5, 2});
    auto b = NDArrayFactory::create<double>('c', {1}, {0.0f});
    auto e = NDArrayFactory::create<double>('c', {1}, {1.0f});
    auto s = NDArrayFactory::create<double>('c', {1}, {1.0f});

    auto exp = NDArrayFactory::create<double>('c', {2}, {1.0f, 2.0f});

    matrix.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix, &b, &e, &s}, {}, {0, 0, 0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(exp, *z);

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_StridedSlice_Once_Again_3) {
    auto matrix = NDArrayFactory::create<double>(10);
    auto b = NDArrayFactory::create<double>(0);
    auto e = NDArrayFactory::create<double>(0);
    auto s = NDArrayFactory::create<double>(1.0);

    //auto exp = NDArrayFactory::create<double>('c', {2}, {1.0f, 2.0f});

    //matrix.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix, &b, &e, &s}, {}, {0, 0, 0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    //z->printShapeInfo("SS OS shape");
    ASSERT_TRUE(z->isEmpty());
    //ASSERT_EQ(exp, *z);

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_StridedSlice_Once_Again_4) {
    auto matrix = NDArrayFactory::create<double>('c', {1}, {10});
    auto b = NDArrayFactory::create<double>('c', {1}, {0.});
    auto e = NDArrayFactory::create<double>('c', {1}, {0.});
    auto s = NDArrayFactory::create<double>('c', {1}, {1.0});

    auto exp = NDArrayFactory::create<double>(10);

    //matrix.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix, &b, &e, &s}, {}, {0, 0, 0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    z->printShapeInfo("SS OS shape");
    z->printIndexedBuffer("SS OS out");
    ASSERT_TRUE(z->equalsTo(exp));
    //ASSERT_EQ(exp, *z);

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_StridedSlice_Once_Again_04) {
    auto matrix = NDArrayFactory::create<double>('c', {1}, {10});
    auto b = NDArrayFactory::create_<int>('c', {1}, {1});
    auto e = NDArrayFactory::create_<int>('c', {1}, {(int)0});
    auto s = NDArrayFactory::create_<int>('c', {1}, {1});
    nd4j::ops::ones_as opOnes;
    //auto exp = NDArrayFactory::create<double>('c', {2}, {1.0f, 2.0f});
    auto onesRes = opOnes.execute({&matrix}, {}, {});
    //matrix.linspace(1);
    ASSERT_EQ(onesRes->status(), Status::OK());

    auto ones = onesRes->at(0);
    ones->printShapeInfo("Shape ones");
    *ones *= 10;
    auto onesD = ones->dup();

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, onesD);
    variableSpace->putVariable(-2, b);
    variableSpace->putVariable(-3, e);
    variableSpace->putVariable(-4, s);
    auto block = new Context(1, variableSpace, false);  // not-in-place
    block->fillInputs({-1});
    block->fillInputs({-2});
    block->fillInputs({-3});
    block->fillInputs({-4});
    block->getIArguments()->push_back(0);
    block->getIArguments()->push_back(0);
    block->getIArguments()->push_back(1);
    block->getIArguments()->push_back(0);
    block->getIArguments()->push_back(0);
    auto inputShapes = new ShapeList({ones->getShapeInfo(), b->getShapeInfo(), e->getShapeInfo(), s->getShapeInfo()});
    nd4j::ops::strided_slice op;
    auto result = op.calculateOutputShape(inputShapes, *block); //execute({ones, &b, &e, &s}, {}, {0, 1, 0, 0, 0});
    ASSERT_EQ(result->size(), 1);
    shape::printShapeInfoLinear(result->at(0));
    //auto z = result->at(0);
//    z->printShapeInfo("SS OS shape");
    ASSERT_TRUE(shape::isEmpty(result->at(0)));
    //ASSERT_EQ(exp, *z);
    delete block;

    delete onesRes;
    delete result;
    delete variableSpace;
    delete inputShapes;
}

TEST_F(DeclarableOpsTests6, Test_StridedSlice_Once_Again_5) {
    auto matrix = NDArrayFactory::create<double>('c', {3, 2, 2});
    auto b = NDArrayFactory::create<int>('c', {1}, {2});
    auto e = NDArrayFactory::create<int>('c', {1}, {3});
    auto s = NDArrayFactory::create<int>('c', {1}, {1});

    auto exp = NDArrayFactory::create<double>('c', {2,2}, {0.0f, 0.0f, 0., 0.});

    //matrix.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix, &b, &e, &s}, {}, {0, 0, 0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    z->printShapeInfo("Output shape");
    z->printIndexedBuffer("Output");
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_StridedSlice_Once_Again_6) {
    auto matrix = NDArrayFactory::create<double>('c', {3, 2, 2});
    auto b = NDArrayFactory::create<int>('c', {1}, {2});
    auto e = NDArrayFactory::create<int>('c', {1}, {3});
    auto s = NDArrayFactory::create<int>('c', {1}, {1});

    auto exp = NDArrayFactory::create<double>('c', {1,2,2}, {0.0f, 0.0f, 0., 0.});

    //matrix.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix, &b, &e, &s}, {}, {0, 0, 0, 0, 2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    z->printShapeInfo("Output shape");
    z->printIndexedBuffer("Output");
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_StridedSlice_Once_Again_7) {
    int zero = 0;
    auto matrix = NDArrayFactory::create<double>('c', {5, 4});
    auto b = NDArrayFactory::create<int>('c', {1}, {zero});
    auto e = NDArrayFactory::create<int>('c', {1}, {zero});
    auto s = NDArrayFactory::create<int>('c', {1}, {1});

    //auto exp = NDArrayFactory::create<double>('c', {1,2,2}, {0.0f, 0.0f, 0., 0.});

    //matrix.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix, &b, &e, &s}, {}, {1, 0, 0, 0, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    z->printShapeInfo("Output shape");
    z->printIndexedBuffer("Output");
    //ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Simple_Scalar_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 1}, {2.0f});
    auto exp = NDArrayFactory::create<double>('c', {1, 1}, {4.0f});

    nd4j::ops::test_scalar op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_gatherNd_Edge_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 4, 2, 2});
    auto indices = NDArrayFactory::create<int>('c', {3, 3}, {0,2,1, 0,1,0, 1,3,1});
    auto exp = NDArrayFactory::create<double>('c', {3,2}, {11.f, 12.f, 5.f, 6.f, 31.f, 32.f});
    x.linspace(1);

    nd4j::ops::gather_nd op;
    auto result = op.execute({&x, &indices}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer();
    //z->printShapeInfo("z shape");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Order_1) {
    auto x = NDArrayFactory::create<double>('f', {2, 3});
    auto exp = NDArrayFactory::create<double>('c', {2, 3});
    x.linspace(1);
    exp.linspace(1);

    nd4j::ops::order op;
    auto result = op.execute({&x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    z->printIndexedBuffer("O Output");
    exp.printIndexedBuffer("O Expect");
    ASSERT_TRUE(exp.equalsTo(z));
    ASSERT_NE(x.ordering(), z->ordering());

    delete result;
}

TEST_F(DeclarableOpsTests6, cumSum_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 4}, {1.f, 2.f, 3.f, 4.f});
    auto exp = NDArrayFactory::create<float>('c', {1, 4}, {1.f, 3.f, 6.f, 10.f});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    // z->printIndexedBuffer("CumSum1");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, cumSum_2) {
    auto x= NDArrayFactory::create<float>('c', {2, 4}, {1, 2, 3, 4, 1, 2, 3, 4});
    auto exp= NDArrayFactory::create<float>('c', {2, 4}, {1, 3, 6, 10, 1, 3, 6, 10});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 0, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printIndexedBuffer("CumSum1");
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, cumSum_3) {
    auto x= NDArrayFactory::create<float>('c', {2, 4}, {1, 2, 3, 4, 1, 2, 3, 4});
    auto exp= NDArrayFactory::create<float>('c', {2, 4}, {1, 2, 3, 4, 2, 4, 6, 8});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 0, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, cumSum_4) {
    auto x = NDArrayFactory::create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto exp = NDArrayFactory::create<double>('c', {3, 3}, {12., 15., 18., 11., 13., 15., 7., 8., 9.});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 1, 0}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printBuffer();

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, cumSum_5) {
    auto x = NDArrayFactory::create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto exp = NDArrayFactory::create<double>('c', {3, 3}, {6.f, 5.f, 3.f, 15.f, 11.f, 6.f, 24.f, 17.f, 9.f,});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 1, 1}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, cumSum_6) {
    auto x = NDArrayFactory::create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto exp = NDArrayFactory::create<double>('c', {3, 3}, {11.f, 13.f, 15.f, 7.f, 8.f, 9.f, 0.f, 0.f, 0.f});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {1, 1, 0}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, cumSum_7) {
    auto x = NDArrayFactory::create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto exp = NDArrayFactory::create<double>('c', {3, 3}, {5.f, 3.f, 0.f, 11.f, 6.f, 0.f, 17.f, 9.f, 0.f});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {1, 1, 1}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, cumSum_8) {
    auto x = NDArrayFactory::create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto axis = NDArrayFactory::create<Nd4jLong>('c', {1}, {1});
    auto exp = NDArrayFactory::create<double>('c', {3, 3}, {5.f, 3.f, 0.f, 11.f, 6.f, 0.f, 17.f, 9.f, 0.f});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x, &axis}, {}, {1, 1}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_9) {

    auto inputC = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
    auto axis = NDArrayFactory::create<Nd4jLong>(1);

    auto expFF = NDArrayFactory::create<double>('c', {3, 5}, {1.,  3.,  6., 10., 15., 6., 13., 21., 30., 40., 11., 23., 36., 50., 65.});
    auto expTF = NDArrayFactory::create<double>('c', {3, 5}, {0., 1., 3.,  6., 10., 0.,  6., 13., 21., 30., 0., 11., 23., 36., 50.});

    auto expFT = NDArrayFactory::create<double>('c', {3, 5}, {15, 14, 12, 9, 5,40, 34, 27, 19, 10,65, 54, 42, 29, 15});    //+++
    auto expTT = NDArrayFactory::create<double>('c', {3, 5}, {14, 12, 9, 5, 0,34, 27, 19, 10, 0,54, 42, 29, 15, 0});

    int exclusive, reverse;

    //************************************//
    exclusive = 0; reverse = 0;

    nd4j::ops::cumsum op;
    auto result = op.execute({&inputC, &axis}, {}, {exclusive, reverse}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
    ASSERT_TRUE(expFF.equalsTo(z));
    delete result;

    //************************************//
    exclusive = 1; reverse = 0;

    result = op.execute({&inputC, &axis}, {}, {exclusive, reverse}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());
    z = result->at(0);
    ASSERT_TRUE(expTF.equalsTo(z));
    delete result;

    //************************************//
    exclusive = 0; reverse = 1;

    result = op.execute({&inputC, &axis}, {}, {exclusive, reverse}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());
    z = result->at(0);
    ASSERT_TRUE(expFT.equalsTo(z));
    delete result;

    //************************************//
    exclusive = 1; reverse = 1;

    result = op.execute({&inputC, &axis}, {}, {exclusive, reverse}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());
    z = result->at(0);
    ASSERT_TRUE(expTT.equalsTo(z));
    delete result;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_10) {
    auto x = NDArrayFactory::create<double>('c', {4, 16, 16, 1});
    auto y = NDArrayFactory::create<int>(-3);

    nd4j::ops::cumsum op;
    auto result = op.execute({&x, &y}, {}, {1, 1}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_11) {

    NDArray x('c', {3, 3, 3}, nd4j::DataType::DOUBLE);
    auto exp = NDArrayFactory::create<double>('c', {3,3,3}, {12., 15., 18.,11., 13., 15.,7.,  8.,  9., 39., 42., 45.,29., 31., 33.,16., 17., 18., 66., 69., 72.,47., 49., 51.,25., 26., 27.});

    x.linspace(1);

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 1, 1}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_12) {

    NDArray x('c', {3, 3, 3}, nd4j::DataType::DOUBLE);
    auto exp = NDArrayFactory::create<double>('c', {3,3,3}, {1.,  2.,  3.,5.,  7.,  9.,12., 15., 18., 10., 11., 12.,23., 25., 27.,39., 42., 45., 19., 20., 21.,41., 43., 45., 66., 69., 72.});

    x.linspace(1);

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 0, 1}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_13) {

    NDArray x('c', {3, 3, 3}, nd4j::DataType::DOUBLE);
    auto exp = NDArrayFactory::create<double>('c', {3,3,3}, {11., 13., 15.,7.,  8.,  9.,0.,  0.,  0., 29., 31., 33.,16., 17., 18.,0.,  0.,  0., 47., 49., 51.,25., 26., 27.,0.,  0.,  0.});

    x.linspace(1);

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {1, 1, 1}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_14) {

    NDArray x('c', {3, 3, 3}, nd4j::DataType::DOUBLE);
    auto exp = NDArrayFactory::create<double>('c', {3,3,3}, {29., 31., 33.,35., 37., 39.,41., 43., 45., 19., 20., 21.,22., 23., 24.,25., 26., 27.,  0.,  0.,  0.,0.,  0.,  0.,0.,  0.,  0.});

    x.linspace(1);

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {1, 1, 0}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_15) {

    NDArray x('c', {3, 3, 3}, nd4j::DataType::DOUBLE);
    auto exp = NDArrayFactory::create<double>('c', {3,3,3}, {6.,  5.,  3.,15., 11.,  6.,24., 17.,  9., 33., 23., 12.,42., 29., 15.,51., 35., 18., 60., 41., 21.,69., 47., 24.,78., 53., 27.});

    x.linspace(1);

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 1, 2}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_16) {

    NDArray x('f', {3, 4}, nd4j::DataType::FLOAT32);

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printShapeInfo();
    // x.printShapeInfo();

    ASSERT_TRUE(z->ews() == 1);
    ASSERT_TRUE(x.ews() == 1);

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_17) {

    NDArray x('c', {2, 1500}, nd4j::DataType::FLOAT32);
    NDArray x0 = x(0, {0});
    NDArray x1 = x(1, {0});
    x0.linspace(1);
    x1.linspace(1);

    NDArray exp('c', {2, 1500}, nd4j::DataType::FLOAT32);
    NDArray exp0 = exp(0, {0});
    NDArray exp1 = exp(1, {0});

    exp0.p<float>(0, 1.);
    exp1.p<float>(0, 1.);

    for (int i = 1; i < 1500; ++i) {
        const auto prev = exp0.e<float>(i-1);
        exp0.p<float>(i, prev + i + 1);
        exp1.p<float>(i, prev + i + 1);
    }

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_18) {

    NDArray x('c', {2, 1500}, nd4j::DataType::FLOAT32);
    NDArray x0 = x(0, {0});
    NDArray x1 = x(1, {0});
    x0.linspace(1);
    x1.linspace(1);

    NDArray exp('c', {2, 1500}, nd4j::DataType::FLOAT32);
    NDArray exp0 = exp(0, {0});
    NDArray exp1 = exp(1, {0});

    exp0.p<float>(0, 0.);
    exp1.p<float>(0, 0.);

    for (int i = 1; i < 1500; ++i) {
        const auto prev = exp0.e<float>(i-1);
        exp0.p<float>(i, prev + i);
        exp1.p<float>(i, prev + i);
    }

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {1, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_19) {

    NDArray x('c', {2, 1500}, nd4j::DataType::FLOAT32);
    NDArray x0 = x(0, {0});
    NDArray x1 = x(1, {0});
    x0.linspace(1);
    x1.linspace(1);

    NDArray exp('c', {2, 1500}, nd4j::DataType::FLOAT32);
    NDArray exp0 = exp(0, {0});
    NDArray exp1 = exp(1, {0});

    exp0.p<float>(1499, 1500.);
    exp1.p<float>(1499, 1500.);

    for (int i = 1498; i >= 0; --i) {
        const auto prev = exp0.e<float>(i + 1);
        exp0.p<float>(i, prev + i + 1);
        exp1.p<float>(i, prev + i + 1);
    }

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // exp0.printBuffer();

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, cumSum_20) {

    NDArray x('c', {2, 1500}, nd4j::DataType::FLOAT32);
    NDArray x0 = x(0, {0});
    NDArray x1 = x(1, {0});
    x0.linspace(1);
    x1.linspace(1);

    NDArray exp('c', {2, 1500}, nd4j::DataType::FLOAT32);
    NDArray exp0 = exp(0, {0});
    NDArray exp1 = exp(1, {0});

    exp0.p<float>(1499, 0.);
    exp1.p<float>(1499, 0.);

    for (int i = 1498; i >= 0; --i) {
        const auto prev = exp0.e<float>(i + 1);
        exp0.p<float>(i, prev + i + 2);
        exp1.p<float>(i, prev + i + 2);
    }

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {1, 1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, TestMergeMaxIndex_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    auto y = NDArrayFactory::create<double>('c', {2, 2, 2}, {10.f, 2.f, 30.f, 4.f, 50.f, 6.f, 70.f, 8.f});
    auto z = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.f, 20.f, 3.f, 40.f, 5.f, 60.f, 7.f, 80.f});
    auto exp = NDArrayFactory::create<int>('c', {2, 2, 2}, {1, 2, 1, 2, 1, 2, 1, 2});
    nd4j::ops::mergemaxindex op;

    auto ress = op.execute({&x, &y, &z}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
//    ress->at(0)->printIndexedBuffer("MergeMaxIndex Result is ");
//    ress->at(0)->printShapeInfo("Shape info for MergeMaxIdex");
//    x.printIndexedBuffer("Input is");
    ASSERT_TRUE(ress->at(0)->equalsTo(exp));
    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, TestMergeMaxIndex_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    auto y = NDArrayFactory::create<double>('c', {2, 2, 2}, {10.f, 2.f, 30.f, 4.f, 50.f, 6.f, 70.f, 8.f});
    auto z = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.f, 20.f, 3.f, 40.f, 5.f, 60.f, 7.f, 80.f});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {2, 2, 2}, {1, 2, 1, 2, 1, 2, 1, 2});
    nd4j::ops::mergemaxindex op;

    auto ress = op.execute({&x, &y, &z}, {}, {nd4j::DataType::INT64}, {});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
//    ress->at(0)->printIndexedBuffer("MergeMaxIndex2 Result is ");
//    ress->at(0)->printShapeInfo("Shape info for MergeMaxIdex2");
//    x.printIndexedBuffer("Input is");
    ASSERT_TRUE(ress->at(0)->equalsTo(exp));
    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, TestDropout_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    auto shape = NDArrayFactory::create<Nd4jLong>({2, 2});
    nd4j::ops::dropout op;

    auto ress = op.execute({&x, &shape}, {0.2f}, {113}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    //ress->at(0)->printIndexedBuffer("Result is ");
    //x.printIndexedBuffer("Input is");

    delete ress;
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, TestMod_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    auto y = NDArrayFactory::create<double>('c', {2, 2, 2}, {10.f, 2.f, 30.f, 4.f, 50.f, 6.f, 70.f, 8.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 2, 2}, {1, 0, 3, 0, 5, 0, 7, 0});
    nd4j::ops::mod op;

    auto ress = op.execute({&x, &y}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
//    ress->at(0)->printIndexedBuffer("MOD Result is ");
//    x.printIndexedBuffer("Input is");
    ASSERT_TRUE(ress->at(0)->equalsTo(exp));
    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, TestMod_BP_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    auto y = NDArrayFactory::create<double>('c', {2, 2, 2}, {10.f, 2.f, 30.f, 4.f, 50.f, 6.f, 70.f, 8.f});
    auto eps = NDArrayFactory::create<double>('c', {2, 2, 2}, {10.f, 2.f, 30.f, 4.f, 50.f, 6.f, 70.f, 8.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 2, 2});
    nd4j::ops::mod_bp op;

    auto ress = op.execute({&x, &y, &eps}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
//    ress->at(0)->printIndexedBuffer("MOD_BP Result is ");

    //    x.printIndexedBuffer("Input is");
    ASSERT_TRUE(ress->at(0)->equalsTo(exp));
    delete ress;
}

///////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, TestRank_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    auto y = NDArrayFactory::create<double>('c', {2, 2, 2}, {10.f, 2.f, 30.f, 4.f, 50.f, 6.f, 70.f, 8.f});
    auto eps = NDArrayFactory::create<double>('c', {2, 2, 2}, {10.f, 2.f, 30.f, 4.f, 50.f, 6.f, 70.f, 8.f});
    auto exp = NDArrayFactory::create<int>(3);
    nd4j::ops::rank op;

    auto ress = op.execute({&x}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ress->at(0)->printIndexedBuffer("RANK Result is ");

    //    x.printIndexedBuffer("Input is");
    ASSERT_TRUE(ress->at(0)->equalsTo(exp));
    delete ress;
}
TEST_F(DeclarableOpsTests6, TestDropout_2) {
//    auto x0 = NDArrayFactory::create<double>('c', {10, 10});
//    auto x1 = NDArrayFactory::create<double>('c', {10, 10});
    auto x = NDArrayFactory::create<double>('c', {3, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});

    nd4j::ops::dropout op;

    auto ress = op.execute({&x}, {0.4f}, {113}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    //x.printIndexedBuffer("Input is");
    //ress->at(0)->printIndexedBuffer("Result is ");

    delete ress;
}

TEST_F(DeclarableOpsTests6, TestDropout_3) {
//    auto x0 = NDArrayFactory::create<double>('c', {10, 10});
//    auto x1 = NDArrayFactory::create<double>('c', {10, 10});
    auto x = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    auto shape = NDArrayFactory::create<int>({1, 2});

    nd4j::ops::dropout op;

    auto ress = op.execute({&x, &shape}, {0.4f}, {113}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    //x.printIndexedBuffer("Input is");
    //ress->at(0)->printIndexedBuffer("Result is ");

    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MaxPoolWithArgmax_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 2, 4}, {5.5, 0.,   0.3,  5.5,1.5, 0.,   1.3,  6.5,8.6, 0.,    0.,  0.4,2.5, 1.,   0.3,  4.5,
                                                                1.5, 1.,   1.3,  1.5, 3.5, 0.,   1.3,  2.5, 2.6, 2.,    3.,  1.4, 4.5, 1.,   0.3,  0.5});
    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 2, 2, 4}, {0,  1,  2,  3,4,  5,  6,  7,8,  9, 10, 11,12, 13, 14, 15,
                                                                0,  1,  2,  3,4,  5,  6,  7,8,  9, 10, 11,12, 13, 14, 15});

    nd4j::ops::max_pool_with_argmax op;

    auto ress = op.execute({&x}, {}, {1,1,1,1,1,1,1,1,1});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ASSERT_TRUE(expI.isSameShape(ress->at(0)));
    ASSERT_TRUE(expI.isSameShape(ress->at(1)));
    ASSERT_TRUE(x.equalsTo(ress->at(0)));
    ASSERT_TRUE(expI.equalsTo(ress->at(1)));
    //x.printIndexedBuffer("Input is");

    ASSERT_TRUE(expI.equalsTo(ress->at(1)));

    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, SufficientStatistics_1) {
//    auto x0 = NDArrayFactory::create<double>('c', {10, 10});
//    auto x1 = NDArrayFactory::create<double>('c', {10, 10});
    auto x = NDArrayFactory::create<double>('c', {2, 2, 2, 4}, {5.5, 0.,  0.3, 5.5,1.5, 0.,  1.3, 6.5,8.6, 0.,   0., 0.4,2.5, 1.,  0.3, 4.5,1.5, 1.,
                                                                1.3, 1.5,3.5, 0.,  1.3, 2.5,2.6, 2.,   3., 1.4,4.5, 1.,  0.3, 0.5});
// ------------------------------------
    double count = 8.0;
    auto sumExp = NDArrayFactory::create<double>({30.2, 5., 7.8, 22.8});
    auto sqrExp = NDArrayFactory::create<double>({154.22,   7.,    14.34, 103.62});

    auto axis = NDArrayFactory::create<Nd4jLong>({0, 1, 2});

    nd4j::ops::sufficient_statistics op;

    auto ress = op.execute({&x, &axis}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ASSERT_EQ(ress->at(0)->e<double>(0), count);
    ASSERT_TRUE(sumExp.equalsTo(ress->at(1)));
    ASSERT_TRUE(sqrExp.equalsTo(ress->at(2)));

    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, SufficientStatistics_2) {
//    auto x0 = NDArrayFactory::create<double>('c', {10, 10});
//    auto x1 = NDArrayFactory::create<double>('c', {10, 10});
    auto x = NDArrayFactory::create<double>('c', {2, 2, 2, 4}, {5.5, 0.,  0.3, 5.5,1.5, 0.,  1.3, 6.5,8.6, 0.,   0., 0.4,2.5, 1.,  0.3, 4.5,
                                                                1.5, 1.,  1.3, 1.5,3.5, 0.,  1.3, 2.5,2.6, 2.,   3., 1.4,4.5, 1.,  0.3, 0.5});
// ------------------------------------
    double count = 4.0;
    auto sumExp = NDArrayFactory::create<double>('c', {2, 4}, {
        18.2,        3.,         4.6,        8.8,
        12.,         2.,         3.2,        14.}
    );

    auto sqrExp = NDArrayFactory::create<double>('c', {2, 4}, {
        113.22, 5., 10.78, 34.62,
           41., 2.,  3.56, 69.}
    );

    auto axis = NDArrayFactory::create<int>({0, 1});

    nd4j::ops::sufficient_statistics op;

    auto ress = op.execute({&x, &axis}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ASSERT_EQ(ress->at(0)->e<double>(0), count);
    ASSERT_TRUE(sumExp.equalsTo(ress->at(1)));
    ASSERT_TRUE(sqrExp.equalsTo(ress->at(2)));

    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_1) {

    auto x = NDArrayFactory::create<int>('c', {2, 2, 2}, {
        1, 2, 0, 1, 2, 2, 1, 2}
    );
// ------------------------------------

    NDArray exp('c', {3}, {1, 3, 4}, nd4j::DataType::INT32);

    nd4j::ops::bincount op;

    auto res = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_2) {

    auto x = NDArrayFactory::create<int>('c', {2, 2, 2}, {
        1, 2, 0, 1, 2, 2, 1, 2}
    );

    auto weights = NDArrayFactory::create<double>('c', {2, 2, 2}, {
        2, 1, 3, 1, 5, 1, 1, 6}
    );

// ------------------------------------

    auto exp = NDArrayFactory::create<double>({3., 4., 13.});

    nd4j::ops::bincount op;

    auto res = op.execute({&x, &weights}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_3) {

    auto x = NDArrayFactory::create<int>('c', {2, 2, 2}, {
        1, 2, 0, 1, 2, 2, 1, 2}
    );

    auto weights = NDArrayFactory::create<double>('c', {2, 2, 2}, {
        2, 1, 3, 1, 5, 1, 1, 6}
    );

// ------------------------------------

    auto exp = NDArrayFactory::create<double>({3., 4.});

    nd4j::ops::bincount op;

    auto res = op.execute({&x, &weights}, {}, {0, 2});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_4) {

    auto x = NDArrayFactory::create<int>('c', {2, 2, 2}, {
        1, 2, 0, 1, 2, 2, 1, 2}
    );

    auto weights = NDArrayFactory::create<double>('c', {2, 2, 2}, {
        2, 1, 3, 1, 5, 1, 1, 6}
    );

// ------------------------------------

    auto exp = NDArrayFactory::create<double>({3., 4.,  13., 0.0});

    nd4j::ops::bincount op;

    auto res = op.execute({&x, &weights}, {}, {4, 4});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_5) {

    auto x = NDArrayFactory::create<int>('c', {2, 2, 2}, {
            1, 2, 0, 1, 2, 2, 1, 2}
    );

    auto weights = NDArrayFactory::create<double>('c', {2, 2, 2}, {
            2, 1, 3, 1, 5, 1, 1, 6}
    );
    auto minV = NDArrayFactory::create(4);
    auto maxV = NDArrayFactory::create(4);
// ------------------------------------

    auto exp = NDArrayFactory::create<double>({3., 4., 13., 0.0});

    nd4j::ops::bincount op;

    auto res = op.execute({&x, &weights, &minV, &maxV}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    // res->at(0)->printBuffer("BC out");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_1) {

    auto x = NDArrayFactory::create<int>( {2, 2, 2} );

    auto y = NDArrayFactory::create<int>({ 2, 1, 2});

// ------------------------------------

    auto exp = NDArrayFactory::create<int>({2, 2, 2});

    nd4j::ops::broadcast_dynamic_shape op;

    auto res = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::INT32);

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_2) {

    auto x = NDArrayFactory::create<Nd4jLong>( {2, 2} );

    auto y = NDArrayFactory::create<Nd4jLong>({2, 1, 2});

// ------------------------------------
    auto exp = NDArrayFactory::create<Nd4jLong>({2, 2, 2});

    nd4j::ops::broadcast_dynamic_shape op;

    auto res = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::INT64);
    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_3) {

    auto x = NDArrayFactory::create<Nd4jLong>( {2, 2, 2} );

    auto y = NDArrayFactory::create<Nd4jLong>({ 2, 1});

// ------------------------------------

    auto exp = NDArrayFactory::create<Nd4jLong>({2, 2, 2});

    nd4j::ops::broadcast_dynamic_shape op;

    auto res = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::INT64);

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_SGO_4) {

    auto x = NDArrayFactory::create<Nd4jLong>( {2, 1} );

    auto y = NDArrayFactory::create<Nd4jLong>('c', {1}, { 4,});

// ------------------------------------

    auto exp = NDArrayFactory::create<Nd4jLong>({2, 4});

    nd4j::ops::broadcast_dynamic_shape op;

    auto res = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::INT64);

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    //res->at(0)->printBuffer("Shape SGO 4");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}
/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_5) {

    auto x = NDArrayFactory::create<Nd4jLong>({2, 2, 2});

    auto y = NDArrayFactory::create<Nd4jLong>({2, 2});

// ------------------------------------

    auto exp = NDArrayFactory::create<Nd4jLong>({2, 2, 2});

    nd4j::ops::broadcast_dynamic_shape op;
    auto res = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::INT64);

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
//    res->at(0)->printIndexedBuffer("Output");
//    exp.printIndexedBuffer("Expect");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_SGO_5) {

    auto x = NDArrayFactory::create<Nd4jLong>({2, 1, 2});

    auto y = NDArrayFactory::create<Nd4jLong>({2, 2, 4});

// ------------------------------------

    auto exp = NDArrayFactory::create<Nd4jLong>({2, 2, 4});

    nd4j::ops::broadcast_dynamic_shape op;
    auto res = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::INT64);

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    // res->at(0)->printIndexedBuffer("Output SGO 5");
//    exp.printIndexedBuffer("Expect");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}
/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_SGO_6) {

    auto x = NDArrayFactory::create<Nd4jLong>({2, 1, 4});

    auto y = NDArrayFactory::create<Nd4jLong>({2, 2, 4});

// ------------------------------------

    auto exp = NDArrayFactory::create<Nd4jLong>({2, 2, 4});

    nd4j::ops::broadcast_dynamic_shape op;
    auto res = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::INT64);

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    // res->at(0)->printIndexedBuffer("Output SGO 6");
//    exp.printIndexedBuffer("Expect");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_SGO_7) {

    auto x = NDArrayFactory::create<Nd4jLong>({1, 1, 3});

    auto y = NDArrayFactory::create<Nd4jLong>({2, 4, 1});

// ------------------------------------

    auto exp = NDArrayFactory::create<Nd4jLong>({2, 4, 3});

    nd4j::ops::broadcast_dynamic_shape op;
    auto res = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::INT64);

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    // res->at(0)->printIndexedBuffer("Output SGO 7");
//    exp.printIndexedBuffer("Expect");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_SGO_8) {

    auto x = NDArrayFactory::create<int>('c', {1}, {1});

    auto y = NDArrayFactory::create<int>('c', {1}, {4});

// ------------------------------------

    auto exp = NDArrayFactory::create<int>('c', {1}, {4});

    nd4j::ops::broadcast_dynamic_shape op;
    auto res = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::INT32);

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
//    res->at(0)->printIndexedBuffer("Output SGO 8");
//    exp.printIndexedBuffer("Expect");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_SGO_9) {

    auto x = NDArrayFactory::create<int>('c', {2}, {2,2});

    auto y = NDArrayFactory::create<int>('c', {1}, {1});

// ------------------------------------

    auto exp = NDArrayFactory::create<int>('c', {2}, {2,2});

    nd4j::ops::broadcast_dynamic_shape op;
    auto res = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::INT32);

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    res->at(0)->printIndexedBuffer("Output SGO 9");
    exp.printIndexedBuffer("Expect9");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, ClipByGlobalNorm_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0}
    );

    auto exp = NDArrayFactory::create<double>('c', {2, 3, 3}, {
            -0.2771281,  0.,          0.,
            0.36950415,  0.,          0.,
            -0.2771281,  0.,          0.,
            0.36950415,  0.,          0.,
            -0.2771281,  0.,          0.,
            0.36950415,  0.,          0.}
    );
//    8.660254
//    auto expNorm(8.660254);

    nd4j::ops::clip_by_global_norm op;
    auto result = op.execute({&x}, {0.8}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto norm = result->at(1);
    //z->printIndexedBuffer("Output");
    //exp.printIndexedBuffer("Expected");
    //norm->printIndexedBuffer("Norm");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
//    ASSERT_TRUE(expNorm.equalsTo(norm));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, ClipByGlobalNorm_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0}
    );

    auto a = NDArrayFactory::create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0}
    );

    auto exp = NDArrayFactory::create<double>('c', {2, 3, 3}, {
                                    -0.44090813,   0.,          0.,
                                      0.5878775,   0.,          0.,
                                    -0.44090813,   0.,          0.,
                                      0.5878775,   0.,          0.,
                                    -0.44090813,   0.,          0.,
                                      0.5878775,   0.,          0.}
//12.247449

    );

    nd4j::ops::clip_by_global_norm op;
    auto result = op.execute({&x, &a}, {1.8}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto y = result->at(1);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.isSameShape(y));
    ASSERT_TRUE(exp.equalsTo(z));
    ASSERT_TRUE(exp.equalsTo(y));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, ClipByGlobalNorm_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    auto a = NDArrayFactory::create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 3}, {
            -0.19595918,  0.,          0.,
              0.2612789,  0.,          0.,
            -0.19595918,  0.,          0.,
              0.2612789,  0.,          0.,
            -0.19595918,  0.,          0.,
              0.2612789,   0.,          0.}
    );

    nd4j::ops::clip_by_global_norm op;
    auto result = op.execute({&x, &a}, {0.8}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto y = result->at(1);
    //z->printIndexedBuffer("Output 1");
    //y->printIndexedBuffer("Output 2");
    //result->at(2)->printIndexedBuffer("Global norm is");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.isSameShape(y));
    ASSERT_TRUE(result->at(2)->isScalar());
    ASSERT_TRUE(exp.equalsTo(z));
    ASSERT_TRUE(exp.equalsTo(y));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, -3.0, 4.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 4.0});
    auto exp = NDArrayFactory::create<double>({36.0, -48.0});

    nd4j::ops::matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0});
    auto exp = NDArrayFactory::create<double>({-2.0, -2.0});

    nd4j::ops::matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_3) {

    auto x = NDArrayFactory::create<double>('c', {1, 3, 3}, {3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0});
    NDArray exp('c', {1}, {-54.0});

    nd4j::ops::matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_4) {

    auto x = NDArrayFactory::create<double>('c', {1, 3, 3}, {12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 13.0});
    auto exp = NDArrayFactory::create<double>('c', {1}, {189.0});

    nd4j::ops::matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    // z->printIndexedBuffer("Output ");
    // exp.printIndexedBuffer("Expected ");
    // z->printShapeInfo("Output shape");
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_5) {

    auto x = NDArrayFactory::create<double>('c', {1, 4, 4});
    NDArray exp('c', {1}, {-16.0});
    x.linspace(1);
    x.p(5, 4.0);
    x.p(12, 12.0);

    nd4j::ops::matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_6) {

    auto x = NDArrayFactory::create<double>('c', {4, 4});
    auto exp = NDArrayFactory::create<double>(-16.0);
    x.linspace(1);
    x.p(5, 4.0);
    x.p(12, 12.0);

    nd4j::ops::matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //z->printShapeInfo("Shape");
    //exp.printIndexedBuffer("Expected ");
    ASSERT_TRUE(z->isScalar());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, LogMatrixDeterminant_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, -3.0, 4.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 4.0});
    auto exp = NDArrayFactory::create<double>({3.58351893845611, 3.871201010907891});

    nd4j::ops::log_matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, LogDet_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 3}, {4,12,-16,12,37,-43,-16,-43,98, 4,1.2,-1.6,1.2,3.7,-4.3,-1.6,-4.3,9.8});
    auto exp = NDArrayFactory::create<double>({ 3.5835189, 4.159008});

    //x.printIndexedBuffer("Input");
    nd4j::ops::logdet op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("LogDet Output1 ");
//    exp.printIndexedBuffer("LogDet Expected1 ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, LogDet_2) {

    auto x = NDArrayFactory::create<double>('c', {1, 3, 3}, {4,12,-16,12,37,-43,-16,-43,98});
    auto exp = NDArrayFactory::create<double>('c', {1}, { 3.5835189});

    //x.printIndexedBuffer("Input");
    nd4j::ops::logdet op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("LogDet Output2 ");
//    z->printShapeInfo("Shape");
//    exp.printIndexedBuffer("LogDet Expected2 ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, LogDet_3) {

    auto x = NDArrayFactory::create<double>('c', {3, 3}, {4,12,-16,12,37,-43,-16,-43,98});
    auto exp = NDArrayFactory::create<double>( 3.5835189);

    //x.printIndexedBuffer("Input");
    nd4j::ops::logdet op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("LogDet Output3 ");
//    z->printShapeInfo("Shape");
//    exp.printIndexedBuffer("LogDet Expected3 ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixInverse_1) {

    auto x = NDArrayFactory::create<float>('c', {2, 5, 5}, {
                    2.,  4., 60.,  8., 10.,
                    0.,  1.,  2.,  3.,  4.,
                    0.,  0.,  2.,  4.,  6.,
                    0.,  0.,  0.,  1.,  2.,
                    0.,  0.,  0.,  0.,  4.,

                     1.,  0.,  0.,  0.,  0.,
                     2.,  1.,  0.,  0.,  0.,
                    30.,  2.,  1.,  0.,  0.,
                     4.,  3.,  2.,  1.,  0.,
                     5.,  4.,  3.,  2.,  1.,
    });

    auto exp = NDArrayFactory::create<float>('c', {2, 5, 5}, {
                    0.5, -2.0, -13.0, 54.0, -6.75,
                    0.0,  1.0,  -1.0,  1.0,   0.0,
                      0,    0,   0.5, -2.0,  0.25,
                      0,    0,     0,  1.0,  -0.5,
                      0,    0,     0,    0,  0.25,

                    1.0,  0.0,  0.0,  0.0, 0.,
                   -2.0,  1.0,   0.,   0., 0.,
                  -26.0, -2.0,    1,    0, 0.,
                   54.0,  1.0, -2.0,    1, 0.,
                  -27.0,  0.0,  1.0, -2.0, 1.
    });

    nd4j::ops::matrix_inverse op;
    auto result = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::FLOAT32);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Output ");
//    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixInverse_01) {

    auto x = NDArrayFactory::create<float>('c', {1, 5, 5}, {
            2.,  4., 60.,  8., 10.,
            0.,  1.,  2.,  3.,  4.,
            0.,  0.,  2.,  4.,  6.,
            0.,  0.,  0.,  1.,  2.,
            0.,  0.,  0.,  0.,  4.

    });

    auto exp = NDArrayFactory::create<float>('c', {1, 5, 5}, {
            0.5, -2.0, -13.0, 54.0, -6.75,
            0.0,  1.0,  -1.0,  1.0,   0.0,
            0,    0,   0.5, -2.0,  0.25,
            0,    0,     0,  1.0,  -0.5,
            0,    0,     0,    0,  0.25

    });

    nd4j::ops::matrix_inverse op;
    auto result = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::FLOAT32);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Output ");
//    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixInverse_02) {

    auto x = NDArrayFactory::create<float>('c', {1, 5, 5}, {
            1.,  0.,  0.,  0.,  0.,
            2.,  1.,  0.,  0.,  0.,
            30.,  2.,  1.,  0.,  0.,
            4.,  3.,  2.,  1.,  0.,
            5.,  4.,  3.,  2.,  1.
    });

    auto exp = NDArrayFactory::create<float>('c', {1, 5, 5}, {
            1.0,  0.0,  0.0,  0.0, 0.,
            -2.0,  1.0,   0.,   0., 0.,
            -26.0, -2.0,    1,    0, 0.,
            54.0,  1.0, -2.0,    1, 0.,
            -27.0,  0.0,  1.0, -2.0, 1.
    });

    nd4j::ops::matrix_inverse op;
    auto result = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::FLOAT32);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Output ");
//    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
/*
TEST_F(DeclarableOpsTests6, MatrixInverse_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 5, 5}, {
                    1.,  2., 30.,  4.,  5.,
                    0.,  1.,  2.,  3.,  4.,
                    0.,  0.,  1.,  2.,  3.,
                    0.,  0.,  0.,  1.,  2.,
                    0.,  0.,  0.,  0.,  1.,

                     4.,   0.,  0.,  0.,  0.,
                     4.,   2.,  0.,  0.,  0.,
                    30.,   2.,  1.,  0.,  0.,
                     8.,   6.,  4.,  2.,  0.,
                    15.,  12.,  9.,  6.,  3.,
    });

    auto exp = NDArrayFactory::create<double>('c', {2, 5, 5}, {
     1.0,  -2.0,  -26.0,  54.0, -27.0,
     0.0,   1.0,  -2.0,    1.0,   0.0,
     0.0,   0.0,   1.0,   -2.0,   1.0,
     0.0,   0.0,   0.0,    1.0,  -2.0,
     0.0,   0.0,   0.0,    0.0,   1.0,

     0.25,  0.0,    0.0,   0.0,   0.0,
    -0.50,  0.5,    0.0,   0.0,   0.0,
    -6.50, -1.0,    1.0,   0.0,   0.0,
    13.50,  0.5,   -2.0,   0.5,   0.0,
    -6.75,  0.0,    1.0,  -1.0,   0.33333333
    });

    nd4j::ops::matrix_inverse op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    z->printIndexedBuffer("Output ");
    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}
*/
TEST_F(DeclarableOpsTests6, MatrixInverse_03) {

    auto x = NDArrayFactory::create<float>('c', {5, 5}, {
            4.,   0.,  0.,  0.,  0.,
            4.,   2.,  0.,  0.,  0.,
           30.,   2.,  1.,  0.,  0.,
            8.,   6.,  4.,  2.,  0.,
           15.,  12.,  9.,  6.,  3.,
    });

    auto exp = NDArrayFactory::create<float>('c', {5, 5}, {
            0.25,  0.0,    0.0,   0.0,   0.0,
            -0.50,  0.5,    0.0,   0.0,   0.0,
            -6.50, -1.0,    1.0,   0.0,   0.0,
            13.50,  0.5,   -2.0,   0.5,   0.0,
            -6.75,  0.0,    1.0,  -1.0,   0.33333333
    });

    nd4j::ops::matrix_inverse op;
    auto result = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::FLOAT32);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Output ");
//    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixInverse_3) {

    auto x = NDArrayFactory::create<double>('c', {5, 5}, {
                     4.,   0.,  0.,  0.,  0.,
                     4.,   2.,  0.,  0.,  0.,
                    30.,   2.,  1.,  0.,  0.,
                     8.,   6.,  4.,  2.,  0.,
                    15.,  12.,  9.,  6.,  3.,
    });

    auto exp = NDArrayFactory::create<double>('c', {5, 5}, {
     0.25,  0.0,    0.0,   0.0,   0.0,
    -0.50,  0.5,    0.0,   0.0,   0.0,
    -6.50, -1.0,    1.0,   0.0,   0.0,
    13.50,  0.5,   -2.0,   0.5,   0.0,
    -6.75,  0.0,    1.0,  -1.0,   0.33333333
    });

    nd4j::ops::matrix_inverse op;
    auto result = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixInverse_4) {

    auto x = NDArrayFactory::create<float>('c', {5, 5}, {
                    1.,  2., 30.,  4.,  5.,
                    0.,  1.,  2.,  3.,  4.,
                    0.,  0.,  1.,  2.,  3.,
                    0.,  0.,  0.,  1.,  2.,
                    0.,  0.,  0.,  0.,  1.
    });

    auto exp = NDArrayFactory::create<float>('c', {5, 5}, {
     1.0,  -2.0,  -26.0,  54.0, -27.0,
     0.0,   1.0,  -2.0,    1.0,   0.0,
     0.0,   0.0,   1.0,   -2.0,   1.0,
     0.0,   0.0,   0.0,    1.0,  -2.0,
     0.0,   0.0,   0.0,    0.0,   1.0
    });

    nd4j::ops::matrix_inverse op;
    auto result = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::FLOAT32);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    z->printIndexedBuffer("Output ");
    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, ReluLayer_1) {
    auto x = NDArrayFactory::create<double>('c', {3, 4}, {1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0, 8.0, 9.0, -10.0, 11.0, 12});
    auto w = NDArrayFactory::create<double>('c', {4, 3}, {0.5, 0.1, 0.8, 0.5, 0.2, 0.5, 0.5, 0.25, 0.5, 0.1, 0.0, 0.25});
    auto b = NDArrayFactory::create<double>({20.0, 30.0, 50.0});



    auto exp = NDArrayFactory::create<double>('c', {3, 3}, {
                        21.4,  30.45, 52.3,
                        23.8,  31.05, 56.5,
                        26.2,  31.65, 60.7});

    nd4j::ops::relu_layer op;
    auto result = op.execute({&x, &w, &b}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    // z->printShapeInfo("Output shape");
    // z->printIndexedBuffer("Output ");
    // exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Reduce3_Edge) {
    auto x = NDArrayFactory::create<double>('c', {3, 4, 5});
    auto y = NDArrayFactory::create<double>('c', {3, 4, 5});


    std::vector<int> dims = {0, 1};
    auto z = x.applyReduce3(reduce3::CosineSimilarity, &y, dims, nullptr);
    ASSERT_TRUE(z != nullptr);

    delete z;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_rnn_test1) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::create<double>('c', {bS, numUnits});
    auto maxTimeStep = NDArrayFactory::create<double>('c', {bS}, {time-1, time-3});

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::create<double>('c', {time, bS, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0.69882484, 0.69882484, 0.69882484, 0.69882484, 0.9312333 , 0.9312333 , 0.9312333 , 0.9312333 ,
                                                          0.93751527, 0.93751527, 0.93751527, 0.93751527,0.97136768, 0.97136768, 0.97136768, 0.97136768,0., 0., 0., 0.        ,
                                                          0.97732812, 0.97732812, 0.97732812, 0.97732812,0., 0., 0., 0.        ,0., 0., 0., 0.,0., 0., 0., 0.});

    auto expHFinal = NDArrayFactory::create<double>('c', {bS, numUnits},       {0.97732812, 0.97732812, 0.97732812, 0.97732812, 0.93751527, 0.93751527, 0.93751527, 0.93751527});

    nd4j::ops::static_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_rnn_test2) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::create<double>('c', {bS, numUnits});

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::create<double>('c', {time, bS, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0.69882484, 0.69882484, 0.69882484, 0.69882484,0.9312333 , 0.9312333 , 0.9312333 , 0.9312333,
                                                          0.93751527, 0.93751527, 0.93751527, 0.93751527,0.97136768, 0.97136768, 0.97136768, 0.97136768,0.97338548, 0.97338548, 0.97338548, 0.97338548,
                                                          0.97732812, 0.97732812, 0.97732812, 0.97732812,0.97864398, 0.97864398, 0.97864398, 0.97864398,0.98000654, 0.98000654, 0.98000654, 0.98000654,
                                                          0.98112648, 0.98112648, 0.98112648, 0.98112648});

    auto expHFinal = NDArrayFactory::create<double>('c', {bS, numUnits},       {0.98000654, 0.98000654, 0.98000654, 0.98000654,0.98112648, 0.98112648, 0.98112648, 0.98112648});

    nd4j::ops::static_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_rnn_test3) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::create<double>('c', {bS, numUnits});
    auto maxTimeStep = NDArrayFactory::create<double>('c', {bS}, {time-1, 0});

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::create<double>('c', {time, bS, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0., 0., 0., 0., 0.9312333, 0.9312333, 0.9312333, 0.9312333,
                                                          0., 0., 0., 0.           , 0.97136768, 0.97136768, 0.97136768, 0.97136768,0., 0., 0., 0. ,
                                                          0.97732812, 0.97732812, 0.97732812, 0.97732812,0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.});

    auto expHFinal = NDArrayFactory::create<double>('c', {bS, numUnits},       {0.97732812, 0.97732812, 0.97732812, 0.97732812, 0.2       , 0.2       , 0.2       , 0.2});

    nd4j::ops::static_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_rnn_test4) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::create<double>('c', {bS, numUnits});
    auto maxTimeStep = NDArrayFactory::create<double>('c', {bS}, {time-1, time-3});

    x.linspace(0.01, 0.01);
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::create<double>('c', {time, bS, numUnits}, {0.47615493, 0.47615493, 0.47615493, 0.47615493,0.49676344, 0.49676344, 0.49676344, 0.49676344, 0.87018664, 0.87018664, 0.87018664, 0.87018664,
                                                          0.88400882, 0.88400882, 0.88400882, 0.88400882, 0.96529784, 0.96529784, 0.96529784, 0.96529784,0., 0., 0., 0.        ,
                                                          0.97688859, 0.97688859, 0.97688859, 0.97688859,0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.});

    auto expHFinal = NDArrayFactory::create<double>('c', {bS, numUnits},       {0.97688859, 0.97688859, 0.97688859, 0.97688859, 0.88400882, 0.88400882, 0.88400882, 0.88400882});

    nd4j::ops::static_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_rnn_test5) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::create<double>('c', {bS, numUnits});

    x.linspace(0.01, 0.01);
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::create<double>('c', {time, bS, numUnits}, {0.47615493, 0.47615493, 0.47615493, 0.47615493,0.49676344, 0.49676344, 0.49676344, 0.49676344, 0.87018664, 0.87018664, 0.87018664, 0.87018664,
                                                          0.88400882, 0.88400882, 0.88400882, 0.88400882, 0.96529784, 0.96529784, 0.96529784, 0.96529784,0.96849345, 0.96849345, 0.96849345, 0.96849345,
                                                          0.97688859, 0.97688859, 0.97688859, 0.97688859,0.97831069, 0.97831069, 0.97831069, 0.97831069, 0.97997868, 0.97997868, 0.97997868, 0.97997868,
                                                          0.98110653, 0.98110653, 0.98110653, 0.98110653});

    auto expHFinal = NDArrayFactory::create<double>('c', {bS, numUnits},       {0.97997868, 0.97997868, 0.97997868, 0.97997868, 0.98110653, 0.98110653, 0.98110653, 0.98110653});

    nd4j::ops::static_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_bidir_rnn_test1) {

    const int bS         = 4;
    const int inSize     = 4;
    const int numUnitsFW = 3;
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::create<double>('c', {time, bS, inSize});
    auto WxFW = NDArrayFactory::create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::create<double>('c', {numUnitsFW, numUnitsFW});
    auto bFW  = NDArrayFactory::create<double>('c', {2*numUnitsFW});

    auto h0FW = NDArrayFactory::create<double>('c', {bS, numUnitsFW});
    auto h0BW = NDArrayFactory::create<double>('c', {bS, numUnitsBW});
    auto maxTimeStep = NDArrayFactory::create<double>('c', {bS}, {time-1, time-3, time-4, 0});

    x.linspace(0.01, 0.01);
    h0FW = 0.2;
    h0BW = 0.25;
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expH      = NDArrayFactory::create<double>('c', {time, bS, numUnitsFW+numUnitsBW}, {0.43819931, 0.43819931, 0.43819931, 0.86708881, 0.86708881,0.86708881,0.47615493, 0.47615493, 0.47615493, 0.78347842, 0.78347842,0.78347842,
                                                                       0.51241561, 0.51241561, 0.51241561, 0.55529176, 0.55529176,0.55529176,0., 0., 0., 0., 0.,0.,0.73880324, 0.73880324, 0.73880324, 0.90935605, 0.90935605,
                                                                       0.90935605, 0.77843476, 0.77843476, 0.77843476, 0.64692945, 0.64692945,0.64692945,0., 0., 0., 0., 0.,0.,0., 0., 0., 0., 0.,0.,
                                                                       0.9052501, 0.9052501, 0.9052501, 0.9181592, 0.9181592, 0.9181592,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,
                                                                       0.9555734, 0.9555734, 0.9555734, 0.8026439, 0.8026439, 0.8026439,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,
                                                                       0., 0., 0., 0., 0., 0.,       0., 0., 0., 0., 0., 0.,       0., 0., 0., 0., 0., 0.,       0., 0., 0., 0., 0., 0.});

    auto expHFWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsFW},  {0.9555734 , 0.9555734 , 0.9555734 , 0.77843476, 0.77843476, 0.77843476, 0.51241561, 0.51241561, 0.51241561, 0.2, 0.2, 0.2});
    auto expHBWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsBW},  {0.86708881, 0.86708881, 0.86708881, 0.78347842, 0.78347842, 0.78347842, 0.55529176, 0.55529176, 0.55529176, 0.25, 0.25, 0.25});

    nd4j::ops::static_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &h0FW, &h0BW, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFWfinal = results->at(1);
    auto hBWfinal = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_bidir_rnn_test2) {

    const int bS         = 4;
    const int inSize     = 4;
    const int numUnitsFW = 3;
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::create<double>('c', {time, bS, inSize});
    auto WxFW = NDArrayFactory::create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::create<double>('c', {numUnitsFW, numUnitsFW});
    auto bFW  = NDArrayFactory::create<double>('c', {2*numUnitsFW});

    auto maxTimeStep = NDArrayFactory::create<double>('c', {bS}, {time-1, time-3, time-4, 0});

    x.linspace(0.01, 0.01);
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expH      = NDArrayFactory::create<double>('c', {time, bS, numUnitsFW+numUnitsBW}, {0.22602835, 0.22602835, 0.22602835, 0.86518273, 0.86518273,0.86518273,0.27105303, 0.27105303, 0.27105303, 0.66617761, 0.66617761,0.66617761,
                                                                       0.31492203, 0.31492203, 0.31492203, 0.31492203, 0.31492203,0.31492203,0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,
                                                                       0.60005558, 0.60005558, 0.60005558, 0.9029975 , 0.9029975 ,0.9029975 ,0.66138054, 0.66138054, 0.66138054, 0.43819931, 0.43819931,0.43819931,
                                                                       0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,
                                                                       0.87023975, 0.87023975, 0.87023975, 0.88852032, 0.88852032,0.88852032,0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,
                                                                       0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,
                                                                       0.95177305, 0.95177305, 0.95177305, 0.66737775, 0.66737775,0.66737775,0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,
                                                                       0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,
                                                                       0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.});

    auto expHFWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsFW},  {0.95177305, 0.95177305, 0.95177305, 0.66138054, 0.66138054, 0.66138054, 0.31492203, 0.31492203, 0.31492203, 0.        , 0.        , 0.});
    auto expHBWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsBW},  {0.86518273, 0.86518273, 0.86518273, 0.66617761, 0.66617761, 0.66617761, 0.31492203, 0.31492203, 0.31492203, 0.        , 0.        , 0.});

    nd4j::ops::static_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFWfinal = results->at(1);
    auto hBWfinal = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_bidir_rnn_test3) {

    const int bS         = 4;
    const int inSize     = 4;
    const int numUnitsFW = 3;
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::create<double>('c', {time, bS, inSize});
    auto WxFW = NDArrayFactory::create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::create<double>('c', {numUnitsFW, numUnitsFW});
    auto bFW  = NDArrayFactory::create<double>('c', {2*numUnitsFW});

    x.linspace(0.01, 0.01);
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expH      = NDArrayFactory::create<double>('c', {time, bS, numUnitsFW+numUnitsBW}, {0.22602835, 0.22602835, 0.22602835, 0.86841012, 0.86841012,0.86841012,0.27105303, 0.27105303, 0.27105303, 0.88207531, 0.88207531,0.88207531,
                                                                       0.31492203, 0.31492203, 0.31492203, 0.8941667 , 0.8941667 ,0.8941667 ,0.35748551, 0.35748551, 0.35748551, 0.90489713, 0.90489713,
                                                                       0.90489713, 0.60005558, 0.60005558, 0.60005558, 0.91381375, 0.91381375,0.91381375,0.66138054, 0.66138054, 0.66138054, 0.92253504, 0.92253504,
                                                                       0.92253504,0.71429879, 0.71429879, 0.71429879, 0.93027876, 0.93027876,0.93027876,0.75947891, 0.75947891, 0.75947891, 0.9371767 , 0.9371767 ,
                                                                       0.9371767 , 0.87023975, 0.87023975, 0.87023975, 0.94014274, 0.94014274,0.94014274,0.89680574, 0.89680574, 0.89680574, 0.94648926, 0.94648926,
                                                                       0.94648926,0.91657261, 0.91657261, 0.91657261, 0.95204779, 0.95204779,0.95204779,0.93146896, 0.93146896, 0.93146896, 0.95694206, 0.95694206,
                                                                       0.95694206, 0.95177305, 0.95177305, 0.95177305, 0.93773086, 0.93773086,0.93773086,0.95874689, 0.95874689, 0.95874689, 0.94579176, 0.94579176,
                                                                       0.94579176,0.96416067, 0.96416067, 0.96416067, 0.95267886, 0.95267886,0.95267886,0.96851506, 0.96851506, 0.96851506, 0.95857985, 0.95857985,
                                                                       0.95857985, 0.97269956, 0.97269956, 0.97269956, 0.76075293, 0.76075293,0.76075293,0.97557464, 0.97557464, 0.97557464, 0.78024637, 0.78024637,
                                                                       0.78024637,0.97806922, 0.97806922, 0.97806922, 0.79833344, 0.79833344,0.79833344,0.98026195, 0.98026195, 0.98026195, 0.81508646, 0.81508646,0.81508646});

    auto expHFWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsFW},  {0.97269956, 0.97269956, 0.97269956, 0.97557464, 0.97557464, 0.97557464, 0.97806922, 0.97806922, 0.97806922, 0.98026195, 0.98026195, 0.98026195});
    auto expHBWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsBW},  {0.86841012, 0.86841012, 0.86841012, 0.88207531, 0.88207531, 0.88207531, 0.8941667 , 0.8941667 , 0.8941667 , 0.90489713, 0.90489713, 0.90489713});

    nd4j::ops::static_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFWfinal = results->at(1);
    auto hBWfinal = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_rnn_test1) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::create<double>('c', {bS, numUnits});
    auto maxTimeStep = NDArrayFactory::create<Nd4jLong>('c', {bS}, {time-1, time-3});

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::create<double>('c', {time, bS, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0.69882484, 0.69882484, 0.69882484, 0.69882484,0.9312333 , 0.9312333 , 0.9312333 , 0.9312333 ,
                                                          0.93751527, 0.93751527, 0.93751527, 0.93751527,0.97136768, 0.97136768, 0.97136768, 0.97136768,0.        , 0.        , 0.        , 0.        ,
                                                          0.97732812, 0.97732812, 0.97732812, 0.97732812,0.    , 0.  , 0.  , 0. ,0.   , 0.  , 0.   , 0.  ,0.      , 0.        , 0.        , 0.        });

    auto expHFinal = NDArrayFactory::create<double>('c', {bS, numUnits},       {0.97732812, 0.97732812, 0.97732812, 0.97732812, 0.93751527, 0.93751527, 0.93751527, 0.93751527});

    nd4j::ops::dynamic_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0, &maxTimeStep}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_rnn_test2) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::create<double>('c', {bS, time, inSize});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::create<double>('c', {bS, numUnits});
    auto maxTimeStep = NDArrayFactory::create<int>('c', {bS}, {time-1, time});

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::create<double>('c', {bS, time, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0.92755601, 0.92755601, 0.92755601, 0.92755601,0.96778334, 0.96778334, 0.96778334,
                                                          0.96778334,0.97309129, 0.97309129, 0.97309129, 0.97309129,0.        , 0.        , 0.        , 0.        ,
                                                          0.75001965, 0.75001965, 0.75001965, 0.75001965,0.95449491, 0.95449491, 0.95449491, 0.95449491,0.97732828, 0.97732828, 0.97732828,
                                                          0.97732828,0.98000655, 0.98000655, 0.98000655, 0.98000655,0.98120782, 0.98120782, 0.98120782, 0.98120782});

    auto expHFinal = NDArrayFactory::create<double>('c', {bS, numUnits},       {0.97309129, 0.97309129, 0.97309129, 0.97309129, 0.98120782, 0.98120782, 0.98120782, 0.98120782});

    nd4j::ops::dynamic_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_rnn_test3) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::create<double>('c', {bS, time, inSize});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::create<double>('c', {bS, numUnits});

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::create<double>('c', {bS, time, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0.92755601, 0.92755601, 0.92755601, 0.92755601,0.96778334, 0.96778334, 0.96778334, 0.96778334,0.97309129,
                                                          0.97309129, 0.97309129, 0.97309129,0.97491207, 0.97491207, 0.97491207, 0.97491207,0.75001965, 0.75001965, 0.75001965, 0.75001965,0.95449491, 0.95449491,
                                                          0.95449491, 0.95449491,0.97732828, 0.97732828, 0.97732828, 0.97732828,0.98000655, 0.98000655, 0.98000655, 0.98000655,0.98120782, 0.98120782, 0.98120782, 0.98120782});

    auto expHFinal = NDArrayFactory::create<double>('c', {bS, numUnits},       {0.97491207, 0.97491207, 0.97491207, 0.97491207, 0.98120782, 0.98120782, 0.98120782, 0.98120782});

    nd4j::ops::dynamic_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_rnn_test4) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::create<double>('c', {bS, time, inSize});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits});
    auto maxTimeStep = NDArrayFactory::create<double>('c', {bS}, {time-1, time-4});

    x.linspace(0.01, 0.01);
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::create<double>('c', {bS, time, numUnits}, {0.47615493, 0.47615493, 0.47615493, 0.47615493,0.86347567, 0.86347567, 0.86347567, 0.86347567,0.96059545, 0.96059545,
                                                          0.96059545, 0.96059545,0.9724738 , 0.9724738 , 0.9724738 , 0.9724738 ,0.        , 0.        , 0.        , 0.        ,
                                                          0.57368608, 0.57368608, 0.57368608, 0.57368608,0. , 0. , 0  , 0. ,0., 0. , 0, 0.,0., 0., 0. , 0. ,0. , 0. , 0., 0. });

    auto expHFinal = NDArrayFactory::create<double>('c', {bS, numUnits},       {0.9724738 , 0.9724738 , 0.9724738 , 0.9724738 ,0.57368608, 0.57368608, 0.57368608, 0.57368608});

    nd4j::ops::dynamic_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_rnn_test5) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::create<double>('c', {bS, time, inSize});
    auto Wx = NDArrayFactory::create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::create<double>('c', {2*numUnits});

    x.linspace(0.01, 0.01);
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::create<double>('c', {bS, time, numUnits}, {0.47615493, 0.47615493, 0.47615493, 0.47615493,0.86347567, 0.86347567, 0.86347567, 0.86347567,0.96059545, 0.96059545, 0.96059545, 0.96059545,
                                                          0.9724738 , 0.9724738 , 0.9724738 , 0.9724738 ,0.97486307, 0.97486307, 0.97486307, 0.97486307,0.57368608, 0.57368608, 0.57368608, 0.57368608,
                                                          0.92135149, 0.92135149, 0.92135149, 0.92135149,0.97482354, 0.97482354, 0.97482354, 0.97482354,0.97984727, 0.97984727, 0.97984727, 0.97984727,
                                                          0.98119833, 0.98119833, 0.98119833, 0.98119833});

    auto expHFinal = NDArrayFactory::create<double>('c', {bS, numUnits},       {0.97486307, 0.97486307, 0.97486307, 0.97486307,0.98119833, 0.98119833, 0.98119833, 0.98119833});

    nd4j::ops::dynamic_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_bidir_rnn_test1) {

    const int bS         = 4;
    const int inSize     = 4;
    const int numUnitsFW = 3;
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::create<double>('c', {time, bS, inSize});
    auto WxFW = NDArrayFactory::create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::create<double>('c', {numUnitsFW, numUnitsFW});
    auto bFW  = NDArrayFactory::create<double>('c', {2*numUnitsFW});

    auto h0FW = NDArrayFactory::create<double>('c', {bS, numUnitsFW});
    auto h0BW = NDArrayFactory::create<double>('c', {bS, numUnitsBW});
    auto maxTimeStep = NDArrayFactory::create<int>('c', {bS}, {time-1, time-3, time-4, 0});

    x.linspace(0.01, 0.01);
    h0FW = 0.2;
    h0BW = 0.25;
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expHFW  = NDArrayFactory::create<double>('c', {time, bS, numUnitsFW}, {0.43819931, 0.43819931, 0.43819931,0.47615493, 0.47615493, 0.47615493,0.51241561, 0.51241561, 0.51241561,0.        , 0.        , 0.        ,
                                                          0.73880324, 0.73880324, 0.73880324,0.77843476, 0.77843476, 0.77843476,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.9052501 , 0.9052501 , 0.9052501 ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.9555734 , 0.9555734 , 0.9555734 ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });

    auto expHBW  = NDArrayFactory::create<double>('c', {time, bS, numUnitsBW}, {0.86708881, 0.86708881, 0.86708881,0.78347842, 0.78347842, 0.78347842,0.55529176, 0.55529176, 0.55529176,0.        , 0.        , 0.        ,
                                                          0.90935605, 0.90935605, 0.90935605,0.64692945, 0.64692945, 0.64692945,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.9181592 , 0.9181592 , 0.9181592 ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.8026439 , 0.8026439 , 0.8026439 ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });

    auto expHFWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsFW},  {0.9555734 , 0.9555734 , 0.9555734 , 0.77843476, 0.77843476, 0.77843476, 0.51241561, 0.51241561, 0.51241561, 0.2       , 0.2       , 0.2});
    auto expHBWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsBW},  {0.86708881, 0.86708881, 0.86708881, 0.78347842, 0.78347842, 0.78347842, 0.55529176, 0.55529176, 0.55529176, 0.25      , 0.25      , 0.25});

    nd4j::ops::dynamic_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &h0FW, &h0BW, &maxTimeStep}, {}, {1}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto hFW = results->at(0);
    auto hBW = results->at(1);
    auto hFWfinal = results->at(2);
    auto hBWfinal = results->at(3);

    ASSERT_TRUE(expHFW.isSameShape(hFW));
    ASSERT_TRUE(expHFW.equalsTo(hFW));
    ASSERT_TRUE(expHBW.isSameShape(hBW));
    ASSERT_TRUE(expHBW.equalsTo(hBW));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_bidir_rnn_test2) {

    const int bS         = 4;
    const int inSize     = 4;
    const int numUnitsFW = 3;
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::create<double>('c', {bS, time, inSize});
    auto WxFW = NDArrayFactory::create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::create<double>('c', {numUnitsFW, numUnitsFW});
    auto bFW  = NDArrayFactory::create<double>('c', {2*numUnitsFW});

    auto h0FW = NDArrayFactory::create<double>('c', {bS, numUnitsFW});
    auto h0BW = NDArrayFactory::create<double>('c', {bS, numUnitsBW});
    auto maxTimeStep = NDArrayFactory::create<int>('c', {bS}, {time-1, time-3, time-4, 0});

    x.linspace(0.01, 0.01);
    h0FW = 0.2;
    h0BW = 0.25;
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expHFW  = NDArrayFactory::create<double>('c', {bS, time, numUnitsFW}, {0.43819931, 0.43819931, 0.43819931,0.66617761, 0.66617761, 0.66617761,0.80944357, 0.80944357, 0.80944357,0.87294706, 0.87294706, 0.87294706,0.        , 0.        , 0.        ,
                                                          0.61067683, 0.61067683, 0.61067683,0.84851124, 0.84851124, 0.84851124,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.73978305, 0.73978305, 0.73978305,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });

    auto expHBW  = NDArrayFactory::create<double>('c', {bS, time, numUnitsBW}, {0.84345207, 0.84345207, 0.84345207,0.83584708, 0.83584708, 0.83584708,0.77435951, 0.77435951, 0.77435951,0.58760492, 0.58760492, 0.58760492,0.        , 0.        , 0.        ,
                                                          0.85615841, 0.85615841, 0.85615841,0.67397984, 0.67397984, 0.67397984,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.76576202, 0.76576202, 0.76576202,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });

    auto expHFWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsFW},  {0.87294706, 0.87294706, 0.87294706,0.84851124, 0.84851124, 0.84851124,0.73978305, 0.73978305, 0.73978305,0.2       , 0.2       , 0.2});
    auto expHBWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsBW},  {0.84345207, 0.84345207, 0.84345207, 0.85615841, 0.85615841, 0.85615841, 0.76576202, 0.76576202, 0.76576202, 0.25      , 0.25      , 0.25});

    nd4j::ops::dynamic_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &h0FW, &h0BW, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto hFW = results->at(0);
    auto hBW = results->at(1);
    auto hFWfinal = results->at(2);
    auto hBWfinal = results->at(3);

    ASSERT_TRUE(expHFW.isSameShape(hFW));
    ASSERT_TRUE(expHFW.equalsTo(hFW));
    ASSERT_TRUE(expHBW.isSameShape(hBW));
    ASSERT_TRUE(expHBW.equalsTo(hBW));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_bidir_rnn_test3) {

    const int bS         = 4;
    const int inSize     = 4;
    const int numUnitsFW = 3;
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::create<double>('c', {bS, time, inSize});
    auto WxFW = NDArrayFactory::create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::create<double>('c', {numUnitsFW, numUnitsFW});
    auto bFW  = NDArrayFactory::create<double>('c', {2*numUnitsFW});

    auto maxTimeStep = NDArrayFactory::create<int>('c', {bS}, {time-1, time-3, time-4, 0});

    x.linspace(0.01, 0.01);
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expHFW  = NDArrayFactory::create<double>('c', {bS, time, numUnitsFW}, {0.22602835, 0.22602835, 0.22602835,0.49994591, 0.49994591, 0.49994591,0.72869307, 0.72869307, 0.72869307,0.84784327, 0.84784327, 0.84784327,0.        , 0.        , 0.        ,
                                                          0.43819931, 0.43819931, 0.43819931,0.7793996 , 0.7793996 , 0.7793996 ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.61067683, 0.61067683, 0.61067683,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });

    auto expHBW  = NDArrayFactory::create<double>('c', {bS, time, numUnitsBW}, {0.82273707, 0.82273707, 0.82273707,0.77935851, 0.77935851, 0.77935851,0.6381121 , 0.6381121 , 0.6381121 ,0.35748551, 0.35748551, 0.35748551,0.        , 0.        , 0.        ,
                                                          0.77843476, 0.77843476, 0.77843476,0.47615493, 0.47615493, 0.47615493,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.61067683, 0.61067683, 0.61067683,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });

    auto expHFWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsFW},  {0.84784327, 0.84784327, 0.84784327, 0.7793996 , 0.7793996 , 0.7793996 , 0.61067683, 0.61067683, 0.61067683, 0.        , 0.        , 0.});
    auto expHBWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsBW},  {0.82273707, 0.82273707, 0.82273707, 0.77843476, 0.77843476, 0.77843476, 0.61067683, 0.61067683, 0.61067683, 0.        , 0.        , 0.});

    nd4j::ops::dynamic_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto hFW = results->at(0);
    auto hBW = results->at(1);
    auto hFWfinal = results->at(2);
    auto hBWfinal = results->at(3);

    ASSERT_TRUE(expHFW.isSameShape(hFW));
    ASSERT_TRUE(expHFW.equalsTo(hFW));
    ASSERT_TRUE(expHBW.isSameShape(hBW));
    ASSERT_TRUE(expHBW.equalsTo(hBW));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_bidir_rnn_test4) {

    const int bS         = 4;
    const int inSize     = 4;
    const int numUnitsFW = 3;
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::create<double>('c', {bS, time, inSize});
    auto WxFW = NDArrayFactory::create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::create<double>('c', {numUnitsFW, numUnitsFW});
    auto bFW  = NDArrayFactory::create<double>('c', {2*numUnitsFW});

    auto h0FW = NDArrayFactory::create<double>('c', {bS, numUnitsFW});
    auto h0BW = NDArrayFactory::create<double>('c', {bS, numUnitsBW});

    x.linspace(0.01, 0.01);
    h0FW = 0.2;
    h0BW = 0.25;
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expHFW  = NDArrayFactory::create<double>('c', {bS, time, numUnitsFW}, {0.43819931, 0.43819931, 0.43819931,0.66617761, 0.66617761, 0.66617761,0.80944357, 0.80944357, 0.80944357,0.87294706, 0.87294706, 0.87294706,0.89948899, 0.89948899, 0.89948899,
                                                          0.61067683, 0.61067683, 0.61067683,0.84851124, 0.84851124, 0.84851124,0.91925737, 0.91925737, 0.91925737,0.93751395, 0.93751395, 0.93751395,0.94544483, 0.94544483, 0.94544483,
                                                          0.73978305, 0.73978305, 0.73978305,0.92827068, 0.92827068, 0.92827068,0.95791111, 0.95791111, 0.95791111,0.96427356, 0.96427356, 0.96427356,0.96797541, 0.96797541, 0.96797541,
                                                          0.83057887, 0.83057887, 0.83057887,0.96365083, 0.96365083, 0.96365083,0.97585698, 0.97585698, 0.97585698,0.97866981, 0.97866981, 0.97866981,0.9807326 , 0.9807326 , 0.9807326 });

    auto expHBW  = NDArrayFactory::create<double>('c', {bS, time, numUnitsBW}, {0.85301722, 0.85301722, 0.85301722,0.86427295, 0.86427295, 0.86427295,0.8599919 , 0.8599919 , 0.8599919 ,0.80609463, 0.80609463, 0.80609463,0.61814662, 0.61814662, 0.61814662,
                                                          0.91888753, 0.91888753, 0.91888753,0.92652672, 0.92652672, 0.92652672,0.92939674, 0.92939674, 0.92939674,0.90661931, 0.90661931, 0.90661931,0.74516764, 0.74516764, 0.74516764,
                                                          0.95254269, 0.95254269, 0.95254269,0.95710717, 0.95710717, 0.95710717,0.96021584, 0.96021584, 0.96021584,0.95222547, 0.95222547, 0.95222547,0.83426363, 0.83426363, 0.83426363,
                                                          0.97154357, 0.97154357, 0.97154357,0.97424915, 0.97424915, 0.97424915,0.97644817, 0.97644817, 0.97644817,0.97410547, 0.97410547, 0.97410547,0.89409962, 0.89409962, 0.89409962});

    auto expHFWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsFW},  {0.89948899, 0.89948899, 0.89948899, 0.94544483, 0.94544483, 0.94544483, 0.96797541, 0.96797541, 0.96797541, 0.9807326 , 0.9807326 , 0.9807326 });
    auto expHBWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsBW},  {0.85301722, 0.85301722, 0.85301722, 0.91888753, 0.91888753, 0.91888753, 0.95254269, 0.95254269, 0.95254269, 0.97154357, 0.97154357, 0.97154357});

    nd4j::ops::dynamic_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &h0FW, &h0BW}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto hFW = results->at(0);
    auto hBW = results->at(1);
    auto hFWfinal = results->at(2);
    auto hBWfinal = results->at(3);

    ASSERT_TRUE(expHFW.isSameShape(hFW));
    ASSERT_TRUE(expHFW.equalsTo(hFW));
    ASSERT_TRUE(expHBW.isSameShape(hBW));
    ASSERT_TRUE(expHBW.equalsTo(hBW));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

TEST_F(DeclarableOpsTests6, dynamic_bidir_rnn_test5) {

    const int bS         = 4;
    const int inSize     = 4;
    const int numUnitsFW = 3;
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::create<double>('c', {bS, time, inSize});
    auto WxFW = NDArrayFactory::create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::create<double>('c', {numUnitsFW, numUnitsFW});
    auto bFW  = NDArrayFactory::create<double>('c', {2*numUnitsFW});

    x.linspace(0.01, 0.01);
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expHFW  = NDArrayFactory::create<double>('c', {bS, time, numUnitsFW}, {0.22602835, 0.22602835, 0.22602835,0.49994591, 0.49994591, 0.49994591,0.72869307, 0.72869307, 0.72869307,0.84784327, 0.84784327, 0.84784327,0.89357928, 0.89357928, 0.89357928,
                                                          0.43819931, 0.43819931, 0.43819931,0.7793996 , 0.7793996 , 0.7793996 ,0.9053792 , 0.9053792 , 0.9053792 ,0.93546593, 0.93546593, 0.93546593,0.94518339, 0.94518339, 0.94518339,
                                                          0.61067683, 0.61067683, 0.61067683,0.90347408, 0.90347408, 0.90347408,0.95538786, 0.95538786, 0.95538786,0.96406045, 0.96406045, 0.96406045,0.96795929, 0.96795929, 0.96795929,
                                                          0.73978305, 0.73978305, 0.73978305,0.95499984, 0.95499984, 0.95499984,0.97535671, 0.97535671, 0.97535671,0.97864446, 0.97864446, 0.97864446,0.98073144, 0.98073144, 0.98073144});

    auto expHBW  = NDArrayFactory::create<double>('c', {bS, time, numUnitsBW}, {0.84882345, 0.84882345, 0.84882345,0.85160683, 0.85160683, 0.85160683,0.81997657, 0.81997657, 0.81997657,0.69228829, 0.69228829, 0.69228829,0.39861399, 0.39861399, 0.39861399,
                                                          0.91865453, 0.91865453, 0.91865453,0.92528094, 0.92528094, 0.92528094,0.92212167, 0.92212167, 0.92212167,0.86418213, 0.86418213, 0.86418213,0.57969286, 0.57969286, 0.57969286,
                                                          0.95252666, 0.95252666, 0.95252666,0.95696305, 0.95696305, 0.95696305,0.95878749, 0.95878749, 0.95878749,0.93722463, 0.93722463, 0.93722463,0.71727031, 0.71727031, 0.71727031,
                                                          0.97154234, 0.97154234, 0.97154234,0.97423089, 0.97423089, 0.97423089,0.976149  , 0.976149  , 0.976149  ,0.96878298, 0.96878298, 0.96878298,0.81508646, 0.81508646, 0.81508646});

    auto expHFWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsFW},  {0.89357928, 0.89357928, 0.89357928, 0.94518339, 0.94518339, 0.94518339, 0.96795929, 0.96795929, 0.96795929, 0.98073144, 0.98073144, 0.98073144});
    auto expHBWfinal = NDArrayFactory::create<double>('c', {bS, numUnitsBW},  {0.84882345, 0.84882345, 0.84882345, 0.91865453, 0.91865453, 0.91865453, 0.95252666, 0.95252666, 0.95252666, 0.97154234, 0.97154234, 0.97154234});

    nd4j::ops::dynamic_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto hFW = results->at(0);
    auto hBW = results->at(1);
    auto hFWfinal = results->at(2);
    auto hBWfinal = results->at(3);

    ASSERT_TRUE(expHFW.isSameShape(hFW));
    ASSERT_TRUE(expHFW.equalsTo(hFW));
    ASSERT_TRUE(expHBW.isSameShape(hBW));
    ASSERT_TRUE(expHBW.equalsTo(hBW));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}


TEST_F(DeclarableOpsTests6, Test_Diag_119_1) {
    auto x = NDArrayFactory::create<double>('c', {3}, {0.15f, 0.25f, 0.35f});
    auto e = NDArrayFactory::create<double>('c', {3, 3}, {0.15f, 0.0f, 0.0f,   0.0f, 0.25f, 0.0f,   0.0f, 0.0f, 0.35f});

    nd4j::ops::diag op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Diag_119_2) {
    auto x = NDArrayFactory::create<double>('c', {1}, {0.15f});
    auto e = NDArrayFactory::create<double>('c', {1, 1}, {0.15f});

    nd4j::ops::diag op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Diag_119_3) {
    auto x = NDArrayFactory::create<double>(0.15f);
    auto e = NDArrayFactory::create<double>('c', {1, 1}, {0.15f});

    nd4j::ops::diag op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}


