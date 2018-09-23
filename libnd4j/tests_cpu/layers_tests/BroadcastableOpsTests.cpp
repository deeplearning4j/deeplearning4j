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
// Created by raver119 on 23.11.17.
//


#include "testlayers.h"
#include <Graph.h>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::graph;

class BroadcastableOpsTests : public testing::Test {
public:

};

TEST_F(BroadcastableOpsTests, Test_Add_1) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    auto y = NDArrayFactory::_create<float>('c', {1, 5});
    auto exp = NDArrayFactory::_create<float>('c', {5, 5});
    x.linspace(1);
    y.linspace(1);
    exp.linspace(1);

    exp.applyBroadcast(broadcast::Add, {1}, &y);


    nd4j::ops::add op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_Multiply_1) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    auto y = NDArrayFactory::_create<float>('c', {1, 5});
    auto exp = NDArrayFactory::_create<float>('c', {5, 5});
    x.linspace(1);
    y.linspace(1);
    exp.linspace(1);

    exp.applyBroadcast(broadcast::Multiply, {1}, &y);


    nd4j::ops::multiply op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_SquaredSubtract_1) {
    auto x = NDArrayFactory::_create<float>('c', {5, 5});
    auto y  = NDArrayFactory::_create<float>('c', {1, 5});
    auto exp = NDArrayFactory::_create<float>('c', {5, 5});
    x.linspace(1);
    y.linspace(1);
    exp.linspace(1);

    exp.applyBroadcast(broadcast::SquaredSubtract, {1}, &y);


    nd4j::ops::squaredsubtract op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_ScalarBroadcast_1) {
    auto x = NDArrayFactory::_create<float>('c', {1, 1}, {1});
    auto y = NDArrayFactory::_create<float>('c', {1, 3}, {0, 1, 2});
    auto exp = NDArrayFactory::_create<float>('c', {1,3}, {1, 0, -1});

    nd4j::ops::subtract op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_ScalarBroadcast_2) {
    auto x = NDArrayFactory::_create<float>('c', {1, 1}, {1});
    auto y = NDArrayFactory::_create<float>('c', {1, 3}, {0, 1, 2});
    auto exp = NDArrayFactory::_create<float>('c', {1,3}, {1, 2, 3});

    nd4j::ops::add op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_Maximum_1) {
    auto x = NDArrayFactory::_create<float>('c', {2, 3}, {1, 2, 1, 2, 3, 2});
    auto row = NDArrayFactory::_create<float>('c', {1, 3}, {2, 2, 2});
    auto exp = NDArrayFactory::_create<float>('c', {2, 3}, {2, 2, 2, 2, 3, 2});

    nd4j::ops::maximum op;
    auto result = op.execute({&x, &row}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_Minimum_1) {
    auto x = NDArrayFactory::_create<float>('c', {2, 3}, {1, 2, 1, 2, 3, 2});
    auto col = NDArrayFactory::_create<float>('c', {2, 1}, {2, 1});
    auto exp = NDArrayFactory::_create<float>('c', {2, 3}, {1, 2, 1, 1, 1, 1});

    nd4j::ops::minimum op;
    auto result = op.execute({&x, &col}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_Shape_1) {
    nd4j::ops::minimum op;

    Nd4jLong shapeX[] = {2, 2, 5, 5, 1, 8192, 1, 99};
    Nd4jLong shapeY[] = {2, 2, 5, 5, 1, 8192, 1, 99};
    ShapeList inputShape({shapeX, shapeY});
    VariableSpace vs;
    Context ctx(1, &vs, false);

    auto shapes = op.calculateOutputShape(&inputShape, ctx);

    auto shapeZ = shapes->at(0);
    ASSERT_TRUE(shape::shapeEquals(shapeX, shapeZ));

    shapes->destroy();
    delete shapes;
}

TEST_F(BroadcastableOpsTests, Test_Shape_2) {
    nd4j::ops::minimum op;

    Nd4jLong shapeX[] = {2, 1, 1, 1, 1, 8192, 1, 99};
    Nd4jLong shapeY[] = {2, 2, 5, 5, 1, 8192, 1, 99};
    ShapeList inputShape({shapeX, shapeY});
    VariableSpace vs;
    Context ctx(1, &vs, false);

    auto shapes = op.calculateOutputShape(&inputShape, ctx);

    auto shapeZ = shapes->at(0);
    ASSERT_TRUE(shape::shapeEquals(shapeY, shapeZ));

    shapes->destroy();
    delete shapes;
}


TEST_F(BroadcastableOpsTests, Test_Shape_3) {
    nd4j::ops::minimum op;

    Nd4jLong shapeX[] = {2, 5, 3, 1, 1, 8192, 1, 99};
    Nd4jLong shapeY[] = {2, 1, 3, 3, 1, 8192, 1, 99};
    ShapeList inputShape({shapeX, shapeY});
    VariableSpace vs;
    Context ctx(1, &vs, false);

    auto shapes = op.calculateOutputShape(&inputShape, ctx);

    auto shapeZ = shapes->at(0);
    ASSERT_TRUE(shape::shapeEquals(shapeX, shapeZ));

    shapes->destroy();
    delete shapes;
}


TEST_F(BroadcastableOpsTests, Test_Shape_4) {
    nd4j::ops::minimum op;

    Nd4jLong shapeX[] = {2, 5, 3, 1, 1, 8192, 1, 99};
    Nd4jLong shapeY[] = {2, 5, 1, 1, 1, 8192, 1, 99};
    ShapeList inputShape({shapeX, shapeY});
    VariableSpace vs;
    Context ctx(1, &vs, false);

    auto shapes = op.calculateOutputShape(&inputShape, ctx);

    auto shapeZ = shapes->at(0);
    ASSERT_TRUE(shape::shapeEquals(shapeX, shapeZ));

    shapes->destroy();
    delete shapes;
}

// (2,1,3) + (4,3) = (2,4,3)

TEST_F(BroadcastableOpsTests, Test_Shape_5) {
    nd4j::ops::minimum op;

    Nd4jLong shapeX[] = {3, 2, 1, 3, 3, 3, 1, 8192, 1, 99};
    Nd4jLong shapeY[] = {2, 4, 3, 3, 1, 8192, 1, 99};
    Nd4jLong shapeE[] = {3, 2, 4, 3, 12, 3, 1, 8192, 1, 99};
    ShapeList inputShape({shapeX, shapeY});
    VariableSpace vs;
    Context ctx(1, &vs, false);

    auto shapes = op.calculateOutputShape(&inputShape, ctx);

    auto shapeZ = shapes->at(0);
    ASSERT_TRUE(shape::shapeEquals(shapeE, shapeZ));

    shapes->destroy();
    delete shapes;
}

TEST_F(BroadcastableOpsTests, Test_Scalar_Add_1) {
    auto x = NDArrayFactory::_create<float>('c', {2, 2}, {1, 2, 3, 4});
    auto y  = NDArrayFactory::_scalar<float>(2.0f);
    auto exp = NDArrayFactory::_create<float>('c', {2, 2}, {3, 4, 5, 6});

    nd4j::ops::add op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(BroadcastableOpsTests, Test_Inplace_Output_1) {
    auto x = NDArrayFactory::_create<float>('c', {2, 1, 3});
    auto y = NDArrayFactory::_create<float>('c', {4, 3});
    auto o = NDArrayFactory::_create<float>('c', {2, 4, 3});
    auto e = NDArrayFactory::_create<float>('c', {2, 4, 3});
    auto buffO1 = reinterpret_cast<float *>(o.buffer());
    y.assign(1.0f);
    e.assign(1.0f);

    nd4j::ops::add op;
    auto result = op.execute({&x, &y}, {&o}, {}, {});
    ASSERT_EQ(Status::OK(), result);

    auto buffO2 = reinterpret_cast<float *>(o.buffer());

    ASSERT_TRUE(e.isSameShape(o));
    ASSERT_TRUE(e.equalsTo(o));

    ASSERT_TRUE(buffO1 == buffO2);
}

TEST_F(BroadcastableOpsTests, Test_Subtract_1) {

    auto x = NDArrayFactory::_scalar<float>(1.0f);
    auto y = NDArrayFactory::_create<float>('c', {2}, {0.0f, 1.0f});
    auto e = NDArrayFactory::_create<float>('c', {2}, {1.0f, 0.0f});

    auto z = x - y;

    ASSERT_TRUE(e.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Subtract_2) {
    auto x = NDArrayFactory::_create<float>(1.0f);
    auto y = NDArrayFactory::_create<float>('c', {2}, {0.0f, 1.0f});
    auto e = NDArrayFactory::_create<float>('c', {2}, {1.0f, 0.0f});

    nd4j::ops::subtract op;
    auto result = op.execute({&x, &y}, {}, {});
    auto z = result->at(0);

    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

TEST_F(BroadcastableOpsTests, Test_Subtract_3) {
    auto x = NDArrayFactory::_create<float>(1.0f);
    auto y = NDArrayFactory::_create<float>('c', {2}, {0.0f, 1.0f});
    auto z = NDArrayFactory::_create<float>('c', {2}, {0.0f, 0.0f});
    auto e = NDArrayFactory::_create<float>('c', {2}, {1.0f, 0.0f});

    nd4j::ops::subtract op;
    auto result = op.execute({&x, &y}, {&z}, {}, {});

    ASSERT_EQ(Status::OK(), result);
    ASSERT_TRUE(e.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Subtract_4) {
    auto x = NDArrayFactory::_create<float>(1.0f);
    auto y = NDArrayFactory::_create<float>('c', {2}, {0.0f, 1.0f});
    auto e = NDArrayFactory::_create<float>('c', {2}, {1.0f, 0.0f});

    auto z = x.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), y);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Subtract_5) {
    auto x = NDArrayFactory::_create<float>(1.0f);
    auto y = NDArrayFactory::_create<float>('c', {2}, {0.0f, 1.0f});
    auto e = NDArrayFactory::_create<float>('c', {2}, {-1., 0.});

    auto z = y - x;

    ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Subtract_6) {
    auto x = NDArrayFactory::_create<float>(1.0f);
    auto y = NDArrayFactory::_create<float>(4.f);
    auto e = NDArrayFactory::_create<float>(3.f);

    auto z = y - x;

    ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Subtract_7) {
    auto x = NDArrayFactory::_create<float>(1.0f);
    auto y = NDArrayFactory::_create<float>(4.f);
    auto e = NDArrayFactory::_create<float>(-3.f);

    auto z = x - y;

    ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Add_2) {

    auto x = NDArrayFactory::_create<float>(1.0f);
    auto y = NDArrayFactory::_create<float>('c', {2}, {0.0f, 1.0f});
    auto e = NDArrayFactory::_create<float>('c', {2}, {1.f, 2.f});

    auto z = x + y;

    ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Add_3) {

    auto x = NDArrayFactory::_create<float>(1.0f);
    auto y = NDArrayFactory::_create<float>('c', {2}, {0.0f, 1.0f});
    auto e = NDArrayFactory::_create<float>('c', {2}, {1.f, 2.f});

    auto z = y + x;

    ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Add_4) {
    
    auto x = NDArrayFactory::_create<float>(1.0f);
    auto y = NDArrayFactory::_create<float>(4.f);
    auto e = NDArrayFactory::_create<float>(5.f);

    auto z = x + y;

    ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Add_5) {
    
    auto x = NDArrayFactory::_create<float>(1.0f);
    auto y = NDArrayFactory::_create<float>(4.f);
    auto e = NDArrayFactory::_create<float>(5.f);

    auto z = y + x;

    ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Multiply_2) {
    
    auto x = NDArrayFactory::_create<float>(2.0f);
    auto y = NDArrayFactory::_create<float>('c', {2}, {3.f, 4.f});
    auto e = NDArrayFactory::_create<float>('c', {2}, {6.f, 8.f});

    auto z = y * x;

    ASSERT_TRUE(e.equalsTo(z));
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Multiply_3) {
    
    auto x = NDArrayFactory::_create<float>(2.0f);
    auto y = NDArrayFactory::_create<float>('c', {2}, {3.f, 4.f});
    auto e = NDArrayFactory::_create<float>('c', {2}, {6.f, 8.f});

    auto z = x * y;

    ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Multiply_4) {
    
    auto x = NDArrayFactory::_create<float>(2.0f);
    auto y = NDArrayFactory::_create<float>(4.f);
    auto e = NDArrayFactory::_create<float>(8.f);

    auto z = y * x;

    ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Multiply_5) {
    
    auto x = NDArrayFactory::_create<float>(2.0f);
    auto y = NDArrayFactory::_create<float>(4.f);
    auto e = NDArrayFactory::_create<float>(8.f);

    auto z = x * y;

    ASSERT_TRUE(e.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Multiply_6) {
    auto x = NDArrayFactory::_create<float>(2.0f);
    auto y = NDArrayFactory::_create<float>('c', {1}, {4.f});
    auto e = NDArrayFactory::_create<float>('c', {1}, {8.f});

    auto z = x * y;

    ASSERT_TRUE(e.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Multiply_7) {
    auto x = NDArrayFactory::_create<float>(2.0f);
    auto y = NDArrayFactory::_create<float>('c', {1}, {4.f});
    auto e = NDArrayFactory::_create<float>('c', {1}, {8.f});

    nd4j::ops::multiply op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

TEST_F(BroadcastableOpsTests, Test_Multiply_8) {
    auto x = NDArrayFactory::_create<float>(2.0f);
    auto y = NDArrayFactory::_create<float>('c', {1, 1}, {4.f});
   auto e = NDArrayFactory::_create<float>('c', {1, 1}, {8.f});

    nd4j::ops::multiply op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}