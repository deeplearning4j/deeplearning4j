/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <graph/Graph.h>
#include <graph/Node.h>
#include <ops/declarable/CustomOperations.h>
#include "testlayers.h"

using namespace sd;
using namespace sd::graph;



class BroadcastableOpsTests : public NDArrayTests {
 public:
};

TEST_F(BroadcastableOpsTests, Test_Add_1) {
  NDArray x('c', {5, 5}, FLOAT32);
  NDArray y('c', {1, 5}, FLOAT32);
  NDArray exp('c', {5, 5}, FLOAT32);
  x.linspace(1);
  y.linspace(1);
  exp.linspace(1);
  std::vector<LongType> dims = {1};


  exp.applyBroadcast(broadcast::Add, &dims, y, exp);

  ops::add op;
  auto result = op.evaluate({&x, &y});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);


ASSERT_EQ(exp,*z);
}

TEST_F(BroadcastableOpsTests, Test_Multiply_1) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  auto y = NDArrayFactory::create<float>('c', {1, 5});
  auto exp = NDArrayFactory::create<float>('c', {5, 5});
  x.linspace(1);
  y.linspace(1);
  exp.linspace(1);
  std::vector<LongType> dims = {1};

  exp.applyBroadcast(broadcast::Multiply, &dims, y, exp);

  ops::multiply op;
  auto result = op.evaluate({&x, &y});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

ASSERT_EQ(exp,*z);
}

TEST_F(BroadcastableOpsTests, Test_SquaredSubtract_1) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  auto y = NDArrayFactory::create<float>('c', {1, 5});
  auto exp = NDArrayFactory::create<float>('c', {5, 5});
  x.linspace(1);
  y.linspace(1);
  exp.linspace(1);

  std::vector<LongType> dims = {1};
  exp.applyBroadcast(broadcast::SquaredSubtract, &dims, y, exp);

  ops::squaredsubtract op;
  auto result = op.evaluate({&x, &y});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

ASSERT_EQ(exp,*z);
}

TEST_F(BroadcastableOpsTests, Test_ScalarBroadcast_1) {
  auto x = NDArrayFactory::create<float>('c', {1, 1}, {1});
  auto y = NDArrayFactory::create<float>('c', {1, 3}, {0, 1, 2});
  auto exp = NDArrayFactory::create<float>('c', {1, 3}, {1, 0, -1});

  ops::subtract op;
  auto result = op.evaluate({&x, &y});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

ASSERT_EQ(exp,*z);
}

TEST_F(BroadcastableOpsTests, Test_ScalarBroadcast_2) {
  auto x = NDArrayFactory::create<float>('c', {1, 1}, {1});
  auto y = NDArrayFactory::create<float>('c', {1, 3}, {0, 1, 2});
  auto exp = NDArrayFactory::create<float>('c', {1, 3}, {1, 2, 3});

  ops::add op;
  auto result = op.evaluate({&x, &y});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

ASSERT_EQ(exp,*z);
}

TEST_F(BroadcastableOpsTests, Test_Maximum_1) {
  auto x = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 1, 2, 3, 2});
  auto row = NDArrayFactory::create<float>('c', {1, 3}, {2, 2, 2});
  auto exp = NDArrayFactory::create<float>('c', {2, 3}, {2, 2, 2, 2, 3, 2});

  ops::maximum op;
  auto result = op.evaluate({&x, &row});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

ASSERT_EQ(exp,*z);
}

TEST_F(BroadcastableOpsTests, Test_Minimum_1) {
  auto x = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 1, 2, 3, 2});
  auto col = NDArrayFactory::create<float>('c', {2, 1}, {2, 1});
  auto exp = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 1, 1, 1, 1});

  ops::minimum op;
  auto result = op.evaluate({&x, &col});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

  ASSERT_TRUE(exp.isSameShape(z));

  ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Shape_1) {
  ops::minimum op;

  LongType shapeX[] = {2, 2, 5, 5, 1, 8192, 1, 99};
  LongType shapeY[] = {2, 2, 5, 5, 1, 8192, 1, 99};
  ShapeList inputShape({shapeX, shapeY});
  VariableSpace vs;
  Context ctx(1, &vs, false);

  auto shapes = op.calculateOutputShape(&inputShape, ctx);

  auto shapeZ = shapes->at(0);
  ASSERT_TRUE(shape::shapeEquals(shapeX, shapeZ));

  delete shapes;
}

TEST_F(BroadcastableOpsTests, Test_Shape_2) {
  ops::minimum op;

  const LongType shapeX[] = {2, 1, 1, 1, 1, 8192, 1, 99};
  const LongType shapeY[] = {2, 2, 5, 5, 1, 8192, 1, 99};
  ShapeList inputShape({shapeX, shapeY});
  VariableSpace vs;
  Context ctx(1, &vs, false);

  auto shapes = op.calculateOutputShape(&inputShape, ctx);

  auto shapeZ = shapes->at(0);
  ASSERT_TRUE(shape::shapeEquals(shapeY, shapeZ));

  delete shapes;
}

TEST_F(BroadcastableOpsTests, Test_Shape_3) {
  ops::minimum op;

  const LongType shapeX[] = {2, 5, 3, 1, 1, 8192, 1, 99};
  const LongType shapeY[] = {2, 1, 3, 3, 1, 8192, 1, 99};
  ShapeList inputShape({shapeX, shapeY});
  VariableSpace vs;
  Context ctx(1, &vs, false);

  auto shapes = op.calculateOutputShape(&inputShape, ctx);

  auto shapeZ = shapes->at(0);
  ASSERT_TRUE(shape::shapeEquals(shapeX, shapeZ));

  delete shapes;
}

TEST_F(BroadcastableOpsTests, Test_Shape_4) {
  ops::minimum op;

  const LongType shapeX[] = {2, 5, 3, 1, 1, 8192, 1, 99};
  const LongType shapeY[] = {2, 5, 1, 1, 1, 8192, 1, 99};
  ShapeList inputShape({shapeX, shapeY});
  VariableSpace vs;
  Context ctx(1, &vs, false);

  auto shapes = op.calculateOutputShape(&inputShape, ctx);

  auto shapeZ = shapes->at(0);
  ASSERT_TRUE(shape::shapeEquals(shapeX, shapeZ));

  delete shapes;
}

// (2,1,3) + (4,3) = (2,4,3)

TEST_F(BroadcastableOpsTests, Test_Shape_5) {
  ops::minimum op;

  const LongType shapeX[] = {3, 2, 1, 3, 3, 3, 1, 8192, 1, 99};
  const LongType shapeY[] = {2, 4, 3, 3, 1, 8192, 1, 99};
  const LongType shapeE[] = {3, 2, 4, 3, 12, 3, 1, 8192, 1, 99};
  ShapeList inputShape({shapeX, shapeY});
  VariableSpace vs;
  Context ctx(1, &vs, false);

  auto shapes = op.calculateOutputShape(&inputShape, ctx);

  auto shapeZ = shapes->at(0);
  ASSERT_TRUE(shape::shapeEquals(shapeE, shapeZ));

  delete shapes;
}

TEST_F(BroadcastableOpsTests, Test_Scalar_Add_1) {
  auto x = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
  auto y = NDArrayFactory::create<float>(2.0f);
  auto exp = NDArrayFactory::create<float>('c', {2, 2}, {3, 4, 5, 6});

  ops::add op;
  auto result = op.evaluate({&x, &y});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

ASSERT_EQ(exp,*z);
}

TEST_F(BroadcastableOpsTests, Test_Inplace_Output_1) {
  auto x = NDArrayFactory::create<float>('c', {2, 1, 3});
  auto y = NDArrayFactory::create<float>('c', {4, 3});
  auto o = NDArrayFactory::create<float>('c', {2, 4, 3});
  auto e = NDArrayFactory::create<float>('c', {2, 4, 3});
  auto buffO1 = reinterpret_cast<float *>(o.buffer());
  y.assign(1.0f);
  e.assign(1.0f);

  ops::add op;
  auto result = op.execute({&x, &y}, {&o}, {}, {}, {});
  ASSERT_EQ(sd::Status::OK, result);

  auto buffO2 = reinterpret_cast<float *>(o.buffer());

  ASSERT_TRUE(e.isSameShape(o));
  ASSERT_TRUE(e.equalsTo(o));

  ASSERT_TRUE(buffO1 == buffO2);
}

TEST_F(BroadcastableOpsTests, Test_Subtract_1) {
  auto x = NDArrayFactory::create<float>(1.0f);
  auto y = NDArrayFactory::create<float>('c', {2}, {0.0f, 1.0f});
  auto e = NDArrayFactory::create<float>('c', {2}, {1.0f, 0.0f});

  auto z = x - y;

  ASSERT_TRUE(e.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Subtract_2) {
  auto x = NDArrayFactory::create<float>(1.0f);
  auto y = NDArrayFactory::create<float>('c', {2}, {0.0f, 1.0f});
  auto e = NDArrayFactory::create<float>('c', {2}, {1.0f, 0.0f});

  ops::subtract op;
  auto result = op.evaluate({&x, &y});
  auto z = result.at(0);

  ASSERT_TRUE(e.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Subtract_3) {
  auto x = NDArrayFactory::create<float>(1.0f);
  auto y = NDArrayFactory::create<float>('c', {2}, {0.0f, 1.0f});
  auto z = NDArrayFactory::create<float>('c', {2}, {0.0f, 0.0f});
  auto e = NDArrayFactory::create<float>('c', {2}, {1.0f, 0.0f});

  ops::subtract op;
  auto result = op.execute({&x, &y}, {&z}, {}, {}, {});

  ASSERT_EQ(sd::Status::OK, result);
  ASSERT_TRUE(e.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Subtract_4) {
  auto x = NDArrayFactory::create<float>(1.0f);
  auto y = NDArrayFactory::create<float>('c', {2}, {0.0f, 1.0f});
  auto e = NDArrayFactory::create<float>('c', {2}, {1.0f, 0.0f});

  auto z = x.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), y);

  ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Subtract_5) {
  auto x = NDArrayFactory::create<float>(1.0f);
  auto y = NDArrayFactory::create<float>('c', {2}, {0.0f, 1.0f});
  auto e = NDArrayFactory::create<float>('c', {2}, {-1., 0.});

  auto z = y - x;

  ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Subtract_6) {
  auto x = NDArrayFactory::create<float>(1.0f);
  auto y = NDArrayFactory::create<float>(4.f);
  auto e = NDArrayFactory::create<float>(3.f);

  auto z = y - x;
  sd_printf("Data type of z is %s and e is %s\n",DataTypeUtils::asString(z.dataType()).c_str(),DataTypeUtils::asString(e.dataType()).c_str());
  ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Subtract_7) {
  auto x = NDArrayFactory::create<float>(1.0f);
  auto y = NDArrayFactory::create<float>(4.f);
  auto e = NDArrayFactory::create<float>(-3.f);

  auto z = x - y;

  ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Add_2) {
  auto x = NDArrayFactory::create<float>(1.0f);
  auto y = NDArrayFactory::create<float>('c', {2}, {0.0f, 1.0f});
  auto e = NDArrayFactory::create<float>('c', {2}, {1.f, 2.f});

  auto z = x + y;

  ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Add_3) {
  auto x = NDArrayFactory::create<float>(1.0f);
  auto y = NDArrayFactory::create<float>('c', {2}, {0.0f, 1.0f});
  auto e = NDArrayFactory::create<float>('c', {2}, {1.f, 2.f});

  auto z = y + x;

  ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Add_4) {
  auto x = NDArrayFactory::create<float>(1.0f);
  auto y = NDArrayFactory::create<float>(4.f);
  auto e = NDArrayFactory::create<float>(5.f);

  auto z = x + y;

  ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Add_5) {
  auto x = NDArrayFactory::create<float>(1.0f);
  auto y = NDArrayFactory::create<float>(4.f);
  auto e = NDArrayFactory::create<float>(5.f);

  auto z = y + x;

  ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Multiply_2) {
  auto x = NDArrayFactory::create<float>(2.0f);
  auto y = NDArrayFactory::create<float>('c', {2}, {3.f, 4.f});
  auto e = NDArrayFactory::create<float>('c', {2}, {6.f, 8.f});

  auto z = y * x;

  ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Multiply_3) {
  auto x = NDArrayFactory::create<float>(2.0f);
  auto y = NDArrayFactory::create<float>('c', {2}, {3.f, 4.f});
  auto e = NDArrayFactory::create<float>('c', {2}, {6.f, 8.f});

  auto z = x * y;

  ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Multiply_4) {
  auto x = NDArrayFactory::create<float>(2.0f);
  auto y = NDArrayFactory::create<float>(4.f);
  auto e = NDArrayFactory::create<float>(8.f);

  auto z = y * x;

  ASSERT_TRUE(e.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, Test_Multiply_5) {
  auto x = NDArrayFactory::create<float>(2.0f);
  auto y = NDArrayFactory::create<float>(4.f);
  auto e = NDArrayFactory::create<float>(8.f);

  auto z = x * y;

  ASSERT_TRUE(e.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Multiply_6) {
  auto x = NDArrayFactory::create<float>(2.0f);
  auto y = NDArrayFactory::create<float>('c', {1}, {4.f});
  auto e = NDArrayFactory::create<float>('c', {1}, {8.f});

  auto z = x * y;

  ASSERT_TRUE(e.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Multiply_7) {
  auto x = NDArrayFactory::create<float>(2.0f);
  auto y = NDArrayFactory::create<float>('c', {1}, {4.f});
  auto e = NDArrayFactory::create<float>('c', {1}, {8.f});

  ops::multiply op;
  auto result = op.evaluate({&x, &y});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

  ASSERT_TRUE(e.equalsTo(z));
}

TEST_F(BroadcastableOpsTests, Test_Multiply_8) {
  auto x = NDArrayFactory::create<float>(2.0f);
  auto y = NDArrayFactory::create<float>('c', {1, 1}, {4.f});
  auto e = NDArrayFactory::create<float>('c', {1, 1}, {8.f});

  ops::multiply op;
  auto result = op.evaluate({&x, &y});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

  ASSERT_TRUE(e.equalsTo(z));
}

//////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, broadcast_add_1) {
  NDArray x('c', {4}, {1, 1, 1, 1});
  NDArray y('c', {1, 4}, {1, 2, 3, 4});
  NDArray z('c', {1, 4}, DOUBLE);
  NDArray exp('c', {1, 4}, {2, 3, 4, 5}, DOUBLE);

  ops::add op;
  auto status = op.execute({&x, &y}, {&z});

  ASSERT_EQ(sd::Status::OK, status);
  ASSERT_TRUE(z.equalsTo(exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, broadcast_equals_1) {
  NDArray x('c', {1, 4}, {1, 2, 3, 4});
  NDArray y('c', {3, 4}, {0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4});
  NDArray z('c', {3, 4}, BOOL);
  NDArray exp('c', {3, 4}, {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}, BOOL);

  ops::equals op;
  auto status = op.execute({&x, &y}, {&z});

  ASSERT_EQ(sd::Status::OK, status);
  ASSERT_TRUE(z.equalsTo(exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(BroadcastableOpsTests, broadcast_empty_1) {
  NDArray y('c', {3, 4}, {0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4});
  NDArray x(DOUBLE, y.getContext(), false);
  NDArray z(DOUBLE, y.getContext(), false);
  NDArray zExp(DOUBLE, y.getContext(), false);

  ops::multiply op;
  auto status = op.execute({&x, &y}, {&z}, {}, {}, {});

  ASSERT_EQ(sd::Status::OK, status);
  ASSERT_TRUE(z.isSameShape(zExp));
  ASSERT_TRUE(z.equalsTo(zExp));
}

TEST_F(BroadcastableOpsTests, broadcast_empty_2) {
  NDArray y('c', {1, 4}, {1, 2, 3, 4});
  NDArray x = NDArrayFactory::create<double>('c', {0, 4});
  NDArray e = NDArrayFactory::create<double>('c', {0, 4});
  ;

  ops::multiply op;
  auto status = op.execute({&x, &y}, {&x}, {}, {}, {});

  ASSERT_EQ(sd::Status::OK, status);
  ASSERT_TRUE(e.isSameShape(x));
  ASSERT_TRUE(e.equalsTo(x));
}

TEST_F(BroadcastableOpsTests, broadcast_empty_3) {
  NDArray x = NDArrayFactory::create<float>('c', {1, 0, 2});
  NDArray y('c', {}, std::vector<double>{0.1}, FLOAT32);
  NDArray e = NDArrayFactory::create<float>('c', {1, 0, 2});
  ;

  ops::maximum op;
  auto result = op.evaluate({&x, &y});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

  ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

TEST_F(BroadcastableOpsTests, broadcast_empty_4) {
  NDArray x = NDArrayFactory::create<float>('c', {1, 0, 1});
  NDArray y = NDArrayFactory::create<float>('c', {1, 0, 2});
  NDArray e = NDArrayFactory::create<float>('c', {1, 0, 2});
  ;

  ops::maximum op;
  auto result = op.evaluate({&x, &y});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

  ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

TEST_F(BroadcastableOpsTests, broadcast_empty_5) {
  NDArray x = NDArrayFactory::create<float>('c', {1, 0, 1});
  NDArray y = NDArrayFactory::create<float>('c', {1, 0, 2});
  NDArray e = NDArrayFactory::create<float>('c', {1, 0, 2});
  ;

  ops::realdiv op;
  auto result = op.evaluate({&x, &y});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

  ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

TEST_F(BroadcastableOpsTests, broadcast_empty_6) {
  NDArray x = NDArrayFactory::create<float>('c', {1, 0, 1});
  NDArray y = NDArrayFactory::create<float>('c', {1, 2}, {2, 2});
  NDArray e = NDArrayFactory::create<float>('c', {1, 0, 2});
  ;

  ops::realdiv op;
  auto result = op.evaluate({&x, &y});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

  ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

TEST_F(BroadcastableOpsTests, broadcast_empty_7) {
  NDArray x = NDArrayFactory::create<float>('c', {1, 0, 2, 1});
  NDArray y = NDArrayFactory::create<float>('c', {1, 2, 0});
  NDArray e = NDArrayFactory::create<float>('c', {1, 0, 2, 0});
  ;

  ops::realdiv op;
  auto result = op.evaluate({&x, &y});

  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

  ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

TEST_F(BroadcastableOpsTests, broadcast_bool_empty_1) {
  NDArray y('c', {3, 4}, {0, 0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4});
  NDArray x(DOUBLE, y.getContext(), false);
  NDArray z(BOOL, y.getContext(), false);
  NDArray zExp(BOOL, y.getContext(), false);

  ops::greater op;
  auto status = op.execute({&x, &y}, {&z}, {}, {}, {});

  ASSERT_EQ(sd::Status::OK, status);
  ASSERT_TRUE(z.isSameShape(zExp));
  ASSERT_TRUE(z.equalsTo(zExp));
}

TEST_F(BroadcastableOpsTests, broadcast_bool_empty_2) {
  NDArray y('c', {1, 4}, {1, 2, 3, 4});
  NDArray x = NDArrayFactory::create<double>('c', {0, 4});
  NDArray e = NDArrayFactory::create<bool>('c', {0, 4});
  ;

  ops::greater op;
  auto result = op.evaluate({&x, &y});

  auto z = result.at(0);

  ASSERT_EQ(sd::Status::OK, result.status());
  ASSERT_TRUE(e.isSameShape(z));
  ASSERT_TRUE(e.equalsTo(*z));
}

TEST_F(BroadcastableOpsTests, broadcast_bool_1) {
  NDArray x('c', {3, 1, 2}, FLOAT32);
  NDArray y('c', {2, 2}, FLOAT32);
  NDArray z('c', {3, 2, 2}, BOOL);
  NDArray e('c', {3, 2, 2}, BOOL);

  x.assign(4.f);
  y.assign(2.f);
  e.assign(true);

  ops::greater op;

  auto status = op.execute({&x, &y}, {&z});

  ASSERT_EQ(sd::Status::OK, status);


  ASSERT_TRUE(z.isSameShape(e));
  ASSERT_TRUE(z.equalsTo(e));
}

TEST_F(BroadcastableOpsTests, broadcast_bool_2) {
  NDArray x('c', {3, 1, 2}, FLOAT32);
  NDArray y('c', {2, 2}, FLOAT32);
  NDArray z('c', {3, 2, 2}, BOOL);
  NDArray e('c', {3, 2, 2}, BOOL);

  x.assign(1.f);
  y.assign(2.f);
  e.assign(false);

  ops::equals op;

  auto status = op.execute({&x, &y}, {&z}, {}, {}, {});

  ASSERT_EQ(sd::Status::OK, status);


  ASSERT_TRUE(z.isSameShape(e));
  ASSERT_TRUE(z.equalsTo(e));
}

TEST_F(BroadcastableOpsTests, broadcast_bool_3) {
  auto x = NDArrayFactory::create<int>(0);
  auto y = NDArrayFactory::create<int>('c', {3}, {2, 1, 2});
  NDArray z('c', {3}, BOOL);
  NDArray e('c', {3}, BOOL);

  e.assign(true);

  ops::less op;
  auto status = op.execute({&x, &y}, {&z}, {}, {}, {});

  ASSERT_EQ(sd::Status::OK, status);


  ASSERT_TRUE(z.isSameShape(e));
  ASSERT_TRUE(z.equalsTo(e));
}

TEST_F(BroadcastableOpsTests, broadcast_2) {
  NDArray x('c', {3, 1, 2}, FLOAT32);
  NDArray y('c', {2, 2}, FLOAT32);
  NDArray z('c', {3, 2, 2}, FLOAT32);
  NDArray e('c', {3, 2, 2}, FLOAT32);

  x = 4.f;
  y = 2.f;
  e = -2.f;

  ops::reversesubtract op;  // z = y - x;

  auto status = op.execute({&x, &y}, {&z}, {}, {}, {});

  ASSERT_EQ(sd::Status::OK, status);


  ASSERT_TRUE(z.isSameShape(e));
  ASSERT_TRUE(z.equalsTo(e));
}

TEST_F(BroadcastableOpsTests, broadcast_3) {
  auto x = NDArrayFactory::create<int>(0);
  auto y = NDArrayFactory::create<int>('c', {3}, {2, 1, 2});
  NDArray z('c', {3}, INT32);
  auto e = NDArrayFactory::create<int>('c', {3}, {2, 1, 2});

  ops::add op;
  auto status = op.execute({&x, &y}, {&z}, {}, {}, {});

  ASSERT_EQ(sd::Status::OK, status);


  ASSERT_TRUE(z.isSameShape(e));
  ASSERT_TRUE(z.equalsTo(e));
}

TEST_F(BroadcastableOpsTests, test_bert_multiply_1) {
  auto x = NDArrayFactory::create<float>('c', {4, 128, 1});
  auto y = NDArrayFactory::create<float>('c', {4, 1, 128});
  auto z = NDArrayFactory::create<float>('c', {4, 128, 128});
  auto e = NDArrayFactory::create<float>('c', {4, 128, 128});

  x.assign(0.f);
  y.assign(1.f);
  z.assign(119.f);
  e.assign(0.f);

  x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), y, z);


  ASSERT_EQ(e, z);
}

TEST_F(BroadcastableOpsTests, test_bert_multiply_2) {
  auto x = NDArrayFactory::create<float>('c', {4, 128, 1});
  auto y = NDArrayFactory::create<float>('c', {768});
  auto z = NDArrayFactory::create<float>('c', {4, 128, 768});
  auto e = NDArrayFactory::create<float>('c', {4, 128, 768});

  x.assign(1.f);
  y.assign(2.f);
  z.assign(119.f);
  e.assign(2.f);

  x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), y, z);

  ASSERT_EQ(e, z);
}
