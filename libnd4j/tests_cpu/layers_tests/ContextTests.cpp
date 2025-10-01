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
// Created by raver119 on 30.10.2017.
//
#include <ops/declarable/CustomOperations.h>

#include "testlayers.h"

using namespace sd;
using namespace sd::ops;
using namespace sd::graph;

class ContextTests : public NDArrayTests {
 public:
};

TEST_F(ContextTests, Basic_Test_1) {
  VariableSpace variableSpace;

  auto _20 = NDArrayFactory::create_<float>('c', {2, 2});
  auto _21 = NDArrayFactory::create_<float>('c', {2, 2});

  _20->assign(1.0f);
  _21->assign(2.0f);

  variableSpace.putVariable(2, 0, _20);
  variableSpace.putVariable(2, 1, _21);

  Context block(1, &variableSpace);

  block.pickInput(2, 0);
  block.pickInput(2, 1);

  ASSERT_EQ(2, block.inputs()->size());
  ASSERT_EQ(2, block.width());

  ASSERT_TRUE(variableSpace.hasVariable(2, 0));
  ASSERT_TRUE(variableSpace.hasVariable(2, 1));

  ASSERT_NEAR(1.0f, block.variable(0)->getNDArray()->meanNumber().e<float>(0), 1e-5);
  ASSERT_NEAR(2.0f, block.variable(1)->getNDArray()->meanNumber().e<float>(0), 1e-5);
}

TEST_F(ContextTests, Basic_Test_2) {
  VariableSpace variableSpace;

  auto _20 = NDArrayFactory::create_<float>('c', {2, 2});
  auto _21 = NDArrayFactory::create_<float>('c', {2, 2});

  _20->assign(1.0f);
  _21->assign(2.0f);

  variableSpace.putVariable(-1, _20);
  variableSpace.putVariable(-2, _21);

  Context block(1, &variableSpace);

  block.pickInput(-1);
  block.pickInput(-2);

  ASSERT_EQ(2, block.inputs()->size());
  ASSERT_EQ(2, block.width());

  ASSERT_TRUE(variableSpace.hasVariable(-1));
  ASSERT_TRUE(variableSpace.hasVariable(-2));

  ASSERT_NEAR(1.0f, block.variable(0)->getNDArray()->meanNumber().e<float>(0), 1e-5);
  ASSERT_NEAR(2.0f, block.variable(1)->getNDArray()->meanNumber().e<float>(0), 1e-5);
}

TEST_F(ContextTests, Basic_Test_3) {
  VariableSpace variableSpace;

  Context ctx(1, &variableSpace);

  auto _20 = NDArrayFactory::create_<float>('c', {2, 2});

  ctx.pushNDArrayToVariableSpace(1, 1, _20);

  ASSERT_TRUE(variableSpace.hasVariable(1, 1));
}

TEST_F(ContextTests, Basic_Test_4) {
  VariableSpace variableSpace;

  Context ctx(1, &variableSpace);

  auto _20 = NDArrayFactory::create_<float>('c', {2, 2});
  _20->linspace(1);

  auto _21 = NDArrayFactory::create_<float>('c', {2, 2});
  _21->linspace(10);

  ctx.pushNDArrayToVariableSpace(1, 1, _20);

  ASSERT_TRUE(variableSpace.hasVariable(1, 1));

  ctx.pushNDArrayToVariableSpace(1, 1, _21);

  auto vA = ctx.variable(1, 1);

  ASSERT_TRUE(vA->getNDArray()->equalsTo(_21));
}

TEST_F(ContextTests, Basic_Test_5) {
  VariableSpace variableSpace;

  Context ctx(1, &variableSpace);

  auto _20 = NDArrayFactory::create_<float>('c', {2, 2});
  _20->linspace(1);

  auto exp = _20->dup();

  ctx.pushNDArrayToVariableSpace(1, 1, _20);

  ASSERT_TRUE(variableSpace.hasVariable(1, 1));

  ctx.pushNDArrayToVariableSpace(1, 1, _20);

  auto vA = ctx.variable(1, 1);

  ASSERT_TRUE(vA->getNDArray() == _20);

  ASSERT_TRUE(vA->getNDArray()->equalsTo(exp));

  delete exp;
}

TEST_F(ContextTests, Basic_Test_6) {
  VariableSpace variableSpace;

  Context ctx(1, &variableSpace);

  auto v0 = ctx.ensureVariable();
  auto v1 = ctx.ensureVariable(1);

  ASSERT_TRUE(variableSpace.hasVariable(1, 0));
  ASSERT_TRUE(variableSpace.hasVariable(1, 1));

  auto var0 = variableSpace.getVariable(1, 0);
  auto var1 = variableSpace.getVariable(1, 1);

  ASSERT_TRUE(v0 == var0);
  ASSERT_TRUE(v1 == var1);
}

TEST_F(ContextTests, Basic_Test_7) {
  VariableSpace variableSpace;

  Context ctx(1, &variableSpace);

  auto v0 = ctx.ensureVariable();
  auto v1 = ctx.ensureVariable(1);

  ASSERT_TRUE(variableSpace.hasVariable(1, 0));
  ASSERT_TRUE(variableSpace.hasVariable(1, 1));

  auto var0 = variableSpace.getVariable(1, 0);
  auto var1 = variableSpace.getVariable(1, 1);

  ASSERT_TRUE(v0 == var0);
  ASSERT_TRUE(v1 == var1);

  auto _10 = NDArrayFactory::create_<float>('c', {2, 2});
  _10->linspace(1);

  auto _11 = NDArrayFactory::create_<float>('c', {2, 2});
  _11->linspace(10);

  ctx.pushNDArrayToVariableSpace(1, 0, _10);
  ctx.pushNDArrayToVariableSpace(1, 1, _11);

  auto z0 = variableSpace.getVariable(1, 0);
  auto z1 = variableSpace.getVariable(1, 1);

  ASSERT_TRUE(v0 == z0);
  ASSERT_TRUE(v1 == z1);
}

TEST_F(ContextTests, Basic_Test_8) {
  VariableSpace variableSpace;

  Context ctx(1, &variableSpace);

  auto _10 = NDArrayFactory::create_<float>('c', {2, 2});
  _10->linspace(1);

  auto _11 = NDArrayFactory::create_<float>('c', {2, 2});
  _11->linspace(10);

  ctx.pushNDArrayToVariableSpace(1, 0, _10);
  ctx.pushNDArrayToVariableSpace(1, 1, _11);

  auto z0 = variableSpace.getVariable(1, 0);
  auto z1 = variableSpace.getVariable(1, 1);

  auto v0 = ctx.ensureVariable();
  auto v1 = ctx.ensureVariable(1);

  ASSERT_TRUE(v0 == z0);
  ASSERT_TRUE(v1 == z1);
}

TEST_F(ContextTests, Basic_Test_9) {
  VariableSpace variableSpace;

  auto in = NDArrayFactory::create<float>('c', {5, 5});

  Context ctx(1, &variableSpace, true);
  ctx.pushNDArrayToVariableSpace(1, 1, &in, false);
}

TEST_F(ContextTests, Basic_Test_10) {
  VariableSpace variableSpace;

  Context ctx(119, &variableSpace);
}

TEST_F(ContextTests, Prototype_Test_1) {
  ContextPrototype prototype(nullptr, 119, true);
  prototype.pickInput(12, 3);
  prototype.pickInput(12, 4);

  prototype.getTArguments()->push_back(2.0);
  prototype.getTArguments()->push_back(-2.0);

  prototype.getIArguments()->push_back(17);
  prototype.getIArguments()->push_back(119);

  Context ctx(&prototype, nullptr);

  ASSERT_EQ(ctx.nodeId(), prototype.nodeId());
  ASSERT_EQ(ctx.isInplace(), prototype.isInplace());

  ASSERT_EQ(2, ctx.inputs()->size());
  ASSERT_EQ(2, ctx.getTArguments()->size());
  ASSERT_EQ(2, ctx.getIArguments()->size());

  ASSERT_EQ(2.0, ctx.getTArguments()->at(0));
  ASSERT_EQ(-2.0, ctx.getTArguments()->at(1));

  ASSERT_EQ(17, ctx.getIArguments()->at(0));
  ASSERT_EQ(119, ctx.getIArguments()->at(1));
}

TEST_F(ContextTests, Prototype_Test_2) {
  ContextPrototype prototype(nullptr, 119, false);
  prototype.setOpNum(179);

  Context ctx(&prototype, nullptr);

  ASSERT_EQ(ctx.isInplace(), prototype.isInplace());
  ASSERT_EQ(ctx.opNum(), prototype.opNum());

  ASSERT_EQ(0, ctx.inputs()->size());
  ASSERT_EQ(0, ctx.getTArguments()->size());
  ASSERT_EQ(0, ctx.getIArguments()->size());
}

TEST_F(ContextTests, test_short_context_1) {
  //note this used to be stack allocation based but appears to cause issues with cuda and smart pointers
  auto array0 = new NDArray(NDArrayFactory::create<float>('c', {3, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
  auto array1 = new NDArray(NDArrayFactory::create<float>('c', {3, 2}, {-1.f, -2.f, -3.f, -4.f, -5.f, -6.f}));
  Context ctx(1);

  ctx.setInputArray(0, array0->buffer(), array0->shapeInfo(), array0->specialBuffer(), array0->specialShapeInfo());
  ctx.setInputArray(1, array1->buffer(), array1->shapeInfo(), array1->specialBuffer(), array1->specialShapeInfo());

  ASSERT_EQ(2, ctx.width());

  auto input0 = ctx.array(0);
  ASSERT_TRUE(input0 != nullptr);

  auto input1 = ctx.array(1);
  ASSERT_TRUE(input1 != nullptr);


  ASSERT_TRUE(input0->buffer() == array0->buffer());
  ASSERT_TRUE(input0->shapeInfo() == array0->shapeInfo());

  ASSERT_TRUE(input0->specialBuffer() == array0->specialBuffer());
  ASSERT_TRUE(input0->specialShapeInfo() == array0->specialShapeInfo());

  ASSERT_TRUE(input1->buffer() == array1->buffer());
  ASSERT_TRUE(input1->shapeInfo() == array1->shapeInfo());

  ASSERT_TRUE(input1->specialBuffer() == array1->specialBuffer());
  ASSERT_TRUE(input1->specialShapeInfo() == array1->specialShapeInfo());
}

TEST_F(ContextTests, test_short_context_2) {
  //note this used to be stack allocation based but appears to cause issues with cuda and smart pointers
  auto array0 = new NDArray(NDArrayFactory::create<float>('c', {3, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
  auto array1 = new NDArray(NDArrayFactory::create<float>('c', {3, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
  auto z = new NDArray(NDArrayFactory::create<float>('c', {3, 2}));
  auto exp = new NDArray(NDArrayFactory::create<float>('c', {3, 2}, {2.f, 4.f, 6.f, 8.f, 10.f, 12.f}));
  Context ctx(1);

  ctx.setInputArray(0, array0->buffer(), array0->shapeInfo(), array0->specialBuffer(), array0->specialShapeInfo());
  ctx.setInputArray(1, array1->buffer(), array1->shapeInfo(), array1->specialBuffer(), array1->specialShapeInfo());
  ctx.setOutputArray(0, z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  ASSERT_EQ(2, ctx.width());
  add op;
  op.execute(&ctx);

  ASSERT_EQ(*exp, *z);
}

TEST_F(ContextTests, test_short_context_3) {
  //note this used to be stack allocation based but appears to cause issues with cuda and smart pointers
  auto array0 = new NDArray(NDArrayFactory::create<float>('c', {3, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
  auto array1 = new NDArray(NDArrayFactory::create<float>('c', {3, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));

  auto exp = new NDArray(NDArrayFactory::create<float>('c', {3, 2}, {2.f, 4.f, 6.f, 8.f, 10.f, 12.f}));
  Context ctx(1);

  ctx.setInputArray(0, array0->buffer(), array0->shapeInfo(), array0->specialBuffer(), array0->specialShapeInfo());
  ctx.setInputArray(1, array1->buffer(), array1->shapeInfo(), array1->specialBuffer(), array1->specialShapeInfo());

  ASSERT_EQ(2, ctx.width());

  add op;
  op.execute(&ctx);

  ASSERT_EQ(1, ctx.fastpath_out().size());

  auto z = ctx.fastpath_out()[0];

  ASSERT_EQ(*exp, *z);
}
