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

#ifndef LIBND4J_VARIABLETESTS_H
#define LIBND4J_VARIABLETESTS_H

#include "testlayers.h"
#include <NDArray.h>
#include <graph/Variable.h>
#include <flatbuffers/flatbuffers.h>

using namespace nd4j;
using namespace nd4j::graph;

class VariableTests : public testing::Test {
public:

};

TEST_F(VariableTests, TestClone_1) {
    auto array1 = NDArrayFactory::create_<float>('c', {5, 5});
    array1->assign(1.0);

    auto var1 = new Variable(array1, "alpha");
    var1->setId(119);


    auto var2 = var1->clone();

    ASSERT_FALSE(var1->getNDArray() == var2->getNDArray());
    auto array2 = var2->getNDArray();

    ASSERT_TRUE(array1->equalsTo(array2));
    ASSERT_EQ(var1->id(), var2->id());
    ASSERT_EQ(*var1->getName(), *var2->getName());

    delete var1;

    std::string str("alpha");
    ASSERT_EQ(*var2->getName(), str);
    array2->assign(2.0);

    ASSERT_NEAR(2.0, array2->meanNumber().e<float>(0), 1e-5);

    delete var2;
}

TEST_F(VariableTests, Test_FlatVariableDataType_1) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto original = NDArrayFactory::create<float>('c', {5, 10});
    original.linspace(1);

    auto vec = original.asByteVector();

    auto fShape = builder.CreateVector(original.getShapeInfoAsFlatVector());
    auto fBuffer = builder.CreateVector(vec);
    auto fVid = CreateIntPair(builder, 1, 12);

    auto fArray = CreateFlatArray(builder, fShape, fBuffer, nd4j::graph::DataType::FLOAT);

    auto flatVar = CreateFlatVariable(builder, fVid, 0, nd4j::graph::DataType::FLOAT, 0, fArray);

    builder.Finish(flatVar);

    auto ptr = builder.GetBufferPointer();

    auto restoredVar = GetFlatVariable(ptr);

    auto rv = new Variable(restoredVar);

    ASSERT_EQ(1, rv->id());
    ASSERT_EQ(12, rv->index());

    auto restoredArray = rv->getNDArray();

    ASSERT_TRUE(original.isSameShape(restoredArray));
    ASSERT_TRUE(original.equalsTo(restoredArray));

    delete rv;
}

TEST_F(VariableTests, Test_FlatVariableDataType_2) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto original = NDArrayFactory::create<double>('c', {5, 10});
    original.linspace(1);

    auto vec = original.asByteVector();

    auto fShape = builder.CreateVector(original.getShapeInfoAsFlatVector());
    auto fBuffer = builder.CreateVector(vec);
    auto fVid = CreateIntPair(builder, 1, 12);

    auto fArray = CreateFlatArray(builder, fShape, fBuffer, nd4j::graph::DataType::DOUBLE);

    auto flatVar = CreateFlatVariable(builder, fVid, 0, nd4j::graph::DataType::DOUBLE, 0, fArray);

    builder.Finish(flatVar);

    auto ptr = builder.GetBufferPointer();

    auto restoredVar = GetFlatVariable(ptr);

    auto rv = new Variable(restoredVar);

    ASSERT_EQ(1, rv->id());
    ASSERT_EQ(12, rv->index());

    auto restoredArray = rv->getNDArray();

    ASSERT_TRUE(original.isSameShape(restoredArray));
    ASSERT_TRUE(original.equalsTo(restoredArray));

    delete rv;
}


TEST_F(VariableTests, Test_FlatVariableDataType_3) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto original = NDArrayFactory::create<double>('c', {5, 10});
    auto floating = NDArrayFactory::create<float>('c', {5, 10});
    original.linspace(1);
    floating.linspace(1);

    auto vec = original.asByteVector();

    auto fShape = builder.CreateVector(original.getShapeInfoAsFlatVector());
    auto fBuffer = builder.CreateVector(vec);
    auto fVid = CreateIntPair(builder, 1, 12);

    auto fArray = CreateFlatArray(builder, fShape, fBuffer, nd4j::graph::DataType::DOUBLE);

    auto flatVar = CreateFlatVariable(builder, fVid, 0, nd4j::graph::DataType::DOUBLE, 0, fArray);

    builder.Finish(flatVar);

    auto ptr = builder.GetBufferPointer();

    auto restoredVar = GetFlatVariable(ptr);

    auto rv = new Variable(restoredVar);

    ASSERT_EQ(1, rv->id());
    ASSERT_EQ(12, rv->index());

    auto restoredArray = rv->getNDArray();

    ASSERT_TRUE(floating.isSameShape(restoredArray));
    ASSERT_TRUE(floating.equalsTo(restoredArray));

    delete rv;
}


TEST_F(VariableTests, Test_FlatVariableDataType_4) {
    flatbuffers::FlatBufferBuilder builder(1024);
    auto original = NDArrayFactory::create<float>('c', {5, 10});


    auto vec = original.asByteVector();

    auto fShape = builder.CreateVector(original.getShapeInfoAsFlatVector());
    auto fVid = CreateIntPair(builder, 37, 12);

    auto flatVar = CreateFlatVariable(builder, fVid, 0, nd4j::graph::DataType::FLOAT, fShape, 0);

    builder.Finish(flatVar);

    auto ptr = builder.GetBufferPointer();

    auto restoredVar = GetFlatVariable(ptr);

    auto rv = new Variable(restoredVar);

    ASSERT_EQ(37, rv->id());
    ASSERT_EQ(12, rv->index());

    auto restoredArray = rv->getNDArray();

    ASSERT_TRUE(original.isSameShape(restoredArray));
    ASSERT_TRUE(original.equalsTo(restoredArray));

    delete rv;
}

TEST_F(VariableTests, Test_Dtype_Conversion_1) {
    auto x = NDArrayFactory::create_<float>('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    Variable v(x, "alpha", 12, 3);

    auto vd = v.template asT<double>();
    auto vf = vd->template asT<float>();

    ASSERT_EQ(*v.getName(), *vf->getName());
    ASSERT_EQ(v.id(), vf->id());
    ASSERT_EQ(v.index(), vf->index());

    auto xf = vf->getNDArray();

    ASSERT_TRUE(x->isSameShape(xf));
    ASSERT_TRUE(x->equalsTo(xf));

    delete vd;
    delete vf;
}

#endif //LIBND4J_VARIABLETESTS_H
