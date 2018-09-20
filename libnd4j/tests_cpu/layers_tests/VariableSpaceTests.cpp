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
#include <flatbuffers/flatbuffers.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <NDArray.h>

using namespace nd4j;
using namespace nd4j::graph;

class VariableSpaceTest : public testing::Test {
public:
    int *cShape = new int[8]{2, 2, 2, 2, 1, 0, 1, 99};
    int *fShape = new int[8]{2, 2, 2, 1, 2, 0, 1, 102};


    ~VariableSpaceTest() {
        delete[] cShape;
        delete[] fShape;
    }
};


TEST_F(VariableSpaceTest, SettersGettersTest1) {
    auto space1 = new VariableSpace();
    auto arrayA = NDArrayFactory::create<float>('c', {5, 5});
    auto arrayB = NDArrayFactory::create<float>('c', {3, 3});

    space1->putVariable(1, arrayA);
    space1->putVariable(2, arrayB);

    auto arrayRA = space1->getVariable(1);
    auto arrayRB = space1->getVariable(2);

    ASSERT_TRUE(arrayA == arrayRA->getNDArray());
    ASSERT_TRUE(arrayB == arrayRB->getNDArray());

    // we should survive this call
    delete space1;
}


TEST_F(VariableSpaceTest, SettersGettersTest2) {
    auto space1 = new VariableSpace();
    auto arrayA = NDArrayFactory::create<float>('c', {5, 5});
    auto arrayB = NDArrayFactory::create<float>('c', {3, 3});

    auto varA = new Variable(arrayA);
    auto varB = new Variable(arrayB);

    varA->markExternal(true);

    space1->putVariable(-1, varA);
    space1->putVariable(2, varB);

    Nd4jLong expExternal = (25 * 4) + (8 * 8);
    Nd4jLong expInternal = (9 * 4) + (8 * 8);

    ASSERT_EQ(expExternal, space1->externalMemory());
    ASSERT_EQ(expInternal, space1->internalMemory());

    delete space1;
}

TEST_F(VariableSpaceTest, EqualityTest1) {
    VariableSpace space;

    std::string name("myvar");

    auto arrayA = NDArrayFactory::create<float>('c', {3, 3});
    auto variableA = new Variable(arrayA, name.c_str());

    space.putVariable(1, variableA);

    std::pair<int,int> pair(1,0);

    ASSERT_TRUE(space.hasVariable(1));
    ASSERT_TRUE(space.hasVariable(pair));
    ASSERT_TRUE(space.hasVariable(&name));

    auto rV1 = space.getVariable(1);
    auto rV2 = space.getVariable(pair);
    auto rV3 = space.getVariable(&name);

    ASSERT_TRUE(rV1 == rV2);
    ASSERT_TRUE(rV2 == rV3);
}

TEST_F(VariableSpaceTest, EqualityTest2) {
    VariableSpace space;

    auto arrayA = NDArrayFactory::create<float>('c', {3, 3});

    space.putVariable(1, arrayA);

    std::pair<int,int> pair(1,0);

    ASSERT_TRUE(space.hasVariable(1));
    ASSERT_TRUE(space.hasVariable(pair));

    auto rV1 = space.getVariable(1);
    auto rV2 = space.getVariable(pair);

    ASSERT_TRUE(rV1 == rV2);
}

TEST_F(VariableSpaceTest, CloneTests_1) {
    VariableSpace spaceA;

    auto arrayA = NDArrayFactory::create<float>('c', {3, 3});
    arrayA->assign(1.0);

    spaceA.putVariable(1, arrayA);

    auto spaceB = spaceA.clone();

    std::pair<int,int> pair(1,0);

    ASSERT_TRUE(spaceB->hasVariable(1));
    ASSERT_TRUE(spaceB->hasVariable(pair));

    auto arrayB = spaceB->getVariable(1)->getNDArray();

    ASSERT_TRUE(arrayA->equalsTo(arrayB));

    arrayB->assign(2.0);

    ASSERT_FALSE(arrayA->equalsTo(arrayB));

    delete spaceB;
}

TEST_F(VariableSpaceTest, CloneTests_2) {
    VariableSpace spaceA;

    auto arrayA = NDArrayFactory::create<float>('c', {3, 3});
    arrayA->assign(1.0);

    auto variableA = new Variable(arrayA, "alpha");

    std::string str("alpha");
    std::pair<int, int> pair(2, 3);

    spaceA.putVariable(pair, variableA);

    ASSERT_TRUE(spaceA.hasVariable(&str));
    ASSERT_TRUE(spaceA.hasVariable(pair));

    auto spaceB = spaceA.clone();

    ASSERT_FALSE(spaceB->hasVariable(1));
    ASSERT_FALSE(spaceB->hasVariable(2));
    ASSERT_TRUE(spaceB->hasVariable(pair));
    ASSERT_TRUE(spaceB->hasVariable(&str));

    auto arrayB = spaceB->getVariable(pair)->getNDArray();

    ASSERT_TRUE(arrayA->equalsTo(arrayB));

    arrayB->assign(2.0);

    ASSERT_FALSE(arrayA->equalsTo(arrayB));

    delete spaceB;

    ASSERT_TRUE(spaceA.hasVariable(&str));
    ASSERT_TRUE(spaceA.hasVariable(pair));
}


TEST_F(VariableSpaceTest, Test_DType_Conversion_1) {
    /*
    VariableSpace spaceA;

    auto arrayA = NDArrayFactory::create<float>('c', {3, 3});
    arrayA->assign(1.0);

    auto variableA = new Variable(arrayA, "alpha");

    std::string str("alpha");
    std::pair<int, int> pair(2, 3);

    spaceA.putVariable(pair, variableA);


    auto sd = spaceA.template asT<double>();
    auto sf = sd->template asT<float>();

    ASSERT_TRUE(sf->hasVariable(pair));

    auto xf = sf->getVariable(pair)->getNDArray();

    ASSERT_TRUE(arrayA->isSameShape(xf));
    ASSERT_TRUE(arrayA->equalsTo(xf));

    delete sd;
    delete sf;
    */
}