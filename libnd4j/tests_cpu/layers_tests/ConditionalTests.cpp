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
// Created by raver119 on 16.10.2017.
//

#include "testlayers.h"
#include <Graph.h>
#include <GraphExecutioner.h>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::graph;

class ConditionalTests : public testing::Test {
public:
    ConditionalTests(){
        //Environment::getInstance()->setVerbose(true);
        //Environment::getInstance()->setDebug(true);
    }

    ~ConditionalTests(){
        //Environment::getInstance()->setVerbose(false);
        //Environment::getInstance()->setDebug(false);
    }
};


TEST_F(ConditionalTests, BasicTests_1) {
    Graph<float> graph;

    auto x  =     NDArray<float>::valueOf({2, 2}, 1.0);
    auto y0 =     NDArray<float>::valueOf({2, 2}, 5.0);
    auto y1 =     NDArray<float>::valueOf({2, 2}, -5.0);
    auto scalar = NDArray<float>::scalar(1.0);

    auto variableSpace = graph.getVariableSpace();

    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y0);
    variableSpace->putVariable(-3, y1);
    variableSpace->putVariable(-4, scalar);


    auto scopeCondition = new Node<float>(OpType_LOGIC, 10, 1);
    scopeCondition->setName("scopeCondition");

    auto scopeFalse = new Node<float>(OpType_LOGIC, 10, 2);
    scopeFalse->setName("scopeFalse");

    auto scopeTrue = new Node<float>(OpType_LOGIC, 10, 3);
    scopeTrue->setName("scopeTrue");

    auto nodeF = new Node<float>(OpType_PAIRWISE, 0, 5, {-1, -2});
    nodeF->setScopeInfo(2, "scopeFalse");

    auto nodeT = new Node<float>(OpType_PAIRWISE, 1, 6, {-1, -2});
    nodeT->setScopeInfo(3, "scopeTrue");

    auto nodeC0 = new Node<float>(OpType_ACCUMULATION, 1, 7, {-1});
    nodeC0->setScopeInfo(1, "scopeCondition");

    auto nodeC1 = new Node<float>(OpType_BOOLEAN, 0, 8, {7, -4});
    nd4j::ops::eq_scalar<float> op;
    nodeC1->setCustomOp(&op);
    nodeC1->setScopeInfo(1, "scopeCondition");

    graph.addNode(scopeCondition);
    graph.addNode(scopeFalse);
    graph.addNode(scopeTrue);
    graph.addNode(nodeF);
    graph.addNode(nodeT);
    graph.addNode(nodeC0);
    graph.addNode(nodeC1);

    // at this point graph should ounly have Nodes referring to the Scopes: condition scope, true scope and false scope
    ASSERT_EQ(3, graph.totalNodes());

    // now we're adding Condition op, that'll take all of those in
    auto nodeCondition = new Node<float>(OpType_LOGIC, 20, 10, {1, 2, 3});
    graph.addNode(nodeCondition);

    ASSERT_EQ(4, graph.totalNodes());

    Nd4jStatus status = GraphExecutioner<float>::execute(&graph);
    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(variableSpace->hasVariable(10, 0));
    auto conditionalResult = variableSpace->getVariable(10, 0)->getNDArray();
    ASSERT_NE(nullptr, conditionalResult);

    ASSERT_NEAR(6.0, conditionalResult->meanNumber(), 1e-5);
}

/**
 * Condition is False
 */
TEST_F(ConditionalTests, Flat_Test_1) {
    nd4j::ops::identity<float> op0;

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/simpleif_0_1.fb");
    auto varSpace = graph->getVariableSpace();
    //varSpace->getVariable(1)->getNDArray()->assign(2.0);
    //varSpace->getVariable(2)->getNDArray()->assign(0.0);

    //graph->printOut();

    auto status = GraphExecutioner<float>::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(varSpace->hasVariable(15));

    auto z = varSpace->getVariable(15)->getNDArray();

    ASSERT_NE(nullptr, z);

    NDArray<float> exp('c', {2, 2}, {-2, -2, -2, -2});

    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}

/**
 * Condition is True
 */
TEST_F(ConditionalTests, Flat_Test_2) {
    nd4j::ops::identity<float> op0;

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/simpleif_0.fb");
    auto varSpace = graph->getVariableSpace();
    varSpace->getVariable(1)->getNDArray()->assign(-1.0);

    //graph->printOut();

    auto status = GraphExecutioner<float>::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(varSpace->hasVariable(15));

    auto z = varSpace->getVariable(15)->getNDArray();

    ASSERT_NE(nullptr, z);

    NDArray<float> exp('c', {2, 2}, {1, 1, 1, 1});

    ASSERT_TRUE(exp.equalsTo(z));
    delete graph;
}


/**
 * Condition is false here, so there loop will be skipped
 */
TEST_F(ConditionalTests, Flat_Test_3) {
    nd4j::ops::identity<float> op0;

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/simplewhile_0_3.fb");
    auto varSpace = graph->getVariableSpace();
    varSpace->getVariable(1)->getNDArray()->assign(1.0);

    //graph->printOut();

    auto status = GraphExecutioner<float>::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(varSpace->hasVariable(17));

    auto z = varSpace->getVariable(17)->getNDArray();

    ASSERT_NE(nullptr, z);

    NDArray<float> exp('c', {2, 2}, {1, 1, 1, 1});
    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}

/**
 * just one cycle in body
 */
TEST_F(ConditionalTests, Flat_Test_4) {
    nd4j::ops::identity<float> op0;

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/simplewhile_0_4.fb");
    auto varSpace = graph->getVariableSpace();
    varSpace->getVariable(2)->getNDArray()->assign(4.0);

    //graph->printOut();

    auto status = GraphExecutioner<float>::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(varSpace->hasVariable(17));

    auto z = varSpace->getVariable(17)->getNDArray();

    ASSERT_NE(nullptr, z);

    // 0.0 + 2.0 = 2.0 in each element
    NDArray<float> exp('c', {2, 2}, {2, 2, 2, 2});
    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}


/**
 * just two cycles in body
 */
TEST_F(ConditionalTests, Flat_Test_5) {
    nd4j::ops::identity<float> op0;

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/simplewhile_0_4.fb");
    auto varSpace = graph->getVariableSpace();
    varSpace->getVariable(2)->getNDArray()->assign(9.0);

    //graph->printOut();

    auto status = GraphExecutioner<float>::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(varSpace->hasVariable(17));

    auto z = varSpace->getVariable(17)->getNDArray();

    ASSERT_NE(nullptr, z);

    // 0.0 + 2.0 + 2.0 = 4.0 in each element
    NDArray<float> exp('c', {2, 2}, {4, 4, 4, 4});
    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}

/**
 * While loop with multiple variables
 */
TEST_F(ConditionalTests, Flat_Test_6) {
    nd4j::ops::identity<float> op0;

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/simplewhile_1.fb");
    auto varSpace = graph->getVariableSpace();
    varSpace->getVariable(1)->getNDArray()->assign(-4.0f);
    varSpace->getVariable(2)->getNDArray()->assign(1.0f);

    //graph->printOut();

    auto status = GraphExecutioner<float>::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(varSpace->hasVariable(25));

    auto z = varSpace->getVariable(25)->getNDArray();

    ASSERT_NE(nullptr, z);

    //z->printIndexedBuffer();

    NDArray<float> exp('c', {2, 2}, {-1, -1, -1, -1});
    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}

TEST_F(ConditionalTests, Flat_Test_7) {
    nd4j::ops::identity<float> op0;

    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/simplewhile_1.fb");
    auto varSpace = graph->getVariableSpace();
    varSpace->getVariable(1)->getNDArray()->assign(-9.0f);
    varSpace->getVariable(2)->getNDArray()->assign(1.0f);

    //graph->printOut();

    auto status = GraphExecutioner<float>::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(varSpace->hasVariable(25));

    auto z = varSpace->getVariable(25)->getNDArray();

    ASSERT_NE(nullptr, z);

    //z->printIndexedBuffer();

    NDArray<float> exp('c', {2, 2}, {-3, -3, -3, -3});
    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}

/**
 * This test checks nested while execution
 */
TEST_F(ConditionalTests, Flat_Test_8) {
    auto graph = GraphExecutioner<float>::importFromFlatBuffers("./resources/simplewhile_nested.fb");
    auto varSpace = graph->getVariableSpace();
    //graph->printOut();

    auto status = GraphExecutioner<float>::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(varSpace->hasVariable(52));

    auto z = varSpace->getVariable(52)->getNDArray();

    ASSERT_NE(nullptr, z);

    //val exp = Nd4j.create(2, 2).assign(15.0);
    NDArray<float> exp('c', {2, 2}, {15, 15, 15, 15});
    ASSERT_TRUE(exp.equalsTo(z));

    delete graph;
}