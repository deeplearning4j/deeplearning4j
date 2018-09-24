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
// Created by raver119 on 13.10.2017.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class SwitchTests : public testing::Test {
public:

};

TEST_F(SwitchTests, SwitchTest1) {
    Graph graph;

    FlowPath flowPath;

    auto variableSpace = graph.getVariableSpace();
    variableSpace->setFlowPath(&flowPath);

    auto input = NDArrayFactory::create<float>('c',{32, 100});
    input->assign(-119.0f);

    auto condtionX = NDArrayFactory::create<float>('c', {1, 1});
    condtionX->p(0, 0.0f);
    auto condtionY = NDArrayFactory::create<float>('c', {1, 1});
    condtionY->p(0, 0.0f);

    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, condtionX);
    variableSpace->putVariable(-3, condtionY);

    // this is just 2 ops, that are executed sequentially. We don't really care bout them
    auto nodeA = new Node(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node(OpType_TRANSFORM, 0, 2, {1}, {3});

    // this is our condition op, we'll be using Equals condition, on variables conditionX and conditionY (ids -2 and -3 respectively)
    auto nodeCondition = new Node(OpType_BOOLEAN, 0, 119, {-2, -3});

    // we're creating this op manually in tests, as always.
    nd4j::ops::eq_scalar eqOp;
    nodeCondition->setCustomOp(&eqOp);

    // now, this is Switch operation. It takes BooleanOperation operation in,
    // and based on evaluation result (true/false) - it'll pass data via :0 or :1 output
    // other idx will be considered disabled, and that graph branch won't be executed
    auto nodeSwitch = new Node(OpType_CUSTOM, 0, 3, {2, 119}, {4, 5});

    nd4j::ops::Switch switchOp;
    nodeSwitch->setCustomOp(&switchOp);


    // these 2 ops are connected to FALSE and TRUE outputs. output :0 considered FALSE, and output :1 considered TRUE
    auto nodeZ0 = new Node(OpType_TRANSFORM, 0, 4, {}, {});
    nodeZ0->pickInput(3, 0);
    auto nodeZ1 = new Node(OpType_TRANSFORM, 35, 5, {}, {});
    nodeZ1->pickInput(3, 1);


    graph.addNode(nodeA);
    graph.addNode(nodeB);
    graph.addNode(nodeCondition);
    graph.addNode(nodeSwitch);
    graph.addNode(nodeZ0);
    graph.addNode(nodeZ1);

    graph.buildGraph();

    // we're making sure nodes connected to the Switch have no other inputs in this Graph
    ASSERT_EQ(1, nodeZ0->input()->size());
    ASSERT_EQ(1, nodeZ1->input()->size());

    // just validating topo sort
    ASSERT_EQ(0, nodeA->getLayer());
    ASSERT_EQ(0, nodeCondition->getLayer());
    ASSERT_EQ(1, nodeB->getLayer());
    ASSERT_EQ(2, nodeSwitch->getLayer());
    ASSERT_EQ(3, nodeZ0->getLayer());
    ASSERT_EQ(3, nodeZ1->getLayer());

    // executing graph
    Nd4jStatus status = GraphExecutioner::execute(&graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    // nd4j_printf("Z0: [%i]; Z1: [%i]\n", flowPath.isNodeActive(nodeZ0->id()), flowPath.isNodeActive(nodeZ1->id()));

    // we know that Switch got TRUE evaluation, so :0 should be inactive
    ASSERT_FALSE(flowPath.isNodeActive(nodeZ0->id()));

    // and :1 should be active
    ASSERT_TRUE(flowPath.isNodeActive(nodeZ1->id()));

    std::pair<int,int> unexpected(4,0);
    std::pair<int,int> expectedResultIndex(5,0);
    ASSERT_TRUE(variableSpace->hasVariable(expectedResultIndex));

    // getting output of nodeZ1
    auto output = variableSpace->getVariable(expectedResultIndex)->getNDArray();

    // and veryfing it against known expected value
    ASSERT_NEAR(-118.0f, output->e<float>(0), 1e-5f);
}

TEST_F(SwitchTests, SwitchTest2) {
    Graph graph;

    FlowPath flowPath;
    auto variableSpace = graph.getVariableSpace();
    variableSpace->setFlowPath(&flowPath);

    auto input = NDArrayFactory::create<float>('c',{32, 100});
    input->assign(-119.0f);

    auto condtionX = NDArrayFactory::create<float>('c', {1, 1});
    condtionX->p(0, 1.0f);
    auto condtionY = NDArrayFactory::create<float>('c', {1, 1});
    condtionY->p(0, 1.0f);


    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, condtionX);
    variableSpace->putVariable(-3, condtionY);


    auto nodeA = new Node(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node(OpType_TRANSFORM, 0, 2, {1}, {3});

    auto scopeCondition = new Node(OpType_LOGIC, 10, 3);
    scopeCondition->setName("scopeCondition");

    auto nodeCondition = new Node(OpType_BOOLEAN, 0, 119, {-2, -3});
    nodeCondition->setScopeInfo(3, "scopeCondition");

    nd4j::ops::eq_scalar eqOp;
    nodeCondition->setCustomOp(&eqOp);

    auto nodeSwitch = new Node(OpType_LOGIC, 30, 5, {3, 2});

    nd4j::ops::Switch switchOp;
    nodeSwitch->setCustomOp(&switchOp);


    // these 2 ops are connected to FALSE and TRUE outputs. output :0 considered FALSE, and output :1 considered TRUE
    auto nodeZ0 = new Node(OpType_TRANSFORM, 0, 6, {}, {});
    nodeZ0->pickInput(5, 0);
    auto nodeZ1 = new Node(OpType_TRANSFORM, 35, 7, {}, {});
    nodeZ1->pickInput(5, 1);

    graph.addNode(nodeA);
    graph.addNode(nodeB);
    graph.addNode(scopeCondition);
    graph.addNode(nodeCondition);
    graph.addNode(nodeSwitch);
    graph.addNode(nodeZ0);
    graph.addNode(nodeZ1);

    Nd4jStatus status = GraphExecutioner::execute(&graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(!flowPath.isNodeActive(nodeZ0->id()));
    ASSERT_TRUE(flowPath.isNodeActive(nodeZ1->id()));

    auto z = graph.getVariableSpace()->getVariable(7)->getNDArray();

    // abs(-119) = 119; 1 - 119 = -118
    ASSERT_NEAR(-118.f, z->e<float>(0), 1e-5);
}

TEST_F(SwitchTests, SwitchTest3) {
    Graph graph;

    FlowPath flowPath;
    auto variableSpace = graph.getVariableSpace();
    variableSpace->setFlowPath(&flowPath);

    auto input = NDArrayFactory::create<float>('c',{32, 100});
    input->assign(-119.0f);

    auto condtionX = NDArrayFactory::create<float>('c', {1, 1});
    condtionX->p(0, 2.0f);
    auto condtionY = NDArrayFactory::create<float>('c', {1, 1});
    condtionY->p(0, 1.0f);


    variableSpace->putVariable(-1, input);
    variableSpace->putVariable(-2, condtionX);
    variableSpace->putVariable(-3, condtionY);


    auto nodeA = new Node(OpType_TRANSFORM, 0, 1, {-1}, {2});
    auto nodeB = new Node(OpType_TRANSFORM, 0, 2, {1}, {3});

    auto scopeCondition = new Node(OpType_LOGIC, 10, 3);
    scopeCondition->setName("scopeCondition");

    auto nodeCondition = new Node(OpType_BOOLEAN, 0, 119, {-2, -3});
    nodeCondition->setScopeInfo(3, "scopeCondition");

    nd4j::ops::eq_scalar eqOp;
    nodeCondition->setCustomOp(&eqOp);

    auto nodeSwitch = new Node(OpType_LOGIC, 30, 5, {3, 2});

    nd4j::ops::Switch switchOp;
    nodeSwitch->setCustomOp(&switchOp);


    // these 2 ops are connected to FALSE and TRUE outputs. output :0 considered FALSE, and output :1 considered TRUE
    auto nodeZ0 = new Node(OpType_TRANSFORM, 6, 6, {}, {});
    nodeZ0->pickInput(5, 0);
    auto nodeZ1 = new Node(OpType_TRANSFORM, 35, 7, {}, {});
    nodeZ1->pickInput(5, 1);

    graph.addNode(nodeA);
    graph.addNode(nodeB);
    graph.addNode(scopeCondition);
    graph.addNode(nodeCondition);
    graph.addNode(nodeSwitch);
    graph.addNode(nodeZ0);
    graph.addNode(nodeZ1);

    Nd4jStatus status = GraphExecutioner::execute(&graph);

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(flowPath.isNodeActive(nodeZ0->id()));
    ASSERT_TRUE(!flowPath.isNodeActive(nodeZ1->id()));

    auto z = graph.getVariableSpace()->getVariable(6)->getNDArray();

    // abs(-119) = 119; Neg(119) = 119
    ASSERT_NEAR(-119.f, z->e<float>(0), 1e-5);
}