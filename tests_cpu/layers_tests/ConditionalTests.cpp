//
// Created by raver119 on 16.10.2017.
//

#include "testlayers.h"
#include <Graph.h>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::graph;

class ConditionalTests : public testing::Test {
public:

};


TEST_F(ConditionalTests, BasicTests_1) {
    Graph<float> graph;

    auto x = NDArrayFactory<float>::valueOf({2, 2}, 1.0);
    auto y0 = NDArrayFactory<float>::valueOf({2, 2}, 5.0);
    auto y1 = NDArrayFactory<float>::valueOf({2, 2}, -5.0);
    auto scalar = NDArrayFactory<float>::scalar(1.0);

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

    auto nodeF = new Node<float>(OpType_TRANSFORM, 0, 5, {-1, -2});
    nodeF->setScopeInfo(2, "scopeFalse");

    auto nodeT = new Node<float>(OpType_TRANSFORM, 1, 6, {-1, -2});
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

    ASSERT_NEAR(6.0, conditionalResult->meanNumber(), 1e-5);

}