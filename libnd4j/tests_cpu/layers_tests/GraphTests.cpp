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
// @author raver119@gmail.com
//
#include <array/NDArray.h>
#include <flatbuffers/flatbuffers.h>
#include <graph/Graph.h>
#include <graph/GraphUtils.h>
#include <graph/Node.h>
#include <graph/scheme/graph_generated.h>
#include <graph/scheme/node_generated.h>
#include <ops/declarable/DeclarableOp.h>

#include <ops/declarable/generic/parity_ops.cpp>

#include "testlayers.h"

using namespace sd;
using namespace sd::graph;

class GraphTests : public NDArrayTests {
 public:
  GraphTests() {
  }
};

TEST_F(GraphTests, SingleInput1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0f);

  graph->getVariableSpace()->putVariable(-1, x);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_STRICT, transform::Cosine, 2, {1}, {3});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Abs, 3, {2}, {});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(3, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(3));

  auto node3 = graph->getVariableSpace()->getVariable(3)->getNDArray();

  ASSERT_NEAR(0.4161468, node3->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, DoubleInput1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto y = NDArrayFactory::create_<float>('c', {5, 5});
  y->assign(-1.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, y);
  graph->getVariableSpace()->putVariable(-3, z);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {3});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {-2}, {3});
  auto nodeC = new Node(OpType_PAIRWISE, pairwise::Add, 3, {1, 2}, {-3});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);

  ASSERT_EQ(2, graph->rootNodes());
  ASSERT_EQ(3, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(3.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, SingleInput3) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto v0 = NDArrayFactory::create_<float>('c', {5, 5});
  auto v1 = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, v0);
  graph->getVariableSpace()->putVariable(-3, v1);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2, 3});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {-2});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Ones, 3, {1}, {-3});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(3, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(1.4142135, v0->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
  ASSERT_NEAR(1.0, v1->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, SingleInput4) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto v0 = NDArrayFactory::create_<float>('c', {5, 5});
  auto v1 = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, v0);
  graph->getVariableSpace()->putVariable(-3, v1);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {3});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Neg, 3, {2}, {4, 5});

  auto nodeS = new Node(OpType_TRANSFORM_SAME, transform::Ones, 4, {3}, {-2});
  auto nodeE = new Node(OpType_TRANSFORM_SAME, transform::Identity, 5, {3}, {-3});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);
  graph->addNode(nodeS);
  graph->addNode(nodeE);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(5, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(1.0, v0->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
  ASSERT_NEAR(-1.4142135, v1->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, DoubleInput2) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto y = NDArrayFactory::create_<float>('c', {5, 5});
  y->assign(-1.0);

  auto z0 = NDArrayFactory::create_<float>('c', {5, 5});
  auto z1 = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, y);
  graph->getVariableSpace()->putVariable(-3, z0);
  graph->getVariableSpace()->putVariable(-4, z1);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {3});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Neg, 3, {2}, {-3});

  auto nodeT = new Node(OpType_TRANSFORM_SAME, transform::Abs, 11, {-2}, {12});
  auto nodeU = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 12, {11}, {13});
  auto nodeV = new Node(OpType_TRANSFORM_SAME, transform::Neg, 13, {12}, {-4});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);
  graph->addNode(nodeT);
  graph->addNode(nodeU);
  graph->addNode(nodeV);

  ASSERT_EQ(2, graph->rootNodes());
  ASSERT_EQ(6, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(-1.4142135, z0->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
  ASSERT_NEAR(-1.0, z1->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, DoubleInput3) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto y = NDArrayFactory::create_<float>('c', {5, 5});
  y->assign(-1.0);

  auto z0 = NDArrayFactory::create_<float>('c', {5, 5});
  auto z1 = NDArrayFactory::create_<float>('c', {5, 5});

  auto w = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, y);
  graph->getVariableSpace()->putVariable(-3, z0);
  graph->getVariableSpace()->putVariable(-4, z1);
  graph->getVariableSpace()->putVariable(-5, w);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {3});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Neg, 3, {2}, {-3, 21});

  auto nodeT = new Node(OpType_TRANSFORM_SAME, transform::Abs, 11, {-2}, {12});
  auto nodeU = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 12, {11}, {13});
  auto nodeV = new Node(OpType_TRANSFORM_SAME, transform::Neg, 13, {12}, {-4, 21});

  auto nodeW = new Node(OpType_PAIRWISE, pairwise::Add, 21, {3, 13}, {22});
  auto nodeZ = new Node(OpType_TRANSFORM_SAME, transform::Abs, 22, {21}, {-5});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);
  graph->addNode(nodeT);
  graph->addNode(nodeU);
  graph->addNode(nodeV);
  graph->addNode(nodeW);
  graph->addNode(nodeZ);

  ASSERT_EQ(2, graph->rootNodes());
  ASSERT_EQ(8, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(-1.4142135, z0->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
  ASSERT_NEAR(-1.0, z1->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  ASSERT_NEAR(2.4142135, w->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, QuadInput1) {
  auto graph = new Graph();

  auto x0 = NDArrayFactory::create_<float>('c', {5, 5});
  x0->assign(0.0);

  auto x1 = NDArrayFactory::create_<float>('c', {5, 5});
  x1->assign(-1.0);

  auto x2 = NDArrayFactory::create_<float>('c', {5, 5});
  x2->assign(-2.0);

  auto x3 = NDArrayFactory::create_<float>('c', {5, 5});
  x3->assign(-3.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});
  z->assign(119.0);

  graph->getVariableSpace()->putVariable(-1, x0);
  graph->getVariableSpace()->putVariable(-2, x1);
  graph->getVariableSpace()->putVariable(-3, x2);
  graph->getVariableSpace()->putVariable(-4, x3);
  graph->getVariableSpace()->putVariable(-5, z);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {11});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {-2}, {11});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Abs, 3, {-3}, {21});
  auto nodeD = new Node(OpType_TRANSFORM_SAME, transform::Abs, 4, {-4}, {21});

  auto nodeP1 = new Node(OpType_PAIRWISE, pairwise::Add, 11, {1, 2}, {31});
  auto nodeP2 = new Node(OpType_PAIRWISE, pairwise::Add, 21, {3, 4}, {31});

  auto nodeZ = new Node(OpType_PAIRWISE, pairwise::Add, 31, {11, 21}, {-5});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);
  graph->addNode(nodeD);
  graph->addNode(nodeP1);
  graph->addNode(nodeP2);
  graph->addNode(nodeZ);

  ASSERT_EQ(4, graph->rootNodes());
  ASSERT_EQ(7, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(6.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, InternalBranching1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(0.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, z);

  // 1.0
  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Ones, 1, {-1}, {11, 21});

  // -1
  auto nodeK = new Node(OpType_TRANSFORM_SAME, transform::Neg, 11, {1}, {12});

  // 2.0
  auto nodeL = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 12, {11}, {31});

  // -1
  auto nodeR = new Node(OpType_TRANSFORM_SAME, transform::Neg, 21, {1}, {22});

  // 1
  auto nodeS = new Node(OpType_TRANSFORM_SAME, transform::Neg, 22, {21}, {31});

  // 1.0
  auto nodeZ = new Node(OpType_PAIRWISE, pairwise::Add, 31, {12, 22}, {-2});

  graph->addNode(nodeA);
  graph->addNode(nodeK);
  graph->addNode(nodeL);
  graph->addNode(nodeR);
  graph->addNode(nodeS);
  graph->addNode(nodeZ);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(6, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_EQ(3, nodeZ->getLayer());

  ASSERT_NEAR(3.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, ReductionsTest1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  for (int r = 0; r < x->rows(); r++) {
    for (int c = 0; c < x->columns(); c++) {
      x->p(r, c, -c);
    }
  }

  auto z = NDArrayFactory::create_<float>('c', {5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, z);

  auto nodeA = new Node(OpType_REDUCE_FLOAT, reduce::Mean, 1, {-1}, {2}, {1}, {});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {-2});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(2, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(2.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, IndexReductionsTest1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  for (int r = 0; r < x->rows(); r++) {
    for (int c = 0; c < x->columns(); c++) {
      x->p(r, c, -c);
    }
  }

  auto z = NDArrayFactory::create_<sd::LongType>('c', {5, 1});
  auto axis = NDArrayFactory::create_<sd::LongType>('c', {1}, {1});
  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, z);

  auto nodeA = new Node(OpType_INDEX_REDUCE, indexreduce::IndexMin, 1, {-1}, {2}, {1});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {-2});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(2, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(4.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
  delete axis;
}

#if 0
TEST_F(GraphTests, AutoOutput1) {
    auto graph = new Graph();
    auto x = NDArrayFactory::create_<float>('c', {5, 5});
    x->assign(-2.0);

    graph->getVariableSpace()->putVariable(-1, x);

    auto nodeA = new Node(OpType_TRANSFORM_FLOAT, 0, 1, {-1}, {2});
    auto nodeB = new Node(OpType_TRANSFORM_FLOAT, 35, 2, {1}, {});

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(2, graph->totalNodes());

    graph->buildGraph();

    ASSERT_TRUE(graph->getVariableSpace()->getVariable(2) != nullptr);

    GraphExecutioner::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(1, outputs->size());

    ASSERT_TRUE(outputs->at(0) != nullptr);

    ASSERT_NEAR(-1.0, outputs->at(0)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

    delete outputs;
    delete graph;
}


TEST_F(GraphTests, AutoOutput2) {
    auto graph = new Graph();
    auto x = NDArrayFactory::create_<float>('c', {5, 5});
    x->assign(-2.0);

    graph->getVariableSpace()->putVariable(-1, x);

    auto nodeA = new Node(OpType_TRANSFORM_SAME, 0, 1, {-1}, {2, 3, -1});
    auto nodeB = new Node(OpType_TRANSFORM_SAME, 35, 2, {1}, {});
    auto nodeC = new Node(OpType_TRANSFORM_SAME, 6, 3, {1}, {});

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeC);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(3, graph->totalNodes());

    graph->buildGraph();

    ASSERT_TRUE(graph->getVariableSpace()->getVariable(-1) != nullptr);
    ASSERT_TRUE(graph->getVariableSpace()->getVariable(2) != nullptr);
    ASSERT_TRUE(graph->getVariableSpace()->getVariable(3) != nullptr);

    GraphExecutioner::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(2, outputs->size());

    ASSERT_TRUE(outputs->at(0) != nullptr);

    ASSERT_NEAR(-1.0, outputs->at(0)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
    ASSERT_NEAR(-2.0, outputs->at(1)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

    delete graph;
    delete outputs;
}
#endif

TEST_F(GraphTests, BroadcastTest1) {
  auto graph = new Graph();
  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(0.f);

  auto y = NDArrayFactory::create_<float>('c', {1, 5});
  for (int e = 0; e < y->columns(); e++) {
    y->p(e, (float)e + 1);
  }

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, y);
  graph->getVariableSpace()->putVariable(-3, z);

  auto nodeA = new Node(OpType_BROADCAST, broadcast::Subtract, 1, {-1, -2}, {2}, {1});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Neg, 2, {1}, {-3});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(3.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, ScalarTest1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, z);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {3});
  auto nodeE = new Node(OpType_SCALAR, scalar::Add, 3, {2}, {-2}, {}, 1.3f);

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeE);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(3, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(2.714213, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, SymbolicLookupTest1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {-2});

  std::string p("phi");
  std::string t("theta");

  nodeA->setName(&p);
  nodeB->setName(&t);

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  auto rX = graph->getVariableSpace()->getVariable(&a);
  auto rZ = graph->getVariableSpace()->getVariable(&o);

  std::string om("omicron");

  ASSERT_TRUE(rX->getNDArray() == vX->getNDArray());
  ASSERT_TRUE(rZ->getNDArray() == vZ->getNDArray());
  ASSERT_FALSE(graph->getVariableSpace()->hasVariable(&om));

  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1));
  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(2));

  GraphExecutioner::execute(graph);

  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(&p));
  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(&t));

  ASSERT_NEAR(1.4142135, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, OutputValidation1) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_EXPLICIT;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {-2});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(0, outputs->size());

  delete graph;
  delete outputs;
}

TEST_F(GraphTests, OutputValidation2) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_EXPLICIT;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {-2});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  graph->addOutput(-2);

  GraphExecutioner::execute(graph);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(1, outputs->size());

  ASSERT_NEAR(1.4142135, outputs->at(0)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
  delete outputs;
}

TEST_F(GraphTests, OutputValidation3) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_IMPLICIT;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  GraphExecutioner::execute(graph);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(1, outputs->size());

  ASSERT_NEAR(1.4142135, outputs->at(0)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
  delete outputs;
}

TEST_F(GraphTests, OutputValidation4) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_EXPLICIT_AND_IMPLICIT;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {-2});

  graph->addOutput(-1);

  // not a typo. we want this value only once
  graph->addOutput(-1);

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  GraphExecutioner::execute(graph);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(2, outputs->size());

  ASSERT_NEAR(1.4142135, outputs->at(1)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
  delete outputs;
}

TEST_F(GraphTests, OutputValidation5) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_VARIABLE_SPACE;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Sqrt, 2, {1}, {-2});

  graph->addOutput(-1);

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  GraphExecutioner::execute(graph);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(4, outputs->size());

  delete graph;
  delete outputs;
}

TEST_F(GraphTests, OutputValidation6) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_VARIABLE_SPACE;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {});

  // graph->addOutput(-1);

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  GraphExecutioner::execute(graph);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(4, outputs->size());
  delete graph;
  delete outputs;
}

TEST_F(GraphTests, TestMultiOutput1) {
  sd::ops::testop2i2o op1;
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto y = NDArrayFactory::create_<float>('c', {5, 5});
  y->assign(-3.0);

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, y);

  // Abs
  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {11});
  nodeA0->markInplace(false);
  auto nodeB0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {-2}, {11});
  nodeB0->markInplace(false);

  auto op = sd::ops::OpRegistrator::getInstance().getOperation("testop2i2o");

  // this op will add 1.0 to first input, and 2.0 for second input
  auto nodeT = new Node(op, 11, {1, 2}, {21, 31}, {}, 0.0f);
  nodeT->setName("TestOp2i2o");
  nodeT->markInplace(false);

  // this op will subtract this value from 1.0
  auto nodeX = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 21);
  nodeX->markInplace(false);
  nodeX->pickInput(11, 0);

  // this op will subtract this value from 1.0
  auto nodeY = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 31);
  nodeY->markInplace(false);
  nodeY->pickInput(11, 1);

  graph->addNode(nodeA0);
  graph->addNode(nodeB0);
  graph->addNode(nodeT);
  graph->addNode(nodeX);
  graph->addNode(nodeY);

  std::pair<int, int> pair0(11, 0);
  std::pair<int, int> pair1(11, 1);

  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(pair0));
  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(pair1));

  sd::Status status = GraphExecutioner::execute(graph);

  ASSERT_EQ(sd::Status::OK, status);

  ASSERT_NEAR(-2.0f, graph->getVariableSpace()->getVariable(21)->getNDArray()->meanNumber().e<float>(0), 1e-5);
  ASSERT_NEAR(-4.0f, graph->getVariableSpace()->getVariable(31)->getNDArray()->meanNumber().e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, TestDivergentNode1) {
  auto op = sd::ops::OpRegistrator::getInstance().getOperation("Switch");
  auto nodeY = new Node(op, 1);

  ASSERT_TRUE(nodeY->isDivergencePoint());
  ASSERT_TRUE(nodeY->isActive());

  delete nodeY;
}

TEST_F(GraphTests, MemoryEstimationTest1) {
  Graph graph;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  graph.getVariableSpace()->putVariable(-1, x);

  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {});
  nodeA1->markInplace(false);

  graph.addNode(nodeA0);
  graph.addNode(nodeA1);

  ASSERT_EQ(2, graph.totalNodes());
  ASSERT_EQ(1, graph.rootNodes());

  auto memReq = graph.estimateRequiredMemory();

  ASSERT_EQ(25 * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, MemoryEstimationTest2) {
  Graph graph;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  graph.getVariableSpace()->putVariable(-1, x);

  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {});

  graph.addNode(nodeA0);
  graph.addNode(nodeA1);

  ASSERT_EQ(2, graph.totalNodes());
  ASSERT_EQ(1, graph.rootNodes());

  auto memReq = graph.estimateRequiredMemory();

  ASSERT_EQ(0, memReq);
}

TEST_F(GraphTests, MemoryEstimationTest3) {
  Graph graph;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  graph.getVariableSpace()->putVariable(-1, x);

  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {3});
  auto nodeA2 = new Node(OpType_REDUCE_FLOAT, reduce::Mean, 3, {2}, {}, {});
  nodeA1->markInplace(false);

  graph.addNode(nodeA0);
  graph.addNode(nodeA1);
  graph.addNode(nodeA2);

  ASSERT_EQ(3, graph.totalNodes());
  ASSERT_EQ(1, graph.rootNodes());

  auto memReq = graph.estimateRequiredMemory();

  ASSERT_EQ(26 * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, MemoryEstimationTest4) {
  Graph graph;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  graph.getVariableSpace()->putVariable(-1, x);

  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {3});
  auto nodeA2 = new Node(OpType_REDUCE_FLOAT, reduce::Mean, 3, {2}, {}, {1});
  nodeA1->markInplace(false);

  graph.addNode(nodeA0);
  graph.addNode(nodeA1);
  graph.addNode(nodeA2);

  ASSERT_EQ(3, graph.totalNodes());
  ASSERT_EQ(1, graph.rootNodes());

  auto memReq = graph.estimateRequiredMemory();

  ASSERT_EQ(30 * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, MemoryEstimationTest5) {
  Graph graph;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  graph.getVariableSpace()->putVariable(-1, x);

  sd::ops::testcustom op;

  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {3});
  auto nodeA2 = new Node(&op, 3, {2}, {}, {});
  nodeA1->markInplace(false);

  graph.addNode(nodeA0);
  graph.addNode(nodeA1);
  graph.addNode(nodeA2);

  graph.buildGraph();

  ASSERT_EQ(3, graph.totalNodes());
  ASSERT_EQ(1, graph.rootNodes());

  auto memReq = graph.estimateRequiredMemory();

  ASSERT_EQ((25 + 100) * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, TestGraphInGraph_1) {
  // this one is external graph
  Graph graphA;

  // and this ons is embedded
  Graph graphB;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-5.0);

  auto modifier = NDArrayFactory::create_<float>('c', {5, 5});
  modifier->assign(3.0);

  graphA.getVariableSpace()->putVariable(-1, x);
  graphB.getVariableSpace()->putVariable(-2, modifier);

  // this is placeholder variable
  graphB.getVariableSpace()->putVariable(-1, new Variable(true));

  // abs, result is 5
  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  // 1-, result -4
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 2, {1}, {3});

  // graph should return 12: abs(3.0 x -4)
  auto nodeA2 = new Node(OpType_GRAPH, -1, 3, {2}, {4});

  // 1 - 12 = -11
  auto nodeA3 = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 4, {3}, {});

  nodeA2->setGraph(&graphB);

  graphA.addNode(nodeA0);
  graphA.addNode(nodeA1);
  graphA.addNode(nodeA2);
  graphA.addNode(nodeA3);

  // this is going to be PWT
  auto nodeB0 = new Node(OpType_PAIRWISE, pairwise::Multiply, 1, {-1, -2}, {2});
  auto nodeB1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {});

  graphB.addNode(nodeB0);
  graphB.addNode(nodeB1);

  graphB.buildGraph();
  graphA.buildGraph();

  ASSERT_EQ(0, nodeA0->getLayer());
  ASSERT_EQ(1, nodeA1->getLayer());
  ASSERT_EQ(2, nodeA2->getLayer());
  ASSERT_EQ(3, nodeA3->getLayer());

  ASSERT_EQ(0, nodeB0->getLayer());
  ASSERT_EQ(1, nodeB1->getLayer());

  sd::Status status = GraphExecutioner::execute(&graphA);
  ASSERT_EQ(sd::Status::OK, status);

  float m = graphA.getVariableSpace()->getVariable(4)->getNDArray()->meanNumber().e<float>(0);


  ASSERT_NEAR(-11.0, m, 1e-5);
}

// test for symbolic lookup
TEST_F(GraphTests, TestGraphInGraph_2) {
  // this one is external graph
  Graph graphA;

  // and this ons is embedded
  Graph graphB;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-5.0);

  auto modifier = NDArrayFactory::create_<float>('c', {5, 5});
  modifier->assign(3.0);

  std::string nameA1("_nodeA1");

  graphA.getVariableSpace()->putVariable(-1, x);
  graphB.getVariableSpace()->putVariable(-2, modifier);

  // this is placeholder variable
  auto placeHolder = new Variable(true);
  placeHolder->setName(&nameA1);
  graphB.getVariableSpace()->putVariable(-1, placeHolder);

  // abs, result is 5
  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  // 1-, result -4
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 2, {1}, {3});
  nodeA1->setName(nameA1);

  // graph should return 12: abs(3.0 x -4)
  auto nodeA2 = new Node(OpType_GRAPH, -1, 3, {2}, {4});

  // 1 - 12 = -11
  auto nodeA3 = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 4, {3}, {});

  nodeA2->setGraph(&graphB);

  graphA.addNode(nodeA0);
  graphA.addNode(nodeA1);
  graphA.addNode(nodeA2);
  graphA.addNode(nodeA3);

  // this is going to be PWT
  auto nodeB0 = new Node(OpType_PAIRWISE, pairwise::Multiply, 1, {-1, -2}, {2});
  auto nodeB1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {});

  graphB.addNode(nodeB0);
  graphB.addNode(nodeB1);

  graphB.buildGraph();
  graphA.buildGraph();

  ASSERT_EQ(0, nodeA0->getLayer());
  ASSERT_EQ(1, nodeA1->getLayer());
  ASSERT_EQ(2, nodeA2->getLayer());
  ASSERT_EQ(3, nodeA3->getLayer());

  ASSERT_EQ(0, nodeB0->getLayer());
  ASSERT_EQ(1, nodeB1->getLayer());

  sd::Status status = GraphExecutioner::execute(&graphA);
  ASSERT_EQ(sd::Status::OK, status);

  float m = graphA.getVariableSpace()->getVariable(4)->getNDArray()->meanNumber().e<float>(0);


  ASSERT_NEAR(-11.0, m, 1e-5);
}


TEST_F(GraphTests, Test_Inplace_Outputs_1) {
  auto x = NDArrayFactory::create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  auto exp = NDArrayFactory::create<float>('c', {6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  auto z = NDArrayFactory::create<float>('c', {2, 3});

  sd::ops::test_output_reshape op;
  auto result = op.execute({&x}, {&z}, {}, {}, {});
  ASSERT_EQ(sd::Status::OK, result);

  ASSERT_TRUE(exp.isSameShape(z));
  ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(GraphTests, Test_Inplace_Outputs_2) {
#ifndef __APPLE_OS__
  // we dont want testing this on apple. due to try/catch

  auto x = NDArrayFactory::create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  auto exp = NDArrayFactory::create<float>('c', {6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  auto z = NDArrayFactory::create<float>('c', {3, 3});

  bool failed = false;
  sd::ops::test_output_reshape op;
  try {
    op.execute({&x}, {&z}, {}, {}, {});

  } catch (const std::runtime_error& e) {
    failed = true;
  }

  ASSERT_TRUE(failed);
#endif
}



TEST_F(GraphTests, Test_Minifier_2) {
  // run preprocessor to produce single header
  // if all ok - return value is 0, if error - non-zero value will be returned
  ASSERT_EQ(0, GraphUtils::runPreprocessor("../include/ops/specials.h", "libnd4j_mini2.hpp"));
  // remove file from filesystem
#ifdef __linux__
  ASSERT_EQ(0, unlink("libnd4j_mini2.hpp"));
#endif
}

TEST_F(GraphTests, Test_Minifier_3) {
  // run preprocessor to produce single header
  // if all ok - return value is 0, if error - non-zero value will be returned
#ifdef __linux__
  ASSERT_EQ(0x100, GraphUtils::runPreprocessor("/include/ops/ops.h", "libnd4j_mini3.hpp"));
#endif

}
