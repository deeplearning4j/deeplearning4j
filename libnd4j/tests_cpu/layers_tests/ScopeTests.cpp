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
// Created by raver119 on 15.10.2017.
//
#include <graph/Graph.h>
#include <graph/Node.h>
#include <ops/declarable/CustomOperations.h>

#include "testlayers.h"

using namespace sd;
using namespace sd::graph;

class ScopeTests : public NDArrayTests {
 public:
};

TEST_F(ScopeTests, BasicTests_1) {
  Graph graph;

  auto x = NDArrayFactory::create_<float>('c', {2, 2});
  x->assign(0.0f);

  auto variableSpace = graph.getVariableSpace();
  variableSpace->putVariable(-1, x);

  ops::OpScope opScope;

  auto scopeBody = new Node(OpType_LOGIC, 10, 1);
  scopeBody->setName("scopeBody");
  scopeBody->setCustomOp(&opScope);

  graph.addNode(scopeBody);

  ASSERT_EQ(1, graph.totalNodes());

  auto scopedB0 = new Node(OpType_SCALAR, 0, 6, {-1}, {}, {}, 1.0f);
  scopedB0->markInplace(true);
  scopedB0->setScopeInfo(1, "scopeBody");

  graph.addNode(scopedB0);

  ASSERT_EQ(1, graph.totalNodes());
}

