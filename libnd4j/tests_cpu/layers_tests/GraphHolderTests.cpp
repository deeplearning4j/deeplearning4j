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
// Created by raver119 on 11.12.17.
//
#include <graph/GraphHolder.h>

#include "testlayers.h"

using namespace sd;
using namespace sd::ops;
using namespace sd::graph;

class GraphHolderTests : public NDArrayTests {
 public:
};

TEST_F(GraphHolderTests, SimpleTests_1) {
  Graph graph;
  sd::LongType graphId = 119;
  GraphHolder::getInstance().registerGraph(graphId, &graph);

  ASSERT_TRUE(GraphHolder::getInstance().hasGraph(graphId));

  GraphHolder::getInstance().forgetGraph(graphId);

  ASSERT_FALSE(GraphHolder::getInstance().hasGraph(graphId));
}

TEST_F(GraphHolderTests, SimpleTests_2) {
  auto graph = new Graph;
  sd::LongType graphId = 117;
  GraphHolder::getInstance().registerGraph(graphId, graph);

  ASSERT_TRUE(GraphHolder::getInstance().hasGraph(graphId));

  auto graph2 = GraphHolder::getInstance().cloneGraph(graphId);

  ASSERT_TRUE(graph != graph2);
  ASSERT_TRUE(graph2 != nullptr);

  GraphHolder::getInstance().forgetGraph(graphId);

  ASSERT_FALSE(GraphHolder::getInstance().hasGraph(graphId));

  delete graph;
  delete graph2;
}

TEST_F(GraphHolderTests, SimpleTests_3) {
  auto graph = new Graph;
  sd::LongType graphId = 117;
  GraphHolder::getInstance().registerGraph(graphId, graph);

  ASSERT_TRUE(GraphHolder::getInstance().hasGraph(graphId));

  auto graph2 = GraphHolder::getInstance().cloneGraph(graphId);

  ASSERT_TRUE(graph != graph2);
  ASSERT_TRUE(graph2 != nullptr);

  GraphHolder::getInstance().dropGraph(graphId);

  ASSERT_FALSE(GraphHolder::getInstance().hasGraph(graphId));

  delete graph2;
}
