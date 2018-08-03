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
// Created by raver119 on 11.12.17.
//

#include "testlayers.h"
#include <graph/GraphHolder.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class GraphHolderTests : public testing::Test {
public:

};

TEST_F(GraphHolderTests, SimpleTests_1) {
    Graph<float> graph;
    Nd4jLong graphId = 119;
    GraphHolder::getInstance()->registerGraph(graphId, &graph);

    ASSERT_TRUE(GraphHolder::getInstance()->hasGraph<float>(graphId));

    GraphHolder::getInstance()->forgetGraph<float>(graphId);

    ASSERT_FALSE(GraphHolder::getInstance()->hasGraph<float>(graphId));
}



TEST_F(GraphHolderTests, SimpleTests_2) {
    auto graph = new Graph<float>;
    Nd4jLong graphId = 117;
    GraphHolder::getInstance()->registerGraph(graphId, graph);

    ASSERT_TRUE(GraphHolder::getInstance()->hasGraph<float>(graphId));

    auto graph2 = GraphHolder::getInstance()->cloneGraph<float>(graphId);

    ASSERT_TRUE(graph != graph2);
    ASSERT_TRUE(graph2 != nullptr);

    GraphHolder::getInstance()->forgetGraph<float>(graphId);

    ASSERT_FALSE(GraphHolder::getInstance()->hasGraph<float>(graphId));

    delete graph;
    delete graph2;
}


TEST_F(GraphHolderTests, SimpleTests_3) {
    auto graph = new Graph<float>;
    Nd4jLong graphId = 117;
    GraphHolder::getInstance()->registerGraph(graphId, graph);

    ASSERT_TRUE(GraphHolder::getInstance()->hasGraph<float>(graphId));

    auto graph2 = GraphHolder::getInstance()->cloneGraph<float>(graphId);

    ASSERT_TRUE(graph != graph2);
    ASSERT_TRUE(graph2 != nullptr);

    GraphHolder::getInstance()->dropGraph<float>(graphId);

    ASSERT_FALSE(GraphHolder::getInstance()->hasGraph<float>(graphId));


    delete graph2;
}