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
// Created by raver119 on 11.10.2017.
//

#include "testlayers.h"
#include <vector>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpTuple.h>
#include <ops/declarable/OpRegistrator.h>
#include <GraphExecutioner.h>
#include <memory/MemoryReport.h>
#include <memory/MemoryUtils.h>
#include <MmulHelper.h>

using namespace nd4j;
using namespace nd4j::ops;

class OneOffTests : public testing::Test {
public:

};

TEST_F(OneOffTests, test_avg_pool_3d_1) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/avg_pooling3d.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    delete graph;
}

TEST_F(OneOffTests, test_non2d_0A_1) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/non2d_0A.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    delete graph;
}

TEST_F(OneOffTests, test_assert_scalar_float32_1) {
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/scalar_float32.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    delete graph;
}

TEST_F(OneOffTests, test_pad_1D_1) {
    auto e = NDArrayFactory::create<float>('c', {7}, {10.f,    0.778786f, 0.801198f, 0.724375f, 0.230894f, 0.727141f,   10.f});
    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/pad_1D.fb");

    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(4));

    auto z = graph->getVariableSpace()->getVariable(4)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    z->printIndexedBuffer("z");

    ASSERT_EQ(e, *z);
    delete graph;
}

TEST_F(OneOffTests, test_scatter_nd_update_1) {
    auto e = NDArrayFactory::create<float>('c', {10, 7}, {1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 0.20446908f, 0.37918627f, 0.99792874f, 0.71881700f, 0.18677747f, 0.78299069f, 0.55216062f, 0.40746713f, 0.92128086f, 0.57195139f, 0.44686234f, 0.30861020f, 0.31026053f, 0.09293187f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 0.95073712f, 0.45613325f, 0.95149803f, 0.88341522f, 0.54366302f, 0.50060666f, 0.39031255f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f, 1.00000000f});

    auto graph = GraphExecutioner::importFromFlatBuffers("./resources/scatter_nd_update.fb");
    ASSERT_TRUE(graph != nullptr);

    graph->printOut();

    Nd4jStatus status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(6));

    auto z = graph->getVariableSpace()->getVariable(6)->getNDArray();
    ASSERT_TRUE(z != nullptr);

    z->printIndexedBuffer("z");

    ASSERT_EQ(e, *z);

    delete graph;
}