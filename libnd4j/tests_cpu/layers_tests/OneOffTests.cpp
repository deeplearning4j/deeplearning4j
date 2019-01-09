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