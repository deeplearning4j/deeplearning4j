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
// Created by raver119 on 15.12.17.
//
#include "testlayers.h"
#include <Graph.h>
#include <chrono>
#include <Node.h>
#include <helpers/OpTracker.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::graph;

class OpTrackerTests : public testing::Test {
public:
    int numIterations = 10;
    int poolSize = 10;

    OpTrackerTests() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(OpTrackerTests, Test_Existence_1) {
    nd4j::_loader loader;

    // nd4j_printf("Groups: %i; Operations: %i\n", OpTracker::getInstance()->totalGroups(), OpTracker::getInstance()->totalOperations());

    ASSERT_TRUE(OpTracker::getInstance()->totalGroups() > 0);
    ASSERT_TRUE(OpTracker::getInstance()->totalOperations() > 0);

    OpTracker::getInstance()->exportOperations();
}

TEST_F(OpTrackerTests, Test_Ops_List_1) {
    nd4j::ops::less op;
    auto vec = OpRegistrator::getInstance()->getAllHashes();

    // nd4j_printf("Total ops: %lld\n", vec.size());
    // nd4j_printf("Less hash: %lld\n", op.getOpHash());

    for (const auto &v: vec) {
        if (v == 5484196977525668316L) {
            auto op = OpRegistrator::getInstance()->getOperation(v);
            // nd4j_printf("OpName: %s\n", op->getOpName()->c_str());
        }
    }
}



