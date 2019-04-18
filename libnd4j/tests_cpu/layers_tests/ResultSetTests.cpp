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
// Created by raver on 4/18/2019.
//

#include "testlayers.h"
#include <Graph.h>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::graph;

class ResultSetTests : public testing::Test {
public:

};

TEST_F(ResultSetTests, basic_test_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});

    auto tensors = x.allTensorsAlongDimension({1});
    ASSERT_EQ(3, tensors->size());

    ResultSet set = *tensors;
    ASSERT_EQ(3, tensors->size());
    ASSERT_EQ(3, set.size());

    for (int e = 0; e < set.size(); e++)
        ASSERT_EQ(5, set.at(e)->lengthOf());

    for (int e = 0; e < tensors->size(); e++)
        ASSERT_EQ(5, tensors->at(e)->lengthOf());
}