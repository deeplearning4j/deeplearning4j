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
#include <NDArray.h>
#include <ops/declarable/OpTuple.h>

using namespace nd4j;
using namespace nd4j::ops;

class OpTupleTests : public testing::Test {
    public:
};

TEST_F(OpTupleTests, DirectConstructorTest1) {
    auto alpha = NDArrayFactory::create<float>('c', {1, 2});
    auto beta = NDArrayFactory::create<float>('c', {1, 2});
    OpTuple tuple("dummy", {alpha, beta}, {12.0f}, {1,2, 3});

    ASSERT_EQ("dummy", tuple._opName);
    ASSERT_EQ(2, tuple._inputs.size());
    ASSERT_EQ(0, tuple._outputs.size());
    ASSERT_EQ(1, tuple._tArgs.size());
    ASSERT_EQ(3, tuple._iArgs.size());
}

TEST_F(OpTupleTests, BuilderTest1) {
    auto alpha = NDArrayFactory::create<float>('c', {1, 2});
    auto beta = NDArrayFactory::create<float>('c', {1, 2});
    OpTuple tuple("dummy");
    tuple.addInput(alpha)
            ->addInput(beta)
            ->setTArgs({12.0f})
            ->setIArgs({1, 2, 3});


    ASSERT_EQ("dummy", tuple._opName);
    ASSERT_EQ(2, tuple._inputs.size());
    ASSERT_EQ(0, tuple._outputs.size());
    ASSERT_EQ(1, tuple._tArgs.size());
    ASSERT_EQ(3, tuple._iArgs.size());
}