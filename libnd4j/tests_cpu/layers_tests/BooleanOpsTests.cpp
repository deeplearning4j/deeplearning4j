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
// Created by raver119 on 13.10.2017.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class BooleanOpsTests : public testing::Test {
public:

};


TEST_F(BooleanOpsTests, LtTest_1) {
    auto x = NDArray<float>::scalar(1.0f);
    auto y = NDArray<float>::scalar(2.0f);

    nd4j::ops::lt_scalar<float> op;


    ASSERT_TRUE(op.evaluate({x, y}));

    delete x;
    delete y;
}

TEST_F(BooleanOpsTests, LtTest_2) {
    auto x = NDArray<float>::scalar(2.0f);
    auto y = NDArray<float>::scalar(1.0f);

    nd4j::ops::lt_scalar<float> op;


    ASSERT_FALSE(op.evaluate({x, y}));

    delete x;
    delete y;
}

TEST_F(BooleanOpsTests, Is_non_decreasing_1) {
    NDArray<float> x('c', {2 , 2}, {1, 2, 4, 4});

    nd4j::ops::is_non_decreasing<float> op;

    ASSERT_TRUE(op.evaluate({&x}));

}

TEST_F(BooleanOpsTests, Is_non_decreasing_2) {
    NDArray<float> x('c', {2 , 2}, {1, 2, 4, 3});

    nd4j::ops::is_non_decreasing<float> op;

    ASSERT_FALSE(op.evaluate({&x}));

}

TEST_F(BooleanOpsTests, Is_strictly_increasing_1) {
    NDArray<float> x('c', {2 , 2}, {1, 2, 4, 5});

    nd4j::ops::is_strictly_increasing<float> op;

    ASSERT_TRUE(op.evaluate({&x}));

}

TEST_F(BooleanOpsTests, Is_strictly_increasing_2) {
    NDArray<float> x('c', {2 , 2}, {1, 2, 3, 3});

    nd4j::ops::is_strictly_increasing<float> op;

    ASSERT_FALSE(op.evaluate({&x}));

}

TEST_F(BooleanOpsTests, Is_strictly_increasing_3) {
    NDArray<float> x('c', {2 , 2}, {1, 2, 4, 3});

    nd4j::ops::is_strictly_increasing<float> op;

    ASSERT_FALSE(op.evaluate({&x}));

}

TEST_F(BooleanOpsTests, Is_numeric_tensor_1) {
    NDArray<float> x('c', {2 , 2}, {1, 2, 4, 3});

    nd4j::ops::is_numeric_tensor<float> op;

    ASSERT_TRUE(op.evaluate({&x}));

}
