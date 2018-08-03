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
//  @author raver119@gmail.com
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <NativeOps.h>
#include <helpers/BitwiseUtils.h>

using namespace nd4j;
using namespace nd4j::graph;

class SingleDimTests : public testing::Test {
public:

};

TEST_F(SingleDimTests, Test_Create_1) {
    NDArray<float> x('c', {5}, {1, 2, 3, 4, 5});
    ASSERT_EQ(5, x.lengthOf());
    ASSERT_EQ(1, x.rankOf());
    ASSERT_TRUE(x.isVector());
    ASSERT_TRUE(x.isRowVector());
    ASSERT_FALSE(x.isMatrix());
}

TEST_F(SingleDimTests, Test_Add_1) {
    NDArray<float> x('c', {3}, {1, 2, 3});
    NDArray<float> exp('c', {3}, {2, 3, 4});

    x += 1.0f;

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}


TEST_F(SingleDimTests, Test_Pairwise_1) {
    NDArray<float> x('c', {3}, {1, 2, 3});
    NDArray<float> exp('c', {3}, {2, 4, 6});

    x += x;

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(SingleDimTests, Test_Concat_1) {
    NDArray<float> x('c', {3}, {1, 2, 3});
    NDArray<float> y('c', {3}, {4, 5, 6});
    NDArray<float> exp('c', {6}, {1, 2, 3, 4, 5, 6});

    nd4j::ops::concat<float> op;
    auto result = op.execute({&x, &y}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(SingleDimTests, Test_Reduce_1) {
    NDArray<float> x('c', {3}, {1, 2, 3});

    float r = x.template reduceNumber<simdOps::Sum<float>>();

    ASSERT_NEAR(6.0f, r, 1e-5f);
}

TEST_F(SingleDimTests, Test_IndexReduce_1) {
    NDArray<float> x('c', {3}, {1, 2, 3});

    float r = x.template indexReduceNumber<simdOps::IndexMax<float>>();

    ASSERT_NEAR(2.0f, r, 1e-5f);
}


TEST_F(SingleDimTests, Test_ExpandDims_1) {
    NDArray<float> x('c', {3}, {1, 2, 3});
    NDArray<float> exp('c', {1, 3}, {1, 2, 3});

    nd4j::ops::expand_dims<float> op;
    auto result = op.execute({&x}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(SingleDimTests, Test_ExpandDims_2) {
    NDArray<float> x('c', {3}, {1, 2, 3});
    NDArray<float> exp('c', {3, 1}, {1, 2, 3});

    nd4j::ops::expand_dims<float> op;
    auto result = op.execute({&x}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(SingleDimTests, Test_Squeeze_1) {
    std::vector<Nd4jLong> vecS({1});
    std::vector<float> vecB({3.0f});
    NDArray<float> x('c', vecS, vecB);
    NDArray<float> exp(3.0f);

    nd4j::ops::squeeze<float> op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_EQ(exp.rankOf(), z->rankOf());
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(SingleDimTests, Test_Squeeze_2) {
    NDArray<float> x('c', {3}, {1, 2, 3});
    NDArray<float> exp('c', {3}, {1, 2, 3});

    nd4j::ops::squeeze<float> op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(SingleDimTests, Test_Reshape_1) {
    NDArray<float> x('c', {1, 3}, {1, 2, 3});
    NDArray<float> exp('c', {3}, {1, 2, 3});

    nd4j::ops::reshape<float> op;
    auto result = op.execute({&x}, {}, {-99, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(SingleDimTests, Test_Reshape_2) {
    NDArray<float> x('c', {3}, {1, 2, 3});
    NDArray<float> exp('c', {1, 3}, {1, 2, 3});

    nd4j::ops::reshape<float> op;
    auto result = op.execute({&x}, {}, {-99, 1, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(SingleDimTests, Test_Permute_1) {
    NDArray<float> x('c', {3}, {1, 2, 3});
    NDArray<float> exp('c', {3}, {1, 2, 3});

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}