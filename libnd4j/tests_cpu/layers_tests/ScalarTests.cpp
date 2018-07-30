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

class ScalarTests : public testing::Test {
public:

};

TEST_F(ScalarTests, Test_Create_1) {
    NDArray<float> x(2.0f);

    ASSERT_EQ(0, x.rankOf());
    ASSERT_EQ(1, x.lengthOf());
    ASSERT_TRUE(x.isScalar());
    ASSERT_FALSE(x.isVector());
    ASSERT_FALSE(x.isRowVector());
    ASSERT_FALSE(x.isColumnVector());
    ASSERT_FALSE(x.isMatrix());
}

TEST_F(ScalarTests, Test_Add_1) {
    NDArray<float> x(2.0f);
    NDArray<float> exp(5.0f);

    x += 3.0f;

    ASSERT_NEAR(5.0f, x.getScalar(0), 1e-5f);
    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(ScalarTests, Test_Add_2) {
    NDArray<float> x(2.0f);
    NDArray<float> y(3.0f);
    NDArray<float> exp(5.0f);

    x += y;

    ASSERT_NEAR(5.0f, x.getScalar(0), 1e-5f);
    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(ScalarTests, Test_Add_3) {
    NDArray<float> x('c', {3}, {1, 2, 3});
    NDArray<float> y(3.0f);
    NDArray<float> exp('c', {3}, {4, 5, 6});

    x += y;

    ASSERT_TRUE(exp.isSameShape(&x));

    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(ScalarTests, Test_EQ_1) {
    NDArray<float> x(2.0f);
    NDArray<float> y(3.0f);

    ASSERT_TRUE(y.isSameShape(&x));
    ASSERT_FALSE(y.equalsTo(&x));
}

TEST_F(ScalarTests, Test_Concat_1) {
    NDArray<float> t(1.0f);
    NDArray<float> u(2.0f);
    NDArray<float> v(3.0f);
    NDArray<float> exp('c', {3}, {1, 2, 3});

    nd4j::ops::concat<float> op;
    auto result = op.execute({&t, &u, &v}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ScalarTests, Test_Concat_2) {
    NDArray<float> t(1.0f);
    NDArray<float> u('c', {3}, {2, 3, 4});
    NDArray<float> v(5.0f);
    NDArray<float> exp('c', {5}, {1, 2, 3, 4, 5});

    nd4j::ops::concat<float> op;
    auto result = op.execute({&t, &u, &v}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ScalarTests, Test_Concat_3) {
    NDArray<float> t('c', {3}, {1, 2, 3});
    NDArray<float> u(4.0f);
    NDArray<float> v(5.0f);
    NDArray<float> exp('c', {5}, {1, 2, 3, 4, 5});

    nd4j::ops::concat<float> op;
    auto result = op.execute({&t, &u, &v}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printShapeInfo("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ScalarTests, Test_ExpandDims_1) {
    NDArray<float> x(2.0f);
    NDArray<float> exp('c', {1}, {2.0f});

    nd4j::ops::expand_dims<float> op;
    auto result = op.execute({&x}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ScalarTests, Test_Squeeze_1) {
    NDArray<float> x(2.0f);
    NDArray<float> exp(2.0f);

    nd4j::ops::squeeze<float> op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ScalarTests, Test_Reshape_1) {
    NDArray<float> x(2.0f);
    NDArray<float> exp('c', {1, 1, 1}, {2.0f});

    nd4j::ops::reshape<float> op;
    auto result = op.execute({&x}, {}, {-99, 1, 1, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ScalarTests, Test_Permute_1) {
    NDArray<float> x(3.0f);
    NDArray<float> exp(3.0f);

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ScalarTests, Test_Stack_1) {
    NDArray<float> t(1.0f);
    NDArray<float> u(2.0f);
    NDArray<float> v(3.0f);
    NDArray<float> exp('c', {3}, {1, 2, 3});

    nd4j::ops::stack<float> op;
    auto result = op.execute({&t, &u, &v}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ScalarTests, Test_Stack_2) {
    NDArray<float> t('c', {1, 1}, {1.0f});
    NDArray<float> u('c', {1, 1}, {2.0f});
    NDArray<float> v('c', {1, 1}, {3.0f});
    NDArray<float> w('c', {1, 1}, {4.0f});
    NDArray<float> exp('c', {4, 1, 1}, {1, 2, 3, 4});

    nd4j::ops::stack<float> op;
    auto result = op.execute({&t, &u, &v, &w}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printShapeInfo("z shape");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ScalarTests, Test_Concat_Scalar_1) {
    NDArray<float> t('c', {1, 1}, {1.0f});
    NDArray<float> u('c', {1, 1}, {2.0f});
    NDArray<float> v('c', {1, 1}, {3.0f});
    NDArray<float> w('c', {1, 1}, {4.0f});
    NDArray<float> exp('c', {4, 1}, {1, 2, 3, 4});

    nd4j::ops::concat<float> op;
    auto result = op.execute({&t, &u, &v, &w}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);    

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ScalarTests, Test_Concat_Scalar_2) {
    NDArray<float> t('c', {1, 1}, {1.0f});
    NDArray<float> u('c', {1, 1}, {2.0f});
    NDArray<float> v('c', {1, 1}, {3.0f});
    NDArray<float> w('c', {1, 1}, {4.0f});
    NDArray<float> exp('c', {1, 4}, {1, 2, 3, 4});

    nd4j::ops::concat<float> op;
    auto result = op.execute({&t, &u, &v, &w}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}