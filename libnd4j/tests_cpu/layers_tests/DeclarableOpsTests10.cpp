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
// Created by raver on 8/4/2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>


using namespace nd4j;


class DeclarableOpsTests10 : public testing::Test {
public:

    DeclarableOpsTests10() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests10, Test_ArgMax_1) {
    NDArray<double> x('c', {3, 3});
    NDArray<double> e(8);

    x.linspace(1.0);


    nd4j::ops::argmax<double> op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());


    auto z = *result->at(0);

    ASSERT_EQ(e, z);

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_ArgMax_2) {
    NDArray<double> x('c', {3, 3});
    NDArray<double> y('c', {1}, {1.0});
    NDArray<double> e('c', {3}, {2.0, 2.0, 2.0});

    x.linspace(1.0);

    nd4j::ops::argmax<double> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = *result->at(0);

    //z.printIndexedBuffer("z");
    //z.printShapeInfo("z shape");

    ASSERT_EQ(e, z);

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_And_1) {
    NDArray<double> x('c', {4}, {1, 1, 0, 1});
    NDArray<double> y('c', {4}, {0, 0, 0, 1});
    NDArray<double> e('c', {4}, {0, 0, 0, 1});

    nd4j::ops::boolean_and<double> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_Or_1) {
    NDArray<double> x('c', {4}, {1, 1, 0, 1});
    NDArray<double> y('c', {4}, {0, 0, 0, 1});
    NDArray<double> e('c', {4}, {1, 1, 0, 1});

    nd4j::ops::boolean_or<double> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_Not_1) {
    NDArray<double> x('c', {4}, {1, 1, 0, 1});
    NDArray<double> y('c', {4}, {0, 0, 0, 1});
    NDArray<double> e('c', {4}, {1, 1, 1, 0});

    nd4j::ops::boolean_not<double> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_Size_at_1) {
    NDArray<double> x('c', {10, 20, 30});
    NDArray<double> e(20.0);

    nd4j::ops::size_at<double> op;
    auto result = op.execute({&x}, {}, {1});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Pad_SGO_Test_1) {

    NDArray<double> in({1., 1., 1., 1., 1.});
//    NDArray<double> pad('c', {1, 2}, {1., 1.});// = Nd4j.create(new double[]{1, 1}, new long[]{1, 2});
    NDArray<double> pad('c', {1, 2}, {1., 1.});
//    NDArray<double> value(10.0);

    NDArray<double> exp({10., 1., 1., 1., 1., 1., 10.});

    nd4j::ops::pad<double> op;

    auto res = op.execute({&in, &pad}, {10.0}, {0});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Unique_SGO_Test_1) {
    NDArray<double> input({3., 4., 3., 1., 3., 0., 2., 4., 2., 4.});
    NDArray<double> expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    NDArray<double> exp({3., 4., 1., 0., 2.});

    nd4j::ops::unique<double> op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    //res->at(0)->printIndexedBuffer("Unique values");
    //res->at(1)->printIndexedBuffer("Unique idxs");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

