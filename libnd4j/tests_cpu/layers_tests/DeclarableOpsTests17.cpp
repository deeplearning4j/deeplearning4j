/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>
#include <array>


using namespace nd4j;


class DeclarableOpsTests17 : public testing::Test {
public:

    DeclarableOpsTests17() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests17, test_sparse_to_dense_1) {
    auto values = NDArrayFactory::create<float>({1.f, 2.f, 3.f});
    auto shape = NDArrayFactory::create<Nd4jLong>({3, 3});
    auto ranges = NDArrayFactory::create<Nd4jLong>({0,0, 1,1, 2,2});
    auto def = NDArrayFactory::create<float>(0.f);
    auto exp = NDArrayFactory::create<float>('c', {3, 3}, {1.f,0.f,0.f,  0.f,2.f,0.f,  0.f,0.f,3.f});


    nd4j::ops::compat_sparse_to_dense op;
    auto result = op.evaluate({&ranges, &shape, &values, &def});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

TEST_F(DeclarableOpsTests17, test_sparse_to_dense_2) {
    auto values = NDArrayFactory::string('c', {3}, {"alpha", "beta", "gamma"});
    auto shape = NDArrayFactory::create<Nd4jLong>({3, 3});
    auto ranges = NDArrayFactory::create<Nd4jLong>({0,0, 1,1, 2,2});
    auto def = NDArrayFactory::string("d");
    auto exp = NDArrayFactory::string('c', {3, 3}, {"alpha","d","d",  "d","beta","d",  "d","d","gamma"});


    nd4j::ops::compat_sparse_to_dense op;
    auto result = op.evaluate({&ranges, &shape, &values, &def});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

TEST_F(DeclarableOpsTests17, test_compat_string_split_1) {
    auto x = NDArrayFactory::string('c', {2}, {"first string", "second"});
    auto delimiter = NDArrayFactory::string(" ");

    auto exp0 = NDArrayFactory::create<Nd4jLong>({0,0, 0,1, 1,0});
    auto exp1 = NDArrayFactory::string('c', {3}, {"first", "string", "second"});

    nd4j::ops::compat_string_split op;
    auto result = op.evaluate({&x, &delimiter});
    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_EQ(2, result->size());

    auto z0 = result->at(0);
    auto z1 = result->at(1);

    ASSERT_TRUE(exp0.isSameShape(z0));
    ASSERT_TRUE(exp1.isSameShape(z1));

    ASSERT_EQ(exp0, *z0);
    ASSERT_EQ(exp1, *z1);

    delete result;
}