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


class DeclarableOpsTests14 : public testing::Test {
public:

    DeclarableOpsTests14() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests14, Test_Inf_Comparison_1) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 1.0/0.0, 5});
    auto y = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 1.0/0.0, 5});

    ASSERT_EQ(x, y);
}

TEST_F(DeclarableOpsTests14, Test_Inf_Comparison_2) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 1.0/0.0, 5});
    auto y = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, -1.0/0.0, 5});

    ASSERT_NE(x, y);
}

TEST_F(DeclarableOpsTests14, Test_Diag_Zeros_1) {
    auto x = NDArrayFactory::create<double>('c', {2}, {1, 2});
    auto z = NDArrayFactory::create<double>('c', {2, 2}, {-119, -119, -119, -119});
    auto exp = NDArrayFactory::create<double>('c', {2, 2}, {1, 0, 0, 2});

    nd4j::ops::diag op;
    auto status = op.execute({&x}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(exp, z);
}