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


class DeclarableOpsTests11 : public testing::Test {
public:

    DeclarableOpsTests11() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests11, test_mixed_biasadd_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 3});
    auto y = NDArrayFactory::create<float>('c', {3}, {1.f, 2.f, 3.f});
    auto z = NDArrayFactory::create<float>('c', {2, 3});
    auto exp = NDArrayFactory::create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 1.f, 2.f, 3.f});

    nd4j::ops::biasadd op;
    auto status = op.execute({&x, &y}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(exp, z);
}
