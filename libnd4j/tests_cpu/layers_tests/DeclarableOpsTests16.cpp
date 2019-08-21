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


class DeclarableOpsTests16 : public testing::Test {
public:

    DeclarableOpsTests16() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests16, test_repeat_119) {
    auto x = NDArrayFactory::create<double>('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    auto e = NDArrayFactory::create<double>('c', {4, 3}, {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6});

    nd4j::ops::repeat op;
    auto result = op.execute({&x}, {}, {2, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests16, test_scatter_update_119) {
    auto x = NDArrayFactory::create<float>('c', {3}, {1, 1, 1});
    auto y = NDArrayFactory::create<int>(0);
    auto w = NDArrayFactory::create<float>(3.0f);
    auto e = NDArrayFactory::create<float>('c', {3}, {3.f, 1.f, 1.f});

    nd4j::ops::scatter_upd op;
    auto result = op.execute({&x, &y, &w}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}