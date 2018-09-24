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
// Created by raver119 on 13.01.2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class BackpropTests : public testing::Test {
public:

};

TEST_F(BackpropTests, Test_Add_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto y = NDArrayFactory::create<float>('c', {3, 4});
    auto e = NDArrayFactory::create<float>('c', {2, 3, 4});

    nd4j::ops::add_bp op;
    auto result = op.execute({&x, &y, &e}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());

    auto eps = result->at(0);
    auto grad = result->at(1);

    ASSERT_TRUE(x.isSameShape(eps));
    ASSERT_TRUE(y.isSameShape(grad));

    delete result;
}