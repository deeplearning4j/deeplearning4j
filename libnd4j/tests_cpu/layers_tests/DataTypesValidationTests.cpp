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
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <NDArray.h>
#include <Context.h>
#include <Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/col2im.h>

using namespace nd4j;

using namespace nd4j;
using namespace nd4j::graph;

class DataTypesValidationTests : public testing::Test {
public:

};

TEST_F(DataTypesValidationTests, Basic_Test_1) {
    auto input = NDArrayFactory::create<int8_t>('c', {1, 1, 1, 4});
    auto weights = NDArrayFactory::create<int8_t>('c', {1, 1, 1, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 4, 1, 4}, {2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8.});

    weights.assign(2.0);
    input.linspace(1);

    nd4j::ops::conv2d op;
    auto result = op.execute({&input, &weights}, {}, {1, 1, 1, 1, 0, 0, 1, 1, 0, 0}, {});

    ASSERT_EQ(ND4J_STATUS_VALIDATION, result->status());

    delete result;
}

TEST_F(DataTypesValidationTests, Basic_Test_2) {
    auto input = NDArrayFactory::create<float16>('c', {1, 1, 1, 4});
    auto weights = NDArrayFactory::create<float16>('c', {1, 1, 1, 4});
    auto exp = NDArrayFactory::create<float16>('c', {1, 4, 1, 4}, {2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8.});

    weights.assign(2.0);
    input.linspace(1);

    nd4j::ops::conv2d op;
    auto result = op.execute({&input, &weights}, {}, {1, 1, 1, 1, 0, 0, 1, 1, 0, 0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DataTypesValidationTests, Basic_Test_3) {
    auto input = NDArrayFactory::create<bfloat16>('c', {1, 1, 1, 4});
    auto weights = NDArrayFactory::create<bfloat16>('c', {1, 1, 1, 4});
    auto exp = NDArrayFactory::create<bfloat16>('c', {1, 4, 1, 4}, {2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8.});
    auto out = NDArrayFactory::create<bfloat16>('c', {1, 4, 1, 4});

    weights.assign(2.0);
    input.linspace(1);

    nd4j::ops::conv2d op;
    auto result = op.execute({&input, &weights}, {&out}, {}, {1, 1, 1, 1, 0, 0, 1, 1, 0, 0}, {});
    ASSERT_EQ(Status::OK(), result);

    ASSERT_EQ(exp, out);
}

TEST_F(DataTypesValidationTests, Basic_Test_4) {
    auto input = NDArrayFactory::create<int8_t>('c', {1, 1, 1, 4});
    auto weights = NDArrayFactory::create<float16>('c', {1, 1, 1, 4});
    auto exp = NDArrayFactory::create<float>('c', {1, 4, 1, 4}, {2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8.});
    auto out = NDArrayFactory::create<int8_t>('c', {1, 4, 1, 4});

    weights.assign(2.0);
    input.linspace(1);

    nd4j::ops::conv2d op;
    auto result = op.execute({&input, &weights}, {&out}, {}, {1, 1, 1, 1, 0, 0, 1, 1, 0, 0}, {});
    ASSERT_EQ(ND4J_STATUS_VALIDATION, result);
}

TEST_F(DataTypesValidationTests, cast_1) {

    float16 x = static_cast<float16>(1.f);
    float y = static_cast<float16>(x);

    ASSERT_TRUE(1.f == x);
    ASSERT_TRUE(y == x);
}
