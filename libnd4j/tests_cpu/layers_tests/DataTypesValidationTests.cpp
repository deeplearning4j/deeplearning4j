/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/NDArray.h>
#include <graph/Context.h>
#include <graph/Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/col2im.h>
#include <helpers/RandomLauncher.h>

using namespace sd;
using namespace sd::graph;

class DataTypesValidationTests : public testing::Test {
public:

};

TEST_F(DataTypesValidationTests, Basic_Test_1) {
    auto input = NDArrayFactory::create<int8_t>('c', {1, 1, 1, 4});
    auto weights = NDArrayFactory::create<int8_t>('c', {1, 1, 1, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 4, 1, 4}, {2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8.});

    weights.assign(2.0);
    input.linspace(1);

    sd::ops::conv2d op;
    auto result = op.evaluate({&input, &weights}, {1, 1, 1, 1, 0, 0, 1, 1, 0, 0});

    ASSERT_EQ(ND4J_STATUS_VALIDATION, result.status());
}

TEST_F(DataTypesValidationTests, Basic_Test_2) {
    auto input = NDArrayFactory::create<float16>('c', {1, 1, 1, 4});
    auto weights = NDArrayFactory::create<float16>('c', {1, 1, 1, 4});
    auto exp = NDArrayFactory::create<float16>('c', {1, 4, 1, 4}, {2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8.});

    weights.assign(2.0);
    input.linspace(1);

    sd::ops::conv2d op;
    auto result = op.evaluate({&input, &weights}, {1, 1, 1, 1, 0, 0, 1, 1, 0, 0});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

    ASSERT_TRUE(exp.equalsTo(z));

}


TEST_F(DataTypesValidationTests, Basic_Test_3) {
    auto input = NDArrayFactory::create<bfloat16>('c', {1, 1, 1, 4});
    auto weights = NDArrayFactory::create<bfloat16>('c', {1, 1, 1, 4});
    auto exp = NDArrayFactory::create<bfloat16>('c', {1, 4, 1, 4}, {2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6., 8.});
    auto out = NDArrayFactory::create<bfloat16>('c', {1, 4, 1, 4});

    weights.assign(2.0);
    input.linspace(1);

    sd::ops::conv2d op;
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

    sd::ops::conv2d op;
    auto result = op.execute({&input, &weights}, {&out}, {}, {1, 1, 1, 1, 0, 0, 1, 1, 0, 0}, {});
    ASSERT_EQ(ND4J_STATUS_VALIDATION, result);
}

TEST_F(DataTypesValidationTests, test_bfloat16_rand_1) {
    auto x = NDArrayFactory::create<bfloat16>('c', {5, 10});
    RandomGenerator gen(119, 120);
    RandomLauncher::fillUniform(LaunchContext::defaultContext(), gen, &x, 1, 6);

    ASSERT_TRUE(x.sumNumber().e<float>(0) != 0.f);
}

TEST_F(DataTypesValidationTests, test_bfloat16_rand_2) {
    auto x = NDArrayFactory::create<bfloat16>('c', {5, 10});
    RandomGenerator gen(119, 120);
    RandomLauncher::fillGaussian(LaunchContext::defaultContext(), gen, &x, 0, 1);

    ASSERT_TRUE(x.sumNumber().e<float>(0) != 0.f);
}

TEST_F(DataTypesValidationTests, cast_1) {

    float16 x = static_cast<float16>(1.f);
    float y = static_cast<float16>(x);

    ASSERT_TRUE(static_cast<float16>(1.f) == x);
    ASSERT_TRUE(y == static_cast<float>(x));
}

TEST_F(DataTypesValidationTests, test_bits_hamming_distance_1) {
    auto x = NDArrayFactory::create<int>('c', {3}, {0b01011000, 0b01011111, 0b01111110});
    auto y = NDArrayFactory::create<int>('c', {3}, {0b00010110, 0b01011000, 0b01011000});
    auto z = NDArrayFactory::create<uint64_t>(0);

    Context ctx(1);
    ctx.setInputArray(0, &x);
    ctx.setInputArray(1, &y);
    ctx.setOutputArray(0, &z);

    sd::ops::bits_hamming_distance op;
    auto status = op.execute(&ctx);
    ASSERT_NE(Status::OK(), status);
}

TEST_F(DataTypesValidationTests, test_bits_hamming_distance_2) {
    auto x = NDArrayFactory::create<int>('c', {3}, {0b01011000, 0b01011111, 0b01111110});
    auto y = NDArrayFactory::create<int>('c', {3}, {0b00010110, 0b01011000, 0b01011000});
    auto z = NDArrayFactory::create<Nd4jLong>(0);

    Context ctx(1);
    ctx.setInputArray(0, &x);
    ctx.setInputArray(1, &y);
    ctx.setOutputArray(0, &z);

    sd::ops::bits_hamming_distance op;
    auto status = op.execute(&ctx);
    ASSERT_EQ(Status::OK(), status);
}