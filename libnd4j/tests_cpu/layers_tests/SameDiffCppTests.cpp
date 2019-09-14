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
#include <chrono>
#include <NDArray.h>
#include <helpers/RandomLauncher.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <ops/declarable/CustomOperations.h>
#include <samediff/samediff_cpp.h>

using namespace nd4j;

class SameDiffCppTests : public testing::Test {
public:

};

TEST_F(SameDiffCppTests, basic_cpp_create_test_1) {
    auto e = NDArrayFactory::create<float>('c', {4}, {2.f, 3.f, 4.f, 5.f});

    auto sd = samediff::create();

    auto x = sd.variable(NDArrayFactory::create<float>('c', {4}, {1.f, 2.f, 3.f, 4.f}), true,"x");
    auto y = sd.variable(NDArrayFactory::create<float>('c', {4}, {1.f, 1.f, 1.f, 1.f}),  true, "y");

    auto z = samediff::arithmetic::Add(sd, x, y);

    sd.execute();

    auto result = z.array();

    ASSERT_EQ(e, result);
}

TEST_F(SameDiffCppTests, basic_cpp_create_test_2) {
    auto e = NDArrayFactory::create<float>('c', {4}, {-2.f, -3.f, -4.f, -5.f});

    auto sd = samediff::create();

    auto x = sd.variable(NDArrayFactory::create<float>('c', {4}, {1.f, 2.f, 3.f, 4.f}), true, "x");
    auto y = sd.variable(NDArrayFactory::create<float>('c', {4}, {1.f, 1.f, 1.f, 1.f}), true, "y");

    auto sum = samediff::arithmetic::Add(sd, x, y);
    auto z = samediff::arithmetic::Neg(sd, sum);

    sd.execute();

    auto result = z.array();

    ASSERT_EQ(e, result);
}

TEST_F(SameDiffCppTests, basic_cpp_operators_test_1) {
    auto e = NDArrayFactory::create<float>('c', {4}, {2.f, 3.f, 4.f, 5.f});

    auto sd = samediff::create();

    auto x = sd.variable(NDArrayFactory::create<float>('c', {4}, {1.f, 2.f, 3.f, 4.f}),  true,"x");
    auto y = sd.variable(NDArrayFactory::create<float>('c', {4}, {1.f, 1.f, 1.f, 1.f}), true, "y");

    auto z = x + y;

    sd.execute();

    auto result = z.array();

    ASSERT_EQ(e, result);
}

TEST_F(SameDiffCppTests, basic_cpp_primitives_test_1) {
    auto e = NDArrayFactory::create<float>('c', {4}, {3.f, 4.f, 5.f, 6.f});

    auto sd = samediff::create();

    auto x = sd.variable(NDArrayFactory::create<float>('c', {4}, {1.f, 2.f, 3.f, 4.f}), true, "x");

    auto y = 1.0f + x;
    auto z = y + 1.0f;

    sd.execute();

    auto result = z.array();

    ASSERT_EQ(e, result);
}