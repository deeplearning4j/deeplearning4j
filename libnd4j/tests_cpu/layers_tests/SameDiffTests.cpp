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

class SameDiffTests : public testing::Test {
public:

};

TEST_F(SameDiffTests, basic_cpp_create_test_1) {
    auto e = NDArrayFactory::create<float>('c', {4}, {2.f, 3.f, 4.f, 5.f});

    auto sd = samediff::create();

    auto x = sd.variable("x", NDArrayFactory::create<float>('c', {4}, {1.f, 2.f, 3.f, 4.f}));
    auto y = sd.variable("y", NDArrayFactory::create<float>('c', {4}, {1.f, 1.f, 1.f, 1.f}));

    auto z = samediff::arithmetic::add(sd, x, y);

    samediff::execute(sd);

    auto result = z.array();

    ASSERT_EQ(e, result);
}

TEST_F(SameDiffTests, basic_cpp_create_test_2) {
    auto e = NDArrayFactory::create<float>('c', {4}, {-2.f, -3.f, -4.f, -5.f});

    auto sd = samediff::create();

    auto x = sd.variable("x", NDArrayFactory::create<float>('c', {4}, {1.f, 2.f, 3.f, 4.f}));
    auto y = sd.variable("y", NDArrayFactory::create<float>('c', {4}, {1.f, 1.f, 1.f, 1.f}));

    auto sum = samediff::arithmetic::add(sd, x, y);
    auto z = samediff::arithmetic::neg(sd, sum);

    samediff::execute(sd);

    auto result = z.array();

    ASSERT_EQ(e, result);
}