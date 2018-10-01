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
#include <array/ArrayOptions.h>
#include <NDArray.h>
#include <NDArrayFactory.h>


using namespace nd4j;

class MultiDataTypeTests : public testing::Test {
public:

};

TEST_F(MultiDataTypeTests, DataTypeUtils_Test_1) {
    auto dtype = DataTypeUtils::pickPairwiseResultType(nd4j::INT32, nd4j::FLOAT32);

    ASSERT_EQ(nd4j::FLOAT32, dtype);
}

TEST_F(MultiDataTypeTests, DataTypeUtils_Test_2) {
    auto dtype = DataTypeUtils::pickPairwiseResultType(nd4j::INT32, nd4j::DOUBLE);
    ASSERT_EQ(nd4j::DOUBLE, dtype);

    ASSERT_EQ(nd4j::DOUBLE, DataTypeUtils::pickPairwiseResultType(nd4j::DOUBLE, nd4j::INT32));
}

TEST_F(MultiDataTypeTests, DataTypeUtils_Test_3) {
    auto dtype = DataTypeUtils::pickPairwiseResultType(nd4j::FLOAT32, nd4j::DOUBLE);
    ASSERT_EQ(nd4j::FLOAT32, dtype);
}


TEST_F(MultiDataTypeTests, Basic_Test_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto y = NDArrayFactory::create<double>('c', {2, 3}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
    auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

    auto z = x + y;

    ASSERT_EQ(e, z);
}

TEST_F(MultiDataTypeTests, Basic_Test_2) {
    auto x = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto y = NDArrayFactory::create<double>(2.0);
    auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

    auto z = x * y;

    ASSERT_EQ(e, z);
}


TEST_F(MultiDataTypeTests, Basic_Test_3) {
    auto x = NDArrayFactory::create<int>('c', {2, 3}, {0, 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>(2.0);
    auto e = NDArrayFactory::create<double>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

    auto z = x * y;

    ASSERT_EQ(e, z);
}

TEST_F(MultiDataTypeTests, Basic_Test_4) {
    auto x = NDArrayFactory::create<double>('c', {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto y = NDArrayFactory::create<float>(2.0);
    auto e = NDArrayFactory::create<double>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

    auto z = x * y;

    ASSERT_EQ(e, z);
}

TEST_F(MultiDataTypeTests, Basic_Test_5) {
    auto x = NDArrayFactory::create<int>('c', {2, 3}, {0, 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<Nd4jLong>(2);
    auto e = NDArrayFactory::create<int>('c', {2, 3}, {0, 2, 4, 6, 8, 10});

    auto z = x * y;

    ASSERT_EQ(e, z);
}

TEST_F(MultiDataTypeTests, Basic_Test_6) {
    auto x = NDArrayFactory::create<Nd4jLong>('c', {2, 3}, {0, 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<int>(2);
    auto e = NDArrayFactory::create<Nd4jLong >('c', {2, 3}, {0, 2, 4, 6, 8, 10});

    auto z = x * y;

    ASSERT_EQ(e, z);
}