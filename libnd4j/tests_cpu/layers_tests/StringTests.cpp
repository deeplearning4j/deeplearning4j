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


#include <NDArray.h>
#include <NDArrayFactory.h>
#include "testlayers.h"
#include <graph/Stash.h>

using namespace nd4j;

class StringTests : public testing::Test {
public:

};

TEST_F(StringTests, Basic_Test_1) {
    std::string f("alpha");
    auto array = NDArrayFactory::string(f);
    ASSERT_EQ(nd4j::DataType::UTF8, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto z = array.e<std::string>(0);

    ASSERT_EQ(f, z);
}

TEST_F(StringTests, Basic_Test_2) {
    std::string f("alpha");
    auto array = NDArrayFactory::string(f.c_str());
    ASSERT_EQ(nd4j::DataType::UTF8, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto z = array.e<std::string>(0);

    ASSERT_EQ(f, z);
}

TEST_F(StringTests, Basic_Test_3) {
    auto array = NDArrayFactory::string('c', {3, 2}, {"alpha", "beta", "gamma", "phi", "theta", "omega"});

    ASSERT_EQ(6, array.lengthOf());
    ASSERT_EQ(2, array.rankOf());

    array.printIndexedBuffer("String array");
}


TEST_F(StringTests, Export_Test_1) {
    auto array = NDArrayFactory::string('c', {3}, {"alpha", "beta", "gamma"});

    auto vector = array.asByteVector();
}

TEST_F(StringTests, Basic_dup_1) {
    std::string f("alpha");
    auto array = NDArrayFactory::string(f);
    ASSERT_EQ(nd4j::DataType::UTF8, array.dataType());

    ASSERT_EQ(1, array.lengthOf());
    ASSERT_EQ(0, array.rankOf());

    auto dup = new NDArray(array.dup());

    auto z0 = array.e<std::string>(0);
    auto z1 = dup->e<std::string>(0);

    ASSERT_EQ(f, z0);
    ASSERT_EQ(f, z1);

    delete dup;
}