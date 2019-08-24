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

#include <NDArray.h>
#include <NDArrayFactory.h>
#include "testlayers.h"
#include <graph/Stash.h>
#include <FlatUtils.h>

using namespace nd4j;

class FlatUtilsTests : public testing::Test {
public:

};

TEST_F(FlatUtilsTests, flat_float_serde_1) {
    auto array = NDArrayFactory::create<float>('c', {4}, {1.f, 2.f, 3.f, 4.f});

    flatbuffers::FlatBufferBuilder builder(1024);
    auto flatArray = FlatUtils::toFlatArray(builder, array);
    builder.Finish(flatArray);


    auto pfArray = GetFlatArray(builder.GetBufferPointer());

    auto restored = FlatUtils::fromFlatArray(pfArray);

    ASSERT_EQ(array, *restored);

    delete restored;
}

TEST_F(FlatUtilsTests, flat_int_serde_1) {
    auto array = NDArrayFactory::create<int>('c', {4}, {1, 2, 3, 4});

    flatbuffers::FlatBufferBuilder builder(1024);
    auto flatArray = FlatUtils::toFlatArray(builder, array);
    builder.Finish(flatArray);


    auto pfArray = GetFlatArray(builder.GetBufferPointer());

    auto restored = FlatUtils::fromFlatArray(pfArray);

    ASSERT_EQ(array, *restored);

    delete restored;
}

TEST_F(FlatUtilsTests, flat_bool_serde_1) {
    auto array = NDArrayFactory::create<bool>('c', {4}, {true, false, true, false});

    flatbuffers::FlatBufferBuilder builder(1024);
    auto flatArray = FlatUtils::toFlatArray(builder, array);
    builder.Finish(flatArray);


    auto pfArray = GetFlatArray(builder.GetBufferPointer());

    auto restored = FlatUtils::fromFlatArray(pfArray);

    ASSERT_EQ(array, *restored);

    delete restored;
}

TEST_F(FlatUtilsTests, flat_string_serde_1) {
    auto array = NDArrayFactory::string('c', {3}, {"alpha", "beta", "gamma"});

    flatbuffers::FlatBufferBuilder builder(1024);
    auto flatArray = FlatUtils::toFlatArray(builder, array);
    builder.Finish(flatArray);


    auto pfArray = GetFlatArray(builder.GetBufferPointer());

    auto restored = FlatUtils::fromFlatArray(pfArray);

    ASSERT_EQ(array, *restored);

    delete restored;
}