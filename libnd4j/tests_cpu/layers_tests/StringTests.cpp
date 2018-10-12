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
using namespace nd4j;

class StringTests : public testing::Test {
public:

};

TEST_F(StringTests, Basic_Test_1) {
    std::string f("alpha");
    auto array = NDArrayFactory::string(f);
    ASSERT_EQ(nd4j::DataType::UTF8, array.dataType());

    ASSERT_EQ(5, array.lengthOf());
    ASSERT_EQ(1, array.rankOf());

    auto ptr = reinterpret_cast<char *>(array.buffer());
    std::string z(ptr);

    ASSERT_EQ(f, z);
}
