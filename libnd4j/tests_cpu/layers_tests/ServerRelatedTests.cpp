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
#include <GraphExecutioner.h>

using namespace nd4j;
using namespace nd4j::graph;

class ServerRelatedTests : public testing::Test {
public:

};

TEST_F(ServerRelatedTests, Basic_Output_Test_1) {
    flatbuffers::FlatBufferBuilder builder(4096);

    auto array1 = new NDArray<float>('c', {10, 10});
    auto array2 = new NDArray<float>('c', {10, 10});
    auto array3 = new NDArray<float>('c', {10, 10});

    array1->assign(1.0f);
    array2->assign(2.0f);
    array3->assign(3.0f);

    Variable<float> var1(array1, "first", 1);
    Variable<float> var2(array2, "second", 2);
    Variable<float> var3(array3, "second indexed", 2, 1);

    ExecutionResult<float> result({&var1, &var2, &var3});

    ASSERT_EQ(*array1, *result.at(0)->getNDArray());
    ASSERT_EQ(*array2, *result.at(1)->getNDArray());
    ASSERT_EQ(*array3, *result.at(2)->getNDArray());

    ASSERT_EQ(*array1, *result.byId("first")->getNDArray());
    ASSERT_EQ(*array2, *result.byId("second")->getNDArray());
    ASSERT_EQ(*array3, *result.byId("second indexed")->getNDArray());

    auto flatResult = result.asFlatResult(builder);
    builder.Finish(flatResult);
    auto ptr = builder.GetBufferPointer();
    auto received = GetFlatResult(ptr);

    ExecutionResult<float> restored(received);
    ASSERT_EQ(3, restored.size());

    ASSERT_EQ(*array1, *restored.at(0)->getNDArray());
    ASSERT_EQ(*array2, *restored.at(1)->getNDArray());
    ASSERT_EQ(*array3, *restored.at(2)->getNDArray());

    ASSERT_EQ(*array1, *restored.byId("first")->getNDArray());
    ASSERT_EQ(*array2, *restored.byId("second")->getNDArray());
    ASSERT_EQ(*array3, *restored.byId("second indexed")->getNDArray());
}