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
#include <ops/declarable/CustomOperations.h>
#include <ConstantShapeHelper.h>
#include <ShapeDescriptor.h>
#include <DataBuffer.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class ConstantShapeHelperTests : public testing::Test {
public:

};

TEST_F(ConstantShapeHelperTests, basic_test_1) {
    auto ptr = ShapeBuilders::createShapeInfo(nd4j::DataType::BFLOAT16, 'f', {5, 10, 15});
    ShapeDescriptor descriptor(ptr);

    ASSERT_EQ(1, descriptor.ews());
    ASSERT_EQ(3, descriptor.rank());
    ASSERT_EQ('f', descriptor.order());
    ASSERT_EQ(nd4j::DataType::BFLOAT16, descriptor.dataType());
    ASSERT_FALSE(descriptor.isEmpty());

    ASSERT_FALSE(ConstantShapeHelper::getInstance()->checkBufferExistanceForShapeInfo(descriptor));

    auto buffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(descriptor);

    ASSERT_TRUE(ConstantShapeHelper::getInstance()->checkBufferExistanceForShapeInfo(descriptor));
}