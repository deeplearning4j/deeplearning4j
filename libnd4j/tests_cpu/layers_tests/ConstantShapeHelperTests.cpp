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
#include <helpers/PointersManager.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class ConstantShapeHelperTests : public testing::Test {
public:

};

TEST_F(ConstantShapeHelperTests, basic_test_1) {
    auto ptr = ShapeBuilders::createShapeInfo(nd4j::DataType::BFLOAT16, 'f', {5, 10, 15});
    ShapeDescriptor descriptor(ptr);
    ShapeDescriptor descriptor2(ptr);

    ASSERT_EQ(descriptor, descriptor2);

    ASSERT_EQ(1, descriptor.ews());
    ASSERT_EQ(3, descriptor.rank());
    ASSERT_EQ('f', descriptor.order());
    ASSERT_EQ(nd4j::DataType::BFLOAT16, descriptor.dataType());
    ASSERT_FALSE(descriptor.isEmpty());

    ASSERT_FALSE(ConstantShapeHelper::getInstance()->checkBufferExistanceForShapeInfo(descriptor));

    auto buffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(descriptor);

    ASSERT_TRUE(ConstantShapeHelper::getInstance()->checkBufferExistanceForShapeInfo(descriptor));

    auto buffer2 = ConstantShapeHelper::getInstance()->bufferForShapeInfo(descriptor2);


    ASSERT_TRUE(buffer.primary() != nullptr);
    ASSERT_TRUE(buffer.primary() == buffer2.primary());
    ASSERT_TRUE(buffer.special() == buffer2.special());

    shape::printShapeInfoLinear("0", reinterpret_cast<Nd4jLong *>(buffer.primary()));
    shape::printShapeInfoLinear("1", reinterpret_cast<Nd4jLong *>(buffer2.primary()));
}

TEST_F(ConstantShapeHelperTests, basic_test_2) {
    auto array = NDArrayFactory::create<float>('c', {128});

    auto p = array.isShapeOwner();
    ASSERT_FALSE(p.first);
    ASSERT_FALSE(p.second);
}

TEST_F(ConstantShapeHelperTests, basic_test_3) {
    auto array = NDArrayFactory::create_<float>('c', {128});

    auto p = array->isShapeOwner();
    ASSERT_FALSE(p.first);
    ASSERT_FALSE(p.second);

    ASSERT_TRUE(array->shapeInfo() != nullptr);

#ifdef __CUDABLAS__
    ASSERT_TRUE(array->specialShapeInfo() != nullptr);
#endif

    delete array;
}


TEST_F(ConstantShapeHelperTests, basic_test_4) {
    auto array = NDArrayFactory::create_<float>('c', {128, 256});

    auto dup = array->dup('f');

    auto p = dup->isShapeOwner();
    ASSERT_FALSE(p.first);
    ASSERT_FALSE(p.second);

    ASSERT_TRUE(dup->shapeInfo() != nullptr);

#ifdef __CUDABLAS__
    ASSERT_TRUE(dup->specialShapeInfo() != nullptr);
    PointersManager manager(graph::LaunchContext::defaultContext(), "test");
    manager.printDevContentOnDev<Nd4jLong>(dup->specialShapeInfo(), shape::shapeInfoLength(2), 0);
#endif

    delete array;
    delete dup;
}


TEST_F(ConstantShapeHelperTests, basic_test_5) {
    
    auto arrayA = NDArrayFactory::create<int>(1);
    auto arrayB = NDArrayFactory::create_<float>('c', {128, 256});

    arrayA.printShapeInfo("A");
    arrayB->printShapeInfo("B");
    ASSERT_EQ(0, arrayA.rankOf());
    ASSERT_EQ(2, arrayB->rankOf());
    ASSERT_NE(arrayA.dataType(), arrayB->dataType());

    delete arrayB;
}