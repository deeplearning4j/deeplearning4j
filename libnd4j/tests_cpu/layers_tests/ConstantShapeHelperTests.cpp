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
#include <array/ConstantDataBuffer.h>
#include <helpers/PointersManager.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class ConstantShapeHelperTests : public testing::Test {
public:

};

class ConstantHelperTests : public testing::Test {
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

    ASSERT_FALSE(ConstantShapeHelper::getInstance()->checkBufferExistenceForShapeInfo(descriptor));

    auto buffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(descriptor);

    ASSERT_TRUE(ConstantShapeHelper::getInstance()->checkBufferExistenceForShapeInfo(descriptor));

    auto buffer2 = ConstantShapeHelper::getInstance()->bufferForShapeInfo(descriptor2);


    ASSERT_TRUE(buffer.primary() != nullptr);
    ASSERT_TRUE(buffer.primary() == buffer2.primary());
    ASSERT_TRUE(buffer.special() == buffer2.special());

    delete []ptr;
}

TEST_F(ConstantShapeHelperTests, basic_test_3) {
    auto array = NDArrayFactory::create_<float>('c', {128});

    ASSERT_TRUE(array->shapeInfo() != nullptr);

#ifdef __CUDABLAS__
    ASSERT_TRUE(array->specialShapeInfo() != nullptr);
#endif

    delete array;
}


TEST_F(ConstantShapeHelperTests, basic_test_4) {
    auto array = NDArrayFactory::create_<float>('c', {128, 256});

    auto dup = array->dup('f');

    ASSERT_TRUE(dup->shapeInfo() != nullptr);

#ifdef __CUDABLAS__
    ASSERT_TRUE(dup->specialShapeInfo() != nullptr);
    PointersManager manager(nd4j::LaunchContext ::defaultContext(), "test");
    // manager.printDevContentOnDev<Nd4jLong>(dup->specialShapeInfo(), shape::shapeInfoLength(2), 0);
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

TEST_F(ConstantShapeHelperTests, basic_test_6) {
    ShapeDescriptor descriptorA(nd4j::DataType::INT32, 'c', {});
    ShapeDescriptor descriptorB(nd4j::DataType::FLOAT32, 'c', {10, 10});

    // ASSERT_FALSE(descriptorA < descriptorB);
    // ASSERT_TRUE(descriptorB < descriptorA);

    ASSERT_TRUE(descriptorA < descriptorB);
    ASSERT_FALSE(descriptorB < descriptorA);
}

TEST_F(ConstantShapeHelperTests, basic_test_7) {
    auto array = NDArrayFactory::create_<float>('c', {32, 256});

    IndicesList indices({NDIndex::all(), NDIndex::interval(0,1)});
    auto strided = array->subarray(indices);
    strided->assign(1.0f);

    //strided->printIndexedBuffer("column");

    delete array;
    delete strided;
}

TEST_F(ConstantHelperTests, basic_test_1) {

    ConstantDescriptor descriptor({1, 2, 3});

    ConstantDataBuffer* fBuffer = ConstantHelper::getInstance()->constantBuffer(descriptor, nd4j::DataType::FLOAT32);
    auto fPtr = fBuffer->primaryAsT<float>();

    ASSERT_NEAR(1.f, fPtr[0], 1e-5);
    ASSERT_NEAR(2.f, fPtr[1], 1e-5);
    ASSERT_NEAR(3.f, fPtr[2], 1e-5);

    auto iBuffer = ConstantHelper::getInstance()->constantBuffer(descriptor, nd4j::DataType::INT32);
    auto iPtr = iBuffer->primaryAsT<int>();

    ASSERT_EQ(1, iPtr[0]);
    ASSERT_EQ(2, iPtr[1]);
    ASSERT_EQ(3, iPtr[2]);
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConstantShapeHelperTests, ShapeDescriptor_1) {

    Nd4jLong shapeInfo1[] = {4, 2, 5, 5, 2, 25, 5, 1, 50, 8192, 0, 99};
    Nd4jLong shapeInfo2[] = {4, 2, 5, 5, 2, 50, 10, 2, 1, 8192, 1, 99};

    ShapeDescriptor descr1(shapeInfo1);
    ShapeDescriptor descr2(shapeInfo2);

    ASSERT_FALSE(descr1 == descr2);
}