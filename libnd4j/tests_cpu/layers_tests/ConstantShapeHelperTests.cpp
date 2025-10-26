/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/ConstantDataBuffer.h>
#include <array/ShapeDescriptor.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/CustomOperations.h>

#include "testlayers.h"

using namespace sd;
using namespace sd::ops;
using namespace sd::graph;

class ConstantShapeHelperTests : public NDArrayTests {
 public:
};

class ConstantHelperTests : public NDArrayTests {
 public:
};

class ConstantTadHelperTests : public NDArrayTests {
 public:
};

TEST_F(ConstantShapeHelperTests, test_cachedAmount_1) {
  auto ttlBefore = ConstantShapeHelper::getInstance().totalCachedEntries();

  auto arrayA = NDArrayFactory::create<bool>('c', {7, 11, 17, 23, 31, 43});

  auto ttlMiddle = ConstantShapeHelper::getInstance().totalCachedEntries();

  auto arrayB = NDArrayFactory::create<bool>('c', {7, 11, 17, 23, 31, 43});

  auto ttlAfter = ConstantShapeHelper::getInstance().totalCachedEntries();

  ASSERT_TRUE(ttlBefore <= ttlMiddle);
  ASSERT_EQ(ttlMiddle, ttlAfter);
}

TEST_F(ConstantTadHelperTests, test_cachedAmount_1) {
  auto arrayA = NDArrayFactory::create<bool>('c', {7, 11, 17, 23, 31, 43});
  auto ttlBefore = ConstantTadHelper::getInstance().totalCachedEntries();

  std::vector<LongType> dimensions = {3, 4};
  auto packAA = ConstantTadHelper::getInstance().tadForDimensions(arrayA.shapeInfo(), &dimensions);

  auto ttlMiddle = ConstantTadHelper::getInstance().totalCachedEntries();

  auto packAB = ConstantTadHelper::getInstance().tadForDimensions(arrayA.shapeInfo(), &dimensions);

  auto ttlAfter = ConstantTadHelper::getInstance().totalCachedEntries();

  ASSERT_TRUE(ttlBefore <= ttlMiddle);
}

TEST_F(ConstantShapeHelperTests, basic_test_1) {
  auto ptr = ShapeBuilders::createShapeInfo(sd::DataType::BFLOAT16, 'f', {5, 10, 15});
  ShapeDescriptor descriptor(ptr);
  ShapeDescriptor descriptor2(ptr);

  ASSERT_EQ(descriptor, descriptor2);

  ASSERT_EQ(1, descriptor.ews());
  ASSERT_EQ(3, descriptor.rank());
  ASSERT_EQ('f', descriptor.order());
  ASSERT_EQ(sd::DataType::BFLOAT16, descriptor.dataType());
  ASSERT_FALSE(descriptor.isEmpty());

  ASSERT_FALSE(ConstantShapeHelper::getInstance().checkBufferExistenceForShapeInfo(&descriptor));

  auto buffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(&descriptor);

  ASSERT_TRUE(ConstantShapeHelper::getInstance().checkBufferExistenceForShapeInfo(&descriptor));

  auto buffer2 = ConstantShapeHelper::getInstance().bufferForShapeInfo(&descriptor2);

  ASSERT_TRUE(buffer->primary() != nullptr);
  ASSERT_TRUE(buffer->primary() == buffer2->primary());
  ASSERT_TRUE(buffer->special() == buffer2->special());

  delete[] ptr;
}

TEST_F(ConstantShapeHelperTests, stress_test_1) {
  for (auto x = 0; x < 1000; x++) {
    auto ptr = ShapeBuilders::createShapeInfo(FLOAT32, 'c', {5, x + 10, x + 1});
    ShapeDescriptor descriptor(ptr);
    ConstantShapeHelper::getInstance().createShapeInfo(&descriptor);
    delete[] ptr;
  }
  ShapeDescriptor aShape(FLOAT32, 'c', {(LongType)5, (LongType)382, (LongType)373});

  auto timeStart = std::chrono::system_clock::now();
  ASSERT_TRUE(ConstantShapeHelper::getInstance().checkBufferExistenceForShapeInfo(&aShape));
  auto timeEnd = std::chrono::system_clock::now();

  auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
  sd_printf("Total time (us) %lld\n", outerTime);
}

TEST_F(ConstantShapeHelperTests, basic_test_3) {
  auto array = NDArrayFactory::create_<float>('c', {128});

  ASSERT_TRUE(array->shapeInfo() != nullptr);

#ifdef SD_CUDA
  ASSERT_TRUE(array->specialShapeInfo() != nullptr);
#endif

  delete array;
}

TEST_F(ConstantShapeHelperTests, basic_test_4) {
  auto array = NDArrayFactory::create_<float>('c', {128, 256});

  auto dup = new NDArray(array->dup('f'));

  ASSERT_TRUE(dup->shapeInfo() != nullptr);

#ifdef SD_CUDA
  ASSERT_TRUE(dup->specialShapeInfo() != nullptr);
  PointersManager manager(LaunchContext ::defaultContext(), "test");
#endif

  delete array;
  delete dup;
}

TEST_F(ConstantShapeHelperTests, basic_test_5) {
  auto arrayA = NDArrayFactory::create<int>(1);
  auto arrayB = NDArrayFactory::create_<float>('c', {128, 256});
  ASSERT_EQ(0, arrayA.rankOf());
  ASSERT_EQ(2, arrayB->rankOf());
  ASSERT_NE(arrayA.dataType(), arrayB->dataType());

  delete arrayB;
}

TEST_F(ConstantShapeHelperTests, basic_test_6) {
  ShapeDescriptor descriptorA(INT32, 'c', {});
  ShapeDescriptor descriptorB(FLOAT32, 'c', {10, 10});

  ASSERT_TRUE(descriptorA < descriptorB);
  ASSERT_FALSE(descriptorB < descriptorA);
}

TEST_F(ConstantShapeHelperTests, basic_test_7) {
  auto array = NDArrayFactory::create_<float>('c', {32, 256});

  IndicesList indices({NDIndex::all(), NDIndex::interval(0, 1)});
  auto strided = array->subarray(indices);
  strided.assign(1.0f);


  delete array;
}

TEST_F(ConstantHelperTests, basic_test_1) {
  ConstantDescriptor descriptor({1, 2, 3});

  ConstantDataBuffer* fBuffer = ConstantHelper::getInstance().constantBuffer(descriptor, FLOAT32);
  auto fPtr = fBuffer->primaryAsT<float>();

  ASSERT_NEAR(1.f, fPtr[0], 1e-5);
  ASSERT_NEAR(2.f, fPtr[1], 1e-5);
  ASSERT_NEAR(3.f, fPtr[2], 1e-5);

  auto iBuffer = ConstantHelper::getInstance().constantBuffer(descriptor, INT32);
  auto iPtr = iBuffer->primaryAsT<int>();

  ASSERT_EQ(1, iPtr[0]);
  ASSERT_EQ(2, iPtr[1]);
  ASSERT_EQ(3, iPtr[2]);
}

TEST_F(ConstantHelperTests, basic_test_2) {
  double array[] = {1., 2., 3.};
  ConstantDescriptor descriptor(array, 3);

  ConstantDataBuffer* fBuffer = ConstantHelper::getInstance().constantBuffer(descriptor, FLOAT32);
  auto fPtr = fBuffer->primaryAsT<float>();

  ASSERT_NEAR(1.f, fPtr[0], 1e-5);
  ASSERT_NEAR(2.f, fPtr[1], 1e-5);
  ASSERT_NEAR(3.f, fPtr[2], 1e-5);

  auto iBuffer = ConstantHelper::getInstance().constantBuffer(descriptor, INT32);
  auto iPtr = iBuffer->primaryAsT<int>();

  ASSERT_EQ(1, iPtr[0]);
  ASSERT_EQ(2, iPtr[1]);
  ASSERT_EQ(3, iPtr[2]);
}

//////////////////////////////////////////////////////////////////////
TEST_F(ConstantShapeHelperTests, ShapeDescriptor_1) {
  LongType shapeInfo1[] = {4, 2, 5, 5, 2, 25, 5, 1, 50, 8192, 0, 99};
  LongType shapeInfo2[] = {4, 2, 5, 5, 2, 50, 10, 2, 1, 8192, 1, 99};

  ShapeDescriptor descr1(shapeInfo1);
  ShapeDescriptor descr2(shapeInfo2);

  ASSERT_FALSE(descr1 == descr2);
}

TEST_F(ConstantShapeHelperTests, ShapeDescriptor_validation) {
  // for c order
  std::vector<LongType> shape{2, 3, 4, 5};
  std::vector<LongType> incorrectStride1{20, 20, 5, 1};
  std::vector<LongType> incorrectStride2{60, 20, 5, 5};
  std::vector<LongType> correctStride1{60, 20, 5, 1};
  std::vector<LongType> correctStride2{300, 100, 25, 5};
  std::vector<LongType> correctStride3{800, 200, 40, 5};

  auto shapeDesc = ShapeDescriptor(FLOAT32, 'c', shape, incorrectStride1, 1);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_INCORRECT_STRIDES);
  shapeDesc = ShapeDescriptor(FLOAT32, 'c', shape, correctStride1, 1);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_OK);
  shapeDesc = ShapeDescriptor(FLOAT32, 'c', shape, incorrectStride2, 1);
  ASSERT_TRUE(shapeDesc.validate() == (SHAPE_DESC_INCORRECT_STRIDES | SHAPE_DESC_INCORRECT_EWS));
  shapeDesc = ShapeDescriptor(FLOAT32, 'c', shape, correctStride2, 1);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_INCORRECT_EWS);
  shapeDesc = ShapeDescriptor(FLOAT32, 'c', shape, correctStride2, 5);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_OK);
  shapeDesc = ShapeDescriptor(FLOAT32, 'c', shape, correctStride3, 1);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_INCORRECT_EWS);
  shapeDesc = ShapeDescriptor(FLOAT32, 'c', shape, correctStride3, 0);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_OK);

  // order f
  std::reverse(std::begin(shape), std::end(shape));
  std::reverse(std::begin(incorrectStride1), std::end(incorrectStride1));
  std::reverse(std::begin(incorrectStride2), std::end(incorrectStride2));
  std::reverse(std::begin(correctStride1), std::end(correctStride1));
  std::reverse(std::begin(correctStride2), std::end(correctStride2));
  std::reverse(std::begin(correctStride3), std::end(correctStride3));

  shapeDesc = ShapeDescriptor(FLOAT32, 'f', shape, incorrectStride1, 1);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_INCORRECT_STRIDES);
  shapeDesc = ShapeDescriptor(FLOAT32, 'f', shape, correctStride1, 1);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_OK);
  shapeDesc = ShapeDescriptor(FLOAT32, 'f', shape, incorrectStride2, 1);
  ASSERT_TRUE(shapeDesc.validate() == (SHAPE_DESC_INCORRECT_STRIDES | SHAPE_DESC_INCORRECT_EWS));
  shapeDesc = ShapeDescriptor(FLOAT32, 'f', shape, correctStride2, 1);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_INCORRECT_EWS);
  shapeDesc = ShapeDescriptor(FLOAT32, 'f', shape, correctStride2, 5);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_OK);
  shapeDesc = ShapeDescriptor(FLOAT32, 'f', shape, correctStride3, 1);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_INCORRECT_EWS);
  shapeDesc = ShapeDescriptor(FLOAT32, 'f', shape, correctStride3, 0);
  ASSERT_TRUE(shapeDesc.validate() == SHAPE_DESC_OK);

  std::vector<LongType> shape1;
  shape1.resize(SD_MAX_RANK + 1);
  shapeDesc = ShapeDescriptor(FLOAT32, 'f', shape1, correctStride3, 0);
  ASSERT_TRUE((shapeDesc.validate() & SHAPE_DESC_INCORRECT_RANK) == SHAPE_DESC_INCORRECT_RANK);
}

TEST_F(ConstantShapeHelperTests, ShapeDescriptor_paddedBuffer) {
  constexpr int n = 2;
  constexpr int c = 3;
  constexpr int h = 4;
  constexpr int w = 5;
  constexpr int n_pad = 2;
  constexpr int c_pad = 3;
  constexpr int h_pad = 4;
  constexpr int w_pad = 5;
  char orders[] = {'c', 'f'};

  for (auto& order : orders) {
    auto shapeDesc1 =
        ShapeDescriptor::paddedBufferDescriptor(FLOAT32, order, {n, c, h, w}, {n_pad, c_pad, h_pad, w_pad});
    auto shapeDesc2 = ShapeDescriptor(FLOAT32, order, {n + n_pad, c + c_pad, h + h_pad, w + w_pad});
    auto shapeDesc3 = ShapeDescriptor::paddedBufferDescriptor(FLOAT32, order, {n, c, h, w}, {n_pad, c_pad});
    auto shapeDesc4 = ShapeDescriptor(FLOAT32, order, {n + n_pad, c + c_pad, h, w});
    auto shapeDesc5 =
        ShapeDescriptor::paddedBufferDescriptor(FLOAT32, order, {n, c, h, w}, {0, 0, h_pad, w_pad});
    auto shapeDesc6 = ShapeDescriptor(FLOAT32, order, {n, c, h + h_pad, w + w_pad});

    ASSERT_TRUE(shapeDesc1->validate() == SHAPE_DESC_OK);
    ASSERT_TRUE(shapeDesc2.validate() == SHAPE_DESC_OK);
    ASSERT_TRUE(shapeDesc3->validate() == SHAPE_DESC_OK);
    ASSERT_TRUE(shapeDesc4.validate() == SHAPE_DESC_OK);
    ASSERT_TRUE(shapeDesc5->validate() == SHAPE_DESC_OK);
    ASSERT_TRUE(shapeDesc6.validate() == SHAPE_DESC_OK);

    ASSERT_EQ(shapeDesc1->allocLength(), shapeDesc2.allocLength());
    ASSERT_EQ(shapeDesc3->allocLength(), shapeDesc4.allocLength());
    ASSERT_EQ(shapeDesc5->allocLength(), shapeDesc6.allocLength());

    const auto& v1 = shapeDesc1->stridesPtr();
    const auto& v2 = shapeDesc2.stridesPtr();
    const auto& v3 = shapeDesc3->stridesPtr();
    const auto& v4 = shapeDesc4.stridesPtr();
    const auto& v5 = shapeDesc5->stridesPtr();
    const auto& v6 = shapeDesc6.stridesPtr();

    for (int i = 0; i < shapeDesc1->rank(); i++) {

      ASSERT_EQ(v1[i], v2[i]);

    }
    for (int i = 0; i < shapeDesc3->rank(); i++) {
      ASSERT_EQ(v3[i], v4[i]);
    }
    for (int i = 0; i < shapeDesc5->rank(); i++) {
      ASSERT_EQ(v5[i], v6[i]);
    }
  }
}
