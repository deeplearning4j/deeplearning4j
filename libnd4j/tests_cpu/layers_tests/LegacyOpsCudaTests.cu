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
#include <array/NDArray.h>
#include <helpers/ShapeUtils.h>
#include <loops/reduce3.h>
#include <ops/declarable/LegacyTransformOp.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyReduceSameOp.h>
#include <ops/declarable/LegacyReduceFloatOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyBroadcastOp.h>
#include <helpers/TAD.h>
#include <helpers/ConstantTadHelper.h>

using namespace sd;
using namespace sd::ops;

class LegacyOpsCudaTests : public testing::Test {

};


TEST_F(LegacyOpsCudaTests, test_sortTad_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 5}, {1.f, 3.f, 0.f, 2.f, 4.f,
                                                         6.f, 5.f, 9.f, 7.f, 8.f,
                                                         10.f, 11.f, 14.f, 12.f, 13.f});

    auto e = NDArrayFactory::create<float>('c', {3, 5}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f});

    int axis = 1;
    auto packX = ConstantTadHelper::getInstance()->tadForDimensions(x.shapeInfo(), axis);

    Nd4jPointer extras[2] = {nullptr, LaunchContext::defaultContext()->getCudaStream()};

    x.syncToDevice();
    sortTad(extras, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), &axis, 1, packX.platformShapeInfo(), packX.platformOffsets(), false);
    x.tickWriteDevice();

    ASSERT_EQ(e, x);
}

TEST_F(LegacyOpsCudaTests, test_sort_1) {
  auto x = NDArrayFactory::create<float>('c', {4}, {4.f, 2.f, 1.f, 3.f});
  auto e = NDArrayFactory::create<float>('c', {4}, {1.f, 2.f, 3.f, 4.f});

  Nd4jPointer extras[2] = {nullptr, LaunchContext::defaultContext()->getCudaStream()};

  NDArray::prepareSpecialUse({&x}, {&x});
  ::sort(extras, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), false);
  NDArray::registerSpecialUse({&x});

  ASSERT_EQ(e, x);
}

TEST_F(LegacyOpsCudaTests, test_sort_2) {
  auto x = NDArrayFactory::create<float>('c', {4}, {4.f, 2.f, 1.f, 3.f});
  auto e = NDArrayFactory::create<float>('c', {4}, {4.f, 3.f, 2.f, 1.f});

  Nd4jPointer extras[2] = {nullptr, LaunchContext::defaultContext()->getCudaStream()};

  NDArray::prepareSpecialUse({&x}, {&x});
  ::sort(extras, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), true);
  NDArray::registerSpecialUse({&x});

  ASSERT_EQ(e, x);
}

TEST_F(LegacyOpsCudaTests, test_sort_3) {
  auto x = NDArrayFactory::create<double>('c', {4}, {0.5, 0.4, 0.1, 0.2});
  auto e = NDArrayFactory::create<double>('c', {4}, {0.1, 0.2, 0.4, 0.5});

  Nd4jPointer extras[2] = {nullptr, LaunchContext::defaultContext()->getCudaStream()};

  NDArray::prepareSpecialUse({&x}, {&x});
  ::sort(extras, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), false);
  NDArray::registerSpecialUse({&x});

  ASSERT_EQ(e, x);
}

TEST_F(LegacyOpsCudaTests, test_sort_4) {
  auto x = NDArrayFactory::create<double>('c', {4}, {7, 4, 9, 2});
  auto e = NDArrayFactory::create<double>('c', {4}, {2, 4, 7, 9});

  Nd4jPointer extras[2] = {nullptr, LaunchContext::defaultContext()->getCudaStream()};

  NDArray::prepareSpecialUse({&x}, {&x});
  ::sort(extras, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), false);
  NDArray::registerSpecialUse({&x});

  ASSERT_EQ(e, x);
}