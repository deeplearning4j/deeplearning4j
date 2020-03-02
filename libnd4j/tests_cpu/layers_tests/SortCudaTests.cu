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
//  @author raver119@gmail.com
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <array/NDArray.h>
#include <legacy/NativeOps.h>
#include <helpers/BitwiseUtils.h>

using namespace sd;
using namespace sd::graph;

class SortCudaTests : public testing::Test {
public:

};


TEST_F(SortCudaTests, test_linear_sort_by_key_1) {
    auto k = NDArrayFactory::create<Nd4jLong>('c', {10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});

    Nd4jPointer extras[2] = {nullptr, LaunchContext::defaultContext()->getCudaStream()};

    sortByKey(extras, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), false);
    k.tickWriteDevice();
    v.tickWriteDevice();

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}

TEST_F(SortCudaTests, test_linear_sort_by_val_1) {
    auto k = NDArrayFactory::create<Nd4jLong>('c', {10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});

    Nd4jPointer extras[2] = {nullptr, LaunchContext::defaultContext()->getCudaStream()};

    sortByValue(extras, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), false);
    k.tickWriteDevice();
    v.tickWriteDevice();

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}

TEST_F(SortCudaTests, test_linear_sort_by_val_2) {
    auto k = NDArrayFactory::create<int>('c', {6}, {0, 1, 2, 3, 4, 5});
//    auto v = NDArrayFactory::create<double>('c', {6}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});
    NDArray v = NDArrayFactory::create<double>('c', {6}, {0.9f, .75f, .6f, .95f, .5f, .3f});
    auto ek = NDArrayFactory::create<int>('c', {6}, {3, 0, 1, 2, 4, 5});
    auto ev = NDArrayFactory::create<double>('c', {6}, {0.95, 0.9, 0.75, 0.6, 0.5, 0.3});

    Nd4jPointer extras[2] = {nullptr, LaunchContext::defaultContext()->getCudaStream()};

    sortByValue(extras, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), true);
    k.tickWriteDevice();
    v.tickWriteDevice();
    // k.printIndexedBuffer("KEYS");
    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}

TEST_F(SortCudaTests, test_tad_sort_by_key_1) {
    auto k = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8,   1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {2, 10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5,   1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,     0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {2, 10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,     0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});

    Nd4jPointer extras[2] = {nullptr, LaunchContext::defaultContext()->getCudaStream()};

    int axis = 1;
    sortTadByKey(extras, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), &axis, 1, false);
    k.tickWriteDevice();
    v.tickWriteDevice();

    // k.printIndexedBuffer("k");
    // v.printIndexedBuffer("v");

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}

TEST_F(SortCudaTests, test_tad_sort_by_val_1) {
    auto k = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8,   1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {2, 10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5,   1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,     0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {2, 10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,     0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});

    Nd4jPointer extras[2] = {nullptr, LaunchContext::defaultContext()->getCudaStream()};

    int axis = 1;
    sortTadByValue(extras, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), &axis, 1, false);
    k.tickWriteDevice();
    v.tickWriteDevice();

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}
