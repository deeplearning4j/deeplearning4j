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
#include <NDArray.h>
#include <NativeOps.h>
#include <helpers/BitwiseUtils.h>

using namespace nd4j;
using namespace nd4j::graph;

class SortCpuTests : public testing::Test {
public:

};


TEST_F(SortCpuTests, test_linear_sort_by_key_1) {
    if (!Environment::getInstance()->isCPU())
        return;

    auto k = NDArrayFactory::create<Nd4jLong>('c', {10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});


    NativeOps nativeOps;
    nativeOps.sortByKey(nullptr, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), false);

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}

TEST_F(SortCpuTests, test_linear_sort_by_val_1) {
    if (!Environment::getInstance()->isCPU())
        return;

    auto k = NDArrayFactory::create<Nd4jLong>('c', {10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});


    NativeOps nativeOps;
    nativeOps.sortByValue(nullptr, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), false);

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}

TEST_F(SortCpuTests, test_tad_sort_by_key_1) {
    if (!Environment::getInstance()->isCPU())
        return;

    auto k = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8,   1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {2, 10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5,   1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,     0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {2, 10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,     0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});


    int axis = 1;
    NativeOps nativeOps;
    nativeOps.sortTadByKey(nullptr, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), &axis, 1, false);

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}

TEST_F(SortCpuTests, test_tad_sort_by_val_1) {
    if (!Environment::getInstance()->isCPU())
        return;

    auto k = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8,   1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {2, 10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5,   1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,     0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {2, 10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,     0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});


    int axis = 1;
    NativeOps nativeOps;
    nativeOps.sortTadByValue(nullptr, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), &axis, 1, false);

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}