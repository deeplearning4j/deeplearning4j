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
// Created by raver on 6/18/2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
// #include <array/NDArrayList.h>

using namespace nd4j;


class EmptyTests : public testing::Test {
public:

    EmptyTests() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(EmptyTests, Test_Create_Empty) {
    auto empty = NDArrayFactory::empty<float>();
    ASSERT_TRUE(empty->isEmpty());

    ASSERT_EQ(0, empty->lengthOf());
    ASSERT_TRUE(empty->buffer() == nullptr);

    ASSERT_TRUE(shape::isEmpty(empty->shapeInfo()));

    delete empty;
}

TEST_F(EmptyTests, Test_Concat_1) {
    auto empty = NDArrayFactory::empty<float>();
    auto vector = NDArrayFactory::create_<float>('c', {1}, {1.0f});

    ASSERT_TRUE(empty->isEmpty());

    nd4j::ops::concat op;
    auto result = op.execute({empty, vector}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

//    z->printShapeInfo("z shape");
//    z->printIndexedBuffer("z buffr");

    ASSERT_EQ(*vector, *z);

    delete empty;
    delete vector;
    delete result;
}


TEST_F(EmptyTests, Test_Concat_2) {
    auto empty = NDArrayFactory::empty<float>();
    auto scalar1 =  NDArrayFactory::create_<float>(1.0f);
    auto scalar2  = NDArrayFactory::create_<float>(2.0f);
    auto exp = NDArrayFactory::create<float>('c', {2}, {1.f, 2.f});

    ASSERT_TRUE(empty->isEmpty());

    nd4j::ops::concat op;
    auto result = op.execute({empty, scalar1, scalar2}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

//    z->printShapeInfo("z shape");
//    z->printIndexedBuffer("z buffr");

    ASSERT_EQ(exp, *z);

    delete empty;
    delete scalar1;
    delete scalar2;
    delete result;
}

TEST_F(EmptyTests, Test_Reshape_1) {
    auto vector = NDArrayFactory::create<float>('c', {1}, {119.0f});
    auto exp = NDArrayFactory::create<float>(119.0f);
    auto empty = NDArrayFactory::empty<float>();

    nd4j::ops::reshape op;
    auto result = op.execute({&vector, empty}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(exp, *result->at(0));

    delete empty;
    delete result;
}

TEST_F(EmptyTests, Test_Reshape_2) {
    auto vector = NDArrayFactory::create<float>('c', {1}, {119.0f});
    auto exp = NDArrayFactory::create<float>(119.0f);
    auto empty = NDArrayFactory::empty<float>();

    nd4j::ops::reshape op;
    auto result = op.execute({&vector, empty}, {}, {}, true);

    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(exp, *result->at(0));
    ASSERT_EQ(exp, vector);

    delete empty;
    delete result;
}