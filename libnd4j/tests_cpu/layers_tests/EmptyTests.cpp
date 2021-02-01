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
// Created by raver on 6/18/2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <array/NDArray.h>
// #include <array/NDArrayList.h>

using namespace sd;


class EmptyTests : public testing::Test {
public:

    EmptyTests() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(EmptyTests, Test_Create_Empty_1) {
    auto empty = NDArrayFactory::empty_<float>();
    ASSERT_TRUE(empty->isEmpty());

    ASSERT_EQ(0, empty->lengthOf());
    ASSERT_TRUE(empty->buffer() == nullptr);

    ASSERT_TRUE(shape::isEmpty(empty->shapeInfo()));

    delete empty;
}

TEST_F(EmptyTests, Test_Create_Empty_2) {
    auto empty = NDArrayFactory::empty<float>();
    ASSERT_TRUE(empty.isEmpty());

    ASSERT_EQ(0, empty.lengthOf());
    ASSERT_TRUE(empty.buffer() == nullptr);

    ASSERT_TRUE(shape::isEmpty(empty.shapeInfo()));
    ASSERT_TRUE(empty.isEmpty());
}

TEST_F(EmptyTests, Test_Concat_1) {
//    auto empty = NDArrayFactory::empty_<float>();
    auto empty = new NDArray('c',  {0}, sd::DataType::FLOAT32);//NDArrayFactory::create_<float>('c', {(Nd4jLong)0}};
    auto vector = NDArrayFactory::create_<float>('c', {1}, {1.0f});

    ASSERT_TRUE(empty->isEmpty());

    sd::ops::concat op;
    auto result = op.evaluate({empty, vector}, {}, {0});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

//    z->printShapeInfo("z shape");
//    z->printIndexedBuffer("z buffr");

    ASSERT_EQ(*vector, *z);

    delete empty;
    delete vector;
}


TEST_F(EmptyTests, Test_Concat_2) {
    auto empty = new NDArray('c',  {0}, sd::DataType::FLOAT32); //NDArrayFactory::empty_<float>();
    auto scalar1 =  NDArrayFactory::create_<float>('c', {1}, {1.0f});
    auto scalar2  = NDArrayFactory::create_<float>('c', {1}, {2.0f});
    auto exp = NDArrayFactory::create<float>('c', {2}, {1.f, 2.f});

    ASSERT_TRUE(empty->isEmpty());

    sd::ops::concat op;
    auto result = op.evaluate({empty, scalar1, scalar2}, {}, {0});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

//    z->printShapeInfo("z shape");
//    z->printIndexedBuffer("z buffr");

    ASSERT_EQ(exp, *z);

    delete empty;
    delete scalar1;
    delete scalar2;
}

TEST_F(EmptyTests, Test_Concat_3) {
    auto empty = NDArrayFactory::empty<float>(); //NDArrayFactory::empty_<float>();
    auto scalar1 =  NDArrayFactory::create<float>(1.0f);
    auto scalar2  = NDArrayFactory::create<float>(2.0f);
    auto exp = NDArrayFactory::create<float>('c', {2}, {1.f, 2.f});

    ASSERT_TRUE(empty.isEmpty());

    sd::ops::concat op;
    auto result = op.evaluate({&empty, &scalar1, &scalar2}, {}, {0});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

    ASSERT_EQ(exp, *z);

}

TEST_F(EmptyTests, Test_Concat_4) {
    auto empty = NDArrayFactory::empty<float>(); //NDArrayFactory::empty_<float>();
    auto scalar1 =  NDArrayFactory::create<float>(1.0f);
    auto scalar2  = NDArrayFactory::create<float>(2.0f);
    auto exp = NDArrayFactory::create<float>('c', {2}, {1.f, 2.f});

    ASSERT_TRUE(empty.isEmpty());

    sd::ops::concat op;
    auto result = op.evaluate({&scalar1, &empty, &scalar2}, {}, {0});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);

    ASSERT_EQ(exp, *z);
}

TEST_F(EmptyTests, Test_dup_1) {
    auto empty = NDArrayFactory::empty<int>();
    auto dup = new NDArray(empty.dup());

    ASSERT_TRUE(dup->isEmpty());
    ASSERT_EQ(empty, *dup);

    delete dup;
}

TEST_F(EmptyTests, test_empty_scatter_1) {
    auto x = NDArrayFactory::create<float>('c', {5});
    auto indices = NDArrayFactory::create<int>('c', {0});
    auto updates = NDArrayFactory::create<float>('c', {0});

    x.linspace(1.0f);

    sd::ops::scatter_upd op;
    auto result = op.evaluate({&x, &indices, &updates}, {}, {}, {true});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);
    ASSERT_EQ(x, *z);

}

TEST_F(EmptyTests, test_empty_scatter_2) {
    NDArray x ('c', {5}, sd::DataType::FLOAT32);
    NDArray z ('c', {5}, sd::DataType::FLOAT32);
    auto indices = NDArrayFactory::create<int>('c', {0});
    auto updates = NDArrayFactory::create<float>('c', {0});

    x.linspace(1.0f);

    sd::ops::scatter_upd op;
    auto status = op.execute({&x, &indices, &updates}, {&z}, {}, {}, {true});

    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(x, z);
}

TEST_F(EmptyTests, test_shaped_empty_1) {
    auto empty = NDArrayFactory::create<float>('c', {2, 0, 3});
    std::vector<Nd4jLong> shape = {2, 0, 3};

    ASSERT_EQ(sd::DataType::FLOAT32, empty.dataType());
    ASSERT_EQ(0, empty.lengthOf());
    ASSERT_TRUE(empty.isEmpty());
    ASSERT_EQ(shape, empty.getShapeAsVector());
    ASSERT_EQ(3, empty.rankOf());
}

TEST_F(EmptyTests, test_shaped_empty_2) {
    auto empty = NDArrayFactory::create<float>('c', {0, 3});
    std::vector<Nd4jLong> shape = {0, 3};

    ASSERT_EQ(sd::DataType::FLOAT32, empty.dataType());
    ASSERT_EQ(0, empty.lengthOf());
    ASSERT_TRUE(empty.isEmpty());
    ASSERT_EQ(shape, empty.getShapeAsVector());
    ASSERT_EQ(2, empty.rankOf());
}

TEST_F(EmptyTests, test_shaped_empty_3) {
    auto empty = NDArrayFactory::create<float>('c', {0});
    std::vector<Nd4jLong> shape = {0};

    ASSERT_EQ(sd::DataType::FLOAT32, empty.dataType());
    ASSERT_EQ(0, empty.lengthOf());
    ASSERT_TRUE(empty.isEmpty());
    ASSERT_EQ(shape, empty.getShapeAsVector());
    ASSERT_EQ(1, empty.rankOf());
}

TEST_F(EmptyTests, test_shaped_empty_4) {
    const auto shape = ConstantShapeHelper::getInstance().vectorShapeInfo(0, sd::DataType::FLOAT32);
    NDArray array(shape, true, sd::LaunchContext::defaultContext());
    std::vector<Nd4jLong> shapeOf({0});

    ASSERT_TRUE(array.isEmpty());
    ASSERT_EQ(1, array.rankOf());
    ASSERT_EQ(shapeOf, array.getShapeAsVector());
}


TEST_F(EmptyTests, test_empty_matmul_1) {
    auto x = NDArrayFactory::create<float>('c', {0, 1});
    auto y = NDArrayFactory::create<float>('c', {1, 0});
    auto e = NDArrayFactory::create<float>('c', {0, 0});

    sd::ops::matmul op;
    auto result = op.evaluate({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);
    ASSERT_EQ(e, *z);

}

TEST_F(EmptyTests, test_empty_matmul_2) {
    auto x = NDArrayFactory::create<float>('c', {1, 0, 4});
    auto y = NDArrayFactory::create<float>('c', {1, 4, 0});
    auto e = NDArrayFactory::create<float>('c', {1, 0, 0});

    sd::ops::matmul op;
    auto result = op.evaluate({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result.status());

    auto z = result.at(0);
    ASSERT_EQ(e, *z);
}
