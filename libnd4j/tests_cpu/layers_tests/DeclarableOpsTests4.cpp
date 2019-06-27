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
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests4 : public testing::Test {
public:

    DeclarableOpsTests4() {
        printf("\n");
        fflush(stdout);

        nd4j::ops::adjust_hue op0;
        nd4j::ops::adjust_saturation op1;
    }
};

template <typename T>
class TypedDeclarableOpsTests4 : public testing::Test {
public:

    TypedDeclarableOpsTests4() {
        printf("\n");
        fflush(stdout);

        nd4j::ops::adjust_hue op0;
        nd4j::ops::adjust_saturation op1;
    }
};

typedef ::testing::Types<double, float> TestingTypes;
TYPED_TEST_CASE(TypedDeclarableOpsTests4, TestingTypes);

TYPED_TEST(TypedDeclarableOpsTests4, Test_Pooling_Parity_1) {
    auto x = NDArrayFactory::create<TypeParam>('c', {2, 4, 4, 2});
    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2}, {6.f, 7.f,  10.f,  11.f,  22.f,  23.f,  26.f,  27.f,  38.f,  39.f,  42.f,  43.f,  54.f,  55.f,  58.f, 59.f});

    x.linspace(1);

    nd4j::ops::avgpool2d op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TYPED_TEST(TypedDeclarableOpsTests4, Test_Pooling_Parity_2) {
    auto x = NDArrayFactory::create<TypeParam>('c', {2, 4, 4, 2});
    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2}, {6.f, 7.f,  10.f,  11.f,  22.f,  23.f,  26.f,  27.f,  38.f,  39.f,  42.f,  43.f,  54.f,  55.f,  58.f, 59.f});

    x.linspace(1);


    nd4j::ops::avgpool2d op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TYPED_TEST(TypedDeclarableOpsTests4, Test_Pooling_Parity_5) {
    auto x = NDArrayFactory::create<TypeParam>('c', {2, 5, 5, 2});
    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 3, 3, 2}, {7.f,    8.f,   11.f,   12.f,   14.f,   15.f,   27.f,   28.f,   31.f,   32.f,   34.f,   35.f, 42.f,   43.f,   46.f,   47.f,   49.f,   50.f,   57.f,   58.f,   61.f,   62.f,   64.f,   65.f, 77.f,   78.f,   81.f,   82.f,   84.f,   85.f,   92.f,   93.f,   96.f,   97.f,   99.f,  100.f,});

    x.linspace(1);


    nd4j::ops::avgpool2d op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 1, 0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TYPED_TEST(TypedDeclarableOpsTests4, Test_Pooling_Parity_6) {
    auto x = NDArrayFactory::create<TypeParam>('c', {2, 5, 5, 2});
    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2}, {7.f,   8.f,  11.f,  12.f,  27.f,  28.f,  31.f,  32.f,  57.f,  58.f,  61.f,  62.f,  77.f,  78.f,  81.f, 82.f});

    x.linspace(1);

    nd4j::ops::avgpool2d op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TYPED_TEST(TypedDeclarableOpsTests4, Test_Pooling_Parity_8) {
    auto x = NDArrayFactory::create<TypeParam>('c', {2, 2, 5, 5});
    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 3, 3}, {1.f, 2.5f, 4.5f, 8.5f, 10.f, 12.f, 18.5f, 20.f, 22.f, 26.f, 27.5f, 29.5f, 33.5f, 35.f, 37.f, 43.5f, 45.f, 47.f,  51.f, 52.5f, 54.5f,  58.5f, 60.f, 62.f, 68.5f, 70.f, 72.f,  76.f, 77.5f, 79.5f, 83.5f, 85.f, 87.f,  93.5f, 95.f, 97.f});

    x.linspace(1);


    nd4j::ops::avgpool2d op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TYPED_TEST(TypedDeclarableOpsTests4, Test_Pooling_Parity_9) {
    auto x = NDArrayFactory::create<TypeParam>('c', {2, 2, 5, 5});
    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 3, 3}, {0.25f, 1.25f, 2.25f,  4.25f, 10.f, 12.f, 9.25f, 20.f, 22.f, 6.5f, 13.75f, 14.75, 16.75f, 35.f, 37.f,  21.75f, 45.f, 47.f,  12.75f, 26.25f, 27.25f,  29.25f, 60.f, 62.f, 34.25f, 70.f, 72.f, 19.f, 38.75f, 39.75f, 41.75f, 85.f, 87.f, 46.75f, 95.f, 97.f});

    x.linspace(1);


    nd4j::ops::avgpool2d op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 1, 1, 1, 1, 0, 1, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TYPED_TEST(TypedDeclarableOpsTests4, Test_Pooling_Parity_10) {
    auto x = NDArrayFactory::create<TypeParam>('c', {2, 2, 5, 5});
    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 3, 3}, {4.f, 6.f, 7.5f, 14.f, 16.f, 17.5f,  21.5f, 23.5f, 25.f, 29.f, 31.f, 32.5f, 39.f, 41.f, 42.5f, 46.5f, 48.5f, 50.f, 54.f, 56.f, 57.5f,  64.f, 66.f, 67.5f, 71.5f, 73.5f, 75.f, 79.f, 81.f, 82.5f, 89.f, 91.f, 92.5f,  96.5f, 98.5f, 100.f});

    x.linspace(1);


    nd4j::ops::avgpool2d op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TYPED_TEST(TypedDeclarableOpsTests4, Test_Pooling_Parity_11) {
    auto x = NDArrayFactory::create<TypeParam>('c', {1, 1, 3, 3});
    auto exp = NDArrayFactory::create<TypeParam>('c', {1, 1, 2, 2}, {3, 4, 6, 7});

    x.linspace(1);


    nd4j::ops::avgpool2d op;
    auto result = op.execute({&x}, {}, {2, 2, 1, 1, 0, 0, 1, 1, 0, 0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TYPED_TEST(TypedDeclarableOpsTests4, Test_Pooling_Parity_12) {
    auto x = NDArrayFactory::create<TypeParam>('c', {1, 1, 3, 3});
    auto exp = NDArrayFactory::create<TypeParam>('c', {1, 1, 3, 3}, {3.f, 4.f, 4.5f, 6.f, 7.f, 7.5f, 7.5f, 8.5f, 9.f});

    x.linspace(1);


    nd4j::ops::avgpool2d op;
    auto result = op.execute({&x}, {}, {2, 2, 1, 1, 0, 0, 1, 1, 1, 0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printShapeInfo("z shape:");
    //z->printBuffer("z buffer:");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_BiasAdd_NHWC_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 3, 3, 2});
    auto bias = NDArrayFactory::create<double>('c', {1, 2}, {1, 2});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 3, 2}, {1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f, 1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f});

    nd4j::ops::biasadd op;
    auto result = op.execute({&x, &bias}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_BiasAdd_NCHW_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 2, 3, 3});
    auto bias = NDArrayFactory::create<double>('c', {1, 2}, {1, 2});
    auto exp = NDArrayFactory::create<double>('c', {2, 2, 3, 3}, {1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f, 1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f});

    nd4j::ops::biasadd op;
    auto result = op.execute({&x, &bias}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Fill_1) {
    auto x = NDArrayFactory::create<int>('c', {1, 3}, {3, 2, 4});
    auto v = NDArrayFactory::create<double>(2.);
    auto exp = NDArrayFactory::create<double>('c', {3, 2, 4});
    exp.assign(2.0f);

    nd4j::ops::fill op;
    auto result = op.execute({&x, &v}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Reshape_Again) {
    auto x = NDArrayFactory::create<double>('c', {4, 3});
    auto exp = NDArrayFactory::create<double>('c', {4, 3});

    x.linspace(1);
    exp.linspace(1);

    nd4j::ops::reshape op;
    auto result = op.execute({&x}, {}, {-99, 4, 3});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Gemv_Transpose_1) {
    auto x = NDArrayFactory::create<double>('c', {4, 3});
    auto y = NDArrayFactory::create<double>('c', {4, 1});
    auto exp = NDArrayFactory::create<double>('c',{ 3, 1}, {70, 80, 90});

    x.linspace(1);
    y.linspace(1);

    nd4j::ops::matmul op;
    auto result = op.execute({&x, &y}, {}, {1, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Split_1) {
    auto x = NDArrayFactory::create<double>('c', {5, 30});
    auto sizes = NDArrayFactory::create<int>('c', {1, 3}, {4, 15, 11});

    std::vector<Nd4jLong> list0({0,0, 0,4});
    std::vector<Nd4jLong> list1({0,0, 4,19});
    std::vector<Nd4jLong> list2({0,0, 19,30});

    auto sub0 = x(list0, true);
    auto sub1 = x(list1, true);
    auto sub2 = x(list2, true);

    sub0.assign(0.0);
    sub1.assign(1.0);
    sub2.assign(2.0);


    nd4j::ops::split_v op;
    auto result = op.execute({&x, &sizes}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, result->size());

    auto z0 = result->at(0);
    auto z1 = result->at(1);
    auto z2 = result->at(2);

    ASSERT_TRUE(sub0.isSameShape(z0));
    ASSERT_TRUE(sub1.isSameShape(z1));
    ASSERT_TRUE(sub2.isSameShape(z2));

    ASSERT_TRUE(sub0.equalsTo(z0));
    ASSERT_TRUE(sub1.equalsTo(z1));
    ASSERT_TRUE(sub2.equalsTo(z2));

    delete result;
}

// special test for TF mode, when axis goes first
TEST_F(DeclarableOpsTests4, Test_Split_2) {
    auto x = NDArrayFactory::create<double>('c', {5, 12});
    auto axis = NDArrayFactory::create<double>('c', {1, 1}, {1.f});

    std::vector<Nd4jLong> list0 = {0,0, 0,3};
    std::vector<Nd4jLong> list1 = {0,0, 3,6};
    std::vector<Nd4jLong> list2 = {0,0, 6,9};
    std::vector<Nd4jLong> list3 = {0,0, 9,12};

    auto sub0 = x(list0, true);
    auto sub1 = x(list1, true);
    auto sub2 = x(list2, true);
    auto sub3 = x(list3, true);

    sub0.assign(0.0f);
    sub1.assign(1.0f);
    sub2.assign(2.0f);
    sub3.assign(3.0f);


    nd4j::ops::split op;
    auto result = op.execute({&axis, &x}, {}, {4});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z0 = result->at(0);
    auto z1 = result->at(1);
    auto z2 = result->at(2);
    auto z3 = result->at(3);

    ASSERT_TRUE(sub0.isSameShape(z0));
    ASSERT_TRUE(sub1.isSameShape(z1));
    ASSERT_TRUE(sub2.isSameShape(z2));
    ASSERT_TRUE(sub3.isSameShape(z3));

    ASSERT_TRUE(sub0.equalsTo(z0));
    ASSERT_TRUE(sub1.equalsTo(z1));
    ASSERT_TRUE(sub2.equalsTo(z2));
    ASSERT_TRUE(sub3.equalsTo(z3));

    delete result;
}

// special test for TF mode, when axis goes first
TEST_F(DeclarableOpsTests4, Test_Split_3) {
    auto x = NDArrayFactory::create<double>('c', {6, 12});
    auto axis = NDArrayFactory::create<double>('c', {1, 1}, {0.f});

    std::vector<Nd4jLong> list0 = {0,2, 0,0};
    std::vector<Nd4jLong> list1 = {2,4, 0,0};
    std::vector<Nd4jLong> list2 = {4,6, 0,0};

    auto sub0 = x(list0, true);
    auto sub1 = x(list1, true);
    auto sub2 = x(list2, true);

    sub0.assign(0.0f);
    sub1.assign(1.0f);
    sub2.assign(2.0f);

    nd4j::ops::split op;
    auto result = op.execute({&axis, &x}, {}, {3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z0 = result->at(0);
    auto z1 = result->at(1);
    auto z2 = result->at(2);

    ASSERT_TRUE(sub0.isSameShape(z0));
    ASSERT_TRUE(sub1.isSameShape(z1));
    ASSERT_TRUE(sub2.isSameShape(z2));

    ASSERT_TRUE(sub0.equalsTo(z0));
    ASSERT_TRUE(sub1.equalsTo(z1));
    ASSERT_TRUE(sub2.equalsTo(z2));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Stack_4) {
    auto t = NDArrayFactory::create<double>('c', {2, 3, 5});
    auto u = NDArrayFactory::create<double>('c', {2, 3, 5});
    auto v = NDArrayFactory::create<double>('c', {2, 3, 5});
    auto exp = NDArrayFactory::create<double>('c', {3, 2, 3, 5});

    nd4j::ops::stack op;
    auto result = op.execute({&t, &u, &v}, {}, {-4});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());


    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Squeeze_args_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 1, 1, 1, 2}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {2, 1, 2}, {1, 2, 3, 4});

    nd4j::ops::squeeze op;
    auto result = op.execute({&x}, {}, {1, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Squeeze_args_2) {
    auto x = NDArrayFactory::create<double>('c', {2, 1, 1, 1, 2}, {1, 2, 3, 4});
    auto y = NDArrayFactory::create<double>('c', {2}, {1.f, 3.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 1, 2}, {1, 2, 3, 4});

    nd4j::ops::squeeze op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Squeeze_args_3) {
    auto x = NDArrayFactory::create<double>('c', {2, 1, 1, 1, 2}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {2, 1, 2}, {1, 2, 3, 4});

    nd4j::ops::squeeze op;
    auto result = op.execute({&x}, {}, {-2, -3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_BiasAdd_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 3});
    auto row = NDArrayFactory::create<double>('c', {3}, {1, 2, 3});
    auto exp = NDArrayFactory::create<double>('c', {2, 3}, {1, 2, 3, 1, 2, 3});

    nd4j::ops::biasadd op;
    auto result = op.execute({&x, &row}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_1D_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 3});

    nd4j::ops::unstack op;
    auto result = op.execute({&x}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, result->size());

    for (int e = 0; e < 3; e++)
        ASSERT_EQ(1, result->at(e)->rankOf());

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_SpaceToDepth_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 1, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::space_to_depth op;
    auto result = op.execute({&x}, {}, {2, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_SpaceToDepth_2) {
    auto x = NDArrayFactory::create<double>('c', {1, 3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {1, 12, 1, 1}, {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});

    nd4j::ops::space_to_depth op;
    auto result = op.execute({&x}, {}, {2, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_DepthToSpace_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 1, 1, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::depth_to_space op;
    auto result = op.execute({&x}, {}, {2, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_DepthToSpace_2) {
    auto x = NDArrayFactory::create<double>('c', {1, 12, 1, 1}, {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
    auto exp = NDArrayFactory::create<double>('c', {1, 3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::depth_to_space op;
    auto result = op.execute({&x}, {}, {2, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_DepthToSpace_3) {
    auto x = NDArrayFactory::create<double>('c', {4, 4, 16, 16});
    auto exp = NDArrayFactory::create<double>('c', {4, 16, 64, 1});

    nd4j::ops::depth_to_space op;
    auto result = op.execute({&x}, {}, {4, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Cross_1) {
    auto a = NDArrayFactory::create<double>('c', {3}, {1, 2, 3});
    auto b = NDArrayFactory::create<double>('c', {3}, {6, 7, 8});
    auto exp = NDArrayFactory::create<double>('c', {3}, {-5, 10, -5});

    nd4j::ops::cross op;
    auto result = op.execute({&a, &b}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Cross_2) {
    auto a = NDArrayFactory::create<double>('c', {2, 3}, {1, 2, 3, 1, 2, 3});
    auto b = NDArrayFactory::create<double>('c', {2, 3}, {6, 7, 8, 6, 7, 8});
    auto exp = NDArrayFactory::create<double>('c', {2, 3}, {-5, 10, -5, -5, 10, -5});

    nd4j::ops::cross op;
    auto result = op.execute({&a, &b}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Cross_3) {
    auto a = NDArrayFactory::create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto b = NDArrayFactory::create<double>('c', {3, 3}, {2, 3, 4, 7, 6, 5, 6, 3, 2});
    auto exp = NDArrayFactory::create<double>('c', {3, 3}, { -1,   2,  -1, -11,  22, -11, -11,  40, -27});

    nd4j::ops::cross op;
    auto result = op.execute({&a, &b}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Matmul_YATS_1) {
    auto a = NDArrayFactory::create<double>('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto b = NDArrayFactory::create<double>('c', {4}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {30, 70, 110});

    nd4j::ops::matmul op;
    auto result = op.execute({&a, &b}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Matmul_YATS_2) {
    auto a = NDArrayFactory::create<double>('c', {4}, {1, 2, 3, 4});
    auto b = NDArrayFactory::create<double>('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {3}, {70, 80, 90});

    nd4j::ops::matmul op;
    auto result = op.execute({&a, &b}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Matmul_YATS_3) {
    auto a = NDArrayFactory::create<double>('c', {1, 4}, {1, 2, 3, 4});
    auto b = NDArrayFactory::create<double>('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {1, 3}, {70, 80, 90});

    nd4j::ops::matmul op;
    auto result = op.execute({&a, &b}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Add_119) {
    auto a = NDArrayFactory::create<double>('c', {1, 4}, {1, 2, 3, 4});
    auto b = NDArrayFactory::create<double>('c', {4}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 4}, {2, 4, 6, 8});

    nd4j::ops::add op;
    auto result = op.execute({&a, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_EQ(2, z->rankOf());

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Reshape_Negative_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto shape = NDArrayFactory::create<Nd4jLong>('c', {2}, {-1, 2});
    auto exp = NDArrayFactory::create<double>('c', {4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});

    nd4j::ops::reshape op;
    auto result = op.execute({&x, &shape}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_TileToShape_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 1, 3});
    auto exp = NDArrayFactory::create<double>('c', {2, 4, 3}, {1.f, 2.f, 3.f,1.f, 2.f, 3.f,1.f, 2.f, 3.f,1.f, 2.f, 3.f,
                                        4.f, 5.f, 6.f,4.f, 5.f, 6.f,4.f, 5.f, 6.f,4.f, 5.f, 6.f});
    x.linspace(1.f);

    nd4j::ops::tile_to_shape op;
    auto result = op.execute({&x},{}, {2, 4, 3}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_StridedSlice_Alex_1) {
    auto x = NDArrayFactory::create<double>('c', {3, 4, 5});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {1,3,4,5});
    exp.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&x}, {}, {0,0,0,1,0, -999,0,0,0, -999,3,4,5, -999,1,1,1});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_StridedSlice_Alex_2) {
    auto x = NDArrayFactory::create<double>('c', {3, 4, 5});
    auto begin = NDArrayFactory::create<double>('c', {4}, {-999,0,0,0});
    auto end = NDArrayFactory::create<double>('c', {4}, {-999,3,4,5});
    auto stride = NDArrayFactory::create<double>('c', {4}, {-999,1,1,1});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {1,3,4,5});
    exp.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&x, &begin, &end, &stride}, {}, {0,0,0,1,0});

    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_StridedSlice_Alex_3) {
    auto x = NDArrayFactory::create<double>('c', {1}, {10});
    auto begin = NDArrayFactory::create<int>('c', {1}, {(int)0});
    auto end = NDArrayFactory::create<int>('c', {1}, {(int)0});
    auto stride = NDArrayFactory::create<int>('c', {1}, {1});
    //x.linspace(1);
    //auto exp = NDArrayFactory::create<double>('c', {1,3,4,5});
    //exp.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&x, &begin, &end, &stride}, {}, {1,0,0,0,0});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    z->printShapeInfo("Emply shape expected");
    ASSERT_TRUE(z->isEmpty());

    delete result;
}
TEST_F(DeclarableOpsTests4, Test_StridedSlice_Alex_4) {
    auto x = NDArrayFactory::create<double>('c', {1,3}, {1, 2, 3});
    auto begin = NDArrayFactory::create<double>('c', {2}, {0, 0});
    auto end = NDArrayFactory::create<double>('c', {2}, {0,1});
    auto stride = NDArrayFactory::create<double>('c', {2}, {1,1});
//    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {1}, {1});
    //exp.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&x, &begin, &end, &stride}, {}, {1,0,1,0,2});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    z->printBuffer("Strided Slice");
    z->printShapeInfo("Vector size 1 shape expected");
    exp.printShapeInfo("Expected shape");
    ASSERT_TRUE(z->lengthOf() == 1);
    ASSERT_TRUE(exp.equalsTo(z));
    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test1) {

    auto x1 = NDArrayFactory::create<double>('c', {2,2,2});
    auto x2 = NDArrayFactory::create<double>('c', {2,2,2});
    auto x3 = NDArrayFactory::create<double>('c', {2,2,2});
    x1.linspace(1);
    x2.linspace(9);
    x3.linspace(17);

    auto expected = NDArrayFactory::create<double>('c', {3,2,2,2});
    expected.linspace(1);

    nd4j::ops::parallel_stack op;
    auto  results = op.execute({&x1, &x2, &x3}, {}, {});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test2) {

    auto x1 = NDArrayFactory::create<double>('c', {1,2}, {1,2});
    auto x2 = NDArrayFactory::create<double>('c', {1,2}, {3,4});
    auto x3 = NDArrayFactory::create<double>('c', {1,2}, {5,6});

    auto expected = NDArrayFactory::create<double>('c', {3,1,2}, {1,2,3,4,5,6});

    nd4j::ops::parallel_stack op;
    auto  results = op.execute({&x1, &x2, &x3}, {}, {});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test3) {

    auto x1 = NDArrayFactory::create<double>('c', {2,1}, {1,2});
    auto x2 = NDArrayFactory::create<double>('c', {2,1}, {3,4});
    auto x3 = NDArrayFactory::create<double>('c', {2,1}, {5,6});

    auto expected = NDArrayFactory::create<double>('c', {3,2,1}, {1,2,3,4,5,6});

    nd4j::ops::parallel_stack op;
    auto  results = op.execute({&x1, &x2, &x3}, {}, {});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}
\
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test4) {

    auto x1 = NDArrayFactory::create<double>('c', {2}, {1,2});
    auto x2 = NDArrayFactory::create<double>('c', {2}, {3,4});
    auto x3 = NDArrayFactory::create<double>('c', {2}, {5,6});

    auto expected = NDArrayFactory::create<double>('c', {3,2}, {1,2,3,4,5,6});

    nd4j::ops::parallel_stack op;
    auto  results = op.execute({&x1, &x2, &x3}, {}, {});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test5) {

    auto x1 = NDArrayFactory::create<double>('c', {1}, {1});
    auto x2 = NDArrayFactory::create<double>('c', {1}, {3});
    auto x3 = NDArrayFactory::create<double>('c', {1}, {5});

    auto expected = NDArrayFactory::create<double>('c', {3,1}, {1,3,5});

    nd4j::ops::parallel_stack op;
    auto  results = op.execute({&x1, &x2, &x3}, {}, {});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test6) {

    auto x1 = NDArrayFactory::create<double>(1.);
    auto x2 = NDArrayFactory::create<double>(3.);
    auto x3 = NDArrayFactory::create<double>(5.);

    auto expected = NDArrayFactory::create<double>('c', {3}, {1,3,5});

    nd4j::ops::parallel_stack op;
    auto  results = op.execute({&x1, &x2, &x3}, {}, {});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test7) {

    auto x1 = NDArrayFactory::create<double>(1.);
    auto expected = NDArrayFactory::create<double>('c', {1}, {1.});

    nd4j::ops::parallel_stack op;
    auto  results = op.execute({&x1}, {}, {});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test1) {

    auto in0 = NDArrayFactory::create<double>('c', {2}, {1, 2});
    auto in1 = NDArrayFactory::create<double>('c', {3}, {10, 20, 30});
    auto in2 = NDArrayFactory::create<double>('c', {4}, {100, 200, 300, 400});
    auto exp0 = NDArrayFactory::create<double>('c', {2,3,4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    auto exp1 = NDArrayFactory::create<double>('c', {2,3,4}, {10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30});
    auto exp2 = NDArrayFactory::create<double>('c', {2,3,4}, {100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400});

    nd4j::ops::meshgrid op;
    auto  results = op.execute({&in0, &in1, &in2}, {}, {0});
    auto  out0 = results->at(0);
    auto  out1 = results->at(1);
    auto  out2 = results->at(2);

    // out0->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test2) {

    auto in0 = NDArrayFactory::create<double>('c', {2}, {1, 2});
    auto in1 = NDArrayFactory::create<double>('c', {3}, {10, 20, 30});
    auto in2 = NDArrayFactory::create<double>('c', {4}, {100, 200, 300, 400});
    auto exp0 = NDArrayFactory::create<double>('c', {3,2,4}, {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2});
    auto exp1 = NDArrayFactory::create<double>('c', {3,2,4}, {10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30});
    auto exp2 = NDArrayFactory::create<double>('c', {3,2,4}, {100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400});

    nd4j::ops::meshgrid op;
    auto  results = op.execute({&in0, &in1, &in2}, {}, {});
    auto  out0 = results->at(0);
    auto  out1 = results->at(1);
    auto  out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test3) {

    auto in0 = NDArrayFactory::create<double>('c', {2}, {1, 2});
    auto in1 = NDArrayFactory::create<double>('c', {1,3}, {10, 20, 30});
    auto in2 = NDArrayFactory::create<double>('c', {2,2}, {100, 200, 300, 400});
    auto exp0 = NDArrayFactory::create<double>('c', {3,2,4}, {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2});
    auto exp1 = NDArrayFactory::create<double>('c', {3,2,4}, {10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30});
    auto exp2 = NDArrayFactory::create<double>('c', {3,2,4}, {100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400});

    nd4j::ops::meshgrid op;
    auto  results = op.execute({&in0, &in1, &in2}, {}, {});
    auto  out0 = results->at(0);
    auto  out1 = results->at(1);
    auto  out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test4) {

    auto in0 = NDArrayFactory::create<double>('c', {1,2}, {1, 2});
    auto in1 = NDArrayFactory::create<double>('c', {3,1}, {10, 20, 30});
    auto in2 = NDArrayFactory::create<double>('c', {1,4,1}, {100, 200, 300, 400});
    auto exp0 = NDArrayFactory::create<double>('c', {2,3,4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    auto exp1 = NDArrayFactory::create<double>('c', {2,3,4}, {10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30});
    auto exp2 = NDArrayFactory::create<double>('c', {2,3,4}, {100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400});

    nd4j::ops::meshgrid op;
    auto  results = op.execute({&in0, &in1, &in2}, {}, {0});
    auto  out0 = results->at(0);
    auto  out1 = results->at(1);
    auto  out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test5) {

    auto in0 = NDArrayFactory::create<double>(1);
    auto in1 = NDArrayFactory::create<double>(2);
    auto in2 = NDArrayFactory::create<double>(3);
    auto exp0 = NDArrayFactory::create<double>('c', {1,1,1}, {1});
    auto exp1 = NDArrayFactory::create<double>('c', {1,1,1}, {2});
    auto exp2 = NDArrayFactory::create<double>('c', {1,1,1}, {3});

    nd4j::ops::meshgrid op;
    auto  results = op.execute({&in0, &in1, &in2}, {}, {0});
    auto  out0 = results->at(0);
    auto  out1 = results->at(1);
    auto  out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test6) {

    auto in0 = NDArrayFactory::create<double>('c', {2,2},{1,2,3,4});
    auto in1 = NDArrayFactory::create<double>(5);
    auto in2 = NDArrayFactory::create<double>(6);
    auto exp0 = NDArrayFactory::create<double>('c', {4,1,1}, {1,2,3,4});
    auto exp1 = NDArrayFactory::create<double>('c', {4,1,1}, {5,5,5,5});
    auto exp2 = NDArrayFactory::create<double>('c', {4,1,1}, {6,6,6,6});

    nd4j::ops::meshgrid op;
    auto  results = op.execute({&in0, &in1, &in2}, {}, {0});
    auto  out0 = results->at(0);
    auto  out1 = results->at(1);
    auto  out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test7) {

    auto in0 = NDArrayFactory::create<double>('c', {2,2},{1,2,3,4});
    auto in1 = NDArrayFactory::create<double>(5);
    auto in2 = NDArrayFactory::create<double>(6);
    auto exp0 = NDArrayFactory::create<double>('c', {1,4,1}, {1,2,3,4});
    auto exp1 = NDArrayFactory::create<double>('c', {1,4,1}, {5,5,5,5});
    auto exp2 = NDArrayFactory::create<double>('c', {1,4,1}, {6,6,6,6});

    nd4j::ops::meshgrid op;
    auto  results = op.execute({&in0, &in1, &in2}, {}, {1});
    auto  out0 = results->at(0);
    auto  out1 = results->at(1);
    auto  out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test8) {

    auto in0 = NDArrayFactory::create<double>(5);
    auto exp0 = NDArrayFactory::create<double>('c', {1}, {5});

    nd4j::ops::meshgrid op;
    auto  results = op.execute({&in0}, {}, {0});
    auto  out0 = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test9) {

    auto in0 = NDArrayFactory::create<double>(5);
    auto exp0 = NDArrayFactory::create<double>('c', {1}, {5});

    nd4j::ops::meshgrid op;
    auto  results = op.execute({&in0}, {}, {1});
    auto  out0 = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, WeightedCrossEntropyWithLogits_1) {


    auto input    = NDArrayFactory::create<double>('c', {2, 3}, {11.f, 13.f,  4.f, 15.f,  6.f,  3.f});
    auto targets  = NDArrayFactory::create<double>('c', {2, 3}, {15.5f, 15.7f,  5.f , 15.f,   5.f,   6.f});
    auto weight  = NDArrayFactory::create<double>(0.7f);
    auto expected = NDArrayFactory::create<double>('c', {2, 3}, {-159.50006,  -191.1, -16.009075, -210., -24.001238, -15.03887});

//Targets {15.5f, 15.7f,  5.f , 15.f,   5.f,   6.f};
//----------
//Inputs {11.f, 13.f,  4.f, 15.f,  6.f,  3.f};
//----------
//Weights [0.7]
//Result {-159.50006,  -191.1,       -16.009075, -210., -24.001238,  -15.03887}

    nd4j::ops::weighted_cross_entropy_with_logits op;
    auto results = op.execute({&targets, &input, &weight}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto  output = results->at(0);

    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, WeightedCrossEntropyWithLogits_2) {


    auto input    = NDArrayFactory::create<double>('c', {2, 3}, {11.f,   13.f,  4.f, 15.f,  6.f,  3.f});
    auto targets  = NDArrayFactory::create<double>('c', {2, 3}, {15.5f, 15.7f,  5.f, 15.f,  5.f,  6.f});
    auto weights  = NDArrayFactory::create<double>({0.5f, 0.7f, 1.0f}) ;
    auto expected = NDArrayFactory::create<double>('c', {2, 3}, {-159.5001f, -191.1f, -15.98185f, -210.f,  -24.001238f, -14.951412f});

    nd4j::ops::weighted_cross_entropy_with_logits op;
    auto results = op.execute({&targets, &input, &weights}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto  output = results->at(0);

    output->printIndexedBuffer("Result is ");
    expected.printIndexedBuffer("Expected is ");

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, lstm_test1) {

    const int time      = 5;
    const int batchSize = 3;
    const int inSize    = 3;
    const int numProj   = 3;
    const int numUnits  = 3;

    auto x   = NDArrayFactory::create<double>('c', {time, batchSize, inSize});
    auto h0  = NDArrayFactory::create<double>('c', {batchSize, numProj});
    auto c0  = NDArrayFactory::create<double>('c', {batchSize, numUnits});
    auto Wx  = NDArrayFactory::create<double>('c', {inSize, 4*numUnits});
    auto Wh  = NDArrayFactory::create<double>('c', {numProj, 4*numUnits});
    auto Wc  = NDArrayFactory::create<double>('c', {3*numUnits});
    auto Wp  = NDArrayFactory::create<double>('c', {numUnits, numProj});
    auto b   = NDArrayFactory::create<double>('c', {4*numUnits});

    x.linspace(0.5, 0.5);
    h0 = 1.;
    c0 = 2.;
    Wx = 0.003;
    Wh = 0.006;
    Wc = 0.;
    Wp = 0.;
    b = 0.5;

    auto expH = NDArrayFactory::create<double>('c', {time, batchSize, numProj}, {0.57574,0.57574,0.57574,0.58006,0.58006,0.58006,0.58434,0.58434,0.58434,
                                                           0.55114,0.55114,0.55114,0.55732,0.55732,0.55732,0.56338,0.56338,0.56338,
                                                           0.53763,0.53763,0.53763,0.54534,0.54534,0.54534,0.55287,0.55287,0.55287,
                                                           0.53626,0.53626,0.53626,0.54487,0.54487,0.54487,0.55327,0.55327,0.55327,
                                                           0.54484,0.54484,0.54484,0.55379,0.55379,0.55379,0.5625 ,0.5625 ,0.5625});

    auto expClast = NDArrayFactory::create<double>('c', {1, batchSize, numProj}, {1.1589154,1.1589154,1.1589154,1.1892855,1.1892855,1.1892855,1.219861 ,1.219861 ,1.219861});

    nd4j::ops::lstm op;
    auto results = op.execute({&x, &h0, &c0, &Wx, &Wh, &Wc, &Wp, &b}, {0., 0., 0.}, {0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *h = results->at(0);
    auto *c = results->at(1);
    auto cLast = (*c)({4,5,0,0,0,0},true);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));

    ASSERT_TRUE(expClast.isSameShape(&cLast));
    ASSERT_TRUE(expClast.equalsTo(&cLast));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, relu6_test1) {

    auto input  = NDArrayFactory::create<double>('c', {2,4}, {-13.,10,-5,0,2,7,6,12});
    auto expected  = NDArrayFactory::create<double>('c', {2,4}, {0., 6., 0., 0.,2., 6., 6., 6.});

    nd4j::ops::relu6 op;
    auto results = op.execute({&input}, {0.}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto  output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, relu6_bp_test1) {

    auto input  = NDArrayFactory::create<double>('c', {2,4}, {-13.,10, -5,  0,  2,  7,  6,  5});
    auto gradO  = NDArrayFactory::create<double>('c', {2,4}, {-1., -2., 0., 4., 5., 6., 7., 8.});

    auto expected  = NDArrayFactory::create<double>('c', {2,4}, {0., 0., 0., 0., 5., 0., 0., 8.});

    nd4j::ops::relu6_bp op;
    auto results = op.execute({&input, &gradO}, {0.}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto  output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests4, LrnTest_1) {

    auto x = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2}, { 5.5, 0., 0.3, 5.5,
                                            8.6, 0.,  0., 0.4,
                                            1.5, 1., 1.3, 1.5,
                                            2.6, 2.,  3., 1.4}
    );

    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2}, {
                                            0.98386997,        0.,  0.05358852,  0.9824562,
                                            0.99330735,        0.,          0., 0.37139067,
                                            0.72760683, 0.4850712,   0.5848977, 0.67488194,
                                            0.7581754,  0.58321184, 0.86747235, 0.4048204}
    );

    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {5}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    // out->printIndexedBuffer("LRN out");
    // exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}


////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests4, LrnTest_2) {

    auto x = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2}, { 5.5, 0., 0.3, 5.5,
                                            8.6, 0.,  0., 0.4,
                                            1.5, 1., 1.3, 1.5,
                                            2.6, 2.,  3., 1.4});

    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 2}, {
                                            0.98386997,        0.,  0.05358852,  0.9824562,
                                            0.99330735,        0.,          0., 0.37139067,
                                            0.72760683, 0.4850712,   0.5848977, 0.67488194,
                                            0.7581754,  0.58321184, 0.86747235, 0.4048204});

    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    // out->printIndexedBuffer("LRN out");
    // exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests4, LrnTest_3) {

    auto x = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 4}, {

                5.5, 0., 0.3, 5.5,
                1.5, 0., 1.3, 6.5,
                8.6, 0.,  0., 0.4,
                2.5, 1., 0.3, 4.5,
                1.5, 1., 1.3, 1.5,
                3.5, 0., 1.3, 2.5,
                2.6, 2.,  3., 1.4,
                4.5, 1., 0.3, 0.5}
    );

    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 4}, {
                     0.9824562,          0., 0.03822664, 0.9824562,
                    0.67488194,          0., 0.18924236, 0.96960944,
                    0.99330735,          0.,         0., 0.37139067,
                    0.86567914,  0.18702209, 0.05610663, 0.9520745,
                     0.6154575,  0.34942827, 0.45425674, 0.6154575,
                      0.905509,  0.        ,  0.2824086, 0.8361251,
                    0.57063663,  0.41959068,   0.629386, 0.3504383,
                     0.9520745,  0.21039814, 0.06311944, 0.3268602 }
    );

    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    // out->printIndexedBuffer("LRN out");
    // exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests4, LrnTest_4) {

    auto x = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 4}, {

                    5.5, 0., 0.3, 5.5,
                    1.5, 0., 1.3, 6.5,
                    8.6, 0.,  0., 0.4,
                    2.5, 1., 0.3, 4.5,
                    1.5, 1., 1.3, 1.5,
                    3.5, 0., 1.3, 2.5,
                    2.6, 2.,  3., 1.4,
                    4.5, 1., 0.3, 0.5}
    );

    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 4}, {
                    0.70082176,         0., 0.03822664, 0.70082176,
                    0.21835658,         0., 0.18924236,  0.9462118,
                     0.9922489,         0.,         0., 0.04615111,
                    0.46755522, 0.18702209, 0.05610663,  0.8415994,
                     0.5241424, 0.34942827, 0.45425674,  0.5241424,
                    0.76033086,         0.,  0.2824086, 0.54309344,
                    0.54546785, 0.41959068,   0.629386, 0.29371348,
                    0.94679165, 0.21039814, 0.06311944, 0.10519907}
    );

    nd4j::ops::lrn op;
    auto  results = op.execute({&x}, {1.0, 1.0, 0.5}, {5}, {}, false, nd4j::DataType::DOUBLE);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    // out->printIndexedBuffer("LRN out");
    // exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests4, LrnTest_5) {

    auto x = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 4}, {

                5.5,0., 0.3, 5.5,
                1.5,0., 1.3, 6.5,
                8.6,0.,  0., 0.4,
                2.5,1., 0.3, 4.5,
                1.5,1., 1.3, 1.5,
                3.5,0., 1.3, 2.5,
                2.6,2.,  3., 1.4,
                4.5,1., 0.3, 0.5}
    );

    auto eps = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 4}, {
                0.70082176, 0.,         0.03822664, 0.70082176,
                0.21835658, 0.,         0.18924236,  0.9462118,

                0.9922489,  0.,         0.        , 0.04615111,
                0.46755522, 0.18702209, 0.05610663,  0.8415994,


                0.5241424,  0.34942827, 0.45425674,  0.5241424,
                0.76033086, 0.,         0.2824086 , 0.54309344,

                0.54546785, 0.41959068, 0.629386  , 0.29371348,
                0.94679165, 0.21039814, 0.06311944, 0.10519907}
    );

    auto exp = NDArrayFactory::create<TypeParam>('c', {2, 2, 2, 4});

    nd4j::ops::lrn_bp op;
    auto  results = op.execute({&x, &eps}, {1.0, 1.0, 0.5}, {5}, {}, false, typeid(TypeParam) == typeid(float) ? nd4j::DataType::FLOAT32 : nd4j::DataType::DOUBLE);
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    // out->printIndexedBuffer("LRN out");
    // exp.printIndexedBuffer("LRN exp");
//    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test1) {

    const int rows = 3;
    const int cols = 5;

    auto expected = NDArrayFactory::create<float>('c', {rows, cols}, {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0});

    nd4j::ops::tri op;
    auto results = op.execute({}, {}, {rows, cols});
    auto  output = results->at(0);

    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test2) {

    const int rows = 3;
    const int cols = 5;
    const int diag = 2;

    auto expected = NDArrayFactory::create<float>('c', {rows, cols}, {1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});

    nd4j::ops::tri op;
    auto results = op.execute({}, {}, {rows, cols, diag});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test3) {

    const int rows = 3;
    const int cols = 5;
    const int diag = -1;

    auto expected = NDArrayFactory::create<float>('c', {rows, cols}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0});

    nd4j::ops::tri op;
    auto results = op.execute({}, {}, {rows, cols, diag});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test4) {

    const int rows = 3;
    const int cols = 5;
    const int diag = -2;

    auto expected = NDArrayFactory::create<float>('c', {rows, cols}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0});

    nd4j::ops::tri op;
    auto results = op.execute({}, {}, {rows, cols, diag});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test5) {

    const int rows = 5;

    auto expected = NDArrayFactory::create<float>('c', {rows, rows}, {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});

    nd4j::ops::tri op;
    auto results = op.execute({}, {}, {rows});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test6) {

    const int rows = 3;
    const int cols = 5;
    const int diag = -20;

    auto expected = NDArrayFactory::create<float>('c', {rows, cols}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

    nd4j::ops::tri op;
    auto results = op.execute({}, {}, {rows, cols, diag});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test7) {

    const int rows = 3;
    const int cols = 5;
    const int diag = 20;

    auto expected = NDArrayFactory::create<float>('c', {rows, cols}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    nd4j::ops::tri op;
    auto results = op.execute({}, {}, {rows, cols, diag});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test1) {

    auto input = NDArrayFactory::create<double>('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto expected = NDArrayFactory::create<double>('c', {4, 3}, {1,  2,  3, 0, 5, 6, 0,  0,  9, 0,  0, 0});

    nd4j::ops::triu op;
    auto results = op.execute({&input}, {}, {});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));


    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test2) {

    auto input = NDArrayFactory::create<double>('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto expected = NDArrayFactory::create<double>('c', {4, 3}, {1,  2,  3,4,  5,  6,0,  8,  9,0,  0, 12});

    nd4j::ops::triu op;
    auto results = op.execute({&input}, {}, {-1});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test3) {

    auto input = NDArrayFactory::create<double>('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto expected = NDArrayFactory::create<double>('c', {2, 3, 2}, {1, 2,3, 4,0, 6,7, 8,9,10,0,12});

    nd4j::ops::triu op;
    auto results = op.execute({&input}, {}, {-1});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test4) {

    auto input = NDArrayFactory::create<double>('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto expected = NDArrayFactory::create<double>('c', {2, 3, 2}, {1,  2,0,  4,0,  0,7,  8,0, 10,0,  0});

    nd4j::ops::triu op;
    auto results = op.execute({&input}, {}, {});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test5) {

    auto input = NDArrayFactory::create<double>('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto expected = NDArrayFactory::create<double>('c', {2, 3, 2}, {0, 2,0,  0,0,  0,0,  8,0, 0,0,  0});

    nd4j::ops::triu op;
    auto results = op.execute({&input}, {}, {1});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test6) {

    auto input = NDArrayFactory::create<double>('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto expected = NDArrayFactory::create<double>('c', {2, 3, 2}, {0, 0,0,  0,0,  0,0,  0,0, 0,0,  0});

    nd4j::ops::triu op;
    auto results = op.execute({&input}, {}, {10});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test7) {

    auto input = NDArrayFactory::create<double>('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto expected = NDArrayFactory::create<double>('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::triu op;
    auto results = op.execute({&input}, {}, {-10});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test8) {

    auto input = NDArrayFactory::create<double>('c', {6}, {1, 2, 3, 4, 5, 6});
    auto expected = NDArrayFactory::create<double>('c', {6, 6}, {1, 2, 3, 4, 5, 6,0, 2, 3, 4, 5, 6,0, 0, 3, 4, 5, 6,0, 0, 0, 4, 5, 6,0, 0, 0, 0, 5, 6,0, 0, 0, 0, 0, 6});

    nd4j::ops::triu op;
    auto results = op.execute({&input}, {}, {});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test9) {

    auto input = NDArrayFactory::create<double>('c', {6}, {1, 2, 3, 4, 5, 6});
    auto expected = NDArrayFactory::create<double>('c', {6, 6}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 0, 2, 3, 4, 5, 6, 0, 0, 3, 4, 5, 6});

    nd4j::ops::triu op;
    auto results = op.execute({&input}, {}, {-3});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test10) {

    auto input = NDArrayFactory::create<double>('c', {6}, {1, 2, 3, 4, 5, 6});
    auto expected = NDArrayFactory::create<double>('c', {6, 6}, {0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

    nd4j::ops::triu op;
    auto results = op.execute({&input}, {}, {3});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test11) {

    auto input = NDArrayFactory::create<double>('c', {6}, {1, 2, 3, 4, 5, 6});
    auto expected = NDArrayFactory::create<double>('c', {6, 6}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});

    nd4j::ops::triu op;
    auto results = op.execute({&input}, {}, {-58});
    auto  output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_bp_test1) {

    auto input = NDArrayFactory::create<double>('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto gradO = NDArrayFactory::create<double>('c', {2, 3, 2});
    gradO = 0.5;

    auto expected = NDArrayFactory::create<double>('c', {2, 3, 2}, {0.,0.5,0.,0. ,0.,0. ,0.,0.5,0.,0. ,0.,0.});

    nd4j::ops::triu_bp op;
    auto results = op.execute({&input, &gradO}, {}, {1});
    auto  gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(gradI));
    ASSERT_TRUE(expected.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_bp_test2) {

    auto input = NDArrayFactory::create<double>('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto gradO = NDArrayFactory::create<double>('c', {2, 3, 2});
    gradO = 0.5;

    auto expected = NDArrayFactory::create<double>('c', {2, 3, 2}, {0.5,0.5,0. ,0.5,0. ,0. ,0.5,0.5,0. ,0.5,0. ,0.});

    nd4j::ops::triu_bp op;
    auto results = op.execute({&input, &gradO}, {}, {});
    auto  gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(gradI));
    ASSERT_TRUE(expected.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_bp_test3) {

    auto input = NDArrayFactory::create<double>('c', {6}, {1, 2, 3, 4, 5, 6});
    auto gradO = NDArrayFactory::create<double>('c', {6,6});
    gradO = 0.5;

    auto expected = NDArrayFactory::create<double>('c', {6,6}, {0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0. , 0.5, 0.5, 0.5, 0.5, 0.5,0. , 0. , 0.5, 0.5, 0.5, 0.5,0. , 0. , 0. , 0.5, 0.5, 0.5});

    nd4j::ops::triu_bp op;
    auto results = op.execute({&input, &gradO}, {}, {-2});
    auto  gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(gradI));
    ASSERT_TRUE(expected.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_bp_test4) {

    auto input = NDArrayFactory::create<double>('c', {2,3}, {1, 2, 3, 4, 5, 6});
    auto gradO = NDArrayFactory::create<double>('c', {2,3});
    gradO = 0.5;

    auto expected = NDArrayFactory::create<double>('c', {2,3}, {0., 0., 0., 0., 0., 0.});

    nd4j::ops::triu_bp op;
    auto results = op.execute({&input, &gradO}, {}, {10});
    auto  gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());

    ASSERT_TRUE(expected.isSameShape(gradI));
    ASSERT_TRUE(expected.equalsTo(gradI));

    delete results;
}

