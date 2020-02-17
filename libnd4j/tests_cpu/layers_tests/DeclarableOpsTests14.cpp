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
// Created by raver on 8/4/2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>


using namespace nd4j;


class DeclarableOpsTests14 : public testing::Test {
public:

    DeclarableOpsTests14() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests14, Test_Validation_Edge_1) {
    auto x = NDArrayFactory::create<int>('c', {2}, {2, 2});
    auto exp = NDArrayFactory::create('c', {2, 2}, Environment::getInstance()->defaultFloatDataType());
    exp.assign(4.0f);

    nd4j::ops::fill op;
    auto result = op.evaluate({&x}, {4.0f});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(exp, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, Test_Reshape_CF_1) {
    auto x = NDArrayFactory::create<double>('f', {2, 3}, {1.0, 4.0, 2.0, 5.0, 3.0, 6.0});
    auto e = NDArrayFactory::create<double>('f', {3, 2}, {1.0, 3.0, 5.0, 2.0, 4.0, 6.0});

    auto r = x.reshape('c', {3, 2});;
    r.streamline('f');

    nd4j::ops::reshape op;
    auto result = op.evaluate({&x}, {3, 2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    delete result;
}

TEST_F(DeclarableOpsTests14, Test_Inf_Comparison_1) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, std::numeric_limits<double>::infinity(), 5});
    auto y = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, std::numeric_limits<double>::infinity(), 5});

    ASSERT_EQ(x, y);
}

TEST_F(DeclarableOpsTests14, Test_Inf_Comparison_2) {
#ifdef FFAST_MATH
    if (1 > 0)
        return;
#endif

    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, std::numeric_limits<double>::infinity(), 5});
    auto y = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, -std::numeric_limits<double>::infinity(), 5});

    ASSERT_NE(x, y);
}

TEST_F(DeclarableOpsTests14, Multiply_test) {

    for(int k=2;k<10;k++){
        //nd4j_printf("k=%d\n", k);
        NDArray x = NDArrayFactory::create<double>('c', {k, 1});
        NDArray y = NDArrayFactory::create<double>('c', {k});
        NDArray e = NDArrayFactory::create<double>('c', {k, k});
        x.assign(1.0);
        y.assign(1.0);
        e.assign(1.0);

        nd4j::ops::multiply op;
        auto result = op.evaluate({&x, &y});
        auto f = result->at(0);
        NDArray r = *f;

        ASSERT_EQ(e, r);
        ASSERT_EQ(e, *f);

        delete result;
    }
}

TEST_F(DeclarableOpsTests14, Test_EvalReductionShape_1) {
    auto x = NDArrayFactory::create<int>('c', {3}, {5, 3, 4});
    auto y = NDArrayFactory::create<int>('c', {1}, {1});
    auto e = NDArrayFactory::create<Nd4jLong>('c', {2}, {5, 4});

    nd4j::ops::evaluate_reduction_shape op;
    auto result = op.evaluate({&x, &y}, {}, {}, {false, false});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, Test_EvalReductionShape_2) {
    auto x = NDArrayFactory::create<int>('c', {3}, {5, 3, 4});
    auto y = NDArrayFactory::create<int>('c', {1}, {1});
    auto e = NDArrayFactory::create<Nd4jLong>('c', {3}, {5, 1, 4});

    nd4j::ops::evaluate_reduction_shape op;
    auto result = op.evaluate({&x, &y}, {}, {}, {true, false});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, Test_Reduce_Min_Small_0) {
    auto x = NDArrayFactory::create<float>('c', {3, 4}, {-999.f, 0.2236f, 0.7973f, 0.0962f, 0.7231f, 0.3381f, -0.7301f, 0.9115f, -0.5094f, 0.9749f, -2.1340f, 0.6023f});
    auto z = NDArrayFactory::create<float>('c', {4});
    auto e = NDArrayFactory::create<float>('c', {4}, {-999.f, 0.2236f, -2.1340f, 0.0962f});

    nd4j::ops::reduce_min op;
    op.execute({&x}, {&z}, {}, {0}, {});

    //z.printIndexedBuffer("Z");

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests14, Test_Reduce_Min_Small_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 4}, {-999.f, 0.2236f, 0.7973f, 0.0962f, 0.7231f, 0.3381f, -0.7301f, 0.9115f, -0.5094f, 0.9749f, -2.1340f, 0.6023f});
    auto z = NDArrayFactory::create<float>('c', {3});
    auto e = NDArrayFactory::create<float>('c', {3}, {-999.f, -0.7301f, -2.1340f});

    nd4j::ops::reduce_min op;
    op.execute({&x}, {&z}, {}, {1}, {});

    //z.printIndexedBuffer("Z");

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests14, Test_Diag_Zeros_1) {
    auto x = NDArrayFactory::create<double>('c', {2}, {1, 2});
    auto z = NDArrayFactory::create<double>('c', {2, 2}, {-119, -119, -119, -119});
    auto exp = NDArrayFactory::create<double>('c', {2, 2}, {1, 0, 0, 2});

    nd4j::ops::diag op;
    auto status = op.execute({&x}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(exp, z);
}

TEST_F(DeclarableOpsTests14, Test_scalar_broadcast_1) {
    auto x = NDArrayFactory::create<float>(1.0f);
    auto y = NDArrayFactory::create<float>('c', {5, 10});
    auto e = NDArrayFactory::create<float>('c', {5, 10});
    e.assign(1.0);


    nd4j::ops::add op;
    auto result = op.evaluate({&x, &y});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests14, Test_scalar_broadcast_2) {
    auto x = NDArrayFactory::create<float>(1.0f);
    auto y = NDArrayFactory::create<float>('c', {5, 10});
    auto e = NDArrayFactory::create<float>('c', {5, 10});
    y.assign(2.0f);
    e.assign(-1.0f);


    nd4j::ops::subtract op;
    auto result = op.evaluate({&x, &y});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests14, test_empty_fill_1) {
    auto x = NDArrayFactory::empty<int>();
    auto y = NDArrayFactory::create<int>(1);

    nd4j::ops::fill op;
    auto result = op.evaluate({&x, &y});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(y, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, test_lstmBlockCell_1) {
    auto a = NDArrayFactory::create<double>('c', {1, 5}, {0.7787856f, 0.80119777f, 0.72437465f, 0.23089433f, 0.72714126f});
    auto b = NDArrayFactory::create<double>('c', {1, 3});
    auto c = NDArrayFactory::create<double>('c', {1, 3});
    auto d = NDArrayFactory::create<double>('c', {8, 12}, {-0.15320599,-0.120416045,0.33126968,0.13921785,-0.32313538,-0.43956736,0.4756174,0.4335605,-0.5450856,-0.3943429,-0.28687626,0.068032146,-0.2793799,0.17298919,-0.36553562,-0.097853184,-0.2544747,-0.39872527,-0.14556861,-0.31479517,0.2559092,0.47166896,-0.31330687,0.47313118,0.5134543,-0.4678212,-0.12853557,0.26142156,0.43472284,-0.42842552,-0.1895876,0.538689,0.508651,-0.020272732,0.112327516,0.2704304,-0.046546757,0.32570732,-0.15148133,-0.19145513,0.18631572,-0.024152994,0.41603214,-0.3421499,0.0106860995,-0.2966229,-0.36713937,0.25841123,0.0843398,0.49082482,0.10800403,0.1874243,-0.26379472,-0.22531849,0.24924624,0.23119557,0.49940765,-0.051413506,0.20315129,-0.41888732,0.44097036,0.40453392,0.013338983,0.23434466,0.23942488,0.47894,-0.19898453,0.09253675,-0.032358468,-0.15213022,-0.3441009,-0.15600958,-0.08235118,0.12165731,-0.4481289,-0.4842423,-0.45797008,-0.4606034,0.08163166,-0.2981107,0.50207126,0.44195646,0.13850057,0.072246075,-0.34388685,0.030900061,0.35821778,0.47900867,0.5094063,0.23683065,0.18020362,-0.1369732,0.015235603,0.2786904,0.07954317,0.12543976});
    auto e = NDArrayFactory::create<double>('c', {3});
    auto f = NDArrayFactory::create<double>('c', {3});
    auto g = NDArrayFactory::create<double>('c', {3});
    auto h = NDArrayFactory::create<double>('c', {12});

    auto z0 = NDArrayFactory::create<double>('c', {1, 3});
    auto z1 = NDArrayFactory::create<double>('c', {1, 3});
    auto z2 = NDArrayFactory::create<double>('c', {1, 3});
    auto z3 = NDArrayFactory::create<double>('c', {1, 3});
    auto z4 = NDArrayFactory::create<double>('c', {1, 3});
    auto z5 = NDArrayFactory::create<double>('c', {1, 3});
    auto z6 = NDArrayFactory::create<double>('c', {1, 3});

    nd4j::ops::lstmBlockCell op;
    auto result = op.execute({&a, &b, &c, &d, &e, &f, &g, &h}, {&z0, &z1, &z2, &z3, &z4, &z5, &z6}, {1.0, -1.0}, {0}, {});
    ASSERT_EQ(Status::OK(), result);
}

TEST_F(DeclarableOpsTests14, test_empty_stack_1) {
    auto x = NDArrayFactory::create<float>('c', {0});
    auto e = NDArrayFactory::create<float>('c', {1, 0});

    nd4j::ops::stack op;
    auto result = op.evaluate({&x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);
    nd4j::ops::reduce_min sumOp;
    auto res2 = sumOp.evaluate({&e}, {1.}, {1});
    ASSERT_EQ(res2->status(), Status::OK());
    auto out = res2->at(0);

    ASSERT_EQ(out->e<float>(0), DataTypeUtils::infOrMax<float>());
    delete res2;
    delete result;
}

TEST_F(DeclarableOpsTests14, test_empty_stack_2) {
    auto x = NDArrayFactory::empty<float>();
    auto e = NDArrayFactory::create<float>('c', {0});

    nd4j::ops::stack op;
    auto result = op.evaluate({&x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, test_empty_stack_3) {
    auto x = NDArrayFactory::empty<float>();
    auto e = NDArrayFactory::create<float>('c', {2, 0});

    nd4j::ops::stack op;
    auto result = op.evaluate({&x, &x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, test_empty_stack_4) {
    auto x = NDArrayFactory::create<float>('c', {0});
    auto e = NDArrayFactory::create<float>('c', {2, 0});

    nd4j::ops::stack op;
    auto result = op.evaluate({&x, &x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, test_empty_reduce_min_1) {

    auto e = NDArrayFactory::create<float>('c', {1, 0});
    nd4j::ops::reduce_min sumOp;
    auto res2 = sumOp.evaluate({&e}, {1.}, {1});
    ASSERT_EQ(res2->status(), Status::OK());
    auto out = res2->at(0);

    ASSERT_EQ(out->e<float>(0), DataTypeUtils::infOrMax<float>());
    delete res2;
}

TEST_F(DeclarableOpsTests14, test_empty_reduce_max_1) {

    auto e = NDArrayFactory::create<float>('c', {1, 0});
    nd4j::ops::reduce_max sumOp;
    auto res2 = sumOp.evaluate({&e}, {1.}, {1});
    ASSERT_EQ(res2->status(), Status::OK());
    auto out = res2->at(0);

    ASSERT_EQ(out->e<float>(0), -DataTypeUtils::infOrMax<float>());
    delete res2;
}

TEST_F(DeclarableOpsTests14, test_empty_reduce_sum_1) {
#ifdef FFAST_MATH
    if (1 > 0)
        return;
#endif

    auto e = NDArrayFactory::create<float>('c', {1, 0});
    nd4j::ops::reduce_sum sumOp;
    auto res2 = sumOp.evaluate({&e}, {1.}, {1});
    ASSERT_EQ(res2->status(), Status::OK());
    auto out = res2->at(0);
    ASSERT_EQ(out->e<float>(0), 0.f);
    delete res2;
}

TEST_F(DeclarableOpsTests14, test_empty_reduce_mean_1) {
#ifdef FFAST_MATH
    if (1 > 0)
        return;
#endif

    auto e = NDArrayFactory::create<float>('c', {1, 0});
    nd4j::ops::reduce_mean sumOp;
    auto res2 = sumOp.evaluate({&e}, {1.}, {1});
    ASSERT_EQ(res2->status(), Status::OK());
    auto out = res2->at(0);
    // out->printShapeInfo("ReduceMean empty shape with keep dims");
    // out->printIndexedBuffer("ReduceMean scalar");
    ASSERT_TRUE(std::isnan(out->e<float>(0)));
    delete res2;
}

TEST_F(DeclarableOpsTests14, Test_StridedSliceZeros_1) {
    auto matrix = NDArrayFactory::create<double>('c', {1, 2, 0, 4});
    auto b = NDArrayFactory::create<int>('c', {3}, {0, 0, 0});
    auto e = NDArrayFactory::create<int>('c', {3}, {2,0,2});
    auto s = NDArrayFactory::create<int>('c', {3}, {1,1,1});

    auto exp = NDArrayFactory::create<double>('c', {1,0,0,4});

    matrix.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.evaluate({&matrix, &b, &e, &s}, {}, {0, 0, 0, 0, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}

TEST_F(DeclarableOpsTests14, Test_StridedSliceZeros_2) {
    auto matrix = NDArrayFactory::create<double>('c', {1, 2, 0, 4});
    auto b = NDArrayFactory::create<int>('c', {3}, {0, 0, 0});
    auto e = NDArrayFactory::create<int>('c', {3}, {2,0,2});
    auto s = NDArrayFactory::create<int>('c', {3}, {1,1,1});

    auto exp = NDArrayFactory::create<double>('c', {0,0,4});

    matrix.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.evaluate({&matrix, &b, &e, &s}, {}, {0, 0, 0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}

TEST_F(DeclarableOpsTests14, test_empty_argmax_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 0});
    auto y = NDArrayFactory::create<int>(0);
    auto e = NDArrayFactory::create<Nd4jLong>('c', {0});

    nd4j::ops::argmax op;
    //nd4j::ops::reduce_max op;

    auto result = op.evaluate({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, test_empty_argmax_2) {
    auto x = NDArrayFactory::create<float>('c', {1, 0});
    auto y = NDArrayFactory::create<int>(1);

    nd4j::ops::argmax op;
    try {
        auto result = op.execute({&x, &y}, {&y}, {}, {}, {});
        ASSERT_TRUE(false);
    } catch (std::exception &e) {
        //
    }
}

TEST_F(DeclarableOpsTests14, test_empty_tanh_5) {
    auto x = NDArrayFactory::create<float>('c', {32, 0});

    nd4j::ops::tanh op;
    auto result = op.evaluate({&x}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x.isSameShape(z));
    ASSERT_EQ(x, *z);

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, repeat_1) {

    NDArray x('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    NDArray e('c', {4, 3}, {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6});

    nd4j::ops::repeat op;
    auto result = op.evaluate({&x}, {}, {2, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, repeat_2) {

    NDArray x('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    NDArray e('c', {2, 6}, {1, 1, 2, 2, 3, 3,4, 4, 5, 5, 6, 6});

    nd4j::ops::repeat op;
    auto result = op.evaluate({&x}, {}, {2, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, repeat_3) {

    NDArray x('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    NDArray e('c', {2, 6}, {1, 2, 2, 3, 3, 3,4, 5, 5, 6, 6, 6});

    nd4j::ops::repeat op;
    auto result = op.evaluate({&x}, {}, {1,2,3,  1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, repeat_4) {

    NDArray x('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    NDArray e('c', {7, 3}, {1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6});

    nd4j::ops::repeat op;
    auto result = op.evaluate({&x}, {}, {3,4,  0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, repeat_5) {

    NDArray x('c', {2, 3, 4}, {1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray e('c', {2, 4, 4}, {1,  2,  3,  4, 5,  6,  7,  8, 5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 17, 18, 19, 20, 21, 22, 23, 24});

    nd4j::ops::repeat op;
    auto result = op.evaluate({&x}, {}, {1,2,1,  1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}
/////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest) {

    auto y = NDArray('c', { 3 }, nd4j::DataType::FLOAT32);
    auto x = NDArray('c', { 5, 2, 1 }, nd4j::DataType::FLOAT32);

    auto e = NDArray('c', { 5, 2, 3 }, { 2., 2., 2., 3., 3., 3., 4., 4., 4., 5., 5., 5., 6., 6., 6., 7., 7., 7., 8., 8., 8., 9., 9., 9., 10., 10., 10., 11., 11., 11. }, nd4j::DataType::FLOAT32);

    y.assign(1.0);
    x.linspace(1.0);

    nd4j::ops::add op;
    auto result = op.evaluate({ &x, &y });
    ASSERT_EQ(Status::OK(), result->status());

    auto res = *result->at(0);

    ASSERT_EQ(e, res);

    delete result;
}
/////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest2) {

    auto y = NDArray('c', { 1, 3 }, nd4j::DataType::FLOAT32);
    auto x = NDArray('c', { 5, 2, 1 }, nd4j::DataType::FLOAT32);

    auto e = NDArray('c', { 5, 2, 3 }, { 2., 2., 2., 3., 3., 3., 4., 4., 4., 5., 5., 5., 6., 6., 6., 7., 7., 7., 8., 8., 8., 9., 9., 9., 10., 10., 10., 11., 11., 11. }, nd4j::DataType::FLOAT32);

    y.assign(1.0);
    x.linspace(1.0);

    nd4j::ops::add op;
    auto result = op.evaluate({ &x, &y });
    ASSERT_EQ(Status::OK(), result->status());

    auto res = *result->at(0);

    ASSERT_EQ(e, res);

    delete result;
}

///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest3) {

    auto x = NDArray('c', { 3, 5, 1 }, nd4j::DataType::FLOAT32);
    auto y = NDArray('c', { 3, 1, 4 }, nd4j::DataType::FLOAT32);
    auto z = NDArray('c', { 3, 5, 4 }, nd4j::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 3, 5, 4 }, { 10., 11., 12., 13., 20., 22., 24., 26., 30., 33., 36., 39., 40., 44., 48., 52., 50., 55., 60., 65., 84., 90., 96., 102., 98., 105., 112., 119., 112., 120., 128., 136., 126., 135., 144., 153., 140., 150., 160., 170., 198., 209., 220., 231., 216., 228., 240., 252., 234., 247., 260., 273., 252., 266., 280., 294., 270., 285., 300., 315. }, nd4j::DataType::FLOAT32);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), y, z);
    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest4) {

    auto x = NDArray('c', { 2, 3, 5, 1 }, nd4j::DataType::FLOAT32);
    auto y = NDArray('c', { 2, 3, 1, 4 }, nd4j::DataType::FLOAT32);
    auto z = NDArray('c', { 2, 3, 5, 4 }, nd4j::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 2, 3, 5, 4 }, { 10., 11., 12., 13.,20., 22., 24., 26.,30., 33., 36., 39.,40., 44., 48., 52.,50., 55., 60., 65.,84., 90., 96., 102.,98., 105., 112., 119.,112., 120., 128., 136.,126., 135., 144., 153.,140., 150., 160., 170.,198., 209., 220., 231.,216., 228., 240., 252.,234., 247., 260., 273.,252., 266., 280., 294.,270., 285., 300., 315.,352., 368., 384., 400.,374., 391., 408., 425.,396., 414., 432., 450.,418., 437., 456., 475.,440., 460., 480., 500.,546., 567., 588., 609.,572., 594., 616., 638.,598., 621., 644., 667.,624., 648., 672., 696.,650., 675., 700., 725.,780., 806., 832., 858.,810., 837., 864., 891.,840., 868., 896., 924.,870., 899., 928., 957.,900., 930., 960., 990. }, nd4j::DataType::FLOAT32);
    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), y, z);
    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest5) {

    auto x = NDArray('c', { 3, 5, 1 }, nd4j::DataType::FLOAT32);
    auto y = NDArray('c', { 3, 1, 4 }, nd4j::DataType::FLOAT32);
    auto z = NDArray('c', { 3, 5, 4 }, nd4j::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 3, 5, 4 }, { 0.1, 0.090909, 0.083333, 0.076923,0.2, 0.181818, 0.166667, 0.153846,0.3, 0.272727, 0.250000, 0.230769,0.4, 0.363636, 0.333333, 0.307692,0.5, 0.454545, 0.416667, 0.384615, 0.428571, 0.400000, 0.375000, 0.352941,  0.500000, 0.466667, 0.437500, 0.411765,  0.571429, 0.533333, 0.500000, 0.470588,  0.642857, 0.600000, 0.562500, 0.529412,  0.714286, 0.666667, 0.625000, 0.588235,  0.611111, 0.578947, 0.550000, 0.523810,  0.666667, 0.631579, 0.600000, 0.571429,  0.722222, 0.684211, 0.650000, 0.619048,   0.777778, 0.736842, 0.700000, 0.666667,  0.833333, 0.789474, 0.750000, 0.714286 }, nd4j::DataType::FLOAT32);
    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Divide(), y, z);
    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest6) {

    auto x = NDArray('c', { 2, 3, 5, 1 }, nd4j::DataType::FLOAT32);
    auto y = NDArray('c', { 2, 3, 1, 4 }, nd4j::DataType::FLOAT32);
    auto z = NDArray('c', { 2, 3, 5, 4 }, nd4j::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 2, 3, 5, 4 }, { 0.1, 0.090909, 0.083333, 0.076923,0.2, 0.181818, 0.166667, 0.153846,0.3, 0.272727, 0.250000, 0.230769,0.4, 0.363636, 0.333333, 0.307692,0.5, 0.454545, 0.416667, 0.384615,  0.428571, 0.400000, 0.375000, 0.352941,  0.500000, 0.466667, 0.437500, 0.411765,  0.571429, 0.533333, 0.500000, 0.470588,  0.642857, 0.600000, 0.562500, 0.529412,  0.714286, 0.666667, 0.625000, 0.588235,0.611111, 0.578947, 0.550000, 0.523810,0.666667, 0.631579, 0.600000, 0.571429,0.722222, 0.684211, 0.650000, 0.619048,0.777778, 0.736842, 0.700000, 0.666667,0.833333, 0.789474, 0.750000, 0.714286, 0.727273, 0.695652, 0.666667, 0.64, 0.772727, 0.739130, 0.708333, 0.68, 0.818182, 0.782609, 0.750000, 0.72, 0.863636, 0.826087, 0.791667, 0.76, 0.909091, 0.869565, 0.833333, 0.80,  0.807692, 0.777778, 0.750000, 0.724138,  0.846154, 0.814815, 0.785714, 0.758621,  0.884615, 0.851852, 0.821429, 0.793103,  0.923077, 0.888889, 0.857143, 0.827586,  0.961538, 0.925926, 0.892857, 0.862069,  0.866667, 0.838710, 0.812500, 0.787879,  0.900000, 0.870968, 0.843750, 0.818182,  0.933333, 0.903226, 0.875000, 0.848485, 0.966667, 0.935484, 0.906250, 0.878788,  1.000000, 0.967742, 0.937500, 0.909091 }, nd4j::DataType::FLOAT32);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Divide(), y, z);
    ASSERT_EQ(e, z);
}

///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest7) {

    auto x = NDArray('c', { 3, 5, 1 }, nd4j::DataType::FLOAT32);
    auto y = NDArray('c', { 3, 1, 4 }, nd4j::DataType::FLOAT32);
    auto z = NDArray('c', { 3, 5, 4 }, nd4j::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 3, 5, 4 }, { -9., -10., -11., -12.,-8., -9., -10., -11., -7., -8., -9., -10.,-6., -7., -8., -9.,-5., -6., -7., -8.,-8., -9., -10., -11.,-7., -8., -9., -10.,-6., -7., -8., -9.,-5., -6., -7., -8.,-4., -5., -6., -7.,-7., -8.000000, -9.000000, -10.00,-6.000000, -7.000000, -8.000000, -9.000,-5.000000, -6.000000, -7.000000, -8.000,-4.000000, -5.000000, -6.000000, -7.000,-3.000000, -4.000000, -5.000000, -6.000 }, nd4j::DataType::FLOAT32);
    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), y, z);
    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest8) {

    auto x = NDArray('c', { 2, 3, 5, 1 }, nd4j::DataType::FLOAT32);
    auto y = NDArray('c', { 2, 3, 1, 4 }, nd4j::DataType::FLOAT32);
    auto z = NDArray('c', { 2, 3, 5, 4 }, nd4j::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 2, 3, 5, 4 }, { -9.0, -10., -11., -12.,-8., -9., -10., -11.0,-7., -8., -9., -10.,-6., -7., -8., -9.,-5., -6., -7., -8.,-8., -9., -10., -11.,-7., -8., -9., -10.,-6., -7., -8., -9.,-5., -6., -7., -8.,-4., -5., -6., -7.,-7., -8., -9., -10.,-6., -7., -8., -9.,-5., -6., -7., -8.,-4., -5., -6., -7.,-3., -4., -5., -6.,-6., -7., -8., -9.,-5., -6., -7., -8.,-4., -5., -6., -7.,-3., -4., -5., -6.,-2., -3., -4., -5.,-5., -6., -7., -8.,-4., -5., -6., -7.,-3., -4., -5., -6.,-2., -3., -4., -5.,-1., -2., -3., -4.,-4., -5., -6., -7.,-3., -4., -5., -6.,-2., -3., -4., -5.,-1., -2., -3., -4., 0., -1., -2., -3. }, nd4j::DataType::FLOAT32);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), y, z);
    ASSERT_EQ(e, z);
}
