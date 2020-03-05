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
#include <array/NDArray.h>
#include <ops/ops.h>
#include <helpers/GradCheck.h>


using namespace sd;


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

    sd::ops::fill op;
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

    sd::ops::reshape op;
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

        sd::ops::multiply op;
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

    sd::ops::evaluate_reduction_shape op;
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

    sd::ops::evaluate_reduction_shape op;
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

    sd::ops::reduce_min op;
    op.execute({&x}, {&z}, {}, {0}, {});

    //z.printIndexedBuffer("Z");

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests14, Test_Reduce_Min_Small_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 4}, {-999.f, 0.2236f, 0.7973f, 0.0962f, 0.7231f, 0.3381f, -0.7301f, 0.9115f, -0.5094f, 0.9749f, -2.1340f, 0.6023f});
    auto z = NDArrayFactory::create<float>('c', {3});
    auto e = NDArrayFactory::create<float>('c', {3}, {-999.f, -0.7301f, -2.1340f});

    sd::ops::reduce_min op;
    op.execute({&x}, {&z}, {}, {1}, {});

    //z.printIndexedBuffer("Z");

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests14, Test_Diag_Zeros_1) {
    auto x = NDArrayFactory::create<double>('c', {2}, {1, 2});
    auto z = NDArrayFactory::create<double>('c', {2, 2}, {-119, -119, -119, -119});
    auto exp = NDArrayFactory::create<double>('c', {2, 2}, {1, 0, 0, 2});

    sd::ops::diag op;
    auto status = op.execute({&x}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(exp, z);
}

TEST_F(DeclarableOpsTests14, Test_scalar_broadcast_1) {
    auto x = NDArrayFactory::create<float>(1.0f);
    auto y = NDArrayFactory::create<float>('c', {5, 10});
    auto e = NDArrayFactory::create<float>('c', {5, 10});
    e.assign(1.0);


    sd::ops::add op;
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


    sd::ops::subtract op;
    auto result = op.evaluate({&x, &y});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests14, test_empty_fill_1) {
    auto x = NDArrayFactory::empty<int>();
    auto y = NDArrayFactory::create<int>(1);

    sd::ops::fill op;
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

    sd::ops::lstmBlockCell op;
    auto result = op.execute({&a, &b, &c, &d, &e, &f, &g, &h}, {&z0, &z1, &z2, &z3, &z4, &z5, &z6}, {1.0, -1.0}, {0}, {});
    ASSERT_EQ(Status::OK(), result);
}

TEST_F(DeclarableOpsTests14, test_empty_reduce_min_1) {

    auto e = NDArrayFactory::create<float>('c', {1, 0});
    sd::ops::reduce_min sumOp;
    auto res2 = sumOp.evaluate({&e}, {1.}, {1});
    ASSERT_EQ(res2->status(), Status::OK());
    auto out = res2->at(0);

    ASSERT_EQ(out->e<float>(0), DataTypeUtils::infOrMax<float>());
    delete res2;
}

TEST_F(DeclarableOpsTests14, test_empty_reduce_max_1) {

    auto e = NDArrayFactory::create<float>('c', {1, 0});
    sd::ops::reduce_max sumOp;
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
    sd::ops::reduce_sum sumOp;
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
    sd::ops::reduce_mean sumOp;
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

    sd::ops::strided_slice op;
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

    sd::ops::strided_slice op;
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

    sd::ops::argmax op;
    //sd::ops::reduce_max op;

    auto result = op.evaluate({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, test_empty_argmax_2) {
    auto x = NDArrayFactory::create<float>('c', {1, 0});
    auto y = NDArrayFactory::create<int>(1);

    sd::ops::argmax op;
    try {
        auto result = op.execute({&x, &y}, {&y}, {}, {}, {});
        ASSERT_TRUE(false);
    } catch (std::exception &e) {
        //
    }
}

TEST_F(DeclarableOpsTests14, test_empty_tanh_5) {
    auto x = NDArrayFactory::create<float>('c', {32, 0});

    sd::ops::tanh op;
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

    sd::ops::repeat op;
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

    sd::ops::repeat op;
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

    sd::ops::repeat op;
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

    sd::ops::repeat op;
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

    sd::ops::repeat op;
    auto result = op.evaluate({&x}, {}, {1,2,1,  1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}
/////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest) {

    auto y = NDArray('c', { 3 }, sd::DataType::FLOAT32);
    auto x = NDArray('c', { 5, 2, 1 }, sd::DataType::FLOAT32);

    auto e = NDArray('c', { 5, 2, 3 }, { 2., 2., 2., 3., 3., 3., 4., 4., 4., 5., 5., 5., 6., 6., 6., 7., 7., 7., 8., 8., 8., 9., 9., 9., 10., 10., 10., 11., 11., 11. }, sd::DataType::FLOAT32);

    y.assign(1.0);
    x.linspace(1.0);

    sd::ops::add op;
    auto result = op.evaluate({ &x, &y });
    ASSERT_EQ(Status::OK(), result->status());

    auto res = *result->at(0);

    ASSERT_EQ(e, res);

    delete result;
}
/////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest2) {

    auto y = NDArray('c', { 1, 3 }, sd::DataType::FLOAT32);
    auto x = NDArray('c', { 5, 2, 1 }, sd::DataType::FLOAT32);

    auto e = NDArray('c', { 5, 2, 3 }, { 2., 2., 2., 3., 3., 3., 4., 4., 4., 5., 5., 5., 6., 6., 6., 7., 7., 7., 8., 8., 8., 9., 9., 9., 10., 10., 10., 11., 11., 11. }, sd::DataType::FLOAT32);

    y.assign(1.0);
    x.linspace(1.0);

    sd::ops::add op;
    auto result = op.evaluate({ &x, &y });
    ASSERT_EQ(Status::OK(), result->status());

    auto res = *result->at(0);

    ASSERT_EQ(e, res);

    delete result;
}

///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest3) {

    auto x = NDArray('c', { 3, 5, 1 }, sd::DataType::FLOAT32);
    auto y = NDArray('c', { 3, 1, 4 }, sd::DataType::FLOAT32);
    auto z = NDArray('c', { 3, 5, 4 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 3, 5, 4 }, { 10., 11., 12., 13., 20., 22., 24., 26., 30., 33., 36., 39., 40., 44., 48., 52., 50., 55., 60., 65., 84., 90., 96., 102., 98., 105., 112., 119., 112., 120., 128., 136., 126., 135., 144., 153., 140., 150., 160., 170., 198., 209., 220., 231., 216., 228., 240., 252., 234., 247., 260., 273., 252., 266., 280., 294., 270., 285., 300., 315. }, sd::DataType::FLOAT32);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), y, z);
    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest4) {

    auto x = NDArray('c', { 2, 3, 5, 1 }, sd::DataType::FLOAT32);
    auto y = NDArray('c', { 2, 3, 1, 4 }, sd::DataType::FLOAT32);
    auto z = NDArray('c', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 2, 3, 5, 4 }, { 10., 11., 12., 13.,20., 22., 24., 26.,30., 33., 36., 39.,40., 44., 48., 52.,50., 55., 60., 65.,84., 90., 96., 102.,98., 105., 112., 119.,112., 120., 128., 136.,126., 135., 144., 153.,140., 150., 160., 170.,198., 209., 220., 231.,216., 228., 240., 252.,234., 247., 260., 273.,252., 266., 280., 294.,270., 285., 300., 315.,352., 368., 384., 400.,374., 391., 408., 425.,396., 414., 432., 450.,418., 437., 456., 475.,440., 460., 480., 500.,546., 567., 588., 609.,572., 594., 616., 638.,598., 621., 644., 667.,624., 648., 672., 696.,650., 675., 700., 725.,780., 806., 832., 858.,810., 837., 864., 891.,840., 868., 896., 924.,870., 899., 928., 957.,900., 930., 960., 990. }, sd::DataType::FLOAT32);
    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), y, z);
    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest5) {

    auto x = NDArray('c', { 3, 5, 1 }, sd::DataType::FLOAT32);
    auto y = NDArray('c', { 3, 1, 4 }, sd::DataType::FLOAT32);
    auto z = NDArray('c', { 3, 5, 4 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 3, 5, 4 }, { 0.1, 0.090909, 0.083333, 0.076923,0.2, 0.181818, 0.166667, 0.153846,0.3, 0.272727, 0.250000, 0.230769,0.4, 0.363636, 0.333333, 0.307692,0.5, 0.454545, 0.416667, 0.384615, 0.428571, 0.400000, 0.375000, 0.352941,  0.500000, 0.466667, 0.437500, 0.411765,  0.571429, 0.533333, 0.500000, 0.470588,  0.642857, 0.600000, 0.562500, 0.529412,  0.714286, 0.666667, 0.625000, 0.588235,  0.611111, 0.578947, 0.550000, 0.523810,  0.666667, 0.631579, 0.600000, 0.571429,  0.722222, 0.684211, 0.650000, 0.619048,   0.777778, 0.736842, 0.700000, 0.666667,  0.833333, 0.789474, 0.750000, 0.714286 }, sd::DataType::FLOAT32);
    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Divide(), y, z);
    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest6) {

    auto x = NDArray('c', { 2, 3, 5, 1 }, sd::DataType::FLOAT32);
    auto y = NDArray('c', { 2, 3, 1, 4 }, sd::DataType::FLOAT32);
    auto z = NDArray('c', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 2, 3, 5, 4 }, { 0.1, 0.090909, 0.083333, 0.076923,0.2, 0.181818, 0.166667, 0.153846,0.3, 0.272727, 0.250000, 0.230769,0.4, 0.363636, 0.333333, 0.307692,0.5, 0.454545, 0.416667, 0.384615,  0.428571, 0.400000, 0.375000, 0.352941,  0.500000, 0.466667, 0.437500, 0.411765,  0.571429, 0.533333, 0.500000, 0.470588,  0.642857, 0.600000, 0.562500, 0.529412,  0.714286, 0.666667, 0.625000, 0.588235,0.611111, 0.578947, 0.550000, 0.523810,0.666667, 0.631579, 0.600000, 0.571429,0.722222, 0.684211, 0.650000, 0.619048,0.777778, 0.736842, 0.700000, 0.666667,0.833333, 0.789474, 0.750000, 0.714286, 0.727273, 0.695652, 0.666667, 0.64, 0.772727, 0.739130, 0.708333, 0.68, 0.818182, 0.782609, 0.750000, 0.72, 0.863636, 0.826087, 0.791667, 0.76, 0.909091, 0.869565, 0.833333, 0.80,  0.807692, 0.777778, 0.750000, 0.724138,  0.846154, 0.814815, 0.785714, 0.758621,  0.884615, 0.851852, 0.821429, 0.793103,  0.923077, 0.888889, 0.857143, 0.827586,  0.961538, 0.925926, 0.892857, 0.862069,  0.866667, 0.838710, 0.812500, 0.787879,  0.900000, 0.870968, 0.843750, 0.818182,  0.933333, 0.903226, 0.875000, 0.848485, 0.966667, 0.935484, 0.906250, 0.878788,  1.000000, 0.967742, 0.937500, 0.909091 }, sd::DataType::FLOAT32);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Divide(), y, z);
    ASSERT_EQ(e, z);
}

///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest7) {

    auto x = NDArray('c', { 3, 5, 1 }, sd::DataType::FLOAT32);
    auto y = NDArray('c', { 3, 1, 4 }, sd::DataType::FLOAT32);
    auto z = NDArray('c', { 3, 5, 4 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 3, 5, 4 }, { -9., -10., -11., -12.,-8., -9., -10., -11., -7., -8., -9., -10.,-6., -7., -8., -9.,-5., -6., -7., -8.,-8., -9., -10., -11.,-7., -8., -9., -10.,-6., -7., -8., -9.,-5., -6., -7., -8.,-4., -5., -6., -7.,-7., -8.000000, -9.000000, -10.00,-6.000000, -7.000000, -8.000000, -9.000,-5.000000, -6.000000, -7.000000, -8.000,-4.000000, -5.000000, -6.000000, -7.000,-3.000000, -4.000000, -5.000000, -6.000 }, sd::DataType::FLOAT32);
    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), y, z);
    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_SpecialCaseTest8) {

    auto x = NDArray('c', { 2, 3, 5, 1 }, sd::DataType::FLOAT32);
    auto y = NDArray('c', { 2, 3, 1, 4 }, sd::DataType::FLOAT32);
    auto z = NDArray('c', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 2, 3, 5, 4 }, { -9.0, -10., -11., -12.,-8., -9., -10., -11.0,-7., -8., -9., -10.,-6., -7., -8., -9.,-5., -6., -7., -8.,-8., -9., -10., -11.,-7., -8., -9., -10.,-6., -7., -8., -9.,-5., -6., -7., -8.,-4., -5., -6., -7.,-7., -8., -9., -10.,-6., -7., -8., -9.,-5., -6., -7., -8.,-4., -5., -6., -7.,-3., -4., -5., -6.,-6., -7., -8., -9.,-5., -6., -7., -8.,-4., -5., -6., -7.,-3., -4., -5., -6.,-2., -3., -4., -5.,-5., -6., -7., -8.,-4., -5., -6., -7.,-3., -4., -5., -6.,-2., -3., -4., -5.,-1., -2., -3., -4.,-4., -5., -6., -7.,-3., -4., -5., -6.,-2., -3., -4., -5.,-1., -2., -3., -4., 0., -1., -2., -3. }, sd::DataType::FLOAT32);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), y, z);
    ASSERT_EQ(e, z);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test1) {

    auto x  = NDArrayFactory::create<double>('c', {3, 4});
    auto y  = NDArrayFactory::create<double>('c', {4, 3});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {35.,  79., 123., 40.,  92., 144., 45., 105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test2) {

    auto x  = NDArrayFactory::create<double>('c', {3, 4});
    auto y  = NDArrayFactory::create<double>('f', {4, 3});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test3) {

    auto x  = NDArrayFactory::create<double>('f', {3, 4});
    auto y  = NDArrayFactory::create<double>('c', {4, 3});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test4) {

    auto x = NDArrayFactory::create<double> ('f', {3, 4});
    auto y  = NDArrayFactory::create<double>('f', {4, 3});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test5) {

    auto x  = NDArrayFactory::create<double>('c', {4, 3});
    auto y  = NDArrayFactory::create<double>('c', {4, 3});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {83.,  94., 105., 94., 107., 120., 105., 120., 135.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test6) {

    auto x  = NDArrayFactory::create<double>('c', {4, 3});
    auto y  = NDArrayFactory::create<double>('f', {3, 4});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {35.,  40.,  45., 79.,  92., 105., 123., 144., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test7) {

    auto x  = NDArrayFactory::create<double>('c', {5,  3,4});
    auto y  = NDArrayFactory::create<double>('f', {5,  3,4});
    auto exp = NDArrayFactory::create<double>('f',{5,  3,3}, {3. ,  84.6, 281.4, 593.4, 1020.6, 7. , 107.8, 323.8, 655. , 1101.4,11. , 131. , 366.2, 716.6, 1182.2,
                                        7. , 107.8, 323.8, 655. , 1101.4,17.4, 137.4, 372.6, 723. , 1188.6,27.8, 167. , 421.4, 791. , 1275.8,
                                       11. , 131. , 366.2, 716.6, 1182.2,27.8, 167. , 421.4, 791. , 1275.8,44.6, 203. , 476.6, 865.4, 1369.4,});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test8) {

    auto x  = NDArrayFactory::create<double>('c', {2,5,  3,4});
    auto y  = NDArrayFactory::create<double>('f', {2,5,  3,4});
    auto exp = NDArrayFactory::create<double>('f',{2,5,  3,3}, {3. , 1563. ,  84.6, 2220.6, 281.4, 2993.4, 593.4, 3881.4,1020.6, 4884.6,   7. , 1663. , 107.8, 2339.8, 323.8, 3131.8, 655. , 4039. ,1101.4, 5061.4,
                                          11. , 1763. , 131. , 2459. , 366.2, 3270.2, 716.6, 4196.6,1182.2, 5238.2,   7. , 1663. , 107.8, 2339.8, 323.8, 3131.8, 655. , 4039. ,1101.4, 5061.4,
                                          17.4, 1769.4, 137.4, 2465.4, 372.6, 3276.6, 723. , 4203. ,1188.6, 5244.6,  27.8, 1875.8, 167. , 2591. , 421.4, 3421.4, 791. , 4367. ,1275.8, 5427.8,
                                          11. , 1763. , 131. , 2459. , 366.2, 3270.2, 716.6, 4196.6,1182.2, 5238.2,  27.8, 1875.8, 167. , 2591. , 421.4, 3421.4, 791. , 4367. ,1275.8, 5427.8,
                                          44.6, 1988.6, 203. , 2723. , 476.6, 3572.6, 865.4, 4537.4,1369.4, 5617.4});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test9) {

    auto x  = NDArrayFactory::create<double>('c', {2,5,  4,3});
    auto y  = NDArrayFactory::create<double>('f', {2,5,  3,4});
    auto exp = NDArrayFactory::create<double>('f',{2,5,  3,3}, {7. , 1639. , 103. , 2311. , 314.2, 3098.2, 640.6, 4000.6,1082.2, 5018.2,   8. , 1664. , 108.8, 2340.8, 324.8, 3132.8, 656. , 4040. ,1102.4, 5062.4,
                                          9. , 1689. , 114.6, 2370.6, 335.4, 3167.4, 671.4, 4079.4,1122.6, 5106.6,  15.8, 1743.8, 131. , 2435. , 361.4, 3241.4, 707. , 4163. ,1167.8, 5199.8,
                                          18.4, 1770.4, 138.4, 2466.4, 373.6, 3277.6, 724. , 4204. ,1189.6, 5245.6,  21. , 1797. , 145.8, 2497.8, 385.8, 3313.8, 741. , 4245. ,1211.4, 5291.4,
                                          24.6, 1848.6, 159. , 2559. , 408.6, 3384.6, 773.4, 4325.4,1253.4, 5381.4,  28.8, 1876.8, 168. , 2592. , 422.4, 3422.4, 792. , 4368. ,1276.8, 5428.8,
                                          33. , 1905. , 177. , 2625. , 436.2, 3460.2, 810.6, 4410.6,1300.2, 5476.2});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

TEST_F(DeclarableOpsTests14, matmul_test10) {

    auto x = NDArrayFactory::create_<float>('c', {3, 5});
    x->linspace(1);

    auto y = NDArrayFactory::create_<float>('c', {5, 3});
    y->linspace(1);

    float _expB[]{135.0f, 310.0f, 485.0f, 150.0f, 350.0f, 550.0f, 165.0f, 390.0f, 615.0f};
    Nd4jLong _expS[] {2, 3, 3, 1, 3, 0, 1, 102}; // expected shape
    ArrayOptions::setDataType(_expS, sd::DataType::FLOAT32);
    NDArray exp(_expB, _expS);

    auto variableSpace = new VariableSpace();
    variableSpace->putVariable(-1, x);
    variableSpace->putVariable(-2, y);
    variableSpace->putVariable(1, new Variable());

    auto block = new Context(1, variableSpace, false);
    block->fillInputs({-1, -2});

    sd::ops::matmul op;

    Nd4jStatus status = op.execute(block);
    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(variableSpace->hasVariable(1));

    auto result = variableSpace->getVariable(1)->getNDArray();

    ASSERT_TRUE(result->equalsTo(&exp));

    delete block;
    delete variableSpace;
}

TEST_F(DeclarableOpsTests14, matmul_test11) {
    auto A = NDArrayFactory::create<float>('c', {3, 3});
    auto B = NDArrayFactory::create<float>('c', {3, 1});
    auto exp = NDArrayFactory::create<float>('c', {3, 1}, {14.00f,  32.00f,  50.00f});

    A.linspace(1);
    B.linspace(1);

    sd::ops::matmul op;

    auto result = op.evaluate({&A, &B}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests14, matmul_test12) {
    auto x= NDArrayFactory::create<double>('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12});
    auto y= NDArrayFactory::create<double>('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12});
    auto exp= NDArrayFactory::create<double>('f', {4, 4}, {38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0, 128.0, 152.0, 176.0, 200.0, 173.0, 206.0, 239.0, 272.0});

    sd::ops::matmul op;
    auto result = op.evaluate({&x, &y}, {}, {1, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


    delete result;
}


TEST_F(DeclarableOpsTests14, matmul_test13) {
    auto x= NDArrayFactory::create<double>('c', {1, 3}, {1, 2, 3});
    auto y= NDArrayFactory::create<double>('c', {1, 4}, {1, 2, 3, 4});
    auto exp= NDArrayFactory::create<double>('f', {3, 4}, {1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0});

    sd::ops::matmul op;
    auto result = op.evaluate({&x, &y}, {}, {1, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests14, matmul_test14) {
    auto x= NDArrayFactory::create<double>('c', {3, 1}, {1, 2, 3});
    auto y= NDArrayFactory::create<double>('c', {4, 1}, {1, 2, 3, 4});
    auto exp= NDArrayFactory::create<double>('f', {3, 4}, {1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0});

    sd::ops::matmul op;
    auto result = op.evaluate({&x, &y}, {}, {0, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests14, matmul_test15) {
    auto x= NDArrayFactory::create<double>('c', {3, 1}, {1, 2, 3});
    auto y= NDArrayFactory::create<double>('c', {1, 4}, {1, 2, 3, 4});
    auto exp= NDArrayFactory::create<double>('f', {3, 4}, {1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0});

    sd::ops::matmul op;
    auto result = op.evaluate({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests14, matmul_test16) {
    auto x= NDArrayFactory::create<double>('c', {4, 1}, {1, 2, 3, 4});
    auto y= NDArrayFactory::create<double>('c', {1, 4}, {1, 2, 3, 4});
    auto exp= NDArrayFactory::create<double>('f', {4, 4}, {1,2, 3, 4,2,4, 6, 8,3,6, 9,12,4,8,12,16});

    sd::ops::matmul op;
    auto result = op.evaluate({&x, &y});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests14, matmul_test17) {
    auto x = NDArrayFactory::create<double>('c', {1, 2}, {2.0f, 2.0f});
    auto y = NDArrayFactory::create<double>('c', {2, 1}, {2.0f, 2.0f});
    auto exp = NDArrayFactory::create<double>('c', {1, 1}, {8.0f});

    sd::ops::matmul op;
    auto result = op.evaluate({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(exp, *result->at(0));

    delete result;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test18) {

    auto x  = NDArrayFactory::create<double>('c', {1, 4, 3});
    auto y  = NDArrayFactory::create<double>('f', {1, 3, 4});
    auto exp = NDArrayFactory::create<double>('f', {1, 3, 3}, {35.,  40.,  45., 79.,  92., 105., 123., 144., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test19) {

    auto x  = NDArrayFactory::create<double>('c', {4, 1});
    auto y  = NDArrayFactory::create<double>('f', {1, 4});
    auto exp = NDArrayFactory::create<double>('f', {1, 1}, {15});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1, 1});
    ASSERT_EQ(Status::OK(), results->status());

    auto z = results->at(0);
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test20) {

    auto x  = NDArrayFactory::create<double>('c', {1, 4, 1});
    auto y  = NDArrayFactory::create<double>('f', {1, 1, 4});
    auto exp = NDArrayFactory::create<double>('f', {1, 1, 1}, {15});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1, 1});

    ASSERT_EQ(Status::OK(), results->status());
    auto z = results->at(0);
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test21) {

    auto x  = NDArrayFactory::create<double>('c', {2, 3});
    auto y  = NDArrayFactory::create<double>('c', {3, 5});
    auto exp = NDArrayFactory::create<double>('f', {5, 2}, {23. , 26. , 29. , 32. , 35., 50. , 57.5, 65. , 72.5, 80.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {0, 0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test22) {

    auto x  = NDArrayFactory::create<double>('c', {3, 2});
    auto y  = NDArrayFactory::create<double>('c', {3, 5});
    auto exp = NDArrayFactory::create<double>('f', {5, 2}, {37. , 41.5, 46. , 50.5, 55., 46. , 52. , 58. , 64. , 70.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1, 0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test23) {

    auto x  = NDArrayFactory::create<double>('c', {3, 2});
    auto y  = NDArrayFactory::create<double>('c', {3, 5});
    auto exp = NDArrayFactory::create<double>('f', {5, 2}, {37. , 41.5, 46. , 50.5, 55., 46. , 52. , 58. , 64. , 70.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1, 0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test24) {

    auto x  = NDArrayFactory::create<double>('c', {2,2,  3,5});
    auto y  = NDArrayFactory::create<double>('c', {2,2,  4,3});
    auto exp = NDArrayFactory::create<double>('f',{2,2,  4,5}, {4.6, 281.8, 89.2, 582.4, 10. , 314.2,108.1, 628.3, 15.4, 346.6,127. , 674.2, 20.8, 379. ,145.9, 720.1,  5.2, 289.6, 93.4, 593.8,
                                          11.5, 322.9,113.2, 640.6, 17.8, 356.2,133. , 687.4, 24.1, 389.5,152.8, 734.2,  5.8, 297.4, 97.6, 605.2, 13. , 331.6,118.3, 652.9,
                                          20.2, 365.8,139. , 700.6, 27.4, 400. ,159.7, 748.3,  6.4, 305.2,101.8, 616.6, 14.5, 340.3,123.4, 665.2, 22.6, 375.4,145. , 713.8,
                                          30.7, 410.5,166.6, 762.4,  7. , 313. ,106. , 628. , 16. , 349. ,128.5, 677.5, 25. , 385. ,151. , 727. , 34. , 421. ,173.5, 776.5});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1, 1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test25) {

    auto x  = NDArrayFactory::create<double>('f', {4, 3});
    auto y  = NDArrayFactory::create<double>('c', {4});
    auto exp = NDArrayFactory::create<double>('f',{3}, {7., 8., 9.});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1, 0});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test26) {

    auto x  = NDArrayFactory::create<double>('f', {3});
    auto y  = NDArrayFactory::create<double>('c', {4, 3});
    auto exp = NDArrayFactory::create<double>('f',{4}, {1.4, 3.2, 5., 6.8});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test27) {

    auto x  = NDArrayFactory::create<double>('f', {1, 1});
    auto y  = NDArrayFactory::create<double>('c', {1, 1});
    auto exp = NDArrayFactory::create<double>('f',{1, 1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test28) {

    auto x  = NDArrayFactory::create<double>('f', {1, 1});
    auto y  = NDArrayFactory::create<double>('c', {1, 1});
    auto exp = NDArrayFactory::create<double>('f',{1, 1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1,1,1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test29) {

    auto x  = NDArrayFactory::create<double>('f', {1});
    auto y  = NDArrayFactory::create<double>('c', {1, 1});
    auto exp = NDArrayFactory::create<double>('f',{1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test30) {

    auto x  = NDArrayFactory::create<double>('f', {1,1});
    auto y  = NDArrayFactory::create<double>('c', {1});
    auto exp = NDArrayFactory::create<double>('f',{1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test31) {

    auto x  = NDArrayFactory::create<double>('f', {4});
    auto y  = NDArrayFactory::create<double>('c', {4});
    auto exp = NDArrayFactory::create<double>(3.);

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test32) {

    auto x  = NDArrayFactory::create<double>('f', {1}, {2.});
    auto y  = NDArrayFactory::create<double>('c', {1}, {3.});
    auto exp = NDArrayFactory::create<double>(6.);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}
/////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test33) {
    auto x = NDArrayFactory::create<double>('c', {4, 3});
    auto y = NDArrayFactory::create<double>('c', {4, 1});
    auto exp = NDArrayFactory::create<double>('c',{ 3, 1}, {70, 80, 90});

    x.linspace(1);
    y.linspace(1);

    sd::ops::matmul op;
    auto result = op.evaluate({&x, &y}, {}, {1, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}
//////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test34) {
    auto a = NDArrayFactory::create<double>('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto b = NDArrayFactory::create<double>('c', {4}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {3}, {30, 70, 110});

    sd::ops::matmul op;
    auto result = op.evaluate({&a, &b});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}
/////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test35) {
    auto a = NDArrayFactory::create<double>('c', {4}, {1, 2, 3, 4});
    auto b = NDArrayFactory::create<double>('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {3}, {70, 80, 90});

    sd::ops::matmul op;
    auto result = op.evaluate({&a, &b});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}
////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test36) {
    auto a = NDArrayFactory::create<double>('c', {1, 4}, {1, 2, 3, 4});
    auto b = NDArrayFactory::create<double>('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {1, 3}, {70, 80, 90});

    sd::ops::matmul op;
    auto result = op.evaluate({&a, &b});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, matmul_test37) {

    NDArray a('c', {32, 12, 128, 64},  sd::DataType::FLOAT32);
    NDArray b('c', {32, 12, 128, 64}, sd::DataType::FLOAT32);
    NDArray c('c', {32,12,128,128}, sd::DataType::FLOAT32);
    NDArray cExp('c', {32,12,128,128}, sd::DataType::FLOAT32);

    a = 1;
    b = 1;
    cExp = 64;      //Each entry in output c is sum of 64 (1.0 x 1.0) multiplications

    sd::ops::matmul op;
    auto status = op.execute({&a, &b}, {&c}, {}, {0,1});

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(cExp.isSameShape(c));
    ASSERT_TRUE(cExp.equalsTo(c));
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_3D_1) {

    // x[4, 12, 128] * y[4, 128] = z[4, 12, 128]

    auto x = NDArray('c', { 2, 3, 5 }, sd::DataType::FLOAT32);
    auto y = NDArray('c', { 2, 5 }, sd::DataType::FLOAT32);
    auto z = NDArray('c', { 2, 3, 5 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 2, 3, 5 }, { 10.000000, 22.000000, 36.000000, 52.000000, 70.000000, 60.000000, 77.000000, 96.000000, 117.000000, 140.000000, 110.000000, 132.000000, 156.000000, 182.000000, 210.000000, 240.000000, 272.000000, 306.000000, 342.000000, 380.000000, 315.000000, 352.000000, 391.000000, 432.000000, 475.000000, 390.000000, 432.000000, 476.000000, 522.000000, 570.000000 }, sd::DataType::FLOAT32);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyBroadcast(sd::broadcast::Multiply, { 0,2 }, y, z);
    //z.printBuffer();
    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_3D_2) {

    auto x = NDArray('f', { 2, 3, 5 }, sd::DataType::FLOAT32);
    auto y = NDArray('f', { 2, 5 }, sd::DataType::FLOAT32);
    auto z = NDArray('f', { 2, 3, 5 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto eC = NDArray('c', { 2, 3, 5 }, { 0.100000, 0.181818, 0.250000, 0.307692, 0.357143, 0.600000, 0.636364, 0.666667, 0.692308, 0.714286, 1.100000, 1.090909, 1.083333, 1.076923, 1.071429, 1.066667, 1.062500, 1.058824, 1.055556, 1.052632, 1.400000, 1.375000, 1.352941, 1.333333, 1.315789, 1.733333, 1.687500, 1.647059, 1.611111, 1.578947 }, sd::DataType::FLOAT32);

    auto e = NDArray('f', { 2, 3, 5 }, sd::DataType::FLOAT32);

    e.assign(eC);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyBroadcast(sd::broadcast::Divide, { 0,2 }, y, z);

    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_4D_1) {

    auto x = NDArray('c', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    auto y = NDArray('c', { 2, 5, 4 }, sd::DataType::FLOAT32);
    auto z = NDArray('c', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 2, 3, 5, 4 }, { 10.000000, 22.000000, 36.000000, 52.000000, 70.000000, 90.000000, 112.000000, 136.000000, 162.000000, 190.000000, 220.000000, 252.000000, 286.000000, 322.000000, 360.000000, 400.000000, 442.000000, 486.000000, 532.000000, 580.000000, 210.000000, 242.000000, 276.000000, 312.000000, 350.000000, 390.000000, 432.000000, 476.000000, 522.000000, 570.000000, 620.000000, 672.000000, 726.000000, 782.000000, 840.000000, 900.000000, 962.000000, 1026.000000, 1092.000000, 1160.000000, 410.000000, 462.000000, 516.000000, 572.000000, 630.000000, 690.000000, 752.000000, 816.000000, 882.000000, 950.000000, 1020.000000, 1092.000000, 1166.000000, 1242.000000, 1320.000000, 1400.000000, 1482.000000, 1566.000000, 1652.000000, 1740.000000, 1830.000000, 1922.000000, 2016.000000, 2112.000000, 2210.000000, 2310.000000, 2412.000000, 2516.000000, 2622.000000, 2730.000000, 2840.000000, 2952.000000, 3066.000000, 3182.000000, 3300.000000, 3420.000000, 3542.000000, 3666.000000, 3792.000000, 3920.000000, 2430.000000, 2542.000000, 2656.000000, 2772.000000, 2890.000000, 3010.000000, 3132.000000, 3256.000000, 3382.000000, 3510.000000, 3640.000000, 3772.000000, 3906.000000, 4042.000000, 4180.000000, 4320.000000, 4462.000000, 4606.000000, 4752.000000, 4900.000000, 3030.000000, 3162.000000, 3296.000000, 3432.000000, 3570.000000, 3710.000000, 3852.000000, 3996.000000, 4142.000000, 4290.000000, 4440.000000, 4592.000000, 4746.000000, 4902.000000, 5060.000000, 5220.000000, 5382.000000, 5546.000000, 5712.000000, 5880.000000 }, sd::DataType::FLOAT32);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyBroadcast(sd::broadcast::Multiply, { 0,2,3 }, y, z);

    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_4D_2) {

    auto x = NDArray('f', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    auto y = NDArray('f', { 2, 5, 4 }, sd::DataType::FLOAT32);
    auto z = NDArray('f', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto eC = NDArray('c', { 2, 3, 5, 4 }, { 0.100000,0.181818,0.250000,0.307692,0.357143,0.400000,0.437500,0.470588,0.500000,0.526316,0.550000,0.571429, 0.590909,0.608696,0.625000,0.640000, 0.653846,0.666667,0.678571,0.689655, 2.100000,2.000000,1.916667, 1.846154, 1.785714, 1.733333,1.687500, 1.647059,1.611111, 1.578947,1.550000, 1.523810,1.500000, 1.478261,1.458333, 1.440000,1.423077, 1.407407,1.392857, 1.379310,4.100000, 3.818182,3.583333, 3.384615, 3.214286, 3.066667,2.937500, 2.823529,2.722222, 2.631579,2.550000, 2.476191,2.409091, 2.347826,2.291667, 2.240000,2.192308, 2.148148,2.107143, 2.068965,2.033333, 2.000000,1.968750, 1.939394,1.911765, 1.885714,1.861111, 1.837838,1.815789, 1.794872,1.775000, 1.756098,1.738095, 1.720930,1.704545, 1.688889,1.673913, 1.659575,1.645833,1.632653,2.700000,2.645161,2.593750,2.545455,2.500000,2.457143,2.416667,2.378378,2.342105,2.307692,2.275000,2.243902,2.214286,2.186047,2.159091,2.133333,2.108696,2.085106,2.062500,2.040816,3.366667,3.290323,3.218750,3.151515,3.088235,3.028571,2.972222,2.918919,2.868421,2.820513,2.775000,2.731707,2.690476,2.651163,2.613636,2.577778,2.543478,2.510638,2.479167,2.448980 }, sd::DataType::FLOAT32);

    auto e = NDArray('f', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);

    e.assign(eC);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyBroadcast(sd::broadcast::Divide, { 0,2,3 }, y, z);

    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_4D_3) {

    auto x = NDArray('f', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    auto y = NDArray('f', { 2, 5 }, sd::DataType::FLOAT32);
    auto z = NDArray('f', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto eC = NDArray('c', { 2, 3, 5, 4 }, { 0.100000, 0.200000, 0.300000, 0.400000, 0.454545, 0.545455, 0.636364, 0.727273, 0.750000, 0.833333, 0.916667, 1.000000, 1.000000, 1.076923, 1.153846, 1.230769, 1.214286, 1.285714, 1.357143, 1.428571, 2.100000, 2.200000, 2.300000, 2.400000, 2.272727, 2.363636, 2.454545, 2.545455, 2.416667, 2.500000, 2.583333, 2.666667, 2.538461, 2.615385, 2.692308, 2.769231, 2.642857, 2.714286, 2.785714, 2.857143, 4.100000, 4.200000, 4.300000, 4.400000, 4.090909, 4.181818, 4.272727, 4.363636, 4.083333, 4.166667, 4.250000, 4.333333, 4.076923, 4.153846, 4.230769, 4.307693, 4.071429, 4.142857, 4.214286, 4.285714, 4.066667, 4.133333, 4.200000, 4.266667, 4.062500, 4.125000, 4.187500, 4.250000, 4.058824, 4.117647, 4.176471, 4.235294, 4.055555, 4.111111, 4.166667, 4.222222, 4.052631, 4.105263, 4.157895, 4.210526, 5.400000, 5.466667, 5.533333, 5.600000, 5.312500, 5.375000, 5.437500, 5.500000, 5.235294, 5.294117, 5.352941, 5.411765, 5.166667, 5.222222, 5.277778, 5.333333, 5.105263, 5.157895, 5.210526, 5.263158, 6.733333, 6.800000, 6.866667, 6.933333, 6.562500, 6.625000, 6.687500, 6.750000, 6.411765, 6.470588, 6.529412, 6.588235, 6.277778, 6.333333, 6.388889, 6.444445, 6.157895, 6.210526, 6.263158, 6.315790 }, sd::DataType::FLOAT32);

    auto e = NDArray('f', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);

    e.assign(eC);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyBroadcast(sd::broadcast::Divide, { 0,2 }, y, z);

    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_4D_4) {

    // x[4, 12, 128, 128] * y[4, 1, 128, 1] = z[4, 12, 128, 128]

    auto x = NDArray('f', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    auto y = NDArray('f', { 2, 1, 5, 1 }, sd::DataType::FLOAT32);
    auto z = NDArray('f', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto eC = NDArray('c', { 2, 3, 5, 4 }, { 0.100000, 0.200000, 0.300000, 0.400000, 0.454545, 0.545455, 0.636364, 0.727273, 0.750000, 0.833333, 0.916667, 1.000000, 1.000000, 1.076923, 1.153846, 1.230769, 1.214286, 1.285714, 1.357143, 1.428571, 2.100000, 2.200000, 2.300000, 2.400000, 2.272727, 2.363636, 2.454545, 2.545455, 2.416667, 2.500000, 2.583333, 2.666667, 2.538461, 2.615385, 2.692308, 2.769231, 2.642857, 2.714286, 2.785714, 2.857143, 4.100000, 4.200000, 4.300000, 4.400000, 4.090909, 4.181818, 4.272727, 4.363636, 4.083333, 4.166667, 4.250000, 4.333333, 4.076923, 4.153846, 4.230769, 4.307693, 4.071429, 4.142857, 4.214286, 4.285714, 4.066667, 4.133333, 4.200000, 4.266667, 4.062500, 4.125000, 4.187500, 4.250000, 4.058824, 4.117647, 4.176471, 4.235294, 4.055555, 4.111111, 4.166667, 4.222222, 4.052631, 4.105263, 4.157895, 4.210526, 5.400000, 5.466667, 5.533333, 5.600000, 5.312500, 5.375000, 5.437500, 5.500000, 5.235294, 5.294117, 5.352941, 5.411765, 5.166667, 5.222222, 5.277778, 5.333333, 5.105263, 5.157895, 5.210526, 5.263158, 6.733333, 6.800000, 6.866667, 6.933333, 6.562500, 6.625000, 6.687500, 6.750000, 6.411765, 6.470588, 6.529412, 6.588235, 6.277778, 6.333333, 6.388889, 6.444445, 6.157895, 6.210526, 6.263158, 6.315790 }, sd::DataType::FLOAT32);

    auto e = NDArray('f', { 2, 3, 5, 4 }, sd::DataType::FLOAT32);
    e.assign(eC);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Divide(), y, z);

    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_5D_1) {
    // x[4, 12, 128, 128, 128] * y[4, 1, 128, 128, 128] = z[4, 12, 128, 128, 128]
    auto x = NDArray('c', { 2, 3, 5, 4, 3 }, sd::DataType::FLOAT32);
    auto y = NDArray('c', { 2, 1, 5, 4, 3 }, sd::DataType::FLOAT32);
    auto z = NDArray('c', { 2, 3, 5, 4, 3 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto e = NDArray('c', { 2, 3, 5, 4, 3 }, { 10.000000, 22.000000, 36.000000, 52.000000, 70.000000, 90.000000, 112.000000, 136.000000, 162.000000, 190.000000, 220.000000, 252.000000, 286.000000, 322.000000, 360.000000, 400.000000, 442.000000, 486.000000, 532.000000, 580.000000, 630.000000, 682.000000, 736.000000, 792.000000, 850.000000, 910.000000, 972.000000, 1036.000000, 1102.000000, 1170.000000, 1240.000000, 1312.000000, 1386.000000, 1462.000000, 1540.000000, 1620.000000, 1702.000000, 1786.000000, 1872.000000, 1960.000000, 2050.000000, 2142.000000, 2236.000000, 2332.000000, 2430.000000, 2530.000000, 2632.000000, 2736.000000, 2842.000000, 2950.000000, 3060.000000, 3172.000000, 3286.000000, 3402.000000, 3520.000000, 3640.000000, 3762.000000, 3886.000000, 4012.000000, 4140.000000, 610.000000, 682.000000, 756.000000, 832.000000, 910.000000, 990.000000, 1072.000000, 1156.000000, 1242.000000, 1330.000000, 1420.000000, 1512.000000, 1606.000000, 1702.000000, 1800.000000, 1900.000000, 2002.000000, 2106.000000, 2212.000000, 2320.000000, 2430.000000, 2542.000000, 2656.000000, 2772.000000, 2890.000000, 3010.000000, 3132.000000, 3256.000000, 3382.000000, 3510.000000, 3640.000000, 3772.000000, 3906.000000, 4042.000000, 4180.000000, 4320.000000, 4462.000000, 4606.000000, 4752.000000, 4900.000000, 5050.000000, 5202.000000, 5356.000000, 5512.000000, 5670.000000, 5830.000000, 5992.000000, 6156.000000, 6322.000000, 6490.000000, 6660.000000, 6832.000000, 7006.000000, 7182.000000, 7360.000000, 7540.000000, 7722.000000, 7906.000000, 8092.000000, 8280.000000, 1210.000000, 1342.000000, 1476.000000, 1612.000000, 1750.000000, 1890.000000, 2032.000000, 2176.000000, 2322.000000, 2470.000000, 2620.000000, 2772.000000, 2926.000000, 3082.000000, 3240.000000, 3400.000000, 3562.000000, 3726.000000, 3892.000000, 4060.000000, 4230.000000, 4402.000000, 4576.000000, 4752.000000, 4930.000000, 5110.000000, 5292.000000, 5476.000000, 5662.000000, 5850.000000, 6040.000000, 6232.000000, 6426.000000, 6622.000000, 6820.000000, 7020.000000, 7222.000000, 7426.000000, 7632.000000, 7840.000000, 8050.000000, 8262.000000, 8476.000000, 8692.000000, 8910.000000, 9130.000000, 9352.000000, 9576.000000, 9802.000000, 10030.000000, 10260.000000, 10492.000000, 10726.000000, 10962.000000, 11200.000000, 11440.000000, 11682.000000, 11926.000000, 12172.000000, 12420.000000, 12670.000000, 12922.000000, 13176.000000, 13432.000000, 13690.000000, 13950.000000, 14212.000000, 14476.000000, 14742.000000, 15010.000000, 15280.000000, 15552.000000, 15826.000000, 16102.000000, 16380.000000, 16660.000000, 16942.000000, 17226.000000, 17512.000000, 17800.000000, 18090.000000, 18382.000000, 18676.000000, 18972.000000, 19270.000000, 19570.000000, 19872.000000, 20176.000000, 20482.000000, 20790.000000, 21100.000000, 21412.000000, 21726.000000, 22042.000000, 22360.000000, 22680.000000, 23002.000000, 23326.000000, 23652.000000, 23980.000000, 24310.000000, 24642.000000, 24976.000000, 25312.000000, 25650.000000, 25990.000000, 26332.000000, 26676.000000, 27022.000000, 27370.000000, 27720.000000, 28072.000000, 28426.000000, 28782.000000, 29140.000000, 29500.000000, 29862.000000, 30226.000000, 30592.000000, 30960.000000, 16870.000000, 17182.000000, 17496.000000, 17812.000000, 18130.000000, 18450.000000, 18772.000000, 19096.000000, 19422.000000, 19750.000000, 20080.000000, 20412.000000, 20746.000000, 21082.000000, 21420.000000, 21760.000000, 22102.000000, 22446.000000, 22792.000000, 23140.000000, 23490.000000, 23842.000000, 24196.000000, 24552.000000, 24910.000000, 25270.000000, 25632.000000, 25996.000000, 26362.000000, 26730.000000, 27100.000000, 27472.000000, 27846.000000, 28222.000000, 28600.000000, 28980.000000, 29362.000000, 29746.000000, 30132.000000, 30520.000000, 30910.000000, 31302.000000, 31696.000000, 32092.000000, 32490.000000, 32890.000000, 33292.000000, 33696.000000, 34102.000000, 34510.000000, 34920.000000, 35332.000000, 35746.000000, 36162.000000, 36580.000000, 37000.000000, 37422.000000, 37846.000000, 38272.000000, 38700.000000, 21070.000000, 21442.000000, 21816.000000, 22192.000000, 22570.000000, 22950.000000, 23332.000000, 23716.000000, 24102.000000, 24490.000000, 24880.000000, 25272.000000, 25666.000000, 26062.000000, 26460.000000, 26860.000000, 27262.000000, 27666.000000, 28072.000000, 28480.000000, 28890.000000, 29302.000000, 29716.000000, 30132.000000, 30550.000000, 30970.000000, 31392.000000, 31816.000000, 32242.000000, 32670.000000, 33100.000000, 33532.000000, 33966.000000, 34402.000000, 34840.000000, 35280.000000, 35722.000000, 36166.000000, 36612.000000, 37060.000000, 37510.000000, 37962.000000, 38416.000000, 38872.000000, 39330.000000, 39790.000000, 40252.000000, 40716.000000, 41182.000000, 41650.000000, 42120.000000, 42592.000000, 43066.000000, 43542.000000, 44020.000000, 44500.000000, 44982.000000, 45466.000000, 45952.000000, 46440.000000 }, sd::DataType::FLOAT32);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), y, z);
    // z.printBuffer();
    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_5D_2) {

    auto x = NDArray('f', { 2, 3, 5, 4, 3 }, sd::DataType::FLOAT32);
    auto y = NDArray('f', { 2, 5, 4, 3 }, sd::DataType::FLOAT32);
    auto z = NDArray('f', { 2, 3, 5, 4, 3 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto eC = NDArray('c', { 2, 3, 5, 4, 3 }, { 0.100000, 0.181818, 0.250000, 0.307692, 0.357143, 0.400000, 0.437500, 0.470588, 0.500000, 0.526316, 0.550000, 0.571429, 0.590909, 0.608696, 0.625000, 0.640000, 0.653846, 0.666667, 0.678571, 0.689655, 0.700000, 0.709677, 0.718750, 0.727273, 0.735294, 0.742857, 0.750000, 0.756757, 0.763158, 0.769231, 0.775000, 0.780488, 0.785714, 0.790698, 0.795455, 0.800000, 0.804348, 0.808511, 0.812500, 0.816327, 0.820000, 0.823529, 0.826923, 0.830189, 0.833333, 0.836364, 0.839286, 0.842105, 0.844828, 0.847458, 0.850000, 0.852459, 0.854839, 0.857143, 0.859375, 0.861538, 0.863636, 0.865672, 0.867647, 0.869565, 6.100000, 5.636364, 5.250000, 4.923077, 4.642857, 4.400000, 4.187500, 4.000000, 3.833333, 3.684211, 3.550000, 3.428571, 3.318182, 3.217391, 3.125000, 3.040000, 2.961539, 2.888889, 2.821429, 2.758621, 2.700000, 2.645161, 2.593750, 2.545455, 2.500000, 2.457143, 2.416667, 2.378378, 2.342105, 2.307692, 2.275000, 2.243902, 2.214286, 2.186047, 2.159091, 2.133333, 2.108696, 2.085106, 2.062500, 2.040816, 2.020000, 2.000000, 1.980769, 1.962264, 1.944444, 1.927273, 1.910714, 1.894737, 1.879310, 1.864407, 1.850000, 1.836066, 1.822581, 1.809524, 1.796875, 1.784615, 1.772727, 1.761194, 1.750000, 1.739130, 12.100000, 11.090909, 10.250000, 9.538462, 8.928572, 8.400000, 7.937500, 7.529412, 7.166667, 6.842105, 6.550000, 6.285714, 6.045455, 5.826087, 5.625000, 5.440000, 5.269231, 5.111111, 4.964286, 4.827586, 4.700000, 4.580645, 4.468750, 4.363636, 4.264706, 4.171429, 4.083333, 4.000000, 3.921053, 3.846154, 3.775000, 3.707317, 3.642857, 3.581395, 3.522727, 3.466667, 3.413043, 3.361702, 3.312500, 3.265306, 3.220000, 3.176471, 3.134615, 3.094340, 3.055556, 3.018182, 2.982143, 2.947368, 2.913793, 2.881356, 2.850000, 2.819672, 2.790323, 2.761905, 2.734375, 2.707692, 2.681818, 2.656716, 2.632353, 2.608696, 2.585714, 2.563380, 2.541667, 2.520548, 2.500000, 2.480000, 2.460526, 2.441558, 2.423077, 2.405063, 2.387500, 2.370370, 2.353658, 2.337349, 2.321429, 2.305882, 2.290698, 2.275862, 2.261364, 2.247191, 2.233333, 2.219780, 2.206522, 2.193548, 2.180851, 2.168421, 2.156250, 2.144330, 2.132653, 2.121212, 2.110000, 2.099010, 2.088235, 2.077670, 2.067308, 2.057143, 2.047170, 2.037383, 2.027778, 2.018349, 2.009091, 2.000000, 1.991071, 1.982301, 1.973684, 1.965217, 1.956897, 1.948718, 1.940678, 1.932773, 1.925000, 1.917355, 1.909836, 1.902439, 1.895161, 1.888000, 1.880952, 1.874016, 1.867188, 1.860465, 3.442857, 3.408451, 3.375000, 3.342466, 3.310811, 3.280000, 3.250000, 3.220779, 3.192308, 3.164557, 3.137500, 3.111111, 3.085366, 3.060241, 3.035714, 3.011765, 2.988372, 2.965517, 2.943182, 2.921348, 2.900000, 2.879121, 2.858696, 2.838710, 2.819149, 2.800000, 2.781250, 2.762887, 2.744898, 2.727273, 2.710000, 2.693069, 2.676471, 2.660194, 2.644231, 2.628572, 2.613208, 2.598131, 2.583333, 2.568807, 2.554545, 2.540540, 2.526786, 2.513274, 2.500000, 2.486957, 2.474138, 2.461539, 2.449152, 2.436975, 2.425000, 2.413223, 2.401639, 2.390244, 2.379032, 2.368000, 2.357143, 2.346457, 2.335938, 2.325581, 4.300000, 4.253521, 4.208333, 4.164383, 4.121622, 4.080000, 4.039474, 4.000000, 3.961539, 3.924051, 3.887500, 3.851852, 3.817073, 3.783133, 3.750000, 3.717647, 3.686047, 3.655172, 3.625000, 3.595506, 3.566667, 3.538461, 3.510870, 3.483871, 3.457447, 3.431579, 3.406250, 3.381443, 3.357143, 3.333333, 3.310000, 3.287129, 3.264706, 3.242718, 3.221154, 3.200000, 3.179245, 3.158879, 3.138889, 3.119266, 3.100000, 3.081081, 3.062500, 3.044248, 3.026316, 3.008696, 2.991379, 2.974359, 2.957627, 2.941176, 2.925000, 2.909091, 2.893443, 2.878049, 2.862903, 2.848000, 2.833333, 2.818898, 2.804688, 2.790698 }, sd::DataType::FLOAT32);

    auto e = NDArray('f', { 2, 3, 5, 4, 3 }, sd::DataType::FLOAT32);

    e.assign(eC);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyBroadcast(sd::broadcast::Divide, { 0,2,3,4 }, y, z);

    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_5D_3) {

    auto x = NDArray('f', { 2, 3, 5, 4, 3 }, sd::DataType::FLOAT32);
    auto y = NDArray('f', { 2, 5 }, sd::DataType::FLOAT32);
    auto z = NDArray('f', { 2, 3, 5, 4, 3 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto eC = NDArray('c', { 2, 3, 5, 4, 3 }, { 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000, 0.800000, 0.900000, 1.000000, 1.100000, 1.200000, 1.181818, 1.272727, 1.363636, 1.454545, 1.545455, 1.636364, 1.727273, 1.818182, 1.909091, 2.000000, 2.090909, 2.181818, 2.083333, 2.166667, 2.250000, 2.333333, 2.416667, 2.500000, 2.583333, 2.666667, 2.750000, 2.833333, 2.916667, 3.000000, 2.846154, 2.923077, 3.000000, 3.076923, 3.153846, 3.230769, 3.307692, 3.384615, 3.461539, 3.538461, 3.615385, 3.692308, 3.500000, 3.571429, 3.642857, 3.714286, 3.785714, 3.857143, 3.928571, 4.000000, 4.071429, 4.142857, 4.214286, 4.285714, 6.100000, 6.200000, 6.300000, 6.400000, 6.500000, 6.600000, 6.700000, 6.800000, 6.900000, 7.000000, 7.100000, 7.200000, 6.636364, 6.727273, 6.818182, 6.909091, 7.000000, 7.090909, 7.181818, 7.272727, 7.363636, 7.454545, 7.545455, 7.636364, 7.083333, 7.166667, 7.250000, 7.333333, 7.416667, 7.500000, 7.583333, 7.666667, 7.750000, 7.833333, 7.916667, 8.000000, 7.461538, 7.538462, 7.615385, 7.692307, 7.769231, 7.846154, 7.923077, 8.000000, 8.076923, 8.153846, 8.230769, 8.307693, 7.785714, 7.857143, 7.928571, 8.000000, 8.071428, 8.142858, 8.214286, 8.285714, 8.357142, 8.428572, 8.500000, 8.571428, 12.100000, 12.200000, 12.300000, 12.400000, 12.500000, 12.600000, 12.700000, 12.800000, 12.900000, 13.000000, 13.100000, 13.200000, 12.090909, 12.181818, 12.272727, 12.363636, 12.454545, 12.545455, 12.636364, 12.727273, 12.818182, 12.909091, 13.000000, 13.090909, 12.083333, 12.166667, 12.250000, 12.333333, 12.416667, 12.500000, 12.583333, 12.666667, 12.750000, 12.833333, 12.916667, 13.000000, 12.076923, 12.153846, 12.230769, 12.307693, 12.384615, 12.461538, 12.538462, 12.615385, 12.692307, 12.769231, 12.846154, 12.923077, 12.071428, 12.142858, 12.214286, 12.285714, 12.357142, 12.428572, 12.500000, 12.571428, 12.642858, 12.714286, 12.785714, 12.857142, 12.066667, 12.133333, 12.200000, 12.266666, 12.333333, 12.400000, 12.466666, 12.533334, 12.600000, 12.666667, 12.733334, 12.800000, 12.062500, 12.125000, 12.187500, 12.250000, 12.312500, 12.375000, 12.437500, 12.500000, 12.562500, 12.625000, 12.687500, 12.750000, 12.058824, 12.117647, 12.176471, 12.235294, 12.294118, 12.352942, 12.411765, 12.470589, 12.529411, 12.588235, 12.647058, 12.705882, 12.055555, 12.111111, 12.166667, 12.222222, 12.277778, 12.333333, 12.388889, 12.444445, 12.500000, 12.555555, 12.611111, 12.666667, 12.052631, 12.105263, 12.157895, 12.210526, 12.263158, 12.315789, 12.368421, 12.421053, 12.473684, 12.526316, 12.578947, 12.631579, 16.066668, 16.133333, 16.200001, 16.266666, 16.333334, 16.400000, 16.466667, 16.533333, 16.600000, 16.666666, 16.733334, 16.799999, 15.812500, 15.875000, 15.937500, 16.000000, 16.062500, 16.125000, 16.187500, 16.250000, 16.312500, 16.375000, 16.437500, 16.500000, 15.588235, 15.647058, 15.705882, 15.764706, 15.823529, 15.882353, 15.941176, 16.000000, 16.058823, 16.117647, 16.176470, 16.235294, 15.388889, 15.444445, 15.500000, 15.555555, 15.611111, 15.666667, 15.722222, 15.777778, 15.833333, 15.888889, 15.944445, 16.000000, 15.210526, 15.263158, 15.315789, 15.368421, 15.421053, 15.473684, 15.526316, 15.578947, 15.631579, 15.684211, 15.736842, 15.789474, 20.066668, 20.133333, 20.200001, 20.266666, 20.333334, 20.400000, 20.466667, 20.533333, 20.600000, 20.666666, 20.733334, 20.799999, 19.562500, 19.625000, 19.687500, 19.750000, 19.812500, 19.875000, 19.937500, 20.000000, 20.062500, 20.125000, 20.187500, 20.250000, 19.117647, 19.176470, 19.235294, 19.294117, 19.352942, 19.411764, 19.470589, 19.529411, 19.588236, 19.647058, 19.705883, 19.764706, 18.722221, 18.777779, 18.833334, 18.888889, 18.944445, 19.000000, 19.055555, 19.111111, 19.166666, 19.222221, 19.277779, 19.333334, 18.368422, 18.421053, 18.473684, 18.526316, 18.578947, 18.631578, 18.684210, 18.736841, 18.789474, 18.842106, 18.894737, 18.947369 }, sd::DataType::FLOAT32);

    auto e = NDArray('f', { 2, 3, 5, 4, 3 }, sd::DataType::FLOAT32);

    e.assign(eC);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyBroadcast(sd::broadcast::Divide, { 0,2 }, y, z);

    ASSERT_EQ(e, z);
}
///////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Test_broadcast_5D_4) {

    auto x = NDArray('f', { 2, 3, 5, 4, 3 }, sd::DataType::FLOAT32);
    auto y = NDArray('f', { 2, 1, 5, 1, 1 }, sd::DataType::FLOAT32);
    auto z = NDArray('f', { 2, 3, 5, 4, 3 }, sd::DataType::FLOAT32);
    // recieved by main algorithm
    auto eC = NDArray('c', { 2, 3, 5, 4, 3 }, { 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000, 0.800000, 0.900000, 1.000000, 1.100000, 1.200000, 1.181818, 1.272727, 1.363636, 1.454545, 1.545455, 1.636364, 1.727273, 1.818182, 1.909091, 2.000000, 2.090909, 2.181818, 2.083333, 2.166667, 2.250000, 2.333333, 2.416667, 2.500000, 2.583333, 2.666667, 2.750000, 2.833333, 2.916667, 3.000000, 2.846154, 2.923077, 3.000000, 3.076923, 3.153846, 3.230769, 3.307692, 3.384615, 3.461539, 3.538461, 3.615385, 3.692308, 3.500000, 3.571429, 3.642857, 3.714286, 3.785714, 3.857143, 3.928571, 4.000000, 4.071429, 4.142857, 4.214286, 4.285714, 6.100000, 6.200000, 6.300000, 6.400000, 6.500000, 6.600000, 6.700000, 6.800000, 6.900000, 7.000000, 7.100000, 7.200000, 6.636364, 6.727273, 6.818182, 6.909091, 7.000000, 7.090909, 7.181818, 7.272727, 7.363636, 7.454545, 7.545455, 7.636364, 7.083333, 7.166667, 7.250000, 7.333333, 7.416667, 7.500000, 7.583333, 7.666667, 7.750000, 7.833333, 7.916667, 8.000000, 7.461538, 7.538462, 7.615385, 7.692307, 7.769231, 7.846154, 7.923077, 8.000000, 8.076923, 8.153846, 8.230769, 8.307693, 7.785714, 7.857143, 7.928571, 8.000000, 8.071428, 8.142858, 8.214286, 8.285714, 8.357142, 8.428572, 8.500000, 8.571428, 12.100000, 12.200000, 12.300000, 12.400000, 12.500000, 12.600000, 12.700000, 12.800000, 12.900000, 13.000000, 13.100000, 13.200000, 12.090909, 12.181818, 12.272727, 12.363636, 12.454545, 12.545455, 12.636364, 12.727273, 12.818182, 12.909091, 13.000000, 13.090909, 12.083333, 12.166667, 12.250000, 12.333333, 12.416667, 12.500000, 12.583333, 12.666667, 12.750000, 12.833333, 12.916667, 13.000000, 12.076923, 12.153846, 12.230769, 12.307693, 12.384615, 12.461538, 12.538462, 12.615385, 12.692307, 12.769231, 12.846154, 12.923077, 12.071428, 12.142858, 12.214286, 12.285714, 12.357142, 12.428572, 12.500000, 12.571428, 12.642858, 12.714286, 12.785714, 12.857142, 12.066667, 12.133333, 12.200000, 12.266666, 12.333333, 12.400000, 12.466666, 12.533334, 12.600000, 12.666667, 12.733334, 12.800000, 12.062500, 12.125000, 12.187500, 12.250000, 12.312500, 12.375000, 12.437500, 12.500000, 12.562500, 12.625000, 12.687500, 12.750000, 12.058824, 12.117647, 12.176471, 12.235294, 12.294118, 12.352942, 12.411765, 12.470589, 12.529411, 12.588235, 12.647058, 12.705882, 12.055555, 12.111111, 12.166667, 12.222222, 12.277778, 12.333333, 12.388889, 12.444445, 12.500000, 12.555555, 12.611111, 12.666667, 12.052631, 12.105263, 12.157895, 12.210526, 12.263158, 12.315789, 12.368421, 12.421053, 12.473684, 12.526316, 12.578947, 12.631579, 16.066668, 16.133333, 16.200001, 16.266666, 16.333334, 16.400000, 16.466667, 16.533333, 16.600000, 16.666666, 16.733334, 16.799999, 15.812500, 15.875000, 15.937500, 16.000000, 16.062500, 16.125000, 16.187500, 16.250000, 16.312500, 16.375000, 16.437500, 16.500000, 15.588235, 15.647058, 15.705882, 15.764706, 15.823529, 15.882353, 15.941176, 16.000000, 16.058823, 16.117647, 16.176470, 16.235294, 15.388889, 15.444445, 15.500000, 15.555555, 15.611111, 15.666667, 15.722222, 15.777778, 15.833333, 15.888889, 15.944445, 16.000000, 15.210526, 15.263158, 15.315789, 15.368421, 15.421053, 15.473684, 15.526316, 15.578947, 15.631579, 15.684211, 15.736842, 15.789474, 20.066668, 20.133333, 20.200001, 20.266666, 20.333334, 20.400000, 20.466667, 20.533333, 20.600000, 20.666666, 20.733334, 20.799999, 19.562500, 19.625000, 19.687500, 19.750000, 19.812500, 19.875000, 19.937500, 20.000000, 20.062500, 20.125000, 20.187500, 20.250000, 19.117647, 19.176470, 19.235294, 19.294117, 19.352942, 19.411764, 19.470589, 19.529411, 19.588236, 19.647058, 19.705883, 19.764706, 18.722221, 18.777779, 18.833334, 18.888889, 18.944445, 19.000000, 19.055555, 19.111111, 19.166666, 19.222221, 19.277779, 19.333334, 18.368422, 18.421053, 18.473684, 18.526316, 18.578947, 18.631578, 18.684210, 18.736841, 18.789474, 18.842106, 18.894737, 18.947369 }, sd::DataType::FLOAT32);

    auto e = NDArray('f', { 2, 3, 5, 4, 3 }, sd::DataType::FLOAT32);
    e.assign(eC);

    x.linspace(1.f);
    y.linspace(10.f);
    z.assign(0.f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Divide(), y, z);

    ASSERT_EQ(e, z);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_1) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    Nd4jLong shape1[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 2, 3, 4, 12, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, sd::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input2}, {}, {0});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_2) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1,2,3,4, 13, 14, 16, 16, 5,6,7,8, 17, 18, 19, 20, 9, 10, 11, 12, 21, 22, 23, 24};
    Nd4jLong shape1[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 3, 4, 4, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 3, 2, 4, 8, 4, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, sd::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input2}, {}, {1});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_3) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    Nd4jLong shape1[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 2, 1, 12, 12, 12, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, sd::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input2}, {}, {0});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_4) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    Nd4jLong shape1[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 1, 12, 12, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 1, 2, 12, 24, 12, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, sd::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input2}, {}, {1});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_5) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,16,16,17,18,19,20,21,22,23,24};
    Nd4jLong shape1[]    = {2, 12, 1, 1,1, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 12, 1, 1,1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 2, 12, 1, 12, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, sd::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input2}, {}, {0});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_6) {

    float buff1[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12};
    float buff2[]   = {13,14,16,16,17,18,19,20,21,22,23,24};
    float expBuff[] = {1 ,13 ,2 ,14 ,3 ,16 ,4 ,16 ,5 ,17 ,6 ,18 ,7 ,19 ,8 ,20 ,9 ,21 ,10 ,22 ,11 ,23 ,12 ,24};
    Nd4jLong shape1[]    = {2, 12, 1, 1, 12, 0, 1, 99};
    Nd4jLong shape2[]    = {2, 12, 1, 1, 12, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 12, 2, 1, 2, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(shape2, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, sd::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray input2(buff2, shape2);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input2}, {}, {1});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_7) {

    float buff1[]   = {1};
    float expBuff[] = {1, 1, 1};
    Nd4jLong shape1[]    = {2, 1, 1, 1, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 3, 1, 1, 1, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, sd::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input1, &input1}, {}, {0});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_8) {

    float buff1[]   = {1};
    float expBuff[] = {1, 1, 1};
    Nd4jLong shape1[]    = {1, 1, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {2, 3, 1, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, sd::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input1, &input1}, {}, {0});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_9) {

    float buff1[]   = {1};
    float expBuff[] = {1, 1, 1};
    Nd4jLong shape1[]    = {2, 1, 1, 1, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {3, 1, 3, 1, 3, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, sd::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input1, &input1}, {}, {1});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_10) {

    float buff1[]   = {1};
    float expBuff[] = {1, 1, 1};
    Nd4jLong shape1[]    = {1, 1, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {2, 1, 3, 3, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, sd::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input1, &input1}, {}, {1});
    auto output = results->at(0);

    //expected.printShapeInfo("exp");
    //output->printShapeInfo("out");

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

TEST_F(DeclarableOpsTests14, Stack_11) {

    float buff1[]   = {1};
    float expBuff[] = {1, 1, 1};
    Nd4jLong shape1[]    = {1, 1, 1, 0, 1, 99};
    Nd4jLong expShape[]  = {2, 3, 1, 1, 1, 0, 1, 99};
    ArrayOptions::setDataType(shape1, sd::DataType::FLOAT32);
    ArrayOptions::setDataType(expShape, sd::DataType::FLOAT32);

    NDArray input1(buff1, shape1);
    NDArray expected(expBuff, expShape);

    sd::ops::stack op;
    auto results = op.evaluate({&input1, &input1, &input1}, {}, {});
    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_12) {
    float inBuff[]  = {1.0f, 2.0f, 3.0f};
    float expBuff[] = {1.0f, 2.0f, 3.0f};

    auto input = NDArrayFactory::create<float>(inBuff, 'c', {1, 3});

    auto exp = NDArrayFactory::create<float>(expBuff, 'c', {1, 1, 3});

    sd::ops::stack op;

    auto result = op.evaluate({&input}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_13) {
    float inBuff[]  = {1.0f, 2.0f, 3.0f};
    float expBuff[] = {1.0f, 2.0f, 3.0f};

    auto input = NDArrayFactory::create<float>(inBuff, 'c', {1, 1, 3});

    auto exp = NDArrayFactory::create<float>(expBuff, 'c', {1, 1, 1, 3});

    sd::ops::stack op;

    auto result = op.evaluate({&input}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests14, Stack_14) {
    float inBuff[]  = {1.0f, 2.0f, 3.0f};
    float expBuff[] = {1.0f, 2.0f, 3.0f};

    auto input = NDArrayFactory::create<float>(inBuff, 'c', {1, 3});

    auto exp = NDArrayFactory::create<float>(expBuff, 'c', {1, 1, 3});

    sd::ops::stack op;

    auto result = op.evaluate({&input}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printShapeInfo();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests14, Stack_15) {
    auto t = NDArrayFactory::create<double>('c', {2, 3, 5});
    auto u = NDArrayFactory::create<double>('c', {2, 3, 5});
    auto v = NDArrayFactory::create<double>('c', {2, 3, 5});
    auto exp = NDArrayFactory::create<double>('c', {3, 2, 3, 5});

    sd::ops::stack op;
    auto result = op.evaluate({&t, &u, &v}, {}, {-4});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());


    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests14, Stack_16) {
    auto t = NDArrayFactory::create<float>(1.0f);
    auto u = NDArrayFactory::create<float>(2.0f);
    auto v = NDArrayFactory::create<float>(3.0f);
    auto exp = NDArrayFactory::create<float>('c', {3}, {1, 2, 3});

    sd::ops::stack op;
    auto result = op.evaluate({&t, &u, &v}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests14, Stack_17) {
    auto t = NDArrayFactory::create<float>('c', {1, 1}, {1.0f});
    auto u = NDArrayFactory::create<float>('c', {1, 1}, {2.0f});
    auto v = NDArrayFactory::create<float>('c', {1, 1}, {3.0f});
    auto w = NDArrayFactory::create<float>('c', {1, 1}, {4.0f});
    auto exp = NDArrayFactory::create<float>('c', {4, 1, 1}, {1, 2, 3, 4});

    sd::ops::stack op;
    auto result = op.evaluate({&t, &u, &v, &w}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    // z->printShapeInfo("z shape");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests14, Stack_18) {
    auto x = NDArrayFactory::create<float>('c', {0});
    auto e = NDArrayFactory::create<float>('c', {1, 0});

    sd::ops::stack op;
    auto result = op.evaluate({&x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);
    sd::ops::reduce_min sumOp;
    auto res2 = sumOp.evaluate({&e}, {1.}, {1});
    ASSERT_EQ(res2->status(), Status::OK());
    auto out = res2->at(0);

    ASSERT_EQ(out->e<float>(0), DataTypeUtils::infOrMax<float>());
    delete res2;
    delete result;
}

TEST_F(DeclarableOpsTests14, Stack_19) {
    auto x = NDArrayFactory::empty<float>();
    auto e = NDArrayFactory::create<float>('c', {0});

    sd::ops::stack op;
    auto result = op.evaluate({&x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, Stack_20) {
    auto x = NDArrayFactory::empty<float>();
    auto e = NDArrayFactory::create<float>('c', {2, 0});

    sd::ops::stack op;
    auto result = op.evaluate({&x, &x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, Stack_21) {

    NDArray x1('c', {3,2}, sd::DataType::FLOAT32);
    NDArray x2('c', {3,2}, sd::DataType::FLOAT32);
    x1.linspace(0);
    x2.linspace(6);

    sd::ops::stack opStack;
    auto resultStack = opStack.evaluate({&x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, resultStack->status());


    sd::ops::concat opConcat;
    auto resultConcat = opConcat.evaluate({&x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, resultConcat->status());

    auto outStack  = resultStack->at(0);
    auto outConcat = resultConcat->at(0);

    outConcat->reshapei({2,3,2});

    ASSERT_TRUE(outStack->isSameShape(outConcat));
    ASSERT_TRUE(outStack->equalsTo(outConcat));

    delete resultStack;
    delete resultConcat;
}


