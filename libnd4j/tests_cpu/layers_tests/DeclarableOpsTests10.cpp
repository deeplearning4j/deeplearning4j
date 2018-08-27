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
#include <helpers/RandomLauncher.h>

using namespace nd4j;


class DeclarableOpsTests10 : public testing::Test {
public:

    DeclarableOpsTests10() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests10, Test_ArgMax_1) {
    NDArray<double> x('c', {3, 3});
    NDArray<double> e(8);

    x.linspace(1.0);


    nd4j::ops::argmax<double> op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());


    auto z = *result->at(0);

    ASSERT_EQ(e, z);

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_ArgMax_2) {
    NDArray<double> x('c', {3, 3});
    NDArray<double> y('c', {1}, {1.0});
    NDArray<double> e('c', {3}, {2.0, 2.0, 2.0});

    x.linspace(1.0);

    nd4j::ops::argmax<double> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = *result->at(0);

    //z.printIndexedBuffer("z");
    //z.printShapeInfo("z shape");

    ASSERT_EQ(e, z);

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_And_1) {
    NDArray<double> x('c', {4}, {1, 1, 0, 1});
    NDArray<double> y('c', {4}, {0, 0, 0, 1});
    NDArray<double> e('c', {4}, {0, 0, 0, 1});

    nd4j::ops::boolean_and<double> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_Or_1) {
    NDArray<double> x('c', {4}, {1, 1, 0, 1});
    NDArray<double> y('c', {4}, {0, 0, 0, 1});
    NDArray<double> e('c', {4}, {1, 1, 0, 1});

    nd4j::ops::boolean_or<double> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_Not_1) {
    NDArray<double> x('c', {4}, {1, 1, 0, 1});
    NDArray<double> y('c', {4}, {0, 0, 0, 1});
    NDArray<double> e('c', {4}, {1, 1, 1, 0});

    nd4j::ops::boolean_not<double> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_Size_at_1) {
    NDArray<double> x('c', {10, 20, 30});
    NDArray<double> e(20.0);

    nd4j::ops::size_at<double> op;
    auto result = op.execute({&x}, {}, {1});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Pad_SGO_Test_1) {

    NDArray<double> in({1., 1., 1., 1., 1.});
//    NDArray<double> pad('c', {1, 2}, {1., 1.});// = Nd4j.create(new double[]{1, 1}, new long[]{1, 2});
    NDArray<double> pad('c', {1, 2}, {1., 1.});
//    NDArray<double> value(10.0);

    NDArray<double> exp({10., 1., 1., 1., 1., 1., 10.});

    nd4j::ops::pad<double> op;

    auto res = op.execute({&in, &pad}, {10.0}, {0});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Unique_SGO_Test_1) {
    NDArray<double> input({3., 4., 3., 1., 3., 0., 2., 4., 2., 4.});
    NDArray<double> expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    NDArray<double> exp({3., 4., 1., 0., 2.});

    nd4j::ops::unique<double> op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    //res->at(0)->printIndexedBuffer("Unique values");
    //res->at(1)->printIndexedBuffer("Unique idxs");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Where_SGO_Test_1) {
    NDArray<double> input('c', {3, 3}, {1., 0., 0., 1., 1., 0., 1., 1., 1.});
    //NDArray<double> expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    NDArray<double> exp('c', {6, 2}, {0., 0., 1., 0., 1., 1., 2., 0., 2., 1., 2., 2.});

    nd4j::ops::Where<double> op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    NDArray<double>* resA = res->at(0);

    ASSERT_TRUE(exp.equalsTo(resA));
    ASSERT_TRUE(exp.isSameShape(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, WhereNP_SGO_Test_1) {
    NDArray<double> cond3d('c', {2, 2, 2}, {1., 0., 0., 1., 1., 1., 1., 0.});
//    NDArray<double> expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    NDArray<double> exp1({0., 0., 1., 1., 1.});
    NDArray<double> exp2({0., 1., 0., 0., 1.});
    NDArray<double> exp3({0., 1., 0., 1., 0.});
    nd4j::ops::where_np<double> op;
    auto res = op.execute({&cond3d}, {}, {});
    ASSERT_TRUE(res->size() == 3);
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    ASSERT_TRUE(exp1.equalsTo(res->at(0)));
    ASSERT_TRUE(exp2.equalsTo(res->at(1)));
    ASSERT_TRUE(exp3.equalsTo(res->at(2)));
    //ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, WhereNP_SGO_Test_2) {
    NDArray<double> cond2d('c', {3, 5}, {1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1.});
//    NDArray<double> expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    NDArray<double> exp1({0., 0., 0., 1., 1., 1., 1., 1., 2., 2., 2., 2.});
    NDArray<double> exp2({0., 1., 4., 0., 1., 2., 3., 4., 1., 2., 3., 4.});
    nd4j::ops::where_np<double> op;
    auto res = op.execute({&cond2d}, {}, {});
    ASSERT_TRUE(res->size() == 2);
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    ASSERT_TRUE(exp1.equalsTo(res->at(0)));
    ASSERT_TRUE(exp2.equalsTo(res->at(1)));
    //ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, svd_test11) {

    NDArray<double> x('c', {3,3}, {1.,2.,3.,4.,5.,6.,7.,8.,9.});
    NDArray<double> expS('c', {3});
    NDArray<double> expU('c', {3,3});
    NDArray<double> expV('c', {3,3});

    nd4j::ops::svd<double> op;
    nd4j::ResultSet<double>* results = op.execute({&x}, {}, {0, 1, 16});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *s = results->at(0);
    NDArray<double> *u = results->at(1);
    NDArray<double> *v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, TruncatedNormalDistribution_Test1) {

    NDArray<double> x('c', {10,10});
    NDArray<double> z('c', {10,10});
//    NDArray<double> expS('c', {3});
//    NDArray<double> expU('c', {3,3});
//    NDArray<double> expV('c', {3,3});
    x.linspace(1);
    NativeOps nops;
    std::vector<Nd4jLong> buffer(50000);
    double restParams[2] = {1., 2.};
    random::RandomBuffer* rng = (nd4j::random::RandomBuffer *)nops.initRandom(nullptr, 123L, buffer.size(), reinterpret_cast<Nd4jPointer>(&buffer[0]));
    //x.applyRandom<randomOps::GaussianDistribution<double>>(rng, nullptr, &z, restParams);

    //x.printIndexedBuffer("Input");
    //z.printIndexedBuffer("Output");
    RandomLauncher<double>::fillTruncatedNormal(rng, &z, 1., 12.);
    RandomLauncher<double>::fillGaussian(rng, &x, 1., 12.);
    z.printIndexedBuffer("Truncated Output");
    double mean = 0, variance = 0;
    double meanX = 0, varianceX = 0;

    for (Nd4jLong e = 0; e < z.lengthOf(); ++e) {
        mean += z(e);
        meanX += x(e);
    }
    mean /= 100.0;
    meanX /= 100.0;
    for (Nd4jLong e = 0; e < z.lengthOf(); ++e) {
        variance += nd4j::math::nd4j_sqrt(nd4j::math::nd4j_pow((mean - z(e)), 2.));
        varianceX += nd4j::math::nd4j_sqrt(nd4j::math::nd4j_pow((meanX - x(e)), 2.));
    }
    variance /= 100.0;
    varianceX /= 100.0;
    nd4j_printf("Mean: %f; Variance: %f\n", mean, variance);
    nd4j_printf("MeanG: %f; VarianceG: %f\n", meanX, varianceX);
    ASSERT_EQ(mean, 0.);
    ASSERT_EQ(variance, 1.);
/*    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *s = results->at(0);
    NDArray<double> *u = results->at(1);
    NDArray<double> *v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    delete results;
    */
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, TruncatedNormalDistribution_Test2) {

    NDArray<float> x('c', {10,10});
    NDArray<float> z('c', {10,10});
//    NDArray<double> expS('c', {3});
//    NDArray<double> expU('c', {3,3});
//    NDArray<double> expV('c', {3,3});
    x.linspace(1);
    NativeOps nops;
    std::vector<Nd4jLong> buffer(50000);
    float restParams[2] = {1.f,2.f};
    random::RandomBuffer* rng = (nd4j::random::RandomBuffer *)nops.initRandom(nullptr, 123L, buffer.size(), reinterpret_cast<Nd4jPointer>(&buffer[0]));
    //x.applyRandom<randomOps::GaussianDistribution<double>>(rng, nullptr, &z, restParams);

    //x.printIndexedBuffer("Input");
    //z.printIndexedBuffer("Output");
    RandomLauncher<float>::fillTruncatedNormal(rng, &z, 1.f, 12.f);
    RandomLauncher<float>::fillGaussian(rng, &x, 1.f, 12.f);
    z.printIndexedBuffer("Truncated Output");
    float mean = 0, variance = 0;
    float meanX = 0, varianceX = 0;

    for (Nd4jLong e = 0; e < z.lengthOf(); ++e) {
        mean += z(e);
        meanX += x(e);
    }
    mean /= 100.0f;
    meanX /= 100.0f;
    for (Nd4jLong e = 0; e < z.lengthOf(); ++e) {
        variance += nd4j::math::nd4j_sqrt(nd4j::math::nd4j_pow((mean - z(e)), 2.f));
        varianceX += nd4j::math::nd4j_sqrt(nd4j::math::nd4j_pow((meanX - x(e)), 2.f));
    }
    variance /= 100.0f;
    varianceX /= 100.0f;
    nd4j_printf("Mean: %f; Variance: %f\n", mean, variance);
    nd4j_printf("MeanG: %f; VarianceG: %f\n", meanX, varianceX);
    ASSERT_EQ(meanX, 1.f);
    ASSERT_EQ(varianceX, 12.f);
/*    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *s = results->at(0);
    NDArray<double> *u = results->at(1);
    NDArray<double> *v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    delete results;
    */
}
