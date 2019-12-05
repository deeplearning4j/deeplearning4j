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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 22.06.2018
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>
#include <loops/random.h>


using namespace nd4j;


class DeclarableOpsTests9 : public testing::Test {
public:

    DeclarableOpsTests9() {
        printf("\n");
        fflush(stdout);
    }
};

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, reduceStDevBP_test3) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {3,1}, {1.,2.,3.});
    auto gradO2 = NDArrayFactory::create<double>('c', {3}, {1.,2.,3.});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {-0.335410, -0.111803, 0.111803, 0.335410, -0.670820, -0.223607, 0.223607, 0.670820, -1.006231, -0.335410, 0.335410, 1.006231});

    x.linspace(1);

    nd4j::ops::reduce_stdev_bp op;

    auto result = op.execute({&x, &gradO2}, {0,0}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);
    // output->printIndexedBuffer();
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, reduceStDevBP_test03) {

    auto x = NDArrayFactory::create<double>('c', {3,4});
    auto gradO1 = NDArrayFactory::create<double>('c', {3,1}, {1.,2.,3.});
    auto gradO2 = NDArrayFactory::create<double>('c', {3}, {1.,2.,3.});
    auto exp = NDArrayFactory::create<double>('c', {3,4}, {-0.335410, -0.111803, 0.111803, 0.335410, -0.670820, -0.223607, 0.223607, 0.670820, -1.006231, -0.335410, 0.335410, 1.006231});
    auto axis = NDArrayFactory::create<int>('c', {1}, {1});
    x.linspace(1);

    nd4j::ops::reduce_stdev_bp op;

    auto result = op.execute({&x, &gradO2, &axis}, {}, {}, {false, false});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);
    // output->printIndexedBuffer();
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

    result = op.execute({&x, &gradO1}, {1,0}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    output = result->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;

}
/*

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, exponentialDistributionInv_test1) {

    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda;
    const double std    = mean;

    auto x = NDArrayFactory::create<double>('c', {N});
    double extraParams[] = {lambda};

    Nd4jLong *buffer = new Nd4jLong[N];
    auto rng = (nd4j::random::RandomBuffer *) initRandom(nullptr, 123, N, (Nd4jPointer) buffer);
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistributionInv_test1: RNG initialization failed !");

    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistributionInv<double>>(rng, x.getBuffer(), x.getShapeInfo(), extraParams);
    const double actualMean = x.meanNumber().e<double>(0);
    const double actualStd  = x.varianceNumber(variance::SummaryStatsStandardDeviation, true).e<double>(0);

    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);

    destroyRandom((Nd4jPointer) rng);
    delete[] buffer;

}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, exponentialDistributionInv_test2) {

    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda;
    const double std    = mean;
    double extraParams[] = {lambda};

    auto x = NDArrayFactory::create<double>('c', {N});
    auto y = NDArrayFactory::create<double>('c', {N});
    y.linspace(0., 1./N);  // [0, 1)


    Nd4jLong *buffer = new Nd4jLong[N];
    auto rng = (nd4j::random::RandomBuffer *) initRandom(nullptr, 123, N, (Nd4jPointer) buffer);
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistributionInv_test2: RNG initialization failed !");

    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistributionInv<double>>(rng, y.getBuffer(), y.getShapeInfo(), x.getBuffer(), x.getShapeInfo(), extraParams);

    const double actualMean = x.meanNumber().e<double>(0);
    const double actualStd  = x.varianceNumber(variance::SummaryStatsStandardDeviation, true).e<double>(0);

    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);

    destroyRandom((Nd4jPointer) rng);
    delete[] buffer;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, exponentialDistribution_test1) {

    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda;
    const double std    = mean;

    auto x = NDArrayFactory::create<double>('c', {N});
    double extraParams[] = {lambda};

    Nd4jLong *buffer = new Nd4jLong[N];
    auto rng = (nd4j::random::RandomBuffer *) initRandom(nullptr, 123, N, (Nd4jPointer) buffer);
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistribution_test1: RNG initialization failed !");

    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistribution<double>>(rng, x.getBuffer(), x.getShapeInfo(), extraParams);
    const double actualMean = x.meanNumber().e<double>(0);
    const double actualStd  = x.varianceNumber(variance::SummaryStatsStandardDeviation, true).e<double>(0);

    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);

    destroyRandom((Nd4jPointer) rng);
    delete[] buffer;
}


//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, exponentialDistribution_test2) {

    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda;
    const double std    = mean;
    double extraParams[] = {lambda};

    auto x = NDArrayFactory::create<double>('c', {N});
    auto y = NDArrayFactory::create<double>('c', {N});
    y.linspace(-N/2.);  // [-25000, 25000)


    Nd4jLong *buffer = new Nd4jLong[N];
   // Nd4jPointer extra[2];
#ifndef __CUDABLAS__
    nd4j::random::RandomBuffer* rng = (nd4j::random::RandomBuffer *) initRandom(nullptr, 123, N, (Nd4jPointer) buffer);
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistribution_test2: RNG initialization failed !");

    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistribution<double>>(rng, y.getBuffer(), y.getShapeInfo(), x.getBuffer(), x.getShapeInfo(), extraParams);

    destroyRandom((Nd4jPointer) rng);
#endif
    const double actualMean = x.meanNumber().e<double>(0);
    const double actualStd  = x.varianceNumber(variance::SummaryStatsStandardDeviation, true).e<double>(0);
    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);




    delete[] buffer;
}
*/

TEST_F(DeclarableOpsTests9, ScalarOpTest_MixedOrders_1) {
    auto x = NDArrayFactory::create<double>('f', {2, 2}, {1.0, 3.0, 2.0, 4.0});
    auto e = NDArrayFactory::create<double>('c', {2, 2}, {2.0, 3.0, 4.0, 5.0});
    auto z = NDArrayFactory::create<double>('c', {2, 2}, {0.0, 0.0, 0.0, 0.0});

    x.applyScalar(scalar::Add, 1.0, &z);

    ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test1) {

    auto x0 = NDArrayFactory::create<double>('c', {2,3,4});
    auto x1 = NDArrayFactory::create<double>('c', {2,2,4});
    auto x2 = NDArrayFactory::create<double>('c', {2,1,4});
    auto exp = NDArrayFactory::create<double>('c', {2,6,4}, {1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                     13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test2) {

    auto x0 = NDArrayFactory::create<double>('c', {1,3,1});
    auto x1 = NDArrayFactory::create<double>('c', {1,2,1});
    auto x2 = NDArrayFactory::create<double>('c', {1,1,1});
    auto exp = NDArrayFactory::create<double>('c', {1,6,1}, {1.f, 2.f, 3.f, 1.f, 2.f, 1.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test3) {

    auto x0 = NDArrayFactory::create<double>('c', {3});
    auto x1 = NDArrayFactory::create<double>('c', {2});
    auto x2 = NDArrayFactory::create<double>('c', {1});
    auto exp = NDArrayFactory::create<double>('c', {6}, {1.f, 2.f, 3.f, 1.f, 2.f, 1.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test4) {

    auto x0 = NDArrayFactory::create<double>('c', {1,1,1}, {1.f});
    auto x1 = NDArrayFactory::create<double>('c', {1,1,1}, {2.f});
    auto x2 = NDArrayFactory::create<double>('c', {1,1,1}, {3.f});
    auto exp = NDArrayFactory::create<double>('c', {1,3,1}, {1.f, 2.f, 3.f});

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test5) {

    auto x0 = NDArrayFactory::create<double>(1.f);
    auto x1 = NDArrayFactory::create<double>('c', {1}, {2.f});
    auto x2 = NDArrayFactory::create<double>(3.f);
    auto exp = NDArrayFactory::create<double>('c', {3}, {1.f, 2.f, 3.f});

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test6) {

    auto x0 = NDArrayFactory::create<double>(1.f);
    auto x1 = NDArrayFactory::create<double>('c', {2}, {2.f, 20.f});
    auto x2 = NDArrayFactory::create<double>(3.f);
    auto exp = NDArrayFactory::create<double>('c', {4}, {1.f, 2.f, 20.f, 3.f});

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test7) {

    auto x0 = NDArrayFactory::create<double>(1.f);
    auto x1 = NDArrayFactory::create<double>(2.f);
    auto x2 = NDArrayFactory::create<double>(3.f);
    auto exp = NDArrayFactory::create<double>('c', {3}, {1.f, 2.f, 3.f});

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test8) {

    auto x0 = NDArrayFactory::create<double>(1.f);
    auto exp = NDArrayFactory::create<double>('c', {1}, {1.f});

    nd4j::ops::concat op;

    auto result = op.execute({&x0}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test9) {

    auto x0 = NDArrayFactory::create<double>('c', {1}, {1.f});
    auto exp = NDArrayFactory::create<double>('c', {1}, {1.f});

    nd4j::ops::concat op;

    auto result = op.execute({&x0}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test10) {

    auto x0 = NDArrayFactory::create<double>('c', {2,3,4});
    auto x1 = NDArrayFactory::create<double>('f', {2,2,4});
    auto x2 = NDArrayFactory::create<double>('c', {2,1,4});
    auto exp = NDArrayFactory::create<double>('c', {2,6,4}, { 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                      13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test11) {

    auto x0 = NDArrayFactory::create<double>('c', {2,3,4});
    auto x1 = NDArrayFactory::create<double>('f', {2,2,4});
    auto x2 = NDArrayFactory::create<double>('f', {2,1,4});
    auto exp = NDArrayFactory::create<double>('c', {2,6,4}, { 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                      13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test12) {

    auto x0 = NDArrayFactory::create<double>('c', {2,3,4});
    auto x1 = NDArrayFactory::create<double>('f', {2,2,4});
    auto x2 = NDArrayFactory::create<double>('f', {2,1,4});
    auto exp = NDArrayFactory::create<double>('c', {2,6,4}, { 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                      13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test13) {

    auto x0 = NDArrayFactory::create<double>('f', {2,3,4});
    auto x1 = NDArrayFactory::create<double>('f', {2,2,4});
    auto x2 = NDArrayFactory::create<double>('f', {2,1,4});
    auto exp = NDArrayFactory::create<double>('f', {2,6,4}, { 1.f, 13.f, 5.f, 17.f, 9.f, 21.f, 1.f,  9.f, 5.f, 13.f, 1.f,  5.f, 2.f, 14.f, 6.f, 18.f,10.f, 22.f, 2.f, 10.f, 6.f, 14.f, 2.f,  6.f,
                                       3.f, 15.f, 7.f, 19.f,11.f, 23.f, 3.f, 11.f, 7.f, 15.f, 3.f,  7.f, 4.f, 16.f, 8.f, 20.f,12.f, 24.f, 4.f, 12.f, 8.f, 16.f, 4.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);


    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

TEST_F(DeclarableOpsTests9, concat_test14) {

    NDArray x0('c', {1, 40, 60}, nd4j::DataType::DOUBLE);
    NDArray x1('c', {1, 40, 60}, nd4j::DataType::DOUBLE);

    x0 = 1.;
    x1 = 2.;

    nd4j::ops::concat op;
    auto result = op.execute({&x0, &x1}, {}, {0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    Nd4jLong numOfTads= ShapeUtils::getNumOfSubArrs(z->getShapeInfo(), {0});
    ASSERT_TRUE(2 == numOfTads);

    for (int e = 0; e < numOfTads; ++e) {
        NDArray tad  = (*z)(e, {0});
        auto mean = tad.meanNumber().e<double>(0);
        ASSERT_NEAR((e+1)*1., mean, 1e-5);
    }

    delete result;
}

TEST_F(DeclarableOpsTests9, concat_test15) {
    auto x = NDArrayFactory::create<double>('c', {2}, {1, 0});
    auto y = NDArrayFactory::create<double> (3.0f);
    auto exp = NDArrayFactory::create<double>('c', {3}, {1, 0, 3});

    nd4j::ops::concat op;
    auto result = op.execute({&x, &y}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test16) {

    auto x = NDArrayFactory::create<double>('c', {0,2,3});
    auto y = NDArrayFactory::create<double>('c', {0,2,3});
    auto exp = NDArrayFactory::create<double>('c', {0,2,3});

    nd4j::ops::concat op;
    auto result = op.execute({&x, &y}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test17) {

    NDArray x0('c', {1, 55, 40}, nd4j::DataType::DOUBLE);
    NDArray x1('c', {1, 55, 40}, nd4j::DataType::DOUBLE);

    x0 = 1.;
    x1 = 2.;

    nd4j::ops::concat op;
    auto result = op.execute({&x0, &x1}, {}, {0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printShapeInfo();
    // z->printIndexedBuffer();

    Nd4jLong numOfTads= ShapeUtils::getNumOfSubArrs(z->getShapeInfo(), {0});
    ASSERT_TRUE(2 == numOfTads);

    for (int e = 0; e < numOfTads; ++e) {
        NDArray tad  = (*z)(e, {0});
        auto mean = tad.meanNumber().e<double>(0);
        ASSERT_NEAR((e+1)*1., mean, 1e-5);
    }

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test18) {
    Context context(1);
    Nd4jLong axis = 0;

    // we crate bunch of arrays, filled with specific values
    for (int e = 0; e < 2000; e++) {
        auto array = NDArrayFactory::create_<float>('c', {1, 300});
        array->assign(e);
        context.setInputArray(e, array, true);
    }

    auto z = NDArrayFactory::create<float>('c', {2000, 300});
    context.setOutputArray(0, &z, false);
    context.setIArguments(&axis, 1);

    nd4j::ops::concat op;
    op.execute(&context);

    for (int e = 0; e < 2000; e++) {
        auto row = z.tensorAlongDimension(e, {1});

        ASSERT_NEAR((float) e, row->e<float>(0), 1e-5f);

        delete row;
    }
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test19) {

    Context context(1);
    Nd4jLong axis = 0;

    // we crate bunch of arrays, filled with specific values
    for (int e = 0; e < 10; e++) {
        auto array = NDArrayFactory::create_<float>('c', {1, 5, 20});
        array->assign(e);
        context.setInputArray(e, array, true);
    }

    auto z = NDArrayFactory::create<float>('c', {10, 5, 20});
    context.setOutputArray(0, &z, false);
    context.setIArguments(&axis, 1);

    nd4j::ops::concat op;
    op.execute(&context);

    for (int e = 0; e < 10; e++)
        ASSERT_NEAR((float) e, z(e, {0}).meanNumber().e<float>(0), 1e-5f);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test20) {
    auto x0 = NDArrayFactory::create<double>('c', {1, 100, 150});
    auto x1 = NDArrayFactory::create<double>('c', {1, 100, 150});
    auto x2 = NDArrayFactory::create<double>('c', {1, 100, 150});
    auto x3 = NDArrayFactory::create<double>('c', {1, 100, 150});

    x0.assign(1.0);
    x1.assign(2.0);
    x2.assign(3.0);
    x3.assign(4.0);

    nd4j::ops::concat op;
    auto result = op.execute({&x0, &x1, &x2, &x3}, {}, {0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    Nd4jLong numOfTads= ShapeUtils::getNumOfSubArrs(z->getShapeInfo(), {0});
    ASSERT_TRUE(4 == numOfTads);

    for (int e = 0; e < numOfTads; e++) {
        NDArray tad  = (*z)(e, {0});
        auto mean = tad.meanNumber().e<double>(0);
        ASSERT_NEAR((double) e+1, mean, 1e-5);
    }

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test21) {

    NDArray x0('c', {1,4,5}, nd4j::DataType::FLOAT32);
    NDArray x1('c', {2,4,5}, nd4j::DataType::FLOAT32);
    NDArray  z('f', {3,4,5}, nd4j::DataType::FLOAT32);

    x0 = 0.;
    x1 = 1.;

    nd4j::ops::concat op;
    auto status = op.execute({&x0, &x1}, {&z}, {}, {0}, {});
    ASSERT_EQ(ND4J_STATUS_OK, status);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test22) {

    NDArray x0('c', {1,6}, {1,2,3,4,5,6});
    NDArray x1('c', {1,6}, {7,8,9,10,11,12});
    NDArray output('f', {2,6}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {2,6}, {1,2,3,4,5,6,7,8,9,10,11,12});

    nd4j::ops::concat op;

    auto status = op.execute({&x0, &x1}, {&output}, {}, {0}, {});
    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test23) {

    NDArray x0('c', {1,4}, {1,2,3,4});
    NDArray x1('c', {1,4}, {5,6,7,8});
    NDArray output('c', {2,4}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {2,4}, {1,2,3,4,5,6,7,8});

    nd4j::ops::concat op;

    auto status = op.execute({&x0, &x1}, {&output}, {}, {0}, {});
    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(exp.equalsTo(output));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test24) {
    auto x = NDArrayFactory::create<double>('c', {2, 1}, {1, 1});
    auto y = NDArrayFactory::create<double>('c', {2, 1}, {0, 0});
    auto e = NDArrayFactory::create<double>('c', {2, 2}, {1, 0, 1, 0});
    auto z = NDArrayFactory::create<double>('c', {2, 2});

    nd4j::ops::concat op;
    auto status = op.execute({&x, &y}, {&z}, {}, {1}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test25) {

    auto x0 = NDArrayFactory::create<double>('c', {1,4}, {1,2,3,4});
    auto x1 = NDArrayFactory::create<double>('c', {1,4}, {5,6,7,8});
    auto axis = NDArrayFactory::create<double>('c', {1}, {0.});
    auto exp = NDArrayFactory::create<double>('c', {2,4}, {1,2,3,4,5,6,7,8});

    nd4j::ops::concat op;

    auto result = op.execute({&x0, &x1, &axis}, {}, {}, {true});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test1) {

    auto input    = NDArrayFactory::create<double>('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    auto gradO    = NDArrayFactory::create<double>('c', {4, 9});
    auto gradIExp = NDArrayFactory::create<double>('c', {2, 3}, {0.78, 0.84, 0.9,1.32, 1.38, 1.44});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp op;
    auto results = op.execute({&input, &gradO}, {}, {2, 3});
    auto gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test2) {

    auto input    = NDArrayFactory::create<double>('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    auto gradO    = NDArrayFactory::create<double>('c', {2, 9});
    auto gradIExp = NDArrayFactory::create<double>('c', {2, 3}, {0.12, 0.15, 0.18, 0.39, 0.42, 0.45});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp op;
    auto results = op.execute({&input, &gradO}, {}, {1, 3});
    auto gradI = results->at(0);
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test3) {

    auto input    = NDArrayFactory::create<double>('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    auto gradO    = NDArrayFactory::create<double>('c', {2, 3});
    auto gradIExp = NDArrayFactory::create<double>('c', {2, 3}, {0.01, 0.02, 0.03,0.04, 0.05, 0.06});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp op;
    auto results = op.execute({&input, &gradO}, {}, {1, 1});
    auto gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test4) {

    auto input    = NDArrayFactory::create<double>('c', {6}, {1.,2.,3.,4.,5.,6.});
    auto gradO    = NDArrayFactory::create<double>('c', {12});
    auto gradIExp = NDArrayFactory::create<double>('c', {6}, {0.08, 0.1 , 0.12, 0.14, 0.16, 0.18});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp op;
    auto results = op.execute({&input, &gradO}, {}, {2});
    auto gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test5) {

    auto input    = NDArrayFactory::create<double>('c', {1}, {1.});
    auto gradO    = NDArrayFactory::create<double>('c', {1});
    auto gradIExp = NDArrayFactory::create<double>('c', {1}, {0.01});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp op;
    auto results = op.execute({&input, &gradO}, {}, {1});
    auto gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test6) {

    auto input    = NDArrayFactory::create<double>('c', {2, 1, 3}, {1.,2.,3.,4.,5.,6.});
    auto gradO    = NDArrayFactory::create<double>('c', {2, 3, 6});
    auto gradIExp = NDArrayFactory::create<double>('c', {2, 1, 3}, {0.51, 0.57, 0.63, 1.59, 1.65, 1.71});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp op;
    auto results = op.execute({&input, &gradO}, {}, {1, 3, 2});
    auto gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test7) {

    auto input    = NDArrayFactory::create<double>('c', {2, 1, 3}, {1.,2.,3.,4.,5.,6.});
    auto reps     = NDArrayFactory::create<int>('c', {1, 3}, {1, 3, 2});
    auto gradO    = NDArrayFactory::create<double>('c', {2, 3, 6});
    auto gradIExp = NDArrayFactory::create<double>('c', {2, 1, 3}, {0.51, 0.57, 0.63, 1.59, 1.65, 1.71});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp op;
    auto results = op.execute({&input, &reps, &gradO}, {}, {});
    auto gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_test1) {

    auto input  = NDArrayFactory::create<double>('c', {1, 6}, {1.,2.,3.,4.,5.,6.});
    auto reps   = NDArrayFactory::create<int>('c', {1, 2}, {2, 1});
    auto expOut = NDArrayFactory::create<double>('c', {2, 6,}, {1.,2.,3.,4.,5.,6., 1.,2.,3.,4.,5.,6.});

    nd4j::ops::tile op;
    auto results = op.execute({&input, &reps}, {}, {});
    auto out = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOut.isSameShape(out));
    ASSERT_TRUE(expOut.equalsTo(out));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test1) {

    auto x  = NDArrayFactory::create<double>('c', {3, 4});
    auto y  = NDArrayFactory::create<double>('c', {4, 3});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {35.,  79., 123., 40.,  92., 144., 45., 105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test2) {

    auto x  = NDArrayFactory::create<double>('c', {3, 4});
    auto y  = NDArrayFactory::create<double>('f', {4, 3});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test3) {

    auto x  = NDArrayFactory::create<double>('f', {3, 4});
    auto y  = NDArrayFactory::create<double>('c', {4, 3});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test4) {

    auto x = NDArrayFactory::create<double> ('f', {3, 4});
    auto y  = NDArrayFactory::create<double>('f', {4, 3});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test5) {

    auto x  = NDArrayFactory::create<double>('c', {4, 3});
    auto y  = NDArrayFactory::create<double>('c', {4, 3});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {83.,  94., 105., 94., 107., 120., 105., 120., 135.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test6) {

    auto x  = NDArrayFactory::create<double>('c', {4, 3});
    auto y  = NDArrayFactory::create<double>('f', {3, 4});
    auto exp = NDArrayFactory::create<double>('f', {3, 3}, {35.,  40.,  45., 79.,  92., 105., 123., 144., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test7) {

    auto x  = NDArrayFactory::create<double>('c', {5,  3,4});
    auto y  = NDArrayFactory::create<double>('f', {5,  3,4});
    auto exp = NDArrayFactory::create<double>('f',{5,  3,3}, {3. ,  84.6, 281.4, 593.4, 1020.6, 7. , 107.8, 323.8, 655. , 1101.4,11. , 131. , 366.2, 716.6, 1182.2,
                                        7. , 107.8, 323.8, 655. , 1101.4,17.4, 137.4, 372.6, 723. , 1188.6,27.8, 167. , 421.4, 791. , 1275.8,
                                       11. , 131. , 366.2, 716.6, 1182.2,27.8, 167. , 421.4, 791. , 1275.8,44.6, 203. , 476.6, 865.4, 1369.4,});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test8) {

    auto x  = NDArrayFactory::create<double>('c', {2,5,  3,4});
    auto y  = NDArrayFactory::create<double>('f', {2,5,  3,4});
    auto exp = NDArrayFactory::create<double>('f',{2,5,  3,3}, {3. , 1563. ,  84.6, 2220.6, 281.4, 2993.4, 593.4, 3881.4,1020.6, 4884.6,   7. , 1663. , 107.8, 2339.8, 323.8, 3131.8, 655. , 4039. ,1101.4, 5061.4,
                                          11. , 1763. , 131. , 2459. , 366.2, 3270.2, 716.6, 4196.6,1182.2, 5238.2,   7. , 1663. , 107.8, 2339.8, 323.8, 3131.8, 655. , 4039. ,1101.4, 5061.4,
                                          17.4, 1769.4, 137.4, 2465.4, 372.6, 3276.6, 723. , 4203. ,1188.6, 5244.6,  27.8, 1875.8, 167. , 2591. , 421.4, 3421.4, 791. , 4367. ,1275.8, 5427.8,
                                          11. , 1763. , 131. , 2459. , 366.2, 3270.2, 716.6, 4196.6,1182.2, 5238.2,  27.8, 1875.8, 167. , 2591. , 421.4, 3421.4, 791. , 4367. ,1275.8, 5427.8,
                                          44.6, 1988.6, 203. , 2723. , 476.6, 3572.6, 865.4, 4537.4,1369.4, 5617.4});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test9) {

    auto x  = NDArrayFactory::create<double>('c', {2,5,  4,3});
    auto y  = NDArrayFactory::create<double>('f', {2,5,  3,4});
    auto exp = NDArrayFactory::create<double>('f',{2,5,  3,3}, {7. , 1639. , 103. , 2311. , 314.2, 3098.2, 640.6, 4000.6,1082.2, 5018.2,   8. , 1664. , 108.8, 2340.8, 324.8, 3132.8, 656. , 4040. ,1102.4, 5062.4,
                                          9. , 1689. , 114.6, 2370.6, 335.4, 3167.4, 671.4, 4079.4,1122.6, 5106.6,  15.8, 1743.8, 131. , 2435. , 361.4, 3241.4, 707. , 4163. ,1167.8, 5199.8,
                                          18.4, 1770.4, 138.4, 2466.4, 373.6, 3277.6, 724. , 4204. ,1189.6, 5245.6,  21. , 1797. , 145.8, 2497.8, 385.8, 3313.8, 741. , 4245. ,1211.4, 5291.4,
                                          24.6, 1848.6, 159. , 2559. , 408.6, 3384.6, 773.4, 4325.4,1253.4, 5381.4,  28.8, 1876.8, 168. , 2592. , 422.4, 3422.4, 792. , 4368. ,1276.8, 5428.8,
                                          33. , 1905. , 177. , 2625. , 436.2, 3460.2, 810.6, 4410.6,1300.2, 5476.2});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, TestDropout_BP_1) {

    NDArray x('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    NDArray errs('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    NDArray shape('c', {2}, {2, 2});
    nd4j::ops::dropout_bp op;

    auto ress = op.execute({&x, &errs, &shape}, {0.2f}, {113});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    //ress->at(0)->printIndexedBuffer("Result is ");
    //x.printIndexedBuffer("Input is");
    ASSERT_FALSE(ress->at(0)->equalsTo(errs));
    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, TestDropout_1) {

    NDArray x('c', {10, 10}, nd4j::DataType::FLOAT32);
//    NDArray<float> errs('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    //NDArray<float> shape({2.f, 2.f});
    nd4j::ops::dropout op;
    x.linspace(1);
    auto ress = op.execute({&x}, {0.2f}, {113});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    NDArray* res = ress->at(0); //->printIndexedBuffer("Result is ");
    //x.printIndexedBuffer("Input is");
    //res->printIndexedBuffer("Result for Dropout_1");
    auto countZero = res->reduceNumber(reduce::CountZero);
    ASSERT_NEAR(countZero.e<Nd4jLong>(0), 80, 5);
    auto ress2 = op.execute({&x}, {0.2f}, {113});

    ASSERT_EQ(ND4J_STATUS_OK, ress2->status());
    NDArray* res2 = ress2->at(0);

    countZero = res->reduceNumber(reduce::CountZero);
    ASSERT_NEAR(countZero.e<Nd4jLong>(0), 80, 5);
    //res2->printIndexedBuffer("Result for Dropout_2");
    ASSERT_TRUE(res->equalsTo(res2));
    //res->printIndexedBuffer("FF dropout");
    //res2->printIndexedBuffer("BP dropout");

    delete ress;
    delete ress2;
}

TEST_F(DeclarableOpsTests9, Test_DropoutInverted_01) {
    NDArray x0('c', {10, 10}, nd4j::DataType::FLOAT32);
    NDArray x1('c', {10, 10}, nd4j::DataType::FLOAT32);

    x0.linspace(1);
    x1.linspace(1);
/*
    float prob[] = {0.5f};
    Nd4jLong* _bufferA = new Nd4jLong[100000];
    long _seed = 119L;
    auto _rngA = (nd4j::random::RandomBuffer *) initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferA);

    x0. applyTransform(random::DropOutInverted, &x0, prob);
//    x1.template applyRandom<randomOps::DropOutInverted<float>>(_rngB, nullptr, &x1, prob);
//    x0.printIndexedBuffer("01Result1");
    int count = 0;
    for (int e = 0; e < x0.lengthOf(); e++)
        if (x0.e<float>(e) != 0.f)
            count++;
//    nd4j_printf("\nX0 count %i\n", count);
//    ASSERT_TRUE(x0.equalsTo(&x1));

    // this check is required to ensure we're calling wrong signature
//    ASSERT_FALSE(x0.equalsTo(nexp0));
//    ASSERT_FALSE(x0.equalsTo(nexp1));
//    ASSERT_FALSE(x0.equalsTo(nexp2));
    destroyRandom(_rngA);
    delete [] _bufferA;
*/
    nd4j::ops::dropout op;

    auto ress = op.execute({&x1}, {0.5f}, {119});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    //ress->at(0)->printIndexedBuffer("01Dropout result is ");
    auto count = ress->at(0)->reduceNumber(reduce::CountNonZero);
//    nd4j_printf("\n01Dropout count %i\n\n", count);

    nd4j::ops::dropout_bp op2;
    //NDArray<float> exp('c', {10,10}, {4.f, 0.f, 12.f, 0.f, 20.f, 24.f, 0.f, 32.f, 0.f, 0.f, 0.f, 0.f, 52.f, 56.f, 60.f, 0.f, 0.f, 0.f, 0.f, 0.f, 84.f, 88.f, 0.f, 0.f, 0.f, 0.f, 108.f, 0.f, 0.f, 120.f, 0.f, 0.f, 132.f, 0.f, 0.f, 0.f, 0.f, 0.f, 156.f, 0.f, 164.f, 168.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 200.f, 204.f, 0.f, 0.f, 0.f, 220.f, 0.f, 0.f, 232.f, 236.f, 240.f, 0.f, 248.f, 0.f, 0.f, 260.f, 0.f, 0.f, 0.f, 276.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 316.f, 0.f, 324.f, 0.f, 0.f, 336.f, 0.f, 0.f, 0.f, 0.f, 356.f, 0.f, 0.f, 368.f, 0.f, 0.f, 0.f, 384.f, 388.f, 0.f, 0.f, 400.f});
    //02Dropout result is  [4.000000, 0.000000, 12.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 36.000000, 0.000000, 0.000000, 0.000000, 0.000000, 56.000000, 60.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 88.000000, 0.000000, 96.000000, 0.000000, 0.000000, 108.000000, 0.000000, 0.000000, 120.000000, 0.000000, 128.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 156.000000, 0.000000, 164.000000, 0.000000, 0.000000, 0.000000, 0.000000, 184.000000, 0.000000, 0.000000, 0.000000, 200.000000, 0.000000, 0.000000, 0.000000, 216.000000, 0.000000, 0.000000, 0.000000, 232.000000, 0.000000, 240.000000, 0.000000, 248.000000, 0.000000, 0.000000, 260.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 308.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 348.000000, 0.000000, 356.000000, 0.000000, 0.000000, 0.000000, 0.000000, 376.000000, 0.000000, 384.000000, 0.000000, 0.000000, 0.000000, 400.000000]

    auto ressX = op2.execute({&x1, &x1}, {0.5f}, {119}); // , false, nd4j::DataType::FLOAT32); // skipped due given by default
    //x0.printIndexedBuffer("X0");
    //x1.printIndexedBuffer("X1");
    ASSERT_EQ(ND4J_STATUS_OK, ressX->status());
    auto ressY = op2.execute({&x1, &x0}, {0.5f}, {119});
    ASSERT_EQ(ND4J_STATUS_OK, ressY->status());
    //ressY->at(0)->printIndexedBuffer("BP");
    //ress->at(0)->printIndexedBuffer("FF");
    bool ret = true;
    for (int e = 0; e < ress->at(0)->lengthOf(); e++) {
        if (ress->at(0)->e<float>(e) == 0.f)
            if (ressX->at(0)->e<float>(e) != ress->at(0)->e<float>(e)) {
                ret = false;
                break;
            }
    }
    ASSERT_TRUE(ret);
    //    ASSERT_FALSE(ressX->at(0)->equalsTo(ressY->at(0)));
    //ressX->at(0)->printIndexedBuffer("02Dropout result is ");
/*    float countZero = ressX->at(0)->template reduceNumber<simdOps::CountZero<float>>();
    ASSERT_NEAR(countZero, 50.f, 5.f);
    countZero = ress->at(0)->template reduceNumber<simdOps::CountZero<float>>();
    ASSERT_NEAR(countZero, 50.f, 5.f);
    countZero = ressY->at(0)->template reduceNumber<simdOps::CountZero<float>>();
    ASSERT_NEAR(countZero, 50.f, 5.f);
    */
//    ASSERT_TRUE(exp.equalsTo(ressX->at(0)));
    delete ressX;
    delete ressY;
    delete ress;
}

TEST_F(DeclarableOpsTests9, Test_Dropout_BP_2) {
    NDArray x('c', {10, 10}, nd4j::DataType::FLOAT32);

    x.linspace(1);

    nd4j::ops::dropout op;

    auto ress = op.execute({&x}, {0.5f}, {119});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
//    ress->at(0)->printIndexedBuffer("01Dropout result is ");

    nd4j::ops::dropout_bp op2;

    auto ressX = op2.execute({&x, &x}, {0.5f}, {119});

    ASSERT_EQ(ND4J_STATUS_OK, ressX->status());
    auto ressY = op2.execute({&x, &x}, {0.5f}, {119});
    ASSERT_EQ(ND4J_STATUS_OK, ressY->status());

    //ress->at(0)->printIndexedBuffer("FF Dropout result is ");
    //ressY->at(0)->printIndexedBuffer("BP Dropout result is ");


    auto countZero = ress->at(0)->reduceNumber(reduce::CountZero);
    ASSERT_NEAR(countZero.e<float>(0), 50.f, 10.f);
    countZero = ressX->at(0)->reduceNumber(reduce::CountZero);
    //nd4j_printf("X zero count is %f\n", countZero);
    ASSERT_NEAR(countZero.e<float>(0), 50.f, 10.f);
    countZero = ressY->at(0)->reduceNumber(reduce::CountZero);
    //nd4j_printf("Y zero count is %f\n", countZero);
    ASSERT_NEAR(countZero.e<float>(0), 50.f, 10.f);
//    ASSERT_TRUE(exp.equalsTo(ressX->at(0)));
    ASSERT_TRUE(ressX->at(0)->equalsTo(ressY->at(0)));
    delete ressX;
    delete ressY;
    delete ress;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, Test_AlphaDropout_BP_1) {
    NDArray x('c', {10, 10}, nd4j::DataType::FLOAT32);
    NDArray eps('c', {10, 10}, nd4j::DataType::FLOAT32);

    x.linspace(1);
    eps.linspace(1);

    nd4j::ops::alpha_dropout_bp op;

    auto ress = op.execute({&x, &eps}, {0.5f, 0.5f, 1.5f, 1.6f}, {119});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    NDArray* res = ress->at(0);

    auto ress2 = op.execute({&x, &eps}, {0.5f, 0.5f, 1.5f, 1.6f}, {119});

    ASSERT_EQ(ND4J_STATUS_OK, ress2->status());
    NDArray* res2 = ress2->at(0);
    //res->printIndexedBuffer("Result1AlphaBP1");
    //res2->printIndexedBuffer("Result1AlphaBP2");
    ASSERT_TRUE(res2->equalsTo(res));


    delete ress;
    delete ress2;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test10) {

    auto x  = NDArrayFactory::create<double>('c', {1, 4, 3});
    auto y  = NDArrayFactory::create<double>('f', {1, 3, 4});
    auto exp = NDArrayFactory::create<double>('f', {1, 3, 3}, {35.,  40.,  45., 79.,  92., 105., 123., 144., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test11) {

    auto x  = NDArrayFactory::create<double>('c', {4, 1});
    auto y  = NDArrayFactory::create<double>('f', {1, 4});
    auto exp = NDArrayFactory::create<double>('f', {1, 1}, {15});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1, 1});
    ASSERT_EQ(Status::OK(), results->status());

    auto z = results->at(0);
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test12) {

    auto x  = NDArrayFactory::create<double>('c', {1, 4, 1});
    auto y  = NDArrayFactory::create<double>('f', {1, 1, 4});
    auto exp = NDArrayFactory::create<double>('f', {1, 1, 1}, {15});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1, 1});

    ASSERT_EQ(Status::OK(), results->status());
    auto z = results->at(0);
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test13) {

    auto x  = NDArrayFactory::create<double>('c', {2, 3});
    auto y  = NDArrayFactory::create<double>('c', {3, 5});
    auto exp = NDArrayFactory::create<double>('f', {5, 2}, {23. , 26. , 29. , 32. , 35., 50. , 57.5, 65. , 72.5, 80.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {0, 0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test14) {

    auto x  = NDArrayFactory::create<double>('c', {3, 2});
    auto y  = NDArrayFactory::create<double>('c', {3, 5});
    auto exp = NDArrayFactory::create<double>('f', {5, 2}, {37. , 41.5, 46. , 50.5, 55., 46. , 52. , 58. , 64. , 70.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1, 0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test15) {

    auto x  = NDArrayFactory::create<double>('c', {3, 2});
    auto y  = NDArrayFactory::create<double>('c', {3, 5});
    auto exp = NDArrayFactory::create<double>('f', {5, 2}, {37. , 41.5, 46. , 50.5, 55., 46. , 52. , 58. , 64. , 70.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1, 0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test16) {

    auto x  = NDArrayFactory::create<double>('c', {2,2,  3,5});
    auto y  = NDArrayFactory::create<double>('c', {2,2,  4,3});
    auto exp = NDArrayFactory::create<double>('f',{2,2,  4,5}, {4.6, 281.8, 89.2, 582.4, 10. , 314.2,108.1, 628.3, 15.4, 346.6,127. , 674.2, 20.8, 379. ,145.9, 720.1,  5.2, 289.6, 93.4, 593.8,
                                          11.5, 322.9,113.2, 640.6, 17.8, 356.2,133. , 687.4, 24.1, 389.5,152.8, 734.2,  5.8, 297.4, 97.6, 605.2, 13. , 331.6,118.3, 652.9,
                                          20.2, 365.8,139. , 700.6, 27.4, 400. ,159.7, 748.3,  6.4, 305.2,101.8, 616.6, 14.5, 340.3,123.4, 665.2, 22.6, 375.4,145. , 713.8,
                                          30.7, 410.5,166.6, 762.4,  7. , 313. ,106. , 628. , 16. , 349. ,128.5, 677.5, 25. , 385. ,151. , 727. , 34. , 421. ,173.5, 776.5});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1, 1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test17) {

    auto x  = NDArrayFactory::create<double>('f', {4, 3});
    auto y  = NDArrayFactory::create<double>('c', {4});
    auto exp = NDArrayFactory::create<double>('f',{3}, {7., 8., 9.});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1, 0});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test18) {

    auto x  = NDArrayFactory::create<double>('f', {3});
    auto y  = NDArrayFactory::create<double>('c', {4, 3});
    auto exp = NDArrayFactory::create<double>('f',{4}, {1.4, 3.2, 5., 6.8});

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {0, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test19) {

    auto x  = NDArrayFactory::create<double>('f', {1, 1});
    auto y  = NDArrayFactory::create<double>('c', {1, 1});
    auto exp = NDArrayFactory::create<double>('f',{1, 1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test20) {

    auto x  = NDArrayFactory::create<double>('f', {1, 1});
    auto y  = NDArrayFactory::create<double>('c', {1, 1});
    auto exp = NDArrayFactory::create<double>('f',{1, 1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1,1,1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test21) {

    auto x  = NDArrayFactory::create<double>('f', {1});
    auto y  = NDArrayFactory::create<double>('c', {1, 1});
    auto exp = NDArrayFactory::create<double>('f',{1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test22) {

    auto x  = NDArrayFactory::create<double>('f', {1,1});
    auto y  = NDArrayFactory::create<double>('c', {1});
    auto exp = NDArrayFactory::create<double>('f',{1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test23) {

    auto x  = NDArrayFactory::create<double>('f', {4});
    auto y  = NDArrayFactory::create<double>('c', {4});
    auto exp = NDArrayFactory::create<double>(3.);

    x.linspace(1.);
    y.linspace(0.1, 0.1);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test24) {

    auto x  = NDArrayFactory::create<double>('f', {1}, {2.});
    auto y  = NDArrayFactory::create<double>('c', {1}, {3.});
    auto exp = NDArrayFactory::create<double>(6.);

    nd4j::ops::matmul op;
    auto results = op.execute({&x, &y}, {}, {1, 1});
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

TEST_F(DeclarableOpsTests9, test_range_int_1) {
    auto x0 = NDArrayFactory::create<int>(0);
    auto x1 = NDArrayFactory::create<int>(2);
    auto x2 = NDArrayFactory::create<int>(1);

    nd4j::ops::range op;
    auto result = op.execute({&x0, &x1, &x2}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    delete result;
}

TEST_F(DeclarableOpsTests9, test_range_empty_1) {
    auto x0 = NDArrayFactory::create<int>(0);
    auto x1 = NDArrayFactory::create<int>(0);
    auto x2 = NDArrayFactory::create<int>(1);

    nd4j::ops::range op;
    auto result = op.execute({&x0, &x1, &x2}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(z->isEmpty());

    delete result;
}


TEST_F(DeclarableOpsTests9, test_broadcast_bool_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 3, 2, 4, 4});
    auto y = NDArrayFactory::create<double>('c', {1, 2, 4, 4});
    auto z = NDArrayFactory::create<bool>('c', {1, 3, 2, 4, 4});

    std::vector<int> dims = {0, 2, 3, 4};
    x.applyBroadcast(broadcast::LessThan, dims, &y, &z, nullptr);
}

TEST_F(DeclarableOpsTests9, test_broadcast_bool_2) {
    auto orig = NDArrayFactory::create<double>('c', {1, 7, 4, 4});
    std::vector<Nd4jLong>  list = {0,0, 0,2, 0,0, 0,0};
    auto x = NDArrayFactory::create<double>('c', {1, 3, 2, 4, 4});

    auto y = orig(list, true);

    auto z = NDArrayFactory::create<bool>('c', {1, 3, 2, 4, 4});

    std::vector<int> dims = {0, 2, 3, 4};
    x.applyBroadcast(broadcast::LessThan, dims, &y, &z, nullptr);

}

TEST_F(DeclarableOpsTests9, test_unstack_1) {
    auto x = NDArrayFactory::create<double>('c', {5, 5});
    x.linspace(1.0);

    nd4j::ops::unstack op;
    auto result = op.execute({&x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_EQ(5, result->size());

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, test_unstack_SGO_1) {
    auto x = NDArrayFactory::create<double>({1, 2, 3, 4, 5});
    x.linspace(1.0);
    auto z1 = NDArrayFactory::create<double>(1);
    auto z2 = NDArrayFactory::create<double>(2);
    auto z3 = NDArrayFactory::create<double>(3);
    auto z4 = NDArrayFactory::create<double>(4);
    auto z5 = NDArrayFactory::create<double>(5);
    std::vector<NDArray*> z({&z1, &z2, &z3, &z4, &z5});
    nd4j::ops::unstack op;
    auto result = op.execute({&x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_EQ(5, result->size());
    for (size_t i = 0; i < result->size(); i++) {
        ASSERT_TRUE(result->at(i)->isSameShape(z[i]));
        ASSERT_TRUE(result->at(i)->equalsTo(z[i]));
    }
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, clipbynorm_test12) {

    const int bS   = 5;
    const int nOut = 4;
    const int axis = 0;
    const double clip = 2.;

    auto x = NDArrayFactory::create<double>('c', {bS, nOut}, {0.412 ,0.184 ,0.961 ,0.897 ,0.173 ,0.931 ,0.736 ,0.540 ,0.953 ,0.278 ,0.573 ,0.787 ,0.320 ,0.776 ,0.338 ,0.311 ,0.835 ,0.909 ,0.890 ,0.290});    // uniform random in range [0,1]
    auto colVect = NDArrayFactory::create<double>('c', {bS, 1}, {0.9, 0.95, 1.00, 1.05, 1.1});
    auto expect = NDArrayFactory::create<double>('c', {bS, nOut});

    auto norm2 = x.reduceAlongDims(reduce::Norm2, {axis}, true); // norm2 has shape [1, nOut]

    auto y = ( (x / norm2) * clip) * colVect ;
    auto temp = (x / norm2) * clip;

    for (int j = 0; j < nOut; ++j) {
        auto yCol = y({0,0, j,j+1});
        const double norm2Col = yCol.reduceNumber(reduce::Norm2).e<double>(0);
        if (norm2Col <= clip)
            expect({0,0, j,j+1}).assign(yCol);
        else
            expect({0,0, j,j+1}).assign ( yCol * (clip / norm2Col) );
    }

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&y}, {clip}, {axis}, {}, false, nd4j::DataType::DOUBLE);
    auto outFF = result->at(0);

    ASSERT_TRUE(expect.isSameShape(outFF));
    ASSERT_TRUE(expect.equalsTo(outFF));

    delete result;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, clipbynorm_bp_test1) {

    const int bS   = 2;
    const int nOut = 3;
    const double clip = 0.7;

    auto x = NDArrayFactory::create<double>('c', {bS, nOut}, {0.412 ,0.184 ,0.961 ,0.173 ,0.736 ,0.540 });    // uniform random in range [0,1]
    auto gradO = NDArrayFactory::create<double>('c', {bS, nOut});

    const OpArgsHolder argsHolderFF({&x}, {clip}, {});
    const OpArgsHolder argsHolderBP({&x, &gradO}, {clip}, {});

    nd4j::ops::clipbynorm opFF;
    nd4j::ops::clipbynorm_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, clipbynorm_bp_test2) {

    const int bS   = 2;
    const int nOut = 3;
    const int axis = 0;
    const double clip = 0.7;

    auto x = NDArrayFactory::create<double>('c', {bS, nOut}, {0.412 ,0.184 ,0.961 ,0.173 ,0.736 ,0.540 });    // uniform random in range [0,1]
    auto gradO = NDArrayFactory::create<double>('c', {bS, nOut});

    const OpArgsHolder argsHolderFF({&x}, {clip}, {axis});
    const OpArgsHolder argsHolderBP({&x, &gradO}, {clip}, {axis});

    nd4j::ops::clipbynorm opFF;
    nd4j::ops::clipbynorm_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, clipbynorm_bp_test3) {

    const int bS   = 2;
    const int nOut = 3;
    const int axis = 1;
    const double clip = 1.;

    auto x = NDArrayFactory::create<double>('c', {bS, nOut}, {0.412 ,0.184 ,0.961 ,0.173 ,0.736 ,0.540 });    // uniform random in range [0,1]
    auto gradO = NDArrayFactory::create<double>('c', {bS, nOut});

    const OpArgsHolder argsHolderFF({&x}, {clip}, {axis});
    const OpArgsHolder argsHolderBP({&x, &gradO}, {clip}, {axis});

    nd4j::ops::clipbynorm opFF;
    nd4j::ops::clipbynorm_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_1) {

    auto inputC = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
    auto axis = NDArrayFactory::create<Nd4jLong>(1);

    auto expFF = NDArrayFactory::create<double>('c', {3, 5}, {1.,   2.,   6.,    24.,   120., 6.,  42., 336.,  3024., 30240.,11., 132.,1716., 24024.,360360.});
    auto expTF = NDArrayFactory::create<double>('c', {3, 5}, {1, 1, 2, 6, 24,1, 6, 42, 336, 3024,1, 11, 132, 1716, 24024});

    auto expFT = NDArrayFactory::create<double>('c', {3, 5}, {120, 120, 60, 20, 5,30240, 5040, 720, 90, 10,360360, 32760, 2730, 210, 15});    //+++
    auto expTT = NDArrayFactory::create<double>('c', {3, 5}, {120, 60, 20, 5, 1,5040, 720, 90, 10, 1,32760, 2730, 210, 15, 1});

    int exclusive, reverse;

    //************************************//
    exclusive = 0; reverse = 0;

    nd4j::ops::cumprod op;
    auto result = op.execute({&inputC, &axis}, {}, {exclusive, reverse}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());
    auto z = result->at(0);
    ASSERT_TRUE(expFF.equalsTo(z));
    delete result;

    //************************************//
    exclusive = 1; reverse = 0;

    result = op.execute({&inputC, &axis}, {}, {exclusive, reverse}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());
    z = result->at(0);
    ASSERT_TRUE(expTF.equalsTo(z));
    delete result;

    //************************************//
    exclusive = 0; reverse = 1;

    result = op.execute({&inputC, &axis}, {}, {exclusive, reverse}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());
    z = result->at(0);
    ASSERT_TRUE(expFT.equalsTo(z));
    delete result;

    //************************************//
    exclusive = 1; reverse = 1;

    result = op.execute({&inputC, &axis}, {}, {exclusive, reverse}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), result->status());
    z = result->at(0);
    ASSERT_TRUE(expTT.equalsTo(z));
    delete result;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_2) {

    NDArray x('c', {2, 1500}, nd4j::DataType::FLOAT32);
    NDArray x0 = x(0, {0});
    NDArray x1 = x(1, {0});
    x0.linspace(1, 0.1);
    x1.linspace(1, 0.1);

    NDArray exp('c', {2, 1500}, nd4j::DataType::FLOAT32);
    NDArray exp0 = exp(0, {0});
    NDArray exp1 = exp(1, {0});

    exp0.p(0, 1.f);
    exp1.p(0, 1.f);

    for (int i = 1; i < 1500; ++i) {
        const auto prev = exp0.e<float>(i-1);
        exp0.p(i, prev * x0.e<float>(i));
        exp1.p(i, prev * x1.e<float>(i));
    }

    nd4j::ops::cumprod op;
    auto result = op.execute({&x}, {}, {0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_bp_check_1) {

    auto       x = NDArrayFactory::create<double>('c', {4, 4});
    auto   gradO = NDArrayFactory::create<double>('c', {4, 4});

    x.linspace(1);

    const OpArgsHolder argsHolderFF({&x},         {}, {0, 0});
    const OpArgsHolder argsHolderBP({&x, &gradO}, {}, {0, 0});

    nd4j::ops::cumprod opFF;
    nd4j::ops::cumprod_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_bp_check_2) {

    auto       x = NDArrayFactory::create<double>('c', {4, 4});
    auto   gradO = NDArrayFactory::create<double>('c', {4, 4});

    x.linspace(1);

    const OpArgsHolder argsHolderFF({&x},         {}, {1, 1});
    const OpArgsHolder argsHolderBP({&x, &gradO}, {}, {1, 1});

    nd4j::ops::cumprod opFF;
    nd4j::ops::cumprod_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_bp_check_3) {

    auto       x = NDArrayFactory::create<double>('c', {4, 4});
    auto   gradO = NDArrayFactory::create<double>('c', {4, 4});

    x.linspace(1);

    const OpArgsHolder argsHolderFF({&x},         {}, {1, 0});
    const OpArgsHolder argsHolderBP({&x, &gradO}, {}, {1, 0});

    nd4j::ops::cumprod opFF;
    nd4j::ops::cumprod_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_bp_check_4) {

    auto       x = NDArrayFactory::create<double>('c', {4, 4});
    auto   gradO = NDArrayFactory::create<double>('c', {4, 4});

    x.linspace(1);

    const OpArgsHolder argsHolderFF({&x},         {}, {0, 1});
    const OpArgsHolder argsHolderBP({&x, &gradO}, {}, {0, 1});

    nd4j::ops::cumprod opFF;
    nd4j::ops::cumprod_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumsum_bp_check_2) {

    auto       x = NDArrayFactory::create<double>('c', {4, 4});
    auto   gradO = NDArrayFactory::create<double>('c', {4, 4});

    x.linspace(1);

    const OpArgsHolder argsHolderFF({&x},         {}, {1, 1});
    const OpArgsHolder argsHolderBP({&x, &gradO}, {}, {1, 1});

    nd4j::ops::cumsum opFF;
    nd4j::ops::cumsum_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_test1) {

    auto inputC = NDArrayFactory::create<double>('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});
    auto axis = NDArrayFactory::create<double>(1.);

    auto expFF = NDArrayFactory::create<double>('c', {3, 5}, {1.,   2.,   6.,    24.,   120., 6.,  42., 336.,  3024., 30240.,11., 132.,1716., 24024.,360360.});
    auto expTF = NDArrayFactory::create<double>('c', {3, 5}, {1, 1, 2, 6, 24,1, 6, 42, 336, 3024,1, 11, 132, 1716, 24024});

    auto expFT = NDArrayFactory::create<double>('c', {3, 5}, {120, 120, 60, 20, 5,30240, 5040, 720, 90, 10,360360, 32760, 2730, 210, 15});    //+++
    auto expTT = NDArrayFactory::create<double>('c', {3, 5}, {120, 60, 20, 5, 1,5040, 720, 90, 10, 1,32760, 2730, 210, 15, 1});
    auto gradO = NDArrayFactory::create<double>('c', {3, 5});

    int exclusive, reverse;

    //************************************//
    exclusive = 0; reverse = 0;

    const OpArgsHolder argsHolderFF({&inputC, &axis}, {}, {exclusive, reverse});
    const OpArgsHolder argsHolderBP({&inputC, &axis, &gradO}, {}, {exclusive, reverse});

    nd4j::ops::cumprod opFF;
    nd4j::ops::cumprod_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);

    //************************************//
/*    exclusive = 1; reverse = 0;

    result = op.execute({&inputC, &axis}, {}, {exclusive, reverse});
    ASSERT_EQ(Status::OK(), result->status());
    z = result->at(0);
    ASSERT_TRUE(expTF.equalsTo(z));
    delete result;
*/
    //************************************//
/*    exclusive = 0; reverse = 1;

    result = op.execute({&inputC, &axis}, {}, {exclusive, reverse});
    ASSERT_EQ(Status::OK(), result->status());
    z = result->at(0);
    ASSERT_TRUE(expFT.equalsTo(z));
    delete result;
*/
    //************************************//
/*    exclusive = 1; reverse = 1;

    result = op.execute({&inputC, &axis}, {}, {exclusive, reverse});
    ASSERT_EQ(Status::OK(), result->status());
    z = result->at(0);
    ASSERT_TRUE(expTT.equalsTo(z));
    delete result;
*/
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_test2) {

    auto inputC = NDArrayFactory::create<double>('c', {2, 2});
    auto axis = NDArrayFactory::create<double>(1.);

//    auto expFF = NDArrayFactory::create<double>('c', {3, 5}, {1.,   2.,   6.,    24.,   120., 6.,  42., 336.,  3024., 30240.,11., 132.,1716., 24024.,360360.});
//    auto expTF = NDArrayFactory::create<double>('c', {3, 5}, {1, 1, 2, 6, 24,1, 6, 42, 336, 3024,1, 11, 132, 1716, 24024});

//    auto expFT = NDArrayFactory::create<double>('c', {3, 5}, {120, 120, 60, 20, 5,30240, 5040, 720, 90, 10,360360, 32760, 2730, 210, 15});    //+++
//    auto expTT = NDArrayFactory::create<double>('c', {3, 5}, {120, 60, 20, 5, 1,5040, 720, 90, 10, 1,32760, 2730, 210, 15, 1});
    auto gradO = NDArrayFactory::create<double>('c', {2, 2});

    int exclusive, reverse;

    //************************************//
    exclusive = 0; reverse = 0;
    inputC.linspace(1);
    const OpArgsHolder argsHolderFF({&inputC, &axis}, {}, {exclusive, reverse});
    const OpArgsHolder argsHolderBP({&inputC, &axis, &gradO}, {}, {exclusive, reverse});

    nd4j::ops::cumprod opFF;
    nd4j::ops::cumprod_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1, 1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    auto alpha = NDArrayFactory::create<double>('c', {3, 4}, {-0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {7.2f, 5.5f, 4.f, 2.7f, 1.6f, 0.7f, 0.f, -0.5f,-0.8f, -0.9f, -0.8f, -0.5f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});

    nd4j::ops::prelu op;

    auto result = op.execute({&x, &alpha}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    auto alpha = NDArrayFactory::create<double>('c', {3}, {-0.6f, 2.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {7.2f,  6.6f,   6.f,   5.4f, -16.f, -14.f, -12.f, -10.f, -16.f, -12.f,  -8.f,  -4.f, 0.f,   1.f,   2.f,   3.f, 4.f,   5.f,   6.f,   7.f, 8.f,   9.f,  10.f,  11.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    auto alpha = NDArrayFactory::create<double>('c', {3,1}, {-0.6f, 2.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {7.2f,  6.6f,   6.f,   5.4f, -16.f, -14.f, -12.f, -10.f, -16.f, -12.f,  -8.f,  -4.f, 0.f,   1.f,   2.f,   3.f, 4.f,   5.f,   6.f,   7.f, 8.f,   9.f,  10.f,  11.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    auto alpha = NDArrayFactory::create<double>('c', {1, 3}, {-0.6f, 2.f, 4.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {7.2f,  6.6f,   6.f,   5.4f, -16.f, -14.f, -12.f, -10.f, -16.f, -12.f,  -8.f,  -4.f, 0.f,   1.f,   2.f,   3.f, 4.f,   5.f,   6.f,   7.f, 8.f,   9.f,  10.f,  11.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test5) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    auto alpha = NDArrayFactory::create<double>('c', {4}, {-0.6f, 2.f, 4.f, -1.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {7.2f, -22.f, -40.f,   9.f, 4.8f, -14.f, -24.f,   5.f, 2.4f,  -6.f,  -8.f,   1.f, 0.f,   1.f,   2.f,   3.f, 4.f,   5.f,   6.f,   7.f, 8.f,   9.f,  10.f,  11.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {1}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test6) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    auto alpha = NDArrayFactory::create<double>('c', {1,1,1}, {-2.});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {24.f, 22.f, 20.f, 18.f, 16.f, 14.f, 12.f, 10.f, 8.f,  6.f,  4.f,  2.f, 0.f,  1.f,  2.f,  3.f, 4.f,  5.f,  6.f,  7.f, 8.f,  9.f, 10.f, 11.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {1,0}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test7) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    auto alpha = NDArrayFactory::create<double>(-2.f);
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {24.f, 22.f, 20.f, 18.f, 16.f, 14.f, 12.f, 10.f, 8.f,  6.f,  4.f,  2.f, 0.f,  1.f,  2.f,  3.f, 4.f,  5.f,  6.f,  7.f, 8.f,  9.f, 10.f, 11.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {1,0}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test8) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    auto alpha = NDArrayFactory::create<double>(-2.f);
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {24.f, 22.f, 20.f, 18.f, 16.f, 14.f, 12.f, 10.f, 8.f,  6.f,  4.f,  2.f, 0.f,  1.f,  2.f,  3.f, 4.f,  5.f,  6.f,  7.f, 8.f,  9.f, 10.f, 11.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {1,0,1,0,1,0}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test9) {

    auto x = NDArrayFactory::create<double>('c', {2, 4}, {-4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f});
    auto alpha = NDArrayFactory::create<double>(-2.f);
    auto exp = NDArrayFactory::create<double>('c', {2, 4}, {8.f, 6.f, 4.f, 2.f,0.f, 1.f, 2.f, 3.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test10) {

    auto x = NDArrayFactory::create<double>('c', {2, 4}, {-4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f});
    auto alpha = NDArrayFactory::create<double>(-2.f);
    auto exp = NDArrayFactory::create<double>('c', {2, 4}, {8.f, 6.f, 4.f, 2.f,0.f, 1.f, 2.f, 3.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {1}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test11) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4, 5});
    x.linspace(-50.);
    auto alpha = NDArrayFactory::create<double>('c', {4}, {0.f, -0.5f, 0.5f, -1.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4, 5}, {0.f,   0.f,   0.f,   0.f,   0.f, 22.5f,  22.f,  21.5f,  21.f,  20.5f, -20.f, -19.5f, -19.f, -18.5f, -18.f, 35.f,  34.f,  33.f,
                                           32.f,  31.f, 0.f,   0.f,   0.f,   0.f,   0.f, 12.5f,  12.f,  11.5f,  11.f,  10.5f, -10.f,  -9.5f,  -9.f,  -8.5f,  -8.f, 15.f,
                                           14.f,  13.f,  12.f,  11.f, 0.f,   0.f,   0.f,   0.f,   0.f, 2.5f,   2.f,   1.5f,   1.f,   0.5f, 0.f,   1.f,   2.f,   3.f,   4.f,
                                           5.f,   6.f,   7.f,   8.f,   9.f, 10.f,  11.f,  12.f,  13.f,  14.f, 15.f,  16.f,  17.f,  18.f,  19.f, 20.f,  21.f,  22.f,  23.f,
                                           24.f, 25.f,  26.f,  27.f,  28.f,  29.f, 30.f,  31.f,  32.f,  33.f,  34.f, 35.f,  36.f,  37.f,  38.f,  39.f, 40.f,  41.f,  42.f,
                                           43.f,  44.f, 45.f,  46.f,  47.f,  48.f,  49.f, 50.f,  51.f,  52.f,  53.f,  54.f, 55.f,  56.f,  57.f,  58.f,  59.f, 60.f,  61.f,
                                           62.f,  63.f,  64.f, 65.f,  66.f,  67.f,  68.f,  69.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {1,3}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test12) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4, 5});
    x.linspace(-50.);
    auto alpha = NDArrayFactory::create<double>('c', {3,5}, {-0.7f, -0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4, 5}, {35.f, 29.4f, 24.f, 18.8f, 13.8f, 31.5f, 26.4f, 21.5f, 16.8f, 12.3f, 28.f, 23.4f, 19.f, 14.8f, 10.8f, 24.5f, 20.4f, 16.5f, 12.8f,
                                           9.3f, 6.f,  2.9f,  0.f, -2.7f, -5.2f, 5.f,  2.4f,  0.f, -2.2f, -4.2f, 4.f,  1.9f,  0.f, -1.7f, -3.2f, 3.f,  1.4f,  0.f, -1.2f,
                                           -2.2f, -3.f, -3.6f, -4.f, -4.2f, -4.2f, -1.5f, -1.6f, -1.5f, -1.2f, -0.7f, 0.f,  1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f,
                                           9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f,
                                           31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 49.f, 50.f, 51.f, 52.f,
                                           53.f, 54.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 67.f, 68.f, 69.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {-1, 2}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test13) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4, 5});
    x.linspace(-50.);
    auto alpha = NDArrayFactory::create<double>('c', {5,3}, {-0.7f, -0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4, 5}, {35.f, 29.4f, 24.f, 18.8f, 13.8f, 31.5f, 26.4f, 21.5f, 16.8f, 12.3f, 28.f, 23.4f, 19.f, 14.8f, 10.8f, 24.5f, 20.4f, 16.5f, 12.8f,
                                           9.3f, 6.f,  2.9f,  0.f, -2.7f, -5.2f, 5.f,  2.4f,  0.f, -2.2f, -4.2f, 4.f,  1.9f,  0.f, -1.7f, -3.2f, 3.f,  1.4f,  0.f, -1.2f,
                                           -2.2f, -3.f, -3.6f, -4.f, -4.2f, -4.2f, -1.5f, -1.6f, -1.5f, -1.2f, -0.7f, 0.f,  1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f,
                                           9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f,
                                           31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 49.f, 50.f, 51.f, 52.f,
                                           53.f, 54.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 67.f, 68.f, 69.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {-1, 2}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test14) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4, 5});
    x.linspace(-50.);
    auto alpha = NDArrayFactory::create<double>('c', {2,10}, {-0.7f, -0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f, 1.1f, 1.2f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4, 5}, {35.f,  29.4f,  24.f,  18.8f,  13.8f, 9.f,   4.4f,   0.f,  -4.2f,  -8.2f, -12.f, -15.6f, -19.f, -22.2f, -25.2f, -28.f, -30.6f,
                                           -33.f,-35.2f, -37.2f, 21.f,  17.4f,  14.f,  10.8f,   7.8f, 5.f,   2.4f,   0.f,  -2.2f,  -4.2f, -6.f,  -7.6f,  -9.f, -10.2f,
                                           -11.2f, -12.f, -12.6f, -13.f, -13.2f, -13.2f, 7.f,   5.4f,   4.f,   2.8f,   1.8f, 1.f,   0.4f,   0.f,  -0.2f,  -0.2f, 0.f,
                                           1.f,   2.f,   3.f,   4.f, 5.f,   6.f,   7.f,   8.f,   9.f, 10.f,  11.f,  12.f,  13.f,  14.f, 15.f,  16.f,  17.f,  18.f,
                                           19.f, 20.f,  21.f,  22.f,  23.f,  24.f, 25.f,  26.f,  27.f,  28.f,  29.f, 30.f,  31.f,  32.f,  33.f,  34.f, 35.f,  36.f,
                                           37.f,  38.f,  39.f, 40.f,  41.f,  42.f,  43.f,  44.f, 45.f,  46.f,  47.f,  48.f,  49.f, 50.f,  51.f,  52.f,  53.f,  54.f,
                                           55.f,  56.f,  57.f,  58.f,  59.f, 60.f,  61.f,  62.f,  63.f,  64.f, 65.f,  66.f,  67.f,  68.f,  69.f});

    nd4j::ops::prelu op;
    auto result = op.execute({&x, &alpha}, {}, {-2}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, thresholdedrelu_test1) {

    const float theta = 2.f;
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {0.f, 0.f, 0.f, 0.f,0.f, 0.f, 0.f, 0.f,0.f, 0.f, 0.f, 0.f,0.f, 0.f, 0.f, 3.f,4.f, 5.f, 6.f, 7.f,8.f, 9.f,10.f,11.f});

    nd4j::ops::thresholdedrelu op;

    auto result = op.execute({&x}, {theta}, {}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, compare_and_bitpack_test1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});
    auto threshold = NDArrayFactory::create<double>(2.0);
    auto exp = NDArrayFactory::create<uint8_t>('c', {2, 3, 4}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                                0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    nd4j::ops::compare_and_bitpack op;

    auto result = op.execute({&x, &threshold}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);
//    output->printIndexedBuffer("Packed to uint8");
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, thresholdedrelu_test2) {

    const float theta = -2.f;
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {0.f,-4.f, -10.f, -8.f, 0.f, -9.f, -8.f, 5.f, 6.f, 6.f, 9.f, 6.f, -8.f, 5.f, 10.f, -2.f, 3.f, -7.f, 4.f, -8.f, -4.f, -9.f, -9.f, 3.f});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 5.f, 6.f, 6.f, 9.f, 6.f, 0.f, 5.f, 10.f, 0.f, 3.f, 0.f, 4.f, 0.f, 0.f, 0.f, 0.f, 3.f});

    nd4j::ops::thresholdedrelu op;

    auto result = op.execute({&x}, {theta}, {}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_bp_test1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12., -11., -10., -9., -8., -7., -6., -5., -4., -3., -2., -1., 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.});
    auto alpha = NDArrayFactory::create<double>('c', {3, 4}, {-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5});
    auto dLdO = NDArrayFactory::create<double>('c', {2, 3, 4});

    const OpArgsHolder argsHolderFF({&x, &alpha}, {}, {});
    const OpArgsHolder argsHolderBP({&x, &alpha, &dLdO}, {}, {});

    nd4j::ops::prelu opFF;
    nd4j::ops::prelu_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_bp_test2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-12., -11., -10., -9., -8., -7., -6., -5., -4., -3., -2., -1., 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.});
    auto alpha = NDArrayFactory::create<double>('c', {4}, {-0.6, 2., 4., -1.});
    auto dLdO = NDArrayFactory::create<double>('c', {2, 3, 4});

    const OpArgsHolder argsHolderFF({&x, &alpha}, {}, {1});
    const OpArgsHolder argsHolderBP({&x, &alpha, &dLdO}, {}, {1});

    nd4j::ops::prelu opFF;
    nd4j::ops::prelu_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_bp_test3) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 2, 5});
    x.linspace(-30.);
    x.p(30, 0.5);   // avoid zero, since it is points of discontinuity for prelu
    auto alpha = NDArrayFactory::create<double>('c', {5,3}, {-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7});
    auto dLdO = NDArrayFactory::create<double>('c', {2, 3, 2, 5});

    const OpArgsHolder argsHolderFF({&x, &alpha}, {}, {-1, 2});
    const OpArgsHolder argsHolderBP({&x, &alpha, &dLdO}, {}, {-1, 2});

    nd4j::ops::prelu opFF;
    nd4j::ops::prelu_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_bp_test4) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4, 5});
    x.linspace(-50.);
    x.p(50, 0.5);   // avoid zero, since it is points of discontinuity for prele
    auto alpha = NDArrayFactory::create<double>('c', {2,10}, {-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.25, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2});
    auto dLdO = NDArrayFactory::create<double>('c', {2, 3, 4, 5});

    const OpArgsHolder argsHolderFF({&x, &alpha}, {}, {-2});
    const OpArgsHolder argsHolderBP({&x, &alpha, &dLdO}, {}, {-2});

    nd4j::ops::prelu opFF;
    nd4j::ops::prelu_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, thresholdedrelu_bp_test1) {

    const double theta = 0.15;

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {1.2, 1.1, 1., 0.9, 0.8, -0.7, -0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.9, -1.0, -1.1});
    auto dLdO = NDArrayFactory::create<double>('c', {2, 3, 4});

    const OpArgsHolder argsHolderFF({&x}, {theta}, {});
    const OpArgsHolder argsHolderBP({&x, &dLdO}, {theta}, {});

    nd4j::ops::thresholdedrelu opFF;
    nd4j::ops::thresholdedrelu_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_test1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto y = NDArrayFactory::create<double>('c', {4});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {0.1f, 0.4f, 0.9f, 1.6f, 0.5f, 1.2f, 2.1f, 3.2f, 0.9f, 2.f, 3.3f, 4.8f, 1.3f, 2.8f, 4.5f, 6.4f, 1.7f, 3.6f, 5.7f, 8.f, 2.1f, 4.4f, 6.9f, 9.6f});
    x.linspace(1.f);
    y.linspace(0.1f, 0.1f);

    nd4j::ops::multiply op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_test2) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto y = NDArrayFactory::create<double>(0.1);
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.f, 2.1f, 2.2f, 2.3f, 2.4f});
    x.linspace(1.f);
    // y.linspace(0.1f, 0.1f);

    nd4j::ops::multiply op;
    auto result = op.execute({&y, &x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_test3) {

    auto x = NDArrayFactory::create<double>('c', {2, 1, 4});
    auto y = NDArrayFactory::create<double>('c', {3,1});
    auto exp = NDArrayFactory::create<double>('c', {2, 3, 4}, {0.1f, 0.2f, 0.3f, 0.4f, 0.2f, 0.4f, 0.6f, 0.8f, 0.3f, 0.6f, 0.9f, 1.2f, 0.5f, 0.6f, 0.7f, 0.8f, 1.f, 1.2f, 1.4f, 1.6f, 1.5f, 1.8f, 2.1f, 2.4f});
    x.linspace(1.f);
    y.linspace(0.1f, 0.1f);

    nd4j::ops::multiply op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_test4) {

    auto x = NDArrayFactory::create<double>('c', {1, 1});
    auto y = NDArrayFactory::create<double>(0.1f);
    auto exp = NDArrayFactory::create<double>('c', {1, 1}, {0.1f});
    x.linspace(1.f);

    nd4j::ops::multiply op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_test5) {

    auto x = NDArrayFactory::create<double>(1.f);
    auto y = NDArrayFactory::create<double>(0.1f);
    auto exp = NDArrayFactory::create<double>(0.1f);

    nd4j::ops::multiply op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test1) {

    auto x = NDArrayFactory::create<double>('c', {1, 1}, {100.});
    auto y = NDArrayFactory::create<double>(0.1);
    auto dLdz = NDArrayFactory::create<double>('c', {1, 1});

    const OpArgsHolder argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply opFF;
    nd4j::ops::multiply_bp opBP;
    auto resFF = opFF.execute({&x, &y}, {}, {});
    auto resBP = opBP.execute({&x, &y, &dLdz}, {}, {});
//    resFF->at(0)->printIndexedBuffer("Multiply 1x1");
//    resBP->at(0)->printIndexedBuffer("Multiply BP 1x1 x");
//    resBP->at(1)->printIndexedBuffer("Multyply BP 1x1 y");*/
    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);
    delete resFF;
    delete resBP;
    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test2) {

    auto x = NDArrayFactory::create<double>('c', {2, 2}, {1.,2.,3.,4.});
    auto y = NDArrayFactory::create<double>(0.1);
    auto dLdz = NDArrayFactory::create<double>('c', {2, 2});

    const OpArgsHolder argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply opFF;
    nd4j::ops::multiply_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test3) {

    auto y = NDArrayFactory::create<double>('c', {2, 2}, {1.,2.,3.,4.});
    auto x = NDArrayFactory::create<double>(0.1);
    auto dLdz = NDArrayFactory::create<double>('c', {2, 2});

    const OpArgsHolder argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply opFF;
    nd4j::ops::multiply_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test4) {

    auto x = NDArrayFactory::create<double>('c', {2, 2}, {1.,2.,3.,4.});
    auto y = NDArrayFactory::create<double>('c', {2, 2}, {0.1,0.2,0.3,0.4});
    auto dLdz = NDArrayFactory::create<double>('c', {2, 2});

    const OpArgsHolder argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply opFF;
    nd4j::ops::multiply_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test5) {

    auto x = NDArrayFactory::create<double>('c', {2, 2}, {1.,2.,3.,4.});
    auto y = NDArrayFactory::create<double>('c', {2}, {0.1,0.2});
    auto dLdz = NDArrayFactory::create<double>('c', {2, 2});

    const OpArgsHolder argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply opFF;
    nd4j::ops::multiply_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test6) {

    auto y = NDArrayFactory::create<double>('c', {2, 2}, {1.,2.,3.,4.});
    auto x = NDArrayFactory::create<double>('c', {2}, {0.1,0.2});
    auto dLdz = NDArrayFactory::create<double>('c', {2, 2});

    const OpArgsHolder argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply opFF;
    nd4j::ops::multiply_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test7) {

    auto y = NDArrayFactory::create<double>('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    auto x = NDArrayFactory::create<double>('c', {2, 1}, {0.1,0.2});
    auto dLdz = NDArrayFactory::create<double>('c', {2, 3});

    const OpArgsHolder argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply opFF;
    nd4j::ops::multiply_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test8) {

    auto y = NDArrayFactory::create<double>('c', {2, 1, 4});
    auto x = NDArrayFactory::create<double>('c', {1, 3, 4});
    auto dLdz = NDArrayFactory::create<double>('c', {2, 3, 4});
    x.linspace(1., 0.5);
    y.linspace(0.1, 0.05);

    const OpArgsHolder argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply opFF;
    nd4j::ops::multiply_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, Floormod_BP_Test_2) {

    auto y = NDArrayFactory::create<double>('c', {10, 10});
    auto x = NDArrayFactory::create<double>('c', {10, 10});
    auto dLdz = NDArrayFactory::create<double>('c', {10, 10});
    //auto eps = NDArrayFactory::create<double>('c', {10, 10});
    x.linspace(4); //2., 2.0);
    y.linspace(3);
    dLdz.linspace(1);
//    const OpArgsHolder argsHolderFF({&x, &y}, {}, {});
//    const OpArgsHolder argsHolderBP({&x, &y, &dLdz}, {}, {});

//    nd4j::ops::floormod opFF;
//    auto resFF = opFF.execute({&x, &y}, {}, {});
//    resFF->at(0)->printIndexedBuffer("FF floormod");
//    delete resFF;
    nd4j::ops::floormod_bp opBP;
    auto resBP = opBP.execute({&x, &y, &dLdz}, {}, {});
    ASSERT_TRUE(resBP->status() == ND4J_STATUS_OK);

//    resBP->at(0)->printIndexedBuffer("BP floormod /dx");
//    resBP->at(1)->printIndexedBuffer("BP floormod /dy");
    ASSERT_TRUE(dLdz.equalsTo(resBP->at(0)));
    ASSERT_TRUE(dLdz.equalsTo(resBP->at(1)));
    delete resBP;

//    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

//    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, Dynamic_Partition_BP_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
    auto y = NDArrayFactory::create<int>('c', {2, 3}, {0, 1, 2, 1, 0, 2});
    auto dLdzX = NDArrayFactory::create<double>('c', {2, 4});
    auto dLdzY = NDArrayFactory::create<double>('c', {2, 4});
    auto dLdzZ = NDArrayFactory::create<double>('c', {2, 4});
    auto exp = NDArrayFactory::create<double>('c', {2,3,4}, {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3});
    x.linspace(1);
//    dLdzX.linspace(1);
//    dLdzY.linspace(2);
//    dLdzZ.linspace(3);
    dLdzX.assign(1);
    dLdzY.assign(2);
    dLdzZ.assign(3);

    nd4j::ops::dynamic_partition op1;
    auto res1 = op1.execute({&x, &y}, {}, {3});

    nd4j::ops::dynamic_partition_bp op2;
    auto res2 = op2.execute({&x, &y, &dLdzX, &dLdzY, &dLdzZ}, {}, {3});
    ASSERT_TRUE(res2->status() == ND4J_STATUS_OK);
    ASSERT_TRUE(res2->size() == 2);
//    printf("How many: %ul\n", res2->size());
//    res2->at(0)->printBuffer("Ouputput0");
//    res2->at(1)->printBuffer("Ouputput1");
    ASSERT_TRUE(res2->at(0)->equalsTo(exp));
    delete res1;
    delete res2;
}
//////////////////////////////////////////////////////////////////////
//TEST_F(DeclarableOpsTests9, Dynamic_Partition_BP_2) {
//
//    auto x = NDArrayFactory::create<double>('c', {2, 3, 4});
//    auto y = NDArrayFactory::create<int>('c', {2, 3}, {0, 1, 2, 1, 0, 2});
//    auto dLdzX = NDArrayFactory::create<double>('c', {2, 4});
//    auto dLdzY = NDArrayFactory::create<double>('c', {2, 4});
//    auto dLdzZ = NDArrayFactory::create<double>('c', {2, 4});
//    x.linspace(1);
//    dLdzX.linspace(1);
//    dLdzY.linspace(1);
//    dLdzZ.linspace(1);
//
//    const OpArgsHolder argsHolderFF({&x, &y}, {}, {3});
//    const OpArgsHolder argsHolderBP({&x, &y, &dLdzX, &dLdzY, &dLdzZ}, {}, {3});
//
//    nd4j::ops::dynamic_partition opFF;
//    nd4j::ops::dynamic_partition_bp opBP;
//
//    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);
//
//    ASSERT_TRUE(isGradCorrect);
//}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, Floormod_BP_Test_4) {

    auto x = NDArrayFactory::create<double>('c', {2, 1, 3}, {2.0, 6.0, -3.0, 2.0, 6.0, -3.0});
    auto y = NDArrayFactory::create<double>('c', {1, 3}, {-3.0, 2.0, -2.0});
    auto exp = NDArrayFactory::create<double>('c', {1, 3}, {-1.,  0., -1.});
    auto eps = NDArrayFactory::create<double>('c', {2, 1, 3});
    eps.assign(1.f);
    nd4j::ops::floormod_bp op;

    auto result = op.execute({&x, &y, &eps}, {}, {});

    ASSERT_TRUE(result->size() == 2);
    auto gradX = result->at(0);
    auto gradY = result->at(1);

//    gradX->printIndexedBuffer("gradX");
//    gradY->printIndexedBuffer("gradY");
    ASSERT_TRUE(exp.isSameShape(gradY));

    ASSERT_TRUE(exp.equalsTo(gradY));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, batchnorm_bp_test1) {

    NDArray input   ('c', {2,3,4}, nd4j::DataType::FLOAT32);
    NDArray mean    ('c', {4}, {1.1, 1.2, 1.3, 1.4}, nd4j::DataType::FLOAT32);
    NDArray variance('c', {4}, nd4j::DataType::FLOAT32);
    NDArray gamma   ('c', {4}, nd4j::DataType::FLOAT32);
    NDArray beta    ('c', {4}, nd4j::DataType::FLOAT32);
    NDArray gradO   ('c', {2,3,4}, nd4j::DataType::FLOAT32);

    NDArray expdLdI('c', {2,3,4}, {-0.000056, -0.000056, -0.000056, -0.000056, -0.000034, -0.000034, -0.000034, -0.000034, -0.000011, -0.000011, -0.000011, -0.000011, 0.000011, 0.000011, 0.000011, 0.000011, 0.000034, 0.000034, 0.000034, 0.000034, 0.000056, 0.000056, 0.000056, 0.000056}, nd4j::DataType::FLOAT32);
    NDArray expdLdG('c', {4}, {6.148104, 6.148104, 6.148105, 6.148105}, nd4j::DataType::FLOAT32);
    NDArray expdLdB('c', {4}, {3.6, 4.5, 5.4, 6.3}, nd4j::DataType::FLOAT32);

    input.linspace(0.1, 0.1);
    variance.assign(0.46666667);
    gamma.assign(1.2);
    beta.assign(1.);     // has no effect on gradient calculations
    gradO.linspace(-0.9, 0.15);

    nd4j::ops::batchnorm_bp op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta, &gradO}, {1e-5}, {1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto dLdI = results->at(0);
    auto dLdG = results->at(3);
    auto dLdB = results->at(4);

    ASSERT_TRUE(expdLdI.isSameShapeStrict(dLdI));
    ASSERT_TRUE(expdLdI.equalsTo(dLdI));

    ASSERT_TRUE(expdLdG.isSameShapeStrict(dLdG));
    ASSERT_TRUE(expdLdG.equalsTo(dLdG));

    ASSERT_TRUE(expdLdB.isSameShapeStrict(dLdB));
    ASSERT_TRUE(expdLdB.equalsTo(dLdB));

    delete results;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, batchnorm_bp_test2) {

    NDArray input   ('c', {2,3,4}, nd4j::DataType::FLOAT32);
    NDArray mean    ('c', {3}, {1.05, 1.1, 1.15}, nd4j::DataType::FLOAT32);
    NDArray variance('c', {3}, {0.5, 0.6, 0.7}, nd4j::DataType::FLOAT32);
    NDArray gamma   ('c', {3}, {1.2, 1.3, 1.4}, nd4j::DataType::FLOAT32);
    NDArray beta    ('c', {3}, nd4j::DataType::FLOAT32);
    NDArray gradO   ('c', {2,3,4}, nd4j::DataType::FLOAT32);

    NDArray expdLdI('c', {2,3,4}, {-0.601415, -0.521226, -0.441037, -0.360849, -0.456306, -0.395465, -0.334624, -0.273784, 0.396631, 0.343747,
                                    0.290863, 0.237978, 0.360849, 0.441037, 0.521226, 0.601415, 0.273784, 0.334625, 0.395465, 0.456306, -0.237978,
                                    -0.290863, -0.343746, -0.396631}, nd4j::DataType::FLOAT32);
    NDArray expdLdG('c', {3}, {5.81236 ,  7.048771, 12.155388}, nd4j::DataType::FLOAT32);
    NDArray expdLdB('c', {3}, {1.8,  6.6, 11.4}, nd4j::DataType::FLOAT32);

    input.linspace(0.1, 0.1);
    // beta.assign(1.);     // has no effect on gradient calculations
    gradO.linspace(-0.9, 0.15);

    nd4j::ops::batchnorm_bp op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta, &gradO}, {1e-5}, {1,1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto dLdI = results->at(0);
    auto dLdG = results->at(3);
    auto dLdB = results->at(4);

    ASSERT_TRUE(expdLdI.isSameShapeStrict(dLdI));
    ASSERT_TRUE(expdLdI.equalsTo(dLdI));

    ASSERT_TRUE(expdLdG.isSameShapeStrict(dLdG));
    ASSERT_TRUE(expdLdG.equalsTo(dLdG));

    ASSERT_TRUE(expdLdB.isSameShapeStrict(dLdB));
    ASSERT_TRUE(expdLdB.equalsTo(dLdB));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, batchnorm_bp_test3) {

    NDArray input   ('c', {2,3,4}, nd4j::DataType::FLOAT32);
    NDArray mean    ('c', {2,1,4}, {1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4}, nd4j::DataType::FLOAT32);
    NDArray variance('c', {2,1,4}, {0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2}, nd4j::DataType::FLOAT32);
    NDArray gamma   ('c', {2,1,4}, {1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9}, nd4j::DataType::FLOAT32);
    NDArray beta    ('c', {2,1,4}, nd4j::DataType::FLOAT32);
    NDArray gradO   ('c', {2,3,4}, nd4j::DataType::FLOAT32);

    NDArray expdLdI('c', {2,3,4}, {-0.577002, -0.744041, -0.850999, -0.922373, -0.000000, -0.000000, -0.000000, -0.000000, 0.577002,
                                    0.744041, 0.850999, 0.922373, -0.386037, -0.350205, -0.312047, -0.271737, -0.000000, -0.000000,
                                    -0.000000, -0.000000, 0.386037, 0.350205, 0.312047, 0.271736}, nd4j::DataType::FLOAT32);
    NDArray expdLdG('c', {2,1,4}, {1.378844, 0.910144, 0.573706, 0.335408, 2.640487, 2.954985, 3.289431, 3.64234 }, nd4j::DataType::FLOAT32);
    NDArray expdLdB('c', {2,1,4}, {-0.9 , -0.45,  0.  ,  0.45,  4.5 ,  4.95,  5.4 ,  5.85}, nd4j::DataType::FLOAT32);

    input.linspace(0.1, 0.1);
    // beta.assign(1.);     // has no effect on gradient calculations
    gradO.linspace(-0.9, 0.15);

    nd4j::ops::batchnorm_bp op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta, &gradO}, {1e-5}, {1,1,0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto dLdI = results->at(0);
    auto dLdG = results->at(3);
    auto dLdB = results->at(4);

    ASSERT_TRUE(expdLdI.isSameShapeStrict(dLdI));
    ASSERT_TRUE(expdLdI.equalsTo(dLdI));

    ASSERT_TRUE(expdLdG.isSameShapeStrict(dLdG));
    ASSERT_TRUE(expdLdG.equalsTo(dLdG));

    ASSERT_TRUE(expdLdB.isSameShapeStrict(dLdB));
    ASSERT_TRUE(expdLdB.equalsTo(dLdB));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, batchnorm_bp_test4) {

    NDArray input   ('c', {2,4}, nd4j::DataType::FLOAT32);
    NDArray mean    ('c', {4}, {1.05, 1.15, 1.2, 1.3}, nd4j::DataType::FLOAT32);
    NDArray variance('c', {4}, {0.5, 0.7, 0.9,  1.1}, nd4j::DataType::FLOAT32);
    NDArray gamma   ('c', {4}, {-1.2, 1.3, -1.4, 1.5}, nd4j::DataType::FLOAT32);
    NDArray beta    ('c', {4}, nd4j::DataType::FLOAT32);
    NDArray gradO   ('c', {2,4}, nd4j::DataType::FLOAT32);

    NDArray expdLdI('c', {2,4}, {0.162923, -0.289673, 0.354174, -0.386151, -0.162923, 0.289673, -0.354174, 0.386151}, nd4j::DataType::FLOAT32);
    NDArray expdLdG('c', {4}, {1.442483, 0.950200, 0.569207, 0.314641}, nd4j::DataType::FLOAT32);
    NDArray expdLdB('c', {4}, {-1.2, -0.9, -0.6, -0.3}, nd4j::DataType::FLOAT32);

    input.linspace(0.1, 0.1);
    gradO.linspace(-0.9, 0.15);

    nd4j::ops::batchnorm_bp op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta, &gradO}, {1e-5}, {1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto dLdI = results->at(0);
    auto dLdG = results->at(3);
    auto dLdB = results->at(4);

    ASSERT_TRUE(expdLdI.isSameShapeStrict(dLdI));
    ASSERT_TRUE(expdLdI.equalsTo(dLdI));

    ASSERT_TRUE(expdLdG.isSameShapeStrict(dLdG));
    ASSERT_TRUE(expdLdG.equalsTo(dLdG));

    ASSERT_TRUE(expdLdB.isSameShapeStrict(dLdB));
    ASSERT_TRUE(expdLdB.equalsTo(dLdB));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, batchnorm_bp_test5) {

    NDArray input   ('c', {2,4,2,2}, nd4j::DataType::FLOAT32);
    NDArray mean    ('c', {4}, {1.05, 1.15, 1.2, 1.3}, nd4j::DataType::FLOAT32);
    NDArray variance('c', {4}, {0.5, 0.7, 0.9,  1.1}, nd4j::DataType::FLOAT32);
    NDArray gamma   ('c', {4}, {-1.2, 1.3, -1.4, 1.5}, nd4j::DataType::FLOAT32);
    NDArray beta    ('c', {4}, nd4j::DataType::FLOAT32);
    NDArray gradO   ('c', {2,4,2,2}, nd4j::DataType::FLOAT32);

    NDArray expdLdI('c', {2,4,2,2}, {-0.737512, -0.659880, -0.582247, -0.504614, 0.561404, 0.502309, 0.443214, 0.384118, -1.168243,
        -1.045270, -0.922297, -0.799324, 1.899026, 1.699128, 1.499231, 1.299333, 0.504614, 0.582247, 0.659880, 0.737512, -0.384118,
        -0.443214, -0.502308, -0.561404, 0.799324, 0.922297, 1.045270, 1.168243, -1.299334, -1.499231, -1.699129, -1.899026}, nd4j::DataType::FLOAT32);
    NDArray expdLdG('c', {4}, {11.073181, 12.585667, 17.708657, 24.313186}, nd4j::DataType::FLOAT32);
    NDArray expdLdB('c', {4}, {4.2,  9. , 13.8, 18.6}, nd4j::DataType::FLOAT32);

    input.linspace(0.1, 0.1);
    gradO.linspace(-0.9, 0.15);

    nd4j::ops::batchnorm_bp op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta, &gradO}, {1e-5}, {1,1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto dLdI = results->at(0);
    auto dLdG = results->at(3);
    auto dLdB = results->at(4);

    ASSERT_TRUE(expdLdI.isSameShapeStrict(dLdI));
    ASSERT_TRUE(expdLdI.equalsTo(dLdI));

    ASSERT_TRUE(expdLdG.isSameShapeStrict(dLdG));
    ASSERT_TRUE(expdLdG.equalsTo(dLdG));

    ASSERT_TRUE(expdLdB.isSameShapeStrict(dLdB));
    ASSERT_TRUE(expdLdB.equalsTo(dLdB));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, batchnorm_bp_test6) {

    NDArray input   ('c', {2,2,2,4}, nd4j::DataType::FLOAT32);
    NDArray mean    ('c', {4}, {1.05, 1.15, 1.2, 1.3}, nd4j::DataType::FLOAT32);
    NDArray variance('c', {4}, {0.5, 0.7, 0.9,  1.1}, nd4j::DataType::FLOAT32);
    NDArray gamma   ('c', {4}, {-1.2, 1.3, -1.4, 1.5}, nd4j::DataType::FLOAT32);
    NDArray beta    ('c', {4}, nd4j::DataType::FLOAT32);
    NDArray gradO   ('c', {2,2,2,4}, nd4j::DataType::FLOAT32);

    NDArray expdLdI('c', {2,2,2,4}, {-4.989124, 2.540357, -1.515022, 0.791769, -3.563660, 1.814540, -1.082159, 0.565549, -2.138196, 1.088724, -0.649295,
                                    0.339329, -0.712732, 0.362908, -0.216432, 0.113110, 0.712732, -0.362908, 0.216432, -0.113110, 2.138195, -1.088724, 0.649295,
                                    -0.339330, 3.563660,-1.814540, 1.082159, -0.565549, 4.989125, -2.540356, 1.515022, -0.791770}, nd4j::DataType::FLOAT32);
    NDArray expdLdG('c', {4}, {20.364472, 17.856588, 16.949714, 15.903684}, nd4j::DataType::FLOAT32);
    NDArray expdLdB('c', {4}, {9.6, 10.8, 12. , 13.2}, nd4j::DataType::FLOAT32);

    input.linspace(0.1, 0.1);
    gradO.linspace(-0.9, 0.15);

    nd4j::ops::batchnorm_bp op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta, &gradO}, {1e-5}, {1,1,3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto dLdI = results->at(0);
    auto dLdG = results->at(3);
    auto dLdB = results->at(4);

    ASSERT_TRUE(expdLdI.isSameShapeStrict(dLdI));
    ASSERT_TRUE(expdLdI.equalsTo(dLdI));

    ASSERT_TRUE(expdLdG.isSameShapeStrict(dLdG));
    ASSERT_TRUE(expdLdG.equalsTo(dLdG));

    ASSERT_TRUE(expdLdB.isSameShapeStrict(dLdB));
    ASSERT_TRUE(expdLdB.equalsTo(dLdB));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, batchnorm_bp_test7) {

    NDArray input   ('c', {2,2,2,2,4}, nd4j::DataType::FLOAT32);
    NDArray mean    ('c', {4}, {1.05, 1.15, 1.2, 1.3}, nd4j::DataType::FLOAT32);
    NDArray variance('c', {4}, {0.5, 0.7, 0.9,  1.1}, nd4j::DataType::FLOAT32);
    NDArray gamma   ('c', {4}, {-1.2, 1.3, -1.4, 1.5}, nd4j::DataType::FLOAT32);
    NDArray beta    ('c', {4}, nd4j::DataType::FLOAT32);
    NDArray gradO   ('c', {2,2,2,2,4}, nd4j::DataType::FLOAT32);

    NDArray expdLdI('c', {2,2,2,2,4}, {-119.435059, 78.159744, -58.732986, 46.630123, -103.510391, 67.738441, -50.901920, 40.412773, -87.585716, 57.317142,
        -43.070854, 34.195419, -71.661041, 46.895844, -35.239792, 27.978071, -55.736359, 36.474548, -27.408726, 21.760721, -39.811687, 26.053242, -19.577662,
        15.543370, -23.887009, 15.631950, -11.746595, 9.326023, -7.962326, 5.210644, -3.915531, 3.108671, 7.962341, -5.210655, 3.915535, -3.108677, 23.887032,
        -15.631958, 11.746601, -9.326031, 39.811691, -26.053246, 19.577671, -15.543377, 55.736382, -36.474548, 27.408726, -21.760731, 71.661064, -46.895851, 35.239788,
        -27.978077, 87.585732, -57.317154, 43.070866, -34.195431, 103.510384, -67.738464, 50.901920, -40.412777, 119.435097, -78.159744, 58.732998, -46.630131}, nd4j::DataType::FLOAT32);
    NDArray expdLdG('c', {4}, {282.38734 , 244.542027, 224.140995, 207.548793}, nd4j::DataType::FLOAT32);
    NDArray expdLdB('c', {4}, {57.6, 60. , 62.4, 64.8}, nd4j::DataType::FLOAT32);

    input.linspace(0.1, 0.1);
    gradO.linspace(-0.9, 0.15);


    nd4j::ops::batchnorm_bp op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta, &gradO}, {1e-5}, {1,1,4});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto dLdI = results->at(0);
    auto dLdG = results->at(3);
    auto dLdB = results->at(4);

    // dLdI->printBuffer();

    ASSERT_TRUE(expdLdI.isSameShapeStrict(dLdI));
    ASSERT_TRUE(expdLdI.equalsTo(dLdI));

    ASSERT_TRUE(expdLdG.isSameShapeStrict(dLdG));
    ASSERT_TRUE(expdLdG.equalsTo(dLdG));

    ASSERT_TRUE(expdLdB.isSameShapeStrict(dLdB));
    ASSERT_TRUE(expdLdB.equalsTo(dLdB));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, batchnorm_bp_test8) {

    NDArray input   ('c', {2,4,2,2,2}, nd4j::DataType::FLOAT32);
    NDArray mean    ('c', {4}, {1.05, 1.15, 1.2, 1.3}, nd4j::DataType::FLOAT32);
    NDArray variance('c', {4}, {0.5, 0.7, 0.9,  1.1}, nd4j::DataType::FLOAT32);
    NDArray gamma   ('c', {4}, {-1.2, 1.3, -1.4, 1.5}, nd4j::DataType::FLOAT32);
    NDArray beta    ('c', {4}, nd4j::DataType::FLOAT32);
    NDArray gradO   ('c', {2,4,2,2,2}, nd4j::DataType::FLOAT32);

    NDArray expdLdI('c', {2,4,2,2,2}, {-34.373802, -32.611046, -30.848286, -29.085529, -27.322769, -25.560009, -23.797251, -22.034491, 36.146996, 34.293301,
        32.439610, 30.585917, 28.732227, 26.878534, 25.024841, 23.171150, -42.876553, -40.677757, -38.478958, -36.280159, -34.081367, -31.882565, -29.683767,
        -27.484968, 50.674446, 48.075760, 45.477066, 42.878380, 40.279686, 37.681000, 35.082310, 32.483616, 22.034489, 23.797249, 25.560009, 27.322765, 29.085526,
        30.848286, 32.611046, 34.373802, -23.171146, -25.024837, -26.878536, -28.732231, -30.585918, -32.439613, -34.293297, -36.146996, 27.484982, 29.683773,
        31.882572, 34.081364, 36.280178, 38.478970, 40.677776, 42.876560, -32.483627, -35.082329, -37.681023, -40.279701, -42.878403, -45.477081, -48.075775, -50.674484}, nd4j::DataType::FLOAT32);
    NDArray expdLdG('c', {4}, {134.490365, 179.785003, 248.933114, 330.087248}, nd4j::DataType::FLOAT32);
    NDArray expdLdB('c', {4}, {32.4, 51.6, 70.8, 90.}, nd4j::DataType::FLOAT32);

    input.linspace(0.1, 0.1);
    gradO.linspace(-0.9, 0.15);

    nd4j::ops::batchnorm_bp op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta, &gradO}, {1e-5}, {1,1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto dLdI = results->at(0);
    auto dLdG = results->at(3);
    auto dLdB = results->at(4);

    // dLdI->printBuffer();

    ASSERT_TRUE(expdLdI.isSameShapeStrict(dLdI));
    ASSERT_TRUE(expdLdI.equalsTo(dLdI));

    ASSERT_TRUE(expdLdG.isSameShapeStrict(dLdG));
    ASSERT_TRUE(expdLdG.equalsTo(dLdG));

    ASSERT_TRUE(expdLdB.isSameShapeStrict(dLdB));
    ASSERT_TRUE(expdLdB.equalsTo(dLdB));

    delete results;
}
/*
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, gru_cell_bp_test1) {

    const int bS = 2;
    const int iS = 3;
    const int nU = 4;

    NDArray x('c', {bS, iS}, nd4j::DataType::DOUBLE);
    NDArray hi('c', {bS, nU}, nd4j::DataType::DOUBLE);
    NDArray W('c', {iS+nU, 2*nU}, nd4j::DataType::DOUBLE);
    NDArray Wc('c', {iS+nU, nU}, nd4j::DataType::DOUBLE);
    NDArray b('c', {2*nU}, nd4j::DataType::DOUBLE);
    NDArray bc('c', {nU}, nd4j::DataType::DOUBLE);
    NDArray dLdr('c', {bS, nU}, nd4j::DataType::DOUBLE);
    NDArray dLdu('c', {bS, nU}, nd4j::DataType::DOUBLE);
    NDArray dLdc('c', {bS, nU}, nd4j::DataType::DOUBLE);
    NDArray dLdh('c', {bS, nU}, nd4j::DataType::DOUBLE);

    x.linspace(-5, 0.5);
    hi   = 1.;
    W    = 0.003;
    Wc   = 0.006;
    b    = 0.5;
    bc   = 0.35;


    const OpArgsHolder argsHolderFF({&x, &hi, &W, &Wc, &b, &bc}, {}, {});
    nd4j::ops::gruCell op;
    auto results = op.execute(argsHolderFF);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto u = results->at(1);    // [bS, nU]
    auto c = results->at(2);    // [bS, nU]
    auto h = results->at(3);    // [bS, nU]

    dLdh = 1.; // SUM loss

    NDArray Wch = Wc({iS,iS+nU, 0,0}); // [nU, nU]
    NDArray dhdc  = 1. - *u;
    NDArray dhdu  = hi - *c;
    NDArray dcdZc = 1. - *c * *c;
    dLdc.assign(dLdh * dhdc);
    dLdu.assign(dLdh * dhdu);
    dLdr.assign(mmul(dLdc * dcdZc * hi, Wch.transpose()));

    delete results;


    const OpArgsHolder argsHolderBP({&x, &hi, &W, &Wc, &b, &bc, &dLdr, &dLdu, &dLdc, &dLdh}, {}, {});

    nd4j::ops::gruCell opFF;
    nd4j::ops::gruCell_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1, 1, 1 , 1, 1}, {0., 1.}, nd4j::GradCheck::LossFunc::SUM, true);

    ASSERT_TRUE(isGradCorrect);
}
*/

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, Cholesky_Test_1) {

    NDArray x = NDArrayFactory::create<double>('c', {3, 3}, {4,12,-16, 12 ,37,-43, -16, -43, 98});
    NDArray exp = NDArrayFactory::create<double>('c', {3,3}, {2.,  0.,  0., 6., 1.,  0., -8.,  5.,  3.});

    nd4j::ops::cholesky op;

    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(result->status(), ND4J_STATUS_OK);
    auto res = result->at(0);
//    res->printIndexedBuffer("Output for Cholesky1");
    ASSERT_TRUE(exp.equalsTo(res));
    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, Cholesky_Test_2) {

    NDArray x = NDArrayFactory::create<double>('c', {2, 3, 3}, {4, 12,-16, 12 ,37,-43, -16, -43, 98, 1, 1, 1, 1, 2, 2, 1, 2., 6});
    NDArray exp = NDArrayFactory::create<double>('c', {2, 3, 3}, {2.,  0.,  0., 6., 1.,  0., -8.,  5.,  3., 1., 0., 0., 1., 1., 0,1., 1., 2.});

    nd4j::ops::cholesky op;

    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(result->status(), ND4J_STATUS_OK);
    auto res = result->at(0);
//    res->printIndexedBuffer("Output for Cholesky 2");
    ASSERT_TRUE(exp.equalsTo(res));
    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, Cholesky_Test_3) {

    NDArray x = NDArrayFactory::create<float>('c', {2, 3, 3}, {4.f, 12.f, -16.f, 12.f, 37.f, -43.f, -16.f, -43.f, 98.f, 1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 1.f, 2.f, 6.f});
    NDArray exp = NDArrayFactory::create<float>('c', {2, 3, 3}, {2.f,  0.f,  0.f, 6.f, 1.f,  0.f, -8.f,  5.f,  3.f, 1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 2.f});

    nd4j::ops::cholesky op;

    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(result->status(), ND4J_STATUS_OK);
    auto res = result->at(0);
    // res->printIndexedBuffer("Output for Cholesky 3");
    ASSERT_TRUE(exp.equalsTo(res, 1e-4));
    delete result;
}

////////////////////////////////////////////////////////////////////
// TEST_F(DeclarableOpsTests9, gru_bp_test1) {

//     const int time = 5;
//     const int bS   = 2;
//     const int iS   = 3;
//     const int nU   = 4;

//     NDArray<double> x     ('c', {time, bS, iS});
//     NDArray<double> h0    ('c', {bS, nU});
//     NDArray<double> Wx    ('c', {iS, 3*nU});
//     NDArray<double> Wh    ('c', {nU, 3*nU});
//     NDArray<double> b     ('c', {3*nU});
//     NDArray<double> dLdh  ('c', {time, bS, nU});

//     x.linspace(0.5, 0.5);
//     h0 = 1.;
//     Wx = 0.003;
//     Wh = 0.006;
//     b  = 0.5;

//     const OpArgsHolder<double> argsHolderFF({&x, &h0, &Wx, &Wh, &b}, {}, {});
//     const OpArgsHolder<double> argsHolderBP({&x, &h0, &Wx, &Wh, &b, &dLdh}, {}, {});

//     nd4j::ops::gru<double> opFF;
//     nd4j::ops::gru_bp<double> opBP;

//     const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

//     ASSERT_TRUE(isGradCorrect);
// }

//
