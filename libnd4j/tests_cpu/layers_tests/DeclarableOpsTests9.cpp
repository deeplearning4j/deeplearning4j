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

    NDArray<double> x('c', {3,4});
    NDArray<double> gradO1('c', {3,1}, {1.,2.,3.});
    NDArray<double> gradO2('c', {3}, {1.,2.,3.});
    NDArray<double> exp('c', {3,4}, {-0.335410, -0.111803, 0.111803, 0.335410, -0.670820, -0.223607, 0.223607, 0.670820, -1.006231, -0.335410, 0.335410, 1.006231});

    x.linspace(1);

    nd4j::ops::reduce_stdev_bp<double> op;

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
TEST_F(DeclarableOpsTests9, exponentialDistributionInv_test1) {
    
    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda; 
    const double std    = mean;

    NDArray<double> x('c', {N});
    double extraParams[] = {lambda};

    Nd4jLong *buffer = new Nd4jLong[N];
    NativeOps nativeOps;
    nd4j::random::RandomBuffer* rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, N, (Nd4jPointer) buffer);    
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistributionInv_test1: RNG initialization failed !");
    
    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistributionInv<double>>(rng, x.getBuffer(), x.getShapeInfo(), extraParams);
    const double actualMean = x.meanNumber();
    const double actualStd  = x.template varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true);
 
    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);    

    nativeOps.destroyRandom((Nd4jPointer) rng);
    delete[] buffer;
        
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, exponentialDistributionInv_test2) {
    
    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda; 
    const double std    = mean;
    double extraParams[] = {lambda};

    NDArray<double> x('c', {N});
    NDArray<double> y('c', {N});
    y.linspace(0., 1./N);  // [0, 1)


    Nd4jLong *buffer = new Nd4jLong[N];
    NativeOps nativeOps;
    nd4j::random::RandomBuffer* rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, N, (Nd4jPointer) buffer);    
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistributionInv_test2: RNG initialization failed !");
    
    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistributionInv<double>>(rng, y.getBuffer(), y.getShapeInfo(), x.getBuffer(), x.getShapeInfo(), extraParams);

    const double actualMean = x.meanNumber();
    const double actualStd  = x.template varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true);

    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);    

    nativeOps.destroyRandom((Nd4jPointer) rng);
    delete[] buffer;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, exponentialDistribution_test1) {
    
    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda; 
    const double std    = mean;

    NDArray<double> x('c', {N});
    double extraParams[] = {lambda};

    Nd4jLong *buffer = new Nd4jLong[N];
    NativeOps nativeOps;
    nd4j::random::RandomBuffer* rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, N, (Nd4jPointer) buffer);    
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistribution_test1: RNG initialization failed !");
    
    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistribution<double>>(rng, x.getBuffer(), x.getShapeInfo(), extraParams);
    const double actualMean = x.meanNumber();
    const double actualStd  = x.template varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true);
 
    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);    

    nativeOps.destroyRandom((Nd4jPointer) rng);
    delete[] buffer;       
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, exponentialDistribution_test2) {
    
    const int N = 50000;
    const double lambda = 2.;
    const double mean   = 1. / lambda; 
    const double std    = mean;
    double extraParams[] = {lambda};

    NDArray<double> x('c', {N});
    NDArray<double> y('c', {N});
    y.linspace(-N/2.);  // [-25000, 25000)


    Nd4jLong *buffer = new Nd4jLong[N];
    NativeOps nativeOps;
    nd4j::random::RandomBuffer* rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, N, (Nd4jPointer) buffer);    
    if (rng == nullptr)
        throw std::runtime_error("DeclarableOpsTests9.exponentialDistribution_test2: RNG initialization failed !");
    
    functions::random::RandomFunction<double>::template execTransform<randomOps::ExponentialDistribution<double>>(rng, y.getBuffer(), y.getShapeInfo(), x.getBuffer(), x.getShapeInfo(), extraParams);

    const double actualMean = x.meanNumber();
    const double actualStd  = x.template varianceNumber<simdOps::SummaryStatsStandardDeviation<double>>(true);

    ASSERT_NEAR(mean, actualMean, 0.01);
    ASSERT_NEAR(std,  actualStd, 0.01);    

    nativeOps.destroyRandom((Nd4jPointer) rng);
    delete[] buffer;
}

TEST_F(DeclarableOpsTests9, ScalarOpTest_MixedOrders_1) {
    NDArray<double> x('f', {2, 2}, {1.0, 3.0, 2.0, 4.0});
    NDArray<double> e('c', {2, 2}, {2.0, 3.0, 4.0, 5.0});
    NDArray<double> z('c', {2, 2}, {0.0, 0.0, 0.0, 0.0});

    x.template applyScalar<simdOps::Add<double>>(1.0, &z);

    ASSERT_EQ(e, z);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test1) {

    NDArray<double> input   ('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    NDArray<double> gradO   ('c', {4, 9});
    NDArray<double> gradIExp('c', {2, 3}, {0.78, 0.84, 0.9,1.32, 1.38, 1.44});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {2, 3});
    NDArray<double>* gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test2) {

    NDArray<double> input   ('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    NDArray<double> gradO   ('c', {2, 9});
    NDArray<double> gradIExp('c', {2, 3}, {0.12, 0.15, 0.18, 0.39, 0.42, 0.45});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {1, 3});
    NDArray<double>* gradI = results->at(0);
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test1) {

    NDArray<float> x0('c', {2,3,4});
    NDArray<float> x1('c', {2,2,4});
    NDArray<float> x2('c', {2,1,4});
    NDArray<float> exp('c', {2,6,4}, {1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                     13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test2) {

    NDArray<float> x0('c', {1,3,1});
    NDArray<float> x1('c', {1,2,1});
    NDArray<float> x2('c', {1,1,1});
    NDArray<float> exp('c', {1,6,1}, {1.f, 2.f, 3.f, 1.f, 2.f, 1.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test3) {

    NDArray<float> x0('c', {3});
    NDArray<float> x1('c', {2});
    NDArray<float> x2('c', {1});
    NDArray<float> exp('c', {6}, {1.f, 2.f, 3.f, 1.f, 2.f, 1.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test4) {

    NDArray<float> x0('c', {1,1,1}, {1.f});
    NDArray<float> x1('c', {1,1,1}, {2.f});
    NDArray<float> x2('c', {1,1,1}, {3.f});
    NDArray<float> exp('c', {1,3,1}, {1.f, 2.f, 3.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test5) {

    NDArray<float> x0(1.f);
    NDArray<float> x1('c', {1}, {2.f});
    NDArray<float> x2(3.f);
    NDArray<float> exp('c', {3}, {1.f, 2.f, 3.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test6) {

    NDArray<float> x0(1.f);
    NDArray<float> x1('c', {2}, {2.f, 20.f});
    NDArray<float> x2(3.f);
    NDArray<float> exp('c', {4}, {1.f, 2.f, 20.f, 3.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test7) {

    NDArray<float> x0(1.f);
    NDArray<float> x1(2.f);
    NDArray<float> x2(3.f);
    NDArray<float> exp('c', {3}, {1.f, 2.f, 3.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test8) {

    NDArray<float> x0(1.f);
    NDArray<float> exp('c', {1}, {1.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test9) {

    NDArray<float> x0('c', {1}, {1.f});
    NDArray<float> exp('c', {1}, {1.f});

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test3) {

    NDArray<double> input   ('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    NDArray<double> gradO   ('c', {2, 3});
    NDArray<double> gradIExp('c', {2, 3}, {0.01, 0.02, 0.03,0.04, 0.05, 0.06});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {1, 1});
    NDArray<double>* gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test4) {

    NDArray<double> input   ('c', {6}, {1.,2.,3.,4.,5.,6.});
    NDArray<double> gradO   ('c', {12});
    NDArray<double> gradIExp('c', {6}, {0.08, 0.1 , 0.12, 0.14, 0.16, 0.18});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {2});
    NDArray<double>* gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test5) {

    NDArray<double> input   ('c', {1}, {1.});
    NDArray<double> gradO   ('c', {1});
    NDArray<double> gradIExp('c', {1}, {0.01});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {1});
    NDArray<double>* gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, tile_bp_test6) {

    NDArray<double> input   ('c', {2, 1, 3}, {1.,2.,3.,4.,5.,6.});
    NDArray<double> gradO   ('c', {2, 3, 6});
    NDArray<double> gradIExp('c', {2, 1, 3}, {0.51, 0.57, 0.63, 1.59, 1.65, 1.71});

    gradO.linspace(0.01, 0.01);

    nd4j::ops::tile_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {1, 3, 2});
    NDArray<double>* gradI = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(gradIExp.isSameShape(gradI));
    ASSERT_TRUE(gradIExp.equalsTo(gradI));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test1) {

    NDArray<double> x ('c', {3, 4});
    NDArray<double> y ('c', {4, 3});
    NDArray<double> exp('f', {3, 3}, {35.,  79., 123., 40.,  92., 144., 45., 105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test2) {

    NDArray<double> x ('c', {3, 4});
    NDArray<double> y ('f', {4, 3});
    NDArray<double> exp('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test3) {

    NDArray<double> x ('f', {3, 4});
    NDArray<double> y ('c', {4, 3});
    NDArray<double> exp('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test4) {

    NDArray<double> x ('f', {3, 4});
    NDArray<double> y ('f', {4, 3});
    NDArray<double> exp('f', {3, 3}, {35., 79., 123.,40., 92., 144.,45.,105., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test5) {

    NDArray<double> x ('c', {4, 3});
    NDArray<double> y ('c', {4, 3});
    NDArray<double> exp('f', {3, 3}, {83.,  94., 105., 94., 107., 120., 105., 120., 135.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test6) {

    NDArray<double> x ('c', {4, 3});
    NDArray<double> y ('f', {3, 4});
    NDArray<double> exp('f', {3, 3}, {35.,  40.,  45., 79.,  92., 105., 123., 144., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test7) {

    NDArray<double> x ('c', {5,  3,4});
    NDArray<double> y ('f', {5,  3,4});
    NDArray<double> exp('f',{5,  3,3}, {3. ,  84.6, 281.4, 593.4, 1020.6, 7. , 107.8, 323.8, 655. , 1101.4,11. , 131. , 366.2, 716.6, 1182.2,
                                        7. , 107.8, 323.8, 655. , 1101.4,17.4, 137.4, 372.6, 723. , 1188.6,27.8, 167. , 421.4, 791. , 1275.8,
                                       11. , 131. , 366.2, 716.6, 1182.2,27.8, 167. , 421.4, 791. , 1275.8,44.6, 203. , 476.6, 865.4, 1369.4,});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test8) {

    NDArray<double> x ('c', {2,5,  3,4});
    NDArray<double> y ('f', {2,5,  3,4});
    NDArray<double> exp('f',{2,5,  3,3}, {3. , 1563. ,  84.6, 2220.6, 281.4, 2993.4, 593.4, 3881.4,1020.6, 4884.6,   7. , 1663. , 107.8, 2339.8, 323.8, 3131.8, 655. , 4039. ,1101.4, 5061.4,
                                          11. , 1763. , 131. , 2459. , 366.2, 3270.2, 716.6, 4196.6,1182.2, 5238.2,   7. , 1663. , 107.8, 2339.8, 323.8, 3131.8, 655. , 4039. ,1101.4, 5061.4,
                                          17.4, 1769.4, 137.4, 2465.4, 372.6, 3276.6, 723. , 4203. ,1188.6, 5244.6,  27.8, 1875.8, 167. , 2591. , 421.4, 3421.4, 791. , 4367. ,1275.8, 5427.8,
                                          11. , 1763. , 131. , 2459. , 366.2, 3270.2, 716.6, 4196.6,1182.2, 5238.2,  27.8, 1875.8, 167. , 2591. , 421.4, 3421.4, 791. , 4367. ,1275.8, 5427.8,
                                          44.6, 1988.6, 203. , 2723. , 476.6, 3572.6, 865.4, 4537.4,1369.4, 5617.4});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}
 
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test9) {

    NDArray<double> x ('c', {2,5,  4,3});
    NDArray<double> y ('f', {2,5,  3,4});
    NDArray<double> exp('f',{2,5,  3,3}, {7. , 1639. , 103. , 2311. , 314.2, 3098.2, 640.6, 4000.6,1082.2, 5018.2,   8. , 1664. , 108.8, 2340.8, 324.8, 3132.8, 656. , 4040. ,1102.4, 5062.4,
                                          9. , 1689. , 114.6, 2370.6, 335.4, 3167.4, 671.4, 4079.4,1122.6, 5106.6,  15.8, 1743.8, 131. , 2435. , 361.4, 3241.4, 707. , 4163. ,1167.8, 5199.8,
                                          18.4, 1770.4, 138.4, 2466.4, 373.6, 3277.6, 724. , 4204. ,1189.6, 5245.6,  21. , 1797. , 145.8, 2497.8, 385.8, 3313.8, 741. , 4245. ,1211.4, 5291.4,
                                          24.6, 1848.6, 159. , 2559. , 408.6, 3384.6, 773.4, 4325.4,1253.4, 5381.4,  28.8, 1876.8, 168. , 2592. , 422.4, 3422.4, 792. , 4368. ,1276.8, 5428.8,
                                          33. , 1905. , 177. , 2625. , 436.2, 3460.2, 810.6, 4410.6,1300.2, 5476.2});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}
   
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test10) {

    NDArray<double> x ('c', {1, 4, 3});
    NDArray<double> y ('f', {1, 3, 4});
    NDArray<double> exp('f', {1, 3, 3}, {35.,  40.,  45., 79.,  92., 105., 123., 144., 165.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test11) {

    NDArray<double> x ('c', {4, 1});
    NDArray<double> y ('f', {1, 4});
    NDArray<double> exp('f', {1, 1}, {15});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test12) {

    NDArray<double> x ('c', {1, 4, 1});
    NDArray<double> y ('f', {1, 1, 4});
    NDArray<double> exp('f', {1, 1, 1}, {15});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test13) {

    NDArray<double> x ('c', {2, 3});
    NDArray<double> y ('c', {3, 5});
    NDArray<double> exp('f', {5, 2}, {23. , 26. , 29. , 32. , 35., 50. , 57.5, 65. , 72.5, 80.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {0, 0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test14) {

    NDArray<double> x ('c', {3, 2});
    NDArray<double> y ('c', {3, 5});
    NDArray<double> exp('f', {5, 2}, {37. , 41.5, 46. , 50.5, 55., 46. , 52. , 58. , 64. , 70.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test15) {

    NDArray<double> x ('c', {3, 2});
    NDArray<double> y ('c', {3, 5});
    NDArray<double> exp('f', {5, 2}, {37. , 41.5, 46. , 50.5, 55., 46. , 52. , 58. , 64. , 70.});

    x.linspace(1.);
    y.linspace(0.5, 0.5);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test16) {

    NDArray<double> x ('c', {2,2,  3,5});
    NDArray<double> y ('c', {2,2,  4,3});
    NDArray<double> exp('f',{2,2,  4,5}, {4.6, 281.8, 89.2, 582.4, 10. , 314.2,108.1, 628.3, 15.4, 346.6,127. , 674.2, 20.8, 379. ,145.9, 720.1,  5.2, 289.6, 93.4, 593.8,
                                          11.5, 322.9,113.2, 640.6, 17.8, 356.2,133. , 687.4, 24.1, 389.5,152.8, 734.2,  5.8, 297.4, 97.6, 605.2, 13. , 331.6,118.3, 652.9,
                                          20.2, 365.8,139. , 700.6, 27.4, 400. ,159.7, 748.3,  6.4, 305.2,101.8, 616.6, 14.5, 340.3,123.4, 665.2, 22.6, 375.4,145. , 713.8,
                                          30.7, 410.5,166.6, 762.4,  7. , 313. ,106. , 628. , 16. , 349. ,128.5, 677.5, 25. , 385. ,151. , 727. , 34. , 421. ,173.5, 776.5});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test17) {

    NDArray<double> x ('f', {4, 3});
    NDArray<double> y ('c', {4});
    NDArray<double> exp('f',{3}, {7., 8., 9.});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 0});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test18) {

    NDArray<double> x ('f', {3});
    NDArray<double> y ('c', {4, 3});
    NDArray<double> exp('f',{4}, {1.4, 3.2, 5., 6.8});

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {0, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test19) {

    NDArray<double> x ('f', {1, 1});
    NDArray<double> y ('c', {1, 1});
    NDArray<double> exp('f',{1, 1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test20) {

    NDArray<double> x ('f', {1, 1});
    NDArray<double> y ('c', {1, 1});
    NDArray<double> exp('f',{1, 1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1,1,1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test21) {

    NDArray<double> x ('f', {1});
    NDArray<double> y ('c', {1, 1});
    NDArray<double> exp('f',{1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test22) {

    NDArray<double> x ('f', {1,1});
    NDArray<double> y ('c', {1});
    NDArray<double> exp('f',{1}, {0.2});

    x.linspace(2.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test23) {

    NDArray<double> x ('f', {4});   
    NDArray<double> y ('c', {4});
    NDArray<double> exp(3.);

    x.linspace(1.);
    y.linspace(0.1, 0.1);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, matmul_test24) {

    NDArray<double> x ('f', {1}, {2.});   
    NDArray<double> y ('c', {1}, {3.});
    NDArray<double> exp(6.);
 
    nd4j::ops::matmul<double> op;
    ResultSet<double>* results = op.execute({&x, &y}, {}, {1, 1});
    NDArray<double>* z = results->at(0);
 
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete results;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test10) {

    NDArray<float> x0('c', {2,3,4});
    NDArray<float> x1('f', {2,2,4});
    NDArray<float> x2('c', {2,1,4});
    NDArray<float> exp('c', {2,6,4}, { 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                      13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test11) {

    NDArray<float> x0('c', {2,3,4});
    NDArray<float> x1('f', {2,2,4});
    NDArray<float> x2('f', {2,1,4});
    NDArray<float> exp('c', {2,6,4}, { 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                      13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test12) {

    NDArray<float> x0('c', {2,3,4});
    NDArray<float> x1('f', {2,2,4});
    NDArray<float> x2('f', {2,1,4});
    NDArray<float> exp('c', {2,6,4}, { 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 9.f, 10.f, 11.f, 12.f, 1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f, 1.f,  2.f,  3.f,  4.f,
                                      13.f, 14.f, 15.f, 16.f,17.f, 18.f, 19.f, 20.f,21.f, 22.f, 23.f, 24.f, 9.f, 10.f, 11.f, 12.f,13.f, 14.f, 15.f, 16.f, 5.f,  6.f,  7.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, concat_test13) {

    NDArray<float> x0('f', {2,3,4});
    NDArray<float> x1('f', {2,2,4});
    NDArray<float> x2('f', {2,1,4});
    NDArray<float> exp('f', {2,6,4}, { 1.f, 13.f, 5.f, 17.f, 9.f, 21.f, 1.f,  9.f, 5.f, 13.f, 1.f,  5.f, 2.f, 14.f, 6.f, 18.f,10.f, 22.f, 2.f, 10.f, 6.f, 14.f, 2.f,  6.f,
                                       3.f, 15.f, 7.f, 19.f,11.f, 23.f, 3.f, 11.f, 7.f, 15.f, 3.f,  7.f, 4.f, 16.f, 8.f, 20.f,12.f, 24.f, 4.f, 12.f, 8.f, 16.f, 4.f,  8.f});

    x0.linspace(1);
    x1.linspace(1);
    x2.linspace(1);

    nd4j::ops::concat<float> op;

    auto result = op.execute({&x0, &x1, &x2}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);


    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, clipbynorm_test12) {
    
    const int bS   = 5;
    const int nOut = 4;
    const int axis = 0;
    const double clip = 2.;
    
    NDArray<double> x('c', {bS, nOut}, {0.412 ,0.184 ,0.961 ,0.897 ,0.173 ,0.931 ,0.736 ,0.540 ,0.953 ,0.278 ,0.573 ,0.787 ,0.320 ,0.776 ,0.338 ,0.311 ,0.835 ,0.909 ,0.890 ,0.290});    // uniform random in range [0,1]
    NDArray<double> colVect('c', {bS, 1}, {0.9, 0.95, 1.00, 1.05, 1.1});
    NDArray<double> expect('c', {bS, nOut});

    NDArray<double> norm2 = x.template reduceAlongDims<simdOps::Norm2<double>>({axis}, true); // norm2 has shape [1, nOut]
    
    NDArray<double> y = ( (x / norm2) * clip) * colVect ;    
    NDArray<double> temp = (x / norm2) * clip;    

    for (int j = 0; j < nOut; ++j) {
        NDArray<double> yCol = y({{}, {j, j+1}});
        const double norm2Col = yCol.template reduceNumber<simdOps::Norm2<double>>();
        if (norm2Col <= clip) 
            expect({{}, {j,j+1}}).assign(yCol);
        else 
            expect({{}, {j,j+1}}).assign ( yCol * (clip / norm2Col) );
    }
    
    nd4j::ops::clipbynorm<double> op;
    auto result = op.execute({&y}, {clip}, {axis});
    auto outFF = result->at(0);        
    
    ASSERT_TRUE(expect.isSameShape(outFF));
    ASSERT_TRUE(expect.equalsTo(outFF));

    delete result;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, clipbynorm_bp_test1) {
    
    const int bS   = 2;
    const int nOut = 3;
    const int axis = 0;
    const double clip = 0.7;
    
    NDArray<double> x('c', {bS, nOut}, {0.412 ,0.184 ,0.961 ,0.173 ,0.736 ,0.540 });    // uniform random in range [0,1]
    NDArray<double> gradO('c', {bS, nOut});

    const OpArgsHolder<double> argsHolderFF({&x}, {clip}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &gradO}, {clip}, {});

    nd4j::ops::clipbynorm<double> opFF;
    nd4j::ops::clipbynorm_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, clipbynorm_bp_test2) {
    
    const int bS   = 2;
    const int nOut = 3;
    const int axis = 0;
    const double clip = 0.7;
    
    NDArray<double> x('c', {bS, nOut}, {0.412 ,0.184 ,0.961 ,0.173 ,0.736 ,0.540 });    // uniform random in range [0,1]
    NDArray<double> gradO('c', {bS, nOut});

    const OpArgsHolder<double> argsHolderFF({&x}, {clip}, {axis});
    const OpArgsHolder<double> argsHolderBP({&x, &gradO}, {clip}, {axis});

    nd4j::ops::clipbynorm<double> opFF;
    nd4j::ops::clipbynorm_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, clipbynorm_bp_test3) {
    
    const int bS   = 2;
    const int nOut = 3;
    const int axis = 1;
    const double clip = 1.;
    
    NDArray<double> x('c', {bS, nOut}, {0.412 ,0.184 ,0.961 ,0.173 ,0.736 ,0.540 });    // uniform random in range [0,1]
    NDArray<double> gradO('c', {bS, nOut});

    const OpArgsHolder<double> argsHolderFF({&x}, {clip}, {axis});
    const OpArgsHolder<double> argsHolderBP({&x, &gradO}, {clip}, {axis});

    nd4j::ops::clipbynorm<double> opFF;
    nd4j::ops::clipbynorm_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_bp_check_1) {

    NDArray<double>       x('c', {4, 4});
    NDArray<double>   gradO('c', {4, 4});

    x.linspace(1);

    const OpArgsHolder<double> argsHolderFF({&x},         {}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &gradO}, {}, {});

    nd4j::ops::cumprod<double> opFF;
    nd4j::ops::cumprod_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_bp_check_2) {

    NDArray<double>       x('c', {4, 4});
    NDArray<double>   gradO('c', {4, 4});

    x.linspace(1);

    const OpArgsHolder<double> argsHolderFF({&x},         {}, {1, 1});
    const OpArgsHolder<double> argsHolderBP({&x, &gradO}, {}, {1, 1});

    nd4j::ops::cumprod<double> opFF;
    nd4j::ops::cumprod_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_bp_check_3) {

    NDArray<double>       x('c', {4, 4});
    NDArray<double>   gradO('c', {4, 4});

    x.linspace(1);

    const OpArgsHolder<double> argsHolderFF({&x},         {}, {1, 0});
    const OpArgsHolder<double> argsHolderBP({&x, &gradO}, {}, {1, 0});

    nd4j::ops::cumprod<double> opFF;
    nd4j::ops::cumprod_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_bp_check_4) {

    NDArray<double>       x('c', {4, 4});
    NDArray<double>   gradO('c', {4, 4});

    x.linspace(1);

    const OpArgsHolder<double> argsHolderFF({&x},         {}, {0, 1});
    const OpArgsHolder<double> argsHolderBP({&x, &gradO}, {}, {0, 1});

    nd4j::ops::cumprod<double> opFF;
    nd4j::ops::cumprod_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumsum_bp_check_2) {

    NDArray<double>       x('c', {4, 4});
    NDArray<double>   gradO('c', {4, 4});

    x.linspace(1);

    const OpArgsHolder<double> argsHolderFF({&x},         {}, {1, 1});
    const OpArgsHolder<double> argsHolderBP({&x, &gradO}, {}, {1, 1});

    nd4j::ops::cumsum<double> opFF;
    nd4j::ops::cumsum_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_test1) {
    
    NDArray<double> inputC('c', {3, 5},   {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.});    
    NDArray<double> axis(1.);

    NDArray<double> expFF('c', {3, 5}, {1.,   2.,   6.,    24.,   120., 6.,  42., 336.,  3024., 30240.,11., 132.,1716., 24024.,360360.});
    NDArray<double> expTF('c', {3, 5}, {1, 1, 2, 6, 24,1, 6, 42, 336, 3024,1, 11, 132, 1716, 24024});

    NDArray<double> expFT('c', {3, 5}, {120, 120, 60, 20, 5,30240, 5040, 720, 90, 10,360360, 32760, 2730, 210, 15});    //+++
    NDArray<double> expTT('c', {3, 5}, {120, 60, 20, 5, 1,5040, 720, 90, 10, 1,32760, 2730, 210, 15, 1});
    NDArray<double> gradO('c', {3, 5});

    int exclusive, reverse;    

    //************************************//
    exclusive = 0; reverse = 0;

    const OpArgsHolder<double> argsHolderFF({&inputC, &axis}, {}, {exclusive, reverse});
    const OpArgsHolder<double> argsHolderBP({&inputC, &axis, &gradO}, {}, {exclusive, reverse});

    nd4j::ops::cumprod<double> opFF;
    nd4j::ops::cumprod_bp<double> opBP;

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
    
    NDArray<double> inputC('c', {2, 2});
    NDArray<double> axis(1.);

//    NDArray<double> expFF('c', {3, 5}, {1.,   2.,   6.,    24.,   120., 6.,  42., 336.,  3024., 30240.,11., 132.,1716., 24024.,360360.});
//    NDArray<double> expTF('c', {3, 5}, {1, 1, 2, 6, 24,1, 6, 42, 336, 3024,1, 11, 132, 1716, 24024});

//    NDArray<double> expFT('c', {3, 5}, {120, 120, 60, 20, 5,30240, 5040, 720, 90, 10,360360, 32760, 2730, 210, 15});    //+++
//    NDArray<double> expTT('c', {3, 5}, {120, 60, 20, 5, 1,5040, 720, 90, 10, 1,32760, 2730, 210, 15, 1});
    NDArray<double> gradO('c', {2, 2});

    int exclusive, reverse;    

    //************************************//
    exclusive = 0; reverse = 0;
    inputC.linspace(1);
    const OpArgsHolder<double> argsHolderFF({&inputC, &axis}, {}, {exclusive, reverse});
    const OpArgsHolder<double> argsHolderBP({&inputC, &axis, &gradO}, {}, {exclusive, reverse});

    nd4j::ops::cumprod<double> opFF;
    nd4j::ops::cumprod_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP, {1, 1, 1, 1}, {1, 1},GradCheck::MEAN);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, cumprod_test3) {
    
    NDArray<double> inputC('c', {2, 2});
    NDArray<double> axis(1.);

//    NDArray<double> expFF('c', {3, 5}, {1.,   2.,   6.,    24.,   120., 6.,  42., 336.,  3024., 30240.,11., 132.,1716., 24024.,360360.});
//    NDArray<double> expTF('c', {3, 5}, {1, 1, 2, 6, 24,1, 6, 42, 336, 3024,1, 11, 132, 1716, 24024});

//    NDArray<double> expFT('c', {3, 5}, {120, 120, 60, 20, 5,30240, 5040, 720, 90, 10,360360, 32760, 2730, 210, 15});    //+++
//    NDArray<double> expTT('c', {3, 5}, {120, 60, 20, 5, 1,5040, 720, 90, 10, 1,32760, 2730, 210, 15, 1});
    NDArray<double> gradO('c', {2, 2});

    int exclusive, reverse;    

    //************************************//
    exclusive = 0; reverse = 0;
    inputC.linspace(1);
//    const OpArgsHolder<double> argsHolderFF({&inputC, &axis}, {}, {exclusive, reverse});
//    const OpArgsHolder<double> argsHolderBP({&inputC, &axis, &gradO}, {}, {exclusive, reverse});

    nd4j::ops::cumprod<double> opFF;
//    nd4j::ops::cumprod_bp<double> opBP;
    auto res = opFF.execute({&inputC, &axis}, {}, {exclusive, reverse});
//    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    res->at(0)->printIndexedBuffer("Cumulative product of 4 ints");
//    ASSERT_TRUE(isGradCorrect);
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test1) {
    
    NDArray<float> x('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});    
    NDArray<float> alpha('c', {3, 4}, {-0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f});
    NDArray<float> exp('c', {2, 3, 4}, {7.2f, 5.5f, 4.f, 2.7f, 1.6f, 0.7f, 0.f, -0.5f,-0.8f, -0.9f, -0.8f, -0.5f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});    

    nd4j::ops::prelu<float> op;

    auto result = op.execute({&x, &alpha}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test2) {
    
    NDArray<float> x('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});    
    NDArray<float> alpha('c', {3}, {-0.6f, 2.f, 4.f});
    NDArray<float> exp('c', {2, 3, 4}, {7.2f,  6.6f,   6.f,   5.4f, -16.f, -14.f, -12.f, -10.f, -16.f, -12.f,  -8.f,  -4.f, 0.f,   1.f,   2.f,   3.f, 4.f,   5.f,   6.f,   7.f, 8.f,   9.f,  10.f,  11.f});    

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test3) {
    
    NDArray<float> x('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});    
    NDArray<float> alpha('c', {3,1}, {-0.6f, 2.f, 4.f});
    NDArray<float> exp('c', {2, 3, 4}, {7.2f,  6.6f,   6.f,   5.4f, -16.f, -14.f, -12.f, -10.f, -16.f, -12.f,  -8.f,  -4.f, 0.f,   1.f,   2.f,   3.f, 4.f,   5.f,   6.f,   7.f, 8.f,   9.f,  10.f,  11.f});    

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test4) {
    
    NDArray<float> x('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});    
    NDArray<float> alpha('c', {1, 3}, {-0.6f, 2.f, 4.f});
    NDArray<float> exp('c', {2, 3, 4}, {7.2f,  6.6f,   6.f,   5.4f, -16.f, -14.f, -12.f, -10.f, -16.f, -12.f,  -8.f,  -4.f, 0.f,   1.f,   2.f,   3.f, 4.f,   5.f,   6.f,   7.f, 8.f,   9.f,  10.f,  11.f});    

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test5) {
    
    NDArray<float> x('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});    
    NDArray<float> alpha('c', {4}, {-0.6f, 2.f, 4.f, -1.f});
    NDArray<float> exp('c', {2, 3, 4}, {7.2f, -22.f, -40.f,   9.f, 4.8f, -14.f, -24.f,   5.f, 2.4f,  -6.f,  -8.f,   1.f, 0.f,   1.f,   2.f,   3.f, 4.f,   5.f,   6.f,   7.f, 8.f,   9.f,  10.f,  11.f});    

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test6) {
    
    NDArray<float> x('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});    
    NDArray<float> alpha('c', {1,1,1}, {-2.});
    NDArray<float> exp('c', {2, 3, 4}, {24.f, 22.f, 20.f, 18.f, 16.f, 14.f, 12.f, 10.f, 8.f,  6.f,  4.f,  2.f, 0.f,  1.f,  2.f,  3.f, 4.f,  5.f,  6.f,  7.f, 8.f,  9.f, 10.f, 11.f});    

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {1,0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test7) {
    
    NDArray<float> x('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});    
    NDArray<float> alpha(-2.f);
    NDArray<float> exp('c', {2, 3, 4}, {24.f, 22.f, 20.f, 18.f, 16.f, 14.f, 12.f, 10.f, 8.f,  6.f,  4.f,  2.f, 0.f,  1.f,  2.f,  3.f, 4.f,  5.f,  6.f,  7.f, 8.f,  9.f, 10.f, 11.f});    

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {1,0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test8) {
    
    NDArray<float> x('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});    
    NDArray<float> alpha(-2.f);
    NDArray<float> exp('c', {2, 3, 4}, {24.f, 22.f, 20.f, 18.f, 16.f, 14.f, 12.f, 10.f, 8.f,  6.f,  4.f,  2.f, 0.f,  1.f,  2.f,  3.f, 4.f,  5.f,  6.f,  7.f, 8.f,  9.f, 10.f, 11.f});    

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {1,0,1,0,1,0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test9) {
    
    NDArray<float> x('c', {2, 4}, {-4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f});    
    NDArray<float> alpha(-2.f);
    NDArray<float> exp('c', {2, 4}, {8.f, 6.f, 4.f, 2.f,0.f, 1.f, 2.f, 3.f});

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test10) {
    
    NDArray<float> x('c', {2, 4}, {-4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f});    
    NDArray<float> alpha(-2.f);
    NDArray<float> exp('c', {2, 4}, {8.f, 6.f, 4.f, 2.f,0.f, 1.f, 2.f, 3.f});

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test11) {
    
    NDArray<float> x('c', {2, 3, 4, 5});    
    x.linspace(-50.);
    NDArray<float> alpha('c', {4}, {0.f, -0.5f, 0.5f, -1.f});
    NDArray<float> exp('c', {2, 3, 4, 5}, {0.f,   0.f,   0.f,   0.f,   0.f, 22.5f,  22.f,  21.5f,  21.f,  20.5f, -20.f, -19.5f, -19.f, -18.5f, -18.f, 35.f,  34.f,  33.f,  
                                           32.f,  31.f, 0.f,   0.f,   0.f,   0.f,   0.f, 12.5f,  12.f,  11.5f,  11.f,  10.5f, -10.f,  -9.5f,  -9.f,  -8.5f,  -8.f, 15.f,  
                                           14.f,  13.f,  12.f,  11.f, 0.f,   0.f,   0.f,   0.f,   0.f, 2.5f,   2.f,   1.5f,   1.f,   0.5f, 0.f,   1.f,   2.f,   3.f,   4.f, 
                                           5.f,   6.f,   7.f,   8.f,   9.f, 10.f,  11.f,  12.f,  13.f,  14.f, 15.f,  16.f,  17.f,  18.f,  19.f, 20.f,  21.f,  22.f,  23.f,  
                                           24.f, 25.f,  26.f,  27.f,  28.f,  29.f, 30.f,  31.f,  32.f,  33.f,  34.f, 35.f,  36.f,  37.f,  38.f,  39.f, 40.f,  41.f,  42.f,  
                                           43.f,  44.f, 45.f,  46.f,  47.f,  48.f,  49.f, 50.f,  51.f,  52.f,  53.f,  54.f, 55.f,  56.f,  57.f,  58.f,  59.f, 60.f,  61.f,  
                                           62.f,  63.f,  64.f, 65.f,  66.f,  67.f,  68.f,  69.f});

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {1,3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test12) {
    
    NDArray<float> x('c', {2, 3, 4, 5});    
    x.linspace(-50.);
    NDArray<float> alpha('c', {3,5}, {-0.7f, -0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f});
    NDArray<float> exp('c', {2, 3, 4, 5}, {35.f, 29.4f, 24.f, 18.8f, 13.8f, 31.5f, 26.4f, 21.5f, 16.8f, 12.3f, 28.f, 23.4f, 19.f, 14.8f, 10.8f, 24.5f, 20.4f, 16.5f, 12.8f,  
                                           9.3f, 6.f,  2.9f,  0.f, -2.7f, -5.2f, 5.f,  2.4f,  0.f, -2.2f, -4.2f, 4.f,  1.9f,  0.f, -1.7f, -3.2f, 3.f,  1.4f,  0.f, -1.2f, 
                                           -2.2f, -3.f, -3.6f, -4.f, -4.2f, -4.2f, -1.5f, -1.6f, -1.5f, -1.2f, -0.7f, 0.f,  1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f,  
                                           9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 
                                           31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 49.f, 50.f, 51.f, 52.f, 
                                           53.f, 54.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 67.f, 68.f, 69.f});

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {-1, 2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test13) {
    
    NDArray<float> x('c', {2, 3, 4, 5});    
    x.linspace(-50.);
    NDArray<float> alpha('c', {5,3}, {-0.7f, -0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f});
    NDArray<float> exp('c', {2, 3, 4, 5}, {35.f, 29.4f, 24.f, 18.8f, 13.8f, 31.5f, 26.4f, 21.5f, 16.8f, 12.3f, 28.f, 23.4f, 19.f, 14.8f, 10.8f, 24.5f, 20.4f, 16.5f, 12.8f,  
                                           9.3f, 6.f,  2.9f,  0.f, -2.7f, -5.2f, 5.f,  2.4f,  0.f, -2.2f, -4.2f, 4.f,  1.9f,  0.f, -1.7f, -3.2f, 3.f,  1.4f,  0.f, -1.2f, 
                                           -2.2f, -3.f, -3.6f, -4.f, -4.2f, -4.2f, -1.5f, -1.6f, -1.5f, -1.2f, -0.7f, 0.f,  1.f,  2.f,  3.f,  4.f, 5.f,  6.f,  7.f,  8.f,  
                                           9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 
                                           31.f, 32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f, 48.f, 49.f, 50.f, 51.f, 52.f, 
                                           53.f, 54.f, 55.f, 56.f, 57.f, 58.f, 59.f, 60.f, 61.f, 62.f, 63.f, 64.f, 65.f, 66.f, 67.f, 68.f, 69.f});

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {-1, 2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_test14) {
    
    NDArray<float> x('c', {2, 3, 4, 5});    
    x.linspace(-50.);
    NDArray<float> alpha('c', {2,10}, {-0.7f, -0.6f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f, 0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f, 1.1f, 1.2f});
    NDArray<float> exp('c', {2, 3, 4, 5}, {35.f,  29.4f,  24.f,  18.8f,  13.8f, 9.f,   4.4f,   0.f,  -4.2f,  -8.2f, -12.f, -15.6f, -19.f, -22.2f, -25.2f, -28.f, -30.6f, 
                                           -33.f,-35.2f, -37.2f, 21.f,  17.4f,  14.f,  10.8f,   7.8f, 5.f,   2.4f,   0.f,  -2.2f,  -4.2f, -6.f,  -7.6f,  -9.f, -10.2f, 
                                           -11.2f, -12.f, -12.6f, -13.f, -13.2f, -13.2f, 7.f,   5.4f,   4.f,   2.8f,   1.8f, 1.f,   0.4f,   0.f,  -0.2f,  -0.2f, 0.f,   
                                           1.f,   2.f,   3.f,   4.f, 5.f,   6.f,   7.f,   8.f,   9.f, 10.f,  11.f,  12.f,  13.f,  14.f, 15.f,  16.f,  17.f,  18.f,  
                                           19.f, 20.f,  21.f,  22.f,  23.f,  24.f, 25.f,  26.f,  27.f,  28.f,  29.f, 30.f,  31.f,  32.f,  33.f,  34.f, 35.f,  36.f,  
                                           37.f,  38.f,  39.f, 40.f,  41.f,  42.f,  43.f,  44.f, 45.f,  46.f,  47.f,  48.f,  49.f, 50.f,  51.f,  52.f,  53.f,  54.f, 
                                           55.f,  56.f,  57.f,  58.f,  59.f, 60.f,  61.f,  62.f,  63.f,  64.f, 65.f,  66.f,  67.f,  68.f,  69.f});

    nd4j::ops::prelu<float> op;
    auto result = op.execute({&x, &alpha}, {}, {-2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, thresholdedrelu_test1) {
    
    const float theta = 2.f;
    NDArray<float> x('c', {2, 3, 4}, {-12.f, -11.f, -10.f, -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f});    
    NDArray<float> exp('c', {2, 3, 4}, {0.f, 0.f, 0.f, 0.f,0.f, 0.f, 0.f, 0.f,0.f, 0.f, 0.f, 0.f,0.f, 0.f, 0.f, 3.f,4.f, 5.f, 6.f, 7.f,8.f, 9.f,10.f,11.f});

    nd4j::ops::thresholdedrelu<float> op;

    auto result = op.execute({&x}, {theta}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}
 
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, thresholdedrelu_test2) {
    
    const float theta = -2.f;
    NDArray<float> x('c', {2, 3, 4}, {0.f,-4.f, -10.f, -8.f, 0.f, -9.f, -8.f, 5.f, 6.f, 6.f, 9.f, 6.f, -8.f, 5.f, 10.f, -2.f, 3.f, -7.f, 4.f, -8.f, -4.f, -9.f, -9.f, 3.f});
    NDArray<float> exp('c', {2, 3, 4}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 5.f, 6.f, 6.f, 9.f, 6.f, 0.f, 5.f, 10.f, 0.f, 3.f, 0.f, 4.f, 0.f, 0.f, 0.f, 0.f, 3.f});

    nd4j::ops::thresholdedrelu<float> op;

    auto result = op.execute({&x}, {theta}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto output = result->at(0);    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_bp_test1) {
    
    NDArray<double> x('c', {2, 3, 4}, {-12., -11., -10., -9., -8., -7., -6., -5., -4., -3., -2., -1., 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.});
    NDArray<double> alpha('c', {3, 4}, {-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5});
    NDArray<double> dLdO('c', {2, 3, 4});

    const OpArgsHolder<double> argsHolderFF({&x, &alpha}, {}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &alpha, &dLdO}, {}, {});

    nd4j::ops::prelu<double> opFF;
    nd4j::ops::prelu_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_bp_test2) {
    
    NDArray<double> x('c', {2, 3, 4}, {-12., -11., -10., -9., -8., -7., -6., -5., -4., -3., -2., -1., 0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.});    
    NDArray<double> alpha('c', {4}, {-0.6, 2., 4., -1.});
    NDArray<double> dLdO('c', {2, 3, 4});

    const OpArgsHolder<double> argsHolderFF({&x, &alpha}, {}, {1});
    const OpArgsHolder<double> argsHolderBP({&x, &alpha, &dLdO}, {}, {1});

    nd4j::ops::prelu<double> opFF;
    nd4j::ops::prelu_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_bp_test3) {
    
    NDArray<double> x('c', {2, 3, 2, 5});    
    x.linspace(-30.);
    x(30) = 0.5;   // avoid zero, since it is points of discontinuity for prelu
    NDArray<double> alpha('c', {5,3}, {-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7});
    NDArray<double> dLdO('c', {2, 3, 2, 5});

    const OpArgsHolder<double> argsHolderFF({&x, &alpha}, {}, {-1, 2});
    const OpArgsHolder<double> argsHolderBP({&x, &alpha, &dLdO}, {}, {-1, 2});

    nd4j::ops::prelu<double> opFF;
    nd4j::ops::prelu_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, prelu_bp_test4) {
    
    NDArray<double> x('c', {2, 3, 4, 5});    
    x.linspace(-50.);
    x(50) = 0.5;   // avoid zero, since it is points of discontinuity for prele
    NDArray<double> alpha('c', {2,10}, {-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.25, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2});
    NDArray<double> dLdO('c', {2, 3, 4, 5});

    const OpArgsHolder<double> argsHolderFF({&x, &alpha}, {}, {-2});
    const OpArgsHolder<double> argsHolderBP({&x, &alpha, &dLdO}, {}, {-2});

    nd4j::ops::prelu<double> opFF;
    nd4j::ops::prelu_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, thresholdedrelu_bp_test1) {
        
    const double theta = 0.15;

    NDArray<double> x('c', {2, 3, 4}, {1.2, 1.1, 1., 0.9, 0.8, -0.7, -0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.9, -1.0, -1.1});    
    NDArray<double> dLdO('c', {2, 3, 4});

    const OpArgsHolder<double> argsHolderFF({&x}, {theta}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &dLdO}, {theta}, {});

    nd4j::ops::thresholdedrelu<double> opFF;
    nd4j::ops::thresholdedrelu_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_test1) {
    
    NDArray<float> x('c', {2, 3, 4});        
    NDArray<float> y('c', {4});
    NDArray<float> exp('c', {2, 3, 4}, {0.1f, 0.4f, 0.9f, 1.6f, 0.5f, 1.2f, 2.1f, 3.2f, 0.9f, 2.f, 3.3f, 4.8f, 1.3f, 2.8f, 4.5f, 6.4f, 1.7f, 3.6f, 5.7f, 8.f, 2.1f, 4.4f, 6.9f, 9.6f});
    x.linspace(1.f);
    y.linspace(0.1f, 0.1f);

    nd4j::ops::multiply<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_test2) {
    
    NDArray<float> x('c', {2, 3, 4});        
    NDArray<float> y(0.1);
    NDArray<float> exp('c', {2, 3, 4}, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.f, 2.1f, 2.2f, 2.3f, 2.4f});
    x.linspace(1.f);
    // y.linspace(0.1f, 0.1f);

    nd4j::ops::multiply<float> op;
    auto result = op.execute({&y, &x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_test3) {
    
    NDArray<float> x('c', {2, 1, 4});        
    NDArray<float> y('c', {3,1});
    NDArray<float> exp('c', {2, 3, 4}, {0.1f, 0.2f, 0.3f, 0.4f, 0.2f, 0.4f, 0.6f, 0.8f, 0.3f, 0.6f, 0.9f, 1.2f, 0.5f, 0.6f, 0.7f, 0.8f, 1.f, 1.2f, 1.4f, 1.6f, 1.5f, 1.8f, 2.1f, 2.4f});
    x.linspace(1.f);
    y.linspace(0.1f, 0.1f);

    nd4j::ops::multiply<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_test4) {
    
    NDArray<float> x('c', {1, 1});        
    NDArray<float> y(0.1f);
    NDArray<float> exp('c', {1, 1}, {0.1f});
    x.linspace(1.f);    

    nd4j::ops::multiply<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_test5) {
    
    NDArray<float> x(1.f);        
    NDArray<float> y(0.1f);
    NDArray<float> exp(0.1f);
    
    nd4j::ops::multiply<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test1) {
            
    NDArray<double> x('c', {1, 1}, {100.});
    NDArray<double> y(0.1);
    NDArray<double> dLdz('c', {1, 1});

    const OpArgsHolder<double> argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply<double> opFF;
    nd4j::ops::multiply_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test2) {
            
    NDArray<double> x('c', {2, 2}, {1.,2.,3.,4.});
    NDArray<double> y(0.1);
    NDArray<double> dLdz('c', {2, 2});

    const OpArgsHolder<double> argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply<double> opFF;
    nd4j::ops::multiply_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test3) {
            
    NDArray<double> y('c', {2, 2}, {1.,2.,3.,4.});
    NDArray<double> x(0.1);
    NDArray<double> dLdz('c', {2, 2});

    const OpArgsHolder<double> argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply<double> opFF;
    nd4j::ops::multiply_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test4) {
            
    NDArray<double> x('c', {2, 2}, {1.,2.,3.,4.});
    NDArray<double> y('c', {2, 2}, {0.1,0.2,0.3,0.4});
    NDArray<double> dLdz('c', {2, 2});

    const OpArgsHolder<double> argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply<double> opFF;
    nd4j::ops::multiply_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test5) {
            
    NDArray<double> x('c', {2, 2}, {1.,2.,3.,4.});
    NDArray<double> y('c', {2}, {0.1,0.2});
    NDArray<double> dLdz('c', {2, 2});

    const OpArgsHolder<double> argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply<double> opFF;
    nd4j::ops::multiply_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test6) {
            
    NDArray<double> y('c', {2, 2}, {1.,2.,3.,4.});
    NDArray<double> x('c', {2}, {0.1,0.2});
    NDArray<double> dLdz('c', {2, 2});

    const OpArgsHolder<double> argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply<double> opFF;
    nd4j::ops::multiply_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test7) {
            
    NDArray<double> y('c', {2, 3}, {1.,2.,3.,4.,5.,6.});
    NDArray<double> x('c', {2, 1}, {0.1,0.2});
    NDArray<double> dLdz('c', {2, 3});

    const OpArgsHolder<double> argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply<double> opFF;
    nd4j::ops::multiply_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests9, multiply_bp_test8) {
            
    NDArray<double> y('c', {2, 1, 4});
    NDArray<double> x('c', {1, 3, 4});
    NDArray<double> dLdz('c', {2, 3, 4});
    x.linspace(1., 0.5);
    y.linspace(0.1, 0.05);

    const OpArgsHolder<double> argsHolderFF({&x, &y}, {}, {});
    const OpArgsHolder<double> argsHolderBP({&x, &y, &dLdz}, {}, {});

    nd4j::ops::multiply<double> opFF;
    nd4j::ops::multiply_bp<double> opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}
