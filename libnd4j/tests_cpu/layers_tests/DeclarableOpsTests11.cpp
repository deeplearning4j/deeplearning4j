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


class DeclarableOpsTests11 : public testing::Test {
public:

    DeclarableOpsTests11() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests11, test_listdiff_1) {
    auto x = NDArrayFactory::create<int>('c', {4}, {0, 1, 2, 3});
    auto y = NDArrayFactory::create<int>('c',{2}, {3, 1});

    nd4j::ops::listdiff op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test1) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-12.49997,-13.04346, -13.63635, -14.28571,-14.99999,-15.78947, -16.66666, -17.64705,-18.75   ,-20.     , -21.42857, -23.07692,
                                   -24.99999,-27.27272, -29.99999, -33.33332,-37.49999,-42.85713, -49.99998, -59.99998,-74.99995,-99.99992,-149.99986,-299.99911});
    NDArray dLdwExp('c', {2,3,4}, {3.21887,  4.96807,  6.10512,  6.80726,  7.15461,  7.19051,  6.93973,  6.41584,  5.62456,  4.56548,  3.2326 ,  1.61444,
                                   -0.30659, -2.55529, -5.16569, -8.18417,-11.67468,-15.72734,-20.47379,-26.11644,-32.9902 ,-41.71318,-53.64824,-73.05434});
    NDArray dLdlExp('c', {2,3,4}, {1.58903, 1.22117, 0.99621, 0.82911, 0.69315, 0.57634, 0.47223, 0.37689, 0.28768, 0.20273, 0.12058, 0.04002,
                                   -0.04002,-0.12058,-0.20273,-0.28768,-0.37689,-0.47223,-0.57634,-0.69315,-0.82911,-0.99621,-1.22117,-1.58903});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {0}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test2) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,1,4}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {2,1,4}, {15.99805, 16.72406, 16.27746,  14.83754,-44.97147,-59.99582,-79.28771,-107.35497});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test3) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights(nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-12.49997,-13.04346, -13.63635, -14.28571,-14.99999,-15.78947, -16.66666, -17.64705,-18.75   ,-20.     , -21.42857, -23.07692,
                                   -24.99999,-27.27272, -29.99999, -33.33332,-37.49999,-42.85713, -49.99998, -59.99998,-74.99995,-99.99992,-149.99986,-299.99911});
    NDArray dLdwExp('c', {}, {-227.77286});
    NDArray dLdlExp('c', {2,3,4}, {1.58903, 1.22117, 0.99621, 0.82911, 0.69315, 0.57634, 0.47223, 0.37689, 0.28768, 0.20273, 0.12058, 0.04002,
                                   -0.04002,-0.12058,-0.20273,-0.28768,-0.37689,-0.47223,-0.57634,-0.69315,-0.82911,-0.99621,-1.22117,-1.58903});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test4) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {4.8876 , -46.29156, -186.36887});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);
    // dLdw->printIndexedBuffer();
    // dLdw->printShapeInfo();

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test5) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-1.04166,-1.08696, -1.13636, -1.19048,-1.25   ,-1.31579, -1.38889, -1.47059,-1.5625 ,-1.66667, -1.78571, -1.92308,
                                   -2.08333,-2.27273, -2.5    , -2.77778,-3.125  ,-3.57143, -4.16667, -5.     ,-6.25   ,-8.33333,-12.49999,-24.99993});
    NDArray dLdwExp('c', {2,3,4}, {1.05912, 1.20488, 1.29964, 1.35815, 1.3871 , 1.39009, 1.36919, 1.32553, 1.25959, 1.17133, 1.06026, 0.92541,
                                   0.76533, 0.57794, 0.3604 , 0.10886,-0.18201,-0.51973,-0.91527,-1.38549,-1.95831,-2.68522,-3.67981,-5.29698});
    NDArray dLdlExp('c', {2,3,4}, {0.13242, 0.10176, 0.08302, 0.06909, 0.05776, 0.04803, 0.03935, 0.03141, 0.02397, 0.01689, 0.01005, 0.00334,
                                   -0.00334,-0.01005,-0.01689,-0.02397,-0.03141,-0.03935,-0.04803,-0.05776,-0.06909,-0.08302,-0.10176,-0.13242});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test6) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {6.73432, 2.46939,-9.20372});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test7) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights(nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {}, {0.});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test8) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {0.     , 0.     ,  0.     ,  0.     ,-1.5    ,-1.57895, -1.66667, -1.76471,-1.875  ,-2.     , -2.14286, -2.30769,
                                  -2.5    ,-2.72727, -3.     , -3.33333,-3.75   ,-4.28571, -5.     , -6.     ,-7.49999,-9.99999,-14.99999,-29.99991});
    NDArray dLdwExp('c', {2,3,4}, {1.56625, 1.74117, 1.85487, 1.92509, 1.95982, 1.96341, 1.93833, 1.88594, 1.80682, 1.70091, 1.56762, 1.4058 ,
                                   1.2137 , 0.98883, 0.72779, 0.42594, 0.07689,-0.32837,-0.80302,-1.36728,-2.05466,-2.92696,-4.12046,-6.06107});
    NDArray dLdlExp('c', {2,3,4}, {0.     , 0.     , 0.     , 0.     , 0.06931, 0.05763, 0.04722, 0.03769, 0.02877, 0.02027, 0.01206, 0.004,
                                  -0.004  ,-0.01206,-0.02027,-0.02877,-0.03769,-0.04722,-0.05763,-0.06931,-0.08291,-0.09962,-0.12212,-0.1589});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);
    weights.p(3, 0.);

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test9) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.52083,-0.54348,-0.56818, -0.59524,-0.625  ,-0.65789,-0.69444, -0.73529,-0.78125,-0.83333,-0.89286, -0.96154,
                                   -1.04167,-1.13636,-1.25   , -1.38889,-1.5625 ,-1.78571,-2.08333, -2.5    ,-3.125  ,-4.16666,-6.24999,-12.49996});
    NDArray dLdwExp('c', {2,3,4}, {0.13412, 0.207  , 0.25438, 0.28364, 0.29811, 0.2996 , 0.28916, 0.26733, 0.23436, 0.19023, 0.13469, 0.06727,
                                  -0.01277,-0.10647,-0.21524,-0.34101,-0.48645,-0.65531,-0.85307,-1.08819,-1.37459,-1.73805,-2.23534,-3.04393});
    NDArray dLdlExp('c', {2,3,4}, {0.06621, 0.05088, 0.04151, 0.03455, 0.02888, 0.02401, 0.01968, 0.0157 , 0.01199, 0.00845, 0.00502, 0.00167,
                                  -0.00167,-0.00502,-0.00845,-0.01199,-0.0157 ,-0.01968,-0.02401,-0.02888,-0.03455,-0.04151,-0.05088,-0.06621});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test10) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,1}, {-9.49054});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test11) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {0.20365,-1.92882,-7.76537});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test12) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, { 0.      , 0.      , 0.      ,  0.      ,-0.75    ,-0.789473,-0.833333, -0.882353,-0.9375  ,-1.      ,-1.071428, -1.153846,
                                -1.25    ,-1.363636,-1.5     , -1.666666,-1.875   ,-2.142857,-2.499999, -2.999999,-3.749997,-4.999997,-7.499993,-14.999956});
    NDArray dLdwExp('c', {2,3,4}, {0.16094, 0.2484 , 0.30526, 0.34036, 0.35773, 0.35953, 0.34699, 0.32079, 0.28123, 0.22827, 0.16163, 0.08072,
                                   -0.01533,-0.12776,-0.25828,-0.40921,-0.58373,-0.78637,-1.02369,-1.30582,-1.64951,-2.08566,-2.68241,-3.65272});
    NDArray dLdlExp('c', {2,3,4}, {0.     , 0.     , 0.     , 0.     , 0.03466, 0.02882, 0.02361, 0.01884, 0.01438, 0.01014, 0.00603, 0.002  ,
                                  -0.002  ,-0.00603,-0.01014,-0.01438,-0.01884,-0.02361,-0.02882,-0.03466,-0.04146,-0.04981,-0.06106,-0.07945});


    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.t<double>(0) = 0.;
    weights.t<double>(1) = 0.;
    weights.t<double>(2) = 0.;
    weights.t<double>(3) = 0.;


    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, log_loss_grad_test13) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {0.     , 0.     ,  0.     ,  0.     , 0.     , 0.     ,  0.     ,  0.     , 0.     , 0.     ,  0.     ,  0.     ,
                                  -2.08333,-2.27273, -2.5    , -2.77778,-3.125  ,-3.57143, -4.16667, -5.     ,-6.25   ,-8.33333,-12.49999,-24.99993});
    NDArray dLdwExp('c', {2,3,1}, {1.75828,  2.30839,  1.25309, -1.35098, -6.16602,-16.78383});
    NDArray dLdlExp('c', {2,3,4}, {0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     , 0.     ,
                                   -0.00334,-0.01005,-0.01689,-0.02397,-0.03141,-0.03935,-0.04803,-0.05776,-0.06909,-0.08302,-0.10176,-0.13242});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.t<double>(0) = 0.;
    weights.t<double>(1) = 0.;
    weights.t<double>(2) = 0.;

    nd4j::ops::log_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, summaryStatsData_test1) {

    functions::summarystats::SummaryStatsData<double> var1;
    functions::summarystats::SummaryStatsData<double> var2;
    var2.n = var2.mean = var2.M2 = var2.M3 = var2.M4 = var2.bias = 5;

    functions::summarystats::SummaryStatsData<double>* arr = new functions::summarystats::SummaryStatsData<double>[2];
    arr[0] = var1;
    arr[1] = var2;
    arr[0] = arr[1];

    functions::summarystats::SummaryStatsData<double> var3(var1);

    ASSERT_TRUE(arr[0].n == arr[0].mean && arr[0].M2 == arr[0].M3 && arr[0].n == 5);
    ASSERT_TRUE(arr[1].n == arr[1].mean && arr[1].M2 == arr[1].M3 && arr[1].n == 5);
    ASSERT_TRUE(var3.n == var3.mean && var3.M2 == var3.M3 && var3.n == 0);

    delete []arr;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test1) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.96, -1.92, -2.88, -3.84, -4.8 , -5.76, -6.72, -7.68, -8.64, -9.6 ,-10.56,-11.52,
                                   -12.48,-13.44,-14.4 ,-15.36,-16.32,-17.28,-18.24,-19.2 ,-20.16,-21.12,-22.08,-23.04});
    NDArray dLdwExp('c', {2,3,4}, {0.9216 ,  3.6864 ,  8.2944 , 14.7456 , 23.04   , 33.1776 , 45.1584 , 58.9824 , 74.6496 , 92.16   ,111.51361,132.7104 ,
                                   155.75038,180.63359,207.35999,235.9296 ,266.34238,298.59842,332.6976 ,368.64001,406.4256 ,446.05444,487.5264 ,530.84161});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto dLdp = results->at(0);
    auto dLdw = results->at(1);
    auto dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test2) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,1,4}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {2,1,4}, {98.61121,129.024  , 164.9664 , 206.4384 , 828.51837,925.28644,1027.58398,1135.41113});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test3) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights(nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.96, -1.92, -2.88, -3.84, -4.8 , -5.76, -6.72, -7.68, -8.64, -9.6 ,-10.56,-11.52,
                                   -12.48,-13.44,-14.4 ,-15.36,-16.32,-17.28,-18.24,-19.2 ,-20.16,-21.12,-22.08,-23.04});
    NDArray dLdwExp('c', {}, {4515.84});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test4) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {807.32153, 1426.63684, 2281.88159});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test5) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.08,-0.16,-0.24,-0.32,-0.4 ,-0.48,-0.56,-0.64,-0.72,-0.8 ,-0.88,-0.96,
                                   -1.04,-1.12,-1.2 ,-1.28,-1.36,-1.44,-1.52,-1.6 ,-1.68,-1.76,-1.84,-1.92});
    NDArray dLdwExp('c', {2,3,4}, {-15.6032,-15.3728,-14.9888,-14.4512,-13.76  ,-12.9152,-11.9168,-10.7648, -9.4592, -8.    , -6.3872, -4.6208,
                                   -2.7008, -0.6272,  1.6   ,  3.9808,  6.5152,  9.2032, 12.0448, 15.04  , 18.1888, 21.4912, 24.9472, 28.5568});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test6) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {-58.16319, -6.5536 , 64.71682});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test7) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights(nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {}, {0.});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test8) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {0. ,0. ,0. ,0. ,-0.48 ,-0.576,-0.672,-0.768,-0.864,-0.96 ,-1.056,-1.152,
                                 -1.248,-1.344,-1.44 ,-1.536,-1.632,-1.728,-1.824,-1.92 ,-2.016,-2.112,-2.208,-2.304});
    NDArray dLdwExp('c', {2,3,4}, {-22.3488 ,-22.07232,-21.61152,-20.9664 ,-20.13696,-19.1232 ,-17.92512,-16.54272,-14.976  ,-13.22496,-11.2896 , -9.16992,
                                   -6.86592, -4.3776 , -1.70496,  1.152  ,  4.19328,  7.41888, 10.8288 , 14.42304, 18.2016 , 22.16449, 26.31168, 30.6432 });

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);
    weights.p(3, 0.);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test9) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.04,-0.08,-0.12,-0.16,-0.2 ,-0.24,-0.28,-0.32,-0.36,-0.4 ,-0.44,-0.48,
                                   -0.52,-0.56,-0.6 ,-0.64,-0.68,-0.72,-0.76,-0.8 ,-0.84,-0.88,-0.92,-0.96});
    NDArray dLdwExp('c', {2,3,4}, {0.0384, 0.1536, 0.3456, 0.6144, 0.96  , 1.3824, 1.8816, 2.4576, 3.1104, 3.84  , 4.6464, 5.5296,
                                    6.4896, 7.5264, 8.64  , 9.8304,11.0976,12.4416,13.8624,15.36  ,16.9344,18.5856,20.3136,22.1184});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test10) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,1}, {188.16});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test11) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {33.6384 ,59.4432 ,95.07841});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test12) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {0.,0.,0.,0., -0.24 ,-0.288,-0.336,-0.384,-0.432,-0.48 ,-0.528,-0.576,
                                  -0.624,-0.672,-0.72 ,-0.768,-0.816,-0.864,-0.912,-0.96 ,-1.008,-1.056,-1.104,-1.152});
    NDArray dLdwExp('c', {2,3,4}, {0.04608, 0.18432, 0.41472, 0.73728, 1.152  , 1.65888, 2.25792, 2.94912, 3.73248, 4.608  , 5.57568, 6.63552,
                                   7.78752, 9.03168,10.368  ,11.79648,13.31712,14.92992,16.63488,18.432  ,20.32128,22.30272,24.37632,26.54208});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.t<double>(0) = 0.;
    weights.t<double>(1) = 0.;
    weights.t<double>(2) = 0.;
    weights.t<double>(3) = 0.;

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, mean_sqerr_loss_grad_test13) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  -1.04,-1.12,-1.2 ,-1.28,-1.36,-1.44,-1.52,-1.6 ,-1.68,-1.76,-1.84,-1.92});
    NDArray dLdwExp('c', {2,3,1}, {2.304  , 13.3632 , 34.2528 , 64.97279,105.5232 ,155.90401});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.t<double>(0) = 0.;
    weights.t<double>(1) = 0.;
    weights.t<double>(2) = 0.;

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

TEST_F(DeclarableOpsTests11, SquaredSubtractTest_Test1) {
    auto x = NDArrayFactory::create<float>('c', {4}, {0, 1, 2, 3});
    auto y = NDArrayFactory::create<float>('c',{4}, {3, 2, 1, 0});
    auto exp = NDArrayFactory::create<float>('c', {4}, {9, 1,1, 9});
    nd4j::ops::squaredsubtract op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(exp.equalsTo(result->at(0)));

    delete result;
}

TEST_F(DeclarableOpsTests11, SquaredSubtractTest_Test2) {
    auto x = NDArrayFactory::create<float>('c', {2, 4}, {0, 1, 2, 3, 0, 1, 2, 3});
    auto y = NDArrayFactory::create<float>('c',{4}, {3, 2, 1, 0});
    auto exp = NDArrayFactory::create<float>('c', {2, 4}, {9, 1,1, 9, 9, 1, 1, 9});
    nd4j::ops::squaredsubtract op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

TEST_F(DeclarableOpsTests11, SquaredSubtractTest_Test3) {
    auto x = NDArrayFactory::create<float>('c', {2, 4}, {0, 1, 2, 3, 0, 1, 2, 3});
    auto y = NDArrayFactory::create<float>('c',{4}, {3, 2, 1, 0});
    auto exp = NDArrayFactory::create<float>('c', {2, 4}, {-6, -4, 6, 24, -30, -12, 14, 48});
    auto eps = NDArrayFactory::create<float>('c', {2, 4}, {1,2,3,4,5,6,7,8});
    nd4j::ops::squaredsubtract_bp op;
    auto result = op.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(exp.equalsTo(result->at(0)));
    delete result;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test1) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,
                                   -0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5});
    NDArray dLdwExp('c', {2,3,4}, {0.96, 1.92, 2.88, 3.84, 4.8 , 5.76, 6.72, 7.68, 8.64, 9.6 ,10.56,11.52,
                                  12.48,13.44,14.4 ,15.36,16.32,17.28,18.24,19.2 ,20.16,21.12,22.08,23.04});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto dLdp = results->at(0);
    auto dLdw = results->at(1);
    auto dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test2) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,1,4}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {2,1,4}, {14.4 , 17.28, 20.16, 23.04, 48.96, 51.84, 54.72, 57.6});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test3) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights(nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,
                                   -0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5});
    NDArray dLdwExp('c', {}, {288.});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test4) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {65.28, 96., 126.72001});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test5) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,
                                   -0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167,-0.04167});
    NDArray dLdwExp('c', {2,3,4}, {-0.92,-0.84,-0.76,-0.68,-0.6 ,-0.52,-0.44,-0.36,-0.28,-0.2 ,-0.12,-0.04,
                                     0.04, 0.12, 0.2 , 0.28, 0.36, 0.44, 0.52, 0.6 , 0.68, 0.76, 0.84, 0.92});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test6) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {-2.56, 0., 2.56});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test7) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights(nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {}, {0.});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test8) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.  ,-0.  ,-0.  ,-0.  ,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,
                                   -0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05});
    NDArray dLdwExp('c', {2,3,4}, {-1.296,-1.2  ,-1.104,-1.008,-0.912,-0.816,-0.72 ,-0.624,-0.528,-0.432,-0.336,-0.24 ,
                                    -0.144,-0.048, 0.048, 0.144, 0.24 , 0.336, 0.432, 0.528, 0.624, 0.72 , 0.816, 0.912});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);
    weights.p(3, 0.);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test9) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.02083, -0.02083, -0.02083, -0.02083,-0.02083, -0.02083, -0.02083, -0.02083,-0.02083, -0.02083, -0.02083, -0.02083,
                                   -0.02083, -0.02083, -0.02083, -0.02083,-0.02083, -0.02083, -0.02083, -0.02083,-0.02083, -0.02083, -0.02083, -0.02083});
    NDArray dLdwExp('c', {2,3,4}, {0.04, 0.08, 0.12, 0.16, 0.2 , 0.24, 0.28, 0.32,0.36, 0.4 , 0.44, 0.48,
                                   0.52, 0.56, 0.6 , 0.64,0.68, 0.72, 0.76, 0.8 ,0.84, 0.88, 0.92, 0.96});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test10) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,1}, {12.});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test11) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {2.72, 4., 5.28});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test12) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {0., 0., 0., 0., -0.025, -0.025, -0.025, -0.025,-0.025, -0.025, -0.025, -0.025,
                                   -0.025, -0.025, -0.025, -0.025,-0.025, -0.025, -0.025, -0.025,-0.025, -0.025, -0.025, -0.025});
    NDArray dLdwExp('c', {2,3,4}, {0.048, 0.096, 0.144, 0.192,0.24 , 0.288, 0.336, 0.384,0.432, 0.48 , 0.528, 0.576,
                                   0.624, 0.672, 0.72 , 0.768,0.816, 0.864, 0.912, 0.96 ,1.008, 1.056, 1.104, 1.152});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.t<double>(0) = 0.;
    weights.t<double>(1) = 0.;
    weights.t<double>(2) = 0.;
    weights.t<double>(3) = 0.;

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, absolute_difference_loss_grad_test13) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.,
                                  -0.04167, -0.04167, -0.04167, -0.04167,-0.04167, -0.04167, -0.04167, -0.04167,-0.04167, -0.04167, -0.04167, -0.04167});
    NDArray dLdwExp('c', {2,3,1}, {0.8 ,2.08,3.36,4.64,5.92,7.2 });

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.t<double>(0) = 0.;
    weights.t<double>(1) = 0.;
    weights.t<double>(2) = 0.;

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdpExp.isSameShape(-*dLdl));
    ASSERT_TRUE(dLdpExp.equalsTo(-*dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, BFloat16_Test_1) {

    NDArray x = NDArrayFactory::create<bfloat16>('c', {2,3,4});
    NDArray y = NDArrayFactory::create<bfloat16>('c', {2,3,4});//('c', {2,3,4}, nd4j::DataType::BFLOAT16);
    NDArray exp = NDArrayFactory::create<bfloat16>('c', {2,3,4});//('c', {2,3,4}, nd4j::DataType::BFLOAT16);

    x.linspace(1);
    y.linspace(1);
    exp.linspace(2,2);
    nd4j::ops::add op;
    auto results = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto res = results->at(0);
    ASSERT_TRUE(res->equalsTo(exp));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, BFloat16_Test_2) {

    NDArray x = NDArrayFactory::create<float16>('c', {2,3,4});
    NDArray y = NDArrayFactory::create<bfloat16>('c', {2,3,4});//('c', {2,3,4}, nd4j::DataType::BFLOAT16);
    NDArray exp = NDArrayFactory::create<float16>('c', {2,3,4});//('c', {2,3,4}, nd4j::DataType::BFLOAT16);

    x.linspace(1);
    y.linspace(1);
    exp.linspace(2,2);
    nd4j::ops::add op;
    auto results = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto res = results->at(0);
    ASSERT_TRUE(res->equalsTo(exp));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, BFloat16_Test_3) {

    NDArray x('c', {2,3,4}, nd4j::DataType::BFLOAT16);
    NDArray y('c', {2,3,4}, nd4j::DataType::BFLOAT16);
    NDArray exp('c', {2,3,4}, nd4j::DataType::BFLOAT16);

    x.linspace(1);
    y.linspace(1);
    exp.linspace(2,2);
    nd4j::ops::add op;
    auto results = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto res = results->at(0);
    ASSERT_TRUE(res->equalsTo(exp));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test1) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.25999, -0.755  , -1.25   , -1.745  , -2.24001, -2.73502, -3.23004, -3.72508, -4.22014, -4.71523, -5.21034, -5.70548,
                                   -6.20066, -6.69587, -7.19113, -7.68643, -8.18177, -8.67717, -9.17262, -9.66813,-10.1637 ,-10.65932,-11.15501,-11.65077});
    NDArray dLdwExp('c', {2,3,4}, {0.73395,  0.75335,  0.69315,  0.55335,  0.33395,  0.03495, -0.34366, -0.80186, -1.33967, -1.95708, -2.65411, -3.43074,
                                  -4.28698, -5.22285, -6.23833, -7.33343, -8.50815, -9.76251,-11.0965 ,-12.51013,-14.00341,-15.57633,-17.2289 ,-18.96113});
    NDArray dLdlExp('c', {2,3,4}, {0.04, 0.02,-0.  ,-0.02,-0.04,-0.06,-0.08,-0.1 ,-0.12,-0.14,-0.16,-0.18,
                                   -0.2 ,-0.22,-0.24,-0.26,-0.28,-0.3 ,-0.32,-0.34,-0.36,-0.38,-0.4 ,-0.42});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test2) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,1,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.18499,-0.53   ,-0.875  ,-1.22   ,-1.56501,-1.91002,-2.25504,-2.60008,-2.94514,-3.29023,-3.63534,-3.98048,
                                   -4.32566,-4.67087,-5.01613,-5.36143,-5.70677,-6.05217,-6.39762,-6.74313,-7.0887 ,-7.43432,-7.78001,-8.12577});
    NDArray dLdwExp('c', {2,1,4}, {0.43622, -0.19079, -0.98462, -1.94525,-18.09855,-20.72768,-23.52373,-26.48669});
    NDArray dLdlExp('c', {2,3,4}, {0.028,  0.014, -0.   , -0.014,-0.028, -0.042, -0.056, -0.07 ,-0.084, -0.098, -0.112, -0.126,
                                   -0.14 , -0.154, -0.168, -0.182,-0.196, -0.21 , -0.224, -0.238,-0.252, -0.266, -0.28 , -0.294});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test3) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights(nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.18499,-0.53   ,-0.875  ,-1.22   ,-1.56501,-1.91002,-2.25504,-2.60008,-2.94514,-3.29023,-3.63534,-3.98048,
                                   -4.32566,-4.67087,-5.01613,-5.36143,-5.70677,-6.05217,-6.39762,-6.74313,-7.0887 ,-7.43432,-7.78001,-8.12577});
    NDArray dLdwExp('c', {}, {-91.52109});
    NDArray dLdlExp('c', {2,3,4}, {0.028,  0.014, -0., -0.014,-0.028, -0.042, -0.056, -0.07 ,-0.084, -0.098, -0.112, -0.126,
                                   -0.14 , -0.154, -0.168, -0.182,-0.196, -0.21 , -0.224, -0.238,-0.252, -0.266, -0.28 , -0.294});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test4) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {-12.54779,-28.13393,-50.83936});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test5) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.01542,-0.04417,-0.07292,-0.10167,-0.13042,-0.15917,-0.18792,-0.21667,-0.24543,-0.27419,-0.30294,-0.33171,
                                   -0.36047,-0.38924,-0.41801,-0.44679,-0.47556,-0.50435,-0.53314,-0.56193,-0.59072,-0.61953,-0.64833,-0.67715});
    NDArray dLdwExp('c', {2,3,4}, {0.37794, 0.37906, 0.37554, 0.36739, 0.35461, 0.33719, 0.31514, 0.28846, 0.25714, 0.22119, 0.18061, 0.13539,
                                   0.08553, 0.03104,-0.02808,-0.09184,-0.16023,-0.23326,-0.31093,-0.39323,-0.48017,-0.57175,-0.66796,-0.76881});
    NDArray dLdlExp('c', {2,3,4}, {0.00233, 0.00117,-0.,-0.00117,-0.00233,-0.0035 ,-0.00467,-0.00583,-0.007  ,-0.00817,-0.00933,-0.0105,
                                   -0.01167,-0.01283,-0.014  ,-0.01517,-0.01633,-0.0175 ,-0.01867,-0.01983,-0.021  ,-0.02217,-0.02333,-0.0245});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test6) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {1.4966 , 0.19776,-1.69436});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test7) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights(nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {}, {0.});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test8) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, { 0.     , 0.     , 0.     , 0.     ,-0.1565 ,-0.191  ,-0.2255 ,-0.26001,-0.29451,-0.32902,-0.36353,-0.39805,
                                   -0.43257,-0.46709,-0.50161,-0.53614,-0.57068,-0.60522,-0.63976,-0.67431,-0.70887,-0.74343,-0.778  ,-0.81258});
    NDArray dLdwExp('c', {2,3,4}, {0.54353, 0.54487, 0.54065, 0.53087, 0.51553, 0.49463, 0.46817, 0.43615, 0.39857, 0.35543, 0.30672, 0.25246,
                                   0.19264, 0.12725, 0.0563 ,-0.02021,-0.10228,-0.18992,-0.28312,-0.38188,-0.48621,-0.5961 ,-0.71156,-0.83258});
    NDArray dLdlExp('c', {2,3,4}, {-0.    ,-0.    , 0.    , 0.    ,-0.0028,-0.0042,-0.0056,-0.007 ,-0.0084,-0.0098,-0.0112,-0.0126,
                                  -0.014 ,-0.0154,-0.0168,-0.0182,-0.0196,-0.021 ,-0.0224,-0.0238,-0.0252,-0.0266,-0.028 ,-0.0294});
    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.p(0, 0.);
    weights.p(1, 0.);
    weights.p(2, 0.);
    weights.p(3, 0.);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test9) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.00771, -0.02208, -0.03646, -0.05083,-0.06521, -0.07958, -0.09396, -0.10834,-0.12271, -0.13709, -0.15147, -0.16585,
                                   -0.18024, -0.19462, -0.20901, -0.22339,-0.23778, -0.25217, -0.26657, -0.28096,-0.29536, -0.30976, -0.32417, -0.33857});
    NDArray dLdwExp('c', {2,3,4}, {0.03008,  0.03064,  0.02888,  0.02481, 0.01841,  0.00971, -0.00132, -0.01466,-0.03032, -0.0483 , -0.06859, -0.0912 ,
                                   -0.11612, -0.14337, -0.17293, -0.20481,-0.23901, -0.27552, -0.31435, -0.35551,-0.39898, -0.44476, -0.49287, -0.5433 });
    NDArray dLdlExp('c', {2,3,4}, {0.00117,  0.00058, -0.     , -0.00058,-0.00117, -0.00175, -0.00233, -0.00292,-0.0035 , -0.00408, -0.00467, -0.00525,
                                   -0.00583, -0.00642, -0.007  , -0.00758,-0.00817, -0.00875, -0.00933, -0.00992,-0.0105 , -0.01108, -0.01167, -0.01225});
    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test10) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,1}, {-3.81338});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test11) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdwExp('c', {1,3,1}, {-0.52282,-1.17225,-2.11831});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdw = results->at(1);

    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test12) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {0.     ,  0.     ,  0.     ,  0.     ,-0.07825, -0.0955 , -0.11275, -0.13   ,-0.14726, -0.16451, -0.18177, -0.19902,
                                   -0.21628, -0.23354, -0.25081, -0.26807,-0.28534, -0.30261, -0.31988, -0.33716,-0.35443, -0.37172, -0.389  , -0.40629});
    NDArray dLdwExp('c', {2,3,4}, {0.0361 ,  0.03677,  0.03466,  0.02977, 0.0221 ,  0.01165, -0.00158, -0.01759,-0.03638, -0.05795, -0.08231, -0.10944,
                                   -0.13935, -0.17204, -0.20752, -0.24577,-0.28681, -0.33063, -0.37723, -0.42661,-0.47877, -0.53372, -0.59144, -0.65196});
    NDArray dLdlExp('c', {2,3,4}, {-0.    , -0.    ,  0.    ,  0.    ,-0.0014, -0.0021, -0.0028, -0.0035,-0.0042, -0.0049, -0.0056, -0.0063,
                                   -0.007 , -0.0077, -0.0084, -0.0091,-0.0098, -0.0105, -0.0112, -0.0119,-0.0126, -0.0133, -0.014 , -0.0147});
    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.t<double>(0) = 0.;
    weights.t<double>(1) = 0.;
    weights.t<double>(2) = 0.;
    weights.t<double>(3) = 0.;


    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sigm_cross_entropy_loss_grad_test13) {

    NDArray labels('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,1}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {0.     ,  0.     ,  0.     ,  0.     , 0.     ,  0.     ,  0.     ,  0.     , 0.     ,  0.     ,  0.     ,  0.     ,
                                   -0.36047, -0.38924, -0.41801, -0.44679,-0.47556, -0.50435, -0.53314, -0.56193,-0.59072, -0.61953, -0.64833, -0.67715});
    NDArray dLdwExp('c', {2,3,1}, {0.22882, 0.02428,-0.4768 ,-1.27447,-2.36878,-3.75981,});
    NDArray dLdlExp('c', {2,3,4}, {-0.     , -0.     ,  0.     ,  0.     , 0.     ,  0.     ,  0.     ,  0.     , 0.     ,  0.     ,  0.     ,  0.,
                                    -0.01167, -0.01283, -0.014  , -0.01517,-0.01633, -0.0175 , -0.01867, -0.01983,-0.021  , -0.02217, -0.02333, -0.0245});
    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);
    weights.t<double>(0) = 0.;
    weights.t<double>(1) = 0.;
    weights.t<double>(2) = 0.;

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));
    ASSERT_TRUE(dLdlExp.isSameShape(dLdl));
    ASSERT_TRUE(dLdlExp.equalsTo(dLdl));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, BFloat16_Test_4) {

    NDArray x = NDArrayFactory::create<float>('c', {2,3,4});
    NDArray y = NDArrayFactory::create<bfloat16>('c', {2,3,4});//('c', {2,3,4}, nd4j::DataType::BFLOAT16);
    NDArray exp = NDArrayFactory::create<float>('c', {2,3,4});//('c', {2,3,4}, nd4j::DataType::BFLOAT16);

    x.linspace(1);
    y.linspace(1);
    exp.linspace(2,2);
    nd4j::ops::add op;
    auto results = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto res = results->at(0);
    ASSERT_TRUE(res->equalsTo(exp));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, BFloat16_Test_5) {

    NDArray x = NDArrayFactory::create<float>('c', {2,3,4});
    NDArray y = NDArrayFactory::create<bfloat16>('c', {2,3,4});//('c', {2,3,4}, nd4j::DataType::BFLOAT16);
    NDArray exp = NDArrayFactory::create<float>('c', {2,3,4});//('c', {2,3,4}, nd4j::DataType::BFLOAT16);

    x.linspace(2, 2);
    y.linspace(1);
    exp.linspace(1);
    nd4j::ops::subtract op;
    auto results = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto res = results->at(0);
    ASSERT_TRUE(res->equalsTo(exp));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, BFloat16_Test_6) {

    NDArray x = NDArrayFactory::create<bfloat16>('c', {2,3,4});
    NDArray y = NDArrayFactory::create<double>('c', {2,3,4});//('c', {2,3,4}, nd4j::DataType::BFLOAT16);
    NDArray exp = NDArrayFactory::create<bfloat16>('c', {2,3,4});//('c', {2,3,4}, nd4j::DataType::BFLOAT16);

    x.linspace(2, 2);
    y.linspace(1);
    exp.linspace(1);
    nd4j::ops::subtract op;
    auto results = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto res = results->at(0);
    ASSERT_TRUE(res->equalsTo(exp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmax_cross_entropy_loss_grad_test1) {

    NDArray labels('c', {2,4}, {0,0,1,0, 0,1,0,0}, nd4j::DataType::INT32);
    NDArray logits('c', {2,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,4}, {0.1176,  0.1224, -0.3726,  0.1326, 0.1176, -0.3776,  0.1274,  0.1326});
    NDArray dLdwExp('c', {2}, {1.36729, 1.40729});

    logits.linspace(-0.08, 0.04);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.execute({&logits, &weights, &labels}, {0.}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmax_cross_entropy_loss_grad_test2) {

    NDArray labels('c', {4}, {0,0,1,0}, nd4j::DataType::INT32);
    NDArray logits('c', {4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {4}, {0.125,  0.125, -0.375,  0.125});
    NDArray dLdwExp('c', {1}, {1.38629});

    logits = 2.;
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.execute({&logits, &weights, &labels}, {0.}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmax_cross_entropy_loss_grad_test3) {

    NDArray labels('c', {4}, {0,0,1,0}, nd4j::DataType::INT32);
    NDArray logits('c', {4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {}, {0}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {4}, {0.125,  0.125, -0.375,  0.125});
    NDArray dLdwExp('c', {}, {1.38629});

    logits = 2.;
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.execute({&logits, &weights, &labels}, {0.}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmax_cross_entropy_loss_grad_test4) {

    NDArray labels('c', {4}, {0,0,1,0}, nd4j::DataType::INT32);
    NDArray logits('c', {4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {}, {0}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {4}, {0.23521,  0.2448 , -0.7452 ,  0.26519});
    NDArray dLdwExp('c', {}, {0.});

    logits.linspace(-0.08, 0.04);
    weights = 0.5;

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.execute({&logits, &weights, &labels}, {0.}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmax_cross_entropy_loss_grad_test5) {

    NDArray labels('c', {4}, {0,0,1,0}, nd4j::DataType::INT32);
    NDArray logits('c', {4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {4}, {0.1176,  0.1224, -0.3726,  0.1326});
    NDArray dLdwExp('c', {1}, {1.36729});

    logits.linspace(-0.08, 0.04);
    weights = 0.5;

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.execute({&logits, &weights, &labels}, {0.}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmax_cross_entropy_loss_grad_test6) {

    NDArray labels('c', {2,4}, {0,0,1,0, 0,1,0,0}, nd4j::DataType::INT32);
    NDArray logits('c', {2,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,4}, {0.0801,  0.0849, -0.2601,  0.0951, 0.0801, -0.2651,  0.0899,  0.0951});
    NDArray dLdwExp('c', {2}, {-0.014000, 0.014000});

    logits.linspace(-0.08, 0.04);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.execute({&logits, &weights, &labels}, {0.3}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmax_cross_entropy_loss_grad_test7) {

    NDArray labels('c', {2,3,4}, {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1,0,0,0, 0,1,0,0}, nd4j::DataType::INT32);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3}, {0.5, 0., 1.5});

    NDArray dLdpExp('c', {2,3,4}, {-0.0956 , 0.0306 , 0.03185, 0.03315, 0.,-0., 0., 0., 0.0882 , 0.0918 ,-0.27945, 0.09945,
                                   0.0294 , 0.0306 , 0.03185,-0.09185,-0., 0., 0., 0., 0.0882 ,-0.2832 , 0.09555, 0.09945});
    NDArray dLdwExp('c', {1,3}, {0.69365, 0.71365, 0.69365});

    logits.linspace(-0.08, 0.04);

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.execute({&logits, &weights, &labels}, {0.}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmax_cross_entropy_loss_grad_test8) {

    NDArray labels('c', {2,3,4,5}, {1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,
                                    0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,
                                    0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0}, nd4j::DataType::INT32);

    NDArray logits('c', {2,3,4,5}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,1,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4,5}, {-0.03399, 0.00799, 0.00832, 0.00866, 0.00901, 0.00768,-0.03367, 0.00832, 0.00866, 0.00901, 0.00768, 0.00799,-0.03335,
                                    0.00866, 0.00901, 0.00768, 0.00799, 0.00832,-0.03301, 0.00901, 0.00768, 0.00799, 0.00832, 0.00866,-0.03265,-0.03399,
                                    0.00799, 0.00832, 0.00866, 0.00901, 0.00768,-0.03367, 0.00832, 0.00866, 0.00901, 0.00768, 0.00799,-0.03335, 0.00866,
                                    0.00901, 0.00768, 0.00799, 0.00832,-0.03301, 0.00901, 0.00768, 0.00799, 0.00832, 0.00866,-0.03265,-0.03399, 0.00799,
                                    0.00832, 0.00866, 0.00901, 0.00768,-0.03367, 0.00832, 0.00866, 0.00901, 0.00768, 0.00799,-0.03335, 0.00866, 0.00901,
                                    0.00768, 0.00799, 0.00832,-0.03301, 0.00901, 0.00768, 0.00799, 0.00832, 0.00866,-0.03265,-0.03399, 0.00799, 0.00832,
                                    0.00866, 0.00901, 0.00768,-0.03367, 0.00832, 0.00866, 0.00901, 0.00768, 0.00799,-0.03335, 0.00866, 0.00901, 0.00768,
                                    0.00799, 0.00832,-0.03301, 0.00901, 0.00768, 0.00799, 0.00832, 0.00866,-0.03265,-0.03399, 0.00799, 0.00832, 0.00866,
                                    0.00901, 0.00768,-0.03367, 0.00832, 0.00866, 0.00901, 0.00768, 0.00799,-0.03335, 0.00866, 0.00901, 0.00768, 0.00799, 0.00832,-0.03301, 0.00901});

    NDArray dLdwExp('c', {1,1,4}, {0.005,  0.00167, -0.00167, -0.005});
    logits.linspace(-0.08, 0.04);
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.execute({&logits, &weights, &labels}, {0.}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);
    auto *dLdw = results->at(1);
    auto *dLdl = results->at(2);

    // dLdp->printIndexedBuffer();

    // ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    // ASSERT_TRUE(dLdpExp.equalsTo(dLdp));
    ASSERT_TRUE(dLdwExp.isSameShape(dLdw));
    ASSERT_TRUE(dLdwExp.equalsTo(dLdw));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, SafeDivideMixed_Test1) {

    NDArray labels('c', {2, 3}, {1.0, 2.0, 3.0, -1.0, 2.0, 1.0});
    auto sumDiff = labels.reduceAlongDims(reduce::Sum, {1}, true);

    NDArray numOfNonZero(sumDiff.getShapeInfo(), nd4j::DataType::INT64, false);
    numOfNonZero.assign(1);
    sumDiff.applyPairwiseTransform(pairwise::SafeDivide, &numOfNonZero, &sumDiff, nullptr);
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test1) {

    NDArray labels('c', {2,3,4}, {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1,0,0,0, 0,1,0,0});
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.76479, 0.2448, 0.2548, 0.26519, 0.23521,-0.7552, 0.2548, 0.26519, 0.23521, 0.2448,-0.7452, 0.26519,
                                   0.23521, 0.2448, 0.2548,-0.73481,-0.76479, 0.2448, 0.2548, 0.26519, 0.23521,-0.7552, 0.2548, 0.26519});
    logits.linspace(-0.08, 0.04);

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&logits, &labels}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test2) {

    NDArray labels('c', {2,3,4}, {1,0,0,0, 0,1,0,1, 0,0,1,0, 0,0,0,1, 1,0,1,0, 0,1,0,0});
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.71836,  0.28164,  0.28164,  0.28164, 0.33051, -0.66949,  0.33051, -0.66949, 0.38785,  0.38785, -0.61215,  0.38785,
                                    0.28164,  0.28164,  0.28164, -0.71836,-0.66949,  0.33051, -0.66949,  0.33051, 0.38785, -0.61215,  0.38785,  0.38785});
    logits.linspace(-0.08, 0.04);

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&logits, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test3) {

    NDArray labels('c', {2,3}, {1,0,0, 0,1,1});
    NDArray logits('c', {2,3}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3}, {-0.52996,  0.47004,  0.47004, 0.52996, -0.47004, -0.47004});
    logits.linspace(-0.08, 0.04);

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&logits, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test4) {

    NDArray labels('c', {2,1}, {1,1});
    NDArray logits('c', {2,1}, {-0.04, 0.04});

    NDArray dLdpExp('c', {2,1}, {0., 0.});

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&logits, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test5) {

    NDArray labels('c', {2,1}, {1,0});
    NDArray logits('c', {2,1}, {-0.04, 0.04});

    NDArray dLdpExp('c', {2,1}, {-0.51999, 0.51999});

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&logits, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test6) {

    NDArray labels('c', {1,2}, {1,1});
    NDArray logits('c', {1,2}, {-0.04, 0.04});

    NDArray dLdpExp('c', {1,2}, {0, 0});

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&logits, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test7) {

    NDArray labels('c', {2}, {0,1});
    NDArray logits('c', {2}, {-0.04, 0.04});

    NDArray dLdpExp('c', {2}, {0.48001, -0.48001});

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&logits, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test8) {

    NDArray labels('c', {1}, {1});
    NDArray logits('c', {1}, {0.04});

    NDArray dLdpExp('c', {1}, {0});

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&logits, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, Multiply_BP_Test1) {

    NDArray x('c', {3,4,5}, nd4j::DataType::DOUBLE);
    NDArray y('c', {1,1,1}, nd4j::DataType::DOUBLE);

    NDArray dLdp('c', {3,4,5}, nd4j::DataType::DOUBLE);
    NDArray dLdpExp('c', {3,4,5}, nd4j::DataType::DOUBLE);

    x.assign(1.0);//linspace(0.1, 0.1);
    y.assign(1.0);
    dLdp.assign(1.0);
    dLdpExp.assign(1.0);
    nd4j::ops::multiply_bp op;

    auto results = op.execute({&x, &y, &dLdp}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdo = results->at(0);
    ASSERT_TRUE(dLdpExp.isSameShape(dLdo));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdo));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sparseSoftmaxCrossEntropyWithLogits_grad_test1) {

    NDArray labels('c', {2}, {2,1}, nd4j::DataType::INT64);
    NDArray logits('c', {2,3}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3}, {0.30061,  0.33222, -0.63283, 0.30061, -0.66778,  0.36717});

    logits.linspace(0.1, 0.1);

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sparseSoftmaxCrossEntropyWithLogits_grad_test2) {

    NDArray labels('c', {2}, {0,1}, nd4j::DataType::INT64);
    NDArray logits('c', {2,3}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3}, {-0.69939,  0.33222,  0.36717, 0.30061, -0.66778,  0.36717});

    logits.linspace(-0.1, 0.1);

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sparseSoftmaxCrossEntropyWithLogits_grad_test3) {

    NDArray labels('c', {}, {1}, nd4j::DataType::INT64);
    NDArray logits('c', {2}, {-0.2, 0.3});

    NDArray dLdpExp('c', {2}, {0.37754, -0.37754});

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sparseSoftmaxCrossEntropyWithLogits_grad_test4) {

    NDArray labels('c', {2,3}, {0,1,1, 3,3,2}, nd4j::DataType::INT64);
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.78616,  0.23633,  0.26118,  0.28865, 0.21384, -0.76367,  0.26118,  0.28865, 0.21384, -0.76367,  0.26118,  0.28865,
                                  0.21384,  0.23633,  0.26118, -0.71135, 0.21384,  0.23633,  0.26118, -0.71135, 0.21384,  0.23633, -0.73882,  0.28865});
    logits.linspace(-0.5, 0.1);

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sparseSoftmaxCrossEntropyWithLogits_grad_test5) {

    NDArray labels('c', {1,1}, {0}, nd4j::DataType::INT64);
    NDArray logits('c', {1,1,2}, {-0.3,0.2});

    NDArray dLdpExp('c', {1,1,2}, {-0.62246,  0.62246});

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.execute({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}



