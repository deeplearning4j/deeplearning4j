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
    auto result = op.evaluate({&x, &y}, {}, {});
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
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {0}, {});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {0});

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
    NDArray dLdwExp('c', {},  std::vector<double>{-227.77286});
    NDArray dLdlExp('c', {2,3,4}, {1.58903, 1.22117, 0.99621, 0.82911, 0.69315, 0.57634, 0.47223, 0.37689, 0.28768, 0.20273, 0.12058, 0.04002,
                                   -0.04002,-0.12058,-0.20273,-0.28768,-0.37689,-0.47223,-0.57634,-0.69315,-0.82911,-0.99621,-1.22117,-1.58903});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {1});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {1});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {2});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {2});

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

    NDArray dLdwExp('c', {},  std::vector<double>{0.});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {2});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {2});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {3});

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

    NDArray dLdwExp('c', {1,1}, std::vector<double>{-9.49054});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::log_loss_grad op;
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {3});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {3});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {3});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {1e-7}, {3});

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

TEST_F(DeclarableOpsTests11, ImageResizeBicubic_Test1) {

    NDArray input    = NDArrayFactory::create<float>('c', {1, 7, 7, 1}, {
             1.f, 2.1f, 3.15f,  4.2f, 5.15f,  6.1f,   7.f,
             8.f, 9.1f,  10.f,  11.f, 12.9f, 13.1f,  14.f,
            15.f, 16.f,  17.f,  18.f,  19.f,  20.f,  21.f,
            22.f, 23.f,  24.f,  25.f,  26.f,   27.f,  28.f,
            30.f, 31.f,  32.f,  33.f,  34.f,   35.f,  36.f,
            37.f, 38.f,  39.f,  40.f,  41.f,   42.f,  43.f,
            44.f, 45.f,  46.f,  47.f,  48.f,   49.f,  50.f
    });
    NDArray expected = NDArrayFactory::create<float>('c', {1, 30, 30, 1}, {
               1.f,  1.1976162f,  1.4174359f,  1.6775769f,   1.9961575f,  2.3283265f,
         2.550918f,  2.7360606f,  2.9655411f,  3.2929654f,   3.5441515f,  3.7380352f,
         3.948995f,   4.248106f,  4.5073795f,  4.6843743f,   4.8572845f,   5.104302f,
        5.3869915f,   5.581401f,  5.7539616f,   5.974285f,    6.272836f,  6.5204263f,
         6.718899f,  6.8871036f,   7.039068f,   7.099216f,   7.0784245f,  7.0281887f,
         2.247592f,   2.446947f,  2.6694887f,  2.9312382f,    3.248216f,  3.5745337f,
          3.78931f,  3.9656973f,   4.186417f,  4.5046535f,    4.740569f,  4.9217057f,
         5.133866f,   5.459533f,  5.7744613f,  6.0197873f,    6.254011f,   6.535633f,
        6.8097296f,  6.9607787f,  7.0749416f,   7.241601f,   7.5094895f,  7.7499495f,
         7.954571f,   8.131972f,   8.286526f,   8.346463f,    8.325745f,   8.275683f,
        3.6286845f,   3.830573f,  4.0569587f,  4.3211575f,   4.6364856f,  4.9556503f,
         5.160583f,  5.3258467f,   5.535462f,    5.84216f,    6.058749f,   6.223753f,
         6.437597f,   6.797369f,  7.1836042f,  7.5164022f,   7.8290343f,   8.154773f,
         8.417635f,   8.512958f,     8.5521f,   8.649708f,     8.87788f,   9.108794f,
         9.320926f,   9.509781f,   9.667375f,    9.72694f,    9.706349f,   9.656599f,
         5.276778f,   5.480438f,   5.709702f,  5.9754477f,    6.288551f,  6.6005697f,
         6.796207f,  6.9511423f,  7.1503997f,  7.4461427f,    7.644651f,   7.794562f,
         8.009684f,   8.400473f,   8.851847f,    9.26469f,    9.649218f,  10.015648f,
        10.268647f,  10.313368f, 10.2843275f,  10.319379f,   10.512033f,  10.734956f,
        10.954604f,  11.154507f,  11.315369f,  11.374779f,   11.354242f,  11.304622f,
         7.325373f,  7.5284843f,   7.757575f,   8.022221f,    8.331997f,   8.638187f,
         8.827649f,   8.976217f,   9.168955f,    9.45726f,   9.6442375f,   9.784517f,
         9.999621f,  10.407702f,  10.896234f,  11.355122f,   11.781423f,  12.172186f,
        12.420712f, 12.4374485f,  12.370511f,  12.371386f,   12.545973f,  12.766424f,
        12.992249f,   13.20012f,  13.364252f,  13.424109f,    13.40342f,  13.353425f,
         9.493208f,   9.692467f,  9.9169445f,  10.176801f,   10.482199f,   10.78547f,
        10.974367f,  11.123442f,   11.31637f,  11.603645f,   11.790616f,  11.930889f,
        12.144082f,  12.546447f,  13.024898f,    13.4723f,   13.889232f,  14.276275f,
        14.528972f,  14.555555f,   14.50145f,  14.515459f,   14.700572f,  14.927055f,
        15.156046f,  15.366046f,  15.532901f,  15.594008f,  15.5728855f,  15.521847f,
        10.970133f,  11.163599f,  11.380694f,  11.633735f,   11.935032f,  12.238887f,
         12.43254f,  12.588294f,  12.787534f,  13.079956f,    13.27752f,  13.426631f,
        13.636713f,  14.013844f,  14.441672f,  14.827978f,   15.191209f,  15.549808f,
         15.81343f,  15.881828f,  15.883522f,  15.950411f,    16.16933f,   16.40794f,
        16.636436f,  16.842583f,  17.010887f,   17.07363f,    17.05194f,  16.999537f,
        12.219155f,  12.406129f,  12.614796f,  12.860335f,  13.157928f,   13.464224f,
        13.665207f,  13.830567f,  14.039036f,  14.339629f,  14.552863f,   14.715049f,
        14.921564f,  15.264454f,  15.622843f,  15.924977f,  16.213829f,   16.532364f,
          16.8099f,  16.934835f,  17.012146f,  17.150164f,  17.413412f,   17.666712f,
        17.892765f,   18.09207f,  18.261044f,  18.325531f,  18.303238f,   18.249378f,
       13.7663965f,  13.947391f,  14.148263f,  14.386917f,  14.681246f,   14.990087f,
        15.198166f,  15.372728f,  15.590062f,  15.898583f,  16.126892f,   16.301655f,
         16.50487f,  16.815214f,  17.107498f,  17.329458f,  17.547403f,   17.827654f,
        18.118288f,  18.296928f,    18.4461f,  18.651634f,  18.956806f,    19.22382f,
        19.447308f,  19.639887f,  19.809319f,  19.875397f,  19.852556f,   19.797365f,
       15.9419365f,  16.118704f,  16.314133f,  16.547867f,  16.839561f,    17.14954f,
        17.361883f,  17.542162f,  17.764957f,  18.078188f,  18.315733f,   18.498205f,
        18.699116f,  18.988684f,  19.238989f,  19.410137f,  19.583265f,   19.839512f,
         20.13878f,   20.35177f,  20.546844f,  20.795671f,  21.128067f,   21.404358f,
        21.626736f,    21.8155f,   21.98561f,  22.052843f,  22.029604f,   21.973448f,
         17.53522f,   17.71077f,  17.904636f,   18.13695f,   18.42784f,   18.738056f,
        18.951529f,  19.133352f,  19.357613f,  19.672083f,  19.912102f,   20.096638f,
        20.296894f,  20.580765f,  20.819603f,  20.976887f,  21.137802f,   21.387535f,
        21.689209f,  21.911621f,  22.119276f,   22.37999f,   22.71991f,   22.998823f,
         23.22097f,   23.40876f,   23.57911f,  23.646685f,  23.623325f,   23.566887f,
        18.746353f,  18.922657f,  19.117487f,  19.350685f,   19.64207f,   19.952137f,
        20.164913f,  20.345781f,  20.569134f,   20.88284f,   21.12133f,    21.30459f,
        21.505253f,  21.792645f,  22.038572f,  22.204426f,   22.37289f,   22.626648f,
        22.926834f,  23.143423f,  23.343302f,  23.596668f,  23.931936f,   24.209232f,
        24.431519f,  24.619913f,   24.79011f,  24.857473f,   24.83419f,   24.777927f,
         20.16656f,  20.344206f,  20.540766f,  20.775532f,  21.067804f,   21.377607f,
        21.589132f,  21.768297f,   21.99003f,  22.302366f,  22.538124f,   22.719105f,
        22.920494f,  23.214176f,  23.472767f,  23.653934f,   23.83589f,   24.096842f,
        24.394371f,  24.600555f,  24.786541f,  25.026773f,  25.353731f,    25.62813f,
        25.850672f,   26.04014f,  26.210072f,  26.277063f,  26.253906f,   26.197956f,
        22.363024f,   22.54125f,  22.738552f,  22.973991f,  23.266647f,    23.57634f,
        23.787327f,   23.96576f,  24.186796f,  24.498543f,  24.733124f,   24.913122f,
        25.114826f,  25.411213f,  25.675262f,  25.863028f,  26.050789f,   26.314838f,
        26.611223f,  26.812925f,  26.992926f,  27.227505f,  27.550882f,   27.824034f,
        28.046684f,  28.236614f,  28.406433f,  28.473265f,  28.450163f,   28.394344f,
        24.429443f,   24.60767f,   24.80497f,   25.04041f,  25.333065f,   25.642756f,
        25.853743f,  26.032173f,   26.25321f,  26.564959f,   26.79954f,    26.97954f,
        27.181242f,   27.47763f,   27.74168f,  27.929441f,  28.117207f,   28.381254f,
        28.677637f,  28.879343f,  29.059345f,  29.293922f,  29.617298f,   29.890451f,
        30.113104f,  30.303034f,  30.472853f,  30.539684f,  30.516582f,   30.460762f,
              26.f,  26.178228f,  26.375526f,   26.61097f,  26.903624f,   27.213314f,
        27.424305f,  27.602734f,  27.823772f,  28.135519f,    28.3701f,   28.550098f,
          28.7518f,   29.04819f,  29.312237f,       29.5f,  29.687763f,   29.951813f,
          30.2482f,  30.449903f,  30.629902f,  30.864483f,  31.187859f,   31.461012f,
        31.683659f,  31.873592f,  32.043407f,   32.11024f,  32.087135f,    32.03132f,
        27.570559f,  27.748787f,  27.946087f,  28.181528f,  28.474184f,   28.783876f,
        28.994865f,  29.173294f,   29.39433f,   29.70608f,  29.940659f,   30.120655f,
         30.32236f,  30.618746f,  30.882797f,  31.070557f,   31.25832f,   31.522371f,
        31.818754f,   32.02046f,   32.20046f,   32.43504f,  32.758415f,   33.031567f,
         33.25422f,   33.44415f,  33.613964f,  33.680794f,  33.657696f,    33.60188f,
        29.636976f,  29.815207f,    30.0125f,  30.247944f,    30.5406f,    30.85029f,
        31.061283f,  31.239712f,   31.46075f,    31.7725f,   32.00708f,   32.187077f,
         32.38878f,  32.685165f,  32.949215f,   33.13698f,   33.32474f,    33.58879f,
        33.885178f,  34.086884f,   34.26688f,  34.501457f,  34.824837f,    35.09799f,
        35.320637f,  35.510574f,   35.68039f,  35.747215f,  35.724117f,     35.6683f,
         31.83344f,  32.011665f,   32.20897f,  32.444412f,   32.73707f,   33.046757f,
        33.257744f,  33.436176f,  33.657207f,   33.96896f,  34.203537f,   34.383537f,
         34.58524f,   34.88163f,  35.145676f,   35.33344f,  35.521206f,   35.785255f,
        36.081642f,   36.28334f,   36.46334f,   36.69792f,  37.021297f,   37.294453f,
        37.517097f,  37.707027f,  37.876846f,   37.94368f,  37.920578f,   37.864758f,
        33.253647f,  33.431873f,   33.62917f,  33.864613f,   34.15727f,   34.466957f,
        34.677948f,  34.856377f,  35.077415f,   35.38916f,  35.623745f,   35.803745f,
        36.005447f,  36.301834f,  36.565884f,  36.753647f,  36.941406f,   37.205456f,
         37.50184f,  37.703545f,  37.883545f,  38.118122f,    38.4415f,   38.714653f,
          38.9373f,  39.127235f,  39.297054f,  39.363884f,  39.340782f,    39.28496f,
        34.464783f,   34.64301f,  34.840305f,  35.075752f,  35.368404f,     35.6781f,
        35.889088f,  36.067516f,   36.28855f,    36.6003f,  36.834885f,   37.014877f,
        37.216583f,   37.51297f,   37.77702f,  37.964783f,  38.152546f,   38.416595f,
         38.71298f,  38.914684f,  39.094685f,   39.32926f,  39.652645f,   39.925793f,
         40.14844f,  40.338375f,  40.508194f,  40.575024f,   40.55192f,   40.496105f,
        36.058067f,   36.23629f,   36.43359f,  36.669033f,  36.961685f,   37.271378f,
         37.48237f,    37.6608f,  37.881836f,   38.19359f,   38.42817f,   38.608162f,
        38.809868f,   39.10625f,    39.3703f,  39.558064f,   39.74583f,    40.00988f,
        40.306267f,   40.50797f,   40.68797f,   40.92255f,  41.245926f,   41.519077f,
        41.741722f,  41.931652f,  42.101475f,  42.168304f,  42.145203f,   42.089386f,
        38.315002f,  38.493233f,  38.690533f,  38.925976f,  39.218628f,    39.52832f,
        39.739307f,  39.917736f,  40.138775f,   40.45052f,  40.685104f,   40.865097f,
        41.066803f,   41.36319f,  41.627243f,  41.815002f,  42.002766f,    42.26682f,
          42.5632f,  42.764908f,  42.944904f,  43.179485f,   43.50286f,   43.776016f,
        43.998665f,  44.188595f,  44.358418f,  44.425247f,  44.402145f,    44.34633f,
         40.22708f,   40.40531f,  40.602608f,   40.83805f,  41.130707f,   41.440395f,
        41.651382f,   41.82982f,  42.050854f,    42.3626f,  42.597183f,    42.77718f,
         42.97888f,   43.27527f,   43.53932f,   43.72708f,  43.914845f,   44.178894f,
         44.47528f,  44.676983f,  44.856983f,   45.09156f,   45.41494f,    45.68809f,
         45.91074f,  46.100674f,  46.270493f,  46.337322f,   46.31422f,     46.2584f,
        41.785618f,  41.963844f,  42.161144f,  42.396584f,   42.68924f,   42.998936f,
        43.209923f,  43.388355f,  43.609394f,  43.921143f,   44.15572f,   44.335716f,
         44.53742f,  44.833805f,   45.09786f,  45.285614f,  45.473377f,   45.737427f,
        46.033817f,  46.235523f,  46.415524f,  46.650105f,  46.973476f,    47.24663f,
        47.469276f,   47.65921f,   47.82903f,  47.895855f,  47.872753f,    47.81694f,
         43.11514f,  43.293365f,  43.490665f,  43.726105f,  44.018764f,   44.328457f,
        44.539444f,  44.717873f,   44.93891f,   45.25066f,   45.48524f,   45.665237f,
         45.86694f,  46.163326f,  46.427376f,  46.615143f,  46.802902f,   47.066956f,
        47.363342f,   47.56505f,   47.74505f,  47.979626f,  48.302998f,   48.576153f,
        48.798798f,   48.98873f,  49.158546f,  49.225376f,  49.202282f,   49.146458f,
        44.303867f,  44.482094f,  44.679394f,  44.914833f,  45.207493f,    45.51718f,
         45.72817f,    45.9066f,   46.12764f,  46.439384f,  46.673965f,   46.853966f,
        47.055668f,  47.352055f,    47.6161f,  47.803867f,   47.99163f,    48.25568f,
        48.552063f,   48.75377f,  48.933773f,   49.16835f,  49.491726f,   49.764877f,
        49.987526f,   50.17746f,  50.347275f,    50.4141f,  50.391006f,   50.335186f,
        44.771675f,  44.949905f,    45.1472f,  45.382645f,    45.6753f,    45.98499f,
        46.195976f,  46.374413f,  46.595448f,  46.907196f,  47.141773f,   47.321774f,
        47.523476f,  47.819862f,   48.08391f,   48.27168f,  48.459446f,    48.72349f,
        49.019882f,   49.22158f,  49.401585f,   49.63616f,  49.959538f,   50.232693f,
        50.455338f,   50.64527f,   50.81509f,   50.88192f,  50.858818f,      50.803f,
        44.609966f,  44.788193f,  44.985493f,  45.220936f,   45.51359f,    45.82328f,
         46.03427f,    46.2127f,  46.433743f,   46.74549f,   46.98007f,   47.160065f,
         47.36177f,  47.658157f,  47.922207f,   48.10997f,  48.297733f,   48.561783f,
        48.858166f,  49.059875f,  49.239872f,   49.47445f,   49.79783f,    50.07098f,
        50.293625f,   50.48356f,  50.653378f,  50.720203f,    50.6971f,    50.64128f,
        44.219246f,  44.397472f,  44.594772f,   44.83021f,  45.122868f,    45.43256f,
        45.643543f,   45.82198f,   46.04302f,  46.354763f,  46.589344f,    46.76934f,
        46.971046f,  47.267433f,  47.531483f,  47.719242f,  47.907005f,    48.17105f,
        48.467438f,   48.66914f,  48.849144f,   49.08372f,    49.4071f,   49.680256f,
        49.902905f,  50.092834f,  50.262653f,  50.329483f,   50.30638f,    50.25057f});

    auto size = NDArrayFactory::create<int>({30, 30});
    nd4j::ops::resize_bicubic op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());
    NDArray* result = results->at(0);

//    result->printBuffer("Resized to 30x30");
//    expected.printBuffer("Expect for 30x30");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}
TEST_F(DeclarableOpsTests11, ImageResizeBicubic_Test2) {

    NDArray input    = NDArrayFactory::create<double>('c', {2, 5, 4, 3});
    NDArray expected = NDArrayFactory::create<float>('c', {2, 10, 8, 3}, {
            1.000000f, 2.000000f, 3.000000f, 2.218750f, 3.218750f, 4.218750f, 4.000000f, 5.000000f, 6.000000f,
            5.500000f, 6.500000f, 7.500000f, 7.000000f, 8.000000f, 9.000000f, 8.781250f, 9.781250f, 10.781250f,
            10.000000f, 11.000000f, 12.000000f, 10.281250f, 11.281250f, 12.281250f, 5.875000f, 6.875000f, 7.875000f,
            7.093750f, 8.093750f, 9.093750f, 8.875000f, 9.875000f, 10.875000f, 10.375000f, 11.375000f, 12.375000f,
            11.875000f, 12.875000f, 13.875000f, 13.656250f, 14.656250f, 15.656250f, 14.875000f, 15.875000f, 16.875000f,
            15.156250f, 16.156250f, 17.156250f, 13.000000f, 14.000000f, 15.000000f, 14.218750f, 15.218750f, 16.218750f,
            16.000000f, 17.000000f, 18.000000f, 17.500000f, 18.500000f, 19.500000f, 19.000000f, 20.000000f, 21.000000f,
            20.781250f, 21.781250f, 22.781250f, 22.000000f, 23.000000f, 24.000000f, 22.281250f, 23.281250f, 24.281250f,
            19.000000f, 20.000000f, 21.000000f, 20.218750f, 21.218750f, 22.218750f, 22.000000f, 23.000000f, 24.000000f,
            23.500000f, 24.500000f, 25.500000f, 25.000000f, 26.000000f, 27.000000f, 26.781250f, 27.781250f, 28.781250f,
            28.000000f, 29.000000f, 30.000000f, 28.281250f, 29.281250f, 30.281250f, 25.000000f, 26.000000f, 27.000000f,
            26.218750f, 27.218750f, 28.218750f, 28.000000f, 29.000000f, 30.000000f, 29.500000f, 30.500000f, 31.500000f,
            31.000000f, 32.000000f, 33.000000f, 32.781250f, 33.781250f, 34.781250f, 34.000000f, 35.000000f, 36.000000f,
            34.281250f, 35.281250f, 36.281250f, 31.000000f, 32.000000f, 33.000000f, 32.218750f, 33.218750f, 34.218750f,
            34.000000f, 35.000000f, 36.000000f, 35.500000f, 36.500000f, 37.500000f, 37.000000f, 38.000000f, 39.000000f,
            38.781250f, 39.781250f, 40.781250f, 40.000000f, 41.000000f, 42.000000f, 40.281250f, 41.281250f, 42.281250f,
            37.000000f, 38.000000f, 39.000000f, 38.218750f, 39.218750f, 40.218750f, 40.000000f, 41.000000f, 42.000000f,
            41.500000f, 42.500000f, 43.500000f, 43.000000f, 44.000000f, 45.000000f, 44.781250f, 45.781250f, 46.781250f,
            46.000000f, 47.000000f, 48.000000f, 46.281250f, 47.281250f, 48.281250f, 44.125000f, 45.125000f, 46.125000f,
            45.343750f, 46.343750f, 47.343750f, 47.125000f, 48.125000f, 49.125000f, 48.625000f, 49.625000f, 50.625000f,
            50.125000f, 51.125000f, 52.125000f, 51.906250f, 52.906250f, 53.906250f, 53.125000f, 54.125000f, 55.125000f,
            53.406250f, 54.406250f, 55.406250f, 49.000000f, 50.000000f, 51.000000f, 50.218750f, 51.218750f, 52.218750f,
            52.000000f, 53.000000f, 54.000000f, 53.500000f, 54.500000f, 55.500000f, 55.000000f, 56.000000f, 57.000000f,
            56.781250f, 57.781250f, 58.781250f, 58.000000f, 59.000000f, 60.000000f, 58.281250f, 59.281250f, 60.281250f,
            50.125000f, 51.125000f, 52.125000f, 51.343750f, 52.343750f, 53.343750f, 53.125000f, 54.125000f, 55.125000f,
            54.625000f, 55.625000f, 56.625000f, 56.125000f, 57.125000f, 58.125000f, 57.906250f, 58.906250f, 59.906250f,
            59.125000f, 60.125000f, 61.125000f, 59.406250f, 60.406250f, 61.406250f, 61.000000f, 62.000000f, 63.000000f,
            62.218750f, 63.218750f, 64.218750f, 64.000000f, 65.000000f, 66.000000f, 65.500000f, 66.500000f, 67.500000f,
            67.000000f, 68.000000f, 69.000000f, 68.781250f, 69.781250f, 70.781250f, 70.000000f, 71.000000f, 72.000000f,
            70.281250f, 71.281250f, 72.281250f, 65.875000f, 66.875000f, 67.875000f, 67.093750f, 68.093750f, 69.093750f,
            68.875000f, 69.875000f, 70.875000f, 70.375000f, 71.375000f, 72.375000f, 71.875000f, 72.875000f, 73.875000f,
            73.656250f, 74.656250f, 75.656250f, 74.875000f, 75.875000f, 76.875000f, 75.156250f, 76.156250f, 77.156250f,
            73.000000f, 74.000000f, 75.000000f, 74.218750f, 75.218750f, 76.218750f, 76.000000f, 77.000000f, 78.000000f,
            77.500000f, 78.500000f, 79.500000f, 79.000000f, 80.000000f, 81.000000f, 80.781250f, 81.781250f, 82.781250f,
            82.000000f, 83.000000f, 84.000000f, 82.281250f, 83.281250f, 84.281250f, 79.000000f, 80.000000f, 81.000000f,
            80.218750f, 81.218750f, 82.218750f, 82.000000f, 83.000000f, 84.000000f, 83.500000f, 84.500000f, 85.500000f,
            85.000000f, 86.000000f, 87.000000f, 86.781250f, 87.781250f, 88.781250f, 88.000000f, 89.000000f, 90.000000f,
            88.281250f, 89.281250f, 90.281250f, 85.000000f, 86.000000f, 87.000000f, 86.218750f, 87.218750f, 88.218750f,
            88.000000f, 89.000000f, 90.000000f, 89.500000f, 90.500000f, 91.500000f, 91.000000f, 92.000000f, 93.000000f,
            92.781250f, 93.781250f, 94.781250f, 94.000000f, 95.000000f, 96.000000f, 94.281250f, 95.281250f, 96.281250f,
            91.000000f, 92.000000f, 93.000000f, 92.218750f, 93.218750f, 94.218750f, 94.000000f, 95.000000f, 96.000000f,
            95.500000f, 96.500000f, 97.500000f, 97.000000f, 98.000000f, 99.000000f, 98.781250f, 99.781250f, 100.781250f,
            100.000000f, 101.000000f, 102.000000f, 100.281250f, 101.281250f, 102.281250f, 97.000000f, 98.000000f,
            99.000000f, 98.218750f, 99.218750f, 100.218750f, 100.000000f, 101.000000f, 102.000000f, 101.500000f,
            102.500000f, 103.500000f, 103.000000f, 104.000000f, 105.000000f, 104.781250f, 105.781250f, 106.781250f,
            106.000000f, 107.000000f, 108.000000f, 106.281250f, 107.281250f, 108.281250f, 104.125000f, 105.125000f,
            106.125000f, 105.343750f, 106.343750f, 107.343750f, 107.125000f, 108.125000f, 109.125000f, 108.625000f,
            109.625000f, 110.625000f, 110.125000f, 111.125000f, 112.125000f, 111.906250f, 112.906250f, 113.906250f,
            113.125000f, 114.125000f, 115.125000f, 113.406250f, 114.406250f, 115.406250f, 109.000000f, 110.000000f,
            111.000000f, 110.218750f, 111.218750f, 112.218750f, 112.000000f, 113.000000f, 114.000000f, 113.500000f,
            114.500000f, 115.500000f, 115.000000f, 116.000000f, 117.000000f, 116.781250f, 117.781250f, 118.781250f,
            118.000000f, 119.000000f, 120.000000f, 118.281250f, 119.281250f, 120.281250f, 110.125000f, 111.125000f,
            112.125000f, 111.343750f, 112.343750f, 113.343750f, 113.125000f, 114.125000f, 115.125000f, 114.625000f,
            115.625000f, 116.625000f, 116.125000f, 117.125000f, 118.125000f, 117.906250f, 118.906250f, 119.906250f,
            119.125000f, 120.125000f, 121.125000f, 119.406250f, 120.406250f, 121.406250f
    });    //input = 1.f;
    input.linspace(1);
    auto size = NDArrayFactory::create<int>({10, 8});
    nd4j::ops::resize_bicubic op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Resized to 10x8");
//    expected.printBuffer("Expect for 10x8");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeBicubic_Test3) {

    NDArray input    = NDArrayFactory::create<double>('c', {1, 3, 3, 4});
    NDArray expected = NDArrayFactory::create<float>('c', {1, 6, 6, 4}, {
            1.000000f, 2.000000f, 3.000000f, 4.000000f, 2.625000f, 3.625000f, 4.625000f, 5.625000f, 5.000000f,
            6.000000f, 7.000000f, 8.000000f, 7.375000f, 8.375000f, 9.375000f, 10.375000f, 9.000000f, 10.000000f,
            11.000000f, 12.000000f, 9.375000f, 10.375000f, 11.375000f, 12.375000f, 5.875000f, 6.875000f, 7.875000f,
            8.875000f, 7.500000f, 8.500000f, 9.500000f, 10.500000f, 9.875000f, 10.875000f, 11.875000f, 12.875000f,
            12.250000f, 13.250000f, 14.250000f, 15.250000f, 13.875000f, 14.875000f, 15.875000f, 16.875000f, 14.250000f,
            15.250000f, 16.250000f, 17.250000f, 13.000000f, 14.000000f, 15.000000f, 16.000000f, 14.625000f, 15.625000f,
            16.625000f, 17.625000f, 17.000000f, 18.000000f, 19.000000f, 20.000000f, 19.375000f, 20.375000f, 21.375000f,
            22.375000f, 21.000000f, 22.000000f, 23.000000f, 24.000000f, 21.375000f, 22.375000f, 23.375000f, 24.375000f,
            20.125000f, 21.125000f, 22.125000f, 23.125000f, 21.750000f, 22.750000f, 23.750000f, 24.750000f, 24.125000f,
            25.125000f, 26.125000f, 27.125000f, 26.500000f, 27.500000f, 28.500000f, 29.500000f, 28.125000f, 29.125000f,
            30.125000f, 31.125000f, 28.500000f, 29.500000f, 30.500000f, 31.500000f, 25.000000f, 26.000000f, 27.000000f,
            28.000000f, 26.625000f, 27.625000f, 28.625000f, 29.625000f, 29.000000f, 30.000000f, 31.000000f, 32.000000f,
            31.375000f, 32.375000f, 33.375000f, 34.375000f, 33.000000f, 34.000000f, 35.000000f, 36.000000f, 33.375000f,
            34.375000f, 35.375000f, 36.375000f, 26.125000f, 27.125000f, 28.125000f, 29.125000f, 27.750000f, 28.750000f,
            29.750000f, 30.750000f, 30.125000f, 31.125000f, 32.125000f, 33.125000f, 32.500000f, 33.500000f, 34.500000f,
            35.500000f, 34.125000f, 35.125000f, 36.125000f, 37.125000f, 34.500000f, 35.500000f, 36.500000f, 37.500000f
    });
    input.linspace(1);
    auto size = NDArrayFactory::create<int>({6, 6});
    nd4j::ops::resize_bicubic op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Resized to 6x6");
//    expected.printBuffer("Expect for 6x6");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeBicubic_Test4) {

    NDArray input    = NDArrayFactory::create<double>('c', {1, 3, 4, 3});
    NDArray expected = NDArrayFactory::create<float>('c', {1, 6, 8, 3}, {
            1.000000f, 2.000000f, 3.000000f, 2.218750f, 3.218750f, 4.218750f, 4.000000f, 5.000000f, 6.000000f,
            5.500000f, 6.500000f, 7.500000f, 7.000000f, 8.000000f, 9.000000f, 8.781250f, 9.781250f, 10.781250f,
            10.000000f, 11.000000f, 12.000000f, 10.281250f, 11.281250f, 12.281250f, 5.875000f, 6.875000f, 7.875000f,
            7.093750f, 8.093750f, 9.093750f, 8.875000f, 9.875000f, 10.875000f, 10.375000f, 11.375000f, 12.375000f,
            11.875000f, 12.875000f, 13.875000f, 13.656250f, 14.656250f, 15.656250f, 14.875000f, 15.875000f, 16.875000f,
            15.156250f, 16.156250f, 17.156250f, 13.000000f, 14.000000f, 15.000000f, 14.218750f, 15.218750f, 16.218750f,
            16.000000f, 17.000000f, 18.000000f, 17.500000f, 18.500000f, 19.500000f, 19.000000f, 20.000000f, 21.000000f,
            20.781250f, 21.781250f, 22.781250f, 22.000000f, 23.000000f, 24.000000f, 22.281250f, 23.281250f, 24.281250f,
            20.125000f, 21.125000f, 22.125000f, 21.343750f, 22.343750f, 23.343750f, 23.125000f, 24.125000f, 25.125000f,
            24.625000f, 25.625000f, 26.625000f, 26.125000f, 27.125000f, 28.125000f, 27.906250f, 28.906250f, 29.906250f,
            29.125000f, 30.125000f, 31.125000f, 29.406250f, 30.406250f, 31.406250f, 25.000000f, 26.000000f, 27.000000f,
            26.218750f, 27.218750f, 28.218750f, 28.000000f, 29.000000f, 30.000000f, 29.500000f, 30.500000f, 31.500000f,
            31.000000f, 32.000000f, 33.000000f, 32.781250f, 33.781250f, 34.781250f, 34.000000f, 35.000000f, 36.000000f,
            34.281250f, 35.281250f, 36.281250f, 26.125000f, 27.125000f, 28.125000f, 27.343750f, 28.343750f, 29.343750f,
            29.125000f, 30.125000f, 31.125000f, 30.625000f, 31.625000f, 32.625000f, 32.125000f, 33.125000f, 34.125000f,
            33.906250f, 34.906250f, 35.906250f, 35.125000f, 36.125000f, 37.125000f, 35.406250f, 36.406250f, 37.406250f
            });
    input.linspace(1);
    auto size = NDArrayFactory::create<int>({6, 8});
    nd4j::ops::resize_bicubic op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Resized to 6x8");
//    expected.printBuffer("Expect for 6x8");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeBicubic_Test5) {

    NDArray input    = NDArrayFactory::create<double>('c', {1, 4, 4, 3});
    NDArray expected = NDArrayFactory::create<float>('c', {1, 8, 8, 3}, {
            1.000000f, 2.000000f, 3.000000f, 2.218750f, 3.218750f, 4.218750f, 4.000000f, 5.000000f, 6.000000f,
            5.500000f, 6.500000f, 7.500000f, 7.000000f, 8.000000f, 9.000000f, 8.781250f, 9.781250f, 10.781250f,
            10.000000f, 11.000000f, 12.000000f, 10.281250f, 11.281250f, 12.281250f, 5.875000f, 6.875000f, 7.875000f,
            7.093750f, 8.093750f, 9.093750f, 8.875000f, 9.875000f, 10.875000f, 10.375000f, 11.375000f, 12.375000f,
            11.875000f, 12.875000f, 13.875000f, 13.656250f, 14.656250f, 15.656250f, 14.875000f, 15.875000f, 16.875000f,
            15.156250f, 16.156250f, 17.156250f, 13.000000f, 14.000000f, 15.000000f, 14.218750f, 15.218750f, 16.218750f,
            16.000000f, 17.000000f, 18.000000f, 17.500000f, 18.500000f, 19.500000f, 19.000000f, 20.000000f, 21.000000f,
            20.781250f, 21.781250f, 22.781250f, 22.000000f, 23.000000f, 24.000000f, 22.281250f, 23.281250f, 24.281250f,
            19.000000f, 20.000000f, 21.000000f, 20.218750f, 21.218750f, 22.218750f, 22.000000f, 23.000000f, 24.000000f,
            23.500000f, 24.500000f, 25.500000f, 25.000000f, 26.000000f, 27.000000f, 26.781250f, 27.781250f, 28.781250f,
            28.000000f, 29.000000f, 30.000000f, 28.281250f, 29.281250f, 30.281250f, 25.000000f, 26.000000f, 27.000000f,
            26.218750f, 27.218750f, 28.218750f, 28.000000f, 29.000000f, 30.000000f, 29.500000f, 30.500000f, 31.500000f,
            31.000000f, 32.000000f, 33.000000f, 32.781250f, 33.781250f, 34.781250f, 34.000000f, 35.000000f, 36.000000f,
            34.281250f, 35.281250f, 36.281250f, 32.125000f, 33.125000f, 34.125000f, 33.343750f, 34.343750f, 35.343750f,
            35.125000f, 36.125000f, 37.125000f, 36.625000f, 37.625000f, 38.625000f, 38.125000f, 39.125000f, 40.125000f,
            39.906250f, 40.906250f, 41.906250f, 41.125000f, 42.125000f, 43.125000f, 41.406250f, 42.406250f, 43.406250f,
            37.000000f, 38.000000f, 39.000000f, 38.218750f, 39.218750f, 40.218750f, 40.000000f, 41.000000f, 42.000000f,
            41.500000f, 42.500000f, 43.500000f, 43.000000f, 44.000000f, 45.000000f, 44.781250f, 45.781250f, 46.781250f,
            46.000000f, 47.000000f, 48.000000f, 46.281250f, 47.281250f, 48.281250f, 38.125000f, 39.125000f, 40.125000f,
            39.343750f, 40.343750f, 41.343750f, 41.125000f, 42.125000f, 43.125000f, 42.625000f, 43.625000f, 44.625000f,
            44.125000f, 45.125000f, 46.125000f, 45.906250f, 46.906250f, 47.906250f, 47.125000f, 48.125000f, 49.125000f,
            47.406250f, 48.406250f, 49.406250f,
        });
    input.linspace(1);
    auto size = NDArrayFactory::create<int>({8, 8});
    nd4j::ops::resize_bicubic op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Resized to 8x8");
//    expected.printBuffer("Expect for 8x8");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeBicubic_Test6) {

    NDArray input    = NDArrayFactory::create<float>('c', {7, 7, 1}, {
             1.f, 2.1f,  3.15f,  4.2f,  5.15f,   6.1f,   7.f,
             8.f, 9.1f,   10.f,  11.f,  12.9f,  13.1f,  14.f,
            15.f, 16.f,   17.f,  18.f,   19.f,   20.f,  21.f,
            22.f, 23.f,   24.f,  25.f,   26.f,   27.f,  28.f,
            30.f, 31.f,   32.f,  33.f,   34.f,   35.f,  36.f,
            37.f, 38.f,   39.f,  40.f,   41.f,   42.f,  43.f,
            44.f, 45.f,   46.f,  47.f,   48.f,   49.f,  50.f
    });

    NDArray expected = NDArrayFactory::create<float>('c', {30, 30, 1}, {
             1.000000f,  1.197616f,  1.417436f,  1.677577f,  1.996158f,  2.328327f,  2.550918f,  2.736061f,  2.965541f,
             3.292965f,  3.544151f,  3.738035f,  3.948995f,  4.248106f,  4.507379f,  4.684374f,  4.857284f,  5.104302f,
             5.386991f,  5.581401f,  5.753962f,  5.974285f,  6.272836f,  6.520426f,  6.718899f,  6.887104f,  7.039068f,
             7.099216f,  7.078424f,  7.028189f,  2.247592f,  2.446947f,  2.669489f,  2.931238f,  3.248216f,  3.574534f,
             3.789310f,  3.965697f,  4.186417f,  4.504653f,  4.740569f,  4.921706f,  5.133866f,  5.459533f,  5.774461f,
             6.019787f,  6.254011f,  6.535633f,  6.809730f,  6.960779f,  7.074942f,  7.241601f,  7.509489f,  7.749949f,
             7.954571f,  8.131972f,  8.286526f,  8.346463f,  8.325745f,  8.275683f,  3.628684f,  3.830573f,  4.056959f,
             4.321157f,  4.636486f,  4.955650f,  5.160583f,  5.325847f,  5.535462f,  5.842160f,  6.058749f,  6.223753f,
             6.437597f,  6.797369f,  7.183604f,  7.516402f,  7.829034f,  8.154773f,  8.417635f,  8.512958f,  8.552100f,
             8.649708f,  8.877880f,  9.108794f,  9.320926f,  9.509781f,  9.667375f,  9.726940f,  9.706349f,  9.656599f,
             5.276778f,  5.480438f,  5.709702f,  5.975448f,  6.288551f,  6.600570f,  6.796207f,  6.951142f,  7.150400f,
             7.446143f,  7.644651f,  7.794562f,  8.009684f,  8.400473f,  8.851847f,  9.264690f,  9.649218f, 10.015648f,
            10.268647f, 10.313368f, 10.284327f, 10.319379f, 10.512033f, 10.734956f, 10.954604f, 11.154507f, 11.315369f,
            11.374779f, 11.354242f, 11.304622f,  7.325373f,  7.528484f,  7.757575f,  8.022221f,  8.331997f,  8.638187f,
             8.827649f,  8.976217f,  9.168955f,  9.457260f,  9.644237f,  9.784517f,  9.999621f, 10.407702f, 10.896234f,
            11.355122f, 11.781423f, 12.172186f, 12.420712f, 12.437449f, 12.370511f, 12.371386f, 12.545973f, 12.766424f,
            12.992249f, 13.200120f, 13.364252f, 13.424109f, 13.403420f, 13.353425f,  9.493208f,  9.692467f,  9.916944f,
            10.176801f, 10.482199f, 10.785470f, 10.974367f, 11.123442f, 11.316370f, 11.603645f, 11.790616f, 11.930889f,
            12.144082f, 12.546447f, 13.024898f, 13.472300f, 13.889232f, 14.276275f, 14.528972f, 14.555555f, 14.501450f,
            14.515459f, 14.700572f, 14.927055f, 15.156046f, 15.366046f, 15.532901f, 15.594008f, 15.572885f, 15.521847f,
            10.970133f, 11.163599f, 11.380694f, 11.633735f, 11.935032f, 12.238887f, 12.432540f, 12.588294f, 12.787534f,
            13.079956f, 13.277520f, 13.426631f, 13.636713f, 14.013844f, 14.441672f, 14.827978f, 15.191209f, 15.549808f,
            15.813430f, 15.881828f, 15.883522f, 15.950411f, 16.169330f, 16.407940f, 16.636436f, 16.842583f, 17.010887f,
            17.073630f, 17.051940f, 16.999537f, 12.219155f, 12.406129f, 12.614796f, 12.860335f, 13.157928f, 13.464224f,
            13.665207f, 13.830567f, 14.039036f, 14.339629f, 14.552863f, 14.715049f, 14.921564f, 15.264454f, 15.622843f,
            15.924977f, 16.213829f, 16.532364f, 16.809900f, 16.934835f, 17.012146f, 17.150164f, 17.413412f, 17.666712f,
            17.892765f, 18.092070f, 18.261044f, 18.325531f, 18.303238f, 18.249378f, 13.766397f, 13.947391f, 14.148263f,
            14.386917f, 14.681246f, 14.990087f, 15.198166f, 15.372728f, 15.590062f, 15.898583f, 16.126892f, 16.301655f,
            16.504870f, 16.815214f, 17.107498f, 17.329458f, 17.547403f, 17.827654f, 18.118288f, 18.296928f, 18.446100f,
            18.651634f, 18.956806f, 19.223820f, 19.447308f, 19.639887f, 19.809319f, 19.875397f, 19.852556f, 19.797365f,
            15.941937f, 16.118704f, 16.314133f, 16.547867f, 16.839561f, 17.149540f, 17.361883f, 17.542162f, 17.764957f,
            18.078188f, 18.315733f, 18.498205f, 18.699116f, 18.988684f, 19.238989f, 19.410137f, 19.583265f, 19.839512f,
            20.138780f, 20.351770f, 20.546844f, 20.795671f, 21.128067f, 21.404358f, 21.626736f, 21.815500f, 21.985610f,
            22.052843f, 22.029604f, 21.973448f, 17.535220f, 17.710770f, 17.904636f, 18.136950f, 18.427840f, 18.738056f,
            18.951529f, 19.133352f, 19.357613f, 19.672083f, 19.912102f, 20.096638f, 20.296894f, 20.580765f, 20.819603f,
            20.976887f, 21.137802f, 21.387535f, 21.689209f, 21.911621f, 22.119276f, 22.379990f, 22.719910f, 22.998823f,
            23.220970f, 23.408760f, 23.579110f, 23.646685f, 23.623325f, 23.566887f, 18.746353f, 18.922657f, 19.117487f,
            19.350685f, 19.642070f, 19.952137f, 20.164913f, 20.345781f, 20.569134f, 20.882840f, 21.121330f, 21.304590f,
            21.505253f, 21.792645f, 22.038572f, 22.204426f, 22.372890f, 22.626648f, 22.926834f, 23.143423f, 23.343302f,
            23.596668f, 23.931936f, 24.209232f, 24.431519f, 24.619913f, 24.790110f, 24.857473f, 24.834190f, 24.777927f,
            20.166560f, 20.344206f, 20.540766f, 20.775532f, 21.067804f, 21.377607f, 21.589132f, 21.768297f, 21.990030f,
            22.302366f, 22.538124f, 22.719105f, 22.920494f, 23.214176f, 23.472767f, 23.653934f, 23.835890f, 24.096842f,
            24.394371f, 24.600555f, 24.786541f, 25.026773f, 25.353731f, 25.628130f, 25.850672f, 26.040140f, 26.210072f,
            26.277063f, 26.253906f, 26.197956f, 22.363024f, 22.541250f, 22.738552f, 22.973991f, 23.266647f, 23.576340f,
            23.787327f, 23.965760f, 24.186796f, 24.498543f, 24.733124f, 24.913122f, 25.114826f, 25.411213f, 25.675262f,
            25.863028f, 26.050789f, 26.314838f, 26.611223f, 26.812925f, 26.992926f, 27.227505f, 27.550882f, 27.824034f,
            28.046684f, 28.236614f, 28.406433f, 28.473265f, 28.450163f, 28.394344f, 24.429443f, 24.607670f, 24.804970f,
            25.040410f, 25.333065f, 25.642756f, 25.853743f, 26.032173f, 26.253210f, 26.564959f, 26.799540f, 26.979540f,
            27.181242f, 27.477630f, 27.741680f, 27.929441f, 28.117207f, 28.381254f, 28.677637f, 28.879343f, 29.059345f,
            29.293922f, 29.617298f, 29.890451f, 30.113104f, 30.303034f, 30.472853f, 30.539684f, 30.516582f, 30.460762f,
            26.000000f, 26.178228f, 26.375526f, 26.610970f, 26.903624f, 27.213314f, 27.424305f, 27.602734f, 27.823772f,
            28.135519f, 28.370100f, 28.550098f, 28.751800f, 29.048190f, 29.312237f, 29.500000f, 29.687763f, 29.951813f,
            30.248200f, 30.449903f, 30.629902f, 30.864483f, 31.187859f, 31.461012f, 31.683659f, 31.873592f, 32.043407f,
            32.110240f, 32.087135f, 32.031320f, 27.570559f, 27.748787f, 27.946087f, 28.181528f, 28.474184f, 28.783876f,
            28.994865f, 29.173294f, 29.394330f, 29.706080f, 29.940659f, 30.120655f, 30.322360f, 30.618746f, 30.882797f,
            31.070557f, 31.258320f, 31.522371f, 31.818754f, 32.020460f, 32.200460f, 32.435040f, 32.758415f, 33.031567f,
            33.254220f, 33.444150f, 33.613964f, 33.680794f, 33.657696f, 33.601880f, 29.636976f, 29.815207f, 30.012500f,
            30.247944f, 30.540600f, 30.850290f, 31.061283f, 31.239712f, 31.460750f, 31.772500f, 32.007080f, 32.187077f,
            32.388780f, 32.685165f, 32.949215f, 33.136980f, 33.324740f, 33.588790f, 33.885178f, 34.086884f, 34.266880f,
            34.501457f, 34.824837f, 35.097990f, 35.320637f, 35.510574f, 35.680390f, 35.747215f, 35.724117f, 35.668300f,
            31.833440f, 32.011665f, 32.208970f, 32.444412f, 32.737070f, 33.046757f, 33.257744f, 33.436176f, 33.657207f,
            33.968960f, 34.203537f, 34.383537f, 34.585240f, 34.881630f, 35.145676f, 35.333440f, 35.521206f, 35.785255f,
            36.081642f, 36.283340f, 36.463340f, 36.697920f, 37.021297f, 37.294453f, 37.517097f, 37.707027f, 37.876846f,
            37.943680f, 37.920578f, 37.864758f, 33.253647f, 33.431873f, 33.629170f, 33.864613f, 34.157270f, 34.466957f,
            34.677948f, 34.856377f, 35.077415f, 35.389160f, 35.623745f, 35.803745f, 36.005447f, 36.301834f, 36.565884f,
            36.753647f, 36.941406f, 37.205456f, 37.501840f, 37.703545f, 37.883545f, 38.118122f, 38.441500f, 38.714653f,
            38.937300f, 39.127235f, 39.297054f, 39.363884f, 39.340782f, 39.284960f, 34.464783f, 34.643010f, 34.840305f,
            35.075752f, 35.368404f, 35.678100f, 35.889088f, 36.067516f, 36.288550f, 36.600300f, 36.834885f, 37.014877f,
            37.216583f, 37.512970f, 37.777020f, 37.964783f, 38.152546f, 38.416595f, 38.712980f, 38.914684f, 39.094685f,
            39.329260f, 39.652645f, 39.925793f, 40.148440f, 40.338375f, 40.508194f, 40.575024f, 40.551920f, 40.496105f,
            36.058067f, 36.236290f, 36.433590f, 36.669033f, 36.961685f, 37.271378f, 37.482370f, 37.660800f, 37.881836f,
            38.193590f, 38.428170f, 38.608162f, 38.809868f, 39.106250f, 39.370300f, 39.558064f, 39.745830f, 40.009880f,
            40.306267f, 40.507970f, 40.687970f, 40.922550f, 41.245926f, 41.519077f, 41.741722f, 41.931652f, 42.101475f,
            42.168304f, 42.145203f, 42.089386f, 38.315002f, 38.493233f, 38.690533f, 38.925976f, 39.218628f, 39.528320f,
            39.739307f, 39.917736f, 40.138775f, 40.450520f, 40.685104f, 40.865097f, 41.066803f, 41.363190f, 41.627243f,
            41.815002f, 42.002766f, 42.266820f, 42.563200f, 42.764908f, 42.944904f, 43.179485f, 43.502860f, 43.776016f,
            43.998665f, 44.188595f, 44.358418f, 44.425247f, 44.402145f, 44.346330f, 40.227080f, 40.405310f, 40.602608f,
            40.838050f, 41.130707f, 41.440395f, 41.651382f, 41.829820f, 42.050854f, 42.362600f, 42.597183f, 42.777180f,
            42.978880f, 43.275270f, 43.539320f, 43.727080f, 43.914845f, 44.178894f, 44.475280f, 44.676983f, 44.856983f,
            45.091560f, 45.414940f, 45.688090f, 45.910740f, 46.100674f, 46.270493f, 46.337322f, 46.314220f, 46.258400f,
            41.785618f, 41.963844f, 42.161144f, 42.396584f, 42.689240f, 42.998936f, 43.209923f, 43.388355f, 43.609394f,
            43.921143f, 44.155720f, 44.335716f, 44.537420f, 44.833805f, 45.097860f, 45.285614f, 45.473377f, 45.737427f,
            46.033817f, 46.235523f, 46.415524f, 46.650105f, 46.973476f, 47.246630f, 47.469276f, 47.659210f, 47.829030f,
            47.895855f, 47.872753f, 47.816940f, 43.115140f, 43.293365f, 43.490665f, 43.726105f, 44.018764f, 44.328457f,
            44.539444f, 44.717873f, 44.938910f, 45.250660f, 45.485240f, 45.665237f, 45.866940f, 46.163326f, 46.427376f,
            46.615143f, 46.802902f, 47.066956f, 47.363342f, 47.565050f, 47.745050f, 47.979626f, 48.302998f, 48.576153f,
            48.798798f, 48.988730f, 49.158546f, 49.225376f, 49.202282f, 49.146458f, 44.303867f, 44.482094f, 44.679394f,
            44.914833f, 45.207493f, 45.517180f, 45.728170f, 45.906600f, 46.127640f, 46.439384f, 46.673965f, 46.853966f,
            47.055668f, 47.352055f, 47.616100f, 47.803867f, 47.991630f, 48.255680f, 48.552063f, 48.753770f, 48.933773f,
            49.168350f, 49.491726f, 49.764877f, 49.987526f, 50.177460f, 50.347275f, 50.414100f, 50.391006f, 50.335186f,
            44.771675f, 44.949905f, 45.147200f, 45.382645f, 45.675300f, 45.984990f, 46.195976f, 46.374413f, 46.595448f,
            46.907196f, 47.141773f, 47.321774f, 47.523476f, 47.819862f, 48.083910f, 48.271680f, 48.459446f, 48.723490f,
            49.019882f, 49.221580f, 49.401585f, 49.636160f, 49.959538f, 50.232693f, 50.455338f, 50.645270f, 50.815090f,
            50.881920f, 50.858818f, 50.803000f, 44.609966f, 44.788193f, 44.985493f, 45.220936f, 45.513590f, 45.823280f,
            46.034270f, 46.212700f, 46.433743f, 46.745490f, 46.980070f, 47.160065f, 47.361770f, 47.658157f, 47.922207f,
            48.109970f, 48.297733f, 48.561783f, 48.858166f, 49.059875f, 49.239872f, 49.474450f, 49.797830f, 50.070980f,
            50.293625f, 50.483560f, 50.653378f, 50.720203f, 50.697100f, 50.641280f, 44.219246f, 44.397472f, 44.594772f,
            44.830210f, 45.122868f, 45.432560f, 45.643543f, 45.821980f, 46.043020f, 46.354763f, 46.589344f, 46.769340f,
            46.971046f, 47.267433f, 47.531483f, 47.719242f, 47.907005f, 48.171050f, 48.467438f, 48.669140f, 48.849144f,
            49.083720f, 49.407100f, 49.680256f, 49.902905f, 50.092834f, 50.262653f, 50.329483f, 50.306380f, 50.250570f
    });

    auto size = NDArrayFactory::create<int>({30, 30});
    nd4j::ops::resize_bicubic op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());
    NDArray* result = results->at(0);

//    result->printBuffer("Resized to 30x30");
//    expected.printBuffer("Expect for 30x30");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeBicubic_Test7) {

    NDArray input    = NDArrayFactory::create<double>('c', {2, 5, 5, 1}, {
        0.2303, 0.7950, 0.8171, 0.0451, 0.3690, 0.6846, 0.2727, 0.2770, 0.2381, 0.9511,
        0.4116, 0.3997, 0.4075, 0.6275, 0.8018, 0.0678, 0.6221, 0.2982, 0.1524, 0.2613,
        0.7425, 0.6036, 0.7926, 0.5838, 0.1361, 0.4154, 0.3634, 0.3741, 0.2088, 0.2989,
        0.3982, 0.5618, 0.7266, 0.1089, 0.2922, 0.3306, 0.2869, 0.6638, 0.3091, 0.9312,
        0.0240, 0.2893, 0.5632, 0.9625, 0.4189, 0.3854, 0.2743, 0.6754, 0.8820, 0.8699});

    NDArray expected = NDArrayFactory::create<float>('c', {2, 9, 9, 1}, {
                0.2303f,    0.54569f,   0.840649f, 0.92725444f, 0.65660673f,
            0.16641647f, 0.06117659f, 0.33279106f,  0.4023279f,  0.5139505f,
            0.49821317f,  0.4906872f,   0.537642f,  0.4070102f, 0.13030615f,
              0.258801f, 0.65352744f,   0.773368f, 0.69225276f, 0.44177493f,
            0.21910316f, 0.22368976f, 0.24221404f, 0.21399781f,  0.5114972f,
             0.9169859f,  1.0511527f,  0.5608501f, 0.41315168f,  0.2913824f,
             0.2966933f, 0.38585684f, 0.48849702f, 0.71013063f,  0.9086001f,
             0.9794303f, 0.29625386f, 0.39427578f, 0.45971435f, 0.39693952f,
            0.40860707f, 0.51061106f,  0.6181093f, 0.67309624f, 0.69564015f,
            0.06012487f,  0.3863805f, 0.58993465f, 0.40679216f, 0.22607432f,
            0.20093678f, 0.25901243f,  0.3615362f, 0.39371052f, 0.24176767f,
             0.4868709f,   0.650651f,  0.5493148f,  0.3825456f, 0.27788478f,
            0.18927254f, 0.16692996f, 0.15432167f,   0.677519f,  0.6236242f,
            0.61700624f,  0.7214321f,  0.7307374f,  0.6251454f,  0.3924176f,
            0.17802659f, 0.10231908f, 0.81192374f, 0.66878575f,  0.6118803f,
             0.7797006f,  0.8396968f, 0.72889954f, 0.44547448f, 0.16794783f,
            0.07125802f,     0.4154f, 0.38504714f,  0.3623221f,  0.3862173f,
             0.3397379f, 0.23285517f, 0.21876639f,  0.2892362f, 0.30817088f,
            0.41268015f, 0.45587808f, 0.51991886f, 0.60977113f, 0.49489656f,
            0.21313031f, 0.11297428f,  0.2167207f, 0.23940037f, 0.39337245f,
            0.46112412f,   0.583034f, 0.76207364f,  0.6326203f, 0.22189438f,
            0.12071565f,  0.3275853f,  0.3794855f, 0.38497013f, 0.35049653f,
            0.41895086f,   0.671095f, 0.62119365f, 0.22362521f, 0.30189657f,
            0.72530353f, 0.85048175f,  0.2524255f,  0.2182264f,  0.2964637f,
             0.5361996f,  0.6255393f, 0.46424767f,  0.5741281f,  0.8408146f,
            0.92403257f, 0.04648584f, 0.14959256f, 0.32215607f, 0.46194845f,
             0.6642166f, 0.83560026f,  0.7663391f,  0.5284251f,  0.4573109f,
            0.10357999f, 0.17442937f, 0.32116935f, 0.45530772f,  0.7163773f,
             0.9856574f,  0.8976148f,  0.5538923f, 0.45173654f, 0.34958175f,
             0.2680429f, 0.30470955f, 0.51233786f, 0.75128907f, 0.86736864f,
             0.8982046f, 0.83254474f,  0.8168574f,  0.4225865f,  0.2956836f,
            0.29948136f,  0.5276342f, 0.76461166f,  0.8442875f,   0.907862f,
             0.9139262f, 0.92068815f
    });
    auto size = NDArrayFactory::create<int>({9, 9});
    nd4j::ops::resize_bicubic op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Resized to 9x9");
//    expected.printBuffer("Expect for 9x9");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeBicubic_Test8) {

    NDArray input    = NDArrayFactory::create<double>('c', {2, 5, 5, 1}, {
            0.23028551377579154,  0.7949972231516509,  0.8171307820461517, 0.04507309923418412,   0.3689673597428338,
             0.6845757584903018, 0.27268547668219667,  0.2770196372806053,  0.2381478370531429,   0.9511201914609859,
            0.41160882670429033,  0.3997152563642703,  0.4074505147711718,  0.6274595060113246,   0.8017922711300232,
            0.06782045852179475,  0.6220772280691722,  0.2982335327629251,  0.1523603480424196,   0.2612986044295986,
            0.7424762244324299,   0.6036156464824591,  0.7926371071102005,  0.5838270656432538,  0.13607200219168547,
            0.4154002170215956,  0.36340617544852116, 0.37405031188276827, 0.20880251686544882,    0.298919946410666,
            0.39820758164277126,  0.5617728968896589,    0.72660225993937, 0.10888245916813699,  0.29215797784445496,
             0.3305531351746034, 0.28693451964931715,  0.6637635348315494, 0.30913418229827583,   0.9312186188801752,
             0.0239594182399363,  0.2892942758780874,  0.5631691110629038,  0.9625499752246309,   0.4189439089689968,
            0.3854304088214935,  0.27426304203925045,  0.6754051704648238,  0.8820362490795286,   0.8699337744328859});


    auto testData = NDArrayFactory::create<float>('c', {2,9,9,1}, {
        0.230286514f,        0.510566354f,        0.794997215f,        0.931386113f,        0.817130804f,        0.402811885f,        0.045073099f,        0.134639814f,        0.368967354f,
        0.483021289f,        0.501266003f,        0.521932304f,        0.572325349f,        0.534847379f,        0.267853439f,        0.105112493f,        0.349290252f,        0.674043298f,
        0.684575737f,        0.478224277f,        0.272685468f,        0.239882097f,         0.27701965f,        0.191148892f,         0.23814784f,        0.590989769f,        0.951120198f,
        0.622912169f,        0.441326082f,        0.266387194f,        0.232538164f,        0.301838756f,        0.356378645f,        0.495445013f,        0.756725252f,        0.981704295f,
        0.411608815f,         0.40493685f,        0.399715245f,        0.381842017f,        0.407450527f,        0.501836538f,        0.627459526f,        0.735251725f,        0.801792264f,
        0.150875032f,        0.357000858f,        0.524536073f,        0.450354964f,        0.318719596f,        0.319606483f,        0.385957927f,         0.46392554f,        0.529285908f,
         0.06782046f,        0.375309169f,        0.622077227f,        0.525792599f,        0.298233539f,        0.184723631f,         0.15236035f,        0.193153858f,        0.261298597f,

        0.372918189f,        0.512539625f,         0.63369292f,        0.628733814f,        0.535196245f,        0.436597466f,        0.323553175f,        0.215942055f,        0.148014024f,
        0.742476225f,        0.655325174f,        0.603615642f,        0.704684138f,         0.79263711f,        0.747929871f,        0.583827078f,        0.340373576f,        0.136071995f,
        0.415400207f,        0.388405323f,        0.363406181f,        0.379345775f,        0.374050319f,         0.28397581f,        0.208802521f,        0.238369256f,        0.298919946f,
        0.413146496f,        0.444389015f,        0.488355637f,        0.568351328f,        0.556217432f,        0.345546633f,        0.140068889f,        0.148834035f,         0.23562704f,
        0.398207575f,        0.464537472f,        0.561772883f,        0.717433035f,        0.726602256f,        0.416013002f,        0.108882457f,        0.142608985f,        0.292157978f,
        0.391511708f,        0.389470309f,        0.442729384f,        0.651181757f,        0.737665415f,         0.41685915f,        0.138383076f,        0.342548877f,        0.659080088f,

        0.330553144f,        0.273416102f,        0.286934525f,         0.50450629f,        0.663763523f,        0.463456154f,        0.309134185f,        0.586929917f,        0.931218624f,
        0.137025774f,        0.169145152f,        0.263757467f,        0.436182201f,        0.597053051f,        0.657990932f,        0.662163854f,         0.68354249f,        0.692712903f,
        0.023959421f,        0.130951077f,        0.289294273f,        0.413664877f,        0.563169122f,        0.839498401f,        0.962549984f,        0.728188932f,        0.418943912f,
        0.175951749f,        0.198239252f,        0.281999886f,        0.420836329f,        0.609856486f,        0.863734365f,        0.983550847f,        0.825015843f,        0.596413136f,
        0.385430396f,        0.292239636f,        0.274263054f,        0.445040524f,        0.675405145f,        0.817462444f,        0.882036269f,        0.895356655f,        0.869933784f
    });

    auto size = NDArrayFactory::create<int>({9, 9});
    nd4j::ops::resize_bicubic op;
    auto results = op.evaluate({&input, &size}, {}, {}, {true, false});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Resized to 9x9");
//    testData.printBuffer("Expect for 9x9");
    ASSERT_TRUE(testData.isSameShape(result));
    ASSERT_TRUE(testData.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeArea_Test1) {

    NDArray input    = NDArrayFactory::create<double>('c', {1, 3, 3, 4});
    NDArray expected = NDArrayFactory::create<float>('c', {1, 6, 6, 4}, {
             1.f,  2.f,  3.f,  4.f,
             1.f,  2.f,  3.f,  4.f,
             5.f,  6.f,  7.f,  8.f,
             5.f,  6.f,  7.f,  8.f,
             9.f, 10.f, 11.f, 12.f,
             9.f, 10.f, 11.f, 12.f,

             1.f,  2.f,  3.f,  4.f,
             1.f,  2.f,  3.f,  4.f,
             5.f,  6.f,  7.f,  8.f,
             5.f,  6.f,  7.f,  8.f,
             9.f, 10.f, 11.f, 12.f,
             9.f, 10.f, 11.f, 12.f,

            13.f, 14.f, 15.f, 16.f,
            13.f, 14.f, 15.f, 16.f,
            17.f, 18.f, 19.f, 20.f,
            17.f, 18.f, 19.f, 20.f,
            21.f, 22.f, 23.f, 24.f,
            21.f, 22.f, 23.f, 24.f,

            13.f, 14.f, 15.f, 16.f,
            13.f, 14.f, 15.f, 16.f,
            17.f, 18.f, 19.f, 20.f,
            17.f, 18.f, 19.f, 20.f,
            21.f, 22.f, 23.f, 24.f,
            21.f, 22.f, 23.f, 24.f,

            25.f, 26.f, 27.f, 28.f,
            25.f, 26.f, 27.f, 28.f,
            29.f, 30.f, 31.f, 32.f,
            29.f, 30.f, 31.f, 32.f,
            33.f, 34.f, 35.f, 36.f,
            33.f, 34.f, 35.f, 36.f,

            25.f, 26.f, 27.f, 28.f,
            25.f, 26.f, 27.f, 28.f,
            29.f, 30.f, 31.f, 32.f,
            29.f, 30.f, 31.f, 32.f,
            33.f, 34.f, 35.f, 36.f,
            33.f, 34.f, 35.f, 36.f    });
    input.linspace(1);
    auto size = NDArrayFactory::create<int>({6, 6});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 6x6");
//    expected.printBuffer("Area Expect for 6x6");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeArea_Test2) {

    NDArray input    = NDArrayFactory::create<float>('c', {1, 3, 3, 1});
    NDArray expected = NDArrayFactory::create<float>('c', {1, 6, 6, 1}, {
            1.f, 1.f, 2.f, 2.f, 3.f, 3.f,
            1.f, 1.f, 2.f, 2.f, 3.f, 3.f,
            4.f, 4.f, 5.f, 5.f, 6.f, 6.f,
            4.f, 4.f, 5.f, 5.f, 6.f, 6.f,
            7.f, 7.f, 8.f, 8.f, 9.f, 9.f,
            7.f, 7.f, 8.f, 8.f, 9.f, 9.f
    });
    input.linspace(1);
    auto size = NDArrayFactory::create<int>({6, 6});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 6x6");
//    expected.printBuffer("Area Expect for 6x6");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}


TEST_F(DeclarableOpsTests11, ImageResizeArea_Test3) {

    NDArray input    = NDArrayFactory::create<float>('c', {1, 3, 3, 3});
    NDArray expected = NDArrayFactory::create<float>('c', {1, 6, 6, 3}, {
         1.f,  2.f,  3.f,   1.f,  2.f,  3.f,   4.f,  5.f,  6.f,   4.f,  5.f,  6.f,   7.f,  8.f,  9.f,   7.f,  8.f,  9.f,
         1.f,  2.f,  3.f,   1.f,  2.f,  3.f,   4.f,  5.f,  6.f,   4.f,  5.f,  6.f,   7.f,  8.f,  9.f,   7.f,  8.f,  9.f,
        10.f, 11.f, 12.f,  10.f, 11.f, 12.f,  13.f, 14.f, 15.f,  13.f, 14.f, 15.f,  16.f, 17.f, 18.f,  16.f, 17.f, 18.f,
        10.f, 11.f, 12.f,  10.f, 11.f, 12.f,  13.f, 14.f, 15.f,  13.f, 14.f, 15.f,  16.f, 17.f, 18.f,  16.f, 17.f, 18.f,
        19.f, 20.f, 21.f,  19.f, 20.f, 21.f,  22.f, 23.f, 24.f,  22.f, 23.f, 24.f,  25.f, 26.f, 27.f,  25.f, 26.f, 27.f,
        19.f, 20.f, 21.f,  19.f, 20.f, 21.f,  22.f, 23.f, 24.f,  22.f, 23.f, 24.f,  25.f, 26.f, 27.f,  25.f, 26.f, 27.f
    });
    input.linspace(1);
    auto size = NDArrayFactory::create<int>({6, 6});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 6x6");
//    expected.printBuffer("Area Expect for 6x6");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeArea_Test4) {

    NDArray input    = NDArrayFactory::create<float>('c', {2, 3, 3, 3}, {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27
    });

    NDArray expected = NDArrayFactory::create<float>('c', {2, 6, 6, 3}, {
         1.f,  2.f,  3.f,   1.f,  2.f,  3.f,   4.f,  5.f,  6.f,   4.f,  5.f,  6.f,   7.f,  8.f,  9.f,   7.f,  8.f,  9.f,
         1.f,  2.f,  3.f,   1.f,  2.f,  3.f,   4.f,  5.f,  6.f,   4.f,  5.f,  6.f,   7.f,  8.f,  9.f,   7.f,  8.f,  9.f,
        10.f, 11.f, 12.f,  10.f, 11.f, 12.f,  13.f, 14.f, 15.f,  13.f, 14.f, 15.f,  16.f, 17.f, 18.f,  16.f, 17.f, 18.f,
        10.f, 11.f, 12.f,  10.f, 11.f, 12.f,  13.f, 14.f, 15.f,  13.f, 14.f, 15.f,  16.f, 17.f, 18.f,  16.f, 17.f, 18.f,
        19.f, 20.f, 21.f,  19.f, 20.f, 21.f,  22.f, 23.f, 24.f,  22.f, 23.f, 24.f,  25.f, 26.f, 27.f,  25.f, 26.f, 27.f,
        19.f, 20.f, 21.f,  19.f, 20.f, 21.f,  22.f, 23.f, 24.f,  22.f, 23.f, 24.f,  25.f, 26.f, 27.f,  25.f, 26.f, 27.f,

         1.f,  2.f,  3.f,   1.f,  2.f,  3.f,   4.f,  5.f,  6.f,   4.f,  5.f,  6.f,   7.f,  8.f,  9.f,   7.f,  8.f,  9.f,
         1.f,  2.f,  3.f,   1.f,  2.f,  3.f,   4.f,  5.f,  6.f,   4.f,  5.f,  6.f,   7.f,  8.f,  9.f,   7.f,  8.f,  9.f,
        10.f, 11.f, 12.f,  10.f, 11.f, 12.f,  13.f, 14.f, 15.f,  13.f, 14.f, 15.f,  16.f, 17.f, 18.f,  16.f, 17.f, 18.f,
        10.f, 11.f, 12.f,  10.f, 11.f, 12.f,  13.f, 14.f, 15.f,  13.f, 14.f, 15.f,  16.f, 17.f, 18.f,  16.f, 17.f, 18.f,
        19.f, 20.f, 21.f,  19.f, 20.f, 21.f,  22.f, 23.f, 24.f,  22.f, 23.f, 24.f,  25.f, 26.f, 27.f,  25.f, 26.f, 27.f,
        19.f, 20.f, 21.f,  19.f, 20.f, 21.f,  22.f, 23.f, 24.f,  22.f, 23.f, 24.f,  25.f, 26.f, 27.f,  25.f, 26.f, 27.f
    });
    //input.linspace(1);
    auto size = NDArrayFactory::create<int>({6, 6});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 6x6");
//    expected.printBuffer("Area Expect for 6x6");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeArea_Test5) {

    NDArray input    = NDArrayFactory::create<int>('c', {2, 3, 3, 3}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27
    });

    NDArray expected = NDArrayFactory::create<float>('c', {2, 6, 6, 3}, {
            1.f,  2.f,  3.f,   1.f,  2.f,  3.f,   4.f,  5.f,  6.f,   4.f,  5.f,  6.f,   7.f,  8.f,  9.f,   7.f,  8.f,  9.f,
            1.f,  2.f,  3.f,   1.f,  2.f,  3.f,   4.f,  5.f,  6.f,   4.f,  5.f,  6.f,   7.f,  8.f,  9.f,   7.f,  8.f,  9.f,
            10.f, 11.f, 12.f,  10.f, 11.f, 12.f,  13.f, 14.f, 15.f,  13.f, 14.f, 15.f,  16.f, 17.f, 18.f,  16.f, 17.f, 18.f,
            10.f, 11.f, 12.f,  10.f, 11.f, 12.f,  13.f, 14.f, 15.f,  13.f, 14.f, 15.f,  16.f, 17.f, 18.f,  16.f, 17.f, 18.f,
            19.f, 20.f, 21.f,  19.f, 20.f, 21.f,  22.f, 23.f, 24.f,  22.f, 23.f, 24.f,  25.f, 26.f, 27.f,  25.f, 26.f, 27.f,
            19.f, 20.f, 21.f,  19.f, 20.f, 21.f,  22.f, 23.f, 24.f,  22.f, 23.f, 24.f,  25.f, 26.f, 27.f,  25.f, 26.f, 27.f,

            1.f,  2.f,  3.f,   1.f,  2.f,  3.f,   4.f,  5.f,  6.f,   4.f,  5.f,  6.f,   7.f,  8.f,  9.f,   7.f,  8.f,  9.f,
            1.f,  2.f,  3.f,   1.f,  2.f,  3.f,   4.f,  5.f,  6.f,   4.f,  5.f,  6.f,   7.f,  8.f,  9.f,   7.f,  8.f,  9.f,
            10.f, 11.f, 12.f,  10.f, 11.f, 12.f,  13.f, 14.f, 15.f,  13.f, 14.f, 15.f,  16.f, 17.f, 18.f,  16.f, 17.f, 18.f,
            10.f, 11.f, 12.f,  10.f, 11.f, 12.f,  13.f, 14.f, 15.f,  13.f, 14.f, 15.f,  16.f, 17.f, 18.f,  16.f, 17.f, 18.f,
            19.f, 20.f, 21.f,  19.f, 20.f, 21.f,  22.f, 23.f, 24.f,  22.f, 23.f, 24.f,  25.f, 26.f, 27.f,  25.f, 26.f, 27.f,
            19.f, 20.f, 21.f,  19.f, 20.f, 21.f,  22.f, 23.f, 24.f,  22.f, 23.f, 24.f,  25.f, 26.f, 27.f,  25.f, 26.f, 27.f
    });
    //input.linspace(1);
    auto size = NDArrayFactory::create<int>({6, 6});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 6x6");
//    expected.printBuffer("Area Expect for 6x6");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeArea_Test6) {

    NDArray input    = NDArrayFactory::create<int>('c', {2, 3, 3, 1}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            1, 2, 3, 4, 5, 6, 7, 8, 9
    });

    NDArray expected = NDArrayFactory::create<float>('c', {2, 6, 6, 1}, {
            1.f,             1.f,            1.5f,            2.f,            2.f,            3.f,
            1.f,             1.f,            1.5f,            2.f,            2.f,            3.f,
           2.5f,            2.5f,             3.f,            3.5f,           3.5f,          4.5f,
            4.f,             4.f,            4.5f,            5.f,            5.f,            6.f,
            4.f,             4.f,            4.5f,            5.f,            5.f,            6.f,
            7.f,             7.f,            7.5f,            8.f,            8.f,            9.f,

            1.f,             1.f,            1.5f,            2.f,            2.f,            3.f,
            1.f,             1.f,            1.5f,            2.f,            2.f,            3.f,
           2.5f,            2.5f,             3.f,           3.5f,           3.5f,           4.5f,
            4.f,             4.f,            4.5f,            5.f,            5.f,            6.f,
            4.f,             4.f,            4.5f,            5.f,            5.f,            6.f,
            7.f,             7.f,            7.5f,            8.f,            8.f,            9.f
    });
    //input.linspace(1);
    auto size = NDArrayFactory::create<int>({6, 6});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input, &size}, {}, {}, {true});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 6x6");
//    expected.printBuffer("Area Expect for 6x6");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeArea_Test7) {

    NDArray input    = NDArrayFactory::create<int>('c', {2, 3, 3, 1}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            1, 2, 3, 4, 5, 6, 7, 8, 9
    });

    NDArray expected = NDArrayFactory::create<float>('c', {2, 6, 6, 1}, {
            1.f,             1.f,            1.5f,            2.f,            2.f,            3.f,
            1.f,             1.f,            1.5f,            2.f,            2.f,            3.f,
            2.5f,            2.5f,             3.f,            3.5f,           3.5f,          4.5f,
            4.f,             4.f,            4.5f,            5.f,            5.f,            6.f,
            4.f,             4.f,            4.5f,            5.f,            5.f,            6.f,
            7.f,             7.f,            7.5f,            8.f,            8.f,            9.f,

            1.f,             1.f,            1.5f,            2.f,            2.f,            3.f,
            1.f,             1.f,            1.5f,            2.f,            2.f,            3.f,
            2.5f,            2.5f,             3.f,           3.5f,           3.5f,           4.5f,
            4.f,             4.f,            4.5f,            5.f,            5.f,            6.f,
            4.f,             4.f,            4.5f,            5.f,            5.f,            6.f,
            7.f,             7.f,            7.5f,            8.f,            8.f,            9.f
    });
    //input.linspace(1);
//    auto size = NDArrayFactory::create<int>({6, 6});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input}, {}, {6, 6}, {true});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 6x6");
//    expected.printBuffer("Area Expect for 6x6");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests11, ImageResizeArea_Test8) {

    NDArray input    = NDArrayFactory::create<int>('c', {1, 3, 3, 1}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9
    });

    NDArray expected = NDArrayFactory::create<float>('c', {1, 6, 6, 1}, {
            1.f,             1.f,            1.5f,            2.f,            2.f,            3.f,
            1.f,             1.f,            1.5f,            2.f,            2.f,            3.f,
            2.5f,            2.5f,             3.f,           3.5f,           3.5f,           4.5f,
            4.f,             4.f,            4.5f,            5.f,            5.f,            6.f,
            4.f,             4.f,            4.5f,            5.f,            5.f,            6.f,
            7.f,             7.f,            7.5f,            8.f,            8.f,            9.f
    });
    //input.linspace(1);
//    auto size = NDArrayFactory::create<int>({6, 6});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input}, {}, {6, 6}, {true});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 6x6");
//    expected.printBuffer("Area Expect for 6x6");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, ImageResizeArea_Test9) {

    NDArray input    = NDArrayFactory::create<int>('c', {1, 2, 3, 4}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24
    });

    NDArray expected = NDArrayFactory::create<float>('c', {1, 10, 10, 4}, {
            1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 3.666667f, 4.666667f, 5.666667f, 6.666667f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 6.333336f, 7.333336f, 8.333336f, 9.333337f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 8.999998f, 9.999998f, 10.999998f, 11.999998f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 3.666667f, 4.666667f, 5.666667f, 6.666667f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 6.333336f, 7.333336f, 8.333336f, 9.333337f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 8.999998f, 9.999998f, 10.999998f, 11.999998f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 3.666667f, 4.666667f, 5.666667f, 6.666667f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 6.333336f, 7.333336f, 8.333336f, 9.333337f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 8.999998f, 9.999998f, 10.999998f, 11.999998f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 3.666667f, 4.666667f, 5.666667f, 6.666667f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 6.333336f, 7.333336f, 8.333336f, 9.333337f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 8.999998f, 9.999998f, 10.999998f, 11.999998f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 3.666667f, 4.666667f, 5.666667f, 6.666667f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 6.333336f, 7.333336f, 8.333336f, 9.333336f, 8.999999f, 9.999999f, 11.000000f, 11.999999f, 8.999999f, 9.999999f, 11.000000f, 11.999999f, 8.999998f, 9.999997f, 10.999997f, 11.999997f, 13.000003f, 14.000004f, 15.000003f, 16.000004f, 13.000003f, 14.000004f, 15.000003f, 16.000004f, 13.000003f, 14.000004f, 15.000003f, 16.000004f, 15.666671f, 16.666672f, 17.666672f, 18.666672f, 17.000006f, 18.000004f, 19.000006f, 20.000004f, 17.000006f, 18.000004f, 19.000006f, 20.000004f, 18.333344f, 19.333344f, 20.333345f, 21.333344f, 21.000006f, 22.000006f, 23.000006f, 24.000006f, 21.000006f, 22.000006f, 23.000006f, 24.000006f, 21.000002f, 22.000000f, 23.000002f, 24.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 15.666667f, 16.666668f, 17.666668f, 18.666668f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 18.333340f, 19.333340f, 20.333342f, 21.333340f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 20.999996f, 21.999996f, 22.999994f, 23.999996f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 15.666667f, 16.666668f, 17.666668f, 18.666668f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 18.333340f, 19.333340f, 20.333342f, 21.333340f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 20.999996f, 21.999996f, 22.999994f, 23.999996f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 15.666667f, 16.666668f, 17.666668f, 18.666668f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 18.333340f, 19.333340f, 20.333342f, 21.333340f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 20.999996f, 21.999996f, 22.999994f, 23.999996f, 12.999995f, 13.999995f, 14.999994f, 15.999994f, 12.999995f, 13.999995f, 14.999994f, 15.999994f, 12.999995f, 13.999995f, 14.999994f, 15.999994f, 15.666661f, 16.666662f, 17.666660f, 18.666660f, 16.999994f, 17.999994f, 18.999992f, 19.999992f, 16.999994f, 17.999994f, 18.999992f, 19.999992f, 18.333334f, 19.333332f, 20.333334f, 21.333332f, 20.999992f, 21.999992f, 22.999990f, 23.999992f, 20.999992f, 21.999992f, 22.999990f, 23.999992f, 20.999989f, 21.999989f, 22.999987f, 23.999987f

    });
    //input.linspace(1);
    auto size = NDArrayFactory::create<int>({10, 10});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 10x10");
    //    expected.printBuffer("Area Expect for 6x6");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, ImageResizeArea_Test10) {

    NDArray input    = NDArrayFactory::create<int>('c', {1, 2, 3, 4}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24
    });

    NDArray expected = NDArrayFactory::create<float>('c', {1, 10, 10, 4}, {
            1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 3.666667f, 4.666667f, 5.666667f, 6.666667f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 6.333336f, 7.333336f, 8.333336f, 9.333337f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 8.999998f, 9.999998f, 10.999998f, 11.999998f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 3.666667f, 4.666667f, 5.666667f, 6.666667f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 6.333336f, 7.333336f, 8.333336f, 9.333337f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 8.999998f, 9.999998f, 10.999998f, 11.999998f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 3.666667f, 4.666667f, 5.666667f, 6.666667f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 6.333336f, 7.333336f, 8.333336f, 9.333337f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 8.999998f, 9.999998f, 10.999998f, 11.999998f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 3.666667f, 4.666667f, 5.666667f, 6.666667f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 6.333336f, 7.333336f, 8.333336f, 9.333337f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 9.000000f, 10.000000f, 11.000000f, 12.000000f, 8.999998f, 9.999998f, 10.999998f, 11.999998f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 1.000000f, 2.000000f, 3.000000f, 4.000000f, 3.666667f, 4.666667f, 5.666667f, 6.666667f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 5.000000f, 6.000000f, 7.000000f, 8.000000f, 6.333336f, 7.333336f, 8.333336f, 9.333336f, 8.999999f, 9.999999f, 11.000000f, 11.999999f, 8.999999f, 9.999999f, 11.000000f, 11.999999f, 8.999998f, 9.999997f, 10.999997f, 11.999997f, 13.000003f, 14.000004f, 15.000003f, 16.000004f, 13.000003f, 14.000004f, 15.000003f, 16.000004f, 13.000003f, 14.000004f, 15.000003f, 16.000004f, 15.666671f, 16.666672f, 17.666672f, 18.666672f, 17.000006f, 18.000004f, 19.000006f, 20.000004f, 17.000006f, 18.000004f, 19.000006f, 20.000004f, 18.333344f, 19.333344f, 20.333345f, 21.333344f, 21.000006f, 22.000006f, 23.000006f, 24.000006f, 21.000006f, 22.000006f, 23.000006f, 24.000006f, 21.000002f, 22.000000f, 23.000002f, 24.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 15.666667f, 16.666668f, 17.666668f, 18.666668f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 18.333340f, 19.333340f, 20.333342f, 21.333340f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 20.999996f, 21.999996f, 22.999994f, 23.999996f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 15.666667f, 16.666668f, 17.666668f, 18.666668f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 18.333340f, 19.333340f, 20.333342f, 21.333340f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 20.999996f, 21.999996f, 22.999994f, 23.999996f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 13.000000f, 14.000001f, 15.000000f, 16.000000f, 15.666667f, 16.666668f, 17.666668f, 18.666668f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 17.000002f, 18.000000f, 19.000002f, 20.000000f, 18.333340f, 19.333340f, 20.333342f, 21.333340f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 21.000002f, 22.000000f, 22.999998f, 24.000000f, 20.999996f, 21.999996f, 22.999994f, 23.999996f, 12.999995f, 13.999995f, 14.999994f, 15.999994f, 12.999995f, 13.999995f, 14.999994f, 15.999994f, 12.999995f, 13.999995f, 14.999994f, 15.999994f, 15.666661f, 16.666662f, 17.666660f, 18.666660f, 16.999994f, 17.999994f, 18.999992f, 19.999992f, 16.999994f, 17.999994f, 18.999992f, 19.999992f, 18.333334f, 19.333332f, 20.333334f, 21.333332f, 20.999992f, 21.999992f, 22.999990f, 23.999992f, 20.999992f, 21.999992f, 22.999990f, 23.999992f, 20.999989f, 21.999989f, 22.999987f, 23.999987f

    });
    //input.linspace(1);
    //auto size = NDArrayFactory::create<int>({10, 10});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input}, {}, {10, 10});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 10x10");
    //    expected.printBuffer("Area Expect for 6x6");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, ImageResizeArea_Test11) {

    NDArray input    = NDArrayFactory::create<int>('c', {1, 2, 3, 4}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24
    });

//    NDArray expected = NDArrayFactory::create<float>('c', {1, 6, 9, 4}, {
//            1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333336, 8.999999, 9.999999, 11.000000, 11.999999, 8.999999, 9.999999, 11.000000, 11.999999, 8.999998, 9.999997, 10.999997, 11.999997, 13.000003, 14.000004, 15.000003, 16.000004, 13.000003, 14.000004, 15.000003, 16.000004, 13.000003, 14.000004, 15.000003, 16.000004, 15.666671, 16.666672, 17.666672, 18.666672, 17.000006, 18.000004, 19.000006, 20.000004, 17.000006, 18.000004, 19.000006, 20.000004, 18.333344, 19.333344, 20.333345, 21.333344, 21.000006, 22.000006, 23.000006, 24.000006, 21.000006, 22.000006, 23.000006, 24.000006, 21.000002, 22.000000, 23.000002, 24.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 15.666667, 16.666668, 17.666668, 18.666668, 17.000002, 18.000000, 19.000002, 20.000000, 17.000002, 18.000000, 19.000002, 20.000000, 18.333340, 19.333340, 20.333342, 21.333340, 21.000002, 22.000000, 22.999998, 24.000000, 21.000002, 22.000000, 22.999998, 24.000000, 20.999996, 21.999996, 22.999994, 23.999996, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 15.666667, 16.666668, 17.666668, 18.666668, 17.000002, 18.000000, 19.000002, 20.000000, 17.000002, 18.000000, 19.000002, 20.000000, 18.333340, 19.333340, 20.333342, 21.333340, 21.000002, 22.000000, 22.999998, 24.000000, 21.000002, 22.000000, 22.999998, 24.000000, 20.999996, 21.999996, 22.999994, 23.999996, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 15.666667, 16.666668, 17.666668, 18.666668, 17.000002, 18.000000, 19.000002, 20.000000, 17.000002, 18.000000, 19.000002, 20.000000, 18.333340, 19.333340, 20.333342, 21.333340, 21.000002, 22.000000, 22.999998, 24.000000, 21.000002, 22.000000, 22.999998, 24.000000, 20.999996, 21.999996, 22.999994, 23.999996, 12.999995, 13.999995, 14.999994, 15.999994, 12.999995, 13.999995, 14.999994, 15.999994, 12.999995, 13.999995, 14.999994, 15.999994, 15.666661, 16.666662, 17.666660, 18.666660, 16.999994, 17.999994, 18.999992, 19.999992, 16.999994, 17.999994, 18.999992, 19.999992, 18.333334, 19.333332, 20.333334, 21.333332, 20.999992, 21.999992, 22.999990, 23.999992, 20.999992, 21.999992, 22.999990, 23.999992, 20.999989, 21.999989, 22.999987, 23.999987
//
//    });
    //input.linspace(1);
    //auto size = NDArrayFactory::create<int>({10, 10});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input}, {}, {6, 9});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 6x9");
    //    expected.printBuffer("Area Expect for 6x6");
//    ASSERT_TRUE(expected.isSameShape(result));
//    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, ImageResizeArea_Test12) {

    NDArray input    = NDArrayFactory::create<int>('c', {1, 2, 3, 4}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24
    });

//    NDArray expected = NDArrayFactory::create<float>('c', {1, 6, 9, 4}, {
//            1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333336, 8.999999, 9.999999, 11.000000, 11.999999, 8.999999, 9.999999, 11.000000, 11.999999, 8.999998, 9.999997, 10.999997, 11.999997, 13.000003, 14.000004, 15.000003, 16.000004, 13.000003, 14.000004, 15.000003, 16.000004, 13.000003, 14.000004, 15.000003, 16.000004, 15.666671, 16.666672, 17.666672, 18.666672, 17.000006, 18.000004, 19.000006, 20.000004, 17.000006, 18.000004, 19.000006, 20.000004, 18.333344, 19.333344, 20.333345, 21.333344, 21.000006, 22.000006, 23.000006, 24.000006, 21.000006, 22.000006, 23.000006, 24.000006, 21.000002, 22.000000, 23.000002, 24.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 15.666667, 16.666668, 17.666668, 18.666668, 17.000002, 18.000000, 19.000002, 20.000000, 17.000002, 18.000000, 19.000002, 20.000000, 18.333340, 19.333340, 20.333342, 21.333340, 21.000002, 22.000000, 22.999998, 24.000000, 21.000002, 22.000000, 22.999998, 24.000000, 20.999996, 21.999996, 22.999994, 23.999996, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 15.666667, 16.666668, 17.666668, 18.666668, 17.000002, 18.000000, 19.000002, 20.000000, 17.000002, 18.000000, 19.000002, 20.000000, 18.333340, 19.333340, 20.333342, 21.333340, 21.000002, 22.000000, 22.999998, 24.000000, 21.000002, 22.000000, 22.999998, 24.000000, 20.999996, 21.999996, 22.999994, 23.999996, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 15.666667, 16.666668, 17.666668, 18.666668, 17.000002, 18.000000, 19.000002, 20.000000, 17.000002, 18.000000, 19.000002, 20.000000, 18.333340, 19.333340, 20.333342, 21.333340, 21.000002, 22.000000, 22.999998, 24.000000, 21.000002, 22.000000, 22.999998, 24.000000, 20.999996, 21.999996, 22.999994, 23.999996, 12.999995, 13.999995, 14.999994, 15.999994, 12.999995, 13.999995, 14.999994, 15.999994, 12.999995, 13.999995, 14.999994, 15.999994, 15.666661, 16.666662, 17.666660, 18.666660, 16.999994, 17.999994, 18.999992, 19.999992, 16.999994, 17.999994, 18.999992, 19.999992, 18.333334, 19.333332, 20.333334, 21.333332, 20.999992, 21.999992, 22.999990, 23.999992, 20.999992, 21.999992, 22.999990, 23.999992, 20.999989, 21.999989, 22.999987, 23.999987
//
//    });
    //input.linspace(1);
    //auto size = NDArrayFactory::create<int>({10, 10});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input}, {}, {10, 15});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 6x9");
    //    expected.printBuffer("Area Expect for 6x6");
//    ASSERT_TRUE(expected.isSameShape(result));
//    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, ImageResizeArea_Test13) {

    NDArray input    = NDArrayFactory::create<int>('c', {1, 2, 3, 4}, {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24
    });

//    NDArray expected = NDArrayFactory::create<float>('c', {1, 8, 8, 4}, {
//            1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333337, 9.000000, 10.000000, 11.000000, 12.000000, 9.000000, 10.000000, 11.000000, 12.000000, 8.999998, 9.999998, 10.999998, 11.999998, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 1.000000, 2.000000, 3.000000, 4.000000, 3.666667, 4.666667, 5.666667, 6.666667, 5.000000, 6.000000, 7.000000, 8.000000, 5.000000, 6.000000, 7.000000, 8.000000, 6.333336, 7.333336, 8.333336, 9.333336, 8.999999, 9.999999, 11.000000, 11.999999, 8.999999, 9.999999, 11.000000, 11.999999, 8.999998, 9.999997, 10.999997, 11.999997, 13.000003, 14.000004, 15.000003, 16.000004, 13.000003, 14.000004, 15.000003, 16.000004, 13.000003, 14.000004, 15.000003, 16.000004, 15.666671, 16.666672, 17.666672, 18.666672, 17.000006, 18.000004, 19.000006, 20.000004, 17.000006, 18.000004, 19.000006, 20.000004, 18.333344, 19.333344, 20.333345, 21.333344, 21.000006, 22.000006, 23.000006, 24.000006, 21.000006, 22.000006, 23.000006, 24.000006, 21.000002, 22.000000, 23.000002, 24.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 15.666667, 16.666668, 17.666668, 18.666668, 17.000002, 18.000000, 19.000002, 20.000000, 17.000002, 18.000000, 19.000002, 20.000000, 18.333340, 19.333340, 20.333342, 21.333340, 21.000002, 22.000000, 22.999998, 24.000000, 21.000002, 22.000000, 22.999998, 24.000000, 20.999996, 21.999996, 22.999994, 23.999996, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 15.666667, 16.666668, 17.666668, 18.666668, 17.000002, 18.000000, 19.000002, 20.000000, 17.000002, 18.000000, 19.000002, 20.000000, 18.333340, 19.333340, 20.333342, 21.333340, 21.000002, 22.000000, 22.999998, 24.000000, 21.000002, 22.000000, 22.999998, 24.000000, 20.999996, 21.999996, 22.999994, 23.999996, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 13.000000, 14.000001, 15.000000, 16.000000, 15.666667, 16.666668, 17.666668, 18.666668, 17.000002, 18.000000, 19.000002, 20.000000, 17.000002, 18.000000, 19.000002, 20.000000, 18.333340, 19.333340, 20.333342, 21.333340, 21.000002, 22.000000, 22.999998, 24.000000, 21.000002, 22.000000, 22.999998, 24.000000, 20.999996, 21.999996, 22.999994, 23.999996, 12.999995, 13.999995, 14.999994, 15.999994, 12.999995, 13.999995, 14.999994, 15.999994, 12.999995, 13.999995, 14.999994, 15.999994, 15.666661, 16.666662, 17.666660, 18.666660, 16.999994, 17.999994, 18.999992, 19.999992, 16.999994, 17.999994, 18.999992, 19.999992, 18.333334, 19.333332, 20.333334, 21.333332, 20.999992, 21.999992, 22.999990, 23.999992, 20.999992, 21.999992, 22.999990, 23.999992, 20.999989, 21.999989, 22.999987, 23.999987
//
//    });
    //input.linspace(1);
    //auto size = NDArrayFactory::create<int>({10, 10});
    nd4j::ops::resize_area op;
    auto results = op.evaluate({&input}, {}, {9, 9});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printBuffer("Area Resized to 8x8");
    //    expected.printBuffer("Area Expect for 6x6");
//    ASSERT_TRUE(expected.isSameShape(result));
//    ASSERT_TRUE(expected.equalsTo(result));
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

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, Solve_Test_1) {

    auto a = NDArrayFactory::create<float>('c', {3, 3}, {
            2.f, -1.f, -2.f, -4.f, 6.f, 3.f, -4.f, -2.f, 8.f
    });

    auto b = NDArrayFactory::create<float>('c', {3, 1}, {
            2.f, 4.f, 3.f
    });

    auto exp = NDArrayFactory::create<float>('c', {3, 1}, {
            7.625f, 3.25f, 5.f
    });

    nd4j::ops::solve op;

    auto res = op.evaluate({&a, &b});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);

//    z->printIndexedBuffer("Solve of 3x3");

    ASSERT_TRUE(exp.equalsTo(z));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, Solve_Test_2) {

    auto a = NDArrayFactory::create<float>('c', {4, 4}, {
            1.f,  1.f,  1.f,  1.f,
            0.f,  1.f,  1.f,  0.f,
            0.f,  0.f,  2.f,  1.f,
            0.f,  0.f,  0.f,  3.f,
    });

    auto b = NDArrayFactory::create<float>('c', {4, 1}, {
            2.f, 4.f, 2.f, 4.f
    });

    auto exp = NDArrayFactory::create<float>('c', {4, 1}, {
            -3.3333333f,      3.6666666f,         0.333333f,        1.3333333f
    });

    nd4j::ops::solve op;

    auto res = op.evaluate({&a, &b});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);

//    z->printIndexedBuffer("Solve 4x4");

    ASSERT_TRUE(exp.equalsTo(z));
    delete res;
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, Solve_Test_3) {

    auto a = NDArrayFactory::create<float>('c', {2, 4, 4}, {
            1.f,  1.f,  1.f,  1.f,
            0.f,  1.f,  1.f,  0.f,
            0.f,  0.f,  2.f,  1.f,
            0.f,  0.f,  0.f,  3.f,

            3.f,  0.f,  0.f,  0.f,
            2.f,  1.f,  0.f,  0.f,
            1.f,  0.f,  1.f,  0.f,
            1.f,  1.f,  1.f,  1.f

    });

    auto b = NDArrayFactory::create<float>('c', {2, 4, 1}, {
            2.f, 4.f, 2.f, 4.f,
            4.f, 2.f, 4.f, 2.f
    });

    auto exp = NDArrayFactory::create<float>('c', {2, 4, 1}, {
            -3.3333333f,      3.6666666f,         0.333333f,        1.3333333f,
            1.333333f,      -0.6666667f,         2.6666667f,        -1.3333333f
    });

    nd4j::ops::solve op;

    auto res = op.evaluate({&a, &b});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);

//    z->printIndexedBuffer("Solve 4x4");

    ASSERT_TRUE(exp.equalsTo(z));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, Solve_Test_4) {

    auto a = NDArrayFactory::create<float>('c', {2, 2, 2}, {
            0.7788f,    0.8012f,   0.7244f,    0.2309f,
            0.7271f,    0.1804f,   0.5056f,    0.8925f
    });

    auto b = NDArrayFactory::create<float>('c', {2, 2, 2}, {
            0.7717f,    0.9281f, 0.9846f,    0.4838f,
            0.6433f,    0.6041f, 0.6501f,    0.7612f
    });

    auto exp = NDArrayFactory::create<float>('c', {2, 2, 2}, {
//            1.524494767f,    0.432706356f,-0.518630624f,    0.737760842f,
//            0.819143713f,    0.720401764f, 0.264349997f,    0.444699198f
             1.5245394f,   0.4326952f,  -0.51873577f,  0.7377896f,
            0.81915987f,  0.72049433f,    0.2643504f,  0.44472617f
    });

    nd4j::ops::solve op;

    auto res = op.evaluate({&a, &b});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);

//    z->printBuffer("4 Solve 4x4");
//    exp.printBuffer("4 Expec 4x4");

    ASSERT_TRUE(exp.equalsTo(z));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, Solve_Test_5) {

    auto a = NDArrayFactory::create<float>('c', {3, 3}, {
            0.7788f,    0.8012f,    0.7244f,
            0.2309f,    0.7271f,    0.1804f,
            0.5056f,    0.8925f,    0.5461f
    });

    auto b = NDArrayFactory::create<float>('c', {3, 3}, {
            0.7717f,    0.9281f,    0.9846f,
            0.4838f,    0.6433f,    0.6041f,
            0.6501f,    0.7612f,    0.7605f
    });

    auto exp = NDArrayFactory::create<float>('c', {3, 3}, {
             1.5504692f,  1.8953944f,  2.2765768f,
            0.03399149f,  0.2883001f,  0.5377323f,
            -0.8774802f, -1.2155888f, -1.8049058f
    });

    nd4j::ops::solve op;

    auto res = op.evaluate({&a, &b}, {true});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);

    z->printBuffer("4 Solve 4x4");
    exp.printBuffer("4 Expec 4x4");

    ASSERT_TRUE(exp.equalsTo(z));
    delete res;
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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {0});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {0});

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
    NDArray dLdwExp('c', {}, std::vector<double>{4515.84});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {1});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {1});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {2});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {2});

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

    NDArray dLdwExp('c', {}, std::vector<double>{0.});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {2});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {2});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3});

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

    NDArray dLdwExp('c', {1,1}, std::vector<double>{188.16});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::mean_sqerr_loss_grad op;
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3});

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
    auto result = op.evaluate({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(exp.equalsTo(result->at(0)));

    delete result;
}

TEST_F(DeclarableOpsTests11, SquaredSubtractTest_Test2) {
    auto x = NDArrayFactory::create<float>('c', {2, 4}, {0, 1, 2, 3, 0, 1, 2, 3});
    auto y = NDArrayFactory::create<float>('c',{4}, {3, 2, 1, 0});
    auto exp = NDArrayFactory::create<float>('c', {2, 4}, {9, 1,1, 9, 9, 1, 1, 9});
    nd4j::ops::squaredsubtract op;
    auto result = op.evaluate({&x, &y}, {}, {});
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
    auto result = op.evaluate({&x, &y, &eps}, {}, {});
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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {0});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {0});

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
    NDArray dLdwExp('c', {}, std::vector<double>{288.});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {1});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {1});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {2});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {2});

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

    NDArray dLdwExp('c', {}, std::vector<double>{0.});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {2});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {2});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3});

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

    NDArray dLdwExp('c', {1,1}, std::vector<double>{12.});

    predictions.linspace(0.04, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::absolute_difference_loss_grad op;
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3});

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
    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3});

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
    auto results = op.evaluate({&x, &y}, {}, {});

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
    auto results = op.evaluate({&x, &y}, {}, {});

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
    auto results = op.evaluate({&x, &y}, {}, {});

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
    auto results = op.evaluate({&logits, &weights, &labels}, {0.}, {0});

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
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {0});

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
    NDArray dLdwExp('c', {}, std::vector<double>{-91.52109});
    NDArray dLdlExp('c', {2,3,4}, {0.028,  0.014, -0., -0.014,-0.028, -0.042, -0.056, -0.07 ,-0.084, -0.098, -0.112, -0.126,
                                   -0.14 , -0.154, -0.168, -0.182,-0.196, -0.21 , -0.224, -0.238,-0.252, -0.266, -0.28 , -0.294});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {1});

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
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {1});

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
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {2});

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
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {2});

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

    NDArray dLdwExp('c', {}, std::vector<double>{0.});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {2});

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
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {2});

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
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {3});

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

    NDArray dLdwExp('c', {1,1}, std::vector<double>{-3.81338});

    logits.linspace(-0.08, 0.04);
    labels.linspace(1);
    weights.assign(0.5);

    nd4j::ops::sigm_cross_entropy_loss_grad op;
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {3});

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
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {3});

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
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {3});

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
    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {3});

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
    auto results = op.evaluate({&x, &y}, {}, {});

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
    auto results = op.evaluate({&x, &y}, {}, {});

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
    auto results = op.evaluate({&x, &y}, {}, {});

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

    auto results = op.evaluate({&logits, &weights, &labels}, {0.}, {0});

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
    NDArray dLdwExp('c', {1}, std::vector<double>{1.38629});

    logits = 2.;
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.evaluate({&logits, &weights, &labels}, {0.}, {1});

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
    NDArray weights('c', {}, std::vector<double>{0}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {4}, {0.125,  0.125, -0.375,  0.125});
    NDArray dLdwExp('c', {}, std::vector<double>{1.38629});

    logits = 2.;
    weights.assign(0.5);

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.evaluate({&logits, &weights, &labels}, {0.}, {1});

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
    NDArray weights('c', {}, std::vector<double>{0}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {4}, {0.23521,  0.2448 , -0.7452 ,  0.26519});
    NDArray dLdwExp('c', {}, std::vector<double>{0.});

    logits.linspace(-0.08, 0.04);
    weights = 0.5;

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.evaluate({&logits, &weights, &labels}, {0.}, {2});

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
    NDArray dLdwExp('c', {1}, std::vector<double>{1.36729});

    logits.linspace(-0.08, 0.04);
    weights = 0.5;

    nd4j::ops::softmax_cross_entropy_loss_grad op;

    auto results = op.evaluate({&logits, &weights, &labels}, {0.}, {3});

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

    auto results = op.evaluate({&logits, &weights, &labels}, {0.3}, {2});

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

    auto results = op.evaluate({&logits, &weights, &labels}, {0.}, {3});

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

    auto results = op.evaluate({&logits, &weights, &labels}, {0.}, {2});

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
    auto sumDiff = labels.reduceAlongDimension(reduce::Sum, {1}, true);

    NDArray numOfNonZero(sumDiff.getShapeInfo(), nd4j::DataType::INT64, false);
    numOfNonZero.assign(1);
    sumDiff.applyPairwiseTransform(pairwise::SafeDivide, numOfNonZero, sumDiff);
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test1) {

    NDArray labels('c', {2,3,4}, {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 1,0,0,0, 0,1,0,0});
    NDArray logits('c', {2,3,4}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {2,3,4}, {-0.76479, 0.2448, 0.2548, 0.26519, 0.23521,-0.7552, 0.2548, 0.26519, 0.23521, 0.2448,-0.7452, 0.26519,
                                   0.23521, 0.2448, 0.2548,-0.73481,-0.76479, 0.2448, 0.2548, 0.26519, 0.23521,-0.7552, 0.2548, 0.26519});
    logits.linspace(-0.08, 0.04);

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.evaluate({&logits, &labels}, {}, {});

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

    auto results = op.evaluate({&logits, &labels}, {}, {1});

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

    auto results = op.evaluate({&logits, &labels}, {}, {0});

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

    auto results = op.evaluate({&logits, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test5) {

    NDArray labels('c', {2,1}, std::vector<double>{1,0});
    NDArray logits('c', {2,1}, {-0.04, 0.04});

    NDArray dLdpExp('c', {2,1}, {-0.51999, 0.51999});

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.evaluate({&logits, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test6) {

    NDArray labels('c', {1,2}, {1,1.});
    NDArray logits('c', {1,2}, {-0.04, 0.04});

    NDArray dLdpExp('c', {1,2}, {0, 0.});

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.evaluate({&logits, &labels}, {}, {0});

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

    auto results = op.evaluate({&logits, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, softmaxCrossEntropyWithLogits_grad_test8) {

    NDArray labels('c', {1}, std::vector<double>{1});
    NDArray logits('c', {1}, std::vector<double>{0.04});

    NDArray dLdpExp('c', {1}, std::vector<double>{0});

    nd4j::ops::softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.evaluate({&logits, &labels}, {}, {0});

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

    auto results = op.evaluate({&x, &y, &dLdp}, {}, {});

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

    auto results = op.evaluate({&labels, &logits}, {}, {});

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

    auto results = op.evaluate({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sparseSoftmaxCrossEntropyWithLogits_grad_test3) {

    NDArray labels('c', {}, std::vector<double>{1}, nd4j::DataType::INT64);
    NDArray logits('c', {2}, {-0.2, 0.3});

    NDArray dLdpExp('c', {2}, {0.37754, -0.37754});

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.evaluate({&labels, &logits}, {}, {});

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

    auto results = op.evaluate({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests11, sparseSoftmaxCrossEntropyWithLogits_grad_test5) {

    NDArray labels('c', {1,1}, std::vector<double>({0}), nd4j::DataType::INT64);
    NDArray logits('c', {1,1,2}, {-0.3,0.2});

    NDArray dLdpExp('c', {1,1,2}, {-0.62246,  0.62246});

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits_grad op;

    auto results = op.evaluate({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *dLdp = results->at(0);

    ASSERT_TRUE(dLdpExp.isSameShape(dLdp));
    ASSERT_TRUE(dLdpExp.equalsTo(dLdp));

    delete results;
}



