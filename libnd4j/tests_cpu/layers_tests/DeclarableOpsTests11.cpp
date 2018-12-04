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

TEST_F(DeclarableOpsTests11, test_mixed_biasadd_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 3});
    auto y = NDArrayFactory::create<float>('c', {3}, {1.f, 2.f, 3.f});
    auto z = NDArrayFactory::create<float>('c', {2, 3});
    auto exp = NDArrayFactory::create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 1.f, 2.f, 3.f});

    nd4j::ops::biasadd op;
    auto status = op.execute({&x, &y}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(exp, z);
}

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
    auto results = op.execute({&predictions, &weights, &labels}, {1e-7}, {0});

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
    NDArray dLdwExp('c', {0}, {-227.77286});
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
        
    NDArray dLdwExp('c', {0}, {0.});
    
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
    NDArray dLdwExp('c', {0}, {4515.84});
    
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
        
    NDArray dLdwExp('c', {0}, {0.});
    
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
    result->at(0)->printBuffer("Output");

    delete result;
}

TEST_F(DeclarableOpsTests11, SquaredSubtractTest_Test2) {
    auto x = NDArrayFactory::create<float>('c', {2, 4}, {0, 1, 2, 3, 0, 1, 2, 3});
    auto y = NDArrayFactory::create<float>('c',{4}, {3, 2, 1, 0});
    auto exp = NDArrayFactory::create<float>('c', {2, 4}, {9, 1,1, 9, 9, 1, 1, 9});
    nd4j::ops::squaredsubtract op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());
    result->at(0)->printBuffer("Output");
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
    result->at(0)->printBuffer("Output");
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
    NDArray dLdwExp('c', {0}, {288.});
    
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
        
    NDArray dLdwExp('c', {0}, {0.});
    
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
    res->printIndexedBuffer("BFloat16 sum:");
    ASSERT_TRUE(res->equalsTo(exp));

    delete results;
}

