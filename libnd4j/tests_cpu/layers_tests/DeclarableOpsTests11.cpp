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
