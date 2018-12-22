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


class DeclarableOpsTests12 : public testing::Test {
public:

    DeclarableOpsTests12() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests12, test_any_validation_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 1}, {1.0, 2.0});
    auto y = NDArrayFactory::create<int>('c', {2}, {1, 0});

    nd4j::ops::transpose op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(x.dataType(), z->dataType());

    delete result;
}


/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, cosine_distance_loss_grad_test1) {

    NDArray labels('c', {2,4}, {0,1,1,0,1,0,1,0});
    NDArray predictions('c', {2,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,1}, nd4j::DataType::DOUBLE);
    
    NDArray dLdpExp('c', {2,4}, {-0. , -0.5, -0.5, -0., -0.5, -0. , -0.5, -0.});
    NDArray dLdwExp('c', {2,1}, {1.2, -0.2});

    predictions.linspace(-0.4, 0.2);
    weights.assign(0.5);    

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.execute({&predictions, &weights, &labels}, {}, {0, -1});
    
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
TEST_F(DeclarableOpsTests12, cosine_distance_loss_grad_test2) {

    NDArray labels('c', {2,4}, {-0.1, 0.3, 2, -1.4, 2.5, -3, 1.2, 2.2});
    NDArray predictions('c', {2,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,4}, nd4j::DataType::DOUBLE);
    
    NDArray dLdpExp('c', {2,4}, {0.05, -0.15, -1.  ,  0.7 ,-1.25,  1.5 , -0.6 , -1.1 });
    NDArray dLdwExp('c', {1,4}, {-0.04,  2.86,  0.04, -0.92});
    NDArray dLdlExp('c', {2,4}, {0.2,  0.1,  0. , -0.1, -0.2, -0.3, -0.4, -0.5});

    predictions.linspace(-0.4, 0.2);
    weights.assign(0.5);    

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.execute({&predictions, &weights, &labels}, {}, {0, 0});
    
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

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, cosine_distance_loss_grad_test3) {

    NDArray labels('c', {4}, {-0.1, 0.3, 2, -1.4});
    NDArray predictions('c', {4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1}, nd4j::DataType::DOUBLE);
    
    NDArray dLdpExp('c', {4}, {0.05, -0.15, -1.,  0.7});
    NDArray dLdwExp('c', {1}, {1.3});
    NDArray dLdlExp('c', {4}, {0.2,  0.1, -0. , -0.1});
    
    predictions.linspace(-0.4, 0.2);
    weights.assign(0.5);

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.execute({&predictions, &weights, &labels}, {}, {0, 0});
    
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

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, cosine_distance_loss_grad_test4) {

    NDArray labels('c', {1,4}, {-0.1, 0.3, 2, -1.4});
    NDArray predictions('c', {1,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {0}, nd4j::DataType::DOUBLE);
    
    NDArray dLdpExp('c', {1,4}, {0.05, -0.15, -1.,  0.7});
    NDArray dLdwExp('c', {0}, {1.3});
    NDArray dLdlExp('c', {1,4}, {0.2,  0.1, -0. , -0.1});
    
    predictions.linspace(-0.4, 0.2);
    weights.assign(0.5);

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.execute({&predictions, &weights, &labels}, {}, {1, 1});
    
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


/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, cosine_distance_loss_grad_test5) {

    NDArray labels('c', {4}, {-0.1, 0.3, 2, -1.4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,1}, nd4j::DataType::DOUBLE);
    
    NDArray dLdpExp('c', {4}, {0.1, -0.3, -2. ,  1.4});
    NDArray dLdwExp('c', {1,1}, {0.});
    NDArray dLdlExp('c', {4}, {0.4,  0.2, -0. , -0.2});

    predictions.linspace(-0.4, 0.2);
    weights = 0.5;    

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.execute({&predictions, &weights, &labels}, {}, {2, 0});
    
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

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, cosine_distance_loss_grad_test6) {

    NDArray labels('c', {4,1}, {-0.1, 0.3, 2, -1.4}, nd4j::DataType::DOUBLE);
    NDArray predictions('c', {4,1}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {4,1}, nd4j::DataType::DOUBLE);
    
    NDArray dLdpExp('c', {4,1}, {0.0125, -0.0375, -0.25  , 0.175});
    NDArray dLdwExp('c', {4,1}, {0.24 , 0.265, 0.25 , 0.32});
    NDArray dLdlExp('c', {4,1}, {0.05 , 0.025, -0.   , -0.025});

    predictions.linspace(-0.4, 0.2);
    weights = 0.5;    

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.execute({&predictions, &weights, &labels}, {}, {3, 1});
    
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

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, cosine_distance_loss_grad_test7) {

    NDArray labels('c', {2,3,4}, {-0.1, 0.3, 2, -1.4, 2.5, -3, 1.2, 2.2,-0.1, 0.3, 2, -3.4, 2.5, -3, 1.2, 2.2,-0.2, 0.3, 2, -1.4, 2.7, -3, 1.2, 4.2});
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {1,3,1}, nd4j::DataType::DOUBLE);
    
    NDArray dLdpExp('c', {2,3,4}, {0.00833, -0.025  , -0.16667,  0.11667,-0.20833,  0.25   , -0.1    , -0.18333, 0.00833, -0.025  , -0.16667,  0.28333,
                                   -0.20833,  0.25   , -0.1    , -0.18333, 0.01667, -0.025  , -0.16667,  0.11667,-0.225  ,  0.25   , -0.1    , -0.35   });
    NDArray dLdwExp('c', {1,3,1}, {0.50444, 0.89778, -1.40222});
    NDArray dLdlExp('c', {2,3,4}, {0.03333,  0.01667, -0.     , -0.01667,-0.03333, -0.05   , -0.06667, -0.08333,-0.1, -0.11667, -0.13333, -0.15,
                                   -0.16667, -0.18333, -0.2    , -0.21667,-0.23333, -0.25   , -0.26667, -0.28333,-0.3, -0.31667, -0.33333, -0.35   });

    predictions.linspace(-0.4, 0.2);
    weights = 0.5;    

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.execute({&predictions, &weights, &labels}, {}, {2, 0});
    
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

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, cosine_distance_loss_grad_test8) {

    NDArray labels('c', {2,3,4}, {-0.1, 0.3, 2, -1.4, 2.5, -3, 1.2, 2.2,-0.1, 0.3, 2, -3.4, 2.5, -3, 1.2, 2.2,-0.2, 0.3, 2, -1.4, 2.7, -3, 1.2, 4.2});
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,1,1}, nd4j::DataType::DOUBLE);
    
    NDArray dLdpExp('c', {2,3,4}, {0.00625, -0.01875, -0.125  ,  0.0875,-0.15625,  0.1875 , -0.075  , -0.1375, 0.00625, -0.01875, -0.125  ,  0.2125,
                                  -0.15625,  0.1875 , -0.075  , -0.1375, 0.0125 , -0.01875, -0.125  ,  0.0875,-0.16875,  0.1875 , -0.075  , -0.2625});
    NDArray dLdwExp('c', {2,1,1}, {0.57, -3.2175});
    NDArray dLdlExp('c', {2,3,4}, {0.025,  0.0125, -0.  , -0.0125,-0.025, -0.0375, -0.05, -0.0625,-0.075, -0.0875, -0.1 , -0.1125,
                                   -0.125, -0.1375, -0.15, -0.1625,-0.175, -0.1875, -0.2 , -0.2125,-0.225, -0.2375, -0.25, -0.2625});

    predictions.linspace(-0.4, 0.2);
    weights = 0.5;    

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.execute({&predictions, &weights, &labels}, {}, {3, 1});
    
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

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, cosine_distance_loss_grad_test9) {

    NDArray labels('c', {2,3,4}, {-0.1, 0.3, 2, -1.4, 2.5, -3, 1.2, 2.2,-0.1, 0.3, 2, -3.4, 2.5, -3, 1.2, 2.2,-0.2, 0.3, 2, -1.4, 2.7, -3, 1.2, 4.2});
    NDArray predictions('c', {2,3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {2,3,1}, nd4j::DataType::DOUBLE);
    
    NDArray dLdpExp('c', {2,3,4}, {0.05, -0.15, -1.  ,  0.7,-1.25,  1.5 , -0.6 , -1.1, 0.05, -0.15, -1.  ,  1.7,
                                    -1.25,  1.5 , -0.6 , -1.1, 0.1 , -0.15, -1.  ,  0.7,-1.35,  1.5 , -0.6 , -2.1});
    NDArray dLdwExp('c', {2,3,1}, {1.3 , -1.36,  3.62, -6.  , -0.98,-19.76});
    NDArray dLdlExp('c', {2,3,4}, {0.2,  0.1, -0. , -0.1,-0.2, -0.3, -0.4, -0.5,-0.6, -0.7, -0.8, -0.9,
                                    -1. , -1.1, -1.2, -1.3,-1.4, -1.5, -1.6, -1.7,-1.8, -1.9, -2. , -2.1});

    predictions.linspace(-0.4, 0.2);
    weights = 0.5;    

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.execute({&predictions, &weights, &labels}, {}, {0, 2});
    
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

 
/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, hinge_loss_14) {

    NDArray logits('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray weights('c', {0}, {1.});
    NDArray labels('c', {3,4}, {0,1,1,0,1,0,1,0,1,0,1,0});

    NDArray output('c', {0}, nd4j::DataType::DOUBLE);

    logits.linspace(1.);
    weights.assign(1.);    

    nd4j::ops::hinge_loss op;
    Nd4jStatus status = op.execute({&logits, &weights, &labels}, {&output}, {}, {1}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(output.e<double>(0) == 47.);
}