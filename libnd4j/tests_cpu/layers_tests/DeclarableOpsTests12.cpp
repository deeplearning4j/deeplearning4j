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

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TestDivideBP_1) {

    NDArray x('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray y = NDArrayFactory::create<double>(2.);
    NDArray eps('c', {3,4}, nd4j::DataType::DOUBLE);

    NDArray output1('c', {3, 4}, nd4j::DataType::DOUBLE);
    NDArray output2(nd4j::DataType::DOUBLE);

    x.linspace(2., 2.);
    eps.linspace(1.);

    nd4j::ops::divide_bp op;
    Nd4jStatus status = op.execute({&x, &y, &eps}, {&output1, &output2}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    output1.printIndexedBuffer("DivideBP X out");
    output2.printIndexedBuffer("DivideBP Y out");
    //ASSERT_TRUE(output.e<double>(0) == 47.);
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TestDivideBP_2) {

    NDArray x('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray y = NDArrayFactory::create<double>('c', {3,4});
    NDArray eps('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray exp1('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray exp2('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray output1('c', {3, 4}, nd4j::DataType::DOUBLE);
    NDArray output2('c', {3, 4}, nd4j::DataType::DOUBLE);
    exp1.assign(1.);
    exp2.assign(-2.);
    x.linspace(2., 2.);
    y.linspace(1.);
    eps.linspace(1.);

    nd4j::ops::divide_bp op;
    Nd4jStatus status = op.execute({&x, &y, &eps}, {&output1, &output2}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    output1.printIndexedBuffer("2DivideBP X out");
    output2.printIndexedBuffer("2DivideBP Y out");
    ASSERT_TRUE(output1.equalsTo(exp1));
    ASSERT_TRUE(output2.equalsTo(exp2));
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TestReverseDivideBP_1) {

    NDArray x('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray y = NDArrayFactory::create<double>(2.);
    NDArray eps('c', {3,4}, nd4j::DataType::DOUBLE);

    NDArray output1('c', {3, 4}, nd4j::DataType::DOUBLE);
    NDArray output2(nd4j::DataType::DOUBLE);

    x.linspace(2., 2.);
    eps.linspace(1.);

    nd4j::ops::reversedivide_bp op;
    Nd4jStatus status = op.execute({&y, &x, &eps}, {&output2, &output1}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    output1.printIndexedBuffer("RDivideBP X out");
    output2.printIndexedBuffer("RDivideBP Y out");
    //ASSERT_TRUE(output.e<double>(0) == 47.);
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TestReverseDivideBP_2) {

    NDArray x('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray y = NDArrayFactory::create<double>('c', {3,4});
    NDArray eps('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray exp1('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray exp2('c', {3,4}, nd4j::DataType::DOUBLE);

    NDArray output1('c', {3, 4}, nd4j::DataType::DOUBLE);
    NDArray output2('c', {3, 4}, nd4j::DataType::DOUBLE);

    x.linspace(2., 2.);
    y.linspace(1.);
    eps.linspace(1.);
    exp1.assign(1.);
    exp2.assign(-2.);
    nd4j::ops::reversedivide_bp op;
    Nd4jStatus status = op.execute({&y, &x, &eps}, {&output2, &output1}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    output1.printIndexedBuffer("2RDivideBP X out");
    output2.printIndexedBuffer("2RDivideBP Y out");
    ASSERT_TRUE(output1.equalsTo(exp1));
    ASSERT_TRUE(output2.equalsTo(exp2));
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TestSliceBP_1) {

    NDArray x('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray eps('c', {2,2}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {3,4}, {0., 0., 0., 0., 0., 1.,1., 0., 0., 1., 1., 0.});
    //NDArray exp2('c', {3,4}, nd4j::DataType::DOUBLE);

    NDArray output('c', {3, 4}, nd4j::DataType::DOUBLE);
    //NDArray output2('c', {3, 4}, nd4j::DataType::DOUBLE);
    output.assign(119.113);
    x.linspace(1.);
    eps.assign(1.);
    //exp1.assign(1.);
    //exp2.assign(-2.);
    nd4j::ops::slice_bp op;
    Nd4jStatus status = op.execute({&x, &eps}, {&output}, {}, {1,1,2,2}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    output.printIndexedBuffer("SLICE_BP out");
    ASSERT_TRUE(output.equalsTo(exp));
    //ASSERT_TRUE(output2.equalsTo(exp2));
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TestConfusionZero_1) {

    NDArray x('c', {2}, {1,2}, nd4j::DataType::INT64);
    NDArray i('c', {2}, {0,2}, nd4j::DataType::INT64);
    //NDArray eps('c', {2,2}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {4,4}, {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, nd4j::DataType::INT64);
    //NDArray exp2('c', {3,4}, nd4j::DataType::DOUBLE);

    NDArray output('c', {4, 4}, nd4j::DataType::INT64);
    //NDArray output2('c', {3, 4}, nd4j::DataType::DOUBLE);
    output.assign(119.113);
    x.linspace(1.);
    //eps.assign(1.);
    //exp1.assign(1.);
    //exp2.assign(-2.);
    nd4j::ops::confusion_matrix op;
    Nd4jStatus status = op.execute({&x, &i}, {&output}, {}, {4}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    output.printIndexedBuffer("Confusion out");
    ASSERT_TRUE(output.equalsTo(exp));
    //ASSERT_TRUE(output2.equalsTo(exp2));
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TestMaximumBP_1) {

    NDArray x('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray y('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray eps('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray exp1('c', {3,4}, {0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 11, 12}, nd4j::DataType::DOUBLE);
    NDArray exp2('c', {3,4}, {1, 2, 3, 4, 5, 6, 0, 0, 0,  0,  0,  0}, nd4j::DataType::DOUBLE);

    NDArray output1('c', {3, 4}, nd4j::DataType::DOUBLE);
    NDArray output2('c', {3, 4}, nd4j::DataType::DOUBLE);
    output1.assign(119);
    x.linspace(1.);
    y.linspace(12., -1.);
    x.printBuffer("X");
    y.printBuffer("Y");
    eps.linspace(1.);
    //exp1.assign(1.);
    //exp2.assign(-2.);
    nd4j::ops::maximum_bp op;
    Nd4jStatus status = op.execute({&x, &y, &eps}, {&output1, &output2}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    output1.printIndexedBuffer("X max");
    output2.printIndexedBuffer("Y max");
    ASSERT_TRUE(output1.equalsTo(exp1));
    ASSERT_TRUE(output2.equalsTo(exp2));
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TestMinimumBP_1) {

    NDArray x('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray y('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray eps('c', {3,4}, nd4j::DataType::DOUBLE);
    NDArray exp1('c', {3,4}, {0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 11, 12}, nd4j::DataType::DOUBLE);
    NDArray exp2('c', {3,4}, {1, 2, 3, 4, 5, 6, 0, 0, 0,  0,  0,  0}, nd4j::DataType::DOUBLE);

    NDArray output1('c', {3, 4}, nd4j::DataType::DOUBLE);
    NDArray output2('c', {3, 4}, nd4j::DataType::DOUBLE);
    output1.assign(119);
    x.linspace(1.);
    y.linspace(12., -1.);
    x.printBuffer("X");
    y.printBuffer("Y");
    eps.linspace(1.);
    //exp1.assign(1.);
    //exp2.assign(-2.);
    nd4j::ops::minimum_bp op;
    Nd4jStatus status = op.execute({&x, &y, &eps}, {&output2, &output1}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    output2.printIndexedBuffer("X min");
    output1.printIndexedBuffer("Y min");
    ASSERT_TRUE(output1.equalsTo(exp1));
    ASSERT_TRUE(output2.equalsTo(exp2));
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, reverse_test15) {
    
    NDArray x('c', {5}, {1,2,3,4,5}, nd4j::DataType::DOUBLE);
    NDArray axis('c', {0}, {0}, nd4j::DataType::INT32);
    NDArray z('c', {5}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {5}, {5,4,3,2,1}, nd4j::DataType::DOUBLE);
    

    nd4j::ops::reverse op;
    // auto result = op.execute({&x, &axis}, {}, {1}, {});
    Nd4jStatus status = op.execute({&x, &axis}, {&z}, {}, {1}, {});    
    // auto z = result->at(0);
    // z->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));    
    // delete result;
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, mirrorPad_test17) {
    
    NDArray x('c', {2,3}, {1,2,3,4,5,6}, nd4j::DataType::DOUBLE);
    NDArray padding('c', {2,2}, {1,1,2,2}, nd4j::DataType::INT32);
    NDArray z('c', {4,7}, nd4j::DataType::DOUBLE);
    NDArray exp1('c', {4,7}, {6, 5, 4, 5, 6, 5, 4,3, 2, 1, 2, 3, 2, 1,6, 5, 4, 5, 6, 5, 4,3, 2, 1, 2, 3, 2, 1}, nd4j::DataType::DOUBLE);
    NDArray exp2('c', {4,7}, {2, 1, 1, 2, 3, 3, 2,2, 1, 1, 2, 3, 3, 2,5, 4, 4, 5, 6, 6, 5,5, 4, 4, 5, 6, 6, 5}, nd4j::DataType::DOUBLE);
    
    nd4j::ops::mirror_pad op;    
    Nd4jStatus status = op.execute({&x, &padding}, {&z}, {}, {0}, {});      // reflect
    
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(exp1.isSameShape(z));
    ASSERT_TRUE(exp1.equalsTo(z));

    z = 0.;
    status = op.execute({&x, &padding}, {&z}, {}, {1}, {});                 // symmetric

    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(exp2.isSameShape(z));
    ASSERT_TRUE(exp2.equalsTo(z));    
}

/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, mirrorPad_test18) {
    
    NDArray x('c', {3}, {1,2,3}, nd4j::DataType::DOUBLE);
    NDArray padding('c', {2}, {1,1}, nd4j::DataType::INT32);
    NDArray z('c', {5}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {5}, {2,1,2,3,2}, nd4j::DataType::DOUBLE);
        
    nd4j::ops::mirror_pad op;    
    Nd4jStatus status = op.execute({&x, &padding}, {&z}, {}, {0}, {});      // reflect    
    
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));    
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests26) {

    NDArray input('c', {5}, {0.778786, 0.801198, 0.724375, 0.230894, 0.727141}, nd4j::DataType::FLOAT32);
    NDArray paddings('c', {1,2}, {1,1}, nd4j::DataType::INT32);
    NDArray expected('c', {7}, {10., 0.778786, 0.801198, 0.724375, 0.230894, 0.727141, 10.}, nd4j::DataType::FLOAT32);    
    NDArray z('c', {7}, nd4j::DataType::FLOAT32);    

    nd4j::ops::pad op;    
    Nd4jStatus status = op.execute({&input, &paddings}, {&z}, {10}, {0}, {});      // constant 

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.isSameShapeStrict(&z));
    ASSERT_TRUE(expected.equalsTo(z));
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, relu_1) {

    NDArray input('c', {1,5,5,6}, { 0.557449, 0.768277, 1.094015, -0.557449, -0.768277, -1.094015,0.563735, 0.900299, 0.789979, -0.563735, -0.900299, -0.789979,
                                    0.142528, 0.959611, 0.877506, -0.142528, -0.959611, -0.877506,0.448742, 0.995377, 1.171543, -0.448742, -0.995377, -1.171543,
                                    0.603772, 0.799391, 0.560310, -0.603772, -0.799391, -0.560310,0.529753, 0.906786, 0.737630, -0.529753, -0.906786, -0.737630,
                                    0.221464, 0.824996, 0.472221, -0.221464, -0.824996, -0.472221,0.427730, 0.397933, 0.714365, -0.427730, -0.397933, -0.714365,
                                    0.488365, 1.016589, 0.744197, -0.488365, -1.016589, -0.744197,0.789846, 0.940837, 0.838412, -0.789846, -0.940837, -0.838412,
                                    0.404485, 0.677328, 0.754997, -0.404485, -0.677328, -0.754997,0.436760, 0.794765, 0.729766, -0.436760, -0.794765, -0.729766,
                                    0.588081, 0.652226, 0.725522, -0.588081, -0.652226, -0.725522,0.374457, 1.225813, 1.053411, -0.374457, -1.225813, -1.053411,
                                    0.300958, 0.599417, 0.633234, -0.300958, -0.599417, -0.633234,0.241993, 1.025464, 0.695378, -0.241993, -1.025464, -0.695378,
                                    0.236289, 0.907919, 1.012100, -0.236289, -0.907919, -1.012100,0.627402, 0.565187, 0.766926, -0.627402, -0.565187, -0.766926,
                                    0.133276, 0.326284, 0.102804, -0.133276, -0.326284, -0.102804,0.426913, 0.256251, 0.305241, -0.426913, -0.256251, -0.305241,
                                    0.177977, 0.841799, 0.800615, -0.177977, -0.841799, -0.800615,0.001991, 0.518389, 0.439322, -0.001991, -0.518389, -0.439322,
                                    0.166846, 0.508224, 0.486687, -0.166846, -0.508224, -0.486687,0.167493, 0.930932, 0.868717, -0.167493, -0.930932, -0.868717,
                                    0.174864, 0.444607, 0.445000, -0.174864, -0.444607, -0.445000},  nd4j::DataType::FLOAT32);
    
    NDArray expected('c', {1,5,5,6}, { 0.557449, 0.768277, 1.094015, 0., 0., 0., 0.563735, 0.900299, 0.789979, 0., 0., 0.,
                                0.142528, 0.959611, 0.877506, 0., 0., 0., 0.448742, 0.995377, 1.171543, 0., 0., 0.,
                                0.603772, 0.799391, 0.560310, 0., 0., 0., 0.529753, 0.906786, 0.737630, 0., 0., 0.,
                                0.221464, 0.824996, 0.472221, 0., 0., 0., 0.427730, 0.397933, 0.714365, 0., 0., 0.,
                                0.488365, 1.016589, 0.744197, 0., 0., 0., 0.789846, 0.940837, 0.838412, 0., 0., 0.,
                                0.404485, 0.677328, 0.754997, 0., 0., 0., 0.436760, 0.794765, 0.729766, 0., 0., 0.,
                                0.588081, 0.652226, 0.725522, 0., 0., 0., 0.374457, 1.225813, 1.053411, 0., 0., 0.,
                                0.300958, 0.599417, 0.633234, 0., 0., 0., 0.241993, 1.025464, 0.695378, 0., 0., 0.,
                                0.236289, 0.907919, 1.012100, 0., 0., 0., 0.627402, 0.565187, 0.766926, 0., 0., 0.,
                                0.133276, 0.326284, 0.102804, 0., 0., 0., 0.426913, 0.256251, 0.305241, 0., 0., 0.,
                                0.177977, 0.841799, 0.800615, 0., 0., 0., 0.001991, 0.518389, 0.439322, 0., 0., 0.,
                                0.166846, 0.508224, 0.486687, 0., 0., 0., 0.167493, 0.930932, 0.868717, 0., 0., 0.,
                                0.174864, 0.444607, 0.445000, 0., 0., 0.},  nd4j::DataType::FLOAT32);

    NDArray z('c', {1,5,5,6}, nd4j::DataType::FLOAT32);    

    nd4j::ops::relu op;    
    Nd4jStatus status = op.execute({&input}, {&z}, {0}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.isSameShapeStrict(&z));
    ASSERT_TRUE(expected.equalsTo(z));    
}

////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, scatterND_5) {    
        
    NDArray indices('c', {4, 1}, {1, 1, 1, 1}, nd4j::DataType::INT32);
    auto updates = NDArrayFactory::create<float>('c', {4}, {1.f, 2.f, 3.f, 4.f});
    auto shape = NDArrayFactory::create<int>('c', {1}, {8});
    auto exp = NDArrayFactory::create<float>('c', {8}, {0.f, 10.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    
    nd4j::ops::scatter_nd op;
    auto result = op.execute({&indices, &updates, &shape}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0); 

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, scatter_add_8) {

    NDArray input('c', {8}, {1,1,1,1,1,1,1,1}, nd4j::DataType::FLOAT32);    
    NDArray indices('c', {4}, {1, 1, 1, 1}, nd4j::DataType::INT32);
    NDArray updates('c', {4}, {1,2,3,4}, nd4j::DataType::FLOAT32);
    NDArray expected('c', {8}, {1.f, 11.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}, nd4j::DataType::FLOAT32);

    NDArray z('c', {8}, nd4j::DataType::FLOAT32);    

    nd4j::ops::scatter_add op;
    Nd4jStatus status = op.execute({&input, &indices, &updates}, {&z}, {}, {}, {true});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.isSameShapeStrict(&z));
    ASSERT_TRUE(expected.equalsTo(z));
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, gather_6) {
    
    NDArray input('c', {3,5}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, nd4j::DataType::FLOAT32);    
    NDArray indices('c', {1}, {2}, nd4j::DataType::INT32);
    NDArray expected('c', {1,5}, {11, 12, 13, 14, 15.}, nd4j::DataType::FLOAT32);

    nd4j::ops::gather op;

    auto result = op.execute({&input, &indices}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto* output = result->at(0);
    output->printShapeInfo();
    output->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

#include "ops/declarable/helpers/multiUnique.h"
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, multiUnique_1) {

    NDArray input1('c', {3,5}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, nd4j::DataType::INT32);
    NDArray input2('c', {3,4}, {1,2,3,4,5,6,7,8,9,10,11,12}, nd4j::DataType::INT32);
    NDArray input3('c', {2,3}, {10,11,12,13,14,15}, nd4j::DataType::INT32);
    NDArray input4('c', {1,5}, {7,8,9,10,11}, nd4j::DataType::INT32);
    NDArray input5('c', {5,3}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, nd4j::DataType::INT32);

    //NDArray indices('c', {1}, {2}, nd4j::DataType::INT32);
    //NDArray expected('c', {1,5}, {11, 12, 13, 14, 15.}, nd4j::DataType::FLOAT32);

    std::vector<NDArray*> arrayList({&input1, &input2, &input3, &input4, &input5});

    ASSERT_FALSE(nd4j::ops::helpers::multiUnique(arrayList));
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, multiUnique_2) {

    NDArray input1('c', {3,5}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, nd4j::DataType::INT32);
    NDArray input2('c', {3,4}, {21,22,23,24,25,26,27,28,29,210,211,212}, nd4j::DataType::INT32);
    NDArray input3('c', {2,3}, {310,311,312,313,314,315}, nd4j::DataType::INT32);
    NDArray input4('c', {1,5}, {47,48,49,410,411}, nd4j::DataType::INT32);
    NDArray input5('c', {5,3}, {51,52,53,54,55,56,57,58,59,510,511,512,513,514,515}, nd4j::DataType::INT32);

    //NDArray indices('c', {1}, {2}, nd4j::DataType::INT32);
    //NDArray expected('c', {1,5}, {11, 12, 13, 14, 15.}, nd4j::DataType::FLOAT32);

    std::vector<NDArray*> arrayList({&input1, &input2, &input3, &input4, &input5});
    ASSERT_TRUE(nd4j::ops::helpers::multiUnique(arrayList));
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, tensormmul_6) {
    
    NDArray x('c', {1}, {2});
    NDArray y('c', {2,1,2}, {1,2,3,4});
    NDArray exp('c', {2,2}, {2,4,6,8}, nd4j::DataType::FLOAT32);
                                             
    nd4j::ops::tensormmul op;
    auto results = op.execute({&x, &y}, {}, {1,0, 1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);    
    exp.printShapeInfo();
    result->printShapeInfo();
    result->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, concat_test10) {

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
TEST_F(DeclarableOpsTests12, concat_14) {

    NDArray x0('c', {1,6}, {1,2,3,4,5,6});
    NDArray x1('c', {1,6}, {7,8,9,10,11,12});
    NDArray output('f', {2,6}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {2,6}, {1,2,3,4,5,6,7,8,9,10,11,12});
    
    nd4j::ops::concat op;

    auto status = op.execute({&x0, &x1}, {&output}, {}, {0}, {});
    ASSERT_EQ(ND4J_STATUS_OK, status);
    // output.printBuffer();
    // output.printIndexedBuffer();

    ASSERT_TRUE(exp.equalsTo(output));
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, concat_15) {

    NDArray x0('c', {1,4}, {1,2,3,4});
    NDArray x1('c', {1,4}, {5,6,7,8});
    NDArray output('c', {2,4}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {2,4}, {1,2,3,4,5,6,7,8});
    
    nd4j::ops::concat op;

    auto status = op.execute({&x0, &x1}, {&output}, {}, {0}, {});
    ASSERT_EQ(ND4J_STATUS_OK, status);
    // output.printBuffer();
    // output.printIndexedBuffer();

    ASSERT_TRUE(exp.equalsTo(output));
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, reduceMeanBp_4) {

    NDArray x('c', {3,5}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    NDArray grad0('c', {5}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {3,5}, nd4j::DataType::DOUBLE);

    grad0 = 1.;
    exp = 0.333333;
                   
    nd4j::ops::reduce_mean_bp op;
    auto result = op.execute({&x, &grad0}, {}, {0});
    auto output = result->at(0);    

    // output->printShapeInfo();
    // output->printIndexedBuffer();
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, reduceMeanBp_5) {

    NDArray x('c', {3,5}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    NDArray grad0('c', {3}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {3,5}, nd4j::DataType::DOUBLE);

    grad0 = 1.;
    exp = 0.2;
                   
    nd4j::ops::reduce_mean_bp op;
    auto result = op.execute({&x, &grad0}, {}, {1});
    auto output = result->at(0);    

    // output->printShapeInfo();
    // output->printIndexedBuffer();
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, reduceSqnormBp_1) {

    NDArray x('c', {8,6,4}, nd4j::DataType::DOUBLE);
    NDArray grad0('c', {8,6,1}, nd4j::DataType::DOUBLE);
                   
    nd4j::ops::reduce_sqnorm_bp op;
    auto result = op.execute({&x, &grad0}, {1}, {2});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, cumsum_1) {
    
    NDArray x('f', {3, 4}, nd4j::DataType::FLOAT32);

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printShapeInfo();
    // x.printShapeInfo();

    ASSERT_TRUE(z->ews() == 1);
    ASSERT_TRUE(x.ews() == 1);

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pullRows_1) {
    
    NDArray x('c', {5, 1}, {0,1,2,3,4});
    NDArray z('c', {4, 1}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {4, 1}, {0,2,3,4});

    Nd4jLong indexes[] = {0,2,3,4};

    std::vector<int> dims = {1};

    auto xTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x.getShapeInfo(), dims);
    auto zTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(z.getShapeInfo(), dims);
 
    NativeOps op;
    op.pullRows(nullptr, x.buffer(), x.getShapeInfo(), nullptr, nullptr,
                         z.buffer(), z.getShapeInfo(), nullptr, nullptr,
                         4, indexes,
                         xTadPack.primaryShapeInfo(), xTadPack.primaryOffsets(),
                         zTadPack.primaryShapeInfo(), zTadPack.primaryOffsets());
 
    ASSERT_TRUE(z.equalsTo(exp));    
}
