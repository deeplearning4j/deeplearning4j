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
#include <ConstantTadHelper.h>
#include <helpers/PointersManager.h>

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
    auto result = op.evaluate({&x, &y});
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

    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {0, -1});

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

    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {0, 0});

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
    NDArray dLdwExp('c', {1}, std::vector<double>{1.3});
    NDArray dLdlExp('c', {4}, {0.2,  0.1, -0. , -0.1});

    predictions.linspace(-0.4, 0.2);
    weights.assign(0.5);

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {0, 0});

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
    NDArray weights('c', {}, std::vector<double>{0.}, nd4j::DataType::DOUBLE);

    NDArray dLdpExp('c', {1,4}, {0.05, -0.15, -1.,  0.7});
    NDArray dLdwExp('c', {}, std::vector<double>{1.3});
    NDArray dLdlExp('c', {1,4}, {0.2,  0.1, -0. , -0.1});

    predictions.linspace(-0.4, 0.2);
    weights.assign(0.5);

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {1, 1});

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
    NDArray dLdwExp('c', {1,1}, std::vector<double>{0.});
    NDArray dLdlExp('c', {4}, {0.4,  0.2, -0. , -0.2});

    predictions.linspace(-0.4, 0.2);
    weights = 0.5;

    nd4j::ops::cosine_distance_loss_grad op;

    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {2, 0});

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

    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3, 1});

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

    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {2, 0});

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

    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {3, 1});

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

    auto results = op.evaluate({&predictions, &weights, &labels}, {}, {0, 2});

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
    NDArray weights('c', {}, std::vector<double>{1.});
    NDArray labels('c', {3,4}, {0,1,1,0,1,0,1,0,1,0,1,0});

    NDArray output('c', {}, std::vector<double>{0.}, nd4j::DataType::DOUBLE);

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
    Nd4jStatus status = op.execute({&x, &y, &eps}, std::vector<NDArray*>{&output1, &output2}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
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
    Nd4jStatus status = op.execute({&y, &x, &eps}, std::vector<NDArray*>{&output2, &output1}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
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
    Nd4jStatus status = op.execute({&y, &x, &eps}, std::vector<NDArray*>{&output2, &output1}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
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
    eps.linspace(1.);
    //exp1.assign(1.);
    //exp2.assign(-2.);
    nd4j::ops::maximum_bp op;
    Nd4jStatus status = op.execute({&x, &y, &eps}, std::vector<NDArray*>{&output1, &output2}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
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
    eps.linspace(1.);
    //exp1.assign(1.);
    //exp2.assign(-2.);
    nd4j::ops::minimum_bp op;
    Nd4jStatus status = op.execute({&x, &y, &eps}, std::vector<NDArray*>{&output2, &output1}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(output1.equalsTo(exp1));
    ASSERT_TRUE(output2.equalsTo(exp2));
}


/////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, reverse_test15) {

    NDArray x('c', {5}, {1,2,3,4,5}, nd4j::DataType::DOUBLE);
    NDArray axis('c', {}, std::vector<double>{0}, nd4j::DataType::INT32);
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
    NDArray padding('c', {2,2}, {1,1,2,2}, nd4j::DataType::INT64);
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
    NDArray padding('c', {1, 2}, {1,1}, nd4j::DataType::INT32);
    NDArray z('c', {5}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {5}, {2,1,2,3,2}, nd4j::DataType::DOUBLE);

    nd4j::ops::mirror_pad op;
    Nd4jStatus status = op.execute({&x, &padding}, {&z}, {}, {0}, {});      // reflect

    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
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
    ASSERT_TRUE(expected.isSameShapeStrict(z));
    ASSERT_TRUE(expected.equalsTo(z));
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

    NDArray x('c', {1}, std::vector<double>{2}, nd4j::DataType::FLOAT32);
    NDArray y('c', {2,1,2}, {1,2,3,4}, nd4j::DataType::FLOAT32);
    NDArray exp('c', {2,2}, {2,4,6,8}, nd4j::DataType::FLOAT32);

    nd4j::ops::tensormmul op;
    auto results = op.evaluate({&x, &y}, {}, {1,0, 1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // exp.printShapeInfo();
    // result->printShapeInfo();
    // result->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, reduceMeanBp_4) {

    NDArray x('c', {3,5}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    NDArray gradO('c', {5}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {3,5}, nd4j::DataType::DOUBLE);

    gradO = 1.;
    exp = 0.333333;

    nd4j::ops::reduce_mean_bp op;
    auto result = op.evaluate({&x, &gradO}, {}, {0});
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
    NDArray gradO('c', {3}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {3,5}, nd4j::DataType::DOUBLE);

    gradO = 1.;
    exp = 0.2;

    nd4j::ops::reduce_mean_bp op;
    auto result = op.evaluate({&x, &gradO}, {}, {1});
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
    NDArray gradO('c', {8,6,1}, nd4j::DataType::DOUBLE);

    nd4j::ops::reduce_sqnorm_bp op;
    auto result = op.evaluate({&x, &gradO}, {1}, {2});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pullRows_1) {

    NDArray x('c', {5, 1}, {0,1,2,3,4});
    NDArray z('c', {4, 1}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {4, 1}, {0,2,3,4});

    Nd4jLong indexes[] = {0,2,3,4};
    PointersManager pm(LaunchContext::defaultContext(), "pullRows");
    auto pidx = reinterpret_cast<Nd4jLong *>(pm.replicatePointer(indexes, 4 * sizeof(Nd4jLong)));

    std::vector<int> dims = {1};

    auto xTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x.getShapeInfo(), dims);
    auto zTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(z.getShapeInfo(), dims);

    Nd4jPointer nativeStart[2];

#ifdef __CUDABLAS__
    nativeStart[1] = (x.getContext()->getCudaStream());
#endif
    OpaqueDataBuffer xBuf(x.dataBuffer());
    OpaqueDataBuffer zBuf(z.dataBuffer());
    pullRows(nativeStart, &xBuf, x.getShapeInfo(), x.getSpecialShapeInfo(),
                         &zBuf, z.getShapeInfo(), z.specialShapeInfo(),
                         4, pidx,
                         xTadPack.platformShapeInfo(), xTadPack.platformOffsets(),
                         zTadPack.platformShapeInfo(), zTadPack.platformOffsets());

    ASSERT_TRUE(z.equalsTo(exp));
    pm.synchronize();
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pullRows_2) {

    NDArray arr('f', {5, 2}, {0,1,2,3,4,5,6,7,8,9});
    NDArray* y = new NDArray(arr.dup('c'));
    NDArray x = (*y)({0,0, 0,1}, true);     // view, points on first column of y, shape is {5,1}

    NDArray z('c', {4, 1}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {4, 1}, {0,2,3,4});

    Nd4jLong indexes[] = {0,2,3,4};
    PointersManager pm(LaunchContext::defaultContext(), "pullRows");
    auto pidx = reinterpret_cast<Nd4jLong *>(pm.replicatePointer(indexes, 4 * sizeof(Nd4jLong)));

    std::vector<int> dims = {1};

    auto xTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x.getShapeInfo(), dims);
    auto zTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(z.getShapeInfo(), dims);

    Nd4jPointer nativeStart[2];
#ifdef __CUDABLAS__
    nativeStart[1] = (x.getContext()->getCudaStream());
#endif
    OpaqueDataBuffer xBuf(x.dataBuffer());
    OpaqueDataBuffer zBuf(z.dataBuffer());
    pullRows(nativeStart, &xBuf, x.getShapeInfo(), x.specialShapeInfo(),
                         &zBuf, z.getShapeInfo(), z.specialShapeInfo(),
                         4, pidx,
                         xTadPack.platformShapeInfo(), xTadPack.platformOffsets(),
                         zTadPack.platformShapeInfo(), zTadPack.platformOffsets());

    ASSERT_TRUE(z.equalsTo(exp));
    pm.synchronize();
    delete y;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, softmax_9) {
    NDArray  arrC('c', {5,2}, {-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1}, nd4j::DataType::FLOAT32);
    NDArray* arrF = new NDArray(arrC.dup('f'));

    NDArray  outCC('c', {5,2}, nd4j::DataType::FLOAT32);
    NDArray  outCF('f', {5,2}, nd4j::DataType::FLOAT32);
    NDArray  outFC('c', {5,2}, nd4j::DataType::FLOAT32);
    NDArray  outFF('c', {5,2}, nd4j::DataType::FLOAT32);

    nd4j::ops::softmax op;
    auto status1 = op.execute({&arrC}, {&outCC}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, status1);
    auto status2 = op.execute({&arrC}, {&outCF}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, status2);
    auto status3 = op.execute({arrF}, {&outFC}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, status3);
    auto status4 = op.execute({arrF}, {&outFF}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, status4);

    // outCC.printIndexedBuffer("\n");
    // outCF.printIndexedBuffer("\n");
    // outFC.printIndexedBuffer("\n");
    // outFF.printIndexedBuffer("\n");

    ASSERT_EQ(outCC, outCF);
    ASSERT_EQ(outCC, outFC);
    ASSERT_EQ(outCC, outFF);

    delete arrF;
}

TEST_F(DeclarableOpsTests12, maxpool_bp_half_1) {
    auto x = NDArrayFactory::create<bfloat16>('c', {2, 3, 10, 1}, {0.2019043f, 0.6464844f, 0.9116211f, 0.60058594f, 0.34033203f, 0.7036133f, 0.6772461f, 0.3815918f, 0.87353516f, 0.04650879f, 0.67822266f, 0.8618164f, 0.88378906f, 0.7573242f, 0.66796875f, 0.63427734f, 0.33764648f, 0.46923828f, 0.62939453f, 0.76464844f, -0.8618164f, -0.94873047f, -0.9902344f, -0.88916016f, -0.86572266f, -0.92089844f, -0.90722656f, -0.96533203f, -0.97509766f, -0.4975586f, -0.84814453f, -0.984375f, -0.98828125f, -0.95458984f, -0.9472656f, -0.91064453f, -0.80859375f, -0.83496094f, -0.9140625f, -0.82470703f, 0.4802246f, 0.45361328f, 0.28125f, 0.28320312f, 0.79345703f, 0.44604492f, -0.30273438f, 0.11730957f, 0.56396484f, 0.73583984f, 0.1418457f, -0.44848633f, 0.6923828f, -0.40234375f, 0.40185547f, 0.48632812f, 0.14538574f, 0.4638672f, 0.13000488f, 0.5058594f});
    auto y = NDArrayFactory::create<bfloat16>('c', {2, 3, 10, 1}, {0.0f, -0.13391113f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.1751709f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.51904297f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5107422f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    auto z = NDArrayFactory::create<bfloat16>('c', {2, 3, 10, 1});

    nd4j::ops::maxpool2d_bp op;
    Context ctx(1);
    Nd4jLong iArgs[] = {5,1,1, 2,2,0, 1,1,1, 0,0};
    ctx.setIArguments(iArgs, 11);
    ctx.setInputArray(0, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo());
    ctx.setInputArray(1, y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo());
    ctx.setOutputArray(0, z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo());


    auto status = op.execute(&ctx);
    ASSERT_EQ(Status::OK(), status);

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_bp_1) {

    NDArray input('c', {2,3,4,10});
    NDArray gradO('c', {2,3,4,10});
    NDArray exp('c', {2,3,4,10}, {1.00438418e-02,  5.25184907e-03,  1.78685773e-03, -1.14537543e-03, -4.00071684e-03, -5.31899510e-03, -4.97647980e-03, -4.42161644e-03, -3.95395281e-03, -3.59310722e-03, 2.91823584e-04, -2.18498681e-05, -3.12092161e-04, -6.07360795e-04, -9.36298165e-04,
                                -1.02553482e-03, -7.91735307e-04, -6.15672267e-04, -4.71792649e-04, -3.42114770e-04, 4.29357824e-05, -5.46473675e-05, -1.48361753e-04, -2.47166492e-04, -3.61090642e-04, -3.81607766e-04, -2.89086485e-04, -2.17203109e-04, -1.56231865e-04, -9.91634734e-05,
                                8.99407951e-06, -3.76849275e-05, -8.32021178e-05, -1.31939698e-04, -1.89008832e-04, -1.96661276e-04, -1.47534331e-04, -1.08789405e-04, -7.53896020e-05, -4.36357586e-05,
                                1.23124300e-06, -2.60028974e-05, -5.27824741e-05, -8.17063192e-05, -1.15871291e-04, -1.19515295e-04, -8.91248055e-05, -6.49499125e-05, -4.39216528e-05, -2.37579407e-05, -9.34046056e-07, -1.87477999e-05, -3.63574763e-05, -5.54830040e-05, -7.82010393e-05,
                                -8.02115537e-05, -5.95739621e-05, -4.30659420e-05, -2.86241393e-05, -1.47010251e-05, -1.52835810e-06, -1.40790498e-05, -2.65316012e-05, -4.01083526e-05, -5.62983550e-05, -5.75223821e-05, -4.25982689e-05, -3.06141737e-05, -2.00884024e-05, -9.90276021e-06,
                                -1.61666367e-06, -1.09328157e-05, -2.02010433e-05, -3.03347279e-05, -4.24536738e-05, -4.32532870e-05, -3.19610226e-05, -2.28673853e-05, -1.48570880e-05, -7.08444895e-06,
                                -1.53552355e-06, -8.72318924e-06, -1.58886232e-05, -2.37402273e-05, -3.31507035e-05, -3.37014644e-05, -2.48602537e-05, -1.77248403e-05, -1.14254890e-05, -5.30027773e-06, -1.40318230e-06, -7.11624580e-06, -1.28209140e-05, -1.90826468e-05, -2.66006646e-05,
                                -2.69959855e-05, -1.98865000e-05, -1.41387427e-05, -9.05554589e-06, -4.10473058e-06, -1.26330860e-06, -5.91293519e-06, -1.05618501e-05, -1.56718652e-05, -2.18157675e-05, -2.21090413e-05, -1.62681827e-05, -1.15394150e-05, -7.35144840e-06, -3.26711961e-06,
                                -1.13179840e-06, -4.98940426e-06, -8.85062400e-06, -1.30997241e-05, -1.82144904e-05, -1.84380206e-05, -1.35542105e-05, -9.59566933e-06, -6.08572736e-06, -2.65887866e-06,
                                -1.01367493e-06, -4.26561428e-06, -7.52358210e-06, -1.11123145e-05, -1.54364170e-05, -1.56106762e-05, -1.14666063e-05, -8.10436813e-06, -5.12021325e-06, -2.20401580e-06, -9.09635219e-07, -3.68808492e-06, -6.47385696e-06, -9.54499774e-06, -1.32485484e-05,
                                -1.33870126e-05, -9.82651000e-06, -6.93532820e-06, -4.36710525e-06, -1.85539375e-06, -8.18735487e-07, -3.22003825e-06, -5.62928972e-06, -8.28724023e-06, -1.14948289e-05, -1.16066676e-05, -8.51461300e-06, -6.00201292e-06, -3.76846447e-06, -1.58258263e-06,
                                -7.39498375e-07, -2.83553072e-06, -4.93973403e-06, -7.26259532e-06, -1.00675643e-05, -1.01591886e-05, -7.44886802e-06, -5.24508141e-06, -3.28481428e-06, -1.36524977e-06,
                                -6.70378654e-07, -2.51585061e-06, -4.36947221e-06, -6.41683391e-06, -8.89049170e-06, -8.96649362e-06, -6.57134478e-06, -4.62275193e-06, -2.88851857e-06, -1.18941352e-06, -6.09944266e-07, -2.24723408e-06, -3.89250545e-06, -5.71062310e-06, -7.90838203e-06,
                                -7.97212033e-06, -5.84020108e-06, -4.10491293e-06, -2.55976192e-06, -1.04521314e-06, -5.56935277e-07, -2.01937837e-06, -3.48954882e-06, -5.11487451e-06, -7.08044308e-06, -7.13442114e-06, -5.22460778e-06, -3.66942504e-06, -2.28403951e-06, -9.25535005e-07,
                                -5.10270809e-07, -1.82444705e-06, -3.14605040e-06, -4.60769843e-06, -6.37601988e-06, -6.42213308e-06, -4.70144141e-06, -3.29971408e-06, -2.05053857e-06, -8.25151346e-07,
                                -4.69036365e-07, -1.65639949e-06, -2.85086708e-06, -4.17237243e-06, -5.77171340e-06, -5.81141694e-06, -4.25308644e-06, -2.98317354e-06, -1.85106614e-06, -7.40148607e-07, -4.32460268e-07, -1.51051631e-06, -2.59534818e-06, -3.79594053e-06, -5.24941379e-06,
                                -5.28384317e-06, -3.86593183e-06, -2.71007866e-06, -1.67932183e-06, -6.67554332e-07, -3.99893480e-07, -1.38306928e-06, -2.37269478e-06, -3.46823890e-06, -4.79492701e-06, -4.82497671e-06, -3.52932648e-06, -2.47282924e-06, -1.53039912e-06, -6.05077048e-07,
                                -3.70789934e-07, -1.27108103e-06, -2.17750403e-06, -3.18120783e-06, -4.39700398e-06, -4.42338614e-06, -3.23483960e-06, -2.26541715e-06, -1.40042869e-06, -5.50929371e-07});
    input.linspace(1);
    gradO = 1;

    nd4j::ops::lrn_bp op;

    auto results = op.evaluate({&input, &gradO}, {1., 1., 1}, {5});
    auto gradI = results->at(0);

    ASSERT_EQ(*gradI, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_bp_2) {

    NDArray input('c', {2,3,4,10});
    NDArray gradO('c', {2,3,4,10});
    NDArray exp('c', {2,3,4,10}, {-1.06179598e-03, -2.70050880e-03, -4.02126182e-03, -2.58826977e-03, -2.16024881e-03, -2.20575323e-03, -2.75954953e-03, -4.42477595e-03, -2.89176637e-03, -9.46942251e-04, -1.32603094e-03, -3.34868953e-03, -4.98152524e-03, -3.21313459e-03, -2.68880837e-03, -2.75207381e-03, -3.45109636e-03, -5.54159656e-03, -3.61320702e-03, -1.16457068e-03,
                    -1.70158676e-03, -4.26037982e-03, -6.33032294e-03, -4.09416296e-03, -3.43742501e-03, -3.52900685e-03, -4.43827361e-03, -7.13911094e-03, -4.64041065e-03, -1.46419462e-03, -2.26016506e-03, -5.59943309e-03, -8.30824208e-03, -5.39253885e-03, -4.54709725e-03, -4.68666852e-03, -5.91615774e-03, -9.53640230e-03, -6.17204653e-03, -1.89000927e-03,
                    -3.14102764e-03, -7.67878769e-03, -1.13740638e-02, -7.41857197e-03, -6.29213545e-03, -6.51977258e-03, -8.27047508e-03, -1.33656031e-02, -8.59564263e-03, -2.51553906e-03, -4.64272872e-03, -1.11560747e-02, -1.64905936e-02, -1.08321551e-02, -9.26420093e-03, -9.67171416e-03, -1.23506878e-02, -2.00199075e-02, -1.27442302e-02, -3.45497206e-03,
                    -7.49545777e-03, -1.76018942e-02, -2.59558801e-02, -1.72390267e-02, -1.49321631e-02, -1.57669969e-02, -2.03234926e-02, -3.30405571e-02, -2.06389092e-02, -4.78462130e-03, -1.38390735e-02, -3.14943902e-02, -4.63354364e-02, -3.13667879e-02, -2.77508944e-02, -2.98541505e-02, -3.89749333e-02, -6.32867143e-02, -3.77952419e-02, -5.26650995e-03,
                    -3.16195861e-02, -6.90807998e-02, -1.01725549e-01, -7.13700354e-02, -6.54785037e-02, -7.25797564e-02, -9.49372798e-02, -1.47399038e-01, -7.21285641e-02, 2.15010419e-02, -8.06625858e-02, -1.79638922e-01, -2.66877055e-01, -1.64447501e-01, -1.00968637e-01, -2.75682062e-02, 1.13596700e-01,  3.32260162e-01,  5.96845448e-01, 8.13161016e-01,
                    9.52381015e-01,  8.13161016e-01,  5.96845508e-01, 3.32260162e-01,  1.13596708e-01, -2.75682174e-02, -1.37202948e-01, -2.71326721e-01, -1.84127048e-01, -7.94974267e-02, 3.29870060e-02, -7.39035010e-02, -1.60488203e-01, -1.04997143e-01, -8.06594491e-02, -7.25797564e-02, -7.87955597e-02, -1.11791104e-01, -7.58660138e-02, -3.48676592e-02,
                    -4.96974029e-03, -4.04525958e-02, -6.82792515e-02, -4.20900472e-02, -3.21968049e-02, -2.98541524e-02, -3.36477235e-02, -4.95737195e-02, -3.37007530e-02, -1.48636252e-02, -4.92655952e-03, -2.17927732e-02, -3.49853337e-02, -2.15152260e-02, -1.66727621e-02, -1.57669988e-02, -1.81730352e-02, -2.73226351e-02, -1.85334161e-02, -7.91355036e-03,
                    -3.57114570e-03, -1.33136865e-02, -2.09431648e-02, -1.29161589e-02, -1.01064872e-02, -9.67171136e-03, -1.12970043e-02, -1.71830691e-02, -1.16271935e-02, -4.84848116e-03, -2.59314431e-03, -8.91274121e-03, -1.38697922e-02, -8.58002994e-03, -6.75992295e-03, -6.51977304e-03, -7.68158771e-03, -1.17703741e-02, -7.94785097e-03, -3.25604435e-03,
                    -1.94202550e-03, -6.36530807e-03, -9.84015409e-03, -6.10316684e-03, -4.83274320e-03, -4.68666898e-03, -5.55526093e-03, -8.55536573e-03, -5.76688722e-03, -2.33053416e-03, -1.50016253e-03, -4.76644421e-03, -7.33569637e-03, -4.55961144e-03, -3.62428720e-03, -3.52900638e-03, -4.20164689e-03, -6.49448857e-03, -4.37143166e-03, -1.74761284e-03,
                    -1.19028054e-03, -3.69978836e-03, -5.67591935e-03, -3.53418733e-03, -2.81759514e-03, -2.75207404e-03, -3.28776496e-03, -5.09600528e-03, -3.42601724e-03, -1.35771628e-03, -9.65878542e-04, -2.95373448e-03, -4.52052988e-03, -2.81889434e-03, -2.25270819e-03, -2.20575323e-03, -2.64216494e-03, -4.10421193e-03, -2.75646802e-03, -1.08450721e-03,
                    -7.98697409e-04, -2.41194153e-03, -3.68447183e-03, -2.30037421e-03, -1.84193184e-03, -1.80714857e-03, -2.16938392e-03, -3.37567786e-03, -2.26523401e-03, -8.85842834e-04, -6.71049987e-04, -2.00629188e-03, -3.06024216e-03, -1.91263494e-03, -1.53396139e-03, -1.50748459e-03, -1.81288645e-03, -2.82496959e-03, -1.89429161e-03, -7.36965681e-04,
                    -5.71501616e-04, -1.69480499e-03, -2.58198148e-03, -1.61517004e-03, -1.29717519e-03, -1.27655920e-03, -1.53747783e-03, -2.39865575e-03, -1.60740130e-03, -6.22576685e-04, -4.92433901e-04, -1.45049067e-03, -2.20754091e-03, -1.38200901e-03, -1.11122860e-03, -1.09486456e-03, -1.32032647e-03, -2.06194492e-03, -1.38099224e-03, -5.32818493e-04});

    input.linspace(-10, 0.1);
    gradO = 1;

    nd4j::ops::lrn_bp op;

    auto results = op.evaluate({&input, &gradO}, {1., 1., 1}, {2});
    auto gradI = results->at(0);

    ASSERT_EQ(*gradI, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_bp_3) {

    NDArray input('c', {2,3,4,10});
    NDArray gradO('c', {2,3,4,10});
    NDArray exp('c', {2,3,4,10}, {-6.78180193e-04, -1.06947345e-03, -1.50362519e-03, -1.47711602e-03, -1.45060697e-03, -1.42409769e-03, -1.39758852e-03, -1.37107936e-03, -8.79839936e-04, -4.27795108e-04, -8.62496032e-04, -1.34585891e-03, -1.88281795e-03, -1.84591592e-03, -1.80901436e-03, -1.77211256e-03, -1.73521065e-03, -1.69830909e-03, -1.08184782e-03, -5.13895764e-04,
                                 -1.13227055e-03, -1.74428569e-03, -2.42520543e-03, -2.37169350e-03, -2.31818156e-03, -2.26466986e-03, -2.21115816e-03, -2.15764646e-03, -1.36136822e-03, -6.26647263e-04, -1.54878304e-03, -2.34815548e-03, -3.23930010e-03, -3.15753091e-03, -3.07576265e-03, -2.99399323e-03, -2.91222427e-03, -2.83045508e-03, -1.76287338e-03, -7.75904860e-04,
                                 -2.23870482e-03, -3.32566188e-03, -4.54067392e-03, -4.40674182e-03, -4.27281018e-03, -4.13887901e-03, -4.00494691e-03, -3.87101574e-03, -2.36659218e-03, -9.72117065e-04, -3.49745504e-03, -5.05724549e-03, -6.80746930e-03, -6.56589260e-03, -6.32431870e-03, -6.08274434e-03, -5.84116904e-03, -5.59959421e-03, -3.32604628e-03, -1.21081201e-03,
                                 -6.14068285e-03, -8.55270587e-03, -1.12749329e-02, -1.07723922e-02, -1.02698486e-02, -9.76730697e-03, -9.26476624e-03, -8.76222178e-03, -4.94601438e-03, -1.37539487e-03, -1.30690653e-02, -1.72132626e-02, -2.19351258e-02, -2.06174850e-02, -1.92998387e-02, -1.79821979e-02, -1.66645572e-02, -1.53469117e-02, -7.72346184e-03, -5.22134826e-04,
                                 -3.99478227e-02, -4.78655733e-02, -5.70126995e-02, -5.16961850e-02, -4.63796593e-02, -4.10631336e-02, -3.57466117e-02, -3.04300785e-02, -9.11374856e-03, 1.14024431e-02, -2.35893592e-01, -2.17480078e-01, -1.88097835e-01, -1.38812393e-01, -8.95269737e-02, -4.02415469e-02, 9.04385652e-03,  5.83292767e-02,  1.78530529e-01, 2.96026409e-01,
                                 4.16666657e-01,  2.79557735e-01,  1.36546940e-01, 7.49502778e-02,  1.33536234e-02, -4.82430384e-02, -1.09839723e-01, -1.71436355e-01, -2.33033031e-01, -2.74476141e-01, 1.54189002e-02, -8.10869783e-03, -3.24862264e-02, -3.88403721e-02, -4.51945364e-02, -5.15486896e-02, -5.79028539e-02, -6.42570183e-02, -5.45457527e-02, -4.61437553e-02,
                                 -2.29711179e-04, -8.06892477e-03, -1.63567103e-02, -1.78351123e-02, -1.93135180e-02, -2.07919199e-02, -2.22703181e-02, -2.37487257e-02, -1.87229179e-02, -1.43175106e-02, -1.37000845e-03, -5.16320160e-03, -9.21433326e-03, -9.76086594e-03, -1.03073996e-02, -1.08539313e-02, -1.14004640e-02, -1.19469995e-02, -9.08647850e-03, -6.55380823e-03,
                                 -1.23490533e-03, -3.45137389e-03, -5.83263952e-03, -6.09064987e-03, -6.34865928e-03, -6.60666777e-03, -6.86467718e-03, -7.12268520e-03, -5.30054048e-03, -3.67741752e-03, -9.94500006e-04, -2.44303374e-03, -4.00528917e-03, -4.14666394e-03, -4.28803731e-03, -4.42941114e-03, -4.57078544e-03, -4.71215881e-03, -3.45545518e-03, -2.33156094e-03,
                                 -7.93270417e-04, -1.81236281e-03, -2.91444198e-03, -3.00004939e-03, -3.08565609e-03, -3.17126350e-03, -3.25687067e-03, -3.34247784e-03, -2.42513884e-03, -1.60246110e-03, -6.39747130e-04, -1.39506557e-03, -2.21352675e-03, -2.26921216e-03, -2.32489733e-03, -2.38058274e-03, -2.43626791e-03, -2.49195332e-03, -1.79354590e-03, -1.16592250e-03,
                                 -5.23828785e-04, -1.10576022e-03, -1.73730974e-03, -1.77553250e-03, -1.81375467e-03, -1.85197743e-03, -1.89020019e-03, -1.92842260e-03, -1.37922564e-03, -8.84913374e-04, -4.35433642e-04, -8.97393096e-04, -1.39935245e-03, -1.42670958e-03, -1.45406683e-03, -1.48142409e-03, -1.50878134e-03, -1.53613824e-03, -1.09309505e-03, -6.93831593e-04,
                                 -3.66991735e-04, -7.42538832e-04, -1.15100679e-03, -1.17125409e-03, -1.19150116e-03, -1.21174823e-03, -1.23199564e-03, -1.25224248e-03, -8.87364266e-04, -5.58210537e-04, -3.13144788e-04, -6.24410110e-04, -9.63238359e-04, -9.78639582e-04, -9.94040747e-04, -1.00944215e-03, -1.02484343e-03, -1.04024459e-03, -7.34565372e-04, -4.58585098e-04,
                                 -2.70129647e-04, -5.32291830e-04, -8.17865424e-04, -8.29851197e-04, -8.41836852e-04, -8.53822567e-04, -8.65808397e-04, -8.77794111e-04, -6.18013146e-04, -3.83307983e-04, -2.35282409e-04, -4.59096394e-04, -7.03040219e-04, -7.12549896e-04, -7.22059398e-04, -7.31569016e-04, -7.41078693e-04, -7.50588137e-04, -5.27105702e-04, -3.25074652e-04});

    input.linspace(-10, 0.1);
    gradO = 1;

    nd4j::ops::lrn_bp op;

    auto results = op.evaluate({&input, &gradO}, {1., 1., 1}, {7});
    auto gradI = results->at(0);

    ASSERT_EQ(*gradI, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_bp_4) {

    NDArray input('c', {2,3,4,10});
    NDArray gradO('c', {2,3,4,10});
    NDArray exp('c', {2,3,4,10}, {-0.00119282, -0.00116995, -0.00114708, -0.00112421, -0.00110134, -0.00107847, -0.00105559, -0.00103272, -0.00100985, -0.00098698, -0.00150102, -0.00146918, -0.00143734, -0.0014055 , -0.00137366, -0.00134182, -0.00130998, -0.00127814, -0.0012463 , -0.00121446,
                                -0.00194534,-0.00189916, -0.00185299, -0.00180681, -0.00176064, -0.00171446, -0.00166829, -0.00162211, -0.00157593, -0.00152976, -0.0026189 , -0.00254833, -0.00247776, -0.00240719, -0.00233662, -0.00226605, -0.00219548, -0.00212491, -0.00205434, -0.00198377,
                                -0.00370962, -0.00359401, -0.00347839, -0.00336277, -0.00324716, -0.00313154, -0.00301593, -0.00290031, -0.00278469, -0.00266908, -0.00564327, -0.00543464, -0.00522602, -0.00501739, -0.00480876, -0.00460013, -0.0043915 , -0.00418288, -0.00397425, -0.00376562,
                                -0.00955302, -0.00911865, -0.00868428, -0.00824992, -0.00781555, -0.00738118, -0.00694682, -0.00651245, -0.00607808, -0.00564371, -0.01927758, -0.01813637, -0.01699515, -0.01585394, -0.01471272, -0.01357151, -0.01243029, -0.01128908, -0.01014786, -0.00900664,
                                -0.05409876, -0.04945958, -0.04482041, -0.04018124, -0.03554206, -0.03090289, -0.02626371, -0.02162454, -0.01698537, -0.01234619, -0.26145172, -0.214688  , -0.16792431, -0.12116055, -0.07439683, -0.02763309,  0.01913062,  0.06589434, 0.11265809,  0.15942183,
                                0.25974026,  0.19902176,  0.13830325,  0.07758474, 0.01686624, -0.04385226, -0.10457078, -0.16528927, -0.22600779, -0.2867263 , -0.01177884, -0.0173331 , -0.02288735, -0.02844159, -0.03399584, -0.0395501 , -0.04510435, -0.05065861, -0.05621284, -0.0617671 ,
                                -0.00944993, -0.01073084, -0.01201174, -0.01329265, -0.01457355, -0.01585446, -0.01713536, -0.01841627, -0.01969717, -0.02097807, -0.00589878, -0.00637122, -0.00684368, -0.00731612, -0.00778858, -0.00826102, -0.00873347, -0.00920592, -0.00967837, -0.01015082,
                                -0.00390961, -0.00413245, -0.00435528, -0.00457812, -0.00480095, -0.00502378, -0.00524662, -0.00546945, -0.00569229, -0.00591512, -0.00275609, -0.00287813, -0.00300018, -0.00312222, -0.00324427, -0.00336631, -0.00348836, -0.0036104 , -0.00373245, -0.00385449,
                                -0.00203982, -0.00211371, -0.00218759, -0.00226147, -0.00233536, -0.00240924, -0.00248312, -0.00255701, -0.00263089, -0.00270478, -0.00156781, -0.00161586, -0.00166391, -0.00171197, -0.00176002, -0.00180807, -0.00185612, -0.00190417, -0.00195223, -0.00200028,
                                -0.00124141, -0.00127439, -0.00130737, -0.00134035, -0.00137333, -0.00140631, -0.00143929, -0.00147227, -0.00150525, -0.00153822, -0.00100674, -0.00103034, -0.00105394, -0.00107754, -0.00110115, -0.00112475, -0.00114835, -0.00117195, -0.00119556, -0.00121916,
                                -0.00083255, -0.00085002, -0.00086748, -0.00088495, -0.00090242, -0.00091989, -0.00093735, -0.00095482, -0.00097229, -0.00098976, -0.0006998 , -0.00071308, -0.00072637, -0.00073965, -0.00075294, -0.00076623, -0.00077951, -0.0007928 , -0.00080609, -0.00081937,
                                -0.00059635, -0.00060669, -0.00061703, -0.00062737, -0.00063771, -0.00064805, -0.00065839, -0.00066873, -0.00067906, -0.0006894 , -0.0005142 , -0.0005224 , -0.00053061, -0.00053881, -0.00054701, -0.00055522, -0.00056342, -0.00057162, -0.00057983, -0.00058803});

    input.linspace(-10, 0.1);
    gradO = 1;

    nd4j::ops::lrn_bp op;

    auto results = op.evaluate({&input, &gradO}, {1., 1., 1}, {12});
    auto gradI = results->at(0);

    ASSERT_EQ(*gradI, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_bp_5) {

    NDArray input('c', {2,2,2,5});
    NDArray gradO('c', {2,2,2,5});
    NDArray exp('c', {2,2,2,5}, {6.2497472e-03, -3.4008762e-03, -1.5232352e-02, 2.3018382e-04,  1.3257053e-02, 7.1492628e-03, -5.4330104e-03, -2.0878183e-02, 1.5153568e-03,  2.0571884e-02,
                                 6.7926152e-03, -1.0990440e-02, -3.2685306e-02, 7.2436016e-03,  4.2120241e-02, -1.3439789e-02, -3.4284033e-02, -4.4852167e-02, 8.8073254e-02,  2.2223940e-01,
                                 4.0824831e-01,  2.1201703e-01,  3.8555145e-02, -3.1969927e-02, -3.0673094e-02, 5.2034661e-02,  1.0463811e-02, -3.6619946e-02, -1.3280880e-02,  5.9767403e-03,
                                 2.3028374e-02,  2.0452859e-03, -2.2533152e-02, -6.1039329e-03,  7.2805062e-03, 1.4290780e-02,  3.8017845e-04, -1.6107092e-02,-3.6896234e-03,  6.4357026e-03});
    input.linspace(-20, 1);
    // gradO.linspace(0.1, 0.1);
    gradO = 1;

    nd4j::ops::lrn_bp op;

    auto results = op.evaluate({&input, &gradO}, {1., 1., 0.5}, {2});
    auto gradI = results->at(0);

    ASSERT_EQ(*gradI, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_bp_6) {

    NDArray input('c', {1,1,1,5}, {1, 2., 3, 4, 5});
    NDArray gradO('c', {1,1,1,5});
    NDArray exp('c', {1,1,1,5}, {0.06926288,  0.04360996,  0.01795704, -0.00769587, -0.0333488});
    // gradO.linspace(-1.5, 0.1);
    gradO = 1;

    nd4j::ops::lrn_bp op;

    auto results = op.evaluate({&input, &gradO}, {1., 2., 0.5}, {10});
    auto gradI = results->at(0);

    ASSERT_EQ(*gradI, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_bp_7) {

    NDArray input('c', {2,2,2,5});
    NDArray gradO('c', {2,2,2,5});

    input.linspace(-20, 1);
    gradO.linspace(-1.5, 0.1);

    const OpArgsHolder argsHolderFF({&input}, {1,2,0.5}, {2});
    const OpArgsHolder argsHolderBP({&input, &gradO}, {1,2,0.5}, {2});

    nd4j::ops::lrn opFF;
    nd4j::ops::lrn_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_bp_8) {

    NDArray input('c', {1,1,1,5}, {1, 2, 3, 4, 5});
    NDArray gradO('c', {1,1,1,5}, {2, 3, 4, 5, 6});

    const OpArgsHolder argsHolderFF({&input}, {1,2,0.5}, {2});
    const OpArgsHolder argsHolderBP({&input, &gradO}, {1,2,0.5}, {2});

    nd4j::ops::lrn opFF;
    nd4j::ops::lrn_bp opBP;

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_bp_9) {

    NDArray input('c', {1,1,1,5}, {1,2,3,4,5});
    NDArray gradO('c', {1,1,1,5}, {1, 1, 1, 1, 1});
    NDArray exp('c', {1,1,1,5}, {0.1084472 ,  0.03816165,  0.00978456, -0.01859251,-0.02511311});

    nd4j::ops::lrn_bp op;

    auto results = op.evaluate({&input, &gradO}, {1., 2., 0.5}, {3});
    auto gradI = results->at(0);

    // for (int i = 0; i < exp.lengthOf(); ++i)
    //     printf("%10.5f  %10.5f\n", exp.e<double>(i), gradI->e<double>(i));

    ASSERT_EQ(*gradI, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_bp_10) {

    NDArray input('c', {1,1,1,1}, std::vector<double>{1});
    NDArray gradO('c', {1,1,1,1}, std::vector<double>{1});
    NDArray exp('c', {1,1,1,1}, std::vector<double>{0.19245008});

    nd4j::ops::lrn_bp op;

    auto results = op.evaluate({&input, &gradO}, {1., 2., 0.5}, {1});
    auto gradI = results->at(0);

    ASSERT_EQ(*gradI, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_1) {

    NDArray input('c', {2,2,2,5});
    NDArray exp('c', {2,2,2,5}, {-0.42923987, -0.3623817 , -0.3152079 , -0.34268343, -0.3836809, -0.43648192, -0.3652726 , -0.31428117, -0.3379276 , -0.3731494 ,
                                -0.45129365, -0.37083852, -0.3111639 , -0.3260225 , -0.34698898, -0.4975186 , -0.3831305 , -0.2847474 , -0.25607377, -0.18569534,
                                 0., 0.18569534,  0.25607377,  0.38411066,  0.52075565,0.33633637,  0.32117262,  0.30966178,  0.37259716,  0.45631808,
                                 0.36986336,  0.33643705,  0.31394684,  0.36608824,  0.43857202, 0.3821113 ,  0.34197718,  0.31508508,  0.36284128,  0.4303756 });

    input.linspace(-20, 1);

    nd4j::ops::lrn op;

    auto results = op.evaluate({&input}, {1., 2., 0.5}, {2});
    auto output = results->at(0);

    ASSERT_EQ(*output, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_2) {

    NDArray input('c', {1,1,1,5}, {1, 2., 3, 4, 5});
    NDArray exp('c', {1,1,1,5}, {0.09530295, 0.1906059 , 0.28590885, 0.3812118 , 0.47651473});

    nd4j::ops::lrn op;

    auto results = op.evaluate({&input}, {0.1, 2., 0.5}, {5});
    auto output = results->at(0);
    ASSERT_EQ(*output, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_3) {

    NDArray input('c', {1,1,1,1}, std::vector<double>{1.});
    NDArray exp('c', {1,1,1,1}, std::vector<double>{0.69006556});

    nd4j::ops::lrn op;

    auto results = op.evaluate({&input}, {0.1, 2., 0.5}, {5});
    auto output = results->at(0);
    ASSERT_EQ(*output, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_4) {

    NDArray input('c', {1,1,1,1}, std::vector<double>{1.});
    NDArray exp('c', {1,1,1,1}, std::vector<double>{0.69006556});

    nd4j::ops::lrn op;

    auto results = op.evaluate({&input}, {0.1, 2., 0.5}, {0});
    auto output = results->at(0);
    ASSERT_EQ(*output, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, lrn_5) {

    NDArray input('c', {1,1,1,5}, {1, 2., 3, 4, 5});
    NDArray exp('c', {1,1,1,5}, {0.69006556, 0.70272833, 0.7051508 , 0.7060045 , 0.7064008});

    nd4j::ops::lrn op;

    auto results = op.evaluate({&input}, {0.1, 2., 0.5}, {0});
    auto output = results->at(0);
    ASSERT_EQ(*output, exp);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, inTopK_1) {

    NDArray x('c', {4, 5}, {11.0, 14.0, 6.0, 9.0, 3.5, 7.0, 21.0, 3.0,  15.0, 6.0, 9.0, 3.5, 7.0, 11.0, 13.0, 5.0, 16.0, 9.0, 13.5, 7.0});
    NDArray y('c', {4}, {0., 0, 0, 0}, nd4j::DataType::INT64);
    NDArray z('c', {4}, {1., 1, 1, 1}, nd4j::DataType::BOOL);

    NDArray expV('c', {4}, {1., 0, 0, 0}, nd4j::DataType::BOOL);

    nd4j::ops::in_top_k op;
    Nd4jStatus status = op.execute({&x, &y, }, {&z}, {}, {2}, {});

    // z.printIndexedBuffer();
    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(expV.isSameShape(z));
    ASSERT_TRUE(expV.equalsTo(z));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, inTopK_2) {

    auto input = NDArrayFactory::create<double>('c', {4, 5});
    auto idx = NDArrayFactory::create<Nd4jLong>('c', {4});

    auto exp = NDArrayFactory::create<bool>({false, false, false, true});

    int exclusive, reverse;
    input.linspace(1);
    idx.linspace(1);

    nd4j::ops::in_top_k op;

    auto res = op.evaluate({&input, &idx}, {}, {1});

    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    //res->at(0)->printIndexedBuffer("IN_TOP_K output");
    ASSERT_TRUE(res->at(0)->equalsTo(&exp));
    delete res;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, inTopK_3) {
    auto x = NDArrayFactory::create<double>('c', {2, 3}, {1.0, 11.0, 3.0, 14.0, 5.0, 6.0});
    auto y = NDArrayFactory::create<Nd4jLong>('c', {2}, {1, 1});
    auto expV = NDArrayFactory::create<bool>('c', {2}, {true, false});

    nd4j::ops::in_top_k op;
    auto result = op.evaluate({&x, &y}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto v = result->at(0);

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, inTopK_4) {
    auto x = NDArrayFactory::create<double>('c', {6, 4}, {11.0, 3.0, 14.0, 5.0, 6.0, 9.0, 3.5, 7.0, 21.0, 3.0, 14.0, 15.0, 6.0, 9.0, 3.5, 7.0, 11.0, 13.0, 14.0, 5.0, 16.0, 9.0, 13.5, 7.0} );
    auto y = NDArrayFactory::create<Nd4jLong>('c', {6}, {0, 0, 0, 0, 0, 0});
    auto expV = NDArrayFactory::create<bool>('c', {6}, {true, false, true, false, false, true});

    nd4j::ops::in_top_k op;
    auto result = op.evaluate({&x, &y}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto v = result->at(0);

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, inTopK_5) {
    auto x = NDArrayFactory::create<double>('f', {6, 4}, {11.0, 3.0, 14.0, 5.0, 6.0, 9.0, 3.5, 7.0, 21.0, 3.0, 14.0, 15.0, 6.0, 9.0, 3.5, 7.0, 11.0, 13.0, 14.0, 5.0, 16.0, 9.0, 13.5, 7.0} );
    auto y = NDArrayFactory::create<Nd4jLong>('f', {6}, {0, 0, 0, 0, 0, 0});
    auto expV = NDArrayFactory::create<bool>('f', {6}, {true, false, false, false, false, false });

    nd4j::ops::in_top_k op;
    auto result = op.evaluate({&x, &y}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto v = result->at(0);

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, cube_1) {

    NDArray x('c', {2, 3}, {1., 2., 3., 4., 5, 6});
    NDArray exp('c', {2, 3}, {1., 8., 27., 64., 125, 216});

    nd4j::ops::cube op;

    auto result = op.evaluate({&x});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, cube_bp_1) {

    NDArray x('c', {2, 3}, {1., 2., 3., 4., 5, 6});
    NDArray gradO('c', {2, 3}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {2, 3}, {1.5, 6., 13.5, 24., 37.5, 54});

    gradO = 0.5;

    nd4j::ops::cube_bp op;

    auto result = op.evaluate({&x, &gradO});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    // z->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
// CONSTANT mode 2D
TEST_F(DeclarableOpsTests12, pad_tests1) {


    NDArray input('c', {2,3}, {1,2,3,4,5,6}, nd4j::DataType::FLOAT32);
    NDArray paddings('c', {2,2}, {1,1,2,2}, nd4j::DataType::INT32);
    NDArray expected('c', {4,7}, {0,0,0,0,0,0,0, 0,0,1,2,3,0,0, 0,0,4,5,6,0,0, 0,0,0,0,0,0,0}, nd4j::DataType::FLOAT32);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// REFLECT mode 2D
TEST_F(DeclarableOpsTests12, pad_tests2) {

    float inBuff[]  = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    int padBuff[] = {1,1,2,2};
    float expBuff[] = {6.f, 5.f, 4.f, 5.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 2.f, 3.f, 2.f, 1.f, 6.f, 5.f, 4.f, 5.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 2.f, 3.f, 2.f, 1.f};

    auto input    = NDArrayFactory::create<float>(inBuff,  'c', {2,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {2,2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4,7});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// SYMMETRIC mode 2D
TEST_F(DeclarableOpsTests12, pad_tests3) {

    float inBuff[]  = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    Nd4jLong padBuff[] = {1,1,2,2};
    float expBuff[] = {2.f, 1.f, 1.f, 2.f, 3.f, 3.f, 2.f,  2.f,1.f,1.f,2.f,3.f,3.f,2.f, 5.f,4.f,4.f,5.f,6.f,6.f,5.f, 5.f,4.f,4.f,5.f,6.f,6.f,5.f};

    auto input    = NDArrayFactory::create<float>(inBuff,  'c', {2,3});
    auto paddings = NDArrayFactory::create<Nd4jLong>(padBuff, 'c', {2,2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4,7});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// CONSTANT mode 3D
TEST_F(DeclarableOpsTests12, pad_tests4) {

    float inBuff[]  = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f,11.f,12.f,13.f,14.f,15.f,16.f,17.f,18.f};
    int padBuff[] = {1,1,2,2,2,2};
    float expBuff[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 2.f, 3.f, 0.f, 0.f, 0.f, 0.f, 4.f, 5.f, 6.f, 0.f, 0.f, 0.f, 0.f,
                        7.f, 8.f, 9.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 10.f, 11.f, 12.f, 0.f,
                        0.f, 0.f, 0.f, 13.f, 14.f, 15.f, 0.f, 0.f, 0.f, 0.f, 16.f, 17.f, 18.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                        0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    auto input    = NDArrayFactory::create<float>(inBuff,  'c', {2,3,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {3,2});
    auto expected = NDArrayFactory::create<float>(expBuff, 'c', {4,7,7});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    // for(int i = 0; i < expected.lengthOf(); ++i) {
    //     float one = expected.e<float>(i);
    //     float two = result->e<float>(i);
    //     if(one != two)
    //         printf("%i : %f, %f\n", i, one, two);
    // }

    delete results;
}



////////////////////////////////////////////////////////////////////
// REFLECT mode 3D
TEST_F(DeclarableOpsTests12, pad_tests5) {

    double inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    int padBuff[] = {1,1,2,2,2,2};
    double expBuff[] = {18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 15,14,13,14,15,14,13, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 15,14,13,14,15,14,13, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1};
    auto input    = NDArrayFactory::create<double>(inBuff,  'c', {2,3,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {3,2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4,7,7});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// SYMMETRIC mode 3D
TEST_F(DeclarableOpsTests12, pad_tests6) {

    double inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    int padBuff[] = {1,1,2,2,2,2};
    double expBuff[] = {5, 4, 4, 5, 6, 6, 5, 2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2, 5, 4, 4, 5, 6, 6, 5, 8, 7, 7, 8, 9, 9, 8, 8, 7, 7, 8, 9, 9, 8, 5, 4, 4, 5, 6, 6, 5, 5, 4, 4, 5, 6, 6, 5, 2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2, 5, 4, 4, 5, 6, 6, 5, 8, 7, 7, 8, 9, 9, 8, 8, 7, 7, 8, 9, 9, 8, 5, 4, 4, 5, 6, 6, 5, 14,13,13,14,15,15,14, 11,10,10,11,12,12,11, 11,10,10,11,12,12,11, 14,13,13,14,15,15,14, 17,16,16,17,18,18,17, 17,16,16,17,18,18,17, 14,13,13,14,15,15,14, 14,13,13,14,15,15,14, 11,10,10,11,12,12,11, 11,10,10,11,12,12,11, 14,13,13,14,15,15,14, 17,16,16,17,18,18,17, 17,16,16,17,18,18,17, 14,13,13,14,15,15,14};

    auto input    = NDArrayFactory::create<double>(inBuff,  'c', {2,3,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {3,2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4,7,7});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
// CONSTANT mode 4D
TEST_F(DeclarableOpsTests12, pad_tests7)
{

    double inBuff[] =  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    double expBuff[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 0, 0, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto input = NDArrayFactory::create<double>(inBuff, 'c', {2, 2, 2, 2});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {4, 2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
// REFLECT mode 4D
TEST_F(DeclarableOpsTests12, pad_tests8)
{

    double inBuff[] =  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    double expBuff[] = {16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1};
    auto input = NDArrayFactory::create<double>(inBuff, 'c', {2, 2, 2, 2});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {4, 2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////
// SYMMETRIC mode 4D
TEST_F(DeclarableOpsTests12, pad_tests9)
{

    double inBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    double expBuff[] = {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16};
    auto input = NDArrayFactory::create<double>(inBuff, 'c', {2, 2, 2, 2});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {4, 2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests10) {

    auto input    = NDArrayFactory::create<double>('c', {2,3,4});
    auto paddings = NDArrayFactory::create<int>('c', {3,2}, {0,0, 0,1, 0,0});
    auto expected = NDArrayFactory::create<double>('c', {2,4,4}, {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.});

    input = 1.f;
    //input.assign(1.);
    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests11) {

    auto input    = NDArrayFactory::create<double>('c', {2,3,4});
    auto paddings = NDArrayFactory::create<int>('c', {3,2}, {0,0, 0,1, 0,0});
    auto expected = NDArrayFactory::create<double>('c', {2,4,4}, {1., 2., 3., 4., 5., 6., 7., 8., 9.,10.,11.,12., 5., 6., 7., 8.,13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,17.,18.,19.,20.});

    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests12) {

    auto input    = NDArrayFactory::create<double>('c', {2,3,4,5});
    auto paddings = NDArrayFactory::create<int>('c', {4,2}, {0,0, 0,1, 0,1, 0,0});
    auto expected = NDArrayFactory::create<double>('c', {2,4,5,5}, { 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 16., 17., 18., 19., 20.,
                                             21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 36., 37., 38., 39., 40.,
                                             41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 56., 57., 58., 59., 60.,
                                             41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 56., 57., 58., 59., 60.,
                                             61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 80., 76., 77., 78., 79., 80.,
                                             81., 82., 83., 84., 85., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99.,100., 96., 97., 98., 99.,100.,
                                            101.,102.,103.,104.,105.,106.,107.,108.,109.,110.,111.,112.,113.,114.,115.,116.,117.,118.,119.,120.,116.,117.,118.,119.,120.,
                                            101.,102.,103.,104.,105.,106.,107.,108.,109.,110.,111.,112.,113.,114.,115.,116.,117.,118.,119.,120.,116.,117.,118.,119.,120.});
    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests13) {

    auto input    = NDArrayFactory::create<double>('c', {5});
    auto paddings = NDArrayFactory::create<int>('c', {1,2}, {2,3});
    auto expected = NDArrayFactory::create<double>('c', {10}, {3., 2., 1., 2., 3., 4., 5., 4., 3., 2.});
    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests14) {

    auto input    = NDArrayFactory::create<double>('c', {1,5});
    auto paddings = NDArrayFactory::create<int>('c', {2,2}, {0,0,2,3});
    auto expected = NDArrayFactory::create<double>('c', {1,10}, {2., 1., 1., 2., 3., 4., 5., 5., 4., 3.});
    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests15) {

    auto input    = NDArrayFactory::create<double>('c', {1,5});
    auto paddings = NDArrayFactory::create<int>('c', {2,2}, {1,1,0,0});
    auto expected = NDArrayFactory::create<double>('c', {3,5}, {1., 2., 3., 4., 5., 1., 2., 3., 4., 5., 1., 2., 3., 4., 5.});
    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests16) {

    auto input    = NDArrayFactory::create<double>('c', {5,1});
    auto paddings = NDArrayFactory::create<int>('c', {2,2}, {2,3,0,0});
    auto expected = NDArrayFactory::create<double>('c', {10,1}, {3., 2., 1., 2., 3., 4., 5., 4., 3., 2.});
    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests17) {

    auto input    = NDArrayFactory::create<double>('c', {5,1});
    auto paddings = NDArrayFactory::create<int>('c', {2,2}, {0,0,1,0});
    auto expected = NDArrayFactory::create<double>('c', {5,2}, {1.,1., 2.,2., 3.,3., 4.,4., 5.,5.});
    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests18) {

    auto input    = NDArrayFactory::create<double>('c', {5});
    auto paddings = NDArrayFactory::create<int>('c', {1,2}, {0,0});
    auto expected = NDArrayFactory::create<double>('c', {5}, {1.,2.,3.,4.,5.});
    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests19) {

    auto input    = NDArrayFactory::create<double>('c', {5,1});
    auto paddings = NDArrayFactory::create<int>('c', {2,2}, {0,0,0,0});
    auto expected = NDArrayFactory::create<double>('c', {5,1}, {1., 2., 3., 4., 5.});
    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests20) {

    auto input    = NDArrayFactory::create<double>('c', {1,5});
    auto paddings = NDArrayFactory::create<int>('c', {2,2}, {0,0,0,0});
    auto expected = NDArrayFactory::create<double>('c', {1,5}, {1., 2., 3., 4., 5.});
    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests21) {

    auto input    = NDArrayFactory::create<double>('c', {1,3,1,5});
    auto paddings = NDArrayFactory::create<int>('c', {4,2}, {0,0, 0,1, 0,1, 0,0});
    auto expected = NDArrayFactory::create<double>('c', {1,4,2,5}, {1., 2., 3., 4., 5., 1., 2., 3., 4., 5., 6., 7., 8., 9.,10., 6., 7., 8., 9.,10.,
                                             11.,12.,13.,14.,15.,11.,12.,13.,14.,15.,11.,12.,13.,14.,15.,11.,12.,13.,14.,15.});
    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests22) {

    auto input    = NDArrayFactory::create<double>('c', {1,1});
    auto paddings = NDArrayFactory::create<int>('c', {2,2}, {0,0, 0,0});
    auto expected = NDArrayFactory::create<double>('c', {1,1}, {1.});

    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests23) {

    auto input    = NDArrayFactory::create<double>('c', {1,1});
    auto paddings = NDArrayFactory::create<int>('c', {2,2}, {0,0, 1,0});
    auto expected = NDArrayFactory::create<double>('c', {1,2}, {0.,1.});

    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printShapeInfo("r");
    // expected.printShapeInfo("e");

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests24) {

    auto input    = NDArrayFactory::create<double>('c', {1});
    auto paddings = NDArrayFactory::create<int>('c', {1,2}, {0,0});
    auto expected = NDArrayFactory::create<double>('c', {1}, {1.});

    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests25) {

    auto input    = NDArrayFactory::create<double>('c', {1});
    auto paddings = NDArrayFactory::create<int>('c', {1,2}, {1,1});
    auto expected = NDArrayFactory::create<double>('c', {3}, {1.,1.,1});

    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests26) {

    auto input    = NDArrayFactory::create<double>('c', {1});
    auto paddings = NDArrayFactory::create<int>('c', {1,2}, {3,2});
    auto expected = NDArrayFactory::create<double>('c', {6}, {0., 0., 0., 1., 0., 0.});

    input.linspace(1.f);

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests27) {

    NDArray input('c', {2,3}, nd4j::DataType::FLOAT32);
    NDArray paddings('c', {2,2}, {0,0,0,1}, nd4j::DataType::INT32);
    NDArray exp('c', {2,4}, {1,1,1,0,1,1,1,0}, nd4j::DataType::FLOAT32);
    NDArray z('c', {2,4}, nd4j::DataType::FLOAT32);
    input = 1.;

    nd4j::ops::pad op;
    Nd4jStatus status = op.execute({&input, &paddings}, {&z}, {0}, {0}, {});      // constant
    // z.printIndexedBuffer();

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(exp.isSameShapeStrict(z));
    ASSERT_TRUE(exp.equalsTo(z));
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests28) {

    NDArray input('c', {1,111,111,32}, nd4j::DataType::FLOAT32);
    NDArray paddings('c', {4,2}, {0,0,0,1,0,1,0,0}, nd4j::DataType::INT32);
    NDArray z('c', {1,112,112,32}, nd4j::DataType::FLOAT32);
    input = 1.;

    nd4j::ops::pad op;
    Nd4jStatus status = op.execute({&input, &paddings}, {&z}, {0}, {0}, {});      // constant
    // z.printIndexedBuffer();

    NDArray sum = z.reduceNumber(nd4j::reduce::Sum);

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_EQ(sum.e<float>(0), 111*111*32);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests29) {

    auto in = NDArrayFactory::create<double>({1., 1., 1., 1., 1.});
//    auto pad = NDArrayFactory::create<double>('c', {1, 2}, {1., 1.});// = Nd4j.create(new double[]{1, 1}, new long[]{1, 2});
    auto pad = NDArrayFactory::create<int>('c', {1, 2}, {1, 1});
//    auto value(10.0);

    auto exp = NDArrayFactory::create<double>({10., 1., 1., 1., 1., 1., 10.});

    nd4j::ops::pad op;

    auto res = op.evaluate({&in, &pad}, {10.0}, {0});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    delete res;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests30) {

    auto in = NDArrayFactory::create<double>({1., 11., 111., 11., 1.});
    auto pad = NDArrayFactory::create<int>('c', {1, 2}, {1, 1});

    auto exp = NDArrayFactory::create<double>({1., 1., 11., 111., 11., 1., 1.});

    nd4j::ops::pad op;

    auto res = op.evaluate({&in, &pad}, {10.0}, {2});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests31) {

    auto in = NDArrayFactory::create<double>({1., 11., 111., 1111., 11111.});
//    auto pad = NDArrayFactory::create<double>('c', {1, 2}, {1., 1.});// = Nd4j.create(new double[]{1, 1}, new long[]{1, 2});
    auto pad = NDArrayFactory::create<int>('c', {1, 2}, {1, 1});
//    auto value(10.0);

    auto exp = NDArrayFactory::create<double>({11., 1., 11., 111., 1111., 11111., 1111.});

    nd4j::ops::pad op;

    auto res = op.evaluate({&in, &pad}, {10.0}, {1});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    delete res;
}

///////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests32) {

    auto in = NDArrayFactory::create<double>('c', {3,3}, {1., 2., 3., 4., 5.,6,7,8,9});
    auto pad = NDArrayFactory::create<int>('c', {2,2}, {1, 2, 2, 3});

    auto exp = NDArrayFactory::create<double>('c', {6,8}, {2, 1, 1, 2, 3, 3, 2, 1, 2, 1, 1, 2, 3, 3, 2, 1, 5, 4, 4, 5, 6, 6, 5, 4, 8, 7, 7, 8, 9, 9, 8, 7, 8, 7, 7, 8, 9, 9, 8, 7, 5, 4, 4, 5, 6, 6, 5, 4});

    nd4j::ops::pad op;

    auto res = op.evaluate({&in, &pad}, {10.0}, {2});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    delete res;
}
///////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests33) {

    auto in = NDArrayFactory::create<double>('c', {2,3,4}, {1, 2, 3, 4,5, 6, 7, 8,9,10,11,12,13, 14, 15, 16,17, 18, 19, 20,21, 22, 23, 24});

    auto pad = NDArrayFactory::create<int>('c', {3,2}, {1, 2, 2, 3, 3,3});

    auto exp = NDArrayFactory::create<double>('c', {5,8,10}, { 7,6,5,5,6,7,8,8,7,6.,   3,2,1,1,2,3,4,4,3,2.,   3,2,1,1,2,3,4,4,3,2.,   7,6,5,5,6,7,8,8,7,6.,  11,10,9,9,10,11,12,12,11,10.,
                                                            11,10,9,9,10,11,12,12,11,10.,   7,6,5,5,6,7,8,8,7,6.,   3,2,1,1,2,3,4,4,3,2.,  7,6,5,5,6,7,8,8,7,6.,   3,2,1,1,2,3,4,4,3,2.,
                                                             3,2,1,1,2,3,4,4,3,2.,   7,6,5,5,6,7,8,8,7,6.,  11,10,9,9,10,11,12,12,11,10.,  11,10,9,9,10,11,12,12,11,10.,7,6,5,5,6,7,8,8,7,6.,
                                                             3,2,1,1,2,3,4,4,3,2., 19,18,17,17,18,19,20,20,19,18.,  15,14,13,13,14,15,16,16,15,14.,  15,14,13,13,14,15,16,16,15,14.,
                                                            19,18,17,17,18,19,20,20,19,18.,  23,22,21,21,22,23,24,24,23,22.,  23,22,21,21,22,23,24,24,23,22.,  19,18,17,17,18,19,20,20,19,18.,
                                                            15,14,13,13,14,15,16,16,15,14., 19,18,17,17,18,19,20,20,19,18.,  15,14,13,13,14,15,16,16,15,14.,  15,14,13,13,14,15,16,16,15,14.,
                                                            19,18,17,17,18,19,20,20,19,18.,  23,22,21,21,22,23,24,24,23,22.,  23,22,21,21,22,23,24,24,23,22.,  19,18,17,17,18,19,20,20,19,18.,
                                                            15,14,13,13,14,15,16,16,15,14.,  7,6,5,5,6,7,8,8,7,6.,   3,2,1,1,2,3,4,4,3,2.,   3,2,1,1,2,3,4,4,3,2.,   7,6,5,5,6,7,8,8,7,6.,
                                                            11,10,9,9,10,11,12,12,11,10.,  11,10,9,9,10,11,12,12,11,10.,   7,6,5,5,6,7,8,8,7,6.,   3,2,1,1,2,3,4,4,3,2.});
    nd4j::ops::pad op;

    auto res = op.evaluate({&in, &pad}, {10.0}, {2});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    delete res;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, pad_tests34) {

    NDArray input('c', {5}, {0.778786, 0.801198, 0.724375, 0.230894, 0.727141}, nd4j::DataType::FLOAT32);
    NDArray paddings('c', {1,2}, {1,1}, nd4j::DataType::INT32);
    NDArray expected('c', {7}, {10., 0.778786, 0.801198, 0.724375, 0.230894, 0.727141, 10.}, nd4j::DataType::FLOAT32);
    NDArray z('c', {7}, nd4j::DataType::FLOAT32);

    nd4j::ops::pad op;
    Nd4jStatus status = op.execute({&input, &paddings}, {&z}, {10}, {0}, {});      // constant

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(expected.isSameShapeStrict(z));
    ASSERT_TRUE(expected.equalsTo(z));
}

////////////////////////////////////////////////////////////////////
// CONSTANT mode 2D
TEST_F(DeclarableOpsTests12, Pad_1) {

    double inBuff[]  = {1,2,3,4,5,6};
    int padBuff[] = {1,1,2,2};
    double expBuff[] = {0,0,0,0,0,0,0, 0,0,1,2,3,0,0, 0,0,4,5,6,0,0, 0,0,0,0,0,0,0};

    auto input    = NDArrayFactory::create<double>(inBuff,  'c', {2,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {2,2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4,7});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// REFLECT mode 2D
TEST_F(DeclarableOpsTests12, Pad_2) {

    double inBuff[]  = {1,2,3,4,5,6};
    int padBuff[] = {1,1,2,2};
    double expBuff[] = {6,5,4,5,6,5,4, 3,2,1,2,3,2,1, 6,5,4,5,6,5,4, 3,2,1,2,3,2,1};

    auto input    = NDArrayFactory::create<double>(inBuff,  'c', {2,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {2,2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4,7});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// SYMMETRIC mode 2D
TEST_F(DeclarableOpsTests12, Pad_3) {

    double inBuff[]  = {1,2,3,4,5,6};
    int padBuff[] = {1,1,2,2};
    double expBuff[] = {2,1,1,2,3,3,2, 2,1,1,2,3,3,2, 5,4,4,5,6,6,5, 5,4,4,5,6,6,5};

    auto input    = NDArrayFactory::create<double>(inBuff,  'c', {2,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {2,2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4,7});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// CONSTANT mode 3D
TEST_F(DeclarableOpsTests12, Pad_4) {

    double inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    int padBuff[] = {1,1,2,2,2,2};
    double expBuff[] = {0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 1, 2, 3,0,0,0,0, 4, 5, 6,0,0,0,0, 7, 8, 9,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0,10,11,12,0,0,0,0,13,14,15,0,0,0,0,16,17,18,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0,0,0, 0, 0, 0,0,0};

    auto input    = NDArrayFactory::create<double>(inBuff,  'c', {2,3,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {3,2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4,7,7});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}



////////////////////////////////////////////////////////////////////
// REFLECT mode 3D
TEST_F(DeclarableOpsTests12, Pad_5) {

    double inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    int padBuff[] = {1,1,2,2,2,2};
    double expBuff[] = {18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 15,14,13,14,15,14,13, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 15,14,13,14,15,14,13, 18,17,16,17,18,17,16, 15,14,13,14,15,14,13, 12,11,10,11,12,11,10, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1, 6, 5, 4, 5, 6, 5, 4, 9, 8, 7, 8, 9, 8, 7, 6, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 2, 1};
    auto input    = NDArrayFactory::create<double>(inBuff,  'c', {2,3,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {3,2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4,7,7});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


////////////////////////////////////////////////////////////////////
// SYMMETRIC mode 3D
TEST_F(DeclarableOpsTests12, Pad_6) {

    double inBuff[]  = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    int padBuff[] = {1,1,2,2,2,2};
    double expBuff[] = {5, 4, 4, 5, 6, 6, 5, 2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2, 5, 4, 4, 5, 6, 6, 5, 8, 7, 7, 8, 9, 9, 8, 8, 7, 7, 8, 9, 9, 8, 5, 4, 4, 5, 6, 6, 5, 5, 4, 4, 5, 6, 6, 5, 2, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 2, 5, 4, 4, 5, 6, 6, 5, 8, 7, 7, 8, 9, 9, 8, 8, 7, 7, 8, 9, 9, 8, 5, 4, 4, 5, 6, 6, 5, 14,13,13,14,15,15,14, 11,10,10,11,12,12,11, 11,10,10,11,12,12,11, 14,13,13,14,15,15,14, 17,16,16,17,18,18,17, 17,16,16,17,18,18,17, 14,13,13,14,15,15,14, 14,13,13,14,15,15,14, 11,10,10,11,12,12,11, 11,10,10,11,12,12,11, 14,13,13,14,15,15,14, 17,16,16,17,18,18,17, 17,16,16,17,18,18,17, 14,13,13,14,15,15,14};

    auto input    = NDArrayFactory::create<double>(inBuff,  'c', {2,3,3});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {3,2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4,7,7});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
// CONSTANT mode 4D
TEST_F(DeclarableOpsTests12, Pad_7)
{

    double inBuff[] =  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    double expBuff[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 0, 0, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 14, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto input = NDArrayFactory::create<double>(inBuff, 'c', {2, 2, 2, 2});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {4, 2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
// REFLECT mode 4D
TEST_F(DeclarableOpsTests12, Pad_8)
{

    double inBuff[] =  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    double expBuff[] = {16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13, 12, 11, 12, 11, 10, 9, 10, 9, 12, 11, 12, 11, 10, 9, 10, 9, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1, 8, 7, 8, 7, 6, 5, 6, 5, 8, 7, 8, 7, 6, 5, 6, 5, 4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1};
    auto input = NDArrayFactory::create<double>(inBuff, 'c', {2, 2, 2, 2});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {4, 2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

//////////////////////////////////////////////////////////////////
// SYMMETRIC mode 4D
TEST_F(DeclarableOpsTests12, Pad_9)
{

    double inBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int padBuff[] = {1, 1, 1, 1, 1, 1, 1, 1};
    double expBuff[] = {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 9, 9, 10, 10, 9, 9, 10, 10, 11, 11, 12, 12, 11, 11, 12, 12, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16, 13, 13, 14, 14, 13, 13, 14, 14, 15, 15, 16, 16, 15, 15, 16, 16};
    auto input = NDArrayFactory::create<double>(inBuff, 'c', {2, 2, 2, 2});
    auto paddings = NDArrayFactory::create<int>(padBuff, 'c', {4, 2});
    auto expected = NDArrayFactory::create<double>(expBuff, 'c', {4, 4, 4, 4});

    nd4j::ops::pad op;
    auto results = op.evaluate({&input, &paddings}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *result = results->at(0);
    // result->printIndexedBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(*result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

TEST_F(DeclarableOpsTests12, Test_Expose_1) {
    auto input0 = NDArrayFactory::create<double>('c', {2, 3}, {1, 2, 3, 6, 5, 4});
    auto input1 = NDArrayFactory::create<double>('c', {2, 3}, {3, 2, 1, 4, 5, 6});

    nd4j::ops::expose op;

    auto result = op.evaluate({&input0, &input1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z0 = result->at(0);
    auto z1 = result->at(1);

    ASSERT_TRUE(input0.equalsTo(z0));
    ASSERT_TRUE(input1.equalsTo(z1));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, Pad_SGO_Test_1) {

    auto in = NDArrayFactory::create<double>({1., 1., 1., 1., 1.});
//    auto pad = NDArrayFactory::create<double>('c', {1, 2}, {1., 1.});// = Nd4j.create(new double[]{1, 1}, new long[]{1, 2});
    auto pad = NDArrayFactory::create<int>('c', {1, 2}, {1, 1});
//    auto value(10.0);

    auto exp = NDArrayFactory::create<double>({10., 1., 1., 1., 1., 1., 10.});

    nd4j::ops::pad op;

    auto res = op.evaluate({&in, &pad}, {10.0}, {0});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    // res->at(0)->printIndexedBuffer("PAD_SGO");
    // exp.printIndexedBuffer("PAD_EXP");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, LU_Test_1) {

    auto in = NDArrayFactory::create<double>('c', {3,3}, {1., 2., 3., 0., 2., 3., 0., 0., 7.});
    auto exp = NDArrayFactory::create<double>('c', {3,3}, {1., 2., 3., 0., 2., 3., 0., 0., 7});
    auto pExp = NDArrayFactory::create<int>('c', {3}, {0, 1, 2});
    nd4j::ops::lu op;

    auto res = op.evaluate({&in});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);
    auto p = res->at(1);
//    z->printIndexedBuffer("Triangulars");
//    p->printIndexedBuffer("Permutaions");

    ASSERT_TRUE(exp.equalsTo(z));
    ASSERT_TRUE(pExp.equalsTo(p));

    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, LU_Test_2) {
    auto in = NDArrayFactory::create<double>('c', {3,3}, {1, 0, 0, 2, 3, 0, 4, 5, 6});

    auto expLU = NDArrayFactory::create<double>('c', {3,3}, {4.,  5.,  6., 0.25, -1.25, -1.5, 0.5, -0.4, -3.6});
    auto expP = NDArrayFactory::create<int>({2, 0, 1});
    nd4j::ops::lu op;

    auto res = op.evaluate({&in});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);
    auto p = res->at(1);
//    z->printIndexedBuffer("Triangulars2");
//    p->printIndexedBuffer("Permutaions2");
    ASSERT_TRUE(expLU.equalsTo(z));
    ASSERT_TRUE(expP.equalsTo(p));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, LU_Test_3) {
    auto in = NDArrayFactory::create<double>('c', {3,3}, {1,2,3,4,7,9, 11, 12, 13});

    auto expLU = NDArrayFactory::create<double>('c', {3,3}, {
               11.,        12.,         13.,
        0.36363637,  2.6363635,    4.272727,
        0.09090909,  0.3448276,  0.34482753});

    auto expP = NDArrayFactory::create<int>({2, 1, 0});
    nd4j::ops::lu op;

    auto res = op.evaluate({&in});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);
    auto p = res->at(1);
//    z->printIndexedBuffer("Triangulars3");
//    p->printIndexedBuffer("Permutaions3");
    ASSERT_TRUE(expLU.equalsTo(z));
    ASSERT_TRUE(expP.equalsTo(p));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, LU_Test_4) {

    auto in = NDArrayFactory::create<double>('c', {10,10}, {
            1.,   2.,   3.,   4.,   5.,   6.,   7.,  8.,   1.,  15.,
            5.,   1.,  13.,   4.,  15.,   1.,  17.,  9.,  11.,  25.,
            1.,   9.,   1.,   4.,   5.,   2.,  13.,  10,  21.,  15.,
            3.,   9.,   4.,   1.,   5.,   3.,   7.,   1,   1.,   5.,
            2.,   3.,   2.,   5.,   4.,   4.,   7.,   3,   3.,   4.,
            0.,   1.,   3.,   3.,   5.,   1.,   3.,   1,  31.,  15.,
            2.,   1.,   4.,   3.,   1.,   5.,   1.,   2,  31.,  35.,
            3.,   4.,   3.,   3.,   4.,   4.,   4.,  1.,   3.,   1.,
            1.,   1.,   1.,   1.,   5.,   6.,   5.,  4.,   3.,   2.,
            1.,   1.,   1.,   1.,   1.,   1.,   1.,  1.,   1.,   1.});

    auto expLU = NDArrayFactory::create<double>('c', {10,10}, {
            5.0,      1.0,      13.0,       4.0,      15.0,       1.0,      17.0,       9.0,       11.0,       25.0,
            0.2,      8.8,      -1.6,       3.2,       2.0,       1.8,       9.6,       8.2,       18.8,       10.0,
            0.6, 0.386364, -4.181818, -0.636364, -5.772727,  2.704545, -9.909091, -7.568182, -10.863636, -17.863636,
            0.6, 0.954545,  0.543478, -4.108696, -2.771739, -0.788043, -6.978261, -8.114130, -17.641304,  -9.836957,
            0.4, 0.068182,  0.260870, -0.328042, -4.539683,  3.513228, -6.158730, -2.846561,  22.365079,  25.751323,
            0.2, 0.090909,  0.347826, -0.031746, -0.823427,  7.563520, -1.118881,  1.485431,  20.725524,  23.196387,
            0.0, 0.113636, -0.760870, -0.523810,  0.236014,  0.213036, -7.593805, -9.585099,   1.663379, -15.900300,
            0.4, 0.295455,  0.652174, -0.698413,  0.167832,  0.021727, -0.001360, -3.321530, -16.392106, - 9.022119,
            0.2, 0.204545, -0.173913, -0.592593,  0.232517,  0.610602,  0.277466, -0.244631, -39.715757, -18.928178,
            0.2, 0.090909,  0.347826, -0.031746,  0.057692, -0.070344, -0.030154, -0.243578,   0.087256,   0.112695
    });

    auto expP = NDArrayFactory::create<int>({1, 2, 7, 3, 6, 8, 5, 4, 0, 9});
    nd4j::ops::lu op;

    auto res = op.evaluate({&in});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);
    auto p = res->at(1);
//    z->printBuffer("Triangulars4");
//    expLU.printBuffer("TriangulExp4");
//    p->printBuffer("Permutaions4");

    ASSERT_TRUE(expLU.equalsTo(z));
    ASSERT_TRUE(expP.equalsTo(p));
    delete res;
}

TEST_F(DeclarableOpsTests12, LU_Test_5) {

    auto in = NDArrayFactory::create<double>('c', {2, 10,10}, {
            1.,   2.,   3.,   4.,   5.,   6.,   7.,  8.,   1.,  15.,
            5.,   1.,  13.,   4.,  15.,   1.,  17.,  9.,  11.,  25.,
            1.,   9.,   1.,   4.,   5.,   2.,  13.,  10,  21.,  15.,
            3.,   9.,   4.,   1.,   5.,   3.,   7.,   1,   1.,   5.,
            2.,   3.,   2.,   5.,   4.,   4.,   7.,   3,   3.,   4.,
            0.,   1.,   3.,   3.,   5.,   1.,   3.,   1,  31.,  15.,
            2.,   1.,   4.,   3.,   1.,   5.,   1.,   2,  31.,  35.,
            3.,   4.,   3.,   3.,   4.,   4.,   4.,  1.,   3.,   1.,
            1.,   1.,   1.,   1.,   5.,   6.,   5.,  4.,   3.,   2.,
            1.,   1.,   1.,   1.,   1.,   1.,   1.,  1.,   1.,   1.,

            1.,   2.,   3.,   4.,   5.,   6.,   7.,  8.,   1.,  15.,
            5.,   1.,  13.,   4.,  15.,   1.,  17.,  9.,  11.,  25.,
            1.,   9.,   1.,   4.,   5.,   2.,  13.,  10,  21.,  15.,
            3.,   9.,   4.,   1.,   5.,   3.,   7.,   1,   1.,   5.,
            2.,   3.,   2.,   5.,   4.,   4.,   7.,   3,   3.,   4.,
            0.,   1.,   3.,   3.,   5.,   1.,   3.,   1,  31.,  15.,
            2.,   1.,   4.,   3.,   1.,   5.,   1.,   2,  31.,  35.,
            3.,   4.,   3.,   3.,   4.,   4.,   4.,  1.,   3.,   1.,
            1.,   1.,   1.,   1.,   5.,   6.,   5.,  4.,   3.,   2.,
            1.,   1.,   1.,   1.,   1.,   1.,   1.,  1.,   1.,   1.
    });

    auto expLU = NDArrayFactory::create<double>('c', {2, 10,10}, {
            5.0,      1.0,      13.0,       4.0,      15.0,       1.0,      17.0,       9.0,       11.0,       25.0,
            0.2,      8.8,      -1.6,       3.2,       2.0,       1.8,       9.6,       8.2,       18.8,       10.0,
            0.6, 0.386364, -4.181818, -0.636364, -5.772727,  2.704545, -9.909091, -7.568182, -10.863636, -17.863636,
            0.6, 0.954545,  0.543478, -4.108696, -2.771739, -0.788043, -6.978261, -8.114130, -17.641304,  -9.836957,
            0.4, 0.068182,  0.260870, -0.328042, -4.539683,  3.513228, -6.158730, -2.846561,  22.365079,  25.751323,
            0.2, 0.090909,  0.347826, -0.031746, -0.823427,  7.563520, -1.118881,  1.485431,  20.725524,  23.196387,
            0.0, 0.113636, -0.760870, -0.523810,  0.236014,  0.213036, -7.593805, -9.585099,   1.663379, -15.900300,
            0.4, 0.295455,  0.652174, -0.698413,  0.167832,  0.021727, -0.001360, -3.321530, -16.392106, - 9.022119,
            0.2, 0.204545, -0.173913, -0.592593,  0.232517,  0.610602,  0.277466, -0.244631, -39.715757, -18.928178,
            0.2, 0.090909,  0.347826, -0.031746,  0.057692, -0.070344, -0.030154, -0.243578,   0.087256,   0.112695,

            5.0,      1.0,      13.0,       4.0,      15.0,       1.0,      17.0,       9.0,       11.0,       25.0,
            0.2,      8.8,      -1.6,       3.2,       2.0,       1.8,       9.6,       8.2,       18.8,       10.0,
            0.6, 0.386364, -4.181818, -0.636364, -5.772727,  2.704545, -9.909091, -7.568182, -10.863636, -17.863636,
            0.6, 0.954545,  0.543478, -4.108696, -2.771739, -0.788043, -6.978261, -8.114130, -17.641304,  -9.836957,
            0.4, 0.068182,  0.260870, -0.328042, -4.539683,  3.513228, -6.158730, -2.846561,  22.365079,  25.751323,
            0.2, 0.090909,  0.347826, -0.031746, -0.823427,  7.563520, -1.118881,  1.485431,  20.725524,  23.196387,
            0.0, 0.113636, -0.760870, -0.523810,  0.236014,  0.213036, -7.593805, -9.585099,   1.663379, -15.900300,
            0.4, 0.295455,  0.652174, -0.698413,  0.167832,  0.021727, -0.001360, -3.321530, -16.392106, - 9.022119,
            0.2, 0.204545, -0.173913, -0.592593,  0.232517,  0.610602,  0.277466, -0.244631, -39.715757, -18.928178,
            0.2, 0.090909,  0.347826, -0.031746,  0.057692, -0.070344, -0.030154, -0.243578,   0.087256,   0.112695

    });

    auto expP = NDArrayFactory::create<int>('c', {2, 10}, {
            1, 2, 7, 3, 6, 8, 5, 4, 0, 9,
            1, 2, 7, 3, 6, 8, 5, 4, 0, 9
    });
    nd4j::ops::lu op;

    auto res = op.evaluate({&in});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);
    auto p = res->at(1);
//    z->printBuffer("Triangulars5");
//    expLU.printBuffer("TriangulExp5");
//    p->printBuffer("Permutaions5");

    ASSERT_TRUE(expLU.equalsTo(z));
    ASSERT_TRUE(expP.equalsTo(p));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, LU_Test_1_2) {

    auto in = NDArrayFactory::create<double>('c', {2, 3,3}, {1., 2., 3., 0., 2., 3., 0., 0., 7.,1., 2., 3., 0., 2., 3., 0., 0., 7.});
    auto exp = NDArrayFactory::create<double>('c', {2, 3,3}, {1., 2., 3., 0., 2., 3., 0., 0., 7, 1., 2., 3., 0., 2., 3., 0., 0., 7.});

    nd4j::ops::lu op;

    auto res = op.evaluate({&in});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);
    auto p = res->at(1);
//    z->printIndexedBuffer("Triangulars (2,3,3)");
//    p->printIndexedBuffer("Permutaions (2,3,3)");
    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, LU_Test_3_2) {

    auto in = NDArrayFactory::create<double>('c', {2, 3,3}, {1,2,3,4,7,9, 11, 12, 13,1,2,3,4,7,9, 11, 12, 13});

    auto expLU = NDArrayFactory::create<double>('c', {2, 3,3}, {
            11.,        12.,         13.,
            0.36363637,  2.6363635,    4.272727,
            0.09090909,  0.3448276,  0.34482753,

            11.,        12.,         13.,
            0.36363637,  2.6363635,    4.272727,
            0.09090909,  0.3448276,  0.34482753
    });

    auto expP = NDArrayFactory::create<int>('c', {2,3}, {2, 1, 0, 2, 1, 0});
    nd4j::ops::lu op;

    auto res = op.evaluate({&in});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);
    auto p = res->at(1);
//    z->printIndexedBuffer("Triangulars3_2");
//    p->printIndexedBuffer("Permutaions3_2");

    ASSERT_TRUE(expLU.equalsTo(z));
    ASSERT_TRUE(expP.equalsTo(p));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, LU_Test_3_3) {

    auto in = NDArrayFactory::create<double>('c', {2, 3,3}, {1,2,3,4,7,9, 11, 12, 13,13,2,3,4,7,9, 11, 12, 1});
    auto expLU = NDArrayFactory::create<double>('c', {2, 3,3}, {
            11.,        12.,         13.,
            0.36363637,  2.6363635,    4.272727,
            0.09090909,  0.3448276,  0.34482753,

                    13.,        2.,         3.,
             0.84615386, 10.307693, -1.5384617,
             0.30769232,  0.619403,   9.029851});

    auto expP = NDArrayFactory::create<int>('c', {2,3}, {2, 1, 0, 0, 2, 1});
    nd4j::ops::lu op;

    auto res = op.evaluate({&in});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);
    auto p = res->at(1);
//    z->printIndexedBuffer("Triangulars3_3");
//    p->printIndexedBuffer("Permutaions3_3");

    ASSERT_TRUE(expLU.equalsTo(z));
    ASSERT_TRUE(expP.equalsTo(p));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, LU_Test_4_1) {

    auto in = NDArrayFactory::create<float>('c', {2, 2,2}, {
        0.7788f, 0.8012f, 0.7244f, 0.2309f,
        0.7271f, 0.1804f, 0.5056f, 0.8925f
    });

    auto expLU = NDArrayFactory::create<float>('c', {2, 2,2}, {
        0.7788f, 0.8012f, 0.930149f, -0.514335f,
        0.7271f, 0.1804f, 0.695365f,  0.767056f
    });

    auto expP = NDArrayFactory::create<int>('c', {2,2}, {0, 1, 0, 1});
    nd4j::ops::lu op;

    auto res = op.evaluate({&in});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);
    auto p = res->at(1);
//    z->printIndexedBuffer("Triangulars4_1");
//    p->printIndexedBuffer("Permutaions4_1");

    ASSERT_TRUE(expLU.equalsTo(z));
    ASSERT_TRUE(expP.equalsTo(p));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, LU_Test_4_2) {

    auto in = NDArrayFactory::create<float>('c', {2, 2,2}, {
        0.7788f, 0.8012f,  0.7244f,    0.2309f,
        0.7271f, 0.1804f,  0.5056f,    0.8925f
    });

    auto expLU = NDArrayFactory::create<float>('c', {2, 2,2}, {
            0.7788f, 0.8012f, 0.930149f, -0.514335f,
            0.7271f, 0.1804f, 0.695365f, 0.767056f
    });

    auto expP = NDArrayFactory::create<Nd4jLong>('c', {2,2}, {0, 1, 0, 1});
    nd4j::ops::lu op;

    auto res = op.evaluate({&in}, {}, {nd4j::DataType::INT64});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);
    auto p = res->at(1);
//    z->printIndexedBuffer("Triangulars4_2");
//    p->printIndexedBuffer("Permutaions4_2");

    ASSERT_TRUE(expLU.equalsTo(z));
    ASSERT_TRUE(expP.equalsTo(p));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, QR_Test_1) {

    auto in = NDArrayFactory::create<double>('c', {5,3}, {
        12.,  -51.,    4.,        6.,   167.,  -68.,       -4.,    24.,  -41.,       -1.,     1.,    0.,        2.,     0.,    3.
    });
    auto expQ = NDArrayFactory::create<double>('c', {5, 5}, {
              0.8464148,   0.3912908,  -0.3431241,  0.06613743, -0.09146205,            -0.42320737,  -0.9040873,  0.02927014,  0.01737854, -0.04861044,             0.28213826, -0.17042054, -0.93285596, -0.02194202,  0.14371186,             0.07053456, -0.01404065,  0.00109937,  0.99740064,  0.00429488,            -0.14106913,  0.0166551,  0.10577161,  0.00585613,  0.98417485
    });

    auto expR = NDArrayFactory::create<double>('c', {5,3}, {
       -14.177447, -20.666622,       13.401566,               0., -175.04254,       70.080315,               0.,         0.,       35.201546,               0.,         0.,              0.,               0.,         0.,              0. });
    nd4j::ops::qr op;
    auto res = op.evaluate({&in}, {}, {}, {true});

    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto q = res->at(0);
    auto r = res->at(1);
//    q->printIndexedBuffer("Orthogonal 5x5");
//    expQ.printBuffer("Orthogonal Exp");
//    r->printIndexedBuffer("Upper triangular 5x3");
//    expR.printBuffer("Upper triangular Exp");
//    q->printShapeInfo("Q shape");
//    r->printShapeInfo("R shape");
    nd4j::ops::matmul opMul;
    auto res2 = opMul.evaluate({q, r}); //MmulHelper::matmul(q, r, &in, false, false);
    auto exp = res2->at(0);//->printIndexedBuffer("Result as result");
    ASSERT_TRUE(exp->isSameShape(in));
//    ASSERT_TRUE(q->isSameShape(expQ));

    //ASSERT_TRUE(expQ.equalsTo(q));
    ASSERT_TRUE(exp->equalsTo(in));
    delete res2;
    delete res;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, QR_Test_1_1) {

    auto in = NDArrayFactory::create<double>('c', {4, 5, 3}, {
            12.,  -51.,    4.,            6.,   167.,  -68.,            -4.,    24.,  -41.,            -1.,     1.,    0.,            2.,     0.,    3.,
            12.,  -51.,    4.,            6.,   167.,  -68.,            -4.,    24.,  -41.,            -1.,     1.,    0.,            2.,     0.,    3.,
            12.,  -51.,    4.,            6.,   167.,  -68.,            -4.,    24.,  -41.,            -1.,     1.,    0.,            2.,     0.,    3.,
            12.,  -51.,    4.,            6.,   167.,  -68.,            -4.,    24.,  -41.,            -1.,     1.,    0.,            2.,     0.,    3.
    });
    auto expQ = NDArrayFactory::create<double>('c', {4, 5, 5}, {
            0.8464148,   0.3912908,  -0.3431241,  0.06613743, -0.09146205,            -0.42320737,  -0.9040873,  0.02927014,  0.01737854, -0.04861044,            0.28213826, -0.17042054, -0.93285596, -0.02194202,  0.14371186,            0.07053456, -0.01404065,  0.00109937,  0.99740064,  0.00429488,            -0.14106913,  0.0166551,  0.10577161,  0.00585613,  0.98417485,
            0.8464148,   0.3912908,  -0.3431241,  0.06613743, -0.09146205,            -0.42320737,  -0.9040873,  0.02927014,  0.01737854, -0.04861044,            0.28213826, -0.17042054, -0.93285596, -0.02194202,  0.14371186,            0.07053456, -0.01404065,  0.00109937,  0.99740064,  0.00429488,            -0.14106913,  0.0166551,  0.10577161,  0.00585613,  0.98417485,
            0.8464148,   0.3912908,  -0.3431241,  0.06613743, -0.09146205,            -0.42320737,  -0.9040873,  0.02927014,  0.01737854, -0.04861044,            0.28213826, -0.17042054, -0.93285596, -0.02194202,  0.14371186,            0.07053456, -0.01404065,  0.00109937,  0.99740064,  0.00429488,            -0.14106913,  0.0166551,  0.10577161,  0.00585613,  0.98417485,
            0.8464148,   0.3912908,  -0.3431241,  0.06613743, -0.09146205,            -0.42320737,  -0.9040873,  0.02927014,  0.01737854, -0.04861044,            0.28213826, -0.17042054, -0.93285596, -0.02194202,  0.14371186,            0.07053456, -0.01404065,  0.00109937,  0.99740064,  0.00429488,            -0.14106913,  0.0166551,  0.10577161,  0.00585613,  0.98417485
    });

    auto expR = NDArrayFactory::create<double>('c', {4, 5,3}, {
        -14.177447, -20.666622,       13.401566,            0., -175.04254,       70.080315,            0.,         0.,       35.201546,            0.,         0.,              0.,            0.,         0.,              0.,
        -14.177447, -20.666622,       13.401566,            0., -175.04254,       70.080315,            0.,         0.,       35.201546,            0.,         0.,              0.,            0.,         0.,              0.,
        -14.177447, -20.666622,       13.401566,            0., -175.04254,       70.080315,            0.,         0.,       35.201546,            0.,         0.,              0.,            0.,         0.,              0.,
        -14.177447, -20.666622,       13.401566,            0., -175.04254,       70.080315,            0.,         0.,       35.201546,            0.,         0.,              0.,            0.,         0.,              0.
    });
    nd4j::ops::qr op;
    auto res = op.evaluate({&in}, {}, {}, {true});

    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto q = res->at(0);
    auto r = res->at(1);
//    q->printIndexedBuffer("Orthogonal 5x5");
//    expQ.printBuffer("Orthogonal Exp");
//    r->printIndexedBuffer("Upper triangular 5x3");
//    expR.printBuffer("Upper triangular Exp");
//    q->printShapeInfo("Q shape");
//    r->printShapeInfo("R shape");
    nd4j::ops::matmul opMul;
    auto res2 = opMul.evaluate({q, r}); //MmulHelper::matmul(q, r, &in, false, false);
    auto exp = res2->at(0);//->printIndexedBuffer("Result as result");
    ASSERT_TRUE(exp->isSameShape(in));
//    ASSERT_TRUE(q->isSameShape(expQ));

    //ASSERT_TRUE(expQ.equalsTo(q));
    ASSERT_TRUE(exp->equalsTo(in));
    delete res2;
    delete res;

}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, QR_Test_2) {

    auto in = NDArrayFactory::create<double>('c', {5,3}, {
        12.,  -51.,    4.,  6.,   167.,  -68.,  -4.,    24.,  -41.,  -1.,     1.,    0.,   2.,     0.,    3.
    });
    auto expQ = NDArrayFactory::create<double>('c', {5, 3}, {
            0.8464148,   0.3912908,    -0.3431241,            -0.42320737,  -0.9040873,  0.02927014,            0.28213826, -0.17042054,  -0.93285596,            0.07053456, -0.01404065,   0.00109937,            -0.14106913,  0.0166551,   0.10577161
    });

    auto expR = NDArrayFactory::create<double>('c', {3,3}, {
            -14.177447,         -20.666622,       13.401566,                    0.,         -175.04254,       70.080315,                    0.,                 0.,       35.201546
    });

    nd4j::ops::qr op;
    auto res = op.evaluate({&in}, {}, {}, {false});

    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto q = res->at(0);
    auto r = res->at(1);
    ASSERT_TRUE(q->isSameShape(expQ));
    ASSERT_TRUE(r->isSameShape(expR));
//    q->printIndexedBuffer("Orthogonal 5x5");
//    r->printIndexedBuffer("Upper triangular 5x3");

    nd4j::ops::matmul opMul;
    auto res2 = opMul.evaluate({q, r}); //MmulHelper::matmul(q, r, &in, false, false);
    auto exp = res2->at(0);//->printIndexedBuffer("Result as result");
    ASSERT_TRUE(exp->isSameShape(in));
    ASSERT_TRUE(exp->equalsTo(in));
    delete res2;
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TriangularSolve_Test_1) {

    auto a = NDArrayFactory::create<float>('c', {4, 4}, {
            3.f,  0.f,  0.f,  0.f,
            2.f,  1.f,  0.f,  0.f,
            1.f,  0.f,  1.f,  0.f,
            1.f,  1.f,  1.f,  1.f
    });

    auto b = NDArrayFactory::create<float>('c', {4, 1}, {
            4.f, 2.f, 4.f, 2.f
    });

    auto exp = NDArrayFactory::create<float>('c', {4, 1}, {
            1.333333f,      -0.6666667f,         2.6666667f,        -1.3333333f });

    nd4j::ops::triangular_solve op;

    auto res = op.evaluate({&a, &b});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);

//    z->printIndexedBuffer("TriangularSolve");

    ASSERT_TRUE(exp.equalsTo(z));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TriangularSolve_Test_2) {

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
            2.f,      4.f,         1.f,        1.3333333f });

    nd4j::ops::triangular_solve op;

    auto res = op.evaluate({&a, &b});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);

//    z->printIndexedBuffer("TriangularSolve");

    ASSERT_TRUE(exp.equalsTo(z));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TriangularSolve_Test_3) {

    auto a = NDArrayFactory::create<float>('c', {2, 4, 4}, {
            3.f,  0.f,  0.f,  0.f,
            2.f,  1.f,  0.f,  0.f,
            1.f,  0.f,  1.f,  0.f,
            1.f,  1.f,  1.f,  1.f,

            3.f,  0.f,  0.f,  0.f,
            2.f,  1.f,  0.f,  0.f,
            1.f,  0.f,  1.f,  0.f,
            1.f,  1.f,  1.f,  1.f
    });

    auto b = NDArrayFactory::create<float>('c', {2, 4, 1}, {
            4.f, 2.f, 4.f, 2.f,
            4.f, 2.f, 4.f, 2.f
    });

    auto exp = NDArrayFactory::create<float>('c', {2, 4, 1}, {
            1.333333f,      -0.6666667f,         2.6666667f,        -1.3333333f,
            1.333333f,      -0.6666667f,         2.6666667f,        -1.3333333f
    });

    nd4j::ops::triangular_solve op;

    auto res = op.evaluate({&a, &b});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);

//    z->printIndexedBuffer("TriangularSolve");

    ASSERT_TRUE(exp.equalsTo(z));
    delete res;
}
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TriangularSolve_Test_4) {

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

    nd4j::ops::triangular_solve op;

    auto res = op.evaluate({&a, &b}, {}, {}, {false});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);

//    z->printIndexedBuffer("TriangularSolve");

    ASSERT_TRUE(exp.equalsTo(z));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests12, TriangularSolve_Test_5) {

    auto a = NDArrayFactory::create<float>('c', {4, 4}, {
            5.f,  1., -3.f,  3.f,
            0.f,  1.f,  1.f, -1.f,
            0.f,  0.f,  2.f, -9.f,
            0.f,  0.f,  0.f,  4.f
    });

    auto b = NDArrayFactory::create<float>('c', {4, 1}, {
             5.f,             2.f,             0.f,            -3.f
    });

    auto exp = NDArrayFactory::create<float>('c', {4, 1}, {
            1.f,      1.f,         1.f,        1.f
    });

    nd4j::ops::triangular_solve op;

    auto res = op.evaluate({&a, &b}, {}, {}, {false, true});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto z = res->at(0);

//    z->printIndexedBuffer("TriangularSolve with adjoint");

    ASSERT_TRUE(exp.equalsTo(z));
    delete res;
}
