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


//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test1) {

    NDArray<double> y('c', {2, 3, 4}, {-1.001 ,-0.915 ,-0.829 ,-0.743 ,-0.657 ,-0.571 ,-0.485 ,-0.399 ,-0.313 ,-0.227 ,-0.141 ,-0.055 ,0.031 ,0.117 ,0.203 ,0.289 ,0.375 ,0.461 ,0.547 ,0.633 ,0.719 ,0.805 ,0.891 ,0.977});
    NDArray<double> x('c', {2, 3, 4}, {-0.51, -0.46, -0.41, -0.36, -0.31, -0.26, -0.21, -0.16, -0.11, -0.06, -0.01, 0.04, 0.09, 0.14, 0.19, 0.24, 0.29, 0.34, 0.39, 0.44, 0.49, 0.54, 0.59, 0.61});

    NDArray<double> exp('c', {2,3,4}, {-2.04201, -2.03663, -2.03009, -2.02199,-2.01166, -1.99808, -1.97941, -1.95217,-1.90875, -1.8292 , -1.6416 , -0.942  ,
                                       0.33172,  0.69614,  0.81846,  0.87776, 0.91253,  0.93533,  0.95141,  0.96336, 0.97259,  0.97993,  0.98591,  1.01266,});

    nd4j::ops::tf_atan2<double> op;
    auto result = op.execute({&y, &x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test2) {

    NDArray<double> y('c', {2, 3, 4}, {-1.001 ,-0.915 ,-0.829 ,-0.743 ,-0.657 ,-0.571 ,-0.485 ,-0.399 ,-0.313 ,-0.227 ,-0.141 ,-0.055 ,0.031 ,0.117 ,0.203 ,0.289 ,0.375 ,0.461 ,0.547 ,0.633 ,0.719 ,0.805 ,0.891 ,0.977});
    NDArray<double> x('c', {   3, 4}, {-1.05, -0.82, -0.639, -0.458, -0.277, -0.096, 0.085, 0.266, 0.447, 0.628, 0.809, 0.99});

    NDArray<double> exp('c', {2,3,4}, {-2.38008, -2.30149, -2.22748, -2.1232 ,-1.96979, -1.73736, -1.3973 , -0.98279,-0.61088, -0.34685, -0.17256, -0.0555 ,
                                       3.11208,  2.99987,  2.83399,  2.57869, 2.207  ,  1.77611,  1.41664,  1.17298, 1.01458,  0.90829,  0.8336 ,  0.77879});

    nd4j::ops::tf_atan2<double> op;
    auto result = op.execute({&y, &x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test3) {

    NDArray<double> y('c', {2, 3, 4}, {-1.001 ,-0.915 ,-0.829 ,-0.743 ,-0.657 ,-0.571 ,-0.485 ,-0.399 ,-0.313 ,-0.227 ,-0.141 ,-0.055 ,0.031 ,0.117 ,0.203 ,0.289 ,0.375 ,0.461 ,0.547 ,0.633 ,0.719 ,0.805 ,0.891 ,0.977});
    NDArray<double> x('c', {   3, 4}, {-1.05, -0.82, -0.639, -0.458, -0.277, -0.096, 0.085, 0.266, 0.447, 0.628, 0.809, 0.99});

    NDArray<double> exp('c', {2,3,4}, {-2.33231, -2.41089, -2.48491, -2.58919,-2.74259, -2.97502,  2.9681 ,  2.55359, 2.18167,  1.91765,  1.74335,  1.62629,
                                       -1.54128, -1.42907, -1.2632 , -1.00789,-0.63621, -0.20531,  0.15416,  0.39782, 0.55622,  0.6625 ,  0.7372 ,  0.79201});

    nd4j::ops::tf_atan2<double> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test4) {

    NDArray<double> y('c', {1, 3, 4}, {-1.001 ,-0.829 ,-0.657 ,-0.485 ,-0.313 ,-0.141 ,0.031 ,0.203 ,0.375 ,0.547 ,0.719 ,0.891});
    NDArray<double> x('c', {2, 3, 1}, {-0.82, -0.458, -0.096, 0.085, 0.447, 0.809});

    NDArray<double> exp('c', {2,3,4}, {-2.45527, -2.36165, -2.24628, -2.10492,-2.1703 , -1.86945, -1.50321, -1.15359,-0.25062, -0.17373, -0.13273, -0.10733,
                                        3.05688,  3.03942,  3.01293,  2.9681 , 2.18167,  1.87635,  1.50156,  1.14451, 1.13674,  0.97626,  0.84423,  0.7372 });

    nd4j::ops::tf_atan2<double> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test5) {

    NDArray<double> y('c', {1, 3, 4}, {-1.001 ,-0.829 ,-0.657 ,-0.485 ,-0.313 ,-0.141 ,0.031 ,0.203 ,0.375 ,0.547 ,0.719 ,0.891});
    NDArray<double> x('c', {2, 3, 1}, {-0.82, -0.458, -0.096, 0.085, 0.447, 0.809});

    NDArray<double> exp('c', {2,3,4}, {-2.25712, -2.35074, -2.46611, -2.60747,-2.54209, -2.84294,  3.07401,  2.72438, 1.82141,  1.74453,  1.70353,  1.67813,
                                       -1.48608, -1.46862, -1.44214, -1.3973 ,-0.61088, -0.30556,  0.06924,  0.42629, 0.43405,  0.59453,  0.72657,  0.8336 });

    nd4j::ops::tf_atan2<double> op;
    auto result = op.execute({&y, &x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test6) {

    NDArray<double> y('c', {1, 3, 4}, {-1.001 ,-0.829 ,-0.657 ,-0.485 ,-0.313 ,-0.141 ,0.031 ,0.203 ,0.375 ,0.547 ,0.719 ,0.891});
    NDArray<double> x('c', {      4}, {-0.82, -0.096, 0.085, 0.809});

    NDArray<double> exp('c', {1,3,4}, {-2.25712, -1.68608, -1.44214, -0.54006,-2.77695, -2.16855,  0.34972,  0.24585, 2.71267,  1.74453,  1.45312,  0.8336 });

    nd4j::ops::tf_atan2<double> op;
    auto result = op.execute({&y, &x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, range_test10) {
    
    NDArray<double> limit('c', {1, 3, 4});
    limit = 5.;
    NDArray<double> exp('c', {5}, {0.,1.,2.,3.,4.});

    nd4j::ops::range<double> op;
    auto result = op.execute({&limit}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, range_test11) {
    
    NDArray<double> limit('c', {1, 3, 4});
    NDArray<double> start('c', {2, 4});
    limit = 5.;
    start = 0.5;
    NDArray<double> exp('c', {5}, {0.5,1.5,2.5,3.5,4.5});

    nd4j::ops::range<double> op;
    auto result = op.execute({&start, &limit}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);    

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, range_test12) {
    
    NDArray<double> exp('c', {9}, {0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5});

    nd4j::ops::range<double> op;
    auto result = op.execute({}, {0.5, 5, 0.5}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);    

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, top_k_permuted_test1) {

    NDArray<double> x({7., 3., 1., 2., 5., 0., 4., 6., 9., 8.});
    NDArray<double> expUnsorted({7., 6., 9., 8.}); // Sorted = False
    NDArray<double> expSorted({9., 8., 7., 6., 5.}); // Sorted = False


    nd4j::ops::top_k<double> op;
    auto result = op.execute({&x}, {}, {4, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto zI = result->at(1);
    //z->printIndexedBuffer("TopK(5)");
    //zI->printIndexedBuffer("TopKI(5)");
    ASSERT_TRUE(expUnsorted.isSameShape(z));
    ASSERT_TRUE(expUnsorted.equalsTo(z));

    auto result2 = op.execute({&x}, {}, {5, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result2->status());

    z = result2->at(0);
    zI = result2->at(1);
    //z->printIndexedBuffer("sorted TopK(5)");
    //zI->printIndexedBuffer("sorted TopKI(5)");
    ASSERT_TRUE(expSorted.isSameShape(z));
    ASSERT_TRUE(expSorted.equalsTo(z));

    delete result;
    delete result2;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, top_k_permuted_test2) {

    NDArray<double> x({7., 3., 1., 2., 5., 0., 4., 6., 9., 8.});
    NDArray<double> expUnsorted({7.,    5.,    6.,    9.,    8.}); // Sorted = False
    NDArray<double> expSorted({9., 8., 7., 6., 5.}); // Sorted = False


    nd4j::ops::top_k<double> op;
    auto result = op.execute({&x}, {}, {5, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto zI = result->at(1);
    //z->printIndexedBuffer("TopK(5)");
    //zI->printIndexedBuffer("TopKI(5)");
    ASSERT_TRUE(expUnsorted.isSameShape(z));
    ASSERT_TRUE(expUnsorted.equalsTo(z));

    auto result2 = op.execute({&x}, {}, {5, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result2->status());

    z = result2->at(0);
    zI = result2->at(1);
    //z->printIndexedBuffer("sorted TopK(5)");
    //zI->printIndexedBuffer("sorted TopKI(5)");
    ASSERT_TRUE(expSorted.isSameShape(z));
    ASSERT_TRUE(expSorted.equalsTo(z));

    delete result;
    delete result2;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, sparse_softmax_cross_entropy_loss_with_logits_test1) {
    
    NDArray<double> labels('c', {2,3},{3.,2.,1.,0.,1.,2.});
    NDArray<double> logits('c', {2,3,4});
    NDArray<double> expected('c', {2,3}, {1.24254, 1.34254, 1.44254, 1.54254, 1.44254, 1.34254});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits<double> op;
    nd4j::ResultSet<double>* results = op.execute({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, sparse_softmax_cross_entropy_loss_with_logits_test2) {
    
    NDArray<double> labels('c', {2},{1.,0.});
    NDArray<double> logits('c', {2,3});
    NDArray<double> expected('c', {2}, {1.10194, 1.20194});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits<double> op;
    nd4j::ResultSet<double>* results = op.execute({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, sparse_softmax_cross_entropy_loss_with_logits_test3) {
    
    NDArray<double> labels('c', {1},{0.});
    NDArray<double> logits('c', {1,3});
    NDArray<double> expected('c', {1}, {1.20194});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits<double> op;
    nd4j::ResultSet<double>* results = op.execute({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, sparse_softmax_cross_entropy_loss_with_logits_test4) {
    
    NDArray<double> labels('c', {2},{0.,0.});
    NDArray<double> logits('c', {2,1});
    NDArray<double> expected('c', {2}, {0., 0.});
                                            
    logits.linspace(0.1, 0.1);

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits<double> op;
    nd4j::ResultSet<double>* results = op.execute({&labels, &logits}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}
