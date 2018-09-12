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
TEST_F(DeclarableOpsTests10, InTopK_SGO_Test_1) {

    NDArray<double> input('c', {4, 5});
    NDArray<double> idx('c', {4});

    NDArray<double> exp({0., 0., 0., 1.});

    int exclusive, reverse;
    input.linspace(1);
    idx.linspace(1);
    ////////////////////////////////////////

    nd4j::ops::in_top_k<double> op;

    auto res = op.execute({&input, &idx}, {}, {1});

    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    //res->at(0)->printIndexedBuffer("IN_TOP_K output");
    ASSERT_TRUE(res->at(0)->equalsTo(&exp));
    delete res;
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
TEST_F(DeclarableOpsTests10, MirrorPad_SGO_Test_1) {

    NDArray<double> in({1., 1., 1., 1., 1.});
//    NDArray<double> pad('c', {1, 2}, {1., 1.});// = Nd4j.create(new double[]{1, 1}, new long[]{1, 2});
    NDArray<double> pad('c', {1, 2}, {1., 1.});
//    NDArray<double> value(10.0);

    NDArray<double> exp({1., 1., 1., 1., 1., 1., 1.});

    nd4j::ops::mirror_pad<double> op;

    auto res = op.execute({&in, &pad}, {10.0}, {0});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    //res->at(0)->printIndexedBuffer("Mirror pad:");
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
    ASSERT_EQ(res->status() , ND4J_STATUS_OK);
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

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Where_SGO_Test_2) {
    NDArray<double> input({1., 0., 2., 3., 4.});
    //NDArray<double> expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    NDArray<double> exp('c', {4,1}, {0., 2., 3., 4.});

    nd4j::ops::Where<double> op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    NDArray<double>* resA = res->at(0);
//    resA->printIndexedBuffer("Result A");
//    resA->printShapeInfo("ShapeA");
    ASSERT_TRUE(exp.equalsTo(resA));
    ASSERT_TRUE(exp.isSameShape(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Where_SGO_Test_3) {
    NDArray<double> input('c', {5, 1}, {1., 0., 2., 3., 4.});
    //NDArray<double> expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    NDArray<double> exp('c', {4, 2}, {0., 0., 2., 0., 3., 0., 4., 0.});

    nd4j::ops::Where<double> op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    NDArray<double>* resA = res->at(0);
    //resA->printIndexedBuffer("Result A");
    //resA->printShapeInfo("ShapeA");
    ASSERT_TRUE(exp.equalsTo(resA));
    ASSERT_TRUE(exp.isSameShape(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Where_SGO_Test_4) {
    NDArray<double> input('c', {5, 1}, {0., 0., 0., 0., 0.});
    //NDArray<double> expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    NDArray<double> exp('c', {4, 2}, {0., 0., 2., 0., 3., 0., 4., 0.});

    nd4j::ops::Where<double> op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    NDArray<double>* resA = res->at(0);
    ASSERT_TRUE(resA->isEmpty());
    //resA->printIndexedBuffer("Result A");
    //resA->printShapeInfo("ShapeA");
    //ASSERT_TRUE(exp.equalsTo(resA));
    //ASSERT_TRUE(exp.isSameShape(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, WhereNP_SGO_Test_4) {
    NDArray<double> input('c', {5, 1}, {0., 0., 0., 0., 0.});
    //NDArray<double> expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    NDArray<double> exp('c', {4, 2}, {0., 0., 2., 0., 3., 0., 4., 0.});

    nd4j::ops::where_np<double> op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    NDArray<double>* resA = res->at(0);
    ASSERT_TRUE(resA->isEmpty());
    //resA->printIndexedBuffer("Result A");
    //resA->printShapeInfo("ShapeA");
    //ASSERT_TRUE(exp.equalsTo(resA));
    //ASSERT_TRUE(exp.isSameShape(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
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

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, TestMarixBandPart_Test_1) {

    NDArray<double> x('c', {2, 3, 3});

    NDArray<double> exp('c', {2, 3, 3});
    x.linspace(1);
    exp.linspace(1);
    exp(0, 0, 2)  = 0.;
    exp(1, 0, 2)  = 0.;
    exp(0, 2, 0)  = 0.;
    exp(1, 2, 0)  = 0.;

    nd4j::ops::matrix_band_part<double> op;
    nd4j::ResultSet<double>* results = op.execute({&x}, {}, {1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());
    //results->at(0)->printIndexedBuffer("MBP Test1");
    //exp.printIndexedBuffer("MBP Expec");
    ASSERT_TRUE(exp.equalsTo(results->at(0)));

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

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, split_test4) {
    
    NDArray<float> input('c', {10},{1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f});
    NDArray<float> axis(-1);
    NDArray<float> exp1('c', {5}, {1.f,2.f,3.f,4.f,5.f});
    NDArray<float> exp2('c', {5}, {6.f,7.f,8.f,9.f,10.f});
                                            
    nd4j::ops::split<float> op;
    auto results = op.execute({&input, &axis}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *out1 = results->at(0);
    NDArray<float> *out2 = results->at(1);

    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, split_test5) {
    
    NDArray<float> input('c', {3,8},{1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f,11.f,12.f,13.f,14.f,15.f,16.f,17.f,18.f,19.f,20.f,21.f,22.f,23.f,24.f});
    NDArray<float> exp1('c', {3,4}, {1.f,2.f,3.f,4.f, 9.f,10.f,11.f,12.f, 17.f,18.f,19.f,20.f});
    NDArray<float> exp2('c', {3,4}, {5.f,6.f,7.f,8.f, 13.f,14.f,15.f,16.f, 21.f,22.f,23.f,24.f});
                                            
    nd4j::ops::split<float> op;
    auto results = op.execute({&input}, {}, {2,-1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *out1 = results->at(0);
    NDArray<float> *out2 = results->at(1);

    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, histogram_fixed_width_test1) {
    
    NDArray<float> input('c', {2,3},{-1.f, 0.f, 1.5f, 2.f, 5.f, 15.f});
    NDArray<float> range('c', {2}, {0.f, 5.f});
    NDArray<float> exp('c', {5}, {2.f, 1.f, 1.f, 0.f, 2.f});
                                            
    nd4j::ops::histogram_fixed_width<float> op;
    auto results = op.execute({&input, &range}, {}, {5});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *out = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, histogram_fixed_width_test2) {
    
    NDArray<float> input('c', {2,3,4},{0.f, 5.f, 2.f, 1.f, -1.f, 2.f, 5.f, 3.f, 2.f, 3.f, -1.f, 5.f, 3.f, 2.f, 1.f, 4.f, 2.f, 5.f, 5.f, 5.f, 6.f, 6.f, -1.f, 0.f});
    NDArray<float> range('c', {2}, {0.f, 5.f});
    NDArray<float> exp('c', {5}, {5.f, 2.f, 5.f, 3.f, 9.f});
                                            
    nd4j::ops::histogram_fixed_width<float> op;
    auto results = op.execute({&input, &range}, {}, {5});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *out = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, histogram_fixed_width_test3) {
    
    NDArray<float> input('c', {2,3,1,4,1},{0.f, 5.f, 2.001f, 1.f, -1.f, 2.f, 5.f, 3.f, 2.999f, 3.00001f, -1.f, 3.99999f, 3.f, 2.f, 1.f, 4.f, 2.f, 5.f, 5.f, 5.f, 6.f, 6.f, -1.f, 0.00001f});
    NDArray<float> range('c', {1,2,1}, {0.f, 5.f});
    NDArray<float> exp('c', {5}, {5.f, 2.f, 5.f, 4.f, 8.f});
                                            
    nd4j::ops::histogram_fixed_width<float> op;
    auto results = op.execute({&input, &range}, {}, {5});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *out = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, histogram_fixed_width_test4) {
    
    NDArray<float> input('c', {20,5},{13.8387f,0.1509f,50.39f,30.403f,13.5174f,9.7351f,37.6652f,28.9215f,22.7011f,45.2834f,40.7628f,50.4995f,26.8003f,27.479f,44.633f,6.9109f,48.5004f,
                                      46.5971f,1.6203f,23.6381f,38.9661f,50.8146f,17.2482f,8.0429f,7.5666f,7.9709f,21.8403f,20.1694f,23.3004f,50.9151f,46.239f,38.7323f,29.6946f,32.9876f,
                                      23.0013f,39.7318f,19.4486f,37.6147f,-0.1506f,5.3246f,3.6173f,24.2573f,4.3941f,9.7105f,24.0364f,35.3681f,17.7805f,35.7681f,16.4144f,17.4362f,8.4987f,
                                      26.8108f,36.2937f,31.6442f,29.7221f,8.7445f,33.3301f,4.0939f,13.078f,45.1481f,29.0172f,21.6548f,35.408f,27.1861f,2.2576f,40.6804f,36.2201f,29.7352f,
                                      29.1244f,38.7444f,5.8721f,33.5983f,48.2694f,34.4161f,19.7148f,13.8085f,13.6075f,22.5042f,37.8002f,50.0543f,48.5314f,20.3694f,28.5042f,-0.4679f,4.4245f,
                                      18.9837f,40.7724f,2.7611f,44.0431f,37.186f,27.7361f,14.6001f,9.1721f,14.6087f,21.4072f,49.3344f,11.4668f,14.6171f,15.2502f,5.244f});
    NDArray<float> range('c', {1,2}, {0.00001f, 50.00001f});
    NDArray<float> exp('c', {5}, {22.f, 17.f, 24.f, 19.f, 18.f});
                                            
    nd4j::ops::histogram_fixed_width<float> op;
    auto results = op.execute({&input, &range}, {}, {5});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *out = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, histogram_fixed_width_test5) {
    
    NDArray<float> input('c', {5,20},{20.f, 0.f, 60.f, 40.f, 20.f, 0.f, 40.f, 0.f, 40.f, 40.f,40.f,60.f, 20.f, 20.f, 60.f, 0.f, 40.f,
                                      46.5971f,1.6203f,23.6381f,38.9661f,50.8146f,17.2482f,8.0429f,7.5666f,7.9709f,21.8403f,20.1694f,23.3004f,50.9151f,46.239f,38.7323f,29.6946f,32.9876f,
                                      23.0013f,39.7318f,19.4486f,37.6147f,-0.1506f,5.3246f,3.6173f,24.2573f,4.3941f,9.7105f,24.0364f,35.3681f,17.7805f,35.7681f,16.4144f,17.4362f,8.4987f,
                                      26.8108f,36.2937f,31.6442f,29.7221f,8.7445f,33.3301f,4.0939f,13.078f,45.1481f,29.0172f,21.6548f,35.408f,27.1861f,2.2576f,40.6804f,36.2201f,29.7352f,
                                      29.1244f,38.7444f,5.8721f,33.5983f,48.2694f,34.4161f,19.7148f,13.8085f,13.6075f,22.5042f,37.8002f,50.0543f,48.5314f,20.3694f,28.5042f,-0.4679f,4.4245f,
                                      18.9837f,40.7724f,2.7611f,44.0431f,37.186f,27.7361f,14.6001f,9.1721f,14.6087f,21.4072f,49.3344f,11.4668f,14.6171f,15.2502f,5.244f});
    NDArray<float> range('c', {1,2}, {0.00001f, 50.00001f});
    NDArray<float> exp('c', {5}, {23.f, 19.f, 20.f, 23.f, 15.f});
                                            
    nd4j::ops::histogram_fixed_width<float> op;
    auto results = op.execute({&input, &range}, {}, {5});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *out = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test1) {
    
    NDArray<float> input('c', {3});
    NDArray<float> shape('c', {2}, {3.f, 3.f});
    NDArray<float> exp('c', {3,3}, {1.f, 2.f, 3.f,1.f, 2.f, 3.f,1.f, 2.f, 3.f});

    input.linspace(1.f);
                                            
    nd4j::ops::broadcast_to<float> op;
    auto results = op.execute({&input, &shape}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test2) {
    
    NDArray<float> input('c', {1,3});
    NDArray<float> shape('c', {2}, {3.f, 3.f});
    NDArray<float> exp('c', {3,3}, {1.f, 2.f, 3.f,1.f, 2.f, 3.f,1.f, 2.f, 3.f});

    input.linspace(1.f);
                                            
    nd4j::ops::broadcast_to<float> op;
    auto results = op.execute({&input, &shape}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test3) {
    
    NDArray<float> input('c', {3,1});
    NDArray<float> shape('c', {2}, {3.f, 3.f});
    NDArray<float> exp('c', {3,3}, {1.f, 1.f, 1.f,2.f, 2.f, 2.f,3.f, 3.f, 3.f});

    input.linspace(1.f);
                                            
    nd4j::ops::broadcast_to<float> op;
    auto results = op.execute({&input, &shape}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test4) {
    
    NDArray<float> input(10.f);
    NDArray<float> shape('c', {2}, {3.f, 3.f});
    NDArray<float> exp('c', {3,3}, {10.f, 10.f, 10.f,10.f, 10.f, 10.f, 10.f, 10.f, 10.f});    
                                            
    nd4j::ops::broadcast_to<float> op;
    auto results = op.execute({&input, &shape}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test5) {
    
    NDArray<float> input(10.f);
    NDArray<float> shape('c', {1}, {3.f});
    NDArray<float> exp('c', {3}, {10.f, 10.f, 10.f});    
                                            
    nd4j::ops::broadcast_to<float> op;
    auto results = op.execute({&input, &shape}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test6) {
    
    NDArray<float> input(10.f);
    NDArray<float> shape(1.f);
    NDArray<float> exp('c', {1}, {10.f});    
                                            
    nd4j::ops::broadcast_to<float> op;
    auto results = op.execute({&input, &shape}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test7) {
    
    NDArray<float> input(10.f);
    NDArray<float> shape(0.f);
    NDArray<float> exp(10.f);    
                                            
    nd4j::ops::broadcast_to<float> op;
    auto results = op.execute({&input, &shape}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test8) {
    
    NDArray<float> input('c', {3});
    NDArray<float> shape('c', {3}, {1.f, 3.f, 3.f});
    NDArray<float> exp('c', {1,3,3}, {1.f, 2.f, 3.f,1.f, 2.f, 3.f,1.f, 2.f, 3.f});

    input.linspace(1.f);
                                            
    nd4j::ops::broadcast_to<float> op;
    auto results = op.execute({&input, &shape}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test9) {
    
    NDArray<float> input('c', {5,1,1});
    NDArray<float> shape('c', {5}, {2.f,1.f,5.f,1.f,3.f});
    NDArray<float> exp('c', {2,1,5,1,3}, {1.f, 1.f, 1.f,2.f, 2.f, 2.f,3.f, 3.f, 3.f,4.f, 4.f, 4.f,5.f, 5.f, 5.f,
                                          1.f, 1.f, 1.f,2.f, 2.f, 2.f,3.f, 3.f, 3.f,4.f, 4.f, 4.f,5.f, 5.f, 5.f});
    input.linspace(1.f);
                                            
    nd4j::ops::broadcast_to<float> op;
    auto results = op.execute({&input, &shape}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test10) {
    
    NDArray<float> input('c', {5,1,3});
    NDArray<float> shape('c', {5}, {2.f,1.f,5.f,1.f,3.f});
    NDArray<float> exp('c', {2,1,5,1,3}, {1.f,  2.f,  3.f, 4.f,  5.f,  6.f, 7.f,  8.f,  9.f,10.f, 11.f, 12.f,13.f, 14.f, 15.f,
                                          1.f,  2.f,  3.f, 4.f,  5.f,  6.f, 7.f,  8.f,  9.f,10.f, 11.f, 12.f,13.f, 14.f, 15.f});
    input.linspace(1.f);
                                            
    nd4j::ops::broadcast_to<float> op;
    auto results = op.execute({&input, &shape}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, deconv3d_test1) {

    int bS=2, iD=4,iH=4,iW=4,  iC=2,oC=3,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=3,oH=3,oW=3;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, oD, oH, oW, oC});
    NDArray<double> weights ('c', {kD, kH, kW, iC, oC});    
    NDArray<double> exp('c', {bS, iD, iH, iW, iC}, {0.3 , 0.75, 1.5 , 2.4 , 1.5 , 2.4 , 1.2 , 1.65, 2.4 , 3.3 , 6.6 , 8.4 , 6.6 , 8.4 , 4.2 , 5.1 , 2.4 , 3.3 , 6.6 , 8.4 , 6.6 , 8.4 , 4.2 , 5.1 , 2.1 , 2.55, 5.1 , 6.  , 5.1 , 6.  , 3.  , 3.45,
                                                    4.2 , 5.1 ,10.2 ,12.  ,10.2 ,12.  , 6.  , 6.9 ,12.  ,13.8 ,27.6 ,31.2 ,27.6 ,31.2 ,15.6 ,17.4 ,12.  ,13.8 ,27.6 ,31.2 ,27.6 ,31.2 ,15.6 ,17.4 , 7.8 , 8.7 ,17.4 ,19.2 ,17.4 ,19.2 , 9.6 ,10.5 ,
                                                    4.2 , 5.1 ,10.2 ,12.  ,10.2 ,12.  , 6.  , 6.9 ,12.  ,13.8 ,27.6 ,31.2 ,27.6 ,31.2 ,15.6 ,17.4 ,12.  ,13.8 ,27.6 ,31.2 ,27.6 ,31.2 ,15.6 ,17.4 , 7.8 , 8.7 ,17.4 ,19.2 ,17.4 ,19.2 , 9.6 ,10.5 ,
                                                    3.9 , 4.35, 8.7 , 9.6 , 8.7 , 9.6 , 4.8 , 5.25, 9.6 ,10.5 ,21.  ,22.8 ,21.  ,22.8 ,11.4 ,12.3 , 9.6 ,10.5 ,21.  ,22.8 ,21.  ,22.8 ,11.4 ,12.3 , 5.7 , 6.15,12.3 ,13.2 ,12.3 ,13.2 , 6.6 , 7.05,
                                                    0.3 , 0.75, 1.5 , 2.4 , 1.5 , 2.4 , 1.2 , 1.65, 2.4 , 3.3 , 6.6 , 8.4 , 6.6 , 8.4 , 4.2 , 5.1 , 2.4 , 3.3 , 6.6 , 8.4 , 6.6 , 8.4 , 4.2 , 5.1 , 2.1 , 2.55, 5.1 , 6.  , 5.1 , 6.  , 3.  , 3.45,
                                                    4.2 , 5.1 ,10.2 ,12.  ,10.2 ,12.  , 6.  , 6.9 ,12.  ,13.8 ,27.6 ,31.2 ,27.6 ,31.2 ,15.6 ,17.4 ,12.  ,13.8 ,27.6 ,31.2 ,27.6 ,31.2 ,15.6 ,17.4 , 7.8 , 8.7 ,17.4 ,19.2 ,17.4 ,19.2 , 9.6 ,10.5 ,
                                                    4.2 , 5.1 ,10.2 ,12.  ,10.2 ,12.  , 6.  , 6.9 ,12.  ,13.8 ,27.6 ,31.2 ,27.6 ,31.2 ,15.6 ,17.4 ,12.  ,13.8 ,27.6 ,31.2 ,27.6 ,31.2 ,15.6 ,17.4 , 7.8 , 8.7 ,17.4 ,19.2 ,17.4 ,19.2 , 9.6 ,10.5 ,
                                                    3.9 , 4.35, 8.7 , 9.6 , 8.7 , 9.6 , 4.8 , 5.25, 9.6 ,10.5 ,21.  ,22.8 ,21.  ,22.8 ,11.4 ,12.3 , 9.6 ,10.5 ,21.  ,22.8 ,21.  ,22.8 ,11.4 ,12.3 , 5.7 , 6.15,12.3 ,13.2 ,12.3 ,13.2 , 6.6 , 7.05});
    input = 0.5;
    weights.linspace(0.1, 0.1);

    nd4j::ops::deconv3d<double> op;
    ResultSet<double>* results = op.execute({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    NDArray<double>* output = results->at(0);            

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    
    delete results;
}
  
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, deconv3d_test2) {

    int bS=2, iD=4,iH=4,iW=4,  iC=2,oC=3,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=4,oH=4,oW=4;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, oD, oH, oW, oC});
    NDArray<double> weights ('c', {kD, kH, kW, iC, oC});    
    NDArray<double> exp('c', {bS, iD, iH, iW, iC}, {0.3 ,  0.75, 1.5 ,  2.4 , 1.5 ,  2.4 , 1.5 ,  2.4 , 2.4 ,  3.3 , 6.6 ,  8.4 , 6.6 ,  8.4 , 6.6 ,  8.4 , 2.4 ,  3.3 , 6.6 ,  8.4 , 6.6 ,  8.4 , 6.6 ,  8.4 , 2.4 ,  3.3 , 6.6 ,  8.4 , 6.6 ,  8.4 , 6.6 ,  8.4 ,
                                                    4.2 ,  5.1 ,10.2 , 12.  ,10.2 , 12.  ,10.2 , 12.  ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,
                                                    4.2 ,  5.1 ,10.2 , 12.  ,10.2 , 12.  ,10.2 , 12.  ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,
                                                    4.2 ,  5.1 ,10.2 , 12.  ,10.2 , 12.  ,10.2 , 12.  ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,
                                                    0.3 ,  0.75, 1.5 ,  2.4 , 1.5 ,  2.4 , 1.5 ,  2.4 , 2.4 ,  3.3 , 6.6 ,  8.4 , 6.6 ,  8.4 , 6.6 ,  8.4 , 2.4 ,  3.3 , 6.6 ,  8.4 , 6.6 ,  8.4 , 6.6 ,  8.4 , 2.4 ,  3.3 , 6.6 ,  8.4 , 6.6 ,  8.4 , 6.6 ,  8.4 ,
                                                    4.2 ,  5.1 ,10.2 , 12.  ,10.2 , 12.  ,10.2 , 12.  ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,
                                                    4.2 ,  5.1 ,10.2 , 12.  ,10.2 , 12.  ,10.2 , 12.  ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,
                                                    4.2 ,  5.1 ,10.2 , 12.  ,10.2 , 12.  ,10.2 , 12.  ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 ,12.  , 13.8 ,27.6 , 31.2 ,27.6 , 31.2 ,27.6 , 31.2 });
    input = 0.5;
    weights.linspace(0.1, 0.1);

    nd4j::ops::deconv3d<double> op;
    ResultSet<double>* results = op.execute({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    NDArray<double>* output = results->at(0);            

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    
    delete results;
}
  
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, deconv3d_test3) {

    int bS=2, iD=4,iH=4,iW=4,  iC=2,oC=3,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=3,oH=3,oW=3;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, oC, oD, oH, oW});
    NDArray<double> weights ('c', {oC, iC, kD, kH, kW});
    NDArray<double> exp('c', {bS, iC, iD, iH, iW}, {2.55,  5.25,  5.25,  2.7, 5.4 , 11.1 , 11.1 ,  5.7, 5.4 , 11.1 , 11.1 ,  5.7, 2.85,  5.85,  5.85,  3. , 5.7 , 11.7 , 11.7 ,  6. ,12.  , 24.6 , 24.6 , 12.6,12.  , 24.6 , 24.6 , 12.6, 6.3 , 12.9 , 12.9 ,  6.6,
                                                    5.7 , 11.7 , 11.7 ,  6. ,12.  , 24.6 , 24.6 , 12.6,12.  , 24.6 , 24.6 , 12.6, 6.3 , 12.9 , 12.9 ,  6.6, 3.15,  6.45,  6.45,  3.3, 6.6 , 13.5 , 13.5 ,  6.9, 6.6 , 13.5 , 13.5 ,  6.9, 3.45,  7.05,  7.05,  3.6,
                                                    3.75,  7.65,  7.65,  3.9, 7.8 , 15.9 , 15.9 ,  8.1, 7.8 , 15.9 , 15.9 ,  8.1, 4.05,  8.25,  8.25,  4.2, 8.1 , 16.5 , 16.5 ,  8.4,16.8 , 34.2 , 34.2 , 17.4,16.8 , 34.2 , 34.2 , 17.4, 8.7 , 17.7 , 17.7 ,  9. ,
                                                    8.1 , 16.5 , 16.5 ,  8.4,16.8 , 34.2 , 34.2 , 17.4,16.8 , 34.2 , 34.2 , 17.4, 8.7 , 17.7 , 17.7 ,  9. , 4.35,  8.85,  8.85,  4.5, 9.  , 18.3 , 18.3 ,  9.3, 9.  , 18.3 , 18.3 ,  9.3, 4.65,  9.45,  9.45,  4.8,
                                                    2.55,  5.25,  5.25,  2.7, 5.4 , 11.1 , 11.1 ,  5.7, 5.4 , 11.1 , 11.1 ,  5.7, 2.85,  5.85,  5.85,  3. , 5.7 , 11.7 , 11.7 ,  6. ,12.  , 24.6 , 24.6 , 12.6,12.  , 24.6 , 24.6 , 12.6, 6.3 , 12.9 , 12.9 ,  6.6,
                                                    5.7 , 11.7 , 11.7 ,  6. ,12.  , 24.6 , 24.6 , 12.6,12.  , 24.6 , 24.6 , 12.6, 6.3 , 12.9 , 12.9 ,  6.6, 3.15,  6.45,  6.45,  3.3, 6.6 , 13.5 , 13.5 ,  6.9, 6.6 , 13.5 , 13.5 ,  6.9, 3.45,  7.05,  7.05,  3.6,
                                                    3.75,  7.65,  7.65,  3.9, 7.8 , 15.9 , 15.9 ,  8.1, 7.8 , 15.9 , 15.9 ,  8.1, 4.05,  8.25,  8.25,  4.2, 8.1 , 16.5 , 16.5 ,  8.4,16.8 , 34.2 , 34.2 , 17.4,16.8 , 34.2 , 34.2 , 17.4, 8.7 , 17.7 , 17.7 ,  9. ,
                                                    8.1 , 16.5 , 16.5 ,  8.4,16.8 , 34.2 , 34.2 , 17.4,16.8 , 34.2 , 34.2 , 17.4, 8.7 , 17.7 , 17.7 ,  9. , 4.35,  8.85,  8.85,  4.5, 9.  , 18.3 , 18.3 ,  9.3, 9.  , 18.3 , 18.3 ,  9.3, 4.65,  9.45,  9.45,  4.8});
    input = 0.5;
    weights.linspace(0.1, 0.1);
    weights.permutei({2, 3, 4, 1, 0});

    nd4j::ops::deconv3d<double> op;
    ResultSet<double>* results = op.execute({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    NDArray<double>* output = results->at(0);            

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, deconv3d_test4) {

    int bS=2, iD=2,iH=2,iW=2,  iC=2,oC=3,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=1,pH=1,pW=1,  dD=1,dH=1,dW=1;
    int       oD=3,oH=3,oW=3;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, oC, oD, oH, oW});
    NDArray<double> weights ('c', {oC, iC, kD, kH, kW});
    NDArray<double> exp('c', {bS, iC, iD, iH, iW}, {24.6, 24.6,24.6, 24.6,24.6, 24.6,24.6, 24.6,34.2, 34.2,34.2, 34.2,34.2, 34.2,34.2, 34.2,24.6, 24.6,24.6, 24.6,
                                                    24.6, 24.6,24.6, 24.6,34.2, 34.2,34.2, 34.2,34.2, 34.2,34.2, 34.2});
    input = 0.5;
    weights.linspace(0.1, 0.1);
    weights.permutei({2, 3, 4, 1, 0});

    nd4j::ops::deconv3d<double> op;
    ResultSet<double>* results = op.execute({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    NDArray<double>* output = results->at(0);            

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, deconv3d_bp_test1) {

    int bS=1, iD=3,iH=3,iW=3,  iC=1,oC=2,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, oD, oH, oW, oC});
    NDArray<double> weights ('c', {kD, kH, kW, iC, oC});
    NDArray<double> bias    ('c', {iC});
    NDArray<double> gradO   ('c', {bS, iD, iH, iW, iC});
    
    input = 0.5;
    weights.linspace(0.1, 0.1);
    gradO.linspace(0.5);

    const OpArgsHolder<double> argsHolderFF({&input, &weights, &bias},         {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    const OpArgsHolder<double> argsHolderBP({&input, &weights, &bias, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});

    nd4j::ops::deconv3d<double> opFF;
    nd4j::ops::deconv3d_bp<double> opBP;  

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, deconv3d_bp_test2) {

    int bS=1, iD=2,iH=2,iW=2,  iC=1,oC=2,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=2,oH=2,oW=2;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, oD, oH, oW, oC});
    NDArray<double> weights ('c', {kD, kH, kW, iC, oC});
    NDArray<double> gradO   ('c', {bS, iD, iH, iW, iC});
    
    input = 0.5;
    weights.linspace(0.1, 0.1);
    gradO.linspace(0.5);

    const OpArgsHolder<double> argsHolderFF({&input, &weights},         {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    const OpArgsHolder<double> argsHolderBP({&input, &weights, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});

    nd4j::ops::deconv3d<double> opFF;
    nd4j::ops::deconv3d_bp<double> opBP;  

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, deconv3d_bp_test3) {

    int bS=1, iD=3,iH=3,iW=3,  iC=1,oC=2,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int       oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, oC, oD, oH, oW});
    NDArray<double> weights ('c', {oC, iC, kD, kH, kW});
    NDArray<double> gradO   ('c', {bS, iC, iD, iH, iW});
    
    input = 0.5;
    weights.linspace(0.1, 0.1);
    gradO.linspace(0.5);
    weights.permutei({2, 3, 4, 1, 0});

    const OpArgsHolder<double> argsHolderFF({&input, &weights},         {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    const OpArgsHolder<double> argsHolderBP({&input, &weights, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});

    nd4j::ops::deconv3d<double> opFF;
    nd4j::ops::deconv3d_bp<double> opBP;  

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, deconv3d_bp_test4) {

    int bS=1, iD=2,iH=2,iW=2,  iC=1,oC=2,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=1,pH=1,pW=1,  dD=1,dH=1,dW=1;
    int       oD=3,oH=3,oW=3;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, oC, oD, oH, oW});
    NDArray<double> weights ('c', {oC, iC, kD, kH, kW});
    NDArray<double> gradO   ('c', {bS, iC, iD, iH, iW});
    
    input = 0.5;
    weights.linspace(0.1, 0.1);
    gradO.linspace(0.5);
    weights.permutei({2, 3, 4, 1, 0});

    const OpArgsHolder<double> argsHolderFF({&input, &weights},         {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    const OpArgsHolder<double> argsHolderBP({&input, &weights, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});

    nd4j::ops::deconv3d<double> opFF;
    nd4j::ops::deconv3d_bp<double> opBP;  

    const bool isGradCorrect = GradCheck::checkGrad(opFF, opBP, argsHolderFF, argsHolderBP);

    ASSERT_TRUE(isGradCorrect);
}
