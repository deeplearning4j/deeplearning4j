/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

template <typename T>
class TypedDeclarableOpsTests10 : public testing::Test {
public:

    TypedDeclarableOpsTests10() {
        printf("\n");
        fflush(stdout);
    }
};

typedef ::testing::Types<double, float> TestingTypes;
TYPED_TEST_CASE(TypedDeclarableOpsTests10, TestingTypes);

TEST_F(DeclarableOpsTests10, Test_ArgMax_1) {
    auto x = NDArrayFactory::create<double>('c', {3, 3});
    auto e = NDArrayFactory::create<Nd4jLong>(8);

    x.linspace(1.0);


    nd4j::ops::argmax op;
    auto result = op.execute({&x}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());


    auto z = *result->at(0);

    ASSERT_EQ(e, z);

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_ArgMax_2) {
    auto x = NDArrayFactory::create<double>('c', {3, 3});
    auto y = NDArrayFactory::create<int>('c', {1}, {1});
    auto e = NDArrayFactory::create<Nd4jLong>('c', {3}, {2, 2, 2});

    x.linspace(1.0);

    nd4j::ops::argmax op;
    auto result = op.execute({&x, &y}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = *result->at(0);

    //z.printIndexedBuffer("z");
    //z.printShapeInfo("z shape");

    ASSERT_EQ(e, z);

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_And_1) {
    auto x = NDArrayFactory::create<double>('c', {4}, {1, 1, 0, 1});
    auto y = NDArrayFactory::create<double>('c', {4}, {0, 0, 0, 1});
    auto e = NDArrayFactory::create<double>('c', {4}, {0, 0, 0, 1});

    nd4j::ops::boolean_and op;
    auto result = op.execute({&x, &y}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_Or_1) {
    auto x = NDArrayFactory::create<double>('c', {4}, {1, 1, 0, 1});
    auto y = NDArrayFactory::create<double>('c', {4}, {0, 0, 0, 1});
    auto e = NDArrayFactory::create<double>('c', {4}, {1, 1, 0, 1});

    nd4j::ops::boolean_or op;
    auto result = op.execute({&x, &y}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_Not_1) {
    auto x = NDArrayFactory::create<bool>('c', {4}, {1, 1, 0, 1});
    auto y = NDArrayFactory::create<bool>('c', {4}, {0, 0, 0, 1});
//    auto e = NDArrayFactory::create<bool>('c', {4}, {1, 1, 1, 0});
    auto e = NDArrayFactory::create<bool>('c', {4}, {0, 0, 1, 0});

    nd4j::ops::boolean_not op;
    auto result = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::BOOL);
    ASSERT_EQ(Status::OK(), result->status());
    auto res = result->at(0);

    ASSERT_TRUE(e.equalsTo(res));

    delete result;
}

TEST_F(DeclarableOpsTests10, Test_Size_at_1) {
    auto x = NDArrayFactory::create<double>('c', {10, 20, 30});
    auto e = NDArrayFactory::create<Nd4jLong>(20);

    nd4j::ops::size_at op;
    auto result = op.execute({&x}, {}, {1});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, MirrorPad_SGO_Test_1) {

    auto in = NDArrayFactory::create<double>({1., 2., 3., 4., 5.});
//    auto pad('c', {1, 2}, {1., 1.});// = Nd4j.create(new double[]{1, 1}, new long[]{1, 2});
    auto pad = NDArrayFactory::create<int>('c', {1, 2}, {1, 1});
//    auto value(10.0);

    auto exp = NDArrayFactory::create<double>({2., 1., 2., 3., 4., 5., 4.});

    nd4j::ops::mirror_pad op;

    auto res = op.execute({&in, &pad}, {10.0}, {0}, {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);

    ASSERT_TRUE(exp.equalsTo(res->at(0)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Unique_SGO_Test_1) {
    auto input = NDArrayFactory::create<double>({3., 4., 3., 1., 3., 0., 2., 4., 2., 4.});
    auto expIdx = NDArrayFactory::create<Nd4jLong>({0, 1, 0, 2, 0, 3, 4, 1, 4, 1});
    auto exp = NDArrayFactory::create<double>({3., 4., 1., 0., 2.});

    nd4j::ops::unique op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto res1 = res->at(0);
    auto res2 = res->at(1);

    ASSERT_TRUE(exp.equalsTo(res1));
    ASSERT_TRUE(expIdx.equalsTo(res2));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Where_SGO_Test_1) {
    auto input = NDArrayFactory::create<bool>('c', {3, 3}, {true, false, false, true, true, false, true, true, true});
    //auto expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {6, 2}, {0LL, 0LL, 1LL, 0LL, 1LL, 1LL, 2LL, 0LL, 2LL, 1LL, 2LL, 2LL});

    nd4j::ops::Where op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    auto  resA = res->at(0);

    ASSERT_TRUE(exp.isSameShape(resA));
    ASSERT_TRUE(exp.equalsTo(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Where_SGO_Test_02) {
    auto input = NDArrayFactory::create<bool>('c', {2, 2, 2}, {true, false, false, true, true, true, true, false});
    //auto expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {5, 3}, {0LL, 0LL, 0LL, 0LL, 1LL, 1LL, 1LL, 0LL, 0LL, 1LL, 0LL, 1LL, 1LL, 1LL, 0LL});

    nd4j::ops::Where op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    auto  resA = res->at(0);

    ASSERT_TRUE(exp.equalsTo(resA));
    ASSERT_TRUE(exp.isSameShape(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, WhereNP_SGO_Test_1) {
    auto cond3d = NDArrayFactory::create<bool>('c', {2, 2, 2}, {true, false, false, true, true, true, true, false});
//    auto expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    auto exp1 = NDArrayFactory::create<Nd4jLong>({0, 0, 1, 1, 1});
    auto exp2 = NDArrayFactory::create<Nd4jLong>({0, 1, 0, 0, 1});
    auto exp3 = NDArrayFactory::create<Nd4jLong>({0, 1, 0, 1, 0});
    nd4j::ops::where_np op;
    auto res = op.execute({&cond3d}, {}, {});
    ASSERT_TRUE(res->size() == 3);
    ASSERT_EQ(res->status(), ND4J_STATUS_OK);
    auto res1 = res->at(0);
    auto res2 = res->at(1);
    auto res3 = res->at(2);
//    res1->printShapeInfo("Res1 shape"); res1->printBuffer("Res1");
//    res2->printShapeInfo("Res2 shape"); res2->printBuffer("Res2");
//    res3->printShapeInfo("Res3 shape"); res3->printBuffer("Res3");
    ASSERT_TRUE(exp1.equalsTo(res1));
    ASSERT_TRUE(exp2.equalsTo(res2));
    ASSERT_TRUE(exp3.equalsTo(res3));
    //ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, WhereNP_SGO_Test_2) {
    auto cond2d = NDArrayFactory::create<bool>('c', {3, 5}, {1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1});
//    auto expIdx({0, 1, 0, 2, 0, 3, 4, 1, 4, 1});
    auto exp1 = NDArrayFactory::create<Nd4jLong>({0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2});
    auto exp2 = NDArrayFactory::create<Nd4jLong>({0, 1, 4, 0, 1, 2, 3, 4, 1, 2, 3, 4});
    nd4j::ops::where_np op;
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
    auto input = NDArrayFactory::create<bool>({true, false, true, true, true});
    //auto expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {4,1}, {0, 2, 3, 4});

    nd4j::ops::Where op;
    auto res = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::INT64);
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    auto resA = res->at(0);
//    resA->printIndexedBuffer("Result A");
//    resA->printShapeInfo("ShapeA");
    ASSERT_TRUE(exp.equalsTo(resA));
    ASSERT_TRUE(exp.isSameShape(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Where_SGO_Test_3) {
    auto input = NDArrayFactory::create<bool>('c', {5, 1}, {true, false, true, true, true});
    //auto expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {4, 2}, {0, 0, 2, 0, 3, 0, 4, 0});

    nd4j::ops::Where op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    auto resA = res->at(0);
    //resA->printIndexedBuffer("Result A");
    //resA->printShapeInfo("ShapeA");
    ASSERT_TRUE(exp.equalsTo(resA));
    ASSERT_TRUE(exp.isSameShape(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Where_SGO_Test_4) {
    auto input = NDArrayFactory::create<bool>('c', {5, 1}, {false, false, false, false, false});
    //auto expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {4, 2}, {0, 0, 2, 0, 3, 0, 4, 0});

    nd4j::ops::Where op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    auto resA = res->at(0);
    ASSERT_TRUE(resA->isEmpty());
    //resA->printIndexedBuffer("Result A");
    //resA->printShapeInfo("ShapeA");
    //ASSERT_TRUE(exp.equalsTo(resA));
    //ASSERT_TRUE(exp.isSameShape(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Where_SGO_Test_5) {
    auto input = NDArrayFactory::create<float>('c', {5}, {1, 0, 0, 2, 3});
    //auto expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {3, 1}, {0, 3, 4});

    nd4j::ops::Where op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    auto resA = res->at(0);
    //ASSERT_TRUE(resA->isEmpty());

    ASSERT_TRUE(exp.equalsTo(resA));
    ASSERT_TRUE(exp.isSameShape(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, WhereNP_SGO_Test_4) {
    auto input = NDArrayFactory::create<bool>('c', {5, 1}, {false, false, false, false, false});
    //auto expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {4, 2}, {0, 0, 2, 0, 3, 0, 4, 0});

    nd4j::ops::where_np op;
    auto res = op.execute({&input}, {}, {});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    auto resA = res->at(0);
    ASSERT_TRUE(resA->isEmpty());
    //resA->printIndexedBuffer("Result A");
    //resA->printShapeInfo("ShapeA");
    //ASSERT_TRUE(exp.equalsTo(resA));
    //ASSERT_TRUE(exp.isSameShape(resA));
//    ASSERT_TRUE(expIdx.equalsTo(res->at(1)));
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, CosineDistance_SGO_Test_1) {
    auto labels = NDArrayFactory::create<double>('c', {2, 3}, {1.0, 2.0, 3.0, -1.0, 2.0, 1.0});
    //auto expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    auto predictions = NDArrayFactory::create<double>('c', {2, 3}, {-0.3, -0.2, -0.1, 0, 0.1, 0.2});
    auto weights = NDArrayFactory::create<double>('c', {2, 1}, {0., 1.});
    auto exp = NDArrayFactory::create<double>(0.6);

    nd4j::ops::cosine_distance_loss op;
    auto res = op.execute({&predictions, &weights, &labels}, {}, {3, 1});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    auto resA = res->at(0);

    ASSERT_TRUE(exp.equalsTo(resA));

    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, CosineDistance_SGO_Test_2) {
    auto labels = NDArrayFactory::create<double>('c', {2, 3}, {1.0, 2.0, 3.0, -1.0, 2.0, 1.0});
    //auto expIdx({0., 1., 0., 2., 0., 3., 4., 1., 4., 1.});
    auto predictions = NDArrayFactory::create<double>('c', {2, 3}, {-0.3, -0.2, -0.1, 0, 0.1, 0.2});
    auto weights = NDArrayFactory::create<double>('c', {2, 1}, {0., 1.});
    auto exp = NDArrayFactory::create<double>(0.6);

    nd4j::ops::cosine_distance_loss op;
    auto res = op.execute({&predictions, &weights, &labels}, {}, {2, 1});
    ASSERT_TRUE(res->status() == ND4J_STATUS_OK);
    auto resA = res->at(0);

    ASSERT_TRUE(exp.equalsTo(resA));

    delete res;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, TestMarixBandPart_Test_1) {

    auto x = NDArrayFactory::create<double>('c', {2, 3, 3});

    auto exp = NDArrayFactory::create<double>('c', {2, 3, 3});
    x.linspace(1);
    exp.linspace(1);
    exp.p(0, 0, 2, 0.);
    exp.p(1, 0, 2, 0.);
    exp.p(0, 2, 0, 0.);
    exp.p(1, 2, 0, 0.);

    nd4j::ops::matrix_band_part op;
    auto results = op.execute({&x}, {}, {1, 1}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());
    //results->at(0)->printIndexedBuffer("MBP Test1");
    //exp.printIndexedBuffer("MBP Expec");
    ASSERT_TRUE(exp.equalsTo(results->at(0)));

    delete results;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test1) {

    auto y = NDArrayFactory::create<double>('c', {2, 3, 4}, {-1.001 ,-0.915 ,-0.829 ,-0.743 ,-0.657 ,-0.571 ,-0.485 ,-0.399 ,-0.313 ,-0.227 ,-0.141 ,-0.055 ,0.031 ,0.117 ,0.203 ,0.289 ,0.375 ,0.461 ,0.547 ,0.633 ,0.719 ,0.805 ,0.891 ,0.977});
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {-0.51, -0.46, -0.41, -0.36, -0.31, -0.26, -0.21, -0.16, -0.11, -0.06, -0.01, 0.04, 0.09, 0.14, 0.19, 0.24, 0.29, 0.34, 0.39, 0.44, 0.49, 0.54, 0.59, 0.61});

    auto exp = NDArrayFactory::create<double>('c', {2,3,4}, {-2.04201, -2.03663, -2.03009, -2.02199,-2.01166, -1.99808, -1.97941, -1.95217,-1.90875, -1.8292 , -1.6416 , -0.942  ,
                                       0.33172,  0.69614,  0.81846,  0.87776, 0.91253,  0.93533,  0.95141,  0.96336, 0.97259,  0.97993,  0.98591,  1.01266,});

    nd4j::ops::tf_atan2 op;
    auto result = op.execute({&y, &x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test2) {

    auto y = NDArrayFactory::create<double>('c', {2, 3, 4}, {-1.001 ,-0.915 ,-0.829 ,-0.743 ,-0.657 ,-0.571 ,-0.485 ,-0.399 ,-0.313 ,-0.227 ,-0.141 ,-0.055 ,0.031 ,0.117 ,0.203 ,0.289 ,0.375 ,0.461 ,0.547 ,0.633 ,0.719 ,0.805 ,0.891 ,0.977});
    auto x = NDArrayFactory::create<double>('c', {   3, 4}, {-1.05, -0.82, -0.639, -0.458, -0.277, -0.096, 0.085, 0.266, 0.447, 0.628, 0.809, 0.99});

    auto exp = NDArrayFactory::create<double>('c', {2,3,4}, {-2.38008, -2.30149, -2.22748, -2.1232 ,-1.96979, -1.73736, -1.3973 , -0.98279,-0.61088, -0.34685, -0.17256, -0.0555 ,
                                       3.11208,  2.99987,  2.83399,  2.57869, 2.207  ,  1.77611,  1.41664,  1.17298, 1.01458,  0.90829,  0.8336 ,  0.77879});

    nd4j::ops::tf_atan2 op;
    auto result = op.execute({&y, &x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);
    // z->printIndexedBuffer();

    // x.applyTrueBroadcast(nd4j::BroadcastOpsTuple::custom(scalar::Atan2, pairwise::Atan2, broadcast::Atan2), &y, &z, true);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test3) {

    auto y = NDArrayFactory::create<double>('c', {2, 3, 4}, {-1.001 ,-0.915 ,-0.829 ,-0.743 ,-0.657 ,-0.571 ,-0.485 ,-0.399 ,-0.313 ,-0.227 ,-0.141 ,-0.055 ,0.031 ,0.117 ,0.203 ,0.289 ,0.375 ,0.461 ,0.547 ,0.633 ,0.719 ,0.805 ,0.891 ,0.977});
    auto x = NDArrayFactory::create<double>('c', {   3, 4}, {-1.05, -0.82, -0.639, -0.458, -0.277, -0.096, 0.085, 0.266, 0.447, 0.628, 0.809, 0.99});

    auto exp = NDArrayFactory::create<double>('c', {2,3,4}, {-2.33231, -2.41089, -2.48491, -2.58919,-2.74259, -2.97502,  2.9681 ,  2.55359, 2.18167,  1.91765,  1.74335,  1.62629,
                                       -1.54128, -1.42907, -1.2632 , -1.00789,-0.63621, -0.20531,  0.15416,  0.39782, 0.55622,  0.6625 ,  0.7372 ,  0.79201});

    nd4j::ops::tf_atan2 op;
    auto result = op.execute({&x, &y}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test4) {

    auto y = NDArrayFactory::create<double>('c', {1, 3, 4}, {-1.001 ,-0.829 ,-0.657 ,-0.485 ,-0.313 ,-0.141 ,0.031 ,0.203 ,0.375 ,0.547 ,0.719 ,0.891});
    auto x = NDArrayFactory::create<double>('c', {2, 3, 1}, {-0.82, -0.458, -0.096, 0.085, 0.447, 0.809});

    auto exp = NDArrayFactory::create<double>('c', {2,3,4}, {-2.45527, -2.36165, -2.24628, -2.10492,-2.1703 , -1.86945, -1.50321, -1.15359,-0.25062, -0.17373, -0.13273, -0.10733,
                                        3.05688,  3.03942,  3.01293,  2.9681 , 2.18167,  1.87635,  1.50156,  1.14451, 1.13674,  0.97626,  0.84423,  0.7372 });

    nd4j::ops::tf_atan2 op;
    auto result = op.execute({&x, &y}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test5) {

    auto y = NDArrayFactory::create<double>('c', {1, 3, 4}, {-1.001 ,-0.829 ,-0.657 ,-0.485 ,-0.313 ,-0.141 ,0.031 ,0.203 ,0.375 ,0.547 ,0.719 ,0.891});
    auto x = NDArrayFactory::create<double>('c', {2, 3, 1}, {-0.82, -0.458, -0.096, 0.085, 0.447, 0.809});

    auto exp = NDArrayFactory::create<double>('c', {2,3,4}, {-2.25712, -2.35074, -2.46611, -2.60747,-2.54209, -2.84294,  3.07401,  2.72438, 1.82141,  1.74453,  1.70353,  1.67813,
                                       -1.48608, -1.46862, -1.44214, -1.3973 ,-0.61088, -0.30556,  0.06924,  0.42629, 0.43405,  0.59453,  0.72657,  0.8336 });

    nd4j::ops::tf_atan2 op;
    auto result = op.execute({&y, &x}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, atan2_test6) {

    auto y = NDArrayFactory::create<double>('c', {1, 3, 4}, {-1.001 ,-0.829 ,-0.657 ,-0.485 ,-0.313 ,-0.141 ,0.031 ,0.203 ,0.375 ,0.547 ,0.719 ,0.891});
    auto x = NDArrayFactory::create<double>('c', {      4}, {-0.82, -0.096, 0.085, 0.809});

    auto exp = NDArrayFactory::create<double>('c', {1,3,4}, {-2.25712, -1.68608, -1.44214, -0.54006,-2.77695, -2.16855,  0.34972,  0.24585, 2.71267,  1.74453,  1.45312,  0.8336 });

    nd4j::ops::tf_atan2 op;
    auto result = op.execute({&y, &x}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, IGamma_Test1) {

    auto y = NDArrayFactory::create<double>('c', {1, 3, 4}, {1.1 , 2.1 , 3.1 ,4.1 , 5.1 , 6.1 ,7.1 ,8.1 ,9.1 ,10.1,11.1 ,12.1});
    auto x = NDArrayFactory::create<double>('c', {      4}, {1.2, 2.2, 3.2, 4.2});

    auto exp = NDArrayFactory::create<double>('c', {1,3,4}, {
               0.659917,     0.61757898,  0.59726304,   0.58478117,
           0.0066205109,    0.022211598, 0.040677428,  0.059117373,
        0.0000039433403, 0.000086064574, 0.000436067, 0.0012273735});

    nd4j::ops::igamma op;
    auto result = op.execute({&y, &x}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);
//    z->printBuffer("OUtput");
//    exp.printBuffer("EXpect");
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, IGamma_Test2) {

    auto y = NDArrayFactory::create<double>('c', {1, 3, 4}, {1.1 , 2.1 , 3.1 ,4.1 , 5.1 , 6.1 ,
                                                             7.1 ,8.1 ,9.1 ,10.1,11.1 ,12.1});
    auto x = NDArrayFactory::create<double>('c', {      4}, {1.2, 2.2, 3.2, 4.2});
    auto exp = NDArrayFactory::create<double>('c', {1,3,4}, {0.340083, 0.382421, 0.402737, 0.415221,
                                                             0.993379, 0.977788, 0.959323, 0.940883,
                                                             0.999996, 0.999914, 0.999564, 0.998773});

    nd4j::ops::igammac op;
    auto result = op.execute({&y, &x}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    auto z = result->at(0);
//    z->printBuffer("OUtput");
//    exp.printBuffer("EXpect");
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, range_test10) {

    auto limit = NDArrayFactory::create<double>('c', {1, 3, 4});
    limit = 5.;
    auto exp = NDArrayFactory::create<double>('c', {5}, {0.,1.,2.,3.,4.});

    nd4j::ops::range op;
    auto result = op.execute({&limit}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, range_test11) {

    auto limit = NDArrayFactory::create<double>('c', {1, 3, 4});
    auto start = NDArrayFactory::create<double>('c', {2, 4});
    limit = 5.;
    start = 0.5;
    auto exp = NDArrayFactory::create<double>('c', {5}, {0.5,1.5,2.5,3.5,4.5});

    nd4j::ops::range op;
    auto result = op.execute({&start, &limit}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, range_test12) {

    auto exp = NDArrayFactory::create<float>('c', {9}, {0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5});

    nd4j::ops::range op;
    auto result = op.execute({}, {0.5, 5, 0.5}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, top_k_permuted_test1) {

    auto x = NDArrayFactory::create<double>({7., 3., 1., 2., 5., 0., 4., 6., 9., 8.});
    auto expUnsorted = NDArrayFactory::create<double>({7., 6., 9., 8.}); // Sorted = False
    auto expSorted = NDArrayFactory::create<double>({9., 8., 7., 6., 5.}); // Sorted = False


    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {4}, {false});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto zI = result->at(1);

    ASSERT_TRUE(expUnsorted.isSameShape(z));
    ASSERT_TRUE(expUnsorted.equalsTo(z));

    auto result2 = op.execute({&x}, {}, {5}, {true});

    ASSERT_EQ(ND4J_STATUS_OK, result2->status());

    z = result2->at(0);
    zI = result2->at(1);

    ASSERT_TRUE(expSorted.isSameShape(z));
    ASSERT_TRUE(expSorted.equalsTo(z));

    delete result;
    delete result2;
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, top_k_permuted_test2) {

    auto x = NDArrayFactory::create<double>({7., 3., 1., 2., 5., 0., 4., 6., 9., 8.});
    auto expUnsorted = NDArrayFactory::create<double>({7.,    5.,    6.,    9.,    8.}); // Sorted = False
    auto expSorted = NDArrayFactory::create<double>({9., 8., 7., 6., 5.}); // Sorted = False


    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {5}, {false});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto zI = result->at(1);

    ASSERT_TRUE(expUnsorted.isSameShape(z));
    ASSERT_TRUE(expUnsorted.equalsTo(z));

    auto result2 = op.execute({&x}, {}, {5}, {true});

    ASSERT_EQ(ND4J_STATUS_OK, result2->status());

    z = result2->at(0);
    zI = result2->at(1);

    ASSERT_TRUE(expSorted.isSameShape(z));
    ASSERT_TRUE(expSorted.equalsTo(z));

    delete result;
    delete result2;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, sparse_softmax_cross_entropy_loss_with_logits_test1) {

    auto labels = NDArrayFactory::create<int>('c', {2,3},{3, 2, 1, 0, 1, 2});
    auto logits = NDArrayFactory::create<double>('c', {2,3,4});
    auto expected = NDArrayFactory::create<double>('c', {2,3}, {1.24254, 1.34254, 1.44254, 1.54254, 1.44254, 1.34254});

    logits.linspace(0.1, 0.1);

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&labels, &logits}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, sparse_softmax_cross_entropy_loss_with_logits_test2) {

    auto labels = NDArrayFactory::create<int>('c', {2},{1, 0});
    auto logits = NDArrayFactory::create<double>('c', {2,3});
    auto expected = NDArrayFactory::create<double>('c', {2}, {1.10194, 1.20194});

    logits.linspace(0.1, 0.1);

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&labels, &logits}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, sparse_softmax_cross_entropy_loss_with_logits_test3) {

    NDArray labels('c', {1}, {0}, nd4j::DataType::INT32);
    auto logits = NDArrayFactory::create<double>('c', {1,3});
    auto expected = NDArrayFactory::create<double>('c', {1}, {1.20194});

    logits.linspace(0.1, 0.1);

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&labels, &logits}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, sparse_softmax_cross_entropy_loss_with_logits_test4) {

    auto labels = NDArrayFactory::create<int>('c', {2},{0, 0});
    auto logits = NDArrayFactory::create<double>('c', {2,1});
    auto expected = NDArrayFactory::create<double>('c', {2}, {0., 0.});

    logits.linspace(0.1, 0.1);

    nd4j::ops::sparse_softmax_cross_entropy_loss_with_logits op;
    auto results = op.execute({&labels, &logits}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, split_test4) {

    auto input = NDArrayFactory::create<double>('c', {10},{1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f});
    auto axis = NDArrayFactory::create<double>(-1);
    auto exp1 = NDArrayFactory::create<double>('c', {5}, {1.f,2.f,3.f,4.f,5.f});
    auto exp2 = NDArrayFactory::create<double>('c', {5}, {6.f,7.f,8.f,9.f,10.f});

    nd4j::ops::split op;
    auto results = op.execute({&input, &axis}, {}, {2}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto out1 = results->at(0);
    auto out2 = results->at(1);

    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, split_test5) {

    auto input = NDArrayFactory::create<double>('c', {3,8},{1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f,11.f,12.f,13.f,14.f,15.f,16.f,17.f,18.f,19.f,20.f,21.f,22.f,23.f,24.f});
    auto exp1 = NDArrayFactory::create<double>('c', {3,4}, {1.f,2.f,3.f,4.f, 9.f,10.f,11.f,12.f, 17.f,18.f,19.f,20.f});
    auto exp2 = NDArrayFactory::create<double>('c', {3,4}, {5.f,6.f,7.f,8.f, 13.f,14.f,15.f,16.f, 21.f,22.f,23.f,24.f});

    nd4j::ops::split op;
    auto results = op.execute({&input}, {}, {2,-1},{});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto out1 = results->at(0);
    auto out2 = results->at(1);

    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp1.equalsTo(out1));
    ASSERT_TRUE(exp2.equalsTo(out2));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, histogram_fixed_width_test1) {

    auto input = NDArrayFactory::create<double>('c', {2,3},{-1.f, 0.f, 1.5f, 2.f, 5.f, 15.f});
    auto range = NDArrayFactory::create<double>('c', {2}, {0, 5});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {5}, {2, 1, 1, 0, 2});

    nd4j::ops::histogram_fixed_width op;
    auto results = op.execute({&input, &range}, {}, {5}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto out = results->at(0);

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, histogram_fixed_width_test2) {

    auto input = NDArrayFactory::create<double>('c', {2,3,4},{0.f, 5.f, 2.f, 1.f, -1.f, 2.f, 5.f, 3.f, 2.f, 3.f, -1.f, 5.f, 3.f, 2.f, 1.f, 4.f, 2.f, 5.f, 5.f, 5.f, 6.f, 6.f, -1.f, 0.f});
    auto range = NDArrayFactory::create<double>('c', {2}, {0, 5});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {5}, {5, 2, 5, 3, 9});

    nd4j::ops::histogram_fixed_width op;
    auto results = op.execute({&input, &range}, {}, {5}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto out = results->at(0);

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, histogram_fixed_width_test3) {

    auto input = NDArrayFactory::create<double>('c', {2,3,1,4,1},{0.f, 5.f, 2.001f, 1.f, -1.f, 2.f, 5.f, 3.f, 2.999f, 3.00001f, -1.f, 3.99999f, 3.f, 2.f, 1.f, 4.f, 2.f, 5.f, 5.f, 5.f, 6.f, 6.f, -1.f, 0.00001f});
    auto range = NDArrayFactory::create<double>('c', {1,2,1}, {0, 5});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {5}, {5, 2, 5, 4, 8});

    nd4j::ops::histogram_fixed_width op;
    auto results = op.execute({&input, &range}, {}, {5}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto out = results->at(0);

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, histogram_fixed_width_test4) {

    auto input = NDArrayFactory::create<double>('c', {20,5},{13.8387f,0.1509f,50.39f,30.403f,13.5174f,9.7351f,37.6652f,28.9215f,22.7011f,45.2834f,40.7628f,50.4995f,26.8003f,27.479f,44.633f,6.9109f,48.5004f,
                                      46.5971f,1.6203f,23.6381f,38.9661f,50.8146f,17.2482f,8.0429f,7.5666f,7.9709f,21.8403f,20.1694f,23.3004f,50.9151f,46.239f,38.7323f,29.6946f,32.9876f,
                                      23.0013f,39.7318f,19.4486f,37.6147f,-0.1506f,5.3246f,3.6173f,24.2573f,4.3941f,9.7105f,24.0364f,35.3681f,17.7805f,35.7681f,16.4144f,17.4362f,8.4987f,
                                      26.8108f,36.2937f,31.6442f,29.7221f,8.7445f,33.3301f,4.0939f,13.078f,45.1481f,29.0172f,21.6548f,35.408f,27.1861f,2.2576f,40.6804f,36.2201f,29.7352f,
                                      29.1244f,38.7444f,5.8721f,33.5983f,48.2694f,34.4161f,19.7148f,13.8085f,13.6075f,22.5042f,37.8002f,50.0543f,48.5314f,20.3694f,28.5042f,-0.4679f,4.4245f,
                                      18.9837f,40.7724f,2.7611f,44.0431f,37.186f,27.7361f,14.6001f,9.1721f,14.6087f,21.4072f,49.3344f,11.4668f,14.6171f,15.2502f,5.244f});
    auto range = NDArrayFactory::create<double>('c', {1,2}, {0, 50});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {5}, {22, 17, 24, 19, 18});

    nd4j::ops::histogram_fixed_width op;
    auto results = op.execute({&input, &range}, {}, {5}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto out = results->at(0);

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, histogram_fixed_width_test5) {

    auto input = NDArrayFactory::create<double>('c', {5,20},{20.f, 0.f, 60.f, 40.f, 20.f, 0.f, 40.f, 0.f, 40.f, 40.f,40.f,60.f, 20.f, 20.f, 60.f, 0.f, 40.f,
                                      46.5971f,1.6203f,23.6381f,38.9661f,50.8146f,17.2482f,8.0429f,7.5666f,7.9709f,21.8403f,20.1694f,23.3004f,50.9151f,46.239f,38.7323f,29.6946f,32.9876f,
                                      23.0013f,39.7318f,19.4486f,37.6147f,-0.1506f,5.3246f,3.6173f,24.2573f,4.3941f,9.7105f,24.0364f,35.3681f,17.7805f,35.7681f,16.4144f,17.4362f,8.4987f,
                                      26.8108f,36.2937f,31.6442f,29.7221f,8.7445f,33.3301f,4.0939f,13.078f,45.1481f,29.0172f,21.6548f,35.408f,27.1861f,2.2576f,40.6804f,36.2201f,29.7352f,
                                      29.1244f,38.7444f,5.8721f,33.5983f,48.2694f,34.4161f,19.7148f,13.8085f,13.6075f,22.5042f,37.8002f,50.0543f,48.5314f,20.3694f,28.5042f,-0.4679f,4.4245f,
                                      18.9837f,40.7724f,2.7611f,44.0431f,37.186f,27.7361f,14.6001f,9.1721f,14.6087f,21.4072f,49.3344f,11.4668f,14.6171f,15.2502f,5.244f});
    auto range = NDArrayFactory::create<double>('c', {1,2}, {0, 50});
//    auto exp = NDArrayFactory::create<Nd4jLong>('c', {5}, {23, 19, 20, 23, 15}); // 23, 15, 24, 17, 21
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {5}, {23, 15, 24, 17, 21});

    nd4j::ops::histogram_fixed_width op;
    auto results = op.execute({&input, &range}, {}, {5}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *out = results->at(0);

    ASSERT_TRUE(exp.isSameShape(out));
    // out->printBuffer("5HIST");
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, histogram_fixed_width_test6) {

    auto input = NDArrayFactory::create<double>('c', {7},{0.0, 0.1, 0.1, 0.3, 0.5, 0.5, 0.9});
    auto range = NDArrayFactory::create<double>('c', {2}, {0, 1});
    auto bins  = NDArrayFactory::create<int>(5);

    auto exp = NDArrayFactory::create<Nd4jLong>('c', {5}, {3, 1, 2, 0, 1});

    nd4j::ops::histogram_fixed_width op;
    auto results = op.execute({&input, &range, &bins}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto out = results->at(0);
    // out->printShapeInfo();
    // out->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(out));
    ASSERT_TRUE(exp.equalsTo(out));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, NTH_Element_Test_1) {

    NDArray input = NDArrayFactory::create<float>('c', {12}, {10, 1, 9, 8, 11, 7, 6, 5, 12, 3, 2, 4});
    NDArray n = NDArrayFactory::create<float>(4.f);
    NDArray exp = NDArrayFactory::create<float>(5.f);

    //input.linspace(1.f);

    nd4j::ops::nth_element op;
    auto results = op.execute({&input, &n}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, NTH_Element_Test_2) {

    NDArray input = NDArrayFactory::create<float>('c', {3, 4}, {10, 11, 9, 12, 8, 7, 6, 5, 1, 3, 2, 4});
    NDArray n = NDArrayFactory::create<int>(3);
    NDArray exp = NDArrayFactory::create<float>({12.f, 8.f, 4.f});

//    input.linspace(1.f);

    nd4j::ops::nth_element op;
    auto results = op.execute({&input, &n}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, NTH_Element_Test_3) {

    NDArray input = NDArrayFactory::create<float>('c', {3,4}, {10, 1, 9, 8, 11, 7, 6, 5, 12, 3, 2, 4});
    NDArray n = NDArrayFactory::create<int>(3);
    NDArray exp = NDArrayFactory::create<float>({1.f, 5.f, 2.f});

    //input.linspace(1.f);

    nd4j::ops::nth_element op;
    auto results = op.execute({&input, &n}, {}, {1}); // with reverse = true

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, NTH_Element_Test_4) {

    NDArray input = NDArrayFactory::create<float>('c', {2, 2, 3}, {10, 1, 9, 8, 11, 7, 6, 5, 12, 3, 2, 4});
    NDArray n = NDArrayFactory::create<int>(2);
    NDArray exp = NDArrayFactory::create<float>('c', {2,2}, {10.f, 11.f, 12.f, 4.f});

    //input.linspace(1.f);

    nd4j::ops::nth_element op;
    auto results = op.execute({&input, &n}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}
///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, NTH_Element_Test_04) {

    NDArray input = NDArrayFactory::create<float>('c', {6, 15});
    NDArray n = NDArrayFactory::create<int>(4);
    NDArray exp = NDArrayFactory::create<float>('c', {6}, {5.f, 20.f, 35.f, 50.f, 65.f, 80.f});

    input.linspace(1.f);

    nd4j::ops::nth_element op;
    auto results = op.execute({&input, &n}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}
///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, NTH_Element_Test_5) {

    NDArray input = NDArrayFactory::create<float>('c', {2, 2, 3}, {10, 1, 9, 8, 11, 7, 6, 5, 12, 3, 2, 4});
    NDArray n = NDArrayFactory::create<int>(2);
    NDArray exp = NDArrayFactory::create<float>('c', {2,2}, {1.f, 7.f, 5.f, 2.f});

//    input.linspace(1.f);

    nd4j::ops::nth_element op;
    auto results = op.execute({&input, &n}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, NTH_Element_Test_6) {

    NDArray input = NDArrayFactory::create<float>('c', {12}, {10, 1, 9, 8, 11, 7, 6, 5, 12, 3, 2, 4});
    NDArray n = NDArrayFactory::create<int>(0);
    NDArray exp = NDArrayFactory::create(1.f);//NDArrayFactory::create<float>('c', {2,2}, {1.f, 4.f, 7.f, 10.f});

//    input.linspace(1.f);

    nd4j::ops::nth_element op;
    auto results = op.execute({&input, &n}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* output = results->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}
///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, NTH_Element_Test_06) {

    NDArray input = NDArrayFactory::create<float>('c', {12}, {10, 1, 9, 8, 11, 7, 6, 5, 12, 3, 2, 4});
    NDArray n = NDArrayFactory::create<int>(4);
    NDArray exp = NDArrayFactory::create(8.f);//NDArrayFactory::create<float>('c', {2,2}, {1.f, 4.f, 7.f, 10.f});

//    input.linspace(1.f);

    nd4j::ops::nth_element op;
    auto results = op.execute({&input, &n}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* output = results->at(0);
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}
///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, NTH_Element_Test_7) {

    NDArray input = NDArrayFactory::create<float>('c', {2, 3, 4}, {0.7788f, 0.8012f, 0.7244f, 0.2309f,
                                                                   0.7271f, 0.1804f, 0.5056f, 0.8925f,
                                                                   0.5461f, 0.9234f, 0.0856f, 0.7938f,

                                                                   0.6591f, 0.5555f, 0.1596f, 0.3087f,
                                                                   0.1548f, 0.4695f, 0.9939f, 0.6113f,
                                                                   0.6765f, 0.1800f, 0.6750f, 0.2246f});
    NDArray n = NDArrayFactory::create<int>(2);
    NDArray exp = NDArrayFactory::create<float>('c', {2,3}, {0.7788f, 0.7271f, 0.7938f, 0.5555f, 0.6113f, 0.675f});

    //input.linspace(1.f);

    nd4j::ops::nth_element op;
    auto results = op.execute({&input, &n}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}
///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, NTH_Element_Test_8) {

    NDArray input = NDArrayFactory::create<float>('c', {2, 3, 4}, {0.7788f, 0.8012f, 0.7244f, 0.2309f,
                                                                   0.7271f, 0.1804f, 0.5056f, 0.8925f,
                                                                   0.5461f, 0.9234f, 0.0856f, 0.7938f,

                                                                   0.6591f, 0.5555f, 0.1596f, 0.3087f,
                                                                   0.1548f, 0.4695f, 0.9939f, 0.6113f,
                                                                   0.6765f, 0.1800f, 0.6750f, 0.2246f});
    NDArray n = NDArrayFactory::create<int>(2);
    NDArray exp = NDArrayFactory::create<float>('c', {2,3}, {0.7244f, 0.5056f, 0.5461f, 0.3087f, 0.4695f, 0.2246f});

    //input.linspace(1.f);

    nd4j::ops::nth_element op;
    auto results = op.execute({&input, &n}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test1) {

    auto input = NDArrayFactory::create<Nd4jLong>('c', {3});
    auto shape = NDArrayFactory::create<int>('c', {2}, {3, 3});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {3,3}, {1, 2, 3,1, 2, 3, 1, 2, 3});

    input.linspace(1.f);

    nd4j::ops::broadcast_to op;
    auto results = op.execute({&input, &shape}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test2) {

    auto input = NDArrayFactory::create<double>('c', {1,3});
    auto shape = NDArrayFactory::create<double>('c', {2}, {3.f, 3.f});
    auto exp = NDArrayFactory::create<double>('c', {3,3}, {1.f, 2.f, 3.f,1.f, 2.f, 3.f,1.f, 2.f, 3.f});

    input.linspace(1.f);

    nd4j::ops::broadcast_to op;
    auto results = op.execute({&input, &shape}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test3) {

    auto input = NDArrayFactory::create<double>('c', {3,1});
    auto shape = NDArrayFactory::create<double>('c', {2}, {3.f, 3.f});
    auto exp = NDArrayFactory::create<double>('c', {3,3}, {1.f, 1.f, 1.f,2.f, 2.f, 2.f,3.f, 3.f, 3.f});

    input.linspace(1.f);

    nd4j::ops::broadcast_to op;
    auto results = op.execute({&input, &shape}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test4) {

    auto input = NDArrayFactory::create<double>(10.);
    auto shape = NDArrayFactory::create<double>('c', {2}, {3.f, 3.f});
    auto exp = NDArrayFactory::create<double>('c', {3,3}, {10.f, 10.f, 10.f,10.f, 10.f, 10.f, 10.f, 10.f, 10.f});

    nd4j::ops::broadcast_to op;
    auto results = op.execute({&input, &shape}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test5) {

    auto input = NDArrayFactory::create<double>(10.f);
    auto shape = NDArrayFactory::create<double>('c', {1}, {3.f});
    auto exp = NDArrayFactory::create<double>('c', {3}, {10.f, 10.f, 10.f});

    nd4j::ops::broadcast_to op;
    auto results = op.execute({&input, &shape}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test6) {

    auto input = NDArrayFactory::create<double>(10.f);
    auto shape = NDArrayFactory::create<double>(1.f);
    auto exp = NDArrayFactory::create<double>('c', {1}, {10.f});

    nd4j::ops::broadcast_to op;
    auto results = op.execute({&input, &shape}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test7) {

    auto input = NDArrayFactory::create<double>(10.f);
    auto shape = NDArrayFactory::create<Nd4jLong>(1);
    auto exp = NDArrayFactory::create<double>('c', {1}, {10.});

    nd4j::ops::broadcast_to op;
    auto results = op.execute({&input, &shape}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test8) {

    auto input = NDArrayFactory::create<double>('c', {3});
    auto shape = NDArrayFactory::create<double>('c', {3}, {1.f, 3.f, 3.f});
    auto exp = NDArrayFactory::create<double>('c', {1,3,3}, {1.f, 2.f, 3.f,1.f, 2.f, 3.f,1.f, 2.f, 3.f});

    input.linspace(1.f);

    nd4j::ops::broadcast_to op;
    auto results = op.execute({&input, &shape}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test9) {

    auto input = NDArrayFactory::create<double>('c', {5,1,1});
    auto shape = NDArrayFactory::create<double>('c', {5}, {2.f,1.f,5.f,1.f,3.f});
    auto exp = NDArrayFactory::create<double>('c', {2,1,5,1,3}, {1.f, 1.f, 1.f,2.f, 2.f, 2.f,3.f, 3.f, 3.f,4.f, 4.f, 4.f,5.f, 5.f, 5.f,
                                          1.f, 1.f, 1.f,2.f, 2.f, 2.f,3.f, 3.f, 3.f,4.f, 4.f, 4.f,5.f, 5.f, 5.f});
    input.linspace(1.f);

    nd4j::ops::broadcast_to op;
    auto results = op.execute({&input, &shape}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, broadcast_to_test10) {

    auto input = NDArrayFactory::create<double>('c', {5,1,3});
    auto shape = NDArrayFactory::create<double>('c', {5}, {2.f,1.f,5.f,1.f,3.f});
    auto exp = NDArrayFactory::create<double>('c', {2,1,5,1,3}, {1.f,  2.f,  3.f, 4.f,  5.f,  6.f, 7.f,  8.f,  9.f,10.f, 11.f, 12.f,13.f, 14.f, 15.f,
                                          1.f,  2.f,  3.f, 4.f,  5.f,  6.f, 7.f,  8.f,  9.f,10.f, 11.f, 12.f,13.f, 14.f, 15.f});
    input.linspace(1.f);

    nd4j::ops::broadcast_to op;
    auto results = op.execute({&input, &shape}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, ImageResizeBilinear_Test1) {

    NDArray input    = NDArrayFactory::create<double>('c', {1, 2,3,4});
    //NDArray<float> paddings('c', {3,2}, {0,0, 0,1, 0,0});
    //NDArray<float> expected('c', {2,4,4}, {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.});
    NDArray expected = NDArrayFactory::create<double>('c', {1, 10, 10, 4}, {1.,  2.,   3.,   4.,  2.2,  3.2,  4.2,  5.2, 3.4,  4.4,  5.4,  6.4,
                                                  4.6, 5.6,  6.6,  7.6, 5.8,  6.8,  7.8,  8.8, 7.,   8.,   9.,  10.,
                                                  8.2, 9.2, 10.2, 11.2, 9.,  10.,  11.,  12.,  9.,  10.,  11.,  12.,
                                                  9., 10.,  11.,  12.,  3.4,  4.4,  5.4,  6.4, 4.6,  5.6, 6.6,   7.6,
                                                  5.8, 6.8,  7.8,  8.8, 7.0,   8.,  9.,  10.,  8.2,  9.2, 10.2, 11.2,
                                                  9.4,10.4, 11.4, 12.4,10.6,  11.6,12.6, 13.6,11.4, 12.4, 13.4, 14.4,
                                                 11.4,12.4, 13.4, 14.4,11.4,  12.4,13.4, 14.4, 5.8,  6.8,  7.8,  8.8,
                                                  7.,  8.,   9.,  10.,  8.2,   9.2,10.2, 11.2, 9.4, 10.4, 11.4, 12.4,
                                                 10.6,11.6, 12.6, 13.6,11.8,  12.8,13.8, 14.8,13.0, 14.0, 15.0, 16.,
                                                 13.8,14.8, 15.8, 16.8,13.8,  14.8,15.8, 16.8,13.8, 14.8, 15.8, 16.8,
                                                  8.2, 9.2, 10.2, 11.2, 9.4,  10.4,11.4, 12.4,10.6, 11.6, 12.6, 13.6,
                                                 11.8,12.8, 13.8, 14.8,13.,   14., 15.,  16., 14.2, 15.2, 16.2, 17.2,
                                                 15.4,16.4, 17.4, 18.4,16.2,  17.2,18.2, 19.2,16.2, 17.2, 18.2, 19.2,
                                                 16.2,17.2, 18.2, 19.2,10.6,  11.6,12.6, 13.6,11.8, 12.8, 13.8, 14.8,
                                                 13., 14.,  15.,  16., 14.2,  15.2,16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                 16.6,17.6, 18.6, 19.6,17.8,  18.8,19.8, 20.8,18.6, 19.6, 20.6, 21.6,
                                                 18.6,19.6, 20.6, 21.6,18.6,  19.6,20.6, 21.6,13.,  14.,  15.,  16.,
                                                 14.2,15.2, 16.2, 17.2,15.4,  16.4,17.4, 18.4,16.6, 17.6, 18.6, 19.6,
                                                 17.8,18.8, 19.8, 20.8,19.,   20., 21.,  22., 20.2, 21.2, 22.2, 23.2,
                                                 21., 22.,  23.,  24., 21.,   22., 23.,  24., 21.,  22.,  23.,  24.,
                                                 13., 14.,  15.,  16., 14.2,  15.2,16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                 16.6,17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,19.,  20.,  21.,  22.,
                                                 20.2,21.2, 22.2, 23.2,21.,  22.,  23.,  24., 21.,  22.,  23.,  24.,
                                                 21., 22.,  23.,  24., 13.,  14.,  15.,  16., 14.2, 15.2, 16.2, 17.2,
                                                 15.4,16.4, 17.4, 18.4,16.6, 17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,
                                                 19., 20.,  21.,  22., 20.2, 21.2, 22.2, 23.2,21.,  22.,  23.,  24.,
                                                 21., 22.,  23.,  24., 21.,  22.,  23.,  24., 13.,  14.,  15.,  16.,
                                                 14.2,15.2, 16.2, 17.2,15.4, 16.4, 17.4, 18.4,16.6, 17.6, 18.6, 19.6,
                                                 17.8,18.8, 19.8, 20.8,19.,  20.,  21.,  22., 20.2, 21.2, 22.2, 23.2,
                                                 21., 22.,  23.,  24., 21.,  22.,  23.,  24., 21.,  22.,  23.,  24.,
                                                 13., 14.,  15.,  16., 14.2, 15.2, 16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                 16.6,17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,19.,  20.,  21.,  22.,
                                                 20.2,21.2, 22.2, 23.2,
                                                 21.,  22.,  23.,  24., 21.,  22.,  23.,  24., 21., 22., 23., 24.});
    //input = 1.f;
    input.linspace(1);

    nd4j::ops::resize_bilinear op;
    auto results = op.execute({&input}, {}, {10, 10});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

    //result->printIndexedBuffer("Resized to 10x10");
    //expected.printIndexedBuffer("Expect for 10x10");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests10, ImageResizeBilinear_Test01) {

    NDArray input    = NDArrayFactory::create<double>('c', {2,3,4});
    //NDArray<float> paddings('c', {3,2}, {0,0, 0,1, 0,0});
    //NDArray<float> expected('c', {2,4,4}, {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.});
    NDArray expected = NDArrayFactory::create<double>('c', {10, 10, 4}, {1.,  2.,   3.,   4.,  2.2,  3.2,  4.2,  5.2, 3.4,  4.4,  5.4,  6.4,
                                                                            4.6, 5.6,  6.6,  7.6, 5.8,  6.8,  7.8,  8.8, 7.,   8.,   9.,  10.,
                                                                            8.2, 9.2, 10.2, 11.2, 9.,  10.,  11.,  12.,  9.,  10.,  11.,  12.,
                                                                            9., 10.,  11.,  12.,  3.4,  4.4,  5.4,  6.4, 4.6,  5.6, 6.6,   7.6,
                                                                            5.8, 6.8,  7.8,  8.8, 7.0,   8.,  9.,  10.,  8.2,  9.2, 10.2, 11.2,
                                                                            9.4,10.4, 11.4, 12.4,10.6,  11.6,12.6, 13.6,11.4, 12.4, 13.4, 14.4,
                                                                            11.4,12.4, 13.4, 14.4,11.4,  12.4,13.4, 14.4, 5.8,  6.8,  7.8,  8.8,
                                                                            7.,  8.,   9.,  10.,  8.2,   9.2,10.2, 11.2, 9.4, 10.4, 11.4, 12.4,
                                                                            10.6,11.6, 12.6, 13.6,11.8,  12.8,13.8, 14.8,13.0, 14.0, 15.0, 16.,
                                                                            13.8,14.8, 15.8, 16.8,13.8,  14.8,15.8, 16.8,13.8, 14.8, 15.8, 16.8,
                                                                            8.2, 9.2, 10.2, 11.2, 9.4,  10.4,11.4, 12.4,10.6, 11.6, 12.6, 13.6,
                                                                            11.8,12.8, 13.8, 14.8,13.,   14., 15.,  16., 14.2, 15.2, 16.2, 17.2,
                                                                            15.4,16.4, 17.4, 18.4,16.2,  17.2,18.2, 19.2,16.2, 17.2, 18.2, 19.2,
                                                                            16.2,17.2, 18.2, 19.2,10.6,  11.6,12.6, 13.6,11.8, 12.8, 13.8, 14.8,
                                                                            13., 14.,  15.,  16., 14.2,  15.2,16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                                            16.6,17.6, 18.6, 19.6,17.8,  18.8,19.8, 20.8,18.6, 19.6, 20.6, 21.6,
                                                                            18.6,19.6, 20.6, 21.6,18.6,  19.6,20.6, 21.6,13.,  14.,  15.,  16.,
                                                                            14.2,15.2, 16.2, 17.2,15.4,  16.4,17.4, 18.4,16.6, 17.6, 18.6, 19.6,
                                                                            17.8,18.8, 19.8, 20.8,19.,   20., 21.,  22., 20.2, 21.2, 22.2, 23.2,
                                                                            21., 22.,  23.,  24., 21.,   22., 23.,  24., 21.,  22.,  23.,  24.,
                                                                            13., 14.,  15.,  16., 14.2,  15.2,16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                                            16.6,17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,19.,  20.,  21.,  22.,
                                                                            20.2,21.2, 22.2, 23.2,21.,  22.,  23.,  24., 21.,  22.,  23.,  24.,
                                                                            21., 22.,  23.,  24., 13.,  14.,  15.,  16., 14.2, 15.2, 16.2, 17.2,
                                                                            15.4,16.4, 17.4, 18.4,16.6, 17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,
                                                                            19., 20.,  21.,  22., 20.2, 21.2, 22.2, 23.2,21.,  22.,  23.,  24.,
                                                                            21., 22.,  23.,  24., 21.,  22.,  23.,  24., 13.,  14.,  15.,  16.,
                                                                            14.2,15.2, 16.2, 17.2,15.4, 16.4, 17.4, 18.4,16.6, 17.6, 18.6, 19.6,
                                                                            17.8,18.8, 19.8, 20.8,19.,  20.,  21.,  22., 20.2, 21.2, 22.2, 23.2,
                                                                            21., 22.,  23.,  24., 21.,  22.,  23.,  24., 21.,  22.,  23.,  24.,
                                                                            13., 14.,  15.,  16., 14.2, 15.2, 16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                                            16.6,17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,19.,  20.,  21.,  22.,
                                                                            20.2,21.2, 22.2, 23.2,
                                                                            21.,  22.,  23.,  24., 21.,  22.,  23.,  24., 21., 22., 23., 24.});
    //input = 1.f;
    input.linspace(1);

    nd4j::ops::resize_bilinear op;
    auto results = op.execute({&input}, {}, {10, 10});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

    //result->printIndexedBuffer("Resized to 10x10");
    //expected.printIndexedBuffer("Expect for 10x10");
//    result->printShapeInfo("Output shape");
//    expected.printShapeInfo("Expect shape");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

TEST_F(DeclarableOpsTests10, ImageResizeBilinear_Test02) {

    NDArray input    = NDArrayFactory::create<float>('c', {2, 5,5,3}, {0.7788f,    0.8012f,    0.7244f,
                                                                        0.2309f,    0.7271f,    0.1804f,
                                                                        0.5056f,    0.8925f,    0.5461f,
                                                                        0.9234f,    0.0856f,    0.7938f,
                                                                        0.6591f,    0.5555f,    0.1596f,
                                                                        0.3087f,    0.1548f,    0.4695f,
                                                                        0.9939f,    0.6113f,    0.6765f,
                                                                        0.1800f,    0.6750f,    0.2246f,
                                                                        0.0509f,    0.4601f,    0.8284f,
                                                                        0.2354f,    0.9752f,    0.8361f,
                                                                        0.2585f,    0.4189f,    0.7028f,
                                                                        0.7679f,    0.5373f,    0.7234f,
                                                                        0.2690f,    0.0062f,    0.0327f,
                                                                        0.0644f,    0.8428f,    0.7494f,
                                                                        0.0755f,    0.6245f,    0.3491f,
                                                                        0.5793f,    0.5730f,    0.1822f,
                                                                        0.6420f,    0.9143f,    0.3019f,
                                                                        0.3574f,    0.1704f,    0.8395f,
                                                                        0.5468f,    0.0744f,    0.9011f,
                                                                        0.6574f,    0.4124f,    0.2445f,
                                                                        0.4248f,    0.5219f,    0.6952f,
                                                                        0.4900f,    0.2158f,    0.9549f,
                                                                        0.1386f,    0.1544f,    0.5365f,
                                                                        0.0134f,    0.4163f,    0.1456f,
                                                                        0.4109f,    0.2484f,    0.3330f,
                                                                        0.2974f,    0.6636f,    0.3808f,
                                                                        0.8664f,    0.1896f,    0.7530f,
                                                                        0.7215f,    0.6612f,    0.7270f,
                                                                        0.5704f,    0.2666f,    0.7453f,
                                                                        0.0444f,    0.3024f,    0.4850f,
                                                                        0.7982f,    0.0965f,    0.7843f,
                                                                        0.5075f,    0.0844f,    0.8370f,
                                                                        0.6103f,    0.4604f,    0.6087f,
                                                                        0.8594f,    0.4599f,    0.6714f,
                                                                        0.2744f,    0.1981f,    0.4143f,
                                                                        0.7821f,    0.3505f,    0.5040f,
                                                                        0.1180f,    0.8307f,    0.1817f,
                                                                        0.8442f,    0.5074f,    0.4471f,
                                                                        0.5105f,    0.6666f,    0.2576f,
                                                                        0.2341f,    0.6801f,    0.2652f,
                                                                        0.5394f,    0.4690f,    0.6146f,
                                                                        0.1210f,    0.2576f,    0.0769f,
                                                                        0.4643f,    0.1628f,    0.2026f,
                                                                        0.3774f,    0.0506f,    0.3462f,
                                                                        0.5720f,    0.0838f,    0.4228f,
                                                                        0.0588f,    0.5362f,    0.4756f,
                                                                        0.2530f,    0.1778f,    0.0751f,
                                                                        0.8977f,    0.3648f,    0.3065f,
                                                                        0.4739f,    0.7014f,    0.4473f,
                                                                        0.5171f,    0.1744f,    0.3487f});

    NDArray expected = NDArrayFactory::create<float>('c', {10, 10, 4}, {1.,  2.,   3.,   4.,  2.2,  3.2,  4.2,  5.2, 3.4,  4.4,  5.4,  6.4,
                                                                         4.6, 5.6,  6.6,  7.6, 5.8,  6.8,  7.8,  8.8, 7.,   8.,   9.,  10.,
                                                                         8.2, 9.2, 10.2, 11.2, 9.,  10.,  11.,  12.,  9.,  10.,  11.,  12.,
                                                                         9., 10.,  11.,  12.,  3.4,  4.4,  5.4,  6.4, 4.6,  5.6, 6.6,   7.6,
                                                                         5.8, 6.8,  7.8,  8.8, 7.0,   8.,  9.,  10.,  8.2,  9.2, 10.2, 11.2,
                                                                         9.4,10.4, 11.4, 12.4,10.6,  11.6,12.6, 13.6,11.4, 12.4, 13.4, 14.4,
                                                                         11.4,12.4, 13.4, 14.4,11.4,  12.4,13.4, 14.4, 5.8,  6.8,  7.8,  8.8,
                                                                         7.,  8.,   9.,  10.,  8.2,   9.2,10.2, 11.2, 9.4, 10.4, 11.4, 12.4,
                                                                         10.6,11.6, 12.6, 13.6,11.8,  12.8,13.8, 14.8,13.0, 14.0, 15.0, 16.,
                                                                         13.8,14.8, 15.8, 16.8,13.8,  14.8,15.8, 16.8,13.8, 14.8, 15.8, 16.8,
                                                                         8.2, 9.2, 10.2, 11.2, 9.4,  10.4,11.4, 12.4,10.6, 11.6, 12.6, 13.6,
                                                                         11.8,12.8, 13.8, 14.8,13.,   14., 15.,  16., 14.2, 15.2, 16.2, 17.2,
                                                                         15.4,16.4, 17.4, 18.4,16.2,  17.2,18.2, 19.2,16.2, 17.2, 18.2, 19.2,
                                                                         16.2,17.2, 18.2, 19.2,10.6,  11.6,12.6, 13.6,11.8, 12.8, 13.8, 14.8,
                                                                         13., 14.,  15.,  16., 14.2,  15.2,16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                                         16.6,17.6, 18.6, 19.6,17.8,  18.8,19.8, 20.8,18.6, 19.6, 20.6, 21.6,
                                                                         18.6,19.6, 20.6, 21.6,18.6,  19.6,20.6, 21.6,13.,  14.,  15.,  16.,
                                                                         14.2,15.2, 16.2, 17.2,15.4,  16.4,17.4, 18.4,16.6, 17.6, 18.6, 19.6,
                                                                         17.8,18.8, 19.8, 20.8,19.,   20., 21.,  22., 20.2, 21.2, 22.2, 23.2,
                                                                         21., 22.,  23.,  24., 21.,   22., 23.,  24., 21.,  22.,  23.,  24.,
                                                                         13., 14.,  15.,  16., 14.2,  15.2,16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                                         16.6,17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,19.,  20.,  21.,  22.,
                                                                         20.2,21.2, 22.2, 23.2,21.,  22.,  23.,  24., 21.,  22.,  23.,  24.,
                                                                         21., 22.,  23.,  24., 13.,  14.,  15.,  16., 14.2, 15.2, 16.2, 17.2,
                                                                         15.4,16.4, 17.4, 18.4,16.6, 17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,
                                                                         19., 20.,  21.,  22., 20.2, 21.2, 22.2, 23.2,21.,  22.,  23.,  24.,
                                                                         21., 22.,  23.,  24., 21.,  22.,  23.,  24., 13.,  14.,  15.,  16.,
                                                                         14.2,15.2, 16.2, 17.2,15.4, 16.4, 17.4, 18.4,16.6, 17.6, 18.6, 19.6,
                                                                         17.8,18.8, 19.8, 20.8,19.,  20.,  21.,  22., 20.2, 21.2, 22.2, 23.2,
                                                                         21., 22.,  23.,  24., 21.,  22.,  23.,  24., 21.,  22.,  23.,  24.,
                                                                         13., 14.,  15.,  16., 14.2, 15.2, 16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                                         16.6,17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,19.,  20.,  21.,  22.,
                                                                         20.2,21.2, 22.2, 23.2,
                                                                         21.,  22.,  23.,  24., 21.,  22.,  23.,  24., 21., 22., 23., 24.});
    //input.linspace(1);

    nd4j::ops::resize_bilinear op;
    auto results = op.execute({&input}, {}, {9, 9});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

    result->printIndexedBuffer("Resized to 9x9");
    //expected.printIndexedBuffer("Expect for 10x10");
    result->printShapeInfo("Output shape");
//    expected.printShapeInfo("Expect shape");
//    ASSERT_TRUE(expected.isSameShape(result));
//    ASSERT_TRUE(expected.equalsTo(result));
    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, ImageResizeBilinear_Test2) {

    NDArray input    = NDArrayFactory::create<double>('c', {1, 2,3,4});
    NDArray size = NDArrayFactory::create<int>({10, 10});
    //NDArray<float> expected('c', {2,4,4}, {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.});
    NDArray expected = NDArrayFactory::create<double>('c', {1, 10, 10, 4}, {1.,  2.,   3.,   4.,  2.2,  3.2,  4.2,  5.2, 3.4,  4.4,  5.4,  6.4,
                                                  4.6, 5.6,  6.6,  7.6, 5.8,  6.8,  7.8,  8.8, 7.,   8.,   9.,  10.,
                                                  8.2, 9.2, 10.2, 11.2, 9.,  10.,  11.,  12.,  9.,  10.,  11.,  12.,
                                                  9., 10.,  11.,  12.,  3.4,  4.4,  5.4,  6.4, 4.6,  5.6, 6.6,   7.6,
                                                  5.8, 6.8,  7.8,  8.8, 7.0,   8.,  9.,  10.,  8.2,  9.2, 10.2, 11.2,
                                                  9.4,10.4, 11.4, 12.4,10.6,  11.6,12.6, 13.6,11.4, 12.4, 13.4, 14.4,
                                                  11.4,12.4, 13.4, 14.4,11.4,  12.4,13.4, 14.4, 5.8,  6.8,  7.8,  8.8,
                                                  7.,  8.,   9.,  10.,  8.2,   9.2,10.2, 11.2, 9.4, 10.4, 11.4, 12.4,
                                                  10.6,11.6, 12.6, 13.6,11.8,  12.8,13.8, 14.8,13.0, 14.0, 15.0, 16.,
                                                  13.8,14.8, 15.8, 16.8,13.8,  14.8,15.8, 16.8,13.8, 14.8, 15.8, 16.8,
                                                  8.2, 9.2, 10.2, 11.2, 9.4,  10.4,11.4, 12.4,10.6, 11.6, 12.6, 13.6,
                                                  11.8,12.8, 13.8, 14.8,13.,   14., 15.,  16., 14.2, 15.2, 16.2, 17.2,
                                                  15.4,16.4, 17.4, 18.4,16.2,  17.2,18.2, 19.2,16.2, 17.2, 18.2, 19.2,
                                                  16.2,17.2, 18.2, 19.2,10.6,  11.6,12.6, 13.6,11.8, 12.8, 13.8, 14.8,
                                                  13., 14.,  15.,  16., 14.2,  15.2,16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                  16.6,17.6, 18.6, 19.6,17.8,  18.8,19.8, 20.8,18.6, 19.6, 20.6, 21.6,
                                                  18.6,19.6, 20.6, 21.6,18.6,  19.6,20.6, 21.6,13.,  14.,  15.,  16.,
                                                  14.2,15.2, 16.2, 17.2,15.4,  16.4,17.4, 18.4,16.6, 17.6, 18.6, 19.6,
                                                  17.8,18.8, 19.8, 20.8,19.,   20., 21.,  22., 20.2, 21.2, 22.2, 23.2,
                                                  21., 22.,  23.,  24., 21.,   22., 23.,  24., 21.,  22.,  23.,  24.,
                                                  13., 14.,  15.,  16., 14.2,  15.2,16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                  16.6,17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,19.,  20.,  21.,  22.,
                                                  20.2,21.2, 22.2, 23.2,21.,  22.,  23.,  24., 21.,  22.,  23.,  24.,
                                                  21., 22.,  23.,  24., 13.,  14.,  15.,  16., 14.2, 15.2, 16.2, 17.2,
                                                  15.4,16.4, 17.4, 18.4,16.6, 17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,
                                                  19., 20.,  21.,  22., 20.2, 21.2, 22.2, 23.2,21.,  22.,  23.,  24.,
                                                  21., 22.,  23.,  24., 21.,  22.,  23.,  24., 13.,  14.,  15.,  16.,
                                                  14.2,15.2, 16.2, 17.2,15.4, 16.4, 17.4, 18.4,16.6, 17.6, 18.6, 19.6,
                                                  17.8,18.8, 19.8, 20.8,19.,  20.,  21.,  22., 20.2, 21.2, 22.2, 23.2,
                                                  21., 22.,  23.,  24., 21.,  22.,  23.,  24., 21.,  22.,  23.,  24.,
                                                  13., 14.,  15.,  16., 14.2, 15.2, 16.2, 17.2,15.4, 16.4, 17.4, 18.4,
                                                  16.6,17.6, 18.6, 19.6,17.8, 18.8, 19.8, 20.8,19.,  20.,  21.,  22.,
                                                  20.2,21.2, 22.2, 23.2,
                                                  21.,  22.,  23.,  24., 21.,  22.,  23.,  24., 21., 22., 23., 24.});
    //input = 1.f;
    input.linspace(1);

    nd4j::ops::resize_bilinear op;
    auto results = op.execute({&input, &size}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, ImageResizeBilinear_Test3) {

    NDArray input    = NDArrayFactory::create<double>('c', {1, 2,3,4});
    //NDArray<float> paddings('c', {3,2}, {0,0, 0,1, 0,0});
    //NDArray<float> expected('c', {2,4,4}, {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.});
    NDArray expected = NDArrayFactory::create<double>('c', {1, 10, 10, 4},
            { 1.,         2.,         3.,         4. ,
             1.8888888,  2.8888888,  3.8888888,  4.888889,
             2.7777777,  3.7777777,  4.7777777,  5.7777777,
             3.6666667,  4.666667 ,  5.666667,   6.666667 ,
             4.5555553,  5.5555553,  6.5555553,  7.5555553,
             5.4444447,  6.4444447,  7.4444447,  8.444445,
             6.3333335,  7.3333335,  8.333334,   9.333334,
             7.2222223,  8.222222,   9.222222,  10.222222,
             8.111111,   9.111111,  10.111111,  11.111111,
             9.,        10.,        11.,        12.,

             2.3333335,  3.3333335,  4.3333335,  5.3333335,
             3.2222223,  4.2222223,  5.2222223,  6.2222223,
             4.111111,   5.111111,   6.111111,   7.111111,
             5.,         6.,         7.,         8.,
             5.888889,   6.888889,   7.888889,   8.888888,
             6.777778,   7.777778,   8.777778,   9.777778,
             7.666667,   8.666667,   9.666667,  10.666667,
             8.555555,   9.555555,  10.555555,  11.555555,
             9.444444,  10.444444,  11.444444,  12.444444,
            10.333333,  11.333333,  12.333333,  13.333333,

             3.6666667,  4.666667,   5.666667,   6.666667,
             4.5555553,  5.5555553,  6.5555553,  7.5555553,
             5.4444447,  6.4444447,  7.4444447,  8.444445 ,
             6.3333335,  7.3333335,  8.333334,   9.333334 ,
             7.2222223,  8.222222,   9.222222,  10.222222 ,
             8.111112,   9.111112,  10.111112,  11.111112 ,
             9.,        10.,        11.000001,  12.000001 ,
             9.888889,  10.888889,  11.888889,  12.888889 ,
            10.777778,  11.777778,  12.777778,  13.777778 ,
            11.666667,  12.666667,  13.666667,  14.666667,

             5.,        6.,        7.,        8.,
             5.888889,  6.888889,  7.888889,  8.888889,
             6.7777777, 7.7777777, 8.777779,  9.777779,
             7.666667,  8.666667,  9.666667, 10.666667,
             8.555555,  9.555555, 10.555555, 11.555555,
             9.444445, 10.444445, 11.444445, 12.444445,
            10.333334, 11.333334, 12.333334, 13.333334,
            11.222222, 12.222222, 13.222222, 14.222222,
            12.111111, 13.111111, 14.111111, 15.111111,
            13.,       14.,       15.,       16.,

             6.3333335, 7.3333335, 8.333334,  9.333334,
             7.2222223, 8.222222,  9.222222, 10.222222,
             8.111111,  9.111111, 10.111112, 11.111112,
             9.,       10.,       11.,       12.,
             9.888889, 10.888889, 11.888889, 12.888889,
            10.777779, 11.777779, 12.777779, 13.777779,
            11.666667, 12.666667, 13.666668, 14.666668,
            12.555555, 13.555555, 14.555555, 15.555555,
            13.444445, 14.444445, 15.444445, 16.444445,
            14.333334, 15.333334, 16.333334, 17.333334,
             7.666667,  8.666667,  9.666667, 10.666667,
             8.555555,  9.555555, 10.555555, 11.555555,
             9.444445, 10.444445, 11.444445, 12.444445,
            10.333334, 11.333334, 12.333334, 13.333334,
            11.222222, 12.222222, 13.222222, 14.222222,
            12.111112, 13.111112, 14.111112, 15.111112,
            13.,       14.,       15.0,      16.,
            13.888889, 14.888889, 15.888889, 16.88889,
            14.777778, 15.777778, 16.777779, 17.777779,
            15.666667, 16.666668, 17.666668, 18.666668,

             9.,       10.,       11.,       12.,
             9.888889, 10.888889, 11.888889, 12.888889,
            10.777778, 11.777778, 12.777779, 13.777779,
            11.666667, 12.666666, 13.666666, 14.666666,
            12.555555, 13.555555, 14.555555, 15.555555,
            13.444445, 14.444445, 15.444445, 16.444445,
            14.333334, 15.333334, 16.333334, 17.333334,
            15.222221, 16.222221, 17.222221, 18.222221,
            16.11111,  17.11111,  18.11111,  19.11111,
            17.,       18.,       19.,       20.,

            10.333334, 11.333334, 12.333334, 13.333334,
            11.222223, 12.222223, 13.222223, 14.222223,
            12.111112, 13.111112, 14.111112, 15.111112,
            13.000001, 14.,       15.,       16.,
            13.888889, 14.888889, 15.888889, 16.88889,
            14.777779, 15.777779, 16.777779, 17.777779,
            15.666668, 16.666668, 17.666668, 18.666668,
            16.555555, 17.555555, 18.555555, 19.555555,
            17.444445, 18.444445, 19.444445, 20.444445,
            18.333334, 19.333334, 20.333334, 21.333334,
            11.666667, 12.666667, 13.666667, 14.666667,
            12.555555, 13.555555, 14.555555, 15.555555,
            13.444445, 14.444445, 15.444446, 16.444447,
            14.333334, 15.333333, 16.333332, 17.333332,
            15.222222, 16.222221, 17.222221, 18.222221,
            16.11111,  17.11111,  18.11111,  19.11111,
            17.,       18.,       19.,       20.,
            17.88889,  18.88889,  19.88889,  20.88889,
            18.777779, 19.777779, 20.777779, 21.777779,
            19.666668, 20.666668, 21.666668, 22.666668,

            13.,        14.,        15.,        16.,
            13.888889,  14.888889,  15.888889,  16.88889,
            14.777778,  15.777778,  16.777779,  17.777779,
            15.666667,  16.666666,  17.666666,  18.666666,
            16.555555,  17.555555,  18.555555,  19.555555,
            17.444445,  18.444445,  19.444445,  20.444445,
            18.333334,  19.333334,  20.333334,  21.333334,
            19.222221,  20.222221,  21.222221,  22.222221,
            20.11111,   21.11111,   22.11111,   23.11111,
            21.,        22.,        23.,        24.});
    //input = 1.f;
    input.linspace(1);

    nd4j::ops::resize_bilinear op;
    auto results = op.execute({&input}, {}, {10, 10, 1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, ImageResizeBilinear_Test4) {

    NDArray input    = NDArrayFactory::create<double>('c', {1, 2,3,4});
    NDArray size = NDArrayFactory::create<int>({10, 10});
    NDArray expected = NDArrayFactory::create<double>('c', {1, 10, 10, 4},
                            { 1.,         2.,         3.,         4. ,
                              1.8888888,  2.8888888,  3.8888888,  4.888889,
                              2.7777777,  3.7777777,  4.7777777,  5.7777777,
                              3.6666667,  4.666667 ,  5.666667,   6.666667 ,
                              4.5555553,  5.5555553,  6.5555553,  7.5555553,
                              5.4444447,  6.4444447,  7.4444447,  8.444445,
                              6.3333335,  7.3333335,  8.333334,   9.333334,
                              7.2222223,  8.222222,   9.222222,  10.222222,
                              8.111111,   9.111111,  10.111111,  11.111111,
                              9.,        10.,        11.,        12.,

                              2.3333335,  3.3333335,  4.3333335,  5.3333335,
                              3.2222223,  4.2222223,  5.2222223,  6.2222223,
                              4.111111,   5.111111,   6.111111,   7.111111,
                              5.,         6.,         7.,         8.,
                              5.888889,   6.888889,   7.888889,   8.888888,
                              6.777778,   7.777778,   8.777778,   9.777778,
                              7.666667,   8.666667,   9.666667,  10.666667,
                              8.555555,   9.555555,  10.555555,  11.555555,
                              9.444444,  10.444444,  11.444444,  12.444444,
                              10.333333,  11.333333,  12.333333,  13.333333,

                              3.6666667,  4.666667,   5.666667,   6.666667,
                              4.5555553,  5.5555553,  6.5555553,  7.5555553,
                              5.4444447,  6.4444447,  7.4444447,  8.444445 ,
                              6.3333335,  7.3333335,  8.333334,   9.333334 ,
                              7.2222223,  8.222222,   9.222222,  10.222222 ,
                              8.111112,   9.111112,  10.111112,  11.111112 ,
                              9.,        10.,        11.000001,  12.000001 ,
                              9.888889,  10.888889,  11.888889,  12.888889 ,
                              10.777778,  11.777778,  12.777778,  13.777778 ,
                              11.666667,  12.666667,  13.666667,  14.666667,

                              5.,        6.,        7.,        8.,
                              5.888889,  6.888889,  7.888889,  8.888889,
                              6.7777777, 7.7777777, 8.777779,  9.777779,
                              7.666667,  8.666667,  9.666667, 10.666667,
                              8.555555,  9.555555, 10.555555, 11.555555,
                              9.444445, 10.444445, 11.444445, 12.444445,
                              10.333334, 11.333334, 12.333334, 13.333334,
                              11.222222, 12.222222, 13.222222, 14.222222,
                              12.111111, 13.111111, 14.111111, 15.111111,
                              13.,       14.,       15.,       16.,

                              6.3333335, 7.3333335, 8.333334,  9.333334,
                              7.2222223, 8.222222,  9.222222, 10.222222,
                              8.111111,  9.111111, 10.111112, 11.111112,
                              9.,       10.,       11.,       12.,
                              9.888889, 10.888889, 11.888889, 12.888889,
                              10.777779, 11.777779, 12.777779, 13.777779,
                              11.666667, 12.666667, 13.666668, 14.666668,
                              12.555555, 13.555555, 14.555555, 15.555555,
                              13.444445, 14.444445, 15.444445, 16.444445,
                              14.333334, 15.333334, 16.333334, 17.333334,
                              7.666667,  8.666667,  9.666667, 10.666667,
                              8.555555,  9.555555, 10.555555, 11.555555,
                              9.444445, 10.444445, 11.444445, 12.444445,
                              10.333334, 11.333334, 12.333334, 13.333334,
                              11.222222, 12.222222, 13.222222, 14.222222,
                              12.111112, 13.111112, 14.111112, 15.111112,
                              13.,       14.,       15.0,      16.,
                              13.888889, 14.888889, 15.888889, 16.88889,
                              14.777778, 15.777778, 16.777779, 17.777779,
                              15.666667, 16.666668, 17.666668, 18.666668,

                              9.,       10.,       11.,       12.,
                              9.888889, 10.888889, 11.888889, 12.888889,
                              10.777778, 11.777778, 12.777779, 13.777779,
                              11.666667, 12.666666, 13.666666, 14.666666,
                              12.555555, 13.555555, 14.555555, 15.555555,
                              13.444445, 14.444445, 15.444445, 16.444445,
                              14.333334, 15.333334, 16.333334, 17.333334,
                              15.222221, 16.222221, 17.222221, 18.222221,
                              16.11111,  17.11111,  18.11111,  19.11111,
                              17.,       18.,       19.,       20.,

                              10.333334, 11.333334, 12.333334, 13.333334,
                              11.222223, 12.222223, 13.222223, 14.222223,
                              12.111112, 13.111112, 14.111112, 15.111112,
                              13.000001, 14.,       15.,       16.,
                              13.888889, 14.888889, 15.888889, 16.88889,
                              14.777779, 15.777779, 16.777779, 17.777779,
                              15.666668, 16.666668, 17.666668, 18.666668,
                              16.555555, 17.555555, 18.555555, 19.555555,
                              17.444445, 18.444445, 19.444445, 20.444445,
                              18.333334, 19.333334, 20.333334, 21.333334,
                              11.666667, 12.666667, 13.666667, 14.666667,
                              12.555555, 13.555555, 14.555555, 15.555555,
                              13.444445, 14.444445, 15.444446, 16.444447,
                              14.333334, 15.333333, 16.333332, 17.333332,
                              15.222222, 16.222221, 17.222221, 18.222221,
                              16.11111,  17.11111,  18.11111,  19.11111,
                              17.,       18.,       19.,       20.,
                              17.88889,  18.88889,  19.88889,  20.88889,
                              18.777779, 19.777779, 20.777779, 21.777779,
                              19.666668, 20.666668, 21.666668, 22.666668,

                              13.,        14.,        15.,        16.,
                              13.888889,  14.888889,  15.888889,  16.88889,
                              14.777778,  15.777778,  16.777779,  17.777779,
                              15.666667,  16.666666,  17.666666,  18.666666,
                              16.555555,  17.555555,  18.555555,  19.555555,
                              17.444445,  18.444445,  19.444445,  20.444445,
                              18.333334,  19.333334,  20.333334,  21.333334,
                              19.222221,  20.222221,  21.222221,  22.222221,
                              20.11111,   21.11111,   22.11111,   23.11111,
                              21.,        22.,        23.,        24.});
    //input = 1.f;
    input.linspace(1);

    nd4j::ops::resize_bilinear op;
    auto results = op.execute({&input, &size}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);
//    result->printIndexedBuffer("Resized to 10x10");
//    expected.printIndexedBuffer("Expected of 10x10");
//    result->printShapeInfo("Resized to 10x10 shape");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, LinSpace_Test1) {

    NDArray start = NDArrayFactory::create<double>(1.);
    NDArray finish = NDArrayFactory::create<double>(12.);
    NDArray num = NDArrayFactory::create<int>(23);
    NDArray expect = NDArrayFactory::create<double>({1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5,
                                                        8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12.});

    nd4j::ops::lin_space op;
    auto result = op.execute({&start, &finish, &num}, {}, {});
    ASSERT_EQ(result->status(), ND4J_STATUS_OK);
    auto res = result->at(0);

    ASSERT_TRUE(expect.equalsTo(res));
    delete result;
}
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, ImageResizeNeighbor_Test1) {

    NDArray input    = NDArrayFactory::create<double>('c', {1, 2, 3, 4});
    //NDArray<float> paddings('c', {3,2}, {0,0, 0,1, 0,0});
    //NDArray<float> expected('c', {2,4,4}, {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.});
    NDArray expected = NDArrayFactory::create<double>('c', {1, 4, 5, 4}, { 1,  2,  3,  4,
     1,  2,  3,  4,
     5,  6,  7,  8,
     5,  6,  7,  8,
     9, 10, 11, 12,

     1,  2,  3,  4,
     1,  2,  3,  4,
     5,  6,  7,  8,
     5,  6,  7,  8,
     9, 10, 11, 12,

    13, 14, 15, 16,
    13, 14, 15, 16,
    17, 18, 19, 20,
    17, 18, 19, 20,
    21, 22, 23, 24,

    13, 14, 15, 16,
    13, 14, 15, 16,
    17, 18, 19, 20,
    17, 18, 19, 20,
    21, 22, 23, 24
    });
    //input = 1.f;
    input.linspace(1);

    nd4j::ops::resize_nearest_neighbor op;
    auto results = op.execute({&input}, {}, {4, 5});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

//    result->printIndexedBuffer("Resized to 4x5");
//    expected.printIndexedBuffer("Expect for 4x5");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

TEST_F(DeclarableOpsTests10, ImageResizeNeighbor_Test01) {

    NDArray input    = NDArrayFactory::create<double>('c', {2, 3, 4});
    //NDArray<float> paddings('c', {3,2}, {0,0, 0,1, 0,0});
    //NDArray<float> expected('c', {2,4,4}, {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.});
    NDArray expected = NDArrayFactory::create<double>('c', {4, 5, 4},    { 1,  2,  3,  4,
                                                                           1,  2,  3,  4,
                                                                           5,  6,  7,  8,
                                                                           5,  6,  7,  8,
                                                                           9, 10, 11, 12,

                                                                           1,  2,  3,  4,
                                                                           1,  2,  3,  4,
                                                                           5,  6,  7,  8,
                                                                           5,  6,  7,  8,
                                                                           9, 10, 11, 12,

                                                                           13, 14, 15, 16,
                                                                           13, 14, 15, 16,
                                                                           17, 18, 19, 20,
                                                                           17, 18, 19, 20,
                                                                           21, 22, 23, 24,

                                                                           13, 14, 15, 16,
                                                                           13, 14, 15, 16,
                                                                           17, 18, 19, 20,
                                                                           17, 18, 19, 20,
                                                                           21, 22, 23, 24
    });
    //input = 1.f;
    input.linspace(1);

    nd4j::ops::resize_nearest_neighbor op;
    auto results = op.execute({&input}, {}, {4, 5});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);

    //result->printIndexedBuffer("Resized to 4x5");
    //expected.printIndexedBuffer("Expect for 4x5");
    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, ReduceLogSumExpTest_1) {

    NDArray input   = NDArrayFactory::create<double> ('c', {3,3}, {0, 1, 0, 0, 1, 0, 0, 0, 0});

    NDArray expected = NDArrayFactory::create<double>(2.5206409f);

    nd4j::ops::reduce_logsumexp op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, ReduceLogSumExpTest_2) {

    NDArray input = NDArrayFactory::create<double>('c', {3,3}, {0, 1, 0, 0, 1, 0, 0, 0, 0});

    NDArray expected = NDArrayFactory::create<double>({1.0986123f, 1.8619947f, 1.0986123f});

    nd4j::ops::reduce_logsumexp op;
    auto results = op.execute({&input}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
//    result->printIndexedBuffer("REDUCE_LOGSUMEXP");
//    expected.printIndexedBuffer("LSE EXPECTED");
    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, ReduceLogSumExpTest_3) {

    NDArray input = NDArrayFactory::create<float>('c', {3,3}, {0, 1, 0, 0, 1, 0, 0, 0, 0});

    NDArray expected = NDArrayFactory::create<float>('c', {1,3}, {1.0986123f, 1.8619947f, 1.0986123f});

    nd4j::ops::reduce_logsumexp op;
    auto results = op.execute({&input}, {1.f}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
//    result->printIndexedBuffer("REDUCE_LOGSUMEXP");
//    expected.printIndexedBuffer("LSE EXPECTED");
    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_NonMaxSuppressing_1) {

    NDArray boxes    = NDArrayFactory::create<float>('c', {3,4});
    NDArray scores = NDArrayFactory::create<float>('c', {3}, {1, 2, 3});
    NDArray expected = NDArrayFactory::create<int>('c', {3}, {2, 1, 0});
    boxes.linspace(1.f);

    nd4j::ops::non_max_suppression op;
    auto results = op.execute({&boxes, &scores}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);
    //result->printIndexedBuffer("OOOOUUUUTTT");

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_NonMaxSuppressing_2) {

    NDArray boxes    = NDArrayFactory::create<double>('c', {6,4}, {0, 0, 1, 1, 0, 0.1f, 1, 1.1f, 0, -0.1f, 1.f, 0.9f,
                                         0, 10, 1, 11, 0, 10.1f, 1.f, 11.1f, 0, 100, 1, 101});
    NDArray scales = NDArrayFactory::create<double>('c', {6}, {0.9f, .75f, .6f, .95f, .5f, .3f}); //3, 0, 1, 2, 4, 5
    NDArray expected = NDArrayFactory::create<int>('c', {3}, {3,0,5});

    nd4j::ops::non_max_suppression op;
    auto results = op.execute({&boxes, &scales}, {0.5}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);
//    result->printBuffer("NonMaxSuppression OUtput2");
    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_NonMaxSuppressingOverlap_1) {

    NDArray boxes    = NDArrayFactory::create<double>('c', {4,4}, {
        0,     0, 1,    1,
        0,   0.1, 1,  1.1,
        0,  -0.1, 1,  0.9,
        0,    10, 1,   11});
    NDArray scores = NDArrayFactory::create<double>('c', {4}, {0.9, .75, .6, .95}); //3
    NDArray max_num = NDArrayFactory::create<int>(3);
    NDArray expected = NDArrayFactory::create<int>('c', {1,}, {3});

    nd4j::ops::non_max_suppression_overlaps op;
    auto results = op.execute({&boxes, &scores, &max_num}, {0.5, 0.}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);
//    result->printBuffer("NonMaxSuppressionOverlap1 Output");
    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_NonMaxSuppressingOverlap_2) {

    NDArray boxes    = NDArrayFactory::create<double>('c', {4,4}, {
            0,     0, 1,    1,
            0,   0.1, 1,  1.1,
            0,  -0.1, 1,  0.9,
            0,    10, 1,   11});
    NDArray scores = NDArrayFactory::create<double>('c', {4}, {0.9, .95, .6, .75}); //3
    NDArray max_num = NDArrayFactory::create<int>(3);
    NDArray expected = NDArrayFactory::create<int>('c', {3,}, {1,1,1});

    nd4j::ops::non_max_suppression_overlaps op;
    auto results = op.execute({&boxes, &scores, &max_num}, {0.5, 0.}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);
//    result->printBuffer("NonMaxSuppressionOverlap Output");
    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_NonMaxSuppressingOverlap_3) {

    NDArray boxes    = NDArrayFactory::create<double>('c', {4,4}, {
            0,     0, 1,    1,
            0,   0.1, 1,  1.1,
            0,  -0.1, 1,  0.9,
            0,    10, 1,   11});
    NDArray scores = NDArrayFactory::create<double>('c', {4}, {0.5, .95, -.6, .75}); //3
    NDArray max_num = NDArrayFactory::create<int>(5);
    NDArray expected = NDArrayFactory::create<int>('c', {5,}, {1,1,1,1,1});

    nd4j::ops::non_max_suppression_overlaps op;
    auto results = op.execute({&boxes, &scores, &max_num}, {0.5, 0.}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray* result = results->at(0);
//    result->printBuffer("NonMaxSuppressionOverlap Output");
    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_CropAndResize_1) {
    int axis = 0;
    NDArray images = NDArrayFactory::create<double>('c', {1,2,2,1}, {1,2,3,4});
    NDArray boxes = NDArrayFactory::create<float>('c', {1,4}, {0,0,1,1});
    NDArray boxI = NDArrayFactory::create<int>('c', {1}, {axis});
    NDArray cropSize = NDArrayFactory::create<int>({1, 1});

    //NDArray<float> ('c', {6}, {0.9f, .75f, .6f, .95f, .5f, .3f});
    NDArray expected = NDArrayFactory::create<double>('c', {1,1,1,1}, {2.5f});

    nd4j::ops::crop_and_resize op;
    auto results = op.execute({&images, &boxes, &boxI, &cropSize}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
//    result->printIndexedBuffer("Cropped and Resized");

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_CropAndResize_2) {
    int axis = 0;
    NDArray images    = NDArrayFactory::create<float>('c', {1,2,2,1}, {1.f, 2.f, 3.f, 4.f});
    NDArray boxes = NDArrayFactory::create<float>('c', {1,4}, {0.f, 0.f, 1.f, 1.f});
    NDArray boxI = NDArrayFactory::create<int>('c', {1}, {axis});
    NDArray cropSize = NDArrayFactory::create<int>({1, 1});

    //NDArray<float> ('c', {6}, {0.9f, .75f, .6f, .95f, .5f, .3f});
    NDArray expected = NDArrayFactory::create<float>('c', {1,1,1,1}, {4.f});

    nd4j::ops::crop_and_resize op;
    auto results = op.execute({&images, &boxes, &boxI, &cropSize}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_CropAndResize_3) {

    NDArray images   ('c', {1,2,2,1}, {1,2,3,4}, nd4j::DataType::FLOAT32);
    NDArray boxes('c', {1,4}, {0,0,1,1}, nd4j::DataType::FLOAT32);
    NDArray boxI('c', {1}, {0}, nd4j::DataType::INT64);
    NDArray cropSize = NDArrayFactory::create<Nd4jLong>({3, 3});

    //NDArray<float> ('c', {6}, {0.9f, .75f, .6f, .95f, .5f, .3f});
    NDArray expected('c', {1,3,3,1}, {1, 1.5f, 2., 2.f, 2.5f, 3.f, 3.f, 3.5f, 4.f}, nd4j::DataType::FLOAT32);

    nd4j::ops::crop_and_resize op;
    auto results = op.execute({&images, &boxes, &boxI, &cropSize}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_CropAndResize_4) {

    NDArray images('c', {1,2,2,1}, {1, 2, 3, 4}, nd4j::DataType::FLOAT32);
    NDArray boxes('c', {1,4}, {0,0,1,1}, nd4j::DataType::FLOAT32);
    NDArray boxI('c', {1}, {0}, nd4j::DataType::INT32);
    NDArray cropSize = NDArrayFactory::create<int>({3, 3});

    //NDArray<float> ('c', {6}, {0.9f, .75f, .6f, .95f, .5f, .3f});
    NDArray expected('c', {1,3,3,1}, {1, 2.f, 2.f, 3.f, 4, 4.f, 3.f, 4.f, 4.f}, nd4j::DataType::FLOAT32);

    nd4j::ops::crop_and_resize op;
    auto results = op.execute({&images, &boxes, &boxI, &cropSize}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer("Cropped and Resized");
    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_CropAndResize_5) {

    NDArray images('c', {1, 100, 100, 3}, nd4j::DataType::FLOAT32);
    NDArray boxes('c', {1,4}, {0,0,1,1}, nd4j::DataType::FLOAT32);
    NDArray boxI('c', {2}, {1,1}, nd4j::DataType::INT32);
    NDArray cropSize = NDArrayFactory::create<int>({10, 10});

    //NDArray<float> ('c', {6}, {0.9f, .75f, .6f, .95f, .5f, .3f});
    NDArray expected('c', {1, 10, 10,3}, nd4j::DataType::FLOAT32);

    nd4j::ops::crop_and_resize op;
    auto results = op.execute({&images, &boxes, &boxI, &cropSize}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(result));
    //ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_DrawBoundingBoxes_1) {
    NDArray images = NDArrayFactory::create<float>('c', {2,4,5,3});
    NDArray boxes = NDArrayFactory::create<float>('c', {2, 2, 4}, {
        0.f , 0.f , 1.f , 1.f ,     0.1f, 0.2f, 0.9f, 0.8f,
        0.3f, 0.3f, 0.7f, 0.7f,     0.4f, 0.4f, 0.6f, 0.6f
    });

    NDArray colors = NDArrayFactory::create<float>('c', {2, 3}, {201.f, 202.f, 203.f, 127.f, 128.f, 129.f});

    //NDArray<float> ('c', {6}, {0.9f, .75f, .6f, .95f, .5f, .3f});
    NDArray expected = NDArrayFactory::create<float>('c', {2,4,5,3}, {
        127.f, 128.f,  129.f,   127.f, 128.f, 129.f,    127.f, 128.f, 129.f,    127.f, 128.f, 129.f,    201.f, 202.f, 203.f,
        127.f, 128.f,  129.f,    19.f,  20.f,  21.f,     22.f,  23.f,  24.f,    127.f, 128.f, 129.f,    201.f, 202.f, 203.f,
        127.f, 128.f,  129.f,   127.f, 128.f, 129.f,    127.f, 128.f, 129.f,    127.f, 128.f, 129.f,    201.f, 202.f, 203.f,
        201.f, 202.f,  203.f,   201.f, 202.f, 203.f,    201.f, 202.f, 203.f,    201.f, 202.f, 203.f,    201.f, 202.f, 203.f,

        61.f,  62.f,   63.f,    201.f, 202.f, 203.f,    201.f, 202.f, 203.f,     70.f,  71.f,  72.f,     73.f,  74.f,  75.f,
        76.f,  77.f,   78.f,    127.f, 128.f, 129.f,    127.f, 128.f, 129.f,     85.f,  86.f,  87.f,     88.f,  89.f,  90.f,
        91.f,  92.f,   93.f,    201.f, 202.f, 203.f,    201.f, 202.f, 203.f,    100.f, 101.f, 102.f,    103.f, 104.f, 105.f,
       106.f, 107.f,  108.f,    109.f, 110.f, 111.f,    112.f, 113.f, 114.f,    115.f, 116.f, 117.f,    118.f, 119.f, 120.f
    });
    images.linspace(1.);
    nd4j::ops::draw_bounding_boxes op;
    auto results = op.execute({&images, &boxes, &colors}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    result->syncToHost();
//    result->printBuffer("Bounded boxes");
//    expected.printBuffer("Bounded expec");
    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_DrawBoundingBoxes_2) {
    NDArray images = NDArrayFactory::create<float>('c', {1,9,9,1});
    NDArray boxes = NDArrayFactory::create<float>('c', {1, 1, 4}, {0.2f, 0.2f, 0.7f, 0.7f});
    NDArray colors = NDArrayFactory::create<float>('c', {1, 1}, {0.95f});

    //NDArray<float> ('c', {6}, {0.9f, .75f, .6f, .95f, .5f, .3f});
    NDArray expected = NDArrayFactory::create<float>('c', {1,9,9,1}, {
             1.1f ,   2.1f,  3.1f,   4.1f,    5.1f,   6.1f,  7.1f ,  8.1f ,  9.1f ,
            10.1f ,  0.95f,  0.95f, 0.95f,   0.95f,  0.95f, 16.1f , 17.1f , 18.1f ,
            19.1f ,  0.95f,  21.1f, 22.1f,   23.1f,  0.95f, 25.1f , 26.1f , 27.1f ,
            28.1f ,  0.95f,  30.1f, 31.1f,   32.1f,  0.95f, 34.1f , 35.1f , 36.1f ,
            37.1f ,  0.95f,  39.1f, 40.1f,   41.1f,  0.95f, 43.1f , 44.1f , 45.1f ,
            46.1f ,  0.95f,  0.95f, 0.95f,   0.95f,  0.95f, 52.1f , 53.1f , 54.1f ,
            55.1f ,  56.1f,  57.1f, 58.1f,   59.1f , 60.1f, 61.1f , 62.1f , 63.1f ,
            64.1f ,  65.1f,  66.1f, 67.1f,   68.1f , 69.1f, 70.1f , 71.1f , 72.1f ,
            73.1f ,  74.1f,  75.1f, 76.1f,   77.1f , 78.1f, 79.1f , 80.1f , 81.1f    });
    images.linspace(1.1);
    nd4j::ops::draw_bounding_boxes op;
    auto results = op.execute({&images, &boxes, &colors}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
//    result->syncToHost();
//    result->printBuffer("Bounded boxes 2");
//    expected.printBuffer("Bounded expec 2");
    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, Image_DrawBoundingBoxes_3) {
    NDArray images = NDArrayFactory::create<float>('c', {2,5,5,1}, {0.7788f, 0.8012f, 0.7244f, 0.2309f, 0.7271f, 0.1804f,
                                                                    0.5056f, 0.8925f, 0.5461f, 0.9234f, 0.0856f, 0.7938f,
                                                                    0.6591f, 0.5555f, 0.1596f, 0.3087f, 0.1548f, 0.4695f,
                                                                    0.9939f, 0.6113f, 0.6765f, 0.1800f, 0.6750f, 0.2246f,
                                                                    0.0509f, 0.4601f, 0.8284f, 0.2354f, 0.9752f, 0.8361f,
                                                                    0.2585f, 0.4189f, 0.7028f, 0.7679f, 0.5373f, 0.7234f,
                                                                    0.2690f, 0.0062f, 0.0327f, 0.0644f, 0.8428f, 0.7494f,
                                                                    0.0755f, 0.6245f, 0.3491f, 0.5793f, 0.5730f, 0.1822f,
                                                                    0.6420f, 0.9143f});

            NDArray boxes = NDArrayFactory::create<float>('c', {2, 2, 4}, {0.7717f,     0.9281f,     0.9846f,     0.4838f,
                                                                           0.6433f,     0.6041f,     0.6501f,     0.7612f,
                                                                           0.7605f,     0.3948f,     0.9493f,     0.8600f,
                                                                           0.7876f,     0.8945f,     0.4638f,     0.7157f});
    NDArray colors = NDArrayFactory::create<float>('c', {1, 2}, {0.9441f, 0.5957f});

    //NDArray<float> ('c', {6}, {0.9f, .75f, .6f, .95f, .5f, .3f});
//    NDArray expected = NDArrayFactory::create<float>('c', {2,5,5,1}, {
//            0.7788f, 0.8012f, 0.7244f, 0.2309f, 0.7271f,
//            0.1804f, 0.5056f, 0.8925f, 0.5461f, 0.9234f, 0.0856f, 0.7938f, 0.9441f,
//            0.9441f, 0.1596f, 0.3087f, 0.1548f, 0.4695f, 0.9939f, 0.6113f, 0.6765f,
//            0.1800f, 0.6750f, 0.2246f, 0.0509f, 0.4601f, 0.8284f, 0.2354f, 0.9752f, 0.8361f,
//            0.2585f, 0.4189f,0.7028f,0.7679f,0.5373f,0.7234f,0.2690f,0.0062f,0.0327f,0.0644f,
//            0.8428f, 0.9441f,0.9441f,0.9441f,0.3491f,0.5793f,0.5730f,0.1822f,0.6420f,0.9143f  });
    NDArray expected = NDArrayFactory::create<float>('c', {2,5,5,1}, {
                                                                       0.7788f, 0.8012f,  0.7244f,   0.2309f,  0.7271f,
                                                                       0.1804f, 0.5056f,  0.8925f,   0.5461f,  0.9234f,
                                                                       0.0856f, 0.7938f,  0.9441f,   0.9441f,  0.1596f,
                                                                       0.3087f, 0.1548f,  0.4695f,   0.9939f,  0.6113f,
                                                                       0.6765f, 0.18f  ,  0.675f ,   0.2246f,  0.0509f,

                                                                       0.4601f, 0.8284f,  0.2354f,   0.9752f,  0.8361f,
                                                                       0.2585f, 0.4189f,  0.7028f,   0.7679f,  0.5373f,
                                                                       0.7234f, 0.269f ,  0.0062f,   0.0327f,  0.0644f,
                                                                       0.8428f, 0.9441f,  0.9441f,   0.9441f,  0.3491f,
                                                                       0.5793f, 0.573f ,  0.1822f,   0.642f ,  0.9143f});
    nd4j::ops::draw_bounding_boxes op;
    auto results = op.execute({&images, &boxes, &colors}, {}, {});
     ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
//    result->printBuffer("Boxes3 output");
//    expected.printBuffer("Boxes3 expect");

//    result->syncToHost();
//    result->printBuffer("Bounded boxes 2");
//    expected.printBuffer("Bounded expec 2");
    ASSERT_TRUE(expected.isSameShapeStrict(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, FakeQuantWithMinMaxVars_Test_1) {

    NDArray x('c', {2,3}, {-63.80f, -63.75f, -63.70f, -63.5f, 0.0f, 0.1f}, nd4j::DataType::FLOAT32);
    NDArray exp('c', {2,3},  {-63.75, -63.75, -63.75, -63.5, 0., 0.}, nd4j::DataType::FLOAT32);
    NDArray min('c', {},  {-63.65f}, nd4j::DataType::FLOAT32);
    NDArray max('c', {},  {0.1f}, nd4j::DataType::FLOAT32);

    nd4j::ops::fake_quant_with_min_max_vars op;
    auto results = op.execute({&x, &min, &max}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
//    result->printBuffer("Quantized");
//    exp.printBuffer("Expected");
    ASSERT_TRUE(exp.isSameShapeStrict(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, FakeQuantWithMinMaxVars_Test_2) {

    NDArray x = NDArrayFactory::create<double>('c', {2,3}, {-63.80, -63.75, -63.4, -63.5, 0.0, 0.1});
    NDArray exp = NDArrayFactory::create<double>('c', {2,3},  {-63.75, -63.75, -63.5 , -63.5 ,   0.  ,   0. });
    NDArray min = NDArrayFactory::create<double>(-63.65);
    NDArray max = NDArrayFactory::create<double>(0.1);

    nd4j::ops::fake_quant_with_min_max_vars op;
    auto results = op.execute({&x, &min, &max}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer("Quantized2");
    ASSERT_TRUE(exp.isSameShapeStrict(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, FakeQuantWithMinMaxVars_Test_3) {

    NDArray x = NDArrayFactory::create<double>('c', {1,2,3,1}, {-63.80, -63.75, -63.4, -63.5, 0.0, 0.1});
    NDArray exp = NDArrayFactory::create<double>('c', {1,2,3,1},  {-63.75, -63.75, -63.5 , -63.5 ,   0.  ,   0. });
    NDArray min = NDArrayFactory::create<double>('c', {1},{-63.65});
    NDArray max = NDArrayFactory::create<double>('c', {1}, {0.1});

    nd4j::ops::fake_quant_with_min_max_vars_per_channel op;
    auto results = op.execute({&x, &min, &max}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
    // result->printIndexedBuffer("Quantized2");
    ASSERT_TRUE(exp.isSameShapeStrict(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, FakeQuantWithMinMaxVars_Test_4) {

    NDArray x = NDArrayFactory::create<float>('c', {2,4,5,3});
    NDArray exp = NDArrayFactory::create<float>('c', {2,4,5,3},{
                  1.0588236,  1.9607843,  3.019608,  4.0588236,  5.098039,  6.039216,  7.0588236,  8.039216,  9.058824,
                 10.058824,  10.980392,  12.078432, 13.058824,  13.921569, 15.09804,  16.058825,  17.058825, 18.117647,
                 19.058825,  20.,        21.137257, 22.058825,  22.941177, 23.882355, 25.058825,  26.078432, 26.901962,
                 28.058825, 29.019608,   29.92157,  31.058825,  31.960785, 32.941177, 34.058823,  35.09804,  35.960785,
                 37.058823, 38.039215,   38.980392, 40.058823,  40.980392, 42.000004, 43.058826,  43.92157,  45.01961,
                 45.,       47.058823,   48.03922,  45.,        50.,       51.058826, 45.,        50.,       54.078434,
                 45.,       50.,         57.09804,  45.,        50.,       60.11765,  45.,        50.,       62.862747,
                 45.,       50.,         65.882355, 45.,        50.,       68.90196,  45.,        50.,       70.,
                 45.,       50.,         70.,       45.,        50.,       70.,       45.,        50.,       70.,
                 45.,       50.,         70.,       45.,        50.,       70.,       45.,        50.,       70.,
                 45.,       50.,         70.,       45.,        50.,       70.,       45.,        50.,       70.,
                 45.,       50.,         70.,       45.,        50.,       70.,       45.,        50.,       70.,
                 45.,       50.,         70.,       45.,        50.,       70.,       45.,        50.,       70.,
                 45.,       50.,        70.});
    NDArray min = NDArrayFactory::create<float>({20., 20., 20.});
    NDArray max = NDArrayFactory::create<float>({65., 70., 90.});
    x.linspace(1.);
    nd4j::ops::fake_quant_with_min_max_vars_per_channel op;
    auto results = op.execute({&x, &min, &max}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
//    result->printBuffer("Quantized per channels 4");
//    exp.printBuffer("Quantized per channest E");
//    auto diff = *result - exp;
//    diff.printIndexedBuffer("Difference");
    ASSERT_TRUE(exp.isSameShapeStrict(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

TEST_F(DeclarableOpsTests10, FakeQuantWithMinMaxVars_Test_5) {
    NDArray x = NDArrayFactory::create<float>('c', {2, 3, 5, 4});
    NDArray exp = NDArrayFactory::create<float>('c', {2, 3, 5, 4},{
            -19.92157   , -18.980392  , -18.039217  , -16.941177  ,
            -19.92157   , -18.980392  , -18.039217  , -16.941177  ,
            -19.92157   , -18.980392  , -18.039217  , -16.941177  ,
            -19.92157   , -18.980392  , -18.039217  , -16.941177  ,
            -19.92157   , -18.980392  , -18.039217  , -16.941177  ,
            -19.92157   , -18.980392  , -18.039217  , -16.941177  ,
            -19.92157   , -18.980392  , -18.039217  , -16.941177  ,
            -19.92157   , -18.980392  , -18.039217  , -16.941177  ,
            -19.92157   , -18.980392  , -18.039217  , -16.941177  ,
            -19.92157   , -18.980392  , -18.039217  , -16.941177  ,
            -19.92157   , -18.980392  , -18.039217  , -16.941177  ,
            -16.        , -15.058824  , -13.960785  , -13.0196085 ,
            -11.92157   , -10.980392  , -10.039217  ,  -8.941177  ,
            -8.000001  ,  -7.0588236 ,  -5.960785  ,  -5.0196085 ,
            -3.9215698 ,  -2.9803925 ,  -2.039217  ,  -0.94117737,
            0.        ,   0.94117737,   2.039215  ,   2.9803925 ,
            4.07843   ,   5.0196075 ,   5.960783  ,   7.0588226 ,
            8.        ,   8.941177  ,  10.039215  ,  10.980392  ,
            12.07843   ,  13.019608  ,  13.960783  ,  15.058823  ,
            16.        ,  16.941177  ,  18.039217  ,  18.980392  ,
            20.07843   ,  21.019608  ,  21.960783  ,  23.058823  ,
            20.07843   ,  21.019608  ,  21.960783  ,  23.058823  ,
            20.07843   ,  21.019608  ,  21.960783  ,  23.058823  ,
            20.07843   ,  21.019608  ,  21.960783  ,  23.058823  ,
            20.07843   ,  21.019608  ,  21.960783  ,  23.058823  ,
            20.07843   ,  21.019608  ,  21.960783  ,  23.058823  ,
            20.07843   ,  21.019608  ,  21.960783  ,  23.058823  ,
            20.07843   ,  21.019608  ,  21.960783  ,  23.058823  ,
            20.07843   ,  21.019608  ,  21.960783  ,  23.058823  ,
            20.07843   ,  21.019608  ,  21.960783  ,  23.058823
    });
    NDArray min = NDArrayFactory::create<float>({-20., -19., -18., -17});
    NDArray max = NDArrayFactory::create<float>({20., 21., 22., 23});
    x.linspace(-60.);
    nd4j::ops::fake_quant_with_min_max_vars_per_channel op;
    auto results = op.execute({&x, &min, &max}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
//    result->printBuffer("Quantized per channels 5");
//    exp.printBuffer("Quantized per channest E");
//    auto diff = *result - exp;
//    diff.printIndexedBuffer("Difference");

    ASSERT_TRUE(exp.isSameShapeStrict(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

/*    public void testFakeQuantAgainstTF_1() {
        INDArray x = Nd4j.createFromArray(new float[]{ 0.7788f,0.8012f, 0.7244f, 0.2309f,0.7271f,
                0.1804f,    0.5056f,    0.8925f,    0.5461f,    0.9234f,
                0.0856f,    0.7938f,    0.6591f,    0.5555f,    0.1596f}).reshape(3,5);
        INDArray min = Nd4j.createFromArray(new float[]{-0.2283f,   -0.0719f,   -0.0154f,   -0.5162f,   -0.3567f}).reshape(1,5);
        INDArray max = Nd4j.createFromArray(new float[]{0.9441f,    0.5957f,    0.8669f,    0.3502f,    0.5100f}).reshape(1,5);

        INDArray out = Nd4j.createUninitialized(x.shape());
        val op = new FakeQuantWithMinMaxVarsPerChannel(x,min,max,out);

        INDArray expected = Nd4j.createFromArray(new float[]{0.7801f,    0.5966f,    0.7260f,   0.2320f,    0.5084f,
                0.1800f,    0.5046f,    0.8684f,    0.3513f,    0.5084f,
                0.0877f,    0.5966f,    0.6600f,    0.3513f,    0.1604f}).reshape(3,5);

        assertEquals(expected, out);
    }*/
TEST_F(DeclarableOpsTests10, FakeQuantWithMinMaxVars_Test_6) {
    NDArray x = NDArrayFactory::create<float>('c', {3, 5}, {0.7788f,0.8012f, 0.7244f, 0.2309f,0.7271f,
                                                            0.1804f,    0.5056f,    0.8925f,    0.5461f,    0.9234f,
                                                            0.0856f,    0.7938f,    0.6591f,    0.5555f,    0.1596f});
//    NDArray exp = NDArrayFactory::create<float>('c', {3, 5},{
//            0.7801f,    0.5966f,    0.7260f,   0.2320f,    0.5084f,
//            0.1800f,    0.5046f,    0.8684f,    0.3513f,    0.5084f,
//            0.0877f,    0.5966f,    0.6600f,    0.3513f,    0.1604f
//    });

    NDArray exp = NDArrayFactory::create<float>('c', {3,5}, {0.77700233, 0.596913,   0.72314,    0.23104,    0.50982356,
    0.17930824, 0.50528157, 0.86846,    0.34995764, 0.50982356,
    0.08735529, 0.596913,   0.6574,     0.34995764, 0.15974471});
    NDArray min = NDArrayFactory::create<float>('c', {5}, {-0.2283f,   -0.0719f,   -0.0154f,   -0.5162f,   -0.3567f});
    NDArray max = NDArrayFactory::create<float>('c', {5}, {0.9441f,    0.5957f,    0.8669f,    0.3502f,    0.5100f});
   // x.linspace(-60.);
    nd4j::ops::fake_quant_with_min_max_vars_per_channel op;
    auto results = op.execute({&x, &min, &max}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto result = results->at(0);
//    result->printBuffer("Quantized per channels 5");
//    exp.printBuffer("Quantized per channest E");
//    auto diff = *result - exp;
//    diff.printIndexedBuffer("Difference");

    ASSERT_TRUE(exp.isSameShapeStrict(result));
    ASSERT_TRUE(exp.equalsTo(result));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, batchnorm_test1) {

    NDArray input   ('c', {2,4}, nd4j::DataType::FLOAT32);
    NDArray mean    ('c', {4}, {1.05, 1.15, 1.2, 1.3}, nd4j::DataType::FLOAT32);
    NDArray variance('c', {4}, {0.5, 0.7, 0.9,  1.1},  nd4j::DataType::FLOAT32);
    NDArray gamma   ('c', {4}, {-1.2, 1.3, -1.4, 1.5}, nd4j::DataType::FLOAT32);
    NDArray beta    ('c', {4}, {10, 20, -10, -20},     nd4j::DataType::FLOAT32);

    NDArray expected('c', {2,4}, {11.61218734,  18.52390321,  -8.67185076, -21.28716864, 10.93337162,  19.14541765, -9.26213931, -20.71509369}, nd4j::DataType::FLOAT32);

    input.linspace(0.1, 0.1);

    nd4j::ops::batchnorm op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta}, {1e-5}, {1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);
    // output->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests10, batchnorm_test2) {

    auto input    = NDArrayFactory::create<TypeParam>('c', {2,3,4});
    auto mean     = NDArrayFactory::create<TypeParam>('c', {4});
    auto variance = NDArrayFactory::create<TypeParam>('c', {4});
    auto gamma    = NDArrayFactory::create<TypeParam>('c', {4});
    auto beta     = NDArrayFactory::create<TypeParam>('c', {4});

    auto expected = NDArrayFactory::create<TypeParam>('c', {2,3,4}, {-0.52733537,-0.35763144,-0.18792751,-0.01822358, 0.15148035, 0.32118428, 0.49088821, 0.66059214, 0.83029607, 1.        , 1.16970393, 1.33940786,
                                            1.50911179, 1.67881572, 1.84851965, 2.01822358, 2.18792751, 2.35763144, 2.52733537, 2.6970393 , 2.86674323, 3.03644717, 3.2061511 , 3.37585503});

    input.linspace(0.1, 0.1);
    mean.assign(1.);
    variance.assign(0.5);
    gamma.assign(1.2);
    beta.assign(1.);

    nd4j::ops::batchnorm op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta}, {1e-5}, {1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);
    // output->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests10, batchnorm_test3) {

    auto input    = NDArrayFactory::create<TypeParam>('c', {2,3,4});
    auto mean     = NDArrayFactory::create<TypeParam>('c', {3}, {1.05, 1.1, 1.15});
    auto variance = NDArrayFactory::create<TypeParam>('c', {3}, {0.5, 0.6, 0.7});
    auto gamma    = NDArrayFactory::create<TypeParam>('c', {3}, {1.2, 1.3, 1.4});
    auto beta     = NDArrayFactory::create<TypeParam>('c', {3}, {0.1, 0.2, 0.3});

    auto expected = NDArrayFactory::create<TypeParam>('c', {2,3,4}, {-1.51218734,-1.34248341,-1.17277948,-1.00307555,-0.80696728,-0.6391394 ,-0.47131152,-0.30348364,-0.11832703, 0.04900378, 0.21633459, 0.38366541,
                                            0.52425983, 0.69396376, 0.86366769, 1.03337162, 1.20696728, 1.37479516, 1.54262304, 1.71045092, 1.8896427 , 2.05697351, 2.22430432, 2.39163513,});

    input.linspace(0.1, 0.1);

    nd4j::ops::batchnorm op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta}, {1e-5}, {1,1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////
TYPED_TEST(TypedDeclarableOpsTests10, batchnorm_test4) {

    auto input    = NDArrayFactory::create<TypeParam>('c', {2,3,4});
    auto mean     = NDArrayFactory::create<TypeParam>('c', {2,1,4}, {1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4});
    auto variance = NDArrayFactory::create<TypeParam>('c', {2,1,4}, {0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2});
    auto gamma    = NDArrayFactory::create<TypeParam>('c', {2,1,4}, {1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9});
    auto beta     = NDArrayFactory::create<TypeParam>('c', {2,1,4}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.66, 0.7, 0.8});

    auto expected = NDArrayFactory::create<TypeParam>('c', {2,3,4}, {-1.51218734,-1.31045092,-1.12231189,-0.9416324 ,-0.83337162,-0.6391394 ,-0.45298865,-0.2708162 ,-0.1545559 , 0.03217212, 0.21633459, 0.4,
                                            0.58432694, 0.82999915, 0.95743373, 1.14688951, 1.25894242, 1.50999575, 1.64392367, 1.84066852, 1.93355791, 2.18999235, 2.33041362, 2.53444754});

    input.linspace(0.1, 0.1);

    nd4j::ops::batchnorm op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta}, {1e-5}, {1,1,0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, batchnorm_test5) {

    NDArray input   ('c', {2,4,2,2}, nd4j::DataType::FLOAT32);
    NDArray mean    ('c', {4}, {1.05, 1.15, 1.2, 1.3}, nd4j::DataType::FLOAT32);
    NDArray variance('c', {4}, {0.5, 0.7, 0.9,  1.1},  nd4j::DataType::FLOAT32);
    NDArray gamma   ('c', {4}, {-1.2, 1.3, -1.4, 1.5}, nd4j::DataType::FLOAT32);
    NDArray beta    ('c', {4}, {10, 20, -10, -20},     nd4j::DataType::FLOAT32);

    NDArray expected('c', {2,4,2,2}, {11.612187,  11.442483, 11.272779,  11.103076, 18.990039,  19.145418, 19.300796,  19.456175, -9.557284,  -9.704856, -9.852428, -10., -20.,
                                      -19.856981, -19.713963, -19.570944, 8.896924,   8.727221, 8.557517,   8.387813, 21.476097,  21.631475, 21.786854,  21.942233, -11.918438,
                                      -12.06601 , -12.213582, -12.361154, -17.7117, -17.568681, -17.425663, -17.282644}, nd4j::DataType::FLOAT32);
    input.linspace(0.1, 0.1);

    nd4j::ops::batchnorm op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta}, {1e-5}, {1,1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);
    // output->printBuffer();

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, batchnorm_test6) {

    NDArray input   ('c', {2,2,2,4}, nd4j::DataType::FLOAT32);
    NDArray mean    ('c', {4}, {1.05, 1.15, 1.2, 1.3}, nd4j::DataType::FLOAT32);
    NDArray variance('c', {4}, {0.5, 0.7, 0.9,  1.1},  nd4j::DataType::FLOAT32);
    NDArray gamma   ('c', {4}, {-1.2, 1.3, -1.4, 1.5}, nd4j::DataType::FLOAT32);
    NDArray beta    ('c', {4}, {10, 20, -10, -20},     nd4j::DataType::FLOAT32);

    NDArray expected('c', {2,2,2,4}, {11.612187,  18.523903,  -8.671851, -21.287169, 10.933372,  19.145418,  -9.262139, -20.715094, 10.254556,  19.766932,  -9.852428, -20.143019, 9.57574 ,
                                    20.388447, -10.442716, -19.570944,8.896924,  21.009961, -11.033005, -18.998869, 8.218109,  21.631475, -11.623294, -18.426794, 7.539293,  22.25299 ,
                                    -12.213582, -17.854719, 6.860477,  22.874504, -12.803871, -17.282644}, nd4j::DataType::FLOAT32);
    input.linspace(0.1, 0.1);

    nd4j::ops::batchnorm op;

    auto results = op.execute({&input, &mean, &variance, &gamma, &beta}, {1e-5}, {1,1,3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, bool_broadcast_test_1) {

    NDArray arr1('c', {2,2,1}, {1, 2, 3, 4}, nd4j::DataType::INT32);
    NDArray arr2('c', {  2,2}, {0, 1, 0, 4}, nd4j::DataType::INT32);

    NDArray expd('c', {2,2,2}, {0,1,0,0, 0,0,0,1}, nd4j::DataType::BOOL);

    NDArray result('c', {2,2,2}, nd4j::DataType::BOOL);

    arr1.applyTrueBroadcast(nd4j::BroadcastBoolOpsTuple::custom(scalar::EqualTo, pairwise::EqualTo, broadcast::EqualTo), &arr2, &result, true, nullptr);
    // result.printIndexedBuffer();
    // expd.printIndexedBuffer();

    ASSERT_TRUE(expd.isSameShape(result));
    ASSERT_TRUE(expd.equalsTo(result));
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests10, printIndexedTest_1) {

    NDArray arr('c', {2,2,2,2}, {1, 2, 3, 4, 5, 6, 7, 8,9, 10, 11, 12, 13, 14, 15, 16}, nd4j::DataType::INT32);
//    NDArray arr2('c', {  2,2}, {0, 1, 0, 4}, nd4j::DataType::INT32);

//    NDArray expd('c', {2,2,2}, {0,1,0,0, 0,0,0,1}, nd4j::DataType::BOOL);

//    NDArray result('c', {2,2,2}, nd4j::DataType::BOOL);

//    arr1.applyTrueBroadcast(nd4j::BroadcastBoolOpsTuple::custom(scalar::EqualTo, pairwise::EqualTo, broadcast::EqualTo), &arr2, &result, true, nullptr);
    // result.printIndexedBuffer();
    // expd.printIndexedBuffer();

//    ASSERT_TRUE(expd.isSameShape(result));
//    ASSERT_TRUE(expd.equalsTo(result));
    // arr.printIndexedBuffer("Test Print"); // output as [1, 2, 3, 4, 5, 6, 7, 8]
//
// we want output as
//  [[[1 2]
//    [3 4]]
//
//   [[5 6]
//    [7 8]]]
//
    ResultSet* lastDims = arr.allTensorsAlongDimension({3}); // last dim
    size_t k = 0; // k from 0 to lastDims->size()
    Nd4jLong rank = 4; // in this case
    printf("[");
    for (Nd4jLong i = 0; i < rank - 1; i++) {

        for (Nd4jLong l = 0; l < i; ++l)
            printf("\n");
        printf("[");
        for (Nd4jLong j = 0; j < arr.sizeAt(i); j++) {
            //    if (!i)
            //        printf("[");
            //    else
            //        printf(" ");
            lastDims->at(k++)->printBuffer();
        //if (k == arr.sizeAt(i))
        //    printf("]\n");
        }
        printf("]\n");
    }
    printf("]\n");
    delete lastDims;

}


