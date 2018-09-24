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
// Created by raver119 on 09.02.18.
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests6 : public testing::Test {
public:

    DeclarableOpsTests6() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests6, Test_Dilation2D_Again_1) {
    auto x = NDArrayFactory::_create<double>('c', {4, 128, 128, 4});
    auto w = NDArrayFactory::_create<double>('c', {4, 5, 4});
    auto exp = NDArrayFactory::_create<double>('c', {4, 64, 43, 4});


    nd4j::ops::dilation2d op;
    auto result = op.execute({&x, &w}, {}, {1, 1,5,7,1, 1,2,3,1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests6, Test_Dilation2D_Again_2) {
    auto x = NDArrayFactory::_create<double>('c', {4, 26, 19, 4});
    auto w = NDArrayFactory::_create<double>('c', {11, 7, 4});

    nd4j::ops::dilation2d op;
    auto result = op.execute({&x, &w}, {}, {0, 1,2,3,1, 1,3,2,1});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_StridedSlice_Once_Again_1) {
    auto matrix = NDArrayFactory::_create<double>('c', {5, 2});
    auto b = NDArrayFactory::_create<double>(0.0f);
    auto e = NDArrayFactory::_create<double>(1.0f);
    auto s = NDArrayFactory::_create<double>(1.0f);

    auto exp = NDArrayFactory::_create<double>('c', {2}, {1.0f, 2.0f});

    matrix.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix, &b, &e, &s}, {}, {0, 0, 0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);    

    ASSERT_EQ(exp, *z);

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_StridedSlice_Once_Again_2) {
    auto matrix = NDArrayFactory::_create<double>('c', {5, 2});
    auto b = NDArrayFactory::_create<double>('c', {1}, {0.0f});
    auto e = NDArrayFactory::_create<double>('c', {1}, {1.0f});
    auto s = NDArrayFactory::_create<double>('c', {1}, {1.0f});

    auto exp = NDArrayFactory::_create<double>('c', {2}, {1.0f, 2.0f});

    matrix.linspace(1);

    nd4j::ops::strided_slice op;
    auto result = op.execute({&matrix, &b, &e, &s}, {}, {0, 0, 0, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);    

    ASSERT_EQ(exp, *z);

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Simple_Scalar_1) {
    auto x = NDArrayFactory::_create<double>('c', {1, 1}, {2.0f});
    auto exp = NDArrayFactory::_create<double>('c', {1, 1}, {4.0f});

    nd4j::ops::test_scalar op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests6, Test_gather_Edge_1) {
    auto x = NDArrayFactory::_create<double>('c', {2, 4, 3, 2});
    auto indices = NDArrayFactory::_create<double>('c', {2}, {1.f, 0.f});

    nd4j::ops::gather op;
    auto result = op.execute({&x, &indices}, {}, {-2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_gatherNd_Edge_1) {
    auto x = NDArrayFactory::_create<double>('c', {2, 4, 2, 2});
    auto indices = NDArrayFactory::_create<double>('c', {3, 3}, {0,2,1, 0,1,0, 1,3,1});
    auto exp = NDArrayFactory::_create<double>('c', {3,2}, {11.f, 12.f, 5.f, 6.f, 31.f, 32.f});
    x.linspace(1);

    nd4j::ops::gather_nd op;
    auto result = op.execute({&x, &indices}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer();
    //z->printShapeInfo("z shape");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}



TEST_F(DeclarableOpsTests6, Test_StB_1) {
    auto x = NDArrayFactory::_create<double>('c', {4, 64, 64, 4});
    auto blocks = NDArrayFactory::_create<double>('c', {2}, {8, 8});
    auto paddings = NDArrayFactory::_create<double>('c', {2, 2}, {12, 12, 16, 16});

    x.assign(1.0f);

    nd4j::ops::space_to_batch op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    //nd4j_printf("Mean: %f\n", z->meanNumber());

    delete result;

}

TEST_F(DeclarableOpsTests6, Test_StB_2) {
    auto x = NDArrayFactory::_create<double>('c', {2, 6, 6, 2});
    auto blocks = NDArrayFactory::_create<double>('c', {2}, {2, 2});
    auto paddings = NDArrayFactory::_create<double>('c', {2, 2}, {2, 2, 2, 2});

    x.assign(1.0f);

    nd4j::ops::space_to_batch op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    delete result;

}

TEST_F(DeclarableOpsTests6, Test_BtS_1) {
    auto x = NDArrayFactory::_create<double>('f', {256, 8, 8, 2});
    auto blocks = NDArrayFactory::_create<double>('c',{2}, {8, 8});
    auto crops = NDArrayFactory::_create<double>('c', {2, 2});

    nd4j::ops::batch_to_space op;
    auto result = op.execute({&x, &blocks, &crops}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Order_1) {
    auto x = NDArrayFactory::_create<double>('f', {2, 3});
    auto exp = NDArrayFactory::_create<double>('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    x.linspace(1);

    nd4j::ops::order op;
    auto result = op.execute({&x}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));
    ASSERT_NE(x.ordering(), z->ordering());

    delete result;
}


TEST_F(DeclarableOpsTests6, Test_CumSum_Inclusive_Reverse_1) {
    auto x = NDArrayFactory::_create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto exp = NDArrayFactory::_create<double>('c', {3, 3}, {12.f, 15.f, 18.f, 11.f, 13.f, 15.f, 7.f, 8.f, 9.f});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_CumSum_Inclusive_Reverse_2) {
    auto x = NDArrayFactory::_create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto exp = NDArrayFactory::_create<double>('c', {3, 3}, {6.f, 5.f, 3.f, 15.f, 11.f, 6.f, 24.f, 17.f, 9.f,});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {0, 1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_CumSum_Exclusive_Reverse_1) {
    auto x = NDArrayFactory::_create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto exp = NDArrayFactory::_create<double>('c', {3, 3}, {11.f, 13.f, 15.f, 7.f, 8.f, 9.f, 0.f, 0.f, 0.f});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {1, 1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_CumSum_Exclusive_Reverse_2) {
    auto x = NDArrayFactory::_create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto exp = NDArrayFactory::_create<double>('c', {3, 3}, {5.f, 3.f, 0.f, 11.f, 6.f, 0.f, 17.f, 9.f, 0.f});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x}, {}, {1, 1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_CumSum_Exclusive_Reverse_2_1) {
    auto x = NDArrayFactory::_create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto axis = NDArrayFactory::_create<double>('c', {1}, {1});
    auto exp = NDArrayFactory::_create<double>('c', {3, 3}, {5.f, 3.f, 0.f, 11.f, 6.f, 0.f, 17.f, 9.f, 0.f});

    nd4j::ops::cumsum op;
    auto result = op.execute({&x, &axis}, {}, {1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, TestDropout_1) {

    auto x = NDArrayFactory::_create<double>('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    auto shape = NDArrayFactory::_create<double>({2.f, 2.f});
    nd4j::ops::dropout op;

    auto ress = op.execute({&x, &shape}, {0.2f}, {113});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    //ress->at(0)->printIndexedBuffer("Result is ");
    //x.printIndexedBuffer("Input is");

    delete ress;
}


TEST_F(DeclarableOpsTests6, TestDropout_2) {
//    auto x0 = NDArrayFactory::_create<double>('c', {10, 10});
//    auto x1 = NDArrayFactory::_create<double>('c', {10, 10});
    auto x = NDArrayFactory::_create<double>('c', {3, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});

    nd4j::ops::dropout op;

    auto ress = op.execute({&x}, {0.4f}, {113});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    //x.printIndexedBuffer("Input is");
    //ress->at(0)->printIndexedBuffer("Result is ");

    delete ress;
}

TEST_F(DeclarableOpsTests6, TestDropout_3) {
//    auto x0 = NDArrayFactory::_create<double>('c', {10, 10});
//    auto x1 = NDArrayFactory::_create<double>('c', {10, 10});
    auto x = NDArrayFactory::_create<double>('c', {2, 2, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    auto shape = NDArrayFactory::_create<double>({1.f, 2.f});

    nd4j::ops::dropout op;

    auto ress = op.execute({&x, &shape}, {0.4f}, {113});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    //x.printIndexedBuffer("Input is");
    //ress->at(0)->printIndexedBuffer("Result is ");

    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MaxPoolWithArgmax_1) {

    auto x = NDArrayFactory::_create<double>('c', {2, 2, 2, 4}, {
             5.5, 0.,   0.3,  5.5,
             1.5, 0.,   1.3,  6.5,
             8.6, 0.,    0.,  0.4,
             2.5, 1.,   0.3,  4.5,
             1.5, 1.,   1.3,  1.5,
             3.5, 0.,   1.3,  2.5,
             2.6, 2.,    3.,  1.4,
             4.5, 1.,   0.3,  0.5}
    );       
    auto expI = NDArrayFactory::_create<double>('c', {2, 2, 2, 4}, {
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15,
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15}
    );

    nd4j::ops::max_pool_with_argmax op;

    auto ress = op.execute({&x}, {}, {1,1,1,1,1,1,1,1,1});

    
    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ASSERT_TRUE(expI.isSameShape(ress->at(0)));
    ASSERT_TRUE(expI.isSameShape(ress->at(1)));
    ASSERT_TRUE(x.equalsTo(ress->at(0)));
    ASSERT_TRUE(expI.equalsTo(ress->at(1)));
    //x.printIndexedBuffer("Input is");
    //ress->at(0)->printIndexedBuffer("Result is ");
    ASSERT_TRUE(expI.equalsTo(ress->at(1)));
    
    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, SufficientStatistics_1) {
//    auto x0 = NDArrayFactory::_create<double>('c', {10, 10});
//    auto x1 = NDArrayFactory::_create<double>('c', {10, 10});
    auto x = NDArrayFactory::_create<double>('c', {2, 2, 2, 4}, {
        5.5, 0.,  0.3, 5.5,
        1.5, 0.,  1.3, 6.5,
        8.6, 0.,   0., 0.4,
        2.5, 1.,  0.3, 4.5,
        1.5, 1.,  1.3, 1.5,
        3.5, 0.,  1.3, 2.5,
        2.6, 2.,   3., 1.4,
        4.5, 1.,  0.3, 0.5}
    );
// ------------------------------------
    double count = 8.0;
    auto sumExp = NDArrayFactory::_create<double>({30.2, 5., 7.8, 22.8});
    auto sqrExp = NDArrayFactory::_create<double>({154.22,   7.,    14.34, 103.62});

    auto axis = NDArrayFactory::_create<double>({0.0, 1.0, 2.0});

    nd4j::ops::sufficient_statistics op;

    auto ress = op.execute({&x, &axis}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ASSERT_EQ(ress->at(0)->e<double>(0), count);
    ASSERT_TRUE(sumExp.equalsTo(ress->at(1)));
    ASSERT_TRUE(sqrExp.equalsTo(ress->at(2)));

    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, SufficientStatistics_2) {
//    auto x0 = NDArrayFactory::_create<double>('c', {10, 10});
//    auto x1 = NDArrayFactory::_create<double>('c', {10, 10});
    auto x = NDArrayFactory::_create<double>('c', {2, 2, 2, 4}, {
        5.5, 0.,  0.3, 5.5,
        1.5, 0.,  1.3, 6.5,
        8.6, 0.,   0., 0.4,
        2.5, 1.,  0.3, 4.5,
        1.5, 1.,  1.3, 1.5,
        3.5, 0.,  1.3, 2.5,
        2.6, 2.,   3., 1.4,
        4.5, 1.,  0.3, 0.5}
    );
// ------------------------------------
    double count = 4.0;
    auto sumExp = NDArrayFactory::_create<double>('c', {2, 4}, {
        18.2,        3.,         4.6,        8.8,
        12.,         2.,         3.2,        14.}
    );

    auto sqrExp = NDArrayFactory::_create<double>('c', {2, 4}, {
        113.22, 5., 10.78, 34.62,
           41., 2.,  3.56, 69.}
    );

    auto axis = NDArrayFactory::_create<double>({0.0, 1.0});

    nd4j::ops::sufficient_statistics op;

    auto ress = op.execute({&x, &axis}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, ress->status());
    ASSERT_EQ(ress->at(0)->e<double>(0), count);
    ASSERT_TRUE(sumExp.equalsTo(ress->at(1)));
    ASSERT_TRUE(sqrExp.equalsTo(ress->at(2)));

    delete ress;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_1) {

    auto x = NDArrayFactory::_create<double>('c', {2, 2, 2}, {
        1, 2, 0, 1, 2, 2, 1, 2}
    );
// ------------------------------------

    auto exp = NDArrayFactory::_create<double>({1., 3., 4.});

    nd4j::ops::bincount op;

    auto res = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_2) {

    auto x = NDArrayFactory::_create<double>('c', {2, 2, 2}, {
        1, 2, 0, 1, 2, 2, 1, 2}
    );

    auto weights = NDArrayFactory::_create<double>('c', {2, 2, 2}, {
        2, 1, 3, 1, 5, 1, 1, 6}
    );

// ------------------------------------

    auto exp = NDArrayFactory::_create<double>({3., 4., 13.});

    nd4j::ops::bincount op;

    auto res = op.execute({&x, &weights}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_3) {

    auto x = NDArrayFactory::_create<double>('c', {2, 2, 2}, {
        1, 2, 0, 1, 2, 2, 1, 2}
    );

    auto weights = NDArrayFactory::_create<double>('c', {2, 2, 2}, {
        2, 1, 3, 1, 5, 1, 1, 6}
    );

// ------------------------------------

    auto exp = NDArrayFactory::_create<double>({3., 4.});

    nd4j::ops::bincount op;

    auto res = op.execute({&x, &weights}, {}, {0, 2});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BinCount_4) {

    auto x = NDArrayFactory::_create<double>('c', {2, 2, 2}, {
        1, 2, 0, 1, 2, 2, 1, 2}
    );

    auto weights = NDArrayFactory::_create<double>('c', {2, 2, 2}, {
        2, 1, 3, 1, 5, 1, 1, 6}
    );

// ------------------------------------

    auto exp = NDArrayFactory::_create<double>({3., 4.,  13., 0.0});

    nd4j::ops::bincount op;

    auto res = op.execute({&x, &weights}, {}, {4, 4});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_1) {

    auto x = NDArrayFactory::_create<double>( {2., 2., 2.} );

    auto y = NDArrayFactory::_create<double>({ 2., 1., 2.});

// ------------------------------------

    auto exp = NDArrayFactory::_create<double>({2., 2., 2.});

    nd4j::ops::broadcast_dynamic_shape op;

    auto res = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_2) {

    auto x = NDArrayFactory::_create<double>( {2., 2.} );

    auto y = NDArrayFactory::_create<double>({2.0, 1.0, 2.0});

// ------------------------------------
    auto exp = NDArrayFactory::_create<double>({2., 2., 2.});

    nd4j::ops::broadcast_dynamic_shape op;

    auto res = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_3) {

    auto x = NDArrayFactory::_create<double>( {2., 2., 2.} );

    auto y = NDArrayFactory::_create<double>({ 2.0, 1.0});

// ------------------------------------

    auto exp = NDArrayFactory::_create<double>({2., 2., 2.});

    nd4j::ops::broadcast_dynamic_shape op;

    auto res = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

/////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, BroadcastDynamicShape_4) {

    auto x = NDArrayFactory::_create<double>({2., 2., 2.});

    auto y = NDArrayFactory::_create<double>({2., 2.});

// ------------------------------------

    auto exp = NDArrayFactory::_create<double>({2., 2., 2.});

    nd4j::ops::broadcast_dynamic_shape op;

    auto res = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, res->status());
    ASSERT_TRUE(exp.equalsTo(res->at(0)));

    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, ClipByGlobalNorm_1) {

    auto x = NDArrayFactory::_create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0}
    );

    auto exp = NDArrayFactory::_create<double>('c', {2, 3, 3}, {
            -0.2771281,  0.,          0.,
            0.36950415,  0.,          0.,
            -0.2771281,  0.,          0.,
            0.36950415,  0.,          0.,
            -0.2771281,  0.,          0.,
            0.36950415,  0.,          0.}
    );
//    8.660254
//    auto expNorm(8.660254);

    nd4j::ops::clip_by_global_norm op;
    auto result = op.execute({&x}, {0.8}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto norm = result->at(1);
    //z->printIndexedBuffer("Output");
    //exp.printIndexedBuffer("Expected");
    //norm->printIndexedBuffer("Norm");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
//    ASSERT_TRUE(expNorm.equalsTo(norm));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, ClipByGlobalNorm_2) {

    auto x = NDArrayFactory::_create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0}
    );

    auto a = NDArrayFactory::_create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                                      -3.0, 0.0, 0.0, 4.0, 0.0, 0.0}
    );

    auto exp = NDArrayFactory::_create<double>('c', {2, 3, 3}, {
                                    -0.44090813,   0.,          0.,
                                      0.5878775,   0.,          0.,
                                    -0.44090813,   0.,          0.,
                                      0.5878775,   0.,          0.,
                                    -0.44090813,   0.,          0.,
                                      0.5878775,   0.,          0.}
//12.247449

    );

    nd4j::ops::clip_by_global_norm op;
    auto result = op.execute({&x, &a}, {1.8}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto y = result->at(1);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.isSameShape(y));
    ASSERT_TRUE(exp.equalsTo(z));
    ASSERT_TRUE(exp.equalsTo(y));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, ClipByGlobalNorm_3) {

    auto x = NDArrayFactory::_create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    auto a = NDArrayFactory::_create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    auto exp = NDArrayFactory::_create<double>('c', {2, 3, 3}, {
            -0.19595918,  0.,          0.,
              0.2612789,  0.,          0.,
            -0.19595918,  0.,          0.,
              0.2612789,  0.,          0.,
            -0.19595918,  0.,          0.,
              0.2612789,   0.,          0.}
    );

    nd4j::ops::clip_by_global_norm op;
    auto result = op.execute({&x, &a}, {0.8}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    auto y = result->at(1);
    //z->printIndexedBuffer("Output 1");
    //y->printIndexedBuffer("Output 2");
    //result->at(2)->printIndexedBuffer("Global norm is");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.isSameShape(y));
    ASSERT_TRUE(result->at(2)->isScalar());
    ASSERT_TRUE(exp.equalsTo(z));
    ASSERT_TRUE(exp.equalsTo(y));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_1) {

    auto x = NDArrayFactory::_create<double>('c', {2, 3, 3}, {-3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, -3.0, 4.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 4.0});
    auto exp = NDArrayFactory::_create<double>({36.0, -48.0});

    nd4j::ops::matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_2) {

    auto x = NDArrayFactory::_create<double>('c', {2, 2, 2}, {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0});
    auto exp = NDArrayFactory::_create<double>({-2.0, -2.0});

    nd4j::ops::matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_3) {

    auto x = NDArrayFactory::_create<double>('c', {1, 3, 3}, {3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0});
    auto exp = NDArrayFactory::_create<double>({-54.0});

    nd4j::ops::matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_4) {

    auto x = NDArrayFactory::_create<double>('c', {1, 3, 3}, {12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 13.0});
    auto exp = NDArrayFactory::_create<double>({189.0});

    nd4j::ops::matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixDeterminant_5) {

    auto x = NDArrayFactory::_create<double>('c', {1, 4, 4});
    auto exp = NDArrayFactory::_create<double>({-16.0});
    x.linspace(1);
    x.p(5, 4.0);
    x.p(12, 12.0);

    nd4j::ops::matrix_determinant op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixInverse_1) {

    auto x = NDArrayFactory::_create<double>('c', {2, 5, 5}, {
                    2.,  4., 60.,  8., 10.,
                    0.,  1.,  2.,  3.,  4.,
                    0.,  0.,  2.,  4.,  6.,
                    0.,  0.,  0.,  1.,  2.,
                    0.,  0.,  0.,  0.,  4.,

                     1.,  0.,  0.,  0.,  0.,
                     2.,  1.,  0.,  0.,  0.,
                    30.,  2.,  1.,  0.,  0.,
                     4.,  3.,  2.,  1.,  0.,
                     5.,  4.,  3.,  2.,  1.,
    });

    auto exp = NDArrayFactory::_create<double>('c', {2, 5, 5}, {
                    0.5, -2.0, -13.0, 54.0, -6.75,
                    0.0,  1.0,  -1.0,  1.0,   0.0,
                      0,    0,   0.5, -2.0,  0.25,
                      0,    0,     0,  1.0,  -0.5,
                      0,    0,     0,    0,  0.25,
    
                    1.0,  0.0,  0.0,  0.0, 0.,
                   -2.0,  1.0,   0.,   0., 0.,
                  -26.0, -2.0,    1,    0, 0.,
                   54.0,  1.0, -2.0,    1, 0.,
                  -27.0,  0.0,  1.0, -2.0, 1.
    });

    nd4j::ops::matrix_inverse op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Output ");
//    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


////////////////////////////////////////////////////////////////////////////////
/*
TEST_F(DeclarableOpsTests6, MatrixInverse_2) {

    auto x = NDArrayFactory::_create<double>('c', {2, 5, 5}, {
                    1.,  2., 30.,  4.,  5.,
                    0.,  1.,  2.,  3.,  4.,
                    0.,  0.,  1.,  2.,  3.,
                    0.,  0.,  0.,  1.,  2.,
                    0.,  0.,  0.,  0.,  1.,

                     4.,   0.,  0.,  0.,  0.,
                     4.,   2.,  0.,  0.,  0.,
                    30.,   2.,  1.,  0.,  0.,
                     8.,   6.,  4.,  2.,  0.,
                    15.,  12.,  9.,  6.,  3.,
    });

    auto exp = NDArrayFactory::_create<double>('c', {2, 5, 5}, {
     1.0,  -2.0,  -26.0,  54.0, -27.0,
     0.0,   1.0,  -2.0,    1.0,   0.0,
     0.0,   0.0,   1.0,   -2.0,   1.0, 
     0.0,   0.0,   0.0,    1.0,  -2.0, 
     0.0,   0.0,   0.0,    0.0,   1.0, 

     0.25,  0.0,    0.0,   0.0,   0.0,
    -0.50,  0.5,    0.0,   0.0,   0.0,
    -6.50, -1.0,    1.0,   0.0,   0.0,
    13.50,  0.5,   -2.0,   0.5,   0.0,
    -6.75,  0.0,    1.0,  -1.0,   0.33333333
    });

    nd4j::ops::matrix_inverse op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    z->printIndexedBuffer("Output ");
    exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}
*/
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixInverse_3) {

    auto x = NDArrayFactory::_create<double>('c', {5, 5}, {
                     4.,   0.,  0.,  0.,  0.,
                     4.,   2.,  0.,  0.,  0.,
                    30.,   2.,  1.,  0.,  0.,
                     8.,   6.,  4.,  2.,  0.,
                    15.,  12.,  9.,  6.,  3.,
    });

    auto exp = NDArrayFactory::_create<double>('c', {5, 5}, {
     0.25,  0.0,    0.0,   0.0,   0.0,
    -0.50,  0.5,    0.0,   0.0,   0.0,
    -6.50, -1.0,    1.0,   0.0,   0.0,
    13.50,  0.5,   -2.0,   0.5,   0.0,
    -6.75,  0.0,    1.0,  -1.0,   0.33333333
    });

    nd4j::ops::matrix_inverse op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, MatrixInverse_4) {

    auto x = NDArrayFactory::_create<double>('c', {5, 5}, {
                    1.,  2., 30.,  4.,  5.,
                    0.,  1.,  2.,  3.,  4.,
                    0.,  0.,  1.,  2.,  3.,
                    0.,  0.,  0.,  1.,  2.,
                    0.,  0.,  0.,  0.,  1.
    });

    auto exp = NDArrayFactory::_create<double>('c', {5, 5}, {
     1.0,  -2.0,  -26.0,  54.0, -27.0,
     0.0,   1.0,  -2.0,    1.0,   0.0,
     0.0,   0.0,   1.0,   -2.0,   1.0, 
     0.0,   0.0,   0.0,    1.0,  -2.0, 
     0.0,   0.0,   0.0,    0.0,   1.0 
    });

    nd4j::ops::matrix_inverse op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    //z->printIndexedBuffer("Output ");
    //exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, ReluLayer_1) {
    auto x = NDArrayFactory::_create<double>('c', {3, 4}, {1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0, 8.0, 9.0, -10.0, 11.0, 12});
    auto w = NDArrayFactory::_create<double>('c', {4, 3}, {0.5, 0.1, 0.8, 0.5, 0.2, 0.5, 0.5, 0.25, 0.5, 0.1, 0.0, 0.25}); 
    auto b = NDArrayFactory::_create<double>({20.0, 30.0, 50.0});



    auto exp = NDArrayFactory::_create<double>('c', {3, 3}, {
                        21.4,  30.45, 52.3, 
                        23.8,  31.05, 56.5, 
                        26.2,  31.65, 60.7}); 

    nd4j::ops::relu_layer op;
    auto result = op.execute({&x, &w, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    // z->printShapeInfo("Output shape");
    // z->printIndexedBuffer("Output ");
    // exp.printIndexedBuffer("Expected ");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Gather_Discrepancy_119) {
    auto x = NDArrayFactory::_create<double>('c', {2, 2}, {1, 2, 3, 4});
    auto indices = NDArrayFactory::_create<double>('c', {2}, {1, 0});
    auto e = NDArrayFactory::_create<double>('c', {2, 2}, {3, 4, 1, 2});

    nd4j::ops::gather op;
    auto result = op.execute({&x, &indices}, {}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Gather_Discrepancy_119_2) {
    auto x = NDArrayFactory::_create<double>('c', {2, 2}, {1, 2, 3, 4});
    auto e = NDArrayFactory::_create<double>('c', {2, 2}, {3, 4, 1, 2});

    nd4j::ops::gather op;
    auto result = op.execute({&x}, {}, {0, 1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Reduce3_Edge) {
    auto x = NDArrayFactory::_create<double>('c', {3, 4, 5});
    auto y = NDArrayFactory::_create<double>('c', {3, 4, 5});


    std::vector<int> dims = {0, 1};
    auto z = x.applyReduce3(reduce3::CosineSimilarity, &y, dims, nullptr);
    ASSERT_TRUE(z != nullptr);

    delete z;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_rnn_test1) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::_create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::_create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::_create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::_create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::_create<double>('c', {bS, numUnits});
    auto maxTimeStep = NDArrayFactory::_create<double>('c', {bS}, {time-1, time-3});

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::_create<double>('c', {time, bS, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0.69882484, 0.69882484, 0.69882484, 0.69882484, 0.9312333 , 0.9312333 , 0.9312333 , 0.9312333 ,
                                                          0.93751527, 0.93751527, 0.93751527, 0.93751527,0.97136768, 0.97136768, 0.97136768, 0.97136768,0., 0., 0., 0.        ,
                                                          0.97732812, 0.97732812, 0.97732812, 0.97732812,0., 0., 0., 0.        ,0., 0., 0., 0.,0., 0., 0., 0.});
    
    auto expHFinal = NDArrayFactory::_create<double>('c', {bS, numUnits},       {0.97732812, 0.97732812, 0.97732812, 0.97732812, 0.93751527, 0.93751527, 0.93751527, 0.93751527});

    nd4j::ops::static_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_rnn_test2) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::_create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::_create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::_create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::_create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::_create<double>('c', {bS, numUnits});

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::_create<double>('c', {time, bS, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0.69882484, 0.69882484, 0.69882484, 0.69882484,0.9312333 , 0.9312333 , 0.9312333 , 0.9312333,
                                                          0.93751527, 0.93751527, 0.93751527, 0.93751527,0.97136768, 0.97136768, 0.97136768, 0.97136768,0.97338548, 0.97338548, 0.97338548, 0.97338548,
                                                          0.97732812, 0.97732812, 0.97732812, 0.97732812,0.97864398, 0.97864398, 0.97864398, 0.97864398,0.98000654, 0.98000654, 0.98000654, 0.98000654,
                                                          0.98112648, 0.98112648, 0.98112648, 0.98112648});

    auto expHFinal = NDArrayFactory::_create<double>('c', {bS, numUnits},       {0.98000654, 0.98000654, 0.98000654, 0.98000654,0.98112648, 0.98112648, 0.98112648, 0.98112648});

    nd4j::ops::static_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_rnn_test3) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::_create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::_create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::_create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::_create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::_create<double>('c', {bS, numUnits});
    auto maxTimeStep = NDArrayFactory::_create<double>('c', {bS}, {time-1, 0});

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::_create<double>('c', {time, bS, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0., 0., 0., 0., 0.9312333, 0.9312333, 0.9312333, 0.9312333,
                                                          0., 0., 0., 0.           , 0.97136768, 0.97136768, 0.97136768, 0.97136768,0., 0., 0., 0. , 
                                                          0.97732812, 0.97732812, 0.97732812, 0.97732812,0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.});
    
    auto expHFinal = NDArrayFactory::_create<double>('c', {bS, numUnits},       {0.97732812, 0.97732812, 0.97732812, 0.97732812, 0.2       , 0.2       , 0.2       , 0.2});

    nd4j::ops::static_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_rnn_test4) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::_create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::_create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::_create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::_create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::_create<double>('c', {bS, numUnits});
    auto maxTimeStep = NDArrayFactory::_create<double>('c', {bS}, {time-1, time-3});

    x.linspace(0.01, 0.01);
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::_create<double>('c', {time, bS, numUnits}, {0.47615493, 0.47615493, 0.47615493, 0.47615493,0.49676344, 0.49676344, 0.49676344, 0.49676344, 0.87018664, 0.87018664, 0.87018664, 0.87018664,
                                                          0.88400882, 0.88400882, 0.88400882, 0.88400882, 0.96529784, 0.96529784, 0.96529784, 0.96529784,0., 0., 0., 0.        , 
                                                          0.97688859, 0.97688859, 0.97688859, 0.97688859,0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.});
    
    auto expHFinal = NDArrayFactory::_create<double>('c', {bS, numUnits},       {0.97688859, 0.97688859, 0.97688859, 0.97688859, 0.88400882, 0.88400882, 0.88400882, 0.88400882});

    nd4j::ops::static_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_rnn_test5) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::_create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::_create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::_create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::_create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::_create<double>('c', {bS, numUnits});

    x.linspace(0.01, 0.01);
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::_create<double>('c', {time, bS, numUnits}, {0.47615493, 0.47615493, 0.47615493, 0.47615493,0.49676344, 0.49676344, 0.49676344, 0.49676344, 0.87018664, 0.87018664, 0.87018664, 0.87018664,
                                                          0.88400882, 0.88400882, 0.88400882, 0.88400882, 0.96529784, 0.96529784, 0.96529784, 0.96529784,0.96849345, 0.96849345, 0.96849345, 0.96849345,
                                                          0.97688859, 0.97688859, 0.97688859, 0.97688859,0.97831069, 0.97831069, 0.97831069, 0.97831069, 0.97997868, 0.97997868, 0.97997868, 0.97997868,
                                                          0.98110653, 0.98110653, 0.98110653, 0.98110653});

    auto expHFinal = NDArrayFactory::_create<double>('c', {bS, numUnits},       {0.97997868, 0.97997868, 0.97997868, 0.97997868, 0.98110653, 0.98110653, 0.98110653, 0.98110653});

    nd4j::ops::static_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_bidir_rnn_test1) {
    
    const int bS         = 4;
    const int inSize     = 4;    
    const int numUnitsFW = 3;    
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::_create<double>('c', {time, bS, inSize});
    auto WxFW = NDArrayFactory::_create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::_create<double>('c', {numUnitsFW, numUnitsFW});    
    auto bFW  = NDArrayFactory::_create<double>('c', {2*numUnitsFW});

    auto h0FW = NDArrayFactory::_create<double>('c', {bS, numUnitsFW});
    auto h0BW = NDArrayFactory::_create<double>('c', {bS, numUnitsBW});
    auto maxTimeStep = NDArrayFactory::_create<double>('c', {bS}, {time-1, time-3, time-4, 0});

    x.linspace(0.01, 0.01);
    h0FW = 0.2;    
    h0BW = 0.25;
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expH      = NDArrayFactory::_create<double>('c', {time, bS, numUnitsFW+numUnitsBW}, {0.43819931, 0.43819931, 0.43819931, 0.86708881, 0.86708881,0.86708881,0.47615493, 0.47615493, 0.47615493, 0.78347842, 0.78347842,0.78347842,
                                                                       0.51241561, 0.51241561, 0.51241561, 0.55529176, 0.55529176,0.55529176,0., 0., 0., 0., 0.,0.,0.73880324, 0.73880324, 0.73880324, 0.90935605, 0.90935605,
                                                                       0.90935605, 0.77843476, 0.77843476, 0.77843476, 0.64692945, 0.64692945,0.64692945,0., 0., 0., 0., 0.,0.,0., 0., 0., 0., 0.,0.,
                                                                       0.9052501, 0.9052501, 0.9052501, 0.9181592, 0.9181592, 0.9181592,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,
                                                                       0.9555734, 0.9555734, 0.9555734, 0.8026439, 0.8026439, 0.8026439,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,
                                                                       0., 0., 0., 0., 0., 0.,       0., 0., 0., 0., 0., 0.,       0., 0., 0., 0., 0., 0.,       0., 0., 0., 0., 0., 0.});
    
    auto expHFWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsFW},  {0.9555734 , 0.9555734 , 0.9555734 , 0.77843476, 0.77843476, 0.77843476, 0.51241561, 0.51241561, 0.51241561, 0.2, 0.2, 0.2});
    auto expHBWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsBW},  {0.86708881, 0.86708881, 0.86708881, 0.78347842, 0.78347842, 0.78347842, 0.55529176, 0.55529176, 0.55529176, 0.25, 0.25, 0.25});

    nd4j::ops::static_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &h0FW, &h0BW, &maxTimeStep}, {}, {});    

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFWfinal = results->at(1);
    auto hBWfinal = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_bidir_rnn_test2) {
    
    const int bS         = 4;
    const int inSize     = 4;    
    const int numUnitsFW = 3;    
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::_create<double>('c', {time, bS, inSize});
    auto WxFW = NDArrayFactory::_create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::_create<double>('c', {numUnitsFW, numUnitsFW});    
    auto bFW  = NDArrayFactory::_create<double>('c', {2*numUnitsFW});

    auto maxTimeStep = NDArrayFactory::_create<double>('c', {bS}, {time-1, time-3, time-4, 0});

    x.linspace(0.01, 0.01);
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expH      = NDArrayFactory::_create<double>('c', {time, bS, numUnitsFW+numUnitsBW}, {0.22602835, 0.22602835, 0.22602835, 0.86518273, 0.86518273,0.86518273,0.27105303, 0.27105303, 0.27105303, 0.66617761, 0.66617761,0.66617761,
                                                                       0.31492203, 0.31492203, 0.31492203, 0.31492203, 0.31492203,0.31492203,0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 
                                                                       0.60005558, 0.60005558, 0.60005558, 0.9029975 , 0.9029975 ,0.9029975 ,0.66138054, 0.66138054, 0.66138054, 0.43819931, 0.43819931,0.43819931,
                                                                       0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 
                                                                       0.87023975, 0.87023975, 0.87023975, 0.88852032, 0.88852032,0.88852032,0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,
                                                                       0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,
                                                                       0.95177305, 0.95177305, 0.95177305, 0.66737775, 0.66737775,0.66737775,0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,
                                                                       0.        , 0.        , 0.        , 0.        , 0.        ,0.        ,0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 
                                                                       0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0.});
    
    auto expHFWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsFW},  {0.95177305, 0.95177305, 0.95177305, 0.66138054, 0.66138054, 0.66138054, 0.31492203, 0.31492203, 0.31492203, 0.        , 0.        , 0.});
    auto expHBWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsBW},  {0.86518273, 0.86518273, 0.86518273, 0.66617761, 0.66617761, 0.66617761, 0.31492203, 0.31492203, 0.31492203, 0.        , 0.        , 0.});

    nd4j::ops::static_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &maxTimeStep}, {}, {});    

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFWfinal = results->at(1);
    auto hBWfinal = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, static_bidir_rnn_test3) {
    
    const int bS         = 4;
    const int inSize     = 4;    
    const int numUnitsFW = 3;    
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::_create<double>('c', {time, bS, inSize});
    auto WxFW = NDArrayFactory::_create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::_create<double>('c', {numUnitsFW, numUnitsFW});    
    auto bFW  = NDArrayFactory::_create<double>('c', {2*numUnitsFW});

    x.linspace(0.01, 0.01);
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expH      = NDArrayFactory::_create<double>('c', {time, bS, numUnitsFW+numUnitsBW}, {0.22602835, 0.22602835, 0.22602835, 0.86841012, 0.86841012,0.86841012,0.27105303, 0.27105303, 0.27105303, 0.88207531, 0.88207531,0.88207531,
                                                                       0.31492203, 0.31492203, 0.31492203, 0.8941667 , 0.8941667 ,0.8941667 ,0.35748551, 0.35748551, 0.35748551, 0.90489713, 0.90489713,
                                                                       0.90489713, 0.60005558, 0.60005558, 0.60005558, 0.91381375, 0.91381375,0.91381375,0.66138054, 0.66138054, 0.66138054, 0.92253504, 0.92253504,
                                                                       0.92253504,0.71429879, 0.71429879, 0.71429879, 0.93027876, 0.93027876,0.93027876,0.75947891, 0.75947891, 0.75947891, 0.9371767 , 0.9371767 ,
                                                                       0.9371767 , 0.87023975, 0.87023975, 0.87023975, 0.94014274, 0.94014274,0.94014274,0.89680574, 0.89680574, 0.89680574, 0.94648926, 0.94648926,
                                                                       0.94648926,0.91657261, 0.91657261, 0.91657261, 0.95204779, 0.95204779,0.95204779,0.93146896, 0.93146896, 0.93146896, 0.95694206, 0.95694206,
                                                                       0.95694206, 0.95177305, 0.95177305, 0.95177305, 0.93773086, 0.93773086,0.93773086,0.95874689, 0.95874689, 0.95874689, 0.94579176, 0.94579176,
                                                                       0.94579176,0.96416067, 0.96416067, 0.96416067, 0.95267886, 0.95267886,0.95267886,0.96851506, 0.96851506, 0.96851506, 0.95857985, 0.95857985,
                                                                       0.95857985, 0.97269956, 0.97269956, 0.97269956, 0.76075293, 0.76075293,0.76075293,0.97557464, 0.97557464, 0.97557464, 0.78024637, 0.78024637,
                                                                       0.78024637,0.97806922, 0.97806922, 0.97806922, 0.79833344, 0.79833344,0.79833344,0.98026195, 0.98026195, 0.98026195, 0.81508646, 0.81508646,0.81508646});
    
    auto expHFWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsFW},  {0.97269956, 0.97269956, 0.97269956, 0.97557464, 0.97557464, 0.97557464, 0.97806922, 0.97806922, 0.97806922, 0.98026195, 0.98026195, 0.98026195});
    auto expHBWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsBW},  {0.86841012, 0.86841012, 0.86841012, 0.88207531, 0.88207531, 0.88207531, 0.8941667 , 0.8941667 , 0.8941667 , 0.90489713, 0.90489713, 0.90489713});

    nd4j::ops::static_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW}, {}, {});    

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFWfinal = results->at(1);
    auto hBWfinal = results->at(2);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_rnn_test1) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::_create<double>('c', {time, bS, inSize});
    auto Wx = NDArrayFactory::_create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::_create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::_create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::_create<double>('c', {bS, numUnits});
    auto maxTimeStep = NDArrayFactory::_create<double>('c', {bS}, {time-1, time-3});

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::_create<double>('c', {time, bS, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0.69882484, 0.69882484, 0.69882484, 0.69882484,0.9312333 , 0.9312333 , 0.9312333 , 0.9312333 ,
                                                          0.93751527, 0.93751527, 0.93751527, 0.93751527,0.97136768, 0.97136768, 0.97136768, 0.97136768,0.        , 0.        , 0.        , 0.        ,
                                                          0.97732812, 0.97732812, 0.97732812, 0.97732812,0.    , 0.  , 0.  , 0. ,0.   , 0.  , 0.   , 0.  ,0.      , 0.        , 0.        , 0.        });
    
    auto expHFinal = NDArrayFactory::_create<double>('c', {bS, numUnits},       {0.97732812, 0.97732812, 0.97732812, 0.97732812, 0.93751527, 0.93751527, 0.93751527, 0.93751527});

    nd4j::ops::dynamic_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0, &maxTimeStep}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_rnn_test2) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::_create<double>('c', {bS, time, inSize});
    auto Wx = NDArrayFactory::_create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::_create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::_create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::_create<double>('c', {bS, numUnits});
    auto maxTimeStep = NDArrayFactory::_create<double>('c', {bS}, {time-1, time});

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::_create<double>('c', {bS, time, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0.92755601, 0.92755601, 0.92755601, 0.92755601,0.96778334, 0.96778334, 0.96778334, 
                                                          0.96778334,0.97309129, 0.97309129, 0.97309129, 0.97309129,0.        , 0.        , 0.        , 0.        ,
                                                          0.75001965, 0.75001965, 0.75001965, 0.75001965,0.95449491, 0.95449491, 0.95449491, 0.95449491,0.97732828, 0.97732828, 0.97732828, 
                                                          0.97732828,0.98000655, 0.98000655, 0.98000655, 0.98000655,0.98120782, 0.98120782, 0.98120782, 0.98120782});
    
    auto expHFinal = NDArrayFactory::_create<double>('c', {bS, numUnits},       {0.97309129, 0.97309129, 0.97309129, 0.97309129, 0.98120782, 0.98120782, 0.98120782, 0.98120782});

    nd4j::ops::dynamic_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_rnn_test3) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::_create<double>('c', {bS, time, inSize});
    auto Wx = NDArrayFactory::_create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::_create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::_create<double>('c', {2*numUnits});
    auto h0 = NDArrayFactory::_create<double>('c', {bS, numUnits});    

    x.linspace(0.01, 0.01);
    h0 = 0.2;
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::_create<double>('c', {bS, time, numUnits}, {0.68474828, 0.68474828, 0.68474828, 0.68474828,0.92755601, 0.92755601, 0.92755601, 0.92755601,0.96778334, 0.96778334, 0.96778334, 0.96778334,0.97309129, 
                                                          0.97309129, 0.97309129, 0.97309129,0.97491207, 0.97491207, 0.97491207, 0.97491207,0.75001965, 0.75001965, 0.75001965, 0.75001965,0.95449491, 0.95449491, 
                                                          0.95449491, 0.95449491,0.97732828, 0.97732828, 0.97732828, 0.97732828,0.98000655, 0.98000655, 0.98000655, 0.98000655,0.98120782, 0.98120782, 0.98120782, 0.98120782});

    auto expHFinal = NDArrayFactory::_create<double>('c', {bS, numUnits},       {0.97491207, 0.97491207, 0.97491207, 0.97491207, 0.98120782, 0.98120782, 0.98120782, 0.98120782});

    nd4j::ops::dynamic_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &h0}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_rnn_test4) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::_create<double>('c', {bS, time, inSize});
    auto Wx = NDArrayFactory::_create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::_create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::_create<double>('c', {2*numUnits});    
    auto maxTimeStep = NDArrayFactory::_create<double>('c', {bS}, {time-1, time-4});

    x.linspace(0.01, 0.01);    
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::_create<double>('c', {bS, time, numUnits}, {0.47615493, 0.47615493, 0.47615493, 0.47615493,0.86347567, 0.86347567, 0.86347567, 0.86347567,0.96059545, 0.96059545, 
                                                          0.96059545, 0.96059545,0.9724738 , 0.9724738 , 0.9724738 , 0.9724738 ,0.        , 0.        , 0.        , 0.        ,
                                                          0.57368608, 0.57368608, 0.57368608, 0.57368608,0. , 0. , 0  , 0. ,0., 0. , 0, 0.,0., 0., 0. , 0. ,0. , 0. , 0., 0. });
    
    auto expHFinal = NDArrayFactory::_create<double>('c', {bS, numUnits},       {0.9724738 , 0.9724738 , 0.9724738 , 0.9724738 ,0.57368608, 0.57368608, 0.57368608, 0.57368608});

    nd4j::ops::dynamic_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_rnn_test5) {

    const int bS       = 2;
    const int inSize   = 3;
    const int numUnits = 4;
    const int time     = 5;

    auto x  = NDArrayFactory::_create<double>('c', {bS, time, inSize});
    auto Wx = NDArrayFactory::_create<double>('c', {inSize, numUnits});
    auto Wh = NDArrayFactory::_create<double>('c', {numUnits, numUnits});
    auto b  = NDArrayFactory::_create<double>('c', {2*numUnits});    

    x.linspace(0.01, 0.01);    
    Wx = 0.3;
    Wh = 0.4;
    b  = 0.25;

    auto expH      = NDArrayFactory::_create<double>('c', {bS, time, numUnits}, {0.47615493, 0.47615493, 0.47615493, 0.47615493,0.86347567, 0.86347567, 0.86347567, 0.86347567,0.96059545, 0.96059545, 0.96059545, 0.96059545,
                                                          0.9724738 , 0.9724738 , 0.9724738 , 0.9724738 ,0.97486307, 0.97486307, 0.97486307, 0.97486307,0.57368608, 0.57368608, 0.57368608, 0.57368608,
                                                          0.92135149, 0.92135149, 0.92135149, 0.92135149,0.97482354, 0.97482354, 0.97482354, 0.97482354,0.97984727, 0.97984727, 0.97984727, 0.97984727,
                                                          0.98119833, 0.98119833, 0.98119833, 0.98119833});
    
    auto expHFinal = NDArrayFactory::_create<double>('c', {bS, numUnits},       {0.97486307, 0.97486307, 0.97486307, 0.97486307,0.98119833, 0.98119833, 0.98119833, 0.98119833});

    nd4j::ops::dynamic_rnn op;
    auto results = op.execute({&x, &Wx, &Wh, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto h = results->at(0);
    auto hFinal = results->at(1);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));
    ASSERT_TRUE(expHFinal.isSameShape(hFinal));
    ASSERT_TRUE(expHFinal.equalsTo(hFinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_bidir_rnn_test1) {
    
    const int bS         = 4;
    const int inSize     = 4;    
    const int numUnitsFW = 3;    
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::_create<double>('c', {time, bS, inSize});
    auto WxFW = NDArrayFactory::_create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::_create<double>('c', {numUnitsFW, numUnitsFW});    
    auto bFW  = NDArrayFactory::_create<double>('c', {2*numUnitsFW});

    auto h0FW = NDArrayFactory::_create<double>('c', {bS, numUnitsFW});
    auto h0BW = NDArrayFactory::_create<double>('c', {bS, numUnitsBW});
    auto maxTimeStep = NDArrayFactory::_create<double>('c', {bS}, {time-1, time-3, time-4, 0});

    x.linspace(0.01, 0.01);
    h0FW = 0.2;    
    h0BW = 0.25;
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expHFW  = NDArrayFactory::_create<double>('c', {time, bS, numUnitsFW}, {0.43819931, 0.43819931, 0.43819931,0.47615493, 0.47615493, 0.47615493,0.51241561, 0.51241561, 0.51241561,0.        , 0.        , 0.        ,
                                                          0.73880324, 0.73880324, 0.73880324,0.77843476, 0.77843476, 0.77843476,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.9052501 , 0.9052501 , 0.9052501 ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.9555734 , 0.9555734 , 0.9555734 ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });
    
    auto expHBW  = NDArrayFactory::_create<double>('c', {time, bS, numUnitsBW}, {0.86708881, 0.86708881, 0.86708881,0.78347842, 0.78347842, 0.78347842,0.55529176, 0.55529176, 0.55529176,0.        , 0.        , 0.        ,
                                                          0.90935605, 0.90935605, 0.90935605,0.64692945, 0.64692945, 0.64692945,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.9181592 , 0.9181592 , 0.9181592 ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.8026439 , 0.8026439 , 0.8026439 ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });
    
    auto expHFWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsFW},  {0.9555734 , 0.9555734 , 0.9555734 , 0.77843476, 0.77843476, 0.77843476, 0.51241561, 0.51241561, 0.51241561, 0.2       , 0.2       , 0.2});
    auto expHBWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsBW},  {0.86708881, 0.86708881, 0.86708881, 0.78347842, 0.78347842, 0.78347842, 0.55529176, 0.55529176, 0.55529176, 0.25      , 0.25      , 0.25});

    nd4j::ops::dynamic_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &h0FW, &h0BW, &maxTimeStep}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto hFW = results->at(0);
    auto hBW = results->at(1);
    auto hFWfinal = results->at(2);
    auto hBWfinal = results->at(3);    

    ASSERT_TRUE(expHFW.isSameShape(hFW));
    ASSERT_TRUE(expHFW.equalsTo(hFW));
    ASSERT_TRUE(expHBW.isSameShape(hBW));
    ASSERT_TRUE(expHBW.equalsTo(hBW));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_bidir_rnn_test2) {
    
    const int bS         = 4;
    const int inSize     = 4;    
    const int numUnitsFW = 3;    
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::_create<double>('c', {bS, time, inSize});
    auto WxFW = NDArrayFactory::_create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::_create<double>('c', {numUnitsFW, numUnitsFW});    
    auto bFW  = NDArrayFactory::_create<double>('c', {2*numUnitsFW});

    auto h0FW = NDArrayFactory::_create<double>('c', {bS, numUnitsFW});
    auto h0BW = NDArrayFactory::_create<double>('c', {bS, numUnitsBW});
    auto maxTimeStep = NDArrayFactory::_create<double>('c', {bS}, {time-1, time-3, time-4, 0});

    x.linspace(0.01, 0.01);
    h0FW = 0.2;    
    h0BW = 0.25;
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expHFW  = NDArrayFactory::_create<double>('c', {bS, time, numUnitsFW}, {0.43819931, 0.43819931, 0.43819931,0.66617761, 0.66617761, 0.66617761,0.80944357, 0.80944357, 0.80944357,0.87294706, 0.87294706, 0.87294706,0.        , 0.        , 0.        ,
                                                          0.61067683, 0.61067683, 0.61067683,0.84851124, 0.84851124, 0.84851124,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.73978305, 0.73978305, 0.73978305,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });
    
    auto expHBW  = NDArrayFactory::_create<double>('c', {bS, time, numUnitsBW}, {0.84345207, 0.84345207, 0.84345207,0.83584708, 0.83584708, 0.83584708,0.77435951, 0.77435951, 0.77435951,0.58760492, 0.58760492, 0.58760492,0.        , 0.        , 0.        ,
                                                          0.85615841, 0.85615841, 0.85615841,0.67397984, 0.67397984, 0.67397984,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.76576202, 0.76576202, 0.76576202,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });
    
    auto expHFWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsFW},  {0.87294706, 0.87294706, 0.87294706,0.84851124, 0.84851124, 0.84851124,0.73978305, 0.73978305, 0.73978305,0.2       , 0.2       , 0.2});
    auto expHBWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsBW},  {0.84345207, 0.84345207, 0.84345207, 0.85615841, 0.85615841, 0.85615841, 0.76576202, 0.76576202, 0.76576202, 0.25      , 0.25      , 0.25});

    nd4j::ops::dynamic_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &h0FW, &h0BW, &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto hFW = results->at(0);
    auto hBW = results->at(1);
    auto hFWfinal = results->at(2);
    auto hBWfinal = results->at(3);    

    ASSERT_TRUE(expHFW.isSameShape(hFW));
    ASSERT_TRUE(expHFW.equalsTo(hFW));
    ASSERT_TRUE(expHBW.isSameShape(hBW));
    ASSERT_TRUE(expHBW.equalsTo(hBW));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_bidir_rnn_test3) {
    
    const int bS         = 4;
    const int inSize     = 4;    
    const int numUnitsFW = 3;    
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::_create<double>('c', {bS, time, inSize});
    auto WxFW = NDArrayFactory::_create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::_create<double>('c', {numUnitsFW, numUnitsFW});    
    auto bFW  = NDArrayFactory::_create<double>('c', {2*numUnitsFW});

    auto maxTimeStep = NDArrayFactory::_create<double>('c', {bS}, {time-1, time-3, time-4, 0});

    x.linspace(0.01, 0.01);
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expHFW  = NDArrayFactory::_create<double>('c', {bS, time, numUnitsFW}, {0.22602835, 0.22602835, 0.22602835,0.49994591, 0.49994591, 0.49994591,0.72869307, 0.72869307, 0.72869307,0.84784327, 0.84784327, 0.84784327,0.        , 0.        , 0.        ,
                                                          0.43819931, 0.43819931, 0.43819931,0.7793996 , 0.7793996 , 0.7793996 ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.61067683, 0.61067683, 0.61067683,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });
    
    auto expHBW  = NDArrayFactory::_create<double>('c', {bS, time, numUnitsBW}, {0.82273707, 0.82273707, 0.82273707,0.77935851, 0.77935851, 0.77935851,0.6381121 , 0.6381121 , 0.6381121 ,0.35748551, 0.35748551, 0.35748551,0.        , 0.        , 0.        ,
                                                          0.77843476, 0.77843476, 0.77843476,0.47615493, 0.47615493, 0.47615493,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.61067683, 0.61067683, 0.61067683,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,
                                                          0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        ,0.        , 0.        , 0.        });
    
    auto expHFWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsFW},  {0.84784327, 0.84784327, 0.84784327, 0.7793996 , 0.7793996 , 0.7793996 , 0.61067683, 0.61067683, 0.61067683, 0.        , 0.        , 0.});
    auto expHBWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsBW},  {0.82273707, 0.82273707, 0.82273707, 0.77843476, 0.77843476, 0.77843476, 0.61067683, 0.61067683, 0.61067683, 0.        , 0.        , 0.});

    nd4j::ops::dynamic_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &maxTimeStep}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto hFW = results->at(0);
    auto hBW = results->at(1);
    auto hFWfinal = results->at(2);
    auto hBWfinal = results->at(3);    

    ASSERT_TRUE(expHFW.isSameShape(hFW));
    ASSERT_TRUE(expHFW.equalsTo(hFW));
    ASSERT_TRUE(expHBW.isSameShape(hBW));
    ASSERT_TRUE(expHBW.equalsTo(hBW));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, dynamic_bidir_rnn_test4) {
    
    const int bS         = 4;
    const int inSize     = 4;    
    const int numUnitsFW = 3;    
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::_create<double>('c', {bS, time, inSize});
    auto WxFW = NDArrayFactory::_create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::_create<double>('c', {numUnitsFW, numUnitsFW});    
    auto bFW  = NDArrayFactory::_create<double>('c', {2*numUnitsFW});

    auto h0FW = NDArrayFactory::_create<double>('c', {bS, numUnitsFW});
    auto h0BW = NDArrayFactory::_create<double>('c', {bS, numUnitsBW});

    x.linspace(0.01, 0.01);
    h0FW = 0.2;    
    h0BW = 0.25;
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expHFW  = NDArrayFactory::_create<double>('c', {bS, time, numUnitsFW}, {0.43819931, 0.43819931, 0.43819931,0.66617761, 0.66617761, 0.66617761,0.80944357, 0.80944357, 0.80944357,0.87294706, 0.87294706, 0.87294706,0.89948899, 0.89948899, 0.89948899,
                                                          0.61067683, 0.61067683, 0.61067683,0.84851124, 0.84851124, 0.84851124,0.91925737, 0.91925737, 0.91925737,0.93751395, 0.93751395, 0.93751395,0.94544483, 0.94544483, 0.94544483,
                                                          0.73978305, 0.73978305, 0.73978305,0.92827068, 0.92827068, 0.92827068,0.95791111, 0.95791111, 0.95791111,0.96427356, 0.96427356, 0.96427356,0.96797541, 0.96797541, 0.96797541,
                                                          0.83057887, 0.83057887, 0.83057887,0.96365083, 0.96365083, 0.96365083,0.97585698, 0.97585698, 0.97585698,0.97866981, 0.97866981, 0.97866981,0.9807326 , 0.9807326 , 0.9807326 });
    
    auto expHBW  = NDArrayFactory::_create<double>('c', {bS, time, numUnitsBW}, {0.85301722, 0.85301722, 0.85301722,0.86427295, 0.86427295, 0.86427295,0.8599919 , 0.8599919 , 0.8599919 ,0.80609463, 0.80609463, 0.80609463,0.61814662, 0.61814662, 0.61814662,
                                                          0.91888753, 0.91888753, 0.91888753,0.92652672, 0.92652672, 0.92652672,0.92939674, 0.92939674, 0.92939674,0.90661931, 0.90661931, 0.90661931,0.74516764, 0.74516764, 0.74516764,
                                                          0.95254269, 0.95254269, 0.95254269,0.95710717, 0.95710717, 0.95710717,0.96021584, 0.96021584, 0.96021584,0.95222547, 0.95222547, 0.95222547,0.83426363, 0.83426363, 0.83426363,
                                                          0.97154357, 0.97154357, 0.97154357,0.97424915, 0.97424915, 0.97424915,0.97644817, 0.97644817, 0.97644817,0.97410547, 0.97410547, 0.97410547,0.89409962, 0.89409962, 0.89409962});
    
    auto expHFWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsFW},  {0.89948899, 0.89948899, 0.89948899, 0.94544483, 0.94544483, 0.94544483, 0.96797541, 0.96797541, 0.96797541, 0.9807326 , 0.9807326 , 0.9807326 });
    auto expHBWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsBW},  {0.85301722, 0.85301722, 0.85301722, 0.91888753, 0.91888753, 0.91888753, 0.95254269, 0.95254269, 0.95254269, 0.97154357, 0.97154357, 0.97154357});

    nd4j::ops::dynamic_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW,  &h0FW, &h0BW}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto hFW = results->at(0);
    auto hBW = results->at(1);
    auto hFWfinal = results->at(2);
    auto hBWfinal = results->at(3);    

    ASSERT_TRUE(expHFW.isSameShape(hFW));
    ASSERT_TRUE(expHFW.equalsTo(hFW));
    ASSERT_TRUE(expHBW.isSameShape(hBW));
    ASSERT_TRUE(expHBW.equalsTo(hBW));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}

TEST_F(DeclarableOpsTests6, dynamic_bidir_rnn_test5) {
    
    const int bS         = 4;
    const int inSize     = 4;    
    const int numUnitsFW = 3;    
    const int numUnitsBW = 3;
    const int time       = 5;

    auto x  = NDArrayFactory::_create<double>('c', {bS, time, inSize});
    auto WxFW = NDArrayFactory::_create<double>('c', {inSize, numUnitsFW});
    auto WhFW = NDArrayFactory::_create<double>('c', {numUnitsFW, numUnitsFW});    
    auto bFW  = NDArrayFactory::_create<double>('c', {2*numUnitsFW});

    x.linspace(0.01, 0.01);
    WxFW = 0.3;
    WhFW = 0.4;
    bFW  = 0.1;

    auto expHFW  = NDArrayFactory::_create<double>('c', {bS, time, numUnitsFW}, {0.22602835, 0.22602835, 0.22602835,0.49994591, 0.49994591, 0.49994591,0.72869307, 0.72869307, 0.72869307,0.84784327, 0.84784327, 0.84784327,0.89357928, 0.89357928, 0.89357928,
                                                          0.43819931, 0.43819931, 0.43819931,0.7793996 , 0.7793996 , 0.7793996 ,0.9053792 , 0.9053792 , 0.9053792 ,0.93546593, 0.93546593, 0.93546593,0.94518339, 0.94518339, 0.94518339,
                                                          0.61067683, 0.61067683, 0.61067683,0.90347408, 0.90347408, 0.90347408,0.95538786, 0.95538786, 0.95538786,0.96406045, 0.96406045, 0.96406045,0.96795929, 0.96795929, 0.96795929,
                                                          0.73978305, 0.73978305, 0.73978305,0.95499984, 0.95499984, 0.95499984,0.97535671, 0.97535671, 0.97535671,0.97864446, 0.97864446, 0.97864446,0.98073144, 0.98073144, 0.98073144});
    
    auto expHBW  = NDArrayFactory::_create<double>('c', {bS, time, numUnitsBW}, {0.84882345, 0.84882345, 0.84882345,0.85160683, 0.85160683, 0.85160683,0.81997657, 0.81997657, 0.81997657,0.69228829, 0.69228829, 0.69228829,0.39861399, 0.39861399, 0.39861399,
                                                          0.91865453, 0.91865453, 0.91865453,0.92528094, 0.92528094, 0.92528094,0.92212167, 0.92212167, 0.92212167,0.86418213, 0.86418213, 0.86418213,0.57969286, 0.57969286, 0.57969286,
                                                          0.95252666, 0.95252666, 0.95252666,0.95696305, 0.95696305, 0.95696305,0.95878749, 0.95878749, 0.95878749,0.93722463, 0.93722463, 0.93722463,0.71727031, 0.71727031, 0.71727031,
                                                          0.97154234, 0.97154234, 0.97154234,0.97423089, 0.97423089, 0.97423089,0.976149  , 0.976149  , 0.976149  ,0.96878298, 0.96878298, 0.96878298,0.81508646, 0.81508646, 0.81508646});
    
    auto expHFWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsFW},  {0.89357928, 0.89357928, 0.89357928, 0.94518339, 0.94518339, 0.94518339, 0.96795929, 0.96795929, 0.96795929, 0.98073144, 0.98073144, 0.98073144});
    auto expHBWfinal = NDArrayFactory::_create<double>('c', {bS, numUnitsBW},  {0.84882345, 0.84882345, 0.84882345, 0.91865453, 0.91865453, 0.91865453, 0.95252666, 0.95252666, 0.95252666, 0.97154234, 0.97154234, 0.97154234});

    nd4j::ops::dynamic_bidirectional_rnn op;
    auto results = op.execute({&x, &WxFW,&WhFW,&bFW,  &WxFW,&WhFW,&bFW}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto hFW = results->at(0);
    auto hBW = results->at(1);
    auto hFWfinal = results->at(2);
    auto hBWfinal = results->at(3);    

    ASSERT_TRUE(expHFW.isSameShape(hFW));
    ASSERT_TRUE(expHFW.equalsTo(hFW));
    ASSERT_TRUE(expHBW.isSameShape(hBW));
    ASSERT_TRUE(expHBW.equalsTo(hBW));
    ASSERT_TRUE(expHFWfinal.isSameShape(hFWfinal));
    ASSERT_TRUE(expHFWfinal.equalsTo(hFWfinal));
    ASSERT_TRUE(expHBWfinal.isSameShape(hBWfinal));
    ASSERT_TRUE(expHBWfinal.equalsTo(hBWfinal));

    delete results;
}


TEST_F(DeclarableOpsTests6, Test_Diag_119_1) {
    auto x = NDArrayFactory::_create<double>('c', {3}, {0.15f, 0.25f, 0.35f});
    auto e = NDArrayFactory::_create<double>('c', {3, 3}, {0.15f, 0.0f, 0.0f,   0.0f, 0.25f, 0.0f,   0.0f, 0.0f, 0.35f});

    nd4j::ops::diag op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Diag_119_2) {
    auto x = NDArrayFactory::_create<double>('c', {1}, {0.15f});
    auto e = NDArrayFactory::_create<double>('c', {1, 1}, {0.15f});

    nd4j::ops::diag op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests6, Test_Diag_119_3) {
    auto x = NDArrayFactory::_create<double>(0.15f);
    auto e = NDArrayFactory::_create<double>('c', {1, 1}, {0.15f});

    nd4j::ops::diag op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, deconv2d_test1) {

    int bS=2, iH=4,iW=4,  iC=5,oC=10,  kH=2,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=3,oW=3;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW    

    auto input    = NDArrayFactory::_create<double>('c', {bS, oH, oW, oC});
    auto weights  = NDArrayFactory::_create<double>('c', {kH, kW, iC, oC});    
    auto exp = NDArrayFactory::_create<double>('c', {bS, iH, iW, iC}, {  2.75,   7.75,  12.75,  17.75,  22.75, 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 27.75,  32.75,  37.75,  42.75,  47.75,
                                                  55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,105.5 , 115.5 , 125.5 , 135.5 , 145.5 ,
                                                  55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,105.5 , 115.5 , 125.5 , 135.5 , 145.5 ,
                                                  52.75,  57.75,  62.75,  67.75,  72.75,130.5 , 140.5 , 150.5 , 160.5 , 170.5 ,130.5 , 140.5 , 150.5 , 160.5 , 170.5 , 77.75,  82.75,  87.75,  92.75,  97.75,
                                                   2.75,   7.75,  12.75,  17.75,  22.75, 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 27.75,  32.75,  37.75,  42.75,  47.75,
                                                  55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,105.5 , 115.5 , 125.5 , 135.5 , 145.5 ,
                                                  55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,105.5 , 115.5 , 125.5 , 135.5 , 145.5 ,
                                                  52.75,  57.75,  62.75,  67.75,  72.75,130.5 , 140.5 , 150.5 , 160.5 , 170.5 ,130.5 , 140.5 , 150.5 , 160.5 , 170.5 , 77.75,  82.75,  87.75,  92.75,  97.75});
    input = 0.5;
    weights.linspace(0.1, 0.1);
    
    nd4j::ops::deconv2d op;
    auto results = op.execute({&input, &weights}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto output = results->at(0);            

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, deconv2d_test2) {

    int bS=2, iH=4,iW=4,  iC=5,oC=10,  kH=2,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=4,oW=4;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW    

    auto input    = NDArrayFactory::_create<double>('c', {bS, oH, oW, oC});
    auto weights  = NDArrayFactory::_create<double>('c', {kH, kW, iC, oC});
    auto exp = NDArrayFactory::_create<double>('c', {bS, iH, iW, iC}, {2.75,   7.75,  12.75,  17.75,  22.75, 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,
                                                 2.75,   7.75,  12.75,  17.75,  22.75, 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  });    
    input = 0.5;
    weights.linspace(0.1, 0.1);
    
    nd4j::ops::deconv2d op;
    auto results = op.execute({&input, &weights}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto output = results->at(0);            

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, deconv2d_tf_test1) {

    int bS=2, iH=4,iW=4,  iC=5,oC=10,  kH=2,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=3,oW=3;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW    

    auto input    = NDArrayFactory::_create<double>('c', {bS, oH, oW, oC});
    auto weights  = NDArrayFactory::_create<double>('c', {kH, kW, iC, oC});
    auto outShape = NDArrayFactory::_create<double>('c', {4}, {static_cast<double>(bS), static_cast<double>(iH), static_cast<double>(iW), static_cast<double>(iC)});
    auto exp = NDArrayFactory::_create<double>('c', {bS, iH, iW, iC}, {  2.75,   7.75,  12.75,  17.75,  22.75, 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 27.75,  32.75,  37.75,  42.75,  47.75,
                                                  55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,105.5 , 115.5 , 125.5 , 135.5 , 145.5 ,
                                                  55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,105.5 , 115.5 , 125.5 , 135.5 , 145.5 ,
                                                  52.75,  57.75,  62.75,  67.75,  72.75,130.5 , 140.5 , 150.5 , 160.5 , 170.5 ,130.5 , 140.5 , 150.5 , 160.5 , 170.5 , 77.75,  82.75,  87.75,  92.75,  97.75,
                                                   2.75,   7.75,  12.75,  17.75,  22.75, 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 27.75,  32.75,  37.75,  42.75,  47.75,
                                                  55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,105.5 , 115.5 , 125.5 , 135.5 , 145.5 ,
                                                  55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,105.5 , 115.5 , 125.5 , 135.5 , 145.5 ,
                                                  52.75,  57.75,  62.75,  67.75,  72.75,130.5 , 140.5 , 150.5 , 160.5 , 170.5 ,130.5 , 140.5 , 150.5 , 160.5 , 170.5 , 77.75,  82.75,  87.75,  92.75,  97.75});
    input = 0.5;
    weights.linspace(0.1, 0.1);
    
    nd4j::ops::deconv2d_tf op;
    auto results = op.execute({&outShape, &weights, &input}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto output = results->at(0);            

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests6, deconv2d_tf_test2) {

    int bS=2, iH=4,iW=4,  iC=5,oC=10,  kH=2,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=4,oW=4;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW    

    auto input    = NDArrayFactory::_create<double>('c', {bS, oH, oW, oC});
    auto weights  = NDArrayFactory::_create<double>('c', {kH, kW, iC, oC});
    auto outShape = NDArrayFactory::_create<double>('c', {4}, {static_cast<double>(bS), static_cast<double>(iH), static_cast<double>(iW), static_cast<double>(iC)});
    auto exp = NDArrayFactory::_create<double>('c', {bS, iH, iW, iC}, {2.75,   7.75,  12.75,  17.75,  22.75, 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,
                                                 2.75,   7.75,  12.75,  17.75,  22.75, 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 , 30.5 ,  40.5 ,  50.5 ,  60.5 ,  70.5 ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,
                                                55.5 ,  65.5 ,  75.5 ,  85.5 ,  95.5 ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  ,161.  , 181.  , 201.  , 221.  , 241.  });    
    input = 0.5;
    weights.linspace(0.1, 0.1);
    
    nd4j::ops::deconv2d_tf op;
    auto results = op.execute({&outShape, &weights, &input}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto output = results->at(0);            

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));
    
    delete results;
}

  

  
