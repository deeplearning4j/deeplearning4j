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
// @author raver119@gmail.com
//


#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests5 : public testing::Test {
public:

    DeclarableOpsTests5() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests5, Test_PermuteEquality_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    auto exp = NDArrayFactory::create<double>('c', {3, 5, 4}, {1.0, 6.0, 11.0, 16.0, 2.0, 7.0, 12.0, 17.0, 3.0, 8.0, 13.0, 18.0, 4.0, 9.0, 14.0, 19.0, 5.0, 10.0, 15.0, 20.0, 21.0, 26.0, 31.0, 36.0, 22.0, 27.0, 32.0, 37.0, 23.0, 28.0, 33.0, 38.0, 24.0, 29.0, 34.0, 39.0, 25.0, 30.0, 35.0, 40.0, 41.0, 46.0, 51.0, 56.0, 42.0, 47.0, 52.0, 57.0, 43.0, 48.0, 53.0, 58.0, 44.0, 49.0, 54.0, 59.0, 45.0, 50.0, 55.0, 60.0});
    x.linspace(1);
    x.reshapei('c', {3, 4, 5});

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {0, 2, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_0) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{0, 1, 2} shape");
//    x.printBuffer("{0, 1, 2} data");

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {0, 1, 2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_PermuteEquality_2) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {4, 3, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 21.0, 22.0, 23.0, 24.0, 25.0, 41.0, 42.0, 43.0, 44.0, 45.0, 6.0, 7.0, 8.0, 9.0, 10.0, 26.0, 27.0, 28.0, 29.0, 30.0, 46.0, 47.0, 48.0, 49.0, 50.0, 11.0, 12.0, 13.0, 14.0, 15.0, 31.0, 32.0, 33.0, 34.0, 35.0, 51.0, 52.0, 53.0, 54.0, 55.0, 16.0, 17.0, 18.0, 19.0, 20.0, 36.0, 37.0, 38.0, 39.0, 40.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{1, 0, 2} shape");
//    x.printBuffer("{1, 0, 2} data");

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {1, 0, 2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_3) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {4, 5, 3}, {1.0, 21.0, 41.0, 2.0, 22.0, 42.0, 3.0, 23.0, 43.0, 4.0, 24.0, 44.0, 5.0, 25.0, 45.0, 6.0, 26.0, 46.0, 7.0, 27.0, 47.0, 8.0, 28.0, 48.0, 9.0, 29.0, 49.0, 10.0, 30.0, 50.0, 11.0, 31.0, 51.0, 12.0, 32.0, 52.0, 13.0, 33.0, 53.0, 14.0, 34.0, 54.0, 15.0, 35.0, 55.0, 16.0, 36.0, 56.0, 17.0, 37.0, 57.0, 18.0, 38.0, 58.0, 19.0, 39.0, 59.0, 20.0, 40.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{1, 2, 0} shape");
//    x.printBuffer("{1, 2, 0} data");

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {1, 2, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_4) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {5, 3, 4}, {1.0, 6.0, 11.0, 16.0, 21.0, 26.0, 31.0, 36.0, 41.0, 46.0, 51.0, 56.0, 2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 3.0, 8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0, 58.0, 4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0, 39.0, 44.0, 49.0, 54.0, 59.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{2, 0, 1} shape");
//    x.printBuffer("{2, 0, 1} data");

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {2, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_5) {
    auto x = NDArrayFactory::create<double>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {5, 4, 3}, {1.0, 21.0, 41.0, 6.0, 26.0, 46.0, 11.0, 31.0, 51.0, 16.0, 36.0, 56.0, 2.0, 22.0, 42.0, 7.0, 27.0, 47.0, 12.0, 32.0, 52.0, 17.0, 37.0, 57.0, 3.0, 23.0, 43.0, 8.0, 28.0, 48.0, 13.0, 33.0, 53.0, 18.0, 38.0, 58.0, 4.0, 24.0, 44.0, 9.0, 29.0, 49.0, 14.0, 34.0, 54.0, 19.0, 39.0, 59.0, 5.0, 25.0, 45.0, 10.0, 30.0, 50.0, 15.0, 35.0, 55.0, 20.0, 40.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{2, 1, 0} shape");
//    x.printBuffer("{2, 1, 0} data");

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {2, 1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_TTS_bp_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 1, 3});
    auto eps = NDArrayFactory::create<double>('c', {2, 4, 3});

    auto exp = NDArrayFactory::create<double>('c', {2, 1, 3}, {22.f, 26.f, 30.f, 70.f, 74.f, 78.f});

    eps.linspace(1.f);

    nd4j::ops::tile_to_shape_bp op;
    auto result = op.execute({&x, &eps}, {}, {2, 4, 3});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printShapeInfo("RES shape");
    // x.printShapeInfo("EXP shape");
    // z->printIndexedBuffer("RES output");
    ASSERT_TRUE(x.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_Rdiv_bp_1) {
    auto x = NDArrayFactory::create<double>('c', {3, 1}, {1, 2, 3});
    auto y = NDArrayFactory::create<double>('c', {1, 4}, {1, 2, 3, 4});
    auto eps = NDArrayFactory::create<double>('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});


    nd4j::ops::reversedivide op_ff;
    auto result_ff = op_ff.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result_ff->status());

    auto z_ff = result_ff->at(0);
    ASSERT_TRUE(eps.isSameShape(z_ff));

    nd4j::ops::reversedivide_bp op_bp;
    auto result_bp = op_bp.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(Status::OK(), result_bp->status());

    auto z_bp = result_bp->at(0);
    ASSERT_TRUE(x.isSameShape(z_bp));

    delete result_ff;
    delete result_bp;
}


TEST_F(DeclarableOpsTests5, Test_Boolean_diff_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 1}, {1.0f});
    auto y = NDArrayFactory::create<double>(2.0f);

    nd4j::ops::less op;
    auto result = op.execute({&x, &y}, {}, {}, {}, false, nd4j::DataType::BOOL);
    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_EQ(result->at(0)->t<bool>(0), true);
    delete result;
}

TEST_F(DeclarableOpsTests5, Test_SetSeed_1) {
    auto x = NDArrayFactory::create<int>('c', {1, 1}, {120});
    auto y = NDArrayFactory::create<int>(5);

    nd4j::ops::set_seed op;
    auto result = op.execute({&x, &y}, {}, {120, 5}, {}, false, nd4j::DataType::INT32);

    ASSERT_EQ(Status::OK(), result->status());
//    result->at(0)->printIndexedBuffer("RES SEED");
    nd4j::ops::get_seed getOp;
    auto getRes = getOp.execute({}, {}, {});
    ASSERT_EQ(Status::OK(), getRes->status());
//    getRes->at(0)->printIndexedBuffer("Output RES GET SEED");
//    ASSERT_EQ(result->at(0)->t<bool>(0), true);
    delete result;
    delete getRes;
}
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, scatterMul_test1) {
    auto matrix = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    NDArray idc('c', {1}, {0LL}, nd4j::DataType::INT64);
    auto updates = NDArrayFactory::create<float>('c', {1, 2}, {10, 1});
    auto exp = NDArrayFactory::create<float>('c', {2, 2}, {10, 2, 3, 4});

    nd4j::ops::scatter_mul op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, scatterDiv_test1) {
    auto matrix = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    NDArray idc('c', {1}, {0LL}, nd4j::DataType::INT64);
    auto updates = NDArrayFactory::create<float>('c', {1, 2}, {10, 1});
    auto exp = NDArrayFactory::create<float>('c', {2, 2}, {0.10, 2, 3, 4});

    nd4j::ops::scatter_div op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Scatter Div");
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, scatterSub_test1) {
    auto matrix = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    NDArray idc('c', {1}, {0LL}, nd4j::DataType::INT64);
    auto updates = NDArrayFactory::create<float>('c', {1, 2}, {10, 1});
    auto exp = NDArrayFactory::create<float>('c', {2, 2}, {-9, 1, 3, 4});

    nd4j::ops::scatter_sub op;
    auto result = op.execute({&matrix, &idc, &updates}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Scatter Sub");
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, hardsigmoid_test1) {
    auto matrix = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<float>('c', {2, 2}, {0.7, 0.9, 1, 1});

    nd4j::ops::hardsigmoid op;
    auto result = op.execute({&matrix}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, hardsigmoid_test2) {
    auto matrix = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    auto eps = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<float>('c', {2, 2}, {0.2, 0.4, 0, 0});

    nd4j::ops::hardsigmoid_bp op;
    auto result = op.execute({&matrix, &eps}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, hardtanh_test1) {
    auto matrix = NDArrayFactory::create<float>('c', {3, 3}, {-4, -3, -2, -1, 0, 1, 2, 3, 4});
    auto exp = NDArrayFactory::create<float>('c', {3, 3}, {-1, -1, -1, -1, 0, 1, 1, 1, 1});

    nd4j::ops::hardtanh op;
    auto result = op.execute({&matrix}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Hardtanh 2x2");
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, hardtanh_test2) {
    auto matrix = NDArrayFactory::create<float>('c', {3, 3}, {-4, -3, -2, -1, 0, 1, 2, 3, 4});
    auto eps = NDArrayFactory::create<float>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto exp = NDArrayFactory::create<float>('c', {3, 3}, {0, 0, 0, 4, 5, 6, 0, 0, 0});

    nd4j::ops::hardtanh_bp op;
    auto result = op.execute({&matrix, &eps}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Hardtanh_bp 2x2");
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, histogram_test1) {
    auto matrix = NDArrayFactory::create<double>('c', {3, 3}, {-4, -3, -2, -1, 0, 1, 2, 3, 4});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {3}, {3, 3, 3});

    nd4j::ops::histogram op;
    auto result = op.execute({&matrix}, {}, {3}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Histogram3");
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, histogram_test2) {
    auto matrix = NDArrayFactory::create<double>('c', {3}, {1, 2, 1});
    auto exp = NDArrayFactory::create<Nd4jLong>('c', {4}, {2, 0, 0, 1});

    nd4j::ops::histogram op;
    auto result = op.execute({&matrix}, {}, {4}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Identity_test1) {
    auto matrix = NDArrayFactory::create<float>('c', {3, 3}, {-4, -3, -2, -1, 0, 1, 2, 3, 4});
//    auto exp = NDArrayFactory::create<Nd4jLong>('c', {3, 3}, {3, 3, 3});

    nd4j::ops::identity op;
    auto result = op.execute({&matrix}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    ASSERT_TRUE(matrix.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Identity_test2) {
    auto matrix = NDArrayFactory::create<float>('c', {3, 3}, {-4, -3, -2, -1, 0, 1, 2, 3, 4});
    auto eps = NDArrayFactory::create<float>('c', {3, 3}, {1,2,3,4,5,6,7,8,9});
//    auto exp = NDArrayFactory::create<float>('c', {3,3});
    nd4j::ops::identity_bp op;
    auto result = op.execute({&matrix, &eps}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    ASSERT_TRUE(z->equalsTo(eps));

    delete result;
}
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Log1p_test1) {
    auto matrix = NDArrayFactory::create<float>('c', {3, 3}, {4, 3, 2, 1, 0, 1, 2, 3, 4});
    auto y = NDArrayFactory::create<float>('c', {3,3}, {5,4,3,2,1,2,3,4,5});
    //  auto eps = NDArrayFactory::create<float>('c', {3, 3}, {1,2,3,4,5,6,7,8,9});
//    auto exp = NDArrayFactory::create<float>('c', {3,3});
    nd4j::ops::Log1p op;
    y.applyTransform(nd4j::transform::Log, nullptr, nullptr);
    auto result = op.execute({&matrix}, {}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    ASSERT_TRUE(z->equalsTo(y));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_1) {

    auto x = NDArrayFactory::create<double>('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {0, 0, 0, 0});

    nd4j::ops::space_to_batch op;
    auto result = op.execute({&x, &paddings}, {}, {2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_2) {
    auto x = NDArrayFactory::create<double>('c', {1, 2, 2, 1}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {4, 1, 1, 1}, {1, 2, 3, 4});
    auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {0, 0, 0, 0});

    nd4j::ops::space_to_batch op;
    auto result = op.execute({&x, &paddings}, {}, {2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto paddings = NDArrayFactory::create<int>('c', {2, 2}, {0, 0, 2, 0});
    auto exp = NDArrayFactory::create<double>('c', {8, 1, 3, 1}, {0, 1, 3, 0,  9, 11,0, 2, 4, 0, 10, 12,0, 5, 7, 0, 13, 15,0, 6, 8, 0, 14, 16});

    nd4j::ops::space_to_batch op;
    auto result = op.execute({&x, &paddings}, {}, {2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_4) {

    const int blockSize = 2;
    NDArray x('c', {3, 3*blockSize - 1 - 2, 4*blockSize - 2 - 3, 2}, {147, 148, 219, 220, 149, 150, 11,  12, 83,  84, 13,  14, 155, 156, 227, 228, 157, 158, 171, 172, 243, 244, 173, 174, 35,  36, 107, 108, 37,  38, 179, 180, 251, 252, 181, 182, 195, 196, 267, 268, 197, 198, 59,  60, 131, 132, 61,  62, 203, 204, 275, 276, 205, 206}, nd4j::DataType::FLOAT32);
    NDArray paddings = NDArrayFactory::create<int>('c', {2, 2}, {1, 2, 2, 3});

    NDArray exp('c', {3*blockSize*blockSize, 3, 4, 2}, {0,0, 0,0, 0,0, 0,0, 0,0, 11,12, 13,14, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
        0,0, 0,0, 0,0, 35,36, 37,38, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 59,60, 61,62, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
        0,0, 0,0, 0,0, 0,0, 83,84, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 107, 108, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
        0,0, 0,0, 0,0, 0,0, 0,0, 131, 132, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 147, 148, 149, 150, 0,0, 0,0, 155, 156, 157, 158,
        0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 171, 172, 173, 174, 0,0, 0,0, 179, 180, 181, 182, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 195, 196,
        197, 198, 0,0, 0,0, 203, 204, 205, 206, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 219, 220, 0,0, 0,0, 0,0, 227, 228, 0,0, 0,0, 0,0,
        0,0, 0,0, 0,0, 0,0, 243, 244, 0,0, 0,0, 0,0, 251, 252, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 267, 268, 0,0, 0,0, 0,0, 275,
        276, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0}, nd4j::DataType::FLOAT32);

    nd4j::ops::space_to_batch op;
    auto result = op.execute({&x, &paddings}, {}, {blockSize});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_BatchToSpace_1) {
    auto x = NDArrayFactory::create<double>('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto exp = NDArrayFactory::create<double>('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto crops = NDArrayFactory::create<int>('c', {2, 2}, {0, 0, 0, 0});

    nd4j::ops::batch_to_space op;
    auto result = op.execute({&x, &crops}, {}, {2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    // z->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_BatchToSpace_2) {
    auto x = NDArrayFactory::create<double>('c', {4, 1, 1, 1}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<double>('c', {1, 2, 2, 1}, {1, 2, 3, 4});
    auto crops = NDArrayFactory::create<int>('c', {2, 2}, {0, 0, 0, 0});

    nd4j::ops::batch_to_space op;
    auto result = op.execute({&x, &crops}, {}, {2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_BatchToSpace_3) {
    auto x = NDArrayFactory::create<double>('c', {8, 1, 3, 1}, {0, 1, 3, 0,  9, 11,
                                                                0, 2, 4, 0, 10, 12,
                                                                0, 5, 7, 0, 13, 15,
                                                                0, 6, 8, 0, 14, 16});
    auto exp = NDArrayFactory::create<double>('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto crops = NDArrayFactory::create<int>('c', {2, 2}, {0, 0, 2, 0});

    nd4j::ops::batch_to_space op;
    auto result = op.execute({&x, &crops}, {}, {2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_BatchToSpace_4) {

    const int blockSize = 2;
    NDArray x('c', {3*blockSize*blockSize, 3, 4, 2}, nd4j::DataType::FLOAT32);
    x.linspace(1, 1);
    NDArray crops = NDArrayFactory::create<int>('c', {2, 2}, {1, 2, 2, 3});

    NDArray exp('c', {3, 3*blockSize - 1 - 2, 4*blockSize - 2 - 3, 2}, {147, 148, 219, 220, 149, 150, 11,  12, 83,  84, 13,  14, 155, 156, 227, 228, 157, 158, 171, 172, 243, 244, 173, 174, 35,  36, 107, 108, 37,  38, 179, 180, 251, 252, 181, 182, 195, 196, 267, 268, 197, 198, 59,  60, 131, 132, 61,  62, 203, 204, 275, 276, 205, 206}, nd4j::DataType::FLOAT32);

    nd4j::ops::batch_to_space op;
    auto result = op.execute({&x, &crops}, {}, {blockSize});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test1) {

    auto expected = NDArrayFactory::create<float>('c', {3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});

    nd4j::ops::eye op;
    auto results = op.execute({}, {}, {-99, 3});
    auto output = results->at(0);
    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test2) {

    auto expected = NDArrayFactory::create<float>('c', {3, 4}, {1,  0,  0,  0, 0,  1,  0,  0, 0,  0,  1,  0});

    nd4j::ops::eye op;
    auto results = op.execute({}, {}, {-99, 3, 4});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test3) {

    auto expected = NDArrayFactory::create<int>('c', {2, 3, 4}, {1,  0,  0,  0, 0,  1,  0,  0, 0,  0,  1,  0, 1,  0,  0,  0, 0,  1,  0,  0, 0,  0,  1,  0});

    nd4j::ops::eye op;
    auto results = op.execute({}, {9 /*int*/}, {-99, 3, 4, 2});
    auto output = results->at(0);
     // output->printIndexedBuffer("Output eye");

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test4) {

    auto expected = NDArrayFactory::create<double>('c', {2, 2, 3, 4}, {1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0.});

    nd4j::ops::eye op;
    auto results = op.execute({}, {6/*double*/}, {-99, 3, 4, 2, 2});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test5) {

    nd4j::ops::eye op;
    auto result = op.execute({},{},{3, 2});

    auto z = result->at(0);
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test1) {

    auto input = NDArrayFactory::create<double>('c', {4, 3, 2});
    input.linspace(1);
    auto indices = NDArrayFactory::create<int>('c', {2,2,1}, {3,2,3,2});

    auto expected = NDArrayFactory::create<double>('c', {2,2,3,2}, {19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18});

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test2) {

    auto input = NDArrayFactory::create<double>('c', {4, 3, 2});
    input.linspace(1);
    auto indices = NDArrayFactory::create<int>('c', {2,2,2}, {3,2,1,2, 0,1,0,1});

    auto expected = NDArrayFactory::create<double>('c', {2,2,2}, {23, 24, 11, 12, 3,  4, 3,  4});

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {}, {true});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test3) {

    auto input = NDArrayFactory::create<double>('c', {4, 3, 2});
    input.linspace(1);
    auto indices = NDArrayFactory::create<int>('c', {3}, {3,2,1});
    auto expected = NDArrayFactory::create<double>(24.);

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test4) {

    auto input = NDArrayFactory::create<double>('c', {4, 3, 2});
    input.linspace(1);
    auto indices = NDArrayFactory::create<int>('c', {2,3}, {3,2,1,0,2,1});
    auto expected = NDArrayFactory::create<double>('c',{2}, {24., 6});

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test5) {

    auto input = NDArrayFactory::create<double>('c', {4}, {1,2,3,4});
    auto indices = NDArrayFactory::create<int>('c', {5,1}, {3,2,0,1,1});
    auto expected = NDArrayFactory::create<double>('c',{5}, {4.,3,1,2,2});

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test6) {

    auto input = NDArrayFactory::create<double>('c', {4}, {1,2,3,4});
    std::vector<Nd4jLong> shape = {1};
    auto indices = NDArrayFactory::create<int>('c', shape, {2});
    auto expected = NDArrayFactory::create<double>(3.);

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test7) {

    auto input = NDArrayFactory::create<double>('c', {4, 4});
    input.linspace(1);
    auto indices = NDArrayFactory::create<int>('c', {3,3,2}, {0,2,1, 0,1,0, 1,3,1, 0,2,1, 0,1,0, 1,3,1});
    auto expected = NDArrayFactory::create<double>('c', {3,3}, {3,5,5,8,5,10,2,2,14});

    nd4j::ops::gather_nd op;
    auto results = op.execute({&input, &indices}, {}, {}, {true});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test8) {
    auto x = NDArrayFactory::create<double>('c', {2, 2}, {1., 2., 3., 4.});
    auto y = NDArrayFactory::create<int>('c', {2, 2}, {0, 0, 1, 1});
    auto e = NDArrayFactory::create<double>('c', {2}, {1., 4.});

    nd4j::ops::gather_nd op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests5, gatherNd_test9) {
    auto x = NDArrayFactory::create<double>('c', {2, 4, 2, 2});
    auto indices = NDArrayFactory::create<int>('c', {3, 3}, {0,2,1, 0,1,0, 1,3,1});
    auto exp = NDArrayFactory::create<double>('c', {3,2}, {11.f, 12.f, 5.f, 6.f, 31.f, 32.f});
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

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test10) {

    auto input = NDArrayFactory::create<double>('c', {4, 3, 2});
    auto indices = NDArrayFactory::create<int>('c', {2,2,2}, {30,20,1,2, 0,10,0,1});

    auto output = NDArrayFactory::create<double>('c', {2,2,2});

    nd4j::ops::gather_nd op;

    ASSERT_ANY_THROW(op.execute({&input, &indices}, {&output}, {}, {}, {true}));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test11) {

    auto input = NDArrayFactory::create<double>('c', {4, 4});
    auto indices = NDArrayFactory::create<int>('c', {3,3,2}, {0,2,1, 0,10,0, 1,30,1, 0,20,1, 0,1,0, 1,30,1});
    auto output = NDArrayFactory::create<double>('c', {3,3});

    nd4j::ops::gather_nd op;

    ASSERT_ANY_THROW(op.execute({&input, &indices}, {&output}, {}, {}, {true}));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test1) {

    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<int>('c', {4}, {4,4,4,4});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {4,  3,  2,  1,  5, 9,  8,  7,  6, 10, 14, 13, 12, 11, 15, 19, 18, 17, 16, 20, 24, 23, 22, 21, 25, 29, 28, 27, 26, 30, 34, 33, 32, 31, 35, 39, 38, 37, 36, 40, 44, 43, 42, 41, 45, 49, 48, 47, 46, 50, 54, 53, 52, 51, 55, 59, 58, 57, 56, 60});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {2, 1});
    ASSERT_EQ(Status::OK(), results->status());

    auto output = results->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test2) {

    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<Nd4jLong>('c', {4}, {0,1,2,3});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {1,  2,  3,  4,  5, 6,  7,  8,  9, 10, 12, 11, 13, 14, 15, 18, 17, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31, 33, 34, 35, 38, 37, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 51, 53, 54, 55, 58, 57, 56, 59, 60});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {2, 1});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test3) {

    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<int>('c', {3}, {2,3,4});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {2,  1,  3,  4,  5, 7,  6,  8,  9, 10, 12, 11, 13, 14, 15, 17, 16, 18, 19, 20, 23, 22, 21, 24, 25, 28, 27, 26, 29, 30, 33, 32, 31, 34, 35, 38, 37, 36, 39, 40, 44, 43, 42, 41, 45, 49, 48, 47, 46, 50, 54, 53, 52, 51, 55, 59, 58, 57, 56, 60});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {2, 0});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test4) {

    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<int>('c', {5}, {1, 2, 1, 2, 3});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {1, 22,  3, 24, 45, 6, 27,  8, 29, 50, 11, 32, 13, 34, 55, 16, 37, 18, 39, 60, 21,  2, 23,  4, 25, 26,  7, 28,  9, 30, 31, 12, 33, 14, 35, 36, 17, 38, 19, 40, 41, 42, 43, 44,  5, 46, 47, 48, 49, 10, 51, 52, 53, 54, 15, 56, 57, 58, 59, 20});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {0, 2});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test5) {

    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<int>('c', {5}, {1, 2, 4, 2, 3});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {1,  7, 18,  9, 15, 6,  2, 13,  4, 10, 11, 12,  8, 14,  5, 16, 17,  3, 19, 20, 21, 27, 38, 29, 35, 26, 22, 33, 24, 30, 31, 32, 28, 34, 25, 36, 37, 23, 39, 40, 41, 47, 58, 49, 55, 46, 42, 53, 44, 50, 51, 52, 48, 54, 45, 56, 57, 43, 59, 60});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {1, 2});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

	delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test6) {

    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto seqLengths = NDArrayFactory::create<int>('c', {4}, {1, 2, 3, 2});
    auto exp = NDArrayFactory::create<double>('c', {3, 4, 5}, {1,  2,  3,  4,  5, 26, 27, 28, 29, 30, 51, 52, 53, 54, 55, 36, 37, 38, 39, 40, 21, 22, 23, 24, 25, 6,  7,  8,  9, 10, 31, 32, 33, 34, 35, 16, 17, 18, 19, 20, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 11, 12, 13, 14, 15, 56, 57, 58, 59, 60});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {0, 1});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test7) {

    auto input = NDArrayFactory::create<double>('c', {1, 5});
    input.linspace(1);
    std::vector<int> data = {3};
    auto seqLengths = NDArrayFactory::create<int>('c', {1}, data);
    auto exp = NDArrayFactory::create<double>('c', {1, 5}, {3, 2, 1, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {1, 0});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test8) {

    auto input = NDArrayFactory::create<double>('c', {1, 5});
    input.linspace(1);
    std::vector<int> data = {1,0,1,0,1};
    auto seqLengths = NDArrayFactory::create<int>('c', {5}, data);
    auto exp = NDArrayFactory::create<double>('c', {1, 5}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {0, 1});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test9) {

    auto input = NDArrayFactory::create<double>('c', {5, 1});
    input.linspace(1);
    std::vector<Nd4jLong> data = {1,0,1,0,1};
    auto seqLengths = NDArrayFactory::create<Nd4jLong>('c', {5}, data);
    auto exp = NDArrayFactory::create<double>('c', {5, 1}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {1, 0});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

	delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test10) {

    auto input = NDArrayFactory::create<double>('c', {5, 1});
    input.linspace(1);
    std::vector<int> data = {3};
    auto seqLengths = NDArrayFactory::create<int>('c', {1}, data);
    auto exp = NDArrayFactory::create<double>('c', {5, 1}, {3, 2, 1, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {0, 1});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test11) {

    auto input = NDArrayFactory::create<double>('c', {1, 1, 5, 1});
    input.linspace(1);
    std::vector<int> data = {1, 0, 1, 0, 1};
    auto seqLengths = NDArrayFactory::create<int>('c', {5}, data);
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 5, 1}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {1, 2});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test12) {

    auto input = NDArrayFactory::create<double>('c', {1, 1, 5, 1});
    input.linspace(1);
    std::vector<int> data = {3};
    auto seqLengths = NDArrayFactory::create<int>('c', {1}, data);
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 5, 1}, {3, 2, 1, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {2, 0});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test13) {

    auto input = NDArrayFactory::create<double>('c', {1, 1, 5, 1});
    input.linspace(1);
    std::vector<int> data = {1};
    auto seqLengths = NDArrayFactory::create<int>('c', {1}, data);
    auto exp = NDArrayFactory::create<double>('c', {1, 1, 5, 1}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &seqLengths}, {}, {3, 0});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test14) {
    auto input = NDArrayFactory::create<double>('c', {8, 8, 3, 2}, {0.09753360,             0.76124972,             0.24693797,             0.13813169,             0.33144656,             0.08299957,             0.67197708,             0.80659380,             0.98274191,             0.63566073,             0.21592326,             0.54902743,             0.54555996,             0.23407607,             0.11372584,             0.49965927,             0.15210842,             0.53268608,             0.38700677,             0.68832738,             0.37292716,             0.94616004,             0.77735792,             0.60803430,             0.61523204,             0.64298760,             0.26848351,             0.75015615,             0.28683049,             0.70937606,             0.06478678,             0.68985848,             0.55216783,             0.55382648,             0.34652863,             0.17261296,             0.54193264,             0.05176904,             0.82555761,             0.71106697,             0.04416722,             0.07653656,             0.01034390,             0.99430482,             0.59944390,             0.17973880,             0.36437840,             0.86383673,             0.45025550,             0.97136977,             0.13565978,             0.71567448,             0.92094825,             0.93536442,             0.93630291,             0.67277404,             0.93899264,             0.52422773,             0.44892176,             0.03127759,             0.85910449,             0.18252879,             0.72830945,             0.96736828,             0.89831575,             0.83437150,             0.59050780,             0.36145925,             0.16483070,             0.44021176,             0.76018652,             0.44227383,             0.13052339,             0.18204235,             0.99743733,             0.26885190,             0.87726522,             0.16396056,             0.94943412,             0.40016700,             0.65267938,             0.71073267,             0.40094733,             0.91182634,             0.05391789,             0.49520416,             0.24963864,             0.34847086,             0.74088617,             0.36115701,             0.63074210,             0.97423085,             0.42216846,             0.06326975,             0.07858702,             0.20586622,             0.28752144,             0.38146961,             0.83518735,             0.08207577,             0.82083487,             0.81665728,             0.33309570,             0.67563176,             0.98343578,             0.95919930,             0.66994391,             0.89296165,             0.34755773,             0.63166554,             0.18849320,             0.34828456,             0.98477707,             0.75163124,             0.83306004,             0.14203056,             0.01497920,             0.85727447,             0.71194544,             0.85654019,             0.86160433,             0.79580411,             0.47710411,             0.09318029,             0.31369071,             0.64122249,             0.58399725,             0.26706597,             0.05655339,             0.91025211,             0.30330468,             0.33142930,             0.05668627,             0.02936449,             0.12613087,             0.09960114,             0.16218074,             0.15088139,             0.31239040,             0.55980062,             0.34804391,             0.34941538,             0.61370555,             0.07022964,             0.59757058,             0.31189846,             0.25215345,             0.52546591,             0.55744218,             0.59485650,             0.60553664,             0.07536713,             0.55971796,             0.38764845,             0.20737843,             0.37989120,             0.18361641,             0.48636240,             0.06052657,             0.04241913,             0.66710351,             0.07007925,             0.59371493,             0.74479056,             0.84699625,             0.51210368,             0.12489571,             0.23371067,             0.27274571,             0.83306066,             0.75830824,             0.25963478,             0.87137718,             0.24418835,             0.05032742,             0.52076188,             0.47762345,             0.89829370,             0.34417708,             0.84705151,             0.08203183,             0.10632956,             0.78431292,             0.86441722,             0.36487598,             0.09833603,             0.85863594,             0.11010505,             0.11659283,             0.42500288,             0.02747301,             0.12359903,             0.01753431,             0.41160932,             0.47245979,             0.08268172,             0.21580773,             0.75770279,             0.19736489,             0.44461885,             0.33341706,             0.22519571,             0.31528710,             0.14802902,             0.64171939,             0.52643769,             0.19261234,             0.98032835,             0.15401656,             0.85274458,             0.66408502,             0.23212704,             0.74630026,             0.05713613,             0.49025892,             0.48418810,             0.59541513,             0.09243053,             0.93919152,             0.95357019,             0.52377729,             0.65963871,             0.47934951,             0.49919534,             0.34369898,             0.78211256,             0.13908708,             0.95754117,             0.84107746,             0.09126213,             0.42979124,             0.10295325,             0.34631257,             0.69448345,             0.41720536,             0.15282440,             0.74329854,             0.45775009,             0.12786280,             0.39830299,             0.20386769,             0.59703523,             0.94077086,             0.42255597,             0.80453309,             0.79757204,             0.28653229,             0.60175909,             0.55859623,             0.34318230,             0.63002770,             0.36533324,             0.89689906,             0.73236186,             0.61491989,             0.83787947,             0.67939463,             0.72016694,             0.77499849,             0.72428343,             0.34571059,             0.23143007,             0.20099338,             0.85583142,             0.73174191,             0.54284092,             0.20264181,             0.53037061,             0.30493131,             0.82279766,             0.58542432,             0.72632070,             0.18394258,             0.00608118,             0.23808232,             0.17007573,             0.75245459,             0.84990616,             0.38827634,             0.33809538,             0.01080317,             0.27250145,             0.81769542,             0.15323253,             0.71668395,             0.99427044,             0.11355576,             0.50511923,             0.60248266,             0.36610154,             0.99123140,             0.10519719,             0.18754650,             0.43232584,             0.25247084,             0.47968157,             0.88649124,             0.33588961,             0.92338319,             0.18808573,             0.79433656,             0.12074559,             0.02325163,             0.10117917,             0.83559239,             0.67213900,             0.67265260,             0.11917707,             0.76574855,             0.43842117,             0.28530411,             0.79648090,             0.47939640,             0.73564612,             0.41465671,             0.10995635,             0.20271728,             0.00521771,             0.22952055,             0.78271870,             0.12833592,             0.88639055,             0.76398188,             0.49533508,             0.85447872,             0.15937568,             0.92947480,             0.62705964,             0.85960084,             0.13435660,             0.81845809,             0.60715133,             0.83030708,             0.83071910,             0.38883408,             0.92033237,             0.46066239,             0.48806761,             0.50688779,             0.00654483,             0.32076493,             0.42367646,             0.07381865,             0.22801110,             0.26669388,             0.99691302,             0.12113623,             0.34373057,             0.98977921,             0.96225332,             0.90143562,             0.19559914,             0.08978307,             0.09687492,             0.59820890,             0.75527947,             0.67683355,             0.21847023,             0.29395619,             0.50477953,             0.07112842,             0.54090558,             0.68230725,             0.49713828,             0.41958965,             0.68013847,             0.47691765,             0.63269259,             0.94304095,             0.54587271,             0.72447569,             0.28913523,             0.75766936,             0.52965692,             0.96854824,             0.15589071,             0.84128672,             0.16337522,             0.05771034,             0.21556356,             0.12094140,             0.29721207,             0.00811008,             0.66184926});
    auto lengths = NDArrayFactory::create<Nd4jLong>('c', {8}, {7, 2, 3, 5, 2, 1, 6, 4});
    auto e = NDArrayFactory::create<double>('c', {8, 8, 3, 2}, {0.54193264,             0.05176904,             0.82555761,             0.71106697,             0.04416722,             0.07653656,             0.06478678,             0.68985848,             0.55216783,             0.55382648,             0.34652863,             0.17261296,             0.61523204,             0.64298760,             0.26848351,             0.75015615,             0.28683049,             0.70937606,             0.38700677,             0.68832738,             0.37292716,             0.94616004,             0.77735792,             0.60803430,             0.54555996,             0.23407607,             0.11372584,             0.49965927,             0.15210842,             0.53268608,             0.67197708,             0.80659380,             0.98274191,             0.63566073,             0.21592326,             0.54902743,             0.09753360,             0.76124972,             0.24693797,             0.13813169,             0.33144656,             0.08299957,             0.01034390,             0.99430482,             0.59944390,             0.17973880,             0.36437840,             0.86383673,             0.93630291,             0.67277404,             0.93899264,             0.52422773,             0.44892176,             0.03127759,             0.45025550,             0.97136977,             0.13565978,             0.71567448,             0.92094825,             0.93536442,             0.85910449,             0.18252879,             0.72830945,             0.96736828,             0.89831575,             0.83437150,             0.59050780,             0.36145925,             0.16483070,             0.44021176,             0.76018652,             0.44227383,             0.13052339,             0.18204235,             0.99743733,             0.26885190,             0.87726522,             0.16396056,             0.94943412,             0.40016700,             0.65267938,             0.71073267,             0.40094733,             0.91182634,             0.05391789,             0.49520416,             0.24963864,             0.34847086,             0.74088617,             0.36115701,             0.63074210,             0.97423085,             0.42216846,             0.06326975,             0.07858702,             0.20586622,             0.34755773,             0.63166554,             0.18849320,             0.34828456,             0.98477707,             0.75163124,             0.33309570,             0.67563176,             0.98343578,             0.95919930,             0.66994391,             0.89296165,             0.28752144,             0.38146961,             0.83518735,             0.08207577,             0.82083487,             0.81665728,             0.83306004,             0.14203056,             0.01497920,             0.85727447,             0.71194544,             0.85654019,             0.86160433,             0.79580411,             0.47710411,             0.09318029,             0.31369071,             0.64122249,             0.58399725,             0.26706597,             0.05655339,             0.91025211,             0.30330468,             0.33142930,             0.05668627,             0.02936449,             0.12613087,             0.09960114,             0.16218074,             0.15088139,             0.31239040,             0.55980062,             0.34804391,             0.34941538,             0.61370555,             0.07022964,             0.27274571,             0.83306066,             0.75830824,             0.25963478,             0.87137718,             0.24418835,             0.59371493,             0.74479056,             0.84699625,             0.51210368,             0.12489571,             0.23371067,             0.18361641,             0.48636240,             0.06052657,             0.04241913,             0.66710351,             0.07007925,             0.60553664,             0.07536713,             0.55971796,             0.38764845,             0.20737843,             0.37989120,             0.59757058,             0.31189846,             0.25215345,             0.52546591,             0.55744218,             0.59485650,             0.05032742,             0.52076188,             0.47762345,             0.89829370,             0.34417708,             0.84705151,             0.08203183,             0.10632956,             0.78431292,             0.86441722,             0.36487598,             0.09833603,             0.85863594,             0.11010505,             0.11659283,             0.42500288,             0.02747301,             0.12359903,             0.19736489,             0.44461885,             0.33341706,             0.22519571,             0.31528710,             0.14802902,             0.01753431,             0.41160932,             0.47245979,             0.08268172,             0.21580773,             0.75770279,             0.64171939,             0.52643769,             0.19261234,             0.98032835,             0.15401656,             0.85274458,             0.66408502,             0.23212704,             0.74630026,             0.05713613,             0.49025892,             0.48418810,             0.59541513,             0.09243053,             0.93919152,             0.95357019,             0.52377729,             0.65963871,             0.47934951,             0.49919534,             0.34369898,             0.78211256,             0.13908708,             0.95754117,             0.84107746,             0.09126213,             0.42979124,             0.10295325,             0.34631257,             0.69448345,             0.41720536,             0.15282440,             0.74329854,             0.45775009,             0.12786280,             0.39830299,             0.20386769,             0.59703523,             0.94077086,             0.42255597,             0.80453309,             0.79757204,             0.28653229,             0.60175909,             0.55859623,             0.34318230,             0.63002770,             0.36533324,             0.89689906,             0.73236186,             0.61491989,             0.83787947,             0.67939463,             0.72016694,             0.77499849,             0.72428343,             0.34571059,             0.23143007,             0.20099338,             0.85583142,             0.73174191,             0.54284092,             0.20264181,             0.53037061,             0.30493131,             0.82279766,             0.58542432,             0.72632070,             0.18394258,             0.00608118,             0.23808232,             0.17007573,             0.75245459,             0.84990616,             0.38827634,             0.33809538,             0.01080317,             0.27250145,             0.81769542,             0.15323253,             0.71668395,             0.99427044,             0.11355576,             0.50511923,             0.22952055,             0.78271870,             0.12833592,             0.88639055,             0.76398188,             0.49533508,             0.47939640,             0.73564612,             0.41465671,             0.10995635,             0.20271728,             0.00521771,             0.67265260,             0.11917707,             0.76574855,             0.43842117,             0.28530411,             0.79648090,             0.79433656,             0.12074559,             0.02325163,             0.10117917,             0.83559239,             0.67213900,             0.25247084,             0.47968157,             0.88649124,             0.33588961,             0.92338319,             0.18808573,             0.60248266,             0.36610154,             0.99123140,             0.10519719,             0.18754650,             0.43232584,             0.85447872,             0.15937568,             0.92947480,             0.62705964,             0.85960084,             0.13435660,             0.81845809,             0.60715133,             0.83030708,             0.83071910,             0.38883408,             0.92033237,             0.59820890,             0.75527947,             0.67683355,             0.21847023,             0.29395619,             0.50477953,             0.98977921,             0.96225332,             0.90143562,             0.19559914,             0.08978307,             0.09687492,             0.07381865,             0.22801110,             0.26669388,             0.99691302,             0.12113623,             0.34373057,             0.46066239,             0.48806761,             0.50688779,             0.00654483,             0.32076493,             0.42367646,             0.07112842,             0.54090558,             0.68230725,             0.49713828,             0.41958965,             0.68013847,             0.47691765,             0.63269259,             0.94304095,             0.54587271,             0.72447569,             0.28913523,             0.75766936,             0.52965692,             0.96854824,             0.15589071,             0.84128672,             0.16337522,             0.05771034,             0.21556356,             0.12094140,             0.29721207,             0.00811008,             0.66184926});

    nd4j::ops::reverse_sequence op;
    auto results = op.execute({&input, &lengths}, {}, {1, 0});
    ASSERT_EQ(Status::OK(), results->status());

    auto z = results->at(0);

    ASSERT_EQ(e, *z);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_0) {
    auto x = NDArrayFactory::create<double>('c', {2, 6}, {1.0, 1.0, 1.0, 1.0, 11.0, 3.0, 1.0, 1.0, 1.0, 14.0, 5.0, 6.0});
    auto expV = NDArrayFactory::create<double>('c', {2, 1}, {11.0, 14.0});
    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 1}, {4, 3});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {1, 0}); // without sorting

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);
/*
    v->printShapeInfo("topK_0: shape v");
    expV.printShapeInfo("topK_0: shape expV");

    i->printShapeInfo("topK_0: shape I");
    expI.printShapeInfo("topK_0: shape expI");

    v->printIndexedBuffer("topK_0: v");
    expV.printIndexedBuffer("topK_0: expV");
    i->printIndexedBuffer("topK_0: i");
    expI.printIndexedBuffer("topK_0: expI");
*/

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));
    // repeat res again
    for (int cases = 0; cases < 100; ++cases) {
        op.execute({&x}, {v, i}, {}, {1, 0}, {}); // without sorting
    }
    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 3}, {1.0f, 11.0f, 3.0f, 14.0f, 5.0f, 6.0f});
    auto expV = NDArrayFactory::create<double>('c', {2, 1}, {11.0f, 14.0f});
    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 1}, {1, 0});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {1, 0}); // without sorting

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);

//    v->printShapeInfo("topK_1: shape v");
//    expV.printShapeInfo("topK_1: shape expV");

//    i->printShapeInfo("topK_1: shape I");
//    expI.printShapeInfo("topK_1: shape expI");

//    v->printIndexedBuffer("topK_1: v");
//    expV.printIndexedBuffer("topK_1: expV");
//    i->printIndexedBuffer("topK_1: i");
//    expI.printIndexedBuffer("topK_1: expI");


    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));
    // repeat res again
    for (int cases = 0; cases < 100; ++cases) {
        op.execute({&x}, {v, i}, {}, {1, 0}, {}); // without sorting
    }
    delete result;
}

///////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_2) {
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {11.0,  3.0, 14.0, 5.0,
                                      6.0,  9.0, 3.5, 7.0,
                                      21.0, 3.0, 14.0, 15.0,
                                      6.0, 9.0, 3.5, 7.0,
                                      11.0, 13.0, 14.0, 5.0,
                                      16.0, 9.0, 13.5, 7.0
                     }
    );
// <<<14.>,<9.>>, <<21.>,<9.>>, <<14.>,<16.>>>
    auto expV = NDArrayFactory::create<double>('c', {2, 3, 1}, {14.0f, 9.0f,
                                         21.0f,
                                         9.0f, 14.0f,
                                         16.0f
                        }
    );

    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 3, 1 }, {2, 1, 0, 1, 2, 0});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);

//    v->printShapeInfo("shape v");
//    expV.printShapeInfo("shape expV");

//    i->printShapeInfo("shape I");
//    expI.printShapeInfo("shape expI");

//    v->printIndexedBuffer("v");
//    expV.printIndexedBuffer("expV");
//    i->printIndexedBuffer("i");
//    expI.printIndexedBuffer("expI");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_TopK_3) {
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {11.0,  3.0, 14.0, 5.0,
                                      6.0,  9.0, 3.5, 7.0,
                                      21.0, 3.0, 14.0, 15.0,
                                      6.0, 9.0, 3.5, 7.0,
                                      11.0, 13.0, 14.0, 5.0,
                                      16.0, 9.0, 13.5, 7.0
                     }
    );

    auto expV = NDArrayFactory::create<double>('c', {2, 3, 2}, {14.0f, 11.0f,
                                                                9.0f, 7.0f,
                                                                21.0f, 15.0f,
                                         9.0f, 7.0f,
                                         14.0f, 13.0f,
                                         16.0f, 13.5f
                        }
    );

    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 3, 2 }, {2, 0, 1, 3, 0, 3, 1,  3, 2, 1, 0, 2});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {2, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);

//    v->printShapeInfo("shape v");
//    expV.printShapeInfo("shape expV");

//    i->printShapeInfo("shape I");
//    expI.printShapeInfo("shape expI");

//    v->printIndexedBuffer("v");
//    expV.printIndexedBuffer("expV");
//    i->printIndexedBuffer("i");
//    expI.printIndexedBuffer("expI");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_TopK_3_unsorted) {
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {11.0,  3.0, 14.0, 5.0,
                                                             6.0,  9.0, 3.5, 7.0,
                                                             21.0, 3.0, 14.0, 15.0,
                                                             6.0, 9.0, 3.5, 7.0,
                                                             11.0, 13.0, 14.0, 5.0,
                                                             16.0, 9.0, 13.5, 7.0
                                            }
    );

    auto expV = NDArrayFactory::create<double>('c', {2, 3, 2}, {11.0f, 14.0f,
                                                                9.0f, 7.0f,
                                                                21.0f, 15.0f,
                                                                9.0f, 7.0f,
                                                                13.0f, 14.0f,
                                                                16.0f, 13.5f
                                               }
    );

    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 3, 2 }, {0, 2, 1, 3, 0, 3, 1,  3, 1, 2, 0, 2});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {2}, {false});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_4) {
    auto x = NDArrayFactory::create<double>('c', {2, 3}, {1.0f, 11.0f, 3.0f, 14.0f, 5.0f, 6.0f});
    auto expV = NDArrayFactory::create<double>('c', {2, 2}, {11.0f, 3.0f, 14.0f, 6.0f});
    auto expI = NDArrayFactory::create<Nd4jLong>('c', {2, 2}, {1, 2, 0, 2});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {2, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_5) {
    auto x = NDArrayFactory::create<double>('f', {2, 3}, {1.1, 5.2, 3.1, 14.2, 11.1, 6.2});
    auto expV = NDArrayFactory::create<double>('f', {2, 2}, {11.1, 14.2, 3.1, 6.2});
    auto expI = NDArrayFactory::create<Nd4jLong>('f', {2, 2}, {2, 1, 1, 2});

    nd4j::ops::top_k op;
    auto result = op.execute({&x}, {}, {2, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

///////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_Moments_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );

    auto y = NDArrayFactory::create<double>('c', {3}, {0, 1, 2});
    //auto expV('f', {6}, {1, 0, 0, 0, 0, 0 });

    float expMean = 9.395833f;
    float expDeviation = 22.4579f;
//Mean 9.395833
//Deviance 22.4579

    float inf = 1.e-5f;

    nd4j::ops::moments op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto d = result->at(1);

//    v->printIndexedBuffer("Result is ");
//    d->printIndexedBuffer("Result is ");

    ASSERT_TRUE(v->isScalar());
    ASSERT_NEAR(expMean, v->e<double>(0), inf);
    ASSERT_NEAR(expDeviation, d->e<double>(0), inf);

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_Moments_2) {
    NDArray x('c', {2, 3, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );

    NDArray expV('c', {4}, {11.833333, 7.6666665, 10.416667, 7.6666665});
    NDArray expD('c', {4}, {28.472221, 12.888889, 23.951387, 11.555554});

    nd4j::ops::moments op;
    auto result = op.execute({&x}, {}, {0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto d = result->at(1);

    ASSERT_TRUE(v->isVector());
    ASSERT_TRUE(d->isVector());

    ASSERT_TRUE(v->equalsTo(&expV));
    ASSERT_TRUE(d->equalsTo(&expD));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_Moments_3) {
    auto x = NDArrayFactory::create<double>('c', {2, 3, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );

    auto expV = NDArrayFactory::create<double>('c', {3, 4}, { 8.5f, 6.f , 8.75f,  6.f,
                                       8.5f, 11.f, 8.75f, 6.f,
                                      18.5f, 6.f, 13.75f, 11.f});
    auto expD = NDArrayFactory::create<double>('c', {3, 4}, { 6.25f, 9.f, 27.5625f,  1.f,
                                       6.25f, 4.f, 27.5625f,  1.f,
                                       6.25f, 9.f, 0.0625f,  16.f});

    nd4j::ops::moments op;
    auto result = op.execute({&x}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto d = result->at(1);

    ASSERT_TRUE(v->isMatrix());
    ASSERT_TRUE(d->isMatrix());

    ASSERT_TRUE(v->equalsTo(&expV));
    ASSERT_TRUE(d->equalsTo(&expD));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_Moments_4) {

    auto x = NDArrayFactory::create<double>('f', {2, 3, 4}, {11.0f,  6.0f,  6.0f, 11.0f, 21.0f, 16.0f,  3.0f,  9.0f, 9.0f, 13.0f,  3.0f,  9.0f,
                                      14.0f,  3.5f,  3.5f, 14.0f, 14.0f,  13.5f,  5.0f,  7.0f, 7.0f,  5.0f, 15.0f,  7.0f});


    auto expV = NDArrayFactory::create<double>('c', {3, 4}, { 8.5f, 6.f , 8.75f,  6.f, 8.5f, 11.f, 8.75f, 6.f, 18.5f, 6.f, 13.75f, 11.f});
    auto expD = NDArrayFactory::create<double>('c', {3, 4}, { 6.25f, 9.f, 27.5625f,  1.f, 6.25f, 4.f, 27.5625f,  1.f, 6.25f, 9.f, 0.0625f,  16.f});

    nd4j::ops::moments op;
    auto result = op.execute({&x}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto d = result->at(1);

    ASSERT_TRUE(v->isMatrix());
    ASSERT_TRUE(d->isMatrix());

    // v->printIndexedBuffer("v");
    // expV.printIndexedBuffer("expV");

    // d->printIndexedBuffer("d");
    // expD.printIndexedBuffer("expD");

    ASSERT_TRUE(v->equalsTo(&expV));
    ASSERT_TRUE(d->equalsTo(&expD));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test1) {

    auto input = NDArrayFactory::create<double>('c', {3, 4, 5});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {3}, {40, 120, 200});
    NDArray matrix('c', {3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
    nd4j::ops::trace op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);
    double traceM = matrix.getTrace();
    // nd4j_printf("Trace for matrix is %f\n", traceM);
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    // exp.printIndexedBuffer("EXP TRACE");
    // output->printIndexedBuffer("OUT TRACE");
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test2) {

    auto input = NDArrayFactory::create<double>('c', {4, 5});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>(40.);

    nd4j::ops::trace op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test3) {

    auto input = NDArrayFactory::create<double>('c', {1, 5});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>(1.);

    nd4j::ops::trace op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test4) {

    auto input = NDArrayFactory::create<double>('c', {5, 1});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>(1.);

    nd4j::ops::trace op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test5) {

    auto input = NDArrayFactory::create<double>('c', {3, 4, 5, 6});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {3, 4}, {75,  225,  375,  525, 675,  825,  975, 1125, 1275, 1425, 1575, 1725});

    nd4j::ops::trace op;
    auto results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test1) {

    auto input = NDArrayFactory::create<double>('c', {2, 2, 2});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto output = results->at(0);

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if(output->e<float>(i) == (float)0.)
            haveZeros = true;

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test2) {

    auto input = NDArrayFactory::create<double>('c', {1, 3, 2});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(input.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test3) {

    auto input = NDArrayFactory::create<double>('c', {3, 2, 1});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto output = results->at(0);

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if(output->e<float>(i) == (float)0.)
            haveZeros = true;

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test04) {
    auto input = NDArrayFactory::create<double>('c', {4});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    //NDArray* output;
    auto results = op.execute({&input}, {},  {},  {}, true, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), results->status());
    auto output = &input; //results->at(0);
    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if(output->e<float>(i) == (float)0.)
            haveZeros = true;

    ASSERT_TRUE(input.isSameShape(output));
    //ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test4) {
    auto input = NDArrayFactory::create<double>('c', {4});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    //NDArray* output;
    auto results = op.execute({&input}, {},  {},  {}, false, nd4j::DataType::DOUBLE);
    ASSERT_EQ(Status::OK(), results->status());
    auto output = results->at(0);
    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if(output->e<float>(i) == (float)0.)
            haveZeros = true;

    ASSERT_TRUE(input.isSameShape(output));
    //ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test5) {

    auto input = NDArrayFactory::create<double>('c', {4,1});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto output = results->at(0);

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if(output->e<float>(i) == (float)0.)
            haveZeros = true;

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test6) {

    auto input = NDArrayFactory::create<double>('c', {4,1,1});
    input.linspace(1);

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto output = results->at(0);

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if(output->e<float>(i) == (float)0.)
            haveZeros = true;

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test7) {

    auto input = NDArrayFactory::create<double>('c', {1,4});
    input.linspace(1);
    auto exp = NDArrayFactory::create<double>('c', {1,4}, {1, 2, 3, 4});

    nd4j::ops::random_shuffle op;
    auto results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(input.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, EmbeddingLookup_1) {

    auto x = NDArrayFactory::create<double>('c', {3, 4, 2}, {10, 20, 11, 21, 12, 22, 13, 23,
                                      14, 24, 15, 25, 16, 26, 17, 27,
                                      18, 28, 19, 29, 20, 30, 21, 31});

    auto y = NDArrayFactory::create<int>({1, 1, 1, 0, 0, 0, 2, 2, 2});
    auto exp = NDArrayFactory::create<double>('c', {9, 4, 2}, {14, 24, 15, 25, 16, 26, 17, 27, 14, 24, 15, 25,
                                        16, 26, 17, 27, 14, 24, 15, 25, 16, 26, 17, 27,
                                        10, 20, 11, 21, 12, 22, 13, 23, 10, 20, 11, 21,
                                        12, 22, 13, 23, 10, 20, 11, 21, 12, 22, 13, 23,
                                        18, 28, 19, 29, 20, 30, 21, 31, 18, 28, 19, 29,
                                        20, 30, 21, 31, 18, 28, 19, 29, 20, 30, 21, 31});

    // y.printShapeInfo("y shape");
    // y.printIndexedBuffer("y buffer");

    nd4j::ops::embedding_lookup op;
    auto result = op.execute({&x, &y}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    auto output = result->at(0);
    // x.printShapeInfo("Input");
    // output->printShapeInfo("Output");
    // exp.printShapeInfo("Expected");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_TRUE(exp.isSameShape(output));
    //output->printIndexedBuffer("Output");
    //exp.printIndexedBuffer("Expect");

    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

TEST_F(DeclarableOpsTests5, EmbeddingLookup_2) {

    auto x = NDArrayFactory::create<double>('c', {3, 4, 2}, {10, 20, 30, 40, 50, 60,
                                      70, 80, 90, 10, 11, 12,
                                      13, 14, 15, 16, 17, 18,
                                      19, 20, 21, 22, 23, 24});
                    //1,   0,   1,   0,   1,   0
    auto y = NDArrayFactory::create<Nd4jLong>({1, 0, 1, 0, 1, 0});
    auto exp = NDArrayFactory::create<double>('c', {6, 4, 2}, {90, 10, 11, 12, 13, 14,
                                        15, 16, 10, 20, 30, 40,
                                        50, 60, 70, 80, 90, 10,
                                        11, 12, 13, 14, 15, 16,
                                        10, 20, 30, 40, 50, 60,
                                        70, 80, 90, 10, 11, 12,
                                        13, 14, 15, 16, 10, 20,
                                        30, 40, 50, 60, 70, 80});

    // y.printShapeInfo("y shape");
    // y.printIndexedBuffer("y buffer");

    nd4j::ops::embedding_lookup op;
    auto result = op.execute({&x, &y}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    auto output = result->at(0);
    // x.printShapeInfo("Input");
    // output->printShapeInfo("Output");
    // exp.printShapeInfo("Expected");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_TRUE(exp.isSameShape(output));
    // output->printIndexedBuffer("Output");
    // exp.printIndexedBuffer("Expect");

    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

TEST_F(DeclarableOpsTests5, EmbeddingLookup_3) {


    auto y = NDArrayFactory::create<Nd4jLong>('c', {3,2}, {5, 4, 4, 5, 3, 3});
    auto exp = NDArrayFactory::create<double>('c', {6, 3, 3}, {
                6, 20, 11,    21, 12, 22,    13, 23, 14,
                5, 20, 11,    21, 12, 22,    13, 23, 14,
                5, 20, 11,    21, 12, 22,    13, 23, 14,
                6, 20, 11,    21, 12, 22,    13, 23, 14,
                4, 20, 11,    21, 12, 22,    13, 23, 14,
                4, 20, 11,    21, 12, 22,    13, 23, 14 });

    // y.printShapeInfo("y shape");
    // y.printIndexedBuffer("y buffer");
    auto p1 = NDArrayFactory::create<double>('c', {3,3}, {1, 20, 11, 21, 12, 22, 13, 23, 14});
    auto p2 = NDArrayFactory::create<double>('c', {3,3}, {2, 20, 11, 21, 12, 22, 13, 23, 14});
    auto p3 = NDArrayFactory::create<double>('c', {3,3}, {3, 20, 11, 21, 12, 22, 13, 23, 14});
    auto p4 = NDArrayFactory::create<double>('c', {3,3}, {4, 20, 11, 21, 12, 22, 13, 23, 14});
    auto p5 = NDArrayFactory::create<double>('c', {3,3}, {5, 20, 11, 21, 12, 22, 13, 23, 14});
    auto p6 = NDArrayFactory::create<double>('c', {3,3}, {6, 20, 11, 21, 12, 22, 13, 23, 14});
    auto p7 = NDArrayFactory::create<double>('c', {3,3}, {7, 20, 11, 21, 12, 22, 13, 23, 14});
    auto p8 = NDArrayFactory::create<double>('c', {3,3}, {8, 20, 11, 21, 12, 22, 13, 23, 14});

//    res = tf.nn.embedding_lookup((p1, p2, p3, p4, p5, p6, p7), ids, 'mod')

    nd4j::ops::embedding_lookup op;
    auto result = op.execute({&p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &y}, {}, {1}, {}, false, nd4j::DataType::DOUBLE);
    auto output = result->at(0);
    // x.printShapeInfo("Input");
    // output->printIndexedBuffer("Output");
    // exp.printShapeInfo("Expected");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_TRUE(exp.isSameShape(output));
    // output->printIndexedBuffer("Output");
    // exp.printIndexedBuffer("Expect");

    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}
/* @Test
    public void testDynamicPartition(){
        INDArray data = Nd4j.createFromArray(2, 1, 2, 0);
        INDArray partitions = Nd4j.createFromArray(0, 2, 1, 0);
        INDArray[] out = Nd4j.exec(DynamicCustomOp.builder("dynamic_partition")
                .addOutputs(Nd4j.createUninitialized(DataType.INT, 2), Nd4j.createUninitialized(DataType.INT, 1), Nd4j.createUninitialized(DataType.INT, 1))
                .addIntegerArguments(3) //3 partitions
                .addInputs(data, partitions).build());

        INDArray exp0 = Nd4j.createFromArray(2, 0);
        INDArray exp1 = Nd4j.createFromArray(2);
        INDArray exp2 = Nd4j.createFromArray(1);

        assertEquals(exp0, out[0]);     //Usually just gives [0,0]
        assertEquals(exp1, out[1]);
        assertEquals(exp2, out[2]);
    }*/
TEST_F(DeclarableOpsTests5, DynamicPartition_01) {

    auto x = NDArrayFactory::create<int>({2,1,2,0});

    auto y = NDArrayFactory::create<int>({0,2,1,0});

    int numPartition = 3;
    std::vector<NDArray> exp( { NDArrayFactory::create<int>('c', {2}, {2, 0}),
                                NDArrayFactory::create<int>('c', {1}, {2}),
                                NDArrayFactory::create<int>('c', {1}, {1})});

    nd4j::ops::dynamic_partition op;
    auto result = op.execute({&x, &y}, {}, {numPartition});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(result->size(), numPartition); // result has the same size as given param 4

    for (int e = 0; e < result->size(); e++) {
        auto output = result->at(e);
        // output->printShapeInfo("Output shape> ");
        // output->printIndexedBuffer("Output data> ");
        ASSERT_TRUE(exp[e].isSameShape(output));
        ASSERT_TRUE(exp[e].equalsTo(output));
    }

    delete result;
}

TEST_F(DeclarableOpsTests5, DynamicPartition_1) {

    auto x = NDArrayFactory::create<double>('c', {3, 4, 2}, {10, 20, 11, 21, 12, 22,
                                      13, 23, 14, 24, 15, 25, 16, 26, 17, 27,
                                      18, 28, 19, 29, 20, 30, 21, 31});

    auto y = NDArrayFactory::create<int>('c', {3, 4, 2}, {0, 0, 0, 0, 0, 0,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                      1, 1, 1, 1, 1, 1, 1, 1
                    }
    );
/*    auto y = NDArrayFactory::create<double>('c', {3, 4}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                      2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
                      1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f
                    }
    );
*/
    int numPartition = 3;
    std::vector<NDArray> exp( { NDArrayFactory::create<double>('c', {6}, {10, 20, 11, 21, 12, 22}),
                                NDArrayFactory::create<double>('c', {8}, {18, 28, 19, 29, 20, 30, 21, 31}),
                                NDArrayFactory::create<double>('c', {10}, {13, 23, 14, 24, 15, 25, 16, 26, 17, 27})});

    nd4j::ops::dynamic_partition op;
    auto result = op.execute({&x, &y}, {}, {numPartition});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(result->size(), numPartition); // result has the same size as given param 4

    for (int e = 0; e < result->size(); e++) {
        auto output = result->at(e);
        // output->printShapeInfo("Output shape> ");
        // output->printIndexedBuffer("Output data> ");
        ASSERT_TRUE(exp[e].isSameShape(output));
        ASSERT_TRUE(exp[e].equalsTo(output));
    }

    delete result;
}

////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, DynamicPartition_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 4}, {0.1f, -1.f, 5.2f, 4.3f, -1.f, 7.4f, 0.0f, -2.2f});
    auto y = NDArrayFactory::create<int>('c', {2, 4}, {1, 2, 1, 2, 1, 2, 3, 0});

    std::vector<NDArray> exp( {NDArrayFactory::create<double>('c', {1}, {-2.2}),
                               NDArrayFactory::create<double>('c', {3}, {0.1, 5.2, -1.}),
                               NDArrayFactory::create<double>('c', {3}, {-1., 4.3, 7.4}),
                               NDArrayFactory::create<double>('c', {1}, {0.0})});

    nd4j::ops::dynamic_partition op;
    int numPartition = 4;
    auto result = op.execute({&x, &y}, {}, {numPartition});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(result->size(), numPartition); // result has the same size as given param 4

    for (int e = 0; e < result->size(); e++) {
        auto output = result->at(e);

        ASSERT_TRUE(exp[e].isSameShape(output));
        ASSERT_TRUE(exp[e].equalsTo(output));
    }

    delete result;
}


TEST_F(DeclarableOpsTests5, DynamicPartition_3) {

    auto x = NDArrayFactory::create<double>('c', {2, 4}, {0.1f, -1.f, 5.2f, 4.3f, -1.f, 7.4f, 0.0f, -2.2f});
    auto y = NDArrayFactory::create<Nd4jLong>('c', {2, 4}, {0, 1, 0, 2, 0, 2, 3, 0});

    std::vector<NDArray> exp( {NDArrayFactory::create<double>({0.1f, 5.2f, -1.f, -2.2f}),
                               NDArrayFactory::create<double>('c', {1}, {-1.f}),
                               NDArrayFactory::create<double>({4.3f, 7.4f}),
                               NDArrayFactory::create<double>('c', {1}, {0.0f})});

    nd4j::ops::dynamic_partition op;
    int numPartition = 4;
    auto result = op.execute({&x, &y}, {}, {numPartition});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(result->size(), numPartition); // result has the same size as given param 4

    for (int e = 0; e < result->size(); e++) {
        auto output = result->at(e);
        if (output)
        {
            // output->printShapeInfo("Output shape> ");
            // exp[e].printShapeInfo("Expected shape> ");
            // output->printIndexedBuffer("Output data> ");

            ASSERT_TRUE(exp[e].isSameShape(output));
            ASSERT_TRUE(exp[e].equalsTo(output));
        }
        else
        {
            ASSERT_TRUE(exp[e].lengthOf() == 0);
        }
    }

    delete result;
}

TEST_F(DeclarableOpsTests5, DynamicStitch_empty_1) {
    auto i0 = NDArrayFactory::create<int>('c', {2}, {2, 3});
    auto i1 = NDArrayFactory::empty<int>();
    auto i2 = NDArrayFactory::create<int>('c', {2}, {0, 1});

    auto d0 = NDArrayFactory::create<double>('c', {2, 5}, {0.085571885,0.7937801,0.65908563,0.55552566,0.15962744,0.7787856,0.80119777,0.72437465,0.23089433,0.72714126});
    auto d1 = NDArrayFactory::empty<double>();
    auto d2 = NDArrayFactory::create<double>('c', {2, 5}, {0.94414854,0.5956861,0.8668989,0.3502196,0.5100082,0.061725974,0.6621324,0.034165382,0.32576954,0.51917326});

    nd4j::ops::dynamic_stitch op;
    auto result = op.execute({&i0, &i1, &i2, &d0, &d1, &d2}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

TEST_F(DeclarableOpsTests5, DynamicStitch_empty_2) {
    auto i0 = NDArrayFactory::create<int>('c', {2}, {2, 3});
    auto i1 = NDArrayFactory::create<int>('c', {0});
    auto i2 = NDArrayFactory::create<int>('c', {2}, {0, 1});

    auto d0 = NDArrayFactory::create<double>('c', {2, 5}, {0.085571885,0.7937801,0.65908563,0.55552566,0.15962744,0.7787856,0.80119777,0.72437465,0.23089433,0.72714126});
    auto d1 = NDArrayFactory::create<double>('c', {0, 5});
    auto d2 = NDArrayFactory::create<double>('c', {2, 5}, {0.94414854,0.5956861,0.8668989,0.3502196,0.5100082,0.061725974,0.6621324,0.034165382,0.32576954,0.51917326});

    nd4j::ops::dynamic_stitch op;
    auto result = op.execute({&i0, &i1, &i2, &d0, &d1, &d2}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, DynamicStitch_1) {

    auto x1 = NDArrayFactory::create<int>({1, 3, 5, 0});
    auto x2 = NDArrayFactory::create<int>({2, 4});
    auto y2 = NDArrayFactory::create<double>({-1., -1.});
    auto y1 = NDArrayFactory::create<double>({0.1f, 5.2f, 4.3f, 7.4f});


    auto exp = NDArrayFactory::create<double>({7.4f, 0.1f, -1.f, 5.2f, -1.f, 4.3f});

    nd4j::ops::dynamic_stitch op;
    auto result = op.execute({&x1, &x2, &y1, &y2}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, DynamicStitch_2) {

    auto x1 = NDArrayFactory::create<int>({1, 3});
    auto x2 = NDArrayFactory::create<int>({5, 0, 2, 4});
    auto y1 = NDArrayFactory::create<double>({-1.f, -1.f});
    auto y2 = NDArrayFactory::create<double>({0.1f, 5.2f, 4.3f, 7.4f});


    auto exp = NDArrayFactory::create<double>({5.2f, -1.f, 4.3f, -1.f, 7.4f, 0.1f});

    nd4j::ops::dynamic_stitch op;
    auto result = op.execute({&x1, &x2, &y1, &y2}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto output = result->at(0);

    // output->printShapeInfo("Output shape> ");
    // exp.printShapeInfo("Expected shape> ");
    // output->printIndexedBuffer("Output data> ");
    // exp.printIndexedBuffer("Expected res>");
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test1) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 3, 4});
    x.linspace(1);
    auto scale = NDArrayFactory::create<double>('c', {4});

    scale = 0.5;
    auto offset = NDArrayFactory::create<double>('c', {4});
    offset = 2.;
    auto expY = NDArrayFactory::create<double>('c', {2, 2, 3, 4}, {1.20337462,  1.20337462,  1.20337462,  1.20337462, 1.34821558,  1.34821558,  1.34821558,  1.34821558, 1.49305654,  1.49305654,  1.49305654,  1.49305654, 1.63789749,  1.63789749,  1.63789749,  1.63789749, 1.78273857,  1.78273857,  1.78273857,  1.78273857, 1.92757952,  1.92757952,  1.92757952,  1.92757952, 2.0724206 ,  2.0724206 ,  2.0724206 ,  2.0724206 , 2.21726155,  2.21726155,  2.21726155,  2.21726155, 2.36210251,  2.36210251,  2.36210251,  2.36210251, 2.50694346,  2.50694346,  2.50694346,  2.50694346, 2.65178442,  2.65178442,  2.65178442,  2.65178442, 2.79662538,  2.79662538,  2.79662538,  2.79662538});
    auto expBatchMean = NDArrayFactory::create<double>('c', {4}, {23.,  24.,  25.,  26.});
    auto expBatchVar = NDArrayFactory::create<double>('c', {4}, {208.00001526,  208.00001526,  208.00001526,  208.00001526});


    nd4j::ops::fused_batch_norm op;
    auto results = op.execute({&x, &scale, &offset}, {}, {0,1});
    auto y = results->at(0);
    auto batchMean = results->at(1);
    auto batchVar = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test2) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 3, 4});
    x.linspace(1);

    auto scale = NDArrayFactory::create<double>('c', {4});

    scale = 0.5;
    auto offset = NDArrayFactory::create<double>('c', {4});
    offset = 2.;
    auto expY = NDArrayFactory::create<double>('c', {2, 2, 3, 4}, {1.20347691,  1.20347691,  1.20347691,  1.20347691, 1.34829926,  1.34829926,  1.34829926,  1.34829926, 1.49312162,  1.49312162,  1.49312162,  1.49312162, 1.6379441 ,  1.6379441 ,  1.6379441 ,  1.6379441 , 1.78276646,  1.78276646,  1.78276646,  1.78276646, 1.92758882,  1.92758882,  1.92758882,  1.92758882, 2.0724113 ,  2.0724113 ,  2.0724113 ,  2.0724113 , 2.21723366,  2.21723366,  2.21723366,  2.21723366, 2.36205602,  2.36205602,  2.36205602,  2.36205602, 2.50687838,  2.50687838,  2.50687838,  2.50687838, 2.65170074,  2.65170074,  2.65170074,  2.65170074, 2.79652309,  2.79652309,  2.79652309,  2.79652309});
    auto expBatchMean = NDArrayFactory::create<double>('c', {4}, {23.,  24.,  25.,  26.});
    auto expBatchVar = NDArrayFactory::create<double>('c', {4}, {208.00001526,  208.00001526,  208.00001526,  208.00001526});

    nd4j::ops::fused_batch_norm op;
    auto results = op.execute({&x, &scale, &offset}, {0.05}, {0,1});
    auto y = results->at(0);
    auto batchMean = results->at(1);
    auto batchVar = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test3) {

    auto x = NDArrayFactory::create<double>('c', {2, 4, 2, 3});
    x.linspace(1);

    auto scale = NDArrayFactory::create<double>('c', {4});

    scale = 0.5;
    auto offset = NDArrayFactory::create<double>('c', {4});
    offset = 2.;
    auto expY = NDArrayFactory::create<double>('c', {2, 4, 2, 3}, {1.20337462,  1.20337462,  1.20337462,  1.20337462, 1.34821558,  1.34821558,  1.34821558,  1.34821558, 1.49305654,  1.49305654,  1.49305654,  1.49305654, 1.63789749,  1.63789749,  1.63789749,  1.63789749, 1.78273857,  1.78273857,  1.78273857,  1.78273857, 1.92757952,  1.92757952,  1.92757952,  1.92757952, 2.0724206 ,  2.0724206 ,  2.0724206 ,  2.0724206 , 2.21726155,  2.21726155,  2.21726155,  2.21726155, 2.36210251,  2.36210251,  2.36210251,  2.36210251, 2.50694346,  2.50694346,  2.50694346,  2.50694346, 2.65178442,  2.65178442,  2.65178442,  2.65178442, 2.79662538,  2.79662538,  2.79662538,  2.79662538});
    auto expBatchMean = NDArrayFactory::create<double>('c', {4}, {23.,  24.,  25.,  26.});
    auto expBatchVar = NDArrayFactory::create<double>('c', {4}, {208.00001526,  208.00001526,  208.00001526,  208.00001526});

    nd4j::ops::fused_batch_norm op;
    auto results = op.execute({&x, &scale, &offset}, {}, {1,1});
    auto y = results->at(0);
    auto batchMean = results->at(1);
    auto batchVar = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test4) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 3, 4});
    x.linspace(1);
    std::vector<Nd4jLong> shape = {4};
    auto scale = NDArrayFactory::create<double>('c', shape);
    auto offset = NDArrayFactory::create<double>('c', shape);
    auto mean = NDArrayFactory::create<double>('c', shape);
    auto variance = NDArrayFactory::create<double>('c', shape);

    scale = 0.5;
    offset = 2.;
    mean = 25.;
    variance = 5.;

    auto expY = NDArrayFactory::create<double>('c', {2, 2, 3, 4}, {-3.36602688, -3.14244223, -2.91885757, -2.6952734 , -2.47168875, -2.24810457, -2.02451992, -1.80093551, -1.57735109, -1.35376668, -1.13018227, -0.90659785, -0.68301344, -0.45942879, -0.23584437, -0.01225996, 0.21132445,  0.43490887,  0.65849328,  0.88207781, 1.10566223,  1.32924664,  1.55283117,  1.77641559, 2.        ,  2.22358441,  2.44716883,  2.67075348, 2.89433765,  3.11792231,  3.34150672,  3.56509113, 3.78867555,  4.01225996,  4.23584461,  4.45942879, 4.68301344,  4.90659809,  5.13018227,  5.35376644, 5.57735109,  5.80093575,  6.02451992,  6.24810457, 6.47168875,  6.6952734 ,  6.91885757,  7.14244223});
    auto expBatchMean = NDArrayFactory::create<double>('c', shape, {0.,  0.,  0.,  0.});
    auto expBatchVar = NDArrayFactory::create<double>('c', shape, {0.,  0.,  0.,  0.});


    nd4j::ops::fused_batch_norm op;
    auto results = op.execute({&x, &scale, &offset}, {}, {0,1});
    auto y = results->at(0);
    auto batchMean = results->at(1);
    auto batchVar = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test5) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 3, 4});
    x.linspace(1);
    std::vector<Nd4jLong> shape = {4};
    auto scale = NDArrayFactory::create<double>('c', shape);
    auto offset = NDArrayFactory::create<double>('c', shape);
    auto mean = NDArrayFactory::create<double>('c', shape);
    auto variance = NDArrayFactory::create<double>('c', shape);

    scale = 0.5;
    offset = 2.;
    mean = 25.;
    variance = 5.;

    auto expY = NDArrayFactory::create<double>('c', {2, 2, 3, 4}, {-3.33992958e+00,  -3.11743259e+00,  -2.89493513e+00,  -2.67243814e+00, -2.44994116e+00,  -2.22744417e+00,  -2.00494719e+00,  -1.78244996e+00, -1.55995297e+00,  -1.33745599e+00,  -1.11495876e+00,  -8.92461777e-01, -6.69964790e-01,  -4.47467566e-01,  -2.24970579e-01,  -2.47359276e-03, 2.20023513e-01,   4.42520618e-01,   6.65017605e-01,   8.87514710e-01, 1.11001182e+00,   1.33250880e+00,   1.55500591e+00,   1.77750289e+00, 2.00000000e+00,   2.22249699e+00,   2.44499421e+00,   2.66749120e+00, 2.88998818e+00,   3.11248541e+00,   3.33498240e+00,   3.55747938e+00, 3.77997637e+00,   4.00247383e+00,   4.22497082e+00,   4.44746780e+00, 4.66996479e+00,   4.89246178e+00,   5.11495876e+00,   5.33745575e+00, 5.55995274e+00,   5.78244972e+00,   6.00494719e+00,   6.22744417e+00, 6.44994116e+00,   6.67243814e+00,   6.89493513e+00,   7.11743259e+00});
    auto expBatchMean = NDArrayFactory::create<double>('c', shape, {0.,  0.,  0.,  0.});
    auto expBatchVar = NDArrayFactory::create<double>('c', shape, {0.,  0.,  0.,  0.});


    nd4j::ops::fused_batch_norm op;
    auto results = op.execute({&x, &scale, &offset}, {0.05}, {0,1});
    auto y = results->at(0);
    auto batchMean = results->at(1);
    auto batchVar = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, confusion_matrix_test1) {

    auto labels = NDArrayFactory::create<Nd4jLong>('c', {1, 3}, {1, 2, 4});
    auto predictions = NDArrayFactory::create<Nd4jLong>('c', {1, 3}, {2, 2, 4});
    auto expected = NDArrayFactory::create<Nd4jLong>('c', {5, 5}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1});

    nd4j::ops::confusion_matrix op;
    auto results = op.execute({&labels, &predictions}, {}, {});
    ASSERT_EQ(Status::OK(), results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, confusion_matrix_test2) {

    auto labels = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {1, 2});
    auto predictions = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {0, 2});
    auto expected = NDArrayFactory::create<Nd4jLong>('c', {3, 3}, {0, 0, 0, 1, 0, 0, 0, 0, 1});

    nd4j::ops::confusion_matrix op;
    auto results = op.execute({&labels, &predictions}, {}, {3});
    ASSERT_EQ(Status::OK(), results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, confusion_matrix_test3) {

    auto labels = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {1, 2});
    auto predictions = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {0, 2});
    auto weights = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {100, 200});
    auto expected = NDArrayFactory::create<Nd4jLong>('c', {3, 3}, {0, 0, 0, 100, 0, 0, 0, 0, 200});

    nd4j::ops::confusion_matrix op;
    auto results = op.execute({&labels, &predictions, &weights}, {}, {3});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, confusion_matrix_test4) {

    auto labels = NDArrayFactory::create<int>('c', {1, 2}, {1, 2});
    auto predictions = NDArrayFactory::create<int>('c', {1, 2}, {0, 2});
    auto weights = NDArrayFactory::create<double>('c', {1, 2}, {100, 200});
    auto expected = NDArrayFactory::create<double>('c', {3, 3}, {0, 0, 0, 100, 0, 0, 0, 0, 200});

    nd4j::ops::confusion_matrix op;
    auto results = op.execute({&labels, &predictions, &weights}, {}, {3, nd4j::DataType::DOUBLE});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ZeroFraction_1) {

    auto x = NDArrayFactory::create<double>('c', {3, 4, 2}, {0, 20, 30, 0, 50, 0,
                                      70, 0, 90, 0, 11, 12,
                                      13, 14, 15, 16, 17, 18,
                                      19, 0, 21, 22, 23, 24});

    nd4j::ops::zero_fraction op;
    auto res = op.execute({&x}, {}, {});

    ASSERT_EQ(Status::OK(), res->status());
    ASSERT_TRUE(res->at(0)->isScalar());
    ASSERT_EQ(res->at(0)->e<double>(0), 0.25);

    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ZeroFraction_2) {

    auto x = NDArrayFactory::create<double>('c', {2, 2, 2}, {5.5, 0., 0.3, 5.5, 8.6, 0., 0., 0.4});

    nd4j::ops::zero_fraction op;
    auto res = op.execute({&x}, {}, {});

    ASSERT_EQ(Status::OK(), res->status());
    ASSERT_TRUE(res->at(0)->isScalar());
    ASSERT_EQ(res->at(0)->e<double>(0), 0.375);

    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ZeroFraction_3) {

    auto x = NDArrayFactory::create<double>('f', {2, 2, 2}, {5.5, 0., 0.3, 5.5, 8.6, 0., 0., 0.4});

    nd4j::ops::zero_fraction op;
    auto res = op.execute({&x}, {}, {});

    ASSERT_EQ(Status::OK(), res->status());
    ASSERT_TRUE(res->at(0)->isScalar());
    ASSERT_EQ(res->at(0)->e<double>(0), 0.375);

    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, XWPlusB_1) {

    auto x = NDArrayFactory::create<double>('c', {2,3}, { 1.f, 11.f,  3.f, 14.f,  5.f,  6.f});
    auto y = NDArrayFactory::create<double>('c', {3,2}, { 11.f,  3.f, 4.f,  5.f, 6.f,  2.f});
    auto b = NDArrayFactory::create<double>({100.f, 200.f});

    auto exp = NDArrayFactory::create<double>('c', {2,2}, {173.f, 264.f, 310.f, 279.f});

    nd4j::ops::xw_plus_b op;
    auto result = op.execute({&x, &y, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto output = result->at(0);

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, StopGradient_1) {

    auto x = NDArrayFactory::create<double>('c', {2,3}, { 1.f, 11.f,  3.f, 14.f,  5.f,  6.f});

    nd4j::ops::stop_gradient op;
    auto result = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto output = result->at(0);

    // output->printShapeInfo("Output shape> ");
    // x.printShapeInfo("Expected shape> ");
    // output->printIndexedBuffer("Output data> ");
    // x.printIndexedBuffer("Expected res>");

    ASSERT_TRUE(x.isSameShape(output));
    ASSERT_TRUE(x.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, StopGradient_2) {

    auto x = NDArrayFactory::create<double>('f', {2,3}, { 1.f, 11.f,  3.f, 14.f,  5.f,  6.f});

    nd4j::ops::stop_gradient op;
    auto result = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::DOUBLE);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto output = result->at(0);

    // output->printShapeInfo("Output shape> ");
    // x.printShapeInfo("Expected shape> ");
    // output->printIndexedBuffer("Output data> ");
    // x.printIndexedBuffer("Expected res>");

    ASSERT_TRUE(x.isSameShape(output));
    ASSERT_TRUE(x.equalsTo(output));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test1) {

    auto input = NDArrayFactory::create<double>('c', {3, 3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3, 3}, {-2.16985e+00,-1.69846e-01,-3.16985e+00, -1.31507e+00,-6.31507e+00,-3.15072e-01, -8.00046e+00,-4.58767e-04,-9.00046e+00, -1.31327e+00,-1.23133e+01,-3.13266e-01, -1.40000e+01,-1.13743e-06,-1.50000e+01, -1.31326e+00,-1.83133e+01,-3.13262e-01, -2.00000e+01,-2.81941e-09,-2.10000e+01, -1.31326e+00,-2.43133e+01,-3.13262e-01, -2.73133e+01,-1.31326e+00,-3.13262e-01});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test2) {

    auto input = NDArrayFactory::create<double>('c', {3, 3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3, 3}, {-3.05095e+00,-3.04946e+00,-5.00705e+00, -5.09458e-02,-7.04946e+00,-7.04851e-03, -6.05095e+00,-4.94556e-02,-8.00705e+00, -3.04859e+00,-1.30000e+01,-3.04859e+00, -1.50486e+01,-2.37286e-06,-1.70486e+01, -4.85876e-02,-1.60000e+01,-4.85874e-02, -2.10000e+01,-3.04859e+00,-2.51269e+01, -7.96007e-10,-2.50486e+01,-2.12693e+00, -2.40000e+01,-4.85874e-02,-1.26928e-01});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {1}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test3) {

    auto input = NDArrayFactory::create<double>('c', {3, 3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3, 3}, {-2.16985e+00,-1.69846e-01,-3.16985e+00, -1.31507e+00,-6.31507e+00,-3.15072e-01, -8.00046e+00,-4.58767e-04,-9.00046e+00, -1.31327e+00,-1.23133e+01,-3.13266e-01, -1.40000e+01,-1.13743e-06,-1.50000e+01, -1.31326e+00,-1.83133e+01,-3.13262e-01, -2.00000e+01,-2.81941e-09,-2.10000e+01, -1.31326e+00,-2.43133e+01,-3.13262e-01, -2.73133e+01,-1.31326e+00,-3.13262e-01});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {2}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test5) {

    auto input = NDArrayFactory::create<double>('c', {3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, 5});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3}, {-2.16985, -0.16985, -3.16985, -1.31507, -6.31507, -0.31507, -9.31335, -1.31335, -0.31335});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test6) {

    auto input = NDArrayFactory::create<double>('c', {3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, 5});
    auto expOutput = NDArrayFactory::create<double>('c', {3, 3}, {-3.05095,-3.04946,-7.12773, -0.05095,-7.04946,-2.12773, -6.05095,-0.04946,-0.12773});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test7) {

    auto input = NDArrayFactory::create<double>('c', {1, 5}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {1, 5}, {-4.42414, -2.42414, -5.42414, -1.42414, -0.42414});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test8) {

    auto input = NDArrayFactory::create<double>('c', {1, 5}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {1, 5}, {0, 0, 0, 0, 0});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test9) {

    auto input = NDArrayFactory::create<double>('c', {5, 1}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {5, 1}, {0, 0, 0, 0, 0});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test10) {

    auto input = NDArrayFactory::create<double>('c', {5, 1}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {5, 1}, {-4.42414, -2.42414, -5.42414, -1.42414, -0.42414});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test11) {

    auto input = NDArrayFactory::create<double>('c', {5}, {-1, 1, -2, 2, 3});
    auto expOutput = NDArrayFactory::create<double>('c', {5}, {-4.42414, -2.42414, -5.42414, -1.42414, -0.42414});

    nd4j::ops::log_softmax op;
    auto  results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto z = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test12) {

    auto input = NDArrayFactory::create<double>('c', {1, 4}, {0.1869,   -1.4918,   -0.6497,   -0.8864});
    auto expOutput = NDArrayFactory::create<double>('c', {1, 4}, {-0.6738,   -2.3525,   -1.5104,   -1.7472});

    for (int i = 0; i < 10; ++i)
    {
        nd4j::ops::log_softmax op;
        auto  results = op.execute({&input}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
        auto z = results->at(0);

        ASSERT_EQ(Status::OK(), results->status());
        ASSERT_TRUE(expOutput.isSameShape(z));
        ASSERT_TRUE(expOutput.equalsTo(z, 1e-4));

        delete results;
    }
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_bp_test1) {

    auto input   = NDArrayFactory::create<double>('c', {2, 2}, {1,2,3,4});
    auto epsilon = NDArrayFactory::create<double>('c', {2, 2}, {0.1, 0.2, 0.3, 0.4});
    auto exp = NDArrayFactory::create<double>('c', {2, 2}, {-0.07311,0.02689, -0.07311,0.02689});

    nd4j::ops::log_softmax_bp op;
    auto  results = op.execute({&input, &epsilon}, {}, {}, {}, false, nd4j::DataType::DOUBLE);
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_bp_test2) {

    auto input   = NDArrayFactory::create<double>('c', {2, 2}, {1,2,3,4});
    auto epsilon = NDArrayFactory::create<double>('c', {2, 2}, {0.1, 0.2, 0.3, 0.4});
    auto exp = NDArrayFactory::create<double>('c', {2, 2}, {-0.17616, -0.17616, 0.02384,  0.02384});

    nd4j::ops::log_softmax_bp op;
    auto  results = op.execute({&input, &epsilon}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ELU_1) {

    auto input   = NDArrayFactory::create<double>('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    auto exp     = NDArrayFactory::create<double>('c', {2, 2, 2}, { -0.63212055,  2. , 1.5, -0.753403, 1.,   2.,  2.,   1.});
    auto res     = NDArrayFactory::create<double>('c', {2, 2, 2});

    input.applyScalar(nd4j::scalar::ELU, 1.f, &res);

    ASSERT_TRUE(res.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, L2_Loss_1) {

    auto input   = NDArrayFactory::create<double>('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    double exp(9.605);

    nd4j::ops::l2_loss op;
    auto results = op.execute({&input}, {}, {});
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(output->isScalar());

    ASSERT_EQ(output->e<double>(0), exp);

    delete results;
}

TEST_F(DeclarableOpsTests5, L2_Loss_2) {
    auto x = NDArrayFactory::create<double>(0.7787855863571167);
    auto e = NDArrayFactory::create<double>(0.303254);

    nd4j::ops::l2_loss op;
    auto results = op.execute({&x}, {}, {});
    ASSERT_EQ(Status::OK(), results->status());

    auto z = results->at(0);

    ASSERT_EQ(e, *z);

    delete results;
}

TEST_F(DeclarableOpsTests5, L2_Loss_3) {
    auto x = NDArrayFactory::create<double>(0.7787855863571167);
    auto e = NDArrayFactory::create<double>(0.303254);
    auto z = NDArrayFactory::create<double>(0.0);

    nd4j::ops::l2_loss op;
    auto status = op.execute({&x}, {&z} , {}, {}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(e, z);
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, LogPoissonLoss_1) {
    auto weights = NDArrayFactory::create<double>('c', {1, 1}, {1});

    auto input   = NDArrayFactory::create<double>('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    auto targets = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

    auto exp = NDArrayFactory::create<double>('c', {2, 2, 2}, {1.3678794, 5.389056, 2.981689, 1.6465969, 1.7182817, 5.389056, 5.389056, 1.7182817});

    nd4j::ops::log_poisson_loss op;
    auto results = op.execute({&input, &weights, &targets}, {}, {0}, {}, false, nd4j::DataType::DOUBLE);
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, LogPoissonLoss_2) {

    auto weights = NDArrayFactory::create<double>('c', {1, 1}, {1});

    auto input   = NDArrayFactory::create<double>('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    auto targets = NDArrayFactory::create<double>('c', {2, 2, 2}, {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});

    auto exp = NDArrayFactory::create<double>('c', {2, 2, 2}, {3.0196857, 4.0408626, 2.1334953, 3.6984034, 1.3700882, 4.0408626, 4.0408626, 1.3700882});

    nd4j::ops::log_poisson_loss op;
    auto results = op.execute({&input, &weights, &targets}, {}, {0, 1}, {}, false, nd4j::DataType::DOUBLE);
    auto output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, NormalizeMoments_1) {

    auto means   = NDArrayFactory::create<double>('c', {2, 3, 4}, { 11.,   3.,  14.,   5.,
                                               6.,   9.,  3.5,   7.,
                                              21.,   3.,  14.,  15.,
                                               6.,   9.,  3.5,   7.,
                                              11.,  13.,  14.,   5.,
                                              16.,   9., 13.5,   7.});

    auto deviance = NDArrayFactory::create<double>('c', {2, 3, 4}, { 21.,  13.,  24.,  15.,
                                               16.,  19., 13.5,  17.,
                                               31.,  13.,  24.,  25.,
                                               16.,  19., 13.5,  17.,
                                               21.,  23.,  24.,  15.,
                                               26.,  19., 23.5,  17.});

    auto counts = NDArrayFactory::create<double>(2.0);

    auto expMeans = NDArrayFactory::create<double>('c', {2, 3, 4}, {
                                                 5.5,   1.5,     7.,  2.5,
                                                  3.,   4.5,   1.75,  3.5,
                                                10.5,   1.5,     7.,  7.5,
                                                  3.,   4.5,   1.75,  3.5,
                                                 5.5,   6.5,     7.,  2.5,
                                                  8.,   4.5,   6.75,  3.5});

    auto expDeviance = NDArrayFactory::create<double>('c', {2, 3, 4}, {
                                                -19.75,     4.25,       -37.,   1.25,
                                                   -1.,   -10.75,     3.6875,  -3.75,
                                                -94.75,     4.25,       -37.,  -43.75,
                                                   -1.,   -10.75,     3.6875,  -3.75,
                                                -19.75,   -30.75,       -37.,   1.25,
                                                  -51.,   -10.75,   -33.8125,  -3.75});

    nd4j::ops::normalize_moments op;
    auto results = op.execute({&counts, &means, &deviance}, {0.0}, {});

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_EQ(results->size(), 2);

    auto outputMeans = results->at(0);
    auto outputDeviance = results->at(1);

    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expMeans.isSameShape(outputDeviance));
    ASSERT_TRUE(expDeviance.equalsTo(outputDeviance));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, NormalizeMoments_2) {

    auto means   = NDArrayFactory::create<double>('c', {3, 2, 4}, { 11.,   3.,  14.,   5.,
                                               6.,   9.,  3.5,   7.,
                                              21.,   3.,  14.,  15.,
                                               6.,   9.,  3.5,   7.,
                                              11.,  13.,  14.,   5.,
                                              16.,   9., 13.5,   7.});

    auto deviance = NDArrayFactory::create<double>('c', {3, 2, 4}, { 21.,  13.,  24.,  15.,
                                               16.,  19., 13.5,  17.,
                                               31.,  13.,  24.,  25.,
                                               16.,  19., 13.5,  17.,
                                               21.,  23.,  24.,  15.,
                                               26.,  19., 23.5,  17.});

    auto counts = NDArrayFactory::create<double>(12.0);

    auto expMeans = NDArrayFactory::create<double>('c', {3, 2, 4}, { 0.9166667,     0.25,  1.1666667, 0.4166667,
                                                     0.5,     0.75,  0.2916667, 0.5833334,
                                                    1.75,     0.25,  1.1666667,      1.25,
                                                     0.5,     0.75,  0.2916667, 0.5833334,
                                               0.9166667, 1.0833334, 1.1666667, 0.4166667,
                                               1.3333334,      0.75,     1.125, 0.5833334});

    auto expDeviance = NDArrayFactory::create<double>('c', {3, 2, 4}, {
                                                 0.9097222,  1.0208334,  0.6388887,  1.0763888,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                -0.4791665,  1.0208334,  0.6388887,  0.5208335,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                 0.9097222,  0.7430556,  0.6388887,  1.0763888,
                                                0.38888884,  1.0208334,  0.6927084,   1.076389});

    nd4j::ops::normalize_moments op;
    auto results = op.execute({&counts, &means, &deviance}, {0.0}, {});

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_EQ(results->size(), 2);

    auto outputMeans = results->at(0);
    auto outputDeviance = results->at(1);

    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expMeans.isSameShape(outputDeviance));
    ASSERT_TRUE(expDeviance.equalsTo(outputDeviance));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, NormalizeMoments_3) {

    auto means   = NDArrayFactory::create<double>('c', {3, 2, 4}, { 11.,   3.,  14.,   5.,
                                               6.,   9.,  3.5,   7.,
                                              21.,   3.,  14.,  15.,
                                               6.,   9.,  3.5,   7.,
                                              11.,  13.,  14.,   5.,
                                              16.,   9., 13.5,   7.});

    auto deviance = NDArrayFactory::create<double>('c', {3, 2, 4}, { 21.,  13.,  24.,  15.,
                                               16.,  19., 13.5,  17.,
                                               31.,  13.,  24.,  25.,
                                               16.,  19., 13.5,  17.,
                                               21.,  23.,  24.,  15.,
                                               26.,  19., 23.5,  17.});

    auto counts = NDArrayFactory::create<double>(12.0);
    double shift = 10.0;
    auto expMeans = NDArrayFactory::create<double>('c', {3, 2, 4}, { 10.9166667,     10.25,  11.1666667, 10.4166667,
                                                     10.5,     10.75,  10.2916667, 10.5833334,
                                                    11.75,     10.25,  11.1666667,      11.25,
                                                     10.5,     10.75,  10.2916667, 10.5833334,
                                               10.9166667, 11.0833334, 11.1666667, 10.4166667,
                                               11.3333334,      10.75,     11.125, 10.5833334});

    auto expDeviance = NDArrayFactory::create<double>('c', {3, 2, 4}, {
                                                 0.9097222,  1.0208334,  0.6388887,  1.0763888,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                -0.4791665,  1.0208334,  0.6388887,  0.5208335,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                 0.9097222,  0.7430556,  0.6388887,  1.0763888,
                                                0.38888884,  1.0208334,  0.6927084,   1.076389});

    nd4j::ops::normalize_moments op;
    auto results = op.execute({&counts, &means, &deviance}, {shift}, {});

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_EQ(results->size(), 2);

    auto outputMeans = results->at(0);
    auto outputDeviance = results->at(1);

    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));
    ASSERT_TRUE(expMeans.isSameShape(outputDeviance));
    ASSERT_TRUE(expDeviance.equalsTo(outputDeviance));

    delete results;
}

