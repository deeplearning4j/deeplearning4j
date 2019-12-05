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

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>
#include <MmulHelper.h>
#include <PointersManager.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests3 : public testing::Test {
public:

    DeclarableOpsTests3() {
//
    }
};


TEST_F(DeclarableOpsTests3, Test_Tile_1) {
    auto x= NDArrayFactory::create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto rep_vector= NDArrayFactory::create<int>('c', {1, 2}, {2, 2});
    std::vector<Nd4jLong> reps({2, 2});

    auto exp = x.tile(reps);

    nd4j::ops::tile op;
    auto result = op.execute({&x, &rep_vector}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Tile_2) {
    auto x= NDArrayFactory::create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    std::vector<Nd4jLong> reps({2, 2});

    auto exp = x.tile(reps);

    nd4j::ops::tile op;
    auto result = op.execute({&x}, {}, {2, 2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Permute_1) {
    auto x= NDArrayFactory::create<float>('c', {2, 3, 4});
    auto permute= NDArrayFactory::create<Nd4jLong>('c', {1, 3}, {0, 2, 1});
    auto exp= NDArrayFactory::create<float>('c', {2, 4, 3});

    nd4j::ops::permute op;
    auto result = op.execute({&x, &permute}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Permute_2) {
    auto x= NDArrayFactory::create<float>('c', {2, 3, 4});
    auto exp= NDArrayFactory::create<float>('c', {4, 3, 2});

    nd4j::ops::permute op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Unique_1) {
    auto x= NDArrayFactory::create<float>('c', {1, 5}, {1.f, 2.f, 1.f, 2.f, 3.f});
    auto expV= NDArrayFactory::create<float>('c', {3}, {1.f, 2.f, 3.f});
    auto expI= NDArrayFactory::create<Nd4jLong>('c', {5}, {0, 1, 0, 1, 2});
//    auto expI= NDArrayFactory::create<float>('c', {3}, {0, 1, 4});

    nd4j::ops::unique op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto i = result->at(1);
    // v->printIndexedBuffer("Values");
    // i->printIndexedBuffer("Indices");
    // i->printShapeInfo("Indices shape");
    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Unique_2) {
    auto x= NDArrayFactory::create<float>('c', {1, 5}, {1.f, 2.f, 1.f, 2.f, 3.f});
    auto expV= NDArrayFactory::create<float>('c', {3}, {1.f, 2.f, 3.f});
    auto expI= NDArrayFactory::create<Nd4jLong>('c', {5}, {0, 1, 0, 1, 2});
    auto expC= NDArrayFactory::create<Nd4jLong>('c', {3}, {2, 2, 1});

    nd4j::ops::unique_with_counts op;
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(3, result->size());

    auto v = result->at(0);
    auto i = result->at(1);
    auto c = result->at(2);

     // v->printShapeInfo();
     // v->printIndexedBuffer("Values");
     // i->printShapeInfo();
     // i->printIndexedBuffer("Indices");
     // c->printShapeInfo();
     // c->printIndexedBuffer("Counts");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    ASSERT_TRUE(expC.isSameShape(c));
    ASSERT_TRUE(expC.equalsTo(c));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Rint_1) {
    auto x= NDArrayFactory::create<float>('c', {1, 7}, {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
    auto exp= NDArrayFactory::create<float>('c', {1, 7}, {-2.f, -2.f, -0.f, 0.f, 2.f, 2.f, 2.f});

    nd4j::ops::rint op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Norm_1) {
    auto x = NDArrayFactory::create<float>('c', {100, 100});
    x.linspace(1);

    std::vector<int> empty;
    std::vector<int> dims({1});
    nd4j::ops::norm op;

    auto result0 = op.execute({&x}, {0.}, {});

    auto z0 = result0->at(0);
    auto exp0 = x.reduceAlongDims(reduce::NormFrobenius, empty, false, false);
    ASSERT_TRUE(exp0.isSameShape(z0));
    ASSERT_TRUE(exp0.equalsTo(z0));

    delete result0;

    auto result1 = op.execute({&x}, {1.}, {1});
    ASSERT_EQ(result1->status(), ND4J_STATUS_OK);
    auto z1 = result1->at(0);
    // z1->printIndexedBuffer("Z1");
    auto exp1 = x.reduceAlongDims(reduce::Norm2, dims, false, false);
    // exp1.printIndexedBuffer("EXP1");
    // z1->printShapeInfo("Z1 shape");
    // exp1.printShapeInfo("EXP1 shape");
    ASSERT_TRUE(exp1.isSameShape(z1));
    ASSERT_TRUE(exp1.equalsTo(z1));

    delete result1;

    auto result4 = op.execute({&x}, {4.}, {1});

    auto z4 = result4->at(0);
    auto exp4= x.reduceAlongDims(reduce::NormMax, dims, false, false);
    ASSERT_TRUE(exp4.isSameShape(z4));
    ASSERT_TRUE(exp4.equalsTo(z4));

    delete result4;
}


TEST_F(DeclarableOpsTests3, Test_Norm_2) {
    auto x = NDArrayFactory::create<float>('c', {100, 100});
    x.linspace(1);
    auto axis= NDArrayFactory::create<Nd4jLong>('c', {1, 1}, {1});

    std::vector<int> empty;
    std::vector<int> dims({1});
    nd4j::ops::norm op;

    auto result0 = op.execute({&x}, {0}, {});

    auto z0 = result0->at(0);
    auto exp0 = x.reduceAlongDims(reduce::NormFrobenius, empty, false, false);
    ASSERT_TRUE(exp0.isSameShape(z0));
    ASSERT_TRUE(exp0.equalsTo(z0));

    delete result0;

    auto result1 = op.execute({&x, &axis}, {1}, {});

    auto z1 = result1->at(0);
    auto exp1 = x.reduceAlongDims(reduce::Norm2, dims, false, false);
    ASSERT_TRUE(exp1.isSameShape(z1));
    ASSERT_TRUE(exp1.equalsTo(z1));

    delete result1;

    auto result4 = op.execute({&x, &axis}, {4}, {});

    auto z4 = result4->at(0);
    auto exp4= x.reduceAlongDims(reduce::NormMax, dims, false, false);
    ASSERT_TRUE(exp4.isSameShape(z4));
    ASSERT_TRUE(exp4.equalsTo(z4));

    delete result4;
}


TEST_F(DeclarableOpsTests3, Test_ClipByAvgNorm_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    auto exp = NDArrayFactory::create<double>('c', {2, 3}, {-2.88, 0.0, 0.0, 3.84, 0.0, 0.0});

    nd4j::ops::clipbyavgnorm op;
    auto result = op.execute({&x}, {0.8}, {}, {}, false, nd4j::DataType::DOUBLE);

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_ClipByAvgNorm_2) {
    auto x= NDArrayFactory::create<float>('c', {2, 3}, {-3.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f});
    auto exp= NDArrayFactory::create<float>('c', {2, 3}, {-3.f, 0.0f, 0.0f, 4.f, 0.0f, 0.0f});

    nd4j::ops::clipbyavgnorm op;
    auto result = op.execute({&x}, {0.9}, {});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_ClipByNorm_1) {
    auto x= NDArrayFactory::create<double>('c', {2, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    auto exp= NDArrayFactory::create<double>('c', {2, 3}, {-2.4, 0.0, 0.0, 3.2, 0.0, 0.0});

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {4.0}, {});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_ClipByNorm_2) {
    auto x= NDArrayFactory::create<double>('c', {2, 3}, {-3.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f});
    auto exp= NDArrayFactory::create<double>('c', {2, 3}, {-3.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f});

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {6.0}, {});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, Test_ClipByNorm_3) {

    auto x = NDArrayFactory::create<double>('c', {3, 5});
    auto unities = NDArrayFactory::create<double>('c', {3, 1}, {1., 1., 1.});
    auto scale = NDArrayFactory::create<double>('c', {3, 1}, {1.1, 1., 0.9});

    x.linspace(100.);

    auto xNorm1 = x.reduceAlongDims(reduce::Norm2, {1}, true);
    x /= xNorm1;
    xNorm1 = x.reduceAlongDims(reduce::Norm2,{1}, true);

    ASSERT_TRUE(unities.isSameShape(xNorm1));
    ASSERT_TRUE(unities.equalsTo(xNorm1));

    x *= scale;
    xNorm1 = x.reduceAlongDims(reduce::Norm2, {1}, true);

    nd4j::ops::clipbynorm op;
    auto result = op.execute({&x}, {1.0}, {1}, {}, false, nd4j::DataType::DOUBLE);
    auto z = result->at(0);

    auto zNorm1 = z->reduceAlongDims(reduce::Norm2, {1}, true);
    auto exp = NDArrayFactory::create<double>('c', {3, 1}, {1., 1., xNorm1.e<double>(2)});

    ASSERT_TRUE(exp.isSameShape(&zNorm1));
    ASSERT_TRUE(exp.equalsTo(&zNorm1));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_ListDiff_1) {
    auto x= NDArrayFactory::create<float>('c', {6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto y= NDArrayFactory::create<float>('c', {3}, {1.f, 3.f, 5.f});

    auto exp0= NDArrayFactory::create<float>('c', {3}, {2.f, 4.f, 6.f});
    auto exp1= NDArrayFactory::create<Nd4jLong>('c', {3}, {1, 3, 5});

    nd4j::ops::listdiff op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(Status::OK(), result->status());

    auto z0 = result->at(0);
    auto z1 = result->at(1);

    z0->getDataBuffer()->syncToSpecial(true);   // force sync
    z1->getDataBuffer()->syncToSpecial(true);   // force sync

    ASSERT_TRUE(exp0.isSameShape(z0));
    ASSERT_TRUE(exp0.equalsTo(z0));

    ASSERT_TRUE(exp1.isSameShape(z1));
    ASSERT_TRUE(exp1.equalsTo(z1));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Range_1) {
    auto start = NDArrayFactory::create<float>(0.3f);
    auto stop = NDArrayFactory::create<float>(-5.f);
    auto step = NDArrayFactory::create<float>(-0.33f);
    auto exp= NDArrayFactory::create<float>('c', {17}, { 0.3f, -0.03f, -0.36f, -0.69f, -1.02f, -1.35f, -1.68f, -2.01f, -2.34f, -2.67f, -3.f, -3.33f, -3.66f, -3.99f, -4.32f, -4.65f, -4.98f});

    nd4j::ops::range op;
    auto result = op.execute({&start, &stop, &step}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Range_2) {
    auto start= NDArrayFactory::create<float>('c', {1, 1}, {2.f});
    auto stop= NDArrayFactory::create<float>('c', {1, 1}, {0.f});
    auto step= NDArrayFactory::create<float>('c', {1, 1}, {-1.f});
    auto exp= NDArrayFactory::create<float>('c', {2}, {2.f, 1.f});

    nd4j::ops::range op;
    auto result = op.execute({&start, &stop, &step}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Range_3) {
    auto start= NDArrayFactory::create<float>('c', {1, 1}, {0.f});
    auto stop= NDArrayFactory::create<float>('c', {1, 1}, {2.f});
    auto step= NDArrayFactory::create<float>('c', {1, 1}, {1.f});
    auto exp= NDArrayFactory::create<float>('c', {2}, {0.f, 1.f});

    nd4j::ops::range op;
    auto result = op.execute({&start, &stop, &step}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Range_4) {
    auto exp= NDArrayFactory::create<float>('c', {13}, {-10.f,  -8.334f,  -6.668f,  -5.002f,  -3.336f,  -1.67f,  -0.004f,   1.662f,   3.328f,   4.994f,   6.66f,   8.326f,   9.992f});

    nd4j::ops::range op;
    auto result = op.execute({}, {-10., 10., 1.666}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Range_5) {
    auto exp= NDArrayFactory::create<float>('c', {2}, {2.f, 1.f});

    nd4j::ops::range op;
    auto result = op.execute({}, {2, 0, -1}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Range_6) {
    auto exp= NDArrayFactory::create<float>('c', {2}, {0.f, 1.f});

    nd4j::ops::range op;
    auto result = op.execute({}, {0, 2, 1}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Range_7) {
    auto exp= NDArrayFactory::create<float>('c', {10}, {10.f,  8.334f,  6.668f,  5.002f,  3.336f,  1.67f,  0.004f, -1.662f, -3.328f, -4.994f});

    nd4j::ops::range op;
    auto result = op.execute({}, {10,-5,-1.666}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}



TEST_F(DeclarableOpsTests3, Test_Range_8) {
    auto exp= NDArrayFactory::create<int>('c', {2}, {2, 1});

    nd4j::ops::range op;
    auto result = op.execute({}, {}, {2, 0, -1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Range_9) {
    auto exp= NDArrayFactory::create<int>('c', {2}, {0, 1});

    nd4j::ops::range op;
    auto result = op.execute({}, {}, {0, 2, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Batched_Gemm_1) {
    auto a= NDArrayFactory::create<double>('c', {1, 3}, {1, 1, 1});
    auto b= NDArrayFactory::create<double>('c', {1, 3}, {0, 0, 0});
    auto x= NDArrayFactory::create<double>('f', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto y= NDArrayFactory::create<double>('f', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto exp = MmulHelper::mmul(&x, &y);

    nd4j::ops::batched_gemm op;
    auto result = op.execute({&a, &b, &x, &x, &x, &y, &y, &y}, {}, {111, 111, 3, 3, 3, 3, 3, 3, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, result->size());

    for (int e = 0; e < 3; e++) {
        auto z = result->at(e);

//        exp->printIndexedBuffer("e");
//        z->printIndexedBuffer("z");

        ASSERT_TRUE(exp->isSameShape(z));
        ASSERT_TRUE(exp->equalsTo(z));
    }

    delete exp;
    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Batched_Gemm_2) {
    auto a= NDArrayFactory::create<double>('c', {1, 3}, {1, 1, 1});
    auto b= NDArrayFactory::create<double>('c', {1, 3}, {0, 0, 0});
    auto x= NDArrayFactory::create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto y= NDArrayFactory::create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto exp = MmulHelper::mmul(&x, &y);

    nd4j::ops::batched_gemm op;
    auto result = op.execute({&a, &b, &x, &x, &x, &y, &y, &y}, {}, {112, 112, 3, 3, 3, 3, 3, 3, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, result->size());

    for (int e = 0; e < 3; e++) {
        auto z = result->at(e);

        //exp->printIndexedBuffer("e");
        //z->printIndexedBuffer("z");

        ASSERT_TRUE(exp->isSameShape(z));
        ASSERT_TRUE(exp->equalsTo(z));
    }

    delete exp;
    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Batched_Gemm_3) {
    auto a= NDArrayFactory::create<double>('c', {1, 3}, {1, 1, 1});
    auto b= NDArrayFactory::create<double>('c', {1, 3}, {0, 0, 0});
    auto x= NDArrayFactory::create<double>('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto y= NDArrayFactory::create<double>('f', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto exp = MmulHelper::mmul(&x, &y);

    nd4j::ops::batched_gemm op;
    auto result = op.execute({&a, &b, &x, &x, &x, &y, &y, &y}, {}, {112, 111, 3, 3, 3, 3, 3, 3, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, result->size());

    for (int e = 0; e < 3; e++) {
        auto z = result->at(e);

//        exp->printIndexedBuffer("e");
//        z->printIndexedBuffer("z");

        ASSERT_TRUE(exp->isSameShape(z));
        ASSERT_TRUE(exp->equalsTo(z));
    }

    delete exp;
    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Batched_Gemm_4) {
    auto a= NDArrayFactory::create<double>('c', {1, 3}, {1, 1, 1});
    auto b= NDArrayFactory::create<double>('c', {1, 3}, {0, 0, 0});
    auto x= NDArrayFactory::create<double>('f', {5, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto y= NDArrayFactory::create<double>('f', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    auto exp = MmulHelper::mmul(&x, &y);

    nd4j::ops::batched_gemm op;
    auto result = op.execute({&a, &b, &x, &x, &x, &y, &y, &y}, {}, {111, 111, 5, 4, 3, 5, 3, 5, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, result->size());

    for (int e = 0; e < 3; e++) {
        auto z = result->at(e);

        //exp->printIndexedBuffer("e");
        //z->printIndexedBuffer("z");

        ASSERT_TRUE(exp->isSameShape(z));
        ASSERT_TRUE(exp->equalsTo(z));
    }

    delete exp;
    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Batched_Gemm_5) {
    auto a= NDArrayFactory::create<double>('c', {1, 3}, {1, 1, 1});
    auto b= NDArrayFactory::create<double>('c', {1, 3}, {0, 0, 0});
    auto x= NDArrayFactory::create<double>('c', {5, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto y= NDArrayFactory::create<double>('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    auto exp = MmulHelper::mmul(&x, &y);

    nd4j::ops::batched_gemm op;
    auto result = op.execute({&a, &b, &x, &x, &x, &y, &y, &y}, {}, {112, 112, 5, 4, 3, 3, 4, 5, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, result->size());

    for (int e = 0; e < 3; e++) {
        auto z = result->at(e);

        //exp->printIndexedBuffer("e");
        //z->printIndexedBuffer("z");

        ASSERT_TRUE(exp->isSameShape(z));
        ASSERT_TRUE(exp->equalsTo(z));
    }

    delete exp;
    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Batched_Gemm_6) {
    auto a= NDArrayFactory::create<double>('c', {1, 3}, {1, 1, 1});
    auto b= NDArrayFactory::create<double>('c', {1, 3}, {0, 0, 0});
    auto x= NDArrayFactory::create<double>('f', {2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto y= NDArrayFactory::create<double>('f', {5, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

    auto exp = MmulHelper::mmul(&x, &y);

    nd4j::ops::batched_gemm op;
    auto result = op.execute({&a, &b, &x, &x, &x, &y, &y, &y}, {}, {111, 111, 2, 3, 5, 2, 5, 2, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, result->size());

    for (int e = 0; e < 3; e++) {
        auto z = result->at(e);

        //exp->printIndexedBuffer("e");
        //z->printIndexedBuffer("z");

        ASSERT_TRUE(exp->isSameShape(z));
        ASSERT_TRUE(exp->equalsTo(z));
    }

    delete exp;
    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Batched_Gemm_7) {
    auto a= NDArrayFactory::create<double>('c', {1, 3}, {1, 1, 1});
    auto b= NDArrayFactory::create<double>('c', {1, 3}, {0, 0, 0});
    auto x= NDArrayFactory::create<double>('c', {2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto y= NDArrayFactory::create<double>('c', {5, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

    auto exp = MmulHelper::mmul(&x, &y);

    // exp->printShapeInfo("exp shape");

    nd4j::ops::batched_gemm op;
    auto result = op.execute({&a, &b, &x, &x, &x, &y, &y, &y}, {}, {112, 112, 2, 3, 5, 5, 3, 2, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, result->size());

    for (int e = 0; e < 3; e++) {
        auto z = result->at(e);

        //exp->printIndexedBuffer("e");
        //z->printIndexedBuffer("z");

        ASSERT_TRUE(exp->isSameShape(z));
        ASSERT_TRUE(exp->equalsTo(z));
    }

    delete exp;
    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Batched_Gemm_Validation_1) {
    auto a = NDArrayFactory::create<float>('c', {1, 3}, {1.f, 1.f, 1.f});
    auto b = NDArrayFactory::create<double>('c', {1, 3}, {0.f, 0.f, 0.f});
    auto x = NDArrayFactory::create<float16>('c', {2, 5}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});
    auto y = NDArrayFactory::create<float>('c', {5, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});

    nd4j::ops::batched_gemm op;
    try {
        auto result = op.execute({&a, &b, &x, &x, &x, &y, &y, &y}, {}, {112, 112, 2, 3, 5, 5, 3, 2, 3});
        delete result;
        ASSERT_TRUE(false);
    } catch (std::invalid_argument &e) {
        //
    }
}

TEST_F(DeclarableOpsTests3, Test_Batched_Gemm_Validation_2) {
    auto a = NDArrayFactory::create<float>('c', {1, 3}, {1.f, 1.f, 1.f});
    auto b = NDArrayFactory::create<double>('c', {1, 3}, {0.f, 0.f, 0.f});
    auto x = NDArrayFactory::create<float>('c', {2, 5}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f});
    auto y = NDArrayFactory::create<float>('c', {5, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});

    auto z = NDArrayFactory::create<double>('c', {2, 3});

    nd4j::ops::batched_gemm op;
    try {
        auto result = op.execute({&a, &b, &x, &x, &x, &y, &y, &y}, {&z}, {}, {112, 112, 2, 3, 5, 5, 3, 2, 3}, {});
        ASSERT_TRUE(false);
    } catch (std::invalid_argument &e) {
        //
    }
}

TEST_F(DeclarableOpsTests3, Test_Manual_Gemm_1) {
    auto x= NDArrayFactory::create<double>('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12});
    auto y= NDArrayFactory::create<double>('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12});
    auto exp= NDArrayFactory::create<double>('f', {4, 4}, {38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0, 128.0, 152.0, 176.0, 200.0, 173.0, 206.0, 239.0, 272.0});

    nd4j::ops::matmul op;
    auto result = op.execute({&x, &y}, {}, {1, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Manual_Gemm_2) {
    auto x= NDArrayFactory::create<double>('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12});
    auto y= NDArrayFactory::create<double>('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12});
    auto exp= NDArrayFactory::create<double>('f', {3, 3}, {70.0, 158.0, 246.0, 80.0, 184.0, 288.0, 90.0, 210.0, 330.0});

    nd4j::ops::matmul op;
    auto result = op.execute({&x, &y}, {}, {0, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Manual_Gemm_3) {
    auto x= NDArrayFactory::create<double>('c', {1, 3}, {1, 2, 3});
    auto y= NDArrayFactory::create<double>('c', {1, 4}, {1, 2, 3, 4});
    auto exp= NDArrayFactory::create<double>('f', {3, 4}, {1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0});

    nd4j::ops::matmul op;
    auto result = op.execute({&x, &y}, {}, {1, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Manual_Gemm_4) {
    auto x= NDArrayFactory::create<double>('c', {3, 1}, {1, 2, 3});
    auto y= NDArrayFactory::create<double>('c', {4, 1}, {1, 2, 3, 4});
    auto exp= NDArrayFactory::create<double>('f', {3, 4}, {1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0});

    nd4j::ops::matmul op;
    auto result = op.execute({&x, &y}, {}, {0, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Manual_Gemm_5) {
    auto x= NDArrayFactory::create<double>('c', {3, 1}, {1, 2, 3});
    auto y= NDArrayFactory::create<double>('c', {1, 4}, {1, 2, 3, 4});
    auto exp= NDArrayFactory::create<double>('f', {3, 4}, {1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0});

    nd4j::ops::matmul op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Manual_Gemm_6) {
    auto x= NDArrayFactory::create<double>('c', {4, 1}, {1, 2, 3, 4});
    auto y= NDArrayFactory::create<double>('c', {1, 4}, {1, 2, 3, 4});
    auto exp= NDArrayFactory::create<double>('f', {4, 4}, {1,2, 3, 4,2,4, 6, 8,3,6, 9,12,4,8,12,16});

    nd4j::ops::matmul op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_AvgPool_1) {
    auto x= NDArrayFactory::create<float>('c', {2, 10, 10, 3});
    x.linspace(1);

    nd4j::ops::avgpool2d op;
    //                                  kY  kX  sY  sX  pY  pX  dY  dX  M   P
    auto result = op.execute({&x}, {}, {3,  3,  3,  3,  0,  0,  1,  1,  1,  0,  1});
    //                                  0   1   2   3   4   5   6   7   8   9   10
    auto z = result->at(0);

    // z->printShapeInfo("z shape");
    // z->printIndexedBuffer("z buffr");

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_ReverseDivide_1) {
    auto x= NDArrayFactory::create<double>('c', {1, 3}, {2, 2, 2});
    auto y= NDArrayFactory::create<double>('c', {1, 3}, {4, 6, 8});
    auto exp= NDArrayFactory::create<double>('c', {1, 3}, {2, 3, 4});

    nd4j::ops::reversedivide op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, sruCell_test1) {

    const int batchSize = 2;
    const int inSize    = 5;

    auto xt  = NDArrayFactory::create<float>('c', {batchSize, inSize});
    auto ct_1= NDArrayFactory::create<float>('c', {batchSize, inSize});
    auto w   = NDArrayFactory::create<float>('c', {inSize, 3*inSize});
    auto b   = NDArrayFactory::create<float>('c', {2*inSize});

    xt.assign(1.);
    ct_1.assign(2.);
    w.assign(0.5);
    b.assign(0.7);

    auto expHt= NDArrayFactory::create<float>('c', {batchSize, inSize}, {0.96674103f, 0.96674103f, 0.96674103f, 0.96674103f, 0.96674103f, 0.96674103f, 0.96674103f, 0.96674103f, 0.96674103f, 0.96674103f});
    auto expCt= NDArrayFactory::create<float>('c', {batchSize, inSize}, {2.01958286f, 2.01958286f, 2.01958286f, 2.01958286f, 2.01958286f, 2.01958286f, 2.01958286f, 2.01958286f, 2.01958286f, 2.01958286f});

    nd4j::ops::sruCell op;
    auto results = op.execute({&xt, &ct_1, &w, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, sruCell_test2) {

    const int batchSize = 2;
    const int inSize    = 5;

    auto xt  = NDArrayFactory::create<float>('c', {batchSize, inSize});
    auto ct_1= NDArrayFactory::create<float>('c', {batchSize, inSize});
    auto w   = NDArrayFactory::create<float>('c', {inSize, 3*inSize});
    auto b   = NDArrayFactory::create<float>('c', {2*inSize});

    xt.assign(1.);
    ct_1.assign(2.);
    w.assign(0.5);
    b.assign(-1.);

    auto expHt= NDArrayFactory::create<float>('c', {batchSize, inSize}, {0.97542038f, 0.97542038f, 0.97542038f, 0.97542038f, 0.97542038f, 0.97542038f, 0.97542038f, 0.97542038f, 0.97542038f, 0.97542038f});
    auto expCt= NDArrayFactory::create<float>('c', {batchSize, inSize}, {2.09121276f, 2.09121276f, 2.09121276f, 2.09121276f, 2.09121276f, 2.09121276f, 2.09121276f, 2.09121276f, 2.09121276f, 2.09121276f});

    nd4j::ops::sruCell op;
    auto results = op.execute({&xt, &ct_1, &w, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, sruCell_test3) {

    const int batchSize = 2;
    const int inSize    = 5;

    auto xt  = NDArrayFactory::create<float>('c', {batchSize, inSize});
    auto ct_1= NDArrayFactory::create<float>('c', {batchSize, inSize});
    auto w   = NDArrayFactory::create<float>('c', {inSize, 3*inSize});
    auto b   = NDArrayFactory::create<float>('c', {2*inSize});

    xt.assign(10.);
    ct_1.assign(1.);
    w.assign(0.5);
    b.assign(-1.);

    auto expHt= NDArrayFactory::create<float>('c', {batchSize, inSize}, {0.76159416f, 0.76159416f, 0.76159416f, 0.76159416f, 0.76159416f, 0.76159416f, 0.76159416f, 0.76159416f, 0.76159416f, 0.76159416f});
    auto expCt= NDArrayFactory::create<float>('c', {batchSize, inSize}, {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f});

    nd4j::ops::sruCell op;
    auto results = op.execute({&xt, &ct_1, &w, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(0);
    auto *ct = results->at(1);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));
    ASSERT_TRUE(expCt.isSameShape(ct));
    ASSERT_TRUE(expCt.equalsTo(ct));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, gruCell_test1) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numUnits  = 4;

    auto xt    = NDArrayFactory::create<float>('c', {batchSize, inSize});
    auto ht_1  = NDArrayFactory::create<float>('c', {batchSize, numUnits});
    auto Wru   = NDArrayFactory::create<float>('c', {(inSize+numUnits), 2*numUnits});
    auto Wc    = NDArrayFactory::create<float>('c', {(inSize+numUnits), numUnits});
    auto bru   = NDArrayFactory::create<float>('c', {2*numUnits});
    auto bc    = NDArrayFactory::create<float>('c', {numUnits});

    xt.assign(1.);
    ht_1.assign(2.);
    Wru.assign(0.5);
    Wc.assign(0.5);
    bru.assign(0.7);
    bc.assign(0.7);

    auto expHt = NDArrayFactory::create<float>('c', {batchSize, numUnits}, {1.99993872f, 1.99993872f, 1.99993872f, 1.99993872f, 1.99993872f, 1.99993872f, 1.99993872f, 1.99993872f});

    nd4j::ops::gruCell op;
    auto results = op.execute({&xt, &ht_1, &Wru, &Wc, &bru, &bc}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(3);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, gruCell_test2) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numUnits  = 4;

    auto xt    = NDArrayFactory::create<float>('c', {batchSize, inSize});
    auto ht_1  = NDArrayFactory::create<float>('c', {batchSize, numUnits});
    auto Wru   = NDArrayFactory::create<float>('c', {(inSize+numUnits), 2*numUnits});
    auto Wc    = NDArrayFactory::create<float>('c', {(inSize+numUnits), numUnits});
    auto bru   = NDArrayFactory::create<float>('c', {2*numUnits});
    auto bc    = NDArrayFactory::create<float>('c', {numUnits});

    xt.assign(1.);
    ht_1.assign(0.);
    Wru.assign(1.5);
    Wc.assign(1.5);
    bru.assign(-10);
    bc.assign(-10);

    auto expHt= NDArrayFactory::create<float>('c', {batchSize, numUnits}, {0.00669224f, 0.00669224f, 0.00669224f, 0.00669224f, 0.00669224f, 0.00669224f, 0.00669224f, 0.00669224f});

    nd4j::ops::gruCell op;
    auto results = op.execute({&xt, &ht_1, &Wru, &Wc, &bru, &bc}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(3);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, gruCell_test3) {

    const int batchSize = 2;
    const int inSize    = 10;
    const int numUnits  = 4;

    auto xt  = NDArrayFactory::create<float>('c', {batchSize, inSize});
    auto ht_1= NDArrayFactory::create<float>('c', {batchSize, numUnits});
    auto Wru   = NDArrayFactory::create<float>('c', {(inSize+numUnits), 2*numUnits});
    auto Wc    = NDArrayFactory::create<float>('c', {(inSize+numUnits), numUnits});
    auto bru   = NDArrayFactory::create<float>('c', {2*numUnits});
    auto bc    = NDArrayFactory::create<float>('c', {numUnits});

    xt.assign(1.);
    ht_1.assign(0.);
    Wru.assign(0.1);
    Wc.assign(0.1);
    bru.assign(1);
    bc.assign(1);

    auto expHt= NDArrayFactory::create<float>('c', {batchSize, numUnits}, {0.1149149f, 0.1149149f, 0.1149149f, 0.1149149f, 0.1149149f, 0.1149149f, 0.1149149f, 0.1149149f});

    nd4j::ops::gruCell op;
    auto results = op.execute({&xt, &ht_1, &Wru, &Wc, &bru, &bc}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *ht = results->at(3);

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, invertPermutation_test1) {

    auto input= NDArrayFactory::create<double>('c', {1, 8}, {5,2,7,4,6,3,1,0});
    auto expected= NDArrayFactory::create<double>('c', {1, 8}, {7, 6, 1, 5, 3, 0, 4, 2});

    nd4j::ops::invert_permutation op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, invertPermutation_test2) {

    auto input= NDArrayFactory::create<double>('c', {1, 8}, {5,2,7,4,6,3,1,0});
    auto expected= NDArrayFactory::create<double>('c', {1, 8}, {7, 6, 1, 5, 3, 0, 4, 2});

    nd4j::ops::invert_permutation op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, invertPermutation_test3) {

    auto input= NDArrayFactory::create<double>('c', {1, 8}, {1,2,0,4,6,3,5,7});
    auto expected= NDArrayFactory::create<double>('c', {1, 8}, {2, 0, 1, 5, 3, 6, 4, 7});

    nd4j::ops::invert_permutation op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test1) {

    auto input= NDArrayFactory::create<double>('c', {3, 2});
    input.linspace(1);

    auto expected= NDArrayFactory::create<double>('c', {3,2,3,2}, {1,0,0,0,0,0, 0,2,0,0,0,0, 0,0,3,0,0,0, 0,0,0,4,0,0, 0,0,0,0,5,0, 0,0,0,0,0,6});

    nd4j::ops::diag op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test2) {

    auto input= NDArrayFactory::create<double>('c', {2, 3});
    input.linspace(1);

    auto expected= NDArrayFactory::create<double>('c', {2,3,2,3}, {1,0,0,0,0,0, 0,2,0,0,0,0, 0,0,3,0,0,0, 0,0,0,4,0,0, 0,0,0,0,5,0, 0,0,0,0,0,6});

    nd4j::ops::diag op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test_vector) {


    auto input = NDArrayFactory::linspace<double>(1,4,4);
    auto expected= NDArrayFactory::create<double>('c', {4,4}, {1,0,0,0, 0,2,0,0, 0,0,3,0,0,0,0,4});

    nd4j::ops::diag op;
    auto results = op.execute({input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
    delete input;
}




TEST_F(DeclarableOpsTests3, diag_test_col_vector) {


    auto input = NDArrayFactory::linspace<double>(1,4,4);
    input->reshapei({4,1});
    auto expected= NDArrayFactory::create<double>('c', {4,4}, {1,0,0,0, 0,2,0,0, 0,0,3,0,0,0,0,4});

    nd4j::ops::diag op;
    auto results = op.execute({input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
    delete input;
}
///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test3) {

    auto input= NDArrayFactory::create<double>('c', {1, 3});
    input.linspace(1);

    auto expected= NDArrayFactory::create<double>('c', {3,3}, {1,0,0, 0,2,0, 0,0,3});

    nd4j::ops::diag op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test4) {

    auto input= NDArrayFactory::create<double>('c', {3, 1});
    input.linspace(1);

    auto expected= NDArrayFactory::create<double>('c', {3,3}, {1,0,0, 0,2,0, 0,0,3});

    nd4j::ops::diag op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test5) {

    auto input= NDArrayFactory::create<double>('c', {1, 1});
    input.linspace(2);

    auto expected= NDArrayFactory::create<double>('c', {1,1}, {2});

    nd4j::ops::diag op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test6) {

    auto input= NDArrayFactory::create<double>('c', {2,2,2});
    input.linspace(1);

    auto expected= NDArrayFactory::create<double>('c', {2,2,2,2,2,2}, {1,0,0,0, 0,0,0,0, 0,2,0,0, 0,0,0,0, 0,0,3,0, 0,0,0,0, 0,0,0,4, 0,0,0,0, 0,0,0,0, 5,0,0,0, 0,0,0,0, 0,6,0,0, 0,0,0,0, 0,0,7,0, 0,0,0,0, 0,0,0,8});

    nd4j::ops::diag op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, matrixSetDiag_test1) {

    auto input= NDArrayFactory::create<double>('c', {4,3,2});
    auto diagonal= NDArrayFactory::create<double>('c', {4,2});
    input.assign(0.);
    diagonal.assign(1.);

    auto expected= NDArrayFactory::create<double>('c', {4,3,2}, {1,0,0,1,0,0, 1,0,0,1,0,0, 1,0,0,1,0,0, 1,0,0,1,0,0});

    nd4j::ops::matrix_set_diag op;
    auto results = op.execute({&input, &diagonal}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, matrixSetDiag_test2) {

    auto input= NDArrayFactory::create<float>('c', {1,1,2});
    auto diagonal= NDArrayFactory::create<float>('c', {1,1});
    input.assign(0.);
    diagonal.assign(1.);

    auto expected= NDArrayFactory::create<float>('c', {1,1,2}, {1.f, 0.f});

    nd4j::ops::matrix_set_diag op;
    auto results = op.execute({&input, &diagonal}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, matrixSetDiag_test3) {

    auto input= NDArrayFactory::create<double>('c', {2,1,4});
    auto diagonal= NDArrayFactory::create<double>('c', {2,1});
    input.assign(0.);
    diagonal.assign(1.);

    auto expected= NDArrayFactory::create<double>('c', {2,1,4}, {1,0,0,0,1,0,0,0});

    nd4j::ops::matrix_set_diag op;
    auto results = op.execute({&input, &diagonal}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, matrixSetDiag_test4) {

    auto input= NDArrayFactory::create<double>('c', {2,1,4,1});
    auto diagonal= NDArrayFactory::create<double>('c', {2,1,1});
    input.assign(0.);
    diagonal.assign(1.);

    auto expected= NDArrayFactory::create<double>('c', {2,1,4,1}, {1,0,0,0,1,0,0,0});

    nd4j::ops::matrix_set_diag op;
    auto results = op.execute({&input, &diagonal}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diagPart_test1) {

    auto input= NDArrayFactory::create<double>('c', {2,2});
    input.linspace(1);

    auto expected= NDArrayFactory::create<double>('c', {2}, {1,4});

    nd4j::ops::diag_part op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);
    // output->printBuffer();

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diagPart_test2) {

    auto input= NDArrayFactory::create<double>('c', {2,2,2,2});
    input.linspace(1);

    auto expected= NDArrayFactory::create<double>('c', {2,2}, {1,6,11,16});

    nd4j::ops::diag_part op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diagPart_test3) {

    auto input= NDArrayFactory::create<double>('c', {2,2,2,2,2,2});
    input.linspace(1);

    auto expected= NDArrayFactory::create<double>('c', {2,2,2}, {1,10,19,28,37,46,55,64});

    nd4j::ops::diag_part op;
    auto results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test1) {

    auto a = NDArrayFactory::create<float16>('c', {3,3});
    auto b = NDArrayFactory::create<float16>('c', {3,3});
    auto x = NDArrayFactory::create<float16>('c', {3,3});

    a.linspace((float16)0.1, (float16)0.1);
    b.linspace((float16)0.1, (float16)0.1);
    x.assign(0.1);

    auto expected = NDArrayFactory::create<float16>('c', {3,3}, {0.40638509f, 0.33668978f, 0.28271242f, 0.23973916f, 0.20483276f, 0.17604725f, 0.15203027f, 0.13180567f, 0.114647f});

    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-2));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test2) {

    auto a= NDArrayFactory::create<float>('c', {3,3});
    auto b= NDArrayFactory::create<float>('c', {3,3});
    auto x= NDArrayFactory::create<float>('c', {3,3});

    a.linspace(0.1, 0.1);
    b.linspace(0.1, 0.1);
    x.assign(0.1);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {0.40638509f, 0.33668978f, 0.28271242f, 0.23973916f, 0.20483276f, 0.17604725f, 0.15203027f, 0.13180567f, 0.114647f});

    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test3) {

    auto a= NDArrayFactory::create<float>('c', {3,3});
    auto b= NDArrayFactory::create<float>('c', {3,3});
    auto x= NDArrayFactory::create<float>('c', {3,3});

    a.linspace(0.1, 0.1);
    b.linspace(0.1, 0.1);
    x.assign(0.1);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {0.40638509f, 0.33668978f, 0.28271242f, 0.23973916f, 0.20483276f, 0.17604725f, 0.15203027f, 0.13180567f, 0.114647f});

    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test4) {

    auto a= NDArrayFactory::create<float>('c', {3,3});
    auto b= NDArrayFactory::create<float>('c', {3,3});
    auto x= NDArrayFactory::create<float>('c', {3,3});

    a.linspace(1);
    b.linspace(1);
    x.assign(0.1);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {1.00000000e-01f, 2.80000000e-02f, 8.56000000e-03f, 2.72800000e-03f, 8.90920000e-04f, 2.95706080e-04f, 9.92854864e-05f, 3.36248880e-05f, 1.14644360e-05f});

    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-6));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test5) {

    auto a= NDArrayFactory::create<float>('c', {3,3});
    auto b= NDArrayFactory::create<float>('c', {3,3});
    auto x= NDArrayFactory::create<float>('c', {3,3});

    a.linspace(3200.);
    b.linspace(3200.);
    x.assign(0.1);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-6));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test6) {

    auto a= NDArrayFactory::create<float>('c', {3,3});
    auto b= NDArrayFactory::create<float>('c', {3,3});
    auto x= NDArrayFactory::create<float>('c', {3,3});

    a.linspace(10.);
    b.linspace(10.);
    x.assign(0.1);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {3.92988233e-06f, 1.35306497e-06f, 4.67576826e-07f, 1.62083416e-07f, 5.63356971e-08f, 1.96261318e-08f, 6.85120307e-09f, 2.39594668e-09f, 8.39227685e-10f});

    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-6));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test7) {

    auto a= NDArrayFactory::create<float>('c', {3,3});
    auto b= NDArrayFactory::create<float>('c', {3,3});
    auto x= NDArrayFactory::create<float>('c', {3,3});

    a.linspace(10.);
    b.linspace(10.);
    x.assign(0.9);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {0.99999607f, 0.99999865f, 0.99999953f, 0.99999984f, 0.99999994f, 0.99999998f, 0.99999999f, 1.f, 1.f});

    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-6));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test8) {

    auto a= NDArrayFactory::create<float>('c', {3,3});
    auto b= NDArrayFactory::create<float>('c', {3,3});
    auto x= NDArrayFactory::create<float>('c', {3,3});

    a.linspace(10.);
    b.linspace(10.);
    x.assign(1.);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {1.f, 1.f, 1.f,1.f,1.f,1.f,1.f,1.f,1.f});

    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-6));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test9) {

    auto a= NDArrayFactory::create<float>('c', {3,3});
    auto b= NDArrayFactory::create<float>('c', {3,3});
    auto x= NDArrayFactory::create<float>('c', {3,3});

    a.linspace(10.);
    b.linspace(10.);
    x.assign(0.);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});

    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test10) {

    auto a= NDArrayFactory::create<float>('c', {3,3});
    auto b= NDArrayFactory::create<float>('c', {3,3});
    auto x= NDArrayFactory::create<float>('c', {3,3});

    a.linspace(10.);
    b.linspace(10.);
    x.assign(0.5);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});

    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test11) {

    NDArray a('c', {4}, {0.7788f, 0.8012f, 0.7244f, 0.2309f}, nd4j::DataType::FLOAT32);
    NDArray b('c', {4}, {0.7717f, 0.9281f, 0.9846f, 0.4838f}, nd4j::DataType::FLOAT32);
    NDArray x('c', {4}, {0.9441f, 0.5957f, 0.8669f, 0.3502f}, nd4j::DataType::FLOAT32);

    NDArray expected('c', {4}, {0.912156, 0.634460, 0.898314, 0.624538}, nd4j::DataType::FLOAT32);
    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test12) {

    NDArray a('c', {4}, {8.0091f, 8.2108f, 7.5194f, 3.0780f}, nd4j::DataType::FLOAT32);
    NDArray b('c', {4}, {7.9456f, 9.3527f, 9.8610f, 5.3541f}, nd4j::DataType::FLOAT32);
    NDArray x('c', {4}, {0.9441f, 0.5957f, 0.8669f, 0.3502f}, nd4j::DataType::FLOAT32);

    NDArray expected('c', {4}, {0.9999995 , 0.8594694 , 0.999988  , 0.49124345}, nd4j::DataType::FLOAT32);

    nd4j::ops::betainc op;
    auto results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test1) {

    auto x= NDArrayFactory::create<float>('c', {3,3});
    auto q= NDArrayFactory::create<float>('c', {3,3});

    q.linspace(1.);
    x.assign(2.);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {1.64493407f, 0.64493407f, 0.39493407f, 0.28382296f, 0.22132296f, 0.18132296f, 0.15354518f, 0.13313701f, 0.11751201f});

    nd4j::ops::zeta op;
    auto results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test2) {

    auto x= NDArrayFactory::create<float>('c', {3,3});
    auto q= NDArrayFactory::create<float>('c', {3,3});

    q.linspace(10.);
    x.assign(2.);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {0.10516634f, 0.09516634f, 0.08690187f, 0.07995743f, 0.07404027f, 0.06893823f, 0.06449378f, 0.06058753f, 0.05712733f});

    nd4j::ops::zeta op;
    auto results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test3) {

    auto x= NDArrayFactory::create<float>('c', {3,3});
    auto q= NDArrayFactory::create<float>('c', {3,3});

    q.linspace(100.);
    x.assign(2.);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {0.01005017f, 0.00995017f, 0.00985214f, 0.00975602f, 0.00966176f, 0.0095693f, 0.0094786f, 0.0093896f, 0.00930226f});

    nd4j::ops::zeta op;
    auto results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test4) {

    auto x= NDArrayFactory::create<float>('c', {3,3});
    auto q= NDArrayFactory::create<float>('c', {3,3});

    q.linspace(100.);
    x.assign(2.);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {0.01005017f, 0.00995017f, 0.00985214f, 0.00975602f, 0.00966176f, 0.0095693f, 0.0094786f, 0.0093896f, 0.00930226f});

    nd4j::ops::zeta op;
    auto results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test5) {

    auto x= NDArrayFactory::create<float>('c', {3,3});
    auto q= NDArrayFactory::create<float>('c', {3,3});

    q.linspace(1.);
    x.assign(1.1);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {10.58444846f, 9.58444846f, 9.11793197f, 8.81927915f, 8.60164151f, 8.43137352f, 8.29204706f, 8.17445116f, 8.07291961f});

    nd4j::ops::zeta op;
    auto results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test6) {

    auto x= NDArrayFactory::create<float>('c', {3,3});
    auto q= NDArrayFactory::create<float>('c', {3,3});

    q.linspace(1.);
    x.assign(1.01);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {100.57794334f, 99.57794334f, 99.08139709f, 98.75170576f, 98.50514758f, 98.30834069f, 98.1446337f, 98.00452955f, 97.88210202f});

    nd4j::ops::zeta op;
    auto results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test7) {

    auto x= NDArrayFactory::create<float>('c', {3,3});
    auto q= NDArrayFactory::create<float>('c', {3,3});

    q.linspace(1.);
    x.assign(10.);

    auto expected= NDArrayFactory::create<float>('c', {3,3}, {1.00099458e+00f, 9.94575128e-04f, 1.80126278e-05f, 1.07754001e-06f, 1.23865693e-07f, 2.14656932e-08f, 4.92752156e-09f, 1.38738839e-09f, 4.56065812e-10f});

    nd4j::ops::zeta op;
    auto results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test8) {

    auto x= NDArrayFactory::create<double>('c', {3,4}, {1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.01,1.11,1.12});
    auto q= NDArrayFactory::create<double>('c', {3,4}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.11, 0.12});

    //q.linspace(1.);
    //x.assign(10.);

    auto expected= NDArrayFactory::create<double>('c', {3,4}, {23.014574, 12.184081, 8.275731, 6.1532226, 4.776538, 3.7945523, 3.0541048, 2.4765317, 2.0163891, 205.27448, 21.090889, 19.477398});

    nd4j::ops::zeta op;
    auto results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test9) {

    auto x= NDArrayFactory::create<double>('c', {3,4}, {1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.01,1.11,1.12});
    auto q= NDArrayFactory::create<double>('c', {3,4}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.11, 0.12});
    auto z= NDArrayFactory::create<double>('c', {3,4}, {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.});

    //q.linspace(1.);
    //x.assign(10.);

    auto expected= NDArrayFactory::create<double>('c', {3,4}, {23.014574, 12.184081, 8.275731, 6.1532226, 4.776538, 3.7945523, 3.0541048, 2.4765317, 2.0163891, 205.27448, 21.090889, 19.477398});

    nd4j::ops::zeta op;
    auto results = op.execute({&x, &q}, {&z}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results);

    //auto *output = results->at(0);
    // z.printIndexedBuffer("Zeta output");
    ASSERT_TRUE(expected.isSameShape(z));
    ASSERT_TRUE(expected.equalsTo(z));

//    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test10) {

    auto x= NDArrayFactory::create<double>('c', {3,4}, {1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.01,1.11,1.12});
    auto q= NDArrayFactory::create<double>('c', {3,4}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.11, 0.12});
    auto z= NDArrayFactory::create<double>('c', {3,4});

    //q.linspace(1.);
    //x.assign(10.);

    auto expected= NDArrayFactory::create<double>('c', {3,4}, {23.014574, 12.184081, 8.275731, 6.1532226, 4.776538, 3.7945523, 3.0541048, 2.4765317, 2.0163891, 205.27448, 21.090889, 19.477398});

    nd4j::ops::zeta op;
    auto results = op.execute({&x, &q}, {&z}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results);

    //auto *output = results->at(0);
    // z.printIndexedBuffer("Zeta output");
    ASSERT_TRUE(expected.isSameShape(z));
    ASSERT_TRUE(expected.equalsTo(z));

//    delete results;
}


TEST_F(DeclarableOpsTests3, Test_SplitV_Validation_1) {
    auto x = NDArrayFactory::create<float>('c', {8, 7});
    auto indices = NDArrayFactory::create<int>('c',{2}, {5, 3});
    auto axis = NDArrayFactory::create<int>(-2);

    auto z0 = NDArrayFactory::create<float>('c', {5, 7});
    auto z1 = NDArrayFactory::create<float>('c', {3, 7});

    nd4j::ops::split_v op;
    auto status = op.execute({&x, &indices, &axis}, {&z0, &z1}, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, polygamma_test1) {

    auto n= NDArrayFactory::create<double>('c', {3,3});
    auto x= NDArrayFactory::create<double>('c', {3,3});
//    ASSERT_FALSE(true);
    n.linspace(1.);
    x.assign(0.5);

    auto expected= NDArrayFactory::create<double>('c', {3,3}, {4.934802, -16.828796, 97.409088, -771.474243, 7691.113770, -92203.460938, 1290440.250000, -20644900.000000, 3.71595e+08});

    nd4j::ops::polygamma op;
    auto results = op.execute({&n, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);
    // output->printBuffer();

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, polygamma_test2) {

    auto n= NDArrayFactory::create<double>('c', {3,3});
    auto x= NDArrayFactory::create<double>('c', {3,3});

    n.linspace(10.);
    x.linspace(0.5);

    auto expected= NDArrayFactory::create<double>('c', {3,3}, {-7.43182451e+09, 3.08334759e+05,-3.25669798e+03, 1.55186197e+02,-1.46220433e+01, 2.00905201e+00,-3.48791235e-01, 7.08016273e-02,-1.60476052e-02});

    //ASSERT_FALSE(true);

    nd4j::ops::polygamma op;
    auto results = op.execute({&n, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, polygamma_test3) {

    auto n= NDArrayFactory::create<double>('c', {3,3});
    auto x= NDArrayFactory::create<double>('c', {3,3});

    n.linspace(1.);
    x.linspace(10.);

    auto expected= NDArrayFactory::create<double>('c', {3,3}, {1.05166336e-01,-9.04983497e-03, 1.31009323e-03,-2.44459433e-04, 5.31593880e-05,-1.28049888e-05, 3.31755364e-06,-9.07408791e-07, 2.58758130e-07});
    nd4j::ops::polygamma op;
    auto results = op.execute({&n, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

TEST_F(DeclarableOpsTests3, polygamma_test4) {

    NDArray n('c', {3,4}, {/*0.7788*/0, 0,1,2,3,4,5,6,7,8,9,10}, nd4j::DataType::DOUBLE);
    NDArray x('c', {3,4}, {0.7717,0.9281,0.9846,0.4838,0.6433,0.6041,0.6501,0.7612,0.7605,0.3948,0.9493,0.8600}, nd4j::DataType::DOUBLE);

    NDArray expected('c', {3,4}, {/*std::numeric_limits<double>::quiet_NaN()*/-1.031918,  -7.021327e-01,  1.682743e+00, -1.851378e+01,3.604167e+01, -3.008293e+02,
                                1.596005e+03, -4.876665e+03,4.510025e+04, -1.730340e+08,  6.110257e+05, -1.907087e+07}, nd4j::DataType::DOUBLE);

    nd4j::ops::polygamma op;
    auto results = op.execute({&n, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

TEST_F(DeclarableOpsTests3, digamma_1) {

    NDArray x('c', {18}, {-25, -24.99999, -21.5, -21.2, -5.5, -4.1, -2.1, -0.5, -0.3, 0., 0.2, 1, 1.5, 2.2, 5.2, 19., 21, 22.2}, nd4j::DataType::DOUBLE);

    NDArray expected('c', {18}, {std::numeric_limits<double>::infinity(), -99996.761229, 3.091129, 7.401432, 1.792911,11.196838,10.630354, 0.03649, 2.11331,
                                 std::numeric_limits<double>::infinity(),-5.28904,-0.577216, 0.03649, 0.544293, 1.549434,2.917892, 3.020524, 3.077401}, nd4j::DataType::DOUBLE);

    nd4j::ops::digamma op;
    auto results = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, svd_test1) {

    auto x= NDArrayFactory::create<double>('c', {6,6}, {0. ,-9. ,-6 ,9 ,-10 ,-12 ,2 ,13 ,5 ,-11 ,20 ,-17 ,1 ,-2 ,-11 ,3 ,-8 ,3 ,-14 ,19 ,-20 ,20 ,-17 ,-5 ,6 ,-16 ,0 ,-1 ,-16 ,11 ,7 ,-19 ,2 ,-17 ,17 ,-16});
    auto expS= NDArrayFactory::create<double>('c', {6}, {54.12775, 38.79293, 25.89287, 9.82168, 6.07227, 2.91827});
    auto expU= NDArrayFactory::create<double>('c', {6,6}, {0.14692,-0.11132,-0.69568, 0.59282,-0.14881, 0.32935,-0.38751, 0.60378,-0.04927,-0.01397,-0.69456,-0.01581, 0.19293,-0.12795,-0.18682,-0.69065,-0.20597, 0.62617, 0.66806, 0.4314 ,-0.33849,-0.22166, 0.04099,-0.44967, 0.11121,-0.64065,-0.02138,-0.07378,-0.60568,-0.45216,-0.5765 ,-0.1007 ,-0.60305,-0.34175, 0.29068,-0.3042});
    auto expV= NDArrayFactory::create<double>('c', {6,6}, {-0.24577,-0.24512, 0.00401,-0.04585,-0.62058, 0.70162, 0.27937, 0.75961, 0.43885,-0.06857,-0.3839 , 0.01669,-0.35944,-0.09629, 0.44593, 0.78602,-0.09103,-0.19125, 0.53973, 0.07613,-0.10721, 0.49559, 0.35687, 0.56431,-0.6226 , 0.39742, 0.12785,-0.15716, 0.52372, 0.37297, 0.23113,-0.43578, 0.76204,-0.32414, 0.23996, 0.11543});

    nd4j::ops::svd op;
    auto results = op.execute({&x}, {}, {1, 1, 16});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *s = results->at(0);
    auto *u = results->at(1);
    auto *v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    ASSERT_TRUE(expS.equalsTo(s));

    if(nd4j::Environment::getInstance()->isCPU()) {
        ASSERT_TRUE(expU.equalsTo(u));
        ASSERT_TRUE(expV.equalsTo(v));
    }
    else {
        for(uint i = 0; i < expU.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expU.e<float>(i)), nd4j::math::nd4j_abs(u->e<float>(i)), 1e-5);
        for(uint i = 0; i < expV.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expV.e<float>(i)), nd4j::math::nd4j_abs(v->e<float>(i)), 1e-5);
    }

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, svd_test2) {

    auto x = NDArrayFactory::create<double>('c', {7,6}, {0. ,-9. ,-6 ,9 ,-10 ,-12 ,2 ,13 ,5 ,-11 ,20 ,-17 ,1 ,-2 ,-11 ,3 ,-8 ,3 ,-14 ,19 ,-20 ,20 ,-17 ,-5 ,6 ,-16 ,0 ,-1 ,-16 ,11 ,7 ,-19 ,2 ,-17 ,17 ,-16, 4, -9, 1, -15, 7, -2});
    auto expS= NDArrayFactory::create<double>('c', {6}, {56.76573,  39.11776,  26.00713,  11.83606, 6.16578, 3.99672});
    auto expU= NDArrayFactory::create<double>('c', {7,7}, {-0.13417,-0.12443, -0.68854,  0.5196 ,  0.21706,  0.03974,  0.41683, 0.347  , 0.62666, -0.04964, -0.01912,  0.66932,  0.1457 , -0.12183,-0.17329,-0.14666, -0.19639, -0.55355,  0.0614 ,  0.75729,  0.1619 ,-0.64703, 0.37056, -0.37398, -0.32922, -0.0186 , -0.35656, -0.26134,-0.08027,-0.64405, -0.0127 , -0.06934,  0.59287, -0.14956, -0.44712, 0.55906,-0.06235, -0.58017, -0.12911, -0.359  , -0.00393, -0.44877, 0.30645,-0.11953, -0.09083, -0.54163,  0.14283, -0.50417,  0.56178});
    auto expV= NDArrayFactory::create<double>('c', {6,6}, {0.2508 ,-0.2265 , 0.01689,  0.04486,  0.53132,  0.77537,-0.32281, 0.74559, 0.41845, -0.13821,  0.37642,  0.06315, 0.33139,-0.05528, 0.47186,  0.73171,  0.18905, -0.3055 ,-0.57263, 0.06276,-0.09542,  0.59396, -0.36152,  0.419  , 0.59193, 0.4361 , 0.13557, -0.03632, -0.5755 ,  0.32944,-0.21165,-0.44227, 0.75794, -0.29895, -0.27993,  0.13187});

    nd4j::ops::svd op;
    auto results = op.execute({&x}, {}, {1, 1, 16});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *s = results->at(0);
    auto *u = results->at(1);
    auto *v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    ASSERT_TRUE(expS.equalsTo(s));

    if(nd4j::Environment::getInstance()->isCPU()) {
        ASSERT_TRUE(expU.equalsTo(u));
        ASSERT_TRUE(expV.equalsTo(v));
    }
    else {
        for(uint i = 0; i < expU.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expU.e<float>(i)), nd4j::math::nd4j_abs(u->e<float>(i)), 1e-5);
        for(uint i = 0; i < expV.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expV.e<float>(i)), nd4j::math::nd4j_abs(v->e<float>(i)), 1e-5);
    }

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, svd_test3) {

    auto x= NDArrayFactory::create<double>('c', {7,6}, {0. ,-9. ,-6 ,9 ,-10 ,-12 ,2 ,13 ,5 ,-11 ,20 ,-17 ,1 ,-2 ,-11 ,3 ,-8 ,3 ,-14 ,19 ,-20 ,20 ,-17 ,-5 ,6 ,-16 ,0 ,-1 ,-16 ,11 ,7 ,-19 ,2 ,-17 ,17 ,-16, 4, -9, 1, -15, 7, -2});
    auto expS= NDArrayFactory::create<double>('c', {6}, {56.76573,  39.11776,  26.00713,  11.83606, 6.16578, 3.99672});
    auto expU= NDArrayFactory::create<double>('c', {7,6}, {-0.13417, -0.12443, -0.68854,  0.5196 ,  0.21706,  0.03974, 0.347  ,  0.62666, -0.04964, -0.01912,  0.66932,  0.1457 ,-0.17329, -0.14666, -0.19639, -0.55355,  0.0614 ,  0.75729,-0.64703,  0.37056, -0.37398, -0.32922, -0.0186 , -0.35656,-0.08027, -0.64405, -0.0127 , -0.06934,  0.59287, -0.14956, 0.55906, -0.06235, -0.58017, -0.12911, -0.359  , -0.00393, 0.30645, -0.11953, -0.09083, -0.54163,  0.14283, -0.50417});
    auto expV= NDArrayFactory::create<double>('c', {6,6}, {0.2508 ,-0.2265 , 0.01689,  0.04486,  0.53132,  0.77537,-0.32281, 0.74559, 0.41845, -0.13821,  0.37642,  0.06315, 0.33139,-0.05528, 0.47186,  0.73171,  0.18905, -0.3055 ,-0.57263, 0.06276,-0.09542,  0.59396, -0.36152,  0.419  , 0.59193, 0.4361 , 0.13557, -0.03632, -0.5755 ,  0.32944,-0.21165,-0.44227, 0.75794, -0.29895, -0.27993,  0.13187});

    nd4j::ops::svd op;
    auto results = op.execute({&x}, {}, {0, 1, 16});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *s = results->at(0);
    auto *u = results->at(1);
    auto *v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    ASSERT_TRUE(expS.equalsTo(s));

    if(nd4j::Environment::getInstance()->isCPU()) {
        ASSERT_TRUE(expU.equalsTo(u));
        ASSERT_TRUE(expV.equalsTo(v));
    }
    else {
        for(uint i = 0; i < expU.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expU.e<float>(i)), nd4j::math::nd4j_abs(u->e<float>(i)), 1e-5f);
        for(uint i = 0; i < expV.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expV.e<float>(i)), nd4j::math::nd4j_abs(v->e<float>(i)), 1e-5f);
    }

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, svd_test4) {

    auto x= NDArrayFactory::create<double>('c', {6,7}, {0. ,-9. ,-6 ,9 ,-10 ,-12 ,2 ,13 ,5 ,-11 ,20 ,-17 ,1 ,-2 ,-11 ,3 ,-8 ,3 ,-14 ,19 ,-20 ,20 ,-17 ,-5 ,6 ,-16 ,0 ,-1 ,-16 ,11 ,7 ,-19 ,2 ,-17 ,17 ,-16, 4, -9, 1, -15, 7, -2});
    auto expS= NDArrayFactory::create<double>('c', {6}, {53.11053,  39.09542,  28.1987,   17.7468,   11.61684,   5.36217});
    auto expU= NDArrayFactory::create<double>('c', {6,6}, {-0.16541, 0.21276,  0.51284,  0.20472,  0.74797,  0.25102,-0.49879, 0.12076,  0.37629, -0.7211 , -0.24585,  0.12086,-0.36569,-0.70218, -0.08012,  0.21274, -0.07314,  0.56231,-0.44508, 0.4329 ,  0.1356 ,  0.60909, -0.47398, -0.02164, 0.61238,-0.05674,  0.59489,  0.06588, -0.3874 ,  0.33685,-0.13044,-0.50644,  0.46552,  0.13236, -0.00474, -0.70161});
    auto expV= NDArrayFactory::create<double>('c', {7,7}, {-0.35914,  0.68966, -0.30077, -0.15238, -0.48179,  0.14716, -0.16709, 0.21989, -0.34343,  0.11086, -0.78381, -0.37902,  0.24224, -0.06862, 0.32179,  0.12812, -0.25812,  0.0691 , -0.12891,  0.26979,  0.84807,-0.50833,  0.13793,  0.06658, -0.53001,  0.52572, -0.16194,  0.36692, 0.48118,  0.15876, -0.65132, -0.24602,  0.3963 , -0.16651, -0.27155,-0.31605, -0.46947, -0.50195,  0.0378 , -0.34937, -0.53062,  0.15069, 0.35957,  0.35408,  0.38732, -0.12154, -0.22827, -0.7151 ,  0.13065});

    nd4j::ops::svd op;
    auto results = op.execute({&x}, {}, {1, 1, 16});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *s = results->at(0);
    auto *u = results->at(1);
    auto *v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    ASSERT_TRUE(expS.equalsTo(s));

    if(nd4j::Environment::getInstance()->isCPU()) {
        ASSERT_TRUE(expU.equalsTo(u));
        ASSERT_TRUE(expV.equalsTo(v));
    }
    else {
        for(uint i = 0; i < expU.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expU.e<float>(i)), nd4j::math::nd4j_abs(u->e<float>(i)), 1e-5f);
        for(uint i = 0; i < expV.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expV.e<float>(i)), nd4j::math::nd4j_abs(v->e<float>(i)), 1e-5f);
    }

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, svd_test5) {

    auto x= NDArrayFactory::create<double>('c', {6,7}, {0. ,-9. ,-6 ,9 ,-10 ,-12 ,2 ,13 ,5 ,-11 ,20 ,-17 ,1 ,-2 ,-11 ,3 ,-8 ,3 ,-14 ,19 ,-20 ,20 ,-17 ,-5 ,6 ,-16 ,0 ,-1 ,-16 ,11 ,7 ,-19 ,2 ,-17 ,17 ,-16, 4, -9, 1, -15, 7, -2});
    auto expS= NDArrayFactory::create<double>('c', {6}, {53.11053,  39.09542,  28.1987,   17.7468,   11.61684,   5.36217});
    auto expU= NDArrayFactory::create<double>('c', {6,6}, {-0.16541, 0.21276,  0.51284,  0.20472,  0.74797,  0.25102,-0.49879, 0.12076,  0.37629, -0.7211 , -0.24585,  0.12086,-0.36569,-0.70218, -0.08012,  0.21274, -0.07314,  0.56231,-0.44508, 0.4329 ,  0.1356 ,  0.60909, -0.47398, -0.02164, 0.61238,-0.05674,  0.59489,  0.06588, -0.3874 ,  0.33685,-0.13044,-0.50644,  0.46552,  0.13236, -0.00474, -0.70161});
    auto expV= NDArrayFactory::create<double>('c', {7,6}, {-0.35914,  0.68966, -0.30077, -0.15238, -0.48179,  0.14716, 0.21989, -0.34343,  0.11086, -0.78381, -0.37902,  0.24224, 0.32179,  0.12812, -0.25812,  0.0691 , -0.12891,  0.26979,-0.50833,  0.13793,  0.06658, -0.53001,  0.52572, -0.16194, 0.48118,  0.15876, -0.65132, -0.24602,  0.3963 , -0.16651,-0.31605, -0.46947, -0.50195,  0.0378 , -0.34937, -0.53062, 0.35957,  0.35408,  0.38732, -0.12154, -0.22827, -0.7151});

    nd4j::ops::svd op;
    auto results = op.execute({&x}, {}, {0, 1, 16});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *s = results->at(0);
    auto *u = results->at(1);
    auto *v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    ASSERT_TRUE(expS.equalsTo(s));

    if(nd4j::Environment::getInstance()->isCPU()) {
        ASSERT_TRUE(expU.equalsTo(u));
        ASSERT_TRUE(expV.equalsTo(v));
    }
    else {
        for(uint i = 0; i < expU.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expU.e<float>(i)), nd4j::math::nd4j_abs(u->e<float>(i)), 1e-5f);
        for(uint i = 0; i < expV.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expV.e<float>(i)), nd4j::math::nd4j_abs(v->e<float>(i)), 1e-5f);
    }

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, svd_test6) {

    auto x= NDArrayFactory::create<double>('c', {2,2,5,5}, {-7. ,17 ,4 ,-10 ,5 ,1 ,-5 ,-19 ,13 ,-8 ,9 ,13 ,19 ,13 ,-2
                                                ,-8 ,10 ,-9 ,0 ,-20 ,-2 ,14 ,19 ,5 ,-18 ,4 ,-13 ,12 ,-10 ,5 ,-10 ,-10 ,17 ,-5 ,-2 ,10 ,5 ,-4 ,-11 ,15 ,-3 ,15 ,-17
                                                ,-20 ,-10 ,-4 ,12 ,-9 ,16 ,13 ,10 ,-19 ,2 ,-9 ,-10 ,8 ,-2 ,-4 ,3 ,7 ,10 ,-19 ,-11 ,-4 ,-6 ,2 ,-12 ,6 ,-4 ,-14 ,14
                                                ,16 ,7 ,19 ,-17 ,2 ,-14 ,5 ,-1 ,16 ,19 ,-11 ,-14 ,-16 ,-19 ,15 ,-18 ,-12 ,-16 ,16 ,1 ,5 ,7 ,8 ,2 ,13 ,-3 ,6 ,2 ,-5});
    auto expS= NDArrayFactory::create<double>('c', {2,2,5}, {40.95395,  31.46869,  24.79993,  12.33768,   1.80031,
                                                        38.18412,  31.52287,  23.52755,  11.79484,   1.90195,
                                                        39.34498,  32.54861,  17.52492,   7.03003,   2.2399,
                                                        44.72126,  32.3164 ,  16.60139,   6.88783,   0.78122});
    auto expU= NDArrayFactory::create<double>('c', {2,2,5,5},  {0.25441,  0.16908, -0.68564,  0.58844, -0.30054,
                                           -0.32285, -0.58332,  0.3451 ,  0.4746 , -0.45953,0.58332,  0.10605,  0.51533,  0.50234,  0.36136,0.12588, -0.73123, -0.37812, -0.00215,  0.55361,
                                           0.68915, -0.2919 ,  0.04767, -0.4197 , -0.51132,0.44464, -0.25326, -0.42493, -0.01712, -0.74653,0.516  , -0.16688,  0.1854 , -0.77155,  0.27611,
                                           -0.19321, -0.14317, -0.85886, -0.15224,  0.42585,-0.60155, -0.68323,  0.18819, -0.29053, -0.22696,-0.36993,  0.64862, -0.10956, -0.54483, -0.36552,
                                           -0.57697, -0.32277,  0.11229,  0.55495,  0.4923 ,-0.02937,  0.01689, -0.63257,  0.57075, -0.52245,-0.56002, -0.2036 , -0.53119, -0.6022 ,  0.01017,
                                           -0.33605, -0.35257,  0.53215, -0.04936, -0.69075,0.48958, -0.85427, -0.14796, -0.03449,  0.08633,0.15008,  0.60996,  0.31071, -0.67721,  0.22421,
                                           0.67717, -0.59857,  0.04372, -0.2565 ,  0.33979,0.68116,  0.49852, -0.13441,  0.51374, -0.07421,-0.20066,  0.04504,  0.42865,  0.44418,  0.75939,0.12113, -0.13826,  0.83651,  0.11988, -0.50209});
    auto expV= NDArrayFactory::create<double>('c', {2,2,5,5}, {0.01858,  0.17863,  0.51259,  0.14048,  0.82781,
                                          0.59651, -0.13439, -0.395  ,  0.66979,  0.14654,0.73731,  0.47061,  0.19357, -0.41127, -0.16817,0.1047 , -0.29727,  0.73711,  0.38235, -0.45951,
                                          -0.29873,  0.80012, -0.02078,  0.4651 , -0.23201,-0.05314, -0.0419 , -0.52146,  0.77792,  0.344  ,-0.66438,  0.05648,  0.03756, -0.31531,  0.67422,
                                          0.74471,  0.01504, -0.03081, -0.24335,  0.62049,0.03172,  0.91947,  0.30828,  0.23713,  0.04796,-0.01311,  0.38652, -0.79415, -0.42423, -0.19945,
                                          -0.13783, -0.54667, -0.58527,  0.49955,  0.3001 ,0.85214,  0.01628,  0.02688, -0.02891,  0.52157,0.16608, -0.20181,  0.61371,  0.69894, -0.25794,
                                          0.45726, -0.33952, -0.32659, -0.18938, -0.73015,0.13486,  0.73816, -0.41646,  0.47458, -0.1956 ,0.5536 , -0.137  ,  0.64688,  0.50536,  0.03017,
                                          -0.51827, -0.31837, -0.16732,  0.71378, -0.30425,-0.39314,  0.15266,  0.63693, -0.30945, -0.5663 ,-0.51981,  0.03325,  0.37603,  0.05147,  0.76462,-0.01282,  0.92491, -0.08042,  0.36977, -0.03428});

    nd4j::ops::svd op;
    auto results = op.execute({&x}, {}, {1, 1, 16});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *s = results->at(0);
    auto *u = results->at(1);
    auto *v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    ASSERT_TRUE(expS.equalsTo(s));

    if(nd4j::Environment::getInstance()->isCPU()) {
        ASSERT_TRUE(expU.equalsTo(u));
        ASSERT_TRUE(expV.equalsTo(v));
    }
    else {
        for(uint i = 0; i < expU.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expU.e<float>(i)), nd4j::math::nd4j_abs(u->e<float>(i)), 1e-5f);
        for(uint i = 0; i < expV.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expV.e<float>(i)), nd4j::math::nd4j_abs(v->e<float>(i)), 1e-5f);
    }

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, svd_test7) {

    auto x= NDArrayFactory::create<double>('c', {2,2,5,5}, {-7. ,17 ,4 ,-10 ,5 ,1 ,-5 ,-19 ,13 ,-8 ,9 ,13 ,19 ,13 ,-2
            ,-8 ,10 ,-9 ,0 ,-20 ,-2 ,14 ,19 ,5 ,-18 ,4 ,-13 ,12 ,-10
            ,5 ,-10 ,-10 ,17 ,-5 ,-2 ,10 ,5 ,-4 ,-11 ,15 ,-3 ,15 ,-17
            ,-20 ,-10 ,-4 ,12 ,-9 ,16 ,13 ,10 ,-19 ,2 ,-9 ,-10 ,8 ,-2
            ,-4 ,3 ,7 ,10 ,-19 ,-11 ,-4 ,-6 ,2 ,-12 ,6 ,-4 ,-14 ,14
            ,16 ,7 ,19 ,-17 ,2 ,-14 ,5 ,-1 ,16 ,19 ,-11 ,-14 ,-16
            ,-19 ,15 ,-18 ,-12 ,-16 ,16 ,1 ,5 ,7 ,8 ,2 ,13 ,-3 ,6 ,2 ,-5});
    auto expS= NDArrayFactory::create<double>('c', {2,2,5}, {40.95395,  31.46869,  24.79993,  12.33768,   1.80031,
                                        38.18412,  31.52287,  23.52755,  11.79484,   1.90195,
                                        39.34498,  32.54861,  17.52492,   7.03003,   2.2399,
                                        44.72126,  32.3164 ,  16.60139,   6.88783,   0.78122});

    nd4j::ops::svd op;
    auto results = op.execute({&x}, {}, {0, 0, 16});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *s = results->at(0);

    ASSERT_TRUE(expS.equalsTo(s));
    ASSERT_TRUE(expS.isSameShape(s));
    delete results;
}

///////////////////////////////////////////////////////////////////
// TEST_F(DeclarableOpsTests3, svd_test8) {

//    auto x= NDArrayFactory::create<float>('c', {2,2,11,10}, {3 ,-8 ,0 ,3 ,-5 ,16 ,-3 ,7 ,-4 ,19 ,19 ,13 ,15 ,15 ,9 ,6 ,-7 ,-5 ,-9 ,-12 ,7 ,-1 ,-1 ,6 ,19
//            ,-6 ,16 ,0 ,16 ,16 ,7 ,14 ,18. ,0 ,18 ,-4 ,10 ,-16 ,-17 ,15 ,13 ,-17 ,-14 ,-17 ,-5 ,-9 ,-1 ,-19
//            ,-18 ,5 ,-5 ,-13 ,17 ,-19 ,-5 ,18 ,4 ,10 ,17 ,-7 ,-10 ,16 ,10 ,8 ,-10 ,-3 ,10 ,1 ,-4 ,-16 ,-1
//            ,-1 ,5 ,5 ,17 ,14 ,20 ,15 ,-6 ,19 ,14 ,17 ,0 ,-17 ,-16 ,-8 ,-6 ,3 ,-6 ,-11 ,-4 ,-2 ,-7 ,4 ,-6
//            ,-6 ,-17 ,16 ,-8 ,-20 ,2 ,7 ,-12 ,15 ,-15 ,-19 ,14 ,17 ,9 ,10 ,5 ,18 ,2 ,-6 ,0 ,2 ,-10 ,7 ,8
//            ,-13 ,2 ,8 ,20 ,11 ,-15 ,13 ,-10 ,-14 ,-2 ,20 ,5 ,2 ,16 ,18 ,-3 ,3 ,-18 ,15 ,-11 ,17 ,-8 ,-18
//            ,20 ,-12 ,20 ,20 ,-16 ,20 ,-8. ,19 ,-8 ,3 ,-3 ,17 ,7 ,13 ,9 ,-2 ,11 ,16 ,4 ,-18 ,5 ,0 ,-12 ,9
//            ,-6 ,6 ,0 ,-9 ,-13 ,13 ,17 ,-12 ,3 ,-13 ,17 ,-19 ,17 ,0 ,-8 ,4 ,-19 ,-9 ,-7 ,12 ,-1 ,-12 ,-1
//            ,7 ,2 ,19 ,10 ,19 ,-15 ,-18 ,17 ,-1 ,1 ,14 ,-7 ,-10 ,12 ,-20 ,6 ,-5 ,14 ,5 ,5 ,3 ,-18 ,5 ,17
//            ,-13 ,20 ,-1 ,-2 ,-11 ,-5 ,14 ,8 ,7 ,-13 ,-9 ,-12 ,11 ,3 ,14 ,-6 ,-2 ,13 ,8 ,-15 ,-5 ,-6 ,-7 ,19
//            ,-1 ,6 ,1 ,14 ,8 ,18 ,-20 ,-14 ,-3 ,-5 ,19 ,15 ,13 ,2 ,-20 ,2 ,14 ,13 ,4 ,-15 ,1 ,-14
//            ,0 ,9 ,-1 ,10 ,4 ,6 ,4 ,-7 ,-2 ,-1 ,-15 ,-1 ,-16 ,-5 ,-12 ,-10 ,16 ,-16 ,-15 ,-17 ,-5 ,-6
//            ,18 ,14 ,-3 ,-10 ,8 ,20 ,19 ,20 ,-3 ,-6 ,9 ,10 ,-1 ,-20 ,-5 ,5 ,12 ,8 ,17 ,13 ,-18 ,-14 ,0
//            ,4 ,-11 ,3 ,-12 ,-2 ,-5 ,19 ,-15 ,19 ,16 ,-16 ,13 ,-6 ,11 ,11 ,0 ,-18 ,4 ,5 ,6 ,-12 ,-10
//            ,-3 ,2 ,-18 ,16 ,-5 ,17 ,16 ,-16 ,-20 ,14 ,6 ,10 ,-5 ,-3 ,4 ,20 ,18 ,5 ,1 ,-10 ,15 ,10 ,16
//            ,-18 ,2 ,12 ,20 ,6 ,14 ,8 ,3 ,-2 ,9 ,15 ,-4 ,13 ,-19 ,-5 ,3 ,3 ,-20 ,-4 ,18 ,-11 ,11 ,-10
//            ,3 ,8 ,9 ,20 ,-19 ,6 ,18 ,9 ,20 ,-12 ,4 ,15 ,19 ,3 ,5 ,1 ,2 ,20 ,-3 ,-1 ,-8 ,-3 ,8 ,17 ,
//                                         -14 ,18 ,-10 ,4 ,13 ,-5 ,13 ,-6 ,12 ,-10 ,19 ,4 ,-7 ,-17 ,20 ,8 ,6 ,-3 ,3 ,-7 ,-18 ,17 ,
//                                         -13 ,18 ,-20 ,-16 ,-5 ,12 ,5 ,17 ,-4 ,4 ,7 ,8 ,17 ,-9 ,-12 ,-10 ,8 ,-14 ,-11 ,7 ,19 ,-17});

//    auto expS= NDArrayFactory::create<float>('c', {2,2,10}, { 64.12636,  54.37044,  50.63744,  48.10308,  33.7364 ,  29.96456,
//                                          25.53945,  19.31856,  15.30939,   9.31349,
//                                          67.41342,  59.64963,  58.72687,  39.22496,  32.39772,  29.30833,
//                                          23.1491 ,  16.92442,   6.38613,   3.49563,
//                                          74.37477,  52.07016,  46.10758,  39.10742,  32.02261,  27.05888,
//                                          20.54921,  13.17989,   8.4158 ,   4.39974,
//                                          65.47447,  56.31305,  54.13371,  46.26955,  43.47755,  30.25799,
//                                          20.71463,  16.89671,  10.39572,   7.81631});

//    auto expU= NDArrayFactory::create<float>('c', {2,2,11,11},  {-0.177870, -0.149461, -0.196911, 0.036990, -0.338237, 0.548901,
//                                             -0.074396, 0.497067, -0.083636, -0.111810, -0.466989, -0.010465, 0.434732, 0.337198, 0.305239, -0.292813,
//                                             0.041280, -0.517144, 0.121499, 0.464908, 0.003658, 0.135017, -0.446916, -0.098318, 0.073571, -0.200521,
//                                             0.186776, -0.353022, -0.435582, -0.225959, 0.052972, 0.032390, -0.583801, -0.402790, 0.562809, 0.102744,
//                                             0.066555, 0.206079, 0.115322, 0.217220, -0.062591, -0.273173, -0.569645, 0.005612, 0.092601, 0.350055,
//                                             -0.608007, -0.367743, 0.064860, 0.112656, 0.091576, -0.144262, 0.554655, -0.042100, -0.092023, 0.026986,
//                                             -0.395811, -0.245209, 0.572522, 0.429430, 0.099621, -0.159236, -0.086263, 0.268160, -0.391298, 0.050417,
//                                             0.150175, 0.045253, 0.464173, 0.138376, 0.265551, 0.049691, 0.528778, 0.116951, 0.384609, 0.144416,
//                                             -0.453591, -0.519390, -0.150671, 0.072897, 0.102406, -0.154184, 0.450735, 0.174171, -0.519405, 0.147109,
//                                             0.333670, 0.178053, 0.360763, 0.226976, 0.069976, -0.046765, 0.448897, 0.511309, -0.361050, -0.191690,
//                                             -0.304442, 0.270383, -0.124133, 0.417183, -0.083359, 0.137022, 0.004276, -0.462336, 0.051267, 0.020622,
//                                             -0.566932, -0.051351, -0.417106, -0.292202, -0.021595, -0.315956, 0.396626, -0.604952, 0.155990, 0.258395,
//                                             -0.125080, 0.115404, 0.234517, -0.357460, 0.271271, 0.063771, -0.087400, -0.024710, -0.179892, 0.584339,
//                                             -0.413085, 0.510580, 0.334646, 0.044424, 0.224735, 0.134434, -0.147861, 0.291853, 0.487948, 0.238917,
//                                             0.433893, 0.435884, 0.056370, -0.051216, -0.450902, 0.062411, 0.080733, -0.365211, 0.031931, 0.493926,
//                                             -0.239428, 0.038247, -0.180721, -0.118035, 0.042175, 0.377296, -0.516399, 0.324744, -0.756196, 0.160856,
//                                             -0.152527, -0.046867, -0.092933, -0.044945, 0.137659, 0.246552, -0.071709, 0.032821, -0.529356, -0.029669,
//                                             0.200178, 0.188916, 0.428036, -0.496734, -0.164185, 0.629070, -0.131588, 0.073992, 0.066877, 0.208450,
//                                             -0.156170, -0.253670, -0.000365, -0.121172, 0.067774, 0.618226, 0.230460, -0.118865, 0.579424, 0.324523,
//                                             0.038653, 0.310308, 0.570186, -0.217271, -0.110967, 0.196375, 0.167058, 0.264071, -0.130023, 0.254189,
//                                             -0.459057, -0.301033, 0.069932, -0.033338, -0.070600, 0.685064, 0.130274, 0.074929, -0.206899, 0.574057,
//                                             0.327277, -0.131588, -0.018497, 0.312445, 0.314594, 0.480422, -0.293858, -0.273277, -0.006598, -0.134574,
//                                             0.403501, 0.140025, 0.380693, -0.257039, -0.067012, 0.248776, -0.361838, -0.270296, -0.225844, 0.320245,
//                                             0.055730, 0.454809, -0.212163, -0.063281, 0.563112, -0.200737, 0.537389, -0.210845, 0.109997, 0.166215,
//                                             -0.243725, -0.347349, -0.274348, 0.263950, 0.437134, 0.265820, -0.127520, -0.033325, -0.137156, 0.518557,
//                                             0.246720, 0.389394, -0.600568, 0.062027, -0.047838, -0.338416, 0.032778, -0.141998, -0.338022, -0.381467,
//                                             0.210512, -0.314413, 0.256321, 0.001460, 0.238901, 0.139840, 0.633423, -0.182575, -0.461504, 0.290250,
//                                             -0.025930, 0.336998, -0.211280, -0.662387, -0.207946, -0.003860, -0.147842, 0.157217, 0.123704, 0.345686,
//                                             0.337946, 0.138261, -0.178814, -0.109597, 0.087135, -0.509500, -0.300296, -0.262279, 0.377476, -0.366815,
//                                             0.091787, 0.247495, -0.193812, -0.179714, 0.238552, -0.162305, -0.029549, 0.785426, -0.157586, -0.084533,
//                                             -0.357024, 0.317878, 0.217656, 0.125319, 0.648832, 0.344045, -0.001109, 0.457190, -0.072439, -0.106278,
//                                             0.228962, -0.136139, -0.528342, -0.020840, -0.108908, -0.231661, 0.396864, 0.234925, 0.180894, -0.179430,
//                                             -0.587730, 0.178276, -0.008672, -0.386172, 0.033155, 0.319568, 0.101457, -0.272011, 0.126007, 0.175374,
//                                             -0.081668, 0.112987, -0.296422, -0.713743, 0.269413, -0.082098, -0.338649, 0.131035, -0.518616, 0.022478,
//                                             0.177802, -0.042432, -0.606219, -0.343848, 0.014416, -0.141375, 0.748332, -0.165911, -0.049067, -0.241062,
//                                             0.436318, 0.173318, 0.058066, 0.193764, -0.000647, 0.265777, -0.027847, -0.096305, 0.711632, 0.066506,
//                                             -0.223124, 0.219165, -0.038165, 0.427444, -0.296887, 0.139982, 0.298976, 0.294876, -0.001315, 0.419802,
//                                             0.475401, -0.156256, -0.289477, -0.438761, -0.116348, 0.108350, -0.369368, -0.219943, 0.433088, 0.187565,
//                                             -0.217259, 0.147014, -0.538991, -0.065052, 0.310337, 0.491887, 0.254439, 0.075052, 0.071155, -0.084856,
//                                             0.402098, 0.096270, 0.093662, -0.475769, 0.256832, 0.161394, -0.390050, -0.513551, -0.184665, 0.211506,
//                                             -0.112525, -0.493409, -0.258765, 0.262124, -0.272998, 0.269370, 0.266226, -0.367919, 0.192386, -0.006422,
//                                             -0.466728, -0.481792, 0.090611, -0.156359, 0.178693, -0.371658, -0.214190, -0.469058, -0.006134, 0.081902,
//                                             0.536950, 0.064836, -0.334010, 0.523530, -0.182061, -0.206686, 0.002985, 0.054858, -0.038727, -0.075390,
//                                             0.543839, -0.442964, -0.190550, -0.298127, -0.065323, 0.131415, 0.329899, 0.122096, -0.507075, 0.523751,
//                                             -0.167317, 0.198593, -0.069066, 0.402739, 0.328583, 0.314184, -0.268003, -0.148549, 0.118925, -0.508174,
//                                             0.128716, -0.405597, -0.157224, 0.271021, -0.384444, -0.174935, 0.343919, -0.076726, 0.607931, 0.383931,
//                                             0.198254, 0.133707, 0.321460, -0.232543, 0.099988, -0.321954, -0.366304, -0.137440, 0.232835, -0.290306,
//                                             -0.260804, -0.347721, 0.182895, 0.382311, -0.332847, -0.192469, -0.438258, -0.017533, -0.192976, -0.702531,
//                                             0.124463, 0.039719, -0.221319, -0.224785, 0.096356, -0.302131, -0.462598, 0.194320});


//    auto expV= NDArrayFactory::create<float>('c', {2,2,10,10}, {-0.050761, 0.370975, -0.061567, -0.125530, 0.024081, 0.275524, -0.800334,
//                                            -0.025855, 0.348132, 0.036882, 0.034921, 0.307295, 0.629837, 0.014276, 0.265687, 0.188407, -0.035481, 0.082827,
//                                            -0.490175, 0.391118, -0.180180, 0.169108, 0.206663, 0.623321, 0.260009, 0.081943, 0.004485, 0.136199, 0.060353,
//                                            -0.641224, -0.181559, -0.041761, 0.578416, -0.161798, -0.573128, -0.187563, 0.012533, 0.368041, 0.314619,
//                                            -0.079349, -0.527508, 0.216020, 0.004721, 0.188769, -0.242534, -0.442685, -0.121683, -0.565306, -0.202894,
//                                            0.095280, -0.181900, -0.170627, -0.201655, 0.620259, -0.257996, 0.277656, -0.009623, 0.266775, 0.081952,
//                                            0.539241, -0.452254, -0.136142, 0.177049, -0.144734, 0.494673, 0.101613, 0.280091, -0.186281, 0.548779,
//                                            0.235160, 0.054763, -0.571503, 0.298086, 0.035312, -0.195188, 0.474030, -0.175457, -0.497267, -0.101439,
//                                            -0.170678, -0.060605, -0.557305, 0.073433, 0.057195, 0.352091, -0.486102, -0.483569, 0.252091, -0.121245,
//                                            0.068719, -0.638919, -0.078029, -0.236556, -0.351440, -0.024437, 0.319855, -0.007406, 0.319691, -0.402334,
//                                            -0.197966, 0.058936, -0.360900, 0.233414, -0.251532, 0.105457, 0.048097, 0.029321, 0.002714, -0.845953,
//                                            -0.136344, 0.378037, 0.277491, 0.278420, 0.037491, 0.432117, -0.586745, 0.104573, 0.316569, -0.039848,
//                                            0.239645, -0.320923, 0.555156, 0.145059, -0.546959, 0.267760, 0.298029, 0.177831, -0.191286, -0.032427,
//                                            0.197034, 0.081887, -0.113063, 0.711713, 0.020279, -0.362346, -0.145776, 0.173289, -0.500880, 0.181624,
//                                            0.084391, -0.278967, 0.212143, -0.413382, 0.012879, -0.216886, -0.625774, 0.066795, -0.421937, -0.291320,
//                                            0.011402, -0.416660, -0.134200, 0.043039, 0.554715, 0.126867, 0.147315, 0.474334, 0.094354, -0.156458,
//                                            0.450168, 0.447448, 0.261750, -0.161426, -0.064309, -0.592417, 0.210891, 0.104312, 0.176178, -0.237020,
//                                            0.455579, -0.358056, -0.307454, 0.033700, -0.486831, -0.303963, -0.284916, 0.241549, 0.510701, 0.206104,
//                                            0.062587, 0.248212, 0.132088, -0.122704, 0.026342, -0.011108, 0.066306, 0.763127, 0.009491, 0.038822,
//                                            -0.562773, -0.320104, 0.477773, 0.354169, 0.293329, -0.304227, -0.001662, -0.213324, 0.365277, -0.198056,
//                                            -0.383499, -0.017789, 0.324542, -0.642856, 0.238689, -0.360461, -0.060599, -0.257192, 0.342400, 0.180845,
//                                            0.272810, -0.452278, -0.409323, 0.077013, -0.082561, 0.334893, -0.103309, -0.198049, 0.480416, 0.470593,
//                                            0.029072, -0.300574, 0.532293, 0.250892, -0.355298, 0.079716, -0.319781, 0.259925, 0.277872, -0.251917,
//                                            0.346821, 0.161642, 0.205861, 0.107125, -0.594779, -0.226272, 0.610183, -0.065926, 0.170332, 0.312553,
//                                            -0.108093, 0.368268, -0.183109, -0.192222, -0.544559, 0.136824, -0.412352, -0.398250, -0.257291, 0.019911,
//                                            0.288797, 0.013350, 0.349817, -0.108331, 0.180576, 0.652863, 0.319319, 0.020218, -0.324499, 0.290877,
//                                            0.338518, -0.301776, -0.440871, -0.281683, -0.158759, -0.080281, 0.418260, 0.189926, -0.064112, -0.390914,
//                                            0.485420, -0.464327, 0.211070, 0.044295, -0.032292, 0.043985, 0.147160, -0.702247, -0.198395, -0.352940,
//                                            -0.237014, -0.438235, 0.073448, -0.418712, -0.280275, -0.091373, -0.194273, 0.347558, -0.421767, 0.283011,
//                                            -0.351869, -0.210088, -0.034628, 0.448410, 0.149194, -0.488551, -0.068805, -0.117007, -0.390999, 0.377100,
//                                            0.423252, -0.041944, 0.455115, -0.537818, 0.266732, 0.218202, 0.047475, -0.383506, -0.158858, 0.450881,
//                                            0.072415, 0.355772, 0.002360, 0.138976, 0.541349, -0.295405, 0.463832, 0.400676, -0.168962, 0.259334,
//                                            -0.047960, 0.272197, 0.582658, 0.198052, 0.127300, -0.320468, -0.104858, -0.229698, 0.046672, -0.474224,
//                                            0.370765, -0.246450, 0.212667, 0.024935, -0.344530, -0.238547, 0.185931, 0.269068, 0.487414, 0.421376,
//                                            0.442391, -0.284247, 0.304973, -0.365006, -0.159016, -0.129088, -0.126454, 0.600462, -0.461163, -0.243552,
//                                            -0.049814, -0.381340, -0.054504, 0.436237, 0.126120, -0.359677, -0.409734, -0.179422, -0.414820, 0.371149,
//                                            0.078299, 0.503544, 0.322165, 0.148341, -0.495447, -0.084355, -0.174667, 0.016802, -0.066954, 0.318825,
//                                            -0.480771, -0.060163, 0.144302, -0.041555, 0.459106, 0.029882, -0.565026, 0.282336, 0.528472, 0.044916,
//                                            -0.286167, -0.101052, -0.181529, -0.419406, -0.032204, -0.732282, 0.106833, -0.288881, 0.171516, -0.096242,
//                                            -0.331834, -0.493188, 0.393195, 0.358365, 0.049125, 0.123457, 0.438169, -0.105015, 0.092386, -0.130413, -0.476991});

//    nd4j::ops::svd op;
//    auto results = op.execute({&x}, {}, {1, 1, 7});

//    ASSERT_EQ(ND4J_STATUS_OK, results->status());

//    auto *s = results->at(0);
//    auto *u = results->at(1);
//    auto *v = results->at(2);

    // ASSERT_TRUE(expS.isSameShape(s));
    // ASSERT_TRUE(expU.isSameShape(u));
    // ASSERT_TRUE(expV.isSameShape(v));

    // ASSERT_TRUE(expS.equalsTo(s));

    // if(nd4j::Environment::getInstance()->isCPU()) {
    //     ASSERT_TRUE(expU.equalsTo(u));
    //     ASSERT_TRUE(expV.equalsTo(v));
    // }
    // else {
    //     for(uint i = 0; i < expU.lengthOf(); ++i)
    //         ASSERT_NEAR(nd4j::math::nd4j_abs(expU.e<float>(i)), nd4j::math::nd4j_abs(u->e<float>(i)), 1e-5);
    //     for(uint i = 0; i < expV.lengthOf(); ++i)
    //         ASSERT_NEAR(nd4j::math::nd4j_abs(expV.e<float>(i)), nd4j::math::nd4j_abs(v->e<float>(i)), 1e-5);
    // }

//    delete results;
// }

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, svd_test9) {

    auto x= NDArrayFactory::create<double>('c', {2,2,5,6}, {17 ,-11 ,20 ,-10 ,19 ,13 ,-18 ,6 ,-2 ,-6 ,-10 ,4 ,-6 ,-4 ,3 ,16 ,12 ,
                                       -15 ,8 ,-8 ,12 ,-1 ,20 ,19 ,-13 ,0 ,20 ,17 ,-8 ,16 ,-19 ,7 ,-16 ,-14 ,-5 ,7 ,7 ,-5 ,12 ,-15 ,7 ,8 ,
                                       1 ,-8 ,-17 ,10 ,-11 ,8 ,-10 ,1 ,-6 ,10 ,15 ,19 ,-15 ,8 ,2 ,8 ,12 ,7 ,-5 ,1 ,8 ,4 ,-13 ,2 ,19 ,-2 ,-10 ,
                                       -8 ,11 ,1 ,20 ,-11 ,4 ,1 ,-17 ,-15 ,0 ,-9 ,-4 ,-1 ,-6 ,-9 ,-13 ,10 ,7 ,-2 ,15 ,-10 ,-1 ,11 ,-20 ,-2 ,
                                       -1 ,-18 ,12 ,16 ,8 ,-9 ,-20 ,-7 ,-20 ,3 ,-9 ,12 ,8 ,-19 ,-2 ,2 ,1 ,7 ,10 ,-18 ,13 ,6 ,14 ,0 ,19 ,8});

    auto expS= NDArrayFactory::create<double>('c', {2,2,5}, {50.46507,  35.75599,  28.12787,  12.45245,   9.08545,
                                        38.56035,  30.62846,  26.31646,  19.42605,   3.01162,
                                        38.56369,  29.18881,  19.54565,  10.89746,   2.017  ,
                                        44.99108,  34.95059,  26.00453,  15.43898,   7.18752});

    auto expU= NDArrayFactory::create<double>('c', {2,2,5,5},  {-0.73644, -0.10751,  0.10081, -0.00325,  0.66025,
                                           0.26329,  0.3079 ,  0.38582,  0.77696,  0.28872,
                                           0.03076,  0.03015, -0.9128 ,  0.36387,  0.18039,
                                           -0.61335,  0.10076,  0.01381,  0.40922, -0.66783,
                                           -0.10577,  0.93946, -0.0871 , -0.31058,  0.04677,
                                           0.52823,  0.31163, -0.78777,  0.02322, -0.05234,
                                           -0.23942, -0.45801, -0.34248,  0.71286,  0.32778,
                                           0.26147,  0.60409,  0.39933,  0.46862,  0.43318,
                                           0.62118, -0.37993,  0.30992,  0.34537, -0.50444,
                                           0.45763, -0.42877,  0.08128, -0.3904 ,  0.66912,
                                           -0.05428,  0.53632,  0.19774, -0.32198,  0.75276,
                                           -0.21986, -0.8214 , -0.00392, -0.1659 ,  0.49944,
                                           -0.79443,  0.1633 , -0.45374, -0.31666, -0.18989,
                                           -0.24459,  0.10463, -0.27652,  0.85595,  0.34657,
                                           0.50772,  0.00757, -0.82374, -0.18941,  0.16658,
                                           0.49473, -0.39923, -0.20758,  0.74339, -0.01213,
                                           -0.2024 , -0.80239, -0.35502, -0.3982 , -0.17492,
                                           0.68875,  0.1822 , -0.08046, -0.39238, -0.57619,
                                           0.34555,  0.12488, -0.50703, -0.29269,  0.72267,
                                           -0.34713,  0.3847 , -0.7532 ,  0.22176, -0.33913});

    auto expV= NDArrayFactory::create<double>('c', {2,2,6,6}, {-4.15640000e-01,  -5.30190000e-01,   5.29200000e-02,  -7.15710000e-01,
                                          -1.10690000e-01,   1.37280000e-01,
                                          2.86620000e-01,   5.88200000e-02,   1.68760000e-01,  -2.55000000e-03,
                                          -1.00090000e-01,   9.35890000e-01,
                                          -4.88230000e-01,   4.84470000e-01,  -1.09150000e-01,  -1.46810000e-01,
                                          6.70320000e-01,   2.10040000e-01,
                                          1.00910000e-01,   4.35740000e-01,  -6.90500000e-01,  -3.61090000e-01,
                                          -4.38680000e-01,   1.83200000e-02,
                                          -5.48440000e-01,  -2.86950000e-01,  -4.23900000e-01,   5.78540000e-01,
                                          -2.10060000e-01,   2.41550000e-01,
                                          -4.42450000e-01,   4.56640000e-01,   5.48020000e-01,   3.32100000e-02,
                                          -5.40210000e-01,  -4.97000000e-02,
                                          -6.36070000e-01,   5.57600000e-02,   3.28740000e-01,   3.81950000e-01,
                                          -4.21850000e-01,   4.00490000e-01,
                                          1.83740000e-01,  -1.36190000e-01,  -2.29380000e-01,  -5.11090000e-01,
                                          -2.06580000e-01,   7.68890000e-01,
                                          -4.81880000e-01,  -6.31100000e-01,   3.40000000e-04,  -1.35730000e-01,
                                          5.88210000e-01,   7.12900000e-02,
                                          2.25200000e-01,   4.30600000e-02,   9.08510000e-01,  -3.08940000e-01,
                                          1.51570000e-01,   6.02100000e-02,
                                          1.97510000e-01,  -7.26560000e-01,   1.05370000e-01,   1.10600000e-02,
                                          -5.79750000e-01,  -2.92870000e-01,
                                          4.89620000e-01,  -2.24300000e-01,   5.31200000e-02,   6.92040000e-01,
                                          2.72560000e-01,   3.92350000e-01,
                                          -6.84450000e-01,  -5.18030000e-01,   2.92000000e-02,  -4.96740000e-01,
                                          -1.17970000e-01,  -4.08100000e-02,
                                          4.25340000e-01,  -1.65500000e-02,  -2.82400000e-02,  -5.60180000e-01,
                                          1.93050000e-01,  -6.83340000e-01,
                                          8.08800000e-02,   4.38260000e-01,  -2.48340000e-01,  -6.36220000e-01,
                                          2.37500000e-02,   5.78250000e-01,
                                          -6.10000000e-04,   3.00110000e-01,   1.17290000e-01,  -6.92400000e-02,
                                          -9.19220000e-01,  -2.15420000e-01,
                                          5.41330000e-01,  -6.61130000e-01,  -2.86360000e-01,  -2.13500000e-02,
                                          -3.19580000e-01,   2.92020000e-01,
                                          2.25920000e-01,  -1.10170000e-01,   9.17020000e-01,  -1.71540000e-01,
                                          3.39100000e-02,   2.55590000e-01,
                                          -4.86810000e-01,  -2.32390000e-01,  -4.31500000e-01,   3.75290000e-01,
                                          4.98470000e-01,  -3.65370000e-01,
                                          6.39700000e-02,  -4.04150000e-01,  -5.28310000e-01,   8.90000000e-02,
                                          -7.30460000e-01,  -1.09390000e-01,
                                          -4.94030000e-01,   1.55540000e-01,  -3.46720000e-01,  -7.58460000e-01,
                                          5.20000000e-04,   1.90420000e-01,
                                          2.55960000e-01,   3.17040000e-01,  -3.47800000e-02,  -3.01860000e-01,
                                          -3.57600000e-02,  -8.60450000e-01,
                                          1.31650000e-01,   7.57150000e-01,  -4.89030000e-01,   3.47710000e-01,
                                          -4.39400000e-02,   2.17750000e-01,
                                          -6.57270000e-01,   2.91000000e-01,   4.17280000e-01,   2.52880000e-01,
                                          -4.63400000e-01,  -1.74620000e-01});

    nd4j::ops::svd op;
    auto results = op.execute({&x}, {}, {1, 1, 16});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *s = results->at(0);
    auto *u = results->at(1);
    auto *v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    ASSERT_TRUE(expS.equalsTo(s));

    if(nd4j::Environment::getInstance()->isCPU()) {
        ASSERT_TRUE(expU.equalsTo(u));
        ASSERT_TRUE(expV.equalsTo(v));
    }
    else {
        for(uint i = 0; i < expU.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expU.e<float>(i)), nd4j::math::nd4j_abs(u->e<float>(i)), 1e-5);
        for(uint i = 0; i < expV.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expV.e<float>(i)), nd4j::math::nd4j_abs(v->e<float>(i)), 1e-5);
    }

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, svd_test10) {

    auto x= NDArrayFactory::create<double>('c', {2,2,5,6}, {17 ,-11 ,20 ,-10 ,19 ,13 ,-18 ,6 ,-2 ,-6 ,-10 ,4 ,-6 ,-4 ,3 ,16 ,12 ,
                                       -15 ,8 ,-8 ,12 ,-1 ,20 ,19 ,-13 ,0 ,20 ,17 ,-8 ,16 ,-19 ,7 ,-16 ,-14 ,-5 ,7 ,7 ,-5 ,12 ,-15 ,7 ,8 ,
                                       1 ,-8 ,-17 ,10 ,-11 ,8 ,-10 ,1 ,-6 ,10 ,15 ,19 ,-15 ,8 ,2 ,8 ,12 ,7 ,-5 ,1 ,8 ,4 ,-13 ,2 ,19 ,-2 ,-10 ,
                                       -8 ,11 ,1 ,20 ,-11 ,4 ,1 ,-17 ,-15 ,0 ,-9 ,-4 ,-1 ,-6 ,-9 ,-13 ,10 ,7 ,-2 ,15 ,-10 ,-1 ,11 ,-20 ,-2 ,
                                       -1 ,-18 ,12 ,16 ,8 ,-9 ,-20 ,-7 ,-20 ,3 ,-9 ,12 ,8 ,-19 ,-2 ,2 ,1 ,7 ,10 ,-18 ,13 ,6 ,14 ,0 ,19 ,8});

    auto expS= NDArrayFactory::create<double>('c', {2,2,5}, {50.46507,  35.75599,  28.12787,  12.45245,   9.08545,
                                        38.56035,  30.62846,  26.31646,  19.42605,   3.01162,
                                        38.56369,  29.18881,  19.54565,  10.89746,   2.017  ,
                                        44.99108,  34.95059,  26.00453,  15.43898,   7.18752});

    auto expU= NDArrayFactory::create<double>('c', {2,2,5,5},  {-0.73644, -0.10751,  0.10081, -0.00325,  0.66025,
                                           0.26329,  0.3079 ,  0.38582,  0.77696,  0.28872,
                                           0.03076,  0.03015, -0.9128 ,  0.36387,  0.18039,
                                           -0.61335,  0.10076,  0.01381,  0.40922, -0.66783,
                                           -0.10577,  0.93946, -0.0871 , -0.31058,  0.04677,
                                           0.52823,  0.31163, -0.78777,  0.02322, -0.05234,
                                           -0.23942, -0.45801, -0.34248,  0.71286,  0.32778,
                                           0.26147,  0.60409,  0.39933,  0.46862,  0.43318,
                                           0.62118, -0.37993,  0.30992,  0.34537, -0.50444,
                                           0.45763, -0.42877,  0.08128, -0.3904 ,  0.66912,
                                           -0.05428,  0.53632,  0.19774, -0.32198,  0.75276,
                                           -0.21986, -0.8214 , -0.00392, -0.1659 ,  0.49944,
                                           -0.79443,  0.1633 , -0.45374, -0.31666, -0.18989,
                                           -0.24459,  0.10463, -0.27652,  0.85595,  0.34657,
                                           0.50772,  0.00757, -0.82374, -0.18941,  0.16658,
                                           0.49473, -0.39923, -0.20758,  0.74339, -0.01213,
                                           -0.2024 , -0.80239, -0.35502, -0.3982 , -0.17492,
                                           0.68875,  0.1822 , -0.08046, -0.39238, -0.57619,
                                           0.34555,  0.12488, -0.50703, -0.29269,  0.72267,
                                           -0.34713,  0.3847 , -0.7532 ,  0.22176, -0.33913});

    auto expV= NDArrayFactory::create<double>('c', {2,2,6,5}, {  -4.15640000e-01,  -5.30190000e-01,   5.29200000e-02,  -7.15710000e-01,
                                            -1.10690000e-01,
                                            2.86620000e-01,   5.88200000e-02,   1.68760000e-01,  -2.55000000e-03,
                                            -1.00090000e-01,
                                            -4.88230000e-01,   4.84470000e-01,  -1.09150000e-01,  -1.46810000e-01,
                                            6.70320000e-01,
                                            1.00910000e-01,   4.35740000e-01,  -6.90500000e-01,  -3.61090000e-01,
                                            -4.38680000e-01,
                                            -5.48440000e-01,  -2.86950000e-01,  -4.23900000e-01,   5.78540000e-01,
                                            -2.10060000e-01,
                                            -4.42450000e-01,   4.56640000e-01,   5.48020000e-01,   3.32100000e-02,
                                            -5.40210000e-01,
                                            -6.36070000e-01,   5.57600000e-02,   3.28740000e-01,   3.81950000e-01,
                                            -4.21850000e-01,
                                            1.83740000e-01,  -1.36190000e-01,  -2.29380000e-01,  -5.11090000e-01,
                                            -2.06580000e-01,
                                            -4.81880000e-01,  -6.31100000e-01,   3.40000000e-04,  -1.35730000e-01,
                                            5.88210000e-01,
                                            2.25200000e-01,   4.30600000e-02,   9.08510000e-01,  -3.08940000e-01,
                                            1.51570000e-01,
                                            1.97510000e-01,  -7.26560000e-01,   1.05370000e-01,   1.10600000e-02,
                                            -5.79750000e-01,
                                            4.89620000e-01,  -2.24300000e-01,   5.31200000e-02,   6.92040000e-01,
                                            2.72560000e-01,
                                            -6.84450000e-01,  -5.18030000e-01,   2.92000000e-02,  -4.96740000e-01,
                                            -1.17970000e-01,
                                            4.25340000e-01,  -1.65500000e-02,  -2.82400000e-02,  -5.60180000e-01,
                                            1.93050000e-01,
                                            8.08800000e-02,   4.38260000e-01,  -2.48340000e-01,  -6.36220000e-01,
                                            2.37500000e-02,
                                            -6.10000000e-04,   3.00110000e-01,   1.17290000e-01,  -6.92400000e-02,
                                            -9.19220000e-01,
                                            5.41330000e-01,  -6.61130000e-01,  -2.86360000e-01,  -2.13500000e-02,
                                            -3.19580000e-01,
                                            2.25920000e-01,  -1.10170000e-01,   9.17020000e-01,  -1.71540000e-01,
                                            3.39100000e-02,
                                            -4.86810000e-01,  -2.32390000e-01,  -4.31500000e-01,   3.75290000e-01,
                                            4.98470000e-01,
                                            6.39700000e-02,  -4.04150000e-01,  -5.28310000e-01,   8.90000000e-02,
                                            -7.30460000e-01,
                                            -4.94030000e-01,   1.55540000e-01,  -3.46720000e-01,  -7.58460000e-01,
                                            5.20000000e-04,
                                            2.55960000e-01,   3.17040000e-01,  -3.47800000e-02,  -3.01860000e-01,
                                            -3.57600000e-02,
                                            1.31650000e-01,   7.57150000e-01,  -4.89030000e-01,   3.47710000e-01,
                                            -4.39400000e-02,
                                            -6.57270000e-01,   2.91000000e-01,   4.17280000e-01,   2.52880000e-01,
                                            -4.63400000e-01});

    nd4j::ops::svd op;
    auto results = op.execute({&x}, {}, {0, 1, 16});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto *s = results->at(0);
    auto *u = results->at(1);
    auto *v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    ASSERT_TRUE(expS.equalsTo(s));

    if(nd4j::Environment::getInstance()->isCPU()) {
        ASSERT_TRUE(expU.equalsTo(u));
        ASSERT_TRUE(expV.equalsTo(v));
    }
    else {
        for(uint i = 0; i < expU.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expU.e<float>(i)), nd4j::math::nd4j_abs(u->e<float>(i)), 1e-5);
        for(uint i = 0; i < expV.lengthOf(); ++i)
            ASSERT_NEAR(nd4j::math::nd4j_abs(expV.e<float>(i)), nd4j::math::nd4j_abs(v->e<float>(i)), 1e-5);
    }

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, svd_test11) {

    NDArray x('c', {2,2,3,3}, {0.7788, 0.8012, 0.7244, 0.2309, 0.7271, 0.1804, 0.5056, 0.8925, 0.5461, 0.9234, 0.0856, 0.7938, 0.6591, 0.5555,
                                0.1596, 0.3087, 0.1548, 0.4695, 0.7788, 0.8012, 0.7244, 0.2309, 0.7271, 0.1804, 0.5056, 0.8925, -0.5461, 0.9234,
                                0.0856, -0.7938, 0.6591, 0.5555, 0.1500, 0.3087, 0.1548, 0.4695});
    NDArray expS('c', {2,2,3}, {1.89671, 0.37095, 0.05525,1.51296, 0.52741, 0.17622, 1.69095, 0.90438, 0.24688,1.33551, 0.87475, 0.21571});
    NDArray expU('c', {2,2,3,3}, {6.9205e-01,  6.0147e-01, -3.9914e-01, 3.8423e-01, -7.7503e-01, -5.0170e-01, 6.1110e-01, -1.9384e-01,  7.6746e-01,
                                7.8967e-01,  4.5442e-01, -4.1222e-01, 4.9381e-01, -8.6948e-01, -1.2540e-02, 3.6412e-01,  1.9366e-01,  9.1100e-01,
                                7.1764e-01,  5.9844e-01,  3.5617e-01, 4.4477e-01, -3.1000e-04, -8.9564e-01, 5.3588e-01, -8.0116e-01,  2.6639e-01,
                                8.7050e-01, -4.2088e-01, -2.5513e-01, 4.8622e-01,  6.5499e-01,  5.7843e-01, 7.6340e-02,  6.2757e-01, -7.7481e-01});
    NDArray expV('c', {2,2,3,3}, {0.49383,  0.51614, -0.69981, 0.72718, -0.68641,  0.00688, 0.4768 ,  0.51228,  0.7143 , 0.77137, -0.17763,
                                  -0.6111 , 0.26324, -0.7852 ,  0.56051, 0.57939,  0.59322,  0.55892, 0.55149,  0.06737,  0.83146, 0.81413,
                                  -0.26072, -0.51887, 0.18182,  0.96306, -0.19863, 0.85948,  0.2707 , -0.4336 , 0.26688,  0.48582,  0.83232,
                                  -0.43596,  0.83108, -0.34531});

    nd4j::ops::svd op;
    auto results = op.execute({&x}, {}, {0, 1, 16});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto s = results->at(0);
    auto u = results->at(1);
    auto v = results->at(2);

    ASSERT_TRUE(expS.isSameShape(s));
    ASSERT_TRUE(expU.isSameShape(u));
    ASSERT_TRUE(expV.isSameShape(v));

    ASSERT_TRUE(expS.equalsTo(s));
    ASSERT_TRUE(expU.equalsTo(u));
    ASSERT_TRUE(expV.equalsTo(v));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, elu_test1) {

    auto x = NDArrayFactory::create<double>('c', {3,3}, {0.1, .2, .3, -.4,-.5,-.6, .7, .8, .9});
    auto exp = NDArrayFactory::create<double>('c', {3,3}, {.1, .2, .3, 0.5*-0.32968, 0.5*-0.393469, 0.5*-0.451188, .7, .8, .9});

    nd4j::ops::elu op;
    auto results = op.execute({&x}, {0.5}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto s = results->at(0);
    ASSERT_TRUE(exp.equalsTo(s));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, elu_bp_test1) {

    auto x = NDArrayFactory::create<double>('c', {3, 3}, {0.1, .2, .3, -.4, -.5, -.6, .7, .8, .9});
    auto eps = NDArrayFactory::create<double>('c', {3,3});
    eps.assign(2.);
    auto exp = NDArrayFactory::create<double>('c', {3, 3}, {2, 2, 2, 0.5*1.34064, 0.5*1.213061, 0.5*1.097623, 2, 2, 2});

    nd4j::ops::elu_bp op;
    auto results = op.execute({ &x, &eps }, {0.5}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto s = results->at(0);
    ASSERT_TRUE(exp.equalsTo(s));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, lrelu_test1) {

    auto x = NDArrayFactory::create<double>('c', {3,3}, {1, 2, 3, -4,-5,-6, 7, 8, 9});
    auto exp = NDArrayFactory::create<double>('c', {3,3}, {1, 2, 3, -0.8, -1., -1.2, 7, 8, 9});

    nd4j::ops::lrelu op;
    auto results = op.execute({&x}, {0.2}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto s = results->at(0);
    ASSERT_TRUE(exp.equalsTo(s));

    delete results;
}

TEST_F(DeclarableOpsTests3, lrelu_bp_test1) {

    auto x = NDArrayFactory::create<double>('c', {3,3}, {1, 2, 3, -4,-5,-6, 7, 8, 9});
    auto eps = NDArrayFactory::create<double>('c', {3,3}, {2,2,2,2,2,2,2, 2,2});
    auto exp = NDArrayFactory::create<double>('c', {3,3}, {2, 2, 2, 0.4, 0.4, 0.4, 2, 2, 2});

    nd4j::ops::lrelu_bp op;
    auto results = op.execute({&x, &eps}, {0.2}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto s = results->at(0);
    ASSERT_TRUE(exp.equalsTo(s));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, selu_test1) {

    auto x = NDArrayFactory::create<double>('c', {3,3}, {1, 2, 3, -4,-5,-6, 7, 8, 9});
    auto exp = NDArrayFactory::create<double>('c', {3,3}, {1.050701, 2.101402, 3.152103, -1.725899, -1.746253, -1.753742, 7.354907, 8.405608, 9.456309});

    nd4j::ops::selu op;
    auto results = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto s = results->at(0);
    ASSERT_TRUE(exp.equalsTo(s));

    delete results;
}

TEST_F(DeclarableOpsTests3, selu_test2) {

    auto x = NDArrayFactory::create<double>('c', {3,3}, {1, 2, 3, -4,-5,-6, 7, 8, 9});
//    auto expS = NDArrayFactory::create<double>('c', {3});
    auto eps = NDArrayFactory::create<double>('c', {3,3}, {2,2,2,2,2,2,2, 2,2});
    auto exp = NDArrayFactory::create<double>('c', {3,3}, {2.101401, 2.101402, 2.101402, 0.064401, 0.023692, 0.008716, 2.101402, 2.101402, 2.101402});

    nd4j::ops::selu_bp op;
    auto results = op.execute({&x, &eps}, {0.2}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    auto s = results->at(0);
//    auto u = results->at(1);
//    auto v = results->at(2);
//    s->printIndexedBuffer("SELU_BP");
    ASSERT_TRUE(exp.equalsTo(s));

    delete results;
}

TEST_F(DeclarableOpsTests3, EQScalarTests_1) {
    Graph graph;

    auto x  =     NDArrayFactory::create(1.0f);
    auto scalar = NDArrayFactory::create(1.0f);

    nd4j::ops::eq_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_TRUE(res);

}

TEST_F(DeclarableOpsTests3, EQScalarTests_2) {
    Graph graph;

    auto x  =     NDArrayFactory::create(2.0f);
    auto scalar = NDArrayFactory::create(1.0f);

    nd4j::ops::eq_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_FALSE(res);
}

TEST_F(DeclarableOpsTests3, GTScalarTests_1) {
    Graph graph;

    auto x  =     NDArrayFactory::create(1.0f);
    auto scalar = NDArrayFactory::create(1.0f);

    nd4j::ops::gt_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_FALSE(res);
}

TEST_F(DeclarableOpsTests3, GTScalarTests_2) {
    Graph graph;

    auto x  =     NDArrayFactory::create(2.0f);
    auto scalar = NDArrayFactory::create(1.0f);

    nd4j::ops::gt_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_TRUE(res);
}

TEST_F(DeclarableOpsTests3, GTEScalarTests_1) {
    Graph graph;

    auto x  =     NDArrayFactory::create(1.0f);
    auto scalar = NDArrayFactory::create(1.0f);

    nd4j::ops::gte_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_TRUE(res);
}

TEST_F(DeclarableOpsTests3, GTEScalarTests_2) {
    Graph graph;

    auto x  =     NDArrayFactory::create(2.0f);
    auto scalar = NDArrayFactory::create(1.0f);

    nd4j::ops::gte_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_TRUE(res);
}

TEST_F(DeclarableOpsTests3, GTEScalarTests_3) {
    Graph graph;

    auto x  =     NDArrayFactory::create(1.0f);
    auto scalar = NDArrayFactory::create(2.0f);

    nd4j::ops::gte_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_FALSE(res);
}

TEST_F(DeclarableOpsTests3, LTEScalarTests_1) {
    Graph graph;

    auto x  =     NDArrayFactory::create(1.0f);
    auto scalar = NDArrayFactory::create(1.0f);

    nd4j::ops::lte_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_TRUE(res);
}

TEST_F(DeclarableOpsTests3, LTEScalarTests_2) {
    Graph graph;

    auto x  =     NDArrayFactory::create(2.0f);
    auto scalar = NDArrayFactory::create(1.0f);

    nd4j::ops::lte_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_FALSE(res);
}

TEST_F(DeclarableOpsTests3, LTEScalarTests_3) {
    Graph graph;

    auto x  =     NDArrayFactory::create(1.0f);
    auto scalar = NDArrayFactory::create(2.0f);

    nd4j::ops::lte_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_TRUE(res);
}

TEST_F(DeclarableOpsTests3, NEQScalarTests_1) {
    Graph graph;

    auto x  =     NDArrayFactory::create(1.0f);
    auto scalar = NDArrayFactory::create(1.0f);

    nd4j::ops::neq_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_FALSE(res);

}

TEST_F(DeclarableOpsTests3, NEQScalarTests_2) {
    Graph graph;

    auto x  =     NDArrayFactory::create(2.0f);
    auto scalar = NDArrayFactory::create(1.0f);

    nd4j::ops::neq_scalar op;
    auto res = op.evaluate({&x, &scalar});
    ASSERT_TRUE(res);
}

TEST_F(DeclarableOpsTests3, NOOPTests_1) {
    Graph graph;

    auto x  =     NDArrayFactory::create(2.0f);
    auto scalar = NDArrayFactory::create(1.0f);

    nd4j::ops::noop op;
    auto res = op.execute({&x, &scalar}, {}, {});
    ASSERT_TRUE(res->status() == nd4j::Status::OK());
    delete res;
}
