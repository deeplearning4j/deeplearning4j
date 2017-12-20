#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests3 : public testing::Test {
public:
    
    DeclarableOpsTests3() {
        printf("\n");
        fflush(stdout);
    }
};


TEST_F(DeclarableOpsTests3, Test_Tile_1) {
    NDArray<float> x('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    NDArray<float> rep_vector('c', {1, 2}, {2, 2});
    std::vector<int> reps({2, 2});

    auto exp = x.tile(reps);

    nd4j::ops::tile<float> op;
    auto result = op.execute({&x, &rep_vector}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Tile_2) {
    NDArray<float> x('c', {2, 3}, {1, 2, 3, 4, 5, 6});
    std::vector<int> reps({2, 2});

    auto exp = x.tile(reps);

    nd4j::ops::tile<float> op;
    auto result = op.execute({&x}, {}, {2, 2});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Permute_1) {
    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> permute('c', {1, 3}, {0, 2, 1});
    NDArray<float> exp('c', {2, 4, 3});

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x, &permute}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Permute_2) {
    NDArray<float> x('c', {2, 3, 4});
    NDArray<float> exp('c', {4, 3, 2});

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Unique_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 1, 2, 3});
    NDArray<float> expV('c', {1, 3}, {1, 2, 3});
    NDArray<float> expI('c', {1, 3}, {0, 1, 4});

    nd4j::ops::unique<float> op;
    auto result = op.execute({&x}, {}, {});

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

TEST_F(DeclarableOpsTests3, Test_Rint_1) {
    NDArray<float> x('c', {1, 7}, {-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0});
    NDArray<float> exp('c', {1, 7}, {-2., -2., -0., 0., 2., 2., 2.});

    nd4j::ops::rint<float> op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Norm_1) {
    NDArray<double> x('c', {100, 100});
    NDArrayFactory<double>::linspace(1, x);

    std::vector<int> empty;
    std::vector<int> dims({1});
    nd4j::ops::norm<double> op;

    auto result0 = op.execute({&x}, {0}, {});

    auto z0 = result0->at(0);
    auto exp0 = x.template reduceAlongDims<simdOps::NormFrobenius<double>>(empty, false);
    ASSERT_TRUE(exp0.isSameShape(z0));
    ASSERT_TRUE(exp0.equalsTo(z0));

    delete result0;

    auto result1 = op.execute({&x}, {1}, {1});
    
    auto z1 = result1->at(0);
    auto exp1 = x.template reduceAlongDims<simdOps::Norm2<double>>(dims, false);
    ASSERT_TRUE(exp1.isSameShape(z1));
    ASSERT_TRUE(exp1.equalsTo(z1));

    delete result1;

    auto result4 = op.execute({&x}, {4}, {1});
    
    auto z4 = result4->at(0);
    auto exp4= x.template reduceAlongDims<simdOps::NormMax<double>>(dims, false);
    ASSERT_TRUE(exp4.isSameShape(z4));
    ASSERT_TRUE(exp4.equalsTo(z4));

    delete result4;
}


TEST_F(DeclarableOpsTests3, Test_Norm_2) {
    NDArray<double> x('c', {100, 100});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> axis('c', {1, 1}, {1});

    std::vector<int> empty;
    std::vector<int> dims({1});
    nd4j::ops::norm<double> op;

    auto result0 = op.execute({&x}, {0}, {});

    auto z0 = result0->at(0);
    auto exp0 = x.template reduceAlongDims<simdOps::NormFrobenius<double>>(empty, false);
    ASSERT_TRUE(exp0.isSameShape(z0));
    ASSERT_TRUE(exp0.equalsTo(z0));

    delete result0;

    auto result1 = op.execute({&x, &axis}, {1}, {});

    auto z1 = result1->at(0);
    auto exp1 = x.template reduceAlongDims<simdOps::Norm2<double>>(dims, false);
    ASSERT_TRUE(exp1.isSameShape(z1));
    ASSERT_TRUE(exp1.equalsTo(z1));

    delete result1;

    auto result4 = op.execute({&x, &axis}, {4}, {});

    auto z4 = result4->at(0);
    auto exp4= x.template reduceAlongDims<simdOps::NormMax<double>>(dims, false);
    ASSERT_TRUE(exp4.isSameShape(z4));
    ASSERT_TRUE(exp4.equalsTo(z4));

    delete result4;
}


TEST_F(DeclarableOpsTests3, Test_ClipByAvgNorm_1) { 
    NDArray<double> x('c', {2, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    NDArray<double> exp('c', {2, 3}, {-2.88, 0.0, 0.0, 3.84, 0.0, 0.0});

    nd4j::ops::clipbyavgnorm<double> op;
    auto result = op.execute({&x}, {0.8}, {});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_ClipByAvgNorm_2) { 
    NDArray<double> x('c', {2, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    NDArray<double> exp('c', {2, 3}, {-3, 0.0, 0.0, 4, 0.0, 0.0});

    nd4j::ops::clipbyavgnorm<double> op;
    auto result = op.execute({&x}, {0.9}, {});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_ClipByNorm_1) { 
    NDArray<double> x('c', {2, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    NDArray<double> exp('c', {2, 3}, {-2.4, 0.0, 0.0, 3.2, 0.0, 0.0});

    nd4j::ops::clipbynorm<double> op;
    auto result = op.execute({&x}, {4.0}, {});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_ClipByNorm_2) { 
    NDArray<double> x('c', {2, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0});
    NDArray<double> exp('c', {2, 3}, {-3.0, 0.0, 0.0, 4.0, 0.0, 0.0});

    nd4j::ops::clipbynorm<double> op;
    auto result = op.execute({&x}, {6.0}, {});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_CumSum_1) {
    NDArray<float> x('c', {1, 4}, {1, 2, 3, 4});
    NDArray<float> exp('c', {1, 4}, {1, 3, 6, 10});

    nd4j::ops::cumsum<float> op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_CumSum_2) {
    NDArray<float> x('c', {2, 4}, {1, 2, 3, 4, 1, 2, 3, 4});
    NDArray<float> exp('c', {2, 4}, {1, 3, 6, 10, 1, 3, 6, 10});

    nd4j::ops::cumsum<float> op;
    auto result = op.execute({&x}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_ListDiff_1) {
    NDArray<float> x('c', {1, 6}, {1, 2, 3, 4, 5, 6});
    NDArray<float> y('c', {1, 3}, {1, 3, 5});

    NDArray<float> exp0('c', {1, 3}, {2, 4, 6});
    NDArray<float> exp1('c', {1, 3}, {1, 3, 5});

    nd4j::ops::listdiff<float> op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z0 = result->at(0);
    auto z1 = result->at(1);

    ASSERT_TRUE(exp0.isSameShape(z0));
    ASSERT_TRUE(exp0.equalsTo(z0));

    ASSERT_TRUE(exp1.isSameShape(z1));
    ASSERT_TRUE(exp1.equalsTo(z1));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Range_1) {
    NDArray<float> start('c', {1, 1}, {2});
    NDArray<float> stop('c', {1, 1}, {0});
    NDArray<float> step('c', {1, 1}, {1});
    NDArray<float> exp('c', {1, 2}, {2, 1});

    nd4j::ops::range<float> op;
    auto result = op.execute({&start, &stop, &step}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Range_2) {
    NDArray<float> start('c', {1, 1}, {2});
    NDArray<float> stop('c', {1, 1}, {0});
    NDArray<float> step('c', {1, 1}, {-1});
    NDArray<float> exp('c', {1, 2}, {2, 1});

    nd4j::ops::range<float> op;
    auto result = op.execute({&start, &stop, &step}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Range_3) {
    NDArray<float> start('c', {1, 1}, {0});
    NDArray<float> stop('c', {1, 1}, {2});
    NDArray<float> step('c', {1, 1}, {1});
    NDArray<float> exp('c', {1, 2}, {0, 1});

    nd4j::ops::range<float> op;
    auto result = op.execute({&start, &stop, &step}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Range_4) {
    NDArray<float> exp('c', {1, 2}, {2, 1});

    nd4j::ops::range<float> op;
    auto result = op.execute({}, {2, 0, 1}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Range_5) {
    NDArray<float> exp('c', {1, 2}, {2, 1});

    nd4j::ops::range<float> op;
    auto result = op.execute({}, {2, 0, -1}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Range_6) {
    NDArray<float> exp('c', {1, 2}, {0, 1});

    nd4j::ops::range<float> op;
    auto result = op.execute({}, {0, 2, 1}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Range_7) {
    NDArray<float> exp('c', {1, 2}, {2, 1});

    nd4j::ops::range<float> op;
    auto result = op.execute({}, {}, {2, 0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Range_8) {
    NDArray<float> exp('c', {1, 2}, {2, 1});

    nd4j::ops::range<float> op;
    auto result = op.execute({}, {}, {2, 0, -1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Range_9) {
    NDArray<float> exp('c', {1, 2}, {0, 1});

    nd4j::ops::range<float> op;
    auto result = op.execute({}, {}, {0, 2, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests3, Test_Batched_Gemm_1) {
    NDArray<double> a('c', {1, 3}, {1, 1, 1});
    NDArray<double> b('c', {1, 3}, {0, 0, 0});
    NDArray<double> x('f', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<double> y('f', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto exp = NDArrayFactory<double>::mmulHelper(&x, &y);

    nd4j::ops::batched_gemm<double> op;
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
    NDArray<double> a('c', {1, 3}, {1, 1, 1});
    NDArray<double> b('c', {1, 3}, {0, 0, 0});
    NDArray<double> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<double> y('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto exp = NDArrayFactory<double>::mmulHelper(&x, &y);

    nd4j::ops::batched_gemm<double> op;
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
    NDArray<double> a('c', {1, 3}, {1, 1, 1});
    NDArray<double> b('c', {1, 3}, {0, 0, 0});
    NDArray<double> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<double> y('f', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto exp = NDArrayFactory<double>::mmulHelper(&x, &y);

    nd4j::ops::batched_gemm<double> op;
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
    NDArray<double> a('c', {1, 3}, {1, 1, 1});
    NDArray<double> b('c', {1, 3}, {0, 0, 0});
    NDArray<double> x('f', {5, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    NDArray<double> y('f', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    auto exp = NDArrayFactory<double>::mmulHelper(&x, &y);

    nd4j::ops::batched_gemm<double> op;
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
    NDArray<double> a('c', {1, 3}, {1, 1, 1});
    NDArray<double> b('c', {1, 3}, {0, 0, 0});
    NDArray<double> x('c', {5, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    NDArray<double> y('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    auto exp = NDArrayFactory<double>::mmulHelper(&x, &y);

    nd4j::ops::batched_gemm<double> op;
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
    NDArray<double> a('c', {1, 3}, {1, 1, 1});
    NDArray<double> b('c', {1, 3}, {0, 0, 0});
    NDArray<double> x('f', {2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    NDArray<double> y('f', {5, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

    auto exp = NDArrayFactory<double>::mmulHelper(&x, &y);

    nd4j::ops::batched_gemm<double> op;
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
    NDArray<double> a('c', {1, 3}, {1, 1, 1});
    NDArray<double> b('c', {1, 3}, {0, 0, 0});
    NDArray<double> x('c', {2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    NDArray<double> y('c', {5, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

    auto exp = NDArrayFactory<double>::mmulHelper(&x, &y);

    nd4j::ops::batched_gemm<double> op;
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

TEST_F(DeclarableOpsTests3, Test_Manual_Gemm_1) {
    NDArray<double> x('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12});
    NDArray<double> y('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12});
    NDArray<double> exp('f', {4, 4}, {38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0, 128.0, 152.0, 176.0, 200.0, 173.0, 206.0, 239.0, 272.0});

    nd4j::ops::matmul<double> op;
    auto result = op.execute({&x, &y}, {}, {1, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


    delete result;
}


TEST_F(DeclarableOpsTests3, Test_Manual_Gemm_2) {
    NDArray<double> x('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12});
    NDArray<double> y('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 12});
    NDArray<double> exp('f', {3, 3}, {70.0, 158.0, 246.0, 80.0, 184.0, 288.0, 90.0, 210.0, 330.0});

    nd4j::ops::matmul<double> op;
    auto result = op.execute({&x, &y}, {}, {0, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));


    delete result;
}

TEST_F(DeclarableOpsTests3, Test_ReverseDivide_1) {
    NDArray<float> x('c', {1, 3}, {2, 2, 2});
    NDArray<float> y('c', {1, 3}, {4, 6, 8});
    NDArray<float> exp('c', {1, 3}, {2, 3, 4});

    nd4j::ops::reversedivide<float> op;
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

    NDArray<double> xt  ('c', {batchSize, inSize});
    NDArray<double> ct_1('c', {batchSize, inSize});
    NDArray<double> w   ('c', {inSize, 3*inSize});    
    NDArray<double> b   ('c', {1, 2*inSize});

    xt.assign(1.);    
    ct_1.assign(2.);
    w.assign(0.5);
    b.assign(0.7);

    NDArray<double> expHt('c', {batchSize, inSize}, {0.96674103,0.96674103,0.96674103,0.96674103,0.96674103,0.96674103,0.96674103,0.96674103,0.96674103,0.96674103});
    NDArray<double> expCt('c', {batchSize, inSize}, {2.01958286,2.01958286,2.01958286,2.01958286,2.01958286, 2.01958286,2.01958286,2.01958286,2.01958286,2.01958286});

    nd4j::ops::sruCell<double> op;
    nd4j::ResultSet<double>* results = op.execute({&xt, &ct_1, &w, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *ht = results->at(0);
    NDArray<double> *ct = results->at(1);

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

    NDArray<double> xt  ('c', {batchSize, inSize});
    NDArray<double> ct_1('c', {batchSize, inSize});
    NDArray<double> w   ('c', {inSize, 3*inSize});    
    NDArray<double> b   ('c', {1, 2*inSize});

    xt.assign(1.);    
    ct_1.assign(2.);
    w.assign(0.5);
    b.assign(-1.);

    NDArray<double> expHt('c', {batchSize, inSize}, {0.97542038,0.97542038,0.97542038,0.97542038,0.97542038,0.97542038,0.97542038,0.97542038,0.97542038,0.97542038});
    NDArray<double> expCt('c', {batchSize, inSize}, {2.09121276,2.09121276,2.09121276,2.09121276,2.09121276,2.09121276,2.09121276,2.09121276,2.09121276,2.09121276});

    nd4j::ops::sruCell<double> op;
    nd4j::ResultSet<double>* results = op.execute({&xt, &ct_1, &w, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *ht = results->at(0);
    NDArray<double> *ct = results->at(1);

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

    NDArray<double> xt  ('c', {batchSize, inSize});
    NDArray<double> ct_1('c', {batchSize, inSize});
    NDArray<double> w   ('c', {inSize, 3*inSize});    
    NDArray<double> b   ('c', {1, 2*inSize});

    xt.assign(10.);    
    ct_1.assign(1.);
    w.assign(0.5);
    b.assign(-1.);

    NDArray<double> expHt('c', {batchSize, inSize}, {0.76159416,0.76159416,0.76159416,0.76159416,0.76159416,0.76159416,0.76159416,0.76159416,0.76159416,0.76159416});
    NDArray<double> expCt('c', {batchSize, inSize}, {1.,1.,1.,1.,1.,1.,1.,1.,1.,1.});

    nd4j::ops::sruCell<double> op;
    nd4j::ResultSet<double>* results = op.execute({&xt, &ct_1, &w, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *ht = results->at(0);
    NDArray<double> *ct = results->at(1);

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

    NDArray<double> xt  ('c', {batchSize, inSize});
    NDArray<double> ht_1('c', {batchSize, numUnits});
    NDArray<double> Wx  ('c', {inSize, 3*numUnits});
    NDArray<double> Wh  ('c', {numUnits, 3*numUnits});
    NDArray<double> b   ('c', {1, 3*numUnits});

    xt.assign(1.);    
    ht_1.assign(2.);
    Wx.assign(0.5);
    Wh.assign(0.5);
    b.assign(0.7);

    NDArray<double> expHt('c', {batchSize, numUnits}, {1.99993872,1.99993872,1.99993872,1.99993872,1.99993872,1.99993872,1.99993872,1.99993872});
    
    nd4j::ops::gruCell<double> op;
    nd4j::ResultSet<double>* results = op.execute({&xt, &ht_1, &Wx, &Wh, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *ht = results->at(0);    

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, gruCell_test2) {
    
    const int batchSize = 2;
    const int inSize    = 10;
    const int numUnits  = 4;

    NDArray<double> xt  ('c', {batchSize, inSize});
    NDArray<double> ht_1('c', {batchSize, numUnits});
    NDArray<double> Wx  ('c', {inSize, 3*numUnits});
    NDArray<double> Wh  ('c', {numUnits, 3*numUnits});
    NDArray<double> b   ('c', {1, 3*numUnits});

    xt.assign(1.);    
    ht_1.assign(0.);
    Wx.assign(1.5);
    Wh.assign(1.5);
    b.assign(-10);

    NDArray<double> expHt('c', {batchSize, numUnits}, {0.00669224,0.00669224,0.00669224,0.00669224,0.00669224,0.00669224,0.00669224,0.00669224});
    
    nd4j::ops::gruCell<double> op;
    nd4j::ResultSet<double>* results = op.execute({&xt, &ht_1, &Wx, &Wh, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *ht = results->at(0);    

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, gruCell_test3) {
    
    const int batchSize = 2;
    const int inSize    = 10;
    const int numUnits  = 4;

    NDArray<double> xt  ('c', {batchSize, inSize});
    NDArray<double> ht_1('c', {batchSize, numUnits});
    NDArray<double> Wx  ('c', {inSize, 3*numUnits});
    NDArray<double> Wh  ('c', {numUnits, 3*numUnits});
    NDArray<double> b   ('c', {1, 3*numUnits});

    xt.assign(1.);    
    ht_1.assign(0.);
    Wx.assign(0.1);
    Wh.assign(0.1);
    b.assign(1);

    NDArray<double> expHt('c', {batchSize, numUnits}, {0.1149149,0.1149149,0.1149149,0.1149149,0.1149149,0.1149149,0.1149149,0.1149149});
    
    nd4j::ops::gruCell<double> op;
    nd4j::ResultSet<double>* results = op.execute({&xt, &ht_1, &Wx, &Wh, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *ht = results->at(0);    

    ASSERT_TRUE(expHt.isSameShape(ht));
    ASSERT_TRUE(expHt.equalsTo(ht));

    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, invertPermutation_test1) {
        
    NDArray<double> input('c', {1, 8}, {5,2,7,4,6,3,1,0});
    NDArray<double> expected('c', {1, 8}, {7, 6, 1, 5, 3, 0, 4, 2});
    
    nd4j::ops::invert_permutation<double> op;
    nd4j::ResultSet<double>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, invertPermutation_test2) {
        
    NDArray<double> input('c', {1, 8}, {5,2,7,4,6,3,1,0});
    NDArray<double> expected('c', {1, 8}, {7, 6, 1, 5, 3, 0, 4, 2});
    
    nd4j::ops::invert_permutation<double> op;
    nd4j::ResultSet<double>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, invertPermutation_test3) {
        
    NDArray<double> input('c', {1, 8}, {1,2,0,4,6,3,5,7});
    NDArray<double> expected('c', {1, 8}, {2, 0, 1, 5, 3, 6, 4, 7});
    
    nd4j::ops::invert_permutation<double> op;
    nd4j::ResultSet<double>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test1) {
        
    NDArray<float> input('c', {3, 2});
    NDArrayFactory<float>::linspace(1, input);

    NDArray<float> expected('c', {3,2,3,2}, {1,0,0,0,0,0, 0,2,0,0,0,0, 0,0,3,0,0,0, 0,0,0,4,0,0, 0,0,0,0,5,0, 0,0,0,0,0,6});
    
    nd4j::ops::diag<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test2) {
        
    NDArray<float> input('c', {2, 3});
    NDArrayFactory<float>::linspace(1, input);

    NDArray<float> expected('c', {2,3,2,3}, {1,0,0,0,0,0, 0,2,0,0,0,0, 0,0,3,0,0,0, 0,0,0,4,0,0, 0,0,0,0,5,0, 0,0,0,0,0,6});

    nd4j::ops::diag<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test3) {
        
    NDArray<float> input('c', {1, 3});
    NDArrayFactory<float>::linspace(1, input);

    NDArray<float> expected('c', {3,3}, {1,0,0, 0,2,0, 0,0,3});

    nd4j::ops::diag<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test4) {
        
    NDArray<float> input('c', {3, 1});
    NDArrayFactory<float>::linspace(1, input);

    NDArray<float> expected('c', {3,3}, {1,0,0, 0,2,0, 0,0,3});

    nd4j::ops::diag<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test5) {
        
    NDArray<float> input('c', {1, 1});
    NDArrayFactory<float>::linspace(2, input);

    NDArray<float> expected('c', {1,1}, {2});

    nd4j::ops::diag<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diag_test6) {
        
    NDArray<float> input('c', {2,2,2});
    NDArrayFactory<float>::linspace(1, input);

    NDArray<float> expected('c', {2,2,2,2,2,2}, {1,0,0,0, 0,0,0,0, 0,2,0,0, 0,0,0,0, 0,0,3,0, 0,0,0,0, 0,0,0,4, 0,0,0,0, 0,0,0,0, 5,0,0,0, 0,0,0,0, 0,6,0,0, 0,0,0,0, 0,0,7,0, 0,0,0,0, 0,0,0,8});

    nd4j::ops::diag<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, matrixSetDiag_test1) {
        
    NDArray<float> input('c', {4,3,2});
    NDArray<float> diagonal('c', {4,2});
    input.assign(0.);
    diagonal.assign(1.);

    NDArray<float> expected('c', {4,3,2}, {1,0,0,1,0,0, 1,0,0,1,0,0, 1,0,0,1,0,0, 1,0,0,1,0,0});

    nd4j::ops::matrix_set_diag<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input, &diagonal}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, matrixSetDiag_test2) {
        
    NDArray<float> input('c', {1,1,2});
    NDArray<float> diagonal('c', {1,1});
    input.assign(0.);
    diagonal.assign(1.);

    NDArray<float> expected('c', {1,1,2}, {1,0});

    nd4j::ops::matrix_set_diag<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input, &diagonal}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, matrixSetDiag_test3) {
        
    NDArray<float> input('c', {2,1,4});
    NDArray<float> diagonal('c', {2,1});
    input.assign(0.);
    diagonal.assign(1.);

    NDArray<float> expected('c', {2,1,4}, {1,0,0,0,1,0,0,0});    

    nd4j::ops::matrix_set_diag<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input, &diagonal}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, matrixSetDiag_test4) {
        
    NDArray<float> input('c', {2,1,4,1});
    NDArray<float> diagonal('c', {2,1,1});
    input.assign(0.);
    diagonal.assign(1.);

    NDArray<float> expected('c', {2,1,4,1}, {1,0,0,0,1,0,0,0});    

    nd4j::ops::matrix_set_diag<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input, &diagonal}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diagPart_test1) {
        
    NDArray<float> input('c', {2,2});    
    NDArrayFactory<float>::linspace(1, input);
    
    NDArray<float> expected('c', {1,2}, {1,4});    

    nd4j::ops::diag_part<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);
    // output->printBuffer();

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diagPart_test2) {
        
    NDArray<float> input('c', {2,2,2,2});    
    NDArrayFactory<float>::linspace(1, input);
    
    NDArray<float> expected('c', {2,2}, {1,6,11,16});    

    nd4j::ops::diag_part<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, diagPart_test3) {
        
    NDArray<float> input('c', {2,2,2,2,2,2});    
    NDArrayFactory<float>::linspace(1, input);
    
    NDArray<float> expected('c', {2,2,2}, {1,10,19,28,37,46,55,64});    

    nd4j::ops::diag_part<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test1) {
        
    NDArray<float16> a('c', {3,3});    
    NDArray<float16> b('c', {3,3});    
    NDArray<float16> x('c', {3,3});    

    NDArrayFactory<float16>::linspace((float16)0.1, a, (float16)0.1);
    NDArrayFactory<float16>::linspace((float16)0.1, b, (float16)0.1);
    x.assign(0.1);
    
    NDArray<float16> expected('c', {3,3}, {0.40638509,0.33668978,0.28271242,0.23973916,0.20483276,0.17604725,0.15203027,0.13180567,0.114647});
                                           
    nd4j::ops::betainc<float16> op;
    nd4j::ResultSet<float16>* results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float16> *output = results->at(0);        

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-2));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test2) {
        
    NDArray<float> a('c', {3,3});    
    NDArray<float> b('c', {3,3});    
    NDArray<float> x('c', {3,3});    

    NDArrayFactory<float>::linspace(0.1, a, 0.1);
    NDArrayFactory<float>::linspace(0.1, b, 0.1);
    x.assign(0.1);
    
    NDArray<float> expected('c', {3,3}, {0.40638509,0.33668978,0.28271242,0.23973916,0.20483276,0.17604725,0.15203027,0.13180567,0.114647});

    nd4j::ops::betainc<float> op;
    nd4j::ResultSet<float>* results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test3) {
        
    NDArray<double> a('c', {3,3});    
    NDArray<double> b('c', {3,3});    
    NDArray<double> x('c', {3,3});    

    NDArrayFactory<double>::linspace(0.1, a, 0.1);
    NDArrayFactory<double>::linspace(0.1, b, 0.1);
    x.assign(0.1);
    
    NDArray<double> expected('c', {3,3}, {0.40638509,0.33668978,0.28271242,0.23973916,0.20483276,0.17604725,0.15203027,0.13180567,0.114647});

    nd4j::ops::betainc<double> op;
    nd4j::ResultSet<double>* results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test4) {
        
    NDArray<double> a('c', {3,3});    
    NDArray<double> b('c', {3,3});    
    NDArray<double> x('c', {3,3});    

    NDArrayFactory<double>::linspace(1, a);
    NDArrayFactory<double>::linspace(1, b);
    x.assign(0.1);
    
    NDArray<double> expected('c', {3,3}, {1.00000000e-01,2.80000000e-02,8.56000000e-03,2.72800000e-03,8.90920000e-04,2.95706080e-04,9.92854864e-05,3.36248880e-05,1.14644360e-05});

    nd4j::ops::betainc<double> op;
    nd4j::ResultSet<double>* results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-6));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test5) {
        
    NDArray<double> a('c', {3,3});    
    NDArray<double> b('c', {3,3});    
    NDArray<double> x('c', {3,3});    

    NDArrayFactory<double>::linspace(3200., a);
    NDArrayFactory<double>::linspace(3200., b);
    x.assign(0.1);
    
    NDArray<double> expected('c', {3,3}, {0.,0.,0.,0.,0.,0.,0.,0.,0.});

    nd4j::ops::betainc<double> op;
    nd4j::ResultSet<double>* results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-6));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test6) {
        
    NDArray<double> a('c', {3,3});    
    NDArray<double> b('c', {3,3});    
    NDArray<double> x('c', {3,3});    

    NDArrayFactory<double>::linspace(10., a);
    NDArrayFactory<double>::linspace(10., b);
    x.assign(0.1);
    
    NDArray<double> expected('c', {3,3}, {3.92988233e-06,1.35306497e-06,4.67576826e-07,1.62083416e-07,5.63356971e-08,1.96261318e-08,6.85120307e-09,2.39594668e-09,8.39227685e-10});

    nd4j::ops::betainc<double> op;
    nd4j::ResultSet<double>* results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-6));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test7) {
        
    NDArray<double> a('c', {3,3});    
    NDArray<double> b('c', {3,3});    
    NDArray<double> x('c', {3,3});    

    NDArrayFactory<double>::linspace(10., a);
    NDArrayFactory<double>::linspace(10., b);
    x.assign(0.9);
    
    NDArray<double> expected('c', {3,3}, {0.99999607,0.99999865,0.99999953,0.99999984,0.99999994,0.99999998,0.99999999,1.,1.});

    nd4j::ops::betainc<double> op;
    nd4j::ResultSet<double>* results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-6));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test8) {
        
    NDArray<double> a('c', {3,3});    
    NDArray<double> b('c', {3,3});    
    NDArray<double> x('c', {3,3});    

    NDArrayFactory<double>::linspace(10., a);
    NDArrayFactory<double>::linspace(10., b);
    x.assign(1.);
    
    NDArray<double> expected('c', {3,3}, {1.,1.,1.,1.,1.,1.,1.,1.,1.});

    nd4j::ops::betainc<double> op;
    nd4j::ResultSet<double>* results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output, 1e-6));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test9) {
        
    NDArray<double> a('c', {3,3});    
    NDArray<double> b('c', {3,3});    
    NDArray<double> x('c', {3,3});    

    NDArrayFactory<double>::linspace(10., a);
    NDArrayFactory<double>::linspace(10., b);
    x.assign(0.);
    
    NDArray<double> expected('c', {3,3}, {0.,0.,0.,0.,0.,0.,0.,0.,0.});

    nd4j::ops::betainc<double> op;
    nd4j::ResultSet<double>* results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, betainc_test10) {
        
    NDArray<double> a('c', {3,3});    
    NDArray<double> b('c', {3,3});    
    NDArray<double> x('c', {3,3});    

    NDArrayFactory<double>::linspace(10., a);
    NDArrayFactory<double>::linspace(10., b);
    x.assign(0.5);
    
    NDArray<double> expected('c', {3,3}, {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5});

    nd4j::ops::betainc<double> op;
    nd4j::ResultSet<double>* results = op.execute({&a, &b, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test1) {
        
    NDArray<double> x('c', {3,3});    
    NDArray<double> q('c', {3,3});        

    NDArrayFactory<double>::linspace(1., q);    
    x.assign(2.);
    
    NDArray<double> expected('c', {3,3}, {1.64493407,0.64493407,0.39493407,0.28382296,0.22132296,0.18132296,0.15354518,0.13313701,0.11751201});

    nd4j::ops::zeta<double> op;
    nd4j::ResultSet<double>* results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test2) {
        
    NDArray<double> x('c', {3,3});    
    NDArray<double> q('c', {3,3});        

    NDArrayFactory<double>::linspace(10., q);    
    x.assign(2.);
    
    NDArray<double> expected('c', {3,3}, {0.10516634,0.09516634,0.08690187,0.07995743,0.07404027,0.06893823,0.06449378,0.06058753,0.05712733});

    nd4j::ops::zeta<double> op;
    nd4j::ResultSet<double>* results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test3) {
        
    NDArray<double> x('c', {3,3});    
    NDArray<double> q('c', {3,3});        

    NDArrayFactory<double>::linspace(100., q);    
    x.assign(2.);
    
    NDArray<double> expected('c', {3,3}, {0.01005017,0.00995017,0.00985214,0.00975602,0.00966176,0.0095693 ,0.0094786 ,0.0093896 ,0.00930226});

    nd4j::ops::zeta<double> op;
    nd4j::ResultSet<double>* results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test4) {
        
    NDArray<double> x('c', {3,3});    
    NDArray<double> q('c', {3,3});        

    NDArrayFactory<double>::linspace(100., q);    
    x.assign(2.);
    
    NDArray<double> expected('c', {3,3}, {0.01005017,0.00995017,0.00985214,0.00975602,0.00966176,0.0095693 ,0.0094786 ,0.0093896 ,0.00930226});

    nd4j::ops::zeta<double> op;
    nd4j::ResultSet<double>* results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test5) {
        
    NDArray<double> x('c', {3,3});    
    NDArray<double> q('c', {3,3});        

    NDArrayFactory<double>::linspace(1., q);    
    x.assign(1.1);
    
    NDArray<double> expected('c', {3,3}, {10.58444846,9.58444846,9.11793197, 8.81927915,8.60164151,8.43137352, 8.29204706,8.17445116,8.07291961});

    nd4j::ops::zeta<double> op;
    nd4j::ResultSet<double>* results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test6) {
        
    NDArray<double> x('c', {3,3});    
    NDArray<double> q('c', {3,3});        

    NDArrayFactory<double>::linspace(1., q);    
    x.assign(1.01);
    
    NDArray<double> expected('c', {3,3}, {100.57794334,99.57794334,99.08139709, 98.75170576,98.50514758,98.30834069, 98.1446337 ,98.00452955,97.88210202});

    nd4j::ops::zeta<double> op;
    nd4j::ResultSet<double>* results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, zeta_test7) {
        
    NDArray<double> x('c', {3,3});    
    NDArray<double> q('c', {3,3});        

    NDArrayFactory<double>::linspace(1., q);    
    x.assign(10.);
    
    NDArray<double> expected('c', {3,3}, {1.00099458e+00,9.94575128e-04,1.80126278e-05,1.07754001e-06,1.23865693e-07,2.14656932e-08,4.92752156e-09,1.38738839e-09,4.56065812e-10});

    nd4j::ops::zeta<double> op;
    nd4j::ResultSet<double>* results = op.execute({&x, &q}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, polygamma_test1) {
        
    NDArray<double> n('c', {3,3});    
    NDArray<double> x('c', {3,3});        

    NDArrayFactory<double>::linspace(1., n);        
    x.assign(0.5);
    
    NDArray<double> expected('c', {3,3}, {4.93480220e+00,-1.68287966e+01, 9.74090910e+01,-7.71474250e+02, 7.69111355e+03,-9.22034579e+04, 1.29044022e+06,-2.06449000e+07, 3.71595452e+08});

    nd4j::ops::polygamma<double> op;
    nd4j::ResultSet<double>* results = op.execute({&n, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, polygamma_test2) {
        
    NDArray<double> n('c', {3,3});    
    NDArray<double> x('c', {3,3});        

    NDArrayFactory<double>::linspace(10., n);        
    NDArrayFactory<double>::linspace(0.5, x);        
    
    NDArray<double> expected('c', {3,3}, {-7.43182451e+09, 3.08334759e+05,-3.25669798e+03, 1.55186197e+02,-1.46220433e+01, 2.00905201e+00,-3.48791235e-01, 7.08016273e-02,-1.60476052e-02});

    nd4j::ops::polygamma<double> op;
    nd4j::ResultSet<double>* results = op.execute({&n, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests3, polygamma_test3) {
        
    NDArray<double> n('c', {3,3});    
    NDArray<double> x('c', {3,3});        

    NDArrayFactory<double>::linspace(1., n);        
    NDArrayFactory<double>::linspace(10., x);        
    
    NDArray<double> expected('c', {3,3}, {1.05166336e-01,-9.04983497e-03, 1.31009323e-03,-2.44459433e-04, 5.31593880e-05,-1.28049888e-05, 3.31755364e-06,-9.07408791e-07, 2.58758130e-07});

    nd4j::ops::polygamma<double> op;
    nd4j::ResultSet<double>* results = op.execute({&n, &x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *output = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

