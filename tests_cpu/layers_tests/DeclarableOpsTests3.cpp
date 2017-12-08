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

    std::vector<int> dims({1});
    nd4j::ops::norm<double> op;

    auto result0 = op.execute({&x}, {0}, {1});

    auto z0 = result0->at(0);
    auto exp0 = x.template reduceAlongDims<simdOps::NormFrobenius<double>>(dims, false);
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

