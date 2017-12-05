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






