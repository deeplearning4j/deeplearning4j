#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests2 : public testing::Test {
public:
    
    DeclarableOpsTests2() {
        printf("\n");
    }
};

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, Gather_test_1) {
    
    NDArray<float> input   ('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray<float> indices ('c', {1,6},   {0,1, 2,2, 1,2});
    NDArray<float> expected('c', {2,6,4}, {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 9,10,11,12, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16, 17,18,19,20, 21,22,23,24, 21,22,23,24, 17,18,19,20, 21,22,23,24});

    nd4j::ops::gather<float> op;

    ResultSet<float>* result = op.execute({&input, &indices}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, Gather_test_2) {
    
    NDArray<float> input   ('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray<float> indices ('c', {1,1},   {2});
    NDArray<float> expected('c', {2,4}, {9,10,11,12,21,22,23,24});

    nd4j::ops::gather<float> op;

    ResultSet<float>* result = op.execute({&input, &indices}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, Gather_test_3) {
    
    NDArray<float> input   ('c', {2,3,4},   {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray<float> indices ('c', {2,3},     {0, 1, 2, 2, 1,2} );
    NDArray<float> expected('c', {2,2,3,4}, {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,  9,10,11,12, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24, 21,22,23,24,17,18,19,20,21,22,23,24});

    nd4j::ops::gather<float> op;

    ResultSet<float>* result = op.execute({&input, &indices}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, Gather_test_4) {
    
    NDArray<float> input   ('c', {3,3,4},   {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31,32,33,34,35,36});
    NDArray<float> indices ('c', {2,3},     {0, 1, 2, 2, 1,2} );
    NDArray<float> expected('c', {2,3,3,4}, {1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31,32,33,34,35,36, 25,26,27,28,29,30,31,32,33,34,35,36, 13,14,15,16,17,18,19,20,21,22,23,24, 25,26,27,28,29,30,31,32,33,34,35,36});

    nd4j::ops::gather<float> op;

    ResultSet<float>* result = op.execute({&input, &indices}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, Gather_test_5) {
    
    NDArray<float> input   ('c', {2,3,4},   {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    NDArray<float> indices ('c', {2,3},     {0, 1, 2, 2, 1,2} );
    NDArray<float> expected('c', {2,3,2,3}, {1, 2, 3, 3, 2, 3, 5, 6, 7, 7, 6, 7, 9,10,11,11,10,11, 13,14,15,15,14,15, 17,18,19,19,18,19, 21,22,23,23,22,23});

    nd4j::ops::gather<float> op;

    ResultSet<float>* result = op.execute({&input, &indices}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    ASSERT_TRUE(expected.isSameShapeStrict(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete result;
}


TEST_F(DeclarableOpsTests2, Test_FloorMod_1) {
    NDArray<float> x('c', {1, 3}, {2.0, 6.0, -3.0});
    NDArray<float> y('c', {1, 3}, {-3.0, 2.0, -2.0});
    NDArray<float> exp('c', {1, 3}, {-1.,  0., -1.,});

    nd4j::ops::floormod<float> op;
    
    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests2, Test_CRelu_1) {
    NDArray<float> x('c', {2, 2}, {1.0, 2.0, 3.0, 4.0});
    NDArray<float> exp('c', {2, 4}, {1.0, 2.0, 0, 0, 3.0, 4.0, 0, 0});

    nd4j::ops::crelu<float> op;

    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests2, Test_CRelu_BP_2) {
    NDArray<float> x('c', {2, 2}, {1.0, 2.0, -3.0, 4.0});
    NDArray<float> eps('c', {2, 4}, {1.0, 2.0, 4, 3, 3.0, 4.0, 2, 1});
    NDArray<float> exp('c', {2, 2}, {1, 2, -2, 4});

    nd4j::ops::crelu_bp<float> op;
    auto result = op.execute({&x, &eps}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests2, Test_Concat_BP_2) {
    NDArray<float> x('c', {2, 2});
    NDArray<float> y('c', {2, 2});
    NDArray<float> eps('c', {2, 4}, {1.0, 2.0, 0, 1, 3.0, 4.0, 0, 1});
    NDArray<float> expEX('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> expEY('c', {2, 2}, {0, 1, 0, 1});

    nd4j::ops::concat_bp<float> op;
    auto result = op.execute({&x, &y, &eps}, {}, {-1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto epsX = result->at(0);
    auto epsY = result->at(1);

    ASSERT_TRUE(expEX.isSameShape(epsX));
    ASSERT_TRUE(expEX.equalsTo(epsX));

    ASSERT_TRUE(expEY.isSameShape(epsY));
    ASSERT_TRUE(expEY.equalsTo(epsY));

    delete result;
}

 
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot5) {
    
    NDArray<float> x('c', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    NDArray<float> y('c', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    NDArray<float> expected('c', {2,4,2,4}, {44,110,160, 66,132, 38, 88,154, 68,170,224,102,204, 82,136,238, 92,230,288,138,276,126,184,322, 116,290,352,174,348,170,232,406, 76,190,160,114,228,182,152,266, 100,250,224,150,300,226,200,350, 124,310,288,186,372,270,248,434, 148,370,352,222,444,314,296,518});
                                             
    nd4j::ops::tensormmul<float> op;
    nd4j::ResultSet<float>* results = op.execute({&x, &y}, {}, {1,1,1,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot6) {
    
    NDArray<float> x('c', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    NDArray<float> y('f', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    NDArray<float> expected('c', {2,4,2,4}, {22, 66,110,154, 44, 88,132,176, 34,102,170,238, 68,136,204,272, 46,138,230,322, 92,184,276,368, 58,174,290,406,116,232,348,464, 38,114,190,266, 76,152,228,304, 50,150,250,350,100,200,300,400, 62,186,310,434,124,248,372,496, 74,222,370,518,148,296,444,592});
                                             
    nd4j::ops::tensormmul<float> op;
    nd4j::ResultSet<float>* results = op.execute({&x, &y}, {}, {1,1,1,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot7) {
    
    NDArray<float> x('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    NDArray<float> y('c', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    NDArray<float> expected('c', {2,4,2,4}, {76,166,112,106,196, 62,136,226, 60,174,208, 98,212,230,136,250, 76,214,336,122,260,174,168,306, 124,286,240,178,340,150,232,394, 100,226,176,142,268,106,184,310, 84,234,272,134,284,274,184,334, 100,274,400,158,332,218,216,390, 148,346,304,214,412,194,280,478});
                                             
    nd4j::ops::tensormmul<float> op;
    nd4j::ResultSet<float>* results = op.execute({&x, &y}, {}, {1,1,1,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot8) {
    
    NDArray<float> x('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    NDArray<float> y('f', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    NDArray<float> expected('c', {2,4,2,4}, {30, 90,150,210, 60,120,180,240, 38,114,190,266, 76,152,228,304, 46,138,230,322, 92,184,276,368, 54,162,270,378,108,216,324,432, 42,126,210,294, 84,168,252,336, 50,150,250,350,100,200,300,400, 58,174,290,406,116,232,348,464, 66,198,330,462,132,264,396,528});
                                             
    nd4j::ops::tensormmul<float> op;
    nd4j::ResultSet<float>* results = op.execute({&x, &y}, {}, {1,1,1,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot9) {
    
    NDArray<float> x('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    NDArray<float> y('f', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    NDArray<float> expected('c', {3,4,4,3}, {14, 14, 14, 30, 30, 30, 46, 46, 46, 62, 62, 62, 86, 86, 86,198,198,198,310,310,310,422,422,422, 62, 62, 62,142,142,142,222,222,222,302,302,302, 38, 38, 38, 86, 86, 86,134,134,134,182,182,182, 38, 38, 38, 86, 86, 86,134,134,134,182,182,182, 14, 14, 14, 30, 30, 30, 46, 46, 46, 62, 62, 62, 86, 86, 86,198,198,198,310,310,310,422,422,422, 62, 62, 62,142,142,142,222,222,222,302,302,302, 62, 62, 62,142,142,142,222,222,222,302,302,302, 38, 38, 38, 86, 86, 86,134,134,134,182,182,182, 14, 14, 14, 30, 30, 30, 46, 46, 46, 62, 62, 62, 86, 86, 86,198,198,198,310,310,310,422,422,422});
                                             
    nd4j::ops::tensormmul<float> op;
    nd4j::ResultSet<float>* results = op.execute({&x, &y}, {}, {1,0,1,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot10) {
    
    NDArray<float> x('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    NDArray<float> y('f', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    NDArray<float> expected('c', {4,4}, {114,258,402,546, 138,314,490,666, 162,370,578,786, 186,426,666,906});
                                             
    nd4j::ops::tensormmul<float> op;
    nd4j::ResultSet<float>* results = op.execute({&x, &y}, {}, {2,0,1, 2,0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot11) {
    
    NDArray<float> x('c', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    NDArray<float> y('f', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    NDArray<float> expected('c', {4,4}, {98,218,338,458, 134,302,470,638, 170,386,602,818, 206,470,734,998});
                                             
    nd4j::ops::tensormmul<float> op;
    nd4j::ResultSet<float>* results = op.execute({&x, &y}, {}, {2,0,1, 2,0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot12) {
    
    NDArray<float> x('c', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    NDArray<float> y('c', {2,4,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    NDArray<float> expected('c', {4,4}, {272,292,312,332, 368,396,424,452, 464,500,536,572, 560,604,648,692});
                                             
    nd4j::ops::tensormmul<float> op;
    nd4j::ResultSet<float>* results = op.execute({&x, &y}, {}, {2,0,1, 2,0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot13) {
    
    NDArray<float> x('c', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    NDArray<float> y('c', {4,2,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    NDArray<float> expected('c', {3,3}, {640,560,640, 576,624,576, 640,560,640});
                                             
    nd4j::ops::tensormmul<float> op;
    nd4j::ResultSet<float>* results = op.execute({&x, &y}, {}, {2,0,2, 2,1,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot14) {
    
    NDArray<float> x('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    NDArray<float> y('c', {4,2,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    NDArray<float> expected('c', {3,3}, {648,600,520, 648,536,648, 520,600,648});
                                             
    nd4j::ops::tensormmul<float> op;
    nd4j::ResultSet<float>* results = op.execute({&x, &y}, {}, {2,0,2, 2,1,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, TestTensorDot15) {
    
    NDArray<float> x('f', {2,3,4}, {1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15});
    NDArray<float> y('f', {4,2,3}, {2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16, 2,4,6,8,10,12,14,16});
    NDArray<float> expected('c', {3,3}, {624,624,624, 656,656,656, 624,624,624});
                                             
    nd4j::ops::tensormmul<float> op;
    nd4j::ResultSet<float>* results = op.execute({&x, &y}, {}, {2,0,2, 2,1,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_1) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,3,4,5});
    NDArray<float> expected('c', {2,3,4,5});
                                        
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);
    expected.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_2) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {1,1,4,5});
    NDArray<float> expected('c', {2,3,4,5});
                                        
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);
    expected.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_3) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {1,1,1,5});
    NDArray<float> expected('c', {2,3,4,5});
                                        
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);
    expected.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_4) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,1,1,5});
    NDArray<float> expected('c', {2,3,4,5});
                                        
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);
    expected.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_5) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {1,1});
    NDArray<float> expected('c', {2,3,4,5});
                                        
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);
    expected.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_6) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {1,1});
    NDArray<float> expected('c', {2,3,4,5});
                                        
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.);
    expected.assign(0.);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_7) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,3,4,5});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 60.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_8) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,3,4,5});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 0.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_9) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,1,4,1});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 60.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_10) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {1,1});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 60.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_11) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {1,1});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 1.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_12) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {1,1});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 0.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_13) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,3,4,5});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 1.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_14) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,3,4,5});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);
    weights(1) = 0.;
    weights(2) = 0.;

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 1.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_15) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,3,4,5});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(3, predictions);
    weights.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 2.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_16) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,3,4,5});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(3, predictions);
    weights.assign(0.5);
    predictions(0) = 0.;
    predictions(1) = 0.;
    predictions(2) = 0.;
    predictions(3) = 0.;

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 2.01667, 1e-5);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_17) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,3,4,5});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(3, predictions);
    weights.assign(0.5);
    predictions(0) = 0.;
    predictions(1) = 0.;
    predictions(2) = 0.;
    predictions(3) = 0.;
    labels(0) = 0.;
    labels(1) = 0.;
    labels(2) = 0.;
    labels(3) = 0.;

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 1.93333, 1e-5);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_18) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,1,1,5});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(3, predictions);
    weights.assign(0.5);
    predictions(0) = 0.;
    predictions(1) = 0.;
    predictions(2) = 0.;
    predictions(3) = 0.;
    labels(0) = 0.;
    labels(1) = 0.;
    labels(2) = 0.;
    labels(3) = 0.;

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 1.93333, 1e-5);

    delete results;

}


////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_19) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {1,1});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(3, predictions);
    weights.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_TRUE((*result)(0) == 1.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_20) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,3,4,5});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(3, predictions);
    weights.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_TRUE((*result)(0) == 1.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_21) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,3,1,1});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(3, predictions);
    weights.assign(0.5);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_TRUE((*result)(0) == 1.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_22) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {1,1});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(3, predictions);
    weights.assign(0.);

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_TRUE((*result)(0) == 0.);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, absoluteDifference_test_23) {
    
    NDArray<float> labels('c', {2,3,4,5});
    NDArray<float> predictions('c', {2,3,4,5});
    NDArray<float> weights('c', {2,3,4,5});
                                            
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(3, predictions);
    weights.assign(0.5);
    predictions(0) = 0.;
    predictions(1) = 0.;
    predictions(2) = 0.;
    predictions(3) = 0.;
    labels(0) = 0.;
    labels(1) = 0.;
    labels(2) = 0.;
    labels(3) = 0.;
    weights(40+0) = 0.;
    weights(40+1) = 0.;
    weights(40+2) = 0.;
    weights(40+3) = 0.;

    nd4j::ops::absoluteDifference<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());        
    ASSERT_NEAR((*result)(0), 0.965517, 1e-5);

    delete results;

}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosineDistance_test1) {
    
    NDArray<float> labels('c', {2,3,4});
    NDArray<float> predictions('c', {2,3,4});
    NDArray<float> weights('c', {1,3,4});
    NDArray<float> expected('c', {1,3,4}, {-91.5,-107.5,-125.5,-145.5, -167.5,-191.5,-217.5,-245.5, -275.5,-307.5,-341.5,-377.5});
                                        
    NDArrayFactory<float>::linspace(1, labels);
    NDArrayFactory<float>::linspace(2, predictions);
    weights.assign(0.5);    

    nd4j::ops::cosineDistance<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {0,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}
 
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosineDistance_test2) {
    
    NDArray<float> labels('c', {2,3,4});
    NDArray<float> predictions('c', {2,3,4});
    NDArray<float> weights('c', {2,1,4});
    NDArray<float> expected('c', {2,1,4}, {-3.25, -4., -4.75, -5.5,-12.25,-13.,-13.75,-14.5});
                                        
    NDArrayFactory<float>::linspace(1, labels);
    weights.assign(0.5);    
    predictions.assign(0.5);    

    nd4j::ops::cosineDistance<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {0,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;

}
 

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosineDistance_test3) {
    
    NDArray<float> labels('c', {2,3,4});
    NDArray<float> predictions('c', {2,3,4});
    NDArray<float> weights('c', {2,3,1});
    NDArray<float> expected('c', {2,3,1}, {-2., -6.,-10.,-14.,-18.,-22.});
                                        
    NDArrayFactory<float>::linspace(1, labels);
    weights.assign(0.5);    
    predictions.assign(0.5);    

    nd4j::ops::cosineDistance<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}
 
////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosineDistance_test4) {
    
    NDArray<float> labels('c', {2,3,4});
    NDArray<float> predictions('c', {2,3,4});
    NDArray<float> weights('c', {1,1});
    NDArray<float> expected('c', {2,3,1}, {-2., -6.,-10.,-14.,-18.,-22.});
                                        
    NDArrayFactory<float>::linspace(1, labels);
    weights.assign(0.5);    
    predictions.assign(0.5);    

    nd4j::ops::cosineDistance<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {0,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosineDistance_test5) {
    
    NDArray<float> labels('c', {2,3,4});
    NDArray<float> predictions('c', {2,3,4});
    NDArray<float> weights('c', {2,1,4});    
                                            
    NDArrayFactory<float>::linspace(1, labels);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosineDistance<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == -71.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosineDistance_test6) {
    
    NDArray<float> labels('c', {2,3,4});
    NDArray<float> predictions('c', {2,3,4});
    NDArray<float> weights('c', {1,1});    
                                            
    NDArrayFactory<float>::linspace(1, labels);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosineDistance<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {1,1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == -71.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosineDistance_test7) {
    
    NDArray<float> labels('c', {2,3,4});
    NDArray<float> predictions('c', {2,3,4});
    NDArray<float> weights('c', {1,1,4});    
                                            
    NDArrayFactory<float>::linspace(1, labels);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosineDistance<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {1,0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == -69.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosineDistance_test8) {
    
    NDArray<float> labels('c', {2,3,4});
    NDArray<float> predictions('c', {2,3,4});
    NDArray<float> weights('c', {2,3,1});    
                                            
    NDArrayFactory<float>::linspace(1, labels);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosineDistance<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {2,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == -24.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosineDistance_test9) {
    
    NDArray<float> labels('c', {2,3,4});
    NDArray<float> predictions('c', {2,3,4});
    NDArray<float> weights('c', {1,1});    
                                            
    NDArrayFactory<float>::linspace(1, labels);
    weights.assign(0.5);
    predictions.assign(0.5);

    nd4j::ops::cosineDistance<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {2,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == -24.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, cosineDistance_test10) {
    
    NDArray<float> labels('c', {2,3,4});
    NDArray<float> predictions('c', {2,3,4});
    NDArray<float> weights('c', {2,3,1});    
                                            
    NDArrayFactory<float>::linspace(1, labels);
    weights.assign(0.5);
    predictions.assign(0.5);
    weights(0) = 0.;
    weights(1) = 0.;

    nd4j::ops::cosineDistance<float> op;
    nd4j::ResultSet<float>* results = op.execute({&predictions, &weights, &labels}, {}, {2,2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == -32.);

    delete results;

}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test1) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {2,3,4});
    NDArray<float> expected('c', {2,3,4}, {1., 0. , 0., 2.5,0., 3.5, 0., 4.5,0., 5.5, 0., 6.5, 0., 7.5, 0., 8.5,0., 9.5,10., 0. ,0.,11.5, 0.,12.5});
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test2) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {1,1});
    NDArray<float> expected('c', {2,3,4}, {1., 0. , 0., 2.5,0., 3.5, 0., 4.5,0., 5.5, 0., 6.5, 0., 7.5, 0., 8.5,0., 9.5,10., 0. ,0.,11.5, 0.,12.5});
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test3) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {1,3,1});
    NDArray<float> expected('c', {2,3,4}, {1., 0. , 0., 2.5,0., 3.5, 0., 4.5,0., 5.5, 0., 6.5, 0., 7.5, 0., 8.5,0., 9.5,10., 0. ,0.,11.5, 0.,12.5});
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    
    // result->printBuffer();

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test4) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {2,3,4});    
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);
    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 83.);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test5) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {1,1});    
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);
    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 83.);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test6) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {2,1,1});    
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);
    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());
    ASSERT_TRUE((*result)(0) == 83.);

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test7) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {2,3,4});    
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);
    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 6.91667, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test8) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {1,1});    
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);
    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 6.91667, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test9) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {1,1,4});    
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);
    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 6.91667, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test10) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {2,3,4});    
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);
    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 3.45833, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test11) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {2,1,4});    
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);
    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 3.45833, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test12) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {2,3,4});    
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.5);
    weights(0) = 0.;
    weights(1) = 0.;
    weights(2) = 0.;
    weights(3) = 0.;
    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 3.975, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, hingeLoss_test13) {
    
    NDArray<float> labels('c', {2,3,4},{0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0});
    NDArray<float> logits('c', {2,3,4});
    NDArray<float> weights('c', {1,1});    
                                            
    NDArrayFactory<float>::linspace(1, logits);
    weights.assign(0.);    
    

    nd4j::ops::hingeLoss<float> op;
    nd4j::ResultSet<float>* results = op.execute({&logits, &weights, &labels}, {}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float> *result = results->at(0);    

    ASSERT_TRUE(result->isScalar());    
    ASSERT_TRUE((*result)(0) == 0.);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huberLoss_test1) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4});
    NDArray<double> expected('c', {2,3,4}, {0.0425 ,0.0875 ,0.13250001,0.17749999,0.22250001,0.26750001,0.31250003,0.35749999,0.4025    ,0.44749999,0.49249998,0.53750002, 0.58249998,0.6275    ,0.67250001,0.71749997,0.76249999,0.8075    ,0.85250002,0.89749998,0.9425    ,0.98749995,1.03250015,1.0775001});
                                            
    NDArrayFactory<double>::linspace(0.1, labels, 0.1);
    NDArrayFactory<double>::linspace(1, predictions);
    weights.assign(0.5);    

    nd4j::ops::huberLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {0.1}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huberLoss_test2) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,1}); 
    NDArray<double> expected('c', {2,3,4}, {0.0425 ,0.0875 ,0.13250001,0.17749999,0.22250001,0.26750001,0.31250003,0.35749999,0.4025    ,0.44749999,0.49249998,0.53750002, 0.58249998,0.6275    ,0.67250001,0.71749997,0.76249999,0.8075    ,0.85250002,0.89749998,0.9425    ,0.98749995,1.03250015,1.0775001});
                                            
    NDArrayFactory<double>::linspace(0.1, labels, 0.1);
    NDArrayFactory<double>::linspace(1, predictions);
    weights.assign(0.5);    

    nd4j::ops::huberLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {0.1}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huberLoss_test3) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {1,1}); 
    NDArray<double> expected('c', {2,3,4}, {0.0425 ,0.0875 ,0.13250001,0.17749999,0.22250001,0.26750001,0.31250003,0.35749999,0.4025    ,0.44749999,0.49249998,0.53750002, 0.58249998,0.6275    ,0.67250001,0.71749997,0.76249999,0.8075    ,0.85250002,0.89749998,0.9425    ,0.98749995,1.03250015,1.0775001});
                                            
    NDArrayFactory<double>::linspace(0.1, labels, 0.1);
    NDArrayFactory<double>::linspace(1, predictions);
    weights.assign(0.5);    

    nd4j::ops::huberLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {0.1}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huberLoss_test4) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4}); 
                                            
    NDArrayFactory<double>::linspace(0.1, labels, 0.1);
    NDArrayFactory<double>::linspace(1, predictions);
    weights.assign(0.5);    

    nd4j::ops::huberLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {0.1}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 13.44, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huberLoss_test5) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {1,1}); 
                                            
    NDArrayFactory<double>::linspace(0.1, labels, 0.1);
    NDArrayFactory<double>::linspace(1, predictions);
    weights.assign(0.5);    

    nd4j::ops::huberLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {0.1}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 13.44, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huberLoss_test6) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4}); 
                                            
    NDArrayFactory<double>::linspace(0.1, labels, 0.1);
    NDArrayFactory<double>::linspace(1, predictions);
    weights.assign(0.5);    

    nd4j::ops::huberLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {0.1}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 1.12, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huberLoss_test7) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,1,1}); 
                                            
    NDArrayFactory<double>::linspace(0.1, labels, 0.1);
    NDArrayFactory<double>::linspace(1, predictions);
    weights.assign(0.5);    

    nd4j::ops::huberLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {0.1}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 1.12, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huberLoss_test8) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4}); 
                                            
    NDArrayFactory<double>::linspace(0.1, labels, 0.1);
    NDArrayFactory<double>::linspace(1, predictions);
    weights.assign(0.5);    
    weights(0) = 0.;
    weights(1) = 0.;
    weights(2) = 0.;
    weights(3) = 0.;

    nd4j::ops::huberLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {0.1}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 1.3, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huberLoss_test9) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4}); 
                                            
    NDArrayFactory<double>::linspace(0.1, labels, 0.1);
    NDArrayFactory<double>::linspace(1, predictions);
    weights.assign(0.5);    

    nd4j::ops::huberLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {0.1}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 0.56, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huberLoss_test10) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {1,1}); 
                                            
    NDArrayFactory<double>::linspace(0.1, labels, 0.1);
    NDArrayFactory<double>::linspace(1, predictions);
    weights.assign(0.5);    

    nd4j::ops::huberLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {0.1}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 0.56, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, huberLoss_test11) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4}); 
                                            
    NDArrayFactory<double>::linspace(0.1, labels, 0.1);
    NDArrayFactory<double>::linspace(1, predictions);
    weights.assign(0.5);    
    weights(0) = 0.;
    weights(1) = 0.;
    weights(2) = 0.;
    weights(3) = 0.;

    nd4j::ops::huberLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {0.1}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), 0.65, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test1) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4}); 
    NDArray<double> expected('c', {2,3,4}, {1.60943663,  2.48403668,  3.05256081,  3.40363169,  3.57730675,  3.59525585,  3.46986699,  3.20791793,  2.81228209,  2.28273821,  1.61630058,  0.80721998, -0.15329313, -1.27764463, -2.5828433 , -4.09208679, -5.83734226, -7.8636713 ,-10.23689461,-13.05822182,-16.49509811,-20.85659218,-26.82411766,-36.52717209});
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    

    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test2) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,1,4}); 
    NDArray<double> expected('c', {2,3,4}, {1.60943663,  2.48403668,  3.05256081,  3.40363169,  3.57730675,  3.59525585,  3.46986699,  3.20791793,  2.81228209,  2.28273821,  1.61630058,  0.80721998, -0.15329313, -1.27764463, -2.5828433 , -4.09208679, -5.83734226, -7.8636713 ,-10.23689461,-13.05822182,-16.49509811,-20.85659218,-26.82411766,-36.52717209});
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    

    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}
  
///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test3) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {1,1}); 
    NDArray<double> expected('c', {2,3,4}, {1.60943663,  2.48403668,  3.05256081,  3.40363169,  3.57730675,  3.59525585,  3.46986699,  3.20791793,  2.81228209,  2.28273821,  1.61630058,  0.80721998, -0.15329313, -1.27764463, -2.5828433 , -4.09208679, -5.83734226, -7.8636713 ,-10.23689461,-13.05822182,-16.49509811,-20.85659218,-26.82411766,-36.52717209});
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    

    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete results;
}
  
///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test4) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4}); 
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    

    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), -113.886429, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test5) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {1,3,1}); 
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    

    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), -113.886429, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test6) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {1,1}); 
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    

    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), -113.886429, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test7) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4}); 
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    

    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), -9.490536, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test8) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {1,3,1}); 
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    

    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), -9.490536, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test9) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {1,1}); 
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    

    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), -9.490536, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test10) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4}); 
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    
    weights(0) = 0.;
    weights(1) = 0.;
    weights(2) = 0.;
    weights(3) = 0.;

    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), -12.443609, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test11) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4}); 
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    
 
    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), -4.745268, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test12) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {1,1}); 
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    
 
    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), -4.745268, 1e-5);    

    delete results;
}

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests2, logLoss_test13) {
    
    NDArray<double> labels('c', {2,3,4});
    NDArray<double> predictions('c', {2,3,4});
    NDArray<double> weights('c', {2,3,4}); 
                                            
    NDArrayFactory<double>::linspace(0.04, predictions, 0.04);
    NDArrayFactory<double>::linspace(1, labels);
    weights.assign(0.5);    
    weights(0) = 0.;
    weights(1) = 0.;
    weights(2) = 0.;
    weights(3) = 0.;
 
    nd4j::ops::logLoss<double> op;
    nd4j::ResultSet<double>* results = op.execute({&predictions, &weights, &labels}, {1e-7}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *result = results->at(0);            

    ASSERT_TRUE(result->isScalar());    
    ASSERT_NEAR((*result)(0), -6.221805, 1e-5);    

    delete results;
}
