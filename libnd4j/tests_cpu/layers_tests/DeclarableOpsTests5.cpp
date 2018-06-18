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
    NDArray<float> x('c', {1, 60});
    NDArray<float> exp('c', {3, 5, 4}, {1.0, 6.0, 11.0, 16.0, 2.0, 7.0, 12.0, 17.0, 3.0, 8.0, 13.0, 18.0, 4.0, 9.0, 14.0, 19.0, 5.0, 10.0, 15.0, 20.0, 21.0, 26.0, 31.0, 36.0, 22.0, 27.0, 32.0, 37.0, 23.0, 28.0, 33.0, 38.0, 24.0, 29.0, 34.0, 39.0, 25.0, 30.0, 35.0, 40.0, 41.0, 46.0, 51.0, 56.0, 42.0, 47.0, 52.0, 57.0, 43.0, 48.0, 53.0, 58.0, 44.0, 49.0, 54.0, 59.0, 45.0, 50.0, 55.0, 60.0});
    NDArrayFactory<float>::linspace(1, x);
    x.reshapei('c', {3, 4, 5});

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {0, 2, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_0) {
    NDArray<float> x('c', {1, 60});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{0, 1, 2} shape");
//    x.printBuffer("{0, 1, 2} data");

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {0, 1, 2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_PermuteEquality_2) {
    NDArray<float> x('c', {1, 60});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {4, 3, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 21.0, 22.0, 23.0, 24.0, 25.0, 41.0, 42.0, 43.0, 44.0, 45.0, 6.0, 7.0, 8.0, 9.0, 10.0, 26.0, 27.0, 28.0, 29.0, 30.0, 46.0, 47.0, 48.0, 49.0, 50.0, 11.0, 12.0, 13.0, 14.0, 15.0, 31.0, 32.0, 33.0, 34.0, 35.0, 51.0, 52.0, 53.0, 54.0, 55.0, 16.0, 17.0, 18.0, 19.0, 20.0, 36.0, 37.0, 38.0, 39.0, 40.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{1, 0, 2} shape");
//    x.printBuffer("{1, 0, 2} data");

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {1, 0, 2});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_3) {
    NDArray<float> x('c', {1, 60});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {4, 5, 3}, {1.0, 21.0, 41.0, 2.0, 22.0, 42.0, 3.0, 23.0, 43.0, 4.0, 24.0, 44.0, 5.0, 25.0, 45.0, 6.0, 26.0, 46.0, 7.0, 27.0, 47.0, 8.0, 28.0, 48.0, 9.0, 29.0, 49.0, 10.0, 30.0, 50.0, 11.0, 31.0, 51.0, 12.0, 32.0, 52.0, 13.0, 33.0, 53.0, 14.0, 34.0, 54.0, 15.0, 35.0, 55.0, 16.0, 36.0, 56.0, 17.0, 37.0, 57.0, 18.0, 38.0, 58.0, 19.0, 39.0, 59.0, 20.0, 40.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{1, 2, 0} shape");
//    x.printBuffer("{1, 2, 0} data");

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {1, 2, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_4) {
    NDArray<float> x('c', {1, 60});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {5, 3, 4}, {1.0, 6.0, 11.0, 16.0, 21.0, 26.0, 31.0, 36.0, 41.0, 46.0, 51.0, 56.0, 2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 3.0, 8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0, 58.0, 4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0, 39.0, 44.0, 49.0, 54.0, 59.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{2, 0, 1} shape");
//    x.printBuffer("{2, 0, 1} data");

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {2, 0, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_PermuteEquality_5) {
    NDArray<float> x('c', {1, 60});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {5, 4, 3}, {1.0, 21.0, 41.0, 6.0, 26.0, 46.0, 11.0, 31.0, 51.0, 16.0, 36.0, 56.0, 2.0, 22.0, 42.0, 7.0, 27.0, 47.0, 12.0, 32.0, 52.0, 17.0, 37.0, 57.0, 3.0, 23.0, 43.0, 8.0, 28.0, 48.0, 13.0, 33.0, 53.0, 18.0, 38.0, 58.0, 4.0, 24.0, 44.0, 9.0, 29.0, 49.0, 14.0, 34.0, 54.0, 19.0, 39.0, 59.0, 5.0, 25.0, 45.0, 10.0, 30.0, 50.0, 15.0, 35.0, 55.0, 20.0, 40.0, 60.0});
    x.reshapei('c', {3, 4, 5});

//    x.printShapeInfo("{2, 1, 0} shape");
//    x.printBuffer("{2, 1, 0} data");

    nd4j::ops::permute<float> op;
    auto result = op.execute({&x}, {}, {2, 1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_TTS_bp_1) {
    NDArray<float> x('c', {2, 1, 3});
    NDArray<float> eps('c', {2, 4, 3});


    nd4j::ops::tile_to_shape_bp<float> op;
    auto result = op.execute({&x, &eps}, {}, {2, 4, 3});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_Rdiv_bp_1) {
    NDArray<float> x('c', {3, 1}, {1, 2, 3});
    NDArray<float> y('c', {1, 4}, {1, 2, 3, 4});
    NDArray<float> eps('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});


    nd4j::ops::reversedivide<float> op_ff;
    auto result_ff = op_ff.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result_ff->status());

    auto z_ff = result_ff->at(0);
    ASSERT_TRUE(eps.isSameShape(z_ff));

    nd4j::ops::reversedivide_bp<float> op_bp;
    auto result_bp = op_bp.execute({&x, &y, &eps}, {}, {});
    ASSERT_EQ(Status::OK(), result_bp->status());

    auto z_bp = result_bp->at(0);
    ASSERT_TRUE(x.isSameShape(z_bp));

    delete result_ff;
    delete result_bp;
}


TEST_F(DeclarableOpsTests5, Test_Boolean_diff_1) {
    NDArray<float> x('c', {1, 1}, {1.0f});
    NDArray<float> y(2.0f);

    nd4j::ops::less<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_1) {
    NDArray<float> x('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> paddings('c', {2, 2}, {0, 0, 0, 0});

    NDArray<float> exp('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::space_to_batch<float> op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    //z->printIndexedBuffer("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_1_1) {
    NDArray<float> x('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> exp('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::space_to_batch<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 0, 0, 0, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_2) {
    NDArray<float> x('c', {1, 2, 2, 1}, {1, 2, 3, 4});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> paddings('c', {2, 2}, {0, 0, 0, 0});

    NDArray<float> exp('c', {4, 1, 1, 1}, {1, 2, 3, 4});

    nd4j::ops::space_to_batch<float> op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_3) {
    NDArray<float> x('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> paddings('c', {2, 2}, {0, 0, 2, 0});

    NDArray<float> exp('c', {8, 1, 3, 1}, {0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12, 0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16});

    nd4j::ops::space_to_batch<float> op;
    auto result = op.execute({&x, &blocks, &paddings}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_SpaceToBatch_3_1) {
    NDArray<float> x('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    NDArray<float> exp('c', {8, 1, 3, 1}, {0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12, 0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16});

    nd4j::ops::space_to_batch<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 0, 0, 2, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_BatchToSpace_1) {
    NDArray<float> x('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> exp('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> crops('c', {2, 2}, {0, 0, 0, 0});

    nd4j::ops::batch_to_space<float> op;
    auto result = op.execute({&x, &blocks, &crops}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_BatchToSpace_1_1) {
    NDArray<float> x('c', {4, 1, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> exp('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::batch_to_space<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 0, 0, 0, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_BatchToSpace_2) {
    NDArray<float> x('c', {4, 1, 1, 1}, {1, 2, 3, 4});
    NDArray<float> exp('c', {1, 2, 2, 1}, {1, 2, 3, 4});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> crops('c', {2, 2}, {0, 0, 0, 0});

    nd4j::ops::batch_to_space<float> op;
    auto result = op.execute({&x, &blocks, &crops}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_BatchToSpace_3) {
    NDArray<float> x('c', {8, 1, 3, 1}, {0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12, 0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16});
    NDArray<float> exp('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    NDArray<float> blocks('c', {2}, {2, 2});
    NDArray<float> crops('c', {2, 2}, {0, 0, 2, 0});

    nd4j::ops::batch_to_space<float> op;
    auto result = op.execute({&x, &blocks, &crops}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests5, Test_BatchToSpace_3_1) {
    NDArray<float> x('c', {8, 1, 3, 1}, {0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12, 0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16});
    NDArray<float> exp('c', {2, 2, 4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

    nd4j::ops::batch_to_space<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 0, 0, 2, 0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test1) {
    
    NDArray<float> expected('c', {3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});

    nd4j::ops::eye<float> op;
    ResultSet<float>* results = op.execute({&expected}, {}, {99, 3});
    NDArray<float>* output = results->at(0);
    // output->printIndexedBuffer();
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test2) {
    
    NDArray<float> expected('c', {3, 4}, {1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0.});

    nd4j::ops::eye<float> op;
    ResultSet<float>* results = op.execute({&expected}, {}, {99, 3, 4});
    NDArray<float>* output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test3) {
    
    NDArray<float> expected('c', {2, 3, 4}, {1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0.});

    nd4j::ops::eye<float> op;
    ResultSet<float>* results = op.execute({&expected}, {}, {99, 3, 4, 2});
    NDArray<float>* output = results->at(0);
    // output->printIndexedBuffer();
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_test4) {
    
    NDArray<float> expected('c', {2, 2, 3, 4}, {1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0., 1.,  0.,  0.,  0., 0.,  1.,  0.,  0., 0.,  0.,  1.,  0.});

    nd4j::ops::eye<float> op;
    ResultSet<float>* results = op.execute({&expected}, {}, {99, 3, 4, 2, 2});
    NDArray<float>* output = results->at(0);
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test1) {

    NDArray<float> input('c', {4, 3, 2});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> indices('c', {2,2,1}, {3,2,3,2});

    NDArray<float> expected('c', {2,2,3,2}, {19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15, 16, 17, 18});

    nd4j::ops::gather_nd<float> op;
    ResultSet<float>* results = op.execute({&input, &indices}, {}, {});
    NDArray<float>* output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test2) {

    NDArray<float> input('c', {4, 3, 2});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> indices('c', {2,2,2}, {3,2,1,2, 0,1,0,1});

    NDArray<float> expected('c', {2,2,2}, {23, 24, 11, 12, 3,  4, 3,  4});

    nd4j::ops::gather_nd<float> op;
    ResultSet<float>* results = op.execute({&input, &indices}, {}, {});
    NDArray<float>* output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test3) {

    NDArray<float> input('c', {4, 3, 2});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> indices('c', {3}, {3,2,1});
    NDArray<float> expected(24.);

    nd4j::ops::gather_nd<float> op;
    ResultSet<float>* results = op.execute({&input, &indices}, {}, {});
    NDArray<float>* output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test4) {

    NDArray<float> input('c', {4, 3, 2});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> indices('c', {2,3}, {3,2,1,0,2,1});
    NDArray<float> expected('c',{2}, {24., 6});

    nd4j::ops::gather_nd<float> op;
    ResultSet<float>* results = op.execute({&input, &indices}, {}, {});
    NDArray<float>* output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test5) {

    NDArray<float> input('c', {4}, {1,2,3,4});
    NDArray<float> indices('c', {5,1}, {3,2,0,1,1});
    NDArray<float> expected('c',{5}, {4.,3,1,2,2});

    nd4j::ops::gather_nd<float> op;
    ResultSet<float>* results = op.execute({&input, &indices}, {}, {});
    NDArray<float>* output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, gatherNd_test6) {

    NDArray<float> input('c', {4}, {1,2,3,4});
    std::vector<Nd4jLong> shape = {1};
    NDArray<float> indices('c', shape, {2});
    NDArray<float> expected(3.);

    nd4j::ops::gather_nd<float> op;
    ResultSet<float>* results = op.execute({&input, &indices}, {}, {});
    NDArray<float>* output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test1) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {4}, {4,4,4,4});
    NDArray<float> exp('c', {3, 4, 5}, {4,  3,  2,  1,  5, 9,  8,  7,  6, 10, 14, 13, 12, 11, 15, 19, 18, 17, 16, 20, 24, 23, 22, 21, 25, 29, 28, 27, 26, 30, 34, 33, 32, 31, 35, 39, 38, 37, 36, 40, 44, 43, 42, 41, 45, 49, 48, 47, 46, 50, 54, 53, 52, 51, 55, 59, 58, 57, 56, 60});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {2, 1});
    NDArray<float>* output = results->at(0);
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test2) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {4}, {0,1,2,3});
    NDArray<float> exp('c', {3, 4, 5}, {1,  2,  3,  4,  5, 6,  7,  8,  9, 10, 12, 11, 13, 14, 15, 18, 17, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 31, 33, 34, 35, 38, 37, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 51, 53, 54, 55, 58, 57, 56, 59, 60});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {2, 1});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test3) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {3}, {2,3,4});
    NDArray<float> exp('c', {3, 4, 5}, {2,  1,  3,  4,  5, 7,  6,  8,  9, 10, 12, 11, 13, 14, 15, 17, 16, 18, 19, 20, 23, 22, 21, 24, 25, 28, 27, 26, 29, 30, 33, 32, 31, 34, 35, 38, 37, 36, 39, 40, 44, 43, 42, 41, 45, 49, 48, 47, 46, 50, 54, 53, 52, 51, 55, 59, 58, 57, 56, 60});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {2, 0});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test4) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {5}, {1, 2, 1, 2, 3});
    NDArray<float> exp('c', {3, 4, 5}, {1, 22,  3, 24, 45, 6, 27,  8, 29, 50, 11, 32, 13, 34, 55, 16, 37, 18, 39, 60, 21,  2, 23,  4, 25, 26,  7, 28,  9, 30, 31, 12, 33, 14, 35, 36, 17, 38, 19, 40, 41, 42, 43, 44,  5, 46, 47, 48, 49, 10, 51, 52, 53, 54, 15, 56, 57, 58, 59, 20});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {0, 2});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test5) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {5}, {1, 2, 4, 2, 3});
    NDArray<float> exp('c', {3, 4, 5}, {1,  7, 18,  9, 15, 6,  2, 13,  4, 10, 11, 12,  8, 14,  5, 16, 17,  3, 19, 20, 21, 27, 38, 29, 35, 26, 22, 33, 24, 30, 31, 32, 28, 34, 25, 36, 37, 23, 39, 40, 41, 47, 58, 49, 55, 46, 42, 53, 44, 50, 51, 52, 48, 54, 45, 56, 57, 43, 59, 60});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {1, 2});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

	delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test6) {
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> seqLengths('c', {4}, {1, 2, 3, 2});
    NDArray<float> exp('c', {3, 4, 5}, {1,  2,  3,  4,  5, 26, 27, 28, 29, 30, 51, 52, 53, 54, 55, 36, 37, 38, 39, 40, 21, 22, 23, 24, 25, 6,  7,  8,  9, 10, 31, 32, 33, 34, 35, 16, 17, 18, 19, 20, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 11, 12, 13, 14, 15, 56, 57, 58, 59, 60});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {0, 1});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test7) {
    
    NDArray<float> input('c', {1, 5});
    NDArrayFactory<float>::linspace(1, input);
    std::vector<float> data = {3};
    NDArray<float> seqLengths('c', {1}, data);    
    NDArray<float> exp('c', {1, 5}, {3, 2, 1, 4, 5});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {1, 0});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test8) {
    
    NDArray<float> input('c', {1, 5});
    NDArrayFactory<float>::linspace(1, input);
    std::vector<float> data = {1,0,1,0,1};
    NDArray<float> seqLengths('c', {5}, data);    
    NDArray<float> exp('c', {1, 5}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {0, 1});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test9) {
    
    NDArray<float> input('c', {5, 1});
    NDArrayFactory<float>::linspace(1, input);
    std::vector<float> data = {1,0,1,0,1};
    NDArray<float> seqLengths('c', {5}, data);    
    NDArray<float> exp('c', {5, 1}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {1, 0});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

	delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test10) {
    
    NDArray<float> input('c', {5, 1});
    NDArrayFactory<float>::linspace(1, input);
    std::vector<float> data = {3};
    NDArray<float> seqLengths('c', {1}, data);    
    NDArray<float> exp('c', {5, 1}, {3, 2, 1, 4, 5});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {0, 1});
    NDArray<float>* output = results->at(0);        
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test11) {
    
    NDArray<float> input('c', {1, 1, 5, 1});
    NDArrayFactory<float>::linspace(1, input);
    std::vector<float> data = {1, 0, 1, 0, 1};
    NDArray<float> seqLengths('c', {5}, data);    
    NDArray<float> exp('c', {1, 1, 5, 1}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {1, 2});
    NDArray<float>* output = results->at(0);        
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test12) {
    
    NDArray<float> input('c', {1, 1, 5, 1});
    NDArrayFactory<float>::linspace(1, input);
    std::vector<float> data = {3};
    NDArray<float> seqLengths('c', {1}, data);    
    NDArray<float> exp('c', {1, 1, 5, 1}, {3, 2, 1, 4, 5});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {2, 0});
    NDArray<float>* output = results->at(0);        
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, reverse_sequense_test13) {
    
    NDArray<float> input('c', {1, 1, 5, 1});
    NDArrayFactory<float>::linspace(1, input);
    std::vector<float> data = {1};
    NDArray<float> seqLengths('c', {1}, data);    
    NDArray<float> exp('c', {1, 1, 5, 1}, {1, 2, 3, 4, 5});

    nd4j::ops::reverse_sequence<float> op;
    ResultSet<float>* results = op.execute({&input, &seqLengths}, {}, {3, 0});
    NDArray<float>* output = results->at(0);        
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_0) {
    NDArray<float> x('c', {2, 6}, {1.0f, 1.0f, 1.0f, 1.0f, 11.0f, 3.0f, 1.0f, 1.0f, 1.0f, 14.0f, 5.0f, 6.0f});
    NDArray<float> expV('c', {2, 1}, {11.0f, 14.0f});
    NDArray<float> expI('c', {2, 1}, {4.0f, 3.0f});

    nd4j::ops::top_k<float> op;
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
        op.execute({&x}, {v, i}, {}, {1, 0}); // without sorting
    }
    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_1) {
    NDArray<float> x('c', {2, 3}, {1.0f, 11.0f, 3.0f, 14.0f, 5.0f, 6.0f});
    NDArray<float> expV('c', {2, 1}, {11.0f, 14.0f});
    NDArray<float> expI('c', {2, 1}, {1.0f, 0.0f});

    nd4j::ops::top_k<float> op;
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
        op.execute({&x}, {v, i}, {}, {1, 0}); // without sorting
    }
    delete result;
}

///////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_2) {
    NDArray<float> x('c', {2, 3, 4}, {11.0,  3.0, 14.0, 5.0,
                                      6.0,  9.0, 3.5, 7.0,
                                      21.0, 3.0, 14.0, 15.0,
                                      6.0, 9.0, 3.5, 7.0,
                                      11.0, 13.0, 14.0, 5.0,
                                      16.0, 9.0, 13.5, 7.0
                     }
    );
// <<<14.>,<9.>>, <<21.>,<9.>>, <<14.>,<16.>>>
    NDArray<float> expV('c', {2, 3, 1}, {14.0f, 9.0f,
                                         21.0f,
                                         9.0f, 14.0f,
                                         16.0f
                        }
    );

    NDArray<float> expI('c', {2, 3, 1 }, {2, 1, 0, 1, 2, 0});

    nd4j::ops::top_k<float> op;
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
    NDArray<float> x('c', {2, 3, 4}, {11.0,  3.0, 14.0, 5.0,
                                      6.0,  9.0, 3.5, 7.0,
                                      21.0, 3.0, 14.0, 15.0,
                                      6.0, 9.0, 3.5, 7.0,
                                      11.0, 13.0, 14.0, 5.0,
                                      16.0, 9.0, 13.5, 7.0
                     }
    );

    NDArray<float> expV('c', {2, 3, 2}, {14.0f, 11.0f, 9.0f,
                                         7.0f, 21.0f, 15.0f,
                                         9.0f, 7.0f, 14.0f,
                                         13.0f, 16.0f, 13.5f
                        }
    );

    NDArray<float> expI('c', {2, 3, 2 }, {2, 0, 1, 3, 0, 3, 1,  3, 2, 1, 0, 2});

    nd4j::ops::top_k<float> op;
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

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_4) {
    NDArray<float> x('c', {2, 3}, {1.0f, 11.0f, 3.0f, 14.0f, 5.0f, 6.0f});
    NDArray<float> expV('c', {2, 2}, {11.0f, 3.0f, 14.0f, 6.0f});
    NDArray<float> expI('c', {2, 2}, {1.0f, 2.0f, 0.0f, 2.0f});

    nd4j::ops::top_k<float> op;
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

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_TopK_5) {
    NDArray<float> x('f', {2, 3}, {1.0f, 11.0f, 3.0f, 14.0f, 5.0f, 6.0f});
    NDArray<float> expV('f', {2, 2}, {5.0f, 14.0f, 3.0f, 11.0f});
    NDArray<float> expI('f', {2, 2}, {2.0f, 1.0f, 1.0f, 0.0f});

    nd4j::ops::top_k<float> op;
    auto result = op.execute({&x}, {}, {2, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    NDArray<float>* v = result->at(0);
    NDArray<float>* i = result->at(1);

//    x.printShapeInfo("shape of the source X");
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

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_InTopK_1) {
    NDArray<float> x('c', {2, 3}, {1.0, 11.0, 3.0, 14.0, 5.0, 6.0});
    NDArray<float> y('c', {2}, {1, 1});
    NDArray<float> expV('c', {2}, {1, 0});

    nd4j::ops::in_top_k<float> op;
    auto result = op.execute({&x, &y}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto v = result->at(0);

    // v->printShapeInfo("InTopK: shape v");
    // expV.printShapeInfo("InTopK: shape expV");

    // v->printIndexedBuffer("v");
    // expV.printIndexedBuffer("expV");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_InTopK_2) {
    NDArray<float> x('c', {6, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );

    NDArray<float> y('c', {6}, {0, 0, 0, 0, 0, 0});
    NDArray<float> expV('c', {6}, {1, 0, 1, 0, 0, 1 });

    nd4j::ops::in_top_k<float> op;
    auto result = op.execute({&x, &y}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto v = result->at(0);

    // v->printShapeInfo("InTopK: shape v");
    // expV.printShapeInfo("InTopK: shape expV");

    // v->printIndexedBuffer("v");
    // expV.printIndexedBuffer("expV");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, Test_InTopK_3) {
    NDArray<float> x('f', {6, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );

    NDArray<float> y('f', {6}, {0, 0, 0, 0, 0, 0});
    NDArray<float> expV('f', {6}, {1, 0, 0, 0, 0, 0 });

    nd4j::ops::in_top_k<float> op;
    auto result = op.execute({&x, &y}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto v = result->at(0);

    // v->printShapeInfo("InTopK: shape v");
    // expV.printShapeInfo("InTopK: shape expV");

    // v->printIndexedBuffer("v");
    // expV.printIndexedBuffer("expV");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;
}

///////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, Test_Moments_1) {
    NDArray<float> x('c', {2, 3, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );

    NDArray<float> y('c', {3}, {0, 1, 2});
    //NDArray<float> expV('f', {6}, {1, 0, 0, 0, 0, 0 });

    float expMean = 9.395833f;
    float expDeviation = 22.4579f;
//Mean 9.395833
//Deviance 22.4579

    float inf = 1.e-5f;

    nd4j::ops::moments<float> op;
    auto result = op.execute({&x, &y}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(2, result->size());

    auto v = result->at(0);
    auto d = result->at(1);

//    v->printIndexedBuffer("Result is ");
//    d->printIndexedBuffer("Result is ");

    ASSERT_TRUE(v->isScalar());
    ASSERT_NEAR(expMean, (*v)(0), inf);
    ASSERT_NEAR(expDeviation, (*d)(0), inf);

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_Moments_2) {
    NDArray<float> x('c', {2, 3, 4}, {11.0f, 3.0f, 14.0f, 5.0f,
                                   6.0f, 9.0f, 3.5f, 7.0f,
                                   21.0f, 3.0f, 14.0f, 15.0f,
                                   6.0f, 9.0f, 3.5f, 7.0f,
                                   11.0f, 13.0f, 14.0f, 5.0f,
                                   16.0f, 9.0f, 13.5f, 7.0f}
    );

    NDArray<float> expV('c', {4}, {11.833333f, 7.6666665f, 10.416667f, 7.6666665f});
    NDArray<float> expD('c', {4}, {28.472221f, 12.888889f, 23.951387f, 11.555554f});

    nd4j::ops::moments<float> op;
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
    NDArray<float> x('c', {2, 3, 4}, {11.0, 3.0, 14.0, 5.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   21.0, 3.0, 14.0, 15.0,
                                   6.0, 9.0, 3.5, 7.0,
                                   11.0, 13.0, 14.0, 5.0,
                                   16.0, 9.0, 13.5, 7.0}
    );
    
    NDArray<float> expV('c', {3, 4}, { 8.5f, 6.f , 8.75f,  6.f, 
                                       8.5f, 11.f, 8.75f, 6.f, 
                                      18.5f, 6.f, 13.75f, 11.f});
    NDArray<float> expD('c', {3, 4}, { 6.25f, 9.f, 27.5625f,  1.f,
                                       6.25f, 4.f, 27.5625f,  1.f,
                                       6.25f, 9.f, 0.0625f,  16.f});

    nd4j::ops::moments<float> op;
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
//    NDArray<float> x('c', {2, 3, 4}, {11.0,  3.0,  14.0, 5.0,
//                                       6.0,  9.0,   3.5, 7.0,
//                                     21.0, 3.0, 14.0, 15.0,
//                                      6.0, 9.0,  3.5,  7.0,
//                                      11.0, 13.0, 14.0, 5.0,
//                                      16.0,  9.0, 13.5, 7.0}
//    );
//   the fortran ordered matrix the same as C-ordered above
//
    NDArray<float> x('f', {2, 3, 4}, {11.0f,  6.0f,  6.0f, 11.0f,
                                      21.0f, 16.0f,  3.0f,  9.0f,
                                       9.0f, 13.0f,  3.0f,  9.0f,
                                      14.0f,  3.5f,  3.5f, 14.0f,
                                      14.0f,  13.5f,  5.0f,  7.0f,
                                       7.0f,  5.0f, 15.0f,  7.0f
                                     }
    );


    NDArray<float> expV('c', {3, 4}, { 8.5f, 6.f , 8.75f,  6.f, 
                                       8.5f, 11.f, 8.75f, 6.f, 
                                      18.5f, 6.f, 13.75f, 11.f});
    NDArray<float> expD('c', {3, 4}, { 6.25f, 9.f, 27.5625f,  1.f,
                                       6.25f, 4.f, 27.5625f,  1.f,
                                       6.25f, 9.f, 0.0625f,  16.f});

    nd4j::ops::moments<float> op;
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
    
    NDArray<float> input('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> exp('c', {3}, {40, 120, 200});

    nd4j::ops::trace<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test2) {
    
    NDArray<float> input('c', {4, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> exp(40.);

    nd4j::ops::trace<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test3) {
    
    NDArray<float> input('c', {1, 5});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> exp(1.);

    nd4j::ops::trace<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test4) {
    
    NDArray<float> input('c', {5, 1});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> exp(1.);

    nd4j::ops::trace<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, trace_test5) {
    
    NDArray<float> input('c', {3, 4, 5, 6});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> exp('c', {3, 4}, {75,  225,  375,  525, 675,  825,  975, 1125, 1275, 1425, 1575, 1725});

    nd4j::ops::trace<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test1) {
    
    NDArray<float> input('c', {2, 2, 2});
    NDArrayFactory<float>::linspace(1, input);

    nd4j::ops::random_shuffle<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);    

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if((*output)(i) == (float)0.)
            haveZeros = true;
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test2) {
    
    NDArray<float> input('c', {1, 3, 2});
    NDArrayFactory<float>::linspace(1, input);    

    nd4j::ops::random_shuffle<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);        

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(input.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test3) {
    
    NDArray<float> input('c', {3, 2, 1});
    NDArrayFactory<float>::linspace(1, input);

    nd4j::ops::random_shuffle<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);        

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if((*output)(i) == (float)0.)
            haveZeros = true;
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test4) {
    NDArray<float> input('c', {4});
    NDArrayFactory<float>::linspace(1, input);

    nd4j::ops::random_shuffle<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);            

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if((*output)(i) == (float)0.)
            haveZeros = true;
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test5) {
        
    NDArray<float> input('c', {4,1});
    NDArrayFactory<float>::linspace(1, input);

    nd4j::ops::random_shuffle<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);                

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if((*output)(i) == (float)0.)
            haveZeros = true;
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test6) {
        
    NDArray<float> input('c', {4,1,1});
    NDArrayFactory<float>::linspace(1, input);

    nd4j::ops::random_shuffle<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);                    

    bool haveZeros = false;
    for(int i = 0; i < output->lengthOf(); ++i)
        if((*output)(i) == (float)0.)
            haveZeros = true;
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(!input.equalsTo(output));
    ASSERT_TRUE(!haveZeros);

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, random_shuffle_test7) {
        
    NDArray<float> input('c', {1,4});
    NDArrayFactory<float>::linspace(1, input);
    NDArray<float> exp('c', {1,4}, {1, 2, 3, 4});    

    nd4j::ops::random_shuffle<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);                    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(input.isSameShape(output));
    ASSERT_TRUE(input.equalsTo(output));

    delete results;
}

////////////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, EmbeddingLookup_1) {
    
    NDArray<float> x('c', {3, 4, 2}, {10, 20, 11, 21, 12, 22, 13, 23, 
                                      14, 24, 15, 25, 16, 26, 17, 27,
                                      18, 28, 19, 29, 20, 30, 21, 31});
    
    NDArray<float> y({1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 2.f, 2.f, 2.f});
    NDArray<float> exp('c', {9, 4, 2}, {14, 24, 15, 25, 16, 26, 17, 27, 14, 24, 15, 25,
                                        16, 26, 17, 27, 14, 24, 15, 25, 16, 26, 17, 27,
                                        10, 20, 11, 21, 12, 22, 13, 23, 10, 20, 11, 21,
                                        12, 22, 13, 23, 10, 20, 11, 21, 12, 22, 13, 23,
                                        18, 28, 19, 29, 20, 30, 21, 31, 18, 28, 19, 29,
                                        20, 30, 21, 31, 18, 28, 19, 29, 20, 30, 21, 31});

    // y.printShapeInfo("y shape");
    // y.printIndexedBuffer("y buffer");

    nd4j::ops::embedding_lookup<float> op;
    ResultSet<float>* result = op.execute({&x, &y}, {}, {0});
    NDArray<float>* output = result->at(0);    
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
    
    NDArray<float> x('c', {3, 4, 2}, {10, 20, 30, 40, 50, 60, 
                                      70, 80, 90, 10, 11, 12, 
                                      13, 14, 15, 16, 17, 18, 
                                      19, 20, 21, 22, 23, 24});
                    //1,   0,   1,   0,   1,   0
    NDArray<float> y({1.f, 0.f, 1.f, 0.f, 1.f, 0.f});
    NDArray<float> exp('c', {6, 4, 2}, {90, 10, 11, 12, 13, 14,
                                        15, 16, 10, 20, 30, 40,
                                        50, 60, 70, 80, 90, 10,
                                        11, 12, 13, 14, 15, 16,
                                        10, 20, 30, 40, 50, 60,
                                        70, 80, 90, 10, 11, 12,
                                        13, 14, 15, 16, 10, 20,
                                        30, 40, 50, 60, 70, 80});

    // y.printShapeInfo("y shape");
    // y.printIndexedBuffer("y buffer");

    nd4j::ops::embedding_lookup<float> op;
    ResultSet<float>* result = op.execute({&x, &y}, {}, {0});
    NDArray<float>* output = result->at(0);    
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

TEST_F(DeclarableOpsTests5, DynamicPartition_1) {
    
    NDArray<float> x('c', {3, 4, 2}, {10, 20, 11, 21, 12, 22, 
                                      13, 23, 14, 24, 15, 25, 16, 26, 17, 27,
                                      18, 28, 19, 29, 20, 30, 21, 31});
    
    NDArray<float> y('c', {3, 4, 2}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
                      2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 
                      1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f 
                    }
    );
/*    NDArray<float> y('c', {3, 4}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 
                      2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 2.f, 
                      1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f 
                    }
    );
*/
    int numPartition = 3;
    std::vector<NDArray<float>> exp( { NDArray<float>('c', {6}, {10, 20, 11, 21, 12, 22}), 
                                      NDArray<float>('c', {8}, {18, 28, 19, 29, 20, 30, 21, 31}),
                                      NDArray<float>('c', {10}, {13, 23, 14, 24, 15, 25, 16, 26, 17, 27})});

    nd4j::ops::dynamic_partition<float> op;
    ResultSet<float>* result = op.execute({&x, &y}, {}, {numPartition});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(result->size(), numPartition); // result has the same size as given param 4

    for (int e = 0; e < result->size(); e++) {
        NDArray<float>* output = result->at(e);
        // output->printShapeInfo("Output shape> ");
        // output->printIndexedBuffer("Output data> ");
        ASSERT_TRUE(exp[e].isSameShape(output));
        ASSERT_TRUE(exp[e].equalsTo(output));
    }

    delete result;
}

////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, DynamicPartition_2) {
    
    NDArray<float> x('c', {2, 4}, {0.1f, -1.f, 5.2f, 4.3f, -1.f, 7.4f, 0.0f, -2.2f});
    NDArray<float> y('c', {2, 4}, {1, 2, 1, 2, 1, 2, 3, 0});

    std::vector<NDArray<float>> exp( {NDArray<float>({-2.2f}),
                                      NDArray<float>('c', {3}, {0.1f, 5.2f, -1.f}),
                                      NDArray<float>('c', {3}, {-1.f, 4.3f, 7.4f}),
                                      NDArray<float>({0.0f})
                                     });

    nd4j::ops::dynamic_partition<float> op;
    int numPartition = 4;
    ResultSet<float>* result = op.execute({&x, &y}, {}, {numPartition});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(result->size(), numPartition); // result has the same size as given param 4

    for (int e = 0; e < result->size(); e++) {
        NDArray<float>* output = result->at(e);
        // output->printShapeInfo("Output shape> ");
        // exp[e].printShapeInfo("Expected shape> ");
        // output->printIndexedBuffer("Output data> ");

        ASSERT_TRUE(exp[e].isSameShape(output));
        ASSERT_TRUE(exp[e].equalsTo(output));
    }

    delete result;
}


TEST_F(DeclarableOpsTests5, DynamicPartition_3) {
    
    NDArray<float> x('c', {2, 4}, {0.1f, -1.f, 5.2f, 4.3f, -1.f, 7.4f, 0.0f, -2.2f});
    NDArray<float> y('c', {2, 4}, {0, 1, 0, 2, 0, 2, 3, 0});

    std::vector<NDArray<float>> exp( {NDArray<float>({0.1f, 5.2f, -1.f, -2.2f}),
                                      NDArray<float>({-1.f}),
                                      NDArray<float>({4.3f, 7.4f}),
                                      NDArray<float>({0.0f})
                                     });

    nd4j::ops::dynamic_partition<float> op;
    int numPartition = 4;
    ResultSet<float>* result = op.execute({&x, &y}, {}, {numPartition});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(result->size(), numPartition); // result has the same size as given param 4

    for (int e = 0; e < result->size(); e++) {
        NDArray<float>* output = result->at(e);
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

////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, DynamicStitch_1) {
    
    NDArray<float> x1({1.f, 3.f, 5.f, 0.f});
    NDArray<float> x2({2.f, 4.f});
    NDArray<float> y2({-1.f, -1.f});
    NDArray<float> y1({0.1f, 5.2f, 4.3f, 7.4f});

    
    NDArray<float> exp({7.4f, 0.1f, -1.f, 5.2f, -1.f, 4.3f});

    nd4j::ops::dynamic_stitch<float> op;
    ResultSet<float>* result = op.execute({&x1, &x2, &y1, &y2}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    // output->printShapeInfo("Output shape> ");
    // exp.printShapeInfo("Expected shape> ");
    // output->printIndexedBuffer("Output data> ");
    // exp.printIndexedBuffer("Expected res>");    
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////

TEST_F(DeclarableOpsTests5, DynamicStitch_2) {
    
    NDArray<float> x1({1.f, 3.f});
    NDArray<float> x2({5.f, 0.f, 2.f, 4.f});
    NDArray<float> y1({-1.f, -1.f});
    NDArray<float> y2({0.1f, 5.2f, 4.3f, 7.4f});

    
    NDArray<float> exp({5.2f, -1.f, 4.3f, -1.f, 7.4f, 0.1f});

    nd4j::ops::dynamic_stitch<float> op;
    ResultSet<float>* result = op.execute({&x1, &x2, &y1, &y2}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

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
    
    NDArray<double> x('c', {2, 2, 3, 4});
    NDArrayFactory<double>::linspace(1, x);
    NDArray<double> scale('c', {4});
    
    scale = 0.5;
    NDArray<double> offset('c', {4});
    offset = 2.;
    NDArray<double> expY('c', {2, 2, 3, 4}, {1.20337462,  1.20337462,  1.20337462,  1.20337462, 1.34821558,  1.34821558,  1.34821558,  1.34821558, 1.49305654,  1.49305654,  1.49305654,  1.49305654, 1.63789749,  1.63789749,  1.63789749,  1.63789749, 1.78273857,  1.78273857,  1.78273857,  1.78273857, 1.92757952,  1.92757952,  1.92757952,  1.92757952, 2.0724206 ,  2.0724206 ,  2.0724206 ,  2.0724206 , 2.21726155,  2.21726155,  2.21726155,  2.21726155, 2.36210251,  2.36210251,  2.36210251,  2.36210251, 2.50694346,  2.50694346,  2.50694346,  2.50694346, 2.65178442,  2.65178442,  2.65178442,  2.65178442, 2.79662538,  2.79662538,  2.79662538,  2.79662538});
    NDArray<double> expBatchMean('c', {4}, {23.,  24.,  25.,  26.});
    NDArray<double> expBatchVar('c', {4}, {208.00001526,  208.00001526,  208.00001526,  208.00001526});


    nd4j::ops::fused_batch_norm<double> op;
    ResultSet<double>* results = op.execute({&x, &scale, &offset}, {}, {0,1});
    NDArray<double>* y = results->at(0);    
    NDArray<double>* batchMean = results->at(1);
    NDArray<double>* batchVar = results->at(2);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test2) {
    
    NDArray<double> x('c', {2, 2, 3, 4});
    NDArrayFactory<double>::linspace(1, x);

    NDArray<double> scale('c', {4});
    
    scale = 0.5;
    NDArray<double> offset('c', {4});
    offset = 2.;
    NDArray<double> expY('c', {2, 2, 3, 4}, {1.20347691,  1.20347691,  1.20347691,  1.20347691, 1.34829926,  1.34829926,  1.34829926,  1.34829926, 1.49312162,  1.49312162,  1.49312162,  1.49312162, 1.6379441 ,  1.6379441 ,  1.6379441 ,  1.6379441 , 1.78276646,  1.78276646,  1.78276646,  1.78276646, 1.92758882,  1.92758882,  1.92758882,  1.92758882, 2.0724113 ,  2.0724113 ,  2.0724113 ,  2.0724113 , 2.21723366,  2.21723366,  2.21723366,  2.21723366, 2.36205602,  2.36205602,  2.36205602,  2.36205602, 2.50687838,  2.50687838,  2.50687838,  2.50687838, 2.65170074,  2.65170074,  2.65170074,  2.65170074, 2.79652309,  2.79652309,  2.79652309,  2.79652309});
    NDArray<double> expBatchMean('c', {4}, {23.,  24.,  25.,  26.});
    NDArray<double> expBatchVar('c', {4}, {208.00001526,  208.00001526,  208.00001526,  208.00001526});

    nd4j::ops::fused_batch_norm<double> op;
    ResultSet<double>* results = op.execute({&x, &scale, &offset}, {0.05}, {0,1});
    NDArray<double>* y = results->at(0);    
    NDArray<double>* batchMean = results->at(1);
    NDArray<double>* batchVar = results->at(2);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test3) {
    
    NDArray<double> x('c', {2, 4, 2, 3});
    NDArrayFactory<double>::linspace(1, x);
    
    NDArray<double> scale('c', {4});
    
    scale = 0.5;
    NDArray<double> offset('c', {4});
    offset = 2.;
    NDArray<double> expY('c', {2, 4, 2, 3}, {1.20337462,  1.20337462,  1.20337462,  1.20337462, 1.34821558,  1.34821558,  1.34821558,  1.34821558, 1.49305654,  1.49305654,  1.49305654,  1.49305654, 1.63789749,  1.63789749,  1.63789749,  1.63789749, 1.78273857,  1.78273857,  1.78273857,  1.78273857, 1.92757952,  1.92757952,  1.92757952,  1.92757952, 2.0724206 ,  2.0724206 ,  2.0724206 ,  2.0724206 , 2.21726155,  2.21726155,  2.21726155,  2.21726155, 2.36210251,  2.36210251,  2.36210251,  2.36210251, 2.50694346,  2.50694346,  2.50694346,  2.50694346, 2.65178442,  2.65178442,  2.65178442,  2.65178442, 2.79662538,  2.79662538,  2.79662538,  2.79662538});
    NDArray<double> expBatchMean('c', {4}, {23.,  24.,  25.,  26.});
    NDArray<double> expBatchVar('c', {4}, {208.00001526,  208.00001526,  208.00001526,  208.00001526});

    nd4j::ops::fused_batch_norm<double> op;
    ResultSet<double>* results = op.execute({&x, &scale, &offset}, {}, {1,1});
    NDArray<double>* y = results->at(0);    
    NDArray<double>* batchMean = results->at(1);
    NDArray<double>* batchVar = results->at(2);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test4) {
    
    NDArray<double> x('c', {2, 2, 3, 4});    
    NDArrayFactory<double>::linspace(1, x);
    std::vector<Nd4jLong> shape = {4};
    NDArray<double> scale('c', shape);    
    NDArray<double> offset('c', shape);
    NDArray<double> mean('c', shape);
    NDArray<double> variance('c', shape);
    
    scale = 0.5;    
    offset = 2.;
    mean = 25.;
    variance = 5.;

    NDArray<double> expY('c', {2, 2, 3, 4}, {-3.36602688, -3.14244223, -2.91885757, -2.6952734 , -2.47168875, -2.24810457, -2.02451992, -1.80093551, -1.57735109, -1.35376668, -1.13018227, -0.90659785, -0.68301344, -0.45942879, -0.23584437, -0.01225996, 0.21132445,  0.43490887,  0.65849328,  0.88207781, 1.10566223,  1.32924664,  1.55283117,  1.77641559, 2.        ,  2.22358441,  2.44716883,  2.67075348, 2.89433765,  3.11792231,  3.34150672,  3.56509113, 3.78867555,  4.01225996,  4.23584461,  4.45942879, 4.68301344,  4.90659809,  5.13018227,  5.35376644, 5.57735109,  5.80093575,  6.02451992,  6.24810457, 6.47168875,  6.6952734 ,  6.91885757,  7.14244223});
    NDArray<double> expBatchMean('c', shape, {0.,  0.,  0.,  0.});
    NDArray<double> expBatchVar('c', shape, {0.,  0.,  0.,  0.});


    nd4j::ops::fused_batch_norm<double> op;
    ResultSet<double>* results = op.execute({&x, &scale, &offset}, {}, {0,1});
    NDArray<double>* y = results->at(0);    
    NDArray<double>* batchMean = results->at(1);
    NDArray<double>* batchVar = results->at(2);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, fusedBatchNorm_test5) {
    
    NDArray<double> x('c', {2, 2, 3, 4});    
    NDArrayFactory<double>::linspace(1, x);
    std::vector<Nd4jLong> shape = {4};
    NDArray<double> scale('c', shape);    
    NDArray<double> offset('c', shape);
    NDArray<double> mean('c', shape);
    NDArray<double> variance('c', shape);
    
    scale = 0.5;    
    offset = 2.;
    mean = 25.;
    variance = 5.;

    NDArray<double> expY('c', {2, 2, 3, 4}, {-3.33992958e+00,  -3.11743259e+00,  -2.89493513e+00,  -2.67243814e+00, -2.44994116e+00,  -2.22744417e+00,  -2.00494719e+00,  -1.78244996e+00, -1.55995297e+00,  -1.33745599e+00,  -1.11495876e+00,  -8.92461777e-01, -6.69964790e-01,  -4.47467566e-01,  -2.24970579e-01,  -2.47359276e-03, 2.20023513e-01,   4.42520618e-01,   6.65017605e-01,   8.87514710e-01, 1.11001182e+00,   1.33250880e+00,   1.55500591e+00,   1.77750289e+00, 2.00000000e+00,   2.22249699e+00,   2.44499421e+00,   2.66749120e+00, 2.88998818e+00,   3.11248541e+00,   3.33498240e+00,   3.55747938e+00, 3.77997637e+00,   4.00247383e+00,   4.22497082e+00,   4.44746780e+00, 4.66996479e+00,   4.89246178e+00,   5.11495876e+00,   5.33745575e+00, 5.55995274e+00,   5.78244972e+00,   6.00494719e+00,   6.22744417e+00, 6.44994116e+00,   6.67243814e+00,   6.89493513e+00,   7.11743259e+00});
    NDArray<double> expBatchMean('c', shape, {0.,  0.,  0.,  0.});
    NDArray<double> expBatchVar('c', shape, {0.,  0.,  0.,  0.});


    nd4j::ops::fused_batch_norm<double> op;
    ResultSet<double>* results = op.execute({&x, &scale, &offset}, {0.05}, {0,1});
    NDArray<double>* y = results->at(0);    
    NDArray<double>* batchMean = results->at(1);
    NDArray<double>* batchVar = results->at(2);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expY.isSameShape(y));
    ASSERT_TRUE(expBatchMean.isSameShape(batchMean));
    ASSERT_TRUE(expBatchVar.isSameShape(batchVar));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, confusion_matrix_test1) {

    NDArray<float> labels('c', {1, 3}, {1, 2, 4});
    NDArray<float> predictions('c', {1, 3}, {2, 2, 4});
    NDArray<float> expected('c', {5, 5}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1});

    nd4j::ops::confusion_matrix<float> op;
    ResultSet<float>* results = op.execute({&labels, &predictions}, {}, {});
    NDArray<float>* output = results->at(0);
    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, confusion_matrix_test2) {

    NDArray<float> labels('c', {1, 2}, {1, 2});
    NDArray<float> predictions('c', {1, 2}, {0, 2});
    NDArray<float> expected('c', {3, 3}, {0, 0, 0, 1, 0, 0, 0, 0, 1});

    nd4j::ops::confusion_matrix<float> op;
    ResultSet<float> *results = op.execute({&labels, &predictions}, {}, {3});
    NDArray<float> *output = results->at(0);
    // output->printIndexedBuffer();


    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, confusion_matrix_test3) {

    NDArray<float> labels('c', {1, 2}, {1, 2});
    NDArray<float> predictions('c', {1, 2}, {0, 2});
    NDArray<float> weights('c', {1, 2}, {100, 200});
    NDArray<float> expected('c', {3, 3}, {0, 0, 0, 100, 0, 0, 0, 0, 200});

    nd4j::ops::confusion_matrix<float> op;
    ResultSet<float> *results = op.execute({&labels, &predictions, &weights}, {}, {3});
    NDArray<float> *output = results->at(0);
    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

///////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ZeroFraction_1) {
    
    NDArray<float> x('c', {3, 4, 2}, {0, 20, 30, 0, 50, 0, 
                                      70, 0, 90, 0, 11, 12, 
                                      13, 14, 15, 16, 17, 18, 
                                      19, 0, 21, 22, 23, 24});

    nd4j::ops::zero_fraction<float> op;
    ResultSet<float>* res = op.execute({&x}, {}, {});
    
    ASSERT_EQ(Status::OK(), res->status());
    ASSERT_TRUE(res->at(0)->isScalar());
    ASSERT_EQ(res->at(0)->getScalar(0), 0.25f);
    
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ZeroFraction_2) {
    
    NDArray<float> x('c', {2, 2, 2}, {5.5f, 0.f, 0.3f, 5.5f, 8.6f, 0.f, 0.f, 0.4f});

    nd4j::ops::zero_fraction<float> op;
    ResultSet<float>* res = op.execute({&x}, {}, {});
    
    ASSERT_EQ(Status::OK(), res->status());
    ASSERT_TRUE(res->at(0)->isScalar());
    ASSERT_EQ(res->at(0)->getScalar(0), 0.375f);
    
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ZeroFraction_3) {
    
    NDArray<float> x('f', {2, 2, 2}, {5.5f, 0.f, 0.3f, 5.5f, 8.6f, 0.f, 0.f, 0.4f});

    nd4j::ops::zero_fraction<float> op;
    ResultSet<float>* res = op.execute({&x}, {}, {});
    
    ASSERT_EQ(Status::OK(), res->status());
    ASSERT_TRUE(res->at(0)->isScalar());
    ASSERT_EQ(res->at(0)->getScalar(0), 0.375f);
    
    delete res;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, XWPlusB_1) {

    NDArray<float> x('c', {2,3}, { 1.f, 11.f,  3.f, 14.f,  5.f,  6.f});
    NDArray<float> y('c', {3,2}, { 11.f,  3.f, 4.f,  5.f, 6.f,  2.f});
    NDArray<float> b({100.f, 200.f});

    NDArray<float> exp('c', {2,2}, {173.f, 264.f, 310.f, 279.f});

    nd4j::ops::xw_plus_b<float> op;
    ResultSet<float>* result = op.execute({&x, &y, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

    // output->printShapeInfo("Output shape> ");
    // exp.printShapeInfo("Expected shape> ");
    // output->printIndexedBuffer("Output data> ");
    // exp.printIndexedBuffer("Expected res>");    

    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));

    delete result;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, StopGradient_1) {

    NDArray<float> x('c', {2,3}, { 1.f, 11.f,  3.f, 14.f,  5.f,  6.f});

    nd4j::ops::stop_gradient<float> op;
    ResultSet<float>* result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

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

    NDArray<float> x('f', {2,3}, { 1.f, 11.f,  3.f, 14.f,  5.f,  6.f});

    nd4j::ops::stop_gradient<float> op;
    ResultSet<float>* result = op.execute({&x}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    NDArray<float>* output = result->at(0);

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

    NDArray<double> input('c', {3, 3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14});
    NDArray<double> expOutput('c', {3, 3, 3}, {-2.16985e+00,-1.69846e-01,-3.16985e+00, -1.31507e+00,-6.31507e+00,-3.15072e-01, -8.00046e+00,-4.58767e-04,-9.00046e+00, -1.31327e+00,-1.23133e+01,-3.13266e-01, -1.40000e+01,-1.13743e-06,-1.50000e+01, -1.31326e+00,-1.83133e+01,-3.13262e-01, -2.00000e+01,-2.81941e-09,-2.10000e+01, -1.31326e+00,-2.43133e+01,-3.13262e-01, -2.73133e+01,-1.31326e+00,-3.13262e-01});

    nd4j::ops::log_softmax<double> op;
    ResultSet<double>*  results = op.execute({&input}, {}, {});
    NDArray<double>* z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test2) {

    NDArray<double> input('c', {3, 3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14});
    NDArray<double> expOutput('c', {3, 3, 3}, {-3.05095e+00,-3.04946e+00,-5.00705e+00, -5.09458e-02,-7.04946e+00,-7.04851e-03, -6.05095e+00,-4.94556e-02,-8.00705e+00, -3.04859e+00,-1.30000e+01,-3.04859e+00, -1.50486e+01,-2.37286e-06,-1.70486e+01, -4.85876e-02,-1.60000e+01,-4.85874e-02, -2.10000e+01,-3.04859e+00,-2.51269e+01, -7.96007e-10,-2.50486e+01,-2.12693e+00, -2.40000e+01,-4.85874e-02,-1.26928e-01});

    nd4j::ops::log_softmax<double> op;
    ResultSet<double>*  results = op.execute({&input}, {}, {1});
    NDArray<double>* z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test3) {

    NDArray<double> input('c', {3, 3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, -5,5 ,-6,6, -7,7, -8,8, -9,9, -10,10, -11,11, -12,12, -13,13, 14});
    NDArray<double> expOutput('c', {3, 3, 3}, {-2.16985e+00,-1.69846e-01,-3.16985e+00, -1.31507e+00,-6.31507e+00,-3.15072e-01, -8.00046e+00,-4.58767e-04,-9.00046e+00, -1.31327e+00,-1.23133e+01,-3.13266e-01, -1.40000e+01,-1.13743e-06,-1.50000e+01, -1.31326e+00,-1.83133e+01,-3.13262e-01, -2.00000e+01,-2.81941e-09,-2.10000e+01, -1.31326e+00,-2.43133e+01,-3.13262e-01, -2.73133e+01,-1.31326e+00,-3.13262e-01});

    nd4j::ops::log_softmax<double> op;
    ResultSet<double>*  results = op.execute({&input}, {}, {2});
    NDArray<double>* z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;

}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test5) {

    NDArray<double> input('c', {3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, 5});
    NDArray<double> expOutput('c', {3, 3}, {-2.16985, -0.16985, -3.16985, -1.31507, -6.31507, -0.31507, -9.31335, -1.31335, -0.31335});

    nd4j::ops::log_softmax<double> op;
    ResultSet<double>*  results = op.execute({&input}, {}, {});
    NDArray<double>* z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test6) {

    NDArray<double> input('c', {3, 3}, {-1, 1, -2, 2, -3, 3, -4, 4, 5});
    NDArray<double> expOutput('c', {3, 3}, {-3.05095,-3.04946,-7.12773, -0.05095,-7.04946,-2.12773, -6.05095,-0.04946,-0.12773});

    nd4j::ops::log_softmax<double> op;
    ResultSet<double>*  results = op.execute({&input}, {}, {0});
    NDArray<double>* z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test7) {

    NDArray<double> input('c', {1, 5}, {-1, 1, -2, 2, 3});
    NDArray<double> expOutput('c', {1, 5}, {-4.42414, -2.42414, -5.42414, -1.42414, -0.42414});

    nd4j::ops::log_softmax<double> op;
    ResultSet<double>*  results = op.execute({&input}, {}, {});
    NDArray<double>* z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test8) {

    NDArray<double> input('c', {1, 5}, {-1, 1, -2, 2, 3});
    NDArray<double> expOutput('c', {1, 5}, {0, 0, 0, 0, 0});

    nd4j::ops::log_softmax<double> op;
    ResultSet<double>*  results = op.execute({&input}, {}, {0});
    NDArray<double>* z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test9) {

    NDArray<double> input('c', {5, 1}, {-1, 1, -2, 2, 3});
    NDArray<double> expOutput('c', {5, 1}, {0, 0, 0, 0, 0});

    nd4j::ops::log_softmax<double> op;
    ResultSet<double>*  results = op.execute({&input}, {}, {});
    NDArray<double>* z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test10) {

    NDArray<double> input('c', {5, 1}, {-1, 1, -2, 2, 3});
    NDArray<double> expOutput('c', {5, 1}, {-4.42414, -2.42414, -5.42414, -1.42414, -0.42414});

    nd4j::ops::log_softmax<double> op;
    ResultSet<double>*  results = op.execute({&input}, {}, {0});
    NDArray<double>* z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_test11) {

    NDArray<double> input('c', {5}, {-1, 1, -2, 2, 3});
    NDArray<double> expOutput('c', {5}, {-4.42414, -2.42414, -5.42414, -1.42414, -0.42414});

    nd4j::ops::log_softmax<double> op;
    ResultSet<double>*  results = op.execute({&input}, {}, {});
    NDArray<double>* z = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expOutput.isSameShape(z));
    ASSERT_TRUE(expOutput.equalsTo(z));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_bp_test1) {

    NDArray<double> input  ('c', {2, 2}, {1,2,3,4});
    NDArray<double> epsilon('c', {2, 2}, {0.1, 0.2, 0.3, 0.4});    
    NDArray<double> exp('c', {2, 2}, {-0.07311,0.02689, -0.07311,0.02689});
    
    nd4j::ops::log_softmax_bp<double> op;
    ResultSet<double>*  results = op.execute({&input, &epsilon}, {}, {});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, log_softmax_bp_test2) {

    NDArray<double> input  ('c', {2, 2}, {1,2,3,4});
    NDArray<double> epsilon('c', {2, 2}, {0.1, 0.2, 0.3, 0.4});    
    NDArray<double> exp('c', {2, 2}, {-0.17616, -0.17616, 0.02384,  0.02384});
    
    nd4j::ops::log_softmax_bp<double> op;
    ResultSet<double>*  results = op.execute({&input, &epsilon}, {}, {0});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, ELU_1) {

    NDArray<double> input  ('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    NDArray<double> exp    ('c', {2, 2, 2}, { -0.63212055,  2. , 1.5, -0.753403, 1.,   2.,  2.,   1.});
    NDArray<double> res    ('c', {2, 2, 2});
    
    input.applyTransform<simdOps::ELU<double>>(&res);

    ASSERT_TRUE(res.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, L2_Loss_1) {

    NDArray<double> input  ('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    double exp(9.605);
    
    nd4j::ops::l2_loss<double> op;
    ResultSet<double>* results = op.execute({&input}, {}, {});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(output->isScalar());
    ASSERT_EQ((*output)(0), exp);    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, LogPoisonLoss_1) {

    NDArray<double> input  ('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    NDArray<double> targets('c', {2, 2, 2}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

    NDArray<double> exp('c', {2, 2, 2}, {1.3678794, 5.389056, 2.981689, 1.6465969, 1.7182817, 5.389056, 5.389056, 1.7182817});
    
    nd4j::ops::log_poison_loss<double> op;
    ResultSet<double>* results = op.execute({&targets, &input}, {}, {});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));    

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, LogPoisonLoss_2) {

    NDArray<double> input  ('c', {2, 2, 2}, { -1.,  2. , 1.5, -1.4, 1.,   2.,  2.,   1.});
    NDArray<double> targets('c', {2, 2, 2}, {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});

    NDArray<double> exp('c', {2, 2, 2}, {3.0196857, 4.0408626, 2.1334953, 3.6984034, 1.3700882, 4.0408626, 4.0408626, 1.3700882});
 
    nd4j::ops::log_poison_loss<double> op;
    ResultSet<double>* results = op.execute({&targets, &input}, {}, {1});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(output));
    ASSERT_TRUE(exp.equalsTo(output));    

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, NormalizeMoments_1) {

    NDArray<double> means  ('c', {2, 3, 4}, { 11.,   3.,  14.,   5.,
                                               6.,   9.,  3.5,   7.,
                                              21.,   3.,  14.,  15.,
                                               6.,   9.,  3.5,   7.,
                                              11.,  13.,  14.,   5.,
                                              16.,   9., 13.5,   7.});

    NDArray<double> deviance('c', {2, 3, 4}, { 21.,  13.,  24.,  15.,
                                               16.,  19., 13.5,  17.,
                                               31.,  13.,  24.,  25.,
                                               16.,  19., 13.5,  17.,
                                               21.,  23.,  24.,  15.,
                                               26.,  19., 23.5,  17.});

    NDArray<double> counts(2.0);

    NDArray<double> expMeans('c', {2, 3, 4}, {
                                                 5.5,   1.5,     7.,  2.5,
                                                  3.,   4.5,   1.75,  3.5,
                                                10.5,   1.5,     7.,  7.5,
                                                  3.,   4.5,   1.75,  3.5,
                                                 5.5,   6.5,     7.,  2.5,
                                                  8.,   4.5,   6.75,  3.5});

    NDArray<double> expDeviance('c', {2, 3, 4}, {
                                                -19.75,     4.25,       -37.,   1.25,
                                                   -1.,   -10.75,     3.6875,  -3.75,
                                                -94.75,     4.25,       -37.,  -43.75,
                                                   -1.,   -10.75,     3.6875,  -3.75,
                                                -19.75,   -30.75,       -37.,   1.25,
                                                  -51.,   -10.75,   -33.8125,  -3.75});

    nd4j::ops::normalize_moments<double> op;
    ResultSet<double>* results = op.execute({&counts, &means, &deviance}, {0.0}, {});

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_EQ(results->size(), 2);

    NDArray<double>* outputMeans = results->at(0);    
    NDArray<double>* outputDeviance = results->at(1);    

    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));    
    ASSERT_TRUE(expMeans.isSameShape(outputDeviance));
    ASSERT_TRUE(expDeviance.equalsTo(outputDeviance));    

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, NormalizeMoments_2) {

    NDArray<double> means  ('c', {3, 2, 4}, { 11.,   3.,  14.,   5.,
                                               6.,   9.,  3.5,   7.,
                                              21.,   3.,  14.,  15.,
                                               6.,   9.,  3.5,   7.,
                                              11.,  13.,  14.,   5.,
                                              16.,   9., 13.5,   7.});

    NDArray<double> deviance('c', {3, 2, 4}, { 21.,  13.,  24.,  15.,
                                               16.,  19., 13.5,  17.,
                                               31.,  13.,  24.,  25.,
                                               16.,  19., 13.5,  17.,
                                               21.,  23.,  24.,  15.,
                                               26.,  19., 23.5,  17.});

    NDArray<double> counts(12.0);

    NDArray<double> expMeans('c', {3, 2, 4}, { 0.9166667,     0.25,  1.1666667, 0.4166667,
                                                     0.5,     0.75,  0.2916667, 0.5833334,
                                                    1.75,     0.25,  1.1666667,      1.25,
                                                     0.5,     0.75,  0.2916667, 0.5833334,
                                               0.9166667, 1.0833334, 1.1666667, 0.4166667,
                                               1.3333334,      0.75,     1.125, 0.5833334});

    NDArray<double> expDeviance('c', {3, 2, 4}, {
                                                 0.9097222,  1.0208334,  0.6388887,  1.0763888,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                -0.4791665,  1.0208334,  0.6388887,  0.5208335,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                 0.9097222,  0.7430556,  0.6388887,  1.0763888,
                                                0.38888884,  1.0208334,  0.6927084,   1.076389});

    nd4j::ops::normalize_moments<double> op;
    ResultSet<double>* results = op.execute({&counts, &means, &deviance}, {0.0}, {});

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_EQ(results->size(), 2);

    NDArray<double>* outputMeans = results->at(0);    
    NDArray<double>* outputDeviance = results->at(1);    

    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));    
    ASSERT_TRUE(expMeans.isSameShape(outputDeviance));
    ASSERT_TRUE(expDeviance.equalsTo(outputDeviance));    

    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, NormalizeMoments_3) {

    NDArray<double> means  ('c', {3, 2, 4}, { 11.,   3.,  14.,   5.,
                                               6.,   9.,  3.5,   7.,
                                              21.,   3.,  14.,  15.,
                                               6.,   9.,  3.5,   7.,
                                              11.,  13.,  14.,   5.,
                                              16.,   9., 13.5,   7.});

    NDArray<double> deviance('c', {3, 2, 4}, { 21.,  13.,  24.,  15.,
                                               16.,  19., 13.5,  17.,
                                               31.,  13.,  24.,  25.,
                                               16.,  19., 13.5,  17.,
                                               21.,  23.,  24.,  15.,
                                               26.,  19., 23.5,  17.});

    NDArray<double> counts(12.0);
    double shift = 10.0;
    NDArray<double> expMeans('c', {3, 2, 4}, { 10.9166667,     10.25,  11.1666667, 10.4166667,
                                                     10.5,     10.75,  10.2916667, 10.5833334,
                                                    11.75,     10.25,  11.1666667,      11.25,
                                                     10.5,     10.75,  10.2916667, 10.5833334,
                                               10.9166667, 11.0833334, 11.1666667, 10.4166667,
                                               11.3333334,      10.75,     11.125, 10.5833334});

    NDArray<double> expDeviance('c', {3, 2, 4}, {
                                                 0.9097222,  1.0208334,  0.6388887,  1.0763888,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                -0.4791665,  1.0208334,  0.6388887,  0.5208335,
                                                 1.0833334,  1.0208334,  1.0399306,   1.076389,
                                                 0.9097222,  0.7430556,  0.6388887,  1.0763888,
                                                0.38888884,  1.0208334,  0.6927084,   1.076389});

    nd4j::ops::normalize_moments<double> op;
    ResultSet<double>* results = op.execute({&counts, &means, &deviance}, {shift}, {});

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_EQ(results->size(), 2);

    NDArray<double>* outputMeans = results->at(0);    
    NDArray<double>* outputDeviance = results->at(1);    

    ASSERT_TRUE(expMeans.isSameShape(outputMeans));
    ASSERT_TRUE(expMeans.equalsTo(outputMeans));    
    ASSERT_TRUE(expMeans.isSameShape(outputDeviance));
    ASSERT_TRUE(expDeviance.equalsTo(outputDeviance));    

    delete results;
}

