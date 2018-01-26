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

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests5, eye_trest1) {
    
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
TEST_F(DeclarableOpsTests5, eye_trest2) {
    
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
TEST_F(DeclarableOpsTests5, eye_trest3) {
    
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
TEST_F(DeclarableOpsTests5, eye_trest4) {
    
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

    nd4j::ops::gatherNd<float> op;
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

    nd4j::ops::gatherNd<float> op;
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

    nd4j::ops::gatherNd<float> op;
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

    nd4j::ops::gatherNd<float> op;
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

    nd4j::ops::gatherNd<float> op;
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
    std::vector<int> shape = {1};
    NDArray<float> indices('c', shape, {2});
    NDArray<float> expected(3.);

    nd4j::ops::gatherNd<float> op;
    ResultSet<float>* results = op.execute({&input, &indices}, {}, {});
    NDArray<float>* output = results->at(0);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));

    delete results;
}

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

    v->printShapeInfo("topK_1: shape v");
    expV.printShapeInfo("topK_1: shape expV");

    i->printShapeInfo("topK_1: shape I");
    expI.printShapeInfo("topK_1: shape expI");

    v->printIndexedBuffer("topK_1: v");
    expV.printIndexedBuffer("topK_1: expV");
    i->printIndexedBuffer("topK_1: i");
    expI.printIndexedBuffer("topK_1: expI");


    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

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

    v->printShapeInfo("shape v");
    expV.printShapeInfo("shape expV");

    i->printShapeInfo("shape I");
    expI.printShapeInfo("shape expI");

    v->printIndexedBuffer("v");
    expV.printIndexedBuffer("expV");
    i->printIndexedBuffer("i");
    expI.printIndexedBuffer("expI");

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

    v->printShapeInfo("shape v");
    expV.printShapeInfo("shape expV");

    i->printShapeInfo("shape I");
    expI.printShapeInfo("shape expI");

    v->printIndexedBuffer("v");
    expV.printIndexedBuffer("expV");
    i->printIndexedBuffer("i");
    expI.printIndexedBuffer("expI");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

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

    v->printShapeInfo("shape v");
    expV.printShapeInfo("shape expV");

    i->printShapeInfo("shape I");
    expI.printShapeInfo("shape expI");

    v->printIndexedBuffer("v");
    expV.printIndexedBuffer("expV");
    i->printIndexedBuffer("i");
    expI.printIndexedBuffer("expI");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

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

    x.printShapeInfo("shape of the source X");
    v->printShapeInfo("shape v");
    expV.printShapeInfo("shape expV");

    i->printShapeInfo("shape I");
    expI.printShapeInfo("shape expI");

    v->printIndexedBuffer("v");
    expV.printIndexedBuffer("expV");
    i->printIndexedBuffer("i");
    expI.printIndexedBuffer("expI");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    ASSERT_TRUE(expI.isSameShape(i));
    ASSERT_TRUE(expI.equalsTo(i));

    delete result;
}

TEST_F(DeclarableOpsTests5, Test_InTopK_1) {
    NDArray<float> x('c', {2, 3}, {1.0, 11.0, 3.0, 14.0, 5.0, 6.0});
    NDArray<float> y('c', {2}, {1, 1});
    NDArray<float> expV('c', {2}, {1, 0});

    nd4j::ops::in_top_k<float> op;
    auto result = op.execute({&x, &y}, {}, {2});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    ASSERT_EQ(1, result->size());

    auto v = result->at(0);

    v->printShapeInfo("InTopK: shape v");
    expV.printShapeInfo("InTopK: shape expV");

    v->printIndexedBuffer("v");
    expV.printIndexedBuffer("expV");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;
}

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

    v->printShapeInfo("InTopK: shape v");
    expV.printShapeInfo("InTopK: shape expV");

    v->printIndexedBuffer("v");
    expV.printIndexedBuffer("expV");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;

}

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

    v->printShapeInfo("InTopK: shape v");
    expV.printShapeInfo("InTopK: shape expV");

    v->printIndexedBuffer("v");
    expV.printIndexedBuffer("expV");

    ASSERT_TRUE(expV.isSameShape(v));
    ASSERT_TRUE(expV.equalsTo(v));

    delete result;
}
