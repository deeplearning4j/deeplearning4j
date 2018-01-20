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