//
//  @author raver119@gmail.com
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <helpers/helper_hash.h>
#include <NDArray.h>
#include <array/NDArrayList.h>


using namespace nd4j;
using namespace nd4j::graph;

class DeclarableOpsTests4 : public testing::Test {
public:
    
    DeclarableOpsTests4() {
        printf("\n");
        fflush(stdout);

        nd4j::ops::adjust_hue<float> op0;
        nd4j::ops::adjust_saturation<float> op1;
    }
};

TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_1) {
    NDArray<float> x('c', {2, 4, 4, 2});
    NDArray<float> exp('c', {2, 2, 2, 2}, {6.f, 7.f,  10.f,  11.f,  22.f,  23.f,  26.f,  27.f,  38.f,  39.f,  42.f,  43.f,  54.f,  55.f,  58.f, 59.f});

    NDArrayFactory<float>::linspace(1, x);
    

    nd4j::ops::avgpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_2) {
    NDArray<float> x('c', {2, 4, 4, 2});
    NDArray<float> exp('c', {2, 2, 2, 2}, {6.f, 7.f,  10.f,  11.f,  22.f,  23.f,  26.f,  27.f,  38.f,  39.f,  42.f,  43.f,  54.f,  55.f,  58.f, 59.f});

    NDArrayFactory<float>::linspace(1, x);
    

    nd4j::ops::avgpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_3) {
    NDArray<float> x('c', {2, 4, 4, 2});
    NDArray<float> exp('c', {2, 2, 2, 2}, {11.f,  12.f,  15.f,  16.f,  27.f,  28.f,  31.f,  32.f,  43.f,  44.f,  47.f,  48.f,  59.f,  60.f,  63.f, 64.f});

    NDArrayFactory<float>::linspace(1, x);
    

    nd4j::ops::maxpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_4) {
    NDArray<float> x('c', {2, 4, 4, 2});
    NDArray<float> exp('c', {2, 2, 2, 2}, {11.f,  12.f,  15.f,  16.f,  27.f,  28.f,  31.f,  32.f,  43.f,  44.f,  47.f,  48.f,  59.f,  60.f,  63.f, 64.f});

    NDArrayFactory<float>::linspace(1, x);


    nd4j::ops::maxpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_5) {
    NDArray<float> x('c', {2, 5, 5, 2});
    NDArray<float> exp('c', {2, 3, 3, 2}, {7.f,    8.f,   11.f,   12.f,   14.f,   15.f,   27.f,   28.f,   31.f,   32.f,   34.f,   35.f, 42.f,   43.f,   46.f,   47.f,   49.f,   50.f,   57.f,   58.f,   61.f,   62.f,   64.f,   65.f, 77.f,   78.f,   81.f,   82.f,   84.f,   85.f,   92.f,   93.f,   96.f,   97.f,   99.f,  100.f,});

    NDArrayFactory<float>::linspace(1, x);
    

    nd4j::ops::avgpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 1, 0, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_6) {
    NDArray<float> x('c', {2, 5, 5, 2});
    NDArray<float> exp('c', {2, 2, 2, 2}, {7.f,   8.f,  11.f,  12.f,  27.f,  28.f,  31.f,  32.f,  57.f,  58.f,  61.f,  62.f,  77.f,  78.f,  81.f, 82.f});

    NDArrayFactory<float>::linspace(1, x);
    
    nd4j::ops::avgpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_7) {
    NDArray<float> x('c', {2, 2, 5, 5});
    NDArray<float> exp('c', {2, 2, 2, 2}, {7.f, 9.f, 17.f, 19.f, 32.f, 34.f, 42.f, 44.f, 57.f, 59.f, 67.f, 69.f, 82.f, 84.f, 92.f, 94.f});

    NDArrayFactory<float>::linspace(1, x);


    nd4j::ops::maxpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_8) {
    NDArray<float> x('c', {2, 2, 5, 5});
    NDArray<float> exp('c', {2, 2, 3, 3}, {1.f, 2.5f, 4.5f, 8.5f, 10.f, 12.f, 18.5f, 20.f, 22.f, 26.f, 27.5f, 29.5f, 33.5f, 35.f, 37.f, 43.5f, 45.f, 47.f,  51.f, 52.5f, 54.5f,  58.5f, 60.f, 62.f, 68.5f, 70.f, 72.f,  76.f, 77.5f, 79.5f, 83.5f, 85.f, 87.f,  93.5f, 95.f, 97.f});

    NDArrayFactory<float>::linspace(1, x);


    nd4j::ops::avgpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_9) {
    NDArray<float> x('c', {2, 2, 5, 5});
    NDArray<float> exp('c', {2, 2, 3, 3}, {0.25f, 1.25f, 2.25f,  4.25f, 10.f, 12.f, 9.25f, 20.f, 22.f, 6.5f, 13.75f, 14.75, 16.75f, 35.f, 37.f,  21.75f, 45.f, 47.f,  12.75f, 26.25f, 27.25f,  29.25f, 60.f, 62.f, 34.25f, 70.f, 72.f, 19.f, 38.75f, 39.75f, 41.75f, 85.f, 87.f, 46.75f, 95.f, 97.f});

    NDArrayFactory<float>::linspace(1, x);


    nd4j::ops::avgpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 1, 1, 1, 1, 0, 1, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_10) {
    NDArray<float> x('c', {2, 2, 5, 5});
    NDArray<float> exp('c', {2, 2, 3, 3}, {4.f, 6.f, 7.5f, 14.f, 16.f, 17.5f,  21.5f, 23.5f, 25.f, 29.f, 31.f, 32.5f, 39.f, 41.f, 42.5f, 46.5f, 48.5f, 50.f, 54.f, 56.f, 57.5f,  64.f, 66.f, 67.5f, 71.5f, 73.5f, 75.f, 79.f, 81.f, 82.5f, 89.f, 91.f, 92.5f,  96.5f, 98.5f, 100.f});

    NDArrayFactory<float>::linspace(1, x);


    nd4j::ops::avgpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_11) {
    NDArray<float> x('c', {1, 1, 3, 3});
    NDArray<float> exp('c', {1, 1, 2, 2}, {3, 4, 6, 7});

    NDArrayFactory<float>::linspace(1, x);


    nd4j::ops::avgpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 1, 1, 0, 0, 1, 1, 0, 0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Pooling_Parity_12) {
    NDArray<float> x('c', {1, 1, 3, 3});
    NDArray<float> exp('c', {1, 1, 3, 3}, {3.f, 4.f, 4.5f, 6.f, 7.f, 7.5f, 7.5f, 8.5f, 9.f});

    NDArrayFactory<float>::linspace(1, x);


    nd4j::ops::avgpool2d<float> op;
    auto result = op.execute({&x}, {}, {2, 2, 1, 1, 0, 0, 1, 1, 1, 0, 0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printShapeInfo("z shape:");
    //z->printBuffer("z buffer:");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_BiasAdd_NHWC_1) {
    NDArray<float> x('c', {2, 3, 3, 2});
    NDArray<float> bias('c', {1, 2}, {1, 2});
    NDArray<float> exp('c', {2, 3, 3, 2}, {1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f, 1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f});

    nd4j::ops::biasadd<float> op;
    auto result = op.execute({&x, &bias}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_BiasAdd_NCHW_1) {
    NDArray<float> x('c', {2, 2, 3, 3});
    NDArray<float> bias('c', {1, 2}, {1, 2});
    NDArray<float> exp('c', {2, 2, 3, 3}, {1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f, 1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f,  1.f,  2.f});

    nd4j::ops::biasadd<float> op;
    auto result = op.execute({&x, &bias}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Fill_1) {
    NDArray<float> x('c', {1, 3}, {3, 2, 4});
    NDArray<float> exp('c', {3, 2, 4});
    exp.assign(2.0f);

    nd4j::ops::fill<float> op;
    auto result = op.execute({&x}, {2.0f}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Reshape_Again) {
    NDArray<float> x('c', {4, 3});
    NDArray<float> exp('c', {4, 3});

    NDArrayFactory<float>::linspace(1, x);
    NDArrayFactory<float>::linspace(1, exp);

    nd4j::ops::reshape<float> op;
    auto result = op.execute({&x}, {}, {99, 4, 3});

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Gemv_Transpose_1) {
    NDArray<float> x('c', {4, 3});
    NDArray<float> y('c', {4, 1});
    NDArray<float> exp('c',{ 3, 1}, {70, 80, 90});

    NDArrayFactory<float>::linspace(1, x);
    NDArrayFactory<float>::linspace(1, y);

    nd4j::ops::matmul<float> op;
    auto result = op.execute({&x, &y}, {}, {1, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Split_1) {
    NDArray<float> x('c', {5, 30});
    NDArray<float> sizes('c', {1, 3}, {4, 15, 11});

    IndicesList list0({NDIndex::all(), NDIndex::interval(0, 4)});
    IndicesList list1({NDIndex::all(), NDIndex::interval(4, 19)});
    IndicesList list2({NDIndex::all(), NDIndex::interval(19, 30)});

    auto sub0 = x.subarray(list0);
    auto sub1 = x.subarray(list1);
    auto sub2 = x.subarray(list2);

    sub0->assign(0.0f);
    sub1->assign(1.0f);
    sub2->assign(2.0f);


    nd4j::ops::split_v<float> op;
    auto result = op.execute({&x, &sizes}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, result->size());

    auto z0 = result->at(0);
    auto z1 = result->at(1);
    auto z2 = result->at(2);

    ASSERT_TRUE(sub0->isSameShape(z0));
    ASSERT_TRUE(sub0->equalsTo(z0));

    ASSERT_TRUE(sub1->isSameShape(z1));
    ASSERT_TRUE(sub1->equalsTo(z1));

    ASSERT_TRUE(sub2->isSameShape(z2));
    ASSERT_TRUE(sub2->equalsTo(z2));

    delete sub0;
    delete sub1;
    delete sub2;

    delete result;
}

// special test for TF mode, when axis goes first
TEST_F(DeclarableOpsTests4, Test_Split_2) {
    NDArray<float> x('c', {5, 12});
    NDArray<float> axis('c', {1, 1}, {1.f});

    IndicesList list0({NDIndex::all(), NDIndex::interval(0, 3)});
    IndicesList list1({NDIndex::all(), NDIndex::interval(3, 6)});
    IndicesList list2({NDIndex::all(), NDIndex::interval(6, 9)});
    IndicesList list3({NDIndex::all(), NDIndex::interval(9, 12)});

    auto sub0 = x.subarray(list0);
    auto sub1 = x.subarray(list1);
    auto sub2 = x.subarray(list2);
    auto sub3 = x.subarray(list3);

    sub0->assign(0.0f);
    sub1->assign(1.0f);
    sub2->assign(2.0f);
    sub3->assign(3.0f);


    nd4j::ops::split<float> op;
    auto result = op.execute({&axis, &x}, {}, {4});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z0 = result->at(0);
    auto z1 = result->at(1);
    auto z2 = result->at(2);
    auto z3 = result->at(3);

    ASSERT_TRUE(sub0->isSameShape(z0));
    ASSERT_TRUE(sub1->isSameShape(z1));
    ASSERT_TRUE(sub2->isSameShape(z2));
    ASSERT_TRUE(sub3->isSameShape(z3));

    ASSERT_TRUE(sub0->equalsTo(z0));
    ASSERT_TRUE(sub1->equalsTo(z1));
    ASSERT_TRUE(sub2->equalsTo(z2));
    ASSERT_TRUE(sub3->equalsTo(z3));


    delete sub0;
    delete sub1;
    delete sub2;
    delete sub3;

    delete result;
}

// special test for TF mode, when axis goes first
TEST_F(DeclarableOpsTests4, Test_Split_3) {
    NDArray<float> x('c', {6, 12});
    NDArray<float> axis('c', {1, 1}, {0.f});

    IndicesList list0({NDIndex::interval(0, 2), NDIndex::all()});
    IndicesList list1({NDIndex::interval(2, 4), NDIndex::all()});
    IndicesList list2({NDIndex::interval(4, 6), NDIndex::all()});

    auto sub0 = x.subarray(list0);
    auto sub1 = x.subarray(list1);
    auto sub2 = x.subarray(list2);

    sub0->assign(0.0f);
    sub1->assign(1.0f);
    sub2->assign(2.0f);

    nd4j::ops::split<float> op;
    auto result = op.execute({&axis, &x}, {}, {3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z0 = result->at(0);
    auto z1 = result->at(1);
    auto z2 = result->at(2);

    ASSERT_TRUE(sub0->isSameShape(z0));
    ASSERT_TRUE(sub1->isSameShape(z1));
    ASSERT_TRUE(sub2->isSameShape(z2));

    ASSERT_TRUE(sub0->equalsTo(z0));
    ASSERT_TRUE(sub1->equalsTo(z1));
    ASSERT_TRUE(sub2->equalsTo(z2));

    delete sub0;
    delete sub1;
    delete sub2;

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Stack_4) {
    NDArray<float> t('c', {2, 3, 5});
    NDArray<float> u('c', {2, 3, 5});
    NDArray<float> v('c', {2, 3, 5});
    NDArray<float> exp('c', {3, 2, 3, 5});

    nd4j::ops::stack<float> op;
    auto result = op.execute({&t, &u, &v}, {}, {-4});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());


    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Squeeze_args_1) {
    NDArray<float> x('c', {2, 1, 1, 1, 2}, {1, 2, 3, 4});
    NDArray<float> exp('c', {2, 1, 2}, {1, 2, 3, 4});

    nd4j::ops::squeeze<float> op;
    auto result = op.execute({&x}, {}, {1, 3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Squeeze_args_2) {
    NDArray<float> x('c', {2, 1, 1, 1, 2}, {1, 2, 3, 4});
    NDArray<float> y('c', {2}, {1.f, 3.f});
    NDArray<float> exp('c', {2, 1, 2}, {1, 2, 3, 4});

    nd4j::ops::squeeze<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Squeeze_args_3) {
    NDArray<float> x('c', {2, 1, 1, 1, 2}, {1, 2, 3, 4});
    NDArray<float> exp('c', {2, 1, 2}, {1, 2, 3, 4});

    nd4j::ops::squeeze<float> op;
    auto result = op.execute({&x}, {}, {-2, -3});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_VectorScalar_Concat_1) {
    NDArray<float> x('c', {2}, {1, 0});
    NDArray<float> y(3.0f);
    NDArray<float> exp('c', {3}, {1, 0, 3});

    nd4j::ops::concat<float> op;
    auto result = op.execute({&x, &y}, {}, {0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_BiasAdd_1) {
    NDArray<float> x('c', {2, 3});
    NDArray<float> row('c', {3}, {1, 2, 3});
    NDArray<float> exp('c', {2, 3}, {1, 2, 3, 1, 2, 3});

    nd4j::ops::biasadd<float> op;
    auto result = op.execute({&x, &row}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_1D_1) {
    NDArray<float> x('c', {2, 3});

    nd4j::ops::unstack<float> op;
    auto result = op.execute({&x}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_EQ(3, result->size());

    for (int e = 0; e < 3; e++)
        ASSERT_EQ(1, result->at(e)->rankOf());

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_SpaceToDepth_1) {
    NDArray<float> x('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> exp('c', {1, 1, 1, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::space_to_depth<float> op;
    auto result = op.execute({&x}, {}, {2, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_SpaceToDepth_2) {
    NDArray<float> x('c', {1, 3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> exp('c', {1, 12, 1, 1}, {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});

    nd4j::ops::space_to_depth<float> op;
    auto result = op.execute({&x}, {}, {2, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_DepthToSpace_1) {
    NDArray<float> x('c', {1, 1, 1, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> exp('c', {1, 2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::depth_to_space<float> op;
    auto result = op.execute({&x}, {}, {2, 1});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_DepthToSpace_2) {
    NDArray<float> x('c', {1, 12, 1, 1}, {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
    NDArray<float> exp('c', {1, 3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    nd4j::ops::depth_to_space<float> op;
    auto result = op.execute({&x}, {}, {2, 0});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_DepthToSpace_3) {
    NDArray<float> x('c', {4, 4, 16, 16});
    NDArray<float> exp('c', {4, 16, 64, 1});

    nd4j::ops::depth_to_space<float> op;
    auto result = op.execute({&x}, {}, {4, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Cross_1) {
    NDArray<float> a('c', {3}, {1, 2, 3});
    NDArray<float> b('c', {3}, {6, 7, 8});
    NDArray<float> exp('c', {3}, {-5, 10, -5});

    nd4j::ops::cross<float> op;
    auto result = op.execute({&a, &b}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Cross_2) {
    NDArray<float> a('c', {2, 3}, {1, 2, 3, 1, 2, 3});
    NDArray<float> b('c', {2, 3}, {6, 7, 8, 6, 7, 8});
    NDArray<float> exp('c', {2, 3}, {-5, 10, -5, -5, 10, -5});

    nd4j::ops::cross<float> op;
    auto result = op.execute({&a, &b}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
    
    delete result;
}


TEST_F(DeclarableOpsTests4, Test_Cross_3) {
    NDArray<float> a('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> b('c', {3, 3}, {2, 3, 4, 7, 6, 5, 6, 3, 2});
    NDArray<float> exp('c', {3, 3}, { -1,   2,  -1, -11,  22, -11, -11,  40, -27});

    nd4j::ops::cross<float> op;
    auto result = op.execute({&a, &b}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
    
    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Matmul_YATS_1) {
    NDArray<float> a('c', {3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> b('c', {4}, {1, 2, 3, 4});
    NDArray<float> exp('c', {3}, {30, 70, 110,});

    nd4j::ops::matmul<float> op;
    auto result = op.execute({&a, &b}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Matmul_YATS_2) {
    NDArray<float> a('c', {4}, {1, 2, 3, 4});
    NDArray<float> b('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> exp('c', {1, 3}, {70, 80, 90});

    nd4j::ops::matmul<float> op;
    auto result = op.execute({&a, &b}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    z->printShapeInfo("z");
    
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Matmul_YATS_3) {
    NDArray<float> a('c', {1, 4}, {1, 2, 3, 4});
    NDArray<float> b('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    NDArray<float> exp('c', {1, 3}, {70, 80, 90});

    nd4j::ops::matmul<float> op;
    auto result = op.execute({&a, &b}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);
    
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Add_119) {
    NDArray<float> a('c', {1, 4}, {1, 2, 3, 4});
    NDArray<float> b('c', {4}, {1, 2, 3, 4});
    NDArray<float> exp('c', {1, 4}, {2, 4, 6, 8});

    nd4j::ops::add<float> op;
    auto result = op.execute({&a, &b}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_EQ(2, z->rankOf());

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_Reshape_Negative_1) {
    NDArray<float> x('c', {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    NDArray<float> shape('c', {2}, {-1, 2});
    NDArray<float> exp('c', {4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});

    nd4j::ops::reshape<float> op;
    auto result = op.execute({&x, &shape}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_TileToShape_1) {
    NDArray<float> x('c', {2, 1, 3});
    NDArray<float> exp('c', {2, 4, 3});

    nd4j::ops::tile_to_shape<float> op;
    auto result = op.execute({&x},{}, {2, 4, 3});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));

    delete result;
}


TEST_F(DeclarableOpsTests4, Test_StridedSlice_Alex_1) {
    NDArray<float> x('c', {3, 4, 5});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {1,3,4,5});
    NDArrayFactory<float>::linspace(1, exp);

    nd4j::ops::strided_slice<float> op;
    auto result = op.execute({&x}, {}, {0,0,0,1,0, -999,0,0,0, -999,3,4,5, -999,1,1,1});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(DeclarableOpsTests4, Test_StridedSlice_Alex_2) {
    NDArray<float> x('c', {3, 4, 5});
    NDArray<float> begin('c', {4}, {-999,0,0,0});
    NDArray<float> end('c', {4}, {-999,3,4,5});
    NDArray<float> stride('c', {4}, {-999,1,1,1});
    NDArrayFactory<float>::linspace(1, x);
    NDArray<float> exp('c', {1,3,4,5});
    NDArrayFactory<float>::linspace(1, exp);

    nd4j::ops::strided_slice<float> op;
    auto result = op.execute({&x, &begin, &end, &stride}, {}, {0,0,0,1,0});

    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test1) {

    NDArray<float> x1('c', {2,2,2});
    NDArray<float> x2('c', {2,2,2});
    NDArray<float> x3('c', {2,2,2});
    NDArrayFactory<float>::linspace(1, x1);
    NDArrayFactory<float>::linspace(9, x2);
    NDArrayFactory<float>::linspace(17,x3);

    NDArray<float> expected('c', {3,2,2,2});
    NDArrayFactory<float>::linspace(1, expected);
    
    nd4j::ops::parallel_stack<float> op;
    ResultSet<float>*  results = op.execute({&x1, &x2, &x3}, {}, {});
    NDArray<float>* output = results->at(0);        

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test2) {

    NDArray<float> x1('c', {1,2}, {1,2});
    NDArray<float> x2('c', {1,2}, {3,4});
    NDArray<float> x3('c', {1,2}, {5,6});
    
    NDArray<float> expected('c', {3,1,2}, {1,2,3,4,5,6});    
    
    nd4j::ops::parallel_stack<float> op;
    ResultSet<float>*  results = op.execute({&x1, &x2, &x3}, {}, {});
    NDArray<float>* output = results->at(0);        

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    


    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test3) {

    NDArray<float> x1('c', {2,1}, {1,2});
    NDArray<float> x2('c', {2,1}, {3,4});
    NDArray<float> x3('c', {2,1}, {5,6});
    
    NDArray<float> expected('c', {3,2,1}, {1,2,3,4,5,6});    
    
    nd4j::ops::parallel_stack<float> op;
    ResultSet<float>*  results = op.execute({&x1, &x2, &x3}, {}, {});
    NDArray<float>* output = results->at(0);        

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    

    delete results;
}
\
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test4) {

    NDArray<float> x1('c', {2}, {1,2});
    NDArray<float> x2('c', {2}, {3,4});
    NDArray<float> x3('c', {2}, {5,6});
    
    NDArray<float> expected('c', {3,2}, {1,2,3,4,5,6});    
    
    nd4j::ops::parallel_stack<float> op;
    ResultSet<float>*  results = op.execute({&x1, &x2, &x3}, {}, {});
    NDArray<float>* output = results->at(0);        

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    

    delete results;
} 

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test5) {

    NDArray<float> x1('c', {1}, {1});
    NDArray<float> x2('c', {1}, {3});
    NDArray<float> x3('c', {1}, {5});
    
    NDArray<float> expected('c', {3,1}, {1,3,5});    
    
    nd4j::ops::parallel_stack<float> op;
    ResultSet<float>*  results = op.execute({&x1, &x2, &x3}, {}, {});
    NDArray<float>* output = results->at(0);        

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test6) {

    NDArray<float> x1(1.);
    NDArray<float> x2(3.);
    NDArray<float> x3(5.);
    
    NDArray<float> expected('c', {3}, {1,3,5});    
    
    nd4j::ops::parallel_stack<float> op;
    ResultSet<float>*  results = op.execute({&x1, &x2, &x3}, {}, {});
    NDArray<float>* output = results->at(0);        

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, parallel_stack_test7) {

    NDArray<float> x1(1.);   
    NDArray<float> expected('c', {1}, {1.});    
    
    nd4j::ops::parallel_stack<float> op;
    ResultSet<float>*  results = op.execute({&x1}, {}, {});
    NDArray<float>* output = results->at(0);        

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test1) {

    NDArray<float> in0('c', {2}, {1, 2});
    NDArray<float> in1('c', {3}, {10, 20, 30});
    NDArray<float> in2('c', {4}, {100, 200, 300, 400});
    NDArray<float> exp0('c', {2,3,4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    NDArray<float> exp1('c', {2,3,4}, {10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30});
    NDArray<float> exp2('c', {2,3,4}, {100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400});
    
    nd4j::ops::meshgrid<float> op;
    ResultSet<float>*  results = op.execute({&in0, &in1, &in2}, {}, {0});
    NDArray<float>* out0 = results->at(0);
    NDArray<float>* out1 = results->at(1);
    NDArray<float>* out2 = results->at(2);

    // out0->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));    
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));    
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test2) {

    NDArray<float> in0('c', {2}, {1, 2});
    NDArray<float> in1('c', {3}, {10, 20, 30});
    NDArray<float> in2('c', {4}, {100, 200, 300, 400});
    NDArray<float> exp0('c', {3,2,4}, {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2});
    NDArray<float> exp1('c', {3,2,4}, {10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30});
    NDArray<float> exp2('c', {3,2,4}, {100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400});
    
    nd4j::ops::meshgrid<float> op;
    ResultSet<float>*  results = op.execute({&in0, &in1, &in2}, {}, {});
    NDArray<float>* out0 = results->at(0);
    NDArray<float>* out1 = results->at(1);
    NDArray<float>* out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));    
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));    
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test3) {

    NDArray<float> in0('c', {2}, {1, 2});
    NDArray<float> in1('c', {1,3}, {10, 20, 30});
    NDArray<float> in2('c', {2,2}, {100, 200, 300, 400});
    NDArray<float> exp0('c', {3,2,4}, {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2});
    NDArray<float> exp1('c', {3,2,4}, {10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30});
    NDArray<float> exp2('c', {3,2,4}, {100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400});
    
    nd4j::ops::meshgrid<float> op;
    ResultSet<float>*  results = op.execute({&in0, &in1, &in2}, {}, {});
    NDArray<float>* out0 = results->at(0);
    NDArray<float>* out1 = results->at(1);
    NDArray<float>* out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));    
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));    
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test4) {

    NDArray<float> in0('c', {1,2}, {1, 2});
    NDArray<float> in1('c', {3,1}, {10, 20, 30});
    NDArray<float> in2('c', {1,4,1}, {100, 200, 300, 400});
    NDArray<float> exp0('c', {2,3,4}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    NDArray<float> exp1('c', {2,3,4}, {10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30});
    NDArray<float> exp2('c', {2,3,4}, {100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400});
    
    nd4j::ops::meshgrid<float> op;
    ResultSet<float>*  results = op.execute({&in0, &in1, &in2}, {}, {0});
    NDArray<float>* out0 = results->at(0);
    NDArray<float>* out1 = results->at(1);
    NDArray<float>* out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));    
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));    
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test5) {

    NDArray<float> in0(1);
    NDArray<float> in1(2);
    NDArray<float> in2(3);
    NDArray<float> exp0('c', {1,1,1}, {1});
    NDArray<float> exp1('c', {1,1,1}, {2});
    NDArray<float> exp2('c', {1,1,1}, {3});
    
    nd4j::ops::meshgrid<float> op;
    ResultSet<float>*  results = op.execute({&in0, &in1, &in2}, {}, {0});
    NDArray<float>* out0 = results->at(0);
    NDArray<float>* out1 = results->at(1);
    NDArray<float>* out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));    
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));    
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test6) {

    NDArray<float> in0('c', {2,2},{1,2,3,4});
    NDArray<float> in1(5);
    NDArray<float> in2(6);
    NDArray<float> exp0('c', {4,1,1}, {1,2,3,4});
    NDArray<float> exp1('c', {4,1,1}, {5,5,5,5});
    NDArray<float> exp2('c', {4,1,1}, {6,6,6,6});
    
    nd4j::ops::meshgrid<float> op;
    ResultSet<float>*  results = op.execute({&in0, &in1, &in2}, {}, {0});
    NDArray<float>* out0 = results->at(0);
    NDArray<float>* out1 = results->at(1);
    NDArray<float>* out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));    
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));    
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test7) {

    NDArray<float> in0('c', {2,2},{1,2,3,4});
    NDArray<float> in1(5);
    NDArray<float> in2(6);
    NDArray<float> exp0('c', {1,4,1}, {1,2,3,4});
    NDArray<float> exp1('c', {1,4,1}, {5,5,5,5});
    NDArray<float> exp2('c', {1,4,1}, {6,6,6,6});
    
    nd4j::ops::meshgrid<float> op;
    ResultSet<float>*  results = op.execute({&in0, &in1, &in2}, {}, {1});
    NDArray<float>* out0 = results->at(0);
    NDArray<float>* out1 = results->at(1);
    NDArray<float>* out2 = results->at(2);

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));    
    ASSERT_TRUE(exp1.isSameShape(out1));
    ASSERT_TRUE(exp1.equalsTo(out1));    
    ASSERT_TRUE(exp2.isSameShape(out2));
    ASSERT_TRUE(exp2.equalsTo(out2));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test8) {
    
    NDArray<float> in0(5);
    NDArray<float> exp0('c', {1}, {5});
    
    nd4j::ops::meshgrid<float> op;
    ResultSet<float>*  results = op.execute({&in0}, {}, {0});
    NDArray<float>* out0 = results->at(0);
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, meshgrid_test9) {
    
    NDArray<float> in0(5);
    NDArray<float> exp0('c', {1}, {5});
    
    nd4j::ops::meshgrid<float> op;
    ResultSet<float>*  results = op.execute({&in0}, {}, {1});
    NDArray<float>* out0 = results->at(0);
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp0.isSameShape(out0));
    ASSERT_TRUE(exp0.equalsTo(out0));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, WeightedCrossEntropyWithLogits_1) {
    

    NDArray<float> input   ('c', {2, 3}, {11.f, 13.f,  4.f, 15.f,  6.f,  3.f});
    NDArray<float> targets ('c', {2, 3}, {15.5f, 15.7f,  5.f , 15.f,   5.f,   6.f});
    NDArray<float> weight (0.7f);
    NDArray<float> expected('c', {2, 3}, {-159.50006,  -191.1, -16.009075, -210., -24.001238, -15.03887});
    
//Targets {15.5f, 15.7f,  5.f , 15.f,   5.f,   6.f};
//----------
//Inputs {11.f, 13.f,  4.f, 15.f,  6.f,  3.f};
//----------
//Weights [0.7]
//Result {-159.50006,  -191.1,       -16.009075, -210., -24.001238,  -15.03887}

    nd4j::ops::weighted_cross_entropy_with_logits<float> op;
    ResultSet<float>* results = op.execute({&targets, &input, &weight}, {}, {});
    NDArray<float>* output = results->at(0);
    
    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, WeightedCrossEntropyWithLogits_2) {
    

    NDArray<float> input   ('c', {2, 3}, {11.f,   13.f,  4.f, 15.f,  6.f,  3.f});
    NDArray<float> targets ('c', {2, 3}, {15.5f, 15.7f,  5.f, 15.f,  5.f,  6.f});
    NDArray<float> weights ({0.5f, 0.7f, 1.0f}) ;
    NDArray<float> expected('c', {2, 3}, {-159.5001f, -191.1f, -15.98185f, -210.f,  -24.001238f, -14.951412f});
    
    nd4j::ops::weighted_cross_entropy_with_logits<float> op;
    ResultSet<float>* results = op.execute({&targets, &input, &weights}, {}, {});
    NDArray<float>* output = results->at(0);
    
    output->printIndexedBuffer("Result is ");
    expected.printIndexedBuffer("Expected is ");

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, lstm_test1) {
    
    const int time      = 5;
    const int batchSize = 3;
    const int inSize    = 3;
    const int numProj   = 3;
    const int numUnits  = 3;

    NDArray<double> x  ('c', {time, batchSize, inSize});
    NDArray<double> h0 ('c', {batchSize, numProj});
    NDArray<double> c0 ('c', {batchSize, numUnits});
    NDArray<double> Wx ('c', {inSize, 4*numUnits});
    NDArray<double> Wh ('c', {numProj, 4*numUnits});
    NDArray<double> Wc ('c', {3*numUnits});
    NDArray<double> Wp ('c', {numUnits, numProj});
    NDArray<double> b  ('c', {4*numUnits});

    NDArrayFactory<double>::linspace(0.5, x, 0.5);
    h0 = 1.;
    c0 = 2.;
    Wx = 0.003;
    Wh = 0.006;
    Wc = 0.;
    Wp = 0.;
    b = 0.5;

    NDArray<double> expH('c', {time, batchSize, numProj}, {0.57574,0.57574,0.57574,0.58006,0.58006,0.58006,0.58434,0.58434,0.58434,
                                                           0.55114,0.55114,0.55114,0.55732,0.55732,0.55732,0.56338,0.56338,0.56338,
                                                           0.53763,0.53763,0.53763,0.54534,0.54534,0.54534,0.55287,0.55287,0.55287,
                                                           0.53626,0.53626,0.53626,0.54487,0.54487,0.54487,0.55327,0.55327,0.55327,
                                                           0.54484,0.54484,0.54484,0.55379,0.55379,0.55379,0.5625 ,0.5625 ,0.5625});    

    NDArray<double> expClast('c', {1, batchSize, numProj}, {1.1589154,1.1589154,1.1589154,1.1892855,1.1892855,1.1892855,1.219861 ,1.219861 ,1.219861});

    nd4j::ops::lstm<double> op;
    nd4j::ResultSet<double>* results = op.execute({&x, &h0, &c0, &Wx, &Wh, &Wc, &Wp, &b}, {0., 0., 0.}, {0, 0});    

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *h = results->at(0);    
    NDArray<double> *c = results->at(1);
    NDArray<double> cLast = (*c)({{4,5},{},{}},true);

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));    

    ASSERT_TRUE(expClast.isSameShape(&cLast));
    ASSERT_TRUE(expClast.equalsTo(&cLast));            

    delete results;
} 


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, gru_test1) {
    
    const int time      = 5;
    const int batchSize = 3;
    const int inSize    = 3;    
    const int numUnits  = 3;

    NDArray<double> x  ('c', {time, batchSize, inSize});
    NDArray<double> h0 ('c', {batchSize, numUnits});
    NDArray<double> Wx ('c', {inSize, 3*numUnits});
    NDArray<double> Wh ('c', {numUnits, 3*numUnits});
    NDArray<double> b  ('c', {3*numUnits});

    NDArrayFactory<double>::linspace(0.5, x, 0.5);
    h0 = 1.;    
    Wx = 0.003;
    Wh = 0.006;
    b = 0.5;

    NDArray<double> expH('c', {time, batchSize, numUnits},{0.8062 ,0.8062 ,0.8062 ,0.81167,0.81167,0.81167,0.81702,0.81702,0.81702,
                                                           0.69772,0.69772,0.69772,0.70577,0.70577,0.70577,0.71366,0.71366,0.71366,
                                                           0.64041,0.64041,0.64041,0.64952,0.64952,0.64952,0.65847,0.65847,0.65847,
                                                           0.61392,0.61392,0.61392,0.62331,0.62331,0.62331,0.63254,0.63254,0.63254,
                                                           0.60603,0.60603,0.60603,0.61531,0.61531,0.61531,0.62443,0.62443,0.62443});    
    
    nd4j::ops::gru<double> op;
    nd4j::ResultSet<double>* results = op.execute({&x, &h0, &Wx, &Wh, &b}, {}, {});    

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<double> *h = results->at(0);    

    ASSERT_TRUE(expH.isSameShape(h));
    ASSERT_TRUE(expH.equalsTo(h));    

    delete results;
} 

///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, relu6_test1) {
    
    NDArray<float> input ('c', {2,4}, {-13.,10,-5,0,2,7,6,12});
    NDArray<float> expected ('c', {2,4}, {0., 6., 0., 0.,2., 6., 6., 6.});
    
    nd4j::ops::relu6<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input}, {0.}, {});    

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* output = results->at(0);    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    

    delete results;
} 


///////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, relu6_bp_test1) {
    
    NDArray<float> input ('c', {2,4}, {-13.,10, -5,  0,  2,  7,  6,  5});
    NDArray<float> gradO ('c', {2,4}, {-1., -2., 0., 4., 5., 6., 7., 8.});    
    
    NDArray<float> expected ('c', {2,4}, {0., 0., 0., 0., 5., 0., 0., 8.});
    
    nd4j::ops::relu6_bp<float> op;
    nd4j::ResultSet<float>* results = op.execute({&input, &gradO}, {0.}, {});    

    ASSERT_EQ(ND4J_STATUS_OK, results->status());

    NDArray<float>* output = results->at(0);        

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    

    delete results;
} 
 
////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, LrnTest_1) {

    NDArray<double> x( 'c', {2, 2, 2, 2}, { 5.5, 0., 0.3, 5.5, 
                                            8.6, 0.,  0., 0.4,
                                            1.5, 1., 1.3, 1.5,
                                            2.6, 2.,  3., 1.4}
    );

    NDArray<double> exp('c', {2, 2, 2, 2}, {
                                            0.98386997,        0.,  0.05358852,  0.9824562,
                                            0.99330735,        0.,          0., 0.37139067,
                                            0.72760683, 0.4850712,   0.5848977, 0.67488194,
                                            0.7581754,  0.58321184, 0.86747235, 0.4048204}
    );

    nd4j::ops::lrn<double> op;
    ResultSet<double>*  results = op.execute({&x}, {1.0, 1.0, 0.5}, {5});
    NDArray<double>* out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    out->printIndexedBuffer("LRN out");
    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));    
    
    delete results;
}


////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, LrnTest_2) {

    NDArray<double> x( 'c', {2, 2, 2, 2}, { 5.5, 0., 0.3, 5.5, 
                                            8.6, 0.,  0., 0.4,
                                            1.5, 1., 1.3, 1.5,
                                            2.6, 2.,  3., 1.4});

    NDArray<double> exp('c', {2, 2, 2, 2}, {
                                            0.98386997,        0.,  0.05358852,  0.9824562,
                                            0.99330735,        0.,          0., 0.37139067,
                                            0.72760683, 0.4850712,   0.5848977, 0.67488194,
                                            0.7581754,  0.58321184, 0.86747235, 0.4048204});

    nd4j::ops::lrn<double> op;
    ResultSet<double>*  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2});
    NDArray<double>* out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    out->printIndexedBuffer("LRN out");
    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, LrnTest_3) {

    NDArray<double> x( 'c', {2, 2, 2, 4}, { 

                5.5, 0., 0.3, 5.5,
                1.5, 0., 1.3, 6.5,
                8.6, 0.,  0., 0.4,
                2.5, 1., 0.3, 4.5,
                1.5, 1., 1.3, 1.5,
                3.5, 0., 1.3, 2.5,
                2.6, 2.,  3., 1.4,
                4.5, 1., 0.3, 0.5}
    );

    NDArray<double> exp('c', {2, 2, 2, 4}, {
                     0.9824562,          0., 0.03822664, 0.9824562,
                    0.67488194,          0., 0.18924236, 0.96960944,
                    0.99330735,          0.,         0., 0.37139067,
                    0.86567914,  0.18702209, 0.05610663, 0.9520745,
                     0.6154575,  0.34942827, 0.45425674, 0.6154575,
                      0.905509,  0.        ,  0.2824086, 0.8361251,
                    0.57063663,  0.41959068,   0.629386, 0.3504383,
                     0.9520745,  0.21039814, 0.06311944, 0.3268602 }
    );

    nd4j::ops::lrn<double> op;
    ResultSet<double>*  results = op.execute({&x}, {1.0, 1.0, 0.5}, {2});
    NDArray<double>* out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    out->printIndexedBuffer("LRN out");
    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, LrnTest_4) {

    NDArray<double> x( 'c', {2, 2, 2, 4}, { 

                    5.5, 0., 0.3, 5.5,
                    1.5, 0., 1.3, 6.5,
                    8.6, 0.,  0., 0.4,
                    2.5, 1., 0.3, 4.5,
                    1.5, 1., 1.3, 1.5,
                    3.5, 0., 1.3, 2.5,
                    2.6, 2.,  3., 1.4,
                    4.5, 1., 0.3, 0.5}
    );

    NDArray<double> exp('c', {2, 2, 2, 4}, {
                    0.70082176,         0., 0.03822664, 0.70082176,
                    0.21835658,         0., 0.18924236,  0.9462118,
                     0.9922489,         0.,         0., 0.04615111,
                    0.46755522, 0.18702209, 0.05610663,  0.8415994,
                     0.5241424, 0.34942827, 0.45425674,  0.5241424,
                    0.76033086,         0.,  0.2824086, 0.54309344,
                    0.54546785, 0.41959068,   0.629386, 0.29371348,
                    0.94679165, 0.21039814, 0.06311944, 0.10519907}
    );

    nd4j::ops::lrn<double> op;
    ResultSet<double>*  results = op.execute({&x}, {1.0, 1.0, 0.5}, {5});
    NDArray<double>* out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    out->printIndexedBuffer("LRN out");
    exp.printIndexedBuffer("LRN exp");
    ASSERT_TRUE(exp.equalsTo(out));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, LrnTest_5) {

    NDArray<double> x('c', {2, 2, 2, 4}, { 

                5.5,0., 0.3, 5.5,
                1.5,0., 1.3, 6.5,
                8.6,0.,  0., 0.4,
                2.5,1., 0.3, 4.5,
                1.5,1., 1.3, 1.5,
                3.5,0., 1.3, 2.5,
                2.6,2.,  3., 1.4,
                4.5,1., 0.3, 0.5}
    );

    NDArray<double> eps('c', {2, 2, 2, 4}, {
                0.70082176, 0.,         0.03822664, 0.70082176,
                0.21835658, 0.,         0.18924236,  0.9462118,

                0.9922489,  0.,         0.        , 0.04615111,
                0.46755522, 0.18702209, 0.05610663,  0.8415994,


                0.5241424,  0.34942827, 0.45425674,  0.5241424,
                0.76033086, 0.,         0.2824086 , 0.54309344,

                0.54546785, 0.41959068, 0.629386  , 0.29371348,
                0.94679165, 0.21039814, 0.06311944, 0.10519907}
    );

    NDArray<double> exp('c', {2, 2, 2, 4});

    nd4j::ops::lrn_bp<double> op;
    ResultSet<double>*  results = op.execute({&x, &eps}, {1.0, 1.0, 0.5}, {5});
    NDArray<double>* out = results->at(0);
        
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(exp.isSameShape(out));
    out->printIndexedBuffer("LRN out");
    exp.printIndexedBuffer("LRN exp");
//    ASSERT_TRUE(exp.equalsTo(out));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, avgpool3d_test1) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0, dD=1,dH=1,dW=1;
    int oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME,  0-VALID
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<double> expected('c', {bS, iC, oD, oH, oW}, {10.5, 11.5, 13.5, 14.5, 22.5, 23.5, 25.5, 26.5, 46.5, 47.5, 49.5, 50.5, 58.5, 59.5, 61.5, 62.5,
                                                         82.5, 83.5, 85.5, 86.5, 94.5, 95.5, 97.5, 98.5,118.5,119.5,121.5,122.5,130.5,131.5,133.5,134.5,
                                                        154.5,155.5,157.5,158.5,166.5,167.5,169.5,170.5,190.5,191.5,193.5,194.5,202.5,203.5,205.5,206.5});
    NDArrayFactory<double>::linspace(1., input);
    
    nd4j::ops::avgpool3dnew<double> op;
    ResultSet<double>* results = op.execute({&input}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, avgpool3d_test2) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int oD=3,oH=4,oW=3;
    int paddingMode = 1;             // 1-SAME,  0-VALID
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iD, iH, iW, iC});
    NDArray<double> expected('c', {bS, oD, oH, oW, iC}, {  25. , 26. , 27. , 28. , 29. , 30. , 29.5, 30.5, 31.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 34. , 35. , 36. , 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 43. , 44. , 45. , 43. , 44. , 45. , 46. , 47. , 48. , 47.5, 48.5, 49.5,
                                                           61. , 62. , 63. , 64. , 65. , 66. , 65.5, 66.5, 67.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5, 70. , 71. , 72. , 74.5, 75.5, 76.5, 77.5, 78.5, 79.5, 79. , 80. , 81. , 79. , 80. , 81. , 82. , 83. , 84. , 83.5, 84.5, 85.5,
                                                           79. , 80. , 81. , 82. , 83. , 84. , 83.5, 84.5, 85.5, 83.5, 84.5, 85.5, 86.5, 87.5, 88.5, 88. , 89. , 90. , 92.5, 93.5, 94.5, 95.5, 96.5, 97.5, 97. , 98. , 99. , 97. , 98. , 99. ,100. ,101. ,102. ,101.5,102.5,103.5,
                                                          133. ,134. ,135. ,136. ,137. ,138. ,137.5,138.5,139.5,137.5,138.5,139.5,140.5,141.5,142.5,142. ,143. ,144. ,146.5,147.5,148.5,149.5,150.5,151.5,151. ,152. ,153. ,151. ,152. ,153. ,154. ,155. ,156. ,155.5,156.5,157.5,
                                                          169. ,170. ,171. ,172. ,173. ,174. ,173.5,174.5,175.5,173.5,174.5,175.5,176.5,177.5,178.5,178. ,179. ,180. ,182.5,183.5,184.5,185.5,186.5,187.5,187. ,188. ,189. ,187. ,188. ,189. ,190. ,191. ,192. ,191.5,192.5,193.5,
                                                          187. ,188. ,189. ,190. ,191. ,192. ,191.5,192.5,193.5,191.5,192.5,193.5,194.5,195.5,196.5,196. ,197. ,198. ,200.5,201.5,202.5,203.5,204.5,205.5,205. ,206. ,207. ,205. ,206. ,207. ,208. ,209. ,210. ,209.5,210.5,211.5});
    NDArrayFactory<double>::linspace(1., input);    

    nd4j::ops::avgpool3dnew<double> op;
    ResultSet<double>* results = op.execute({&input}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 0, dataFormat});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
 
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, avgpool3d_test3) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME,  0-VALID
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iD, iH, iW, iC});
    NDArray<double> expected('c', {bS, oD, oH, oW, iC}, {  29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 38.5, 39.5, 40.5, 41.5, 42.5, 43.5, 65.5, 66.5, 67.5, 68.5, 69.5, 70.5,
                                                           74.5, 75.5, 76.5, 77.5, 78.5, 79.5,137.5,138.5,139.5,140.5,141.5,142.5,146.5,147.5,148.5,149.5,150.5,151.5,
                                                          173.5,174.5,175.5,176.5,177.5,178.5,182.5,183.5,184.5,185.5,186.5,187.5});
    NDArrayFactory<double>::linspace(1., input);    

    nd4j::ops::avgpool3dnew<double> op;
    ResultSet<double>* results = op.execute({&input}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, avgpool3d_test4) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=1,pH=1,pW=1,  dD=1,dH=1,dW=1;
    int oD=4,oH=4,oW=4;
    int paddingMode = 0;             // 1-SAME,  0-VALID
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<double> expected('c', {bS, iC, oD, oH, oW},{0.416667, 1.00, 1.333333, 0.75, 1.00, 2.25, 2.75, 1.50, 1.75, 3.75, 4.25, 2.25, 1.416667, 3.00, 3.333333, 1.75, 2.833333, 6.00, 6.666667, 3.50, 5.00, 10.50, 11.50, 6.00, 6.50, 
                                                        13.50, 14.50, 7.50, 4.833333, 10.00, 10.666667, 5.50, 6.833333, 14.00, 14.666667, 7.50, 11.00, 22.50, 23.50, 12.00, 12.50, 25.50, 26.50, 13.50, 8.833333, 18.00, 18.666666, 9.50,
                                                        4.416667, 9.00, 9.333333, 4.75, 7.00, 14.25, 14.75, 7.50, 7.75, 15.75, 16.25, 8.25, 5.416667, 11.00, 11.333333, 5.75, 6.416667, 13.00, 13.333333, 6.75, 10.00, 20.25, 20.75, 
                                                        10.50, 10.75, 21.75, 22.25, 11.25, 7.416667, 15.00, 15.333333, 7.75, 14.833333, 30.00, 30.666666, 15.50, 23.00, 46.50, 47.50, 24.00, 24.50, 49.50, 50.50, 25.50, 16.833334, 
                                                        34.00, 34.666668, 17.50, 18.833334, 38.00, 38.666668, 19.50, 29.00, 58.50, 59.50, 30.00, 30.50, 61.50, 62.50, 31.50, 20.833334, 42.00, 42.666668, 21.50, 10.416667, 21.00, 
                                                        21.333334, 10.75, 16.00, 32.25, 32.75, 16.50, 16.75, 33.75, 34.25, 17.25, 11.416667, 23.00, 23.333334, 11.75, 12.416667, 25.00, 25.333334, 12.75, 19.00, 38.25, 38.75, 19.50,
                                                        19.75, 39.75, 40.25, 20.25, 13.416667, 27.00, 27.333334, 13.75, 26.833334, 54.00, 54.666668, 27.50, 41.00, 82.50, 83.50, 42.00, 42.50, 85.50, 86.50, 43.50, 28.833334, 58.00,
                                                        58.666668, 29.50, 30.833334, 62.00, 62.666668, 31.50, 47.00, 94.50, 95.50, 48.00, 48.50, 97.50, 98.50, 49.50, 32.833332, 66.00, 66.666664, 33.50, 16.416666, 33.00, 33.333332,
                                                        16.75, 25.00, 50.25, 50.75, 25.50, 25.75, 51.75, 52.25, 26.25, 17.416666, 35.00, 35.333332, 17.75, 18.416666, 37.00, 37.333332, 18.75, 28.00, 56.25, 56.75, 28.50, 28.75, 
                                                        57.75, 58.25, 29.25, 19.416666, 39.00, 39.333332, 19.75, 38.833332, 78.00, 78.666664, 39.50, 59.00, 118.50, 119.50, 60.00, 60.50, 121.50, 122.50, 61.50, 40.833332, 82.00,
                                                        82.666664, 41.50, 42.833332, 86.00, 86.666664, 43.50, 65.00, 130.50, 131.50, 66.00, 66.50, 133.50, 134.50, 67.50, 44.833332, 90.00, 90.666664, 45.50, 22.416666, 45.00, 
                                                        45.333332, 22.75, 34.00, 68.25, 68.75, 34.50, 34.75, 69.75, 70.25, 35.25, 23.416666, 47.00, 47.333332, 23.75, 24.416666, 49.00, 49.333332, 24.75, 37.00, 74.25, 74.75, 
                                                        37.50, 37.75, 75.75, 76.25, 38.25, 25.416666, 51.00, 51.333332, 25.75, 50.833332, 102.00, 102.666664, 51.50, 77.00, 154.50, 155.50, 78.00, 78.50, 157.50, 158.50, 79.50,
                                                        52.833332, 106.00, 106.666664, 53.50, 54.833332, 110.00, 110.666664, 55.50, 83.00, 166.50, 167.50, 84.00, 84.50, 169.50, 170.50, 85.50, 56.833332, 114.00, 114.666664, 
                                                        57.50, 28.416666, 57.00, 57.333332, 28.75, 43.00, 86.25, 86.75, 43.50, 43.75, 87.75, 88.25, 44.25, 29.416666, 59.00, 59.333332, 29.75, 30.416666, 61.00, 61.333332, 30.75,
                                                        46.00, 92.25, 92.75, 46.50, 46.75, 93.75, 94.25, 47.25, 31.416666, 63.00, 63.333332, 31.75, 62.833332, 126.00, 126.666664, 63.50, 95.00, 190.50, 191.50, 96.00, 96.50,
                                                        193.50, 194.50, 97.50, 64.833336, 130.00, 130.666672, 65.50, 66.833336, 134.00, 134.666672, 67.50, 101.00, 202.50, 203.50, 102.00, 102.50, 205.50, 206.50, 103.50, 
                                                        68.833336, 138.00, 138.666672, 69.50, 34.416668, 69.00, 69.333336, 34.75, 52.00, 104.25, 104.75, 52.50, 52.75, 105.75, 106.25, 53.25, 35.416668, 71.00, 71.333336, 35.75});
    NDArrayFactory<double>::linspace(1., input);
    
    nd4j::ops::avgpool3dnew<double> op;
    ResultSet<double>* results = op.execute({&input}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));   

    delete results;
}     

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, avgpool3d_bp_test1) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME,  0-VALID
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<double> gradO   ('c', {bS, iC, oD, oH, oW});
    NDArray<double> expected('c', {bS, iC, iD, iH, iW}, {0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.666667, 1.333333, 0.666667,0.666667, 1.333333, 0.666667,0.333333, 0.666667, 0.333333,
                                                         0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,
                                                         0.333333, 0.666667, 0.333333,0.666667, 1.333333, 0.666667,0.666667, 1.333333, 0.666667,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,
                                                         0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.666667, 1.333333, 0.666667,0.666667, 1.333333, 0.666667,0.333333, 0.666667, 0.333333,
                                                         0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,
                                                         0.333333, 0.666667, 0.333333,0.666667, 1.333333, 0.666667,0.666667, 1.333333, 0.666667,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,
                                                         0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.666667, 1.333333, 0.666667,0.666667, 1.333333, 0.666667,0.333333, 0.666667, 0.333333,
                                                         0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,
                                                         0.333333, 0.666667, 0.333333,0.666667, 1.333333, 0.666667,0.666667, 1.333333, 0.666667,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667,0.333333, 0.666667, 0.333333,0.333333, 0.666667, 0.333333,0.166667, 0.333333, 0.166667});
    NDArrayFactory<double>::linspace(1., input);
    gradO = 2.;
    
    nd4j::ops::avgpool3dnew_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, avgpool3d_bp_test2) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=1,pH=1,pW=1,  dD=1,dH=1,dW=1;
    int oD=4,oH=4,oW=4;
    int paddingMode = 0;             // 1-SAME,  0-VALID
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<double> gradO   ('c', {bS, iC, oD, oH, oW});
    NDArray<double> expected('c', {bS, iC, iD, iH, iW}, {1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,
                                                         1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,
                                                         1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,
                                                         1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,
                                                         1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,
                                                         1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,
                                                         1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,
                                                         1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,
                                                         1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333,1.333333, 1.333333, 1.333333,2. , 2. , 2. ,2. , 2. , 2. ,1.333333, 1.333333, 1.333333});
    NDArrayFactory<double>::linspace(1., input);
    gradO = 2.;
    
    nd4j::ops::avgpool3dnew_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, avgpool3d_bp_test3) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int oD=3,oH=4,oW=3;
    int paddingMode = 1;             // 1-SAME,  0-VALID
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iD, iH, iW, iC});
    NDArray<double> gradO   ('c', {bS, oD, oH, oW, iC});
    NDArray<double> expected('c', {bS, iD, iH, iW, iC}, {0.41667, 0.41667, 0.41667,0.83333, 0.83333, 0.83333,1.25, 1.25, 1.25 ,0.58333, 0.58333, 0.58333,1.16667, 1.16667, 1.16667,1.75, 1.75, 1.75 ,0.58333, 0.58333, 0.58333,1.16667, 1.16667, 1.16667,1.75, 1.75, 1.75 ,
                                                         0.41667, 0.41667, 0.41667,0.83333, 0.83333, 0.83333,1.25, 1.25, 1.25 ,0.83333, 0.83333, 0.83333,1.66667, 1.66667, 1.66667,2.5 , 2.5 , 2.5  ,1.16667, 1.16667, 1.16667,2.33333, 2.33333, 2.33333,3.5 , 3.5 , 3.5  ,
                                                         1.16667, 1.16667, 1.16667,2.33333, 2.33333, 2.33333,3.5 , 3.5 , 3.5  ,0.83333, 0.83333, 0.83333,1.66667, 1.66667, 1.66667,2.5 , 2.5 , 2.5  ,1.25   , 1.25   , 1.25   ,2.5    , 2.5    , 2.5    ,3.75, 3.75, 3.75 ,
                                                         1.75   , 1.75   , 1.75   ,3.5    , 3.5    , 3.5    ,5.25, 5.25, 5.25 ,1.75   , 1.75   , 1.75   ,3.5    , 3.5    , 3.5    ,5.25, 5.25, 5.25 ,1.25   , 1.25   , 1.25   ,2.5    , 2.5    , 2.5    ,3.75, 3.75, 3.75 ,
                                                         0.41667, 0.41667, 0.41667,0.83333, 0.83333, 0.83333,1.25, 1.25, 1.25 ,0.58333, 0.58333, 0.58333,1.16667, 1.16667, 1.16667,1.75, 1.75, 1.75 ,0.58333, 0.58333, 0.58333,1.16667, 1.16667, 1.16667,1.75, 1.75, 1.75 ,
                                                         0.41667, 0.41667, 0.41667,0.83333, 0.83333, 0.83333,1.25, 1.25, 1.25 ,0.83333, 0.83333, 0.83333,1.66667, 1.66667, 1.66667,2.5 , 2.5 , 2.5  ,1.16667, 1.16667, 1.16667,2.33333, 2.33333, 2.33333,3.5 , 3.5 , 3.5  ,
                                                         1.16667, 1.16667, 1.16667,2.33333, 2.33333, 2.33333,3.5 , 3.5 , 3.5  ,0.83333, 0.83333, 0.83333,1.66667, 1.66667, 1.66667,2.5 , 2.5 , 2.5  ,1.25   , 1.25   , 1.25   ,2.5    , 2.5    , 2.5    ,3.75, 3.75, 3.75 ,
                                                         1.75   , 1.75   , 1.75   ,3.5    , 3.5    , 3.5    ,5.25, 5.25, 5.25 ,1.75   , 1.75   , 1.75   ,3.5    , 3.5    , 3.5    ,5.25, 5.25, 5.25 ,1.25   , 1.25   , 1.25   ,2.5    , 2.5    , 2.5    ,3.75, 3.75, 3.75 });
    NDArrayFactory<double>::linspace(1., input);
    gradO = 2.;
    
    nd4j::ops::avgpool3dnew_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 0, dataFormat});
    NDArray<double>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, avgpool3d_bp_test4) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int oD=3,oH=4,oW=3;
    int paddingMode = 0;             // 1-SAME,  0-VALID
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iD, iH, iW, iC});
    NDArray<double> gradO   ('c', {bS, oD, oH, oW, iC});
    NDArray<double> expected('c', {bS, iD, iH, iW, iC}, {0.16667, 0.16667, 0.16667,0.33333, 0.33333, 0.33333,0.5 , 0.5  , 0.5  ,0.33333, 0.33333, 0.33333,0.66667, 0.66667, 0.66667,1.  , 1.   , 1.   ,0.58333, 0.58333, 0.58333,1.16667, 1.16667, 1.16667,1.75, 1.75 , 1.75 ,
                                                         0.91667, 0.91667, 0.91667,1.83333, 1.83333, 1.83333,2.75, 2.75 , 2.75 ,0.33333, 0.33333, 0.33333,0.66667, 0.66667, 0.66667,1.  , 1.   , 1.   ,0.66667, 0.66667, 0.66667,1.33333, 1.33333, 1.33333,2.  , 2.   , 2.   ,
                                                         1.16667, 1.16667, 1.16667,2.33333, 2.33333, 2.33333,3.5 , 3.5  , 3.5  ,1.83333, 1.83333, 1.83333,3.66667, 3.66667, 3.66667,5.5 , 5.5  , 5.5  ,0.5    , 0.5    , 0.5    ,1.     , 1.     , 1.     ,1.5 , 1.5  , 1.5  ,
                                                         1.     , 1.     , 1.     ,2.     , 2.     , 2.     ,3.  , 3.   , 3.   ,1.75   , 1.75   , 1.75   ,3.5    , 3.5    , 3.5    ,5.25, 5.25 , 5.25 ,2.75   , 2.75   , 2.75   ,5.5    , 5.5    , 5.5    ,8.25, 8.25 , 8.25 ,
                                                         0.16667, 0.16667, 0.16667,0.33333, 0.33333, 0.33333,0.5 , 0.5  , 0.5  ,0.33333, 0.33333, 0.33333,0.66667, 0.66667, 0.66667,1.  , 1.   , 1.   ,0.58333, 0.58333, 0.58333,1.16667, 1.16667, 1.16667,1.75, 1.75 , 1.75 ,
                                                         0.91667, 0.91667, 0.91667,1.83333, 1.83333, 1.83333,2.75, 2.75 , 2.75 ,0.33333, 0.33333, 0.33333,0.66667, 0.66667, 0.66667,1.  , 1.   , 1.   ,0.66667, 0.66667, 0.66667,1.33333, 1.33333, 1.33333,2.  , 2.   , 2.   ,
                                                         1.16667, 1.16667, 1.16667,2.33333, 2.33333, 2.33333,3.5 , 3.5  , 3.5  ,1.83333, 1.83333, 1.83333,3.66667, 3.66667, 3.66667,5.5 , 5.5  , 5.5  ,0.5    , 0.5    , 0.5    ,1.     , 1.     , 1.     ,1.5 , 1.5  , 1.5  ,
                                                         1.     , 1.     , 1.     ,2.     , 2.     , 2.     ,3.  , 3.   , 3.   ,1.75   , 1.75   , 1.75   ,3.5    , 3.5    , 3.5    ,5.25, 5.25 , 5.25 ,2.75   , 2.75   , 2.75   ,5.5    , 5.5    , 5.5    ,8.25, 8.25 , 8.25 });
    NDArrayFactory<double>::linspace(1., input);
    gradO = 2.;
    
    nd4j::ops::avgpool3dnew_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 0, dataFormat});
    NDArray<double>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
 
    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, maxpool3d_test1) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0, dD=1,dH=1,dW=1;
    int oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME,  0-VALID
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<double> expected('c', {bS, iC, oD, oH, oW}, {20., 21., 23., 24., 32., 33., 35., 36., 56., 57., 59., 60., 68., 69., 71., 72., 92., 93., 95., 96.,104.,105.,107.,108.,
                                                         128.,129.,131.,132.,140.,141.,143.,144.,164.,165.,167.,168.,176.,177.,179.,180.,200.,201.,203.,204.,212.,213.,215.,216.});
    NDArrayFactory<double>::linspace(1., input);
    
    nd4j::ops::maxpool3dnew<double> op;
    ResultSet<double>* results = op.execute({&input}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, maxpool3d_test2) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0, dD=1,dH=1,dW=1;
    int oD=3,oH=4,oW=3;
    int paddingMode = 1;             // 1-SAME,  0-VALID
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iD, iH, iW, iC});
    NDArray<double> expected('c', {bS, oD, oH, oW, iC}, { 49.,  50.,  51., 52.,  53.,  54., 52.,  53.,  54., 58.,  59.,  60., 61.,  62.,  63., 61.,  62.,  63., 67.,  68.,  69., 70.,  71.,  72., 70.,  71.,  72., 67.,  68.,  69., 70.,  71.,  72., 70.,  71.,  72.,
                                                          85.,  86.,  87., 88.,  89.,  90., 88.,  89.,  90., 94.,  95.,  96., 97.,  98.,  99., 97.,  98.,  99.,103., 104., 105.,106., 107., 108.,106., 107., 108.,103., 104., 105.,106., 107., 108.,106., 107., 108.,
                                                          85.,  86.,  87., 88.,  89.,  90., 88.,  89.,  90., 94.,  95.,  96., 97.,  98.,  99., 97.,  98.,  99.,103., 104., 105.,106., 107., 108.,106., 107., 108.,103., 104., 105.,106., 107., 108.,106., 107., 108.,
                                                         157., 158., 159.,160., 161., 162.,160., 161., 162.,166., 167., 168.,169., 170., 171.,169., 170., 171.,175., 176., 177.,178., 179., 180.,178., 179., 180.,175., 176., 177.,178., 179., 180.,178., 179., 180.,
                                                         193., 194., 195.,196., 197., 198.,196., 197., 198.,202., 203., 204.,205., 206., 207.,205., 206., 207.,211., 212., 213.,214., 215., 216.,214., 215., 216.,211., 212., 213.,214., 215., 216.,214., 215., 216.,
                                                         193., 194., 195.,196., 197., 198.,196., 197., 198.,202., 203., 204.,205., 206., 207.,205., 206., 207.,211., 212., 213.,214., 215., 216.,214., 215., 216.,211., 212., 213.,214., 215., 216.,214., 215., 216.});
    NDArrayFactory<double>::linspace(1., input);    

    nd4j::ops::maxpool3dnew<double> op;
    ResultSet<double>* results = op.execute({&input}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);        

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
 
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, maxpool3d_test3) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0, dD=1,dH=1,dW=1;
    int oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME,  0-VALID
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iD, iH, iW, iC});
    NDArray<double> expected('c', {bS, oD, oH, oW, iC}, {58.,  59.,  60., 61.,  62.,  63., 67.,  68.,  69., 70.,  71.,  72., 94.,  95.,  96., 97.,  98.,  99.,103., 104., 105.,106., 107., 108.,
                                                         166., 167., 168.,169., 170., 171.,175., 176., 177.,178., 179., 180.,202., 203., 204.,205., 206., 207.,211., 212., 213.,214., 215., 216.});
    NDArrayFactory<double>::linspace(1., input);    

    nd4j::ops::maxpool3dnew<double> op;
    ResultSet<double>* results = op.execute({&input}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    

    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, maxpool3d_test4) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=1,pH=1,pW=1, dD=1,dH=1,dW=1;
    int oD=4,oH=4,oW=4;
    int paddingMode = 0;             // -SAME,  0-VALID
    int dataFormat  = 0;             // -NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<double> expected('c', {bS, iC, oD, oH, oW},{  4.,   5.,   6.,   6.,  7.,   8.,   9.,   9., 10.,  11.,  12.,  12., 10.,  11.,  12.,  12., 16.,  17.,  18.,  18., 19.,  20.,  21.,  21., 22.,  23.,  24.,  24., 22.,  23.,  24.,  24., 28.,  29.,  30.,  30., 31.,  32.,  33.,  33., 34.,  35.,  36.,  36., 34.,  35.,  36.,  36.,
                                                         28.,  29.,  30.,  30., 31.,  32.,  33.,  33., 34.,  35.,  36.,  36., 34.,  35.,  36.,  36., 40.,  41.,  42.,  42., 43.,  44.,  45.,  45., 46.,  47.,  48.,  48., 46.,  47.,  48.,  48., 52.,  53.,  54.,  54., 55.,  56.,  57.,  57., 58.,  59.,  60.,  60., 58.,  59.,  60.,  60.,
                                                         64.,  65.,  66.,  66., 67.,  68.,  69.,  69., 70.,  71.,  72.,  72., 70.,  71.,  72.,  72., 64.,  65.,  66.,  66., 67.,  68.,  69.,  69., 70.,  71.,  72.,  72., 70.,  71.,  72.,  72., 76.,  77.,  78.,  78., 79.,  80.,  81.,  81., 82.,  83.,  84.,  84., 82.,  83.,  84.,  84.,
                                                         88.,  89.,  90.,  90., 91.,  92.,  93.,  93., 94.,  95.,  96.,  96., 94.,  95.,  96.,  96.,100., 101., 102., 102.,103., 104., 105., 105.,106., 107., 108., 108.,106., 107., 108., 108.,100., 101., 102., 102.,103., 104., 105., 105.,106., 107., 108., 108.,106., 107., 108., 108.,
                                                        112., 113., 114., 114.,115., 116., 117., 117.,118., 119., 120., 120.,118., 119., 120., 120.,124., 125., 126., 126.,127., 128., 129., 129.,130., 131., 132., 132.,130., 131., 132., 132.,136., 137., 138., 138.,139., 140., 141., 141.,142., 143., 144., 144.,142., 143., 144., 144.,
                                                        136., 137., 138., 138.,139., 140., 141., 141.,142., 143., 144., 144.,142., 143., 144., 144.,148., 149., 150., 150.,151., 152., 153., 153.,154., 155., 156., 156.,154., 155., 156., 156.,160., 161., 162., 162.,163., 164., 165., 165.,166., 167., 168., 168.,166., 167., 168., 168.,
                                                        172., 173., 174., 174.,175., 176., 177., 177.,178., 179., 180., 180.,178., 179., 180., 180.,172., 173., 174., 174.,175., 176., 177., 177.,178., 179., 180., 180.,178., 179., 180., 180.,184., 185., 186., 186.,187., 188., 189., 189.,190., 191., 192., 192.,190., 191., 192., 192.,
                                                        196., 197., 198., 198.,199., 200., 201., 201.,202., 203., 204., 204.,202., 203., 204., 204.,208., 209., 210., 210.,211., 212., 213., 213.,214., 215., 216., 216.,214., 215., 216., 216.,208., 209., 210., 210.,211., 212., 213., 213.,214., 215., 216., 216.,214., 215., 216., 216.});
    NDArrayFactory<double>::linspace(1., input);
    
    nd4j::ops::maxpool3dnew<double> op;
    ResultSet<double>* results = op.execute({&input}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));   

    delete results;
}     


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, maxpool3d_bp_test1) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME,  0-VALID
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<double> gradO   ('c', {bS, iC, oD, oH, oW});
    NDArray<double> expected('c', {bS, iC, iD, iH, iW}, {0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0.1, 0.2,0. , 0.3, 0.4,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0.5, 0.6,0. , 0.7, 0.8,
                                                         0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0.9, 1. ,0. , 1.1, 1.2,0. , 0. , 0. ,0. , 0. , 0. ,0. , 1.3, 1.4,0. , 1.5, 1.6,
                                                         0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 1.7, 1.8,0. , 1.9, 2. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 2.1, 2.2,0. , 2.3, 2.4,
                                                         0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 2.5, 2.6,0. , 2.7, 2.8,0. , 0. , 0. ,0. , 0. , 0. ,0. , 2.9, 3. ,0. , 3.1, 3.2,
                                                         0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 3.3, 3.4,0. , 3.5, 3.6,0. , 0. , 0. ,0. , 0. , 0. ,0. , 3.7, 3.8,0. , 3.9, 4. ,
                                                         0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 4.1, 4.2,0. , 4.3, 4.4,0. , 0. , 0. ,0. , 0. , 0. ,0. , 4.5, 4.6,0. , 4.7, 4.8});
    NDArrayFactory<double>::linspace(1., input);
    NDArrayFactory<double>::linspace(0.1, gradO, 0.1);
    
    nd4j::ops::maxpool3dnew_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, maxpool3d_bp_test2) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=1,pH=1,pW=1,  dD=1,dH=1,dW=1;
    int oD=4,oH=4,oW=4;
    int paddingMode = 0;             // 1-SAME,  0-VALID
    int dataFormat  = 0;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<double> gradO   ('c', {bS, iC, oD, oH, oW});
    NDArray<double> expected('c', {bS, iC, iD, iH, iW}, {0.000e+00, 0.000e+00, 0.000e+00,1.000e-01, 2.000e-01, 7.000e-01,5.000e-01, 6.000e-01, 1.500e+00,2.200e+00, 2.400e+00, 5.400e+00,0.000e+00, 0.000e+00, 0.000e+00,1.700e+00, 1.800e+00, 3.900e+00,2.100e+00, 2.200e+00, 4.700e+00,5.400e+00, 5.600e+00, 1.180e+01,
                                                         0.000e+00, 0.000e+00, 0.000e+00,8.200e+00, 8.400e+00, 1.740e+01,9.000e+00, 9.200e+00, 1.900e+01,2.040e+01, 2.080e+01, 4.280e+01,0.000e+00, 0.000e+00, 0.000e+00,6.500e+00, 6.600e+00, 1.350e+01,6.900e+00, 7.000e+00, 1.430e+01,1.500e+01, 1.520e+01, 3.100e+01,
                                                         0.000e+00, 0.000e+00, 0.000e+00,8.100e+00, 8.200e+00, 1.670e+01,8.500e+00, 8.600e+00, 1.750e+01,1.820e+01, 1.840e+01, 3.740e+01,0.000e+00, 0.000e+00, 0.000e+00,2.100e+01, 2.120e+01, 4.300e+01,2.180e+01, 2.200e+01, 4.460e+01,4.600e+01, 4.640e+01, 9.400e+01,
                                                         0.000e+00, 0.000e+00, 0.000e+00,1.290e+01, 1.300e+01, 2.630e+01,1.330e+01, 1.340e+01, 2.710e+01,2.780e+01, 2.800e+01, 5.660e+01,0.000e+00, 0.000e+00, 0.000e+00,1.450e+01, 1.460e+01, 2.950e+01,1.490e+01, 1.500e+01, 3.030e+01,3.100e+01, 3.120e+01, 6.300e+01,
                                                         0.000e+00, 0.000e+00, 0.000e+00,3.380e+01, 3.400e+01, 6.860e+01,3.460e+01, 3.480e+01, 7.020e+01,7.160e+01, 7.200e+01, 1.452e+02,0.000e+00, 0.000e+00, 0.000e+00,1.930e+01, 1.940e+01, 3.910e+01,1.970e+01, 1.980e+01, 3.990e+01,4.060e+01, 4.080e+01, 8.220e+01,
                                                         0.000e+00, 0.000e+00, 0.000e+00,2.090e+01, 2.100e+01, 4.230e+01,2.130e+01, 2.140e+01, 4.310e+01,4.380e+01, 4.400e+01, 8.860e+01,0.000e+00, 0.000e+00, 0.000e+00,4.660e+01, 4.680e+01, 9.420e+01,4.740e+01, 4.760e+01, 9.580e+01,9.720e+01, 9.760e+01, 1.964e+02,
                                                         0.000e+00, 0.000e+00, 0.000e+00,2.570e+01, 2.580e+01, 5.190e+01,2.610e+01, 2.620e+01, 5.270e+01,5.340e+01, 5.360e+01, 1.078e+02,0.000e+00, 0.000e+00, 0.000e+00,2.730e+01, 2.740e+01, 5.510e+01,2.770e+01, 2.780e+01, 5.590e+01,5.660e+01, 5.680e+01, 1.142e+02,
                                                         0.000e+00, 0.000e+00, 0.000e+00,5.940e+01, 5.960e+01, 1.198e+02,6.020e+01, 6.040e+01, 1.214e+02,1.228e+02, 1.232e+02, 2.476e+02,0.000e+00, 0.000e+00, 0.000e+00,3.210e+01, 3.220e+01, 6.470e+01,3.250e+01, 3.260e+01, 6.550e+01,6.620e+01, 6.640e+01, 1.334e+02,
                                                         0.000e+00, 0.000e+00, 0.000e+00,3.370e+01, 3.380e+01, 6.790e+01,3.410e+01, 3.420e+01, 6.870e+01,6.940e+01, 6.960e+01, 1.398e+02,0.000e+00, 0.000e+00, 0.000e+00,7.220e+01, 7.240e+01, 1.454e+02,7.300e+01, 7.320e+01, 1.470e+02,1.484e+02, 1.488e+02, 2.988e+02});
    NDArrayFactory<double>::linspace(1., input);
    NDArrayFactory<double>::linspace(0.1, gradO, 0.1);
    
    nd4j::ops::maxpool3dnew_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, maxpool3d_bp_test3) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int oD=3,oH=4,oW=3;
    int paddingMode = 1;             // 1-SAME,  0-VALID
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iD, iH, iW, iC});
    NDArray<double> gradO   ('c', {bS, oD, oH, oW, iC});
    NDArray<double> expected('c', {bS, iD, iH, iW, iC}, { 0., 0., 0.,  0. ,  0. ,  0. ,  0. ,   0.  ,  0. ,  0. , 0. ,  0. ,  0. ,   0.  ,  0. ,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. ,
                                                          0., 0., 0.,  0. ,  0. ,  0. ,  0. ,   0.  ,  0. ,  0. , 0. ,  0. ,  0. ,   0.  ,  0. ,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. ,  0.1,   0.2    ,  0.3,  1.1,   1.3    ,  1.5,
                                                          0., 0., 0.,  1. ,  1.1,  1.2,  2.9,   3.1 ,  3.3,  0. , 0. ,  0. ,  4.7,   4.9 ,  5.1, 11.2,  11.6    , 12. ,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. ,
                                                          0., 0., 0., 11. , 11.2, 11.4, 23.8,  24.2 , 24.6,  0. , 0. ,  0. , 12.8,  13.  , 13.2, 27.4,  27.8    , 28.2,  0. ,   0.     ,  0. , 31. ,  31.4    , 31.8, 65.6,  66.39999, 67.2,
                                                          0., 0., 0.,  0. ,  0. ,  0. ,  0. ,   0.  ,  0. ,  0. , 0. ,  0. ,  0. ,   0.  ,  0. ,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. ,
                                                          0., 0., 0.,  0. ,  0. ,  0. ,  0. ,   0.  ,  0. ,  0. , 0. ,  0. ,  0. ,   0.  ,  0. ,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. , 10.9,  11.     , 11.1, 22.7,  22.9    , 23.1,
                                                          0., 0., 0., 11.8, 11.9, 12. , 24.5,  24.7 , 24.9,  0. , 0. ,  0. , 26.3,  26.5 , 26.7, 54.4,  54.8    , 55.2,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. ,  0. ,   0.     ,  0. ,
                                                          0., 0., 0., 32.6, 32.8, 33. , 67. ,  67.4 , 67.8,  0. , 0. ,  0. , 34.4,  34.6 , 34.8, 70.6,  71.     , 71.4,  0. ,   0.     ,  0. , 74.2,  74.6    , 75. ,152. , 152.8    ,153.6});
    NDArrayFactory<double>::linspace(1., input);
    NDArrayFactory<double>::linspace(0.1, gradO, 0.1);
    
    nd4j::ops::maxpool3dnew_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

 
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, maxpool3d_bp_test4) {

    int bS=2, iD=3,iH=4,iW=3,  iC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int oD=2,oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME,  0-VALID
    int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW

    NDArray<double> input   ('c', {bS, iD, iH, iW, iC});
    NDArray<double> gradO   ('c', {bS, oD, oH, oW, iC});
    NDArray<double> expected('c', {bS, iD, iH, iW, iC}, {0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,
                                                         0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0.1, 0.2, 0.3,0.4, 0.5, 0.6,0. , 0. , 0. ,0.7, 0.8, 0.9,1. , 1.1, 1.2,
                                                         0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,1.3, 1.4, 1.5,1.6, 1.7, 1.8,0. , 0. , 0. ,1.9, 2. , 2.1,2.2, 2.3, 2.4,
                                                         0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,
                                                         0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,2.5, 2.6, 2.7,2.8, 2.9, 3. ,0. , 0. , 0. ,3.1, 3.2, 3.3,3.4, 3.5, 3.6,
                                                         0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,0. , 0. , 0. ,3.7, 3.8, 3.9,4. , 4.1, 4.2,0. , 0. , 0. ,4.3, 4.4, 4.5,4.6, 4.7, 4.8});
    NDArrayFactory<double>::linspace(1., input);
    NDArrayFactory<double>::linspace(0.1, gradO, 0.1);
    
    nd4j::ops::maxpool3dnew_bp<double> op;
    ResultSet<double>* results = op.execute({&input, &gradO}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  paddingMode, 1, dataFormat});
    NDArray<double>* output = results->at(0);    
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
 
    delete results;
}
 
//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, maxpool2d_test1) {
  
    int bS = 3;                 // batch size (number of samples)
    int iC = 3;                 // input channels
    int iH = 28, iW = 28;       // input height/width
    int kH = 2,  kW = 2;        // kernel (filter) height/width
    int sH = 1,  sW = 1;        // stride height/width
    int pH = 0,  pW = 0;        // padding height/width
    int dH = 1,  dW = 1;        // dilation height/width
    
    int oH = 27, oW = 27;       // output height/width
    
    int isSameMode = 0;         // 1-SAME,  0-VALID

    NDArray<double> input   ('c', {bS, iC, iH, iW});    
    
    nd4j::ops::maxpool2d<double> op;
    ResultSet<double>* results = op.execute({&input}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, isSameMode, 1, 0});
    NDArray<double>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());    
    ASSERT_TRUE(output->isSameShape({bS, iC, oH, oW}));    
    
    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test1) {
  
    const int rows = 3;
    const int cols = 5;
    
    NDArray<float> expected('c', {rows, cols}, {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0});    
    
    nd4j::ops::tri<float> op;
    ResultSet<float>* results = op.execute({}, {}, {rows, cols});
    NDArray<float>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test2) {
  
    const int rows = 3;
    const int cols = 5;
    const int diag = 2;
    
    NDArray<float> expected('c', {rows, cols}, {1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});    
    
    nd4j::ops::tri<float> op;
    ResultSet<float>* results = op.execute({}, {}, {rows, cols, diag});
    NDArray<float>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test3) {
  
    const int rows = 3;
    const int cols = 5;
    const int diag = -1;
    
    NDArray<float> expected('c', {rows, cols}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0});
   
    nd4j::ops::tri<float> op;
    ResultSet<float>* results = op.execute({}, {}, {rows, cols, diag});
    NDArray<float>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test4) {
  
    const int rows = 3;
    const int cols = 5;
    const int diag = -2;
    
    NDArray<float> expected('c', {rows, cols}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0});    
    
    nd4j::ops::tri<float> op;
    ResultSet<float>* results = op.execute({}, {}, {rows, cols, diag});
    NDArray<float>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test5) {
  
    const int rows = 5;    
    
    NDArray<float> expected('c', {rows, rows}, {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1});    
    
    nd4j::ops::tri<float> op;
    ResultSet<float>* results = op.execute({}, {}, {rows});
    NDArray<float>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test6) {
  
    const int rows = 3;
    const int cols = 5;
    const int diag = -20;
    
    NDArray<float> expected('c', {rows, cols}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});    
    
    nd4j::ops::tri<float> op;
    ResultSet<float>* results = op.execute({}, {}, {rows, cols, diag});
    NDArray<float>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, tri_test7) {
  
    const int rows = 3;
    const int cols = 5;
    const int diag = 20;
    
    NDArray<float> expected('c', {rows, cols}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});    
    
    nd4j::ops::tri<float> op;
    ResultSet<float>* results = op.execute({}, {}, {rows, cols, diag});
    NDArray<float>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test1) {    
    
    NDArray<float> input('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});    
    NDArray<float> expected('c', {4, 3}, {1,  2,  3, 0, 5, 6, 0,  0,  9, 0,  0, 0});    
    
    nd4j::ops::triu<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test2) {    
    
    NDArray<float> input('c', {4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});    
    NDArray<float> expected('c', {4, 3}, {1,  2,  3,4,  5,  6,0,  8,  9,0,  0, 12});    
    
    nd4j::ops::triu<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {-1});
    NDArray<float>* output = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test3) {    
    
    NDArray<float> input('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});    
    NDArray<float> expected('c', {2, 3, 2}, {1, 2,3, 4,0, 6,7, 8,9,10,0,12});    
    
    nd4j::ops::triu<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {-1});
    NDArray<float>* output = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test4) {    
    
    NDArray<float> input('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});    
    NDArray<float> expected('c', {2, 3, 2}, {1,  2,0,  4,0,  0,7,  8,0, 10,0,  0});    
    
    nd4j::ops::triu<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test5) {    
    
    NDArray<float> input('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});    
    NDArray<float> expected('c', {2, 3, 2}, {0, 2,0,  0,0,  0,0,  8,0, 0,0,  0});    
    
    nd4j::ops::triu<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {1});
    NDArray<float>* output = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test6) {    
    
    NDArray<float> input('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});    
    NDArray<float> expected('c', {2, 3, 2}, {0, 0,0,  0,0,  0,0,  0,0, 0,0,  0});    
    
    nd4j::ops::triu<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {10});
    NDArray<float>* output = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test7) {    
    
    NDArray<float> input('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});    
    NDArray<float> expected('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});    
    
    nd4j::ops::triu<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {-10});
    NDArray<float>* output = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test8) {    
    
    NDArray<float> input('c', {6}, {1, 2, 3, 4, 5, 6});    
    NDArray<float> expected('c', {6, 6}, {1, 2, 3, 4, 5, 6,0, 2, 3, 4, 5, 6,0, 0, 3, 4, 5, 6,0, 0, 0, 4, 5, 6,0, 0, 0, 0, 5, 6,0, 0, 0, 0, 0, 6});    
    
    nd4j::ops::triu<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {});
    NDArray<float>* output = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test9) {    
    
    NDArray<float> input('c', {6}, {1, 2, 3, 4, 5, 6});    
    NDArray<float> expected('c', {6, 6}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 0, 2, 3, 4, 5, 6, 0, 0, 3, 4, 5, 6});    
    
    nd4j::ops::triu<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {-3});
    NDArray<float>* output = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test10) {    
    
    NDArray<float> input('c', {6}, {1, 2, 3, 4, 5, 6});    
    NDArray<float> expected('c', {6, 6}, {0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    
    nd4j::ops::triu<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {3});
    NDArray<float>* output = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_test11) {    
    
    NDArray<float> input('c', {6}, {1, 2, 3, 4, 5, 6});    
    NDArray<float> expected('c', {6, 6}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6});
    
    nd4j::ops::triu<float> op;
    ResultSet<float>* results = op.execute({&input}, {}, {-58});
    NDArray<float>* output = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_bp_test1) {    
    
    NDArray<float> input('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});    
    NDArray<float> gradO('c', {2, 3, 2});    
    gradO = 0.5;

    NDArray<float> expected('c', {2, 3, 2}, {0.,0.5,0.,0. ,0.,0. ,0.,0.5,0.,0. ,0.,0.});    
    
    nd4j::ops::triu_bp<float> op;
    ResultSet<float>* results = op.execute({&input, &gradO}, {}, {1});
    NDArray<float>* gradI = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(gradI));
    ASSERT_TRUE(expected.equalsTo(gradI));    
       
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_bp_test2) {    
    
    NDArray<float> input('c', {2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});    
    NDArray<float> gradO('c', {2, 3, 2});    
    gradO = 0.5;

    NDArray<float> expected('c', {2, 3, 2}, {0.5,0.5,0. ,0.5,0. ,0. ,0.5,0.5,0. ,0.5,0. ,0.});    
    
    nd4j::ops::triu_bp<float> op;
    ResultSet<float>* results = op.execute({&input, &gradO}, {}, {});
    NDArray<float>* gradI = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(gradI));
    ASSERT_TRUE(expected.equalsTo(gradI));    
       
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_bp_test3) {    
    
    NDArray<float> input('c', {6}, {1, 2, 3, 4, 5, 6});    
    NDArray<float> gradO('c', {6,6});    
    gradO = 0.5;

    NDArray<float> expected('c', {6,6}, {0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0. , 0.5, 0.5, 0.5, 0.5, 0.5,0. , 0. , 0.5, 0.5, 0.5, 0.5,0. , 0. , 0. , 0.5, 0.5, 0.5});    
    
    nd4j::ops::triu_bp<float> op;
    ResultSet<float>* results = op.execute({&input, &gradO}, {}, {-2});
    NDArray<float>* gradI = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(gradI));
    ASSERT_TRUE(expected.equalsTo(gradI));    
       
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, triu_bp_test4) {    
    
    NDArray<float> input('c', {2,3}, {1, 2, 3, 4, 5, 6});    
    NDArray<float> gradO('c', {2,3});    
    gradO = 0.5;

    NDArray<float> expected('c', {2,3}, {0., 0., 0., 0., 0., 0.});    
    
    nd4j::ops::triu_bp<float> op;
    ResultSet<float>* results = op.execute({&input, &gradO}, {}, {10});
    NDArray<float>* gradI = results->at(0);       

    ASSERT_EQ(Status::OK(), results->status());    

    ASSERT_TRUE(expected.isSameShape(gradI));
    ASSERT_TRUE(expected.equalsTo(gradI));    
       
    delete results;
}

