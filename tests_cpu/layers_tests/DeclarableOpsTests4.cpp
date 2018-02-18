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


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, conv3d_test1) {

    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 0;             // 0-SAME,  1-VALID;
    int dataFormat  = 0;             // 0-NDHWC, 1-NCDHW

    NDArray<float> input   ('c', {bS, iD, iH, iW, iC});
    NDArray<float> weights ('c', {kD, kH, kW, iC, oC});
    NDArray<float> expected('c', {2, 3, 4, 3, 3}, {64.,64.,64.,64.,64.,64.,32.,32.,32.,96.,96.,96.,96.,96.,96.,48.,48.,48.,96.,96.,96.,96.,96.,96.,48.,48.,48.,
                                                   64.,64.,64.,64.,64.,64.,32.,32.,32.,64.,64.,64.,64.,64.,64.,32.,32.,32.,96.,96.,96.,96.,96.,96.,48.,48.,48.,
                                                   96.,96.,96.,96.,96.,96.,48.,48.,48.,64.,64.,64.,64.,64.,64.,32.,32.,32.,32.,32.,32.,32.,32.,32.,16.,16.,16.,
                                                   48.,48.,48.,48.,48.,48.,24.,24.,24.,48.,48.,48.,48.,48.,48.,24.,24.,24.,32.,32.,32.,32.,32.,32.,16.,16.,16.,
                                                   64.,64.,64.,64.,64.,64.,32.,32.,32.,96.,96.,96.,96.,96.,96.,48.,48.,48.,96.,96.,96.,96.,96.,96.,48.,48.,48.,
                                                   64.,64.,64.,64.,64.,64.,32.,32.,32.,64.,64.,64.,64.,64.,64.,32.,32.,32.,96.,96.,96.,96.,96.,96.,48.,48.,48.,
                                                   96.,96.,96.,96.,96.,96.,48.,48.,48.,64.,64.,64.,64.,64.,64.,32.,32.,32.,32.,32.,32.,32.,32.,32.,16.,16.,16.,
                                                   48.,48.,48.,48.,48.,48.,24.,24.,24.,48.,48.,48.,48.,48.,48.,24.,24.,24.,32.,32.,32.,32.,32.,32.,16.,16.,16.});
    input = 2.;
    weights = 1.;
    
    nd4j::ops::conv3dNew<float> op;
    ResultSet<float>* results = op.execute({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    NDArray<float>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, conv3d_test2) {
    
    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 0;             // 0-SAME,  1-VALID;
    int dataFormat  = 0;             // 0-NDHWC, 1-NCDHW

    NDArray<float> input   ('c', {bS, iD, iH, iW, iC});
    NDArray<float> weights ('c', {kD, kH, kW, iC, oC});
    NDArray<float> expected('c', {2, 3, 4, 3, 3}, {534.4,540.8,547.2,534.4,540.8,547.2,248. ,251.2,254.4,686.4,696. ,705.6,686.4,696. ,705.6,314.4,319.2,324. ,686.4,696. ,705.6,686.4,696. ,705.6,314.4,319.2,324. ,
                                                   380.8,387.2,393.6,380.8,387.2,393.6,171.2,174.4,177.6,534.4,540.8,547.2,534.4,540.8,547.2,248. ,251.2,254.4,686.4,696. ,705.6,686.4,696. ,705.6,314.4,319.2,324. ,
                                                   686.4,696. ,705.6,686.4,696. ,705.6,314.4,319.2,324. ,380.8,387.2,393.6,380.8,387.2,393.6,171.2,174.4,177.6,152. ,155.2,158.4,152. ,155.2,158.4, 66.4, 68. , 69.6,
                                                   170.4,175.2,180. ,170.4,175.2,180. , 70.8, 73.2, 75.6,170.4,175.2,180. ,170.4,175.2,180. , 70.8, 73.2, 75.6, 75.2, 78.4, 81.6, 75.2, 78.4, 81.6, 28. , 29.6, 31.2,
                                                   534.4,540.8,547.2,534.4,540.8,547.2,248. ,251.2,254.4,686.4,696. ,705.6,686.4,696. ,705.6,314.4,319.2,324. ,686.4,696. ,705.6,686.4,696. ,705.6,314.4,319.2,324. ,
                                                   380.8,387.2,393.6,380.8,387.2,393.6,171.2,174.4,177.6,534.4,540.8,547.2,534.4,540.8,547.2,248. ,251.2,254.4,686.4,696. ,705.6,686.4,696. ,705.6,314.4,319.2,324. ,
                                                   686.4,696. ,705.6,686.4,696. ,705.6,314.4,319.2,324. ,380.8,387.2,393.6,380.8,387.2,393.6,171.2,174.4,177.6,152. ,155.2,158.4,152. ,155.2,158.4, 66.4, 68. , 69.6,
                                                   170.4,175.2,180. ,170.4,175.2,180. , 70.8, 73.2, 75.6,170.4,175.2,180. ,170.4,175.2,180. , 70.8, 73.2, 75.6, 75.2, 78.4, 81.6, 75.2, 78.4, 81.6, 28. , 29.6, 31.2});
    input = 2.;
    NDArrayFactory<float>::linspace(0.1, weights, 0.1);
    
    nd4j::ops::conv3dNew<float> op;
    ResultSet<float>* results = op.execute({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    NDArray<float>* output = results->at(0);
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, conv3d_test3) {
    
    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 1;             // 0-SAME,  1-VALID;
    int dataFormat  = 0;             // 0-NDHWC, 1-NCDHW

    NDArray<float> input   ('c', {bS, iD, iH, iW, iC});
    NDArray<float> weights ('c', {kD, kH, kW, iC, oC});
    NDArray<float> expected('c', {2, 2, 2, 2, 3},  {686.4,696.,705.6,686.4,696.,705.6,686.4,696.,705.6,686.4,696.,705.6,
                                                    686.4,696.,705.6,686.4,696.,705.6,686.4,696.,705.6,686.4,696.,705.6,
                                                    686.4,696.,705.6,686.4,696.,705.6,686.4,696.,705.6,686.4,696.,705.6,
                                                    686.4,696.,705.6,686.4,696.,705.6,686.4,696.,705.6,686.4,696.,705.6});
    input = 2.;
    NDArrayFactory<float>::linspace(0.1, weights, 0.1);
    
    nd4j::ops::conv3dNew<float> op;
    ResultSet<float>* results = op.execute({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    NDArray<float>* output = results->at(0);
    
    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}


//////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, conv3d_test4) {
    
    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 1;             // 0-SAME,  1-VALID;
    int dataFormat = 1;              // 0-NDHWC, 1-NCDHW

    NDArray<float> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<float> weights ('c', {oC, iC, kD, kH, kW});
    NDArray<float> expected('c', {2, 3, 2, 2, 2});
    input = 2.;
    weights = 0.5;
    expected = 48.;
    
    nd4j::ops::conv3dNew<float> op;
    ResultSet<float>* results = op.execute({&input, &weights}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    NDArray<float>* output = results->at(0);    

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, conv3d_test5) {
    
    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 1;             // 0-SAME,  1-VALID;
    int dataFormat = 1;              // 0-NDHWC, 1-NCDHW    

    NDArray<float> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<float> weights ('c', {oC, iC, kD, kH, kW});
    NDArray<float> bias    ('c', {oC});
    NDArray<float> expected('c', {2, 3, 2, 2, 2});

    input = 2.;
    weights = 0.5;
    expected = 49.;
    bias = 1.;
    
    nd4j::ops::conv3dNew<float> op;
    ResultSet<float>* results = op.execute({&input, &weights, &bias}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    NDArray<float>* output = results->at(0);
    
    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, conv3d_test6) {
    
    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 1;             // 0-SAME,  1-VALID;
    int dataFormat = 1;              // 0-NDHWC, 1-NCDHW    

    NDArray<float> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<float> weights ('c', {oC, iC, kD, kH, kW});
    NDArray<float> bias    ('c', {oC},{1,2,3});
    NDArray<float> expected('c', {2, 3, 2, 2, 2},{49., 49.,49., 49., 49., 49.,49., 49., 50., 50.,50., 50., 50., 50.,50., 50., 
                                                  51., 51.,51., 51., 51., 51.,51., 51., 49., 49.,49., 49., 49., 49.,49., 49., 
                                                  50., 50.,50., 50., 50., 50.,50., 50., 51., 51.,51., 51., 51., 51.,51., 51.});
    input = 2.;
    weights = 0.5;    
    
    nd4j::ops::conv3dNew<float> op;
    ResultSet<float>* results = op.execute({&input, &weights, &bias}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    NDArray<float>* output = results->at(0);
    
    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
    delete results;
}

////////////////////////////////////////////////////////////////////
TEST_F(DeclarableOpsTests4, conv3d_test7) {
    
    int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
    int paddingMode = 1;             // 0-SAME,  1-VALID;
    int dataFormat = 1;              // 0-NDHWC, 1-NCDHW    

    NDArray<float> input   ('c', {bS, iC, iD, iH, iW});
    NDArray<float> weights ('c', {oC, iC, kD, kH, kW});
    NDArray<float> bias    ('c', {oC},{1,2,3});
    NDArray<float> expected('c', {2, 3, 2, 2, 2},{236.2, 236.2, 236.2, 236.2, 236.2, 236.2, 236.2, 236.2, 698. , 698. , 698. , 698. ,
                                                  698. , 698. , 698. , 698. ,1159.8,1159.8,1159.8,1159.8,1159.8,1159.8,1159.8,1159.8,
                                                  236.2, 236.2, 236.2, 236.2, 236.2, 236.2, 236.2, 236.2, 698. , 698. , 698. , 698. ,
                                                  698. , 698. , 698. , 698. ,1159.8,1159.8,1159.8,1159.8,1159.8,1159.8,1159.8,1159.8});
    input = 2.;
    NDArrayFactory<float>::linspace(0.1, weights, 0.1);
    
    nd4j::ops::conv3dNew<float> op;
    ResultSet<float>* results = op.execute({&input, &weights, &bias}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW, paddingMode, dataFormat});
    NDArray<float>* output = results->at(0);
    
    // output->printIndexedBuffer();

    ASSERT_EQ(Status::OK(), results->status());
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));    
    
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
    

//Targets {15.5f, 15.7f,  5.f , 15.f,   5.f,   6.f};
//----------
//Inputs {11.f, 13.f,  4.f, 15.f,  6.f,  3.f};
//----------
//Weights [0.7]
//Result {-159.50006,  -191.1,       -16.009075, -210., -24.001238,  -15.03887}

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
    NDArray<double> cLast = (*c)({{4,5},{},{}});    

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




