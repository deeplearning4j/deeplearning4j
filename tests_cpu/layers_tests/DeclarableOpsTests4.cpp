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
    auto result = op.execute({&x}, {}, {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1});

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