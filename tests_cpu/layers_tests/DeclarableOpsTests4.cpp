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
    NDArray<float> exp('c', {2, 2, 2, 2}, {6., 7.,  10.,  11.,  22.,  23.,  26.,  27.,  38.,  39.,  42.,  43.,  54.,  55.,  58., 59.});

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
    NDArray<float> exp('c', {2, 2, 2, 2}, {6., 7.,  10.,  11.,  22.,  23.,  26.,  27.,  38.,  39.,  42.,  43.,  54.,  55.,  58., 59.});

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
    NDArray<float> exp('c', {2, 2, 2, 2}, {11.,  12.,  15.,  16.,  27.,  28.,  31.,  32.,  43.,  44.,  47.,  48.,  59.,  60.,  63., 64.});

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
    NDArray<float> exp('c', {2, 2, 2, 2}, {11.,  12.,  15.,  16.,  27.,  28.,  31.,  32.,  43.,  44.,  47.,  48.,  59.,  60.,  63., 64.});

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
    NDArray<float> exp('c', {2, 3, 3, 2}, {7.,    8.,   11.,   12.,   14.,   15.,   27.,   28.,   31.,   32.,   34.,   35., 42.,   43.,   46.,   47.,   49.,   50.,   57.,   58.,   61.,   62.,   64.,   65., 77.,   78.,   81.,   82.,   84.,   85.,   92.,   93.,   96.,   97.,   99.,  100.,});

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
    NDArray<float> exp('c', {2, 2, 2, 2}, {7.,   8.,  11.,  12.,  27.,  28.,  31.,  32.,  57.,  58.,  61.,  62.,  77.,  78.,  81., 82.});

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
    NDArray<float> exp('c', {2, 3, 3, 2}, {1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2., 1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.});

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
    NDArray<float> exp('c', {2, 2, 3, 3}, {1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2., 1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.,  1.,  2.});

    nd4j::ops::biasadd<float> op;
    auto result = op.execute({&x, &bias}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

