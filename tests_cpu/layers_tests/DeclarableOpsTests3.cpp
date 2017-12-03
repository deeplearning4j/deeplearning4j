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