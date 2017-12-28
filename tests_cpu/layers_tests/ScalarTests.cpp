//
//  @author raver119@gmail.com
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <NativeOps.h>
#include <helpers/BitwiseUtils.h>

using namespace nd4j;
using namespace nd4j::graph;

class ScalarTests : public testing::Test {
public:

};

TEST_F(ScalarTests, Test_Create_1) {
    NDArray<float> x(2.0f);

    ASSERT_EQ(0, x.rankOf());
    ASSERT_EQ(1, x.lengthOf());
    ASSERT_TRUE(x.isScalar());
    ASSERT_FALSE(x.isVector());
    ASSERT_FALSE(x.isRowVector());
    ASSERT_FALSE(x.isColumnVector());
    ASSERT_FALSE(x.isMatrix());
}

TEST_F(ScalarTests, Test_Add_1) {
    NDArray<float> x(2.0f);
    NDArray<float> exp(5.0f);

    x += 3.0f;

    ASSERT_NEAR(5.0f, x.getScalar(0), 1e-5f);
    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(ScalarTests, Test_EQ_1) {
    NDArray<float> x(2.0f);
    NDArray<float> y(3.0f);

    ASSERT_TRUE(y.isSameShape(&x));
    ASSERT_FALSE(y.equalsTo(&x));
}

TEST_F(ScalarTests, Test_Concat_1) {
    NDArray<float> t(1.0f);
    NDArray<float> u(2.0f);
    NDArray<float> v(3.0f);
    NDArray<float> exp('c', {3}, {1, 2, 3});

    nd4j::ops::concat<float> op;
    auto result = op.execute({&t, &u, &v}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ScalarTests, Test_Concat_2) {
    NDArray<float> t(1.0f);
    NDArray<float> u('c', {3}, {2, 3, 4});
    NDArray<float> v(5.0f);
    NDArray<float> exp('c', {5}, {1, 2, 3, 4, 5});

    nd4j::ops::concat<float> op;
    auto result = op.execute({&t, &u, &v}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ScalarTests, Test_Concat_3) {
    NDArray<float> t('c', {3}, {1, 2, 3});
    NDArray<float> u(4.0f);
    NDArray<float> v(5.0f);
    NDArray<float> exp('c', {5}, {1, 2, 3, 4, 5});

    nd4j::ops::concat<float> op;
    auto result = op.execute({&t, &u, &v}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    //z->printShapeInfo("z");

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ScalarTests, Test_ExpandDims_1) {
    NDArray<float> x(2.0f);
    std::vector<int> vecS({1});
    std::vector<float> vecD({2.0f});
    NDArray<float> exp('c', vecS, vecD);

    nd4j::ops::expand_dims<float> op;
    auto result = op.execute({&x}, {}, {0});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}