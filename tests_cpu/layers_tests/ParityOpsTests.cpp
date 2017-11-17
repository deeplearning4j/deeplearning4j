//
// Created by raver119 on 12.10.2017.
//

#include "testlayers.h"
#include <NDArray.h>
#include <ops/declarable/CustomOperations.h>


using namespace nd4j;
using namespace nd4j::ops;

class ParityOpsTests : public testing::Test {
public:

};


TEST_F(ParityOpsTests, TestZeroAs1) {
    NDArray<float> x('c', {10, 10});
    x.assign(1.0);

    NDArray<float> exp('c', {10, 10});
    exp.assign(0.0f);

    nd4j::ops::zeros_as<float> op;

    auto result = op.execute({&x}, {}, {});

    auto z = result->at(0);

    ASSERT_TRUE(z->isSameShape(&x));
    ASSERT_TRUE(z->equalsTo(&exp));

    delete result;
}

TEST_F(ParityOpsTests, TestMaximum1) {
    NDArray<float> x('c', {10, 10});
    x.assign(1.0);

    NDArray<float> y('c', {10, 10});
    y.assign(2.0);

    nd4j::ops::maximum<float> op;

    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);

    ASSERT_TRUE(y.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, TestMinimum1) {
    NDArray<float> x('c', {10, 10});
    x.assign(1.0f);

    NDArray<float> y('c', {10, 10});
    y.assign(-2.0f);


    nd4j::ops::minimum<float> op;

    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);

    ASSERT_TRUE(y.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, TestTear1) {
    NDArray<float> input('c', {10, 5});
    auto tads = NDArrayFactory<float>::allTensorsAlongDimension(&input, {1});
    for (int e = 0; e < tads->size(); e++) {
        ASSERT_EQ(5, tads->at(e)->lengthOf());
        tads->at(e)->assign((float) e + 1);
    }

    nd4j::ops::tear<float> op;

    auto result = op.execute({&input}, {}, {1});

    ASSERT_EQ(10, result->size());

    for (int e = 0; e < result->size(); e++)
        ASSERT_TRUE(tads->at(e)->equalsTo(result->at(e)));

    delete result;
    delete tads;
}

TEST_F(ParityOpsTests, TestUnstack1) {
    NDArray<float> input('c', {10, 5});
    auto tads = NDArrayFactory<float>::allTensorsAlongDimension(&input, {1});
    for (int e = 0; e < tads->size(); e++) {
        ASSERT_EQ(5, tads->at(e)->lengthOf());
        tads->at(e)->assign((float) e + 1);
    }

    nd4j::ops::unstack<float> op;

    auto result = op.execute({&input}, {}, {0});

    ASSERT_EQ(10, result->size());

    for (int e = 0; e < result->size(); e++)
        ASSERT_TRUE(tads->at(e)->equalsTo(result->at(e)));

    delete result;
    delete tads;
}


TEST_F(ParityOpsTests, ExpandDimsTest1) {
    NDArray<float> input('c', {5, 5});
    NDArrayFactory<float>::linspace(1, input);
    auto reshaped = input.reshape('c', {5, 1, 5});

    nd4j::ops::expand_dims<float> op;
    auto result = op.execute({&input}, {}, {1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(reshaped->isSameShape(z));
    ASSERT_TRUE(reshaped->equalsTo(z));

    delete result;
    delete reshaped;

}

TEST_F(ParityOpsTests, Test_Shape_1) {
    NDArray<float> x('c', {3, 4, 5, 6});
    NDArray<float> exp('c', {1, 4}, {3, 4, 5, 6});

    nd4j::ops::shape_of<float> op;
    auto result = op.execute({&x}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, Test_Equals_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {1, 0, 3, 0, 5});
    NDArray<float> exp('c', {1, 5}, {1, 0, 1, 0, 1});

    nd4j::ops::equals<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, Test_NotEquals_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {1, 0, 3, 0, 5});
    NDArray<float> exp('c', {1, 5}, {0, 1, 0, 1, 0});

    nd4j::ops::not_equals<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Less_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {5, 4, 3, 2, 1});
    NDArray<float> exp('c', {1, 5}, {1, 1, 0, 0, 0});

    nd4j::ops::less<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_LessEquals_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {5, 4, 3, 2, 1});
    NDArray<float> exp('c', {1, 5}, {1, 1, 1, 0, 0});

    nd4j::ops::less_equal<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_GreaterEquals_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {5, 4, 3, 2, 1});
    NDArray<float> exp('c', {1, 5}, {0, 0, 1, 1, 1});

    nd4j::ops::greater_equal<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Greater_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});
    NDArray<float> y('c', {1, 5}, {5, 4, 3, 2, 1});
    NDArray<float> exp('c', {1, 5}, {0, 0, 0, 1, 1});

    nd4j::ops::greater<float> op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Where_1) {
    NDArray<float> mask('c', {3, 3}, {1, 1, 1,  0, 0, 0,  1, 1, 1});
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> y('c', {3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1});
    NDArray<float> exp('c', {3, 3}, {1, 2, 3, 6, 5, 4, 7, 8, 9});

    nd4j::ops::where<float> op;
    auto result = op.execute({&mask, &x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Where_2) {
    NDArray<float> mask('c', {1, 3}, {1, 0, 0});
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> y('c', {3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1});
    NDArray<float> exp('c', {3, 3}, {1, 2, 3, 6, 5, 4, 3, 2, 1});

    nd4j::ops::where<float> op;
    auto result = op.execute({&mask, &x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(ParityOpsTests, Test_Where_3) {
    NDArray<float> mask('c', {2, 2, 3}, {0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    NDArray<float> exp('c', {5, 3}, {0, 0, 1, 0, 0, 2, 0, 1, 1, 1, 0, 0, 1, 1, 2});

    nd4j::ops::where<float> op;
    auto result = op.execute({&mask}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Select_1) {
    NDArray<float> mask('c', {1, 3}, {1, 0, 0});
    NDArray<float> x('c', {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    NDArray<float> y('c', {3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1});
    NDArray<float> exp('c', {3, 3}, {1, 2, 3, 6, 5, 4, 3, 2, 1});

    nd4j::ops::select<float> op;
    auto result = op.execute({&mask, &x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Select_2) {
    NDArray<float> mask('c', {2, 2}, {1, 0, 1, 0});
    NDArray<float> x('c', {2, 2}, {1, 2, 3, 4 });
    NDArray<float> y('c', {2, 2}, {9, 8, 7, 6});
    NDArray<float> exp('c', {2, 2}, {1, 8, 3, 6});

    nd4j::ops::select<float> op;
    auto result = op.execute({&mask, &x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Select_3) {
    NDArray<float> mask('c', {1, 1}, {0});
    NDArray<float> x('c', {1, 1}, {1});
    NDArray<float> y('c', {1, 1}, {2});
    NDArray<float> exp('c', {1, 1}, {2});

    nd4j::ops::select<float> op;
    auto result = op.execute({&mask, &x, &y}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(ParityOpsTests, Test_Reshape_TF_1) {
    NDArray<float> x('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> shape('c', {1, 3}, {1, 2, 2});

    NDArray<float> exp('c', {1, 2, 2}, {1, 2, 3, 4});
    
    nd4j::ops::reshape<float> op;

    auto result = op.execute({&x, &shape}, {}, {});
    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}