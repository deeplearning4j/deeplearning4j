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