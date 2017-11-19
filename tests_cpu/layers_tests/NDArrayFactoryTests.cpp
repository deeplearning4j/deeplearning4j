//
// Created by raver119 on 07/11/17.
//


#include <NDArray.h>
#include <NDArrayList.h>
#include <NDArrayFactory.h>
#include "testlayers.h"

using namespace nd4j;

class NDArrayFactoryTests : public testing::Test {
public:

};

TEST_F(NDArrayFactoryTests, Test_Linspace_1) {
    double _expB[] = {1., 2., 3., 4., 5.};
    NDArray<double> exp('c',{1,5});
    exp.setBuffer(_expB);

    NDArray<double> x('c', {1, 5});
    NDArrayFactory<double>::linspace(1, x);

    ASSERT_TRUE(x.equalsTo(&exp));
}

TEST_F(NDArrayFactoryTests, Test_Linspace_2) {
    double _expB[] = {1., 3., 5., 7., 9.};
    NDArray<double> exp('c',{1,5});
    exp.setBuffer(_expB);

    NDArray<double> x('c', {1, 5});
    NDArrayFactory<double>::linspace(1, x, 2);

    ASSERT_TRUE(x.equalsTo(&exp));
}

TEST_F(NDArrayFactoryTests, Test_Linspace_3) {
    double _expB[] = {1., 4., 7., 10., 13.};
    NDArray<double> exp('c',{1,5});
    exp.setBuffer(_expB);

    NDArray<double> x('c', {1, 5});
    NDArrayFactory<double>::linspace(1, x, 3);

    ASSERT_TRUE(x.equalsTo(&exp));
}


TEST_F(NDArrayFactoryTests, Test_Linspace_4) {
    double _expB[] = {-1., -2., -3., -4., -5.};
    NDArray<double> exp('c',{1,5});
    exp.setBuffer(_expB);

    NDArray<double> x('c', {1, 5});
    NDArrayFactory<double>::linspace(-1, x, -1);

    ASSERT_TRUE(x.equalsTo(&exp));
}


TEST_F(NDArrayFactoryTests, Test_Linspace_5) {
    double _expB[] = {9., 8., 7., 6., 5.};
    NDArray<double> exp('c',{1,5});
    exp.setBuffer(_expB);

    NDArray<double> x('c', {1, 5});
    NDArrayFactory<double>::linspace(9, x, -1);

    ASSERT_TRUE(x.equalsTo(&exp));
}

TEST_F(NDArrayFactoryTests, Test_Concat_1) {
    NDArray<float> x('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> y('c', {2, 2}, {-1, -2, -3, -4});
    NDArray<float> exp('c', {2, 4}, {1, 2, -1, -2, 3, 4, -3, -4});

    auto z = NDArrayFactory<float>::concat({&x, &y}, -1);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
}