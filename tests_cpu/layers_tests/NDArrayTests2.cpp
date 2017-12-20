//
// Created by raver119 on 21.11.17.
//

#include "testlayers.h"
#include <memory>
#include <NDArray.h>
#include <NDArrayFactory.h>

using namespace nd4j;

//////////////////////////////////////////////////////////////////////
class NDArrayTest2 : public testing::Test {
public:

};


TEST_F(NDArrayTest2, Test_ByteVector_1) {
    NDArray<float> x('c', {10, 10});
    NDArrayFactory<float>::linspace(1, x);

    auto vec = x.asByteVector();

    auto restored = new NDArray<float>((float *)vec.data(), x.shapeInfo());
    restored->triggerAllocationFlag(false, false);

    ASSERT_TRUE(x.equalsTo(restored));

    delete restored;
}

TEST_F(NDArrayTest2, Test_ByteVector_2) {
    NDArray<float16> x('c', {10, 10});
    NDArrayFactory<float16>::linspace(1, x);

    auto vec = x.asByteVector();

    auto restored = new NDArray<float16>((float16 *)vec.data(), x.shapeInfo());
    restored->triggerAllocationFlag(false, false);

    ASSERT_TRUE(x.equalsTo(restored));

    delete restored;
}

TEST_F(NDArrayTest2, Test_ByteVector_3) {
    NDArray<double> x('c', {10, 10});
    NDArrayFactory<double>::linspace(1, x);

    auto vec = x.asByteVector();

    auto restored = new NDArray<double>((double *)vec.data(), x.shapeInfo());
    restored->triggerAllocationFlag(false, false);

    ASSERT_TRUE(x.equalsTo(restored));

    delete restored;
}

TEST_F(NDArrayTest2, Test_IndexReduce_1) {
    NDArray<float> x('c', {1, 5}, {1, 2, 3, 4, 5});

    float extras[] = {3.0, 0.0, 10};
    int idx = x.template indexReduceNumber<simdOps::FirstIndex<float>>(extras);

    ASSERT_EQ(2, idx);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_1) {

    NDArray<double> x('c', {1, 5});
    NDArray<double> xExp('c', {1, 5}, {1, 0, 0, 0, 0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_2) {

    NDArray<double> x('f', {1, 5});
    NDArray<double> xExp('f', {1, 5}, {1, 0, 0, 0, 0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_3) {

    NDArray<double> x('f', {1, 1});
    NDArray<double> xExp('f', {1, 1}, {1});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_4) {

    NDArray<double> x('f', {2, 1});
    NDArray<double> xExp('f', {2, 1}, {1,0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_5) {

    NDArray<double> x('f', {2, 2});
    NDArray<double> xExp('f', {2, 2}, {1,0,0,1});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_6) {

    NDArray<double> x('c', {3, 2});
    NDArray<double> xExp('c', {3, 2}, {1,0,0,1,0,0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_7) {

    NDArray<double> x('c', {3, 4});
    NDArray<double> xExp('c', {3, 4}, {1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_8) {

    NDArray<double> x('c', {3, 3, 3});
    NDArray<double> xExp('c', {3, 3, 3}, {1.,0.,0. ,0.,0.,0., 0.,0.,0.,   0.,0.,0. ,0.,1.,0., 0.,0.,0.,  0.,0.,0. ,0.,0.,0., 0.,0.,1.});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_AllReduce3_1) {
    NDArray<float> x('c', {2, 3}, {1, 2, 3, 1, 2, 3});
    NDArray<float> y('c', {2, 3}, {2, 3, 4, 2, 3, 4});
    NDArray<float> exp('c', {2, 2}, {1.73205, 1.73205, 1.73205, 1.73205});

    auto z = x.template applyAllReduce3<simdOps::EuclideanDistance<float>>(&y, {1}, nullptr);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_AllReduce3_2) {
    NDArray<float> x('c', {2, 3}, {1, 2, 3, 2, 3, 4 });
    NDArray<float> y('c', {2, 3}, {1, 2, 3, 2, 3, 4});
    NDArray<float> exp('c', {2, 2}, {0., 1.73205, 1.73205, 0.});

    auto z = x.template applyAllReduce3<simdOps::EuclideanDistance<float>>(&y, {1}, nullptr);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
}