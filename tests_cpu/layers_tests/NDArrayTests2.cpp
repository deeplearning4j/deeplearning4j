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

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, mmul_test1) {

    NDArray<float> x('c', {4, 1}, {1, 2, 3, 4});
    NDArray<float> y('c', {1, 4}, {1, 2, 3, 4});
    NDArray<float> exp('c', {4, 4}, {1,2, 3, 4,2,4, 6, 8,3,6, 9,12,4,8,12,16});
                                                     
    NDArray<float> result = mmul(x, y);

    ASSERT_TRUE(exp.isSameShape(&result));
    ASSERT_TRUE(exp.equalsTo(&result));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, mmul_test2) {

    NDArray<float> x('c', {4, 1}, {1, 2, 3, 4});
    NDArray<float> y('c', {1, 4}, {1, 2, 3, 4});
    NDArray<float> exp('c', {1, 1}, {30});
                                                     
    NDArray<float> result = mmul(y ,x);

    ASSERT_TRUE(exp.isSameShape(&result));
    ASSERT_TRUE(exp.equalsTo(&result));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, mmul_test3) {

    NDArray<float> x('c', {4, 1}, {1, 2, 3, 4});
    NDArray<float> exp('c', {4, 4}, {1. ,0.2 ,0.3 ,0.4 ,0.2,0.04,0.06,0.08,0.3,0.06,0.09,0.12,0.4,0.08,0.12,0.16});
    NDArray<float> w((int)x.lengthOf(), 1,  x.ordering(), x.getWorkspace());                            // column-vector
    NDArray<float> wT(1, (int)x.lengthOf(), x.ordering(), x.getWorkspace());                            // row-vector (transposed w)    

    w = x / (float)10.;         
    w(0) = 1.;
    wT.assign(&w);

    NDArray<float> result = mmul(w ,wT);

    ASSERT_TRUE(exp.isSameShape(&result));
    ASSERT_TRUE(exp.equalsTo(&result));    

}


TEST_F(NDArrayTest2, Test_Streamline_1) {
    NDArray<float> x('c', {3, 4, 6});
    NDArray<float> y('c', {3, 4, 6});
    NDArrayFactory<float>::linspace(1, x);
    NDArrayFactory<float>::linspace(1, y);

    x.permutei({1, 0, 2});
    y.permutei({1, 0, 2});

    y.streamline();

    ASSERT_TRUE(x.isSameShape(&y));
    ASSERT_TRUE(x.equalsTo(&y));

    ASSERT_FALSE(x.isSameShapeStrict(&y));
}


TEST_F(NDArrayTest2, Test_Streamline_2) {
    NDArray<float> x('c', {3, 4, 6});
    NDArray<float> y('f', {3, 4, 6});
    NDArrayFactory<float>::linspace(1, x);
    NDArrayFactory<float>::linspace(1, y);

    ASSERT_TRUE(x.isSameShape(&y));
    ASSERT_TRUE(x.equalsTo(&y));
    
    y.streamline('c');

    ASSERT_TRUE(x.isSameShape(&y));
    ASSERT_TRUE(x.equalsTo(&y));
}

TEST_F(NDArrayTest2, Test_Enforce_1) {
    NDArray<float> x('c', {4, 1, 1, 4});
    NDArray<float> exp('c', {4, 4});

    NDArrayFactory<float>::linspace(1, x);
    NDArrayFactory<float>::linspace(1, exp);

    x.enforce({4, 4}, 'c');

    ASSERT_TRUE(exp.isSameShapeStrict(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}
