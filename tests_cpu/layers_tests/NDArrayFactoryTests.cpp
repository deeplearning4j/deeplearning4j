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

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, Test_Linspace_1) {
    double _expB[] = {1., 2., 3., 4., 5.};
    NDArray<double> exp('c',{1,5});
    exp.setBuffer(_expB);

    NDArray<double> x('c', {1, 5});
    NDArrayFactory<double>::linspace(1, x);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, Test_Linspace_2) {
    double _expB[] = {1., 3., 5., 7., 9.};
    NDArray<double> exp('c',{1,5});
    exp.setBuffer(_expB);

    NDArray<double> x('c', {1, 5});
    NDArrayFactory<double>::linspace(1, x, 2);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, Test_Linspace_3) {
    double _expB[] = {1., 4., 7., 10., 13.};
    NDArray<double> exp('c',{1,5});
    exp.setBuffer(_expB);

    NDArray<double> x('c', {1, 5});
    NDArrayFactory<double>::linspace(1, x, 3);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, Test_Linspace_4) {
    double _expB[] = {-1., -2., -3., -4., -5.};
    NDArray<double> exp('c',{1,5});
    exp.setBuffer(_expB);

    NDArray<double> x('c', {1, 5});
    NDArrayFactory<double>::linspace(-1, x, -1);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, Test_Linspace_5) {
    double _expB[] = {9., 8., 7., 6., 5.};
    NDArray<double> exp('c',{1,5});
    exp.setBuffer(_expB);

    NDArray<double> x('c', {1, 5});
    NDArrayFactory<double>::linspace(9, x, -1);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, mmulHelper_test_1) {
    
    NDArray<float> x('c', {3,3}, {10,11,12,13,14,15,16,17,18});
    NDArray<float> y('c', {3,3}, {1,2,3,4,5,6,7,8,9});
    NDArray<float> expected('c', {3,3}, {138.,171.,204. ,174.,216.,258. ,210.,261.,312.});    
                                                 
    NDArray<float>* result = NDArrayFactory<float>::mmulHelper(&x, &y, nullptr, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));

    delete result;

}


////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, mmulHelper_test_2) {
    
    NDArray<float> x('c', {3,3}, {10,11,12,13,14,15,16,17,18});
    NDArray<float> y('c', {3,3}, {1,2,3,4,5,6,7,8,9});
    NDArray<float> expected('c', {3,3}, {138.,171.,204. ,174.,216.,258. ,210.,261.,312.});    
    NDArray<float> result('c', {3,3});
                                                 
    NDArrayFactory<float>::mmulHelper(&x, &y, &result, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, mmulHelper_test_3) {
    
    NDArray<float> x('c', {3,4});  NDArrayFactory<float>::linspace(1, x);
    NDArray<float> y('c', {4,5});  NDArrayFactory<float>::linspace(1, y);
    NDArray<float> expected('c', {3,5}, {110.,120.,130.,140.,150.,246.,272.,298.,324.,350.,382.,424.,466.,508.,550.});    
                                                     
    NDArray<float>* result = NDArrayFactory<float>::mmulHelper(&x, &y, nullptr, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));    

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, mmulHelper_test_4) {
    
    NDArray<float> x('c', {3,4});  NDArrayFactory<float>::linspace(1, x);
    NDArray<float> y('c', {4,5});  NDArrayFactory<float>::linspace(1, y);
    NDArray<float> expected('c', {3,5}, {110.,120.,130.,140.,150.,246.,272.,298.,324.,350.,382.,424.,466.,508.,550.});    
    NDArray<float> result('c', {3,5});
                                                     
    NDArrayFactory<float>::mmulHelper(&x, &y, &result, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));    
}


////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, mmulHelper_test_5) {
    
    NDArray<float> x('c', {4,3});  NDArrayFactory<float>::linspace(1, x);
    NDArray<float> y('c', {3,5});  NDArrayFactory<float>::linspace(1, y);
    NDArray<float> expected('c', {4,5}, {46., 52., 58., 64., 70.,100.,115.,130.,145.,160.,154.,178.,202.,226.,250.,208.,241.,274.,307.,340.});    
                                                     
    NDArray<float>* result = NDArrayFactory<float>::mmulHelper(&x, &y, nullptr, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(result));
    ASSERT_TRUE(expected.equalsTo(result));    

    delete result;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, mmulHelper_test_6) {
    
    NDArray<float> x('c', {4,3});  NDArrayFactory<float>::linspace(1, x);
    NDArray<float> y('c', {3,5});  NDArrayFactory<float>::linspace(1, y);
    NDArray<float> expected('c', {4,5}, {46., 52., 58., 64., 70.,100.,115.,130.,145.,160.,154.,178.,202.,226.,250.,208.,241.,274.,307.,340.});    
    NDArray<float> result('c', {4,5});
                                                     
    NDArrayFactory<float>::mmulHelper(&x, &y, &result, 1., 0.);

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, Test_Concat_1) {
    NDArray<float> x('c', {2, 2}, {1, 2, 3, 4});
    NDArray<float> y('c', {2, 2}, {-1, -2, -3, -4});
    NDArray<float> exp('c', {2, 4}, {1, 2, -1, -2, 3, 4, -3, -4});

    auto z = NDArrayFactory<float>::concat({&x, &y}, -1);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
}