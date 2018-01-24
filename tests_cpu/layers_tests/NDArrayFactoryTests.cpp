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
TEST_F(NDArrayFactoryTests, mmulHelper_test_7) {

    NDArray<float> x('c', {4, 1}, {1, 2, 3, 4});
    NDArray<float> y('c', {1, 4}, {1, 2, 3, 4});
    NDArray<float> exp('c', {4, 4}, {1,2, 3, 4,2,4, 6, 8,3,6, 9,12,4,8,12,16});
    NDArray<float> result('c', {4,4});
                                                     
    NDArrayFactory<float>::mmulHelper(&x, &y, &result, 1., 0.);

    ASSERT_TRUE(exp.isSameShape(&result));
    ASSERT_TRUE(exp.equalsTo(&result));    

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

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, allTensorsAlongDimension_test1) {
    
    NDArray<float> x('c', {4}, {1, 2, 3, 4});
    NDArray<float> exp('c', {4}, {1, 2, 3, 4});    

    ResultSet<float>* set = NDArrayFactory<float>::allTensorsAlongDimension(&x, {0});    
    // set->at(0)->printShapeInfo();
    // set->at(0)->printIndexedBuffer();

    ASSERT_TRUE(set->size() == 1);
    ASSERT_TRUE(exp.isSameShape(set->at(0)));
    ASSERT_TRUE(exp.equalsTo(set->at(0)));

    delete set;    
}

////////////////////////////////////////////////////////////////////
// TEST_F(NDArrayFactoryTests, mmulHelper_test_8) {

//     NDArray<double> x('c', {2, 2}, {-0.063666, -0.997971, -0.997971,  0.063666});
//     NDArray<double> y('c', {2, 2}, {1, -1.477147, 1, -4.044343,});
//     // NDArray<double> y('c', {2, 1}, {-1.477147, -4.044343});
//     NDArray<double>* pY = y.subarray({{0, 2}, {1, 2}});    

//     NDArray<double> result('c', {2, 1}, {0, 0});

//     NDArray<double> exp('c', {2, 1}, {4.130181, 1.216663});
                                      
//     NDArrayFactory<double>::mmulHelper(&x, pY, &result, 1., 0.);
//     result.printBuffer();

//     ASSERT_TRUE(exp.equalsTo(&result));        
//     delete pY;
// }


////////////////////////////////////////////////////////////////////
// TEST_F(NDArrayFactoryTests, mmulHelper_test_9) {

//     NDArray<double> x('c', {4, 4}, {1.524000, 1.756820, 0.233741, 0.289458, 
//                                     0.496646, 1.565497, 0.114189, 3.896555, 
//                                     0.114611, -0.451039, 1.484030, 0.213225, 
//                                     0.229221, -0.272237, 4.160431, 3.902098});

//     NDArray<double> exp('c', {4, 4}, {1.524000, 1.756820, 0.233741, 0.289458, 
//                                       0.496646, 1.565497, 0.114189, 3.896555, 
//                                       0.114611, -0.451039, 1.484030, -0.242331, 
//                                       0.229221, -0.272237, 4.160431, -0.198199});           // this number 0.229221 !!!!!!!

//     NDArray<double> y('c',{2,2}, {-0.063666, -0.997971, -0.997971, 0.063666});
//     // NDArray<double> temp('c', {2,1});
                                                     
//     NDArray<double>* bottomColumn =  x.subarray({{2, 4}, {3, 4}});
//     // NDArrayFactory<double>::mmulHelper(&y, bottomColumn, &temp, 1., 0.);
//     NDArrayFactory<double>::mmulHelper(&y, bottomColumn, bottomColumn, 1., 0.);
//     // bottomColumn->assign(&temp);
//     x.printBuffer();


//     delete bottomColumn;    
//     ASSERT_TRUE(exp.equalsTo(&x));    
    
// }

