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
TEST_F(NDArrayFactoryTests, tensordot_test_1) {

    NDArray<float> a('c', {2, 3, 4});
    NDArray<float> b('c', {2, 5, 3});
                                      
    NDArray<float>* c =  NDArrayFactory<float>::tensorDot(&a, &b, {1}, {2});

    ASSERT_TRUE(c->isSameShape({2,4,2,5}));
    delete c;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, tensordot_test_2) {

    NDArray<float> a('c', {7, 3, 4, 6});
    NDArray<float> b('c', {2, 5, 3, 8, 4});
                                      
    NDArray<float>* c =  NDArrayFactory<float>::tensorDot(&a, &b, {2,1}, {4,2});    

    ASSERT_TRUE(c->isSameShape({7,6,2,5,8}));
    delete c;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, tensordot_test_3) {

    NDArray<float> a('c', {7, 3, 4, 6});
    NDArray<float> b('c', {2, 5, 3, 8, 4});
    NDArray<float> c('f', {7,6,2,8,5});
                                      
    NDArrayFactory<float>::tensorDot(&a, &b, &c, {2,1}, {4,2}, {0,1,2,4,3});

    ASSERT_TRUE(c.isSameShape({7,6,2,8,5}));    
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, tensordot_test_4) {

    NDArray<float> a('c', {7, 3, 4, 3});
    NDArray<float> b('c', {2, 5, 3, 2, 4});
    NDArray<float> c('f', {7,3,2,2,5});
    NDArray<float> expected('c', {7,3,2,2,5}, {  754.5,  2014.5,  3274.5,  4534.5 ,  5794.5,  964.5,  2224.5,  3484.5,  4744.5,  6004.5, 7054.5,  8314.5,  9574.5, 10834.5, 12094.5, 7264.5,  8524.5,  9784.5, 11044.5, 12304.5,  786. ,  2118. ,  3450. ,  4782. ,  6114. , 1008. ,  2340. ,  3672. ,  5004. ,  6336. ,
                                                 7446. ,  8778. , 10110. , 11442. , 12774. , 7668. ,  9000. , 10332. , 11664. , 12996. ,  817.5,  2221.5,  3625.5,  5029.5,  6433.5, 1051.5,  2455.5,  3859.5,  5263.5,  6667.5, 7837.5,  9241.5, 10645.5, 12049.5, 13453.5, 8071.5,  9475.5, 10879.5, 12283.5, 13687.5,
                                                 1888.5,  5740.5,  9592.5, 13444.5, 17296.5, 2530.5,  6382.5, 10234.5, 14086.5, 17938.5,21148.5, 25000.5, 28852.5, 32704.5, 36556.5,21790.5, 25642.5, 29494.5, 33346.5, 37198.5, 1920. ,  5844. ,  9768. , 13692. , 17616. , 2574. ,  6498. , 10422. , 14346. , 18270. ,
                                                21540. , 25464. , 29388. , 33312. , 37236. ,22194. , 26118. , 30042. , 33966. , 37890. , 1951.5,  5947.5,  9943.5, 13939.5, 17935.5, 2617.5,  6613.5, 10609.5, 14605.5, 18601.5,21931.5, 25927.5, 29923.5, 33919.5, 37915.5,22597.5, 26593.5, 30589.5, 34585.5, 38581.5,
                                                 3022.5,  9466.5, 15910.5, 22354.5, 28798.5, 4096.5, 10540.5, 16984.5, 23428.5, 29872.5,35242.5, 41686.5, 48130.5, 54574.5, 61018.5,36316.5, 42760.5, 49204.5, 55648.5, 62092.5, 3054. ,  9570. , 16086. , 22602. , 29118. , 4140. , 10656. , 17172. , 23688. , 30204. ,
                                                35634. , 42150. , 48666. , 55182. , 61698. ,36720. , 43236. , 49752. , 56268. , 62784. , 3085.5,  9673.5, 16261.5, 22849.5, 29437.5, 4183.5, 10771.5, 17359.5, 23947.5, 30535.5,36025.5, 42613.5, 49201.5, 55789.5, 62377.5,37123.5, 43711.5, 50299.5, 56887.5, 63475.5,
                                                 4156.5, 13192.5, 22228.5, 31264.5, 40300.5, 5662.5, 14698.5, 23734.5, 32770.5, 41806.5,49336.5, 58372.5, 67408.5, 76444.5, 85480.5,50842.5, 59878.5, 68914.5, 77950.5, 86986.5, 4188. , 13296. , 22404. , 31512. , 40620. , 5706. , 14814. , 23922. , 33030. , 42138. ,
                                                49728. , 58836. , 67944. , 77052. , 86160. ,51246. , 60354. , 69462. , 78570. , 87678. , 4219.5, 13399.5, 22579.5, 31759.5, 40939.5, 5749.5, 14929.5, 24109.5, 33289.5, 42469.5,50119.5, 59299.5, 68479.5, 77659.5, 86839.5,51649.5, 60829.5, 70009.5, 79189.5, 88369.5,
                                                 5290.5, 16918.5, 28546.5, 40174.5, 51802.5, 7228.5, 18856.5, 30484.5, 42112.5, 53740.5,63430.5, 75058.5, 86686.5, 98314.5,109942.5,65368.5, 76996.5, 88624.5,100252.5,111880.5, 5322. , 17022. , 28722. , 40422. , 52122. , 7272. , 18972. , 30672. , 42372. , 54072. ,
                                                63822. , 75522. , 87222. , 98922. ,110622. ,65772. , 77472. , 89172. ,100872. ,112572. , 5353.5, 17125.5, 28897.5, 40669.5, 52441.5, 7315.5, 19087.5, 30859.5, 42631.5, 54403.5,64213.5, 75985.5, 87757.5, 99529.5,111301.5,66175.5, 77947.5, 89719.5,101491.5,113263.5,
                                                 6424.5, 20644.5, 34864.5, 49084.5, 63304.5, 8794.5, 23014.5, 37234.5, 51454.5, 65674.5,77524.5, 91744.5,105964.5,120184.5,134404.5,79894.5, 94114.5,108334.5,122554.5,136774.5, 6456. , 20748. , 35040. , 49332. , 63624. , 8838. , 23130. , 37422. , 51714. , 66006. ,
                                                77916. , 92208. ,106500. ,120792. ,135084. ,80298. , 94590. ,108882. ,123174. ,137466. , 6487.5, 20851.5, 35215.5, 49579.5, 63943.5, 8881.5, 23245.5, 37609.5, 51973.5, 66337.5,78307.5, 92671.5,107035.5,121399.5,135763.5,80701.5, 95065.5,109429.5,123793.5,138157.5,
                                                 7558.5, 24370.5, 41182.5, 57994.5, 74806.5,10360.5, 27172.5, 43984.5, 60796.5, 77608.5,91618.5,108430.5,125242.5,142054.5,158866.5,94420.5,111232.5,128044.5,144856.5,161668.5, 7590. , 24474. , 41358. , 58242. , 75126. ,10404. , 27288. , 44172. , 61056. , 77940. ,
                                                92010. ,108894. ,125778. ,142662. ,159546. ,94824. ,111708. ,128592. ,145476. ,162360. , 7621.5, 24577.5, 41533.5, 58489.5, 75445.5,10447.5, 27403.5, 44359.5, 61315.5, 78271.5,92401.5,109357.5,126313.5,143269.5,160225.5,95227.5,112183.5,129139.5,146095.5,163051.5});
    
    NDArrayFactory<float>::linspace(0.5, a, 0.5);
    NDArrayFactory<float>::linspace(0.5, b, 0.5);

    NDArrayFactory<float>::tensorDot(&a, &b, &c, {2,1}, {4,2}, {0,1,2,4,3});
    
    ASSERT_TRUE(c.isSameShape(expected));
    ASSERT_TRUE(c.equalsTo(expected));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, tensordot_test_5) {

    NDArray<float> a('c', {2, 3});
    NDArray<float> b('c', {3, 4});
    NDArray<float> c('f', {2, 4});
    NDArray<float> expected('c', {2, 4}, {9.5,11.,12.5 ,14.,20.75 ,24.5,28.25,32.});
    
    NDArrayFactory<float>::linspace(0.5, a, 0.5);
    NDArrayFactory<float>::linspace(0.5, b, 0.5);

    NDArrayFactory<float>::tensorDot(&a, &b, &c, {1}, {0});
    c.printIndexedBuffer();

    ASSERT_TRUE(c.isSameShape(expected));
    ASSERT_TRUE(c.equalsTo(expected));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayFactoryTests, tensordot_test_6) {

    int bS=2, iH=3,iW=2,  iC=2,mC=2,  kH=2,kW=2;
    int       oC=iC*mC;
    int       oH=3,oW=2;        

    NDArray<float> a('c', {bS, iC, kH, kW, oH, oW});
    NDArray<float> b('c', {kH, kW, iC, mC});
    NDArray<float> c('c', {bS, oH, oW, iC*mC});
    NDArray<float> expected('c', {bS, oH, oW, iC*mC}, {100.,110.,336.,370.,107.,118.,345.,380.,114.,126.,354.,390.,121.,134.,363.,400.,128.,142.,372.,410.,135.,150.,381.,420.,
                                                       436.,494.,768.,850.,443.,502.,777.,860.,450.,510.,786.,870.,457.,518.,795.,880.,464.,526.,804.,890.,471.,534.,813.,900.});

    NDArrayFactory<float>::linspace(0.5, a, 0.5);
    NDArrayFactory<float>::linspace(0.5, b, 0.5);

    NDArray<float>* cR = c.reshape(a.ordering(), {bS, oH, oW, iC, mC});
    
    // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]
    NDArrayFactory<float>::tensorDot(&a, &b, cR, {{1,0,4,5,2,3}, {iC,bS*oH*oW,kW*kH}},  {{2,0,1,3},{iC,kH*kW,mC}},  {{3,0,1,2,4},{iC, bS*oH*oW, mC}});
    delete cR;
    
    ASSERT_TRUE(c.isSameShape(expected));
    ASSERT_TRUE(c.equalsTo(expected));
}


TEST_F(NDArrayFactoryTests, mmmulHelperAgain) {
    NDArray<float> x('c', {128, 156});
    NDArray<float> y('c', {156, 256});
    NDArray<float> z('c', {128, 256});
    NDArray<float> e('c', {128, 256});

    x.assign(1.0f);
    y.assign(1.0f);
    e.assign(156.0f);

    NDArrayFactory<float>::mmulHelper(&x, &y, &z);

    ASSERT_TRUE(e.isSameShape(z));
    ASSERT_TRUE(e.equalsTo(z));
}



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



