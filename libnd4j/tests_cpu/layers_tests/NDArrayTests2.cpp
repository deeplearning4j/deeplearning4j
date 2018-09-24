/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 21.11.17.
//

#include "testlayers.h"
#include <memory>
#include <NDArray.h>

using namespace nd4j;

//////////////////////////////////////////////////////////////////////
class NDArrayTest2 : public testing::Test {
public:

};


TEST_F(NDArrayTest2, Test_ByteVector_1) {
    auto x = NDArrayFactory::_create<float>('c', {10, 10});
    x.linspace(1);

    auto vec = x.asByteVector();

    auto restored = new NDArray((float *)vec.data(), x.shapeInfo());
    restored->triggerAllocationFlag(false, false);

    ASSERT_TRUE(x.equalsTo(restored));

    delete restored;
}

TEST_F(NDArrayTest2, Test_ByteVector_2) {
    auto x = NDArrayFactory::_create<float16>('c', {10, 10});
    x.linspace(1);

    auto vec = x.asByteVector();

    auto restored = new NDArray((float16 *)vec.data(), x.shapeInfo());
    restored->triggerAllocationFlag(false, false);

    ASSERT_TRUE(x.equalsTo(restored));

    delete restored;
}

TEST_F(NDArrayTest2, Test_ByteVector_3) {
    auto x = NDArrayFactory::_create<double>('c', {10, 10});
    x.linspace(1);

    auto vec = x.asByteVector();

    auto restored = new NDArray((double *)vec.data(), x.shapeInfo());
    restored->triggerAllocationFlag(false, false);

    ASSERT_TRUE(x.equalsTo(restored));

    delete restored;
}

TEST_F(NDArrayTest2, Test_Reshape_Scalar_1) {
    auto x = NDArrayFactory::_create<double>('c', {1, 1}, {1.0});
    auto e = NDArrayFactory::_create<double>(1.0);

    x.reshapei({});

    ASSERT_EQ(e, x);
    ASSERT_EQ(e.rankOf(), x.rankOf());
}

TEST_F(NDArrayTest2, Test_Reshape_Scalar_2) {
    auto x = NDArrayFactory::_create<double>('c', {1, 1}, {1.0});
    auto e = NDArrayFactory::_create<double>('c', {1}, {1.0});

    x.reshapei({1});

    ASSERT_EQ(e, x);
    ASSERT_EQ(e.rankOf(), x.rankOf());
}

TEST_F(NDArrayTest2, Test_IndexReduce_1) {
    auto x = NDArrayFactory::_create<float>('c', {1, 5}, {1, 2, 3, 4, 5});

    float extras[] = {3.0, 0.0, 10};
    int idx = x.indexReduceNumber(indexreduce::FirstIndex, extras);

    ASSERT_EQ(2, idx);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_1) {

    auto x = NDArrayFactory::_create<double>('c', {1, 5});
    auto xExp = NDArrayFactory::_create<double>('c', {1, 5}, {1, 0, 0, 0, 0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_2) {

    auto x = NDArrayFactory::_create<double>('f', {1, 5});
    auto xExp = NDArrayFactory::_create<double>('f', {1, 5}, {1, 0, 0, 0, 0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_3) {

    auto x = NDArrayFactory::_create<double>('f', {1, 1});
    auto xExp = NDArrayFactory::_create<double>('f', {1, 1}, {1});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_4) {

    auto x = NDArrayFactory::_create<double>('f', {2, 1});
    auto xExp = NDArrayFactory::_create<double>('f', {2, 1}, {1,0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_5) {

    auto x = NDArrayFactory::_create<double>('f', {2, 2});
    auto xExp = NDArrayFactory::_create<double>('f', {2, 2}, {1,0,0,1});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_6) {

    auto x = NDArrayFactory::_create<float>('c', {3, 2});
    auto  xExp = NDArrayFactory::_create<float>('c', {3, 2}, {1,0,0,1,0,0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_7) {

    auto x = NDArrayFactory::_create<float>('c', {3, 4});
    auto xExp = NDArrayFactory::_create<float>('c', {3, 4}, {1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_8) {

    auto x = NDArrayFactory::_create<float>('c', {3, 3, 3});
    auto xExp = NDArrayFactory::_create<float>('c', {3, 3, 3}, {1.,0.,0. ,0.,0.,0., 0.,0.,0.,   0.,0.,0. ,0.,1.,0., 0.,0.,0.,  0.,0.,0. ,0.,0.,0., 0.,0.,1.});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_AllReduce3_1) {
    auto x = NDArrayFactory::_create<float>('c', {2, 3}, {1, 2, 3, 1, 2, 3});
    auto y = NDArrayFactory::_create<float>('c', {2, 3}, {2, 3, 4, 2, 3, 4});
    auto exp = NDArrayFactory::_create<float>('c', {2, 2}, {1.73205, 1.73205, 1.73205, 1.73205});

    auto z = x.applyAllReduce3(reduce3::EuclideanDistance, &y, {1}, nullptr);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_AllReduce3_2) {
    auto x = NDArrayFactory::_create<float>('c', {2, 3}, {1, 2, 3, 2, 3, 4 });
    auto y = NDArrayFactory::_create<float>('c', {2, 3}, {1, 2, 3, 2, 3, 4});
    auto exp = NDArrayFactory::_create<float>('c', {2, 2}, {0., 1.73205, 1.73205, 0.});

    auto z = x.applyAllReduce3(reduce3::EuclideanDistance, &y, {1}, nullptr);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, mmul_test1) {

    auto x = NDArrayFactory::_create<float>('c', {4, 1}, {1, 2, 3, 4});
    auto y = NDArrayFactory::_create<float>('c', {1, 4}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::_create<float>('c', {4, 4}, {1,2, 3, 4,2,4, 6, 8,3,6, 9,12,4,8,12,16});
                                                     
    auto result = mmul(x, y);

    ASSERT_TRUE(exp.isSameShape(&result));
    ASSERT_TRUE(exp.equalsTo(&result));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, mmul_test2) {

    auto x = NDArrayFactory::_create<float>('c', {4, 1}, {1, 2, 3, 4});
    auto y = NDArrayFactory::_create<float>('c', {1, 4}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::_create<float>('c', {1, 1}, {30});
                                                     
    auto result = mmul(y ,x);

    ASSERT_TRUE(exp.isSameShape(&result));
    ASSERT_TRUE(exp.equalsTo(&result));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, mmul_test3) {

    auto x = NDArrayFactory::_create<float>('c', {4, 1}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::_create<float>('c', {4, 4}, {1. ,0.2 ,0.3 ,0.4 ,0.2,0.04,0.06,0.08,0.3,0.06,0.09,0.12,0.4,0.08,0.12,0.16});
    auto w = NDArrayFactory::_create<float>( x.ordering(), {(int)x.lengthOf(), 1},  x.getWorkspace());                            // column-vector
    auto wT = NDArrayFactory::_create<float>(x.ordering(), {1, (int)x.lengthOf()}, x.getWorkspace());                            // row-vector (transposed w)

    w = x / (float)10.;         
    w.p(0, 1.);
    wT.assign(&w);

    auto result = mmul(w ,wT);

    ASSERT_TRUE(exp.isSameShape(&result));
    ASSERT_TRUE(exp.equalsTo(&result));    

}


TEST_F(NDArrayTest2, Test_Streamline_1) {
    auto x = NDArrayFactory::_create<float>('c', {3, 4, 6});
    auto y = NDArrayFactory::_create<float>('c', {3, 4, 6});
    x.linspace(1);
    y.linspace(1);

    x.permutei({1, 0, 2});
    y.permutei({1, 0, 2});

    y.streamline();

    ASSERT_TRUE(x.isSameShape(&y));
    ASSERT_TRUE(x.equalsTo(&y));

    ASSERT_FALSE(x.isSameShapeStrict(&y));
}


TEST_F(NDArrayTest2, Test_Streamline_2) {
    auto x = NDArrayFactory::_create<float>('c', {3, 4, 6});
    auto y = NDArrayFactory::_create<float>('f', {3, 4, 6});
    x.linspace(1);
    y.linspace(1);

    ASSERT_TRUE(x.isSameShape(&y));
    ASSERT_TRUE(x.equalsTo(&y));
    
    y.streamline('c');

    ASSERT_TRUE(x.isSameShape(&y));
    ASSERT_TRUE(x.equalsTo(&y));
}

TEST_F(NDArrayTest2, Test_Enforce_1) {
    auto x = NDArrayFactory::_create<float>('c', {4, 1, 1, 4});
    auto exp = NDArrayFactory::_create<float>('c', {4, 4});

    x.linspace(1);
    exp.linspace(1);

    x.enforce({4, 4}, 'c');

    ASSERT_TRUE(exp.isSameShapeStrict(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(NDArrayTest2, TestVector_1) {
    auto x = NDArrayFactory::_create<float>('c', {2, 3});
    auto row = NDArrayFactory::_create<float>('c', {3}, {1, 2, 3});
    auto exp = NDArrayFactory::_create<float>('c', {2, 3}, {1, 2, 3, 1, 2, 3});

    x.addiRowVector(&row);

    ASSERT_TRUE(exp.equalsTo(&x));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Operator_Plus_Test_5)
{    

    auto x = NDArrayFactory::_create<float>('c', {8, 8, 8});
    auto y = NDArrayFactory::_create<float>('c', {8, 1, 8});
    auto expected = NDArrayFactory::_create<float>('c', {8, 8, 8});

    x = 1.;
    y = 2.;
    expected = 3.;    

    auto result = x + y;

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Operator_Plus_Test_6) {    

    auto x  = NDArrayFactory::_create<float>('c', {3, 3, 3});
    auto y = NDArrayFactory::_create<float>('c', {3, 1, 3});
    auto expected = NDArrayFactory::_create<float>('c', {3, 3, 3}, {2., 4., 6., 5., 7., 9., 8.,10.,12., 14.,16.,18.,17.,19.,21.,20.,22.,24., 26.,28.,30.,29.,31.,33.,32.,34.,36.});
    x.linspace(1);
    y.linspace(1);

    auto result = x + y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, tileToShape_test1) {    

    auto x = NDArrayFactory::_create<float>('c', {2, 2}, {1,2,3,4});
    auto exp = NDArrayFactory::_create<float>('c', {2, 2, 2}, {1,2,3,4,1,2,3,4});
    
    x.tileToShape({2,2,2});        
    
    ASSERT_TRUE(x.isSameShape(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, tileToShape_test2) {    

    auto x = NDArrayFactory::_create<float>('c', {2, 1, 2}, {1,2,3,4});
    auto exp = NDArrayFactory::_create<float>('c', {2, 3, 2}, {1,2,1,2,1,2,3,4,3,4,3,4});
    
    x.tileToShape({2,3,2});        
    
    ASSERT_TRUE(x.isSameShape(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, tileToShape_test3) {

    auto x = NDArrayFactory::_create<float>('c', {2, 2}, {1,2,3,4});
    auto result = NDArrayFactory::_create<float>('c', {2, 2, 2});
    auto exp = NDArrayFactory::_create<float>('c', {2, 2, 2}, {1,2,3,4,1,2,3,4});

    x.tileToShape({2,2,2}, &result);
    // result.printIndexedBuffer();

    ASSERT_TRUE(result.isSameShape(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, tileToShape_test4) {

    auto x = NDArrayFactory::_create<float>('c', {2, 1, 2}, {1,2,3,4});
    auto result = NDArrayFactory::_create<float>('c', {2, 3, 2});
    auto exp = NDArrayFactory::_create<float>('c', {2, 3, 2}, {1,2,1,2,1,2,3,4,3,4,3,4});

    x.tileToShape({2,3,2}, &result);

    ASSERT_TRUE(result.isSameShape(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));
}

TEST_F(NDArrayTest2, Test_TriplewiseLambda_1) {
    auto t = NDArrayFactory::_create<float>('c', {3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto u = NDArrayFactory::_create<float>('c', {3, 3}, {2, 2, 2, 2, 2, 2, 2, 2, 2});
    auto v = NDArrayFactory::_create<float>('c', {3, 3}, {3, 3, 3, 3, 3, 3, 3, 3, 3});
    auto exp = NDArrayFactory::_create<float>('c', {3, 3}, {7, 7, 7, 7, 7, 7, 7, 7, 7});

    float extra = 1.0f;

    auto la = LAMBDA_FFF(_t, _u, _v, extra) {
        return _t + _u + _v + extra;
    };

    t.applyTriplewiseLambda<float>(&u, &v, la);

    ASSERT_TRUE(t.equalsTo(&exp));
}


TEST_F(NDArrayTest2, Test_TriplewiseLambda_2) {
    auto t = NDArrayFactory::_create<float>('c', {3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto u = NDArrayFactory::_create<float>('f', {3, 3}, {2, 2, 2, 2, 2, 2, 2, 2, 2});
    auto v = NDArrayFactory::_create<float>('c', {3, 3}, {3, 3, 3, 3, 3, 3, 3, 3, 3});
    auto exp = NDArrayFactory::_create<float>('c', {3, 3}, {7, 7, 7, 7, 7, 7, 7, 7, 7});

    float extra = 1.0f;

    auto la = LAMBDA_FFF(_t, _u, _v, extra) {
        return _t + _u + _v + extra;
    };

    t.applyTriplewiseLambda<float>(&u, &v, la);

    ASSERT_TRUE(t.equalsTo(&exp));
}

TEST_F(NDArrayTest2, Test_PermuteEquality_1) {
    auto x = NDArrayFactory::_create<float>('c', {1, 60});
    auto exp = NDArrayFactory::_create<float>('c', {3, 5, 4}, {1.0, 6.0, 11.0, 16.0, 2.0, 7.0, 12.0, 17.0, 3.0, 8.0, 13.0, 18.0, 4.0, 9.0, 14.0, 19.0, 5.0, 10.0, 15.0, 20.0, 21.0, 26.0, 31.0, 36.0, 22.0, 27.0, 32.0, 37.0, 23.0, 28.0, 33.0, 38.0, 24.0, 29.0, 34.0, 39.0, 25.0, 30.0, 35.0, 40.0, 41.0, 46.0, 51.0, 56.0, 42.0, 47.0, 52.0, 57.0, 43.0, 48.0, 53.0, 58.0, 44.0, 49.0, 54.0, 59.0, 45.0, 50.0, 55.0, 60.0});
    x.linspace(1);
    x.reshapei('c', {3, 4, 5});

    x.permutei({0, 2, 1});
    x.streamline();

//    x.printShapeInfo("{0, 2, 1} shape");
//    x.printBuffer("{0, 2, 1} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(NDArrayTest2, Test_PermuteEquality_0) {
    auto x = NDArrayFactory::_create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::_create<float>('c', {3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({0, 1, 2});
    x.streamline();

//    x.printShapeInfo("{0, 1, 2} shape");
//    x.printBuffer("{0, 1, 2} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}


TEST_F(NDArrayTest2, Test_PermuteEquality_2) {
    auto x = NDArrayFactory::_create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::_create<float>('c', {4, 3, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 21.0, 22.0, 23.0, 24.0, 25.0, 41.0, 42.0, 43.0, 44.0, 45.0, 6.0, 7.0, 8.0, 9.0, 10.0, 26.0, 27.0, 28.0, 29.0, 30.0, 46.0, 47.0, 48.0, 49.0, 50.0, 11.0, 12.0, 13.0, 14.0, 15.0, 31.0, 32.0, 33.0, 34.0, 35.0, 51.0, 52.0, 53.0, 54.0, 55.0, 16.0, 17.0, 18.0, 19.0, 20.0, 36.0, 37.0, 38.0, 39.0, 40.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({1, 0, 2});
    x.streamline();

//    x.printShapeInfo("{1, 0, 2} shape");
//    x.printBuffer("{1, 0, 2} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(NDArrayTest2, Test_PermuteEquality_3) {
    auto x = NDArrayFactory::_create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::_create<float>('c', {4, 5, 3}, {1.0, 21.0, 41.0, 2.0, 22.0, 42.0, 3.0, 23.0, 43.0, 4.0, 24.0, 44.0, 5.0, 25.0, 45.0, 6.0, 26.0, 46.0, 7.0, 27.0, 47.0, 8.0, 28.0, 48.0, 9.0, 29.0, 49.0, 10.0, 30.0, 50.0, 11.0, 31.0, 51.0, 12.0, 32.0, 52.0, 13.0, 33.0, 53.0, 14.0, 34.0, 54.0, 15.0, 35.0, 55.0, 16.0, 36.0, 56.0, 17.0, 37.0, 57.0, 18.0, 38.0, 58.0, 19.0, 39.0, 59.0, 20.0, 40.0, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({1, 2, 0});
    x.streamline();

//    x.printShapeInfo("{1, 2, 0} shape");
//    x.printBuffer("{1, 2, 0} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(NDArrayTest2, Test_PermuteEquality_4) {
    auto x = NDArrayFactory::_create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::_create<float>('c', {5, 3, 4}, {1.0, 6.0, 11.0, 16.0, 21.0, 26.0, 31.0, 36.0, 41.0, 46.0, 51.0, 56.0, 2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 3.0, 8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0, 58.0, 4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0, 39.0, 44.0, 49.0, 54.0, 59.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({2, 0, 1});
    x.streamline();

//    x.printShapeInfo("{2, 0, 1} shape");
//    x.printBuffer("{2, 0, 1} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(NDArrayTest2, Test_PermuteEquality_5) {
    auto x = NDArrayFactory::_create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::_create<float>('c', {5, 4, 3},
                       {1.0, 21.0, 41.0, 6.0, 26.0, 46.0, 11.0, 31.0, 51.0, 16.0, 36.0, 56.0, 2.0, 22.0, 42.0, 7.0,
                        27.0, 47.0, 12.0, 32.0, 52.0, 17.0, 37.0, 57.0, 3.0, 23.0, 43.0, 8.0, 28.0, 48.0, 13.0, 33.0,
                        53.0, 18.0, 38.0, 58.0, 4.0, 24.0, 44.0, 9.0, 29.0, 49.0, 14.0, 34.0, 54.0, 19.0, 39.0, 59.0,
                        5.0, 25.0, 45.0, 10.0, 30.0, 50.0, 15.0, 35.0, 55.0, 20.0, 40.0, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({2, 1, 0});
    x.streamline();

//    x.printShapeInfo("{2, 0, 1} shape");
//    x.printBuffer("{2, 0, 1} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, setValueInDiagMatrix_test1) {

    auto x   = NDArrayFactory::_create<float>('c', {4, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    auto exp = NDArrayFactory::_create<float>('c', {4, 4}, {1,0,0,0,5,6,0,0,9,10,11,0 ,13,14,15,16});

    x.setValueInDiagMatrix(0., 1, 'u');

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, setValueInDiagMatrix_test2) {

    auto x   = NDArrayFactory::_create<float>('c', {4, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    auto exp = NDArrayFactory::_create<float>('c', {4, 4}, {0,0,0,0,5,0,0,0,9,10,0 ,0 ,13,14,15,0});

    x.setValueInDiagMatrix(0., 0, 'u');

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, setValueInDiagMatrix_test3) {

    auto x   = NDArrayFactory::_create<float>('c', {4, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    auto exp = NDArrayFactory::_create<float>('c', {4, 4}, {1,2,3,4,0,6,7,8,0,0 ,11,12,0 ,0 , 0,16});

    x.setValueInDiagMatrix(0., -1, 'l');

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, setValueInDiagMatrix_test4) {

    auto x   = NDArrayFactory::_create<float>('c', {4, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    auto exp = NDArrayFactory::_create<float>('c', {4, 4}, {0,2,3,4,0,0,7,8,0,0 , 0,12, 0, 0, 0, 0});

    x.setValueInDiagMatrix(0., 0, 'l');

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Indexed_Lambda) {
    auto x = NDArrayFactory::_create<float>('c', {2, 2});
    auto exp = NDArrayFactory::_create<float>('c', {2, 2}, {0, 1, 2, 3});

    auto lambda = ILAMBDA_F(_x) {
        return (float) _idx;
    };

    x.applyIndexedLambda<float>(lambda);

    ASSERT_TRUE(exp.equalsTo(&x));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_DType_Conversion_1) {
    auto x = NDArrayFactory::_create<float>('c', {2, 3}, {1, 2, 3, 4, 5, 6});

    auto xd = x.template asT<double>();

    auto xf = xd->template asT<float>();

    ASSERT_TRUE(x.isSameShape(xf));
    ASSERT_TRUE(x.equalsTo(xf));

    delete xf;
    delete xd;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_ScalarArray_Assign_1) {
    auto x = NDArrayFactory::_create<float>('c', {2, 2});
    auto y = NDArrayFactory::_create<float>(2.0f);
    auto exp = NDArrayFactory::_create<float>('c', {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});

    x.assign(y);

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Reshape_To_Vector_1) {
    auto x = NDArrayFactory::_create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto exp = NDArrayFactory::_create<float>('c', {6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    x.reshapei({-1});

    ASSERT_TRUE(exp.isSameShape(x));
    ASSERT_TRUE(exp.equalsTo(x));
}


TEST_F(NDArrayTest2, Test_toIndexedString_1) {
    auto x = NDArrayFactory::_create<float>('c', {2, 2}, {1.5f, 2.5f, 3.f, 4.5f});

    auto str = x.asIndexedString();
    std::string exp = "[1.5, 2.5, 3, 4.5]";

    ASSERT_EQ(exp, str);
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, permute_test4) {
            
    Nd4jLong arr1ShapeInfo[] = {6, 1, 1, 4, 3, 2, 2,    48, 48, 12, 4,  2,  1, 0, 1,  99};
    Nd4jLong arr2ShapeInfo[] = {6, 1, 2, 2, 1, 4, 3,    48, 2,  1,  48, 12, 4, 0, -1, 99};

    float* arr1Buffer = new float[786432];
    float* arr2Buffer = new float[786432];

    NDArray arr1(arr1Buffer, arr1ShapeInfo, nullptr);
    NDArray arr2(arr2Buffer, arr2ShapeInfo, nullptr);

    const std::vector<int> perm = {0, 4, 5, 1, 2, 3};    
    auto arr1P = arr1.permute(perm);
    // arr1P->printShapeInfo();

    // ASSERT_TRUE(arr1.isSameShapeStrict(&arr2));
    ASSERT_TRUE(arr1P->isSameShapeStrict(&arr2));
    delete arr1P;
    delete []arr1Buffer;
    delete []arr2Buffer;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, TestStdDev3) {
    
    // autoarray('c', {10, 10});
    auto array = NDArrayFactory::_create<double>('c', {2, 2}, {0.2946, 0.2084, 0.0345, 0.7368});
    const int len = array.lengthOf();

    double sum = 0.;
    for(int i=0; i < len; ++i)
        sum += array.e<double>(i);

    const double mean = sum / len;

    double diffSquared = 0.;
    for(int i=0; i < len; ++i)
        diffSquared += (array.e<double>(i) - mean) * (array.e<double>(i) - mean);

    const double trueVariance     = math::nd4j_sqrt<double, double>(diffSquared / len);
    const double trueVarianceCorr = math::nd4j_sqrt<double, double>(diffSquared / (len - 1));

    const double variance     = array.varianceNumber(variance::SummaryStatsStandardDeviation, false).e<double>(0);
    const double varianceCorr = array.varianceNumber(variance::SummaryStatsStandardDeviation, true).e<double>(0);

    // printf("%s  expected %.10f    calculated %.10f\n","variance          :", trueVariance, variance );
    // printf("%s  expected %.10f    calculated %.10f\n","variance corrected:", trueVarianceCorr, varianceCorr);

    ASSERT_NEAR(trueVariance, variance, 1e-8);
    ASSERT_NEAR(trueVarianceCorr, varianceCorr, 1e-8);
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Linspace_1) {
    double _expB[] = {1., 2., 3., 4., 5.};
    auto exp = NDArrayFactory::_create<double>('c',{1,5});
    exp.setBuffer(_expB);

    auto x = NDArrayFactory::_create<double>('c', {1, 5});
    x.linspace(1);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Linspace_2) {
    double _expB[] = {1., 3., 5., 7., 9.};
    auto exp = NDArrayFactory::_create<double>('c',{1,5});
    exp.setBuffer(_expB);

    auto x = NDArrayFactory::_create<double>('c', {1, 5});
    x.linspace(1,2);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Linspace_3) {
    double _expB[] = {1., 4., 7., 10., 13.};
    auto exp = NDArrayFactory::_create<double>('c',{1,5});
    exp.setBuffer(_expB);

    auto x = NDArrayFactory::_create<double>('c', {1, 5});
    x.linspace(1,3);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Linspace_4) {
    double _expB[] = {-1., -2., -3., -4., -5.};
    auto exp = NDArrayFactory::_create<double>('c',{1,5});
    exp.setBuffer(_expB);

    auto x = NDArrayFactory::_create<double>('c', {1, 5});
    x.linspace(-1, -1);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Linspace_5) {
    double _expB[] = {9., 8., 7., 6., 5.};
    auto exp = NDArrayFactory::_create<double>('c',{1,5});
    exp.setBuffer(_expB);

    auto x = NDArrayFactory::_create<double>('c', {1, 5});
    x.linspace(9, -1);

    ASSERT_TRUE(x.equalsTo(&exp));
}


////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, allTensorsAlongDimension_test1) {
    
    auto x = NDArrayFactory::_create<float>('c', {4}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::_create<float>('c', {4}, {1, 2, 3, 4});

    auto set = x.allTensorsAlongDimension({0});
    // set->at(0)->printShapeInfo();
    // set->at(0)->printIndexedBuffer();

    ASSERT_TRUE(set->size() == 1);
    ASSERT_TRUE(exp.isSameShape(set->at(0)));
    ASSERT_TRUE(exp.equalsTo(set->at(0)));

    delete set;
}