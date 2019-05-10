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
#include <DebugHelper.h>
#include <ops/declarable/headers/parity_ops.h>

using namespace nd4j;

//////////////////////////////////////////////////////////////////////
class NDArrayTest2 : public testing::Test {
public:

};


TEST_F(NDArrayTest2, Test_ByteVector_1) {
    auto x = NDArrayFactory::create<float>('c', {10, 10});
    x.linspace(1);

    auto vec = x.asByteVector();

    auto restored = new NDArray((float *)vec.data(), x.shapeInfo());
    restored->triggerAllocationFlag(false, false);

    ASSERT_TRUE(x.equalsTo(restored));

    delete restored;
}

TEST_F(NDArrayTest2, Test_ByteVector_2) {
    auto x = NDArrayFactory::create<float16>('c', {10, 10});
    x.linspace(1);

    auto vec = x.asByteVector();

    auto restored = new NDArray((float16 *)vec.data(), x.shapeInfo());
    restored->triggerAllocationFlag(false, false);

    ASSERT_TRUE(x.equalsTo(restored));

    delete restored;
}

TEST_F(NDArrayTest2, Test_ByteVector_3) {
    auto x = NDArrayFactory::create<double>('c', {10, 10});
    x.linspace(1);

    auto vec = x.asByteVector();

    auto restored = new NDArray((double *)vec.data(), x.shapeInfo());
    restored->triggerAllocationFlag(false, false);

    ASSERT_TRUE(x.equalsTo(restored));

    delete restored;
}

TEST_F(NDArrayTest2, Test_Reshape_Scalar_1) {
    auto x = NDArrayFactory::create<double>('c', {1, 1}, {1.0});
    auto e = NDArrayFactory::create<double>(1.0);

    x.reshapei({});

    ASSERT_EQ(e, x);
    ASSERT_EQ(e.rankOf(), x.rankOf());
}

TEST_F(NDArrayTest2, Test_Reshape_Scalar_2) {
    auto x = NDArrayFactory::create<double>('c', {1, 1}, {1.0});
    auto e = NDArrayFactory::create<double>('c', {1}, {1.0});

    x.reshapei({1});

    ASSERT_EQ(e, x);
    ASSERT_EQ(e.rankOf(), x.rankOf());
}

TEST_F(NDArrayTest2, Test_IndexReduce_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 5}, {1, 2, 3, 4, 5});

    float extras[] = {3.0, 0.0, 10};
    int idx = x.indexReduceNumber(indexreduce::FirstIndex, extras).e<int>(0);

    ASSERT_EQ(2, idx);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_1) {

    auto x = NDArrayFactory::create<double>('c', {1, 5});
    auto xExp = NDArrayFactory::create<double>('c', {1, 5}, {1, 0, 0, 0, 0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_2) {

    auto x = NDArrayFactory::create<double>('f', {1, 5});
    auto xExp = NDArrayFactory::create<double>('f', {1, 5}, {1, 0, 0, 0, 0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_3) {

    auto x = NDArrayFactory::create<double>('f', {1, 1});
    auto xExp = NDArrayFactory::create<double>('f', {1, 1}, {1});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_4) {

    auto x = NDArrayFactory::create<double>('f', {2, 1});
    auto xExp = NDArrayFactory::create<double>('f', {2, 1}, {1,0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_5) {

    auto x = NDArrayFactory::create<double>('f', {2, 2});
    auto xExp = NDArrayFactory::create<double>('f', {2, 2}, {1,0,0,1});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_6) {

    auto x = NDArrayFactory::create<float>('c', {3, 2});
    auto  xExp = NDArrayFactory::create<float>('c', {3, 2}, {1,0,0,1,0,0});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_7) {

    auto x = NDArrayFactory::create<float>('c', {3, 4});
    auto xExp = NDArrayFactory::create<float>('c', {3, 4}, {1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, SetIdentity_test_8) {

    auto x = NDArrayFactory::create<float>('c', {3, 3, 3});
    auto xExp = NDArrayFactory::create<float>('c', {3, 3, 3}, {1.,0.,0. ,0.,0.,0., 0.,0.,0.,   0.,0.,0. ,0.,1.,0., 0.,0.,0.,  0.,0.,0. ,0.,0.,0., 0.,0.,1.});
    
    x.setIdentity();

    ASSERT_TRUE(x.equalsTo(&xExp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_AllReduce3_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 3, 1, 2, 3});
    auto y = NDArrayFactory::create<float>('c', {2, 3}, {2, 3, 4, 2, 3, 4});
    auto exp = NDArrayFactory::create<float>('c', {2, 2}, {1.73205, 1.73205, 1.73205, 1.73205});

    auto z = x.applyAllReduce3(reduce3::EuclideanDistance, &y, {1}, nullptr);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_AllReduce3_2) {
    auto x = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 3, 2, 3, 4 });
    auto y = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 3, 2, 3, 4});
    auto exp = NDArrayFactory::create<float>('c', {2, 2}, {0., 1.73205, 1.73205, 0.});

    auto z = x.applyAllReduce3(reduce3::EuclideanDistance, &y, {1}, nullptr);

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, mmul_test1) {

    auto x = NDArrayFactory::create<float>('c', {4, 1}, {1, 2, 3, 4});
    auto y = NDArrayFactory::create<float>('c', {1, 4}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<float>('c', {4, 4}, {1,2, 3, 4,2,4, 6, 8,3,6, 9,12,4,8,12,16});
                                                     
    auto result = mmul(x, y);

    ASSERT_TRUE(exp.isSameShape(&result));
    ASSERT_TRUE(exp.equalsTo(&result));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, mmul_test2) {

    auto x = NDArrayFactory::create<float>('c', {4, 1}, {1, 2, 3, 4});
    auto y = NDArrayFactory::create<float>('c', {1, 4}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<float>('c', {1, 1}, {30});
                                                     
    auto result = mmul(y ,x);

    ASSERT_TRUE(exp.isSameShape(&result));
    ASSERT_TRUE(exp.equalsTo(&result));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, mmul_test3) {

    auto x = NDArrayFactory::create<float>('c', {4, 1}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<float>('c', {4, 4}, {1. ,0.2 ,0.3 ,0.4 ,0.2,0.04,0.06,0.08,0.3,0.06,0.09,0.12,0.4,0.08,0.12,0.16});
    auto w = NDArrayFactory::create<float>( x.ordering(), {(int)x.lengthOf(), 1},  x.getWorkspace());                            // column-vector
    auto wT = NDArrayFactory::create<float>(x.ordering(), {1, (int)x.lengthOf()}, x.getWorkspace());                            // row-vector (transposed w)

    w = x / (float)10.;         
    w.p(0, 1.);
    wT.assign(&w);

    auto result = mmul(w ,wT);

    ASSERT_TRUE(exp.isSameShape(&result));
    ASSERT_TRUE(exp.equalsTo(&result));    

}


TEST_F(NDArrayTest2, Test_Streamline_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 4, 6});
    auto y = NDArrayFactory::create<float>('c', {3, 4, 6});
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
    auto x = NDArrayFactory::create<double>('c', {3, 4, 6});
    auto y = NDArrayFactory::create<double>('f', {3, 4, 6});
    x.linspace(1);
    y.linspace(1);

    ASSERT_TRUE(x.isSameShape(&y));
    ASSERT_TRUE(x.equalsTo(&y));
    
    y.streamline('c');

    ASSERT_TRUE(x.isSameShape(&y));
    ASSERT_TRUE(x.equalsTo(&y));
}

TEST_F(NDArrayTest2, Test_Enforce_1) {
    auto x = NDArrayFactory::create<float>('c', {4, 1, 1, 4});
    auto exp = NDArrayFactory::create<float>('c', {4, 4});

    x.linspace(1);
    exp.linspace(1);

    x.enforce({4, 4}, 'c');

    ASSERT_TRUE(exp.isSameShapeStrict(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(NDArrayTest2, TestVector_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 3});
    auto row = NDArrayFactory::create<float>('c', {3}, {1, 2, 3});
    auto exp = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 3, 1, 2, 3});

    x.addiRowVector(&row);

    ASSERT_TRUE(exp.equalsTo(&x));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Operator_Plus_Test_5)
{    

    auto x = NDArrayFactory::create<float>('c', {8, 8, 8});
    auto y = NDArrayFactory::create<float>('c', {8, 1, 8});
    auto expected = NDArrayFactory::create<float>('c', {8, 8, 8});

    x = 1.;
    y = 2.;
    expected = 3.;    

    auto result = x + y;

    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Operator_Plus_Test_6) {    

    auto x  = NDArrayFactory::create<float>('c', {3, 3, 3});
    auto y = NDArrayFactory::create<float>('c', {3, 1, 3});
    auto expected = NDArrayFactory::create<float>('c', {3, 3, 3}, {2., 4., 6., 5., 7., 9., 8.,10.,12., 14.,16.,18.,17.,19.,21.,20.,22.,24., 26.,28.,30.,29.,31.,33.,32.,34.,36.});
    x.linspace(1);
    y.linspace(1);

    auto result = x + y;
    
    ASSERT_TRUE(expected.isSameShape(&result));
    ASSERT_TRUE(expected.equalsTo(&result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, tileToShape_test1) {    

    auto x = NDArrayFactory::create<float>('c', {2, 2}, {1,2,3,4});
    auto exp = NDArrayFactory::create<float>('c', {2, 2, 2}, {1,2,3,4,1,2,3,4});
    
    x.tileToShape({2,2,2});        
    
    ASSERT_TRUE(x.isSameShape(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, tileToShape_test2) {    

    auto x = NDArrayFactory::create<float>('c', {2, 1, 2}, {1,2,3,4});
    auto exp = NDArrayFactory::create<float>('c', {2, 3, 2}, {1,2,1,2,1,2,3,4,3,4,3,4});
    
    x.tileToShape({2,3,2});        
    
    ASSERT_TRUE(x.isSameShape(&exp));
    ASSERT_TRUE(x.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, tileToShape_test3) {

    auto x = NDArrayFactory::create<float>('c', {2, 2}, {1,2,3,4});
    auto result = NDArrayFactory::create<float>('c', {2, 2, 2});
    auto exp = NDArrayFactory::create<float>('c', {2, 2, 2}, {1,2,3,4,1,2,3,4});

    x.tileToShape({2,2,2}, &result);
    // result.printIndexedBuffer();

    ASSERT_TRUE(result.isSameShape(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, tileToShape_test4) {

    auto x = NDArrayFactory::create<float>('c', {2, 1, 2}, {1,2,3,4});
    auto result = NDArrayFactory::create<float>('c', {2, 3, 2});
    auto exp = NDArrayFactory::create<float>('c', {2, 3, 2}, {1,2,1,2,1,2,3,4,3,4,3,4});

    x.tileToShape({2,3,2}, &result);

    ASSERT_TRUE(result.isSameShape(&exp));
    ASSERT_TRUE(result.equalsTo(&exp));
}

TEST_F(NDArrayTest2, Test_TriplewiseLambda_1) {
    auto t = NDArrayFactory::create<float>('c', {3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto u = NDArrayFactory::create<float>('c', {3, 3}, {2, 2, 2, 2, 2, 2, 2, 2, 2});
    auto v = NDArrayFactory::create<float>('c', {3, 3}, {3, 3, 3, 3, 3, 3, 3, 3, 3});
    auto exp = NDArrayFactory::create<float>('c', {3, 3}, {7, 7, 7, 7, 7, 7, 7, 7, 7});

    float extra = 1.0f;

    auto la = LAMBDA_FFF(_t, _u, _v, extra) {
        return _t + _u + _v + extra;
    };

    t.applyTriplewiseLambda<float>(&u, &v, la);

    ASSERT_TRUE(t.equalsTo(&exp));
}


TEST_F(NDArrayTest2, Test_TriplewiseLambda_2) {
    auto t = NDArrayFactory::create<float>('c', {3, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto u = NDArrayFactory::create<float>('f', {3, 3}, {2, 2, 2, 2, 2, 2, 2, 2, 2});
    auto v = NDArrayFactory::create<float>('c', {3, 3}, {3, 3, 3, 3, 3, 3, 3, 3, 3});
    auto exp = NDArrayFactory::create<float>('c', {3, 3}, {7, 7, 7, 7, 7, 7, 7, 7, 7});

    float extra = 1.0f;

    auto la = LAMBDA_FFF(_t, _u, _v, extra) {
        return _t + _u + _v + extra;
    };

    t.applyTriplewiseLambda<float>(&u, &v, la);

    ASSERT_TRUE(t.equalsTo(&exp));
}

TEST_F(NDArrayTest2, Test_PermuteEquality_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 60});
    auto exp = NDArrayFactory::create<float>('c', {3, 5, 4}, {1.0, 6.0, 11.0, 16.0, 2.0, 7.0, 12.0, 17.0, 3.0, 8.0, 13.0, 18.0, 4.0, 9.0, 14.0, 19.0, 5.0, 10.0, 15.0, 20.0, 21.0, 26.0, 31.0, 36.0, 22.0, 27.0, 32.0, 37.0, 23.0, 28.0, 33.0, 38.0, 24.0, 29.0, 34.0, 39.0, 25.0, 30.0, 35.0, 40.0, 41.0, 46.0, 51.0, 56.0, 42.0, 47.0, 52.0, 57.0, 43.0, 48.0, 53.0, 58.0, 44.0, 49.0, 54.0, 59.0, 45.0, 50.0, 55.0, 60.0});
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
    auto x = NDArrayFactory::create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<float>('c', {3, 4, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({0, 1, 2});
    x.streamline();

//    x.printShapeInfo("{0, 1, 2} shape");
//    x.printBuffer("{0, 1, 2} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}


TEST_F(NDArrayTest2, Test_PermuteEquality_2) {
    auto x = NDArrayFactory::create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<float>('c', {4, 3, 5}, {1.0, 2.0, 3.0, 4.0, 5.0, 21.0, 22.0, 23.0, 24.0, 25.0, 41.0, 42.0, 43.0, 44.0, 45.0, 6.0, 7.0, 8.0, 9.0, 10.0, 26.0, 27.0, 28.0, 29.0, 30.0, 46.0, 47.0, 48.0, 49.0, 50.0, 11.0, 12.0, 13.0, 14.0, 15.0, 31.0, 32.0, 33.0, 34.0, 35.0, 51.0, 52.0, 53.0, 54.0, 55.0, 16.0, 17.0, 18.0, 19.0, 20.0, 36.0, 37.0, 38.0, 39.0, 40.0, 56.0, 57.0, 58.0, 59.0, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({1, 0, 2});
    x.streamline();

//    x.printShapeInfo("{1, 0, 2} shape");
//    x.printBuffer("{1, 0, 2} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(NDArrayTest2, Test_PermuteEquality_3) {
    auto x = NDArrayFactory::create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<float>('c', {4, 5, 3}, {1.0, 21.0, 41.0, 2.0, 22.0, 42.0, 3.0, 23.0, 43.0, 4.0, 24.0, 44.0, 5.0, 25.0, 45.0, 6.0, 26.0, 46.0, 7.0, 27.0, 47.0, 8.0, 28.0, 48.0, 9.0, 29.0, 49.0, 10.0, 30.0, 50.0, 11.0, 31.0, 51.0, 12.0, 32.0, 52.0, 13.0, 33.0, 53.0, 14.0, 34.0, 54.0, 15.0, 35.0, 55.0, 16.0, 36.0, 56.0, 17.0, 37.0, 57.0, 18.0, 38.0, 58.0, 19.0, 39.0, 59.0, 20.0, 40.0, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({1, 2, 0});
    x.streamline();

//    x.printShapeInfo("{1, 2, 0} shape");
//    x.printBuffer("{1, 2, 0} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(NDArrayTest2, Test_PermuteEquality_4) {
    auto x = NDArrayFactory::create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<float>('c', {5, 3, 4}, {1.0, 6.0, 11.0, 16.0, 21.0, 26.0, 31.0, 36.0, 41.0, 46.0, 51.0, 56.0, 2.0, 7.0, 12.0, 17.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 3.0, 8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0, 58.0, 4.0, 9.0, 14.0, 19.0, 24.0, 29.0, 34.0, 39.0, 44.0, 49.0, 54.0, 59.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0});
    x.reshapei('c', {3, 4, 5});

    x.permutei({2, 0, 1});
    x.streamline();

//    x.printShapeInfo("{2, 0, 1} shape");
//    x.printBuffer("{2, 0, 1} data");

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(NDArrayTest2, Test_PermuteEquality_5) {
    auto x = NDArrayFactory::create<float>('c', {1, 60});
    x.linspace(1);
    auto exp = NDArrayFactory::create<float>('c', {5, 4, 3},
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

    auto x   = NDArrayFactory::create<float>('c', {4, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    auto exp = NDArrayFactory::create<float>('c', {4, 4}, {1,0,0,0,5,6,0,0,9,10,11,0 ,13,14,15,16});

    x.setValueInDiagMatrix(0., 1, 'u');

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, setValueInDiagMatrix_test2) {

    auto x   = NDArrayFactory::create<float>('c', {4, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    auto exp = NDArrayFactory::create<float>('c', {4, 4}, {0,0,0,0,5,0,0,0,9,10,0 ,0 ,13,14,15,0});

    x.setValueInDiagMatrix(0., 0, 'u');

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, setValueInDiagMatrix_test3) {

    auto x   = NDArrayFactory::create<float>('c', {4, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    auto exp = NDArrayFactory::create<float>('c', {4, 4}, {1,2,3,4,0,6,7,8,0,0 ,11,12,0 ,0 , 0,16});

    x.setValueInDiagMatrix(0., -1, 'l');

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));    

}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, setValueInDiagMatrix_test4) {

    auto x   = NDArrayFactory::create<float>('c', {4, 4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    auto exp = NDArrayFactory::create<float>('c', {4, 4}, {0,2,3,4,0,0,7,8,0,0 , 0,12, 0, 0, 0, 0});

    x.setValueInDiagMatrix(0., 0, 'l');

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Indexed_Lambda) {
    auto x = NDArrayFactory::create<float>('c', {2, 2});
    auto exp = NDArrayFactory::create<float>('c', {2, 2}, {0, 1, 2, 3});

    auto lambda = ILAMBDA_F(_x) {
        return (float) _idx;
    };

    x.applyIndexedLambda<float>(lambda);

    ASSERT_TRUE(exp.equalsTo(&x));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_DType_Conversion_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 3, 4, 5, 6});

    auto xd = x.template asT<double>();

    auto xf = xd->template asT<float>();

    ASSERT_TRUE(x.isSameShape(xf));
    ASSERT_TRUE(x.equalsTo(xf));

    delete xf;
    delete xd;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_ScalarArray_Assign_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2});
    auto y = NDArrayFactory::create<float>(2.0f);
    auto exp = NDArrayFactory::create<float>('c', {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});

    x.assign(y);

    ASSERT_TRUE(exp.isSameShape(&x));
    ASSERT_TRUE(exp.equalsTo(&x));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Reshape_To_Vector_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto exp = NDArrayFactory::create<float>('c', {6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    x.reshapei({-1});

    ASSERT_TRUE(exp.isSameShape(x));
    ASSERT_TRUE(exp.equalsTo(x));
}


TEST_F(NDArrayTest2, Test_toIndexedString_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {1.5f, 2.5f, 3.f, 4.5f});

    auto str = x.asIndexedString();
    std::string exp = "[1.5, 2.5, 3, 4.5]";

    ASSERT_EQ(exp, str);
}


//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, permute_test4) {
            
    Nd4jLong arr1ShapeInfo[] = {6, 1, 1, 4, 3, 2, 2,    48, 48, 12, 4,  2,  1, 8192, 1,  99};
    Nd4jLong arr2ShapeInfo[] = {6, 1, 2, 2, 1, 4, 3,    48, 2,  1,  48, 12, 4, 8192, 0, 99};


    auto arr1Buffer = new float[786432];
    auto arr2Buffer = new float[786432];

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
    auto array = NDArrayFactory::create<double>('c', {2, 2}, {0.2946, 0.2084, 0.0345, 0.7368});
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
    auto exp = NDArrayFactory::create<double>('c',{1,5});
    exp.setBuffer(_expB);

    auto x = NDArrayFactory::create<double>('c', {1, 5});
    x.linspace(1);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Linspace_2) {
    double _expB[] = {1., 3., 5., 7., 9.};
    auto exp = NDArrayFactory::create<double>('c',{1,5});
    exp.setBuffer(_expB);

    auto x = NDArrayFactory::create<double>('c', {1, 5});
    x.linspace(1, 2);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Linspace_3) {
    double _expB[] = {1., 4., 7., 10., 13.};
    auto exp = NDArrayFactory::create<double>('c',{1,5});
    exp.setBuffer(_expB);

    auto x = NDArrayFactory::create<double>('c', {1, 5});
    x.linspace(1,3);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Linspace_4) {
    double _expB[] = {-1., -2., -3., -4., -5.};
    auto exp = NDArrayFactory::create<double>('c',{1,5});
    exp.setBuffer(_expB);

    auto x = NDArrayFactory::create<double>('c', {1, 5});
    x.linspace(-1, -1);

    ASSERT_TRUE(x.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, Test_Linspace_5) {
    double _expB[] = {9., 8., 7., 6., 5.};
    auto exp = NDArrayFactory::create<double>('c',{1,5});
    exp.setBuffer(_expB);

    auto x = NDArrayFactory::create<double>('c', {1, 5});
    x.linspace(9, -1);

    ASSERT_TRUE(x.equalsTo(&exp));
}


////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, allTensorsAlongDimension_test1) {
    
    auto x = NDArrayFactory::create<float>('c', {4}, {1, 2, 3, 4});
    auto exp = NDArrayFactory::create<float>('c', {4}, {1, 2, 3, 4});

    auto set = x.allTensorsAlongDimension({0});
    // set->at(0)->printShapeInfo();
    // set->at(0)->printIndexedBuffer();

    ASSERT_TRUE(set->size() == 1);
    ASSERT_TRUE(exp.isSameShape(set->at(0)));
    ASSERT_TRUE(exp.equalsTo(set->at(0)));

    delete set;
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, scalar_get_test1) {
    
    NDArray scalar1('c', {0}, {20.}, nd4j::DataType::FLOAT32);

    NDArray arr('c', {2,2}, {0., 10., 20., 30.}, nd4j::DataType::FLOAT32);

    NDArray scalar2 = arr.e(2); 

    ASSERT_TRUE(scalar1.isSameShape(scalar2));
    ASSERT_TRUE(scalar1.equalsTo(scalar2));
    ASSERT_TRUE(scalar1.dataType() == scalar2.dataType());
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, scalar_get_test2) {
    
    NDArray scalar1('c', {0}, {20.}, nd4j::DataType::FLOAT32);

    NDArray arr('f', {2,2}, {0., 10., 20., 30.}, nd4j::DataType::FLOAT32);

    NDArray scalar2 = arr.e(1); 

    ASSERT_TRUE(scalar1.isSameShape(scalar2));
    ASSERT_TRUE(scalar1.equalsTo(scalar2));
    ASSERT_TRUE(scalar1.dataType() == scalar2.dataType());
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, scalar_set_test1) {
    
    NDArray scalar1('c', {0}, {20.}, nd4j::DataType::FLOAT32);

    NDArray arr('c', {2,2}, {0., 10., -20., 30.}, nd4j::DataType::FLOAT32);
    NDArray exp('c', {2,2}, {0., 10.,  20., 30.}, nd4j::DataType::FLOAT32);

    arr.p(2, scalar1); 

    ASSERT_TRUE(exp.equalsTo(arr));
}


////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, scalar_set_test2) {
    
    NDArray scalar1('c', {0}, {20.}, nd4j::DataType::FLOAT32);

    NDArray arr('f', {2,2}, {0., 10., -20., 30.}, nd4j::DataType::FLOAT32);
    NDArray exp('f', {2,2}, {0., 10.,  20., 30.}, nd4j::DataType::FLOAT32);

    arr.p(1, scalar1); 

    ASSERT_TRUE(exp.equalsTo(arr));
}

TEST_F(NDArrayTest2, big_dup_test) {
    // auto arr = NDArrayFactory::linspace<float>(1.0f, 10000000.0f, 100000000);
    auto arr = NDArrayFactory::linspace<float>(1.0f, 1000.0f, 10000);
    auto dup = arr->dup('c');

    ASSERT_EQ(*arr, *dup);

    delete arr;
    delete dup;
}

TEST_F(NDArrayTest2, debugInfoTest_1) {
    NDArray testArray('c', {2, 4, 4, 4}, {
            91.,  82.,  37.,  64.,     55.,  46.,  73.,  28.,    119.,  12., 112.,  13.,     14., 114.,  16., 117.,
            51.,  42.,  67.,  24.,     15.,  56.,  93.,  28.,    109.,  82.,  12., 113.,    114.,  14., 116.,  11.,
            31.,  22.,  87.,  44.,     55.,  46.,  73.,  28.,    -119.,  12., 112.,  13.,     14., 114.,  16., 117.,
            91.,  -82.,  37.,  64.,    -55.1,  0,  73.,  28.,    -119.,  12., 112.,  13.,     14., 114., 16.2, 117.,
            91.,  -82.,  37.,  64.,     55.,  46.,  73.,  28.,    -119.,  12., 112.,  13.,     14., 114.,  16., 117.,
            51.,  42.,  67.,  24.,     15.,  0.,  93.,  28.,    109.,  82.,  12., 113.,    114.,  14., 116.,  11.,
            31.,  22.,  87.,  44.,     55.,  46.,  73.,  28.,    119.,  12., 112.,  13.,     14., 114.,  16., 117.,
            91.,  82.,  37.,  64.,    -3,  0, 73.,  28.,    119.,  12., 112.,  13.,    140., 110., 160., 107.}, nd4j::DataType::DOUBLE);
    NDArray res(nd4j::DataType::DOUBLE);
    DebugInfo info = DebugHelper::debugStatistics(&testArray);
    DebugInfo exp; // = {}
    nd4j::ops::reduce_min minOp;
    nd4j::ops::reduce_mean meanOp;
    nd4j::ops::reduce_max maxOp;
    nd4j::ops::reduce_stdev stdevOp;

    minOp.execute({&testArray}, {&res}, {}, {}, {});
    exp._minValue = res.e<double>(0);
    meanOp.execute({&testArray}, {&res}, {}, {}, {});
    exp._meanValue = res.e<double>(0);
    maxOp.execute({&testArray}, {&res}, {}, {}, {});
    exp._maxValue = res.e<double>(0);
    stdevOp.execute({&testArray}, {&res}, {}, {}, {});
    exp._stdDevValue = res.e<double>(0);
    exp._zeroCount = 3;
    exp._negativeCount = 7;
    exp._positiveCount = 118;
    exp._infCount = 0;
    exp._nanCount = 0;
    printf("Output statistics %lf %lf %lf %lf\n", info._minValue, info._maxValue, info._meanValue, info._stdDevValue);
    printf("Expect statistics %lf %lf %lf %lf\n", exp._minValue, exp._maxValue, exp._meanValue, exp._stdDevValue);
    printf("%lld %lld %lld %lld %lld\n", info._zeroCount, info._negativeCount, info._positiveCount, info._infCount, info._nanCount);
    ASSERT_EQ(exp, info);
}

TEST_F(NDArrayTest2, debugInfoTest_2) {
    NDArray testArray('c', {2, 4, 4, 4}, {
            91.,  82.,  37.,  64.,     55.,  46.,  73.,  28.,    119.,  12., 112.,  13.,     14., 114.,  16., 117.,
            51.,  42.,  67.,  24.,     15.,  56.,  93.,  28.,    109.,  82.,  12., 113.,    114.,  14., 116.,  11.,
            31.,  22.,  87.,  44.,     55.,  46.,  73.,  28.,    -119.,  12., 112.,  13.,     14., 114.,  16., 117.,
            91.,  -82.,  37.,  64.,    -55.1,  0,  73.,  28.,    -119.,  12., 112.,  13.,     14., 114., 16.2, 117.,
            91.,  -82.,  37.,  64.,     55.,  46.,  73.,  28.,    -119.,  12., 112.,  13.,     14., 114.,  16., 117.,
            51.,  42.,  67.,  24.,     15.,  0.,  93.,  28.,    109.,  82.,  12., 113.,    114.,  14., 116.,  11.,
            31.,  22.,  87.,  44.,     55.,  46.,  73.,  28.,    119.,  12., 112.,  13.,     14., 114.,  16., 117.,
            91.,  82.,  37.,  64.,    -3,  0, 73.,  28.,    119.,  12., 112.,  13.,    140., 110., 160., 107.}, nd4j::DataType::DOUBLE);

    DebugInfo info;
    DebugInfo exp; // = {}
    exp._minValue = -119;
    exp._maxValue = 160.;
    exp._meanValue = 51.328906;
    exp._stdDevValue = 52.385694;
    exp._zeroCount = 3;
    exp._negativeCount = 7;
    exp._positiveCount = 118;
    exp._infCount = 0;
    exp._nanCount = 0;
    DebugHelper::retrieveDebugStatistics(&info, &testArray);
    printf("Output statistics %lf %lf %lf %lf\n", info._minValue, info._maxValue, info._meanValue, info._stdDevValue);
    printf("Expect statistics %lf %lf %lf %lf\n", exp._minValue, exp._maxValue, exp._meanValue, exp._stdDevValue);
    printf("%lld %lld %lld %lld %lld\n", info._zeroCount, info._negativeCount, info._positiveCount, info._infCount, info._nanCount);
    //printf("%lf %lf %lf %lf\n", info._minValue, info._maxValue, info._meanValue, info._stdDevValue);
    //printf("%lld %lld %lld %lld %lld\n", info._zeroCount, info._negativeCount, info._positiveCount, info._infCount, info._nanCount);
    ASSERT_EQ(exp, info);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, test_subarray_ews_1) {

    NDArray x('c', {10, 5}, nd4j::DataType::FLOAT32);
    auto subArr1 = x.subarray({NDIndex::all(), NDIndex::point(2)});

    subArr1->printShapeInfo("subArr1");

    ASSERT_EQ(5, subArr1->ews());
    delete subArr1;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, test_subarray_ews_2) {

    NDArray x('f', {10, 5}, nd4j::DataType::FLOAT32);
    auto subArr1 = x.subarray({NDIndex::all(), NDIndex::point(2)});

    subArr1->printShapeInfo("subArr1");

    ASSERT_EQ(1, subArr1->ews());
    delete subArr1;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, test_subarray_ews_3) {

    NDArray x('c', {10, 5}, nd4j::DataType::FLOAT32);
    auto subArr1 = x.subarray({NDIndex::point(2), NDIndex::all()});

    subArr1->printShapeInfo("subArr1");

    ASSERT_EQ(1, subArr1->ews());
    delete subArr1;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, test_subarray_ews_4) {

    NDArray x('f', {10, 5}, nd4j::DataType::FLOAT32);
    auto subArr1 = x.subarray({NDIndex::point(2), NDIndex::all()});

    subArr1->printShapeInfo("subArr1");

    ASSERT_EQ(10, subArr1->ews());
    delete subArr1;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, subarray_1) {    

    NDArray x('c', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, nd4j::DataType::FLOAT32);
    NDArray y('f', {2,3,4}, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}, nd4j::DataType::FLOAT32);
    
    Nd4jLong shapeExpX0[] = {1, 2, 12, 8192, 12, 99};
    float    buffExpX0[]  = {1.000000, 13.000000};    
    float    buffExpX1[]  = {2.000000, 14.000000};
    Nd4jLong shapeExpX2[] = {3, 2, 1, 1, 12, 4, 1, 8192, 12, 99};
    float    buffExpX2[]  = {1.000000, 13.000000};
    Nd4jLong shapeExpX3[] = {2, 2, 4, 12, 1, 8192, 0, 99};
    float    buffExpX3[]  = {9.000000, 10.000000, 11.000000, 12.000000, 21.000000, 22.000000, 23.000000, 24.000000};
    Nd4jLong shapeExpX4[] = {3, 2, 1, 4, 12, 4, 1, 8192, 0, 99};
    float    buffExpX4[]  = {9.000000, 10.000000, 11.000000, 12.000000, 21.000000, 22.000000, 23.000000, 24.000000};
    Nd4jLong shapeExpX5[] = {2, 2, 3, 12, 4, 8192, 0, 99};
    float    buffExpX5[]  = {4.000000, 8.000000, 12.000000, 16.000000, 20.000000, 24.000000};

    Nd4jLong shapeExpY0[] = {1, 2, 1, 8192, 1, 102};
    float    buffExpY0[]  = {1.000000, 2.000000};    
    float    buffExpY1[]  = {7.000000, 8.000000};
    Nd4jLong shapeExpY2[] = {3, 2, 1, 1, 1, 2, 6, 8192, 1, 102};
    float    buffExpY2[]  = {1.000000, 2.000000};
    Nd4jLong shapeExpY3[] = {2, 2, 4, 1, 6, 8192, 0, 102};
    float    buffExpY3[]  = {5.000000, 11.000000, 17.000000, 23.000000, 6.000000, 12.000000, 18.000000, 24.000000};
    Nd4jLong shapeExpY4[] = {3, 2, 1, 4, 1, 2, 6, 8192, 0, 102};
    float    buffExpY4[]  = {5.000000, 11.000000, 17.000000, 23.000000, 6.000000, 12.000000, 18.000000, 24.000000};
    Nd4jLong shapeExpY5[] = {2, 2, 3, 1, 2, 8192, 1, 102};
    float    buffExpY5[]  = {19.000000, 21.000000, 23.000000, 20.000000, 22.000000, 24.000000};
    

    NDArray x0 = x(0, {1,2});    
    for(int i = 0; i < shape::shapeInfoLength(x0.rankOf()); ++i)
        ASSERT_TRUE(x0.getShapeInfo()[i] == shapeExpX0[i]);
    for(int i = 0; i < x0.lengthOf(); ++i)
        ASSERT_TRUE(x0.e<float>(i) == buffExpX0[i]);

    NDArray x1 = x(1, {1,2});
    for(int i = 0; i < shape::shapeInfoLength(x1.rankOf()); ++i)
        ASSERT_TRUE(x1.getShapeInfo()[i] == shapeExpX0[i]);
    for(int i = 0; i < x1.lengthOf(); ++i)
        ASSERT_TRUE(x1.e<float>(i) == buffExpX1[i]);

    NDArray x2 = x(0, {1,2}, true);
    for(int i = 0; i < shape::shapeInfoLength(x2.rankOf()); ++i)
        ASSERT_TRUE(x2.getShapeInfo()[i] == shapeExpX2[i]);
    for(int i = 0; i < x2.lengthOf(); ++i)
        ASSERT_TRUE(x2.e<float>(i) == buffExpX2[i]);
    
    NDArray x3 = x(2, {1});
    for(int i = 0; i < shape::shapeInfoLength(x3.rankOf()); ++i)
        ASSERT_TRUE(x3.getShapeInfo()[i] == shapeExpX3[i]);
    for(int i = 0; i < x3.lengthOf(); ++i)
        ASSERT_TRUE(x3.e<float>(i) == buffExpX3[i]);

    NDArray x4 = x(2, {1}, true);
    for(int i = 0; i < shape::shapeInfoLength(x4.rankOf()); ++i)
        ASSERT_TRUE(x4.getShapeInfo()[i] == shapeExpX4[i]);
    for(int i = 0; i < x4.lengthOf(); ++i)
        ASSERT_TRUE(x4.e<float>(i) == buffExpX4[i]);

    NDArray x5 = x(3, {2});
    for(int i = 0; i < shape::shapeInfoLength(x5.rankOf()); ++i)
        ASSERT_TRUE(x5.getShapeInfo()[i] == shapeExpX5[i]);
    for(int i = 0; i < x5.lengthOf(); ++i)
        ASSERT_TRUE(x5.e<float>(i) == buffExpX5[i]);

    // ******************* //
    NDArray y0 = y(0, {1,2});    
    for(int i = 0; i < shape::shapeInfoLength(y0.rankOf()); ++i)
        ASSERT_TRUE(y0.getShapeInfo()[i] == shapeExpY0[i]);
    for(int i = 0; i < y0.lengthOf(); ++i)
        ASSERT_TRUE(y0.e<float>(i) == buffExpY0[i]);

    NDArray y1 = y(1, {1,2});
    for(int i = 0; i < shape::shapeInfoLength(y1.rankOf()); ++i)
        ASSERT_TRUE(y1.getShapeInfo()[i] == shapeExpY0[i]);
    for(int i = 0; i < y1.lengthOf(); ++i)
        ASSERT_TRUE(y1.e<float>(i) == buffExpY1[i]);

    NDArray y2 = y(0, {1,2}, true);
    for(int i = 0; i < shape::shapeInfoLength(y2.rankOf()); ++i)
        ASSERT_TRUE(y2.getShapeInfo()[i] == shapeExpY2[i]);
    for(int i = 0; i < y2.lengthOf(); ++i)
        ASSERT_TRUE(y2.e<float>(i) == buffExpY2[i]);
    
    NDArray y3 = y(2, {1});
    for(int i = 0; i < shape::shapeInfoLength(y3.rankOf()); ++i)
        ASSERT_TRUE(y3.getShapeInfo()[i] == shapeExpY3[i]);
    for(int i = 0; i < y3.lengthOf(); ++i)
        ASSERT_TRUE(y3.e<float>(i) == buffExpY3[i]);

    NDArray y4 = y(2, {1}, true);
    for(int i = 0; i < shape::shapeInfoLength(y4.rankOf()); ++i)
        ASSERT_TRUE(y4.getShapeInfo()[i] == shapeExpY4[i]);
    for(int i = 0; i < y4.lengthOf(); ++i)
        ASSERT_TRUE(y4.e<float>(i) == buffExpY4[i]);

    NDArray y5 = y(3, {2});
    for(int i = 0; i < shape::shapeInfoLength(y5.rankOf()); ++i)
        ASSERT_TRUE(y5.getShapeInfo()[i] == shapeExpY5[i]);
    for(int i = 0; i < y5.lengthOf(); ++i)
        ASSERT_TRUE(y5.e<float>(i) == buffExpY5[i]);
        
}

TEST_F(NDArrayTest2, test_subarray_interval_1) {

    NDArray x('f', {10, 10}, nd4j::DataType::FLOAT32);
    auto subArr1 = x.subarray({NDIndex::all(), NDIndex::interval(0,9)});

    subArr1->printShapeInfo("subArr1");

    ASSERT_EQ(10, subArr1->sizeAt(0));
    ASSERT_EQ(9, subArr1->sizeAt(1));
    delete subArr1;
}

TEST_F(NDArrayTest2, test_subarray_interval_2) {

    NDArray x('c', {10, 10}, nd4j::DataType::FLOAT32);
    auto subArr1 = x.subarray({NDIndex::all(), NDIndex::interval(0,9)});

    subArr1->printShapeInfo("subArr1");

    ASSERT_EQ(10, subArr1->sizeAt(0));
    ASSERT_EQ(9, subArr1->sizeAt(1));
    delete subArr1;
}

TEST_F(NDArrayTest2, test_subarray_3d_cf) {
    NDArray f('f', {10, 20, 30}, nd4j::DataType::FLOAT32);
    NDArray c('c', {10, 20, 30}, nd4j::DataType::FLOAT32);

    auto subarrayF = f({0,0, 0,0, 2,3}, true);
    subarrayF.printShapeInfo("F subarray shapeInfo");

    auto subarrayC = c({2,3, 0,0, 0,0}, true);
    subarrayC.printShapeInfo("C subarray shapeInfo");
}

TEST_F(NDArrayTest2, test_broadcast_row_1) {
    auto x = NDArrayFactory::create<float>('c', {10, 5});
    auto y = NDArrayFactory::create<float>('c', {5}, {1.f, 1.f, 1.f, 1.f, 1.f});
    auto e = NDArrayFactory::create<float>('c', {10, 5});
    e.assign(1.0f);

    x += y;

    ASSERT_EQ(e, x);
}

TEST_F(NDArrayTest2, test_broadcast_column_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 10});
    auto y = NDArrayFactory::create<float>('c', {5,  1}, {1.f, 1.f, 1.f, 1.f, 1.f});
    auto e = NDArrayFactory::create<float>('c', {5, 10});
    e.assign(1.0f);

    x += y;

    ASSERT_EQ(e, x);
}

TEST_F(NDArrayTest2, test_broadcast_column_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 10});
    auto y = NDArrayFactory::create<float>('c', {5,  1}, {1.f, 1.f, 1.f, 1.f, 1.f});
    auto e = NDArrayFactory::create<float>('c', {5, 10});
    e.assign(1.0f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Add(), &y, &x, false);
    x.printShapeInfo();
    x.printIndexedBuffer();

    ASSERT_EQ(e, x);
}

TEST_F(NDArrayTest2, test_broadcast_column_3) {
    auto x = NDArrayFactory::create<float>('c', {5, 10});
    auto y = NDArrayFactory::create<float>('c', {5,  1}, {1.f, 1.f, 1.f, 1.f, 1.f});
    auto e = NDArrayFactory::create<float>('c', {5, 10});
    e.assign(1.0f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Add(), &y, &x);

    ASSERT_EQ(e, x);
}

TEST_F(NDArrayTest2, test_broadcast_column_4) {
    auto x = NDArrayFactory::create<float>('f', {10, 5});
    auto y = NDArrayFactory::create<float>('f', {5}, {1.f, 1.f, 1.f, 1.f, 1.f});
    auto e = NDArrayFactory::create<float>('f', {10, 5});
    e.assign(1.0f);

    x.applyTrueBroadcast(BroadcastOpsTuple::Add(), &y, &x);

    ASSERT_EQ(e, x);
}

TEST_F(NDArrayTest2, test_not_tiled_1) {
    auto x = NDArrayFactory::create<float>('c', {4, 12, 128, 128});
    auto y = NDArrayFactory::create<float>('c', {4, 1, 128, 128});
    auto e = NDArrayFactory::create<float>('c', {4, 12, 128, 128});
    y.assign(1.0f);
    e.assign(1.0f);

    x += y;

    ASSERT_EQ(e, x);
}

TEST_F(NDArrayTest2, test_not_tiled_2) {
    auto x = NDArrayFactory::create<float>('c', {4, 128, 768});
    auto y = NDArrayFactory::create<float>('c', {4, 128, 1});
    auto e = NDArrayFactory::create<float>('c', {4, 128, 768});
    y.assign(1.0f);
    e.assign(1.0f);

    x += y;

    ASSERT_EQ(e, x);
}

TEST_F(NDArrayTest2, test_long_sum_1) {
    auto x = NDArrayFactory::create<Nd4jLong>('c', {2, 2}, {1, 2, 3, 4});

    auto z = x.reduceAlongDims(reduce::Sum, {0});

    z.printIndexedBuffer("z long");
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, reshapei_1) {
    
    Nd4jLong shapeInfo1[] = {6,  2,1,2,1,7,1,   7,7,14,28,1,1,   8192, 0, 99};
    Nd4jLong shapeInfo2[] = {2,   4, 7,   7, 1,    8192, 1, 99};

    float buffer[shape::length(shapeInfo1)];
    NDArray x(buffer, shapeInfo1);

    const bool canReshape = x.reshapei({4,7});

    ASSERT_FALSE(canReshape);
    ASSERT_TRUE(shape::equalsStrict(x.getShapeInfo(), shapeInfo2));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, reshapei_2) {
    
    Nd4jLong shapeInfo1[] = {6,  1,2,1,2,7,1,   28,7,7,14,1,1,   8192, 0, 99};
    Nd4jLong shapeInfo2[] = {2,  4, 7,   7, 1,    8192, 1, 99};

    float buffer[shape::length(shapeInfo1)];
    NDArray x(buffer, shapeInfo1);

    const bool canReshape = x.reshapei({4,7});    

    ASSERT_FALSE(canReshape);
    ASSERT_TRUE(shape::equalsStrict(x.getShapeInfo(), shapeInfo2));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, trueBroadcast_1) {

    NDArray x('f', {2, 3}, {1., 2., 3., 4., 5., 6.});
    NDArray y('f', {1, 3}, {5., 4., 3.});
    NDArray z('c', {2, 3}, nd4j::DataType::DOUBLE);

    auto exp = x - y;
    x.applyTrueBroadcast(nd4j::BroadcastOpsTuple::Subtract(), &y, &z, true);

    // exp.printIndexedBuffer();
    // z.printIndexedBuffer();

    ASSERT_TRUE(exp.equalsTo(z));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, reduce_1) {

    NDArray arr6('f', {1, 1, 4, 4, 4, 4}, nd4j::DataType::DOUBLE);
    NDArray exp('f', {1, 1, 4, 4}, nd4j::DataType::DOUBLE);    

    arr6.linspace(1);

    NDArray* arr6s = arr6.reduceAlongDimension(nd4j::reduce::Sum, {2,3});

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double sum = 0;
            for (int x = 0; x < 4; x++) {
                for (int y = 0; y < 4; y++) {
                    Nd4jLong indices[] = {0, 0, x, y, i, j};
                    Nd4jLong offset = shape::getOffset(0, arr6.shapeOf(), arr6.stridesOf(), indices, arr6.rankOf());
                    sum += ((double*)arr6.getBuffer())[offset];
                }
            }
            exp.p<double>(0, 0, i, j, sum);
        }
    }
        
    // arr6s->printShapeInfo();
    // exp.printShapeInfo();
    // exp.printIndexedBuffer();
    // arr6s->printIndexedBuffer();

    ASSERT_TRUE(exp.equalsTo(arr6s));

    delete arr6s;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest2, reduce3_1) {

    NDArray x('c', {1,4}, {1,2,3,4});
    NDArray y('c', {1,4}, {2,3,4,5});
    NDArray exp('c', {4}, {1,1,1,1});

    NDArray* z = x.applyReduce3(nd4j::reduce3::EuclideanDistance, &y, {0}, nullptr);
    // z->printShapeInfo();
    // z->printIndexedBuffer();

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete z;
}

TEST_F(NDArrayTest2, all_tads_1) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});

    auto arrays = x.allTensorsAlongDimension({1});
    ASSERT_EQ(3, arrays->size());

    delete arrays;
}
