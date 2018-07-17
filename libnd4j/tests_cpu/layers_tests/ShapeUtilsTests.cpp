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
// Created by raver119 on 01.11.2017.
//

#include "testlayers.h"
#include <helpers/ShapeUtils.h>
#include <NDArray.h>


using namespace nd4j;
using namespace nd4j::graph;

class ShapeUtilsTests : public testing::Test {
public:

};

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, BasicInject1) {
    std::vector<Nd4jLong> shape({1, 4});

    ShapeUtils<float>::insertDimension(2, shape.data(), -1, 3);
    ASSERT_EQ(4, shape.at(0));
    ASSERT_EQ(3, shape.at(1));
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, BasicInject2) {
    std::vector<Nd4jLong> shape({1, 4});

    ShapeUtils<float>::insertDimension(2, shape.data(), 0, 3);
    ASSERT_EQ(3, shape.at(0));
    ASSERT_EQ(4, shape.at(1));
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, AxisConversionTest_1) {
    std::vector<int> res = ShapeUtils<float>::convertAxisToTadTarget(3, {0});

    ASSERT_EQ(2, res.size());
    ASSERT_EQ(1, res.at(0));
    ASSERT_EQ(2, res.at(1));
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, AxisConversionTest_2) {
    std::vector<int> res = ShapeUtils<float>::convertAxisToTadTarget(4, {2, 3});

    ASSERT_EQ(2, res.size());
    ASSERT_EQ(0, res.at(0));
    ASSERT_EQ(1, res.at(1));
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, EvalBroadcastShapeInfo_1)
{

    Nd4jLong xShapeInfo[]   = {3, 3, 2, 2, 4, 2, 1, 0, 1, 99};
    Nd4jLong yShapeInfo[]   = {2,    1, 2,    2, 1, 0, 1, 99};
    Nd4jLong expShapeInfo[] = {3, 3, 2, 2, 4, 2, 1, 0, 1, 99};

    NDArray<float> x(xShapeInfo);
    NDArray<float> y(yShapeInfo);

    Nd4jLong *newShapeInfo = nullptr;
    ShapeUtils<float>::evalBroadcastShapeInfo(x, y, false, newShapeInfo, nullptr);
        
    ASSERT_TRUE(shape::equalsStrict(expShapeInfo, newShapeInfo));    

    RELEASE(newShapeInfo, x.getWorkspace());
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, EvalBroadcastShapeInfo_2)
{

    Nd4jLong xShapeInfo[]   = {4, 8, 1, 6, 1, 6,   6,  1, 1, 0, 1, 99};    
    Nd4jLong yShapeInfo[]   = {3,    7, 1, 5,      5,  5, 1, 0, 1, 99};
    Nd4jLong expShapeInfo[] = {4, 8, 7, 6, 5, 210, 30, 5, 1, 0, 1, 99};    

    NDArray<float> x(xShapeInfo);
    NDArray<float> y(yShapeInfo);

    Nd4jLong *newShapeInfo = nullptr;
    ShapeUtils<float>::evalBroadcastShapeInfo(x, y, false, newShapeInfo, nullptr);
        
    ASSERT_TRUE(shape::equalsStrict(expShapeInfo, newShapeInfo));    

    RELEASE(newShapeInfo, x.getWorkspace());
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, EvalBroadcastShapeInfo_3)
{

    Nd4jLong xShapeInfo[]   = {3, 15, 3, 5, 15, 5, 1, 0, 1, 99};
    Nd4jLong yShapeInfo[]   = {3, 15, 1, 5,  5, 5, 1, 0, 1, 99};
    Nd4jLong expShapeInfo[] = {3, 15, 3, 5, 15, 5, 1, 0, 1, 99};

    NDArray<float> x(xShapeInfo);
    NDArray<float> y(yShapeInfo);

    Nd4jLong *newShapeInfo = nullptr;
    ShapeUtils<float>::evalBroadcastShapeInfo(x, y, false, newShapeInfo, nullptr);
        
    ASSERT_TRUE(shape::equalsStrict(expShapeInfo, newShapeInfo));    

    RELEASE(newShapeInfo, x.getWorkspace());
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, EvalBroadcastShapeInfo_4)
{

    Nd4jLong xShapeInfo[]   = {3, 8, 1, 3,  3, 3, 1, 0, 1, 99};
    Nd4jLong yShapeInfo[]   = {2,    4, 3,     3, 1, 0, 1, 99};    
    Nd4jLong expShapeInfo[] = {3, 8, 4, 3, 12, 3, 1, 0, 1, 99};

    NDArray<float> x(xShapeInfo);
    NDArray<float> y(yShapeInfo);

    Nd4jLong *newShapeInfo = nullptr;
    ShapeUtils<float>::evalBroadcastShapeInfo(x, y, false, newShapeInfo, nullptr);
    //for(int i=0; i<2*newShapeInfo[0]+4; ++i)
    //        std::cout<<newShapeInfo[i]<<" ";
    //  std::cout<<std::endl;

    ASSERT_TRUE(shape::equalsStrict(expShapeInfo, newShapeInfo));

    RELEASE(newShapeInfo, x.getWorkspace());
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalReduceShapeInfo_test1)
{
    
    NDArray<float> x('c',{2,3,4,5});
    NDArray<float> expected('c', {2,4,5});    
    std::vector<int> dimensions = {1};

    auto newShapeInfo = ShapeUtils<float>::evalReduceShapeInfo('c', dimensions, x.getShapeInfo());    
    
    ASSERT_TRUE(shape::shapeEquals(expected.getShapeInfo(), newShapeInfo));

    delete []newShapeInfo;
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalReduceShapeInfo_test2)
{
    
    NDArray<float> x('c',{2,3,4,5});
    NDArray<float> expected('c', {2,1,4,5});
    std::vector<int> dimensions = {1};

    auto newShapeInfo = ShapeUtils<float>::evalReduceShapeInfo('c', dimensions, x.getShapeInfo(), true);    
    
    ASSERT_TRUE(shape::shapeEquals(expected.getShapeInfo(), newShapeInfo));

    delete []newShapeInfo;
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalReduceShapeInfo_test3)
{
    
    NDArray<float> x('c',{2,3,4,5});
    NDArray<float> expected('c', {1,1,1,5});
    std::vector<int> dimensions = {0,1,2};

    auto newShapeInfo = ShapeUtils<float>::evalReduceShapeInfo('c', dimensions, x.getShapeInfo(), true);    
    
    ASSERT_TRUE(shape::shapeEquals(expected.getShapeInfo(), newShapeInfo));

    delete []newShapeInfo;
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalReduceShapeInfo_test4)
{
    
    NDArray<float> x('c',{2,3,4,5});
    NDArray<float> expected('c', {1,1,1,1});
    std::vector<int> dimensions = {0,1,2,3};

    auto newShapeInfo = ShapeUtils<float>::evalReduceShapeInfo('c', dimensions, x.getShapeInfo(), true);    
    
    ASSERT_TRUE(shape::shapeEquals(expected.getShapeInfo(), newShapeInfo));

    delete []newShapeInfo;
}

TEST_F(ShapeUtilsTests, Test_Strings_1) {
    NDArray<float> x('c', {2, 3, 4, 5});
    std::string exp("[2, 3, 4, 5]");

    auto s = ShapeUtils<float>::shapeAsString(&x);

    ASSERT_EQ(exp, s);
}

TEST_F(ShapeUtilsTests, Test_Backward_Axis_1) {
    NDArray<float> x('c', {2, 4, 3});
    NDArray<float> y('c', {4, 3});
    std::vector<int> exp({0});

    auto z = ShapeUtils<float>::evalBroadcastBackwardAxis(y.shapeInfo(), x.shapeInfo());

    ASSERT_EQ(exp, z);
}

TEST_F(ShapeUtilsTests, Test_Backward_Axis_2) {
    NDArray<float> x('c', {2, 4, 4, 3});
    NDArray<float> y('c', {4, 1, 3});
    std::vector<int> exp({0, 2});

    auto z = ShapeUtils<float>::evalBroadcastBackwardAxis(y.shapeInfo(), x.shapeInfo());

    ASSERT_EQ(exp, z);
}


TEST_F(ShapeUtilsTests, Test_Backward_Axis_3) {
    NDArray<float> x('c', {2, 4, 4, 3});
    NDArray<float> y('c', {2, 1, 1, 3});
    std::vector<int> exp({1, 2});

    auto z = ShapeUtils<float>::evalBroadcastBackwardAxis(y.shapeInfo(), x.shapeInfo());

    ASSERT_EQ(exp, z);
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalPermutFromTo_test1) {
    
    int a=1, b=2, c=3, d=4;
    std::vector<int> expected = {2, 3, 0, 1};    
    
    std::vector<int> result = ShapeUtils<float>::evalPermutFromTo({a,b,c,d}, {c,d,a,b});    
    
    ASSERT_TRUE(std::equal(begin(expected), end(expected), begin(result)));    
    
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalPermutFromTo_test2) {    
    
    int a=1, b=2, c=3, d=4;
    std::vector<int> expected = {0, 1, 3, 2};    
    
    std::vector<int> result = ShapeUtils<float>::evalPermutFromTo({a,b,c,d}, {a,b,d,c});    
    
    ASSERT_TRUE(std::equal(begin(expected), end(expected), begin(result)));    
    
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalPermutFromTo_test3) {    
    
    int a=2, b=2, c=3, d=2;
    std::vector<int> expected = {0, 1, 3, 2};    
    
    std::vector<int> result = ShapeUtils<float>::evalPermutFromTo({a,b,c,d}, {a,b,d,c});    
    
    ASSERT_TRUE(std::equal(begin(expected), end(expected), begin(result)));    
    
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalPermutFromTo_test4) {
    
    int a=2, b=3, c=4, d=5;    
    
    std::vector<int> result = ShapeUtils<float>::evalPermutFromTo({a,b,c,d}, {a,b,c,d});
    
    ASSERT_TRUE(result.empty());    
    
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalPermutFromTo_test5) {
    
    int a=1, b=2, c=3, d=4;
    
    // EXPECT_THROW(ShapeUtils<float>::evalPermutFromTo({a,b,c,d}, {c,d,a,8}), const char*);              
    ASSERT_TRUE(1);
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalPermutFromTo_test6) {
    
    int a=1, b=2, c=3, d=4;
        
    // EXPECT_THROW(ShapeUtils<float>::evalPermutFromTo({a,b,c,d}, {a,b,c,d,d}), const char*);    
    ASSERT_TRUE(1);
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, isPermutNecessary_test1) {
         
    ASSERT_TRUE(ShapeUtils<float>::isPermutNecessary({1,0,2,3}));        
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, isPermutNecessary_test2) {
         
    ASSERT_TRUE(!ShapeUtils<float>::isPermutNecessary({0,1,2,3}));        
}


