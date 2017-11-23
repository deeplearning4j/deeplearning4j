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
    std::vector<int> shape({1, 4});

    ShapeUtils<float>::insertDimension(2, shape.data(), -1, 3);
    ASSERT_EQ(4, shape.at(0));
    ASSERT_EQ(3, shape.at(1));
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, BasicInject2) {
    std::vector<int> shape({1, 4});

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

    int xShapeInfo[]   = {3, 3, 2, 2, 4, 2, 1, 0, 1, 99};
    int yShapeInfo[]   = {2,    1, 2,    2, 1, 0, 1, 99};
    int expShapeInfo[] = {3, 3, 2, 2, 4, 2, 1, 0, 1, 99};

    NDArray<float> x(xShapeInfo);
    NDArray<float> y(yShapeInfo);

    int *newShapeInfo = ShapeUtils<float>::evalBroadcastShapeInfo(x, y);
        
    ASSERT_TRUE(shape::equalsStrict(expShapeInfo, newShapeInfo));    

    RELEASE(newShapeInfo, x.getWorkspace());
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, EvalBroadcastShapeInfo_2)
{

    int xShapeInfo[]   = {4, 8, 1, 6, 1, 6,   6,  1, 1, 0, 1, 99};    
    int yShapeInfo[]   = {3,    7, 1, 5,      5,  5, 1, 0, 1, 99};
    int expShapeInfo[] = {4, 8, 7, 6, 5, 210, 30, 5, 1, 0, 1, 99};    

    NDArray<float> x(xShapeInfo);
    NDArray<float> y(yShapeInfo);

    int *newShapeInfo = ShapeUtils<float>::evalBroadcastShapeInfo(x, y);
        
    ASSERT_TRUE(shape::equalsStrict(expShapeInfo, newShapeInfo));    

    RELEASE(newShapeInfo, x.getWorkspace());
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, EvalBroadcastShapeInfo_3)
{

    int xShapeInfo[]   = {3, 15, 3, 5, 15, 5, 1, 0, 1, 99};
    int yShapeInfo[]   = {3, 15, 1, 5,  5, 5, 1, 0, 1, 99};
    int expShapeInfo[] = {3, 15, 3, 5, 15, 5, 1, 0, 1, 99};

    NDArray<float> x(xShapeInfo);
    NDArray<float> y(yShapeInfo);

    int *newShapeInfo = ShapeUtils<float>::evalBroadcastShapeInfo(x, y);
        
    ASSERT_TRUE(shape::equalsStrict(expShapeInfo, newShapeInfo));    

    RELEASE(newShapeInfo, x.getWorkspace());
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, EvalBroadcastShapeInfo_4)
{

    int xShapeInfo[]   = {3, 8, 1, 3,  3, 3, 1, 0, 1, 99};
    int yShapeInfo[]   = {2,    4, 3,     3, 1, 0, 1, 99};    
    int expShapeInfo[] = {3, 8, 4, 3, 12, 3, 1, 0, 1, 99};

    NDArray<float> x(xShapeInfo);
    NDArray<float> y(yShapeInfo);

    int *newShapeInfo = ShapeUtils<float>::evalBroadcastShapeInfo(x, y);
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

    int *newShapeInfo = ShapeUtils<float>::evalReduceShapeInfo('c', dimensions, x.getShapeInfo());    
    
    ASSERT_TRUE(shape::shapeEquals(expected.getShapeInfo(), newShapeInfo));

    delete []newShapeInfo;
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalReduceShapeInfo_test2)
{
    
    NDArray<float> x('c',{2,3,4,5});
    NDArray<float> expected('c', {2,1,4,5});
    std::vector<int> dimensions = {1};

    int *newShapeInfo = ShapeUtils<float>::evalReduceShapeInfo('c', dimensions, x.getShapeInfo(), true);    
    
    ASSERT_TRUE(shape::shapeEquals(expected.getShapeInfo(), newShapeInfo));

    delete []newShapeInfo;
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalReduceShapeInfo_test3)
{
    
    NDArray<float> x('c',{2,3,4,5});
    NDArray<float> expected('c', {1,1,1,5});
    std::vector<int> dimensions = {0,1,2};

    int *newShapeInfo = ShapeUtils<float>::evalReduceShapeInfo('c', dimensions, x.getShapeInfo(), true);    
    
    ASSERT_TRUE(shape::shapeEquals(expected.getShapeInfo(), newShapeInfo));

    delete []newShapeInfo;
}

//////////////////////////////////////////////////////////////////
TEST_F(ShapeUtilsTests, evalReduceShapeInfo_test4)
{
    
    NDArray<float> x('c',{2,3,4,5});
    NDArray<float> expected('c', {1,1,1,1});
    std::vector<int> dimensions = {0,1,2,3};

    int *newShapeInfo = ShapeUtils<float>::evalReduceShapeInfo('c', dimensions, x.getShapeInfo(), true);    
    
    ASSERT_TRUE(shape::shapeEquals(expected.getShapeInfo(), newShapeInfo));

    delete []newShapeInfo;
}

