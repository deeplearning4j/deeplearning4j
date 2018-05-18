//
// Created by agibsonccc on 2/20/18.
//

#include "testinclude.h"
#include <reduce3.h>
#include <ShapeUtils.h>
#include <vector>

class EuclideanTest : public testing::Test {
public:
    Nd4jLong yShape[4] = {4,4};
    Nd4jLong xShape[2] = {1,4};
    float y[16] ={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float x[4] = {1,2,3,4};
    int dimension[1] = {1};
    int dimensionLength = 1;
    int opNum = 1;
    float extraVals[1] = {0};
    float result[4] = {0.0,0.0,0.0,0.0};

    std::vector<int> dim = {1};
};

TEST_F(EuclideanTest,Test1) {
    auto shapeBuffer = shape::shapeBuffer(2,yShape);
    auto xShapeBuffer = shape::shapeBuffer(2,xShape);

    //int *tadShapeBuffer = shape::computeResultShape(shapeBuffer,dimension,dimensionLength);
    auto tadShapeBuffer = nd4j::ShapeUtils<float>::evalReduceShapeInfo('c', dim, shapeBuffer, false, true, nullptr);
            functions::reduce3::Reduce3<float>::exec(opNum,
                                             x,
                                             xShapeBuffer,
                                             extraVals,
                                             y,
                                             shapeBuffer,
                                             result,
                                             tadShapeBuffer,
                                             dimension,
                                             dimensionLength);

    float distancesAssertion[4] = {0.0,8.0,16.0,24.0};
    for(int i = 0; i < 4; i++) {
        ASSERT_EQ(distancesAssertion[i],result[i]);
    }

    //ASSERT_EQ(result[1],result[0]);
    delete[] shapeBuffer;
    delete[] tadShapeBuffer;
    delete[] xShapeBuffer;
}




