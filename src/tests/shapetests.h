/*
 * shapetests.h
 *
 *  Created on: Jan 2, 2016
 *      Author: agibsonccc
 */

#ifndef SHAPETESTS_H_
#define SHAPETESTS_H_
#include <shape.h>
#include "testhelpers.h"

TEST_GROUP(Shape)
{

    static int output_method(const char* output, ...)
    {
        va_list arguments;
        va_start(arguments, output);
        va_end(arguments);
        return 1;
    }
    void setup()
    {


    }
    void teardown()
    {
    }
};

TEST(Shape, IsVector) {

    int *shape = new int[2];
    shape[0] = 1;
    shape[1] = 2;
    CHECK(shape::isVector(shape,2));
    shape[0] = 2;
    shape[1] = 1;
    CHECK(shape::isVector(shape,2));



    delete[] shape;


}

TEST(Shape,ShapeInformation) {
    int rank = 4;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    shape[2] = 3;
    shape[3] = 2;
    int *stride = shape::calcStrides(shape,rank);

    shape::ShapeInformation *info = (shape::ShapeInformation *) malloc(sizeof(shape::ShapeInformation));
    info->shape = shape;
    info->stride = stride;
    info->elementWiseStride = 1;
    info->rank = rank;

    int rearrange[4] = {2,1,3,0};

    shape::permute(&info,rearrange,rank);
    int shapeAssertion[4] = {3,2,2,2};
    int shapeAssertionCorrect = 1;
    for(int i = 0; i < rank; i++) {
        shapeAssertionCorrect = (shapeAssertionCorrect && shapeAssertion[i] == info->shape[i]);
    }

    CHECK(shapeAssertion);



    free(shape);
    free(stride);

}

TEST(Shape,ShapeInfoBuffer) {
    int rank = 4;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    shape[2] = 3;
    shape[3] = 2;
    int *stride = shape::calcStrides(shape,rank);

    shape::ShapeInformation *info = (shape::ShapeInformation *) malloc(sizeof(shape::ShapeInformation));
    info->shape = shape;
    info->stride = stride;
    info->elementWiseStride = 1;
    info->rank = rank;

    int *shapeInfoBuff = shape::toShapeBuffer(info);
    CHECK(arrsEquals<int>(rank,shape,shape::shapeOf(shapeInfoBuff)));
    CHECK(arrsEquals<int>(rank,stride,shape::stride(shapeInfoBuff)));
    CHECK(info->elementWiseStride == shape::elementWiseStride(shapeInfoBuff));
    free(shape);
    free(stride);
    free(info);
    free(shapeInfoBuff);
}


TEST(Shape,Range) {
    int rangeArr[4] = {0,1,2,3};
    int *testArr = shape::range(0,4);
    int shapeAssertionCorrect = 1;
    for(int i = 0; i < 4; i++) {
        shapeAssertionCorrect = (shapeAssertionCorrect && rangeArr[i] == testArr[i]);
    }

    CHECK(shapeAssertionCorrect);

    free(testArr);

}

TEST(Shape,ReverseCopy) {
    int assertion[4] = {4,3,2,1};
    int *rangeArr = shape::range(1,5);
    int *reverseCopy2 = shape::reverseCopy(rangeArr,4);
    CHECK(arrsEquals<int>(4,assertion,reverseCopy2));
    free(reverseCopy2);
    free(rangeArr);

}

TEST(Shape,Keep) {
    int keepAssertion[2] = {1,2};
    int *rangeArr = shape::range(1,5);
    int keepIndexes[2] = {0,1};
    int *keep = shape::keep(rangeArr,keepIndexes,2,4);

    CHECK(arrsEquals<int>(2,keepAssertion,keep));


    int keepAssertion2[2] = {2,3};
    int keepAssertionIndexes2[2] = {1,2};
    int *keep2 = shape::keep(rangeArr,keepAssertionIndexes2,2,4);
    CHECK(arrsEquals<int>(2,keepAssertion2,keep2));

    free(keep2);
    free(rangeArr);
    free(keep);
}


TEST(Shape,TensorsAlongDimension) {
    int rank = 4;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    shape[2] = 3;
    shape[3] = 2;
    int dimensionLength = 2;
    int *dimension = (int *) malloc(sizeof(int) * dimensionLength);
    dimension[0] = 1;
    dimension[1] = 2;

    int *shapeInfoBuffer = shapeBuffer(rank,shape);
    int tads = shape::tensorsAlongDimension(shapeInfoBuffer,dimension,dimensionLength);
    CHECK(4 == tads);

    free(shapeInfoBuffer);
    free(dimension);

}

TEST(Shape,ReductionIndexForLinear) {
    int rank = 4;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    shape[2] = 3;
    shape[3] = 2;
    int *shapeInfoBuffer = shapeBuffer(rank,shape);
    assertBufferProperties(shapeInfoBuffer);
    int dimensionLength = 1;
    int *dimension = (int *) malloc(sizeof(int) * dimensionLength);
    dimension[0] = 1;
    int *tadShapeInfo = shape::tadShapeInfo(0,shapeInfoBuffer,dimension,dimensionLength);
    assertBufferProperties(tadShapeInfo);
    int tadLength = shape::length(tadShapeInfo);
    CHECK(shape::rank(tadShapeInfo) == 2);
    int *shapeAssertion = (int *) malloc(2 *  sizeof(int));
    shapeAssertion[0] = 1;
    shapeAssertion[1] = 2;
    CHECK(arrsEquals<int>(1,shape::shapeOf(tadShapeInfo),shapeAssertion));
    int elementWiseStride = shape::computeElementWiseStride(shape::rank(tadShapeInfo),shape::shapeOf(tadShapeInfo),shape::stride(tadShapeInfo),0,dimension,dimensionLength);
    CHECK(6 == elementWiseStride);
    int tensorsAlongDimension = shape::tensorsAlongDimension(shapeInfoBuffer,dimension,dimensionLength);
    int idx = shape::reductionIndexForLinear(1,1,tadLength,tensorsAlongDimension,tensorsAlongDimension);
    free(tadShapeInfo);
    free(shapeInfoBuffer);
}

TEST(Shape,ReductionIndexForTad) {

}





#endif /* SHAPETESTS_H_ */
