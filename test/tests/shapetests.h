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

TEST_GROUP(Shape) {

    static int output_method(const char* output, ...) {
        va_list arguments;
        va_start(arguments, output);
        va_end(arguments);
        return 1;
    }
    void setup() {

    }
    void teardown() {
    }
};

TEST(Shape, IsVector) {

    int *shape = new int[2];
    shape[0] = 1;
    shape[1] = 2;
    CHECK(shape::isVector(shape, 2));
    shape[0] = 2;
    shape[1] = 1;
    CHECK(shape::isVector(shape, 2));
    delete[] shape;

}
TEST(Shape,ShapeInformation) {
    int rank = 4;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    shape[2] = 3;
    shape[3] = 2;
    int *stride = shape::calcStrides(shape, rank);

    shape::ShapeInformation *info = (shape::ShapeInformation *) malloc(
            sizeof(shape::ShapeInformation));
    info->shape = shape;
    info->stride = stride;
    info->elementWiseStride = 1;
    info->rank = rank;

    int *rearrange = (int *) malloc(sizeof(int) * 4);
    rearrange[0] = 2;
    rearrange[1] = 1;
    rearrange[2] = 3;
    rearrange[3] = 0;

    shape::permute(&info, rearrange, rank);
    int *shapeAssertion = (int *) malloc(sizeof(int) * 4);
    shapeAssertion[0] = 3;
    shapeAssertion[1] = 2;
    shapeAssertion[2] = 2;
    shapeAssertion[3] = 2;
    int shapeAssertionCorrect = 1;
    for (int i = 0; i < rank; i++) {
        shapeAssertionCorrect = (shapeAssertionCorrect
                                 && shapeAssertion[i] == info->shape[i]);
    }

    CHECK(shapeAssertion);

    free(rearrange);
    free(shapeAssertion);
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
    int *stride = shape::calcStrides(shape, rank);

    shape::ShapeInformation *info = (shape::ShapeInformation *) malloc(
            sizeof(shape::ShapeInformation));
    info->shape = shape;
    info->stride = stride;
    info->elementWiseStride = 1;
    info->rank = rank;

    int *shapeInfoBuff = shape::toShapeBuffer(info);
    CHECK(arrsEquals<int>(rank, shape, shape::shapeOf(shapeInfoBuff)));
    CHECK(arrsEquals<int>(rank, stride, shape::stride(shapeInfoBuff)));
    CHECK(info->elementWiseStride == shape::elementWiseStride(shapeInfoBuff));
    free(shape);
    free(stride);
    free(info);
    free(shapeInfoBuff);
}
TEST(Shape,Range) {
    int *rangeArr = shape::range(0, 4);
    int *testArr = shape::range(0, 4);
    int shapeAssertionCorrect = 1;
    for (int i = 0; i < 4; i++) {
        shapeAssertionCorrect = (shapeAssertionCorrect
                                 && rangeArr[i] == testArr[i]);
    }

    CHECK(shapeAssertionCorrect);
    free(testArr);
    free(rangeArr);

}

TEST(Shape,ReverseCopy) {
    int *assertion = (int *) malloc(sizeof(int) * 4);
    assertion[0] = 4;
    assertion[1] = 3;
    assertion[2] = 2;
    assertion[3] = 1;
    int *rangeArr = shape::range(1, 5);
    int *reverseCopy2 = shape::reverseCopy(rangeArr, 4);
    CHECK(arrsEquals<int>(4, assertion, reverseCopy2));
    free(reverseCopy2);
    free(rangeArr);
    free(assertion);

}

TEST(Shape,Keep) {
    int *keepAssertion = (int *) malloc(sizeof(int) * 2);
    for (int i = 0; i < 2; i++)
        keepAssertion[i] = i + 1;
    int *rangeArr = shape::range(1, 5);
    int *keepIndexes = (int *) malloc(sizeof(int) * 2);
    for (int i = 0; i < 2; i++)
        keepIndexes[i] = i;
    int *keep = shape::keep(rangeArr, keepIndexes, 2, 4);

    CHECK(arrsEquals<int>(2, keepAssertion, keep));

    int *keepAssertion2 = (int *) malloc(sizeof(int) * 2);
    keepAssertion2[0] = 2;
    keepAssertion2[1] = 3;
    int *keepAssertionIndexes2 = (int *) malloc(sizeof(int) * 2);
    for (int i = 0; i < 2; i++)
        keepAssertionIndexes2[i] = i + 1;
    int *keep2 = shape::keep(rangeArr, keepAssertionIndexes2, 2, 4);
    CHECK(arrsEquals<int>(2, keepAssertion2, keep2));

    free(keepIndexes);
    free(keepAssertionIndexes2);
    free(keepAssertion2);
    free(keep2);
    free(rangeArr);
    free(keep);
}

TEST(Shape,TensorsAlong longDimension) {
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

    int *shapeInfoBuffer = shapeBuffer(rank, shape);
    int tads = shape::tensorsAlong longDimension(shapeInfoBuffer, dimension,
                                            dimensionLength);
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
    int *shapeInfoBuffer = shapeBuffer(rank, shape);
    for(int i = 0; i < rank; i++) {
        printf("Stride[%d] was %d\n",i,shape::stride(shapeInfoBuffer)[i]);
    }
    int dimensionLength = 1;
    int *dimension = (int *) malloc(sizeof(int) * dimensionLength);
    dimension[0] = 1;
    int *tadShapeInfo = shape::tadShapeInfo(0, shapeInfoBuffer, dimension,
                                            dimensionLength);
    int tadLength = shape::length(tadShapeInfo);
    CHECK(shape::rank(tadShapeInfo) == 2);
    int *shapeAssertion = (int *) malloc(2 * sizeof(int));
    shapeAssertion[0] = 1;
    shapeAssertion[1] = 2;
    CHECK(arrsEquals<int>(1, shape::shapeOf(tadShapeInfo), shapeAssertion));
    int elementWiseStride = shape::computeElementWiseStride(
            shape::rank(shapeInfoBuffer), shape::shapeOf(shapeInfoBuffer),
            shape::stride(shapeInfoBuffer), 0, dimension, dimensionLength);
    CHECK(6 == elementWiseStride);
    int tensorsAlong longDimension = shape::tensorsAlong longDimension(shapeInfoBuffer,
                                                             dimension, dimensionLength);
    int idx = shape::reductionIndexForLinear(1, elementWiseStride, tadLength,
                                             tensorsAlong longDimension, tensorsAlong longDimension);
    CHECK(idx == 0);
    int idx2 = shape::reductionIndexForLinear(4, 1, tadLength,
                                              tensorsAlong longDimension, tensorsAlong longDimension);
    CHECK(idx2 == 2);
    free(tadShapeInfo);
    free(shapeInfoBuffer);
    free(shapeAssertion);

}

TEST(Shape,PermuteSwap) {
    int *rearrangeIndexes = (int *) malloc(sizeof(int) * 3);
    for (int i = 0; i < 3; i++) {
        rearrangeIndexes[i] = i;
    }
    int *reverse = shape::reverseCopy(rearrangeIndexes, 3);
    free(rearrangeIndexes);
    rearrangeIndexes = reverse;
    int *range = shape::range(0, 3);
    int *permuted = shape::doPermuteSwap(3, range, rearrangeIndexes);
    CHECK(arrsEquals<int>(3, permuted, rearrangeIndexes));
    free(range);
    free(permuted);
    free(rearrangeIndexes);

}

TEST(Shape,Offset) {
    int dimensionLength = 1;
    int rank = 2;
    int *dimension = (int *) malloc(sizeof(int) * dimensionLength);
    dimension[0] = 0;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 3;
    int *shapeBuff = shapeBuffer(rank,shape);
    int offsetAssertion = 1;
    shape::TADPermuteInfo info = shape::tadInfo(shapeBuff,dimension,dimensionLength);
    int offset1 = shape::offset(1,shapeBuff,dimension,1,info);
    if(offset1 != 1)
        printf("Offset != 1 %d\n",offset1);
    CHECK(offset1 == offsetAssertion);


    free(shapeBuff);
    free(dimension);
}

TEST(Shape,Ind2Sub) {
    int rank = 2;
    int *shape = (int *) malloc(sizeof(int) * rank);
    for(int i = 0; i < rank; i++)
        shape[i] = 2;
    int assertion[4][2] = {
            {0,0} ,
            {0,1} ,
            {1,0} ,
            {1,1} ,
    };

    int fAssertion[4][2] = {
            {0,0} ,
            {1,0} ,
            {0,1} ,
            {1,1} ,
    };

    for(int i = 0; i < 4; i++) {
        int *sub = shape::ind2subC(rank,shape,i);
        int *sub2 = shape::ind2sub(rank,shape,i);
        CHECK(arrsEquals<int>(rank, assertion[i], sub));
        CHECK(arrsEquals<int>(rank, fAssertion[i], sub2));
        free(sub);
        free(sub2);

    }

    free(shape);
}


TEST(Shape,ElementWiseStrideLessThanOne) {
    int length = 16;
    int rank = 2;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 2;
    shape[1] = 2;
    int *stride = (int *) malloc(sizeof(int) * rank);
    stride[0] = 8;
    stride[1] = 1;

    double *data = (double *) malloc(sizeof(double) * length);
    for(int i = 0; i < length; i++)
        data[i] = i + 1;

    int elementWise = shape::computeElementWiseStride(rank,shape,stride,0);
    if(elementWise >= 0)
        printf("Element wise failed %d\n",elementWise);
    CHECK(elementWise < 0);
    free(shape);
    free(stride);
    free(data);
}


TEST(Shape,ElementWiseStrideTwoHundred) {
    int length = 16;
    int rank = 2;
    int *shape = (int *) malloc(sizeof(int) * rank);
    shape[0] = 200;
    shape[1] = 200;
    int *stride = (int *) malloc(sizeof(int) * rank);
    stride[0] = 1;
    stride[1] = 200;

    double *data = (double *) malloc(sizeof(double) * length);
    for(int i = 0; i < length; i++)
        data[i] = i + 1;

    int elementWise = shape::computeElementWiseStride(rank,shape,stride,0);
    if(elementWise >= 0)
        printf("Element wise failed %d\n",elementWise);
    CHECK(elementWise < 0);
    free(shape);
    free(stride);
    free(data);

}

TEST(Shape,TadOffset) {
    int rank = 4;
    int dimensionLength = 1;
    int shape[4]= {2,2,3,2};
    int dimension[1] = {1};
    int *shapeBuff = shapeBuffer(rank,shape);
    int tads = shape::tensorsAlong longDimension(rank,24,shape,dimension,dimensionLength);
    CHECK_EQUAL(tads,12);
    int assertionsArr[4][12] = {
            {0,1,2,3,4,5,6,7,8,9,10,11},
            {0,1,2,3,4,5,12,13,14,15,16,17},
            {0,1,6,7,12,13,18,19,0,0,0,0},
            {0,2,4,6,8,10,12,14,16,18,20,22}
    };

    int elementWiseStrides[4] = {12,6,2,1};


    int tadLengths[4] = {12,12,8,12};

    for(int i = 0; i < 4; i++) {
        dimension[0] = i;
        int *currAssertion  = assertionsArr[i];
        for(int j = 0; j < tadLengths[i]; j++) {
            int thisAssertion = currAssertion[j];
            int tadOffset = shape::tadOffset(j,shapeBuff,dimension,dimensionLength);
            CHECK_EQUAL(thisAssertion,tadOffset);
            int tadElementWise = shape::tadElementWiseStride(shapeBuff,dimension,dimensionLength);
            CHECK_EQUAL(elementWiseStrides[i],tadElementWise);
        }

    }

    free(shapeBuff);

}



TEST(Shape,TadLength) {
    int rank = 4;
    int dimensionLength = 1;
    int shape[4]= {2,2,3,2};
    int dimension[1] = {1};
    int *shapeBuff = shapeBuffer(rank,shape);
    int tadLengths[4] = {2,2,3,2};
    for(int i = 0; i < rank; i++) {
        dimension[0] = i;
        int tadLength = shape::tadLength(shapeBuff,dimension,dimensionLength);
        CHECK_EQUAL(tadLengths[i],tadLength);
    }
    free(shapeBuff);
}
/*
TEST(Shape,TadOffsetMulti) {
    int rank = 4;
    int dimensionLength = 1;
    int shape[4]= {2,2,3,2};
    int dimension[2] = {0,1};
    int *shapeBuff = shapeBuffer(rank,shape);
    int tads = shape::tensorsAlong longDimension(rank,24,shape,dimension,dimensionLength);
    CHECK_EQUAL(tads,6);
    int assertionsArr[1][6] = {
            {0,1,2,3,4,5}
    };

    int elementWiseStrides[4] = {12,6,2,1};


    int tadLengths[4] = {12,12,8,12};

    for(int i = 0; i < 4; i++) {
        dimension[0] = i;
        int *currAssertion  = assertionsArr[i];
        for(int j = 0; j < tadLengths[i]; j++) {
            int thisAssertion = currAssertion[j];
            int tadOffset = shape::tadOffset(j,shapeBuff,dimension,dimensionLength);
            CHECK_EQUAL(thisAssertion,tadOffset);
            int tadElementWise = shape::tadElementWiseStride(shapeBuff,dimension,dimensionLength);
            CHECK_EQUAL(elementWiseStrides[i],tadElementWise);
        }

    }

    free(shapeBuff);

}

*/


#endif /* SHAPETESTS_H_ */
