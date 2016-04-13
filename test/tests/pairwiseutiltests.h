//
// Created by agibsonccc on 2/15/16.
//

#ifndef NATIVEOPERATIONS_PAIRWISEUTILTESTS_H
#define NATIVEOPERATIONS_PAIRWISEUTILTESTS_H
#include "testhelpers.h"
#include <pairwise_util.h>
#include <pairwise_transform.h>
#include <reduce3.h>
TEST_GROUP(PairWiseUtil) {

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


TEST(PairWiseUtil,Sort) {
    StridePermutation stridePermutation[4];
    for(int i = 0; i < 4; i++) {
        StridePermutation curr;
        curr.stride = i - 3;
        curr.perm = i;
        stridePermutation[i] = curr;

    }

    quickSort(stridePermutation,4);
    printf("Ran quick sort\n");

    for(int i = 0; i < 3; i++) {
        CHECK(stridePermutation[i].stride < stridePermutation[i + 1].stride);
    }

}

TEST(PairWiseUtil,IterationOne) {
    int shapeIter[MAX_RANK];
    int coord[MAX_RANK];
    int dim;
    int dstStridesIter[MAX_RANK];
    int shape[2] = {2,2};
    int strides[2] = {2,1};
    int rank = 2;

    double data[4];
    for(int i = 0; i < 4; i++)
        data[i] = i;

    double *out_data = NULL;

    printf("Succeeded %d\n",PrepareOneRawArrayIter(rank,shape,data,strides,&rank,shapeIter, &out_data, dstStridesIter));


    ND4J_RAW_ITER_START(dim, rank, coord, shapeIter) {
            /* Process the innermost dimension */
            double *d = out_data;
        } ND4J_RAW_ITER_ONE_NEXT(dim, rank, coord, shapeIter, out_data, dstStridesIter);
}


TEST(PairWiseUtil,DifferentOrderCopy) {
    int shape[3] = {2,2,2};

    double data[8];
    double yData[8];

    for(int i = 0; i < 8; i++)
        data[i] = i + 1;

    for(int i = 0; i < 8; i++) {
        yData[i] = i + 1;
    }

    double resultData[8];

    for(int i = 0; i < 8; i++) {
        resultData[i] = 0.0;
    }

    using namespace functions::pairwise_transforms;
    PairWiseTransform<double> *op  = new ops::Copy<double>();
    int *xShapeBuffer = shape::shapeBuffer(3,shape);
    int *yShapeBuffer = shape::shapeBuffer(3,shape);
    int indexes[] = {8};

    op->exec(data, xShapeBuffer, yData, yShapeBuffer, resultData, xShapeBuffer, NULL);
    for(int i = 0; i < 8; i++) {
        CHECK_EQUAL(resultData[i],yData[i]);
    }

    delete op;
    delete xShapeBuffer;
    delete yShapeBuffer;
}

TEST(PairWiseUtil,PairWiseUtilEuclideanDistance) {
    int shapeArr[2] = {2,2};
    constexpr int rank = 2;
    constexpr int length = 4;

    double data[length];
    double yData[length];

    for(int i = 0; i < length; i++)
        data[i] = i + 1;

    for(int i = 0; i < length; i++) {
        yData[i] = i + 1;
    }

    double assertion = 1.4142135623730951;

    using namespace functions::reduce3;
    Reduce3<double> *op  = new functions::reduce3::ops::EuclideanDistance<double>();
    int *xShapeBuffer = shape::shapeBuffer(rank,shapeArr);
    int *yShapeBuffer = shape::shapeBufferFortran(rank,shapeArr);
    double result = op->execScalar(data,xShapeBuffer,NULL,yData,yShapeBuffer);
    CHECK_EQUAL(assertion,result);

    delete op;
}

TEST(PairWiseUtil,PairWiseUtilEuclideanDistanceDimension) {
    int shapeArr[2] = {2,2};
    constexpr int rank = 2;
    constexpr int length = 4;

    double data[length];
    double yData[length];

    for(int i = 0; i < length; i++)
        data[i] = i + 1;

    for(int i = 0; i < length; i++) {
        yData[i] = i + 1;
    }

    using namespace functions::reduce3;
    Reduce3<double> *op  = new functions::reduce3::ops::EuclideanDistance<double>();
    int *xShapeBuffer = shape::shapeBuffer(rank,shapeArr);
    int *yShapeBuffer = shape::shapeBufferFortran(rank,shapeArr);
    double result[2];
    int resultShape[rank];
    resultShape[0] = 1;
    resultShape[1] = 2;

    constexpr int dimensionLength = 1;
    int dimension[dimensionLength];
    dimension[0] = 0;

    int *resultShapeBuffer = shape::shapeBuffer(rank,resultShape);
    op->exec(data,xShapeBuffer,NULL,yData,yShapeBuffer,result,resultShapeBuffer,dimension,dimensionLength);

    delete xShapeBuffer;
    delete yShapeBuffer;
    delete resultShapeBuffer;
    delete op;
}

TEST(PairWiseUtil,IterationTwo) {
    constexpr int RANK = 3;

    int shapeIter[MAX_RANK];
    int coord[MAX_RANK];
    int dim;
    int xStridesIter[MAX_RANK];
    int yStridesIter[MAX_RANK];
    int resultStridesIter[MAX_RANK];

    int shape[RANK] = {2,2,2};
    int xStrides[RANK] = {4,2,1};
    int yStrides[RANK] = {1,2,4};
    int *resultStrides = xStrides;

    double data[RANK][8];

    for(int i = 0; i < 8; i++)
        data[0][i] = i + 1;

    for(int i = 0; i < 8; i++) {
        data[1][i] = i + 1;
    }

    for(int i = 0; i < 8; i++) {
        data[2][i] = 0;
    }

    double *resultData[RANK];

    int out_rank = 0;

    printf("Succeeded %d\n",PrepareThreeRawArrayIter<double>(RANK,
                                                     shape,
                                                     data[0],
                                                     xStrides,
                                                     data[1],
                                                     yStrides,
                                                     data[2],
                                                     resultStrides,
                                                     out_rank,
                                                     shapeIter,
                                                     &resultData[0],
                                                     xStridesIter,
                                                     &resultData[1],
                                                     yStridesIter,
                                                     &resultData[2],
                                                     resultStridesIter));


    double xAssertion[2][2][2] = {
            {{1,2},{3,4}},{{5,6},{7,8}}
    };

    double yAssertion[2][2][2] = {
            {{1,5},{3,7}},{{2,6},{4,8}}
    };


    ND4J_RAW_ITER_START(dim, out_rank, coord, shapeIter) {
            /* Process the innermost dimension */
            double *xIter = resultData[0];
            double *yIter = resultData[1];
            double *resultIter = resultData[2];

            CHECK_EQUAL(xAssertion[coord[0]][coord[1]][coord[2]],xIter[0]);
            CHECK_EQUAL(yAssertion[coord[0]][coord[1]][coord[2]],yIter[0]);

            printf("Value for x %f y %f result %f\n",xIter[0],yIter[0],resultIter[0]);
        } ND4J_RAW_ITER_THREE_NEXT(dim,
                                   out_rank,
                                   coord,
                                   shapeIter,
                                   resultData[0],
                                   xStridesIter,
                                   resultData[1],
                                   yStridesIter,
                                   resultData[2],
                                   resultStridesIter);
}


#endif //NATIVEOPERATIONS_PAIRWISEUTILTESTS_H
