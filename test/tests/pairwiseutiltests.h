//
// Created by agibsonccc on 2/15/16.
//

#ifndef NATIVEOPERATIONS_PAIRWISEUTILTESTS_H
#define NATIVEOPERATIONS_PAIRWISEUTILTESTS_H
#include "testhelpers.h"
#include <pairwise_util.h>
#include <pairwise_transform.h>

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
    int srcStridesIter[MAX_RANK];
    int dstStridesIter[MAX_RANK];
    int shape[2] = {2,2};
    int strides[2] = {2,1};
    int rank = 2;
    double *data = (double *) malloc(sizeof(data) * 4);
    for(int i = 0; i < 4; i++)
        data[i] = i;

    printf("Succeeded %d\n",PrepareOneRawArrayIter(rank,shape,data,strides,&rank,shapeIter,&data,dstStridesIter));



    ND4J_RAW_ITER_START(dim, rank, coord, shapeIter) {
            /* Process the innermost dimension */
            double *d = data;

            for (int i = 0; i < shape[0]; ++i, d += strides[0]) {
                printf("Data %f\n",d[0]);
            }
        } ND4J_RAW_ITER_ONE_NEXT(dim, rank, coord, shapeIter, data, dstStridesIter);

    free(data);
}


TEST(PairWiseUtil,DifferentOrderCopy) {
    int shape[3] = {2,2,2};
    int xStrides[3] = {4,2,1};
    int yStrides[3] = {1,2,4};
    int *resultStrides = xStrides;
    int rank = 3;
    double *data = (double *) malloc(sizeof(double) * 8);
    for(int i = 0; i < 8; i++)
        data[i] = i + 1;
    double *yData = (double *) malloc(sizeof(double) * 8);
    for(int i = 0; i < 8; i++) {
        yData[i] = i + 1;
    }

    double *resultData = (double *) malloc(sizeof(double) * 48);
    for(int i = 0; i < 8; i++) {
        resultData[i] = 0.0;
    }

    using namespace functions::pairwise_transforms;
    PairWiseTransform<double> *op  = new ops::Copy<double>();
    int *xShapeBuffer = shape::shapeBuffer(3,shape);
    int *yShapeBuffer = shape::shapeBuffer(3,shape);
    op->exec(data,xShapeBuffer,yData,yShapeBuffer,resultData,xShapeBuffer,NULL,8);
    for(int i = 0; i < 8; i++) {
        CHECK_EQUAL(resultData[i],yData[i]);
    }

    delete op;

}

TEST(PairWiseUtil,IterationTwo) {
    int shapeIter[MAX_RANK];
    int coord[MAX_RANK];
    int dim;
    int xStridesIter[MAX_RANK];
    int yStridesIter[MAX_RANK];
    int resultStridesIter[MAX_RANK];

    int shape[3] = {2,2,2};
    int xStrides[3] = {4,2,1};
    int yStrides[3] = {1,2,4};
    int *resultStrides = xStrides;
    int rank = 3;
    double *data = (double *) malloc(sizeof(double) * 8);
    for(int i = 0; i < 8; i++)
        data[i] = i + 1;
    double *yData = (double *) malloc(sizeof(double) * 8);
    for(int i = 0; i < 8; i++) {
        yData[i] = i + 1;
    }

    double *resultData = (double *) malloc(sizeof(double) * 48);
    for(int i = 0; i < 8; i++) {
        resultData[i] = 0.0;
    }

    printf("Succeeded %d\n",PrepareThreeRawArrayIter(rank,
                                                     shape,
                                                     data,
                                                     xStrides,
                                                     yData,
                                                     yStrides,
                                                     resultData,
                                                     resultStrides,
                                                     &rank,
                                                     shapeIter,
                                                     &data,
                                                     xStridesIter,
                                                     &yData,
                                                     yStridesIter,
                                                     &resultData,
                                                     resultStridesIter));


    double xAssertion[2][2][2] = {
            {{1,2},{3,4}},{{5,6},{7,8}}
    };

    double yAssertion[2][2][2] = {
            {{1,5},{3,7}},{{2,6},{4,8}}
    };


    ND4J_RAW_ITER_START(dim, rank, coord, shapeIter) {
            /* Process the innermost dimension */
            double *xIter = data;
            double *yIter = yData;
            double *resultIter = resultData;
            printf("Processing dim %d\n",dim);
            for(int i = 0;i < rank; i++) {
                printf("Coord %d is %d\n",i,coord[i]);
            }
            CHECK_EQUAL(xAssertion[coord[0]][coord[1]][coord[2]],xIter[0]);
            CHECK_EQUAL(yAssertion[coord[0]][coord[1]][coord[2]],yIter[0]);

            printf("Value for x %f y %f result %f\n",xIter[0],yIter[0],resultIter[0]);
        } ND4J_RAW_ITER_THREE_NEXT(dim,
                                   rank,
                                   coord,
                                   shapeIter,
                                   data,
                                   xStridesIter,
                                   yData,
                                   yStridesIter,
                                   resultData,
                                   resultStridesIter);

    free(data);
    free(yData);
    free(resultData);
}


#endif //NATIVEOPERATIONS_PAIRWISEUTILTESTS_H
