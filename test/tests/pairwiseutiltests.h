//
// Created by agibsonccc on 2/15/16.
//

#ifndef NATIVEOPERATIONS_PAIRWISEUTILTESTS_H
#define NATIVEOPERATIONS_PAIRWISEUTILTESTS_H
#include "testhelpers.h"
#include <pairwise_util.h>
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

    for(int i = 0; i < 4; i++) {
        printf("Stride %d with stride %d\n",i,stridePermutation[i].stride);
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
    ND4J_RAW_ITER_START(dim, rank, coord, shapeIter) {
            printf("Data %f\n",data[0]);
        } ND4J_RAW_ITER_ONE_NEXT(dim, rank, coord, shape, data, strides);

}




#endif //NATIVEOPERATIONS_PAIRWISEUTILTESTS_H
