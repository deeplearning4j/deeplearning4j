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

    /*
     *
     *   int idim;
    npy_intp shape_it[NPY_MAXDIMS], dst_strides_it[NPY_MAXDIMS];
    npy_intp coord[NPY_MAXDIMS];

    PyArray_StridedUnaryOp *stransfer = NULL;
    NpyAuxData *transferdata = NULL;
    int aligned, needs_api = 0;
    npy_intp src_itemsize = src_dtype->elsize;

    NPY_BEGIN_THREADS_DEF;

    Check alignment
    aligned = raw_array_is_aligned(ndim, dst_data, dst_strides,
                                   dst_dtype->alignment);
    if (!npy_is_aligned(src_data, src_dtype->alignment)) {
        aligned = 0;
    }

    Use raw iteration with no heap allocation
    if (PyArray_PrepareOneRawArrayIter(
            ndim, shape,
            dst_data, dst_strides,
            &ndim, shape_it,
            &dst_data, dst_strides_it) < 0)*/
    printf("Succeeded %d\n",PrepareOneRawArrayIter(rank,shape,data,strides,&rank,shapeIter,&data,dstStridesIter));



    ND4J_RAW_ITER_START(dim, rank, coord, shapeIter) {
            /* Process the innermost dimension */
            double *d = data;

            for (int i = 0; i < shape[0]; ++i, d += strides[0]) {
               printf("Data %f\n",d[0]);
            }
        } ND4J_RAW_ITER_ONE_NEXT(dim, rank, coord, shapeIter, data, dstStridesIter);

}




#endif //NATIVEOPERATIONS_PAIRWISEUTILTESTS_H
