//
// Created by agibsonccc on 2/15/16.
//

#ifndef NATIVEOPERATIONS_PAIRWISE_UTIL_H
#define NATIVEOPERATIONS_PAIRWISE_UTIL_H
#include <array.h>
#include <dll.h>
#include <nd4jmemset.h>
//Loops adapted from:
//https://github.com/numpy/numpy/blob/009b17a85a22707e63ac9ea1896413992bbf9ce5/numpy/core/src/private/lowlevel_strided_loops.h#L401-L401


/************************************************************
 * A struct used by CreateSortedStridePerm, new in 1.7.
 ************************************************************/

typedef struct {
    int perm, stride;
} StridePermutation;


/**
 * Credit to:
 * http://alienryderflex.com/quicksort/
 *
 * The non recursive implementation is important for being able to run on cuda as well
 *  as host.
 *
 *  In practice the work loads intended for
 *  this won't be so hard
 *  that it matters.
 *
 *  We can work on optimizations later that leverage openmp,
 *  the gpu etc
 */
#ifdef __CUDACC__
__host__
#endif
void quickSort(StridePermutation *arr, int elements);


/* Start raw iteration */
#define ND4J_RAW_ITER_START(idim, ndim, coord, shape) \
        memset((coord), 0, (ndim) * sizeof(coord[0])); \
        do {

/* Increment to the next n-dimensional coordinate for one raw array */
#define ND4J_RAW_ITER_ONE_NEXT(idim, ndim, coord, shape, data, strides) \
            for ((idim) = 0; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (data) -= ((shape)[idim] - 1) * (strides)[idim]; \
                } \
                else { \
                    (data) += (strides)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for two raw arrays */
#define ND4J_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, dataB, stridesB) \
            for ((idim) = 0; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for three raw arrays */
#define ND4J_RAW_ITER_THREE_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, \
                              dataB, stridesB, \
                              dataC, stridesC) \
            for ((idim) = 0; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                    (dataC) -= ((shape)[idim] - 1) * (stridesC)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    (dataC) += (stridesC)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))

/* Increment to the next n-dimensional coordinate for four raw arrays */
#define ND4J_RAW_ITER_FOUR_NEXT(idim, ndim, coord, shape, \
                              dataA, stridesA, \
                              dataB, stridesB, \
                              dataC, stridesC, \
                              dataD, stridesD) \
            for ((idim) = 0; (idim) < (ndim); ++(idim)) { \
                if (++(coord)[idim] == (shape)[idim]) { \
                    (coord)[idim] = 0; \
                    (dataA) -= ((shape)[idim] - 1) * (stridesA)[idim]; \
                    (dataB) -= ((shape)[idim] - 1) * (stridesB)[idim]; \
                    (dataC) -= ((shape)[idim] - 1) * (stridesC)[idim]; \
                    (dataD) -= ((shape)[idim] - 1) * (stridesD)[idim]; \
                } \
                else { \
                    (dataA) += (stridesA)[idim]; \
                    (dataB) += (stridesB)[idim]; \
                    (dataC) += (stridesC)[idim]; \
                    (dataD) += (stridesD)[idim]; \
                    break; \
                } \
            } \
        } while ((idim) < (ndim))






/*NUMPY_API
 *
 * This function populates the first ndim elements
 * of strideperm with sorted descending by their absolute values.
 * For example, the stride array (4, -2, 12) becomes
 * [(2, 12), (0, 4), (1, -2)].
 */
#ifdef __CUDACC__
__host__
#endif
void  SortStrideArray(int ndim, int *strides,
                      StridePermutation *out_strideperm) {

    /* Set up the strideperm values */
    for (int i = 0; i < ndim; i++) {
        out_strideperm[i].perm = i;
        out_strideperm[i].stride = strides[i];
    }

    /* Sort them */
    quickSort(out_strideperm,ndim);

}

/*
 * Prepares shape and strides for a simple raw array iteration.
 * This sorts the strides into FORTRAN order, reverses any negative
 * strides, then coalesces axes where possible. The results are
 * filled in the output parameters.
 *
 * This is intended for simple, lightweight iteration over arrays
 * where no buffering of any kind is needed, and the array may
 * not be stored as a PyArrayObject.
 *
 * The arrays shape, out_shape, strides, and out_strides must all
 * point to different data.
 *
 * Returns 0 on success, -1 on failure.
 */
template <typename T>

#ifdef __CUDACC__
__host__
#endif
int PrepareOneRawArrayIter(int ndim, int *shape,
                           T *data, int *strides,
                           int *out_ndim, int *out_shape,
                           T **out_data, int *out_strides) {
    StridePermutation strideperm[MAX_RANK];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_data = data;
        out_shape[0] = 1;
        out_strides[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        int stride_entry = strides[0], shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride */
        if (stride_entry >= 0) {
            *out_data = data;
            out_strides[0] = stride_entry;
        }
        else {
            *out_data = data + stride_entry * (shape_entry - 1);
            out_strides[0] = -stride_entry;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    SortStrideArray(ndim, strides, strideperm);
    for (i = 0; i < ndim; i++) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_strides[i] = strides[iperm];
    }

    /* Reverse any negative strides */
    for (i = 0; i < ndim; ++i) {
        int stride_entry = out_strides[i], shape_entry = out_shape[i];

        if (stride_entry < 0) {
            data += stride_entry * (shape_entry - 1);
            out_strides[i] = -stride_entry;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_data = data;
            out_shape[0] = 0;
            out_strides[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_strides[i] = out_strides[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_strides[i] * out_shape[i] == out_strides[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_strides[i] = out_strides[j];
        }
    }
    ndim = i+1;

#if 0
    /* DEBUG */
    {
        printf("raw iter ndim %d\n", ndim);
        printf("shape: ");
        for (i = 0; i < ndim; ++i) {
            printf("%d ", (int)out_shape[i]);
        }
        printf("\n");
        printf("strides: ");
        for (i = 0; i < ndim; ++i) {
            printf("%d ", (int)out_strides[i]);
        }
        printf("\n");
    }
#endif

    *out_data = data;
    *out_ndim = ndim;
    return 0;
}




/**
 * Credit to:
 * http://alienryderflex.com/quicksort/
 *
 * The non recursive implementation is important for being able to run on cuda as well
 *  as host.
 *
 *  In practice the work loads intended for
 *  this won't be so hard
 *  that it matters.
 *
 *  We can work on optimizations later that leverage openmp,
 *  the gpu etc
 */
#ifdef __CUDACC__
__host__
#endif
void quickSort(StridePermutation *arr, int elements) {

#define  MAX_LEVELS  300

    int  beg[MAX_LEVELS], end[MAX_LEVELS], i= 0, L, R, swap ;
    StridePermutation piv;
    beg[0] = 0;
    end[0] = elements;
    while (i >= 0) {
        L = beg[i];
        R=  end[i] - 1;
        if (L < R) {
            piv = arr[L];
            while (L < R) {
                while (arr[R].stride >= piv.stride && L < R)
                    R--;
                if (L < R)
                    arr[L++] = arr[R];
                while (arr[L].stride <= piv.stride && L < R)
                    L++;
                if (L<R)
                    arr[R--] = arr[L];
            }

            arr[L] = piv;
            beg[i + 1]= L + 1;
            end[i + 1]= end[i];
            end[i++] = L;
            if (end[i] - beg[i] > end[i - 1] - beg[i - 1]) {
                swap = beg[i];
                beg[i]= beg[i - 1];
                beg[i - 1] = swap;
                swap = end[i];
                end[i] = end[i - 1];
                end[i - 1] = swap;
            }
        }
        else {
            i--;
        }
    }
}

/*
 * The same as PrepareOneRawArrayIter, but for two
 * operands instead of one. Any broadcasting of the two operands
 * should have already been done before calling this function,
 * as the ndim and shape is only specified once for both operands.
 *
 * Only the strides of the first operand are used to reorder
 * the dimensions, no attempt to consider all the strides together
 * is made, as is done in the NpyIter object.
 *
 * You can use this together with NPY_RAW_ITER_START and
 * NPY_RAW_ITER_TWO_NEXT to handle the looping boilerplate of everything
 * but the innermost loop (which is for idim == 0).
 *
 * Returns 0 on success, -1 on failure.
 */
template <typename T>
#ifdef __CUDACC__
__host__
#endif
int PrepareTwoRawArrayIter(int ndim, int *shape,
                           T *dataA, int *stridesA,
                           T *dataB, int *stridesB,
                           int *out_ndim, int *out_shape,
                           T **out_dataA, int *out_stridesA,
                           T **out_dataB, int *out_stridesB)
{
    StridePermutation strideperm[MAX_RANK];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_dataA = dataA;
        *out_dataB = dataB;
        out_shape[0] = 1;
        out_stridesA[0] = 0;
        out_stridesB[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        int stride_entryA = stridesA[0], stride_entryB = stridesB[0];
        int shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride for the first operand */
        if (stride_entryA >= 0) {
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_stridesA[0] = stride_entryA;
            out_stridesB[0] = stride_entryB;
        }
        else {
            *out_dataA = dataA + stride_entryA * (shape_entry - 1);
            *out_dataB = dataB + stride_entryB * (shape_entry - 1);
            out_stridesA[0] = -stride_entryA;
            out_stridesB[0] = -stride_entryB;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    SortStrideArray(ndim, stridesA, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_stridesA[i] = stridesA[iperm];
        out_stridesB[i] = stridesB[iperm];
    }

    /* Reverse any negative strides of operand A */
    for (i = 0; i < ndim; ++i) {
        int stride_entryA = out_stridesA[i];
        int stride_entryB = out_stridesB[i];
        int shape_entry = out_shape
        [i];

        if (stride_entryA < 0) {
            dataA += stride_entryA * (shape_entry - 1);
            dataB += stride_entryB * (shape_entry - 1);
            out_stridesA[i] = -stride_entryA;
            out_stridesB[i] = -stride_entryB;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_shape[0] = 0;
            out_stridesA[0] = 0;
            out_stridesB[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                 out_stridesB[i] * out_shape[i] == out_stridesB[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
    }
    ndim = i+1;

    *out_dataA = dataA;
    *out_dataB = dataB;
    *out_ndim = ndim;
    return 0;
}

/*
 * The same as PrepareOneRawArrayIter, but for three
 * operands instead of one. Any broadcasting of the three operands
 * should have already been done before calling this function,
 * as the ndim and shape is only specified once for all operands.
 *
 * Only the strides of the first operand are used to reorder
 * the dimensions, no attempt to consider all the strides together
 * is made, as is done in the NpyIter object.
 *
 * You can use this together with NPY_RAW_ITER_START and
 * NPY_RAW_ITER_THREE_NEXT to handle the looping boilerplate of everything
 * but the innermost loop (which is for idim == 0).
 *
 * Returns 0 on success, -1 on failure.
 */
template <typename T>
#ifdef __CUDACC__
__host__
#endif
int  PrepareThreeRawArrayIter(int ndim, int *shape,
                              T *dataA, int *stridesA,
                              T *dataB, int *stridesB,
                              T *dataC, int *stridesC,
                              int *out_ndim, int *out_shape,
                              T **out_dataA, int *out_stridesA,
                              T **out_dataB, int *out_stridesB,
                              T **out_dataC, int *out_stridesC)
{
    StridePermutation strideperm[MAX_RANK];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_dataA = dataA;
        *out_dataB = dataB;
        *out_dataC = dataC;
        out_shape[0] = 1;
        out_stridesA[0] = 0;
        out_stridesB[0] = 0;
        out_stridesC[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        int stride_entryA = stridesA[0];
        int stride_entryB = stridesB[0];
        int stride_entryC = stridesC[0];
        int shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride for the first operand */
        if (stride_entryA >= 0) {
            *out_dataA = dataA;
            *out_dataB = dataB;
            *out_dataC = dataC;
            out_stridesA[0] = stride_entryA;
            out_stridesB[0] = stride_entryB;
            out_stridesC[0] = stride_entryC;
        }
        else {
            *out_dataA = dataA + stride_entryA * (shape_entry - 1);
            *out_dataB = dataB + stride_entryB * (shape_entry - 1);
            *out_dataC = dataC + stride_entryC * (shape_entry - 1);
            out_stridesA[0] = -stride_entryA;
            out_stridesB[0] = -stride_entryB;
            out_stridesC[0] = -stride_entryC;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    SortStrideArray(ndim, stridesA, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_stridesA[i] = stridesA[iperm];
        out_stridesB[i] = stridesB[iperm];
        out_stridesC[i] = stridesC[iperm];
    }

    /* Reverse any negative strides of operand A */
    for (i = 0; i < ndim; ++i) {
        int stride_entryA = out_stridesA[i];
        int stride_entryB = out_stridesB[i];
        int stride_entryC = out_stridesC[i];
        int shape_entry = out_shape[i];

        if (stride_entryA < 0) {
            dataA += stride_entryA * (shape_entry - 1);
            dataB += stride_entryB * (shape_entry - 1);
            dataC += stride_entryC * (shape_entry - 1);
            out_stridesA[i] = -stride_entryA;
            out_stridesB[i] = -stride_entryB;
            out_stridesC[i] = -stride_entryC;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_dataA = dataA;
            *out_dataB = dataB;
            *out_dataC = dataC;
            out_shape[0] = 0;
            out_stridesA[0] = 0;
            out_stridesB[0] = 0;
            out_stridesC[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
            out_stridesC[i] = out_stridesC[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                 out_stridesB[i] * out_shape[i] == out_stridesB[j] &&
                 out_stridesC[i] * out_shape[i] == out_stridesC[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
            out_stridesC[i] = out_stridesC[j];
        }
    }
    ndim = i+1;

    *out_dataA = dataA;
    *out_dataB = dataB;
    *out_dataC = dataC;
    *out_ndim = ndim;
    return 0;
}


#endif //NATIVEOPERATIONS_PAIRWISE_UTIL_H
