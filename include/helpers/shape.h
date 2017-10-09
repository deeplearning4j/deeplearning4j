/*
 * shape.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SHAPE_H_
#define SHAPE_H_

#include <cstring>
#include <cstdio>
#include "../dll.h"
#include "../nd4jmalloc.h"
#include "../templatemath.h"
#include "../helpers/logger.h"
#include "../pointercast.h"
#include "../cnpy/cnpy.h"

#define MAX_DIMENSION 0x7fffffff
#define MAX_NUM_THREADS  1024
#define MAX_RANK 32
#define MAX_COORD 3
#define PREALLOC_SIZE 33554432
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <helpers/sharedmem.h>
#endif


#ifdef __CUDACC__
#define INLINEDEF
#else
#define INLINEDEF
#endif

#include "../pairwise_util.h"
namespace shape {

/**
 * Shape information approximating
 * the information on an ndarray
 */
    struct ShapeInformation {
#ifdef __CUDACC__
        __host__ __device__
#endif
        ShapeInformation(int *shape_ = nullptr, int *stride_ = nullptr, char order_ = 0, int rank_ = 0, int offset_ = 0, int elementWiseStride_ = 0)
                : shape(shape_), stride(stride_), order(order_), rank(rank_), offset(offset_), elementWiseStride(elementWiseStride_)
        {}

        int *shape;
        int *stride;
        char order;
        int rank;
        int offset;
        int elementWiseStride;
    };

/**
 * Indexing information
 * for bounds checking
 */
    struct CurrentIndexing {
        int numElementsPerThread;
        int blockStartingIndex;
        int startingThreadIndex;
        int endingThreadIndex;

    };



#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF bool shapeEquals(int shape1Rank,int *shape1,int shape2Rank,int *shape2);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF bool shapeEquals(int *shapeInfo1,int *shapeInfo2);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF bool strideEquals(int shape1Rank,int *shape1,int shape2Rank,int *shape2);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF bool strideEquals(int *shapeInfo1,int *shapeInfo2);
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF bool strideEquals(int *stride1,int rank1,int *stride2,int rank2);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF bool equalsSoft(int *shapeA, int *shapeB);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF bool equalsStrict(int *shapeA, int *shapeB);

#ifdef __CUDACC__
__host__ __device__
#endif
    INLINEDEF void traceNew(int id);


#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int tadIndexForLinear(int linearIndex, int tadLength);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int tadLength(int *shapeInfo, int *dimension, int dimensionLength);

#ifdef __CUDACC__
    __host__
#endif
    INLINEDEF bool canReshape(const int oldRank, int* oldShape, const int newRank, int* newShape, bool isFOrder);

#ifdef __CUDACC__
    __host__
#endif
    INLINEDEF bool reshapeCF(const int oldRank, int* oldShape, const int newRank, int* newShape, bool isFOrder, int* target);
/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *shapeBuffer(int rank, int *shape);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *shapeBuffer(int rank, int *shape, int *buffer);

    /**
 * Get the shape info buffer
 * for the given rank and shape.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *shapeBufferFortran(int rank, int *shape);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *shapeBufferFortran(int rank, int *shape, int *output);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF void doPermuteShapeBuffer(int *shapeBuffer,int *rearrange, int *tmpBuffer);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF void doPermuteShapeBuffer(int rank,int *shapeBuffer,int *rearrange, int *tmpBuffer);

#ifdef __CUDACC__
    template <typename T>
    __device__ INLINEDEF int *cuMalloc(int *buffer, long size, UnifiedSharedMemory *manager);


    __device__ INLINEDEF int *cuMalloc(int *buffer, long size);
#endif



/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int * calcStridesFortran(int *shape, int rank);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int * calcStridesFortran(int *shape, int rank, int* ret);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* calcStrides(int *shape, int rank);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* calcStrides(int *shape, int rank, int* ret);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF void updateStrides(int *shape, const char order);


/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* calcStridesFortran(int *shape, int rank, int startNum);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* calcStridesFortran(int *shape, int rank, int startNum, int* ret);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* calcStrides(int *shape, int rank, int startNum);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* calcStrides(int *shape, int rank, int startNum, int* ret);

/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF ShapeInformation *shapeCopy( ShapeInformation *toCopy);


#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF bool strideDescendingCAscendingF(int *shapeBuffer);

/**
 * Compute the element wise stride
 * for a given shape/stride configuration
 * @param rank the rank of the shape/stride
 * @param shape the shape
 * @param stride the stride
 * @param isFOrder 0 or 1 for whether the array is f
 * ordered or not
 * @return -1 if there is no element wise stride the
 * element wise stride of reshape(1,length) otherwise
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int computeElementWiseStride(int rank, int *shape, int *stride, int isFOrder);

/**
 * Compute the element wise stride
 * for a given shape/stride configuration
 * @param rank the rank of the shape/stride
 * @param shape the shape
 * @param stride the stride
 * @param isFOrder 0 or 1 for whether the array is f
 * ordered or not
 * @return -1 if there is no element wise stride the
 * element wise stride of reshape(1,length) otherwise
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int computeElementWiseStride(int rank, int *shape, int *stride, int isFOrder,
                                           int *dimension, int dimensionLength);
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *shapeInfoOnlyShapeAndStride(int *shapeInfo, int *dimension, int dimensionLength,bool reverseCopyStride);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *shapeInfoOnlyShapeAndStride(int *shapeInfo, int *dimension, int dimensionLength,bool reverseCopyStride, int *buffer);
/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *doPermuteSwap(int length, int *shape, int *rearrange);



/**
 * In place permute swap
 * @param length
 * @param shape
 * @param rearrange
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF void doPermuteSwap(int length, int **shape, int *rearrange);

#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *permuteShapeBuffer(int *shapeBuffer,int *rearrange);

#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF void permuteShapeBufferInPlace(int *shapeBuffer,int *rearrange,int *out);


#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF void doPermuteShapeBuffer(int *shapeBuffer,int *rearrange);

#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF void doPermuteShapeBuffer(int rank,int *shapeBuffer,int *rearrange);
    /**
     * Rearrange the permute indexes
     * according to which  dimensions are specified.
     *
     * For example, dimension is implicitly:
     * 0,1,2
     *
     * If you want to do a reduce along dimensions 0 and 1,
     * you need to permute the indexes to be:
     * 2,0,1
     *
     * which will give us the ability to ierate along an element
     * wise stride.
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *createPermuteIndexes(int originalRank,int *dimension,int dimensionLength);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *computeResultShape(int *originalShapeBuffer,int *dimension,int dimensionLength);


/**
 * Get the ordering for the device
 * @param length
 * @param shape
 * @param stride
 * @param elementStride
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF char getOrder(int length, int *shape, int *stride, int elementStride);

/**
 * Ensure that every value in the re arrange
 * array is unique
 * @param arr
 * @param shape
 * @param arrLength
 * @param shapeLength
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int checkArrangeArray(int *arr, int arrLength, int shapeLength);

/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF void permute(ShapeInformation **info, int *rearrange, int rank);

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of cthe shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int isVector(int *shape, int rank);


    /**
     * When 1 dimension is the whole length of the
     * array
     */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int oneDimEqualToLength(int *shape, int rank);
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int oneDimEqualToLength(int *shapeInfo);
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int isVector(int *shapeInfo);
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF bool isRowVector(int *shapeInfo);

#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF bool isColumnVector(int *shapeInfo);
    /**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int isMatrix(int *shape, int rank);

#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int isMatrix(int *shapeInfo);
/**
 * Returns the shape portion of an information
 * buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int *shapeOf(int *buffer);

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int *copyOf(int length, int *toCopy);


#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *copyOf(int length, int *toCopy, int *ret);

    /**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF void copyTo(int length, int *from, int *to);
    /**
* Return a copy of a buffer.
* This buffer allocates memory
* that must be freed elsewhere.
*/
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF void copyTo(int length, int *from, int *to, int *indexes);

/**
 * Permute the given strides
 * in the given rearrange order
 * @param toPermute the buffer to permute
 * @param shapeRank the length of the buffer to permute
 * @param rearrange the rearrange order (must be 0 based indexes
 * and all must be filled in)
 * @return the rearranged array
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *permutedStrides(int *toPermute, int shapeRank, int *rearrange);

/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *slice(int *shape);
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int slices(int *shapeBuffer);

#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *sliceOfShapeBuffer(int sliceIdx,int *shapeBuffer);
/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 3
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int shapeInfoLength(int rank);

#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int shapeInfoLength(int* shapeInfo);


#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF size_t shapeInfoByteLength(int rank);


#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF size_t shapeInfoByteLength(int* shapeInfo);

/**
 * Returns the rank portion of
 * an information buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int rank( int *buffer);

/**
 * Converts a raw int buffer of the layout:
 * rank
 * shape
 * stride
 * offset
 * elementWiseStride
 *
 * where shape and stride are both straight int pointers
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    ShapeInformation *infoFromBuffer(int *buffer);

/**
 * Returns the stride portion of an information
 * buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int *stride(int *buffer);

/**
 * Compute the length of the given shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    Nd4jIndex length(int *shapeInfo);

#ifdef __CUDACC__
    __host__ __device__
#endif
    Nd4jIndex length(std::initializer_list<int>& shape);

/***
 * Returns the offset portion of an information buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int offset(int *buffer);

/**
 * Returns the ordering
 * for this shape information buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF char order(int *buffer);

/**
 * Returns the element wise stride for this information
 * buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int elementWiseStride(int *buffer);


    /**
 * Returns the element wise stride for this information
 * buffer
     * relative to a dimension and ordering for a reduction index
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int reductionIndexElementWiseStride(int *buffer, int *dimension, int dimensionLength);

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int isScalar(int *info);

/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int isScalar(volatile ShapeInformation *info);

/**
 * Return a copy of this array with the
 * given index omitted
 *
 * @param data  the data to copy
 * @param indexes the index of the item to remove
 * @param dataLength the length of the data array
 * @param indexesLength the length of the data array
 * @return the new array with the omitted
 *
 * item
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF void removeIndex(int *data, int *indexes, int dataLength, int indexesLength,
                               int *out);

    /**
 * Return a copy of this array with the
 * given index omitted
 *
 * @param data  the data to copy
 * @param indexes the index of the item to remove
 * @param dataLength the length of the data array
 * @param indexesLength the length of the data array
 * @return the new array with the omitted
 *
 * item
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int * removeIndex(int *data, int *indexes, int dataLength, int indexesLength);

    /**
     * Iterate over a given set of indexes
     * the begin and end indexes are 0 based.
     * 1 padding is automatically assumed for the ending.
     *
     * For example if you want to iterate over 0 to 4
     * it will go to 4 rather than 3.
     *
     * indexes should be the indexes to exclude
     * indexes length should be the length of indexes
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* everyIndexBut(int *indexes,int indexesLength,int begin,int end);

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
#ifdef __CUDACC__
    __device__
#endif
    INLINEDEF int tadOffset(shape::ShapeInformation *xInfo, int offset);

/**
 * Returns a shape
 * forces the given length to be 2.
 * @param shape the shape to modify
 * @param dimension the dimension (row or column)
 * for the shape to be returned as
 * @return the new shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int* ensureVectorShape(int *shape);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* createScalarShapeInfo();

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* createScalarShapeInfo(int *ret);

/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *range(int from, int to, int increment);

/**
 * Range between from and two with an
 * increment of 1
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *range(int from, int to);

/**
 * Keep the given indexes
 * in the data
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *keep(volatile int *data, int *index, int indexLength, int dataLength);

/**
 * Generate reverse copy of the data
 * @param data
 * @param length
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *reverseCopy(int *data, int length);
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF void reverseCopyTo(int *from, int *to, int length);

#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF void reverseCopyTo(int *from, int *to, int *indexes,int length);
/**
 *
 * @param arr1
 * @param arr1Length
 * @param arr2
 * @param arr2Length
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *concat(int *arr1, int arr1Length, int *arr2, int arr2Length);

/**
 *
 * @param numArrays
 * @param numTotalElements
 * @param arr
 * @param lengths
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int *concat(int numArrays, int numTotalElements, int **arr, int *lengths);

/**
 * Get the length per slice of the
 * given shape and the dimension
 * @param rank the rank of the shape
 * @param shape the shape of to get
 * the length per slice for
 * @param dimension the dimension to
 * get the length per slice for
 * @param dimensionLength the length of the dimension array
 * @return the length per slice of the given shape
 * along the given dimension
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int lengthPerSlice(int rank, int *shape, int *dimension, int dimensionLength);

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int sliceOffsetForTensor(int rank,
                                       int index,
                                       int *shape,
                                       int *tensorShape,
                                       int tensorShapeLength,
                                       int *dimension,
                                       int dimensionLength);

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int sliceOffsetForTensor(int index,int tensorLength,int lengthPerSlice2);
/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int offset(int index,
                         int rank,
                         shape::ShapeInformation *info,
                         int *dimension,
                         int dimensionLength);


/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int tensorsAlongDimension(int rank,
                                        volatile int length,
                                        volatile int *shape,
                                        int *dimension,
                                        int dimensionLength);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int tensorsAlongDimension(int *shapeInfo, int *dimension, int dimensionLength);



/**
 * Returns the tensor along dimension
 * for the given block index
 * @param blockSize
 * @param blockIdx
 * @param i
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int tadForBlockIndex(int blockSize, int blockIdx, int i);

/**
 * Computes the number of tads per block
 *
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int tadsPerBlock(int blockSize, int tads);

#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *tadShapeInfo(int index, int *xShapeInfo, int *dimension,
                                int dimensionLength);

/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *toShapeBuffer( ShapeInformation *info);

#ifdef __CUDACC__
    __host__ __device__
#endif

    INLINEDEF int *toShapeBuffer( ShapeInformation *info, int* ret);

/**
 * Returns the number of elements per thread
 */
#ifdef __CUDACC__
    __device__
#endif
    int numElementsPerThread(int N);

/**
 * Returns the block starting index
 */
#ifdef __CUDACC__
    __device__
#endif
    int blockStartingIndex(int N);

/**
 * Returns the thread starting index
 */
#ifdef __CUDACC__
    __device__
#endif
    int threadStartingIndex(int N, int stride, int offset);

/**
 * Returns the thread ending index
 */
#ifdef __CUDACC__
    __device__
#endif
    int threadEndingIndex(int N, int stride, int offset);

/**
 * Returns indexing information
 * for the current kernel invocation
 */
#ifdef __CUDACC__
    __device__
#endif
    CurrentIndexing *currentIndex(int N, int offset, int stride);

/** Given an linear index, element wise stride
 * and the length of each tad
 * map a linear index to a tad
 * @param i the index to map
 * @param the element wise stride for the tads
 * @param numElementsPerTad the number of elements
 * per tad
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    int tadIndex(int i, int elementWiseStride, int numElementsPerTad);

/**
 * Map a tad to a
 * reduction index.
 * @param tadIndexForOriginal the original tad index for the
 * split up problem (eg: split is dimension 3 mapping to a 2,3 problem)
 * @param tadsForReduced the number of tads for the shrunk down problem (eg: 2,3)
 * @param tadsForOriginal the number of tads for the smaller problem (eg: 3)
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int reductionIndexForTad(int tadIndexForOriginal, int tadsForReduced,
                             int tadsForOriginal);

/**
 * Computes the number of tads
 * per reduce index for the
 * reduction tad.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int tadsPerReduceIndex(int tadsForReduce, int tadsForOriginal);

/**
 * Maps a linear index to a reduction index
 * @param i the linear index to map
 * @param elementWiseStride the element wise stride
 * for the multiple problem
 * @param tadNum the number of tads for the shrunken problem
 * @param originalTadNum the tad number for the reduced version of the problem
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int reductionIndexForLinear(int i, int elementWiseStride, int numElementsPerTad,
                                int tadNum, int originalTadNum);

/**
 * Returns the prod of the data
 * up to the given length
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int prod(int *data, int length);
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF Nd4jIndex prodLong( int *data, int length);

    /**
     * Returns the rear most left over item not present in
     * the dimension array. This assumes that the dimension array is sorted.
     *
     * For example, given a dimension array of:
     * 0,2
     *
     * and
     *
     * 12,4,2,1 in data
     *
     * You end up with 1 (data[3])
     * since the first item won't match
     * the last item of the dimension array
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    int rearMostLeftOverItem(int *data,int length,int *dimension,int dimensionLength);

    /**
* Get an offset for retrieval
* from a data buffer
* based on the given
* shape stride and given indices
* @param baseOffset the offset to start from
* @param shape the shape of the array
* @param stride the stride of the array
* @param indices the indices to iterate over
* @return the double at the specified index
*/
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF Nd4jIndex getOffset(Nd4jIndex baseOffset,  int *shape,  int *stride,  int *indices,int rank);
#ifdef __CUDACC__
    __host__ __device__
#endif
    int* createShapeInfo(int *shape, int *stride, int rank);

#ifdef __CUDACC__
    __host__ __device__
#endif
    int* createShapeInfo(int *shape, int *stride, int rank, int *buffer);

    /**
 * Convert a linear index to
 * the equivalent nd index
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @param numIndices the number of total indices (typically prod of shape(
 * @return the mapped indexes along each dimension
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* ind2sub(int rank,  int *shape,int index,int numIndices);


#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *ind2sub(int rank,  int *shape,int index);

    /**
     * Convert a linear index to
     * the equivalent nd index
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @param numIndices the number of total indices (typically prod of shape(
     * @return the mapped indexes along each dimension
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    void  ind2sub(int rank,int *shape,int index,int numIndices,int *out);

/**
     * Convert a linear index to
     * the equivalent nd index.
     * Infers the number of indices from the specified shape.
     *
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    void ind2sub(int rank, int *shape, int index, int *out);

    /**
  * Convert a linear index to
  * the equivalent nd index
  * @param shape the shape of the dimensions
  * @param index the index to map
  * @param numIndices the number of total indices (typically prod of shape(
  * @return the mapped indexes along each dimension
  */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* ind2subC(int rank, int *shape, int index);
    /**
  * Convert a linear index to
  * the equivalent nd index
  * @param shape the shape of the dimensions
  * @param index the index to map
  * @param numIndices the number of total indices (typically prod of shape(
  * @return the mapped indexes along each dimension
  */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int* ind2subC(int rank, int *shape, int index, int numIndices);

    /**
   * Convert a linear index to
   * the equivalent nd index
   * @param shape the shape of the dimensions
   * @param index the index to map
   * @param numIndices the number of total indices (typically prod of shape(
   * @return the mapped indexes along each dimension
   */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF void  ind2subC(int rank, int *shape, int index, int numIndices, int *out);

/**
     * Convert a linear index to
     * the equivalent nd index.
     * Infers the number of indices from the specified shape.
     *
     * @param shape the shape of the dimensions
     * @param index the index to map
     * @return the mapped indexes along each dimension
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF void ind2subC(int rank, int *shape, int index, int *out);

    /**
  * Convert the given index (such as 1,1)
  * to a linear index
  * @param shape the shape of the indexes to convert
  * @param indices the index to convert
  * @return the linear index given the shape
  * and indices
  */
#ifdef __CUDACC__
    __host__ __device__
#endif
    int sub2Ind(int rank, int *shape, int *indices);

    /**
   * Compute the real linear indices for the given shape and stride
   */
#ifdef __CUDACC__
    __host__ __device__
#endif
    Nd4jIndex *computeIndices(int rank,  int *shape,  int *stride);

    /**
   * Compute the real linear indices for the
     * given shape buffer. Shape,stride and rank are derived
     * from the buffer
   */
#ifdef __CUDACC__
    __host__ __device__
#endif
    Nd4jIndex *computeIndices( int *shapeBuffer);

    /**
 * Convert a linear index to
 * the equivalent nd index
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @param numIndices the number of total indices (typically prod of shape(
 * @return the mapped indexes along each dimension
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    void  ind2subOrder(int *shapeInfo,int index,int numIndices,int *out);

    /**
 * Convert a linear index to
 * the equivalent nd index
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @param numIndices the number of total indices (typically prod of shape(
 * @return the mapped indexes along each dimension
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    void  ind2subOrder(int *shapeInfo,int index,int *out);


#ifdef __CUDACC__
    __host__ __device__
#endif
    void printShapeInfo(int *shapeInfo);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void printShapeInfoLinear(int *shapeInfo);

#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF void printIntArray(int *arr,int length);

#ifdef __CUDACC__
    __host__ __device__
#endif
    void printArray(float *arr,int length);


#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *shapeBufferOfNpy(int rank, unsigned int *shape,bool fortranOrder);


#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *shapeBufferOfNpy(cnpy::NpyArray arr);




#ifdef __CUDACC__
    __host__ __device__
#endif
    INLINEDEF int *shapeBufferOfNpyBuffer(char *buffer);


#ifdef __CUDACC__
    __host__
#endif
    // this function checks the consistence of dimensions with array rank (negative dimensions, too large dimensions, too big number of dimensions)
    // also sort input array of dimensions, this operation is also necessary for creating TAD object
    INLINEDEF void checkDimensions(const int rank, std::vector<int>& dimensions);

//END HEADERS


    //BEGIN IMPLEMENTATIONS

}





#endif /* SHAPE_H_ */