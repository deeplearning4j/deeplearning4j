/*
 * shape.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SHAPE_H_
#define SHAPE_H_


namespace shape {
    const int MAX_DIMENSION = 0x7fffffff;
    const int MAX_NUM_THREADS = 1024;

/**
 * Shape information approximating
 * the information on an ndarray
 */
    typedef struct {
        int *shape;
        int *stride;
        char order;
        int rank;
        int offset;
        int elementWiseStride;
    } ShapeInformation;


/**
 * Indexing information
 * for bounds checking
 */
    typedef struct {
        int numElementsPerThread;
        int blockStartingIndex;
        int startingThreadIndex;
        int endingThreadIndex;

    } CurrentIndexing;


/**
 * TADPermuteInfo is for intermediate information
 * needed for computing tensor along dimension.
 *
 *
 */
    typedef struct {
        int *tensorShape;
        int xRank;
        int *reverseDimensions;
        int *rangeRet;
        int removeLength;
        int *remove;
        int *zeroDimension;
        int *newPermuteDims;
        int *permutedShape;
        int *permutedStrides;
        int tensorShapeLength;
        int tensorShapeProd;
    } TADPermuteInfo;


/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
    __device__ __host__

    ShapeInformation *shapeCopy(ShapeInformation *toCopy);

/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
    __device__ __host__

    int *doPermuteSwap(int length, int *shape, int *rearrange);


/**
 * Get the ordering for the device
 * @param length
 * @param shape
 * @param stride
 * @param elementStride
 * @return
 */
    __device__ __host__

    char getOrder(int length, int *shape, int *stride, int elementStride);

/**
 * Ensure that every value in the re arrange
 * array is unique
 * @param arr
 * @param shape
 * @param arrLength
 * @param shapeLength
 * @return
 */
    __device__ __host__

    int checkArrangeArray(int *arr, int *shape, int arrLength, int shapeLength);

/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
    __device__ __host__

    void permute(ShapeInformation **info, int *rearrange, int rank);

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
    __device__ __host__

    int isVector(int *shape, int rank);


/**
 * Returns the shape portion of an information
 * buffer
 */
    __device__ __host__

    int *shapeOf(int *buffer);


/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
    __device__ __host__

    int *copyOf(int length, int *toCopy);

/**
 * Permute the given strides
 * in the given rearrange order
 * @param toPermute the buffer to permute
 * @param shapeRank the length of the buffer to permute
 * @param rearrange the rearrange order (must be 0 based indexes
 * and all must be filled in)
 * @return the rearranged array
 */
    __device__ __host__

    int *permutedStrides(int *toPermute, int shapeRank, int *rearrange);


/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
    __device__ __host__

    int *slice(int *shape);

/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 3
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
    __device__ __host__

    int shapeInfoLength(int rank);


/**
 * Returns the rank portion of
 * an information buffer
 */
    __device__ __host__

    int rank(int *buffer);


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
    __device__ __host__

    ShapeInformation *infoFromBuffer(int *buffer);

/**
 * Returns the stride portion of an information
 * buffer
 */
    __device__ __host__

    int *stride(int *buffer);


/**
 * Compute the length of the given shape
 */
    __device__ __host__

    int length(int *shapeInfo);

/***
 * Returns the offset portion of an information buffer
 */
    __device__ __host__

    int offset(int *buffer);


/**
 * Returns the ordering
 * for this shape information buffer
 */
    __device__ __host__

    char order(int *buffer);


/**
 * Returns the element wise stride for this information
 * buffer
 */
    __device__ __host__

    int elementWiseStride(int *buffer);

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
    __device__ __host__

    int isScalar(int *info);


/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
    __device__ __host__

    int isScalar(volatile ShapeInformation *info);


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
    __device__ __host__

    void removeIndex(int *data, int *indexes, int dataLength, int indexesLength, int **out);


/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
    __device__ int tadOffset(shape::ShapeInformation *xInfo, int offset);

/**
 * Returns a shape
 * forces the given length to be 2.
 * @param shape the shape to modify
 * @param dimension the dimension (row or column)
 * for the shape to be returned as
 * @return the new shape
 */
    __device__ __host__

    int *ensureVectorShape(int *shape, int dimension);


/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
    __device__ __host__

    int *range(int from, int to, int increment);

/**
 * Range between from and two with an
 * increment of 1
 */
    __device__ __host__

    int *range(int from, int to);


/**
 * Keep the given indexes
 * in the data
 */
    __device__ __host__

    int *keep(volatile int *data, int *index, int indexLength, int dataLength);


/**
 * Generate reverse copy of the data
 * @param data
 * @param length
 * @return
 */
    __device__ __host__

    int *reverseCopy(int *data, int length);


/**
 *
 * @param arr1
 * @param arr1Length
 * @param arr2
 * @param arr2Length
 * @return
 */
    __device__ __host__

    int *concat(int *arr1, int arr1Length, int *arr2, int arr2Length);


/**
 *
 * @param numArrays
 * @param numTotalElements
 * @param arr
 * @param lengths
 * @return
 */
    __device__ __host__

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
    __device__ __host__

    int lengthPerSlice(int rank, int *shape, int *dimension, int dimensionLength);

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
    __device__ __host__

    int sliceOffsetForTensor(int rank, int index, int *shape, int *tensorShape, int tensorShapeLength, int *dimension,
                             int dimensionLength);

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
    __device__ int tadOffset(int *xInfo, int offset);

/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
    __device__ __host__

    int offset(int index, int rank, shape::ShapeInformation *info, int *dimension, int dimensionLength);


/**
 * Given the shape information and dimensions
 * returns common information
 * needed for tensor along dimension
 * calculations
 */
    __device__ __host__

    TADPermuteInfo tadInfo(int *xShapeInfo, int *dimension, int dimensionLength);


/**
 * Frees the permute information
 * @param info the info to free
 */
    __host__ __device__

    void freePermuteInfo(TADPermuteInfo info);


/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
    __device__ __host__

    int tensorsAlongDimension(int rank, volatile int length, volatile int *shape, int *dimension,
                              int dimensionLength);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
    __device__ __host__

    int tensorsAlongDimension(int *shapeInfo, int *dimension, int dimensionLength);


/**
 *
 * @param info
 * @param dimension
 * @param dimensionLength
 * @return
 */
    __device__ __host__

    int tensorsAlongDimension(TADPermuteInfo info, int *dimension, int dimensionLength);

/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
    __device__ __host__

    int offset(int index, int *xShapeInfo, int *dimension, int dimensionLength, TADPermuteInfo info);

/**
 * Returns the tensor along dimension
 * for the given block index
 * @param blockSize
 * @param blockIdx
 * @param i
 * @return
 */
    __device__ __host__

    int tadForBlockIndex(int blockSize, int blockIdx, int i);


/**
 * Computes the number of tads per block
 *
 */
    __device__ __host__

    int tadsPerBlock(int blockSize, int tads);


/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
    __device__ __host__

    int *toShapeBuffer(ShapeInformation *info);


/**

 * Returns the number of elements per thread
 */
    __device__ int numElementsPerThread(int N);

/**
 * Returns the block starting index
 */
    __device__ int blockStartingIndex(int N);


/**
 * Returns the thread starting index
 */
    __device__ int threadStartingIndex(int N, int stride, int offset);


/**
 * Returns the thread ending index
 */
    __device__ int threadEndingIndex(int N, int stride, int offset);


/**
 * Returns indexing information
 * for the current kernel invocation
 */
    __device__ CurrentIndexing
    *

    currentIndex(int N, int offset, int stride);


/** Given an linear index, element wise stride
 * and the length of each tad
 * map a linear index to a tad
 * @param i the index to map
 * @param the element wise stride for the tads
 * @param numElementsPerTad the number of elements
 * per tad
 */
    __device__ __host__

    int tadIndex(int i, int elementWiseStride, int numElementsPerTad);

/**
 * Map a tad to a
 * reduction index.
 * @param tadIndexForOriginal the original tad index for the
 * split up problem (eg: split is dimension 3 mapping to a 2,3 problem)
 * @param tadsForReduced the number of tads for the shrunk down problem (eg: 2,3)
 * @param tadsForOriginal the number of tads for the smaller problem (eg: 3)
 */
    __device__ __host__

    int reductionIndexForTad(int tadIndexForOriginal, int tadsForReduced, int tadsForOriginal);

/**
 * Computes the number of tads
 * per reduce index for the
 * reduction tad.
 */
    __device__ __host__

    int tadsPerReduceIndex(int tadsForReduce, int tadsForOriginal);

/**
 * Maps a linear index to a reduction index
 * @param i the linear index to map
 * @param elementWiseStride the element wise stride
 * for the multiple problem
 * @param tadNum the number of tads for the shrunken problem
 * @param originalTadNum the tad number for the reduced version of the problem
 */
    __device__ __host__

    int reductionIndexForLinear(
            int i, int elementWiseStride, int numElementsPerTad, int tadNum, int originalTadNum);


/**
 * Returns the prod of the data
 * up to the given length
 */
    __device__ __host__

    int prod(int *data, int length);


}

#endif /* SHAPE_H_ */
