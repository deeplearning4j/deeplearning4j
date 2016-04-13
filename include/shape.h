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
#include <dll.h>
#include <nd4jmalloc.h>
#include <templatemath.h>

#include "pointercast.h"
#define MAX_DIMENSION 0x7fffffff
#define MAX_NUM_THREADS  1024
#define MAX_RANK 32
#define PREALLOC_SIZE 5242880

namespace shape {

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
    class TADPermuteInfo {
    public:
        int *tensorShape = NULL;
        int xRank = 0;
        int *reverseDimensions = NULL;
        int *rangeRet = NULL;
        int removeLength;
        int *remove = NULL;
        int *zeroDimension = NULL;
        int *newPermuteDims = NULL;
        int *permutedShape = NULL;
        int *permutedStrides = NULL;
        int tensorShapeLength = 0;
        int tensorShapeProd = 0;
    };

#ifdef __CUDACC__
    __host__ __device__
#endif
    inline bool shapeEquals(int shape1Rank,int *shape1,int shape2Rank,int *shape2);



#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int tadIndexForLinear(int linearIndex, int tadLength);
/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int *shapeBuffer(int rank, int *shape);
    /**
 * Get the shape info buffer
 * for the given rank and shape.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int *shapeBufferFortran(int rank, int *shape);


#ifdef __CUDACC__
    __inline__ __device__ int *cuMalloc(int *buffer, long size);
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
    inline int * calcStridesFortran(int *shape, int rank);

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
    inline int* calcStrides(int *shape, int rank);

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
    inline int* calcStridesFortran(int *shape, int rank, int startNum);

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
    inline int* calcStrides(int *shape, int rank, int startNum);

/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline ShapeInformation *shapeCopy(const ShapeInformation *toCopy);

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
    inline int computeElementWiseStride(int rank, int *shape, int *stride, int isFOrder);

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
    inline int computeElementWiseStride(int rank, int *shape, int *stride, int isFOrder,
                                        int *dimension, int dimensionLength);
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int *shapeInfoOnlyShapeAndStride(int *shapeInfo, int *dimension, int dimensionLength,bool reverseCopyStride);
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

    inline int *doPermuteSwap(int length, int *shape, int *rearrange);




#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int* squeezeDimensions(int *shapeInfo, int* dimension, int dimensionLength, bool *squeezedRef, bool *squeezeDimensionsRef,int wholeRank,int numOnes);

/**
 * In place permute swap
 * @param length
 * @param shape
 * @param rearrange
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void doPermuteSwap(int length, int **shape, int *rearrange);

#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *permuteShapeBuffer(int *shapeBuffer,int *rearrange);
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void doPermuteShapeBuffer(int **shapeBuffer,int *rearrange);

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
    inline int *createPermuteIndexes(int originalRank,int *dimension,int dimensionLength);


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

    inline char getOrder(int length, int *shape, int *stride, int elementStride);

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

    inline int checkArrangeArray(int *arr, int arrLength, int shapeLength);

/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void permute(ShapeInformation **info, int *rearrange, int rank);

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of cthe shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int isVector(int *shape, int rank);
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int isVector(int *shapeInfo);

    /**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int isMatrix(const int *shape, int rank);

#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int isMatrix(const int *shapeInfo);
/**
 * Returns the shape portion of an information
 * buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    const int* shapeOf(const int *buffer);
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

    /**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void copyTo(int length, int *from, int *to);
    /**
* Return a copy of a buffer.
* This buffer allocates memory
* that must be freed elsewhere.
*/
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline void copyTo(int length, int *from, int *to, int *indexes);

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

    inline int *permutedStrides(int *toPermute, int shapeRank, int *rearrange);

/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *slice(int *shape);

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

    inline int shapeInfoLength(int rank);

/**
 * Returns the rank portion of
 * an information buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int rank(const int *buffer);

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

    const int *stride(const int *buffer);
    int *stride(int *buffer);

/**
 * Compute the length of the given shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    int length(int *shapeInfo);

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

    inline char order(int *buffer);

/**
 * Returns the element wise stride for this information
 * buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int elementWiseStride(int *buffer);


    /**
 * Returns the element wise stride for this information
 * buffer
     * relative to a dimension and ordering for a reduction index
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int reductionIndexElementWiseStride(int *buffer, int *dimension, int dimensionLength);

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int isScalar(int *info);

/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int isScalar(volatile ShapeInformation *info);

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

    inline void removeIndex(int *data, int *indexes, int dataLength, int indexesLength,
                            int *out);

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
    inline int* everyIndexBut(int *indexes,int indexesLength,int begin,int end);

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
#ifdef __CUDACC__
    __device__
#endif
    inline int tadOffset(shape::ShapeInformation *xInfo, int offset);

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

    inline int* ensureVectorShape(int *shape);

#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int* createScalarShapeInfo();

    /**
     * Generate a coordinate tuple
     * from the given tad dimension specifications.
     * For example given a shape of:
     * 2,3,2,2
     *
     * Dimension of:
     * 0,2
     *
     * The output from this function will be (given the index):
     * 0 -> (0,0,0,0)
     * 1 -> (0,0,0,1)
     * 2 -> (0,1,0,0)
     * 3 -> (0,1,0,1)
     * 4 -> (0,2,0,0)
     * 5 -> (0,2,0,1)
     *
     *
     * We called the specified dimensions
     * in dimension the hold out dimensions.
     *
     * The hold out dimensions are the indexes of the coordinate
     * tuple that will be frozen at zero while the goal will be to
     * iterate over the possible values (relative to the index)
     * to generate the appropriate coordinate tuple
     * given a tensor along dimension, index, and shape information
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int* tad2Sub(int index, int *dimension, int dimensionLength, int *shapeInfo);


/**l
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *range(int from, int to, int increment);

/**
 * Range between from and two with an
 * increment of 1
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *range(int from, int to);

/**
 * Keep the given indexes
 * in the data
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *keep(volatile int *data, int *index, int indexLength, int dataLength);

/**
 * Generate reverse copy of the data
 * @param data
 * @param length
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int *reverseCopy(int *data, int length);
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void reverseCopyTo(int *from, int *to, int length);

#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void reverseCopyTo(int *from, int *to, int *indexes,int length);
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

    inline int *concat(int *arr1, int arr1Length, int *arr2, int arr2Length);

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

    inline int lengthPerSlice(int rank, int *shape, int *dimension, int dimensionLength);

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

    inline int sliceOffsetForTensor(int rank, int index, int *shape, int *tensorShape,
                                    int tensorShapeLength, int *dimension, int dimensionLength);

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
#ifdef __CUDACC__
    __device__
#endif
    inline int tadOffset(int *xInfo, int offset);

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

    inline int offset(int index, int rank, shape::ShapeInformation *info, int *dimension,
                      int dimensionLength);

/**
 * Given the shape information and dimensions
 * returns common information
 * needed for tensor along dimension
 * calculations
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline TADPermuteInfo tadInfo(int *xShapeInfo, int *dimension, int dimensionLength);

/**
 * Frees the permute information
 * @param info the info to free
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void freePermuteInfo(TADPermuteInfo info);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int tensorsAlongDimension(int rank, volatile int length, volatile int *shape,
                                     int *dimension, int dimensionLength);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int tensorsAlongDimension(int *shapeInfo, int *dimension, int dimensionLength);

/**
 *
 * @param info
 * @param dimension
 * @param dimensionLength
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int tensorsAlongDimension(TADPermuteInfo info);

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

    inline int offset(int index, int *xShapeInfo,int *dimension, int dimensionLength,
                      TADPermuteInfo info);

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

    inline int tadForBlockIndex(int blockSize, int blockIdx, int i);

/**
 * Computes the number of tads per block
 *
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int tadsPerBlock(int blockSize, int tads);

#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *tadShapeInfo(int index, int *xShapeInfo, int *dimension,
                             int dimensionLength);

/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *toShapeBuffer(const ShapeInformation *info);

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
    inline int prodLong(const int *data, int length);

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
    inline int getOffset(int baseOffset, const int *shape, const int *stride, const int *indices,int rank);
#ifdef __CUDACC__
    __host__ __device__
#endif
    int* createShapeInfo(int *shape, int *stride, int rank);

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
    inline int* ind2sub(int rank, const int *shape,int index,int numIndices);


#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int *ind2sub(int rank, const int *shape,int index);

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
    inline int* ind2subC(int rank, int *shape, int index);
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
    inline int* ind2subC(int rank, int *shape, int index, int numIndices);

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
    inline void  ind2subC(int rank, int *shape, int index, int numIndices, int *out);

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
    inline void ind2subC(int rank, int *shape, int index, int *out);

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
    Nd4jIndex *computeIndices(int rank, const int *shape, const int *stride);

    /**
   * Compute the real linear indices for the
     * given shape buffer. Shape,stride and rank are derived
     * from the buffer
   */
#ifdef __CUDACC__
    __host__ __device__
#endif
    Nd4jIndex *computeIndices(const int *shapeBuffer);

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

    /**
   * Tad element wise stride:
   * given the inner most dimension (the sorted dimension of the last)
   * the element wise stride of the tad (disregarding order) is the
   * last dimension's stride.
   *
   * For a given singular dimension this will just be the only entry.
   * For example, given the following c order shape/stride:
   * 2,2,3,2
   * 12,6,2,1
   *
   * The tad element wise stride for 3 will be 1.
   * For zero it wil be 12
   *
   * For 2,3 it's 1
   *
   * Note here that the multi dimensional 2,3 case
   * is equivalent to the singular 3 case.
   *
   *
   * Note that this is for the dimension that ultimately
   * ends up removed.
   *
   * Again: this may not preserve ordering of the tad
   * but maybe used for reductions.
   */
#ifdef __CUDACC__
    __host__ __device__
#endif
    int tadElementWiseStride(int *shapeInfo, int *dimension, int dimensionLength);

    /**
     * Length of a tad given
     * the shape information
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    int tadLength(int *shapeInfo, int *dimension, int dimensionLength);
    /**
     * Compute the tad offset given a dimension.
     *
     * The general pattern for computing a tad offset is as follows:
     * Every $STRIDE that was removed (the first dimension)
     * do a jump by the major stride of the parent array
     * (stride[0] of the parent array)
     *
     * For example given a c ordered 2,2,3,2 with stride 12,6,2,1
     * A tad of dimension 1 will jump 12 every 6 tads.
     *
     * You then end up with offsets of:
     * 0
     * 1
     * 2
     * 3
     * 4
     * 5
     * 12
     * 13
     * 14
     * 15
     * 16
     * 17
     *
     * notice there are 12 tads here. This same incremental jump will happen
     * every time.
     * Note here that by default the
     * stride of element wise stride is used for the hops.
     *
     * Sometimes a jump doesn't happen. If there are less tads
     * than the stride of the dimension you removed, the
     * element wise stride will always be used.
     *
     * For example in a dimension of 0,1, you end up with offsets of:
     * 0,1,2,3,4,5
     *
     * Given that the inner most stride of the dimensions that was removed (1)
     * had a stride of 6, we never need to do a major stride jump.
     *
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    int tadOffset(int index, int *shapeInfo, int *dimension, int dimensionLength);


#ifdef __CUDACC__
    __inline__ __device__ int *cuMalloc(int *buffer, long size) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid * size > PREALLOC_SIZE - size) {
            return (int *) malloc(size);
        } else {
            int *ret = buffer;
            ret += (tid * size);
            return ret;
        }
    }
#endif

    /**
   * Length of a tad given
   * the shape information
   */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int tadLength(int *shapeInfo, int *dimension, int dimensionLength) {
        int *shapeTwo = shape::shapeOf(shapeInfo);
        int rank = shape::rank(shapeInfo);
        if(dimensionLength == 1) {
            return shapeTwo[dimension[0]];
        }
        else {
            int ret = 1;
            for(int i = 0; i < rank; i++) {
                for(int j = 0; j < dimensionLength; j++) {
                    if(i == dimension[j])
                        ret *= shapeTwo[dimension[j]];
                }
            }
            return ret;

        }


    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int * tad2Sub(int index, int *dimension, int dimensionLength, int *shapeInfo) {
        int *shape = shape::shapeOf(shapeInfo);
        int rank = shape::rank(shapeInfo);
        int leftOverIndexLen = rank - dimensionLength;
        int *ret = new int[rank];
        //shape of the tad
        int *tadShape = new int[leftOverIndexLen];
        //indexes not specified in the tad indexes
        int *leftOverIndexes = new int[leftOverIndexLen];
        //every coordinate starts as zero
        for(int i = 0; i < rank; i++)
            ret[i] = 0;


        //find the length of the elements we
        //are iterating over
        int len = 1;
        //left over index cursor for initializing elements
        int leftOverIndex = 0;
        for(int i = 0; i < rank; i++) {
            //look for dimensions NOT found in dimension length (basically compute shape - dimension (set difference)
            bool found = false;
            for(int j = 0; j < dimensionLength; j++) {
                //skip over specified dimensions when computing left over length
                if(i == dimension[j]) {
                    found = true;
                    break;
                }

            }

            //add to the indexes that aren't specified as part of the tad dimension
            //indexes
            if(!found) {
                //accumulate the list of indexes left over used for initializing the return value
                leftOverIndexes[leftOverIndex] = i;
                //accumulate the tad shape
                tadShape[leftOverIndex] = shape[i];
                //accumulate the length (product) of the indexes that will be iterated over
                len *= shape[i];
                leftOverIndex++;

            }
        }


        //sub for indices
        int *sub = shape::ind2subC(leftOverIndexLen,tadShape,index,len);


        for(int i = 0; i < leftOverIndexLen; i++) {
            ret[leftOverIndexes[i]] = sub[i];
        }



        delete[] tadShape;
        delete[] leftOverIndexes;
        delete[] sub;



        return  ret;

    }

    /**
     * Tad element wise stride:
     * given the inner most dimension (the sorted dimension of the last)
     * the element wise stride of the tad (disregarding order) is the
     * last dimension's stride.
     *
     * For a given singular dimension this will just be the only entry.
     * For example, given the following c order shape/stride:
     * 2,2,3,2
     * 12,6,2,1
     *
     * The tad element wise stride for 3 will be 1.
     * For zero it wil be 12
     *
     * For 2,3 it's 1
     *
     * Note here that the multi dimensional 2,3 case
     * is equivalent to the singular 3 case.
     *
     *
     * Note that this is for the dimension that ultimately
     * ends up removed.
     *
     * Again: this may not preserve ordering of the tad
     * but maybe used for reductions.
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int tadElementWiseStride(int *shapeInfo,int *dimension,int dimensionLength) {
        return reductionIndexElementWiseStride(shapeInfo,dimension,dimensionLength);
    }


    /**
    * Compute the tad offset given a dimension.
    *
    * The general pattern for computing a tad offset is as follows:
    * Every $STRIDE that was removed (the first dimension)
    * do a jump by the major stride of the parent array
    * (stride[0] of the parent array)
    *
    * For example given a c ordered 2,2,3,2 with stride 12,6,2,1
    * A tad of dimension 1 will jump 12 every 6 tads.
    *
    * You then end up with offsets of:
    * 0
    * 1
    * 2
    * 3
    * 4
    * 5
    * 12
    * 13
    * 14
    * 15
    * 16
    * 17
    *
    * notice there are 12 tads here. This same incremental jump will happen
    * every time.
    * Note here that by default the
    * stride of element wise stride is used for the hops.
    *
    * Sometimes a jump doesn't happen. If there are less tads
    * than the stride of the dimension you removed, the
    * element wise stride will always be used.
    *
    * For example in a dimension of 0,1, you end up with offsets of:
    * 0,1,2,3,4,5
    *
    * Given that the inner most stride of the dimensions that was removed (1)
    * had a stride of 6, we never need to do a major stride jump.
    *
    */
    inline int tadOffset(int index, int *shapeInfo, int *dimension, int dimensionLength) {
        if(dimensionLength > 1) {
            int *tad2Sub = shape::tad2Sub(index,dimension,dimensionLength,shapeInfo);
            int rank = shape::rank(shapeInfo);
            int *shape = shape::shapeOf(shapeInfo);
            int *stride = shape::stride(shapeInfo);
            int ret = shape::getOffset(0,shape,stride,tad2Sub,rank);
            delete[] tad2Sub;
            return ret;

        }
        else {
            int *tad2Sub = shape::tad2Sub(index,dimension,dimensionLength,shapeInfo);
            int rank = shape::rank(shapeInfo);
            int *shape = shape::shapeOf(shapeInfo);
            int *stride = shape::stride(shapeInfo);
            int ret = shape::getOffset(0,shape,stride,tad2Sub,rank);
            delete[] tad2Sub;
            return ret;
        }



    }


#ifdef __CUDACC__
    __host__ __device__
#endif
    inline bool shapeEquals(int shape1Rank,int *shape1,int shape2Rank,int *shape2) {
        if(shape1Rank != shape2Rank)
            return false;
        //rank not equals
        for(int i = 0; i < shape1Rank; i++) {
            if(shape1[i] != shape2[i])
                return false;
        }

        return true;
    }


#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int *shapeInfoOnlyShapeAndStride(int *shapeInfo, int *dimension, int dimensionLength,bool reverseCopyStride) {
        int *theShape = shape::shapeOf(shapeInfo);
        int *theStride = shape::stride(shapeInfo);
        int rank = dimensionLength == 1 ? 2 : dimensionLength;
        int *ret = new int[shape::shapeInfoLength(rank)];
        //set the rank
        ret[0] = rank;
        int *retShape = shape::shapeOf(ret);
        int *retStride = shape::stride(ret);
        int len = rank;
        if(dimensionLength == 1) {
            if(shape::isMatrix(theShape,shape::rank(shapeInfo))) {
                if(dimension[0] == 0) {
                    int newStride[2] = {theStride[dimension[0]],1};
                    int newShape[2] = {theShape[dimension[0]],1};
                    retShape[0] = newShape[0];
                    retShape[1] = newShape[1];
                    retStride[0] = newStride[0];
                    retStride[1] = newStride[1];
                }
                else {
                    int newStride[2] = {theStride[dimension[0]],1};
                    int newShape[2] = {theShape[dimension[0]],1};
                    retShape[0] = newShape[0];
                    retShape[1] = newShape[1];
                    retStride[0] = newStride[0];
                    retStride[1] = newStride[1];
                }
            }
            else {
                int newStride[2] = {1,theStride[dimension[0]]};
                int newShape[2] = {1,theShape[dimension[0]]};
                retShape[0] = newShape[0];
                retShape[1] = newShape[1];
                retStride[0] = newStride[0];
                retStride[1] = newStride[1];
            }



        }
        else {
            int *newIndexes = dimension;
            if(reverseCopyStride)
                shape::reverseCopyTo(theStride, retStride, newIndexes, len);
            else
                shape::copyTo(len, theStride, retStride, newIndexes);
            shape::copyTo(len, theShape, retShape, newIndexes);

        }


        ret[shape::shapeInfoLength(rank) - 1] = shape::order(shapeInfo);
        return ret;
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int * createShapeInfo(int *shape, int *stride, int rank) {
        int *ret = new int[shape::shapeInfoLength(rank)];
        ret[0] = rank;
        int *retShape = shape::shapeOf(ret);
        int *retStride = shape::stride(ret);
        for(int i = 0;i < rank; i++) {
            retShape[i] = shape[i];
            retStride[i] = stride[i];
        }

        return ret;
    }

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
    inline int * calcStridesFortran(int *shape, int rank, int startNum) {
        if (isVector(shape, rank)) {
            int *ret = new int[2];
            for (int i = 0; i < 2; i++)
                ret[i] = 1;
            return ret;

        }

        int dimensions = rank;
        int *stride = new int[dimensions];
        int st = startNum;
        for (int j = 0; j < rank; j++) {
            stride[j] = st;
            st *= shape[j];
        }

        return stride;
    }

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
    inline int * calcStrides(int *shape, int rank, int startNum) {
        int *stride = new int[rank];

        if (shape::isVector(shape, rank)) {
            for (int i = 0; i < 2; i++)
                stride[i] = 1;
            return stride;

        }

        int st = startNum;
        for (int j = rank - 1; j >= 0; j--) {
            stride[j] = st;
            st *= shape[j];
        }

        return stride;
    }

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
    inline int * calcStridesFortran(int *shape, int rank) {
        return calcStridesFortran(shape, rank, 1);
    }

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
    inline int* calcStrides(int *shape, int rank) {
        return calcStrides(shape, rank, 1);
    }

/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline ShapeInformation *shapeCopy(const ShapeInformation *toCopy) {
        ShapeInformation *copy = new ShapeInformation;

        copy->shape = new int[toCopy->rank];

        memcpy(copy->shape, toCopy->shape, toCopy->rank * sizeof(int));

        copy->stride = new int[toCopy->rank];
        for (int i = 0; i < toCopy->rank; i++) {
            copy->stride[i] = toCopy->stride[i];
        }
        copy->order = toCopy->order;
        copy->rank = toCopy->rank;
        copy->offset = toCopy->offset;
        copy->elementWiseStride = toCopy->elementWiseStride;
        return copy;
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int computeElementWiseStride(int rank, int *shape, int *stride, int isFOrder) {
        int oldnd;
        int *olddims = shape::copyOf(rank, shape);
        int *oldstrides = shape::copyOf(rank, stride);
        int np, op, last_stride;
        int oi, oj, ok, ni, nj, nk;
        int *newStrides = new int[rank];
        oldnd = 0;
        //set the shape to be 1 x length
        int newShapeRank = 2;
        int *newShape = new int[newShapeRank];
        newShape[0] = 1;
        newShape[1] = shape::prodLong(shape, rank);

        /*
         * Remove axes with dimension 1 from the old array. They have no effect
         * but would need special cases since their strides do not matter.
         */
        for (oi = 0; oi < rank; oi++) {
            if (shape[oi] != 1) {
                olddims[oldnd] = shape[oi];
                oldstrides[oldnd] = stride[oi];
                oldnd++;
            }
        }

        np = 1;
        for (ni = 0; ni < newShapeRank; ni++) {
            np *= newShape[ni];
        }
        op = 1;
        for (oi = 0; oi < oldnd; oi++) {
            op *= olddims[oi];
        }
        if (np != op) {
            /* different total sizes; no hope */
            return -1;
        }

        if (np == 0) {
            /* the current code does not handle 0-sized arrays, so give up */
            return -1;
        }

        /* oi to oj and ni to nj give the axis ranges currently worked with */
        oi = 0;
        oj = 1;
        ni = 0;
        nj = 1;
        while (ni < newShapeRank && oi < oldnd) {
            np = newShape[ni];
            op = olddims[oi];

            while (np != op) {
                if (np < op) {
                    /* Misses trailing 1s, these are handled later */
                    np *= newShape[nj++];
                } else {
                    op *= olddims[oj++];
                }
            }

            /* Check whether the original axes can be combined */
            for (ok = oi; ok < oj - 1; ok++) {
                if (isFOrder) {
                    if (oldstrides[ok + 1] != olddims[ok] * oldstrides[ok]) {
                        /* not contiguous enough */
                        return -1;
                    }
                } else {
                    /* C order */
                    if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
                        /* not contiguous enough */
                        return -1;
                    }
                }
            }

            /* Calculate new strides for all axes currently worked with */
            if (isFOrder) {
                newStrides[ni] = oldstrides[oi];
                for (nk = ni + 1; nk < nj; nk++) {
                    newStrides[nk] = newStrides[nk - 1] * newShape[nk - 1];
                }
            } else {
                /* C order */
                newStrides[nj - 1] = oldstrides[oj - 1];
                for (nk = nj - 1; nk > ni; nk--) {
                    newStrides[nk - 1] = newStrides[nk] * newShape[nk];
                }
            }
            ni = nj++;
            oi = oj++;
        }

        /*
         * Set strides corresponding to trailing 1s of the new shape.
         */
        if (ni >= 1) {
            last_stride = newStrides[ni - 1];
        } else {
            last_stride = stride[rank - 1];
        }
        if (isFOrder) {
            if (ni >= 1)
                last_stride *= newShape[ni - 1];
        }
        for (nk = ni; nk < newShapeRank; nk++) {
            newStrides[nk] = last_stride;
        }
        //returns the last element of the new stride array
        int ret = last_stride;
        delete[] newStrides;
        delete[] newShape;
        delete[] oldstrides;
        delete[] olddims;
        return ret;
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int computeElementWiseStride(int rank, int *shape, int *stride, int isFOrder,
                                        int *dimension, int dimensionLength) {
        if(dimensionLength == 1) {
            return stride[dimension[0]];
        }
        return -1;

    }

/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int *shapeBuffer(int rank, int *shape) {
        int *stride = shape::calcStrides(shape, rank);
        shape::ShapeInformation * shapeInfo = new shape::ShapeInformation();
        shapeInfo->shape = shape;
        shapeInfo->stride = stride;
        shapeInfo->offset = 0;
        shapeInfo->rank = rank;
        int elementWiseStride = shape::computeElementWiseStride(rank, shape, stride,
                                                                0);
        shapeInfo->order = 'c';
        shapeInfo->elementWiseStride = elementWiseStride;
        int *shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
        delete shapeInfo;
        return shapeInfoBuffer;
    }

    /**
 * Get the shape info buffer
 * for the given rank and shape.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int *shapeBufferFortran(int rank, int *shape) {
        int *stride = shape::calcStridesFortran(shape,rank);
        shape::ShapeInformation * shapeInfo = new shape::ShapeInformation();
        shapeInfo->shape = shape;
        shapeInfo->stride = stride;
        shapeInfo->offset = 0;
        shapeInfo->rank = rank;
        int elementWiseStride = shape::computeElementWiseStride(rank, shape, stride,
                                                                0);
        shapeInfo->order = 'f';
        shapeInfo->elementWiseStride = elementWiseStride;
        int *shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
        delete shapeInfo;
        return shapeInfoBuffer;
    }


    /**
     * Compute the real linear indices for the given shape and stride
     */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline Nd4jIndex *computeIndices(int rank, const int *shape, const int *stride) {
        int length = shape::prodLong(shape,rank);
        Nd4jIndex *ret = new Nd4jIndex[length];
        for(int i = 0; i < length; i++) {
            int *idx = shape::ind2sub(rank, shape, i);
            ret[i] = shape::getOffset(0, shape, stride, idx, rank);
            delete[] idx;
        }

        return ret;
    }

    /**
  * Compute the real linear indices for the given shape and stride
  */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline Nd4jIndex *computeIndices(const int *shapeBuffer) {
        return computeIndices(shape::rank(shapeBuffer),shape::shapeOf(shapeBuffer),shape::stride(shapeBuffer));
    }


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
    inline int sub2Ind(int rank, int *shape, int *indices) {
        int index = 0;
        int shift = 1;

        for(int i = 0; i < rank; i++) {
            index += shift * indices[i];
            shift *= shape[i];
        }
        return index;
    }

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
    inline int* ind2sub(int rank, const int *shape, int index,int numIndices) {
        int denom = numIndices;
        int *ret = new int[rank];

        for(int i = rank - 1; i >= 0; i--) {
            denom /= shape[i];
            ret[i] = index / denom;
            index %= denom;

        }
        return ret;
    }

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
    inline int* ind2sub(int rank, const int *shape, int index) {
        return ind2sub(rank,shape, index,shape::prodLong(shape,rank));
    }

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
    inline void  ind2sub(int rank, int *shape, int index, int numIndices, int *ret) {
        int denom = numIndices;

        for(int i = rank - 1; i >= 0; i--) {
            denom /= shape[i];
            ret[i] = index / denom;
            index %= denom;
        }
    }

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
    inline void ind2sub(int rank,int *shape,int index, int *out) {
        ind2sub(rank,shape, index,shape::prodLong(shape,rank),out);
    }

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
    inline int * ind2subC(int rank, int *shape, int index, int numIndices) {
        int denom = numIndices;
        int *ret = new int[rank];

        for(int i = 0; i < rank; i++) {
            denom /= shape[i];
            ret[i] = index / denom;
            index %= denom;

        }
        return ret;
    }

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
    inline int *ind2subC(int rank, int *shape, int index) {
        return ind2subC(rank,shape, index, shape::prodLong(shape,rank));
    }

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
    inline void ind2subC(int rank, int *shape, int index, int numIndices, int *ret) {
        int denom = numIndices;
        for(int i = 0; i < rank; i++) {
            denom /= shape[i];
            ret[i] = index / denom;
            index %= denom;

        }
    }

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
    inline void ind2subC(int rank, int *shape, int index, int *out) {
        ind2subC(rank,shape, index,shape::prodLong(shape,rank),out);
    }

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
    inline void  ind2subOrder(int *shapeInfo, int index, int numIndices,int *out) {
        if(shape::order(shapeInfo) == 'f') {
            shape::ind2sub(
                    shape::rank(shapeInfo),
                    shape::shapeOf(shapeInfo),
                    index,
                    numIndices,
                    out);
        }
        else {
            shape::ind2subC(
                    shape::rank(shapeInfo),
                    shape::shapeOf(shapeInfo),
                    index,
                    numIndices,
                    out);

        }
    }

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
    inline void ind2subOrder(int *shapeInfo, int index, int *out) {
        ind2subOrder(shapeInfo,index,shape::length(shapeInfo),out);
    }

/**
 * Convert a linear index to
 * the equivalent nd index
 * @param shape the shape of the dimensions
 * @param index the index to map
 * @param numIndices the number of total indices (typically prod of shape(
 * @return the mapped indexes along each dimension
 */



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

    inline int *doPermuteSwap(int length, int *shape, int *rearrange) {
        int *ret = new int[length];
        for (int i = 0; i < length; i++) {
            ret[i] = shape[rearrange[i]];
        }
        return ret;
    }

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

    inline void doPermuteSwap(int length, int **shape, int *rearrange) {
        int *shapeDeref = *shape;
        for (int i = 0; i < length; i++) {
            int x = shapeDeref[i];
            int j = i;
            while (1) {
                int k = rearrange[j];
                rearrange[j] = j;
                if (k == i)
                    break;
                shapeDeref[j] = shapeDeref[k];
                j = k;

            }

            shapeDeref[j] = x;
        }

    }



#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int* squeezeDimensions(int *shapeInfo, int *dimension, int dimensionLength, bool *squeezedRef,bool *squeezeDimensionsRef,int wholeRank,int numOnes) {
        int *squeezeShape = new int[wholeRank - numOnes];
        int *squeezeStride = new int[wholeRank - numOnes];
        *squeezedRef = true;

        int *shape = shape::shapeOf(shapeInfo);
        int *stride = shape::stride(shapeInfo);

        int numEncountered = 0;
        for(int i = 0; i < wholeRank; i++) {
            if(shape[i] != 1) {
                squeezeShape[numEncountered] = shape[i];
                squeezeStride[numEncountered] = stride[i];
                numEncountered++;
            }
        }

        //for any dimensions specified that are 1,ignore them
        int numDimensionsOne = 0;
        for(int i = 0; i < dimensionLength; i++) {
            if(shape[dimension[i]] == 1)
                numDimensionsOne++;
        }

        if(numDimensionsOne > 0) {
            int *newDimensions = new int[dimensionLength - numDimensionsOne];
            int newDimensionIdx = 0;
            for(int i = 0; i < dimensionLength; i++) {
                if(shape[dimension[i]] != 1)
                    newDimensions[newDimensionIdx++] = dimension[i] - numDimensionsOne;
            }

            //reduce along the new dimensions
            dimension = newDimensions;
            dimensionLength  -= numDimensionsOne;

        }
        //update the stride and shape, note that this will not be a memory leak due to the pointers being declared differently
        //the previous pointer is just a view of a pointer to be reused that was passed in
        shape = squeezeShape;
        stride = squeezeStride;
        wholeRank -= numOnes;
        //adjust dimensions
        for(int i = 0; i < dimensionLength; i++) {
            dimension[i] -= numOnes;
        }

        for(int i = 0; i < dimensionLength; i++) {
            //didn't need to be adjusted
            if(dimension[i] < 0)
                dimension[i] += numDimensionsOne;
        }

        char order = shape::order(shapeInfo);
        int *xShapeInfo = shape::createShapeInfo(shape,stride,wholeRank);
        xShapeInfo[shape::shapeInfoLength(wholeRank) - 1] = order;
        return xShapeInfo;

    }
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *permuteShapeBuffer(int *shapeBuffer,int *rearrange) {
        int len = shape::shapeInfoLength(shape::rank(shapeBuffer));
        int *copy = shape::copyOf(len,shapeBuffer);
        doPermuteShapeBuffer(&copy,rearrange);
        return copy;
    }
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void doPermuteShapeBuffer(int **shapeBuffer,int *rearrange) {
        int *shapeRef = *shapeBuffer;
        //rank of the rearrange array == rank of shape buffer
        int rearrageRank = shape::rank(shapeRef);
        int *shape = shape::shapeOf(shapeRef);
        int *stride = shape::stride(shapeRef);
        int *rearrangeCopy1 = shape::copyOf(rearrageRank,rearrange);
        shape::doPermuteSwap(rearrageRank,&shape,rearrangeCopy1);
        delete[] rearrangeCopy1;
        int *rearrangeCopy2 = shape::copyOf(rearrageRank,rearrange);
        shape::doPermuteSwap(rearrageRank,&stride,rearrangeCopy2);
        delete[] rearrangeCopy2;
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int *createPermuteIndexes(int originalRank,int *dimension,int dimensionLength) {
        int delta = originalRank - dimensionLength;
        int *ret = new int[originalRank];
        for(int i = 0; i < delta; i++) {
            ret[i] = i + dimensionLength;
        }

        for(int i = delta; i  < originalRank; i++) {
            ret[i] = i - delta;
        }

        return ret;
    }

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

    inline char getOrder(int length, int *shape, int *stride, int elementStride) {
        int sd = -1;
        int dim = -1;
        int i = -1;
        int cContiguous = 1;
        int isFortran = 1;

        sd = 1;
        for (i = length - 1; i >= 0; --i) {
            dim = shape[i];

            if (stride[i] != sd) {
                cContiguous = 0;
                break;
            }
            /* contiguous, if it got this far */
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }

        /* check if fortran contiguous */
        sd = elementStride;
        for (i = 0; i < length; ++i) {
            dim = shape[i];
            if (stride[i] != sd) {
                isFortran = 0;
            }
            if (dim == 0) {
                break;
            }
            sd *= dim;

        }

        if (isFortran && cContiguous)
            return 'a';
        else if (isFortran && !cContiguous)
            return 'f';
        else if (!isFortran && !cContiguous)
            return 'c';
        else
            return 'c';

    }





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

    inline int checkArrangeArray(int *arr, int arrLength, int shapeLength) {
        if (arrLength != shapeLength)
            return -1;
        for (int i = 0; i < arrLength; i++) {
            if (arr[i] >= arrLength || arr[i] < 0)
                return -1;
        }
        for (int i = 0; i < arrLength; i++) {
            for (int j = 0; j < arrLength; j++) {
                if (i != j && arr[i] == arr[j])
                    return -1;
            }
        }

        return 1;
    }

/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void permute(ShapeInformation **info, int *rearrange, int rank) {
        ShapeInformation *infoDeref = *info;
        checkArrangeArray(rearrange, rank, rank);
        shape::doPermuteSwap(rank, &infoDeref->shape, rearrange);
        shape::doPermuteSwap(rank, &infoDeref->stride, rearrange);
        char order = getOrder(rank, infoDeref->shape, infoDeref->stride,
                              infoDeref->elementWiseStride);
        infoDeref->order = order;

    }

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int isVector(int *shape, int rank) {
        if (rank > 2)
            return 0;
        else if (rank <= 2) {
            if (shape[0] == 1 || shape[1] == 1)
                return 1;
        }
        return 0;
    }

#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int isVector(int *shapeInfo) {
        return isVector(shape::shapeOf(shapeInfo),shape::rank(shapeInfo));
    }

    /**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int isMatrix(const int *shape, int rank) {
        if (rank > 2)
            return 0;
        else if (rank <= 2) {
            if (shape[0] == 1 || shape[1] == 1)
                return 0;
        }

        return 1;
    }

#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int isMatrix(const int *shapeInfo) {
        return isMatrix(shape::shapeOf(shapeInfo),shape::rank(shapeInfo));
    }

/**
 * Returns the shape portion of an information
 * buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *shapeOf(int *buffer) {
        return buffer + 1;
    }

    inline const int *shapeOf(const int *buffer) {
        return buffer + 1;
    }

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *copyOf(int length, int *toCopy) {
        int *ret = new int[length];
        memcpy(ret, toCopy, sizeof(int)*length);
        return ret;
    }

    /**
* Return a copy of a buffer.
* This buffer allocates memory
* that must be freed elsewhere.
*/
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline void copyTo(int length, int *from, int *to) {
        memcpy(to, from, sizeof(int)*length);
    }

    /**
* Return a copy of a buffer.
* This buffer allocates memory
* that must be freed elsewhere.
*/
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline void copyTo(int length, int *from, int *to, int *indexes) {
        for(int i = 0; i < length; i++) {
            to[i] = from[indexes[i]];
        }
    }

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
    inline int *permutedStrides(int *toPermute, int shapeRank, int *rearrange) {
        int *strideCopy = copyOf(shapeRank, toPermute);
        checkArrangeArray(rearrange, shapeRank, shapeRank);
        int *newStride = doPermuteSwap(shapeRank, strideCopy, rearrange);
        delete[] strideCopy;
        return newStride;
    }

/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *slice(int *shape) {
        return shape + 1;
    }

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

    inline int shapeInfoLength(int rank) {
        //FIXME magic numbers
        return rank * 2 + 4;
    }

/**
 * Returns the rank portion of
 * an information buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int rank(const int *buffer) {
        return buffer[0];
    }

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

    inline ShapeInformation *infoFromBuffer(int *buffer) {
        ShapeInformation *info = new ShapeInformation;
        int length = shapeInfoLength(rank(buffer));
        int rank = buffer[0];

        //start after rank
        info->shape = buffer + 1;
        info->stride = buffer + (1 + rank);
        info->rank = rank;
        info->offset = buffer[length - 3];
        info->elementWiseStride = buffer[length - 2];
        int *stride = buffer + 1 + rank;
        info->stride = stride;
        info->order = (char) buffer[length - 1];
        return info;
    }

/**
 * Returns the stride portion of an information
 * buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline const int *stride(const int *buffer) {
        return buffer + (1 + rank(buffer));
    }

    inline int *stride(int *buffer) {
        return buffer + (1 + rank(buffer));
    }


/**
 * Compute the length of the given shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int length(int *shapeInfo) {
        return shape::prodLong(shape::shapeOf(shapeInfo), shape::rank(shapeInfo));
    }

/***
 * Returns the offset portion of an information buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int offset(int *buffer) {
        int length = shape::shapeInfoLength(shape::rank(buffer));
        return buffer[length - 3];
    }

/**
 * Returns the ordering
 * for this shape information buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline char order(int *buffer) {
        //FIXME magic numbers
        int length = buffer[0] * 2 + 4;
        return (char) buffer[length - 1];
    }

/**
 * Returns the element wise stride for this information
 * buffer
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int elementWiseStride(int *buffer) {
        int length2 = shapeInfoLength(buffer[0]);
        return buffer[length2 - 2];
    }

    /**
* Returns the element wise stride for this information
* buffer relative to a dimension and reduction index
*/
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int reductionIndexElementWiseStride(int *buffer, int *dimension, int dimensionLength) {
        if(dimensionLength > 1) {
            char order = shape::order(buffer);
            if(order == 'f') {
                /**
                        * The element wise stride belongs to a reduction index.
                        * When used out of order, we can get rid of the data
                        * dependencies and rely on using the max dimension
                        * specified for stride instead.
                        * Say we take the sum(0,1) along arr
                        * we can use arr.stride(1) as a representation
                        * along which to iterate.
                        */
                int tadElementWiseStride = shape::stride(buffer)[dimension[dimensionLength - 1]];
                return tadElementWiseStride;
            }
            else {
                /**
                        * The element wise stride belongs to a reduction index.
                        * When used out of order, we can get rid of the data
                        * dependencies and rely on using the max dimension
                        * specified for stride instead.
                        * Say we take the sum(0,1) along arr
                        * we can use arr.stride(1) as a representation
                        * along which to iterate.
                        */
                int tadElementWiseStride = shape::stride(buffer)[dimension[dimensionLength - 1]];
                return tadElementWiseStride;
            }
        }
        else {
            char order = shape::order(buffer);
            if(order == 'f') {
                /**
                        * The element wise stride belongs to a reduction index.
                        * When used out of order, we can get rid of the data
                        * dependencies and rely on using the max dimension
                        * specified for stride instead.
                        * Say we take the sum(0,1) along arr
                        * we can use arr.stride(1) as a representation
                        * along which to iterate.
                        */
                int tadElementWiseStride = shape::stride(buffer)[dimension[0]];
                return tadElementWiseStride;
            }
            else {
                /**
                        * The element wise stride belongs to a reduction index.
                        * When used out of order, we can get rid of the data
                        * dependencies and rely on using the max dimension
                        * specified for stride instead.
                        * Say we take the sum(0,1) along arr
                        * we can use arr.stride(1) as a representation
                        * along which to iterate.
                        */
                int tadElementWiseStride = shape::stride(buffer)[dimension[dimensionLength - 1]];
                return tadElementWiseStride;
            }
        }

    }

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int isScalar(int *info) {
        if (shape::rank(info) > 2)
            return 0;
        if (shape::rank(info) == 1)
            return shape::shapeOf(info)[0] == 1;
        else if (rank(info) == 2) {
            return shape::shapeOf(info)[0] == 1 && shape::shapeOf(info)[1] == 1;
        }
        return 0;
    }

/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int isScalar(volatile ShapeInformation *info) {
        if (info->rank > 2)
            return 0;
        if (info->rank == 1)
            return info->shape[0] == 1;
        else if (info->rank == 2) {
            return info->shape[0] == 1 && info->shape[1] == 1;
        }
        return 0;
    }

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

    inline void removeIndex(int *data, int *indexes, int dataLength, int indexesLength,
                            int *ret) {
        int count = 0;
        int absLength = dataLength - indexesLength;
        for (int i = 0; i < dataLength && count < absLength; i++) {
            int contains = 0;
            for (int j = 0; j < indexesLength; j++) {
                if (i == indexes[j]) {
                    contains = 1;
                    break;
                }
            }

            if (!contains) {
                ret[count] = data[i];
                count++;
            }
        }
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int* everyIndexBut(int *indexes,int indexesLength,int begin,int end) {
        int len = end - indexesLength;
        int *ret = new int[len];
        int retIdx = 0;
        //not here that we do 0 based indexing for end - this assumes things like:
        //0 to 4 are specified
        for(int i = begin; i < end ; i++) {
            bool found = false;
            for(int j = 0; j < indexesLength; j++) {
                if(indexes[j] == i) {
                    found = true;
                    break;
                }

            }

            if(!found) {
                ret[retIdx++] = i;
            }

        }

        return ret;

    }

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
#ifdef __CUDACC__
    __device__ int tadOffset(ShapeInformation *xInfo, int offset) {
	return offset + threadIdx.x * xInfo->elementWiseStride;

}

#endif

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

    inline int *ensureVectorShape(int *shape, int dimension) {
        int *ret = new int[2];

        if (dimension == 0) {
            ret[0] = 1;
            ret[1] = shape[0];
        } else {
            ret[0] = shape[0];
            ret[1] = 1;
        }

        return ret;
    }

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

    inline int *ensureVectorShape(int *shape) {
        return ensureVectorShape(shape, 0);
    }

/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *range(int from, int to, int increment) {
        int diff = nd4j::math::nd4j_abs<int>(from - to);
        int retLength = diff / increment;
        int *ret;
        if(diff / increment < 1)
            ret = new int[1];
        else
            ret = new int[diff / increment];
        if (from < to) {
            int count = 0;
            for (int i = from; i < to; i += increment) {
                if (count >= retLength)
                    break;
                ret[count++] = i;
            }
        } else if (from > to) {
            int count = 0;
            for (int i = from - 1; i >= to; i -= increment) {
                if (count >= retLength)
                    break;
                ret[count++] = i;
            }
        }

        return ret;
    }

/**
 * Generate a range
 * beginning at from and ending at to
 * incrementing by 1
 * @param from the start
 * @param to the end
 * @return the int array starting at from and ending at to
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *range(int from, int to) {
        return range(from, to, 1);
    }

/**
 * Keep the given indexes in the data
 * @param data
 * @param index
 * @param indexLength
 * @param dataLength
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *keep(volatile int *data, int *index, int indexLength, int dataLength) {
        int *ret = new int[indexLength];
        int count = 0;
        for (int i = 0; i < dataLength; i++) {
            int contains = 0;
            for (int j = 0; j < indexLength; j++) {
                if (i == index[j]) {
                    contains = 1;
                    break;
                }
            }

            if (contains)
                ret[count++] = data[i];
        }
        return ret;
    }

/**
 * Generate a reverse
 * copy of the data
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *reverseCopy(int *data, int length) {
        if (length < 1)
            return nullptr;

        int *copy = new int[length];
        for (int i = 0; i <= length / 2; i++) {
            int temp = data[i];
            copy[i] = data[length - i - 1];
            copy[length - i - 1] = temp;
        }
        return copy;
    }


#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void reverseCopyTo(int *from, int *to, int length) {
        if (length < 1)
            return;
        for (int i = 0; i <= length / 2; i++) {
            int temp = from[i];
            to[i] = from[length - i - 1];
            to[length - i - 1] = temp;
        }
    }

#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void reverseCopyTo(int *from, int *to, int *indexes, int length) {
        if (length < 1)
            return;

        for (int i = 0; i <= length / 2; i++) {
            int temp = from[indexes[i]];
            to[i] = from[indexes[length - i - 1]];
            to[length - i - 1] = temp;
        }

    }

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

    inline int *concat(int *arr1, int arr1Length, int *arr2, int arr2Length) {
        int *ret = new int[arr1Length + arr2Length];
        std::memcpy(ret, arr1, arr1Length * sizeof(int));
        std::memcpy(ret + arr1Length, arr2, arr2Length * sizeof(int));
        return ret;
    }

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

    inline int *concat(int numArrays, int numTotalElements, int **arr, int *lengths) {
        int *ret = new int[numTotalElements];
        int count = 0;
#pragma omp simd
        for (int i = 0; i < numArrays; i++) {
            for (int j = 0; j < lengths[i]; j++) {
                ret[count++] = arr[i][j];
            }
        }

        return ret;
    }

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

    inline int lengthPerSlice(int rank, int *shape, int *dimension, int dimensionLength) {
        int absSelta = nd4j::math::nd4j_abs<int>(rank - dimensionLength);
        int *ret2 = new int[absSelta];
        removeIndex(shape, dimension, rank, dimensionLength, ret2);
        int length = rank - dimensionLength;
        int ret = prod(ret2, length);
        delete[] ret2;
        return ret;
    }

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

    inline int sliceOffsetForTensor(int rank, int index, int *shape, int *tensorShape,
                                    int tensorShapeLength, int *dimension, int dimensionLength) {
        int tensorLength = prodLong(tensorShape, tensorShapeLength);
        int lengthPerSlice2 = lengthPerSlice(rank, shape, dimension,
                                             dimensionLength);
        if (lengthPerSlice2 <= 0) {
            return 0;
        }

        int offset = index * tensorLength / lengthPerSlice2;
        return offset;
    }
#ifdef __CUDACC__

    /**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
__device__ int tadOffset(int *xInfo, int offset) {
	return offset + threadIdx.x * elementWiseStride(xInfo);

}
#endif

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

    inline int offset(int index, int rank, ShapeInformation *info, int *dimension,
                      int dimensionLength) {
        int *tensorShape = keep(info->shape, dimension, dimensionLength, rank);
        if (dimensionLength == 1) {
            int *newTensorShape = ensureVectorShape(tensorShape, dimension[0]);
            delete[] tensorShape;
            tensorShape = newTensorShape;
        }

        //change the value
        ShapeInformation *copy = shapeCopy(info);
        info = copy;

        int *reverseDimensions = reverseCopy(dimension, dimensionLength);
        int *rangeRet = range(0, rank);
        int *remove = new int[rank - dimensionLength];
        removeIndex(rangeRet, dimension, rank, dimensionLength, remove);

        int *zeroDimension = new int[1];
        zeroDimension[0] = 0;

        int removeLength = rank - dimensionLength;
        int *newPermuteDims = concat(remove, removeLength, reverseDimensions,
                                     dimensionLength);

        permute(&info, newPermuteDims, rank);

        int *permuted = info->shape;
        int *permutedStrides = info->stride;
        int tensorShapeLength = rank - removeLength;
        if (tensorShapeLength < 2)
            tensorShapeLength = 2;
        int sliceIdx = sliceOffsetForTensor(rank, index, permuted, tensorShape,
                                            tensorShapeLength, zeroDimension, 1);

        //determine offset here

        int *ret2 = slice(info->shape);
        int *ret2Stride = slice(info->stride);

        int ret2Length = prod(ret2, rank - 1);
        int ret2Rank = rank - 1;

        int retOffset = sliceIdx * permutedStrides[0];

        int length = prod(tensorShape, tensorShapeLength);
        int tensorLength = length;
        //__device__ int lengthPerSlice(int rank,int *shape,int *dimension) {
        int offset = index * tensorLength
                     / lengthPerSlice(ret2Rank, ret2, zeroDimension, 1);
        /**
         * Need to do slice(offset) here
         */
        if (sliceIdx == 0
            && length == lengthPerSlice(ret2Rank, ret2, zeroDimension, 1)) {
            /**
             * NOTE STRIDE[1] HERE. WE DO THIS TO AVOID CREATING A NEW SLICE OBJECT.
             */
            //account for shape[i] == 1
            int strideIndex = 1;
            for (int i = 0; i < info->rank; i++) {
                if (info->shape[i] == 1)
                    strideIndex++;
            }

            if (strideIndex >= info->rank)
                strideIndex = info->rank - 1;

            retOffset = info->offset + offset * info->stride[strideIndex];
        }

            //determine offset here
            //note here offset doesn't change, just the shape
            //of the tad
        else if (length == lengthPerSlice(ret2Rank, ret2, zeroDimension, 1)) {
            offset -= ret2[0] * (offset / ret2[0]);
            //set offset here
            ret2 = slice(ret2);
            ret2Rank--;
            //account for shape[i] == 1
            int strideIndex = 1;
            for (int i = 0; i < info->rank; i++) {
                if (info->shape[i] == 1)
                    strideIndex++;
            }

            if (strideIndex >= info->rank)
                strideIndex = info->rank - 1;

            retOffset += info->stride[strideIndex] * offset;
        }

        else {

            while (ret2Length > length) {
                sliceIdx = sliceOffsetForTensor(ret2Rank, index, ret2, tensorShape,
                                                tensorShapeLength, zeroDimension, 1);
                sliceIdx -= ret2[0] * (sliceIdx / ret2[0]);
                //set offset
                ret2 = slice(ret2);
                ret2Stride = slice(ret2Stride);
                ret2Rank--;
                //slice wise offsets are offset + i * majorStride()
                //dividing by the slice index will adjust the offset by a factor of sliceIndex
                ret2Length = prod(ret2, ret2Rank);

            }
        }

        retOffset = info->offset + sliceIdx;

        delete[] reverseDimensions;
        delete[] rangeRet;
        delete[] remove;
        delete[] copy;
        //free the new pointer
        if (rank <= 2) {
            delete[] tensorShape;
        }

        if (retOffset < 0)
            retOffset = 0;

        return retOffset;
    }




/**
 * Given the shape information and dimensions
 * returns common information
 * needed for tensor along dimension
 * calculations
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline TADPermuteInfo tadInfo(int *xShapeInfo, int *dimension, int dimensionLength) {
        int *shapeOfX = shape::shapeOf(xShapeInfo);
        int xRank = shape::rank(xShapeInfo);
        int *tensorShape = shape::keep(shapeOfX, dimension, dimensionLength, xRank);
        if (dimensionLength == 1) {
            int *newTensorShape = shape::ensureVectorShape(tensorShape,
                                                           dimension[0]);
            delete[] tensorShape;
            tensorShape = newTensorShape;
        }


        int removeLength = nd4j::math::nd4j_abs<int>(xRank - dimensionLength);
        int tensorShapeLength = shape::rank(xShapeInfo) - removeLength;
        if (tensorShapeLength < 2)
            tensorShapeLength = 2;

        int tensorShapeProd = shape::prod(tensorShape, tensorShapeLength);
        int *reverseDimensions = shape::reverseCopy(dimension, dimensionLength);
        int *rangeRet = shape::range(0, xRank);

        int *remove = new int[removeLength];
        shape::removeIndex(rangeRet, dimension, xRank, dimensionLength, remove);

        int *zeroDimension = new int[1];
        zeroDimension[0] = 0;

        int *newPermuteDims = shape::concat(remove, removeLength, reverseDimensions,
                                            dimensionLength);
        int *permutedShape = shape::copyOf(shape::rank(xShapeInfo),shape::shapeOf(xShapeInfo));
        int *permutedStrides = shape::copyOf(shape::rank(xShapeInfo),shape::stride(xShapeInfo));
        shape::doPermuteSwap(shape::rank(xShapeInfo),&permutedShape,newPermuteDims);
        shape::doPermuteSwap(shape::rank(xShapeInfo),&permutedStrides,newPermuteDims);
        TADPermuteInfo info;
        info.tensorShape = tensorShape;
        info.xRank = xRank;
        info.reverseDimensions = reverseDimensions;
        info.rangeRet = rangeRet;
        info.removeLength = removeLength;
        info.remove = remove;
        info.zeroDimension = zeroDimension;
        info.newPermuteDims =  newPermuteDims;
        info.permutedShape = permutedShape;
        info.permutedStrides = permutedStrides;
        info.tensorShapeLength = tensorShapeLength;
        info.tensorShapeProd = tensorShapeProd;


        return info;
    }

/**
 * Frees the permute information
 * @param info the info to free
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline void freePermuteInfo(TADPermuteInfo info) {
        if(info.tensorShape != NULL)
            delete[] info.tensorShape;
        if(info.reverseDimensions != NULL)
            delete[] info.reverseDimensions;
        if(info.rangeRet != NULL)
            delete[] info.rangeRet;
        if(info.remove != NULL)
            delete[] info.remove;
        if(info.zeroDimension != NULL)
            delete[] info.zeroDimension;
        if(info.newPermuteDims != NULL)
            delete[] info.newPermuteDims;
        if(info.permutedShape != NULL)
            delete[] info.permutedShape;
        if(info.permutedStrides != NULL)
            delete[] info.permutedStrides;

    }

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int tensorsAlongDimension(volatile int rank, volatile int length,
                                     volatile int *shape, int *dimension, int dimensionLength) {
        int *tensorShape = shape::keep(shape, dimension, dimensionLength, rank);
        int ret = length / shape::prodLong(tensorShape, dimensionLength);
        delete[] tensorShape;
        return ret;
    }

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int tensorsAlongDimension(int *shapeInfo, int *dimension, int dimensionLength) {
        int *keepShape = shape::shapeOf(shapeInfo);
        int *tensorShape = shape::keep(keepShape, dimension, dimensionLength,
                                       rank(shapeInfo));
        int ret = shape::length(shapeInfo)
                  / shape::prodLong(tensorShape, dimensionLength);
        delete[] tensorShape;
        return ret;
    }

/**
 *
 * @param info
 * @param dimension
 * @param dimensionLength
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int tensorsAlongDimension(TADPermuteInfo info) {
        int length = shape::prodLong(info.permutedShape, info.xRank);
        return length / shape::prodLong(info.tensorShape, info.tensorShapeLength);
    }

/**
 *
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int * tadShapeInfo(int index, int *xShapeInfo, int *dimension,
                              int dimensionLength) {
        TADPermuteInfo tadInfo = shape::tadInfo(xShapeInfo, dimension,
                                                dimensionLength);
        int sliceIdx = shape::sliceOffsetForTensor(rank(xShapeInfo), index,
                                                   tadInfo.permutedShape, tadInfo.tensorShape,
                                                   tadInfo.tensorShapeLength, tadInfo.zeroDimension, 1);

        //no more dimensions
        if (tadInfo.xRank <= 2) {
            shape::freePermuteInfo(tadInfo);
            return xShapeInfo;
        }

        //determine offset here

        int *ret2 = shape::slice(tadInfo.permutedShape);
        int *ret2Stride = shape::slice(tadInfo.permutedStrides);
        int ret2Length = shape::prod(ret2, shape::rank(xShapeInfo) - 1);
        int ret2Rank = tadInfo.xRank - 1;

        int retOffset = sliceIdx * tadInfo.permutedStrides[0];
        int tensorShapeProd = tadInfo.tensorShapeProd;
        int delta = nd4j::math::nd4j_abs<int>(
                tadInfo.tensorShapeLength - dimensionLength);
        int tensorShapeRoughlyEquals = dimensionLength == 1 && delta <= 1;
        if ((tensorShapeProd == ret2Length && tensorShapeRoughlyEquals == 1)
            || dimensionLength == tadInfo.tensorShapeLength) {
            ShapeInformation *info = new ShapeInformation();
            //row vector
            if (ret2Rank == 1) {
                ret2Rank++;
                ret2 = shape::ensureVectorShape(ret2);
                ret2Stride = shape::ensureVectorShape(ret2Stride);
            }
            info->shape = ret2;
            info->stride = ret2Stride;
            info->offset = retOffset;
            info->rank = ret2Rank;
            info->elementWiseStride = ret2Stride[ret2Rank - 1];
            int *shapeInfoRet = shape::toShapeBuffer(info);
            delete info;
            shape::freePermuteInfo(tadInfo);
            return shapeInfoRet;
        }

        int length = tadInfo.tensorShapeProd;
        int tensorLength = length;
        int sliceOffset = index * tensorLength
                          / shape::lengthPerSlice(ret2Rank, ret2, tadInfo.zeroDimension, 1);
        /**
         * Need to do slice(offset) here
         */
        int lengthPerSlice2 = shape::lengthPerSlice(ret2Rank, ret2,
                                                    tadInfo.zeroDimension, 1);

        if (sliceIdx == 0 && length == lengthPerSlice2) {
            ret2 = slice(ret2);
            ret2Stride = shape::slice(ret2Stride);
            ret2Rank--;
            ret2Length = shape::prod(ret2, ret2Rank);
            int newStride = ret2Stride[ret2Rank - 1];
            retOffset += (sliceOffset * ret2Length * newStride);

            if (retOffset < 0)
                retOffset = 0;
            //row vector
            if (ret2Rank == 1) {
                ret2Rank++;
                ret2 = shape::ensureVectorShape(ret2);
                ret2Stride = shape::ensureVectorShape(ret2Stride);
            }
            ShapeInformation *info = new ShapeInformation();
            info->shape = ret2;
            info->stride = ret2Stride;
            info->offset = retOffset;
            info->rank = ret2Rank;
            info->elementWiseStride = ret2Stride[ret2Rank - 1];
            int *shapeInfoRet = shape::toShapeBuffer(info);
            delete info;
            shape::freePermuteInfo(tadInfo);
            return shapeInfoRet;
        }

            //determine offset here
            //note here offset doesn't change, just the shape
            //of the tad
        else if (length == lengthPerSlice2) {
            sliceOffset -= ret2[0] * (sliceOffset / ret2[0]);
            //set offset here
            ret2 = slice(ret2);
            ret2Stride = slice(ret2Stride);
            ret2Rank--;
            //accumulate from the slice
            int newStride = ret2Stride[ret2Rank - 1];
            retOffset += (lengthPerSlice2 * newStride * sliceOffset);

            if (retOffset < 0)
                retOffset = 0;

            ShapeInformation *info = new ShapeInformation();
            //row vector
            if (ret2Rank == 1) {
                ret2Rank++;
                ret2 = ensureVectorShape(ret2);
                ret2Stride = ensureVectorShape(ret2Stride);
            }
            info->shape = ret2;
            info->stride = ret2Stride;
            info->offset = retOffset;
            info->rank = ret2Rank;
            info->elementWiseStride = ret2Stride[ret2Rank - 1];
            int *shapeInfoRet = shape::toShapeBuffer(info);
            delete info;
            shape::freePermuteInfo(tadInfo);
            return shapeInfoRet;
        }

        else {
            ret2Length = prod(ret2, ret2Rank);
            //start at zero incrementing whenever we hit a slice > 0
            while (ret2Length > length && ret2Rank > 0) {
                sliceIdx = sliceOffsetForTensor(ret2Rank, index, ret2,
                                                tadInfo.tensorShape, tadInfo.tensorShapeLength,
                                                tadInfo.zeroDimension, 1);
                sliceIdx -= ret2[0] * (sliceIdx / ret2[0]);
                if (sliceIdx > 0) {
                    if (ret2Rank > 1) {
                        retOffset += sliceIdx * ret2Stride[0];
                    } else {
                        retOffset += sliceIdx;
                    }
                }
                //set offset
                ret2 = shape::slice(ret2);
                ret2Stride = shape::slice(ret2Stride);
                //bump the offset wrt the slice idx when its not just truncating output

                ret2Rank--;
                ret2Length = shape::prod(ret2, ret2Rank);
            }

            ShapeInformation *info = new ShapeInformation();
            //row vector
            if (ret2Rank == 1) {
                ret2Rank++;
                ret2 = ensureVectorShape(ret2);
                ret2Stride = ensureVectorShape(ret2Stride);
            }
            info->shape = ret2;
            info->stride = ret2Stride;
            info->offset = retOffset;
            info->rank = ret2Rank;
            info->elementWiseStride = ret2Stride[ret2Rank - 1];

            int *shapeInfoRet = shape::toShapeBuffer(info);
            delete info;
            shape::freePermuteInfo(tadInfo);
            return shapeInfoRet;
        }

    }


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
    int getOffset(int baseOffset, const int *shape, const int *stride, const int *indices, int rank) {
        int offset = baseOffset;
        for(int i = 0; i < rank; i++) {
            if(indices[i] >= shape[i]) {
                printf("Index [%d] must not be >= shape[d].\n", i);
                return -1;
            }

            if(shape[i] != 1) {
                offset += (int)indices[i] * stride[i];
            }
        }

        return offset;
    }


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

    inline int offset(int index, int *xShapeInfo,int *dimension, int dimensionLength,
                      TADPermuteInfo info) {
        int sliceIdx = sliceOffsetForTensor(rank(xShapeInfo), index,
                                            info.permutedShape, info.tensorShape, info.tensorShapeLength,
                                            info.zeroDimension, 1);

        //determine offset here

        int *ret2 = slice(info.permutedShape);
        int *ret2Stride = slice(info.permutedStrides);
        int ret2Length = prod(ret2, rank(xShapeInfo) - 1);
        int ret2Rank = info.xRank - 1;

        int retOffset = sliceIdx * info.permutedStrides[0];
        if(dimension[0] == 0 ) {
            return sliceIdx;
            /*
            char xOrder = order(xShapeInfo);
            if (xOrder == 'c') {
                return sliceIdx;
            } else {
                // special case for F ordering on 0 dimension
                return index * info.tensorShapeProd;
            }
             */
        }

        int tensorShapeProd = info.tensorShapeProd;
        int val = nd4j::math::nd4j_abs<int>(
                info.tensorShapeLength - dimensionLength) <= 1;
        int tensorShapeRoughlyEquals = dimensionLength == 1 && val;
        if ((tensorShapeProd == ret2Length && tensorShapeRoughlyEquals == 1)
            || (dimensionLength == info.tensorShapeLength && dimensionLength > 0)) {
            return retOffset;
        }

        int length = info.tensorShapeProd;
        int tensorLength = length;
        int sliceOffset = index * tensorLength
                          / lengthPerSlice(ret2Rank, ret2, info.zeroDimension, 1);
        /**
         * Need to do slice(offset) here
         */
        int lengthPerSlice2 = lengthPerSlice(ret2Rank, ret2, info.zeroDimension, 1);

        if (sliceIdx == 0 && length == lengthPerSlice2) {
            ret2 = slice(ret2);
            ret2Stride = slice(ret2Stride);
            ret2Rank--;
            ret2Length = prod(ret2, ret2Rank);
            int newStride = ret2Stride[ret2Rank - 1];
            retOffset += (sliceOffset * ret2Length * newStride);

            if (retOffset < 0)
                retOffset = 0;

            return retOffset;
        }

            //determine offset here
            //note here offset doesn't change, just the shape
            //of the tad
        else if (length == lengthPerSlice2) {
            sliceOffset -= ret2[0] * (sliceOffset / ret2[0]);
            //set offset here
            ret2 = slice(ret2);
            ret2Stride = slice(ret2Stride);
            ret2Rank--;
            //accumulate from the slice
            int newStride = ret2Stride[ret2Rank - 1];
            retOffset += (lengthPerSlice2 * newStride * sliceOffset);

            if (retOffset < 0)
                retOffset = 0;

            return retOffset;
        }

        else {
            ret2Length = prod(ret2, ret2Rank);
            //start at zero incrementing whenever we hit a slice > 0
            while (ret2Length > length && ret2Rank > 0) {
                sliceIdx = sliceOffsetForTensor(ret2Rank, index, ret2,
                                                info.tensorShape, info.tensorShapeLength,
                                                info.zeroDimension, 1);
                sliceIdx -= ret2[0] * (sliceIdx / ret2[0]);
                if (sliceIdx > 0) {
                    if (ret2Rank > 1) {
                        retOffset += sliceIdx * ret2Stride[0];
                    } else {
                        retOffset += sliceIdx;
                    }
                }
                //set offset
                ret2 = slice(ret2);
                ret2Stride = slice(ret2Stride);
                //bump the offset wrt the slice idx when its not just truncating output

                ret2Rank--;
                ret2Length = prod(ret2, ret2Rank);
            }

            return retOffset;
        }

    }

/**
 * Returns the tensor along dimension
 * for the given block index
 * @param blockSize
 * @param blockIdx
 * @param i
 * @return
 */
#ifdef __CUDACC__
    __device__ __host__
#endif
    inline int tadForBlockIndex(int blockSize, int blockIdx, int i) {
        int ret = blockIdx + i * blockSize;
        return ret;
    }

/**
 * Computes the number of tads per block
 *
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int tadsPerBlock(int blockSize, int tads) {
        return nd4j::math::nd4j_ceil<double>(tads / (double) blockSize);
    }

/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int *toShapeBuffer(const ShapeInformation *info) {
        int *ret = new int[shapeInfoLength(info->rank)];
        int count = 1;
        int rank = info->rank;

        ret[0] = info->rank;
#pragma omp simd
        for (int i = 0; i < rank; i++) {
            ret[count++] = info->shape[i];
        }
#pragma omp simd
        for (int i = 0; i < rank; i++) {
            ret[count++] = info->stride[i];
        }

        ret[count++] = info->offset;
        ret[count++] = info->elementWiseStride;
        ret[count++] = info->order;

        return ret;
    }

/**
 * Given an linear index, element wise stride
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

    inline int tadIndex(int i, int elementWiseStride, int numElementsPerTad) {
        return i / (numElementsPerTad * elementWiseStride);
    }

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

    inline int reductionIndexForTad(int tadIndexForOriginal, int tadsForReduced,
                                    int tadsForOriginal) {
        if (tadIndexForOriginal == 0)
            return 0;
        return tadIndexForOriginal / (tadsForOriginal / tadsForReduced);
    }

/**
 * Tad index for linear
 * @param linearIndex
 * @param tadLength
 * @return
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int tadIndexForLinear(int linearIndex, int tadLength) {
        return linearIndex % tadLength;
    }

/**
 * Computes the number of tads
 * per reduce index for the
 * reduction tad.
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int tadsPerReduceIndex(int tadsForReduce, int tadsForOriginal) {
        return tadsForOriginal / tadsForReduce;
    }

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
    inline int reductionIndexForLinear(int i, int elementWiseStride, int numElementsPerTad,
                                       int tadNum, int originalTadNum) {
        int tad = tadIndex(i, elementWiseStride, numElementsPerTad);
        return reductionIndexForTad(tad, tadNum, originalTadNum);
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int* createScalarShapeInfo() {
        int *shape = new int[2];
        shape[0] = 1;
        shape[1] = 1;
        int *stride = new int[2];
        stride[0] = 1;
        stride[1] = 1;
        ShapeInformation *shapeInformation2 = new ShapeInformation();
        shapeInformation2->rank = 2;
        shapeInformation2->offset = 0;
        shapeInformation2->stride = stride;
        shapeInformation2->shape = shape;
        shapeInformation2->elementWiseStride = 1;
        int *ret = shape::toShapeBuffer(shapeInformation2);
        delete shapeInformation2;
        return ret;
    }

/**
 * Returns the prod of the data
 * up to the given length
 */
#ifdef __CUDACC__
    __host__ __device__
#endif

    inline int prod(int *data, int length) {
        int prod = 1;
        for (int i = 0; i < length; i++) {
            prod *= data[i];
        }

        return prod;
    }

/**
 * Returns the prod of the data
 * up to the given length
 */
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int prodLong(const int *data, int length) {
        int prod = 1;
        for (int i = 0; i < length; i++) {
            prod *= data[i];
        }

        return prod;
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    inline int rearMostLeftOverItem(int *data, int *dimension,int dimensionLength) {
        int *stride = shape::stride(data);
        //corner case: return the final item when its greater than the max, since its guaranteed to be left over
        //note here that strides are interpreted in reverse for tad
        //start from the front rather than the back

        int rank = shape::rank(data);


        if(shape::order(data) == 'f') {
            int dimIdx = dimensionLength - 1;
            for(int i = rank - 1; i >= 0; i--) {
                /**
                 * Needs to find an algorithm such that:
                 * looping backwards will find the highest dimension left
                 * that isn't included in the dimension index list.
                 *
                 * This can also be thought of as the last item of the first index
                 * of the difference between the full list of indices and
                 * the dimension indices.
                 *
                 * We should avoid excessive object creation by only looping backwards.
                 */
                if(dimension[dimIdx--] != i) {
                    int ret = stride[i];
                    return ret;
                }
            }
        }

        else {
            int dimIdx = dimensionLength - 1;

            for(int i = rank - 1; i >= 0; i--) {
                /**
                 * Needs to find an algorithm such that:
                 * looping backwards will find the highest dimension left
                 * that isn't included in the dimension index list.
                 *
                 * This can also be thought of as the last item of the first index
                 * of the difference between the full list of indices and
                 * the dimension indices.
                 *
                 * We should avoid excessive object creation by only looping backwards.
                 */
                if(dimension[dimIdx--] != i) {
                    int ret = stride[i];
                    return ret;
                }
            }
        }




        int ret = stride[0];
        return ret;
    }

}

#endif /* SHAPE_H_ */
