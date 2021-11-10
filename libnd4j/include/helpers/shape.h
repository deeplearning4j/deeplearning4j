/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/*
 * shape.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SHAPE_H_
#define SHAPE_H_
#include <array/ArrayOptions.h>
#include <cnpy/cnpy.h>
#include <helpers/logger.h>
#include <math/templatemath.h>
#include <stdint.h>
#include <system/op_boilerplate.h>

#include <cstdio>
#include <cstring>

#include "system/pairwise_util.h"

namespace shape {

/**
 * Shape information approximating
 * the information on an ndarray
 */
struct SD_LIB_EXPORT ShapeInformation {
  SD_HOST_DEVICE ShapeInformation(sd::LongType *shape_ = nullptr, sd::LongType *stride_ = nullptr, char order_ = 0,
                                  int rank_ = 0, int offset_ = 0, int elementWiseStride_ = 0)
      : shape(shape_),
        stride(stride_),
        order(order_),
        rank(rank_),
        offset(offset_),
        elementWiseStride(elementWiseStride_) {}

  sd::LongType *shape;
  sd::LongType *stride;
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

SD_LIB_EXPORT SD_HOST_DEVICE bool shapeEquals(const int shape1Rank, const sd::LongType *shape1, const int shape2Rank,
                                              const sd::LongType *shape2);

SD_LIB_EXPORT SD_HOST_DEVICE const sd::LongType *detachShape(const sd::LongType *originalShape);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *copyShape(sd::LongType const *originalShape);

SD_LIB_EXPORT SD_HOST_DEVICE bool shapeEquals(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2);

SD_LIB_EXPORT SD_HOST_DEVICE bool shapeEquals(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2,
                                              const sd::LongType *shapeInfo3);

SD_LIB_EXPORT SD_HOST_DEVICE bool strideEquals(int const shape1Rank, sd::LongType const *shape1, int const shape2Rank,
                                               sd::LongType const *shape2);

SD_LIB_EXPORT SD_HOST_DEVICE bool strideEquals(sd::LongType const *shapeInfo1, sd::LongType const *shapeInfo2);

SD_LIB_EXPORT SD_HOST_DEVICE bool strideEquals(sd::LongType const *stride1, int const rank1,
                                               sd::LongType const *stride2, int const rank2);

SD_LIB_EXPORT SD_HOST_DEVICE bool equalsSoft(const sd::LongType *shapeA, const sd::LongType *shapeB);

SD_LIB_EXPORT SD_HOST_DEVICE bool equalsTypesAndShapesSoft(const sd::LongType *shapeA, const sd::LongType *shapeB);

SD_LIB_EXPORT SD_HOST_DEVICE bool equalsStrict(const sd::LongType *shapeA, const sd::LongType *shapeB);

// returns true if ranks, shapes and strides are the same
SD_LIB_EXPORT SD_HOST_DEVICE bool haveSameShapeAndStrides(const sd::LongType *shapeInfo1,
                                                          const sd::LongType *shapeInfo2);
SD_LIB_EXPORT SD_HOST_DEVICE bool haveSameShapeAndStrides(const sd::LongType *shapeInfo1,
                                                          const sd::LongType *shapeInfo2,
                                                          const sd::LongType *shapeInfo3);

SD_LIB_EXPORT SD_HOST_DEVICE int sizeAt(const sd::LongType *shapeInfo, const int dim);
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType strideAt(const sd::LongType *shapeInfo, const int dim);

template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE void fill(T *buffer, T value, sd::LongType length);

SD_LIB_EXPORT SD_HOST_DEVICE void traceNew(int id);

SD_LIB_EXPORT SD_HOST_DEVICE int tadIndexForLinear(int linearIndex, int tadLength);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType tadLength(const sd::LongType *shapeInfo, int *dimension, int dimensionLength);

SD_LIB_EXPORT SD_HOST_DEVICE bool canReshape(const int oldRank, sd::LongType *oldShape, const int newRank,
                                             sd::LongType *newShape, bool isFOrder);

SD_LIB_EXPORT SD_HOST_DEVICE bool reshapeC(const sd::LongType *oldShapeInfo, const char newOrder, const int newRank,
                                           const sd::LongType *newShape, sd::LongType *newShapeInfo);
/**
 * newShapeInfo contains rank, shape and order only, no strides/ews/type
 */
SD_LIB_EXPORT SD_HOST_DEVICE bool reshapeC(const sd::LongType *oldShapeInfo, sd::LongType *newShapeInfo);

/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeBuffer(int rank, sd::DataType dtype, sd::LongType const *shape);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeBuffer(int rank, sd::DataType dtype, sd::LongType const *shape,
                                                       sd::LongType *buffer);

/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeBufferFortran(int rank, sd::DataType dtype, sd::LongType const *shape);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeBufferFortran(int rank, sd::DataType dtype, sd::LongType const *shape,
                                                              sd::LongType *output);

#ifdef __CUDACC__

SD_DEVICE SD_LIB_EXPORT sd::LongType *cuMalloc(sd::LongType *buffer, long size);
#endif

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank, sd::LongType *ret);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, int rank);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, int rank, sd::LongType *ret);

SD_LIB_EXPORT SD_HOST_DEVICE void updateStrides(sd::LongType *shape, const char order);
SD_LIB_EXPORT SD_HOST_DEVICE void updateStrides(const int rank, const sd::LongType *shapeOnly,
                                                sd::LongType *stridesOnly, const char order);

// check whether input dimensions are permuted, not permuted dimensions order have to be 0,....,rank-1
template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE bool isDimPermuted(const T *dimensions, const int dimSize);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank, int startNum);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank, int startNum,
                                                              sd::LongType *ret);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, int rank, int startNum);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, int rank, int startNum,
                                                       sd::LongType *ret);

/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
SD_LIB_EXPORT SD_HOST_DEVICE ShapeInformation *shapeCopy(ShapeInformation *toCopy);

SD_LIB_EXPORT SD_HOST_DEVICE bool strideDescendingCAscendingF(const sd::LongType *shapeBuffer);

SD_LIB_EXPORT SD_HOST_DEVICE bool isContiguous(const sd::LongType *shapeInfo);

/**
 * copy-past from java hasDefaultStridesForShape function
 * check whether array is not permuted and has contiguous elements in memory
 */
SD_LIB_EXPORT SD_HOST_DEVICE bool areStridesDefault(const sd::LongType *shapeInfo);

/**
 * Compute the element wise stride
 * for a given shape/stride configuration
 * @param rank the rank of the shape/stride
 * @param shape the shape
 * @param stride the stride
 * @param isFOrder 0 or 1 for whether the array is f
 * ordered or not
 * @return 0 if there is no element wise stride the
 * element wise stride of reshape(1,length) otherwise
 */
SD_LIB_EXPORT SD_HOST_DEVICE int computeElementWiseStride(int rank, sd::LongType const *shape,
                                                          sd::LongType const *stride, int isFOrder);

/**
 * Compute the element wise stride
 * for a given shape/stride configuration
 * @param rank the rank of the shape/stride
 * @param shape the shape
 * @param stride the stride
 * @param isFOrder 0 or 1 for whether the array is f
 * ordered or not
 * @return 0 if there is no element wise stride the
 * element wise stride of reshape(1,length) otherwise
 */
SD_LIB_EXPORT SD_HOST_DEVICE int computeElementWiseStride(int rank, sd::LongType const *shape,
                                                          sd::LongType const *stride, int isFOrder,
                                                          sd::LongType const *dimension, int dimensionLength);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeInfoOnlyShapeAndStride(sd::LongType const *shapeInfo,
                                                                       sd::LongType *dimension, int dimensionLength,
                                                                       bool reverseCopyStride);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeInfoOnlyShapeAndStride(const sd::LongType *shapeInfo,
                                                                       sd::LongType *dimension, int dimensionLength,
                                                                       bool reverseCopyStride, sd::LongType *buffer);
/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *doPermuteSwap(int length, sd::LongType *shape, int *rearrange);

/**
 * In place permute swap
 * @param length
 * @param shape
 * @param rearrange
 */
SD_LIB_EXPORT SD_HOST_DEVICE void doPermuteSwap(int length, sd::LongType **shape, int *rearrange);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *permuteShapeBuffer(sd::LongType const *shapeBuffer, int *rearrange);

SD_LIB_EXPORT SD_HOST_DEVICE void permuteShapeBufferInPlace(sd::LongType *shapeBuffer, int *rearrange,
                                                            sd::LongType *out);

SD_LIB_EXPORT SD_HOST_DEVICE void doPermuteShapeInfo(sd::LongType *shapeBuffer, const int *rearrange,
                                                     sd::LongType len = -1);

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

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *createPermuteIndexes(int originalRank, int *dimension, int dimensionLength);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *computeResultShape(const sd::LongType *originalShapeBuffer, int *dimension,
                                                              int dimensionLength);

/**
 * This method does inplace transpose of given shapeBuffer
 *
 * @param shapeBuffer
 */
SD_LIB_EXPORT SD_HOST_DEVICE void transposeInplace(sd::LongType *shapeBuffer);

/**
 * Get the ordering for the device
 * @param length
 * @param shape
 * @param stride
 * @param elementStride
 * @return
 */
SD_LIB_EXPORT SD_HOST_DEVICE char getOrder(int length, sd::LongType *shape, sd::LongType *stride, int elementStride);

/**
 * Ensure that every value in the re arrange
 * array is unique
 * @param arr
 * @param shape
 * @param arrLength
 * @param shapeLength
 * @return
 */
template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE int checkArrangeArray(T *arr, int arrLength, int shapeLength);

/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
SD_LIB_EXPORT SD_HOST_DEVICE void permute(ShapeInformation **info, int *rearrange, int rank);

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of cthe shape
 */
SD_LIB_EXPORT SD_HOST_DEVICE int isVector(sd::LongType const *shape, int rank);

/**
 * When 1 dimension is the whole length of the
 * array
 */
SD_LIB_EXPORT SD_HOST_DEVICE int oneDimEqualToLength(sd::LongType *shape, int rank);

SD_LIB_EXPORT SD_HOST_DEVICE int oneDimEqualToLength(sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE int isVector(const sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE bool isLikeVector(sd::LongType const *shapeInfo, int &posOfNonUnityDim);

SD_LIB_EXPORT SD_HOST_DEVICE bool isCommonVector(const sd::LongType *shapeInfo, int &posOfNonUnityDim);

SD_LIB_EXPORT SD_HOST_DEVICE bool isRowVector(const sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE bool isColumnVector(sd::LongType const *shapeInfo);

/**
 * shape - input inShape is shape only, not shapeInfo
 * returns number of non-unity dimensions in inShape
 */
SD_LIB_EXPORT SD_HOST_DEVICE int numOfNonUnitDims(const int rank, const sd::LongType *inShape);

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */

SD_LIB_EXPORT SD_HOST_DEVICE int isMatrix(const sd::LongType *shape, int rank);

SD_INLINE SD_HOST_DEVICE int isMatrix(const sd::LongType *shapeInfo);
/**
 * Returns the shape portion of an information
 * buffer
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeOf(sd::LongType *shapeInfo);
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeOf(const sd::LongType *shapeInfo);

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */

template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE T *copyOf(sd::LongType length, T const *toCopy);

template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE T *copyOf(sd::LongType length, T const *toCopy, T *ret);

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */

template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE void copyTo(sd::LongType length, T const *from, T *to);
/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
SD_LIB_EXPORT SD_HOST_DEVICE void copyTo(int length, sd::LongType const *from, sd::LongType *to, sd::LongType *indexes);

/**
 * Permute the given strides
 * in the given rearrange order
 * @param toPermute the buffer to permute
 * @param shapeRank the length of the buffer to permute
 * @param rearrange the rearrange order (must be 0 based indexes
 * and all must be filled in)
 * @return the rearranged array
 */
// SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *permutedStrides(sd::LongType *toPermute, int shapeRank, sd::LongType
// *rearrange);

/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *slice(sd::LongType *shape);

SD_LIB_EXPORT SD_HOST_DEVICE int slices(sd::LongType *shapeBuffer);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *sliceOfShapeBuffer(sd::LongType sliceIdx, sd::LongType *shapeBuffer);
/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 3
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
SD_LIB_EXPORT SD_HOST_DEVICE int shapeInfoLength(int rank);

SD_LIB_EXPORT SD_HOST_DEVICE int shapeInfoLength(sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE int shapeInfoLength(const sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE size_t shapeInfoByteLength(int rank);

SD_LIB_EXPORT SD_HOST_DEVICE size_t shapeInfoByteLength(const sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE size_t shapeInfoByteLength(const sd::LongType *shapeInfo);

/**
 * Returns the rank portion of
 * an information buffer
 */
SD_LIB_EXPORT SD_HOST_DEVICE int rank(const sd::LongType *shapeInfo);
SD_LIB_EXPORT SD_HOST_DEVICE int rank(const int *shapeInfo);
SD_LIB_EXPORT SD_HOST_DEVICE int rank(const unsigned int *shapeInfo);

/**
 *  returns pointer on elementWiseStride
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *ews(sd::LongType *shapeInfo);

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
SD_LIB_EXPORT SD_HOST_DEVICE ShapeInformation *infoFromBuffer(sd::LongType *buffer);

/**
 * Returns the stride portion of an information
 * buffer
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *stride(sd::LongType *buffer);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *stride(const sd::LongType *buffer);

/**
 * Compute the length of the given shape
 */
SD_LIB_EXPORT SD_HOST_DEVICE bool isEmpty(const sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType length(const sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType length(std::initializer_list<int> &shape);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType length(std::initializer_list<sd::LongType> &shape);

/***
 * Returns the offset portion of an information buffer
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType offset(sd::LongType *buffer);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType &extra(sd::LongType *buffer);

/**
 * Returns the ordering
 * for this shape information buffer
 */
SD_LIB_EXPORT SD_HOST_DEVICE char order(const sd::LongType *buffer);

/**
 * Returns the type
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType type(const sd::LongType *shapeInfo);

/**
 * Returns the element wise stride for this information
 * buffer
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType elementWiseStride(const sd::LongType *shapeInfo);

/**
 * Returns the element wise stride for this information
 * buffer
 * relative to a dimension and ordering for a reduction index
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType reductionIndexElementWiseStride(sd::LongType *buffer, int *dimension,
                                                                          int dimensionLength);

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
SD_LIB_EXPORT SD_HOST_DEVICE int isScalar(const sd::LongType *info);

/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
SD_LIB_EXPORT SD_HOST_DEVICE int isScalar(volatile ShapeInformation *info);

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
template <typename T1, typename T2>
SD_LIB_EXPORT SD_HOST_DEVICE void removeIndex(T1 const *data, T2 const *indexes, sd::LongType dataLength,
                                              sd::LongType indexesLength, T1 *out);

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

template <typename T1, typename T2>
SD_LIB_EXPORT SD_HOST_DEVICE T1 *removeIndex(T1 const *data, T2 const *indexes, sd::LongType dataLength,
                                             sd::LongType indexesLength);

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
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *everyIndexBut(sd::LongType const *indexes, int indexesLength, int begin,
                                                         int end);

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
//#ifdef __CUDACC__
//    SD_DEVICE
//#endif
//    SD_LIB_EXPORT int tadOffset(shape::ShapeInformation *xInfo, int offset);

/**
 * Returns a shape
 * forces the given length to be 2.
 * @param shape the shape to modify
 * @param dimension the dimension (row or column)
 * for the shape to be returned as
 * @return the new shape
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *ensureVectorShape(sd::LongType *shape);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *createScalarShapeInfo();

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *createScalarShapeInfo(sd::LongType *ret);

/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE T *range(int from, int to, int increment);

/**
 * Range between from and two with an
 * increment of 1
 */
template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE T *range(int from, int to);

/**
 * Keep the given indexes
 * in the data
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *keep(volatile sd::LongType *data, int const *index, int indexLength,
                                                int dataLength);

/**
 * Generate reverse copy of the data
 * @param data
 * @param length
 * @return
 */

template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE T *reverseCopy(T const *data, sd::LongType length);

template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE void reverseCopyTo(T const *from, T *to, sd::LongType length);

template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE void reverseCopyTo(T const *from, T *to, sd::LongType *indexes, sd::LongType length);

template <typename T1, typename T2>
SD_LIB_EXPORT SD_HOST void convertT(T1 *from, T2 *to, sd::LongType length);
/**
 *
 * @param arr1
 * @param arr1Length
 * @param arr2
 * @param arr2Length
 * @return
 */
template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE T *concat(T const *arr1, sd::LongType const arr1Length, T const *arr2,
                                       sd::LongType const arr2Length);

/**
 *
 * @param numArrays
 * @param numTotalElements
 * @param arr
 * @param lengths
 * @return
 */
template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE T *concat(int const numArrays, int const numTotalElements, sd::LongType const **arr,
                                       sd::LongType const *lengths);

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
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType lengthPerSlice(int rank, sd::LongType const *shape, int const *dimension,
                                                         int dimensionLength);

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType sliceOffsetForTensor(int rank, int index, sd::LongType const *shape,
                                                               sd::LongType const *tensorShape, int tensorShapeLength,
                                                               int const *dimension, int dimensionLength);

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType sliceOffsetForTensor(int index, int tensorLength, int lengthPerSlice2);
/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
//    SD_LIB_EXPORT SD_HOST_DEVICE int offset(int index,
//                         int rank,
//                         shape::ShapeInformation *info,
//                         sd::LongType *dimension,
//                         int dimensionLength);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType tensorsAlongDimension(int rank, volatile int length,
                                                                volatile sd::LongType *shape, int *dimension,
                                                                int dimensionLength);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType tensorsAlongDimension(sd::LongType *shapeInfo, int *dimension,
                                                                int dimensionLength);

/**
 * Returns the tensor along dimension
 * for the given block index
 * @param blockSize
 * @param blockIdx
 * @param i
 * @return
 */
SD_LIB_EXPORT SD_HOST_DEVICE int tadForBlockIndex(int blockSize, int blockIdx, int i);

/**
 * Computes the number of tads per block
 *
 */
SD_LIB_EXPORT SD_HOST_DEVICE int tadsPerBlock(int blockSize, int tads);

//    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *tadShapeInfo(int index, sd::LongType *xShapeInfo, sd::LongType
//    *dimension,
//                                int dimensionLength);

/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *toShapeBuffer(ShapeInformation *info);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *toShapeBuffer(ShapeInformation *info, sd::LongType *ret);

/**
 * Returns the number of elements per thread
 */
//#ifdef __CUDACC__
//    SD_DEVICE
//#endif
//    int numElementsPerThread(int N);

/**
 * Returns the block starting index
 */
//#ifdef __CUDACC__
//    SD_DEVICE
//#endif
//    int blockStartingIndex(int N);

/**
 * Returns the thread starting index
 */
//#ifdef __CUDACC__
//    SD_DEVICE
//#endif
//    int threadStartingIndex(int N, int stride, int offset);

/**
 * Returns the thread ending index
 */
//#ifdef __CUDACC__
//    SD_DEVICE
//#endif
//    int threadEndingIndex(int N, int stride, int offset);

/**
 * Returns indexing information
 * for the current kernel invocation
 */
//#ifdef __CUDACC__
//    SD_DEVICE
//#endif
//    CurrentIndexing *currentIndex(int N, int offset, int stride);

/** Given an linear index, element wise stride
 * and the length of each tad
 * map a linear index to a tad
 * @param i the index to map
 * @param the element wise stride for the tads
 * @param numElementsPerTad the number of elements
 * per tad
 */
SD_LIB_EXPORT SD_HOST_DEVICE int tadIndex(int i, int elementWiseStride, int numElementsPerTad);

/**
 * Map a tad to a
 * reduction index.
 * @param tadIndexForOriginal the original tad index for the
 * split up problem (eg: split is dimension 3 mapping to a 2,3 problem)
 * @param tadsForReduced the number of tads for the shrunk down problem (eg: 2,3)
 * @param tadsForOriginal the number of tads for the smaller problem (eg: 3)
 */
SD_LIB_EXPORT SD_HOST_DEVICE int reductionIndexForTad(int tadIndexForOriginal, int tadsForReduced, int tadsForOriginal);

/**
 * Computes the number of tads
 * per reduce index for the
 * reduction tad.
 */
SD_LIB_EXPORT SD_HOST_DEVICE int tadsPerReduceIndex(int tadsForReduce, int tadsForOriginal);

/**
 * Maps a linear index to a reduction index
 * @param i the linear index to map
 * @param elementWiseStride the element wise stride
 * for the multiple problem
 * @param tadNum the number of tads for the shrunken problem
 * @param originalTadNum the tad number for the reduced version of the problem
 */
SD_LIB_EXPORT SD_HOST_DEVICE int reductionIndexForLinear(int i, int elementWiseStride, int numElementsPerTad,
                                                         int tadNum, int originalTadNum);

/**
 * Returns the prod of the data
 * up to the given length
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType prodLong(const sd::LongType *data, int length);

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

//    SD_LIB_EXPORT SD_HOST_DEVICE int rearMostLeftOverItem(sd::LongType *data,int length,sd::LongType *dimension,int
//    dimensionLength);

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

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType getOffset(const sd::LongType *shapeInfo, const sd::LongType *coords,
                                                    sd::LongType baseOffset = 0);
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType getOffset(const sd::LongType *shapeInfo, const int *coords,
                                                    sd::LongType baseOffset = 0);
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType getOffset(const sd::LongType *shapeInfo, const sd::Unsigned *coords,
                                                    sd::LongType baseOffset = 0);
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType getOffset(const sd::LongType *shapeInfo, const int *coords,
                                                    const int *dims);  // length of dims is equal to rank of shapeInfo

// all three arrays should have same rank
// all three arrays should have same dimensions or some of them are 1 (that is satisfy broadcasting principle), strides
// may be different shapeInfo1 - first array should have max length compared to rest of two arrays
SD_LIB_EXPORT SD_HOST_DEVICE void getOffsetBroadcast(const sd::LongType &startInd, const sd::LongType ind,
                                                     const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2,
                                                     const sd::LongType *shapeInfo3, const bool sameOffsets12,
                                                     const bool sameOffsets13, int *coords, sd::LongType &offset1,
                                                     sd::LongType &offset2, sd::LongType &offset3);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *createShapeInfo(sd::LongType *shape, sd::LongType *stride, int rank);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *createShapeInfo(sd::LongType *shape, sd::LongType *stride, int rank,
                                                           sd::LongType *buffer);

/**
 * Convert a linear index to the corresponding coordinates
 * for example if shape is {2, 4}, then index 5 corresponds to coordinates [1, 1]
 */
SD_LIB_EXPORT SD_HOST_DEVICE void index2coords(sd::LongType index, const sd::LongType *shapeInfo, sd::LongType *coords);
SD_LIB_EXPORT SD_HOST_DEVICE void index2coords(sd::LongType index, const sd::LongType *shapeInfo, int *coords);
SD_LIB_EXPORT SD_HOST_DEVICE void index2coords(sd::LongType index, const sd::LongType *shapeInfo, sd::Unsigned *coords);
SD_LIB_EXPORT SD_HOST_DEVICE void index2coords(sd::LongType index, const int rank, const sd::LongType *shape,
                                               sd::LongType *coords);
SD_LIB_EXPORT SD_HOST_DEVICE void index2coords(sd::LongType index, const int rank, const sd::LongType *shape,
                                               int *coords);

SD_LIB_EXPORT SD_HOST_DEVICE void index2coordsCPU(const sd::LongType &startIndex, const sd::LongType &index,
                                                  const sd::LongType *shapeInfo, sd::LongType *coords);
SD_LIB_EXPORT SD_HOST_DEVICE void index2coordsCPU(const sd::LongType &startIndex, const sd::LongType &index,
                                                  const sd::LongType *shapeInfo, int *coords);
// SD_LIB_EXPORT SD_HOST_DEVICE void index2coordsCPU(const sd::LongType& startIndex, const sd::LongType& index, const
// sd::LongType *shapeInfo, const int* dims, sd::LongType *coords);

/**
 * take into account only dimensions stored in tadDims, tadDims must be sorted in increasing order!
 */
SD_LIB_EXPORT SD_HOST_DEVICE void index2coords(sd::LongType index, const sd::LongType *shapeInfo, const int *dims,
                                               const int dimsLen, int *coords);

/**
 * Convert coordinates to the corresponding linear index (sequence number in other words)
 * for example if shape is {2, 4} and coordinates [1, 1] then index 5 is returned
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const sd::LongType *coords);
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const int *coords);
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const sd::Unsigned *coords);
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType coords2index(const int rank, const sd::LongType *shape, const int *coords);
/**
 * take into account only dimensions stored in tadDims, tadDims must be sorted in increasing order!
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const int *dims,
                                                       const int dimsSize, const int *coords);

/**
 * increment n-dimensional array by one iteration by changing coord appropriately
 * for example we have array with shape {2, 3}:
 * - if input coord = {0,1}, then output coord = {0,2}
 * - if input coord = {0,2}, then output coord = {1,0}
 * so the aim is to produce following subsequence of coord: {0,0}, {0,1}, {0,2}, {1,0}, {1,1}, {1,2}
 */

/* calculates an array buffer offset for given "index" using following formula: offset = coord_0*stride_0 +
 * coord_1*stride_1 + ... + coord_{rank-1}*stride_{rank-1}
 */
SD_LIB_EXPORT SD_HOST_DEVICE sd::Unsigned getIndexOffset(sd::Unsigned index, const sd::Unsigned *shapeInfo);
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType getIndexOffset(sd::LongType index, const sd::LongType *shapeInfo);
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType indexOffset(sd::LongType index, const sd::LongType *lShapeInfo,
                                                      const sd::Unsigned *uShapeInfo, const bool useUnsigned);

SD_LIB_EXPORT SD_HOST_DEVICE void printShapeInfo(const sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE void printShapeInfoLinear(const sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE void printShapeInfoLinear(const char *msg, const sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE void printShapeInfoLinear(const char *msg, int rank, const sd::LongType *shape,
                                                       const sd::LongType *strides);

SD_LIB_EXPORT SD_HOST_DEVICE void printIntArray(const sd::LongType *arr, const int length);
SD_LIB_EXPORT SD_HOST_DEVICE void printIntArray(const int *arr, const int length);

SD_LIB_EXPORT SD_HOST_DEVICE void printArray(float *arr, int length);

template <typename T>
SD_LIB_EXPORT SD_HOST_DEVICE void printArray(T *arr, int length, const char *message);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeBufferOfNpy(int rank, unsigned int *shape, bool fortranOrder);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeBufferOfNpy(cnpy::NpyArray arr);

//    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeBufferOfNpyBuffer(char *buffer);

// this function checks the consistence of dimensions with array rank (negative dimensions, too large dimensions, too
// big number of dimensions) also sort input array of dimensions, this operation is also necessary for creating TAD
// object
SD_LIB_EXPORT SD_HOST void checkDimensions(const int rank, std::vector<int> &dimensions);

// function calculates linear index of array min, min is sub-array of max, index to be returned is min-array's index and
// corresponds to maxIdx of max array dimsToExclude - should be sorted in increasing order
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType subArrayIndex(const sd::LongType maxIdx, const sd::LongType *maxShapeInfo,
                                                        const sd::LongType *minShapeInfo,
                                                        const int *dimsToExclude = nullptr, const int dimsLen = -1);

// function calculates absolute offset of min array, min is sub-array of max, offset to be returned corresponds to
// maxIdx of max array dimsToExclude - should be sorted in increasing order
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType subArrayOffset(const sd::LongType maxIdx, const sd::LongType *maxShapeInfo,
                                                         const sd::LongType *minShapeInfo,
                                                         const int *dimsToExclude = nullptr, const int dimsLen = -1);

// max array is outer for min array, min array is sub-array of max array
// function calculates the coordinates of min array (and saves them into minIdxs) given coordinates of max array
// (already stored in maxIdxs) dimsToExclude - should be sorted in increasing order dimsLen - length of dimsToExclude,
// if not set (= -1), then it is calculated as maxRank - minRank
SD_LIB_EXPORT SD_HOST_DEVICE void maxIndToMinInd(int *maxIdxs, int *minIdxs, const sd::LongType *maxShapeInfo,
                                                 const sd::LongType *minShapeInfo, const int *dimsToExclude = nullptr,
                                                 const int dimsLen = -1);

// calculate indexes of max-array, these output indexes correspond to one minIdx index of min-array which is sub-array
// of max-array dimsToExclude - should be sorted in increasing order
SD_LIB_EXPORT SD_HOST_DEVICE int outerArrayIndexes(int *maxIdxs, const sd::LongType minIdx,
                                                   const sd::LongType *maxShapeInfo, const sd::LongType *minShapeInfo,
                                                   const int *dimsToExclude = nullptr);

// calculate offsets of max-array, these offsets correspond to one minIdx index of min-array which is sub-array of
// max-array maxOffsets - will contain calculated offsets of max-array, buffer for maxOffsets should be allocated
// beforehand dimsToExclude - should be sorted in increasing order memBuff - auxiliary memory buffer (size = 2 *
// max_rank) for coordinates and increments storing, should be allocated beforehand
SD_LIB_EXPORT SD_HOST_DEVICE int outerArrayOffsets(sd::LongType *maxOffsets, const sd::LongType minIdx,
                                                   const sd::LongType *maxShapeInfo, const sd::LongType *minShapeInfo,
                                                   int *memBuff, const int *dimsToExclude = nullptr);

// calculates offsets for entities (elements or sub-arrays), shape in context of sub-array means dimensions excluded
// from outer array rank is equal to size of shape
SD_LIB_EXPORT void calcOffsets(const int rank, const sd::LongType *shape, const sd::LongType *strides,
                               sd::LongType *offsets, const char order = 'c');
SD_LIB_EXPORT void calcOffsets(const sd::LongType *shapeInfo, sd::LongType *offsets, const char order = 'c');
// SD_LIB_EXPORT void calcOffsets(const sd::LongType *xShapeInfo, sd::LongType*& xOffsets, const sd::LongType
// *yShapeInfo, sd::LongType*& yOffsets, const char order = 'c'); SD_LIB_EXPORT void calcOffsets(const sd::LongType
// *xShapeInfo, sd::LongType*& xOffsets, const sd::LongType *yShapeInfo, sd::LongType*& yOffsets, const sd::LongType*
// zShapeInfo, sd::LongType*& zOffsets, const char order = 'c');
SD_LIB_EXPORT SD_HOST_DEVICE void shapeOldScalar(sd::DataType dtype, sd::LongType *const buffer, const char order);

// deduce order and element-wise stride
// if array is scalar or unit length vector then ews = 1 and order is preserved
// if array is common vector then ews = stride of non-unity dimension and order is preserved
// if strides are normal/contiguous then ews = 1 and corresponding order is set, otherwise ews = 0 and order is
// preserved
SD_LIB_EXPORT SD_HOST_DEVICE void checkStridesEwsAndOrder(sd::LongType *shapeInfo, const char proposedOrder,
                                                          const int numOfNonUnitDims,
                                                          const sd::LongType *shapeNoUnities,
                                                          const sd::LongType *stridesNoUnities);
SD_LIB_EXPORT SD_HOST_DEVICE void checkStridesEwsAndOrder(sd::LongType *shapeInfo);

/**
 * processes whole set of sub-arrays
 * evaluates shapeInfo of sub-arrays (all sub-arrays have the same shapeInfo) and their buffer offsets (each sub-array
 * has its own unique offset from original this-buffer) arguments: wholeShapeInfo - original shapeInfo of whole array
 * numOfSubArrs - number of sub-arrays, size of subArrOffsets is equal to numOfSubArrs
 * dimsSize - size of dimsToExclude, if dimsSize = array rank or dimsSize = 0 it means sub-array is whole array, copy of
 * wholeShapeInfo and one zero offset will be returned dimsToExclude - MUST BE SORTED, dimensions to evaluate sub-array
 * along, i.e. when shape is [2,3,4,5] and dimsToExclude={0,2}, then there will be 8 sub-arrays with shape [3,5]
 * subArrShapeInfo    - output argument, contains shapeInfo (same for all sub-arrays)
 * subArrOffsets      - output argument, contains successive sub-arrays offsets from original this-buffer
 * keepUnitiesInShape - if false then eliminate unities from sub-array shapeInfo, for example {1,a,1,b} -> {a,b}
 */
SD_LIB_EXPORT SD_HOST_DEVICE void calcSubArrsShapeInfoAndOffsets(
    const sd::LongType *wholeShapeInfo, const sd::LongType numOfSubArrs, const int dimsSize, const int *dimsToExclude,
    sd::LongType *subArrShapeInfo, sd::LongType *subArrOffsets, bool keepUnitiesInShape = false);

/**
 * processes only one sub-array, evaluates shapeInfo of sub-array and its buffer offset from original array
 * arguments:
 * idx - input argument, intervals of indexes which define the sub-array to point on,
 *        when isStrided = false then idx has form {dim0Start,dim0End,  dim1Start,dim1End, ....} and length (2 *
 * maxRank) when isStrided = true  then idx has form {dim0Start,dim0End,dim0Stride,  dim1Start,dim1End,dim1Stride, ....}
 * and length (3 * maxRank) when (dimStart == dimEnd) then whole range will be used for current dimension maxShapeInfo -
 * input argument, shapeInfo of original array minShapeInfo - output argument, shapeInfo of sub-array to be deduced
 * minOffset - output argument, offset of sub-array buffer offsets from original buffer
 * keepUnitiesInShape - input argument, if false then eliminate unities from sub-array shapeInfo, for example {1,a,1,b}
 * -> {a,b} isStrided - input argument, if true then idx has length (3 * this->rankOf()) and contains additional stride
 * numbers which correspond to stride between dimStart and dimEnd, numOfUntiesInMinShape - input argument, number of
 * occurrences in idx when (dimEnd - dimStart) = 1
 */
SD_LIB_EXPORT void calcSubArrShapeInfoAndOffset(const sd::LongType *idx, const sd::LongType *maxShapeInfo,
                                                sd::LongType *minShapeInfo, sd::LongType &minOffset,
                                                const bool keepUnitiesInShape = false, const bool isStrided = false,
                                                const int numOfUntiesInMinShape = 0);

/**
 * for example inShapeInfo is {3, 2,1,4, 4,4,1, 16384,1,99}
 * then output shapeNoUnities will contain {2,4, 4,1} - that is only shape and strides, no rank/type/ews/order
 * stridesNoUnities will point on strides in shapeNoUnities that is on {4,1}
 * returns number of non-unity dimensions in inShapeInfo
 * if there is no unities in inShapeInfo, then no copy procedure will be performed and shapeNoUnities/stridesNoUnities
 * will point on corresponding places in inShapeInfo
 */
SD_LIB_EXPORT SD_HOST_DEVICE int excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo,
                                                             sd::LongType *&shapeNoUnities,
                                                             sd::LongType *&stridesNoUnities);

/**
 * for example inShapeInfo is {3, 2,1,3,1,4,  12,12,4,4,1, 16384,1,99}, dimsToExclude(points on unity dimensions) =
 * {1,3}, dimsSize = 2 then outShapeInfo will contain {3, 2,3,4, 12,4,1, 16384,1,99}
 */
SD_INLINE SD_HOST_DEVICE void excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo, const int *dimsToExclude,
                                                          const int dimsSize, sd::LongType *outShapeInfo);

/**
 * get stride over contiguous axis (contiguous axis must have stride = 1)
 * for example when inShapeInfo is {4, 2,5,4,3,  60,1,5,20, 16384,0,99} then output is 5 (that is smallest stride in
 * inShapeInfo except those equal to 1)
 */
// SD_INLINE SD_HOST_DEVICE sd::LongType strideOverContigAxis(const int axis, const sd::LongType* inShapeInfo);

// END HEADERS

// BEGIN IMPLEMENTATIONS

#ifdef __CUDACC__
/**
 * BEWARE: THIS METHOD DOES NOT CHECKS ALLOCATION BOUNDARIES
 */
SD_DEVICE SD_INLINE sd::LongType *cuMalloc(sd::LongType *buffer, long size) {
  sd::LongType *ret = buffer;
  ret += (threadIdx.x * size);
  return ret;
}
#endif

/**
 * Length of a tad given
 * the shape information
 */
SD_INLINE SD_HOST_DEVICE sd::LongType tadLength(const sd::LongType *shapeInfo, int *dimension, int dimensionLength) {
  if (dimensionLength == 1) {
    return shape::shapeOf(shapeInfo)[dimension[0]];
  } else {
    sd::LongType ret = 1;
    for (int i = 0; i < shape::rank(shapeInfo); i++) {
      for (int j = 0; j < dimensionLength; j++) {
        if (i == dimension[j]) ret *= shape::shapeOf(shapeInfo)[dimension[j]];
      }
    }
    return ret;
  }
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
SD_INLINE SD_HOST_DEVICE int tadElementWiseStride(sd::LongType *shapeInfo, int *dimension, int dimensionLength) {
  return reductionIndexElementWiseStride(shapeInfo, dimension, dimensionLength);
}

SD_INLINE SD_HOST_DEVICE bool shapeEquals(const int shape1Rank, const sd::LongType *shape1, const int shape2Rank,
                                          const sd::LongType *shape2) {
  if (shape1Rank != shape2Rank) return false;
  // rank not equals
  for (int i = 0; i < shape1Rank; i++) {
    if (shape1[i] != shape2[i]) return false;
  }

  return true;
}

SD_INLINE SD_HOST_DEVICE bool shapeEquals(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2) {
  return shape::shapeEquals(shape::rank(shapeInfo1), shape::shapeOf(const_cast<sd::LongType *>(shapeInfo1)),
                            shape::rank(shapeInfo2), shape::shapeOf(const_cast<sd::LongType *>(shapeInfo2)));
}

SD_INLINE SD_HOST_DEVICE bool shapeEquals(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2,
                                          const sd::LongType *shapeInfo3) {
  return shape::shapeEquals(shapeInfo1, shapeInfo2) && shape::shapeEquals(shapeInfo1, shapeInfo3);
}

SD_INLINE SD_HOST_DEVICE bool strideEquals(int const shape1Rank, sd::LongType const *shape1, int const shape2Rank,
                                           sd::LongType const *shape2) {
  if (shape1Rank != shape2Rank) return false;
  // rank not equals
  for (int i = 0; i < shape1Rank; i++) {
    if (shape1[i] != shape2[i]) return false;
  }

  return true;
}

SD_INLINE SD_HOST_DEVICE bool strideEquals(sd::LongType const *shapeInfo1, sd::LongType const *shapeInfo2) {
  return shape::strideEquals(shape::rank(shapeInfo1), shape::stride(shapeInfo1), shape::rank(shapeInfo2),
                             shape::stride(shapeInfo2));
}

SD_INLINE SD_HOST_DEVICE bool strideEquals(sd::LongType const *stride1, int const rank1, sd::LongType const *stride2,
                                           int const rank2) {
  if (rank1 != rank2) return false;

  for (int i = 0; i < rank1; i++) {
    if (stride1[i] != stride2[i]) return false;
  }

  return true;
}

SD_INLINE SD_HOST_DEVICE sd::LongType *computeResultShape(sd::LongType const *originalShapeBuffer, int *dimension,
                                                          int dimensionLength) {
  sd::LongType *retShape;
  int retShapeLength;
  if (dimensionLength == 1 && dimension[0] == 2147483647) {
    retShape = new sd::LongType[2];
    retShape[0] = 1;
    retShape[1] = 1;
    retShapeLength = 2;
  } else {
    retShape = shape::removeIndex<sd::LongType, int>(shape::shapeOf(originalShapeBuffer), dimension,
                                                     shape::shapeInfoLength(shape::rank(originalShapeBuffer)),
                                                     dimensionLength);
    retShapeLength = shape::rank(originalShapeBuffer) - dimensionLength;
  }
  // ensure vector is proper shape
  if (retShapeLength == 1) {
    if (dimension[0] == 0) {
      auto newRetShape = new sd::LongType[2]{1, retShape[0]};
      delete[] retShape;
      retShape = newRetShape;
      retShapeLength = 2;
    } else {
      auto newRetShape = new sd::LongType[2]{retShape[0], 1};
      delete[] retShape;
      retShape = newRetShape;
      retShapeLength = 2;
    }
  } else if (retShapeLength == 0) {
    auto newRetShape = new sd::LongType[2]{1, 1};
    delete[] retShape;
    retShape = newRetShape;
    retShapeLength = 2;
  }

  auto ret = shape::shapeBuffer(retShapeLength, sd::ArrayOptions::dataType(originalShapeBuffer), retShape);
  delete[] retShape;

  return ret;
}

SD_INLINE SD_HOST_DEVICE sd::LongType *shapeInfoOnlyShapeAndStride(const sd::LongType *shapeInfo,
                                                                   sd::LongType *dimension, int dimensionLength,
                                                                   bool reverseCopyStride, sd::LongType *buffer) {
  sd::LongType *theShape = shape::shapeOf(shapeInfo);
  sd::LongType *theStride = shape::stride(shapeInfo);
  int rank = dimensionLength == 1 ? 2 : dimensionLength;
  sd::LongType *ret = buffer;
  // set the rank
  ret[0] = rank;
  sd::LongType *retShape = shape::shapeOf(ret);
  sd::LongType *retStride = shape::stride(ret);
  int len = rank;

  if (dimensionLength == 1) {
    if (shape::isMatrix(theShape, shape::rank(shapeInfo))) {
      if (dimension[0] == 0) {
        sd::LongType newStride[2] = {theStride[dimension[0]], 1};
        sd::LongType newShape[2] = {theShape[dimension[0]], 1};
        retShape[0] = newShape[0];
        retShape[1] = newShape[1];
        retStride[0] = newStride[0];
        retStride[1] = newStride[1];
      } else {
        sd::LongType newStride[2] = {theStride[dimension[0]], 1};
        sd::LongType newShape[2] = {theShape[dimension[0]], 1};
        retShape[0] = newShape[0];
        retShape[1] = newShape[1];
        retStride[0] = newStride[0];
        retStride[1] = newStride[1];
      }
    } else {
      sd::LongType newStride[2] = {1, theStride[dimension[0]]};
      sd::LongType newShape[2] = {1, theShape[dimension[0]]};
      retShape[0] = newShape[0];
      retShape[1] = newShape[1];
      retStride[0] = newStride[0];
      retStride[1] = newStride[1];
    }

  } else {
    sd::LongType *newIndexes = dimension;
    if (reverseCopyStride)
      shape::reverseCopyTo(theStride, retStride, newIndexes, len);
    else
      shape::copyTo(len, theStride, retStride, newIndexes);
    shape::copyTo(len, theShape, retShape, newIndexes);
  }

  ret[shape::shapeInfoLength(rank) - 1] = shape::order(shapeInfo);
  return ret;
}

SD_INLINE SD_HOST_DEVICE sd::LongType *shapeInfoOnlyShapeAndStride(const sd::LongType *shapeInfo,
                                                                   sd::LongType *dimension, int dimensionLength,
                                                                   bool reverseCopyStride) {
  int rank = dimensionLength == 1 ? 2 : dimensionLength;

  traceNew(4);

  sd::LongType *ret = new sd::LongType[shape::shapeInfoLength(rank)];
  return shapeInfoOnlyShapeAndStride(shapeInfo, dimension, dimensionLength, reverseCopyStride, ret);
}

SD_INLINE SD_HOST_DEVICE sd::LongType *createShapeInfo(sd::LongType *shape, sd::LongType *stride, int rank) {
  traceNew(5);

  sd::LongType *ret = new sd::LongType[shape::shapeInfoLength(rank)];

  return createShapeInfo(shape, stride, rank, ret);
}

SD_INLINE SD_HOST_DEVICE sd::LongType *createShapeInfo(sd::LongType *shape, sd::LongType *stride, int rank,
                                                       sd::LongType *buffer) {
  buffer[0] = rank;
  sd::LongType *retShape = shape::shapeOf(buffer);
  sd::LongType *retStride = shape::stride(buffer);
  for (int i = 0; i < rank; i++) {
    retShape[i] = shape[i];
    retStride[i] = stride[i];
  }

  return buffer;
}

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
SD_INLINE SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank, int startNum) {
  // if (isVector(shape, rank)) {

  //     traceNew(5);

  //     sd::LongType *ret = new sd::LongType[2];
  //     for (int i = 0; i < 2; i++)
  //         ret[i] = 1;
  //     return ret;

  // }

  int dimensions = rank;

  traceNew(5);

  sd::LongType *stride = new sd::LongType[dimensions];
  sd::LongType st = startNum;
  for (int j = 0; j < rank; j++) {
    stride[j] = st;
    st *= shape[j];
  }

  return stride;
}

SD_INLINE SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank, int startNum,
                                                          sd::LongType *ret) {
  // if (isVector(shape, rank)) {
  //     for (int i = 0; i < rank; i++)
  //         ret[i] = 1;
  //     return ret;

  // }

  // int dimensions = rank;

  sd::LongType st = startNum;
  for (int j = 0; j < rank; j++) {
    ret[j] = st;
    st *= shape[j];
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
SD_INLINE SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, int rank, int startNum) {
  traceNew(7);

  sd::LongType *stride = new sd::LongType[rank];

  if (rank == 1) {
    stride[0] = 1;
    return stride;
  }

  // if (shape::isVector(shape, rank)) {
  //     for (int i = 0; i < 2; i++)
  //         stride[i] = 1;
  //     return stride;

  // }

  sd::LongType st = startNum;
  for (int j = rank - 1; j >= 0; j--) {
    stride[j] = st;
    st *= shape[j];
  }

  return stride;
}

SD_INLINE SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, int rank, int startNum,
                                                   sd::LongType *ret) {
  if (rank == 1) {
    ret[0] = 1;
    return ret;
  }

  // if (shape::isVector(shape, rank)) {
  //     for (int i = 0; i < 2; i++)
  //         ret[i] = 1;
  //     return ret;

  // }

  sd::LongType st = startNum;
  for (int j = rank - 1; j >= 0; j--) {
    ret[j] = st;
    st *= shape[j];
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
SD_INLINE SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank) {
  return calcStridesFortran(shape, rank, 1);
}

SD_INLINE SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank, sd::LongType *ret) {
  return calcStridesFortran(shape, rank, 1, ret);
}

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
SD_INLINE SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, int rank) {
  return calcStrides(shape, rank, 1);
}

SD_INLINE SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, int rank, sd::LongType *ret) {
  return calcStrides(shape, rank, 1, ret);
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE void updateStrides(sd::LongType *shapeInfo, const char order) {
  int rank = shapeInfo[0];
  int doubleRank = 2 * rank;

  if (rank > 0) {
    if (order == 'c') {
      shapeInfo[doubleRank] = 1;  // set unity as last stride for c order
      for (int j = 1; j < rank; ++j) {
        shapeInfo[doubleRank - j] = shapeInfo[doubleRank - j + 1] * shapeInfo[rank + 1 - j];
      }
    } else {
      shapeInfo[rank + 1] = 1;  // set unity as first stride for f order
      for (int j = rank + 1; j < doubleRank; ++j) {
        shapeInfo[j + 1] = shapeInfo[j] * shapeInfo[j - rank];
      }
    }
  }
  // set last 2 elements in shapeInfo
  shapeInfo[doubleRank + 2] = 1;
  shapeInfo[doubleRank + 3] = (int)order;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE void updateStrides(const int rank, const sd::LongType *shapeOnly, sd::LongType *stridesOnly,
                                            const char order) {
  if (rank > 0) {
    if (order == 'c') {
      stridesOnly[rank - 1] = 1;  // set unity as last stride for c order
      for (int j = 1; j < rank; ++j) stridesOnly[rank - 1 - j] = stridesOnly[rank - j] * shapeOnly[rank - j];
    } else {
      stridesOnly[0] = 1;  // set unity as first stride for f order
      for (int j = 1; j < rank; ++j) {
        stridesOnly[j] = stridesOnly[j - 1] * shapeOnly[j - 1];
      }
    }
  }
}

// check whether input dimensions are permuted, not permuted dimensions order have to be 0,....,rank-1
template <typename T>
SD_INLINE SD_HOST_DEVICE bool isDimPermuted(const T *dimensions, const sd::LongType dimSize) {
  for (int i = 0; i < dimSize - 1; ++i)
    if (dimensions[i] > dimensions[i + 1]) return true;

  return false;
}

/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
SD_INLINE SD_HOST_DEVICE ShapeInformation *shapeCopy(ShapeInformation *toCopy) {
  auto copy = new ShapeInformation;

  traceNew(8);

  copy->shape = new sd::LongType[toCopy->rank];

  memcpy(copy->shape, toCopy->shape, toCopy->rank * sizeof(sd::LongType));

  traceNew(9);

  copy->stride = new sd::LongType[toCopy->rank];
  for (int i = 0; i < toCopy->rank; i++) {
    copy->stride[i] = toCopy->stride[i];
  }
  copy->order = toCopy->order;
  copy->rank = toCopy->rank;
  copy->offset = toCopy->offset;
  copy->elementWiseStride = toCopy->elementWiseStride;
  return copy;
}

SD_INLINE SD_HOST_DEVICE int computeElementWiseStride(int rank, sd::LongType const *shape, sd::LongType const *stride,
                                                      int isFOrder) {
  if (rank == 0) return 1;

  if (shape::isVector(shape, rank)) {
    return stride[rank - 1];
  }

  else {
    int oldnd;
    sd::LongType *oldDims = shape::copyOf(rank, shape);
    sd::LongType *oldStrides = shape::copyOf(rank, stride);
    sd::LongType np, op, last_stride;
    sd::LongType oldStart, oldStop, ok, newStart, newStop, nk;

    traceNew(10);

    auto newStrides = new sd::LongType[rank];
    oldnd = 0;
    // set the shape to be 1 x length
    int newShapeRank = 2;
    auto newShape = new sd::LongType[newShapeRank];
    newShape[0] = 1;
    newShape[1] = shape::prodLong(shape, rank);

    /*
     * Remove axes with dimension 1 from the old array. They have no effect
     * but would need special cases since their strides do not matter.
     */
    for (oldStart = 0; oldStart < rank; oldStart++) {
      if (shape[oldStart] != 1) {
        oldDims[oldnd] = shape[oldStart];
        oldStrides[oldnd] = stride[oldStart];
        oldnd++;
      }
    }

    np = 1;
    for (newStart = 0; newStart < newShapeRank; newStart++) {
      np *= newShape[newStart];
    }
    op = 1;
    for (oldStart = 0; oldStart < oldnd; oldStart++) {
      op *= oldDims[oldStart];
    }
    if (np != op) {
      /* different total sizes; no hope */
      delete[] newStrides;
      delete[] newShape;
      delete[] oldStrides;
      delete[] oldDims;
      return 0;
    }

    if (np == 0) {
      /* the current code does not handle 0-sized arrays, so give up */
      delete[] newStrides;
      delete[] newShape;
      delete[] oldStrides;
      delete[] oldDims;
      return 0;
    }

    /* oldStart to oldStop and newStart to newStop give the axis ranges currently worked with */
    oldStart = 0;
    oldStop = 1;
    newStart = 0;
    newStop = 1;
    while (newStart < newShapeRank && oldStart < oldnd) {
      np = newShape[newStart];
      op = oldDims[oldStart];

      while (np != op) {
        if (np < op) {
          /* Misses trailing 1s, these are handled later */
          np *= newShape[newStop++];
        } else {
          op *= oldDims[oldStop++];
        }
      }

      /* Check whether the original axes can be combined */
      for (ok = oldStart; ok < oldStop - 1; ok++) {
        if (isFOrder) {
          if (oldStrides[ok + 1] != oldDims[ok] * oldStrides[ok]) {
            /* not contiguous enough */
            delete[] newStrides;
            delete[] newShape;
            delete[] oldStrides;
            delete[] oldDims;
            return 0;
          }
        } else {
          /* C order */
          if (oldStrides[ok] != oldDims[ok + 1] * oldStrides[ok + 1]) {
            /* not contiguous enough */
            delete[] newStrides;
            delete[] newShape;
            delete[] oldStrides;
            delete[] oldDims;
            return 0;
          }
        }
      }

      /* Calculate new strides for all axes currently worked with */
      if (isFOrder) {
        newStrides[newStart] = oldStrides[oldStart];
        for (nk = newStart + 1; nk < newStop; nk++) {
          newStrides[nk] = newStrides[nk - 1] * newShape[nk - 1];
        }
      } else {
        /* C order */
        newStrides[newStop - 1] = oldStrides[oldStop - 1];
        for (nk = newStop - 1; nk > newStart; nk--) {
          newStrides[nk - 1] = newStrides[nk] * newShape[nk];
        }
      }
      newStart = newStop++;
      oldStart = oldStop++;
    }

    /*
     * Set strides corresponding to trailing 1s of the new shape.
     */
    if (newStart >= 1) {
      last_stride = newStrides[newStart - 1];
    } else {
      last_stride = stride[rank - 1];
    }
    if (isFOrder) {
      if (newStart >= 1) last_stride *= newShape[newStart - 1];
    }
    for (nk = newStart; nk < newShapeRank; nk++) {
      newStrides[nk] = last_stride;
    }
    // returns the last element of the new stride array
    int ret = last_stride;
    delete[] newStrides;
    delete[] newShape;
    delete[] oldStrides;
    delete[] oldDims;
    return ret;
  }
}

SD_INLINE SD_HOST_DEVICE int computeElementWiseStride(int rank, sd::LongType const *shape, sd::LongType const *stride,
                                                      int isFOrder, sd::LongType const *dimension,
                                                      int dimensionLength) {
  if (dimensionLength == 1) {
    return stride[dimension[0]];
  }
  return 0;
}

/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
SD_INLINE SD_HOST_DEVICE sd::LongType *shapeBuffer(int rank, sd::DataType dtype, sd::LongType const *shape) {
  sd::LongType *stride = shape::calcStrides(shape, rank);

  traceNew(11);

  auto shapeInfo = new shape::ShapeInformation();
  shapeInfo->shape = const_cast<sd::LongType *>(shape);
  shapeInfo->stride = stride;
  shapeInfo->offset = 0;
  shapeInfo->rank = rank;
  int elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);
  shapeInfo->order = 'c';
  shapeInfo->elementWiseStride = elementWiseStride;
  auto shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
  delete[] stride;
  delete shapeInfo;
  sd::ArrayOptions::setDataType(shapeInfoBuffer, dtype);
  return shapeInfoBuffer;
}

/**
 * This is special method, it returns ONLY 2D shapebuffer.
 *
 * This method is used only for SoftMax
 */
SD_INLINE SD_HOST_DEVICE sd::LongType *shapeBuffer(int rank, sd::DataType dtype, sd::LongType const *shape,
                                                   sd::LongType *buffer) {
  sd::LongType stride[SD_MAX_RANK];
  shape::calcStrides(shape, rank, stride);

  shape::ShapeInformation shapeInfo;
  shapeInfo.shape = const_cast<sd::LongType *>(shape);
  shapeInfo.stride = stride;
  shapeInfo.offset = 0;
  shapeInfo.rank = rank;
  auto elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);

  shapeInfo.order = 'c';
  shapeInfo.elementWiseStride = elementWiseStride;
  shape::toShapeBuffer(&shapeInfo, buffer);
  sd::ArrayOptions::setDataType(buffer, dtype);
  return buffer;
}

/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
SD_INLINE SD_HOST_DEVICE sd::LongType *shapeBufferFortran(int rank, sd::DataType dtype, sd::LongType const *shape) {
  auto stride = shape::calcStridesFortran(shape, rank);

  traceNew(12);

  auto shapeInfo = new shape::ShapeInformation();
  shapeInfo->shape = const_cast<sd::LongType *>(shape);
  shapeInfo->stride = stride;
  shapeInfo->offset = 0;
  shapeInfo->rank = rank;
  int elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);

  shapeInfo->order = 'f';
  shapeInfo->elementWiseStride = elementWiseStride;
  auto shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
  delete[] stride;
  delete shapeInfo;
  sd::ArrayOptions::setDataType(shapeInfoBuffer, dtype);
  return shapeInfoBuffer;
}

SD_INLINE SD_HOST_DEVICE sd::LongType *shapeBufferFortran(int rank, sd::DataType dtype, sd::LongType const *shape,
                                                          sd::LongType *output) {
  sd::LongType stride[SD_MAX_RANK];
  shape::calcStridesFortran(shape, rank, stride);

  shape::ShapeInformation shapeInfo;
  shapeInfo.shape = const_cast<sd::LongType *>(shape);
  shapeInfo.stride = stride;
  shapeInfo.offset = 0;
  shapeInfo.rank = rank;
  auto elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);

  shapeInfo.order = 'f';
  shapeInfo.elementWiseStride = elementWiseStride;
  shape::toShapeBuffer(&shapeInfo, output);
  sd::ArrayOptions::setDataType(output, dtype);
  return output;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const sd::LongType *indices) {
  sd::LongType index, shift = 1;
  ;

  index = indices[shapeInfo[0] - 1];
  for (sd::Unsigned i = shapeInfo[0]; i > 1; --i) {
    shift *= shapeInfo[i];
    index += shift * indices[i - 2];
  }

  return index;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const int *coords) {
  sd::LongType index, shift = 1;
  ;

  index = coords[shapeInfo[0] - 1];
  for (sd::Unsigned i = shapeInfo[0]; i > 1; --i) {
    shift *= shapeInfo[i];
    index += shift * coords[i - 2];
  }

  return index;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const sd::Unsigned *coords) {
  sd::LongType index, shift = 1;
  ;

  index = coords[shapeInfo[0] - 1];
  for (sd::Unsigned i = shapeInfo[0]; i > 1; --i) {
    shift *= shapeInfo[i];
    index += shift * coords[i - 2];
  }

  return index;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType coords2index(const int rank, const sd::LongType *shape, const int *indices) {
  sd::LongType index, shift = 1;
  ;

  index = indices[rank - 1];
  for (sd::Unsigned i = rank - 1; i >= 1; --i) {
    shift *= shape[i];
    index += shift * indices[i - 1];
  }

  return index;
}

SD_INLINE SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const int *dims, const int dimsLen,
                                                   const int *coords) {
  sd::LongType index, shift = 1;
  ;

  index = coords[dims[dimsLen - 1]];
  for (sd::Unsigned i = dimsLen - 1; i >= 1; --i) {
    shift *= shapeInfo[dims[i]];
    index += shift * coords[i - 1];
  }

  return index;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE void fill(T *buffer, T value, sd::LongType length) {
  PRAGMA_OMP_SIMD
  for (int e = 0; e < length; e++) buffer[e] = value;
}

// //////////////////////////////////////////////////////////////////////
//     SD_INLINE SD_HOST_DEVICE sd::LongType getIndexOffset(sd::LongType index, const sd::LongType *shapeInfo,
//     sd::LongType arrLen) {

//         const sd::LongType ews = shapeInfo[shapeInfo[0] + shapeInfo[0] + 2];

//         if(ews > 0 && order(shapeInfo) == 'c')
//            if (ews == 1)
//                return index;
//            else
//                return ews * index;

//         sd::LongType offset = 0;
//         sd::LongType rank = shapeInfo[0];
//         for(int i = 1; i <= shapeInfo[0]; ++i) {
//             arrLen /= shapeInfo[i];
//             if(arrLen > 0 && shapeInfo[i] > 1) {
//                 offset += (index / arrLen) * shapeInfo[i + rank];
//                 index %= arrLen;
//             }
//         }
//         return offset;
//     }

//     SD_INLINE SD_HOST_DEVICE sd::Unsigned getIndexOffset(sd::Unsigned index, const sd::Unsigned *shapeInfo,
//     sd::Unsigned arrLen) {

//         const sd::Unsigned rank = shapeInfo[0];
//         const sd::Unsigned ews = shapeInfo[rank + rank + 2];

//         if(ews > 0 && shapeInfo[rank + rank + 3] == 99)
//            if (ews == 1)
//                return index;
//            else
//                return ews * index;

//         sd::Unsigned offset = 0;

//         for(sd::Unsigned i = 1; i <= rank; ++i) {
//             arrLen /= shapeInfo[i];
//             if(arrLen > 0 && shapeInfo[i] > 1) {
//                 offset += (index / arrLen) * shapeInfo[i + rank];
//                 index %= arrLen;
//             }
//         }
//         return offset;
//     }

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType getIndexOffset(sd::LongType index, const sd::LongType *shapeInfo) {
  if (shapeInfo[2 * shapeInfo[0] + 3] == 99) {
    const sd::LongType ews = shapeInfo[2 * shapeInfo[0] + 2];
    if (ews == 1)
      return index;
    else if (ews > 1)
      return ews * index;
  }

  sd::LongType offset = 0;

  for (sd::Unsigned i = shapeInfo[0]; i > 1; --i) {
    offset += (index % shapeInfo[i]) * shapeInfo[i + shapeInfo[0]];
    index /= shapeInfo[i];
  }

  offset += index * shapeInfo[1 + shapeInfo[0]];  // last iteration

  return offset;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::Unsigned getIndexOffset(sd::Unsigned index, const sd::Unsigned *shapeInfo) {
  if (shapeInfo[2 * shapeInfo[0] + 3] == 99) {
    const sd::LongType ews = shapeInfo[2 * shapeInfo[0] + 2];
    if (ews == 1)
      return index;
    else if (ews > 1)
      return ews * index;
  }

  sd::Unsigned offset = 0;

  for (sd::Unsigned i = shapeInfo[0]; i > 1; --i) {
    offset += (index % shapeInfo[i]) * shapeInfo[i + shapeInfo[0]];
    index /= shapeInfo[i];
  }

  offset += index * shapeInfo[1 + shapeInfo[0]];  // last iteration

  return offset;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType indexOffset(sd::LongType index, const sd::LongType *lShapeInfo,
                                                  const sd::Unsigned *uShapeInfo, const bool useUnsigned) {
  if (useUnsigned) return getIndexOffset(static_cast<sd::Unsigned>(index), uShapeInfo);

  return getIndexOffset(index, lShapeInfo);
}

/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
SD_INLINE SD_HOST_DEVICE sd::LongType *doPermuteSwap(int length, sd::LongType *shape, int *rearrange) {
  traceNew(16);
  sd::LongType *ret = new sd::LongType[length];
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
SD_INLINE SD_HOST_DEVICE void doPermuteSwap(int length, sd::LongType **shape, int *rearrange) {
  if (length == 1) {
    return;
  } else {
    sd::LongType *shapeDeref = *shape;
    if (shape::prodLong(shapeDeref, length) < 2) {
      return;
    }
  }

  bool inOrder = true;
  for (int i = 0; i < length - 1; i++) {
    inOrder = inOrder && rearrange[i] + 1 == rearrange[i + 1];
  }

  // all in order, nothing to do
  if (inOrder) return;

  sd::LongType *shapeDeref = *shape;
  // we know they are just reversed, dimension length of 2
  if (length == 2) {
    auto shapeFirst = shapeDeref[0];
    auto shapeSecond = shapeDeref[1];
    shapeDeref[0] = shapeSecond;
    shapeDeref[1] = shapeFirst;
    return;
  } else if (length == 1) {
    // no permute
    return;
  }

  auto temp = new sd::LongType[length];
  memcpy(temp, shapeDeref, sizeof(sd::LongType) * length);
  for (int i = 0; i < length; i++) {
    shapeDeref[i] = temp[rearrange[i]];
  }

  delete[] temp;
}

SD_INLINE SD_HOST_DEVICE void permuteShapeBufferInPlace(sd::LongType *shapeBuffer, int *rearrange, sd::LongType *out) {
  if (shapeBuffer != out) memcpy(out, shapeBuffer, sizeof(sd::LongType) * shape::shapeInfoLength(shapeBuffer));

  shape::doPermuteShapeInfo(out, rearrange);
}

SD_INLINE SD_HOST_DEVICE sd::LongType *permuteShapeBuffer(sd::LongType const *shapeBuffer, int *rearrange) {
  auto len = shape::shapeInfoLength(shape::rank(shapeBuffer));
  sd::LongType *copy = shape::copyOf(len, shapeBuffer);
  shape::doPermuteShapeInfo(copy, rearrange);
  return copy;
}

SD_INLINE SD_HOST_DEVICE void doPermuteShapeInfo(sd::LongType *shapeInfo, const int *rearrange, sd::LongType len) {
  if (len == -1)  // calculate array length if it is not given
    len = shape::length(shapeInfo);

  // check whether shape is like {1} or {1,1} or {1,1,1,1,...} - in this case we don't need permute
  if (len == 1) return;

  const int rank = shape::rank(shapeInfo);

  // check whether rearrange is like {0,1,2,3,...}  - in this case we don't need permute as well
  bool isPermutNecessary = false;
  for (int i = 0; i < rank; ++i)
    if (rearrange[i] != i) {
      isPermutNecessary = true;
      break;
    }

  if (!isPermutNecessary) return;

  // check whether rearrange contains correct indexes
  for (int i = 0; i < rank; ++i) {
      if (rearrange[i] >= rank || rearrange[i] < 0) {
          sd_printf(
                  "shape::doPermuteShapeInfo function failed: rearrange indexes are incorrect. Given permute indices must be < rank and >= 0.  Rearrange at index %d was %d\n",
                  i, rearrange[i]);
          return;
      }
  }
  // if everything is ok then perform permute
  auto temp = new sd::LongType[shape::shapeInfoLength(rank) - 3];
  memcpy(temp, shapeInfo, sizeof(sd::LongType) * (shape::shapeInfoLength(rank) - 3));
  for (int i = 0; i < rank; ++i) {
    shapeInfo[i + 1] = temp[rearrange[i] + 1];
    shapeInfo[i + 1 + rank] = temp[rearrange[i] + 1 + rank];
  }

  shape::checkStridesEwsAndOrder(shapeInfo);

  delete[] temp;
}

SD_INLINE SD_HOST_DEVICE sd::LongType *createPermuteIndexes(int originalRank, int *dimension, int dimensionLength) {
  int delta = originalRank - dimensionLength;

  traceNew(17);

  sd::LongType *ret = new sd::LongType[originalRank];
  for (int i = 0; i < delta; i++) {
    ret[i] = i + dimensionLength;
  }

  for (int i = delta; i < originalRank; i++) {
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
SD_INLINE SD_HOST_DEVICE char getOrder(int length, sd::LongType *shape, sd::LongType *stride, int elementStride) {
  sd::LongType sd = 1;
  int dim = -1;
  int i = -1;
  int cContiguous = 1;
  int isFortran = 1;

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

template <typename T>
SD_INLINE SD_HOST_DEVICE int checkArrangeArray(T *arr, int arrLength, int shapeLength) {
  if (arrLength != shapeLength) return -1;
  for (int i = 0; i < arrLength; i++) {
    if (arr[i] >= arrLength || arr[i] < 0) return -1;
  }

  for (int i = 0; i < arrLength; i++) {
    for (int j = 0; j < arrLength; j++) {
      if (i != j && arr[i] == arr[j]) return -1;
    }
  }

  return 1;
}

SD_INLINE SD_HOST_DEVICE void traceNew(int id){
// printf("new happened: [%i]\n", id);

#ifndef __CUDACC__
// fflush(stdout);
#endif
}

/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
SD_INLINE SD_HOST_DEVICE void permute(ShapeInformation **info, int *rearrange, int rank) {
  ShapeInformation *infoDeref = *info;
  checkArrangeArray(rearrange, rank, rank);
  shape::doPermuteSwap(rank, &infoDeref->shape, rearrange);
  shape::doPermuteSwap(rank, &infoDeref->stride, rearrange);
  char order = getOrder(rank, infoDeref->shape, infoDeref->stride, infoDeref->elementWiseStride);
  infoDeref->order = order;
}

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
SD_INLINE SD_HOST_DEVICE int isVector(sd::LongType const *shape, int rank) {
  if (rank == 0) return 0;

  if (rank == 1) return 1;

  if (rank > 2)
    return 0;
  else if (rank <= 2) {
    if (shape[0] == 1 || shape[1] == 1) return 1;
  }
  return 0;
}

SD_INLINE SD_HOST_DEVICE bool isLikeVector(sd::LongType const *shapeInfo, int &posOfNonUnityDim) {
  int numOfNonUnity = 0;
  for (int i = 1; i <= shapeInfo[0]; ++i) {
    if (shapeInfo[i] != 1) {
      ++numOfNonUnity;
      posOfNonUnityDim = i - 1;
    }
  }

  return numOfNonUnity == 1 && shapeInfo[0] > 2;
}

SD_INLINE SD_HOST_DEVICE bool isCommonVector(const sd::LongType *shapeInfo, int &posOfNonUnityDim) {
  if (rank(shapeInfo) > 0 && length(shapeInfo) == 1) {
    posOfNonUnityDim = -1;
    return true;
  }

  int numOfNonUnity = 0;
  for (int i = 1; i <= shapeInfo[0]; ++i) {
    if (shapeInfo[i] != 1) {
      ++numOfNonUnity;
      posOfNonUnityDim = i - 1;
    }
  }
  return numOfNonUnity == 1;
}

SD_INLINE SD_HOST sd::LongType const *detachShape(sd::LongType const *originalShape) {
  sd::LongType *newShape = new sd::LongType[shape::shapeInfoLength(originalShape)];
  memcpy(newShape, originalShape, shape::shapeInfoByteLength(originalShape));

  return newShape;
}

SD_INLINE SD_HOST sd::LongType *copyShape(sd::LongType const *originalShape) {
  sd::LongType *newShape = new sd::LongType[shape::shapeInfoLength(originalShape)];
  memcpy(newShape, originalShape, shape::shapeInfoByteLength(originalShape));

  return newShape;
}

SD_INLINE SD_HOST_DEVICE int isVector(const sd::LongType *shapeInfo) {
  return isVector(shape::shapeOf(const_cast<sd::LongType *>(shapeInfo)), shape::rank(shapeInfo));
}

SD_INLINE SD_HOST_DEVICE bool isRowVector(const sd::LongType *shapeInfo) {
  bool isVector = shape::isVector(shapeInfo) == 1;
  bool shapeFirstOne = shape::shapeOf(const_cast<sd::LongType *>(shapeInfo))[0] == 1;
  return isVector && shapeFirstOne;
}

SD_INLINE SD_HOST_DEVICE bool isColumnVector(const sd::LongType *shapeInfo) {
  bool isVector = shape::isVector(shapeInfo) == 1;
  bool shapeFirstOne = shape::shapeOf(shapeInfo)[0] == 1;
  return isVector && !shapeFirstOne;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE int numOfNonUnitDims(const int rank, const sd::LongType *inShape) {
  int num = 0;

  for (sd::Unsigned i = 0; i < rank; ++i)
    if (inShape[i] != 1) ++num;

  return num;
}

SD_INLINE SD_HOST_DEVICE int oneDimEqualToLength(sd::LongType *shape, int rank) {
  for (int i = 0; i < rank; i++) {
    if (shape[i] == shape::prodLong(shape, rank)) return 1;
  }

  return 0;
}

SD_INLINE SD_HOST_DEVICE int oneDimEqualToLength(sd::LongType *shapeInfo) {
  return oneDimEqualToLength(shape::shapeOf(shapeInfo), shape::rank(shapeInfo));
}

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
SD_INLINE SD_HOST_DEVICE int isMatrix(const sd::LongType *shape, int rank) {
  if (rank > 2)
    return 0;
  else if (rank <= 2) {
    if (shape[0] == 1 || shape[1] == 1) return 0;
  }

  return 1;
}

SD_INLINE SD_HOST_DEVICE int isMatrix(const sd::LongType *shapeInfo) {
  return isMatrix(shape::shapeOf(shapeInfo), shape::rank(shapeInfo));
}

/**
 * Returns the shape portion of an information
 * buffer
 */
SD_INLINE SD_HOST_DEVICE sd::LongType *shapeOf(sd::LongType *shapeInfo) { return shapeInfo + 1; }

SD_INLINE SD_HOST_DEVICE sd::LongType *shapeOf(const sd::LongType *shapeInfo) {
  return shape::shapeOf(const_cast<sd::LongType *>(shapeInfo));
}

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
template <typename T>
SD_INLINE SD_HOST_DEVICE T *copyOf(sd::LongType length, T const *toCopy) {
  traceNew(18);

  T *ret = new T[length];
  return copyOf(length, toCopy, ret);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T *copyOf(sd::LongType length, T const *toCopy, T *ret) {
  memcpy(ret, toCopy, sizeof(T) * length);
  return ret;
}

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
template <typename T>
SD_INLINE SD_HOST_DEVICE void copyTo(sd::LongType length, T const *from, T *to) {
  memcpy(to, from, sizeof(T) * length);
}

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
SD_INLINE SD_HOST_DEVICE void copyTo(int length, sd::LongType const *from, sd::LongType *to, sd::LongType *indexes) {
  for (int i = 0; i < length; i++) {
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
/*
   SD_INLINE SD_HOST_DEVICE sd::LongType *permutedStrides(sd::LongType *toPermute, int shapeRank, int *rearrange) {
       sd::LongType *strideCopy = copyOf(shapeRank, toPermute);
       checkArrangeArray(rearrange, shapeRank, shapeRank);
       sd::LongType *newStride = doPermuteSwap(shapeRank, strideCopy, rearrange);
       delete[] strideCopy;
       return newStride;
   }
   */

/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
SD_INLINE SD_HOST_DEVICE sd::LongType *slice(sd::LongType *shape) { return shape + 1; }

SD_INLINE SD_HOST_DEVICE int slices(sd::LongType *shapeBuffer) {
  return static_cast<int>(shape::shapeOf(shapeBuffer)[0]);
}

SD_INLINE SD_HOST_DEVICE sd::LongType *sliceOfShapeBuffer(sd::LongType sliceIdx, sd::LongType *shapeBuffer) {
  int rank = shape::rank(shapeBuffer);
  int newRank = rank - 1;
  if (newRank < 2) newRank = 2;
  sd::LongType *newShapeBuffer = new sd::LongType[shape::shapeInfoLength(newRank)];
  newShapeBuffer[0] = newRank;
  sd::LongType *currShape = shape::shapeOf(shapeBuffer);
  sd::LongType *currStride = shape::stride(shapeBuffer);
  // initialize new shape and stride by taking the shape and stride + 1
  // and adding to the shape information
  // a slice is always just taking the existing shape and cutting the first index off
  // of the shape and stride
  sd::LongType *newShape = shape::shapeOf(newShapeBuffer);
  sd::LongType *newStride = shape::stride(newShapeBuffer);
  if (shape::isVector(shapeBuffer)) {
    sd::LongType *currShape = shape::shapeOf(shapeBuffer);
    // row vector: slice index 0 is a valid index, just copy the whole thing
    if (currShape[0] == 1) {
      if (sliceIdx == 0) {
        memcpy(newShapeBuffer, shapeBuffer, shape::shapeInfoByteLength(shape::rank(shapeBuffer)));
        return newShapeBuffer;
      }
    }
    // column vector: this will be a scalar
    else {
      delete[] newShapeBuffer;
      sd::LongType *scalar = shape::createScalarShapeInfo();
      int offset = shape::offset(shapeBuffer);
      scalar[shape::shapeInfoLength(2) - 3] = offset + sliceIdx;
      return scalar;
    }
  } else if (shape::isMatrix(shapeBuffer)) {
    newShape[0] = 1;
    newShape[1] = currShape[1];
    newStride[0] = 1;
    newStride[1] = currStride[1];
  } else {
    for (int i = 0; i < newRank; i++) {
      newShape[i] = currShape[i + 1];
      newStride[i] = currStride[i + 1];
    }
  }

  auto indices = new sd::LongType[rank];
  memset((void *)indices, 0, rank * sizeof(sd::LongType));
  indices[0] = sliceIdx;
  sd::LongType offset = shape::getOffset(newShapeBuffer, indices);
  newShapeBuffer[shape::shapeInfoLength(newRank) - 3] = offset;

  // set current order and ews
  newShapeBuffer[2 * newRank + 2] = shape::elementWiseStride(shapeBuffer);
  newShapeBuffer[2 * newRank + 3] = shape::order(shapeBuffer);

  // correct order and ews if necessary
  shape::checkStridesEwsAndOrder(newShapeBuffer);

  delete[] indices;

  return newShapeBuffer;
}

/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 3
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
SD_INLINE SD_HOST_DEVICE int shapeInfoLength(int rank) {
  // FIXME magic numbers
  return rank * 2 + 4;
}

SD_INLINE SD_HOST_DEVICE int shapeInfoLength(sd::LongType *shape) {
  return shapeInfoLength(static_cast<int>(shape[0]));
}

SD_INLINE SD_HOST_DEVICE int shapeInfoLength(const sd::LongType *shape) {
  return shapeInfoLength(static_cast<int>(shape[0]));
}

SD_INLINE SD_HOST_DEVICE size_t shapeInfoByteLength(int rank) {
  // FIXME magic numbers
  return (rank * 2 + 4) * sizeof(sd::LongType);
}

SD_INLINE SD_HOST_DEVICE size_t shapeInfoByteLength(const sd::LongType *shapeInfo) {
  // FIXME magic numbers
  return shapeInfoByteLength((int)shapeInfo[0]);
}

/**
 * Returns the rank portion of
 * an information buffer
 */
SD_INLINE SD_HOST_DEVICE int rank(const sd::LongType *buffer) { return static_cast<int>(buffer[0]); }

SD_INLINE SD_HOST_DEVICE int rank(const int *buffer) { return buffer[0]; }

SD_INLINE SD_HOST_DEVICE int rank(const unsigned int *buffer) { return static_cast<int>(buffer[0]); }

SD_INLINE SD_HOST_DEVICE sd::LongType *ews(sd::LongType *shapeInfo) { return shapeInfo + 2 * shapeInfo[0] + 2; }

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
SD_INLINE SD_HOST_DEVICE ShapeInformation *infoFromBuffer(sd::LongType *buffer) {
  traceNew(19);

  auto info = new ShapeInformation;
  auto length = shapeInfoLength(rank(buffer));
  auto rank = buffer[0];

  // start after rank
  info->shape = buffer + 1;
  info->stride = buffer + (1 + rank);
  info->rank = rank;
  info->offset = buffer[length - 3];
  info->elementWiseStride = buffer[length - 2];
  sd::LongType *stride = buffer + 1 + rank;
  info->stride = stride;
  info->order = (char)buffer[length - 1];
  return info;
}

/**
 * Returns the stride portion of an information
 * buffer
 */
SD_INLINE SD_HOST_DEVICE sd::LongType *stride(sd::LongType *buffer) { return buffer + (1 + rank(buffer)); }

SD_INLINE SD_HOST_DEVICE sd::LongType *stride(const sd::LongType *buffer) {
  return stride(const_cast<sd::LongType *>(buffer));
}

SD_INLINE SD_HOST_DEVICE bool isEmpty(const sd::LongType *shapeInfo) {
  return ((shape::extra(const_cast<sd::LongType *>(shapeInfo)) & ARRAY_EMPTY) == ARRAY_EMPTY);
}

/**
 * Compute the length of the given shape
 */
SD_INLINE SD_HOST_DEVICE sd::LongType length(const sd::LongType *shapeInfo) {
  const int rank = shape::rank(shapeInfo);

  if (rank == 0) {
    if (isEmpty(shapeInfo)) return 0L;
    return 1L;
  }

  if (rank == 1) return shapeInfo[1];

  // if(shape::elementWiseStride(shapeInfo) == 1) { // contiguous
  //     if(shape::order(shapeInfo) == 'c')
  //         return shapeInfo[1] * shapeInfo[rank + 1];      // first dim * first stride
  //     return shapeInfo[rank] * shapeInfo[2 * rank];       // last  dim * last  stride
  // }

  return shape::prodLong(shape::shapeOf(const_cast<sd::LongType *>(shapeInfo)), rank);
}

SD_INLINE SD_HOST_DEVICE sd::LongType length(std::initializer_list<int> &shape) {
  sd::LongType ret = 1;
  for (auto v : shape) {
    ret *= v;
  }
  return ret;
}

SD_INLINE SD_HOST_DEVICE sd::LongType length(std::initializer_list<sd::LongType> &shape) {
  sd::LongType ret = 1;
  for (auto v : shape) {
    ret *= v;
  }
  return ret;
}

/***
 * Returns the offset
 * portion of an information buffer
 */
SD_INLINE SD_HOST_DEVICE sd::LongType offset(sd::LongType *buffer) {
  return buffer[shape::shapeInfoLength(shape::rank(buffer)) - 3];
}

SD_INLINE SD_HOST_DEVICE sd::LongType &extra(sd::LongType *buffer) {
  return buffer[shape::shapeInfoLength(shape::rank(buffer)) - 3];
}

/**
 * Returns the ordering
 * for this shape information buffer
 */
SD_INLINE SD_HOST_DEVICE char order(const sd::LongType *buffer) {
  // FIXME magic numbers
  return static_cast<char>(buffer[buffer[0] * 2 + 3]);
}

/**
 * Returns type
 */
SD_INLINE SD_HOST_DEVICE sd::LongType type(const sd::LongType *shapeInfo) { return shapeInfo[2 * shapeInfo[0] + 1]; }

/**
 * Returns the element wise stride for this information
 * buffer
 */
SD_INLINE SD_HOST_DEVICE sd::LongType elementWiseStride(const sd::LongType *buffer) {
  return buffer[shapeInfoLength(static_cast<int>(buffer[0])) - 2];
}

/**
 * Returns the element wise stride for this information
 * buffer relative to a dimension and reduction index
 */
SD_INLINE SD_HOST_DEVICE sd::LongType reductionIndexElementWiseStride(sd::LongType *buffer, int *dimension,
                                                                      int dimensionLength) {
  if (dimensionLength > 1) {
    if (shape::order(buffer) == 'f') {
      /**
       * The element wise stride belongs to a reduction index.
       * When used out of order, we can get rid of the data
       * dependencies and rely on using the max dimension
       * specified for stride instead.
       * Say we take the sum(0,1) along arr
       * we can use arr.stride(1) as a representation
       * along which to iterate.
       */
      if (shape::shapeOf(buffer)[dimension[dimensionLength - 1]] != 1) {
        // int tadElementWiseStride = shape::stride(buffer)[dimension[dimensionLength - 1]];
        // return tadElementWiseStride;
        auto tadElementWiseStride = shape::stride(buffer)[dimension[0]];
        return tadElementWiseStride;
      }

      return 1;

    } else {
      /**
       * The element wise stride belongs to a reduction index.
       * When used out of order, we can get rid of the data
       * dependencies and rely on using the max dimension
       * specified for stride instead.
       * Say we take the sum(0,1) along arr
       * we can use arr.stride(1) as a representation
       * along which to iterate.
       */
      if (shape::shapeOf(buffer)[dimension[dimensionLength - 1]] != 1) {
        auto tadElementWiseStride = shape::stride(buffer)[dimension[dimensionLength - 1]];
        return tadElementWiseStride;
      }

      return 1;
    }
  } else {
    if (shape::order(buffer) == 'f') {
      /**
       * The element wise stride belongs to a reduction index.
       * When used out of order, we can get rid of the data
       * dependencies and rely on using the max dimension
       * specified for stride instead.
       * Say we take the sum(0,1) along arr
       * we can use arr.stride(1) as a representation
       * along which to iterate.
       */
      auto tadElementWiseStride = shape::stride(buffer)[dimension[0]];
      return tadElementWiseStride;
    } else {
      /**
       * The element wise stride belongs to a reduction index.
       * When used out of order, we can get rid of the data
       * dependencies and rely on using the max dimension
       * specified for stride instead.
       * Say we take the sum(0,1) along arr
       * we can use arr.stride(1) as a representation
       * along which to iterate.
       */
      auto tadElementWiseStride = shape::stride(buffer)[dimension[dimensionLength - 1]];
      return tadElementWiseStride;
    }
  }
}

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
SD_INLINE SD_HOST_DEVICE int isScalar(const sd::LongType *info) {
  const int rank = shape::rank(info);

  if (rank > 2) return 0;
  if (rank == 0) return 1;
  if (rank == 1) return shape::shapeOf(const_cast<sd::LongType *>(info))[0] == 1;
  if (rank == 2)
    return shape::shapeOf(const_cast<sd::LongType *>(info))[0] == 1 &&
           shape::shapeOf(const_cast<sd::LongType *>(info))[1] == 1;

  return 0;
}

/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
SD_INLINE SD_HOST_DEVICE int isScalar(volatile ShapeInformation *info) {
  const int rank = info->rank;

  if (rank > 2) return 0;
  if (rank == 1) return info->shape[0] == 1;
  if (rank == 2) return info->shape[0] == 1 && info->shape[1] == 1;

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
template <typename T1, typename T2>
SD_INLINE SD_HOST_DEVICE void removeIndex(T1 const *data, T2 const *indexes, sd::LongType dataLength,
                                          sd::LongType indexesLength, T1 *ret) {
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
template <typename T1, typename T2>
SD_INLINE SD_HOST_DEVICE T1 *removeIndex(T1 const *data, T2 const *indexes, sd::LongType dataLength,
                                         sd::LongType indexesLength) {
  auto lengthOfArr = dataLength - indexesLength;
  if (lengthOfArr < 0) {
    printf("Remove index call created a <= 0 length array. This was likely not intended.");
  }

  auto ret = new T1[lengthOfArr];
  memset(ret, 0, sizeof(T1) * lengthOfArr);
  removeIndex<T1, T2>(data, indexes, dataLength, indexesLength, ret);
  return ret;
}

SD_INLINE SD_HOST_DEVICE sd::LongType *everyIndexBut(const sd::LongType *indexes, int indexesLength, int begin,
                                                     int end) {
  int len = end - indexesLength;

  traceNew(20);

  auto ret = new sd::LongType[len];
  int retIdx = 0;
  // not here that we do 0 based indexing for end - this assumes things like:
  // 0 to 4 are specified
  for (int i = begin; i < end; i++) {
    bool found = false;
    for (int j = 0; j < indexesLength; j++) {
      if (indexes[j] == i) {
        found = true;
        break;
      }
    }

    if (!found) {
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
SD_INLINE SD_DEVICE int tadOffset(ShapeInformation *xInfo, int offset) {
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
SD_INLINE SD_HOST_DEVICE sd::LongType *ensureVectorShape(sd::LongType *shape, int dimension) {
  traceNew(21);

  sd::LongType *ret = new sd::LongType[2];

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
SD_INLINE SD_HOST_DEVICE sd::LongType *ensureVectorShape(sd::LongType *shape) { return ensureVectorShape(shape, 0); }

/**
 * This method does STRICT comparison for two shape buffers
 *
 * @param shape
 * @return
 */
SD_INLINE SD_HOST_DEVICE bool equalsStrict(const sd::LongType *shapeA, const sd::LongType *shapeB) {
  if (shapeA[0] != shapeB[0]) return false;

  if (shapeA[0] == 0) return true;

  // we do full comparison here
  int length = shape::shapeInfoLength(shapeA[0]);

  for (int e = 1; e < length; e++)
    if (shapeA[e] != shapeB[e]) return false;

  return true;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE bool haveSameShapeAndStrides(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2) {
  if (shapeInfo1[0] != shapeInfo2[0]) return false;

  if (shapeInfo1[0] == 0) return true;

  for (sd::Unsigned e = 0; e < static_cast<sd::Unsigned>(shape::rank(shapeInfo1)); ++e)
    if (shape::shapeOf(shapeInfo1)[e] != shape::shapeOf(shapeInfo2)[e] ||
        shape::stride(shapeInfo1)[e] != shape::stride(shapeInfo2)[e])
      return false;

  return true;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE bool haveSameShapeAndStrides(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2,
                                                      const sd::LongType *shapeInfo3) {
  return shape::haveSameShapeAndStrides(shapeInfo1, shapeInfo2) &&
         shape::haveSameShapeAndStrides(shapeInfo1, shapeInfo3);
}
SD_INLINE SD_HOST_DEVICE int sizeAt(const sd::LongType *shapeInfo, const int dim) {
  if (0 == rank(shapeInfo)) return 1;
  if (dim >= 0)
    return shapeInfo[1 + dim];
  else
    return shapeInfo[1 + (rank(shapeInfo) + dim)];
}

SD_INLINE SD_HOST_DEVICE sd::LongType strideAt(const sd::LongType *shapeInfo, const int dim) {
  if (0 == rank(shapeInfo)) return 1;
  if (dim >= 0)
    return shapeInfo[1 + rank(shapeInfo) + dim];
  else
    return shapeInfo[1 + 2 * rank(shapeInfo) + dim];
}

/**
 * This method does SOFT comparison for two shape buffers, we compare only rank & shapes
 *
 * @param shape
 * @return
 */
SD_INLINE SD_HOST_DEVICE bool equalsSoft(const sd::LongType *shapeA, const sd::LongType *shapeB) {
  if (shapeA[0] != shapeB[0]) return false;

  if (shape::isEmpty(shapeA) && shape::isEmpty(shapeB)) return true;

  if (shapeA[0] == 0) return true;

  // we compare only shapes, and ignoring stride & ews
  auto length = shapeA[0];

  for (int e = 1; e <= length; e++)
    if (shapeA[e] != shapeB[e]) return false;

  return true;
}

SD_INLINE SD_HOST_DEVICE bool equalsTypesAndShapesSoft(const sd::LongType *shapeA, const sd::LongType *shapeB) {
  return equalsSoft(shapeA, shapeB) && shapeA[shapeInfoLength(shapeA) - 3] == shapeB[shapeInfoLength(shapeB) - 3];
}

/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
template <typename T>
SD_INLINE SD_HOST_DEVICE T *range(int from, int to, int increment) {
  int diff = sd::math::sd_abs<int>(from - to);
  int retLength = diff / increment;
  T *ret;

  traceNew(22);

  if (diff / increment < 1)
    ret = new T[1];
  else
    ret = new T[diff / increment];
  if (from < to) {
    int count = 0;
    for (int i = from; i < to; i += increment) {
      if (count >= retLength) break;
      ret[count++] = i;
    }
  } else if (from > to) {
    int count = 0;
    for (int i = from - 1; i >= to; i -= increment) {
      if (count >= retLength) break;
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

template <typename T>
SD_INLINE SD_HOST_DEVICE T *range(int from, int to) {
  return range<T>(from, to, 1);
}

/**
 * Keep the given indexes in the data
 * @param data
 * @param index
 * @param indexLength
 * @param dataLength
 * @return
 */
SD_INLINE SD_HOST_DEVICE sd::LongType *keep(volatile sd::LongType *data, int const *index, int indexLength,
                                            int dataLength) {
  traceNew(23);

  sd::LongType *ret = new sd::LongType[indexLength];
  int count = 0;
  for (int i = 0; i < dataLength; i++) {
    int contains = 0;
    for (int j = 0; j < indexLength; j++) {
      if (i == index[j]) {
        contains = 1;
        break;
      }
    }

    if (contains) ret[count++] = data[i];
  }
  return ret;
}

/**
 * Generate a reverse
 * copy of the data
 */

template <typename T>
SD_INLINE SD_HOST_DEVICE T *reverseCopy(T const *data, sd::LongType length) {
  if (length < 1) return nullptr;

  traceNew(24);

  T *copy = new T[length];
  for (sd::LongType i = 0; i <= length / 2; i++) {
    T temp = data[i];
    copy[i] = data[length - i - 1];
    copy[length - i - 1] = temp;
  }
  return copy;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE void reverseCopyTo(T const *from, T *to, sd::LongType length) {
  if (length < 1) return;
  for (sd::LongType i = 0; i <= length / 2; i++) {
    T temp = from[i];
    to[i] = from[length - i - 1];
    to[length - i - 1] = temp;
  }
}

template <typename T>
SD_INLINE SD_HOST_DEVICE void reverseCopyTo(T const *from, T *to, sd::LongType *indexes, sd::LongType length) {
  if (length < 1) return;

  for (sd::LongType i = 0; i <= length / 2; i++) {
    T temp = from[indexes[i]];
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
template <typename T>
SD_INLINE SD_HOST_DEVICE T *concat(T const *arr1, sd::LongType const arr1Length, T const *arr2,
                                   sd::LongType const arr2Length) {
  traceNew(25);

  T *ret = new T[arr1Length + arr2Length];
  std::memcpy(ret, arr1, arr1Length * sizeof(T));
  std::memcpy(ret + arr1Length, arr2, arr2Length * sizeof(T));
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
template <typename T>
SD_INLINE SD_HOST_DEVICE T *concat(sd::LongType const numArrays, sd::LongType const numTotalElements, T const **arr,
                                   sd::LongType const *lengths) {
  T *ret = new T[numTotalElements];
  sd::LongType count = 0;

  for (sd::LongType i = 0; i < numArrays; i++) {
    for (sd::LongType j = 0; j < lengths[i]; j++) {
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
SD_INLINE SD_HOST_DEVICE sd::LongType lengthPerSlice(int rank, sd::LongType const *shape, int const *dimension,
                                                     int dimensionLength) {
  if (shape::isVector(shape, rank)) {
    // return total length for row vectors
    if (dimensionLength == 1 && shape[0] == 1) {
      return shape::prodLong(shape, rank);
    }
  } else if (rank == dimensionLength)
    return shape::prodLong(shape, rank);
  int absSelta = sd::math::sd_abs<int>(rank - dimensionLength);
  traceNew(27);
  auto ret2 = shape::removeIndex<sd::LongType>(shape, dimension, rank, dimensionLength);
  auto ret = prodLong(ret2, absSelta);
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
SD_INLINE SD_HOST_DEVICE sd::LongType sliceOffsetForTensor(int rank, int index, sd::LongType const *shape,
                                                           sd::LongType const *tensorShape, int tensorShapeLength,
                                                           int const *dimension, int dimensionLength) {
  auto tensorLength = prodLong(tensorShape, tensorShapeLength);
  auto lengthPerSlice2 = lengthPerSlice(rank, shape, dimension, dimensionLength);
  if (lengthPerSlice2 <= 0) {
    return 0;
  }

  sd::LongType offset = index * tensorLength / lengthPerSlice2;
  return offset;
}

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */

SD_INLINE SD_HOST_DEVICE sd::LongType sliceOffsetForTensor(int index, int tensorLength, int lengthPerSlice2) {
  sd::LongType offset = index * tensorLength / lengthPerSlice2;
  return offset;
}

#ifdef __CUDACC__
/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
SD_INLINE SD_DEVICE int tadOffset(sd::LongType *xInfo, int offset) {
  return offset + threadIdx.x * elementWiseStride(xInfo);
}
#endif

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
SD_INLINE SD_HOST_DEVICE sd::LongType tensorsAlongDimension(volatile int rank, volatile int length,
                                                            volatile sd::LongType *shape, int *dimension,
                                                            int dimensionLength) {
  sd::LongType *tensorShape = shape::keep(shape, dimension, dimensionLength, rank);
  sd::LongType ret = length / shape::prodLong(tensorShape, dimensionLength);
  delete[] tensorShape;
  return ret;
}

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
SD_INLINE SD_HOST_DEVICE sd::LongType tensorsAlongDimension(sd::LongType *shapeInfo, int *dimension,
                                                            int dimensionLength) {
  sd::LongType *keepShape = shape::shapeOf(shapeInfo);
  sd::LongType *tensorShape = shape::keep(keepShape, dimension, dimensionLength, rank(shapeInfo));
  sd::LongType ret = shape::length(shapeInfo) / shape::prodLong(tensorShape, dimensionLength);
  delete[] tensorShape;
  return ret;
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

//////////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType getOffset(const sd::LongType *shapeInfo, const sd::LongType *indices,
                                                sd::LongType baseOffset) {
  sd::LongType offset = baseOffset;

  for (sd::Unsigned i = 1; i <= shapeInfo[0]; ++i)
    if (shapeInfo[i] != 1) offset += indices[i - 1] * shapeInfo[shapeInfo[0] + i];

  return offset;
}

//////////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType getOffset(const sd::LongType *shapeInfo, const sd::Unsigned *coords,
                                                sd::LongType baseOffset) {
  sd::LongType offset = baseOffset;

  for (sd::Unsigned i = 1; i <= shapeInfo[0]; ++i)
    if (shapeInfo[i] != 1) offset += coords[i - 1] * shapeInfo[shapeInfo[0] + i];

  return offset;
}

//////////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType getOffset(const sd::LongType *shapeInfo, const int *coords,
                                                sd::LongType baseOffset) {
  sd::LongType offset = baseOffset;

  for (sd::Unsigned i = 1; i <= shapeInfo[0]; ++i)
    if (shapeInfo[i] != 1) offset += coords[i - 1] * shapeInfo[shapeInfo[0] + i];

  return offset;
}

//////////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType getOffset(const sd::LongType *shapeInfo, const int *coords, const int *dims) {
  sd::LongType offset = 0;

  for (sd::Unsigned i = 1; i <= shapeInfo[0]; ++i)
    if (shapeInfo[i] != 1) offset += coords[dims[i - 1]] * shapeInfo[shapeInfo[0] + i];

  return offset;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE void getOffsetBroadcast(const sd::LongType &startInd, const sd::LongType ind,
                                                 const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2,
                                                 const sd::LongType *shapeInfo3, const bool sameOffsets12,
                                                 const bool sameOffsets13, int *coords, sd::LongType &offset1,
                                                 sd::LongType &offset2, sd::LongType &offset3) {
  const sd::LongType *shape1 = shape::shapeOf(shapeInfo1);
  const sd::LongType *strides1 = shape::stride(shapeInfo1);
  const sd::LongType *shape2 = shape::shapeOf(shapeInfo2);
  const sd::LongType *strides2 = shape::stride(shapeInfo2);
  const sd::LongType *shape3 = shape::shapeOf(shapeInfo3);
  const sd::LongType *strides3 = shape::stride(shapeInfo3);

  if (startInd == ind) {
    if (shape::rank(shapeInfo1) == 0) {
      offset1 = offset2 = offset3 = 0;
      return;
    }

    shape::index2coords(ind, shapeInfo1, coords);
    offset1 = shape::getOffset(shapeInfo1, coords);

    if (sameOffsets12)
      offset2 = offset1;
    else
      offset2 = shape::getOffset(shapeInfo2, coords);

    if (sameOffsets13)
      offset3 = offset1;
    else
      offset3 = shape::getOffset(shapeInfo3, coords);

    return;
  }

  int axis = shapeInfo1[0] - 1;
  while (coords[axis] == shape1[axis] - 1) {
    if (!sameOffsets12 && shape2[axis] != 1) offset2 -= (shape2[axis] - 1) * strides2[axis];
    if (!sameOffsets13 && shape3[axis] != 1) offset3 -= (shape3[axis] - 1) * strides3[axis];
    if (shape1[axis] != 1) offset1 -= (shape1[axis] - 1) * strides1[axis];
    coords[axis--] = 0;
  }

  ++coords[axis];
  offset1 += strides1[axis];

  if (!sameOffsets12 && shape2[axis] != 1) offset2 += strides2[axis];
  if (!sameOffsets13 && shape3[axis] != 1) offset3 += strides3[axis];

  if (sameOffsets12) offset2 = offset1;
  if (sameOffsets13) offset3 = offset1;
}

/**
 * Returns the tensor along dimension
 * for the given block index
 * @param blockSize
 * @param blockIdx
 * @param i
 * @return
 */
SD_INLINE SD_HOST_DEVICE int tadForBlockIndex(int blockSize, int blockIdx, int i) { return blockIdx + i * blockSize; }

/**
 * Computes the number of tads per block
 *
 */
SD_INLINE SD_HOST_DEVICE int tadsPerBlock(int blockSize, int tads) {
  return sd::math::sd_ceil<double, int>(tads / (double)blockSize);
}

/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
SD_INLINE SD_HOST_DEVICE sd::LongType *toShapeBuffer(ShapeInformation *info) {
  traceNew(29);

  auto ret = new sd::LongType[shapeInfoLength(info->rank)];
  int count = 1;
  int rank = info->rank;

  ret[0] = info->rank;

  for (int i = 0; i < rank; i++) {
    ret[count++] = info->shape[i];
  }

  for (int i = 0; i < rank; i++) {
    ret[count++] = info->stride[i];
  }

  ret[count++] = info->offset;
  ret[count++] = info->elementWiseStride;
  ret[count] = info->order;

  return ret;
}

SD_INLINE SD_HOST_DEVICE sd::LongType *toShapeBuffer(ShapeInformation *info, sd::LongType *ret) {
  int count = 1;
  int rank = info->rank;

  ret[0] = info->rank;

  if (ret[0] == 0) {
    ret[1] = 0;
    ret[2] = 1;
    ret[3] = 99;
    return ret;
  }

  for (int i = 0; i < rank; i++) {
    ret[count++] = info->shape[i];
  }

  for (int i = 0; i < rank; i++) {
    ret[count++] = info->stride[i];
  }

  ret[count++] = info->offset;
  ret[count++] = info->elementWiseStride;
  ret[count++] = info->order;

  return ret;
}

SD_INLINE SD_HOST_DEVICE void printIntArray(const sd::LongType *arr, const int length) {
  for (int i = 0; i < length; i++) {
    printf(" %lld ", (long long)arr[i]);
  }

  printf("\n");
}

SD_INLINE SD_HOST_DEVICE void printIntArray(const int *arr, const int length) {
  for (int i = 0; i < length; i++) {
    printf(" %i ", arr[i]);
  }

  printf("\n");
}

SD_INLINE SD_HOST_DEVICE void printShapeInfo(const sd::LongType *shapeInfo) {
  int rank = shape::rank(shapeInfo);
  sd::LongType *shape = shape::shapeOf(shapeInfo);
  printf("Rank %d\n", rank);
  printf("Shape:\n");
  for (int i = 0; i < rank; i++) {
    printf(" %lld ", (long long)shape[i]);
  }

  printf("\n");

  sd::LongType *stride = shape::stride(shapeInfo);
  printf("Stride:\n");
  for (int i = 0; i < rank; i++) {
    printf(" %lld ", (long long)stride[i]);
  }

  printf("\n");

  printf("Order %c\n", shape::order(shapeInfo));
}

SD_INLINE SD_HOST_DEVICE void printShapeInfoLinear(const sd::LongType *shapeInfo) {
  int rank = shape::rank(shapeInfo);
  int lim = shape::shapeInfoLength(rank);
  printf("ShapeInfo: [");
  for (int i = 0; i < lim; i++) {
    printf("%lld", (long long)shapeInfo[i]);

    if (i < lim - 1) {
      printf(", ");
    }
  }
  printf("]\n");
#ifndef __CUDA_ARCH__
  fflush(stdout);
#endif
}

SD_INLINE SD_HOST_DEVICE void printShapeInfoLinear(const char *msg, int rank, const sd::LongType *shape,
                                                   const sd::LongType *strides) {
  printf("%s : [", msg);
  for (int i = 0; i < rank; i++) {
    printf("%lld, ", (long long)shape[i]);
  }

  for (int i = 0; i < rank; i++) {
    printf("%lld", (long long)strides[i]);

    if (i < rank - 1) printf(", ");
  }
  printf("]\n");

#ifndef __CUDA_ARCH__
  fflush(stdout);
#endif
}

SD_INLINE SD_HOST_DEVICE void printShapeInfoLinear(const char *msg, const sd::LongType *shapeInfo) {
  int rank = shape::rank(shapeInfo);
  int lim = shape::shapeInfoLength(rank);
  printf("%s : [", msg);
  for (int i = 0; i < lim; i++) {
    printf("%lld", (long long)shapeInfo[i]);

    if (i < lim - 1) {
      printf(", ");
    }
  }
  printf("]\n");
#ifndef __CUDACC__
  fflush(stdout);
#endif
}

template <typename T>
SD_INLINE SD_HOST_DEVICE void printArray(void *varr, int length, const char *message) {
  auto arr = reinterpret_cast<T *>(varr);
  if (message != nullptr)
    printf("%s: [", message);
  else
    printf("Array: [");

  for (int i = 0; i < length; i++) {
    printf("%f", (float)arr[i]);
    if (i + 1 < length) printf(", ");
  }
  printf("]\n");

#ifndef __CUDACC__
  fflush(stdout);
#endif
}

SD_INLINE SD_HOST_DEVICE void printArray(float *arr, int length) {
  printf("Array: [");
  for (int i = 0; i < length; i++) {
    printf("%f", arr[i]);
    if (i + 1 < length) printf(", ");
  }
  printf("]\n");
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
SD_INLINE SD_HOST_DEVICE int tadIndex(int i, int elementWiseStride, int numElementsPerTad) {
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
SD_INLINE SD_HOST_DEVICE int reductionIndexForTad(int tadIndexForOriginal, int tadsForReduced, int tadsForOriginal) {
  if (tadIndexForOriginal == 0) return 0;
  return tadIndexForOriginal / (tadsForOriginal / tadsForReduced);
}

SD_INLINE SD_HOST_DEVICE void transposeInplace(sd::LongType *shapeBuffer) {
  int rank = shape::rank(shapeBuffer);
  sd::LongType *shape = shape::shapeOf(shapeBuffer);
  sd::LongType *strides = shape::stride(shapeBuffer);

  // swap shape
  for (int e = 0; e < rank / 2; e++) {
    int idx1 = rank - e - 1;
    int idx2 = e;
    int tmp = shape[idx2];
    shape[idx2] = shape[idx1];
    shape[idx1] = tmp;
  }

  // swap strides
  for (int e = 0; e < rank / 2; e++) {
    int idx1 = rank - e - 1;
    int idx2 = e;
    int tmp = strides[idx2];
    strides[idx2] = strides[idx1];
    strides[idx1] = tmp;
  }

  if (shape::order(shapeBuffer) == 'c')
    shapeBuffer[shape::shapeInfoLength(shapeBuffer) - 1] = 102;
  else
    shapeBuffer[shape::shapeInfoLength(shapeBuffer) - 1] = 99;
}

/**
 * Tad index for linear
 * @param linearIndex
 * @param tadLength
 * @return
 */
SD_INLINE SD_HOST_DEVICE int tadIndexForLinear(int linearIndex, int tadLength) { return linearIndex % tadLength; }

/**
 * Computes the number of tads
 * per reduce index for the
 * reduction tad.
 */
SD_INLINE SD_HOST_DEVICE int tadsPerReduceIndex(int tadsForReduce, int tadsForOriginal) {
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
SD_INLINE SD_HOST_DEVICE int reductionIndexForLinear(int i, int elementWiseStride, int numElementsPerTad, int tadNum,
                                                     int originalTadNum) {
  int tad = tadIndex(i, elementWiseStride, numElementsPerTad);
  return reductionIndexForTad(tad, tadNum, originalTadNum);
}

SD_INLINE SD_HOST_DEVICE sd::LongType *createScalarShapeInfo() {
  traceNew(30);

  auto shape = new sd::LongType[1];
  shape[0] = 1;
  auto stride = new sd::LongType[1];
  stride[0] = 1;
  auto shapeInformation2 = new ShapeInformation();
  shapeInformation2->rank = 1;
  shapeInformation2->offset = 0;
  shapeInformation2->stride = stride;
  shapeInformation2->shape = shape;
  shapeInformation2->elementWiseStride = 1;
  shapeInformation2->order = 99;
  sd::LongType *ret = shape::toShapeBuffer(shapeInformation2);
  delete shapeInformation2;
  delete[] shape;
  delete[] stride;
  return ret;
}

SD_INLINE SD_HOST_DEVICE sd::LongType *createScalarShapeInfo(sd::LongType *ret) {
  ret[0] = 2;
  ret[1] = 1;
  ret[2] = 1;
  ret[3] = 1;
  ret[4] = 1;
  ret[5] = 0;
  ret[6] = 1;
  ret[7] = 99;

  return ret;
}

/**
 * Returns the prod of the data
 * up to the given length
 */
SD_INLINE SD_HOST_DEVICE sd::LongType prodLong(const sd::LongType *data, int length) {
  sd::LongType prod = 1;
  for (int i = 0; i < length; i++) {
    prod *= data[i];
  }

  return prod;
}

SD_INLINE SD_HOST_DEVICE int rearMostLeftOverItem(sd::LongType *data, sd::LongType *dimension, int dimensionLength) {
  sd::LongType *stride = shape::stride(data);
  // corner case: return the final item when its greater than the max, since its guaranteed to be left over
  // note here that strides are interpreted in reverse for tad
  // start from the front rather than the back

  int rank = shape::rank(data);

  if (shape::order(data) == 'f') {
    int dimIdx = dimensionLength - 1;
    for (int i = rank - 1; i >= 0; i--) {
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
      if (dimension[dimIdx--] != i) {
        int ret = stride[i];
        return ret;
      }
    }
  }

  else {
    int dimIdx = dimensionLength - 1;

    for (int i = rank - 1; i >= 0; i--) {
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
      if (dimension[dimIdx--] != i) {
        int ret = stride[i];
        return ret;
      }
    }
  }

  int ret = stride[0];
  return ret;
}

#ifdef __CUDACC__
SD_DEVICE SD_INLINE void sweepShapeInfoBuffer(sd::LongType *shapeInfoBuffer, sd::LongType *targetBuffer) {
  // we read first element, to find out length of our shapeInfoBuffer
  int rank = shapeInfoBuffer[0];
  int len = shape::shapeInfoLength(rank);
  for (int i = threadIdx.x; i < len; i += blockDim.x) targetBuffer[i] = shapeInfoBuffer[i];
}
#endif

SD_INLINE SD_HOST_DEVICE sd::LongType *shapeBufferOfNpy(cnpy::NpyArray arr) {
  return shape::shapeBufferOfNpy(arr.shape.size(), (unsigned int *)arr.shape.data(), arr.fortranOrder);
}

//    SD_INLINE SD_HOST_DEVICE sd::LongType *shapeBufferOfNpyBuffer(char *buffer) {
//        unsigned sd::LongType *shape;
//        unsigned int ndims, wordSize;
//        bool fortranOrder;
//        cnpy::parseNpyHeaderStr(std::string(buffer),wordSize,shape,ndims,fortranOrder);
//        sd::LongType * ret =  shape::shapeBufferOfNpy(ndims,shape,fortranOrder);
//        delete[] shape;
//        return ret;
//    }

SD_INLINE SD_HOST_DEVICE sd::LongType *shapeBufferOfNpy(int rank, unsigned int *shape, bool fortranOrder) {
  if (fortranOrder) {
    sd::LongType *shapeBufferRet = shape::shapeBufferFortran(rank, sd::FLOAT32, (sd::LongType *)shape);
    return shapeBufferRet;
  } else {
    sd::LongType *newShape = new sd::LongType[rank];
    for (int i = 0; i < rank; i++) {
      newShape[i] = shape[i];
    }

    sd::LongType *shapeBufferRet = shape::shapeBuffer(rank, sd::FLOAT32, newShape);
    delete[] newShape;
    return shapeBufferRet;
  }
}

SD_INLINE SD_HOST_DEVICE bool strideDescendingCAscendingF(const sd::LongType *shapeBuffer) {
  int rank = shape::rank(shapeBuffer);
  sd::LongType *strides = shape::stride(const_cast<sd::LongType *>(shapeBuffer));
  char order = shape::order(shapeBuffer);

  if (shape::isRowVector(shapeBuffer) && strides[0] == 1 && strides[1] == 1) return true;

  if (order == 'c') {
    for (int i = 1; i < rank; i++)
      if (strides[i - 1] <= strides[i]) return false;
    return true;
  } else if (order == 'f') {
    for (int i = 1; i < rank; i++)
      if (strides[i - 1] >= strides[i]) return false;
    return true;
  } else {
    printf("Unknown order for array!\n");
    return false;
  }
}

SD_INLINE SD_HOST_DEVICE bool isContiguous(const sd::LongType *shapeInfo) {
  return (order(shapeInfo) == 'c') && (elementWiseStride(shapeInfo) > 0);
}

//////////////////////////////////////////////////////////////////////////
// copy-past from java hasDefaultStridesForShape function
SD_INLINE SD_HOST_DEVICE bool areStridesDefault(const sd::LongType *shapeInfo) {
  const int rank = shape::rank(shapeInfo);

  if (rank == 0) return true;
  if (!strideDescendingCAscendingF(shapeInfo)) return false;

  sd::LongType defaultShapeInfo[SD_MAX_SHAPEINFOLENGTH];
  memcpy(defaultShapeInfo, shapeInfo, shape::shapeInfoByteLength(shapeInfo));
  shape::updateStrides(defaultShapeInfo, shape::order(shapeInfo));

  bool result = true;
  for (int i = rank + 1; i <= 2 * rank; ++i)
    if (defaultShapeInfo[i] != shapeInfo[i]) {
      result = false;
      break;
    }

  return result;
}

// SD_INLINE SD_HOST bool reshapeC(const int oldRank, sd::LongType* oldShape, const int newRank, sd::LongType*
// newShapeOf, bool isFOrder, sd::LongType* target) {
//         int oldnd;
//         sd::LongType* olddims = shape::copyOf(oldRank, shape::shapeOf(oldShape));
//         sd::LongType* oldstrides = shape::copyOf(oldRank, shape::stride(oldShape));
//         int np, op, last_stride;
//         int oi, oj, ok, ni, nj, nk;
//         sd::LongType* newStrides = new sd::LongType[newRank];
//         oldnd = 0;

//         /*
//          * Remove axes with dimension 1 from the old array. They have no effect
//          * but would need special cases since their strides do not matter.
//          */
//         for (oi = 0; oi < oldRank; oi++) {
//             if (shape::shapeOf(oldShape)[oi] != 1) {
//                 olddims[oldnd] = shape::shapeOf(oldShape)[oi];
//                 oldstrides[oldnd] = shape::stride(oldShape)[oi];
//                 oldnd++;
//             }
//         }

//         np = 1;
//         for (ni = 0; ni < newRank; ni++) {
//             np *= newShapeOf[ni];
//         }
//         op = 1;
//         for (oi = 0; oi < oldnd; oi++) {
//             op *= olddims[oi];
//         }
//         if (np != op) {
//             /* different total sizes; no hope */
//             delete[] olddims;
//             delete[] oldstrides;
//             delete[] newStrides;

//             return false;
//         }

//         if (np == 0) {
//             /* the current code does not handle 0-sized arrays, so give up */
//             delete[] olddims;
//             delete[] oldstrides;
//             delete[] newStrides;

//             return false;
//         }

//         /* oi to oj and ni to nj give the axis ranges currently worked with */
//         oi = 0;
//         oj = 1;
//         ni = 0;
//         nj = 1;

//         while (ni < newRank && oi < oldnd) {
//             np = newShapeOf[ni];
//             op = olddims[oi];

//             while (np != op) {
//                 if (np < op) {
//                     /* Misses trailing 1s, these are handled later */
//                     np *= newShapeOf[nj++];
//                 } else {
//                     op *= olddims[oj++];
//                 }
//             }

//             /* Check whether the original axes can be combined */
//             for (ok = oi; ok < oj - 1; ok++) {
//                 if (isFOrder) {
//                     if (oldstrides[ok + 1] != olddims[ok] * oldstrides[ok]) {
//                         /* not contiguous enough */
//                         delete[] olddims;
//                         delete[] oldstrides;
//                         delete[] newStrides;

//                         return false;
//                     }
//                 } else {
//                     /* C order */
//                     if (oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1]) {
//                         /* not contiguous enough */
//                         delete[] olddims;
//                         delete[] oldstrides;
//                         delete[] newStrides;

//                         return false;
//                     }
//                 }
//             }

//             /* Calculate new strides for all axes currently worked with */
//             if (isFOrder) {
//                 newStrides[ni] = oldstrides[oi];
//                 for (nk = ni + 1; nk < nj; nk++) {
//                     newStrides[nk] = newStrides[nk - 1] * newShapeOf[nk - 1];
//                 }
//             } else {
//                 /* C order */
//                 newStrides[nj - 1] = oldstrides[oj - 1];
//                 for (nk = nj - 1; nk > ni; nk--) {
//                     newStrides[nk - 1] = newStrides[nk] * newShapeOf[nk];
//                 }
//             }
//             ni = nj++;
//             oi = oj++;
//         }

//         if (ni >= 1) {
//             last_stride = newStrides[ni - 1];
//         } else {
//             last_stride = shape::elementWiseStride(oldShape);
//         }
//         if (isFOrder && ni >= 1) {
//             last_stride *= newShapeOf[ni - 1];
//         }
//         for (nk = ni; nk < newRank; nk++) {
//             newStrides[nk] = last_stride;
//         }

//         target[0] = newRank;
//         int cnt = 1;
//         for (int e = 0; e < newRank; e++)
//             target[cnt++] = newShapeOf[e];

//         for (int e = 0; e < newRank; e++)
//             target[cnt++] = newStrides[e];

//         target[shape::shapeInfoLength(newRank) - 3] = 0;
//         target[shape::shapeInfoLength(newRank) - 2] = 0;
//         target[shape::shapeInfoLength(newRank) - 1] = isFOrder ? 102 : 99;
//         sd::ArrayOptions::setDataType(target, sd::ArrayOptions::dataType(oldShape));

//         delete[] olddims;
//         delete[] oldstrides;
//         delete[] newStrides;

//         return true;
//     }

//////////////////////////////////////////////////////////////////////
// SD_INLINE SD_HOST bool reshapeC(const int oldRank, const sd::LongType* oldShapeInfo, const int newRank, const
// sd::LongType* newShape, sd::LongType* newShapeInfo) {

//         // PLEASE NOTE !: reshaping not-permuted (ews=1) array in f order (except insertion/elimination of unities)
//         will definitely cause allocation of new buffer for array elements
//         // also this function takes into account identical shapes automatically, namely in that case oldShapeInfo is
//         completely copied to newShapeInfo

//         newShapeInfo[0] = newRank;
//         memcpy(newShapeInfo + 1, newShape, newRank * sizeof(sd::LongType));

//         sd::LongType* newStrides       = shape::stride(newShapeInfo);
//         const sd::LongType* oldShape   = shape::shapeOf(const_cast<sd::LongType*>(oldShapeInfo));
//         const sd::LongType* oldStrides = shape::stride(const_cast<sd::LongType*>(oldShapeInfo));
//         sd::LongType oldStart(0), oldStop(1), newStart(0), newStop(1), newDim, oldDim;

//         while (newStart < newRank && oldStart < oldRank) {

//             newDim = newShape[newStart];
//             oldDim = oldShape[oldStart];

//             while (newDim != oldDim && newDim > 0 && oldDim > 0)
//                 if (newDim < oldDim) newDim *= newShape[newStop++];
//                 else                 oldDim *= oldShape[oldStop++];

//             // ------ Check whether the original axes can be combined ------ //
//             for (int step = 1, i = oldStart; i < oldStop - 1; ++i) {
//                 if(oldShape[i] == 1)                // skip unity-dimension and its stride
//                     continue;
//                 while((i + step) < oldRank && oldShape[i + step] == 1)
//                     ++step;                         // skip following unity-dimensions and its strides if such are
//                     present
//                 if((i + step) < oldRank && oldStrides[i] != oldShape[i + step] * oldStrides[i + step])
//                     return false;                   // not contiguous enough
//             }

//             newStrides[newStop - 1] = oldStrides[oldStop - 1];
//             for (int i = newStop - 1; i > newStart; --i)
//                 newStrides[i - 1] = newStrides[i] * newShape[i];

//             newStart = newStop++;
//             oldStart = oldStop++;
//         }

//         // rest of strides should be unities (if there is remainder in strides space, that is newStart < newRank)
//         for (int i = newStart; i < newRank; ++i)
//             newStrides[i] = 1;

//         newShapeInfo[2 * newRank + 3] = shape::order(oldShapeInfo);                 // order
//         newShapeInfo[2 * newRank + 2] = shape::elementWiseStride(oldShapeInfo);     // ews
//         newShapeInfo[2 * newRank + 1] = shape::type(oldShapeInfo);                  // type

//         return true;
//     }

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE bool reshapeC(const sd::LongType *oldShapeInfo, const char newOrder, const int newRank,
                                       const sd::LongType *newShape, sd::LongType *newShapeInfo) {
  // copy shape from newShape into newShapeInfo
  newShapeInfo[0] = newRank;
  memcpy(newShapeInfo + 1, newShape, newRank * sizeof(sd::LongType));

  // copy order
  newShapeInfo[2 * newRank + 3] = newOrder;

  return shape::reshapeC(oldShapeInfo, newShapeInfo);
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE bool reshapeC(const sd::LongType *oldShapeInfo, sd::LongType *newShapeInfo) {
  // newShapeInfo contains rank, shape and order; but no strides, type and ews

  const int newRank = shape::rank(newShapeInfo);

  // if oldShapeInfo is scalar or vector with length=1
  if (shape::length(oldShapeInfo) == 1) {
    for (sd::Unsigned i = 0; i < newRank; ++i) shape::stride(newShapeInfo)[i] = 1;
    newShapeInfo[2 * newRank + 1] = shape::type(oldShapeInfo);
    *shape::ews(newShapeInfo) = 1;
    return true;
  }

  const auto oldOrder = shape::order(oldShapeInfo);
  const auto newOrder = shape::order(newShapeInfo);
  const auto oldEws = shape::elementWiseStride(const_cast<sd::LongType *>(oldShapeInfo));

  if (oldEws > 0 && oldOrder != newOrder) return false;

  // *** FIRST STAGE - exclude unity dimensions from oldShapeInfo and newShapeInfo (if such are present of course),
  // since they don't affect on strides evaluation, however they complicate code

  // FIXME - indeed we don't need to allocate so large memory amount (4*SD_MAX_RANK), sufficient amount is
  // (2*oldNumOfNonUnities + 2*newNumOfNonUnities)
  sd::LongType tempBuffer[4 * SD_MAX_RANK];
  sd::LongType *oldShape = tempBuffer, *newShape = tempBuffer + 2 * SD_MAX_RANK, *oldStrides, *newStrides;

  // exclude unities from oldShapeInfo
  const int oldNumOfNonUnities = shape::excludeUnitiesFromShapeInfo(oldShapeInfo, oldShape, oldStrides);
  const int newNumOfNonUnities = shape::excludeUnitiesFromShapeInfo(newShapeInfo, newShape, newStrides);

  // *** SECOND STAGE - strides evaluation

  int oldStart(0), oldStop(1), newStart(0), newStop(1), newDim, oldDim;

  while (newStart < newNumOfNonUnities && oldStart < oldNumOfNonUnities) {
    newDim = newShape[newStart];
    oldDim = oldShape[oldStart];

    while (newDim != oldDim && newDim > 0 && oldDim > 0) {
      if (newDim < oldDim)
        newDim *= newShape[newStop++];
      else
        oldDim *= oldShape[oldStop++];
    }

    // check c-contiguous of old axes range
    for (sd::Unsigned i = oldStart; i < oldStop - 1; ++i)  // do not check value of last stride, it doesn't matter
      if (oldStrides[i] != oldShape[i + 1] * oldStrides[i + 1]) return false;  // not contiguous

    // fill newStrides in c manner
    newStrides[newStop - 1] = oldStrides[oldStop - 1];  // copy last stride
    for (int i = newStop - 2; i >= newStart; --i) newStrides[i] = newStrides[i + 1] * newShape[i + 1];

    newStart = newStop++;
    oldStart = oldStop++;
  }

  // fill new calculated strides into newShapeInfo, take into account possible unities in shape
  for (int j = 0, i = 0; i < newRank; ++i)
    shape::stride(newShapeInfo)[i] = (shape::shapeOf(newShapeInfo)[i] == 1) ? 1 : newStrides[j++];

  // set ews
  if (oldEws == 0)
    shape::checkStridesEwsAndOrder(newShapeInfo, newOrder, newNumOfNonUnities, newShape,
                                   newStrides);  // set ews and order
  else {
    newShapeInfo[2 * newRank + 3] = oldOrder;  // order
    *shape::ews(newShapeInfo) = oldEws;        // ews
  }
  newShapeInfo[2 * newShapeInfo[0] + 1] = 0;
  sd::ArrayOptions::copyDataType(newShapeInfo, oldShapeInfo);  // type

  return true;
}

SD_INLINE SD_HOST bool canReshape(const int oldRank, sd::LongType *oldShape, const int newRank,
                                  sd::LongType *newShapeOf, bool isFOrder) {
  sd::LongType oldnd;
  sd::LongType *oldDims = shape::copyOf(oldRank, shape::shapeOf(oldShape));
  sd::LongType *oldStrides = shape::copyOf(oldRank, shape::stride(oldShape));
  sd::LongType np, op, last_stride;
  sd::LongType oldStart, oldStop, ok, newStart, newStop, nk;
  auto newStrides = new sd::LongType[newRank];
  oldnd = 0;

  /*
   * Remove axes with dimension 1 from the old array. They have no effect
   * but would need special cases since their strides do not matter.
   */
  for (oldStart = 0; oldStart < oldRank; oldStart++) {
    if (shape::shapeOf(oldShape)[oldStart] != 1) {
      oldDims[oldnd] = shape::shapeOf(oldShape)[oldStart];
      oldStrides[oldnd] = shape::stride(oldShape)[oldStart];
      oldnd++;
    }
  }

  np = 1;
  for (newStart = 0; newStart < newRank; newStart++) {
    np *= newShapeOf[newStart];
  }
  op = 1;
  for (oldStart = 0; oldStart < oldnd; oldStart++) {
    op *= oldDims[oldStart];
  }
  if (np != op) {
    /* different total sizes; no hope */
    delete[] oldDims;
    delete[] oldStrides;
    delete[] newStrides;

    return false;
  }

  if (np == 0) {
    /* the current code does not handle 0-sized arrays, so give up */
    delete[] oldDims;
    delete[] oldStrides;
    delete[] newStrides;

    return false;
  }

  /* oldStart to oldStop and newStart to newStop give the axis ranges currently worked with */
  oldStart = 0;
  oldStop = 1;
  newStart = 0;
  newStop = 1;

  while (newStart < newRank && oldStart < oldnd) {
    np = newShapeOf[newStart];
    op = oldDims[oldStart];

    while (np != op) {
      if (np < op) {
        /* Misses trailing 1s, these are handled later */
        np *= newShapeOf[newStop++];
      } else {
        op *= oldDims[oldStop++];
      }
    }

    /* Check whether the original axes can be combined */
    for (ok = oldStart; ok < oldStop - 1; ok++) {
      if (isFOrder) {
        if (oldStrides[ok + 1] != oldDims[ok] * oldStrides[ok]) {
          /* not contiguous enough */
          delete[] oldDims;
          delete[] oldStrides;
          delete[] newStrides;

          return false;
        }
      } else {
        /* C order */
        if (oldStrides[ok] != oldDims[ok + 1] * oldStrides[ok + 1]) {
          /* not contiguous enough */
          delete[] oldDims;
          delete[] oldStrides;
          delete[] newStrides;

          return false;
        }
      }
    }

    /* Calculate new strides for all axes currently worked with */
    if (isFOrder) {
      newStrides[newStart] = oldStrides[oldStart];
      for (nk = newStart + 1; nk < newStop; nk++) {
        newStrides[nk] = newStrides[nk - 1] * newShapeOf[nk - 1];
      }
    } else {
      /* C order */
      newStrides[newStop - 1] = oldStrides[oldStop - 1];
      for (nk = newStop - 1; nk > newStart; nk--) {
        newStrides[nk - 1] = newStrides[nk] * newShapeOf[nk];
      }
    }
    newStart = newStop++;
    oldStart = oldStop++;
  }

  delete[] oldDims;
  delete[] oldStrides;
  delete[] newStrides;

  return true;
}

// this function checks the consistence of dimensions with array rank (negative dimensions, too large dimensions, too
// big number of dimensions) also it sorts input array of dimensions, this operation is also necessary for creating TAD
// object
SD_INLINE SD_HOST void checkDimensions(const int rank, std::vector<int> &dimensions) {
  int dimSize = dimensions.size();
  if (dimSize == 0) throw std::runtime_error("shape::checkDimensions method: array of dimensions is empty!");
  // check presence of negative dimensions and if they are present transform them to positive ones -dim -> rank - |dim|
  for (auto &dim : dimensions)
    if (dim < 0) dim += rank;
  // sort input array of dimensions, this operation is also necessary for creating TAD object in external methods
  if (dimSize > 1) {
    std::sort(dimensions.begin(), dimensions.end());
    // remove duplicates if they are present
    dimensions.erase(std::unique(dimensions.begin(), dimensions.end()), dimensions.end());
  }
  // check whether number of dimensions is to big (>rank)
  dimSize = dimensions.size();
  if (dimSize > rank)
    throw std::runtime_error(
        "shape::checkDimensions method: number of input dimensions is too big ( > rank of array)!");
  // check if min dimension is still negative and whether max dimension is bigger then rank-1
  if (dimensions[0] < 0 || dimensions.back() > (rank - 1))
    throw std::runtime_error(
        "shape::checkDimensions method: the negative dimension is still present in input array after transform or the "
        "too big dimension is present ( > rank of array) !");
}

// max array is outer for min array, min array is sub-array of max array
// function calculates the coordinates of min array (and saves them into minIdxs) given coordinates of max array
// (already stored in maxIdxs)
SD_INLINE SD_HOST_DEVICE void maxIndToMinInd(int *maxIdxs, int *minIdxs, const sd::LongType *maxShapeInfo,
                                             const sd::LongType *minShapeInfo, const int *dimsToExclude, int dimsLen) {
  const auto maxRank = shape::rank(maxShapeInfo);
  const auto minRank = shape::rank(minShapeInfo);

  // if(minRank >= maxRank)
  //     throw std::runtime_error("shape::maxIndToMinInd method: rank of min array should be smaller then rank of max
  //     array!");

  if (dimsLen == -1) dimsLen = maxRank - minRank;  // if size is not given (= -1) then it is equal to ranks difference

  if (maxRank == minRank) {
    if (dimsToExclude == nullptr) {  // --> means dimsToExclude == {0,1,2,...,dimsLen-1}

      for (int i = 0; i < maxRank; ++i) {
        if (i < dimsLen)
          minIdxs[i] = maxIdxs[i];
        else {
          if (maxIdxs[i] > minShapeInfo[i + 1])
            minIdxs[i] = maxIdxs[i] % minShapeInfo[i + 1];
          else if (maxIdxs[i] == minShapeInfo[i + 1])
            minIdxs[i] = 0;
          else
            minIdxs[i] = maxIdxs[i];
        }
      }
    } else {
      for (int i = 0, dim = 0; i < maxRank; ++i) {
        if (dim < dimsLen && dimsToExclude[dim] == i) {
          minIdxs[i] = maxIdxs[i];
          ++dim;
          continue;
        }

        if (maxIdxs[i] > minShapeInfo[i + 1])
          minIdxs[i] = maxIdxs[i] % minShapeInfo[i + 1];
        else if (maxIdxs[i] == minShapeInfo[i + 1])
          minIdxs[i] = 0;
        else
          minIdxs[i] = maxIdxs[i];
      }
    }
  } else {
    if (dimsToExclude == nullptr) {  // --> means dimsToExclude == {0,1,2,...,dimsLen-1}

      for (int i = 0; i < minRank; ++i) {
        if (maxIdxs[i + dimsLen] > minShapeInfo[i + 1])
          minIdxs[i] = maxIdxs[i + dimsLen] % minShapeInfo[i + 1];
        else if (maxIdxs[i + dimsLen] == minShapeInfo[i + 1])
          minIdxs[i] = 0;
        else
          minIdxs[i] = maxIdxs[i + dimsLen];
      }
    } else {
      for (int minI = 0, maxI = 0, dim = 0; maxI < maxRank; ++maxI) {
        if (dim < dimsLen && dimsToExclude[dim] == maxI) {
          ++dim;
          continue;
        }

        if (maxIdxs[maxI] == minShapeInfo[minI + 1])
          minIdxs[minI] = 0;
        else if (maxIdxs[maxI] > minShapeInfo[minI + 1])
          minIdxs[minI] = maxIdxs[maxI] % minShapeInfo[minI + 1];
        else
          minIdxs[minI] = maxIdxs[maxI];
        ++minI;
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType subArrayIndex(const sd::LongType maxIdx, const sd::LongType *maxShapeInfo,
                                                    const sd::LongType *minShapeInfo, const int *dimsToExclude,
                                                    const int dimsLen) {
  int maxIdxs[SD_MAX_RANK];
  shape::index2coords(const_cast<sd::LongType &>(maxIdx), maxShapeInfo, maxIdxs);

  int minIdxs[SD_MAX_RANK];
  maxIndToMinInd(maxIdxs, minIdxs, maxShapeInfo, minShapeInfo, dimsToExclude, dimsLen);

  return shape::coords2index(minShapeInfo, minIdxs);
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE sd::LongType subArrayOffset(const sd::LongType maxIdx, const sd::LongType *maxShapeInfo,
                                                     const sd::LongType *minShapeInfo, const int *dimsToExclude,
                                                     const int dimsLen) {
  int maxIdxs[SD_MAX_RANK];
  shape::index2coords(const_cast<sd::LongType &>(maxIdx), maxShapeInfo, maxIdxs);

  int minIdxs[SD_MAX_RANK];
  maxIndToMinInd(maxIdxs, minIdxs, maxShapeInfo, minShapeInfo, dimsToExclude, dimsLen);

  return getOffset(minShapeInfo, minIdxs);
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE int outerArrayOffsets(sd::LongType *maxOffsets, const sd::LongType minIdx,
                                               const sd::LongType *maxShapeInfo, const sd::LongType *minShapeInfo,
                                               int *memBuff, const int *dimsToExclude) {
  const auto rankMin = shape::rank(minShapeInfo);
  const auto rankMax = shape::rank(maxShapeInfo);

  // if(rankMin >= rankMax)
  //     throw std::runtime_error("shape::subArrayIndex method: rank of min array should be smaller then rank of max
  //     array!");

  const auto diff = rankMax - rankMin;  // the size of dimsToExclude is equal to diff

  int *indices = memBuff;
  int *increment = memBuff + rankMax;

  int N, minI, maxI;

  // calculate min per-dim-indices which corresponds to absolute minIdx index
  shape::index2coords(minIdx, minShapeInfo, indices);

  // transform storage indices to contain per-dim max indices, purpose - memory saving
  // fill increment array as well
  if (dimsToExclude == nullptr) {  // means dimsToExclude == {0,1,2,...,diff-1}
    for (minI = rankMin - 1, maxI = rankMax - 1; maxI >= diff; --maxI, --minI) {
      increment[maxI] = (maxShapeInfo[maxI + 1] == minShapeInfo[minI + 1]) ? 0 : minShapeInfo[minI + 1];
      indices[maxI] = indices[minI];
    }
    for (maxI = 0; maxI < diff; ++maxI) {
      increment[maxI] = 1;
      indices[maxI] = 0;
    }
  } else {
    for (N = diff - 1, minI = rankMin - 1, maxI = rankMax - 1; maxI >= 0; --maxI) {
      if (N >= 0 && dimsToExclude[N] == maxI) {
        increment[maxI] = 1;
        indices[maxI] = 0;
        --N;
      } else {
        increment[maxI] = (maxShapeInfo[maxI + 1] == minShapeInfo[minI + 1]) ? 0 : minShapeInfo[minI + 1];
        indices[maxI] = indices[minI--];
      }
    }
  }

  maxI = rankMax - 1;
  N = 0;
  int step;
  maxOffsets[N++] = shape::getOffset(maxShapeInfo, indices);

  // nested loops - producing of absolute indices for max array
  while (maxI >= 0) {
    if (increment[maxI] != 0) {
      indices[maxI] += increment[maxI];
      if (indices[maxI] >= maxShapeInfo[maxI + 1]) {
        indices[maxI] %= increment[maxI];  // restore initial value of indices[maxI]
        step = -1;
      } else {
        maxOffsets[N++] = shape::getOffset(maxShapeInfo, indices);
        step = rankMax - 1 - maxI;
      }
    } else if (maxI == rankMax - 1)
      step = -1;

    maxI += step;
  }
  return N;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE int outerArrayIndexes(int *maxIdxs, const sd::LongType minIdx,
                                               const sd::LongType *maxShapeInfo, const sd::LongType *minShapeInfo,
                                               const int *dimsToExclude) {
  const auto rankMin = shape::rank(minShapeInfo);
  const auto rankMax = shape::rank(maxShapeInfo);

  // if(rankMin >= rankMax)
  //     throw std::runtime_error("shape::subArrayIndex method: rank of min array should be smaller then rank of max
  //     array!");
  // if(rankMax > SD_MAX_RANK/2)
  //     throw std::runtime_error("shape::subArrayIndex method: rank of max array should be <= SD_MAX_RANK/2 !");

  const auto diff = rankMax - rankMin;  // the size of dimsToExclude is equal to diff

  int indices[SD_MAX_RANK], increment[SD_MAX_RANK];

  int N, minI, maxI;

  // calculate min per-dim-indices which corresponds to absolute minIdx index
  shape::index2coords(minIdx, minShapeInfo, indices);

  // transform storage indices to contain per-dim max indices, purpose - memory saving
  // fill increment array as well
  if (dimsToExclude == nullptr) {  // means dimsToExclude == {0,1,2,...,diff-1}
    for (minI = rankMin - 1, maxI = rankMax - 1; maxI >= diff; --maxI, --minI) {
      increment[maxI] = (maxShapeInfo[maxI + 1] == minShapeInfo[minI + 1]) ? 0 : minShapeInfo[minI + 1];
      indices[maxI] = indices[minI];
    }
    for (maxI = 0; maxI < diff; ++maxI) {
      increment[maxI] = 1;
      indices[maxI] = 0;
    }
  } else {
    for (N = diff - 1, minI = rankMin - 1, maxI = rankMax - 1; maxI >= 0; --maxI) {
      if (N >= 0 && dimsToExclude[N] == maxI) {
        increment[maxI] = 1;
        indices[maxI] = 0;
        --N;
      } else {
        increment[maxI] = (maxShapeInfo[maxI + 1] == minShapeInfo[minI + 1]) ? 0 : minShapeInfo[minI + 1];
        indices[maxI] = indices[minI--];
      }
    }
  }

  maxI = rankMax - 1;
  N = 0;
  int step;
  maxIdxs[N++] = shape::coords2index(maxShapeInfo, indices);

  // nested loops - producing of absolute indices for max array
  while (maxI >= 0) {
    if (increment[maxI] != 0) {
      indices[maxI] += increment[maxI];
      if (indices[maxI] >= maxShapeInfo[maxI + 1]) {
        indices[maxI] %= increment[maxI];  // restore initial value of indices[maxI]
        step = -1;
      } else {
        maxIdxs[N++] = shape::coords2index(maxShapeInfo, indices);
        step = rankMax - 1 - maxI;
      }
    } else if (maxI == rankMax - 1)
      step = -1;

    maxI += step;
  }
  return N;
}

SD_INLINE SD_HOST_DEVICE void shapeOldScalar(sd::DataType dataType, sd::LongType *const buffer, const char order) {
  buffer[0] = 2;
  buffer[1] = 1;
  buffer[2] = 1;
  buffer[3] = 1;
  buffer[4] = 1;
  buffer[6] = 1;
  buffer[7] = (int)order;

  sd::ArrayOptions::setDataType(buffer, dataType);
}

template <typename T1, typename T2>
SD_INLINE SD_HOST void convertT(T1 *from, T2 *to, sd::LongType length) {
  for (sd::LongType e = 0; e < length; e++) to[e] = (T2)from[e];
};

//////////////////////////////////////////////////////////////////////
SD_INLINE void calcOffsets(const sd::LongType *shapeInfo, sd::LongType *offsets, const char order) {
  // firstly consider simple case when ews > 0
  const sd::LongType ews = shape::elementWiseStride(shapeInfo);

  if (ews > 0) {
    // set offset for first sub-array, it is equal to zero always
    offsets[0] = 0;

    sd::LongType e = 0;
    if (order != shape::order(shapeInfo))
      for (int i = 1; i <= shape::rank(shapeInfo); ++i)
        if (shapeInfo[i] != 1) ++e;  // check whether input is CommonVector

    if (order == shape::order(shapeInfo) || e == 1) {  // e==1 means common vector
      e = 1;
      sd::LongType len = shape::length(shapeInfo);
      while (e < len) {
        offsets[e] = offsets[e - 1] + ews;
        e++;
      }
      return;
    }
  }

  shape::calcOffsets(shape::rank(shapeInfo), shape::shapeOf(const_cast<sd::LongType *>(shapeInfo)),
                     shape::stride(const_cast<sd::LongType *>(shapeInfo)), offsets, order);
}

//////////////////////////////////////////////////////////////////////
SD_INLINE void calcOffsets(const int rank, const sd::LongType *shape, const sd::LongType *strides,
                           sd::LongType *offsets, const char order) {
  const uint64_t len = shape::prodLong(shape, rank);

  // set offset for first sub-array, it is equal to zero always
  offsets[0] = 0;

  sd::Unsigned coords[SD_MAX_RANK];
  memset(coords, 0, sizeof(sd::Unsigned) * rank);

  if (order == 'c') {
    for (uint64_t i = 1; i < len; ++i) {
      int axis = rank - 1;
      offsets[i] = 0;
      while (coords[axis] == shape[axis] - 1) {
        offsets[i] -= (shape[axis] - 1) * strides[axis];
        coords[axis--] = 0;
      }
      ++coords[axis];
      offsets[i] += offsets[i - 1] + strides[axis];
    }
  } else {
    for (uint64_t i = 1; i < len; ++i) {
      int axis = 0;
      offsets[i] = 0;
      while (coords[axis] == shape[axis] - 1) {
        offsets[i] -= (shape[axis] - 1) * strides[axis];
        coords[axis++] = 0;
      }
      ++coords[axis];
      offsets[i] += offsets[i - 1] + strides[axis];
    }
  }

  // sd::LongType init = 0, i = 1;
  // // nested loops - calculation of sub-array offsets
  // if(order == 'c') {

  //     int rankMinusOne = rank - 1, j = rankMinusOne;

  //     while(j >= 0) {

  //         if(shape[j] == 1) { --j; continue; } // ignore dimensions equal to unity

  //         if(j == rankMinusOne) {              // last dimension
  //             for(sd::Unsigned l = 1; l < shape[j]; ++l)
  //                 offsets[i++] = offsets[i - 1] + strides[j];
  //             --j;
  //         }
  //         else if(coords[j] < shape[j] - 1) {
  //             init += strides[j];
  //             offsets[i++] = init;
  //             ++coords[j];
  //             j = rankMinusOne;
  //         }
  //         else {
  //             init -= (shape[j] - 1) * strides[j];
  //             coords[j--] = 0;
  //         }
  //     }
  // }
  // else {

  //     int j = 0;

  //     while(j < rank) {

  //         if(shape[j] == 1) { ++j; continue; } // ignore dimensions equal to unity

  //         if(j == 0) {              // last dimension
  //             for(sd::Unsigned l = 1; l < shape[j]; ++l)
  //                 offsets[i++] = offsets[i - 1] + strides[j];
  //             ++j;
  //         }
  //         else if(coords[j] < shape[j] - 1) {
  //             init += strides[j];
  //             offsets[i++] = init;
  //             ++coords[j];
  //             j = 0;
  //         }
  //         else {
  //             init -= (shape[j] - 1) * strides[j];
  //             coords[j++] = 0;
  //         }
  //     }
  // }
}

//////////////////////////////////////////////////////////////////////
SD_INLINE void SD_HOST_DEVICE checkStridesEwsAndOrder(sd::LongType *shapeInfo) {
  // FIXME - indeed we don't need to allocate so large memory amount (2*SD_MAX_RANK), sufficient amount is
  // (2*oldNumOfNonUnities + 2*newNumOfNonUnities)
  sd::LongType tempBuffer[2 * SD_MAX_RANK];
  sd::LongType *shape = tempBuffer, *strides;

  // exclude unities from shapeInfo
  const int numOfNonUnities = shape::excludeUnitiesFromShapeInfo(shapeInfo, shape, strides);

  shape::checkStridesEwsAndOrder(shapeInfo, shape::order(shapeInfo), numOfNonUnities, shape, strides);
}

//////////////////////////////////////////////////////////////////////
SD_INLINE void SD_HOST_DEVICE checkStridesEwsAndOrder(sd::LongType *shapeInfo, const char proposedOrder,
                                                      const int numOfNonUnities, const sd::LongType *shapeNoUnities,
                                                      const sd::LongType *stridesNoUnities) {
  const int rank = shape::rank(shapeInfo);

  if (shape::length(shapeInfo) == 1) {
    *shape::ews(shapeInfo) = 1;
    shapeInfo[rank * 2 + 3] = (int)proposedOrder;
    return;
  }

  if (numOfNonUnities == 1) {  // case of common vector
    *shape::ews(shapeInfo) = *stridesNoUnities;
    shapeInfo[rank * 2 + 3] = (int)proposedOrder;
    return;
  }

  bool contiguous = true;

  //*** check whether strides are in c contiguous order ***//
  for (sd::Unsigned i = 0; i < numOfNonUnities - 1; ++i) {
    if (stridesNoUnities[i] != shapeNoUnities[i + 1] * stridesNoUnities[i + 1]) {
      contiguous = false;
      break;
    }
  }

  if (contiguous) {
    *shape::ews(shapeInfo) = stridesNoUnities[numOfNonUnities - 1];
    shapeInfo[rank * 2 + 3] = 99;
    return;
  }

  contiguous = true;

  //*** check whether strides are in f contiguous order ***//
  for (sd::Unsigned i = 1; i < numOfNonUnities; ++i) {
    if (stridesNoUnities[i] != shapeNoUnities[i - 1] * stridesNoUnities[i - 1]) {
      contiguous = false;
      break;
    }
  }

  if (contiguous) {
    *shape::ews(shapeInfo) = stridesNoUnities[0];
    shapeInfo[rank * 2 + 3] = 102;
    return;
  }

  *shape::ews(shapeInfo) = 0;
  shapeInfo[rank * 2 + 3] = (int)proposedOrder;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE void calcSubArrsShapeInfoAndOffsets(const sd::LongType *wholeShapeInfo,
                                                             const sd::LongType numOfSubArrs, const int dimsSize,
                                                             const int *dimsToExclude, sd::LongType *subArrShapeInfo,
                                                             sd::LongType *subArrOffsets, bool keepUnitiesInShape) {
  const int rank = shape::rank(wholeShapeInfo);

  if (dimsSize == rank || dimsSize == 0) {  // means there is one sub-array and it coincides with whole array, return
                                            // copy of wholeShapeInfo and one zero offset in this case
    memcpy(subArrShapeInfo, wholeShapeInfo, shape::shapeInfoLength(rank) * sizeof(sd::LongType));
    *subArrOffsets = 0;
    return;
  }

  const int subArrRank = keepUnitiesInShape ? rank : rank - dimsSize;

  subArrShapeInfo[0] = subArrRank;                                     // rank
  subArrShapeInfo[2 * subArrRank + 1] = 0;                             // clear (to avoid uninitialized)
  sd::ArrayOptions::copyDataType(subArrShapeInfo, wholeShapeInfo);     // type
  subArrShapeInfo[2 * subArrRank + 3] = shape::order(wholeShapeInfo);  // order

  sd::LongType *shape = new sd::LongType[dimsSize];
  sd::LongType *strides = new sd::LongType[dimsSize];

  for (int k = subArrRank - 1, j = dimsSize - 1, i = rank - 1; i >= 0; --i) {
    if (j >= 0 && i == dimsToExclude[j]) {
      strides[j] = shape::stride(wholeShapeInfo)[i];
      shape[j--] = shape::shapeOf(wholeShapeInfo)[i];

      if (keepUnitiesInShape) {
        shape::shapeOf(subArrShapeInfo)[k] = 1;
        shape::stride(subArrShapeInfo)[k--] = shape::stride(wholeShapeInfo)[i];
      }
    } else {
      shape::shapeOf(subArrShapeInfo)[k] = shape::shapeOf(wholeShapeInfo)[i];
      shape::stride(subArrShapeInfo)[k--] = shape::stride(wholeShapeInfo)[i];
    }
  }

  // calculation of sub-array offsets (subArrOffsets)
  shape::calcOffsets(dimsSize, shape, strides, subArrOffsets);

  // evaluate ews
  shape::checkStridesEwsAndOrder(subArrShapeInfo);

  delete[] strides;
  delete[] shape;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE void calcSubArrShapeInfoAndOffset(const sd::LongType *idx, const sd::LongType *maxShapeInfo,
                                            sd::LongType *minShapeInfo, sd::LongType &minOffset,
                                            const bool keepUnitiesInShape, const bool isStrided,
                                            const int numOfUntiesInMinShape) {
  const sd::Unsigned maxRank = shape::rank(maxShapeInfo);
  minOffset = 0;
  sd::Unsigned first, last, stride, n(isStrided ? 3 : 2);

  minShapeInfo[0] = keepUnitiesInShape ? maxRank : maxRank - numOfUntiesInMinShape;

  for (sd::Unsigned step = 0, j = 0, i = 0; i < maxRank; ++i, step += n) {
    if (idx[step] == idx[step + 1]) {  // means whole dimension
      shape::shapeOf(minShapeInfo)[j] = shape::shapeOf(maxShapeInfo)[i];
      shape::stride(minShapeInfo)[j++] = shape::stride(maxShapeInfo)[i];
    } else {
      first = idx[step] >= 0 ? idx[step] : idx[step] + shape::sizeAt(maxShapeInfo, i) + 1;
      last = idx[step + 1] >= 0 ? idx[step + 1] : idx[step + 1] + shape::sizeAt(maxShapeInfo, i) + 1;

      if (last < first) throw("shape::calcSubArrShapeInfoAndOffset: negative range in input indexes is found!");

      if (isStrided) {
        stride = idx[step + 2];
        last /*resulting sub-array axis*/ = (last - first + stride - 1) / stride;  // ceil (last - first) / stride;
      } else {
        stride = 1;
        last /*resulting sub-array axis*/ = last - first;
      }

      minOffset += first * shape::stride(maxShapeInfo)[i];

      if (!keepUnitiesInShape && last == 1) continue;

      shape::shapeOf(minShapeInfo)[j] = last;
      shape::stride(minShapeInfo)[j++] =
          last == 1 ? shape::stride(maxShapeInfo)[i] : shape::stride(maxShapeInfo)[i] * stride;
    }
  }

  minShapeInfo[2 * shape::rank(minShapeInfo) + 1] = 0;                           // zero
  minShapeInfo[2 * shape::rank(minShapeInfo) + 3] = shape::order(maxShapeInfo);  // order
  sd::ArrayOptions::copyDataType(minShapeInfo, maxShapeInfo);                    // type

  shape::checkStridesEwsAndOrder(minShapeInfo);
}

//////////////////////////////////////////////////////////////////////
SD_INLINE void SD_HOST_DEVICE index2coords(sd::LongType index, const sd::LongType *shapeInfo, sd::LongType *coords) {
  for (sd::Unsigned i = shapeInfo[0]; i > 1; --i) {
    coords[i - 1] = index % shapeInfo[i];
    index /= shapeInfo[i];
  }
  coords[0] = index;  // last iteration
}

//////////////////////////////////////////////////////////////////////
SD_INLINE void SD_HOST_DEVICE index2coords(sd::LongType index, const sd::LongType *shapeInfo, int *coords) {
  for (sd::Unsigned i = shapeInfo[0]; i > 1; --i) {
    coords[i - 1] = static_cast<int>(index) % static_cast<int>(shapeInfo[i]);
    index /= static_cast<int>(shapeInfo[i]);
  }
  coords[0] = static_cast<int>(index);  // last iteration
}

//////////////////////////////////////////////////////////////////////
SD_INLINE void SD_HOST_DEVICE index2coords(sd::LongType index, const sd::LongType *shapeInfo, sd::Unsigned *coords) {
  for (sd::Unsigned i = shapeInfo[0]; i > 1; --i) {
    coords[i - 1] = static_cast<sd::Unsigned>(index) % static_cast<sd::Unsigned>(shapeInfo[i]);
    index /= static_cast<sd::Unsigned>(shapeInfo[i]);
  }
  coords[0] = static_cast<sd::Unsigned>(index);  // last iteration
}

//////////////////////////////////////////////////////////////////////
SD_INLINE void SD_HOST_DEVICE index2coords(sd::LongType index, const int rank, const sd::LongType *shape,
                                           sd::LongType *coords) {
  for (sd::Unsigned i = rank - 1; i > 0; --i) {
    coords[i] = index % shape[i];
    index /= shape[i];
  }
  coords[0] = index;  // last iteration
}

//////////////////////////////////////////////////////////////////////
SD_INLINE void SD_HOST_DEVICE index2coords(sd::LongType index, const int rank, const sd::LongType *shape, int *coords) {
  for (sd::Unsigned i = rank - 1; i > 0; --i) {
    coords[i] = index % shape[i];
    index /= shape[i];
  }
  coords[0] = index;  // last iteration
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE void index2coords(sd::LongType index, const sd::LongType *shapeInfo, const int *dims,
                                           const int dimsLen, int *coords) {
  for (sd::Unsigned i = dimsLen - 1; i > 0; --i) {
    const auto ind = dims[i];
    coords[ind] = index % shapeInfo[1 + ind];
    index /= shapeInfo[1 + ind];
  }
  coords[dims[0]] = index;  // last iteration
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE void index2coordsCPU(const sd::LongType &startIndex, const sd::LongType &index,
                                              const sd::LongType *shapeInfo, sd::LongType *coords) {
  if (startIndex == index) {
    shape::index2coords(index, shapeInfo, coords);
  } else {
    int axis = shapeInfo[0] - 1;
    while (coords[axis] == shape::sizeAt(shapeInfo, axis) - 1) coords[axis--] = 0;
    ++coords[axis];
  }
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE void index2coordsCPU(const sd::LongType &startIndex, const sd::LongType &index,
                                              const sd::LongType *shapeInfo, int *coords) {
  if (startIndex == index) {
    shape::index2coords(index, shapeInfo, coords);
  } else {
    int axis = shapeInfo[0] - 1;
    while (coords[axis] == shape::sizeAt(shapeInfo, axis) - 1) coords[axis--] = 0;
    ++coords[axis];
  }
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE int excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo, sd::LongType *&shapeNoUnities,
                                                         sd::LongType *&stridesNoUnities) {
  const int rank = shape::rank(inShapeInfo);
  const int numOfNonUnities = shape::numOfNonUnitDims(rank, shape::shapeOf(inShapeInfo));

  if (numOfNonUnities == rank) {  // no unities in shape, no copy procedure
    shapeNoUnities = const_cast<sd::LongType *>(inShapeInfo) + 1;
    stridesNoUnities = const_cast<sd::LongType *>(inShapeInfo) + 1 + rank;
    return numOfNonUnities;
  }

  for (sd::Unsigned j = 0, i = 0; i < rank; ++i) {
    if (shape::shapeOf(inShapeInfo)[i] != 1) {
      shapeNoUnities[j] = shape::shapeOf(inShapeInfo)[i];
      shapeNoUnities[numOfNonUnities + j++] = shape::stride(inShapeInfo)[i];
    }
  }

  stridesNoUnities = shapeNoUnities + numOfNonUnities;

  return numOfNonUnities;
}

//////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE void excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo, const int *dimsToExclude,
                                                          const int dimsSize, sd::LongType *outShapeInfo) {
  outShapeInfo[0] = inShapeInfo[0] - dimsSize;

  for (sd::Unsigned j = 0, k = 0, i = 0; i < inShapeInfo[0]; ++i) {
    if (j < dimsSize && i == dimsToExclude[j]) {
      ++j;
      continue;
    }

    shape::shapeOf(outShapeInfo)[k] = shape::shapeOf(inShapeInfo)[i];
    shape::stride(outShapeInfo)[k++] = shape::stride(inShapeInfo)[i];
  }
  outShapeInfo[2 * outShapeInfo[0] + 1] = 0;
  sd::ArrayOptions::copyDataType(outShapeInfo, inShapeInfo);          // type
  *shape::ews(outShapeInfo) = shape::elementWiseStride(inShapeInfo);  // ews
  outShapeInfo[2 * outShapeInfo[0] + 3] = shape::order(inShapeInfo);  // order
}

//////////////////////////////////////////////////////////////////////
// SD_INLINE SD_HOST_DEVICE void index2coordsCPU(const sd::LongType& startIndex, const sd::LongType& index, const
// sd::LongType *shapeInfo, const int* dims, const int dimsLen, int *coords) {

//     if(startIndex == index) {
//         shape::index2coords(index, shapeInfo, dims, dimsLen, coords);
//     }
//     else {
//         int i = dimsLen - 1;
//         while(coords[dims[i]] == shape::sizeAt(shapeInfo, dims[i]) - 1)
//             coords[dims[i--]] = 0;
//         ++coords[dims[i]];
//     }
// }

//////////////////////////////////////////////////////////////////////
// SD_INLINE SD_HOST_DEVICE void calcOffsets(const sd::LongType *xShapeInfo, sd::LongType*& xOffsets, const sd::LongType
// *yShapeInfo, sd::LongType*& yOffsets, const sd::LongType* zShapeInfo, sd::LongType*& zOffsets, const char order) {

//     // we assume all array have same length
//     const sd::LongType len = shape::length(xShapeInfo);

//     const sd::LongType xEws = shape::elementWiseStride(xShapeInfo);
//     const sd::LongType yEws = shape::elementWiseStride(yShapeInfo);
//     const sd::LongType zEws = shape::elementWiseStride(zShapeInfo);

//     const char xOrder = shape::order(xShapeInfo);
//     const char yOrder = shape::order(yShapeInfo);
//     const char zOrder = shape::order(zShapeInfo);

//     const bool shapesSame = shape::shapeEquals(xShapeInfo, yShapeInfo, zShapeInfo);

//     if (xEws == 1 && yEws == 1 && zEws == 1 && xOrder == yOrder && xOrder == zOrder && (xOrder == 'c' || shapesSame))
//     {
//         xOffsets = yOffsets = zOffsets = nullptr;
//     }
//     else if(xEws == 1 && yEws == 1 && xOrder == yOrder && (xOrder == 'c' || shape::shapeEquals(xShapeInfo,
//     yShapeInfo))) {
//         xOffsets = yOffsets = nullptr;
//         zOffsets = new sd::LongType[len];
//         shape::calcOffsets(zShapeInfo, zOffsets, xOrder);
//     }
//     else if(xEws == 1 && zEws == 1 && xOrder == zOrder && (xOrder == 'c' || shape::shapeEquals(xShapeInfo,
//     zShapeInfo))) {
//         xOffsets = zOffsets = nullptr;
//         yOffsets = new sd::LongType[len];
//         shape::calcOffsets(yShapeInfo, yOffsets, xOrder);
//     }
//     else if(yEws == 1 && zEws == 1 && yOrder == zOrder && (yOrder == 'c' || shape::shapeEquals(yShapeInfo,
//     zShapeInfo))) {
//         yOffsets = zOffsets = nullptr;
//         xOffsets = new sd::LongType[len];
//         shape::calcOffsets(xShapeInfo, xOffsets, yOrder);
//     }
//     else if(xEws == 1) {
//         xOffsets = nullptr;
//         PRAGMA_OMP_PARALLEL_SECTIONS
//         {
//             PRAGMA_OMP_SECTION
//             {
//                 yOffsets = new sd::LongType[len];
//                 shape::calcOffsets(yShapeInfo, yOffsets, xOrder);
//             }
//             PRAGMA_OMP_SECTION
//             {
//                 zOffsets = new sd::LongType[len];
//                 shape::calcOffsets(zShapeInfo, zOffsets, xOrder);
//             }
//         }
//     }
//     else if(yEws == 1) {
//         yOffsets = nullptr;
//         PRAGMA_OMP_PARALLEL_SECTIONS
//         {
//             PRAGMA_OMP_SECTION
//             {
//                 xOffsets = new sd::LongType[len];
//                 shape::calcOffsets(xShapeInfo, xOffsets, yOrder);
//             }
//             PRAGMA_OMP_SECTION
//             {
//                 zOffsets = new sd::LongType[len];
//                 shape::calcOffsets(zShapeInfo, zOffsets, yOrder);
//             }
//         }
//     }
//     else if(zEws == 1) {
//         zOffsets = nullptr;
//         PRAGMA_OMP_PARALLEL_SECTIONS
//         {
//             PRAGMA_OMP_SECTION
//             {
//                 xOffsets = new sd::LongType[len];
//                 shape::calcOffsets(xShapeInfo, xOffsets, zOrder);
//             }
//             PRAGMA_OMP_SECTION
//             {
//                 yOffsets = new sd::LongType[len];
//                 shape::calcOffsets(yShapeInfo, yOffsets, zOrder);
//             }
//         }
//     }
//     else if(shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo, zShapeInfo)) {
//         xOffsets = new sd::LongType[len];
//         shape::calcOffsets(xShapeInfo, xOffsets);
//         yOffsets = zOffsets = xOffsets;
//     }
//     else if(shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)) {
//         PRAGMA_OMP_PARALLEL_SECTIONS
//         {
//             PRAGMA_OMP_SECTION
//             {
//                 xOffsets = new sd::LongType[len];
//                 shape::calcOffsets(xShapeInfo, xOffsets);
//             }
//             PRAGMA_OMP_SECTION
//             {
//                 zOffsets = new sd::LongType[len];
//                 shape::calcOffsets(zShapeInfo, zOffsets);
//             }
//         }
//         yOffsets = xOffsets;
//     }
//     else if(shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
//         PRAGMA_OMP_PARALLEL_SECTIONS
//         {
//             PRAGMA_OMP_SECTION
//             {
//                 xOffsets = new sd::LongType[len];
//                 shape::calcOffsets(xShapeInfo, xOffsets);
//             }
//             PRAGMA_OMP_SECTION
//             {
//                 yOffsets = new sd::LongType[len];
//                 shape::calcOffsets(yShapeInfo, yOffsets);
//             }
//         }
//         zOffsets = xOffsets;
//     }
//     else {
//         PRAGMA_OMP_PARALLEL_SECTIONS
//         {
//             PRAGMA_OMP_SECTION
//             {
//                 xOffsets = new sd::LongType[len];
//                 shape::calcOffsets(xShapeInfo, xOffsets);
//             }
//             PRAGMA_OMP_SECTION
//             {
//                 yOffsets = new sd::LongType[len];
//                 shape::calcOffsets(yShapeInfo, yOffsets);
//             }
//             PRAGMA_OMP_SECTION
//             {
//                 zOffsets = new sd::LongType[len];
//                 shape::calcOffsets(zShapeInfo, zOffsets);
//             }
//         }
//     }
// }

//////////////////////////////////////////////////////////////////////
// SD_INLINE SD_HOST_DEVICE void calcOffsets(const sd::LongType *xShapeInfo, sd::LongType*& xOffsets, const sd::LongType
// *yShapeInfo, sd::LongType*& yOffsets, const char order) {

//     // we assume all array have same length
//     const sd::LongType len = shape::length(xShapeInfo);

//     const sd::LongType xEws = shape::elementWiseStride(xShapeInfo);
//     const sd::LongType yEws = shape::elementWiseStride(yShapeInfo);

//     const char xOrder = shape::order(xShapeInfo);
//     const char yOrder = shape::order(yShapeInfo);

//     const bool shapesSame = shape::shapeEquals(xShapeInfo, yShapeInfo);

//     if (xEws == 1 && yEws == 1 && xOrder == yOrder && (xOrder == 'c' || shapesSame)) {
//         xOffsets = yOffsets = nullptr;
//     }
//     else if(xEws == 1) {
//         xOffsets = nullptr;
//         yOffsets = new sd::LongType[len];
//         shape::calcOffsets(yShapeInfo, yOffsets, xOrder);
//     }
//     else if(yEws == 1) {
//         yOffsets = nullptr;
//         xOffsets = new sd::LongType[len];
//         shape::calcOffsets(xShapeInfo, xOffsets, yOrder);
//     }
//     else if(shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)) {
//         xOffsets = new sd::LongType[len];
//         shape::calcOffsets(xShapeInfo, xOffsets);
//         yOffsets = xOffsets;
//     }
//     else {
//         PRAGMA_OMP_PARALLEL_SECTIONS
//         {
//             PRAGMA_OMP_SECTION
//             {
//                 xOffsets = new sd::LongType[len];
//                 shape::calcOffsets(xShapeInfo, xOffsets);
//             }
//             PRAGMA_OMP_SECTION
//             {
//                 yOffsets = new sd::LongType[len];
//                 shape::calcOffsets(yShapeInfo, yOffsets);
//             }
//         }
//     }
// }

//////////////////////////////////////////////////////////////////////
// SD_INLINE SD_HOST_DEVICE sd::LongType strideOverContigAxis(const int axis, const sd::LongType* inShapeInfo) {

//     sd::LongType result = 9223372036854775807LL;

//     for(sd::Unsigned i = 0; i < shape::rank(inShapeInfo); ++i) {

//         const auto currentStride = shape::stride(inShapeInfo)[i];

//         if(i == axis || shape::shapeOf(inShapeInfo)[i] == 1)
//             continue;

//         if(result > currentStride)
//             result = currentStride;
//     }

//     return result == 9223372036854775807LL ? 1 : result;
// }

}  // namespace shape

#endif /* SHAPE_H_ */
