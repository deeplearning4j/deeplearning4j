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
                                        int rank_ = 0, int offset_ = 0, int elementWiseStride_ = 0, bool isEmpty_ = false)
                : shape(shape_),
                  stride(stride_),
                  order(order_),
                  rank(rank_),
                  offset(offset_),
                  elementWiseStride(elementWiseStride_),
                  isEmpty(isEmpty_) {}

        sd::LongType *shape;
        sd::LongType *stride;
        char order;
        int rank;
        int offset;
        int elementWiseStride;
        bool isEmpty;
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



    template <typename T>
    SD_LIB_EXPORT SD_HOST_DEVICE void fill(T *buffer, T value, sd::LongType length);


    SD_LIB_EXPORT SD_HOST_DEVICE int tadIndexForLinear(int linearIndex, int tadLength);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType tadLength(const sd::LongType *shapeInfo, const sd::LongType *dimension, sd::LongType dimensionLength);

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
    SD_LIB_EXPORT SD_HOST_DEVICE int tadElementWiseStride(sd::LongType *shapeInfo, sd::LongType *dimension,
                                                          sd::LongType dimensionLength);

    SD_LIB_EXPORT SD_HOST_DEVICE bool canReshape(const sd::LongType oldRank, sd::LongType *oldShape, const sd::LongType newRank,
                                                 sd::LongType *newShape, bool isFOrder);

    SD_LIB_EXPORT SD_HOST_DEVICE bool reshapeC(const sd::LongType *oldShapeInfo, const char newOrder, const sd::LongType newRank,
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

    SD_LIB_EXPORT SD_HOST_DEVICE void transposeInplace(sd::LongType *shapeBuffer);

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
    SD_LIB_EXPORT SD_HOST_DEVICE void updateStrides(const long long int rank, const sd::LongType *shapeOnly, sd::LongType *stridesOnly,
                                                    const char order);

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
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, long long int rank,
                                                                  long long int startNum);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank, int startNum,
                                                                  sd::LongType *ret);

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, long long int rank,
                                                           long long int startNum);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, long long int rank,
                                                           long long int startNum,
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
    SD_LIB_EXPORT SD_HOST_DEVICE int computeElementWiseStride(long long int rank, sd::LongType const *shape, sd::LongType const *stride,
                                                              int isFOrder);

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
    SD_LIB_EXPORT SD_HOST_DEVICE int computeElementWiseStride(sd::LongType rank, sd::LongType const *shape, sd::LongType const *stride,
                                                              sd::LongType isFOrder, sd::LongType const *dimension, sd::LongType dimensionLength);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeInfoOnlyShapeAndStride(sd::LongType const *shapeInfo, sd::LongType *dimension,
                                                                           sd::LongType dimensionLength, bool reverseCopyStride);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeInfoOnlyShapeAndStride(const sd::LongType *shapeInfo, sd::LongType *dimension,
                                                                           sd::LongType dimensionLength, bool reverseCopyStride,
                                                                           sd::LongType *buffer);



    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *permuteShapeBuffer(sd::LongType const *shapeBuffer, sd::LongType *rearrange);

    SD_LIB_EXPORT SD_HOST_DEVICE void permuteShapeBufferInPlace(sd::LongType *shapeBuffer, sd::LongType *rearrange, sd::LongType *out);

    SD_LIB_EXPORT SD_HOST_DEVICE void doPermuteShapeInfo(sd::LongType *shapeBuffer, const sd::LongType *rearrange, sd::LongType len = -1);

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
 * which will give us the ability to iterate along an element
 * wise stride.
 */

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *createPermuteIndexes(sd::LongType originalRank, sd::LongType *dimension, sd::LongType dimensionLength);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *computeResultShape(const sd::LongType *originalShapeBuffer, sd::LongType *dimension,
                                                                  sd::LongType dimensionLength);

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
    SD_LIB_EXPORT SD_HOST_DEVICE void permute(ShapeInformation **info, sd::LongType *rearrange, long long int rank);

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

    SD_LIB_EXPORT SD_HOST_DEVICE bool isCommonVector(const sd::LongType *shapeInfo, long long int &posOfNonUnityDim);

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
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *slice(sd::LongType *shape);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType slices(sd::LongType *shapeBuffer);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *sliceOfShapeBuffer(sd::LongType sliceIdx, sd::LongType *shapeBuffer);
/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 3
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType shapeInfoLength(sd::LongType rank);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType shapeInfoLength(sd::LongType *shape);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType shapeInfoLength(const sd::LongType *shape);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType shapeInfoByteLength(sd::LongType rank);

    SD_LIB_EXPORT SD_HOST_DEVICE size_t shapeInfoByteLength(const sd::LongType *shapeInfo);

    SD_LIB_EXPORT SD_HOST_DEVICE size_t shapeInfoByteLength(const sd::LongType *shapeInfo);

/**
 * Returns the rank portion of
 * an information buffer
 */
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType rank(const sd::LongType *shapeInfo);

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
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType reductionIndexElementWiseStride(sd::LongType *buffer, sd::LongType *dimension,
                                                                              sd::LongType dimensionLength);

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
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *everyIndexBut(sd::LongType const *indexes, int indexesLength, int begin, int end);

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
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *keep(volatile sd::LongType *data, const sd::LongType *index, int indexLength,
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
    SD_LIB_EXPORT SD_HOST_DEVICE void convertT(T1 *from, T2 *to, sd::LongType length);
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
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType lengthPerSlice(sd::LongType rank, sd::LongType const *shape, const sd::LongType *dimension,
                                                             sd::LongType dimensionLength);

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType sliceOffsetForTensor(sd::LongType rank, sd::LongType index, sd::LongType const *shape,
                                                                   sd::LongType const *tensorShape, sd::LongType tensorShapeLength,
                                                                   const sd::LongType *dimension, sd::LongType dimensionLength);

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType sliceOffsetForTensor(sd::LongType index, sd::LongType tensorLength, sd::LongType lengthPerSlice2);


/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType tensorsAlongDimension(sd::LongType *shapeInfo, sd::LongType *dimension, sd::LongType dimensionLength);

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


// all three arrays should have same rank
// all three arrays should have same dimensions or some of them are 1 (that is satisfy broadcasting principle), strides
// may be different shapeInfo1 - first array should have max length compared to rest of two arrays
    SD_LIB_EXPORT SD_HOST_DEVICE void getOffsetBroadcast(const sd::LongType &startInd, const sd::LongType ind,
                                                         const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2,
                                                         const sd::LongType *shapeInfo3, const bool sameOffsets12,
                                                         const bool sameOffsets13, sd::LongType *coords, sd::LongType &offset1,
                                                         sd::LongType &offset2, sd::LongType &offset3);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *createShapeInfo(sd::LongType *shape, sd::LongType *stride, long long int rank);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *createShapeInfo(sd::LongType *shape, sd::LongType *stride, long long int rank,
                                                               sd::LongType *buffer);



    SD_LIB_EXPORT SD_HOST_DEVICE void index2coordsCPU(const sd::LongType &startIndex, const sd::LongType &index,
                                                      const sd::LongType *shapeInfo, sd::LongType *coords);
    SD_LIB_EXPORT SD_HOST_DEVICE void index2coordsCPU(const sd::LongType &startIndex, const sd::LongType &index,
                                                      const sd::LongType *shapeInfo, sd::LongType *coords);




/**
 * Convert coordinates to the corresponding linear index (sequence number in other words)
 * for example if shape is {2, 4} and coordinates [1, 1] then index 5 is returned
 */
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const sd::LongType *coords);
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo,sd::LongType *coords);
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, sd::LongType *coords);
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType rank, const sd::LongType *shape,
                                                           sd::LongType *indices);
/**
 * take into account only dimensions stored in tadDims, tadDims must be sorted in increasing order!
 */
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const sd::LongType*dims,
                                                           const sd::LongType dimsLen, const sd::LongType *coords);

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
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType getIndexOffset(sd::LongType index, const sd::LongType *shapeInfo);
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType indexOffset(sd::LongType index, const sd::LongType *lShapeInfo,
                                                          const sd::LongType *uShapeInfo, const bool useUnsigned);

    SD_LIB_EXPORT SD_HOST_DEVICE void printShapeInfo(const sd::LongType *shapeInfo);

    SD_LIB_EXPORT SD_HOST_DEVICE void printShapeInfoLinear(const sd::LongType *shapeInfo);


    SD_LIB_EXPORT SD_HOST const char *shapeToString(const sd::LongType *shapeInfo, const char *message);



    SD_LIB_EXPORT SD_HOST_DEVICE void printShapeInfoLinear(const char *msg, const sd::LongType *shapeInfo);

    SD_LIB_EXPORT SD_HOST_DEVICE void printShapeInfoLinear(const char *msg, int rank, const sd::LongType *shape,
                                                           const sd::LongType *strides);

    SD_LIB_EXPORT SD_HOST_DEVICE void printIntArray(const sd::LongType *arr, const int length);
    SD_LIB_EXPORT SD_HOST_DEVICE void printIntArray(const int *arr, const int length);

    SD_LIB_EXPORT SD_HOST_DEVICE void printArray(float *arr, int length);

    template <typename T>
    SD_LIB_EXPORT SD_HOST_DEVICE void printArray(T *arr, int length, const char *message);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeBufferOfNpy(sd::LongType rank, sd::LongType *shape, bool fortranOrder);

    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeBufferOfNpy(cnpy::NpyArray arr);


// this function checks the consistence of dimensions with array rank (negative dimensions, too large dimensions, too
// big number of dimensions) also sort input array of dimensions, this operation is also necessary for creating TAD
// object
    SD_LIB_EXPORT SD_HOST_DEVICE void checkDimensions(const sd::LongType rank, std::vector<sd::LongType> *dimensions);

// function calculates linear index of array min, min is sub-array of max, index to be returned is min-array's index and
// corresponds to maxIdx of max array dimsToExclude - should be sorted in increasing order








// function calculates absolute offset of min array, min is sub-array of max, offset to be returned corresponds to
// maxIdx of max array dimsToExclude - should be sorted in increasing order
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType subArrayOffset(const sd::LongType maxIdx, const sd::LongType *maxShapeInfo,
                                                             const sd::LongType *minShapeInfo,
                                                             const sd::LongType *dimsToExclude = nullptr,
                                                             const sd::LongType dimsLen = -1);

// max array is outer for min array, min array is sub-array of max array
// function calculates the coordinates of min array (and saves them into minIdxs) given coordinates of max array
// (already stored in maxIdxs) dimsToExclude - should be sorted in increasing order dimsLen - length of dimsToExclude,
// if not set (= -1), then it is calculated as maxRank - minRank
    SD_LIB_EXPORT SD_HOST_DEVICE void maxIndToMinInd(sd::LongType *maxIdxs, sd::LongType *minIdxs, const sd::LongType *maxShapeInfo,
                                                     const sd::LongType *minShapeInfo,
                                                     const sd::LongType *dimsToExclude = nullptr,
                                                     sd::LongType dimsLen = -1);


//////////////////////////////////////////////////////////////////////
    SD_INLINE void SD_HOST_DEVICE index2coords(sd::LongType index, const sd::LongType *shapeInfo, sd::LongType *coords) {
        for (sd::LongType i = shapeInfo[0]; i > 1; --i) {
            coords[i - 1] = index % shapeInfo[i];
            index /= shapeInfo[i];
        }
        coords[0] = index;  // last iteration
    }


//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
    SD_INLINE void SD_HOST_DEVICE index2coords(sd::LongType index, const sd::LongType rank, const sd::LongType *shape,
                                               sd::LongType *coords) {
        for (sd::LongType i = rank - 1; i > 0; --i) {
            coords[i] = index % shape[i];
            index /= shape[i];
        }
        coords[0] = index;  // last iteration
    }

//////////////////////////////////////////////////////////////////////
    SD_INLINE SD_HOST_DEVICE void index2coords(sd::LongType index, const sd::LongType *shapeInfo, const sd::LongType *dims,
                                               const sd::LongType  dimsLen, sd::LongType *coords) {
        for (sd::LongType i = dimsLen - 1; i > 0; --i) {
            const auto ind = dims[i];
            coords[ind] = index % shapeInfo[1 + ind];
            index /= shapeInfo[1 + ind];
        }
        coords[dims[0]] = index;  // last iteration
    }

    SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType subArrayIndex(sd::LongType maxIdx, const sd::LongType *maxShapeInfo,
                                                                      const sd::LongType *minShapeInfo) {
        sd::LongType maxIdxs[SD_MAX_RANK];
        shape::index2coords(const_cast<sd::LongType &>(maxIdx), maxShapeInfo, maxIdxs);

        sd::LongType minIdxs[SD_MAX_RANK];
        maxIndToMinInd(maxIdxs, minIdxs, maxShapeInfo, minShapeInfo, nullptr,-1);

        return shape::coords2index(minShapeInfo, minIdxs);
    }

// calculate indexes of max-array, these output indexes correspond to one minIdx index of min-array which is sub-array
// of max-array dimsToExclude - should be sorted in increasing order
    SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType outerArrayIndexes(sd::LongType *maxIdxs, const sd::LongType minIdx, const sd::LongType *maxShapeInfo,
                                                                const sd::LongType *minShapeInfo, const sd::LongType *dimsToExclude = nullptr);

// calculate offsets of max-array, these offsets correspond to one minIdx index of min-array which is sub-array of
// max-array maxOffsets - will contain calculated offsets of max-array, buffer for maxOffsets should be allocated
// beforehand dimsToExclude - should be sorted in increasing order memBuff - auxiliary memory buffer (size = 2 *
// max_rank) for coordinates and increments storing, should be allocated beforehand
    SD_LIB_EXPORT SD_HOST_DEVICE int outerArrayOffsets(sd::LongType *maxOffsets, const sd::LongType minIdx,
                                                       const sd::LongType *maxShapeInfo, const sd::LongType *minShapeInfo,
                                                       sd::LongType *memBuff,
                                                       const sd::LongType *dimsToExclude = nullptr);

// calculates offsets for entities (elements or sub-arrays), shape in context of sub-array means dimensions excluded
// from outer array rank is equal to size of shape
    SD_LIB_EXPORT void calcOffsets(const long long int rank, const sd::LongType *shape, const sd::LongType *strides,
                                   sd::LongType *offsets, const char order = 'c');
    SD_LIB_EXPORT void calcOffsets(const sd::LongType *shapeInfo, sd::LongType *offsets, const char order = 'c');

    SD_LIB_EXPORT SD_HOST_DEVICE void shapeOldScalar(sd::DataType dtype, sd::LongType *const buffer, const char order);

// deduce order and element-wise stride
// if array is scalar or unit length vector then ews = 1 and order is preserved
// if array is common vector then ews = stride of non-unity dimension and order is preserved
// if strides are normal/contiguous then ews = 1 and corresponding order is set, otherwise ews = 0 and order is
// preserved
    SD_LIB_EXPORT SD_HOST_DEVICE void checkStridesEwsAndOrder(sd::LongType *shapeInfo, const char proposedOrder,
                                                              const long long int numOfNonUnitDims, const sd::LongType *shapeNoUnities,
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
    SD_LIB_EXPORT SD_HOST_DEVICE void calcSubArrsShapeInfoAndOffsets(const sd::LongType *wholeShapeInfo,
                                                                     const sd::LongType numOfSubArrs, const long long int dimsSize,
                                                                     const sd::LongType *dimsToExclude, sd::LongType *subArrShapeInfo,
                                                                     sd::LongType *subArrOffsets, bool keepUnitiesInShape = false);

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
                                                    const long long int numOfUntiesInMinShape = 0);

/**
 * for example inShapeInfo is {3, 2,1,4, 4,4,1, 16384,1,99}
 * then output shapeNoUnities will contain {2,4, 4,1} - that is only shape and strides, no rank/type/ews/order
 * stridesNoUnities will point on strides in shapeNoUnities that is on {4,1}
 * returns number of non-unity dimensions in inShapeInfo
 * if there is no unities in inShapeInfo, then no copy procedure will be performed and shapeNoUnities/stridesNoUnities
 * will point on corresponding places in inShapeInfo
 */
    SD_LIB_EXPORT SD_HOST_DEVICE int excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo, sd::LongType *&shapeNoUnities,
                                                                 sd::LongType *&stridesNoUnities);

/**
 * for example inShapeInfo is {3, 2,1,3,1,4,  12,12,4,4,1, 16384,1,99}, dimsToExclude(points on unity dimensions) =
 * {1,3}, dimsSize = 2 then outShapeInfo will contain {3, 2,3,4, 12,4,1, 16384,1,99}
 */
    SD_LIB_EXPORT SD_HOST_DEVICE void excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo, const sd::LongType *dimsToExclude, const long long int dimsSize, sd::LongType *outShapeInfo);

/**
 * get stride over contiguous axis (contiguous axis must have stride = 1)
 * for example when inShapeInfo is {4, 2,5,4,3,  60,1,5,20, 16384,0,99} then output is 5 (that is smallest stride in
 * inShapeInfo except those equal to 1)
 */

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

    SD_INLINE SD_HOST_DEVICE bool shapeEquals(const int shape1Rank, const sd::LongType *shape1, const int shape2Rank,
                                              const sd::LongType *shape2) {
      printf("shapeEquals\n");
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
    SD_INLINE SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, int rank) { return calcStrides(shape, rank, 1); }

    SD_INLINE SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, int rank, sd::LongType *ret) {
        return calcStrides(shape, rank, 1, ret);
    }

// check whether input dimensions are permuted, not permuted dimensions order have to be 0,....,rank-1
    template <typename T>
    SD_INLINE SD_HOST_DEVICE bool isDimPermuted(const T *dimensions, const sd::LongType dimSize) {
        for (int i = 0; i < dimSize - 1; ++i)
            if (dimensions[i] > dimensions[i + 1]) return true;

        return false;
    }

    SD_INLINE SD_HOST_DEVICE int computeElementWiseStride(sd::LongType rank, const sd::LongType  *shape, const sd::LongType  *stride,
                                                          sd::LongType isFOrder, const sd::LongType  *dimension, sd::LongType dimensionLength) {
        if (dimensionLength == 1) {
            return stride[dimension[0]];
        }
        return 0;
    }

//////////////////////////////////////////////////////////////////////
    SD_INLINE SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const sd::LongType *indices) {
        sd::LongType index, shift = 1;


        index = indices[shapeInfo[0] - 1];
        for (sd::LongType i = shapeInfo[0]; i > 1; --i) {
            shift *= shapeInfo[i];
            index += shift * indices[i - 2];
        }

        return index;
    }

    SD_INLINE SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo,  sd::LongType *indices) {
        return coords2index(shapeInfo, const_cast<const sd::LongType *>(indices));
    }


//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
    SD_INLINE SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType rank, const sd::LongType *shape,
                                                       const sd::LongType *indices) {
        sd::LongType index, shift = 1;
        ;

        index = indices[rank - 1];
        for (sd::LongType i = rank - 1; i >= 1; --i) {
            shift *= shape[i];
            index += shift * indices[i - 1];
        }

        return index;
    }

    SD_INLINE SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType rank, const sd::LongType *shape,
                                                       sd::LongType *indices) {
        return coords2index(rank, shape, const_cast<const sd::LongType *>(indices));
    }

    template <typename T>
    SD_INLINE SD_HOST_DEVICE void fill(T *buffer, T value, sd::LongType length) {
        PRAGMA_OMP_SIMD
        for (int e = 0; e < length; e++) buffer[e] = value;
    }

    SD_INLINE SD_HOST_DEVICE sd::LongType coords2index(const sd::LongType *shapeInfo, const sd::LongType *dims, const sd::LongType dimsLen, const sd::LongType *coords) {
        sd::LongType index, shift = 1;
        ;

        index = coords[dims[dimsLen - 1]];
        for (sd::LongType i = dimsLen - 1; i >= 1; --i) {
            shift *= shapeInfo[dims[i]];
            index += shift * coords[i - 1];
        }

        return index;
    }


//////////////////////////////////////////////////////////////////////
    SD_INLINE SD_HOST_DEVICE sd::LongType getIndexOffset(sd::LongType index, const sd::LongType *shapeInfo) {
        char order = shape::order(shapeInfo);
        const sd::LongType ews = shape::elementWiseStride(shapeInfo);
        if (order == 'c') {
            if (ews == 1)
                return index;
            else if (ews > 1)
                return ews * index;
            else if(ews <= 0) { // not contiguous enough for EWS
                sd::LongType coords[SD_MAX_RANK];
                shape::index2coords(index,shapeInfo,coords);
                auto getOffset = shape::getOffset(shapeInfo,coords,0);
                return getOffset;
            }
        }

        //f ordering
        sd::LongType offset = 0;

        sd::LongType rank = shape::rank(shapeInfo);
        for (sd::LongType i =rank; i > 1; --i) {
            offset += (index % shapeInfo[i]) * shapeInfo[i + shapeInfo[0]];
            index /= shapeInfo[i];
        }

        offset += index * shapeInfo[1 + shapeInfo[0]];  // last iteration

        return offset;
    }

//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
    SD_INLINE SD_HOST_DEVICE sd::LongType indexOffset(sd::LongType index, const sd::LongType *lShapeInfo,
                                                      const sd::LongType *uShapeInfo, const bool useUnsigned) {
        if (useUnsigned) return getIndexOffset(static_cast<sd::LongType>(index), uShapeInfo);

        return getIndexOffset(index, lShapeInfo);
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

    SD_INLINE SD_HOST_DEVICE bool isCommonVector(const sd::LongType *shapeInfo, long long int &posOfNonUnityDim) {
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

    SD_INLINE SD_HOST_DEVICE sd::LongType const *detachShape(sd::LongType const *originalShape) {
        sd::LongType *newShape = new sd::LongType[shape::shapeInfoLength(originalShape)];
        memcpy(newShape, originalShape, shape::shapeInfoByteLength(originalShape));

        return newShape;
    }

    SD_INLINE SD_HOST_DEVICE sd::LongType *copyShape(sd::LongType const *originalShape) {
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

        for (sd::LongType i = 0; i < rank; ++i)
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

    SD_INLINE SD_HOST_DEVICE void setShape(sd::LongType *shapeInfo,sd::LongType *shape) {
        auto shapeOf =  shapeInfo + 1;
        int rank = shape::rank(shapeInfo);
        if(rank < 1) {
            shapeOf[0] = 0;
            return;
        }
        for(int i = 0; i < rank; i++) {
            shapeOf[i] = shape[i];
        }
    }



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
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
    SD_INLINE SD_HOST_DEVICE sd::LongType *slice(sd::LongType *shape) { return shape + 1; }

    SD_INLINE SD_HOST_DEVICE sd::LongType slices(sd::LongType *shapeBuffer) {
        return static_cast<int>(shape::shapeOf(shapeBuffer)[0]);
    }

/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 4
 * A shape buffer contains:
 * rank
 * shape elements
 * stride elements
 * flags such as array type like empty and data type
 * element wise stride
 * offset
 * ordering
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
    SD_INLINE SD_HOST_DEVICE sd::LongType shapeInfoLength(sd::LongType rank) {
        //rank takes up 1 element + usual elements
        if(rank < 1)
            //shape of 0 (scalar) even has elements for shape and stride
            return static_cast<sd::LongType>(1 * 2 + 4);
        // FIXME magic numbers
        return static_cast<sd::LongType>(rank * 2 + 4);
    }

    SD_INLINE SD_HOST_DEVICE sd::LongType shapeInfoLength(sd::LongType *shape) {
        return shapeInfoLength(static_cast<sd::LongType>(shape[0]));
    }

    SD_INLINE SD_HOST_DEVICE sd::LongType shapeInfoLength(const sd::LongType *shape) {
        return shapeInfoLength(static_cast<sd::LongType>(shape[0]));
    }

    SD_INLINE SD_HOST_DEVICE sd::LongType shapeInfoByteLength(sd::LongType rank) {
        //scalar formula isn't correct
        if(rank == 0)
            return static_cast<sd::LongType>(6 * sizeof(sd::LongType));
        // FIXME magic numbers
        return static_cast<sd::LongType>((rank * 2 + 4) * sizeof(sd::LongType));
    }

    SD_INLINE SD_HOST_DEVICE size_t shapeInfoByteLength(const sd::LongType *shapeInfo) {

        // FIXME magic numbers
        return shapeInfoByteLength((sd::LongType)shapeInfo[0]);
    }

/**
 * Returns the rank portion of
 * an information buffer
 */
    SD_INLINE SD_HOST_DEVICE sd::LongType rank(const sd::LongType *buffer) { return static_cast<sd::LongType>(buffer[0]); }



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


    SD_INLINE SD_HOST_DEVICE void setStride(sd::LongType *buffer,sd::LongType *strides) {
        auto stridesRet =  buffer + (1 + rank(buffer));
        int rank = shape::rank(buffer);
        if(rank < 1) {
            buffer[2] = 0;
            return;
        }
        for(int i = 0; i < rank; i++) {
            stridesRet[i] = strides[i];
        }
    }


/**
 * Returns the stride portion of an information
 * buffer
 */
    SD_INLINE SD_HOST_DEVICE sd::LongType *stride(sd::LongType *buffer) { return buffer + (1 + rank(buffer)); }

    SD_INLINE SD_HOST_DEVICE sd::LongType *stride(const sd::LongType *buffer) {
        return stride(const_cast<sd::LongType *>(buffer));
    }

/**
 * Compute the length of the given shape
 */
    SD_INLINE SD_HOST_DEVICE sd::LongType length(const sd::LongType *shapeInfo) {
        const sd::LongType rank = shape::rank(shapeInfo);

        if (rank == 0) {
            if (isEmpty(shapeInfo)) return 0L;
            return 1L;
        }

        if (rank == 1) return shapeInfo[1];

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
    SD_INLINE SD_HOST_DEVICE void setOffset(sd::LongType *buffer,sd::LongType offset) {
        buffer[shape::shapeInfoLength(shape::rank(buffer)) - 2] =  offset;
    }

/***
 * Returns the offset
 * portion of an information buffer
 */
    SD_INLINE SD_HOST_DEVICE sd::LongType offset(sd::LongType *buffer) {
        return buffer[shape::shapeInfoLength(shape::rank(buffer)) - 2];
    }

    SD_INLINE SD_HOST_DEVICE void setExtra(sd::LongType *buffer,sd::LongType extra) {
        if(buffer == nullptr)
            THROW_EXCEPTION("Buffer is nullptr");
        sd::LongType  rank = buffer[0];
        if(rank < 0 || rank > SD_MAX_RANK)
            THROW_EXCEPTION("Invalid shape buffer passed in. Rank is < 0  or > 32. May have been deallocated");
        sd::LongType  idx = 0;

        //rank takes up 1 element + usual elements
        if(rank == 0)
            idx = 3;
        else {
            // FIXME magic numbers
            idx = rank + rank + 1;
        }
        buffer[idx] = extra;
    }

    SD_INLINE SD_HOST_DEVICE sd::LongType &extra(sd::LongType *buffer) {
        sd::LongType  rank = buffer[0];
        sd::LongType  idx = 0;
        //rank takes up 1 element + usual elements
        if(rank == 0)
            idx = 3;
        else
            // FIXME magic numbers
            idx = rank + rank + 1;
        return buffer[idx];
    }

    SD_INLINE SD_HOST_DEVICE sd::LongType extra(const sd::LongType *buffer) {
        sd::LongType  rank = buffer[0];
        sd::LongType  idx = 0;
        //rank takes up 1 element + usual elements
        if(rank == 0)
            idx = 3;
        else
            // FIXME magic numbers
            idx = rank + rank + 1;
        return buffer[idx];
    }

/**
 * Returns the ordering
 * for this shape information buffer
 */
    SD_INLINE SD_HOST_DEVICE char order(const sd::LongType *buffer) {
        // FIXME magic numbers
        int len = shapeInfoLength(buffer[0]);
        return static_cast<char>(buffer[len - 1]);
    }

/**
 * Returns the ordering
 * for this shape information buffer
 */
    SD_INLINE SD_HOST_DEVICE char setOrder(sd::LongType *buffer,char c) {
        // FIXME magic numbers
        if(c != 'c' && c != 'f') {
            std::string errorMessage;
            errorMessage += "Invalid order from shape descriptor: ";
            errorMessage += std::to_string(c);
            THROW_EXCEPTION(errorMessage.c_str());
        }
        int len = shapeInfoLength(buffer[0]);
        buffer[len - 1] = static_cast<sd::LongType>(c);
        return c;
    }


/**
 * Returns type
 */
    SD_INLINE SD_HOST_DEVICE sd::LongType type(const sd::LongType *shapeInfo) {
        if(shapeInfo[0] < 1)
            return shapeInfo[2 * 1 + 1];
        return shapeInfo[2 * shapeInfo[0] + 1];

    }

/**
 * Returns the element wise stride for this information
 * buffer
 */
    SD_INLINE SD_HOST_DEVICE sd::LongType elementWiseStride(const sd::LongType *buffer) {
        return buffer[shapeInfoLength(static_cast<sd::LongType>(buffer[0])) - 2];
    }


/**
 * Returns the element wise stride for this information
 * buffer
 */
    SD_INLINE SD_HOST_DEVICE sd::LongType setElementWiseStride(sd::LongType *buffer,sd::LongType elementWiseStride) {
        return buffer[shapeInfoLength(static_cast<sd::LongType>(buffer[0])) - 2] = elementWiseStride;
    }

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
    SD_INLINE SD_HOST_DEVICE int isScalar(const sd::LongType *info) {
        if(shape::isEmpty(info))
            return 0;
        const sd::LongType rank = shape::rank(info);
        if(rank == 0) return 1;
        auto shape = shape::shapeOf(info);

        if (rank > 2) return 0;
        if (rank == 1) return shape[0] <= 1;
        if (rank == 2)
            return shape[0] == 1 &&
                 shape[1] == 1;

        return 0;
    }

/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
    SD_INLINE SD_HOST_DEVICE int isScalar(volatile ShapeInformation *info) {
        const sd::LongType rank = info->rank;

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

        for (sd::LongType e = 0; e < static_cast<sd::LongType>(shape::rank(shapeInfo1)); ++e)
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

#ifndef __JAVACPP_HACK__

    SD_INLINE SD_HOST_DEVICE sd::LongType sizeAt(const sd::LongType *shapeInfo, const sd::LongType dim) {
        if (0 == rank(shapeInfo)) return 1;
        if (dim >= 0)
            return shapeInfo[1 + dim];
        else
            return shapeInfo[1 + (rank(shapeInfo) + dim)];
    }

    SD_INLINE SD_HOST_DEVICE sd::LongType strideAt(const sd::LongType *shapeInfo, const sd::LongType dim) {
        if (0 == rank(shapeInfo)) return 1;
        if (dim >= 0)
            return shapeInfo[1 + rank(shapeInfo) + dim];
        else
            return shapeInfo[1 + 2 * rank(shapeInfo) + dim];
    }
#endif
/**
 * This method does SOFT comparison for two shape buffers, we compare only rank & shapes
 *
 * @param shape
 * @return
 */
    SD_INLINE SD_HOST_DEVICE bool equalsSoft(const sd::LongType *shapeA, const sd::LongType *shapeB) {
        if (shapeA[0] != shapeB[0]) {
            return false;
        }

        if (shape::isEmpty(shapeA) && shape::isEmpty(shapeB)) {
            return true;
        }

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
 * Generate a reverse
 * copy of the data
 */

    template <typename T>
    SD_INLINE SD_HOST_DEVICE T *reverseCopy(T const *data, sd::LongType length) {
        if (length < 1) return nullptr;

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
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */

    SD_INLINE SD_HOST_DEVICE sd::LongType sliceOffsetForTensor(sd::LongType index, sd::LongType tensorLength, sd::LongType lengthPerSlice2) {
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

        for (sd::LongType i = 1; i <= shapeInfo[0]; ++i)
            if (shapeInfo[i] != 1) offset += indices[i - 1] * shapeInfo[shapeInfo[0] + i];

        return offset;
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

#ifdef __CUDACC__
    SD_DEVICE SD_INLINE void sweepShapeInfoBuffer(sd::LongType *shapeInfoBuffer, sd::LongType *targetBuffer) {
  // we read first element, to find out length of our shapeInfoBuffer
  int rank = shapeInfoBuffer[0];
  int len = shape::shapeInfoLength(rank);
  for (int i = threadIdx.x; i < len; i += blockDim.x) targetBuffer[i] = shapeInfoBuffer[i];
}
#endif

    SD_INLINE SD_HOST_DEVICE bool isContiguous(const sd::LongType *shapeInfo) {
        return (order(shapeInfo) == 'c') && (elementWiseStride(shapeInfo) > 0);
    }

// this function checks the consistence of dimensions with array rank (negative dimensions, too large dimensions, too
// big number of dimensions) also it sorts input array of dimensions, this operation is also necessary for creating TAD
// object
    SD_INLINE SD_HOST_DEVICE void checkDimensions(const sd::LongType rank, std::vector<sd::LongType> *dimensions) {
        int dimSize = dimensions->size();
        if (dimSize == 0) {
            THROW_EXCEPTION("shape::checkDimensions method: array of dimensions is empty!");
        }
        // check presence of negative dimensions and if they are present transform them to positive ones -dim -> rank - |dim|
        for (auto &dim : *dimensions)
            if (dim < 0) dim += rank;
        // sort input array of dimensions, this operation is also necessary for creating TAD object in external methods
        if (dimSize > 1) {
            std::sort(dimensions->begin(), dimensions->end());
            // remove duplicates if they are present
            dimensions->erase(std::unique(dimensions->begin(), dimensions->end()), dimensions->end());
        }
        // check whether number of dimensions is to big (>rank)
        dimSize = dimensions->size();
        if (dimSize > rank)
            THROW_EXCEPTION(
                    "shape::checkDimensions method: number of input dimensions is too big ( > rank of array)!");
        // check if min dimension is still negative and whether max dimension is bigger then rank-1
        if (dimensions->at(0) < 0 || dimensions->back() > (rank - 1))
            THROW_EXCEPTION(
                    "shape::checkDimensions method: the negative dimension is still present in input array after transform or the "
                    "too big dimension is present ( > rank of array) !");
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
    SD_INLINE SD_HOST_DEVICE void convertT(T1 *from, T2 *to, sd::LongType length) {
        for (sd::LongType e = 0; e < length; e++) to[e] = (T2)from[e];
    };



//////////////////////////////////////////////////////////////////////
    SD_INLINE SD_HOST_DEVICE void index2coordsCPU(const sd::LongType &startIndex, const sd::LongType &index,
                                                  const sd::LongType *shapeInfo, sd::LongType *coords) {
        if (startIndex == index) {
            shape::index2coords(index, shapeInfo, coords);
        } else {
            sd::LongType axis = shapeInfo[0] - 1;
            while (coords[axis] == shape::sizeAt(shapeInfo, axis) - 1) coords[axis--] = 0;
            ++coords[axis];
        }
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

// host device codes which were duplicated in shape.cpp but guarded from inclusion
#if defined(SD_CUDA)


//////////////////////////////////////////////////////////////////////

    SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool isEmpty(const sd::LongType *shapeInfo) {
        int result = (static_cast<sd::LongType>((shape::extra(shapeInfo)) & static_cast<sd::LongType>(ARRAY_EMPTY)));
        bool isEmptyResult = result ==  static_cast<sd::LongType>(ARRAY_EMPTY);
        return isEmptyResult;
    }

// max array is outer for min array, min array is sub-array of max array
// function calculates the coordinates of min array (and saves them into minIdxs) given coordinates of max array
// (already stored in maxIdxs)
    SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void maxIndToMinInd(sd::LongType *maxIdxs, sd::LongType *minIdxs, const sd::LongType *maxShapeInfo,
                                                               const sd::LongType *minShapeInfo, const sd::LongType *dimsToExclude, sd::LongType dimsLen) {
        const auto maxRank = shape::rank(maxShapeInfo);
        const auto minRank = shape::rank(minShapeInfo);


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

    SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType subArrayOffset(const sd::LongType maxIdx,
                                                                       const sd::LongType *maxShapeInfo,
                                                                       const sd::LongType *minShapeInfo,
                                                                       const sd::LongType *dimsToExclude, const sd::LongType dimsLen) {
        sd::LongType maxIdxs[SD_MAX_RANK];
        shape::index2coords(const_cast<sd::LongType &>(maxIdx), maxShapeInfo, maxIdxs);

        sd::LongType minIdxs[SD_MAX_RANK];
        maxIndToMinInd(maxIdxs, minIdxs, maxShapeInfo, minShapeInfo, dimsToExclude, dimsLen);

        return getOffset(minShapeInfo, minIdxs);
    }

    SD_LIB_EXPORT SD_INLINE SD_DEVICE SD_HOST_DEVICE int outerArrayOffsets(sd::LongType *maxOffsets, const sd::LongType minIdx,
                                                                           const sd::LongType *maxShapeInfo,
                                                                           const sd::LongType *minShapeInfo, sd::LongType *memBuff,
                                                                           const sd::LongType *dimsToExclude) {
        const auto rankMin = shape::rank(minShapeInfo);
        const auto rankMax = shape::rank(maxShapeInfo);

        const auto diff = rankMax - rankMin;  // the size of dimsToExclude is equal to diff

        sd::LongType *indices = memBuff;
        sd::LongType *increment = memBuff + rankMax;

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

    SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool strideDescendingCAscendingF(const sd::LongType *shapeBuffer) {
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



//////////////////////////////////////////////////////////////////////


#endif

    SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE int tadElementWiseStride(sd::LongType *shapeInfo, sd::LongType *dimension,
                                                                    sd::LongType dimensionLength) {
        return reductionIndexElementWiseStride(shapeInfo, dimension, dimensionLength);
    }

}  // namespace shape

#endif /* SHAPE_H_ */
