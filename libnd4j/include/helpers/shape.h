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
*
*
*  Notes on this file. ALl functions here
*  should be inlined.
*  Inlined functions in both cpu and cuda
*  allow different compilation units to embed the functions.
*
*  We need these functions to be in the header in order to keep
*  the functions agnostic.
*
*  Note that SD_INLINE here at the time of writing (Mar 15 2024) was changed
*  from always_inline from gcc.
*
*
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



/**
* Returns whether the given shape
* info has the flag view set.
*/

SD_HOST_DEVICE bool isViewConst(const sd::LongType *shapeInfo) ;


/**
* Returns whether the
* given shape info has an empty flag set.
*/

SD_HOST_DEVICE bool isEmptyConst(const sd::LongType *shapeInfo);


/**
* Returns whether the given shape
* info has the flag view set.
*/

SD_HOST_DEVICE bool isView(sd::LongType *shapeInfo);

/**
* Returns whether the
* given shape info has an empty flag set.
*/

SD_HOST_DEVICE bool isEmpty(sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE bool shapeEquals(int shape1Rank, const sd::LongType *shape1, int shape2Rank,
                                              const sd::LongType *shape2);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *detachShape(sd::LongType *originalShape);


SD_LIB_EXPORT SD_HOST_DEVICE bool shapeEquals(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2);

SD_LIB_EXPORT SD_HOST_DEVICE bool shapeEquals(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2,
                                              const sd::LongType *shapeInfo3);

SD_LIB_EXPORT SD_HOST_DEVICE bool strideEquals(int shape1Rank, sd::LongType const *shape1, int shape2Rank,
                                               sd::LongType const *shape2);

SD_LIB_EXPORT SD_HOST_DEVICE bool strideEquals(sd::LongType const *shapeInfo1, sd::LongType const *shapeInfo2);



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


SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType tadLength(const sd::LongType *shapeInfo, const sd::LongType *dimension,
                                                    sd::LongType dimensionLength);


/**
* Returns whether the given shape
* info has the flag view set.
*/

SD_LIB_EXPORT SD_HOST_DEVICE bool isView(sd::LongType *shapeInfo);
/**
* Returns whether the
* given shape info has an empty flag set.
*/

SD_LIB_EXPORT SD_HOST_DEVICE bool isEmpty(sd::LongType *shapeInfo);



SD_LIB_EXPORT SD_HOST_DEVICE bool reshapeC(const sd::LongType *oldShapeInfo, const char newOrder, sd::LongType newRank, const sd::LongType *newShape,
                                           sd::LongType *newShapeInfo);
/**
* newShapeInfo contains rank, shape and order only, no strides/ews/type
*/
SD_LIB_EXPORT SD_HOST_DEVICE bool reshapeC(const sd::LongType *oldShapeInfo, sd::LongType *newShapeInfo);

/**
* Get the shape info buffer
* for the given rank and shape.
*/
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeBuffer(sd::LongType rank, sd::DataType dtype, sd::LongType  *shape);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *shapeBuffer(sd::LongType rank, sd::DataType dtype, sd::LongType  *shape,
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
#ifndef __JAVACPP_HACK__

SD_DEVICE SD_LIB_EXPORT sd::LongType *cuMalloc(sd::LongType *buffer, long size);
#endif
#endif


SD_LIB_EXPORT SD_HOST_DEVICE void updateStrides(sd::LongType *shape, const char order, bool resetStridesIfView);
SD_LIB_EXPORT SD_HOST_DEVICE void updateStrides(const sd::LongType rank, const sd::LongType *shapeOnly,
                                                sd::LongType *stridesOnly, const char order);





SD_LIB_EXPORT SD_HOST_DEVICE void doPermuteShapeInfo(sd::LongType *shapeBuffer, const sd::LongType *rearrange,
                                                     sd::LongType len = -1);


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
SD_LIB_EXPORT SD_HOST_DEVICE void permute(ShapeInformation **info, sd::LongType *rearrange, sd::LongType rank);

/**
* Returns whether the
* given shape is a vector or not
* @param shape the shape of the array
* @param rank the rank of cthe shape
*/
SD_LIB_EXPORT SD_HOST_DEVICE int isVector(sd::LongType const *shape, int rank);



SD_LIB_EXPORT SD_HOST_DEVICE int isVector(sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE bool isLikeVector(sd::LongType const *shapeInfo, int &posOfNonUnityDim);

SD_LIB_EXPORT SD_HOST_DEVICE bool isCommonVector(const sd::LongType *shapeInfo, sd::LongType &posOfNonUnityDim);

SD_LIB_EXPORT SD_HOST_DEVICE bool isRowVector(sd::LongType *shapeInfo);


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


SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE int isMatrix(const sd::LongType *shapeInfo);
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
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void copyTo(sd::LongType length, T  *from, T *to) {
  for (sd::LongType i = 0; i < length; i++) {
    to[i] = from[i];
  }
}


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
* Returns the stride portion of an information
* buffer
*/
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *stride(sd::LongType *buffer);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *stride(const sd::LongType *buffer);



SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType length(const sd::LongType *shapeInfo);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType length(std::initializer_list<int> &shape);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType length(std::initializer_list<sd::LongType> &shape);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType &extra(sd::LongType *buffer);

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType sizeAt(const sd::LongType *shapeInfo, const sd::LongType dim);

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType strideAt(const sd::LongType *shapeInfo, const sd::LongType dim);

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void setShape(sd::LongType *shapeInfo, sd::LongType *shape);

SD_LIB_EXPORT SD_HOST_DEVICE void setStrideConst(sd::LongType *buffer, const sd::LongType *strides);



SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE char setOrder(sd::LongType *buffer, char c);




SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void setExtra(sd::LongType *buffer, sd::LongType extra);
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
* Returns a shape buffer
* for the shape information metadata.
*/
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *toShapeBuffer(ShapeInformation *info);

SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType *toShapeBuffer(ShapeInformation *info, sd::LongType *ret);


/**
* Returns the prod of the data
* up to the given length
*/
SD_LIB_EXPORT SD_HOST_DEVICE sd::LongType prodLong(const sd::LongType *data, int length);


SD_LIB_EXPORT SD_HOST_DEVICE void printShapeInfo(const sd::LongType *shapeInfo);





// max array is outer for min array, min array is sub-array of max array
// function calculates the coordinates of min array (and saves them into minIdxs) given coordinates of max array
// (already stored in maxIdxs) dimsToExclude - should be sorted in increasing order dimsLen - length of dimsToExclude,
// if not set (= -1), then it is calculated as maxRank - minRank
SD_LIB_EXPORT SD_HOST_DEVICE void maxIndToMinInd(sd::LongType *maxIdxs, sd::LongType *minIdxs,
                                                 const sd::LongType *maxShapeInfo, const sd::LongType *minShapeInfo,
                                                 const sd::LongType *dimsToExclude = nullptr,
                                                 sd::LongType dimsLen = -1);

SD_INLINE SD_HOST SD_LIB_EXPORT sd::LongType tensorsAlongDimension(const sd::LongType* shapeInfo, sd::LongType* dimensions, sd::LongType dimensionLength) {
  // Guard against null or empty dimensions
  if (dimensions == nullptr || dimensionLength == 0)
    throw std::invalid_argument("Invalid input: dimensions not specified (null or length 0)");

  const sd::LongType rank = shape::rank(shapeInfo);

  // Single tensor case
  if (dimensionLength >= rank || (dimensionLength == 1 && dimensions[0] == SD_MAX_INT))
    return 1;

  // Handle negative dimensions
  for(int i = 0; i < dimensionLength; i++) {
    if(dimensions[i] < 0)
      dimensions[i] += rank;
  }

  // Calculate product of shape along specified dimensions
  sd::LongType tensorLen = 1;
  for(int i = 0; i < dimensionLength; i++) {
    tensorLen *= shape::sizeAt(shapeInfo, dimensions[i]);
  }

  // Handle empty tensor case
  if(tensorLen == 0)
    return 1;

  // Calculate number of tensors
  sd::LongType length = shape::length(shapeInfo);
  sd::LongType numTensors = length / tensorLen;

  if(numTensors >= SD_MAX_INT)
    throw std::invalid_argument("Tensors along dimension cannot be >= Integer.MAX_VALUE");

  return numTensors;
}


/**
* Keep the given indexes in the data
* @param data
* @param index
* @param indexLength
* @param dataLength
* @return
*/
SD_LIB_EXPORT SD_INLINE SD_HOST sd::LongType *keep(volatile sd::LongType *data, const sd::LongType *index, int indexLength, int dataLength) {
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

// Optimized version with special cases for common ranks
#define INDEX2COORDS(linear_index, rank, shape, coords)                    \
    do {                                                                   \
        sd::LongType idx = (linear_index);                                \
        if ((rank) == 0) {                                                \
            /* Rank 0 is a scalar, set to 0 to avoid undefined behavior */\
            (coords)[0] = 0;                                              \
        }                                                                  \
        else if ((rank) == 1) {                                           \
            /* Rank 1 is simple: the linear index is the coordinate */    \
            (coords)[0] = idx;                                            \
        }                                                                  \
        else if ((rank) == 2) {                                           \
            (coords)[1] = idx % (shape)[1];                               \
            (coords)[0] = idx / (shape)[1];                               \
        }                                                                  \
        else if ((rank) == 3) {                                           \
            (coords)[2] = idx % (shape)[2];                               \
            idx /= (shape)[2];                                            \
            (coords)[1] = idx % (shape)[1];                               \
            (coords)[0] = idx / (shape)[1];                               \
        }                                                                  \
        else if ((rank) == 4) {                                           \
            (coords)[3] = idx % (shape)[3];                               \
            idx /= (shape)[3];                                            \
            (coords)[2] = idx % (shape)[2];                               \
            idx /= (shape)[2];                                            \
            (coords)[1] = idx % (shape)[1];                               \
            (coords)[0] = idx / (shape)[1];                               \
        }                                                                  \
        else if ((rank) == 5) {                                           \
            (coords)[4] = idx % (shape)[4];                               \
            idx /= (shape)[4];                                            \
            (coords)[3] = idx % (shape)[3];                               \
            idx /= (shape)[3];                                            \
            (coords)[2] = idx % (shape)[2];                               \
            idx /= (shape)[2];                                            \
            (coords)[1] = idx % (shape)[1];                               \
            (coords)[0] = idx / (shape)[1];                               \
        }                                                                  \
        else if ((rank) == 6) {                                           \
            (coords)[5] = idx % (shape)[5];                               \
            idx /= (shape)[5];                                            \
            (coords)[4] = idx % (shape)[4];                               \
            idx /= (shape)[4];                                            \
            (coords)[3] = idx % (shape)[3];                               \
            idx /= (shape)[3];                                            \
            (coords)[2] = idx % (shape)[2];                               \
            idx /= (shape)[2];                                            \
            (coords)[1] = idx % (shape)[1];                               \
            (coords)[0] = idx / (shape)[1];                               \
        }                                                                  \
        else if ((rank) == 7) {                                           \
            (coords)[6] = idx % (shape)[6];                               \
            idx /= (shape)[6];                                            \
            (coords)[5] = idx % (shape)[5];                               \
            idx /= (shape)[5];                                            \
            (coords)[4] = idx % (shape)[4];                               \
            idx /= (shape)[4];                                            \
            (coords)[3] = idx % (shape)[3];                               \
            idx /= (shape)[3];                                            \
            (coords)[2] = idx % (shape)[2];                               \
            idx /= (shape)[2];                                            \
            (coords)[1] = idx % (shape)[1];                               \
            (coords)[0] = idx / (shape)[1];                               \
        }                                                                  \
        else if ((rank) == 8) {                                           \
            (coords)[7] = idx % (shape)[7];                               \
            idx /= (shape)[7];                                            \
            (coords)[6] = idx % (shape)[6];                               \
            idx /= (shape)[6];                                            \
            (coords)[5] = idx % (shape)[5];                               \
            idx /= (shape)[5];                                            \
            (coords)[4] = idx % (shape)[4];                               \
            idx /= (shape)[4];                                            \
            (coords)[3] = idx % (shape)[3];                               \
            idx /= (shape)[3];                                            \
            (coords)[2] = idx % (shape)[2];                               \
            idx /= (shape)[2];                                            \
            (coords)[1] = idx % (shape)[1];                               \
            (coords)[0] = idx / (shape)[1];                               \
        }                                                                  \
        else {                                                            \
            for (sd::LongType i10 = (rank) - 1; i10 > 0; --i10) {   /* avoid variable name clashes with normal i */           \
                (coords)[i10] = idx % (shape)[i10];                           \
                idx /= (shape)[i10];                                        \
            }                                                             \
            (coords)[0] = idx;                                            \
        }                                                                 \
    } while (0)

#define COORDS2INDEX(rank, strides, coords, index_var)                    \
    do {                                                                  \
        if ((rank) == 0) {                                               \
            /* Rank 0 is a scalar, index is always 0 */                  \
            (index_var) = 0;                                             \
        }                                                                 \
        else if ((rank) == 1) {                                          \
            /* Rank 1 is simple: just use the stride */                  \
            (index_var) = (coords)[0] * (strides)[0];                    \
        }                                                                 \
        else if ((rank) == 2) {                                          \
            (index_var) = (coords)[0] * (strides)[0] +                   \
                         (coords)[1] * (strides)[1];                      \
        }                                                                 \
        else if ((rank) == 3) {                                          \
            (index_var) = (coords)[0] * (strides)[0] +                   \
                         (coords)[1] * (strides)[1] +                     \
                         (coords)[2] * (strides)[2];                      \
        }                                                                 \
        else if ((rank) == 4) {                                          \
            (index_var) = (coords)[0] * (strides)[0] +                   \
                         (coords)[1] * (strides)[1] +                     \
                         (coords)[2] * (strides)[2] +                     \
                         (coords)[3] * (strides)[3];                      \
        }                                                                 \
        else if ((rank) == 5) {                                          \
            (index_var) = (coords)[0] * (strides)[0] +                   \
                         (coords)[1] * (strides)[1] +                     \
                         (coords)[2] * (strides)[2] +                     \
                         (coords)[3] * (strides)[3] +                     \
                         (coords)[4] * (strides)[4];                      \
        }                                                                 \
        else if ((rank) == 6) {                                          \
            (index_var) = (coords)[0] * (strides)[0] +                   \
                         (coords)[1] * (strides)[1] +                     \
                         (coords)[2] * (strides)[2] +                     \
                         (coords)[3] * (strides)[3] +                     \
                         (coords)[4] * (strides)[4] +                     \
                         (coords)[5] * (strides)[5];                      \
        }                                                                 \
        else if ((rank) == 7) {                                          \
            (index_var) = (coords)[0] * (strides)[0] +                   \
                         (coords)[1] * (strides)[1] +                     \
                         (coords)[2] * (strides)[2] +                     \
                         (coords)[3] * (strides)[3] +                     \
                         (coords)[4] * (strides)[4] +                     \
                         (coords)[5] * (strides)[5] +                     \
                         (coords)[6] * (strides)[6];                      \
        }                                                                 \
        else if ((rank) == 8) {                                          \
            (index_var) = (coords)[0] * (strides)[0] +                   \
                         (coords)[1] * (strides)[1] +                     \
                         (coords)[2] * (strides)[2] +                     \
                         (coords)[3] * (strides)[3] +                     \
                         (coords)[4] * (strides)[4] +                     \
                         (coords)[5] * (strides)[5] +                     \
                         (coords)[6] * (strides)[6] +                     \
                         (coords)[7] * (strides)[7];                      \
        }                                                                 \
        else {                                                           \
            (index_var) = 0;                                             \
            for (sd::LongType i10 = 0; i10 < (rank); ++i10) {                  \
                (index_var) += (coords)[i10] * (strides)[i10];               \
            }                                                            \
        }                                                                \
    } while (0)
/////
template <typename T>
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void fill(T *buffer, T value, sd::LongType length) {
  for (int e = 0; e < length; e++) buffer[e] = value;
}


SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType subArrayIndex(sd::LongType maxIdx, const sd::LongType *maxShapeInfo,
                                                                  const sd::LongType *minShapeInfo) {
  sd::LongType maxIdxs[SD_MAX_RANK];
  INDEX2COORDS(maxIdx, shape::rank(maxShapeInfo), shape::shapeOf(maxShapeInfo), maxIdxs);

  sd::LongType minIdxs[SD_MAX_RANK];
  shape::maxIndToMinInd(maxIdxs, minIdxs, maxShapeInfo, minShapeInfo, nullptr, -1);

  sd::LongType minIdx;
  COORDS2INDEX(shape::rank(minShapeInfo), shape::stride(minShapeInfo), minIdxs, minIdx);

  return minIdx;
}


SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool strideDescendingCAscendingF( sd::LongType *shapeBuffer) {
  sd::LongType rank = shape::rank(shapeBuffer);
  sd::LongType *strides = shape::stride(const_cast<sd::LongType *>(shapeBuffer));
  char order = shape::order(shapeBuffer);

  if (shape::isRowVector(shapeBuffer) && strides[0] == 1 && strides[1] == 1) return true;

  if (order == 'c') {
    for (sd::LongType i = 1; i < rank; i++)
      if (strides[i - 1] <= strides[i]) return false;
    return true;
  } else if (order == 'f') {
    for (sd::LongType i = 1; i < rank; i++)
      if (strides[i - 1] >= strides[i]) return false;
    return true;
  } else {
    return false;
  }
}

SD_LIB_EXPORT SD_INLINE SD_HOST int outerArrayOffsets(sd::LongType *maxOffsets, const sd::LongType minIdx,
                                                      const sd::LongType *maxShapeInfo, const sd::LongType *minShapeInfo,
                                                      sd::LongType *memBuff, const sd::LongType *dimsToExclude) {
  const auto rankMin = shape::rank(minShapeInfo);
  const auto rankMax = shape::rank(maxShapeInfo);

  const auto diff = rankMax - rankMin;  // the size of dimsToExclude is equal to diff

  sd::LongType *indices = memBuff;
  sd::LongType *increment = memBuff + rankMax;

  sd::LongType N, minI, maxI;

  // calculate min per-dim-indices which corresponds to absolute minIdx index
  INDEX2COORDS(minIdx, rankMin, shape::shapeOf(minShapeInfo), indices);

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
  sd::LongType offset;
  sd::LongType coords[SD_MAX_RANK];
  COORDS2INDEX(rankMax, shape::stride(maxShapeInfo), indices, offset);
  maxOffsets[N++] = offset;

  // nested loops - producing of absolute indices for max array
  while (maxI >= 0) {
    if (increment[maxI] != 0) {
      indices[maxI] += increment[maxI];
      if (indices[maxI] >= maxShapeInfo[maxI + 1]) {
        indices[maxI] %= increment[maxI];  // restore initial value of indices[maxI]
        step = -1;
      } else {
        COORDS2INDEX(rankMax, shape::stride(maxShapeInfo), indices, offset);
        maxOffsets[N++] = offset;
        step = rankMax - 1 - maxI;
      }
    } else if (maxI == rankMax - 1)
      step = -1;

    maxI += step;
  }
  return N;
}

// max array is outer for min array, min array is sub-array of max array
// function calculates the coordinates of min array (and saves them into minIdxs) given coordinates of max array
// (already stored in maxIdxs)
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void maxIndToMinInd(sd::LongType *maxIdxs, sd::LongType *minIdxs,
                                                           const sd::LongType *maxShapeInfo, const sd::LongType *minShapeInfo,
                                                           const sd::LongType *dimsToExclude, sd::LongType dimsLen) {
  const auto maxRank = shape::rank(maxShapeInfo);
  const auto minRank = shape::rank(minShapeInfo);

  if (dimsLen == -1) dimsLen = maxRank - minRank;  // if size is not given (= -1) then it is equal to ranks difference

  if (maxRank == minRank) {
    if (dimsToExclude == nullptr) {  // --> means dimsToExclude == {0,1,2,...,dimsLen-1}

      for (sd::LongType i = 0; i < maxRank; ++i) {
        if (i < dimsLen)
          minIdxs[i] = maxIdxs[i];
        else {
          // FIX: Ensure proper modulo for tiling with dimensions of size 1
          sd::LongType dimSize = shape::shapeOf(minShapeInfo)[i];
          // Only use modulo for non-zero dimension sizes to avoid division by zero
          if (dimSize > 0) {
            minIdxs[i] = maxIdxs[i] % dimSize;
          } else {
            // Handle the case where dimension size is 0 (should be rare)
            minIdxs[i] = maxIdxs[i];
          }
        }
      }
    } else {
      for (sd::LongType i = 0, dim = 0; i < maxRank; ++i) {
        if (dim < dimsLen && dimsToExclude[dim] == i) {
          minIdxs[i] = maxIdxs[i];
          ++dim;
          continue;
        }

        // FIX: Ensure proper modulo for tiling with dimensions of size 1
        sd::LongType dimSize = shape::shapeOf(minShapeInfo)[i];
        // Only use modulo for non-zero dimension sizes to avoid division by zero
        if (dimSize > 0) {
          minIdxs[i] = maxIdxs[i] % dimSize;
        } else {
          // Handle the case where dimension size is 0 (should be rare)
          minIdxs[i] = maxIdxs[i];
        }
      }
    }
  } else {
    if (dimsToExclude == nullptr) {  // --> means dimsToExclude == {0,1,2,...,dimsLen-1}

      for (sd::LongType i = 0; i < minRank; ++i) {
        // FIX: Ensure proper modulo for tiling with dimensions of size 1
        sd::LongType dimSize = shape::shapeOf(minShapeInfo)[i + 1];
        sd::LongType maxIdx = maxIdxs[i + dimsLen];

        // Only use modulo for non-zero dimension sizes to avoid division by zero
        if (dimSize > 0) {
          minIdxs[i] = maxIdx % dimSize;
        } else {
          // Handle the case where dimension size is 0 (should be rare)
          minIdxs[i] = maxIdx;
        }
      }
    } else {
      for (sd::LongType minI = 0, maxI = 0, dim = 0; maxI < maxRank; ++maxI) {
        if (dim < dimsLen && dimsToExclude[dim] == maxI) {
          ++dim;
          continue;
        }

        // FIX: Ensure proper modulo for tiling with dimensions of size 1
        sd::LongType dimSize = shape::shapeOf(minShapeInfo)[minI + 1];

        // Only use modulo for non-zero dimension sizes to avoid division by zero
        if (dimSize > 0) {
          minIdxs[minI] = maxIdxs[maxI] % dimSize;
        } else {
          // Handle the case where dimension size is 0 (should be rare)
          minIdxs[minI] = maxIdxs[maxI];
        }
        ++minI;
      }
    }
  }
}



// calculate offsets of max-array, these offsets correspond to one minIdx index of min-array which is sub-array of
// max-array maxOffsets - will contain calculated offsets of max-array, buffer for maxOffsets should be allocated
// beforehand dimsToExclude - should be sorted in increasing order memBuff - auxiliary memory buffer (size = 2 *
// max_rank) for coordinates and increments storing, should be allocated beforehand
SD_LIB_EXPORT SD_HOST_DEVICE int outerArrayOffsets(sd::LongType *maxOffsets, const sd::LongType minIdx,
                                                   const sd::LongType *maxShapeInfo, const sd::LongType *minShapeInfo,
                                                   sd::LongType *memBuff, const sd::LongType *dimsToExclude = nullptr);

// calculates offsets for entities (elements or sub-arrays), shape in context of sub-array means dimensions excluded
// from outer array rank is equal to size of shape
SD_LIB_EXPORT void calcOffsets(const sd::LongType rank, const sd::LongType *shape, const sd::LongType *strides,
                               sd::LongType *offsets, const char order = 'c');
SD_LIB_EXPORT void calcOffsets(const sd::LongType *shapeInfo, sd::LongType *offsets, const char order = 'c');

SD_LIB_EXPORT SD_HOST_DEVICE void shapeOldScalar(sd::DataType dtype, sd::LongType *const buffer, const char order);

// deduce order and element-wise stride
// if array is scalar or unit length vector then ews = 1 and order is preserved
// if array is common vector then ews = stride of non-unity dimension and order is preserved
// if strides are normal/contiguous then ews = 1 and corresponding order is set, otherwise ews = 0 and order is
// preserved
SD_LIB_EXPORT SD_HOST_DEVICE void checkStridesEwsAndOrder(sd::LongType *shapeInfo, const char proposedOrder,
                                                          const sd::LongType numOfNonUnitDims,
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
    const sd::LongType *wholeShapeInfo, const sd::LongType numOfSubArrs, const sd::LongType dimsSize,
    const sd::LongType *dimsToExclude, sd::LongType *subArrShapeInfo, sd::LongType *subArrOffsets,
    bool keepUnitiesInShape = false);

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
                                                const sd::LongType numOfUntiesInMinShape = 0);

/**
* for example inShapeInfo is {3, 2,1,4, 4,4,1, 16384,1,99}
* then output shapeNoUnities will contain {2,4, 4,1} - that is only shape and strides, no rank/type/ews/order
* stridesNoUnities will point on strides in shapeNoUnities that is on {4,1}
* returns number of non-unity dimensions in inShapeInfo
* if there is no unities in inShapeInfo, then no copy procedure will be performed and shapeNoUnities/stridesNoUnities
* will point on corresponding places in inShapeInfo
*/
SD_LIB_EXPORT SD_HOST_DEVICE int excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo,
                                                             sd::LongType *shapeNoUnities,
                                                             sd::LongType *stridesNoUnities);

/**
* for example inShapeInfo is {3, 2,1,3,1,4,  12,12,4,4,1, 16384,1,99}, dimsToExclude(points on unity dimensions) =
* {1,3}, dimsSize = 2 then outShapeInfo will contain {3, 2,3,4, 12,4,1, 16384,1,99}
*/
SD_LIB_EXPORT SD_HOST_DEVICE void excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo,
                                                              const sd::LongType *dimsToExclude,
                                                              const sd::LongType dimsSize, sd::LongType *outShapeInfo);

/**
* get stride over contiguous axis (contiguous axis must have stride = 1)
* for example when inShapeInfo is {4, 2,5,4,3,  60,1,5,20, 16384,0,99} then output is 5 (that is smallest stride in
* inShapeInfo except those equal to 1)
*/






SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool strideEquals(int const shape1Rank, sd::LongType const *shape1, int const shape2Rank,
                                                         sd::LongType const *shape2) {
  if (shape1Rank != shape2Rank) return false;
  // rank not equals
  for (int i = 0; i < shape1Rank; i++) {
    if (shape1[i] != shape2[i]) return false;
  }

  return true;
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool strideEquals(sd::LongType const *shapeInfo1, sd::LongType const *shapeInfo2) {
  return strideEquals(rank(shapeInfo1), stride(shapeInfo1), rank(shapeInfo2), stride(shapeInfo2));
}



/**
* Computes the standard packed array strides for a given shape.
*
* @param shape    the shape of a matrix:
* @param startNum the start number for the strides
* @return the strides for a matrix of n dimensions
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, sd::LongType rank, sd::LongType startNum) {
  sd::LongType dimensions = rank;

  sd::LongType *stride = new sd::LongType[dimensions];
  sd::LongType st = startNum;
  for (sd::LongType j = 0; j < rank; j++) {
    stride[j] = st;
    st *= shape[j];
  }

  return stride;
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank, int startNum, sd::LongType *ret) {
  sd::LongType st = startNum;
  for (sd::LongType j = 0; j < rank; j++) {
    ret[j] = st;
    st *= shape[j];
  }

  return ret;
}








//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo, const sd::LongType *dimsToExclude,
                                                                        const sd::LongType dimsSize, sd::LongType *outShapeInfo) {
  outShapeInfo[0] = inShapeInfo[0] - dimsSize;

  for (sd::LongType j = 0, k = 0, i = 0; i < inShapeInfo[0]; ++i) {
    if (j < dimsSize && i == dimsToExclude[j]) {
      ++j;
      continue;
    }

    shapeOf(outShapeInfo)[k] = shapeOf(inShapeInfo)[i];
    stride(outShapeInfo)[k++] = stride(inShapeInfo)[i];
  }
  outShapeInfo[2 * outShapeInfo[0] + 1] = 0;
  sd::ArrayOptions::copyDataType(outShapeInfo, inShapeInfo);                         // type
  outShapeInfo[2 * outShapeInfo[0] + 3] = order(inShapeInfo);                 // order
}



/**
* Returns a shape buffer
* for the shape information metadata.
*/
SD_LIB_EXPORT SD_INLINE SD_HOST sd::LongType *toShapeBuffer(ShapeInformation *info) {
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





SD_LIB_EXPORT SD_INLINE SD_HOST const char *shapeInfoString(const sd::LongType *shapeInfo) {
  if (shapeInfo == nullptr) return "";

  std::string ret;

  if (shapeInfo != nullptr) {
    if (shapeInfo[0] > 32 || shapeInfo[0] < 0)
      THROW_EXCEPTION("Input shape buffer is corrupt. First rank is < 0 or greater than the max rank of 32.");
  }

  sd::LongType rank = shape::rank(shapeInfo);
  std::stringstream ss;
  if (rank == 0) {
    ss << "Rank " << rank << "\n";
    ss << "Buffer is:";
    for (int i = 0; i < shapeInfoLength(rank); i++) {
      ss << " " << shapeInfo[i] << " ";
    }

    auto flags = sd::ArrayOptions::enumerateSetFlags(shapeInfo);
    ss << flags;
    ss << "\n";
    ret += ss.str();
    return (new std::string(ret))->c_str();
  }

  sd::LongType *shape = shapeOf(shapeInfo);
  ss << "Rank " << rank << "\n";
  ss << "Shape:\n";
  for (int i = 0; i < rank; i++) {
    ss << " " << (sd::LongType)shape[i] << " ";
  }

  ss << "\n";

  sd::LongType *stride = shape::stride(shapeInfo);
  ss << "Stride:\n";
  for (int i = 0; i < rank; i++) {
    ss << " " << (sd::LongType)stride[i] << " ";
  }

  ss << "\n";

  ss << "Order " << order(shapeInfo) << "\n";

  ss << "Buffer is:";
  for (int i = 0; i < shapeInfoLength(rank); i++) {
    ss << " " << (sd::LongType)shapeInfo[i] << " ";
  }

  auto flags = sd::ArrayOptions::enumerateSetFlags(shapeInfo);
  ss << flags;
  ss << "\n";

  ret += ss.str();
  return (new std::string(ret))->c_str();
}

SD_LIB_EXPORT SD_INLINE SD_HOST void printShapeInfo(const sd::LongType *shapeInfo) {
  if (shapeInfo == nullptr) return;
  if (shapeInfo != nullptr) {
    if (shapeInfo[0] > 32 || shapeInfo[0] < 0)
      THROW_EXCEPTION("Input shape buffer is corrupt. First rank is < 0 or greater than the max rank of 32.");
  }

  sd::LongType rank = shape::rank(shapeInfo);
  if (rank == 0) {
    printf("Rank %lld\n", rank);
    printf("Buffer is:");
    for (int i = 0; i < shapeInfoLength(rank); i++) {
      printf(" %lld ", shapeInfo[i]);
    }

    auto flags = sd::ArrayOptions::enumerateSetFlags(shapeInfo);
    printf("%s", flags);
    printf("\n");
    return;
  }
  sd::LongType *shape = shapeOf(shapeInfo);
  printf("Rank %lld\n", rank);
  printf("Shape:\n");
  for (int i = 0; i < rank; i++) {
    printf(" %lld ", (sd::LongType)shape[i]);
  }

  printf("\n");

  sd::LongType *stride = shape::stride(shapeInfo);
  printf("Stride:\n");
  for (int i = 0; i < rank; i++) {
    printf(" %lld ", (sd::LongType)stride[i]);
  }

  printf("\n");

  printf("Order %c\n", order(shapeInfo));

  printf("Buffer is:");
  for (int i = 0; i < shapeInfoLength(rank); i++) {
    printf(" %lld ", (sd::LongType)shapeInfo[i]);
  }

  auto flags = sd::ArrayOptions::enumerateSetFlags(shapeInfo);
  printf("%s", flags);
  printf("\n");
}

SD_LIB_EXPORT SD_INLINE SD_HOST void printShapeInfoLinear(const sd::LongType *shapeInfo) {
  sd::LongType rank = shape::rank(shapeInfo);
  sd::LongType lim = shapeInfoLength(rank);
  printf("ShapeInfo: [");
  for (sd::LongType i = 0; i < lim; i++) {
    printf("%lld", shapeInfo[i]);

    if (i < lim - 1) {
      printf(", ");
    }
  }
  printf("]\n");
#ifndef __CUDA_ARCH__
  fflush(stdout);
#endif
}


SD_LIB_EXPORT SD_INLINE SD_HOST void printShapeInfoLinear(const char *msg, const sd::LongType *shapeInfo) {
  int rank = shape::rank(shapeInfo);
  int lim = shapeInfoLength(rank);
  printf("%s : [", msg);
  for (int i = 0; i < lim; i++) {
    printf("%lld", shapeInfo[i]);

    if (i < lim - 1) {
      printf(", ");
    }
  }
  printf("]\n");
#ifndef __CUDACC__
  fflush(stdout);
#endif
}



SD_LIB_EXPORT SD_INLINE SD_HOST void transposeInplace(sd::LongType *shapeBuffer) {
  int rank = shape::rank(shapeBuffer);
  sd::LongType *shape = shapeOf(shapeBuffer);
  sd::LongType *strides = stride(shapeBuffer);

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

  if (order(shapeBuffer) == 'c')
    shapeBuffer[shapeInfoLength(shapeBuffer) - 1] = 102;
  else
    shapeBuffer[shapeInfoLength(shapeBuffer) - 1] = 99;
}


SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *shapeBufferOfNpy(sd::LongType rank, sd::LongType *shape, bool fortranOrder) {
  if (fortranOrder) {
    sd::LongType *shapeBufferRet = shapeBufferFortran(rank, sd::FLOAT32, (sd::LongType *)shape);
    return shapeBufferRet;
  } else {
    sd::LongType *newShape = new sd::LongType[rank];
    for (int i = 0; i < rank; i++) {
      newShape[i] = shape[i];
    }

    sd::LongType *shapeBufferRet = shapeBuffer(rank, sd::FLOAT32, newShape);
    delete[] newShape;
    return shapeBufferRet;
  }
}


SD_INLINE SD_HOST sd::LongType *shapeBufferOfNpy(cnpy::NpyArray arr) {
  return shapeBufferOfNpy(arr.shape.size(), (sd::LongType *)arr.shape.data(), arr.fortranOrder);
}



/**
* Computes the standard packed array strides for a given shape.
*
* @param shape    the shape of a matrix:
* @param startNum the start number for the strides
* @return the strides for a matrix of n dimensions
*/
SD_LIB_EXPORT SD_HOST_DEVICE  SD_INLINE sd::LongType *calcStrides(sd::LongType const *shape, sd::LongType rank, sd::LongType startNum) {
  sd::LongType *stride = new sd::LongType[rank];

  if (rank == 1) {
    stride[0] = 1;
    return stride;
  }

  sd::LongType st = startNum;
  for (sd::LongType j = rank - 1; j >= 0; j--) {
    stride[j] = st;
    st *= shape[j];
  }

  return stride;
}


SD_INLINE SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, sd::LongType rank, sd::LongType startNum,
                                                   sd::LongType *ret) {
  if (rank == 1) {
    ret[0] = 1;
    return ret;
  }

  sd::LongType st = startNum;
  for (sd::LongType j = rank - 1; j >= 0; j--) {
    ret[j] = st;
    st *= shape[j];
  }

  return ret;
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, int rank, sd::LongType *ret) {
  return calcStrides(shape, rank, 1, ret);
}


// function calculates absolute offset of min array, min is sub-array of max, offset to be returned corresponds to
// maxIdx of max array dimsToExclude - should be sorted in increasing order
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType subArrayOffset(const sd::LongType maxIdx, const sd::LongType *maxShapeInfo,
                                                                   const sd::LongType *minShapeInfo,
                                                                   const sd::LongType *dimsToExclude = nullptr,
                                                                   const sd::LongType dimsLen = -1);


SD_LIB_EXPORT SD_INLINE SD_HOST sd::LongType subArrayOffset(const sd::LongType maxIdx, const sd::LongType *maxShapeInfo,
                                                            const sd::LongType *minShapeInfo, const sd::LongType *dimsToExclude,
                                                            const sd::LongType dimsLen) {
  sd::LongType maxIdxs[SD_MAX_RANK];
  INDEX2COORDS(maxIdx, shape::rank(maxShapeInfo), shapeOf(maxShapeInfo), maxIdxs);

  sd::LongType minIdxs[SD_MAX_RANK];
  maxIndToMinInd(maxIdxs, minIdxs, maxShapeInfo, minShapeInfo, dimsToExclude, dimsLen);
  sd::LongType  ret;
  COORDS2INDEX(shape::rank(minShapeInfo), shape::stride(minShapeInfo), minIdxs, ret);
  return ret;
}

SD_LIB_EXPORT SD_INLINE  SD_HOST const char *shapeToString(const sd::LongType *shapeInfo, const char *message) {
  if (shapeInfo == nullptr) {
    auto ret = new std::string("Shape info is empty");
    return ret->c_str();
  }


  std::string shapeInfoString;
  shapeInfoString += message;
  shapeInfoString += "  ";
  sd::LongType rank = shape::rank(shapeInfo);
  if (rank == 0) {
    shapeInfoString += "Rank: ";
    shapeInfoString += std::to_string(rank);
    auto ret = new std::string(shapeInfoString.c_str());
    return ret->c_str();
  }

  shapeInfoString += " Rank ";
  shapeInfoString += std::to_string(rank);

  sd::LongType *shape = shapeOf(shapeInfo);
  shapeInfoString += " Shape: ";
  for (int i = 0; i < rank; i++) {
    shapeInfoString += std::to_string(shape[i]);
    shapeInfoString += " ";
  }

  shapeInfoString += " ";
  sd::LongType *stride = shape::stride(shapeInfo);
  shapeInfoString += (" Stride: ");
  for (int i = 0; i < rank; i++) {
    shapeInfoString += std::to_string(stride[i]);
    shapeInfoString += " ";
  }

  shapeInfoString += (" ");
  shapeInfoString += ("Order: ");
  shapeInfoString += order(shapeInfo);
  shapeInfoString += " ";
  shapeInfoString += " Flags extra value: ";
  shapeInfoString += std::to_string(extra(const_cast<sd::LongType *>(shapeInfo)));
  shapeInfoString += " ";

  shapeInfoString += ("Buffer is:");
  for (int i = 0; i < shapeInfoLength(rank); i++) {
    shapeInfoString += std::to_string(shapeInfo[i]);
    shapeInfoString += " ";
  }
  shapeInfoString += (" ");
  auto ret = new std::string(shapeInfoString.c_str());
  return ret->c_str();
}


/**
* Computes the standard packed array strides for a given shape.
*
* @param shape    the shape of a matrix:
* @param startNum the start number for the strides
* @return the strides for a matrix of n dimensions
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, sd::LongType rank) {
  return calcStridesFortran(shape, rank, 1);
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank, sd::LongType *ret) {
  return calcStridesFortran(shape, rank, 1, ret);
}



/**
* Computes the standard packed array strides for a given shape.
*
* @param shape    the shape of a matrix:
* @param startNum the start number for the strides
* @return the strides for a matrix of n dimensions
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *calcStrides(sd::LongType const *shape, sd::LongType rank) {
  return calcStrides(shape, rank, 1);
}





//////////////////////////////////////////////////////////////////////

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool shapeEquals(const int shape1Rank, const sd::LongType *shape1, const int shape2Rank,
                                                        const sd::LongType *shape2) {
  if (shape1Rank != shape2Rank) return false;
  // rank not equals
  for (int i = 0; i < shape1Rank; i++) {
    if (shape1[i] != shape2[i]) return false;
  }

  return true;
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool shapeEquals(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2) {
  return shapeEquals(rank(shapeInfo1), shapeOf(const_cast<sd::LongType *>(shapeInfo1)), rank(shapeInfo2),
                     shapeOf(const_cast<sd::LongType *>(shapeInfo2)));
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool shapeEquals(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2,
                                                        const sd::LongType *shapeInfo3) {
  return shapeEquals(shapeInfo1, shapeInfo2) && shapeEquals(shapeInfo1, shapeInfo3);
}


#if defined(__CUDACC__)
/**
* BEWARE: THIS METHOD DOES NOT CHECKS ALLOCATION BOUNDARIES
*/
SD_DEVICE SD_INLINE sd::LongType *cuMalloc(sd::LongType *buffer, long size) {
 sd::LongType *ret = buffer;
 ret += (threadIdx.x * size);
 return ret;
}
#endif

//////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////


/**
* Get the ordering for the device
* @param length
* @param shape
* @param stride
* @param elementStride
* @return
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE char getOrder(int length, sd::LongType *shape, sd::LongType *stride, int elementStride) {
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
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE int checkArrangeArray(T *arr, int arrLength, int shapeLength) {
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
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE int isVector(sd::LongType const *shape, int rank) {
  if (rank == 0) return 0;

  if (rank == 1) return 1;

  if (rank > 2) return 0;
  if (rank <= 2) {
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

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool isCommonVector(const sd::LongType *shapeInfo, sd::LongType &posOfNonUnityDim) {
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





SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *detachShape(sd::LongType *originalShape) {
  sd::LongType *newShape = new sd::LongType[shapeInfoLength(originalShape)];
  memcpy(newShape, originalShape, shapeInfoByteLength(originalShape));

  return newShape;
}


SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE int isVector(sd::LongType *shapeInfo) {
  return isVector(shapeOf(const_cast<sd::LongType *>(shapeInfo)), rank(shapeInfo));
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool isRowVector(sd::LongType *shapeInfo) {
  bool isVector = shape::isVector(shapeInfo) == 1;
  bool shapeFirstOne = shapeOf(const_cast<sd::LongType *>(shapeInfo))[0] == 1;
  return isVector && shapeFirstOne;
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool isColumnVector( sd::LongType *shapeInfo) {
  bool isVector = shape::isVector(shapeInfo) == 1;
  bool shapeFirstOne = shapeOf(shapeInfo)[0] == 1;
  return isVector && !shapeFirstOne;
}

//////////////////////////////////////////////////////////////////////
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE int numOfNonUnitDims(const int rank, const sd::LongType *inShape) {
  int num = 0;

  for (sd::LongType i = 0; i < rank; ++i)
    if (inShape[i] != 1) ++num;

  return num;
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE int oneDimEqualToLength(sd::LongType *shape, int rank) {
  for (int i = 0; i < rank; i++) {
    if (shape[i] == prodLong(shape, rank)) return 1;
  }

  return 0;
}



/**
* Returns whether the
* given shape is a vector or not
* @param shape the shape of the array
* @param rank the rank of the shape
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE int isMatrix(const sd::LongType *shape, int rank) {
  if (rank > 2) return 0;
  if (rank <= 2) {
    if (shape[0] == 1 || shape[1] == 1) return 0;
  }

  return 1;
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE int isMatrix(const sd::LongType *shapeInfo) {
  return isMatrix(shapeOf(shapeInfo), rank(shapeInfo));
}

/**
* Returns the shape portion of an information
* buffer
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *shapeOf(sd::LongType *shapeInfo) { return shapeInfo + 1; }

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void setShape(sd::LongType *shapeInfo, sd::LongType *shape) {
  sd::LongType rank = shape::rank(shapeInfo);
  for (int i = 0; i < rank; i++) {
    shapeInfo[i + 1] = shape[i];
    if(shape[i] < 0) {
      std::string errorMessage;
      errorMessage += "Shape value is invalid: ";
      errorMessage += std::to_string(shape[i]);
      errorMessage += " at index ";
      errorMessage += std::to_string(i);
      errorMessage += " in shape ";
      errorMessage += std::to_string(rank);
      errorMessage += "\n";
      THROW_EXCEPTION(errorMessage.c_str());
    }


  }


}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *shapeOf(const sd::LongType *shapeInfo) {
  return shapeOf(const_cast<sd::LongType *>(shapeInfo));
}

/**
* Return a copy of a buffer.
* This buffer allocates memory
* that must be freed elsewhere.
*/
template <typename T>
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE T *copyOf(sd::LongType length, T const *toCopy) {
  T *ret = new T[length];
  return copyOf(length, toCopy, ret);
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
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType shapeInfoLength(sd::LongType rank) {
  if(rank > SD_MAX_RANK) {
#ifndef __CUDACC__
    std::string errorMessage;
    errorMessage += "Rank is too high: ";
    errorMessage += std::to_string(rank);
    errorMessage += "\n";
    THROW_EXCEPTION(errorMessage.c_str());
#else
    printf("Rank is too high: %lld\n", rank);
    return -1;
#endif
  }
  // rank takes up 1 element + usual elements
  if (rank < 1)
    // shape of 0 (scalar) even has elements for shape and stride
    return 1 * 2 + 4;
  // FIXME magic numbers
  return rank * 2 + 4;
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType shapeInfoLength(sd::LongType *shape) {
  return shapeInfoLength(shape[0]);
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType shapeInfoLength(const sd::LongType *shape) {
  return shapeInfoLength(static_cast<sd::LongType>(shape[0]));
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType shapeInfoByteLength(sd::LongType rank) {
  // scalar formula isn't correct
  if (rank == 0) return 6 * sizeof(sd::LongType);
  // FIXME magic numbers
  return (rank * 2 + 4) * sizeof(sd::LongType);
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE size_t shapeInfoByteLength(const sd::LongType *shapeInfo) {
  // FIXME magic numbers
  return shapeInfoByteLength(shapeInfo[0]);
}

/**
* Returns the rank portion of
* an information buffer
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType rank(const sd::LongType *buffer) {
  return buffer[0];
}




SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void setStride(sd::LongType *buffer, sd::LongType *strides) {
  auto stridesRet = buffer + (1 + rank(buffer));
  int rank = shape::rank(buffer);
  if (rank < 1) {
    buffer[2] = 0;
    return;
  }
  for (int i = 0; i < rank; i++) {
    stridesRet[i] = strides[i];
  }
}


/**
* Returns the stride portion of an information
* buffer
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *stride(sd::LongType *buffer) { return buffer + (1 + rank(buffer)); }

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *stride(const sd::LongType *buffer) {
  return stride(const_cast<sd::LongType *>(buffer));
}


SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType length(std::initializer_list<int> &shape) {
  sd::LongType ret = 1;
  for (auto v : shape) {
    ret *= v;
  }
  return ret;
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType length(std::initializer_list<sd::LongType> &shape) {
  sd::LongType ret = 1;
  for (auto v : shape) {
    ret *= v;
  }
  return ret;
}



SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void setExtra(sd::LongType *buffer, sd::LongType extra) {
  buffer[sd::ArrayOptions::extraIndex(buffer)] = extra;
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType &extra(sd::LongType *buffer) {
  if(buffer == nullptr) {
    THROW_EXCEPTION("extra:  shapebuffer is nullptr");
  }
  sd::LongType rank = buffer[0];
  sd::LongType idx = 0;
  // rank takes up 1 element + usual elements
  if (rank == 0) {
    idx = 3;
  } else {
    // FIXME magic numbers
    idx = rank + rank + 1;
  }
  return buffer[idx];
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType extra(const sd::LongType *buffer) {
  if(buffer == nullptr) {
    THROW_EXCEPTION("extra:  shapebuffer is nullptr");
  }
  sd::LongType rank = buffer[0];
  sd::LongType idx = 0;
  // rank takes up 1 element + usual elements
  if (rank == 0)
    idx = 3;
  else
    // FIXME magic numbers
    idx = rank + rank + 1;
  return buffer[idx];
}




/**
* Compute the length of the given shape
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType length(const sd::LongType *shapeInfo) {
  const sd::LongType rank = shape::rank(shapeInfo);

  if (rank == 0) {
    if (isEmptyConst(shapeInfo)) return 0L;
    return 1L;
  }

  if (rank == 1) return shapeInfo[1];

  return prodLong(shapeOf(const_cast<sd::LongType *>(shapeInfo)), rank);
}

/**
* Returns the ordering
* for this shape information buffer
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE char order(const sd::LongType *buffer) {
  // order doesn't matter for scalars
  if (rank(buffer) < 1) return 'c';
  sd::LongType len = shapeInfoLength(buffer[0]);
  auto longValidation = buffer[len  - 1];
  if(longValidation != 99 && longValidation != 102) {
    std::string errorMessage;
    errorMessage += "Invalid order from shape descriptor: ";
    errorMessage += std::to_string(longValidation);
    errorMessage += "  Order should either be 99 (c) or 102 (f)";
    THROW_EXCEPTION(errorMessage.c_str());
  }
  char ret = static_cast<char>(buffer[len - 1]);
  if (ret != 'c' && ret != 'f') {
    std::string errorMessage;
    errorMessage += "Invalid order from shape descriptor: ";
    errorMessage += std::to_string(ret);
    errorMessage += " for buffer ";
    errorMessage += shapeToString(buffer, "Buffer was:");
    THROW_EXCEPTION(errorMessage.c_str());
  }

  return ret;
}

/**
* Returns the ordering
* for this shape information buffer
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE char setOrder(sd::LongType *buffer, char c) {
  if(shape::rank(buffer) < 1) {
    buffer[5] = 'c';
    return 'c';
  }
  // FIXME magic numbers
  if (length(buffer) > 1 && c != 'c' && c != 'f') {
    std::string errorMessage;
    errorMessage += "Invalid order from  descriptor: ";
    errorMessage += std::to_string(c);
    THROW_EXCEPTION(errorMessage.c_str());
  }


  sd::LongType len = shapeInfoLength(buffer[0]);
  buffer[len - 1] = static_cast<sd::LongType>(c);
  return c;
}

/**
* Returns type
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType type(const sd::LongType *shapeInfo) {
  if (shapeInfo[0] < 1) return shapeInfo[2 * 1 + 1];
  return shapeInfo[2 * shapeInfo[0] + 1];
}

/**
* Returns the element wise stride for this information
* buffer
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType elementWiseStride(const sd::LongType *buffer) {
  return buffer[shapeInfoLength(buffer[0]) - 2];
}


/**
* Returns whether
* the given shape info buffer
* represents a scalar shape
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE int isScalar(const sd::LongType *info) {
  if (isEmptyConst(info)) return 0;
  const sd::LongType rank = shape::rank(info);
  if (rank == 0) return 1;
  return 0;
}

/**
* Returns whether
* the given shape information
* represents a scalar
* shape or not
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE int isScalar(volatile ShapeInformation *info) {
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
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE T1 *removeIndex(T1 const *data, T2 const *indexes, sd::LongType dataLength,
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
#if defined(__CUDACC__)
SD_LIB_EXPORT SD_INLINE SD_DEVICE int tadOffset(ShapeInformation *xInfo, int offset) {
 return offset + threadIdx.x * xInfo->elementWiseStride;
}
#else
SD_LIB_EXPORT SD_INLINE SD_HOST int tadOffset(ShapeInformation *xInfo, int offset) {
  return 0;
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
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *ensureVectorShape(sd::LongType *shape, int dimension) {
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
* This method does STRICT comparison for two shape buffers
*
* @param shape
* @return
*/
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool equalsStrict(const sd::LongType *shapeA, const sd::LongType *shapeB) {
  if (shapeA[0] != shapeB[0]) return false;

  if (shapeA[0] == 0) return true;

  // we do full comparison here
  int length = shapeInfoLength(shapeA[0]);

  for (int e = 1; e < length; e++)
    if (shapeA[e] != shapeB[e]) return false;

  return true;
}

//////////////////////////////////////////////////////////////////////
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool haveSameShapeAndStrides(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2) {
  if (shapeInfo1[0] != shapeInfo2[0]) return false;

  if (shapeInfo1[0] == 0) return true;

  for (sd::LongType e = 0; e < rank(shapeInfo1); ++e)
    if (shapeOf(shapeInfo1)[e] != shapeOf(shapeInfo2)[e] || stride(shapeInfo1)[e] != stride(shapeInfo2)[e])
      return false;

  return true;
}

//////////////////////////////////////////////////////////////////////
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool haveSameShapeAndStrides(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2,
                                                                    const sd::LongType *shapeInfo3) {
  return haveSameShapeAndStrides(shapeInfo1, shapeInfo2) && haveSameShapeAndStrides(shapeInfo1, shapeInfo3);
}

#ifndef __JAVACPP_HACK__

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType sizeAt(const sd::LongType *shapeInfo, const sd::LongType dim) {
  sd::LongType inputDim = dim;
  if(inputDim < 0)
    inputDim += rank(shapeInfo);
  if (0 == shape::rank(shapeInfo)) return 1;
  if (inputDim >= 0) return shapeInfo[1 + inputDim];
  return shapeInfo[1 + (rank(shapeInfo) + inputDim)];
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType strideAt(const sd::LongType *shapeInfo, const sd::LongType dim) {
  sd::LongType inputDim = dim;
  if(inputDim < 0)
    inputDim += rank(shapeInfo);
  if (0 == rank(shapeInfo)) return 1;
  if (dim >= 0) return shapeInfo[1 + rank(shapeInfo) + inputDim];
  return shapeInfo[1 + 2 * rank(shapeInfo) + inputDim];
}
#endif


SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool equalsTypesAndShapesSoft(const sd::LongType *shapeA, const sd::LongType *shapeB) {
  return equalsSoft(shapeA, shapeB) && shapeA[shapeInfoLength(shapeA) - 3] == shapeB[shapeInfoLength(shapeB) - 3];
}

/**
* Generate an int buffer
* up to the given length
* at the specified increment
*
*/
template <typename T>
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE T *range(int from, int to, int increment) {
  int diff = sd::math::sd_abs<int,int>(from - to);
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
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE T *range(int from, int to) {
  return range<T>(from, to, 1);
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
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE T *concat(T const *arr1, sd::LongType const arr1Length, T const *arr2,
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
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE T *concat(sd::LongType const numArrays, sd::LongType const numTotalElements, T const **arr,
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



#if defined(__CUDACC__)
/**
* Computes the offset for accessing
* a global element given the shape information
* and the offset to be read.
*/
SD_LIB_EXPORT SD_INLINE SD_DEVICE int tadOffset(sd::LongType *xInfo, int offset) {
 return offset + threadIdx.x * elementWiseStride(xInfo);
}
#else
SD_LIB_EXPORT SD_INLINE SD_HOST int tadOffset(sd::LongType *xInfo, int offset) {
  return 0;
}
#endif




SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *createScalarShapeInfo() {
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
  sd::LongType *ret = toShapeBuffer(shapeInformation2);
  delete shapeInformation2;
  delete[] shape;
  delete[] stride;
  return ret;
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType *createScalarShapeInfo(sd::LongType *ret) {
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
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE sd::LongType prodLong(const sd::LongType *data, int length) {
  sd::LongType prod = 1;
  for (int i = 0; i < length; i++) {
    prod *= data[i];
  }

  return prod;
}

#if defined(__CUDACC__)
SD_DEVICE SD_LIB_EXPORT SD_INLINE void sweepShapeInfoBuffer(sd::LongType *shapeInfoBuffer, sd::LongType *targetBuffer) {
 // we read first element, to find out length of our shapeInfoBuffer
 int rank = shapeInfoBuffer[0];
 int len = shape::shapeInfoLength(rank);
 for (int i = threadIdx.x; i < len; i += blockDim.x) targetBuffer[i] = shapeInfoBuffer[i];
}
#else
SD_HOST SD_LIB_EXPORT SD_INLINE void sweepShapeInfoBuffer(sd::LongType *shapeInfoBuffer, sd::LongType *targetBuffer) {

}
#endif



// this function checks the consistence of dimensions with array rank (negative dimensions, too large dimensions, too
// big number of dimensions) also it sorts input array of dimensions, this operation is also necessary for creating TAD
// object
SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void checkDimensions(const sd::LongType rank, std::vector<sd::LongType> *dimensions) {
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
    THROW_EXCEPTION("shape::checkDimensions method: number of input dimensions is too big ( > rank of array)!");
  // check if min dimension is still negative and whether max dimension is bigger then rank-1
  if (dimensions->at(0) < 0 || dimensions->back() > (rank - 1))
    THROW_EXCEPTION(
        "shape::checkDimensions method: the negative dimension is still present in input array after transform or the "
        "too big dimension is present ( > rank of array) !");
}

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE void shapeOldScalar(sd::DataType dataType, sd::LongType *const buffer, const char order) {
  buffer[0] = 2;
  buffer[1] = 1;
  buffer[2] = 1;
  buffer[3] = 1;
  buffer[4] = 1;
  buffer[6] = 1;
  buffer[7] = order;

  sd::ArrayOptions::setDataType(buffer, dataType);
}








/**
*
* @param length
* @param shape
* @param rearrange
* @return
*/
SD_LIB_EXPORT SD_INLINE  SD_HOST void doPermuteSwap(sd::LongType length, sd::LongType *shape, sd::LongType *rearrange) {
  if (length == 1) {
    return;
  } else {
    if (prodLong(shape, length) < 2) {
      return;
    }
  }

  bool inOrder = true;
  for (sd::LongType i = 0; i < length - 1; i++) {
    inOrder = inOrder && rearrange[i] + 1 == rearrange[i + 1];
  }

  // all in order, nothing to do
  if (inOrder) return;

  // we know they are just reversed, dimension length of 2
  if (length == 2) {
    auto shapeFirst = shape[0];
    auto shapeSecond = shape[1];
    shape[0] = shapeSecond;
    shape[1] = shapeFirst;
    return;
  } else if (length == 1) {
    // no permute
    return;
  }

  auto temp = new sd::LongType[length];
  memcpy(temp, shape, sizeof(sd::LongType) * length);
  for (sd::LongType i = 0; i < length; i++) {
    shape[i] = temp[rearrange[i]];
  }

  delete[] temp;
}



/**
* Permute the shape information
* @param info the shape information to permute
* @param rearrange the order to re arrange
* @param rank the rank of the rearrange array
*/
SD_LIB_EXPORT SD_INLINE SD_HOST void permute(ShapeInformation **info, sd::LongType *rearrange, sd::LongType rank) {
  ShapeInformation *infoDeref = *info;
  checkArrangeArray(rearrange, rank, rank);
  doPermuteSwap(rank, infoDeref->shape, rearrange);
  doPermuteSwap(rank, infoDeref->stride, rearrange);
  char order = getOrder(rank, infoDeref->shape, infoDeref->stride, infoDeref->elementWiseStride);
  infoDeref->order = order;
}



/**
* Returns whether the given shape
* info has the flag view set.
*/

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool isViewConst(const sd::LongType *shapeInfo) {
  return ((shape::extra(shapeInfo) & ARRAY_IS_VIEW) == ARRAY_IS_VIEW);
}

/**
* Returns whether the
* given shape info has an empty flag set.
*/

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool isEmptyConst(const sd::LongType *shapeInfo) {
  return ((shape::extra(shapeInfo) & ARRAY_EMPTY) == ARRAY_EMPTY);
}

/**
* Returns whether the given shape
* info has the flag view set.
*/

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool isView(sd::LongType *shapeInfo) {
  return shape::isViewConst(const_cast<const sd::LongType *>(shapeInfo));
}

/**
* Returns whether the
* given shape info has an empty flag set.
*/

SD_LIB_EXPORT SD_INLINE SD_HOST_DEVICE bool isEmpty(sd::LongType *shapeInfo) {
  return shape::isEmptyConst(const_cast<const sd::LongType *>(shapeInfo));
}

/**
* This method does SOFT comparison for two shape buffers, we compare only rank & shapes
*
* @param shape
* @return
*/
SD_LIB_EXPORT SD_INLINE  SD_HOST_DEVICE bool equalsSoft(const sd::LongType *shapeA, const sd::LongType *shapeB) {
  if (shapeA[0] != shapeB[0]) {
    return false;
  }

  if (isEmptyConst(shapeA) && isEmptyConst(shapeB)) {
    return true;
  }

  if (shapeA[0] == 0) return true;

  // we compare only shapes, and ignoring stride & ews
  auto length = shapeA[0];

  for (int e = 1; e <= length; e++)
    if (shapeA[e] != shapeB[e]) return false;

  return true;
}




SD_LIB_EXPORT  SD_INLINE SD_HOST_DEVICE void setStrideConst(sd::LongType *buffer, const sd::LongType *strides) {
  auto stridesRet = buffer + (1 + rank(buffer));
  int rank = shape::rank(buffer);
  if (rank < 1) {
    buffer[2] = 0;
    return;
  }
  for (int i = 0; i < rank; i++) {
    stridesRet[i] = strides[i];
  }
}



/**
* Get the shape info buffer
* for the given rank and shape.
*/
SD_LIB_EXPORT SD_INLINE SD_HOST sd::LongType *shapeBuffer(sd::LongType rank, sd::DataType dtype, sd::LongType  *shape) {
  sd::LongType *stride = calcStrides(shape, rank);

  auto shapeInfo = new ShapeInformation();
  shapeInfo->shape = const_cast<sd::LongType *>(shape);
  shapeInfo->stride = stride;
  shapeInfo->offset = 0;
  shapeInfo->rank = rank;
  sd::LongType elementWiseStride = -1;
  shapeInfo->order = 'c';
  auto shapeInfoBuffer = toShapeBuffer(shapeInfo);
  delete[] stride;
  delete shapeInfo;
  sd::ArrayOptions::setDataType(shapeInfoBuffer, dtype);
  return shapeInfoBuffer;
}




SD_LIB_EXPORT SD_HOST SD_INLINE sd::LongType *shapeBuffer(sd::LongType rank, sd::DataType dtype, sd::LongType  *shape,
                                                          sd::LongType *output) {
  sd::LongType stride[SD_MAX_RANK];
  calcStrides(shape, rank, stride);

  ShapeInformation shapeInfo;
  shapeInfo.shape = const_cast<sd::LongType *>(shape);
  shapeInfo.stride = stride;
  shapeInfo.offset = 0;
  shapeInfo.rank = rank;
  auto elementWiseStride = -1;

  shapeInfo.order = 'c';
  shapeInfo.elementWiseStride = elementWiseStride;
  toShapeBuffer(&shapeInfo, output);
  sd::ArrayOptions::setDataType(output, dtype);
  return output;
}

/**
* Get the shape info buffer
* for the given rank and shape.
*/
SD_LIB_EXPORT SD_INLINE SD_HOST sd::LongType *shapeBufferFortran(int rank, sd::DataType dtype, sd::LongType const *shape) {
  auto stride = calcStridesFortran(shape, rank);

  auto shapeInfo = new ShapeInformation();
  shapeInfo->shape = const_cast<sd::LongType *>(shape);
  shapeInfo->stride = stride;
  shapeInfo->offset = 0;
  shapeInfo->rank = rank;
  sd::LongType elementWiseStride = -1;

  shapeInfo->order = 'f';
  shapeInfo->elementWiseStride = elementWiseStride;
  auto shapeInfoBuffer = toShapeBuffer(shapeInfo);
  delete[] stride;
  delete shapeInfo;
  sd::ArrayOptions::setDataType(shapeInfoBuffer, dtype);
  return shapeInfoBuffer;
}

SD_LIB_EXPORT SD_HOST SD_INLINE sd::LongType *shapeBufferFortran(int rank, sd::DataType dtype, sd::LongType const *shape,
                                                                 sd::LongType *output) {
  sd::LongType stride[SD_MAX_RANK];
  calcStridesFortran(shape, rank, stride);

  ShapeInformation shapeInfo;
  shapeInfo.shape = const_cast<sd::LongType *>(shape);
  shapeInfo.stride = stride;
  shapeInfo.offset = 0;
  shapeInfo.rank = rank;
  auto elementWiseStride = -1;

  shapeInfo.order = 'f';
  shapeInfo.elementWiseStride = elementWiseStride;
  toShapeBuffer(&shapeInfo, output);
  sd::ArrayOptions::setDataType(output, dtype);
  return output;
}



SD_LIB_EXPORT SD_INLINE SD_HOST sd::LongType *toShapeBuffer(ShapeInformation *info, sd::LongType *ret) {
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

SD_LIB_EXPORT SD_HOST SD_INLINE void calcSubArrsShapeInfoAndOffsets(const sd::LongType *wholeShapeInfo, const sd::LongType numOfSubArrs,
                                                                    const sd::LongType dimsSize, const sd::LongType *dimsToExclude,
                                                                    sd::LongType *subArrShapeInfo, sd::LongType *subArrOffsets,
                                                                    bool keepUnitiesInShape) {
  const sd::LongType rank = shape::rank(wholeShapeInfo);

  if (dimsSize == rank || dimsSize == 0) {  // means there is one sub-array and it coincides with whole array, return
    // copy of wholeShapeInfo and one zero offset in this case
    memcpy(subArrShapeInfo, wholeShapeInfo, shapeInfoLength(rank) * sizeof(sd::LongType));
    *subArrOffsets = 0;
    return;
  }

  const sd::LongType subArrRank = keepUnitiesInShape ? rank : rank - dimsSize;

  subArrShapeInfo[0] = subArrRank;                                     // rank
  subArrShapeInfo[2 * subArrRank + 1] = 0;                             // clear (to avoid uninitialized)
  sd::ArrayOptions::copyDataType(subArrShapeInfo, wholeShapeInfo);     // type
  subArrShapeInfo[2 * subArrRank + 3] = order(wholeShapeInfo);  // order

  sd::LongType *shape = new sd::LongType[dimsSize];
  sd::LongType *strides = new sd::LongType[dimsSize];

  for (sd::LongType k = subArrRank - 1, j = dimsSize - 1, i = rank - 1; i >= 0; --i) {
    if (j >= 0 && i == dimsToExclude[j]) {
      strides[j] = stride(wholeShapeInfo)[i];
      shape[j--] = shapeOf(wholeShapeInfo)[i];

      if (keepUnitiesInShape) {
        shapeOf(subArrShapeInfo)[k] = 1;
        stride(subArrShapeInfo)[k--] = stride(wholeShapeInfo)[i];
      }
    } else {
      shapeOf(subArrShapeInfo)[k] = shapeOf(wholeShapeInfo)[i];
      stride(subArrShapeInfo)[k--] = stride(wholeShapeInfo)[i];
    }
  }

  // calculation of sub-array offsets (subArrOffsets)
  calcOffsets(dimsSize, shape, strides, subArrOffsets);

  // evaluate ews
  checkStridesEwsAndOrder(subArrShapeInfo);

  delete[] strides;
  delete[] shape;
}



SD_LIB_EXPORT SD_INLINE SD_HOST void doPermuteShapeInfo(sd::LongType *shapeInfo,
                                                        const sd::LongType *rearrange,
                                                        sd::LongType len) {
  if (shapeInfo == nullptr || rearrange == nullptr || rank(shapeInfo) < 1) {
    return;
  }

  // note we used to automatically return early here but we can also permute
  // shapes like 1,2,1,0 (aka empty) and the shape there can matter.

  const sd::LongType rank = shape::rank(shapeInfo);

  // check whether rearrange is like {0,1,2,3,...}  - in this case we don't need permute as well
  bool isPermuteNecessary = false;
  for (sd::LongType i = 0; i < rank; i++) {
    if (rearrange[i] != i) {
      isPermuteNecessary = true;
      break;
    }
  }
  if (!isPermuteNecessary) {
    return;
  }

  // check whether rearrange contains correct indexes
  for (sd::LongType i = 0; i < rank; ++i) {
    if (rearrange[i] >= rank || rearrange[i] < 0) {
      std::string errorMessage;
      errorMessage += "shape::doPermuteShapeInfo function failed: rearrange indexes are incorrect. Given permute indices must be < rank and >= 0. Rearrange at index ";
      errorMessage += std::to_string(i);
      errorMessage += " was ";
      errorMessage += std::to_string(rearrange[i]);
      errorMessage += "\n";
      THROW_EXCEPTION(errorMessage.c_str());
    }
  }
  // if everything is ok then perform permute
  sd::LongType len2 = shapeInfoLength(rank);
  auto temp = new sd::LongType[len2];
  // note: it's obvious to do simd or something fancy
  // here it actually seems to cause segfaults. Better to be careful.
  for (int i = 0; i < len2; i++) {
    temp[i] = shapeInfo[i];
  }

  for (sd::LongType i = 0; i < rank; i++) {
    shapeInfo[i + 1] = temp[rearrange[i] + 1];
    shapeInfo[i + 1 + rank] = temp[rearrange[i] + 1 + rank];
  }

  checkStridesEwsAndOrder(shapeInfo);
  delete[] temp;

}

SD_LIB_EXPORT SD_INLINE SD_HOST sd::LongType tadLength(const sd::LongType *shapeInfo, const sd::LongType *dimension,
                                                       sd::LongType dimensionLength) {
  if (shapeInfo == nullptr || dimension == nullptr) {
    std::string errorMessage;
    errorMessage += "shape info null: %d";
    errorMessage += std::to_string(shapeInfo == nullptr);
    errorMessage += " dimension null: %d";
    errorMessage += std::to_string(dimension == nullptr);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (dimensionLength == 0) return 0;

  if (shapeInfo[0] > SD_MAX_RANK || shapeInfo[0] < 0)
    THROW_EXCEPTION("Corrupt shape information found. Potentially dellocated?");

  if (dimensionLength == 1) {
    if (dimension[0] > SD_MAX_RANK || dimension[0] < 0)
      THROW_EXCEPTION("Corrupt dimension information found. Potentially dellocated?");

    return shapeOf(shapeInfo)[dimension[0]];
  } else {
    sd::LongType ret = 1;
    for (sd::LongType i = 0; i < rank(shapeInfo); i++) {
      for (sd::LongType j = 0; j < dimensionLength; j++) {
        if (i == dimension[j]) ret *= shapeOf(shapeInfo)[dimension[j]];
      }
    }

    return ret;
  }
}

SD_LIB_EXPORT SD_INLINE SD_HOST int excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo, sd::LongType *shapeNoUnities,
                                                                sd::LongType *stridesNoUnities) {
  const int rank = shape::rank(inShapeInfo);
  const int numOfNonUnities = numOfNonUnitDims(rank, shapeOf(inShapeInfo));

  if (numOfNonUnities == rank) {  // no unities in shape, no copy procedure
    shapeNoUnities = const_cast<sd::LongType *>(inShapeInfo) + 1;
    stridesNoUnities = const_cast<sd::LongType *>(inShapeInfo) + 1 + rank;
    return numOfNonUnities;
  }

  int j = 0;
  for (int i = 0; i < rank; i++) {
    if (shapeOf(inShapeInfo)[i] != 1) {
      shapeNoUnities[j] = shapeOf(inShapeInfo)[i];
      stridesNoUnities[j++] = stride(inShapeInfo)[i];
    }
  }

  return numOfNonUnities;
}


SD_LIB_EXPORT SD_INLINE void SD_HOST checkStridesEwsAndOrder(sd::LongType *shapeInfo) {
  // FIXME - indeed we don't need to allocate so large memory amount (2*SD_MAX_RANK), sufficient amount is
  // (2*oldNumOfNonUnities + 2*newNumOfNonUnities)
  sd::LongType tempBuffer[2 * SD_MAX_RANK];
  sd::LongType *shape = tempBuffer, *strides = tempBuffer + shape::rank(shapeInfo);

  // exclude unities from shapeInfo
  const sd::LongType numOfNonUnities = excludeUnitiesFromShapeInfo(shapeInfo, shape, strides);

  checkStridesEwsAndOrder(shapeInfo, order(shapeInfo), numOfNonUnities, shape, strides);
}

//////////////////////////////////////////////////////////////////////
SD_LIB_EXPORT SD_INLINE void SD_HOST checkStridesEwsAndOrder(sd::LongType *shapeInfo, const char proposedOrder,
                                                             const sd::LongType numOfNonUnities, const sd::LongType *shapeNoUnities,
                                                             const sd::LongType *stridesNoUnities) {
  if (proposedOrder != 'c' && proposedOrder != 'f') {
    std::string errorMessage;
    errorMessage += "checkStridesEwsAndOrder: ";
    errorMessage += "proposedOrder is invalid !";
    errorMessage += " Expected c or f, but got ";
    errorMessage += proposedOrder;
    errorMessage += " instead !";
    THROW_EXCEPTION(errorMessage.c_str());
  }
  const sd::LongType rank = shape::rank(shapeInfo);
  if (length(shapeInfo) == 1) {
    setOrder(shapeInfo, proposedOrder);
    return;
  }

  if (numOfNonUnities == 1) {  // case of common vector
    setOrder(shapeInfo, proposedOrder);
    return;
  }

  bool contiguous = true;

  //*** check whether strides are in c contiguous order ***//
  for (sd::LongType i = 0; i < numOfNonUnities - 1; ++i) {
    if (stridesNoUnities[i] != shapeNoUnities[i + 1] * stridesNoUnities[i + 1]) {
      contiguous = false;
      break;
    }
  }

  if (contiguous) {
    setOrder(shapeInfo, 'c');
    return;
  }

  contiguous = true;

  //*** check whether strides are in f contiguous order ***//
  for (sd::LongType i = 1; i < numOfNonUnities; ++i) {
    if (stridesNoUnities[i] != shapeNoUnities[i - 1] * stridesNoUnities[i - 1]) {
      contiguous = false;
      break;
    }
  }

  if (contiguous) {
    setOrder(shapeInfo, 'f');
    return;
  }


  setOrder(shapeInfo, proposedOrder);
}


SD_INLINE SD_LIB_EXPORT SD_HOST void calcOffsets(const sd::LongType *shapeInfo, sd::LongType *offsets, const char order) {
  if (shapeInfo == nullptr) THROW_EXCEPTION("calcOffsets: shapeInfo is nullptr !");
  if (offsets == nullptr) THROW_EXCEPTION("calcOffsets: offsets is nullptr !");
  if (shapeInfo[0] < 0 || shapeInfo[0] > SD_MAX_RANK) THROW_EXCEPTION("calcOffsets: shapeInfo[0] is invalid !");
  // firstly consider simple case when ews > 0
  const sd::LongType ews = elementWiseStride(shapeInfo);

  if (ews > 0) {
    // set offset for first sub-array, it is equal to zero always
    offsets[0] = 0;

    sd::LongType e = 0;
    if (order != shape::order(shapeInfo))
      for (sd::LongType i = 1; i <= rank(shapeInfo); ++i)
        if (shapeInfo[i] != 1) ++e;  // check whether input is CommonVector

    if (order == shape::order(shapeInfo) || e == 1) {  // e==1 means common vector
      e = 1;
      sd::LongType len = length(shapeInfo);
      while (e < len) {
        offsets[e] = offsets[e - 1] + ews;
        e++;
      }
      return;
    }
  }

  calcOffsets(rank(shapeInfo), shapeOf(const_cast<sd::LongType *>(shapeInfo)),
              stride(const_cast<sd::LongType *>(shapeInfo)), offsets, order);
}

SD_INLINE SD_LIB_EXPORT SD_HOST void calcOffsets(const sd::LongType rank, const sd::LongType *shape, const sd::LongType *strides, sd::LongType *offsets,
                                                 const char order) {
  const sd::LongType len = prodLong(shape, rank);

  // set offset for first sub-array, it is equal to zero always
  offsets[0] = 0;

  sd::LongType coords[SD_MAX_RANK];
  memset(coords, 0, sizeof(sd::LongType) * rank);

  if (order == 'c') {
    for (sd::LongType i = 1; i < len; ++i) {
      sd::LongType axis = rank - 1;
      offsets[i] = 0;
      while (coords[axis] == shape[axis] - 1) {
        offsets[i] -= (shape[axis] - 1) * strides[axis];
        coords[axis--] = 0;
      }
      ++coords[axis];
      offsets[i] += offsets[i - 1] + strides[axis];
    }
  } else {
    for (sd::LongType i = 1; i < len; ++i) {
      sd::LongType axis = 0;
      offsets[i] = 0;
      while (coords[axis] == shape[axis] - 1) {
        offsets[i] -= (shape[axis] - 1) * strides[axis];
        coords[axis++] = 0;
      }
      ++coords[axis];
      offsets[i] += offsets[i - 1] + strides[axis];
    }
  }
}

//////////////////////////////////////////////////////////////////////
SD_LIB_EXPORT SD_HOST  SD_INLINE void calcSubArrShapeInfoAndOffset(const sd::LongType *idx, const sd::LongType *maxShapeInfo, sd::LongType *minShapeInfo,
                                                                   sd::LongType &minOffset, const bool keepUnitiesInShape, const bool isStrided,
                                                                   const sd::LongType numOfUntiesInMinShape) {
  if (sd::ArrayOptions::dataType(maxShapeInfo) == sd::DataType::UNKNOWN) {
    THROW_EXCEPTION("calcSubArrShapeInfoAndOffset: maxShapeInfo has unknown data type !");
  }

  const sd::LongType maxRank = rank(maxShapeInfo);
  minOffset = 0;
  sd::LongType first, last, stride, n(isStrided ? 3 : 2);

  minShapeInfo[0] = keepUnitiesInShape ? maxRank : maxRank - numOfUntiesInMinShape;

  for (sd::LongType step = 0, j = 0, i = 0; i < maxRank; ++i, step += n) {
    if (idx[step] == idx[step + 1]) {  // means whole dimension
      shapeOf(minShapeInfo)[j] = shapeOf(maxShapeInfo)[i];
      shape::stride(minShapeInfo)[j++] = shape::stride(maxShapeInfo)[i];
    } else {
      first = idx[step] >= 0 ? idx[step] : idx[step] + sizeAt(maxShapeInfo, i) + 1;
      last = idx[step + 1] >= 0 ? idx[step + 1] : idx[step + 1] + sizeAt(maxShapeInfo, i) + 1;

      if (last < first)
        THROW_EXCEPTION("shape::calcSubArrShapeInfoAndOffset: negative range in input indexes is found!");

      if (isStrided) {
        stride = idx[step + 2];
        last /*resulting sub-array axis*/ = (last - first + stride - 1) / stride;  // ceil (last - first) / stride;
      } else {
        stride = 1;
        last /*resulting sub-array axis*/ = last - first;
      }

      minOffset += first * shape::stride(maxShapeInfo)[i];

      if (!keepUnitiesInShape && last == 1) continue;

      shapeOf(minShapeInfo)[j] = last;
      shape::stride(minShapeInfo)[j++] =
          last == 1 ? shape::stride(maxShapeInfo)[i] : shape::stride(maxShapeInfo)[i] * stride;
    }
  }

  setExtra(minShapeInfo, extra(maxShapeInfo));
  setOrder(minShapeInfo, 'c');                                                     // order
  sd::ArrayOptions::setDataType(minShapeInfo, sd::ArrayOptions::dataType(maxShapeInfo));  // type
  checkStridesEwsAndOrder(minShapeInfo);
  if (sd::ArrayOptions::dataType(minShapeInfo) == sd::DataType::UNKNOWN)
    THROW_EXCEPTION("Attempted to set unknown data type for minShapeInfo !");
}

SD_LIB_EXPORT SD_HOST_DEVICE SD_INLINE void updateStrides(sd::LongType *shapeInfo, const char order,
                                                          bool resetStridesIfView) {
  sd::LongType rank = shapeInfo[0];

  if(rank < 0 || rank > SD_MAX_RANK) {
    THROW_EXCEPTION("Invalid rank value. Ensure a rank has been assigned.");
  }

  sd::LongType doubleRank = 2 * rank;
  if (isEmpty(shapeInfo)) {
    auto strides = stride(shapeInfo);
    for (int i = 0; i < rank; i++) {
      strides[i] = 0;
    }
  }

  //strides from views can be strange, we provide a knob here
  //for times where we just need new strides when copying
  //from a view to a standalone array. this is common when working
  //with linear parameter vectors representing a full network
  //but may need a reshape or permute (view creation) where a
  //fresh array and buffer are created with new strides.
  if(resetStridesIfView && shape::isView(shapeInfo)) {
    auto strides = stride(shapeInfo);
    if(order == 'c') {
      shape::calcStrides(shape::shapeOf(shapeInfo), rank, strides);
    } else if(order == 'f') {
      shape::calcStridesFortran(shape::shapeOf(shapeInfo), rank, strides);
    }
  } else {
    if (rank > 0) {
      if (order == 'c') {
        shapeInfo[doubleRank] = 1;  // set unity as last stride for c order
        for (sd::LongType j = 1; j < rank; j++) {
          shapeInfo[doubleRank - j] = shapeInfo[doubleRank - j + 1] * shapeInfo[rank + 1 - j];
        }


      } else {
        shapeInfo[rank + 1] = 1;  // set unity as first stride for f order
        for (sd::LongType j = rank + 1; j < doubleRank; j++) {
          shapeInfo[j + 1] = shapeInfo[j] * shapeInfo[j - rank];
        }
      }
    }
  }

  // set last 2 elements in shapeInfo
  shapeInfo[doubleRank + 2] = 1;
  setOrder(shapeInfo, order);
}

SD_LIB_EXPORT SD_INLINE SD_HOST void updateStrides(const sd::LongType rank, const sd::LongType *shapeOnly, sd::LongType *stridesOnly,
                                                   const char order) {
  if (rank > 0) {
    if (order == 'c') {
      stridesOnly[rank - 1] = 1;  // set unity as last stride for c order
      for (sd::LongType j = 1; j < rank; ++j) stridesOnly[rank - 1 - j] = stridesOnly[rank - j] * shapeOnly[rank - j];
    } else {
      stridesOnly[0] = 1;  // set unity as first stride for f order
      for (sd::LongType j = 1; j < rank; ++j) {
        stridesOnly[j] = stridesOnly[j - 1] * shapeOnly[j - 1];
      }
    }
  }
}
/**
* @param toCopy the shape to copy
* @return a copy of the original struct
*/
SD_LIB_EXPORT SD_INLINE SD_HOST ShapeInformation *shapeCopy(ShapeInformation *toCopy) {
  auto copy = new ShapeInformation;

  copy->shape = new sd::LongType[toCopy->rank];

  memcpy(copy->shape, toCopy->shape, toCopy->rank * sizeof(sd::LongType));

  copy->stride = new sd::LongType[toCopy->rank];
  for (sd::LongType i = 0; i < toCopy->rank; i++) {
    copy->stride[i] = toCopy->stride[i];
  }
  copy->order = toCopy->order;
  copy->rank = toCopy->rank;
  copy->offset = toCopy->offset;
  copy->elementWiseStride = toCopy->elementWiseStride;
  return copy;
}



SD_LIB_EXPORT SD_INLINE SD_HOST bool reshapeC(const sd::LongType *oldShapeInfo,
                                              const char newOrder,
                                              const sd::LongType newRank,
                                              const sd::LongType *newShape,
                                              sd::LongType *newShapeInfo) {
  // copy shape from newShape into newShapeInfo
  newShapeInfo[0] = newRank;
  memcpy(newShapeInfo + 1, newShape, newRank * sizeof(sd::LongType));

  // copy order
  newShapeInfo[2 * newRank + 3] = newOrder;
  sd::ArrayOptions::copyDataType(newShapeInfo, oldShapeInfo);
  setOrder(newShapeInfo, newOrder);

  // inherit old data type
  auto ret =  reshapeC(oldShapeInfo, newShapeInfo);
  return ret;
}

SD_LIB_EXPORT SD_INLINE SD_HOST void fillStrides(sd::LongType  *shapeInfo) {
  // double checks if the _rank and _shape_strides are set correctly before filling strides
  auto _shape = shape::shapeOf(shapeInfo);
  auto _strides = shape::stride(shapeInfo);
  auto rank = shape::rank(shapeInfo);
  auto order = shape::order(shapeInfo);
  if (rank > 0 && !shape::isEmptyConst(shapeInfo)) {
    if (order == 'c')
      shape::calcStrides(_shape, rank, _strides);
    else
      shape::calcStridesFortran(_shape, rank, _strides);

  } else {
    for (int i = 0; i < rank; i++) {
      _strides[i] = 0;
    }
  }
}

//////////////////////////////////////////////////////////////////////


SD_LIB_EXPORT SD_INLINE SD_HOST bool reshapeC(const sd::LongType *oldShapeInfo, sd::LongType *newShapeInfo) {
  // newShapeInfo contains rank, shape and order; but no strides, type and ews
  const sd::LongType newRank = shape::rank(newShapeInfo);

  auto oldDt = sd::ArrayOptions::dataType(oldShapeInfo);
  if (oldDt == sd::DataType::UNKNOWN) {
    THROW_EXCEPTION("Attempting to reshape with an unknown data type");
  }

  // if oldShapeInfo is scalar or vector with length=1
  if (shape::length(oldShapeInfo) <= 1) {
    for (sd::LongType i = 0; i < newRank; ++i) shape::stride(newShapeInfo)[i] = 1;
    sd::ArrayOptions::setDataType(newShapeInfo, sd::ArrayOptions::dataType(oldShapeInfo));
    return true;
  }

  const auto oldOrder = shape::order(oldShapeInfo);
  const auto newOrder = shape::order(newShapeInfo);

  // Calculate new strides
  sd::LongType newStride = 1;
  if (newOrder == 'c') {
    for (int i = newRank - 1; i >= 0; --i) {
      shape::stride(newShapeInfo)[i] = newStride;
      newStride *= shape::shapeOf(newShapeInfo)[i];
    }
  } else { // 'f' order
    for (int i = 0; i < newRank; ++i) {
      shape::stride(newShapeInfo)[i] = newStride;
      newStride *= shape::shapeOf(newShapeInfo)[i];
    }
  }

  // Check if the reshape is valid (total number of elements should remain the same)
  sd::LongType oldLength = shape::length(oldShapeInfo);
  sd::LongType newLength = shape::length(newShapeInfo);

  if (oldLength != newLength) {
    THROW_EXCEPTION("Invalid reshape: total number of elements must remain the same");
  }


  // Set ews and order
  shape::checkStridesEwsAndOrder(newShapeInfo);

  sd::ArrayOptions::setDataType(newShapeInfo, oldDt);
  sd::ArrayOptions::setExtra(newShapeInfo, sd::ArrayOptions::extra(oldShapeInfo));

  return true;
}

}  // namespace shape


#endif // SHAPE_HXX_
